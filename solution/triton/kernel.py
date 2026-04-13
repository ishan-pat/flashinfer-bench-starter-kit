"""
Fused MoE Triton Kernel — Token-Sorted Grouped GEMM variant.

Target: moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048
  - DeepSeek-V3/R1 scale MoE: hidden=7168, intermediate=2048
  - 32 local experts (256 global), top-8 routing
  - FP8 E4M3FN weights with 128-element block scaling
  - DeepSeek routing: 8 groups, top-4 groups, top-8 experts

Optimization: token-sorted grouped GEMM.
  Old: 32 Python-loop iterations × 2 kernels = 64 serial launches.
  New: 3 Triton kernel launches covering all 32 experts simultaneously.

Pipeline:
  routing (PyTorch) → token sort (PyTorch) →
  grouped_gemm1 (FP8 WGMMA) → swiglu (fused) → grouped_gemm2 (BF16×FP8) →
  cast to bf16 output
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_GLOBAL_EXPERTS  = 256
NUM_GROUPS          = 8      # ng8
EXPERTS_PER_GROUP   = 32     # 256 / 8
NUM_GROUPS_SELECTED = 4      # kg4
TOPK                = 8      # topk8
NUM_LOCAL_EXPERTS   = 32     # e32
HIDDEN_DIM          = 7168   # h7168
INTER_DIM           = 2048   # i2048
FP8_BLOCK_SIZE      = 128
GATE_UP_DIM         = INTER_DIM * 2  # 4096


# ---------------------------------------------------------------------------
# Autotune configs for B200
# BLOCK_K is always 128 — must align with FP8 block-scale granularity.
# BLOCK_N=256 is valid: n_blks = n_range // 128 handles multi-block tiles.
# ---------------------------------------------------------------------------
def _gemm_configs():
    configs = []
    for bm in [64, 128]:
        for bn in [128, 256]:
            for ns in [3, 4, 5]:
                for nw in [8, 16]:
                    configs.append(triton.Config(
                        {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": 128},
                        num_stages=ns, num_warps=nw,
                    ))
    return configs


def _bucket(n: int) -> int:
    """Round n up to next power of 2 for autotune key bucketing."""
    for t in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]:
        if n <= t:
            return t
    return 16384


# ---------------------------------------------------------------------------
# Routing (PyTorch — verified numerically correct against reference)
#
# Returns:
#   expert_ids   [seq, 8]   int32   — global expert IDs selected per token
#   weights_full [seq, 256] float32 — normalized routing weight per (token, expert)
#                                     non-zero only for the 8 selected experts;
#                                     RSF is applied later in GEMM2 epilogue.
# ---------------------------------------------------------------------------
def _route(routing_logits, routing_bias, seq_len, device):
    bias_f32     = routing_bias.to(torch.float32)
    s            = torch.sigmoid(routing_logits.float())   # [seq, 256] unbiased
    s_with_bias  = s + bias_f32                            # [seq, 256] biased

    # Group scoring: top-2 sum per group (on biased scores)
    gsm          = s_with_bias.view(seq_len, NUM_GROUPS, EXPERTS_PER_GROUP)
    top2_vals, _ = gsm.topk(2, dim=-1)
    group_scores = top2_vals.sum(-1)                       # [seq, 8]

    _, top_groups = group_scores.topk(NUM_GROUPS_SELECTED, dim=-1)  # [seq, 4]

    group_mask   = torch.zeros_like(group_scores)
    group_mask.scatter_(1, top_groups, 1.0)
    score_mask   = (group_mask.unsqueeze(2)
                    .expand(seq_len, NUM_GROUPS, EXPERTS_PER_GROUP)
                    .reshape(seq_len, NUM_GLOBAL_EXPERTS))

    neg_inf      = torch.finfo(torch.float32).min
    scores_pruned = s_with_bias.masked_fill(score_mask == 0, neg_inf)
    _, top8_ids  = scores_pruned.topk(TOPK, dim=-1)        # [seq, 8]

    # Routing weights: unbiased s, normalized, RSF applied later
    weight_mask  = torch.zeros_like(s)
    weight_mask.scatter_(1, top8_ids, 1.0)
    weights      = s * weight_mask
    weights_sum  = weights.sum(-1, keepdim=True).clamp(min=1e-20)
    weights_full = weights / weights_sum                    # [seq, 256]

    return top8_ids.to(torch.int32), weights_full


# ---------------------------------------------------------------------------
# Token sorting
#
# Flattens all (token, topk_slot) expert assignments, keeps only those mapping
# to local experts [local_offset, local_offset+31], sorts by local expert id,
# and computes cumulative per-expert offsets into the sorted array.
#
# Returns:
#   sorted_tok_ids  [N_assigned] int32   global token row indices, sorted by expert
#   expert_offsets  [33]         int32   expert_offsets[e] = start idx for expert e
#   sorted_r_scores [N_assigned] float32 normalized routing weight (without RSF)
#   N_assigned      int                  total local-expert token assignments
#   max_tok         int                  max tokens assigned to any single local expert
# ---------------------------------------------------------------------------
def _sort_tokens(expert_ids, weights_full, local_offset, device):
    # Identify (token, slot) pairs whose expert is local
    local_mask = ((expert_ids >= local_offset) &
                  (expert_ids <  local_offset + NUM_LOCAL_EXPERTS))
    pairs = local_mask.nonzero(as_tuple=False)              # [N_assigned, 2]

    if pairs.shape[0] == 0:
        empty   = torch.zeros(0, dtype=torch.int32, device=device)
        offsets = torch.zeros(NUM_LOCAL_EXPERTS + 1, dtype=torch.int32, device=device)
        return empty, offsets, empty, 0, 0

    tok_global = pairs[:, 0]                                # int64 token row
    slot       = pairs[:, 1]
    global_exp = expert_ids[tok_global, slot]               # int32 global expert id
    local_exp  = (global_exp - local_offset).to(torch.int32)

    r_scores   = weights_full[tok_global, global_exp.long()].to(torch.float32)

    # Sort by local expert id (stable → tokens within an expert stay in row order)
    order           = torch.argsort(local_exp, stable=True)
    sorted_tok_ids  = tok_global[order].to(torch.int32).contiguous()
    sorted_local    = local_exp[order].contiguous()
    sorted_r_scores = r_scores[order].contiguous()

    counts         = torch.bincount(sorted_local, minlength=NUM_LOCAL_EXPERTS)
    expert_offsets = torch.zeros(NUM_LOCAL_EXPERTS + 1, dtype=torch.int32, device=device)
    expert_offsets[1:] = counts.cumsum(0).to(torch.int32)

    N_assigned = int(sorted_tok_ids.shape[0])
    max_tok    = int(counts.max().item())

    return sorted_tok_ids, expert_offsets, sorted_r_scores, N_assigned, max_tok


# ---------------------------------------------------------------------------
# Kernel 1: Grouped GEMM1
#
# gate_up_buf[sorted_pos] = hidden[sorted_tok_ids[sorted_pos]] @ w1[expert].T
# for all 32 local experts in a single launch.
#
# Grid: (32 * ceil(max_tok / BLOCK_M),  ceil(GATE_UP_DIM / BLOCK_N))
#
# Expert dispatch:
#   tiles_per_expert = ceil(max_tok / BLOCK_M)  [computed INSIDE kernel from constexpr]
#   expert_id        = pid_m // tiles_per_expert
#   m_in_expert      = pid_m %  tiles_per_expert
#
# Scale correctness:
#   After FP8 dot, multiply by h_scales[:, None] * w_scales[None, :]
#   where h_scales[i] = hidden_states_scale[k_blk, tok_ids[i]]
#         w_scales[j] = gemm1_weights_scale[expert_id, n_blks[j], k_blk]
#   n_blks is vectorized across the N tile so BLOCK_N=256 (two scale blocks) works.
# ---------------------------------------------------------------------------
@triton.autotune(configs=_gemm_configs(), key=["n_assigned_bucket", "N"])
@triton.jit
def grouped_gemm1_kernel(
    # Sorted token data
    sorted_tok_ids_ptr,    # [N_assigned] int32
    expert_offsets_ptr,    # [33]         int32
    max_tok,               # int — max tokens per expert (for tiles_per_expert calc)

    # Hidden states (full sequence, gather via tok_ids)
    hidden_ptr,            # [seq_len, 7168]   fp8_e4m3fn
    hscale_ptr,            # [56, seq_len]     float32

    # Per-expert weights  (32 experts flattened)
    w1_ptr,                # [32, 4096, 7168]  fp8_e4m3fn
    w1scale_ptr,           # [32, 32, 56]      float32

    # Output
    gate_up_ptr,           # [N_assigned, 4096] float32
    N_assigned,
    n_assigned_bucket,     # autotune key (bucketed N_assigned)

    # Shape constants
    K: tl.constexpr,       # 7168
    N: tl.constexpr,       # 4096

    # Strides
    stride_h_seq,          # hidden_states.stride(0)          = 7168
    stride_w1_exp,         # gemm1_weights.stride(0)          = 4096*7168
    stride_w1_n,           # gemm1_weights.stride(1)          = 7168
    stride_w1s_exp,        # gemm1_weights_scale.stride(0)    = 32*56
    stride_w1s_nb,         # gemm1_weights_scale.stride(1)    = 56
    stride_hscale_blk,     # hidden_states_scale.stride(0)    = seq_len
    stride_gu_tok,         # gate_up_buf.stride(0)            = 4096

    # Block sizes (constexpr — chosen by autotune)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,  # always 128
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # --- Expert dispatch ---
    # tiles_per_expert is computed from constexpr BLOCK_M so it matches the lambda grid.
    tiles_per_expert = tl.cdiv(max_tok, BLOCK_M)
    expert_id        = pid_m // tiles_per_expert
    m_in_expert      = pid_m %  tiles_per_expert

    e_start      = tl.load(expert_offsets_ptr + expert_id)
    e_end        = tl.load(expert_offsets_ptr + expert_id + 1)
    n_expert_tok = e_end - e_start

    # Exit if this M-tile is beyond this expert's token count
    m_off = m_in_expert * BLOCK_M
    if m_off >= n_expert_tok:
        return

    # Positions in the sorted array for this tile
    m_range  = m_off + tl.arange(0, BLOCK_M)
    m_mask   = m_range < n_expert_tok
    global_m = e_start + m_range                            # indices into sorted arrays

    # Gather global token row indices
    tok_ids = tl.load(sorted_tok_ids_ptr + global_m, mask=m_mask, other=0)  # [BM]

    # N tile
    n_start  = pid_n * BLOCK_N
    n_range  = n_start + tl.arange(0, BLOCK_N)
    n_mask   = n_range < N
    n_blks   = n_range // BLOCK_K                           # [BN] scale block per column

    # Expert weight base pointers
    w1_base  = w1_ptr    + expert_id * stride_w1_exp
    w1s_base = w1scale_ptr + expert_id * stride_w1s_exp

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k_start in tl.range(0, K, BLOCK_K):
        k_blk   = k_start // BLOCK_K
        k_range = k_start + tl.arange(0, BLOCK_K)
        k_mask  = k_range < K

        # Load FP8 hidden  [BLOCK_M, BLOCK_K]  (gather rows by tok_ids)
        h_ptrs  = hidden_ptr + tok_ids[:, None] * stride_h_seq + k_range[None, :]
        h_fp8   = tl.load(h_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)

        # Per-token, per-K-block hidden scale  hscale[k_blk, tok_id]  →  [BM]
        h_scales = tl.load(
            hscale_ptr + k_blk * stride_hscale_blk + tok_ids,
            mask=m_mask, other=1.0,
        )

        # Load FP8 weight  [BLOCK_N, BLOCK_K]
        w_ptrs  = w1_base + n_range[:, None] * stride_w1_n + k_range[None, :]
        w_fp8   = tl.load(w_ptrs, mask=n_mask[:, None] & k_mask[None, :], other=0.0)

        # Per-N-block, per-K-block weight scale (vectorized across columns)  [BN]
        # w1scale[expert, n_blk, k_blk] — stride_w1s_nb = 56
        w_scales = tl.load(
            w1s_base + n_blks * stride_w1s_nb + k_blk,
            mask=n_mask, other=1.0,
        )

        # Native FP8 WGMMA → float32, then apply block scales
        raw  = tl.dot(h_fp8, tl.trans(w_fp8), out_dtype=tl.float32)
        acc += raw * h_scales[:, None] * w_scales[None, :]

    # Write gate+up buffer at sorted positions
    out_m    = e_start + m_range
    out_ptrs = gate_up_ptr + out_m[:, None] * stride_gu_tok + n_range[None, :]
    tl.store(out_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


# ---------------------------------------------------------------------------
# Kernel 2: Fused SwiGLU
#
# inter = silu(gate) * up
# where  gate = gate_up[:, :2048]  (first  half — matches feature/batch_gemms fix)
#        up   = gate_up[:, 2048:]  (second half)
#
# Grid: (ceil(N_assigned * INTER_DIM / BLOCK_SIZE),)
# ---------------------------------------------------------------------------
@triton.jit
def swiglu_kernel(
    gate_up_ptr,           # [N_assigned, 4096] float32  INPUT
    inter_ptr,             # [N_assigned, 2048] float32  OUTPUT
    N_assigned,
    INTER_DIM: tl.constexpr,   # 2048
    BLOCK_SIZE: tl.constexpr,  # 1024
):
    pid  = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N_assigned * INTER_DIM

    tok  = offs // INTER_DIM
    col  = offs %  INTER_DIM
    row_stride = INTER_DIM * 2                              # gate_up row width = 4096

    gate = tl.load(gate_up_ptr + tok * row_stride + col,              mask=mask, other=0.0)
    up   = tl.load(gate_up_ptr + tok * row_stride + INTER_DIM + col,  mask=mask, other=0.0)

    # silu(gate) * up
    out  = gate / (1.0 + tl.exp(-gate)) * up

    tl.store(inter_ptr + offs, out, mask=mask)


# ---------------------------------------------------------------------------
# Kernel 3: Grouped GEMM2
#
# output_f32[tok_ids[m]] += (r_scores[m] * rsf) × inter[m] @ w2[expert].T
# for all 32 local experts in a single launch.
#
# Grid: same structure as GEMM1.
# Accumulation: tl.atomic_add scatter (top-8 routing → multiple experts per token).
# ---------------------------------------------------------------------------
@triton.autotune(configs=_gemm_configs(), key=["n_assigned_bucket", "N"])
@triton.jit
def grouped_gemm2_kernel(
    # Sorted token data
    sorted_tok_ids_ptr,    # [N_assigned] int32
    sorted_r_scores_ptr,   # [N_assigned] float32  (normalized weight, no RSF)
    expert_offsets_ptr,    # [33]         int32
    max_tok,               # int

    # Intermediate (SwiGLU output, indexed by sorted position)
    inter_ptr,             # [N_assigned, 2048] float32

    # Per-expert weights
    w2_ptr,                # [32, 7168, 2048]  fp8_e4m3fn
    w2scale_ptr,           # [32, 56, 16]      float32

    # Output accumulation
    output_ptr,            # [seq_len, 7168]   float32  (zero-initialised, atomic add)
    routed_scaling_factor,

    N_assigned,
    n_assigned_bucket,

    # Shape constants
    K: tl.constexpr,       # 2048
    N: tl.constexpr,       # 7168

    # Strides
    stride_inter_tok,      # inter_buf.stride(0)            = 2048
    stride_w2_exp,         # gemm2_weights.stride(0)        = 7168*2048
    stride_w2_n,           # gemm2_weights.stride(1)        = 2048
    stride_w2s_exp,        # gemm2_weights_scale.stride(0)  = 56*16
    stride_w2s_nb,         # gemm2_weights_scale.stride(1)  = 16
    stride_out_seq,        # output_f32.stride(0)           = 7168

    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    tiles_per_expert = tl.cdiv(max_tok, BLOCK_M)
    expert_id        = pid_m // tiles_per_expert
    m_in_expert      = pid_m %  tiles_per_expert

    e_start      = tl.load(expert_offsets_ptr + expert_id)
    e_end        = tl.load(expert_offsets_ptr + expert_id + 1)
    n_expert_tok = e_end - e_start

    m_off = m_in_expert * BLOCK_M
    if m_off >= n_expert_tok:
        return

    m_range  = m_off + tl.arange(0, BLOCK_M)
    m_mask   = m_range < n_expert_tok
    global_m = e_start + m_range

    # Load routing info
    tok_ids  = tl.load(sorted_tok_ids_ptr  + global_m, mask=m_mask, other=0)
    r_scores = tl.load(sorted_r_scores_ptr + global_m, mask=m_mask, other=0.0)
    weight   = r_scores * routed_scaling_factor             # [BM]

    # N tile
    n_start  = pid_n * BLOCK_N
    n_range  = n_start + tl.arange(0, BLOCK_N)
    n_mask   = n_range < N
    n_blks   = n_range // BLOCK_K                           # [BN]

    w2_base  = w2_ptr    + expert_id * stride_w2_exp
    w2s_base = w2scale_ptr + expert_id * stride_w2s_exp

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k_start in tl.range(0, K, BLOCK_K):
        k_blk   = k_start // BLOCK_K
        k_range = k_start + tl.arange(0, BLOCK_K)
        k_mask  = k_range < K

        # Load intermediate (float32) at sorted positions → cast to bf16
        i_ptrs = inter_ptr + global_m[:, None] * stride_inter_tok + k_range[None, :]
        i_mk   = tl.load(i_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        i_bf16 = i_mk.to(tl.bfloat16)

        # Load FP8 weight  [BLOCK_N, BLOCK_K]
        w_ptrs  = w2_base + n_range[:, None] * stride_w2_n + k_range[None, :]
        w_fp8   = tl.load(w_ptrs, mask=n_mask[:, None] & k_mask[None, :], other=0.0)

        # Per-N-block, per-K-block weight scale  [BN]
        # w2scale[expert, n_blk, k_blk] — stride_w2s_nb = 16
        w_scales = tl.load(
            w2s_base + n_blks * stride_w2s_nb + k_blk,
            mask=n_mask, other=1.0,
        )

        # BF16 × dequantized-FP8 WGMMA  (scale applied to weight before dot)
        w_bf16 = (w_fp8.to(tl.float32) * w_scales[:, None]).to(tl.bfloat16)
        acc   += tl.dot(i_bf16, tl.trans(w_bf16), out_dtype=tl.float32)

    # Apply routing weight and scatter-add to output (multiple experts → same token row)
    acc      = acc * weight[:, None]
    out_ptrs = output_ptr + tok_ids[:, None] * stride_out_seq + n_range[None, :]
    tl.atomic_add(out_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


# ---------------------------------------------------------------------------
# Entry point — matches config.toml: entry_point = "kernel.py::kernel"
# DPS: output tensor pre-allocated by framework and passed as last argument.
# ---------------------------------------------------------------------------
def kernel(
    routing_logits,         # float32,       [seq_len, 256]
    routing_bias,           # bfloat16,      [256]
    hidden_states,          # float8_e4m3fn, [seq_len, 7168]
    hidden_states_scale,    # float32,       [56, seq_len]
    gemm1_weights,          # float8_e4m3fn, [32, 4096, 7168]
    gemm1_weights_scale,    # float32,       [32, 32, 56]
    gemm2_weights,          # float8_e4m3fn, [32, 7168, 2048]
    gemm2_weights_scale,    # float32,       [32, 56, 16]
    local_expert_offset,    # int32 scalar
    routed_scaling_factor,  # float32 scalar
    output,                 # bfloat16,      [seq_len, 7168]  (DPS)
):
    seq_len      = hidden_states.shape[0]
    device       = hidden_states.device
    rsf          = float(routed_scaling_factor)
    local_offset = int(local_expert_offset)

    # -------------------------------------------------------------------------
    # 1. Routing
    # -------------------------------------------------------------------------
    expert_ids, weights_full = _route(
        routing_logits, routing_bias, seq_len, device
    )

    # -------------------------------------------------------------------------
    # 2. Token sorting
    # -------------------------------------------------------------------------
    sorted_tok_ids, expert_offsets, sorted_r_scores, N_assigned, max_tok = _sort_tokens(
        expert_ids, weights_full, local_offset, device
    )

    # -------------------------------------------------------------------------
    # 3. Allocate buffers; handle empty case
    # -------------------------------------------------------------------------
    output_f32 = torch.zeros(seq_len, HIDDEN_DIM, dtype=torch.float32, device=device)

    if N_assigned == 0:
        output.zero_()
        return

    gate_up_buf = torch.empty(N_assigned, GATE_UP_DIM, dtype=torch.float32, device=device)
    inter_buf   = torch.empty(N_assigned, INTER_DIM,   dtype=torch.float32, device=device)

    # -------------------------------------------------------------------------
    # 4. Grouped GEMM1  (FP8 WGMMA for all 32 experts)
    #
    # Lambda grid: autotune picks BLOCK_M, grid M = 32 * ceil(max_tok / BLOCK_M).
    # tiles_per_expert is computed INSIDE the kernel as tl.cdiv(max_tok, BLOCK_M)
    # using the constexpr BLOCK_M — guarantees grid ↔ dispatch consistency.
    # -------------------------------------------------------------------------
    grouped_gemm1_kernel[
        lambda meta: (
            NUM_LOCAL_EXPERTS * triton.cdiv(max_tok, meta["BLOCK_M"]),
            triton.cdiv(GATE_UP_DIM, meta["BLOCK_N"]),
        )
    ](
        sorted_tok_ids, expert_offsets, max_tok,
        hidden_states, hidden_states_scale,
        gemm1_weights, gemm1_weights_scale,
        gate_up_buf,
        N_assigned,
        _bucket(N_assigned),
        K=HIDDEN_DIM,
        N=GATE_UP_DIM,
        stride_h_seq      = hidden_states.stride(0),
        stride_w1_exp     = gemm1_weights.stride(0),
        stride_w1_n       = gemm1_weights.stride(1),
        stride_w1s_exp    = gemm1_weights_scale.stride(0),
        stride_w1s_nb     = gemm1_weights_scale.stride(1),
        stride_hscale_blk = hidden_states_scale.stride(0),
        stride_gu_tok     = gate_up_buf.stride(0),
    )

    # -------------------------------------------------------------------------
    # 5. Fused SwiGLU
    # -------------------------------------------------------------------------
    SWIGLU_BLOCK = 1024
    swiglu_kernel[triton.cdiv(N_assigned * INTER_DIM, SWIGLU_BLOCK),](
        gate_up_buf, inter_buf,
        N_assigned,
        INTER_DIM=INTER_DIM,
        BLOCK_SIZE=SWIGLU_BLOCK,
    )

    # -------------------------------------------------------------------------
    # 6. Grouped GEMM2  (BF16×FP8 WGMMA, scatter-add into output_f32)
    # -------------------------------------------------------------------------
    grouped_gemm2_kernel[
        lambda meta: (
            NUM_LOCAL_EXPERTS * triton.cdiv(max_tok, meta["BLOCK_M"]),
            triton.cdiv(HIDDEN_DIM, meta["BLOCK_N"]),
        )
    ](
        sorted_tok_ids, sorted_r_scores, expert_offsets, max_tok,
        inter_buf,
        gemm2_weights, gemm2_weights_scale,
        output_f32,
        rsf,
        N_assigned,
        _bucket(N_assigned),
        K=INTER_DIM,
        N=HIDDEN_DIM,
        stride_inter_tok  = inter_buf.stride(0),
        stride_w2_exp     = gemm2_weights.stride(0),
        stride_w2_n       = gemm2_weights.stride(1),
        stride_w2s_exp    = gemm2_weights_scale.stride(0),
        stride_w2s_nb     = gemm2_weights_scale.stride(1),
        stride_out_seq    = output_f32.stride(0),
    )

    # -------------------------------------------------------------------------
    # 7. Cast float32 accumulation to bf16 output
    # -------------------------------------------------------------------------
    output.copy_(output_f32.to(torch.bfloat16))
