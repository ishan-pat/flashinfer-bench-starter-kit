"""
Fused MoE Triton Kernel — Token-Sorted Grouped GEMM variant.

Target: moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048
  - DeepSeek-V3/R1 scale MoE: hidden=7168, intermediate=2048
  - 32 local experts (256 global), top-8 routing
  - FP8 E4M3FN weights with 128-element block scaling
  - DeepSeek routing: 8 groups, top-4 groups, top-8 experts

Optimization: token-sorted grouped GEMM + fused SwiGLU epilogue.
  Old: 32 Python-loop iterations × 2 kernels = 64 serial launches.
  New: 2 Triton kernel launches covering all 32 experts simultaneously.
  Also eliminates full upfront weight dequantization (~5.7 GB materialized in baseline)
  and the gate_up_buf round-trip (~0.5 GB at full batch).

Pipeline:
  routing (PyTorch, proven correct) → token sort (PyTorch) →
  grouped_gemm1 (FP8 MMA + fused SwiGLU epilogue, writes inter directly) →
  grouped_gemm2 (FP8 dequant → float32 dot, atomic scatter) →
  cast to bf16 output

SwiGLU convention (verified against baseline passing all 19 workloads):
  w1[:, :INTER_DIM,       :] → up    rows (first_half of gate_up)
  w1[:, INTER_DIM:,       :] → gate  rows (second_half of gate_up)
  output = silu(gate) * up   ==   silu(second_half) * first_half
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
# Routing — mirrors proven-correct PyTorch baseline exactly.
#
# Returns:
#   expert_ids   [seq, 8]   int32   — global expert IDs selected per token
#   weights_full [seq, 256] float32 — normalized routing weight per (token, expert)
#                                     non-zero only for 8 selected experts.
#                                     RSF NOT applied here; applied in GEMM2 epilogue.
# ---------------------------------------------------------------------------
def _route(routing_logits, routing_bias, seq_len, device):
    bias_f32    = routing_bias.to(torch.float32)
    s           = torch.sigmoid(routing_logits.float())   # [seq, 256] unbiased
    s_with_bias = s + bias_f32                            # [seq, 256] biased

    # Group scoring: top-2 sum per group (biased)
    gsm          = s_with_bias.view(seq_len, NUM_GROUPS, EXPERTS_PER_GROUP)
    top2_vals, _ = gsm.topk(2, dim=-1)
    group_scores = top2_vals.sum(-1)                      # [seq, 8]

    _, top_groups = group_scores.topk(NUM_GROUPS_SELECTED, dim=-1)  # [seq, 4]
    group_mask    = torch.zeros_like(group_scores)
    group_mask.scatter_(1, top_groups, 1.0)
    score_mask    = (group_mask.unsqueeze(2)
                     .expand(seq_len, NUM_GROUPS, EXPERTS_PER_GROUP)
                     .reshape(seq_len, NUM_GLOBAL_EXPERTS))

    neg_inf       = torch.finfo(torch.float32).min
    scores_pruned = s_with_bias.masked_fill(score_mask == 0, neg_inf)
    _, top8_ids   = scores_pruned.topk(TOPK, dim=-1)     # [seq, 8]

    # Routing weights: unbiased s, normalized. RSF applied later in GEMM2.
    weight_mask  = torch.zeros_like(s)
    weight_mask.scatter_(1, top8_ids, 1.0)
    weights      = s * weight_mask
    weights_sum  = weights.sum(-1, keepdim=True).clamp(min=1e-20)
    weights_full = weights / weights_sum                  # [seq, 256]

    return top8_ids.to(torch.int32), weights_full


# ---------------------------------------------------------------------------
# Token sorting
#
# Flattens all (token, topk_slot) assignments, keeps only local experts
# [local_offset, local_offset+31], sorts by local expert id, and computes
# cumulative per-expert offsets.
#
# Returns:
#   sorted_tok_ids  [N_assigned] int32   global token row indices, sorted by expert
#   expert_offsets  [33]         int32   expert_offsets[e] = start idx for expert e
#   sorted_r_scores [N_assigned] float32 normalized routing weight (no RSF)
#   N_assigned      int
#   max_tok         int          max tokens assigned to any single local expert
# ---------------------------------------------------------------------------
def _sort_tokens(expert_ids, weights_full, local_offset, device):
    local_mask = ((expert_ids >= local_offset) &
                  (expert_ids <  local_offset + NUM_LOCAL_EXPERTS))
    pairs = local_mask.nonzero(as_tuple=False)            # [N_assigned, 2]

    if pairs.shape[0] == 0:
        empty   = torch.zeros(0, dtype=torch.int32, device=device)
        offsets = torch.zeros(NUM_LOCAL_EXPERTS + 1, dtype=torch.int32, device=device)
        return empty, offsets, empty, 0, 0

    tok_global = pairs[:, 0]                              # int64 global token row
    slot       = pairs[:, 1]
    global_exp = expert_ids[tok_global, slot]             # int32 global expert id
    local_exp  = (global_exp - local_offset).to(torch.int32)
    r_scores   = weights_full[tok_global, global_exp.long()].to(torch.float32)

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
# Kernel 1: Grouped GEMM1 with fused SwiGLU epilogue
#
# For every sorted position m and output column c ∈ [0, INTER_DIM):
#   up   [m, c] = hidden[tok_ids[m]] @ w1[e, c,            :].T
#   gate [m, c] = hidden[tok_ids[m]] @ w1[e, INTER_DIM + c, :].T
#   inter[m, c] = silu(gate[m, c]) * up[m, c]
#
# The grid tiles over INTER_DIM columns (not GATE_UP_DIM), so each tile
# computes a BLOCK_N-wide slice of inter directly. Per inner K iteration we
# run two FP8 MMAs sharing the hidden load — one against the "up" weight
# rows [0, INTER_DIM), one against the "gate" weight rows [INTER_DIM, 2*INTER_DIM).
# Both 128-K-block scale drains happen in-register; SwiGLU is applied once
# at the end and the result is stored to inter_buf.
#
# Grid: (NUM_LOCAL_EXPERTS * ceil(max_tok / BLOCK_M),  ceil(INTER_DIM / BLOCK_N))
# ---------------------------------------------------------------------------
@triton.autotune(configs=_gemm_configs(), key=["n_assigned_bucket", "N"])
@triton.jit
def grouped_gemm1_kernel(
    # Sorted token data
    sorted_tok_ids_ptr,    # [N_assigned] int32
    expert_offsets_ptr,    # [33]         int32
    max_tok,               # int

    # Hidden states (full sequence, gather via tok_ids)
    hidden_ptr,            # [seq_len, 7168]   fp8_e4m3fn
    hscale_ptr,            # [56, seq_len]     float32

    # Per-expert weights
    w1_ptr,                # [32, 4096, 7168]  fp8_e4m3fn
    w1scale_ptr,           # [32, 32, 56]      float32

    # Output (fused SwiGLU target)
    inter_ptr,             # [N_assigned, 2048] float32
    N_assigned,
    n_assigned_bucket,

    # Shape constants
    K: tl.constexpr,       # 7168
    N: tl.constexpr,       # 2048 (= INTER_DIM)

    # Strides
    stride_h_seq,
    stride_w1_exp,
    stride_w1_n,
    stride_w1s_exp,
    stride_w1s_nb,
    stride_hscale_blk,
    stride_inter_tok,

    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,  # always 128
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

    tok_ids = tl.load(sorted_tok_ids_ptr + global_m, mask=m_mask, other=0)

    n_start      = pid_n * BLOCK_N
    n_range      = n_start + tl.arange(0, BLOCK_N)                  # "up"   cols in [0, INTER_DIM)
    n_mask       = n_range < N
    n_range_gate = n_range + N                                       # "gate" rows in [INTER_DIM, 2*INTER_DIM)

    n_blks_up   = n_range      // BLOCK_K                            # weight-scale n-block indices
    n_blks_gate = n_range_gate // BLOCK_K

    w1_base  = w1_ptr      + expert_id * stride_w1_exp
    w1s_base = w1scale_ptr + expert_id * stride_w1s_exp

    acc_up   = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    acc_gate = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k_start in tl.range(0, K, BLOCK_K):
        k_blk   = k_start // BLOCK_K
        k_range = k_start + tl.arange(0, BLOCK_K)
        k_mask  = k_range < K

        # FP8 hidden  [BM, BK]  — shared between the up-MMA and the gate-MMA
        h_ptrs  = hidden_ptr + tok_ids[:, None] * stride_h_seq + k_range[None, :]
        h_fp8   = tl.load(h_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        h_scales = tl.load(
            hscale_ptr + k_blk * stride_hscale_blk + tok_ids,
            mask=m_mask, other=1.0,
        )

        # FP8 weight "up" half   [BN, BK]
        w_up_ptrs  = w1_base + n_range[:, None] * stride_w1_n + k_range[None, :]
        w_up_fp8   = tl.load(w_up_ptrs, mask=n_mask[:, None] & k_mask[None, :], other=0.0)
        w_up_scales = tl.load(
            w1s_base + n_blks_up * stride_w1s_nb + k_blk,
            mask=n_mask, other=1.0,
        )

        # FP8 weight "gate" half [BN, BK]
        w_gate_ptrs  = w1_base + n_range_gate[:, None] * stride_w1_n + k_range[None, :]
        w_gate_fp8   = tl.load(w_gate_ptrs, mask=n_mask[:, None] & k_mask[None, :], other=0.0)
        w_gate_scales = tl.load(
            w1s_base + n_blks_gate * stride_w1s_nb + k_blk,
            mask=n_mask, other=1.0,
        )

        # Two native FP8 MMAs sharing h_fp8; fp32 TC accumulators.
        # DeepSeek-V3 N_C=128 two-level promotion: drain each per 128-K block.
        acc_up_block   = tl.dot(h_fp8, tl.trans(w_up_fp8),   out_dtype=tl.float32)
        acc_gate_block = tl.dot(h_fp8, tl.trans(w_gate_fp8), out_dtype=tl.float32)

        acc_up   += acc_up_block   * h_scales[:, None] * w_up_scales[None, :]
        acc_gate += acc_gate_block * h_scales[:, None] * w_gate_scales[None, :]

    # Fused SwiGLU epilogue: silu(gate) * up, where silu(x) = x * sigmoid(x).
    inter_tile = (acc_gate / (1.0 + tl.exp(-acc_gate))) * acc_up

    out_m    = e_start + m_range
    out_ptrs = inter_ptr + out_m[:, None] * stride_inter_tok + n_range[None, :]
    tl.store(out_ptrs, inter_tile, mask=m_mask[:, None] & n_mask[None, :])


# ---------------------------------------------------------------------------
# Kernel 2: Grouped GEMM2
#
# output_f32[tok_ids[m]] += (r_scores[m] * rsf) × inter[m] @ w2[expert].T
# for all 32 local experts in a single launch.
#
# Uses tl.atomic_add scatter — required because top-8 routing means
# multiple experts contribute to the same output token row.
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=_gemm_configs(),
    key=["n_assigned_bucket", "N"],
    reset_to_zero=("output_ptr",),
)
@triton.jit
def grouped_gemm2_kernel(
    # Sorted token data
    sorted_tok_ids_ptr,
    sorted_r_scores_ptr,
    expert_offsets_ptr,
    max_tok,

    # Intermediate (SwiGLU output)
    inter_ptr,             # [N_assigned, 2048] float32

    # Per-expert weights
    w2_ptr,                # [32, 7168, 2048]  fp8_e4m3fn
    w2scale_ptr,           # [32, 56, 16]      float32

    # Output
    output_ptr,            # [seq_len, 7168]   float32  (zero-init, atomic add)
    routed_scaling_factor,

    N_assigned,
    n_assigned_bucket,

    # Shape constants
    K: tl.constexpr,       # 2048
    N: tl.constexpr,       # 7168

    # Strides
    stride_inter_tok,
    stride_w2_exp,
    stride_w2_n,
    stride_w2s_exp,
    stride_w2s_nb,
    stride_out_seq,

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

    tok_ids  = tl.load(sorted_tok_ids_ptr  + global_m, mask=m_mask, other=0)
    r_scores = tl.load(sorted_r_scores_ptr + global_m, mask=m_mask, other=0.0)
    weight   = r_scores * routed_scaling_factor

    n_start  = pid_n * BLOCK_N
    n_range  = n_start + tl.arange(0, BLOCK_N)
    n_mask   = n_range < N
    n_blks   = n_range // BLOCK_K

    w2_base  = w2_ptr      + expert_id * stride_w2_exp
    w2s_base = w2scale_ptr + expert_id * stride_w2s_exp

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k_start in tl.range(0, K, BLOCK_K):
        k_blk   = k_start // BLOCK_K
        k_range = k_start + tl.arange(0, BLOCK_K)
        k_mask  = k_range < K

        # Load intermediate (float32) — no BF16 cast, matches baseline precision
        i_ptrs = inter_ptr + global_m[:, None] * stride_inter_tok + k_range[None, :]
        i_f32  = tl.load(i_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)

        # FP8 weight  [BN, BK]
        w_ptrs  = w2_base + n_range[:, None] * stride_w2_n + k_range[None, :]
        w_fp8   = tl.load(w_ptrs, mask=n_mask[:, None] & k_mask[None, :], other=0.0)
        w_scales = tl.load(
            w2s_base + n_blks * stride_w2s_nb + k_blk,
            mask=n_mask, other=1.0,
        )

        # Dequant to float32, then dot — no BF16 intermediate, matches baseline precision
        w_f32  = w_fp8.to(tl.float32) * w_scales[:, None]  # [BN, BK]
        acc   += tl.dot(i_f32, tl.trans(w_f32), out_dtype=tl.float32)

    # Apply routing weight and scatter-add (multiple experts → same output token row)
    acc      = acc * weight[:, None]
    out_ptrs = output_ptr + tok_ids[:, None] * stride_out_seq + n_range[None, :]
    tl.atomic_add(out_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


# ---------------------------------------------------------------------------
# Entry point
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

    # 1. Routing
    expert_ids, weights_full = _route(
        routing_logits, routing_bias, seq_len, device
    )

    # 2. Token sorting
    sorted_tok_ids, expert_offsets, sorted_r_scores, N_assigned, max_tok = _sort_tokens(
        expert_ids, weights_full, local_offset, device
    )

    # 3. Allocate output accumulator; handle empty case
    output_f32 = torch.zeros(seq_len, HIDDEN_DIM, dtype=torch.float32, device=device)

    if N_assigned == 0:
        output.zero_()
        return

    inter_buf = torch.empty(N_assigned, INTER_DIM, dtype=torch.float32, device=device)

    # 4. Grouped GEMM1 with fused SwiGLU — writes inter_buf directly
    grouped_gemm1_kernel[
        lambda meta: (
            NUM_LOCAL_EXPERTS * triton.cdiv(max_tok, meta["BLOCK_M"]),
            triton.cdiv(INTER_DIM, meta["BLOCK_N"]),
        )
    ](
        sorted_tok_ids, expert_offsets, max_tok,
        hidden_states, hidden_states_scale,
        gemm1_weights, gemm1_weights_scale,
        inter_buf,
        N_assigned,
        _bucket(N_assigned),
        K=HIDDEN_DIM,
        N=INTER_DIM,
        stride_h_seq      = hidden_states.stride(0),
        stride_w1_exp     = gemm1_weights.stride(0),
        stride_w1_n       = gemm1_weights.stride(1),
        stride_w1s_exp    = gemm1_weights_scale.stride(0),
        stride_w1s_nb     = gemm1_weights_scale.stride(1),
        stride_hscale_blk = hidden_states_scale.stride(0),
        stride_inter_tok  = inter_buf.stride(0),
    )

    # 5. Grouped GEMM2 — atomic scatter-add to output_f32
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

    # 6. Cast float32 accumulation → bf16 output (DPS)
    output.copy_(output_f32.to(torch.bfloat16))
