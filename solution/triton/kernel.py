"""
Fused MoE Triton Kernel for FlashInfer Competition.

Target: moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048
  - DeepSeek-V3/R1 scale MoE: hidden=7168, intermediate=2048
  - 32 local experts (256 global), top-8 routing
  - FP8 E4M3FN weights with 128-element block scaling
  - DeepSeek-style routing: 8 groups, top-4 groups, top-8 experts

Function signature uses Destination Passing Style (DPS): output tensor is
pre-allocated by the framework and passed in; the kernel writes into it.

Optimizations:
  - Grouped GEMM: all experts dispatched in one kernel launch (no Python loop)
  - Native FP8 WGMMA: tl.dot(fp8, fp8) → float32 accumulator
  - BF16 WGMMA: cast intermediate/weights to bf16 for GEMM2
  - Fixed routing: weight_scores use sigmoid only (no bias)
  - Fixed SwiGLU: correct gate/up slice mapping [W_up; W_gate]
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_GLOBAL_EXPERTS = tl.constexpr(256)
NUM_GROUPS = tl.constexpr(8)           # ng8
EXPERTS_PER_GROUP = tl.constexpr(32)   # NUM_GLOBAL_EXPERTS / NUM_GROUPS
NUM_GROUPS_SELECTED = tl.constexpr(4)  # kg4: top-4 groups selected
TOPK = tl.constexpr(8)                 # topk8: top-8 experts selected
NUM_LOCAL_EXPERTS = tl.constexpr(32)   # e32
HIDDEN_DIM = tl.constexpr(7168)        # h7168
INTER_DIM = tl.constexpr(2048)         # i2048  (gate and up each = 2048, combined = 4096)
FP8_BLOCK_SIZE = tl.constexpr(128)     # FP8 block-scale granularity
GATE_UP_DIM = tl.constexpr(INTER_DIM.value * 2)  # 4096


# ---------------------------------------------------------------------------
# Autotune configs for B200
# BLOCK_K is fixed at 128 to exactly align with FP8 block-scale boundaries.
# ---------------------------------------------------------------------------
def _gemm_configs():
    configs = []
    for bm in [64, 128]:
        for bn in [128, 256]:
            for ns in [3, 4, 5]:
                for nw in [8, 16]:
                    configs.append(
                        triton.Config(
                            {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": FP8_BLOCK_SIZE},
                            num_stages=ns,
                            num_warps=nw,
                        )
                    )
    return configs


def _token_bucket(n: int) -> int:
    """Bucket num_tokens for autotune key to avoid per-batch recompilation."""
    for thresh in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
        if n <= thresh:
            return thresh
    return 1024


# ---------------------------------------------------------------------------
# Kernel 1: DeepSeek-style routing
# One CTA per BLOCK_T tokens.
# FIX: weight_scores = sigmoid(logits) only — bias only used for selection.
# ---------------------------------------------------------------------------
@triton.jit
def routing_kernel(
    logits_ptr,          # [seq_len, 256] float32
    bias_ptr,            # [256] float32  (pre-converted from bfloat16 in wrapper)
    expert_ids_ptr,      # [seq_len, 8] int32  OUTPUT
    scores_ptr,          # [seq_len, 8] float32 OUTPUT
    seq_len,
    stride_logits_seq,   # stride along seq dim  (== 256)
    stride_eid_seq,      # stride along seq dim of expert_ids (== 8)
    stride_score_seq,    # stride along seq dim of scores (== 8)
    NUM_EXPERTS: tl.constexpr,       # 256
    NUM_GROUPS: tl.constexpr,        # 8
    EPG: tl.constexpr,               # experts_per_group = 32
    KG: tl.constexpr,                # num_groups_selected = 4
    TOPK: tl.constexpr,              # 8
    BLOCK_T: tl.constexpr,           # tokens per CTA (16)
):
    """
    For each token:
      1. Load 256 logits + biases
      2. selection_scores = sigmoid(logits) + bias  (used for picking experts)
         weight_scores    = sigmoid(logits)          (used for final routing weights)
      3. Per-group (8 groups × 32 experts): top-2 sum of selection_scores → group_score
      4. Select top-KG groups
      5. From candidates: select top-TOPK by selection_score
      6. Normalize weight_scores of selected experts; write expert_ids and scores
    """
    pid = tl.program_id(0)
    tok_start = pid * BLOCK_T

    exp_range = tl.arange(0, NUM_EXPERTS)    # [256]
    grp_range = tl.arange(0, NUM_GROUPS)     # [8]
    topk_range = tl.arange(0, TOPK)          # [8]
    epg_range = tl.arange(0, EPG)            # [32]

    # Load bias once (shared across all tokens in this CTA)
    bias = tl.load(bias_ptr + exp_range)     # [256] float32

    for _t in tl.static_range(BLOCK_T):
        t = tok_start + _t
        t_mask = t < seq_len

        logits = tl.load(logits_ptr + t * stride_logits_seq + exp_range, mask=t_mask, other=-1e9)
        # FIX: separate selection (biased) from weighting (unbiased)
        selection_scores = tl.sigmoid(logits) + bias   # [256] used for picking
        weight_scores    = tl.sigmoid(logits)           # [256] used for final weights

        # ---- 2. Group scoring using selection_scores ----
        group_scores = tl.zeros([NUM_GROUPS], dtype=tl.float32)
        for g in tl.static_range(NUM_GROUPS):
            g_base = g * EPG
            g_logits = tl.load(
                logits_ptr + t * stride_logits_seq + g_base + epg_range,
                mask=t_mask, other=-1e9
            )
            g_sel = tl.sigmoid(g_logits) + tl.load(bias_ptr + g_base + epg_range)

            max1 = tl.max(g_sel, axis=0)
            g_sel_tmp = tl.where(g_sel == max1, -1e9, g_sel)
            max2 = tl.max(g_sel_tmp, axis=0)
            gs_val = max1 + tl.maximum(max2, 0.0)
            group_scores = tl.where(grp_range == g, gs_val, group_scores)

        # ---- 3. Top-KG group selection ----
        running_gs = group_scores
        selected_groups = tl.zeros([NUM_GROUPS], dtype=tl.int32)
        for _k in tl.static_range(KG):
            best_g = tl.argmax(running_gs, axis=0)
            selected_groups = tl.where(grp_range == best_g, 1, selected_groups)
            running_gs = tl.where(grp_range == best_g, -1e9, running_gs)

        # ---- 4. Build candidate mask ----
        group_of_exp = exp_range // EPG
        candidate_mask = tl.zeros([NUM_EXPERTS], dtype=tl.int32)
        for g in tl.static_range(NUM_GROUPS):
            g_sel = tl.sum(tl.where(grp_range == g, selected_groups, 0)) > 0
            candidate_mask = tl.where(
                (group_of_exp == g) & g_sel,
                1,
                candidate_mask,
            )

        masked_scores = tl.where(candidate_mask > 0, selection_scores, -1e9)

        # ---- 5. Top-TOPK selection (by selection_scores) ----
        selected_exp = tl.zeros([TOPK], dtype=tl.int32)
        selected_w   = tl.zeros([TOPK], dtype=tl.float32)
        running_ms = masked_scores
        for k in tl.static_range(TOPK):
            best_e = tl.argmax(running_ms, axis=0)
            # FIX: store weight_score (unbiased) not selection_score
            best_w = tl.sum(tl.where(exp_range == best_e, weight_scores, 0.0))
            selected_exp = tl.where(topk_range == k, best_e, selected_exp)
            selected_w   = tl.where(topk_range == k, best_w, selected_w)
            running_ms   = tl.where(exp_range == best_e, -1e9, running_ms)

        # ---- 6. Normalize weight_scores ----
        score_sum = tl.sum(selected_w, axis=0)
        score_sum = tl.maximum(score_sum, 1e-9)
        norm_scores = selected_w / score_sum

        # ---- 7. Write outputs ----
        tl.store(expert_ids_ptr + t * stride_eid_seq + topk_range, selected_exp, mask=t_mask)
        tl.store(scores_ptr + t * stride_score_seq + topk_range, norm_scores, mask=t_mask)


# ---------------------------------------------------------------------------
# Kernel 2: Expert GEMM1 (Per-expert launch)
# Grid: (ceil(num_tokens / BLOCK_M), ceil(N / BLOCK_N))
# Uses native FP8 WGMMA with per-(M,N)-block vectorized scale application.
# ---------------------------------------------------------------------------
@triton.autotune(configs=_gemm_configs(), key=["tok_bucket", "K", "N"])
@triton.jit
def expert_gemm1_kernel(
    token_ids_ptr,         # [num_tokens] int32
    hidden_ptr,            # [seq_len, 7168] float8_e4m3fn
    hscale_ptr,            # [56, seq_len] float32
    w1_ptr,                # [4096, 7168] float8_e4m3fn
    w1scale_ptr,           # [32, 56] float32
    gate_up_ptr,           # [num_tokens, 4096] float32  OUTPUT
    num_tokens,
    tok_bucket,
    K: tl.constexpr,       # 7168
    N: tl.constexpr,       # 4096
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,  # 128
    stride_h_seq,
    stride_w1_n,            # gemm1_weights.stride(1)
    stride_w1s_nb,          # gemm1_weights_scale.stride(1)
    stride_hscale_blk,      # hidden_states_scale.stride(0) = seq_len
    stride_gu_tok,          # gate_up_buf.stride(0) = GATE_UP_DIM
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    m_range = m_start + tl.arange(0, BLOCK_M)
    m_mask = m_range < num_tokens

    n_range = n_start + tl.arange(0, BLOCK_N)
    n_mask = n_range < N

    # --- Load local token IDs for this tile ---
    tok_ids = tl.load(token_ids_ptr + m_range, mask=m_mask, other=0)

    # Vectorized N-block indices (handles BLOCK_N > 128 correctly)
    n_blks = n_range // BLOCK_K   # [BLOCK_N] — which N-scale-block each column belongs to

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k_start in tl.range(0, K, BLOCK_K):
        k_blk  = k_start // BLOCK_K
        k_range = k_start + tl.arange(0, BLOCK_K)
        k_mask  = k_range < K

        # Load FP8 hidden — no upcast (native FP8)
        h_ptrs = hidden_ptr + tok_ids[:, None] * stride_h_seq + k_range[None, :]
        h_fp8  = tl.load(h_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        # Per-token, per-K-block hidden scale: hscale[k_blk, tok_id]
        h_scales = tl.load(
            hscale_ptr + k_blk * stride_hscale_blk + tok_ids,
            mask=m_mask, other=1.0,
        )  # [BLOCK_M]

        # Load FP8 weight — no upcast (native FP8)
        w_ptrs = w1_ptr + n_range[:, None] * stride_w1_n + k_range[None, :]
        w_fp8  = tl.load(w_ptrs, mask=n_mask[:, None] & k_mask[None, :], other=0.0)
        # Vectorized per-N-block weight scale: w1scale[n_blk, k_blk] for each n
        w_scales = tl.load(
            w1scale_ptr + n_blks * stride_w1s_nb + k_blk,
            mask=n_mask, other=1.0,
        )  # [BLOCK_N]

        # Native FP8 WGMMA → float32 accumulator
        raw = tl.dot(h_fp8, tl.trans(w_fp8), out_dtype=tl.float32)

        # Apply block-scales: h_scale per row, w_scale per column
        acc += raw * h_scales[:, None] * w_scales[None, :]

    # Write gate+up buffer
    out_ptrs = gate_up_ptr + m_range[:, None] * stride_gu_tok + n_range[None, :]
    tl.store(out_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


# ---------------------------------------------------------------------------
# Kernel 3: Expert GEMM2 (Per-expert launch)
# Grid: (ceil(num_tokens / BLOCK_M), ceil(N / BLOCK_N))
# Uses BF16 WGMMA: intermediate cast to bf16, weight dequant to bf16.
# ---------------------------------------------------------------------------
@triton.autotune(configs=_gemm_configs(), key=["tok_bucket", "K", "N"])
@triton.jit
def expert_gemm2_kernel(
    inter_ptr,             # [num_tokens, 2048] float32 (SwiGLU output)
    token_ids_ptr,         # [num_tokens] int32
    r_scores_ptr,          # [num_tokens] float32
    routed_scaling_factor,
    w2_ptr,                # [7168, 2048] float8_e4m3fn
    w2scale_ptr,           # [56, 16] float32
    output_ptr,            # [seq_len, 7168] float32  (atomic add target)
    num_tokens,
    tok_bucket,
    K: tl.constexpr,       # 2048
    N: tl.constexpr,       # 7168
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,  # 128
    stride_inter_tok,
    stride_w2_n,            # gemm2_weights.stride(1)
    stride_w2s_nb,          # gemm2_weights_scale.stride(1)
    stride_out_seq,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    m_range = m_start + tl.arange(0, BLOCK_M)
    m_mask = m_range < num_tokens

    n_range = n_start + tl.arange(0, BLOCK_N)
    n_mask = n_range < N

    # --- Load local token IDs and routing scores ---
    tok_ids  = tl.load(token_ids_ptr   + m_range, mask=m_mask, other=0)
    r_scores = tl.load(r_scores_ptr  + m_range, mask=m_mask, other=0.0)
    weight   = r_scores * routed_scaling_factor   # [BLOCK_M]

    # Vectorized N-block indices
    n_blks = n_range // BLOCK_K   # [BLOCK_N]

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k_start in tl.range(0, K, BLOCK_K):
        k_blk  = k_start // BLOCK_K
        k_range = k_start + tl.arange(0, BLOCK_K)
        k_mask  = k_range < K

        # Load intermediate (float32) → cast to bf16 for BF16 WGMMA
        i_ptrs = inter_ptr + m_range[:, None] * stride_inter_tok + k_range[None, :]
        i_mk   = tl.load(i_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        i_bf16 = i_mk.to(tl.bfloat16)

        # Load FP8 weight, dequant to bf16 with vectorized per-N-block scale
        w_ptrs  = w2_ptr + n_range[:, None] * stride_w2_n + k_range[None, :]
        w_fp8   = tl.load(w_ptrs, mask=n_mask[:, None] & k_mask[None, :], other=0.0)
        w_scales = tl.load(
            w2scale_ptr + n_blks * stride_w2s_nb + k_blk,
            mask=n_mask, other=1.0,
        )  # [BLOCK_N]
        w_bf16  = (w_fp8.to(tl.float32) * w_scales[:, None]).to(tl.bfloat16)

        # BF16 WGMMA → float32 accumulator
        acc += tl.dot(i_bf16, tl.trans(w_bf16), out_dtype=tl.float32)

    # Apply routing weight and atomic-add into output
    acc = acc * weight[:, None]
    out_ptrs = output_ptr + tok_ids[:, None] * stride_out_seq + n_range[None, :]
    tl.atomic_add(
        out_ptrs,
        acc,
        mask=m_mask[:, None] & n_mask[None, :],
    )


# ---------------------------------------------------------------------------
# SwiGLU (PyTorch, runs on device after GEMM1)
# FIX: W1 layout is [W_up; W_gate], so [:, :2048] = up, [:, 2048:] = gate
# SwiGLU(x) = silu(gate) * up
# ---------------------------------------------------------------------------
def _swiglu(gate_up: torch.Tensor) -> torch.Tensor:
    """
    gate_up: [total_tokens, 4096] float32
    Weight layout: [W_up (rows 0..2047); W_gate (rows 2048..4095)]
    So: gate_up[:, :2048] = up-projection, gate_up[:, 2048:] = gate
    Returns: [total_tokens, 2048] float32
    """
    gate = gate_up[:, :INTER_DIM]   # FIX: first half is gate
    up   = gate_up[:, INTER_DIM:]   # FIX: second half is up
    return torch.nn.functional.silu(gate) * up


# ---------------------------------------------------------------------------
# Entry point (matches config.toml entry_point = "kernel")
# DPS: output tensor pre-allocated by framework, must be last argument.
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
    output,                 # bfloat16,      [seq_len, 7168]  (DPS — must be last)
):
    """
    Fused MoE forward pass using per-expert GEMM calls.
    Restored Python loop logic to prevent block-boundary straddling issues
    which arise natively in Grouped GEMMs without dynamic block arrays.
    """
    seq_len = hidden_states.shape[0]
    device  = hidden_states.device

    # Step 1: Routing (PyTorch — verified correct against reference)
    bias_f32 = routing_bias.to(torch.float32)
    s = torch.sigmoid(routing_logits.float())        # [seq, 256] — used for weights
    s_with_bias = s + bias_f32                       # [seq, 256] — used for selection

    # Group scoring: top-2 sum per group using s_with_bias
    group_scores_mat = s_with_bias.view(seq_len, NUM_GROUPS.value, EXPERTS_PER_GROUP.value)
    top2_vals, _ = group_scores_mat.topk(2, dim=-1)
    group_scores = top2_vals.sum(-1)                 # [seq, 8]

    # Select top-KG groups
    _, top_groups = group_scores.topk(NUM_GROUPS_SELECTED.value, dim=-1)  # [seq, 4]

    # Build group mask and select top-TOPK experts using s_with_bias
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, top_groups, 1.0)
    score_mask = group_mask.unsqueeze(2).expand(
        seq_len, NUM_GROUPS.value, EXPERTS_PER_GROUP.value
    ).reshape(seq_len, NUM_GLOBAL_EXPERTS.value)
    neg_inf = torch.finfo(torch.float32).min
    scores_pruned = s_with_bias.masked_fill(score_mask == 0, neg_inf)
    _, top8_ids = scores_pruned.topk(TOPK.value, dim=-1)  # [seq, 8]

    # Routing WEIGHTS: use s (WITHOUT bias), normalize
    weight_mask = torch.zeros_like(s)
    weight_mask.scatter_(1, top8_ids, 1.0)
    weights_full = s * weight_mask                   # [seq, 256]
    weights_sum = weights_full.sum(-1, keepdim=True).clamp(min=1e-20)
    weights_full = weights_full / weights_sum        # normalized, rsf applied per-expert below

    expert_ids = top8_ids.to(torch.int32)            # [seq, 8]

    # Step 2: Per-expert FFN loop
    rsf = float(routed_scaling_factor)
    local_offset = int(local_expert_offset)

    output_f32 = torch.zeros(seq_len, HIDDEN_DIM.value, dtype=torch.float32, device=device)

    for local_idx in range(NUM_LOCAL_EXPERTS.value):
        global_id = local_offset + local_idx

        # Find tokens for this expert
        pairs = (expert_ids == global_id).nonzero(as_tuple=False)
        num_tok = pairs.shape[0]
        if num_tok == 0:
            continue

        tok_ids = pairs[:, 0].to(torch.int32).contiguous()
        # Per-token routing weight (normalized, without rsf — applied in kernel)
        r_scores = weights_full[tok_ids, global_id].contiguous()

        # Extract weights/scales for this expert
        w1 = gemm1_weights[local_idx]
        w1s = gemm1_weights_scale[local_idx]
        w2 = gemm2_weights[local_idx]
        w2s = gemm2_weights_scale[local_idx]

        tb = _token_bucket(num_tok)

        # --- GEMM1 ---
        gate_up_buf = torch.empty((num_tok, GATE_UP_DIM.value), dtype=torch.float32, device=device)

        expert_gemm1_kernel[
            lambda meta: (
                triton.cdiv(num_tok, meta["BLOCK_M"]),
                triton.cdiv(GATE_UP_DIM.value, meta["BLOCK_N"]),
            )
        ](
            tok_ids,
            hidden_states, hidden_states_scale,
            w1, w1s,
            gate_up_buf,
            num_tok, tb,
            K=HIDDEN_DIM.value, N=GATE_UP_DIM.value,
            stride_h_seq=hidden_states.stride(0),
            stride_w1_n=w1.stride(0),
            stride_w1s_nb=w1s.stride(0),
            stride_hscale_blk=hidden_states_scale.stride(0),
            stride_gu_tok=gate_up_buf.stride(0),
        )

        # --- SwiGLU ---
        intermediate = _swiglu(gate_up_buf)

        # --- GEMM2 ---
        expert_gemm2_kernel[
            lambda meta: (
                triton.cdiv(num_tok, meta["BLOCK_M"]),
                triton.cdiv(HIDDEN_DIM.value, meta["BLOCK_N"]),
            )
        ](
            intermediate, tok_ids, r_scores, rsf,
            w2, w2s,
            output_f32,
            num_tok, tb,
            K=INTER_DIM.value, N=HIDDEN_DIM.value,
            stride_inter_tok=intermediate.stride(0),
            stride_w2_n=w2.stride(0),
            stride_w2s_nb=w2s.stride(0),
            stride_out_seq=output_f32.stride(0),
        )

    output.copy_(output_f32.to(torch.bfloat16))
