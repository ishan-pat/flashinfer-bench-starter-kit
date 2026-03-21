"""
Fused MoE Triton Kernel for FlashInfer Competition.

Target: moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048
  - DeepSeek-V3/R1 scale MoE: hidden=7168, intermediate=2048
  - 32 local experts (256 global), top-8 routing
  - FP8 E4M3FN weights with 128-element block scaling
  - DeepSeek-style routing: 8 groups, top-4 groups, top-8 experts

Function signature uses Destination Passing Style (DPS): output tensor is
pre-allocated by the framework and passed in; the kernel writes into it.
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_GLOBAL_EXPERTS = 256
NUM_GROUPS = 8           # ng8
EXPERTS_PER_GROUP = 32   # NUM_GLOBAL_EXPERTS / NUM_GROUPS
NUM_GROUPS_SELECTED = 4  # kg4: top-4 groups selected
TOPK = 8                 # topk8: top-8 experts selected
NUM_LOCAL_EXPERTS = 32   # e32
HIDDEN_DIM = 7168        # h7168
INTER_DIM = 2048         # i2048  (gate and up each = 2048, combined = 4096)
FP8_BLOCK_SIZE = 128     # FP8 block-scale granularity
GATE_UP_DIM = INTER_DIM * 2  # 4096


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
      1. Load 256 logits + biases → sigmoid + bias → expert_scores [256]
      2. Per-group (8 groups × 32 experts): top-2 sum → group_score [8]
      3. Select top-KG groups
      4. From 128 candidates: select top-TOPK by expert_score
      5. Normalize selected scores; write expert_ids and scores
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
        mask = t < seq_len

        # ---- 1. sigmoid(logits) + bias ----
        logits = tl.load(logits_ptr + t * stride_logits_seq + exp_range, mask=mask, other=-1e9)  # [256]
        expert_scores = tl.sigmoid(logits) + bias                          # [256]

        # ---- 2. Group scoring: per group, sum top-2 expert scores ----
        # We compute group_scores[g] = top1_score + top2_score for experts in group g
        group_scores = tl.zeros([NUM_GROUPS], dtype=tl.float32)
        for g in tl.static_range(NUM_GROUPS):
            g_base = g * EPG
            g_scores = tl.load(
                logits_ptr + t * stride_logits_seq + g_base + epg_range,
                mask=mask, other=-1e9
            )  # [32]
            g_scores = tl.sigmoid(g_scores) + tl.load(bias_ptr + g_base + epg_range)

            # top-2 sum: two argmax passes
            max1 = tl.max(g_scores, axis=0)
            g_scores_tmp = tl.where(g_scores == max1, -1e9, g_scores)
            max2 = tl.max(g_scores_tmp, axis=0)
            gs_val = max1 + tl.maximum(max2, 0.0)
            group_scores = tl.where(grp_range == g, gs_val, group_scores)

        # ---- 3. Top-KG group selection ----
        running_gs = group_scores
        selected_groups = tl.zeros([NUM_GROUPS], dtype=tl.int32)
        for _k in tl.static_range(KG):
            best_g = tl.argmax(running_gs, axis=0)
            selected_groups = tl.where(grp_range == best_g, 1, selected_groups)
            running_gs = tl.where(grp_range == best_g, -1e9, running_gs)

        # ---- 4. Build candidate mask over all 256 experts ----
        # expert e is candidate iff its group (e // EPG) is selected
        group_of_exp = exp_range // EPG  # [256]  group index for each expert
        candidate_mask = tl.zeros([NUM_EXPERTS], dtype=tl.int32)
        for g in tl.static_range(NUM_GROUPS):
            g_sel = tl.sum(tl.where(grp_range == g, selected_groups, 0)) > 0
            candidate_mask = tl.where(
                group_of_exp == g,
                tl.where(g_sel, 1, candidate_mask),
                candidate_mask,
            )

        # Mask non-candidates
        masked_scores = tl.where(candidate_mask > 0, expert_scores, -1e9)

        # ---- 5. Top-TOPK from 128 candidates ----
        selected_exp = tl.zeros([TOPK], dtype=tl.int32)
        selected_s = tl.zeros([TOPK], dtype=tl.float32)
        running_ms = masked_scores
        for k in tl.static_range(TOPK):
            best_e = tl.argmax(running_ms, axis=0)
            best_s = tl.max(running_ms, axis=0)
            selected_exp = tl.where(topk_range == k, best_e, selected_exp)
            selected_s = tl.where(topk_range == k, best_s, selected_s)
            running_ms = tl.where(exp_range == best_e, -1e9, running_ms)

        # ---- 6. Normalize ----
        score_sum = tl.sum(selected_s, axis=0)
        score_sum = tl.maximum(score_sum, 1e-9)
        norm_scores = selected_s / score_sum

        # ---- 7. Write outputs ----
        tl.store(expert_ids_ptr + t * stride_eid_seq + topk_range, selected_exp, mask=mask)
        tl.store(scores_ptr + t * stride_score_seq + topk_range, norm_scores, mask=mask)


# ---------------------------------------------------------------------------
# Kernel 2: Expert GEMM1 — FP8 gather-GEMM into gate+up buffer
# Grid: (ceil(num_tokens/BLOCK_M), ceil(N/BLOCK_N))
# ---------------------------------------------------------------------------
@triton.autotune(configs=_gemm_configs(), key=["tok_bucket", "K", "N"])
@triton.jit
def expert_gemm1_kernel(
    hidden_ptr,           # [total_seq_len, 7168] float8_e4m3fn
    hscale_ptr,           # [56, total_seq_len]  float32  (56 = HIDDEN_DIM/128)
    token_ids_ptr,        # [num_tokens] int32
    w1_ptr,               # [4096, 7168] float8_e4m3fn  (this expert's weights)
    w1scale_ptr,          # [32, 56] float32  (32=4096/128, 56=7168/128)
    gate_up_ptr,          # [num_tokens, 4096] float32  OUTPUT
    num_tokens,
    tok_bucket,
    K: tl.constexpr,      # 7168
    N: tl.constexpr,      # 4096
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,  # 128
    stride_h_seq,
    stride_w1_n,
    stride_gu_tok,
    stride_hscale_blk,    # total_seq_len  (hscale is [56, seq_len])
    stride_w1s_nb,        # 56  (w1scale is [32, 56])
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    m_range = m_start + tl.arange(0, BLOCK_M)
    n_range = n_start + tl.arange(0, BLOCK_N)
    m_mask = m_range < num_tokens
    n_mask = n_range < N

    # Gather source token row indices
    tok_ids = tl.load(token_ids_ptr + m_range, mask=m_mask, other=0)

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    n_blk = n_range // BLOCK_K  # [BLOCK_N]

    for k_start in tl.range(0, K, BLOCK_K):
        k_blk = k_start // BLOCK_K
        k_range = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_range < K

        # Load FP8 hidden [BLOCK_M, BLOCK_K] and dequantize
        h_ptrs = hidden_ptr + tok_ids[:, None] * stride_h_seq + k_range[None, :]
        h_fp8 = tl.load(h_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        h_f32 = h_fp8.to(tl.float32)
        # Scale: hscale[k_blk, tok_id]  —  shape [56, seq_len], stride_hscale_blk = seq_len
        h_scales = tl.load(
            hscale_ptr + k_blk * stride_hscale_blk + tok_ids,
            mask=m_mask, other=1.0,
        )  # [BLOCK_M]
        h_f32 = h_f32 * h_scales[:, None]

        # Load FP8 weight [BLOCK_N, BLOCK_K] and dequantize
        w_ptrs = w1_ptr + n_range[:, None] * stride_w1_n + k_range[None, :]
        w_fp8 = tl.load(w_ptrs, mask=n_mask[:, None] & k_mask[None, :], other=0.0)
        w_f32 = w_fp8.to(tl.float32)
        # Scale: w1scale[n_blk, k_blk]
        w_scale = tl.load(w1scale_ptr + n_blk * stride_w1s_nb + k_blk, mask=n_mask, other=1.0) # [BLOCK_N]
        w_f32 = w_f32 * w_scale[:, None]

        acc += tl.dot(h_f32, tl.trans(w_f32))

    # Write gate+up buffer
    out_ptrs = gate_up_ptr + m_range[:, None] * stride_gu_tok + n_range[None, :]
    tl.store(out_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


# ---------------------------------------------------------------------------
# Kernel 3: Expert GEMM2 — FP8 GEMM + weighted atomic-add into output
# Grid: (ceil(num_tokens/BLOCK_M), ceil(N/BLOCK_N))
# ---------------------------------------------------------------------------
@triton.autotune(configs=_gemm_configs(), key=["tok_bucket", "K", "N"])
@triton.jit
def expert_gemm2_accumulate_kernel(
    inter_ptr,            # [num_tokens, 2048] float32 (SwiGLU output)
    w2_ptr,               # [7168, 2048] float8_e4m3fn
    w2scale_ptr,          # [56, 16] float32  (56=7168/128, 16=2048/128)
    token_ids_ptr,        # [num_tokens] int32
    r_scores_ptr,         # [num_tokens] float32
    routed_scaling_factor,
    output_ptr,           # [total_seq_len, 7168] bfloat16  (atomic add target)
    num_tokens,
    tok_bucket,
    K: tl.constexpr,      # 2048
    N: tl.constexpr,      # 7168
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,  # 128
    stride_inter_tok,
    stride_w2_n,
    stride_out_seq,
    stride_w2s_nb,        # 16  (w2scale is [56, 16])
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    m_range = m_start + tl.arange(0, BLOCK_M)
    n_range = n_start + tl.arange(0, BLOCK_N)
    m_mask = m_range < num_tokens
    n_mask = n_range < N

    tok_ids = tl.load(token_ids_ptr + m_range, mask=m_mask, other=0)
    r_scores = tl.load(r_scores_ptr + m_range, mask=m_mask, other=0.0)
    weight = r_scores * routed_scaling_factor  # [BLOCK_M]

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    n_blk = n_range // BLOCK_K  # [BLOCK_N]

    for k_start in tl.range(0, K, BLOCK_K):
        k_blk = k_start // BLOCK_K
        k_range = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_range < K

        # Load intermediate (float32, no dequant needed)
        i_ptrs = inter_ptr + m_range[:, None] * stride_inter_tok + k_range[None, :]
        i_mk = tl.load(i_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)

        # Load FP8 weight and dequantize
        w_ptrs = w2_ptr + n_range[:, None] * stride_w2_n + k_range[None, :]
        w_fp8 = tl.load(w_ptrs, mask=n_mask[:, None] & k_mask[None, :], other=0.0)
        w_f32 = w_fp8.to(tl.float32)
        # Scale: w2scale[n_blk, k_blk]  (shape [56, 16])
        w_scale = tl.load(w2scale_ptr + n_blk * stride_w2s_nb + k_blk, mask=n_mask, other=1.0) # [BLOCK_N]
        w_f32 = w_f32 * w_scale[:, None]

        acc += tl.dot(i_mk, tl.trans(w_f32))

    # Apply routing weight
    acc = acc * weight[:, None]  # [BLOCK_M, BLOCK_N]

    # Atomic add into bfloat16 output
    out_ptrs = output_ptr + tok_ids[:, None] * stride_out_seq + n_range[None, :]
    tl.atomic_add(
        out_ptrs,
        acc.to(tl.bfloat16),
        mask=m_mask[:, None] & n_mask[None, :],
    )


# ---------------------------------------------------------------------------
# SwiGLU (PyTorch, not Triton — runs on device after GEMM1)
# ---------------------------------------------------------------------------
def _swiglu(gate_up: torch.Tensor) -> torch.Tensor:
    """
    gate_up: [num_tokens, 4096] float32
    returns: [num_tokens, 2048] float32
    SwiGLU(x) = silu(up) * gate  where gate=x[:2048], up=x[2048:]
    """
    X1 = gate_up[:, :INTER_DIM]
    X2 = gate_up[:, INTER_DIM:]
    return torch.nn.functional.silu(X2) * X1


# ---------------------------------------------------------------------------
# Entry point (matches config.toml entry_point = "kernel")
# ---------------------------------------------------------------------------
def kernel(
    routing_logits,         # float32,       [seq_len, 256]
    routing_bias,           # bfloat16,      [256]
    hidden_states,          # float8_e4m3fn, [seq_len, 7168]
    hidden_states_scale,    # float32,       [56, seq_len]   ← block_idx is dim 0
    gemm1_weights,          # float8_e4m3fn, [32, 4096, 7168]
    gemm1_weights_scale,    # float32,       [32, 32, 56]
    gemm2_weights,          # float8_e4m3fn, [32, 7168, 2048]
    gemm2_weights_scale,    # float32,       [32, 56, 16]
    local_expert_offset,    # int32 scalar
    routed_scaling_factor,  # float32 scalar
    output,                 # bfloat16,      [seq_len, 7168]  (DPS: write here)
):
    """
    Fused MoE forward pass.

    Only computes for local experts [local_expert_offset, local_expert_offset+32).
    Routing covers all 256 global experts; tokens that route to non-local experts
    are simply skipped (their contribution will be accumulated by other devices).
    """
    import traceback as _tb
    import sys as _sys
    import os as _os

    # === REDIRECT PRINT TO FILE ===
    def print(*args, **kwargs):
        with open("/tmp/kernel_debug.txt", "a") as f:
            f.write(" ".join(str(a) for a in args) + "\n")

    seq_len = hidden_states.shape[0]
    device = hidden_states.device

    # =========================================================================
    # LOG 1: Input validation — verify every argument's type, dtype, shape
    # WHY: Catches framework-level mismatches before any kernel runs.
    #      If the framework passes unexpected dtypes or shapes, we see it here.
    # =========================================================================
    print(f"[DEBUG-1] === KERNEL ENTRY ===", flush=True)
    print(f"[DEBUG-1] seq_len={seq_len}, device={device}", flush=True)
    all_args = {
        "routing_logits": routing_logits,
        "routing_bias": routing_bias,
        "hidden_states": hidden_states,
        "hidden_states_scale": hidden_states_scale,
        "gemm1_weights": gemm1_weights,
        "gemm1_weights_scale": gemm1_weights_scale,
        "gemm2_weights": gemm2_weights,
        "gemm2_weights_scale": gemm2_weights_scale,
        "output": output,
    }
    for name, t in all_args.items():
        print(f"[DEBUG-1]   {name:25s} type={type(t).__name__:12s} dtype={t.dtype}  shape={tuple(t.shape)}  strides={t.stride()}  contiguous={t.is_contiguous()}", flush=True)
    print(f"[DEBUG-1]   local_expert_offset      = {local_expert_offset} (type={type(local_expert_offset).__name__})", flush=True)
    print(f"[DEBUG-1]   routed_scaling_factor     = {routed_scaling_factor} (type={type(routed_scaling_factor).__name__})", flush=True)

    # =========================================================================
    # LOG 2: Shape sanity checks
    # WHY: Validates dimensions match the expected MoE geometry constants.
    #      A mismatch here means the workload definition differs from our code.
    # =========================================================================
    print(f"[DEBUG-2] === SHAPE CHECKS ===", flush=True)
    expected = {
        "routing_logits": (seq_len, NUM_GLOBAL_EXPERTS),
        "routing_bias": (NUM_GLOBAL_EXPERTS,),
        "hidden_states": (seq_len, HIDDEN_DIM),
        "hidden_states_scale": (HIDDEN_DIM // FP8_BLOCK_SIZE, seq_len),
        "gemm1_weights": (NUM_LOCAL_EXPERTS, GATE_UP_DIM, HIDDEN_DIM),
        "gemm1_weights_scale": (NUM_LOCAL_EXPERTS, GATE_UP_DIM // FP8_BLOCK_SIZE, HIDDEN_DIM // FP8_BLOCK_SIZE),
        "gemm2_weights": (NUM_LOCAL_EXPERTS, HIDDEN_DIM, INTER_DIM),
        "gemm2_weights_scale": (NUM_LOCAL_EXPERTS, HIDDEN_DIM // FP8_BLOCK_SIZE, INTER_DIM // FP8_BLOCK_SIZE),
        "output": (seq_len, HIDDEN_DIM),
    }
    for name, exp_shape in expected.items():
        actual = tuple(all_args[name].shape)
        match = "OK" if actual == exp_shape else "MISMATCH!"
        print(f"[DEBUG-2]   {name:25s} expected={exp_shape}  actual={actual}  {match}", flush=True)

    # Step 0: zero output (DPS — tensor is pre-allocated but not zeroed)
    output.zero_()

    # =========================================================================
    # LOG 3: Routing kernel launch
    # WHY: The routing kernel uses complex logic (top-k, group selection).
    #      If Triton fails to compile or produces garbage, we catch it here.
    #      Sync + range check validates output correctness.
    # =========================================================================
    print(f"[DEBUG-3] === ROUTING KERNEL ===", flush=True)
    expert_ids = torch.empty((seq_len, TOPK), dtype=torch.int32, device=device)
    routing_scores = torch.empty((seq_len, TOPK), dtype=torch.float32, device=device)

    bias_f32 = routing_bias.to(torch.float32)
    BLOCK_T = 16
    grid_routing = (triton.cdiv(seq_len, BLOCK_T),)
    print(f"[DEBUG-3]   grid={grid_routing}, BLOCK_T={BLOCK_T}", flush=True)

    try:
        routing_kernel[grid_routing](
            routing_logits, bias_f32,
            expert_ids, routing_scores,
            seq_len,
            routing_logits.stride(0),
            expert_ids.stride(0),
            routing_scores.stride(0),
            NUM_EXPERTS=NUM_GLOBAL_EXPERTS,
            NUM_GROUPS=NUM_GROUPS,
            EPG=EXPERTS_PER_GROUP,
            KG=NUM_GROUPS_SELECTED,
            TOPK=TOPK,
            BLOCK_T=BLOCK_T,
        )
        torch.cuda.synchronize()
        eid_min = expert_ids.min().item()
        eid_max = expert_ids.max().item()
        score_min = routing_scores.min().item()
        score_max = routing_scores.max().item()
        has_nan = torch.isnan(routing_scores).any().item()
        print(f"[DEBUG-3]   expert_ids range=[{eid_min}, {eid_max}]", flush=True)
        print(f"[DEBUG-3]   scores range=[{score_min:.6f}, {score_max:.6f}], has_nan={has_nan}", flush=True)
    except Exception as e:
        print(f"[DEBUG-3]   ROUTING KERNEL FAILED: {e}", flush=True)
        _tb.print_exc(file=_sys.stdout)
        raise

    # =========================================================================
    # Step 2: Per-expert FFN
    # =========================================================================
    rsf = float(routed_scaling_factor)
    local_offset = int(local_expert_offset)
    print(f"[DEBUG-4] === EXPERT LOOP === local_offset={local_offset}, rsf={rsf}", flush=True)

    for local_idx in range(NUM_LOCAL_EXPERTS):
        global_id = local_offset + local_idx

        # Find (token, slot) pairs assigned to this expert
        pairs = (expert_ids == global_id).nonzero(as_tuple=False)  # [K, 2]
        num_tok = pairs.shape[0]
        if num_tok == 0:
            continue

        # =====================================================================
        # LOG 4: Per-expert dispatch info
        # WHY: Shows which experts are active and how many tokens they get.
        #      If num_tok is unexpectedly large, it could cause OOM in buffers.
        # =====================================================================
        print(f"[DEBUG-4]   Expert {local_idx} (global={global_id}): {num_tok} tokens", flush=True)

        tok_ids = pairs[:, 0].to(torch.int32).contiguous()
        slot_ids = pairs[:, 1].contiguous()

        # Routing scores for these token-expert pairs
        r_scores = routing_scores[pairs[:, 0], slot_ids].contiguous()

        # Expert weights
        w1 = gemm1_weights[local_idx]         # [4096, 7168] fp8
        w1s = gemm1_weights_scale[local_idx]  # [32, 56]
        w2 = gemm2_weights[local_idx]         # [7168, 2048] fp8
        w2s = gemm2_weights_scale[local_idx]  # [56, 16]

        tb = _token_bucket(num_tok)

        # =====================================================================
        # LOG 5: GEMM1 launch details
        # WHY: Captures exact grid dimensions, strides, and buffer shapes.
        #      Autotuning may fail if grid dims are 0, or strides are wrong.
        #      FP8 scale indexing bugs would show as wrong stride values.
        # =====================================================================
        gate_up_buf = torch.empty((num_tok, GATE_UP_DIM), dtype=torch.float32, device=device)
        print(f"[DEBUG-5]     GEMM1: num_tok={num_tok}, tb={tb}, K={HIDDEN_DIM}, N={GATE_UP_DIM}", flush=True)
        print(f"[DEBUG-5]       w1.shape={tuple(w1.shape)}, w1.stride={w1.stride()}, w1s.shape={tuple(w1s.shape)}, w1s.stride={w1s.stride()}", flush=True)
        print(f"[DEBUG-5]       hscale.stride(0)={hidden_states_scale.stride(0)}, h.stride(0)={hidden_states.stride(0)}", flush=True)
        print(f"[DEBUG-5]       gate_up_buf.shape={tuple(gate_up_buf.shape)}, stride={gate_up_buf.stride()}", flush=True)

        try:
            expert_gemm1_kernel[
                lambda meta: (
                    triton.cdiv(num_tok, meta["BLOCK_M"]),
                    triton.cdiv(GATE_UP_DIM, meta["BLOCK_N"]),
                )
            ](
                hidden_states, hidden_states_scale,
                tok_ids,
                w1, w1s,
                gate_up_buf,
                num_tok, tb,
                K=HIDDEN_DIM, N=GATE_UP_DIM,
                stride_h_seq=hidden_states.stride(0),
                stride_w1_n=w1.stride(0),
                stride_gu_tok=gate_up_buf.stride(0),
                stride_hscale_blk=hidden_states_scale.stride(0),
                stride_w1s_nb=w1s.stride(0),
            )
            torch.cuda.synchronize()
            g1_has_nan = torch.isnan(gate_up_buf).any().item()
            g1_has_inf = torch.isinf(gate_up_buf).any().item()
            print(f"[DEBUG-5]     GEMM1 OK. has_nan={g1_has_nan}, has_inf={g1_has_inf}, range=[{gate_up_buf.min().item():.4f}, {gate_up_buf.max().item():.4f}]", flush=True)
        except Exception as e:
            print(f"[DEBUG-5]     GEMM1 FAILED at expert {local_idx}: {e}", flush=True)
            _tb.print_exc(file=_sys.stdout)
            raise

        # =====================================================================
        # LOG 6: SwiGLU
        # WHY: Shape mismatch between gate and up projections would crash here.
        #      NaN/Inf from GEMM1 propagates and could cause silent corruption.
        # =====================================================================
        try:
            intermediate = _swiglu(gate_up_buf)
            torch.cuda.synchronize()
            sw_has_nan = torch.isnan(intermediate).any().item()
            print(f"[DEBUG-6]     SwiGLU OK. shape={tuple(intermediate.shape)}, has_nan={sw_has_nan}", flush=True)
        except Exception as e:
            print(f"[DEBUG-6]     SwiGLU FAILED at expert {local_idx}: {e}", flush=True)
            _tb.print_exc(file=_sys.stdout)
            raise

        # =====================================================================
        # LOG 7: GEMM2 + atomic accumulate
        # WHY: atomic_add on bfloat16 requires SM >= 90 (Hopper/Blackwell).
        #      Wrong w2 strides or scale indexing would corrupt output.
        #      This is the most likely failure point for hardware compat issues.
        # =====================================================================
        print(f"[DEBUG-7]     GEMM2: K={INTER_DIM}, N={HIDDEN_DIM}", flush=True)
        print(f"[DEBUG-7]       w2.shape={tuple(w2.shape)}, w2.stride={w2.stride()}, w2s.shape={tuple(w2s.shape)}, w2s.stride={w2s.stride()}", flush=True)
        print(f"[DEBUG-7]       inter.stride(0)={intermediate.stride(0)}, out.stride(0)={output.stride(0)}", flush=True)

        try:
            expert_gemm2_accumulate_kernel[
                lambda meta: (
                    triton.cdiv(num_tok, meta["BLOCK_M"]),
                    triton.cdiv(HIDDEN_DIM, meta["BLOCK_N"]),
                )
            ](
                intermediate, w2, w2s,
                tok_ids, r_scores, rsf,
                output,
                num_tok, tb,
                K=INTER_DIM, N=HIDDEN_DIM,
                stride_inter_tok=intermediate.stride(0),
                stride_w2_n=w2.stride(0),
                stride_out_seq=output.stride(0),
                stride_w2s_nb=w2s.stride(0),
            )
            torch.cuda.synchronize()
            print(f"[DEBUG-7]     GEMM2 OK.", flush=True)
        except Exception as e:
            print(f"[DEBUG-7]     GEMM2 FAILED at expert {local_idx}: {e}", flush=True)
            _tb.print_exc(file=_sys.stdout)
            raise

    # =========================================================================
    # LOG 8: Final check
    # WHY: Confirms the kernel completed without silent CUDA async errors.
    # =========================================================================
    torch.cuda.synchronize()
    out_has_nan = torch.isnan(output.float()).any().item()
    out_nonzero = (output != 0).any().item()
    print(f"[DEBUG-8] === KERNEL COMPLETE === has_nan={out_has_nan}, has_nonzero={out_nonzero}", flush=True)

