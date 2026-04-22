# GEMM2 BF16 WGMMA + Wider Autotune — Final Submission Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Push the MoE kernel from ~11-13x speedup to ~15-18x by switching GEMM2 from TF32 (float32 dequant path) to BF16 WGMMA, widening the autotune config space for GEMM2 to include BLOCK_N=512, and eliminating the float32 output buffer via direct BF16 atomic_add.

**Architecture:** Three targeted edits to `solution/triton/kernel.py` only. Task 1 adds BF16 WGMMA to GEMM2 and splits the autotune configs (GEMM2 gets BLOCK_N=512 added). Task 2 eliminates the float32 output buffer by zeroing and scatter-adding directly into the pre-allocated BF16 output tensor. Each task ends with a Modal benchmark run to confirm 19/19 pass and measure the speedup delta before committing.

**Tech Stack:** Triton 3.5.1, PyTorch, Modal B200 GPU; single file `solution/triton/kernel.py`.

---

## File map

- Modify: `solution/triton/kernel.py` (the only file that changes)
  - `_gemm_configs()` → split into `_gemm1_configs()` and `_gemm2_configs()`
  - `grouped_gemm1_kernel` autotune decorator → use `_gemm1_configs()`
  - `grouped_gemm2_kernel` autotune decorator → use `_gemm2_configs()`
  - `grouped_gemm2_kernel` inner loop → BF16 WGMMA (lines 385-399)
  - `kernel()` entry point → direct BF16 output, remove float32 buffer (lines 438-496)

---

## Task 1: BF16 WGMMA in GEMM2 + split autotune configs

**Files:**
- Modify: `solution/triton/kernel.py:52-62` (split configs)
- Modify: `solution/triton/kernel.py:175` (GEMM1 decorator)
- Modify: `solution/triton/kernel.py:303-307` (GEMM2 decorator)
- Modify: `solution/triton/kernel.py:385-399` (GEMM2 inner loop)

- [ ] **Step 1: Replace `_gemm_configs()` with two separate functions**

Find the current `_gemm_configs()` at line 52 and replace the entire function with two:

```python
def _gemm1_configs():
    """Autotune configs for GEMM1. N=INTER_DIM=2048, BLOCK_N up to 256."""
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


def _gemm2_configs():
    """Autotune configs for GEMM2. N=HIDDEN_DIM=7168, try BLOCK_N=512 too."""
    configs = []
    for bm in [64, 128]:
        for bn in [128, 256, 512]:
            for ns in [3, 4, 5]:
                for nw in [8, 16]:
                    configs.append(triton.Config(
                        {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": 128},
                        num_stages=ns, num_warps=nw,
                    ))
    return configs
```

- [ ] **Step 2: Update the GEMM1 autotune decorator to use `_gemm1_configs()`**

Find line 175:
```python
@triton.autotune(configs=_gemm_configs(), key=["n_assigned_bucket", "N"])
```
Change to:
```python
@triton.autotune(configs=_gemm1_configs(), key=["n_assigned_bucket", "N"])
```

- [ ] **Step 3: Update the GEMM2 autotune decorator to use `_gemm2_configs()`**

Find lines 303-307:
```python
@triton.autotune(
    configs=_gemm_configs(),
    key=["n_assigned_bucket", "N"],
    reset_to_zero=("output_ptr",),
)
```
Change to:
```python
@triton.autotune(
    configs=_gemm2_configs(),
    key=["n_assigned_bucket", "N"],
    reset_to_zero=("output_ptr",),
)
```

- [ ] **Step 4: Rewrite the GEMM2 inner K-loop to use BF16 WGMMA**

Find lines 385-399 (the inner loop body of `grouped_gemm2_kernel`):

```python
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
```

Replace with:

```python
        # Load intermediate (float32), cast to BF16 for BF16 WGMMA
        i_ptrs = inter_ptr + global_m[:, None] * stride_inter_tok + k_range[None, :]
        i_f32  = tl.load(i_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        i_bf16 = i_f32.to(tl.bfloat16)                                  # [BM, BK]

        # FP8 weight  [BN, BK] — dequant to BF16 for BF16 WGMMA
        w_ptrs  = w2_base + n_range[:, None] * stride_w2_n + k_range[None, :]
        w_fp8   = tl.load(w_ptrs, mask=n_mask[:, None] & k_mask[None, :], other=0.0)
        w_scales = tl.load(
            w2s_base + n_blks * stride_w2s_nb + k_blk,
            mask=n_mask, other=1.0,
        )

        # BF16 WGMMA: 2x throughput vs TF32 on B200 (1979 vs 990 TFLOP/s peak)
        w_bf16 = (w_fp8.to(tl.float32) * w_scales[:, None]).to(tl.bfloat16)  # [BN, BK]
        acc   += tl.dot(i_bf16, tl.trans(w_bf16), out_dtype=tl.float32)
```

- [ ] **Step 5: Verify AST parse**

```bash
python3 -c "import ast; ast.parse(open('solution/triton/kernel.py').read()); print('OK')"
```

Expected: `OK`

- [ ] **Step 6: Run Modal benchmark to confirm correctness and measure speedup**

```bash
conda run -n fi-bench python scripts/pack_solution.py && conda run -n fi-bench modal run scripts/run_modal.py 2>&1 | tail -30
```

Expected output format:
```
moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048:
  Workload b8f4f012...: PASSED | X.XXX ms | XX.XXx speedup | ...
  ...
```

All 19 workloads must show `PASSED`. If any show `INCORRECT_NUMERICAL` or `RUNTIME_ERROR`, do NOT commit — revert step 4 and keep the float32 path.

- [ ] **Step 7: Commit (only if all 19 pass)**

```bash
git add solution/triton/kernel.py
git commit -m "perf: BF16 WGMMA in GEMM2 + BLOCK_N=512 autotune — separate gemm1/gemm2 configs"
```

---

## Task 2: Direct BF16 output — eliminate float32 accumulator buffer

**Files:**
- Modify: `solution/triton/kernel.py:401-404` (GEMM2 epilogue — atomic_add dtype)
- Modify: `solution/triton/kernel.py:438-496` (kernel() entry point)

Background: currently `kernel()` allocates `output_f32 [seq, 7168] float32`, GEMM2 scatter-adds float32 into it, then `output.copy_(output_f32.to(bf16))`. This wastes ~3.5-14MB (varies by seq_len) and costs one full memory sweep. We instead zero-init the pre-allocated BF16 `output` tensor and atomic_add the bf16-cast accumulator directly — saving the buffer and the copy.

**Risk gate:** `tl.atomic_add` on BF16 tensors uses hardware CAS emulation in some Triton versions. If it's slower, revert. Modal will tell us immediately.

- [ ] **Step 1: Update GEMM2 epilogue to cast acc to BF16 before atomic_add**

Find lines 401-404:
```python
    # Apply routing weight and scatter-add (multiple experts → same output token row)
    acc      = acc * weight[:, None]
    out_ptrs = output_ptr + tok_ids[:, None] * stride_out_seq + n_range[None, :]
    tl.atomic_add(out_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])
```

Replace with:
```python
    # Apply routing weight and scatter-add (multiple experts → same output token row)
    # Cast acc to BF16 before atomic_add — output tensor is BF16, saves float32 buffer
    acc      = (acc * weight[:, None]).to(tl.bfloat16)
    out_ptrs = output_ptr + tok_ids[:, None] * stride_out_seq + n_range[None, :]
    tl.atomic_add(out_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])
```

- [ ] **Step 2: Update `kernel()` entry point to use BF16 output directly**

Find lines 438-496 in `kernel()`. Replace the output allocation and GEMM2 call:

```python
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
```

Replace with:

```python
    # 3. Zero-init the BF16 output directly — no float32 buffer needed
    output.zero_()

    if N_assigned == 0:
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

    # 5. Grouped GEMM2 — BF16 atomic scatter-add directly into output (DPS tensor)
    grouped_gemm2_kernel[
        lambda meta: (
            NUM_LOCAL_EXPERTS * triton.cdiv(max_tok, meta["BLOCK_M"]),
            triton.cdiv(HIDDEN_DIM, meta["BLOCK_N"]),
        )
    ](
        sorted_tok_ids, sorted_r_scores, expert_offsets, max_tok,
        inter_buf,
        gemm2_weights, gemm2_weights_scale,
        output,                               # BF16 output tensor directly
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
        stride_out_seq    = output.stride(0), # BF16 stride
    )
    # No copy needed — output is already BF16
```

- [ ] **Step 3: Update the `reset_to_zero` reference in GEMM2 autotune decorator**

The autotune decorator has `reset_to_zero=("output_ptr",)`. This still applies — it will now reset the BF16 output tensor between autotune trials. No change needed to the decorator itself.

- [ ] **Step 4: Verify AST parse**

```bash
python3 -c "import ast; ast.parse(open('solution/triton/kernel.py').read()); print('OK')"
```

Expected: `OK`

- [ ] **Step 5: Run Modal benchmark**

```bash
conda run -n fi-bench python scripts/pack_solution.py && conda run -n fi-bench modal run scripts/run_modal.py 2>&1 | tail -30
```

**Decision gate:**
- If all 19 `PASSED` AND avg speedup ≥ Task 1 speedup → commit and continue to Task 3
- If any `INCORRECT_NUMERICAL` → BF16 atomic_add has precision issue; revert Task 2 only (keep Task 1). The float32 path is the fallback.
- If any `RUNTIME_ERROR` → Triton 3.5.1 doesn't support BF16 atomic_add on this hardware; revert Task 2 only.

- [ ] **Step 6: Commit (only if 19/19 pass and speedup ≥ Task 1)**

```bash
git add solution/triton/kernel.py
git commit -m "perf: direct BF16 output — eliminate float32 accumulator buffer and memcpy"
```

---

## Task 3: Final pack, push, and tag for April 24 submission

**Files:**
- `solution.json` (regenerated by pack_solution.py)
- `/tmp/mlsys_moe_fresh/solution/triton/kernel.py` (submission repo)
- `/tmp/mlsys_moe_fresh/solution.json` (submission repo)

- [ ] **Step 1: Pack solution on personal branch**

```bash
conda run -n fi-bench python scripts/pack_solution.py
```

Expected:
```
Solution packed: /Users/ishan/flashinfer-bench-starter-kit/solution.json
  Name: grouped-gemm-moe-v1
  Definition: moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048
  Author: shan-antigravity
  Language: triton
```

- [ ] **Step 2: Commit solution.json and push personal branch**

```bash
git add -f solution.json
git commit -m "chore: repack solution.json for final submission"
git push origin feature/batch_gemms
```

- [ ] **Step 3: Copy kernel and solution.json to submission repo**

```bash
cp solution/triton/kernel.py /tmp/mlsys_moe_fresh/solution/triton/kernel.py
cp solution.json /tmp/mlsys_moe_fresh/solution.json
```

- [ ] **Step 4: Fix author in submission repo's solution.json**

```bash
python3 -c "
import json
with open('/tmp/mlsys_moe_fresh/solution.json') as f:
    d = json.load(f)
d['author'] = 'adastra'
with open('/tmp/mlsys_moe_fresh/solution.json', 'w') as f:
    json.dump(d, f, indent=2)
print('author:', d['author'])
"
```

Expected: `author: adastra`

- [ ] **Step 5: Commit, push, and tag submission-v8**

```bash
git -C /tmp/mlsys_moe_fresh add -f solution/triton/kernel.py solution.json
git -C /tmp/mlsys_moe_fresh commit -m "perf: final kernel — BF16 WGMMA in GEMM2, BLOCK_N=512 autotune, direct BF16 output"
git -C /tmp/mlsys_moe_fresh push origin main
git -C /tmp/mlsys_moe_fresh tag submission-v8
git -C /tmp/mlsys_moe_fresh push origin submission-v8
```

- [ ] **Step 6: Verify tag is live**

```bash
git -C /tmp/mlsys_moe_fresh log --oneline -3
git -C /tmp/mlsys_moe_fresh tag | tail -3
```

Expected: `submission-v8` appears in tag list with latest commit hash.
