# MoE Kernel Optimization Design
**Date:** 2026-04-12  
**Track:** `moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048`  
**Target hardware:** NVIDIA B200 (Blackwell)  
**Repos:** `ishan-pat/flashinfer-bench-starter-kit` (branch: `feature/batch_gemms`) + `nkkrnkl/mlsys_moe` (main)

---

## Problem

The current submission kernel iterates over 32 local experts in a Python loop, launching 2 Triton kernels per expert = **64 serial kernel launches per forward pass**. With ~16 tokens per expert on average (64-token batch), each GEMM is far too small to saturate the B200, and dispatch overhead dominates. The submission repo (`nkkrnkl`) additionally still uses PyTorch weight dequantization + `torch.mm` instead of native FP8 Triton kernels.

The contest baseline `flashinfer_deepgemm_wrapper` uses DeepGEMM's optimized FP8 grouped GEMM — we need to match or beat it.

---

## Solution: Token-Sorted Grouped GEMM + FP8 WGMMA

Replace the per-expert loop with a **token-sort preprocessing step** followed by **2 grouped Triton kernel launches** covering all 32 experts simultaneously.

### Pipeline

```
routing (PyTorch, verified correct)
  ↓
token sort (PyTorch, ~5 GPU ops)
  ↓
grouped_gemm1_kernel  [all 32 experts, FP8 WGMMA]  →  gate_up_buf [N_assigned, 4096]
  ↓
swiglu_kernel  [fused element-wise]  →  inter_buf [N_assigned, 2048]
  ↓
grouped_gemm2_kernel  [all 32 experts, BF16×FP8 WGMMA]  →  atomic scatter to output_f32
  ↓
output.copy_(output_f32.bfloat16())
```

**Before:** 64 kernel launches  
**After:** 4 kernel launches (GEMM1 + SwiGLU + GEMM2 + copy)

---

## Component Specifications

### 1. Routing (PyTorch, unchanged)

Keep the verified-correct PyTorch routing from `feature/batch_gemms` commit `70396c7`.

Inputs: `routing_logits [seq, 256]`, `routing_bias [256]`  
Outputs: `expert_ids [seq, 8] int32`, `weights_full [seq, 256] float32`

Algorithm (matches DeepSeek reference):
1. `s = sigmoid(logits)` — unbiased, used for final weights
2. `s_with_bias = s + bias` — biased, used for selection only
3. Group scoring: `topk(2)` per group on `s_with_bias`, sum → `group_scores [seq, 8]`
4. `top_groups = topk(4)` on `group_scores`
5. Mask non-top-group experts, `top8_ids = topk(8)` on masked `s_with_bias`
6. Routing weights: `s[top8_ids]`, normalized (no bias), `weights_full [seq, 256]`

### 2. Token Sorting (PyTorch preprocessing)

```python
# Flatten all (token, expert_slot) pairs to local experts
local_mask = (expert_ids >= local_offset) & (expert_ids < local_offset + 32)
pairs = local_mask.nonzero()                          # [N_assigned, 2]
tok_ids_flat    = pairs[:, 0].to(torch.int32)
local_exp_flat  = (expert_ids[pairs[:,0], pairs[:,1]] - local_offset).to(torch.int32)
r_scores_flat   = weights_full[pairs[:,0], expert_ids[pairs[:,0], pairs[:,1]]]

# Sort by local expert id
order           = torch.argsort(local_exp_flat, stable=True)
sorted_tok_ids  = tok_ids_flat[order].contiguous()    # [N_assigned] int32
sorted_exp_ids  = local_exp_flat[order].contiguous()  # [N_assigned] int32
sorted_r_scores = r_scores_flat[order].contiguous()   # [N_assigned] float32

# Expert offsets (cumulative token counts)
counts          = torch.bincount(sorted_exp_ids, minlength=32)  # [32] int32
expert_offsets  = torch.zeros(33, dtype=torch.int32, device=device)
expert_offsets[1:] = counts.cumsum(0).to(torch.int32)           # [33] int32
N_assigned      = int(sorted_tok_ids.shape[0])
max_tok         = int(counts.max().item()) if N_assigned > 0 else 0
```

### 3. `grouped_gemm1_kernel` (Triton, FP8 WGMMA)

**Purpose:** Compute `gate_up_buf = hidden[sorted_tok_ids] @ w1[expert_id].T` for all experts.

**Grid:** `(32 × ceil(max_tok / BLOCK_M), ceil(4096 / BLOCK_N))`

**Key logic:**
```
expert_id     = pid_m // tiles_per_expert
m_in_expert   = pid_m % tiles_per_expert
e_start       = expert_offsets[expert_id]
e_end         = expert_offsets[expert_id + 1]
if m_in_expert * BLOCK_M >= (e_end - e_start): return  # out of range, exit early

global_m      = e_start + m_in_expert * BLOCK_M + arange(BLOCK_M)
tok_ids       = sorted_tok_ids[global_m]               # gather

# K-loop: FP8 WGMMA
for k_start in range(0, 7168, 128):
    h_fp8    = hidden[tok_ids, k_start:k_start+128]    # gather rows
    h_scales = hscale[k_blk, tok_ids]                  # [BLOCK_M]
    w_fp8    = w1[expert_id, n_start:n_start+BLOCK_N, k_start:k_start+128]
    w_scales = w1scale[expert_id, n_blks, k_blk]       # [BLOCK_N] vectorized
    raw      = tl.dot(h_fp8, w_fp8.T, out_dtype=float32)
    acc     += raw * h_scales[:, None] * w_scales[None, :]

gate_up_buf[global_m, n_range] = acc
```

**Scale indexing:**
- `h_scales`: `hidden_states_scale[k_blk, tok_id]` — shape `[56, seq_len]`
- `w_scales`: `gemm1_weights_scale[expert_id, n_blk, k_blk]` — shape `[32, 32, 56]`; `n_blks = n_range // 128` (vectorized, one load per column in tile)

**Output:** `gate_up_buf [N_assigned, 4096] float32`

### 4. `swiglu_kernel` (Triton, fused)

**Purpose:** `inter = silu(gate) * up` where `gate = gate_up[:, :2048]`, `up = gate_up[:, 2048:]`

**Grid:** `(ceil(N_assigned × 2048 / BLOCK),)` where BLOCK = 1024

Reads `gate_up_buf`, writes `inter_buf [N_assigned, 2048] float32`.

Note: SwiGLU convention matches `feature/batch_gemms` fix — `gate` is first 2048 columns.

### 5. `grouped_gemm2_kernel` (Triton, BF16×FP8 WGMMA)

**Purpose:** Accumulate `output_f32[tok_id] += weight × inter[m] @ w2[expert_id].T`

**Grid:** `(32 × ceil(max_tok / BLOCK_M), ceil(7168 / BLOCK_N))`

**Key differences from GEMM1:**
- Input `inter_buf` is float32 → cast to BF16 before `tl.dot`
- Weights `w2 [7168, 2048] fp8` → pass directly, scale post-dot
- Routing weight `weight = sorted_r_scores[global_m] * rsf` applied to tile in epilogue
- Output: `tl.atomic_add` scatter to `output_f32[tok_ids, n_range]`

**Scale indexing:**
- `w_scales`: `gemm2_weights_scale[expert_id, n_blk, k_blk]` — shape `[32, 56, 16]`

### 6. Autotune Configuration

Both GEMM kernels share the same config space:

| Parameter | Values |
|-----------|--------|
| `BLOCK_M` | 64, 128 |
| `BLOCK_N` | 128, 256 |
| `BLOCK_K` | 128 (fixed — must align FP8 block scale boundary) |
| `num_stages` | 3, 4, 5 |
| `num_warps` | 8, 16 |

Autotune key: `["N_assigned_bucket", "N"]` where `N_assigned_bucket` rounds to nearest power of 2.

**Note on BLOCK_N=256:** When using BLOCK_N=256, a tile spans two FP8 scale blocks. The `n_blks = n_range // 128` vectorization correctly handles this — each column gets its own scale factor. This is already correct in the GEMM1 kernel design.

---

## Data Shapes Reference

| Tensor | Shape | Dtype |
|--------|-------|-------|
| `hidden_states` | `[seq_len, 7168]` | fp8_e4m3fn |
| `hidden_states_scale` | `[56, seq_len]` | float32 |
| `gemm1_weights` | `[32, 4096, 7168]` | fp8_e4m3fn |
| `gemm1_weights_scale` | `[32, 32, 56]` | float32 |
| `gemm2_weights` | `[32, 7168, 2048]` | fp8_e4m3fn |
| `gemm2_weights_scale` | `[32, 56, 16]` | float32 |
| `sorted_tok_ids` | `[N_assigned]` | int32 |
| `expert_offsets` | `[33]` | int32 |
| `gate_up_buf` | `[N_assigned, 4096]` | float32 |
| `inter_buf` | `[N_assigned, 2048]` | float32 |
| `output_f32` | `[seq_len, 7168]` | float32 |
| `output` | `[seq_len, 7168]` | bfloat16 |

---

## Correctness Invariants

1. **Routing must be numerically identical** to the reference — keep PyTorch routing unchanged.
2. **SwiGLU convention:** `gate = gate_up[:, :2048]` (first half), `up = gate_up[:, 2048:]` (second half). `inter = silu(gate) * up`. This matches `feature/batch_gemms` commit `70396c7`.
3. **FP8 block scale alignment:** `BLOCK_K = 128` always. `n_blks = n_range // 128` for vectorized N-scale lookup.
4. **Empty expert handling:** If `N_assigned == 0`, skip all three Triton kernels entirely. If `counts[e] == 0` for a specific expert, its CTAs exit early without writing.
5. **Atomic correctness:** `output_f32` must be zero-initialized before GEMM2. Multiple experts contribute to the same token row via `tl.atomic_add`.

---

## Deployment Steps

After implementing and testing locally:

1. `python scripts/pack_solution.py` → regenerates `solution.json`
2. Commit + push to `ishan-pat/flashinfer-bench-starter-kit` branch `feature/batch_gemms`
3. Copy `solution/triton/kernel.py` + `solution.json` to `/tmp/mlsys_moe`, commit + push to `nkkrnkl/mlsys_moe` main
4. Tag `nkkrnkl/mlsys_moe` main with `submission-v4` (or next tag)

---

## What We Are Not Doing

- **Persistent kernel / warp specialization**: Triton's `num_stages` autotune already captures most of this; marginal gain unclear without benchmarking.
- **Re-quantizing SwiGLU output to FP8**: Would enable FP8 GEMM2 but risks accuracy loss and adds complexity not worth taking without testing.
- **Custom CUDA kernel**: Triton is faster to iterate and sufficient for B200 given the B200's mature Triton backend.
