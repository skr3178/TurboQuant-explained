# TurboQuant Paper Implementation Plan

## Context

Implementing arXiv:2504.19874 — "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate". The existing codebase has hypersphere visualizations but no quantization code. Goal: reproduce the core algorithms and key experiments (Figs 1–3, Table 2, and KV cache eval as stretch goal) using an NVIDIA RTX 3060 12GB GPU.

---

## File Structure

```
TurboQuant/
├── pyproject.toml                        # add: torch, transformers, datasets, bitsandbytes
├── turboquant/
│   ├── __init__.py
│   ├── rotation.py                       # Π (QR) and S (QJL) matrix generation + caching
│   ├── codebook.py                       # Lloyd-Max solver for Beta distribution
│   ├── quantizer.py                      # TurboQuantMSE, TurboQuantProd classes
│   └── kv_cache/
│       ├── __init__.py
│       ├── outlier.py                    # per-channel outlier detection (2.5/3.5-bit)
│       └── hooks.py                      # transformers KV cache hooks
├── experiments/
│   ├── data_utils.py                     # DBpedia / GloVe loaders + normalization
│   ├── eval_metrics.py                   # MSE, IP error, Recall@k
│   ├── fig1_error_histograms.py          # Reproduce Fig 1
│   ├── fig2_grouped_histograms.py        # Reproduce Fig 2
│   ├── fig3_mse_vs_bitwidth.py           # Reproduce Fig 3 (primary validation)
│   ├── ann_search.py                     # Section 4.4: ANN Recall@1 + timing (Table 2)
│   └── kv_cache_eval.py                  # Sections 4.2/4.3 (stretch goal)
└── turboquant/cache/                     # auto-generated codebooks + rotation matrices
```

---

## Algorithm Reference

### Algorithm 1 — TurboQuant_mse
**Setup** (once per d, b):
1. `Π` ← QR decomposition of random N(0,1) matrix ∈ ℝ^(d×d)
2. Codebook `{c_1,...,c_{2^b}}` ← Lloyd-Max on Beta marginal f_X(x) = Γ(d/2)/(√π·Γ((d-1)/2))·(1-x²)^((d-3)/2)

**QUANT_mse(x)**: `y = Π·x`, then `idx_j = argmin_k |y_j - c_k|` ∀j

**DEQUANT_mse(idx)**: `ỹ_j = c_{idx_j}`, then `x̃ = Πᵀ·ỹ`

### Algorithm 2 — TurboQuant_prod
**Setup**: TurboQuant_mse at b-1 bits + random Gaussian matrix S ∈ ℝ^(d×d) (QJL)

**QUANT_prod(x)**: idx ← QUANT_mse(x); r ← x - DEQUANT_mse(idx); qjl ← sign(S·r); output (idx, qjl, ‖r‖₂)

**DEQUANT_prod(idx, qjl, γ)**: x̃_mse ← DEQUANT_mse(idx); x̃_qjl ← √(π/2)/d · γ · Sᵀ·qjl; output x̃_mse + x̃_qjl

### Theoretical bounds to reproduce
- Upper: D_mse ≤ (√3π/2) · 4^(-b)
- Lower: D_mse ≥ 1/4^b  (TurboQuant ≈ 2.7x from optimal)
- Concrete: D_mse(b=1,2,3,4) ≈ 0.36, 0.117, 0.03, 0.009

---

## Implementation Steps (ordered by dependency)

### Step 1 — Dependencies (`pyproject.toml`)
Add: `torch>=2.3.0` (CUDA 12.x wheel), `transformers>=4.45`, `datasets>=2.20`, `bitsandbytes>=0.43`, `tqdm`

Note: torch installs via `uv pip install torch --index-url https://download.pytorch.org/whl/cu121`. Check bitsandbytes Python 3.13 compatibility first; fallback is llama-cpp-python + GGUF.

### Step 2 — `turboquant/rotation.py`
- `make_rotation_matrix(d, seed, device)`: draw N(0,1) matrix → QR → fix column signs via `sign(diag(R))` → return orthogonal Q. Cache to `cache/rotation_d{d}_seed{seed}.pt`.
- `make_qjl_matrix(d, seed, device)`: raw N(0,1) matrix (no orthogonalization). Cache similarly.
- **Verify**: `Pi @ Pi.T ≈ I` to atol=1e-5.

### Step 3 — `turboquant/codebook.py`
- `beta_marginal_pdf(x, d)`: use `scipy.special.gammaln` for stability. Switch to Gaussian N(0,1/d) approximation for d≥50.
- `lloyd_max_1d(n_centroids, d, n_iter=500, tol=1e-10)`:
  - Init: percentiles of target distribution
  - Iterate: boundaries = midpoints between centroids; update centroids = conditional means via `scipy.integrate.quad`
  - Enforce symmetry: c_0 = -c_{2^b-1}, etc.
- `get_codebook(b, d, device)`: check `cache/codebook_b{b}_d{d}.npy`, compute if absent, return as float32 torch tensor.
- `compute_theoretical_bounds(b)`: returns (upper=(√3π/2)·4^(-b), lower=1/4^b).
- **Verify**: b=1,d=1536 → centroids ≈ ±0.0203; b=2,d=1536 → ≈ ±{0.01156, 0.03853}.

### Step 4 — `turboquant/quantizer.py`
**`TurboQuantMSE(d, b, seed=None, device='cpu')`**
- `quantize(x: Tensor[...,d]) -> Tensor[...,d]` (int indices): `y = x @ Pi.T`; broadcast against codebook; argmin.
- `dequantize(idx) -> Tensor[...,d]`: gather codebook entries; `x̃ = ỹ @ Pi`.
- Add `_chunk_apply(fn, x, chunk_size=8192)` for large batches (avoids OOM at 100K×1536).

**`TurboQuantProd(d, b, seed_mse=None, seed_qjl=None, device='cpu')`**
- `quantize(x) -> (idx, qjl, norm_r)`: compute residual; `qjl = sign(S·r)` with `torch.where(Sr >= 0, 1., -1.)` (no zeros!).
- `dequantize(idx, qjl, gamma) -> Tensor[...,d]`.

**Verify**:
- Empirical D_mse at b=2, d=1536 on 10K unit-sphere samples ≈ 0.117 ± 20%.
- Inner product unbiasedness: E[⟨y, dequant_prod(quant_prod(x))⟩] ≈ ⟨y,x⟩.

### Step 5 — `experiments/data_utils.py`
- `load_dbpedia_1536(n=100_000)`: HuggingFace `datasets` → float32 tensor → L2-normalize.
- `load_dbpedia_3072(n=100_000)`: same for 3072-dim dataset.
- `load_glove(dim=300)`: download from Stanford URL if not cached; normalize.
- `split_train_query(data, n_query=1000)`: return (train, query).

### Step 6 — `experiments/eval_metrics.py`
- `mse_distortion(x_orig, x_recon)`: mean squared L2 error per vector, averaged.
- `inner_product_error(x_orig, x_recon, queries)`: `(x_orig @ queries.T) - (x_recon @ queries.T)` → error array.
- `recall_at_k(database_q, queries, k=1, approx_fn, device)`: brute-force ground truth vs approximate; return fraction of correct top-1.

### Step 7 — `experiments/fig3_mse_vs_bitwidth.py` (primary validation)
1. Load DBpedia 1536-dim (100K train, 1K query).
2. For b ∈ {1,2,3,4}: TurboQuantMSE quantize/dequantize → empirical D_mse; also run TurboQuantProd → IP error.
3. Plot log-scale: empirical curves + upper bound (√3π/2)·4^(-b) + lower bound 1/4^b.
4. Save `results/Fig_3_reproduced.png`. Assert empirical D_mse lies between bounds.

### Step 8 — `experiments/fig1_error_histograms.py`
1. DBpedia 1536-dim. For b ∈ {1,2,3,4}: quantize 100K train vectors with both methods.
2. Compute IP error against 1K queries → histograms.
3. 2×4 panel (prod row / mse row). Save `results/Fig_1_reproduced.png`.

### Step 9 — `experiments/fig2_grouped_histograms.py`
1. DBpedia, b=2. Compute avg IP of each train vector vs queries.
2. Bin by avg IP quartile → 4 groups.
3. Per group: IP error histogram for prod vs mse. Save `results/Fig_2_reproduced.png`.

### Step 10 — `experiments/ann_search.py`
1. Build compressed index (TurboQuantProd, b=4) for DBpedia 1536/3072 and GloVe.
2. ANN search: decode all database vectors → inner product → top-k.
3. Ground truth: brute-force (chunked on GPU).
4. Report Recall@1. Time quantization with `time.perf_counter` for Table 2.
5. PQ baseline via `faiss.IndexPQ`.

### Step 11 (stretch) — KV Cache (`kv_cache/outlier.py`, `hooks.py`, `kv_cache_eval.py`)
- `identify_outlier_channels`: variance-based per-channel; top 32 → 3-bit, next 96 → 2-bit.
- `MixedPrecisionQuantizer`: holds TurboQuantMSE at b=2 and b=3.
- `install_kv_hooks`: forward hook on `model.model.layers[i].self_attn` to compress K,V in-place.
- Load Llama-3.1-8B with `load_in_4bit=True` (bitsandbytes) → ~5GB.
- Needle-in-Haystack: inject needle at varying depths, vary context 4k→104k tokens, report recall score.

---

## Memory Budget (12GB RTX 3060)

| Component | Size |
|---|---|
| Llama-3.1-8B 4-bit | ~5.0 GB |
| Π and S matrices (d=1536) | 18 MB |
| DBpedia 100K×1536 FP32 | ~600 MB |
| Working chunk (8192×1536) | ~50 MB |
| Compressed index (b=2) | ~50 MB |
| **Available for KV cache** | **~6 GB** |

KV cache at 2.5-bit avg for 128K context (Llama-3.1-8B): ~2.6 GB → fits.
Decompress **one layer at a time** during attention (not all 32 simultaneously).

---

## Verification Plan

1. **Codebook unit test**: assert b=1 centroids match paper formula within 1%.
2. **Rotation unit test**: `Pi @ Pi.T ≈ I`.
3. **Quantizer smoke test**: D_mse ≈ 0.117 at b=2 on random unit-sphere vectors (d=1536).
4. **Unbiasedness test**: E[⟨y, dequant_prod(x)⟩] ≈ ⟨y,x⟩ on 1K samples.
5. **Fig 3 bound check**: empirical D_mse lies strictly between upper and lower bounds for b=1..4.
6. **ANN timing**: TurboQuant should be orders of magnitude faster than PQ (paper: 0.0013s vs 239.75s for d=1536).

---

## Key Pitfalls
- `torch.sign(0) = 0` — use `torch.where(Sr >= 0, 1., -1.)` for QJL.
- Row-vector convention throughout: `y = x @ Pi.T`, inverse: `x̃ = ỹ @ Pi`.
- bitsandbytes may not support Python 3.13 — verify before starting KV cache step.
- GloVe vectors are NOT unit-normalized — normalize before quantizing.
- DBpedia embeddings are already near-unit — normalize defensively anyway.
