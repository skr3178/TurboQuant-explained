# TurboQuant — DBpedia Figures 1, 2, 3 Implementation Plan

> **Status**: Phases 1 & 2 complete. This plan covers Phase 3: DBpedia 1536-dim experiments reproducing Figs 1, 2, 3.

---

## Completed ✓

| File | Status |
|---|---|
| `turboquant/rotation.py` | Done — `make_rotation_matrix`, `make_qjl_matrix` with caching |
| `turboquant/codebook.py` | Done — `lloyd_max_1d`, `get_codebook`, `compute_theoretical_bounds` |
| `turboquant/quantizer.py` | Done — `TurboQuantMSE`, `TurboQuantProd` with chunking |
| `turboquant/__init__.py` | Done |
| `experiments/data_utils.py` | Partial — GloVe loader done; **needs `load_dbpedia_1536`** |
| `experiments/eval_metrics.py` | Partial — MSE + summary IP error done; **needs flat error + D_prod functions** |
| `experiments/fig3_mse_vs_bitwidth.py` | Done — synthetic d=1536 version |
| `experiments/glove_mse_validation.py` | Done — GloVe 300d MSE |
| `experiments/glove_ann_search.py` | Done — GloVe ANN + timing |

---

## Phase 3: DBpedia Experiments — Figures 1, 2, 3

### What each figure shows

**Figure 1** — *Error distribution histograms* (2 rows × 4 cols)
- Rows: TurboQuant_prod (top) vs TurboQuant_mse (bottom)
- Cols: bitwidth b = 1, 2, 3, 4
- X-axis: Inner Product Distortion ∈ [-0.1, 0.1]; Y-axis: Frequency (×10^7)
- Scale: 100K vectors × 1K queries = 100M error values per panel
- Key insight: prod is always centered at 0 (unbiased); mse shows rightward shift at low bits

**Figure 2** — *Error histograms grouped by average inner product* (2 rows × 4 cols), fixed b=2
- Rows: TurboQuant_prod (top) vs TurboQuant_mse (bottom)
- Cols: avg IP bins ≈ {0.01, 0.06, 0.10, 0.17} (quartiles of the avg-IP distribution)
- Y-axis: Frequency (×10^6); X-axis: [-0.05, 0.05]
- Key insight: prod histogram width is constant across bins; mse width grows with avg IP

**Figure 3** — *Distortion vs bitwidth*, b = 1..5, log scale (2 panels)
- Panel (a): Inner-product error D_prod — TurboQuant_mse and TurboQuant_prod curves + bounds
- Panel (b): MSE D_mse — TurboQuant_mse only + bounds

---

## Exact Metrics

**Raw IP error** for one (vector x, query y) pair:
```
e(x, y) = ⟨y, x⟩ - ⟨y, x̃⟩    where x̃ = DEQUANT(QUANT(x))
```

**D_prod** (scalar, for Fig 3 panel a):
```
D_prod = E[e²] = mean over all (i,j) of e(x_i, y_j)²
```
Computed as: `((X - X_recon) @ queries.T).pow(2).mean()` — chunked to avoid OOM.

**D_mse** (scalar, for Fig 3 panel b):
```
D_mse = mean over all vectors of ||x - x̃||²
```
Already implemented in `eval_metrics.mse_distortion`.

**Theoretical bounds:**

| Metric | Lower bound | Upper bound |
|---|---|---|
| D_mse | `4^(-b)` | `(√3π/2) · 4^(-b)` ≈ 2.72 · 4^(-b) |
| D_prod (unit-norm x,y) | `(1/d) · 4^(-b)` | `(√3π²/d) · 4^(-b)` ≈ 17.3/d · 4^(-b) |

For d=1536: D_prod lower ≈ 6.51×10^-4 · 4^(-b), upper ≈ 1.13×10^-2 · 4^(-b).

**Fig 2 grouping:** `avg_ip[i] = mean_j(⟨y_j, x_i⟩)` for each database vector.
Split into 4 quartile bins; label each panel with the median avg_ip of that bin.

---

## Step-by-Step Implementation

### Step A — `experiments/data_utils.py`: add `load_dbpedia_1536`

```python
def load_dbpedia_1536(n=100_000, n_query=1_000, cache_dir="data") -> tuple[Tensor, Tensor]:
    # dataset: Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M
    # embedding column: "text-embedding-3-large-1536"
    # Cache to data/dbpedia_1536_{n+n_query}.pt to skip re-download
    # L2-normalize defensively (already near-unit from OpenAI)
    # Return: (database [n, 1536], queries [n_query, 1536])
```

### Step B — `experiments/eval_metrics.py`: add two functions

**`inner_product_errors_flat(x_orig, x_recon, queries, chunk_size=10_000) -> np.ndarray`**
- Returns flat np.float32 array of all N×Q raw errors e(x_i, y_j)
- Loop over database in chunks of `chunk_size`:
  `diff = x_orig[chunk] - x_recon[chunk]`; `errors = (diff @ queries.T).cpu().numpy()`
  Append each chunk's (chunk_size, Q) result, then `np.concatenate` and `.ravel()` at end
- Total: 100K × 1K × 4B = 400 MB on CPU — acceptable

**`ip_distortion(x_orig, x_recon, queries, chunk_size=10_000) -> float`**
- Returns scalar D_prod = E[e²] without storing all 100M errors simultaneously
- Running accumulation:
  `sq_sum += ((diff @ queries.T).pow(2)).sum().item()`
  divide by `N * Q` at end

### Step C — `experiments/fig1_error_histograms.py` (new file)

```python
device = "cuda"
database, queries = load_dbpedia_1536(n=100_000, n_query=1_000)
database, queries = database.to(device), queries.to(device)

fig, axes = plt.subplots(2, 4, figsize=(16, 6), sharey="row")

for col, b in enumerate([1, 2, 3, 4]):
    for row, QuantClass in enumerate([TurboQuantProd, TurboQuantMSE]):
        q = QuantClass(d=1536, b=b, device=device)
        out = q.quantize(database)
        recon = q.dequantize(*out) if isinstance(out, tuple) else q.dequantize(out)
        errors = inner_product_errors_flat(database, recon, queries)
        axes[row, col].hist(errors, bins=200, range=(-0.1, 0.1))
        axes[row, col].set_title(f"Bitwidth = {b}")

# Row 0 title: TurboQuant_prod, Row 1 title: TurboQuant_mse
plt.tight_layout()
plt.savefig("results/Fig_1_dbpedia.png", dpi=150)
```

### Step D — `experiments/fig2_grouped_histograms.py` (new file)

```python
b = 2
database, queries = load_dbpedia_1536(...)
avg_ip = (database @ queries.T).mean(dim=1).cpu().numpy()   # shape [N]
edges = np.percentile(avg_ip, [0, 25, 50, 75, 100])

q_prod = TurboQuantProd(d=1536, b=b, device=device)
q_mse  = TurboQuantMSE(d=1536,  b=b, device=device)
recon_prod = q_prod.dequantize(*q_prod.quantize(database))
recon_mse  = q_mse.dequantize(q_mse.quantize(database))

fig, axes = plt.subplots(2, 4, figsize=(16, 6), sharey="row")

for col, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
    mask_np = (avg_ip >= lo) & (avg_ip < hi)
    mask = torch.from_numpy(mask_np)
    median_ip = float(np.median(avg_ip[mask_np]))

    for row, recon in enumerate([recon_prod, recon_mse]):
        errors = inner_product_errors_flat(database[mask], recon[mask], queries)
        axes[row, col].hist(errors, bins=100, range=(-0.05, 0.05))
        axes[row, col].set_title(f"Avg IP = {median_ip:.2f}")

plt.tight_layout()
plt.savefig("results/Fig_2_dbpedia.png", dpi=150)
```

### Step E — `experiments/fig3_dbpedia.py` (new file)

```python
database, queries = load_dbpedia_1536(...)
d, bitwidths = 1536, [1, 2, 3, 4, 5]

d_mse_vals, d_prod_mse_vals, d_prod_prod_vals = [], [], []

for b in bitwidths:
    q_mse = TurboQuantMSE(d=d, b=b, device=device)
    recon_mse = q_mse.dequantize(q_mse.quantize(database))
    d_mse_vals.append(mse_distortion(database, recon_mse))
    d_prod_mse_vals.append(ip_distortion(database, recon_mse, queries))

    if b >= 2:
        q_prod = TurboQuantProd(d=d, b=b, device=device)
        recon_prod = q_prod.dequantize(*q_prod.quantize(database))
        d_prod_prod_vals.append(ip_distortion(database, recon_prod, queries))

b_arr = np.array(bitwidths)

fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 5))

# Panel (a): inner-product error
ax_a.semilogy(b_arr, d_prod_mse_vals, 'b-o', label="TurboQuant_mse")
ax_a.semilogy(b_arr[1:], d_prod_prod_vals, 'm-o', label="TurboQuant_prod")
ax_a.semilogy(b_arr, (1/d) * 4.**(-b_arr), 'g--', label="Lower bound")
ax_a.semilogy(b_arr, (np.sqrt(3)*np.pi**2/d) * 4.**(-b_arr), 'r--', label="Upper bound")

# Panel (b): MSE
ax_b.semilogy(b_arr, d_mse_vals, 'b-o', label="TurboQuant_mse")
ax_b.semilogy(b_arr, 4.**(-b_arr), 'g--', label="Lower bound")
ax_b.semilogy(b_arr, (np.sqrt(3)*np.pi/2) * 4.**(-b_arr), 'r--', label="Upper bound")

plt.savefig("results/Fig_3_dbpedia.png", dpi=150)
```

---

## Memory Plan (12GB GPU)

| Component | Size |
|---|---|
| Database 100K×1536 FP32 | ~600 MB |
| Queries 1K×1536 FP32 | ~6 MB |
| Π and S matrices (1536×1536) | ~18 MB |
| Chunk diff buffer (10K×1536) | ~60 MB |
| Flat error array (100M float32, CPU) | ~400 MB |

Keep database + queries + rotation/QJL matrices on GPU throughout.
Compute `diff @ queries.T` in chunks of 10K, move each chunk to CPU immediately.
Never materialize the full 100M error array on GPU.

---

## Verification Checks

- **Fig 1 prod rows**: all 4 histograms symmetric about 0; `errors.mean() < 1e-4`
- **Fig 1 mse rows**: b=1 histogram visibly right-shifted; shift shrinks with increasing b
- **Fig 2 prod cols**: histogram std approximately equal across all 4 avg-IP bins
- **Fig 2 mse cols**: histogram std increases monotonically from left to right column
- **Fig 3 bounds**: empirical curves lie between dashed lower and upper bounds for all b;
  slope on log10 scale ≈ -0.602 per bitwidth (4× decrease per step)
