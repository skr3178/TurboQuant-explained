# Table 2 — Quantization Timing Benchmark

## Goal
Reproduce the TurboQuant column of Table 2, and add PQ for comparison. Skip RabitQ.

## Paper's Table 2 (reference numbers)
| Approach             | d=200  | d=1536  | d=3072  |
|----------------------|--------|---------|---------|
| Product Quantization | 37.04  | 239.75  | 494.42  |
| RabitQ               | 597.25 | 2267.59 | 3957.19 |
| TurboQuant           | 0.0007 | 0.0013  | 0.0021  |

Times in seconds, 4-bit quantization, N=100,000 vectors, measured on NVIDIA A100.

## Confirmed Experimental Parameters (from paper Section 4.4)
- **N = 100,000** vectors (training set size used throughout Section 4)
- **Bitwidth = 4**
- **Datasets**: GloVe (d=200), DBpedia OpenAI3 embeddings (d=1536, d=3072)
- **Hardware**: NVIDIA A100 (paper) vs RTX 3060 (ours — times will differ, ratios should hold)

## PQ Configuration (from paper page 20)
> "we opted for a version of PQ that uses LUT256, which contains 256 codewords.
>  For 4-bit quantization, it groups 2 coordinates per lookup."

faiss equivalent:
```python
faiss.ProductQuantizer(d, d // 2, 8)
# M = d/2 sub-vectors, each dim=2, 8 bits (256 centroids) → 4 bits/coord effective
```

## RabitQ
Skip — CPU-only, no vectorization, requires a separate reference implementation.
The paper confirms it is 600–4000s due to these limitations. Not worth reproducing.

## Implementation Plan: `experiments/table2_timing.py`

### Step 1 — Setup
- Generate random float32 unit vectors at each dimension: d ∈ {200, 1536, 3072}
- N = 100,000 vectors
- Pre-build codebook and instantiate quantizers *before* the timing loop (exclude one-time setup)

### Step 2 — TurboQuant timing
- Use `TurboQuantMSE.quantize(X)` — encode only, no dequantize
- Call `get_codebook()` before the loop so it uses the cache
- Warm up with one dry run, then time over 10 repeated runs
- Report mean time (seconds)

### Step 3 — PQ timing
```python
import faiss
pq = faiss.ProductQuantizer(d, d // 2, 8)
pq.train(X)               # one-time, excluded from timing
codes = pq.compute_codes(X)  # this is what gets timed
```
- Train once before the timing loop
- Time `compute_codes` over 10 repeated runs
- Report mean time

### Step 4 — Output
Print a table matching the paper format:
```
Method         d=200     d=1536    d=3072
TurboQuant     X.XXXXs   X.XXXXs   X.XXXXs
PQ             X.XXXXs   X.XXXXs   X.XXXXs
Paper (TQ)     0.0007s   0.0013s   0.0021s
```

## Notes
- TurboQuant runs on CUDA; PQ runs on CPU (faiss default). This matches the paper's setup.
- The key claim to verify: TurboQuant is ~1000× faster than PQ.
- Absolute times will differ from the paper (3060 vs A100), but the order-of-magnitude gap should reproduce.
