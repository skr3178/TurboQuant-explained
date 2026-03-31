# Implementation Details

## Overview

Key topics covered:
- MSE (Mean Square Error)
- Results / graphs
- Equations / math
- Potential datasets

**2-Stage process:**
1. Stage 1 — MSE-optimal quantization
2. Stage 2 — Inner-product correction via QJL residual

---

## Datasets

| ID | Description | Link |
|----|-------------|------|
| D1 | DBpedia OpenAI embeddings — 1536-dim, 1M vectors | [HuggingFace](https://huggingface.co/datasets/Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M) |
| D2 | DBpedia OpenAI embeddings — 3072-dim, 1M vectors | [HuggingFace](https://huggingface.co/datasets/Qdrant/dbpedia-entities-openai3-text-embedding-3-large-3072-1M) |
| D3 | GloVe 6B embeddings | [Stanford](https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip) |
| D4 | Needle in a Haystack benchmark | [GitHub](https://github.com/gkamradt/LLMTest_NeedleInAHaystack) |

---

## Models

| # | Model | Link |
|---|-------|------|
| 1 | `meta-llama/Llama-3.1-8B-Instruct` (FP16) | [HuggingFace](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) |
| 2 | `bartowski/Meta-Llama-3.1-8B-Instruct-GGUF` (quantized) | [HuggingFace](https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF) |

---

## Experimental Setup

**Quantization targets:** 2.5-bit and 3.5-bit

Following the experimental setup of Fu et al. [21], evaluations use the `Llama-3.1-8B-Instruct` model — a sweet spot between what was available, affordable, and convincing for an academic research paper.

### Reference Figures

![Fig 1: Error distribution](../images/Fig_1:%20Error%20distribution.png)

![Fig 2: TurboQuant graph](../images/Fig2_turboquant_graph.png)

---

## Why Shannon's Source Coding Entropy Matters

> How much can you compress information without losing it — and what's the absolute limit?

Shannon asked: given a budget of B bits, what's the minimum distortion you can possibly achieve? The answer is the **distortion-rate function D(B)**, and no algorithm can do better — ever.

TurboQuant's distortion is within a factor of **~2.7** of Shannon's theoretical limit. This means you physically cannot do much better regardless of how clever your algorithm is.

**Core problem:** LLMs are memory-bound, not compute-bound.

---

## What Exactly Is Cached? (KV Cache)

The KV cache stores pre-computed **Keys** and **Values** for each past token.

- When generating token N, the model compares the current query against every previous token's key, then weighted-sums their values.
- Without caching, K and V would be recomputed for all previous tokens at every single step.
- The KV cache saves those already-computed vectors to avoid recomputation.

### KV Cache Memory Example

**Llama 3.1 8B:** 32 layers, 8 KV heads, head_dim = 128, FP16 = 2 bytes/float

```
Per token = layers × heads × 2 (K and V) × head_dim × bytes
           = 32 × 8 × 2 × 128 × 2 bytes
           = 131 KB / token

For 128K context = 131 KB × 131,072 ≈ 16 GB
```

---

## Shannon Lower Bound (Lemma 3)

The absolute minimum distortion achievable given a specific bit budget:

```
D(B) ≥ 2^(−2B/d)
```

**Where:**
- `D(B)` = minimum achievable distortion (MSE)
- `B` = total bit budget
- `d` = dimension of the vector

> **Key insight:** Double your bit budget (B → 2B) and the distortion drops by 4×.
> This is the fundamental compression–quality tradeoff — you pay exponentially in bits to gain linearly in quality.

---

## Pipeline

```
vector → random rotation → Beta distribution (Lemma 1) → Gaussian (high-d) → scalar quantization
```

---

## Stage 1: MSE-Optimal Quantization

**Key idea:** Store a high-dimensional vector into few bits while preserving the most important information.

Growing KV cache sizes in transformers make this critical — we need quantization that preserves both MSE and inner-product structure.

### Distortion Metrics

**MSE distortion** — how close the reconstructed vector is to the original:

```
D_mse = E[ ||x − Q⁻¹(Q(x))||² ]
```

**Inner-product distortion** — how much quantization alters stored information:

```
D_prod = E[ |⟨y, x⟩ − ⟨y, Q⁻¹(Q(x))⟩|² ]
```

Nearest-neighbour / vector search using cosine similarity must remain intact after quantization.

**Primitives:**
- `Q` — Quantizer
- `Q⁻¹` — DeQuantizer

---

## Key Ideas

1. **Random rotation** — makes any vector's coordinates statistically predictable
2. **Scalar quantization per coordinate** — enabled by near-independence after rotation
3. **Inner-product residual correction** — QJL step restores unbiasedness

---

## Stage 2: Why Random Rotation Helps

Consider a KV cache vector with all energy in one dimension:

```
x = [0.98, 0.02, 0.01, 0.003, …]
```

A naïve quantizer would need a different codebook per dimension. Most dimensions are near-zero, a few are high-value — this uneven energy distribution is hard to quantize uniformly.

**Solution:** Apply a random rotation generated from a random Gaussian matrix via QR decomposition to obtain an orthogonal matrix. This spreads energy evenly across all dimensions.

At `d = 128` and above, after rotation we get:
- Beta distribution approximates Gaussian
- Near-independence between coordinates
- Concentrated, symmetric coordinate values

---



## Lemma 1: Coordinate Distribution After Rotation

For a vector uniformly distributed on the unit hypersphere, each coordinate follows a **Beta-related distribution:**

```
f_X(x) = Γ(d/2) / (√π · Γ((d−1)/2)) · (1 − x²)^((d−3)/2)
```

**Where:**
- `f_X(x)` is the probability density function (PDF)
- `Γ(·)` is the gamma function: `Γ(n) = (n−1)!`
- The term `(1 − x²)^((d−3)/2)` controls the shape

In high dimensions (`d → ∞`), this converges to `N(0, 1/d)`.

### Intuition: 3D Sphere Example

For a 3D sphere (`d = 3`): `(1 − x²)^((3−3)/2) = (1 − x²)^0 = 1`

The density is **flat** on `[−1, 1]` — one coordinate of a 3D unit sphere is uniformly distributed.

Think of the Earth:
- Fix the z-axis (latitude)
- At any latitude, the remaining coordinates lie on a circle
- **Poles** → small circle (low probability)
- **Equator** → large circle (higher probability)

### Why Gaussian Properties Are Valuable

| Property | Benefit |
|----------|---------|
| Symmetry around 0 | Simple codebook; few parameters needed |
| Most values near 0 | Many small values → easy to compress |
| Few large values | Can tolerate larger error at the tails |
| Separable structure | Hard d-dimensional quantization → easy 1D quantization |

---


# Stage 3:

## Why Lloyl-Max is the right tool:

- How do we optimally quantize one scalar drawn from a known distribution?
- Find optimal quantization levels that minimize mean squared error (MSE)

Inputs: 
- Probability distribution
- No. of quantization levels K

Outputs: 
- Optimal thresholds
- Optimal reconstruction values



## Other Key Ideas

| Idea | Description |
|------|-------------|
| **Idea 1** | Random rotation makes any vector's coordinates statistically predictable |
| **Idea 2** | Lloyd-Max — optimal scalar quantization for a known distribution |
| **Idea 3** | Why MSE quantizers are biased for inner products |
| **Idea 4** | QJL — the residual fix that restores unbiasedness |

---

## Simple Experiment: Verifying Lemma 1

1. Draw `g ~ N(0, I₃)`
2. Normalize: `x = g / ||g||₂`
3. Record the first coordinate `x₁`
4. Plot histogram → should be uniform on `[−1, 1]` for `d = 3`

---

## How Is the Rotation Matrix R Generated?

- Drawn randomly from a Gaussian distribution
- Must be **orthogonal**: `R Rᵀ = I`
- Method: QR decomposition of a random `N(0,1)` matrix
