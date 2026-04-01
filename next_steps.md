
# 🧭 First — Do you need GPU?

**Short answer: No (for most figures)**

| Figure   | GPU Needed?         | Why                      |
| -------- | ------------------- | ------------------------ |
| Figure 1 | ❌ No                | Just vector operations   |
| Figure 2 | ❌ No                | Same as figure 1         |
| Figure 3 | ❌ No                | MSE + inner product only |
| Figure 5 | ⚠️ Maybe (optional) | Large dataset helpful    |
| Figure 4 | ✅ Yes               | Requires LLM inference   |

So you can **validate TurboQuant without GPU**.

This is actually one of the nice things about this paper.

---

# 🧠 Which Dataset for Which Figure

Here's the **exact mapping**:

# 📊 Dataset → Figure Mapping

## 🥇 Figure 1 — Bias vs Unbiased (Histogram)

Use:

**D1 — DBpedia 1536-dim** (recommended)

or

**D3 — GloVe** (lighter & easier)

Paper used:

* DBpedia OpenAI embeddings (1536-dim)


### My recommendation:

Start with **GloVe (D3)**
It's:

* smaller
* faster
* easier to debug

---

## 🥇 Figure 3 — MSE + Inner Product vs Bitwidth

Use:

**D1 — DBpedia 1536-dim** (paper used)

But you can also use:

* D3 — GloVe (perfectly fine)

Again:
**You don't need 1M samples**
Even **10k samples works**

---

## 🥉 Figure 5 — Nearest Neighbor Recall

Use:

Paper used:

* D3 — GloVe (200-dim)
* D1 — DBpedia 1536
* D2 — DBpedia 3072


You can start with:

**Only GloVe (D3)**

No need to use all three.

---

## ❌ Figure 4 — Needle-in-Haystack

Use:

**D4 — Needle benchmark**

BUT:

This requires:

* LLM inference
* KV cache modification
* GPU

Skip this unless you're reproducing full paper.

---

# 🧠 Practical Recommendation (Best Setup)

Start with:

### Step 1 (Lightweight)

Use:
**D3 — GloVe**

Recreate:

* Figure 1
* Figure 3

No GPU needed.

---

### Step 2 (Better Validation)

Use:
**D1 — DBpedia 1536**

Recreate:

* Figure 3 again
* Figure 5 (optional)

---

# 💻 Do you need full dataset?

No.

Paper uses:

* 100k samples
* 1k queries

You can use:

* 10k samples
* 100 queries

Works fine.

---

# ⚡ Fastest Validation Plan

Use:

Dataset:

* **GloVe (D3)**

Recreate:

1. Figure 3
2. Figure 1

This is enough to validate implementation.

---

# My Honest Recommendation For You

Given your deep interest in the algorithm:

Start with:

1. GloVe dataset
2. Figure 3
3. Figure 1

Then optionally:

4. Figure 5

Skip:

* Figure 4
* Table 1
* Table 2

---

If you'd like, I can also:

* Help pick GloVe version (there are multiple)
* Help with dataset download
* Help design experiment code

## Explaination for figure 1: 
Yes, it's directly caused by the number of sample pairs evaluated. The y-axis frequency is just a count of how many (query, db) error samples fall in each bin.

**Our setup:**
- n_db = 99,000, n_query = 1,000
- Total error samples = 99K × 1K = **99M**
- Peak ≈ `99M × bin_width / (σ × √(2π))` ≈ **1.2×10⁶** ✓

**Paper's implied setup:**
- Peak ≈ 1.5×10⁷ with similar σ → implies **~1.2 billion** total pairs
- That's roughly 10× more, consistent with n_query = **10,000** (instead of 1,000)

So the paper likely uses 100K db × **10K queries** = 1B pairs, whereas our `load_dbpedia_1536` defaults to `n_query=1_000`.

**The distribution shapes are correct** — widths, symmetry, relative narrowing across bitwidths all match. It's purely a volume difference. To match the paper's absolute scale you'd change:

```python
database, queries = load_dbpedia_1536(n_query=10_000)
```

But this would require 10× more GPU memory and compute time for the error computation. The scientific content is identical either way.

# Figures ordered in importance: 

**Figure 2 is the most important one to validate first.** Here's why:

---

## Validation priority

### Figure 2 — **Core theoretical claim** (highest priority)
This directly tests whether `TurboQuantProd` is actually working correctly. The paper's central claim is:

- **TurboQuant_prod** (row a): histogram width stays **constant** across all 4 avg IP values → variance of IP error is independent of the actual inner product magnitude
- **TurboQuant_mse** (row b): histogram gets **wider** as avg IP grows → variance increases with inner product magnitude

If your `TurboQuantProd` implementation is correct, you'll see this behaviour. If both rows look the same, the QJL residual correction isn't working right.

### Figure 1 — **Basic sanity check** (second priority)
Simpler to pass — just checks that both quantizers produce roughly Gaussian error distributions, and that variance shrinks with increasing bitwidth. You already have code for this.

### Figure 3 — **Theoretical bounds check** (third)
Validates that empirical MSE falls between the paper's upper/lower bounds. Also already coded. Good for sanity but less discriminating.

### Figure 5 — **Competitive comparison** (last)
Requires external baselines (PQ, RabitQ) and large datasets. Only meaningful once Figures 1–3 are confirmed.

---

**Recommendation**: Run Figure 2 first. It's the acid test. If the histograms in row (a) all have the same width regardless of avg IP, your implementation is correct. Want me to check the existing `fig2_grouped_histograms.py` and make sure it's set up right?