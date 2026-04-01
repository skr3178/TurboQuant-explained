# criteria to check if the distribution passes: 


## Figure 1 — Basic sanity check (second priority)

Simpler to pass — just checks that both quantizers produce
roughly Gaussian error distributions, and that variance shrinks
with increasing bitwidth. You already have code for this.

## Figure 2:   Figure 2 — Core theoretical claim (highest priority)

This directly tests whether TurboQuantProd is actually working
correctly. The paper's central claim is:

- TurboQuant_prod (row a): histogram width stays constant across
all 4 avg IP values → variance of IP error is independent of the
actual inner product magnitude
- TurboQuant_mse (row b): histogram gets wider as avg IP grows →
variance increases with inner product magnitude

If your TurboQuantProd implementation is correct, you'll see this
behaviour. If both rows look the same, the QJL residual
correction isn't working right.

## Figure 3 — Theoretical bounds check (third)

Validates that empirical MSE falls between the paper's
upper/lower bounds. Also already coded. Good for sanity but less
discriminating.

## Figure 5 — Competitive comparison (last)

Requires external baselines (PQ, RabitQ) and large datasets. Only
meaningful once Figures 1–3 are confirmed.


Here's a summary of all the figures and tables in the paper:

## Figures

| Figure | Content | Priority |
|--------|---------|----------|
| **Fig 1** | Error distributions (TurboQuant_prod vs _mse) across bitwidths 1–4 | 2nd |
| **Fig 2** | Variance vs avg IP — the core theoretical claim | **Highest** |
| **Fig 3** | Empirical MSE vs theoretical upper/lower bounds | 3rd |
| **Fig 4** | Needle-In-A-Haystack heatmaps (SnapKV, KIVI, PolarQuant, TurboQuant, etc.) | — |
| **Fig 5** | Recall@k vs Top-k on GloVe, OpenAI3 datasets | Last |

## Tables

| Table | Content | Notes |
|-------|---------|-------|
| **Table 1** | LongBench-V1 results — Full Cache vs KIVI vs PolarQuant vs **TurboQuant** (2.5×, 3.5× KV size) on Llama-3.1-8B-Instruct and Ministral-7B | TurboQuant at 3.5× matches Full Cache average (50.06) |
| **Table 2** | Quantization time (seconds) at d=200, 1536, 3072 — **TurboQuant is 500–1000× faster** than PQ/RabitQ | Critical competitive claim |

## Key takeaways

**Table 2** is arguably the most important table — it's TurboQuant's killer result: `0.0007s` vs `507s` (PQ) at d=1536. That's the speed claim the paper is built around.

**Table 1** shows TurboQuant achieves competitive NLP task performance at 3.5× compression (matching Full Cache), validating the KV cache application.

For your current validation work, the priority order from `passing_criteria.md` holds: **Fig 2 > Fig 1 > Fig 3 > Fig 5**. Tables 1 & 2 require full LLM inference and competitive baselines respectively, so they come much later.