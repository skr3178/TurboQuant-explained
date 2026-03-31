# Use Case: Similarity Search / Attention

Suppose:
- Query vector: **q**
- Database vector: **x**

We care about inner product accuracy: **q^T x**

But we only store compressed vector: **x_tilde = x_tilde_mse + r_tilde**

So we compute: **q^T x_tilde** to approximate q^T x.

---

## Stage 4 Fixes Bias

Stage 4 uses:

```
x_tilde = x_tilde_mse + r_tilde
```

where the residual estimator satisfies:

```
E[r_tilde] = x - x_tilde_mse
```

This is the key unbiased property.

---

## Inner Product Accuracy

Now compute expectation:

```
E[q^T x_tilde] = q^T E[x_tilde] = q^T x
```

Since:

```
r = x - x_tilde_mse
```

Then:

```
E[x_tilde] = x_tilde_mse + E[r_tilde]
           = x_tilde_mse + (x - x_tilde_mse)
           = x
```

So: **E[x_tilde] = x**

This is the key unbiased property.

---

So TurboQuant ensures:
- Unbiased inner product
- Unbiased similarity
