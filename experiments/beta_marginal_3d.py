"""
Verify the beta marginal distribution on a 3D sphere.

Sample random points uniformly on S^2, record the first coordinate x1,
and compare the histogram against the theoretical beta marginal PDF.

For d=3, the PDF simplifies to f(x) = 0.5 (uniform on [-1, 1]).
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.special import gammaln


def main():
    n = 1_000_000
    d = 3

    # Sample uniform points on the d-sphere
    g = torch.randn(n, d)
    x = g / g.norm(dim=1, keepdim=True)
    x1 = x[:, 0].numpy()

    # Theoretical beta marginal PDF
    log_coeff = gammaln(d / 2) - math.log(math.pi) / 2 - gammaln((d - 1) / 2)
    coeff = math.exp(log_coeff)
    print(f"Beta marginal coeff for d={d}: {coeff:.6f}")

    xs = np.linspace(-1, 1, 500)
    pdf = coeff * (1 - xs**2) ** ((d - 3) / 2)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(x1, bins=200, density=True, alpha=0.6, color="#6ee7b7", label="Sampled x\u2081")
    ax.plot(
        xs, pdf, "r-", linewidth=2,
        label=f"Beta marginal (d={d})\nf(x) = {coeff:.1f} (uniform!)",
    )
    ax.set_xlabel("x\u2081", fontsize=13)
    ax.set_ylabel("Density", fontsize=13)
    ax.set_title(f"Beta Marginal Distribution on {d}D Sphere (n={n:,})", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.patch.set_facecolor("#0e0e0e")
    ax.set_facecolor("#0e0e0e")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_color("#333")
    ax.legend(facecolor="#1a1a1a", edgecolor="#333", labelcolor="white")
    plt.tight_layout()

    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "beta_marginal_3d.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")

    # Stats
    print(f"Mean x\u2081: {x1.mean():.6f} (expected: 0)")
    print(f"Std  x\u2081: {x1.std():.6f} (expected: {1/math.sqrt(d):.6f})")
    print(f"Min  x\u2081: {x1.min():.6f}")
    print(f"Max  x\u2081: {x1.max():.6f}")


if __name__ == "__main__":
    main()