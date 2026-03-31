"""
Validate TurboQuantMSE on real GloVe 300d embeddings.

Reproduces MSE vs bit-width analysis with theoretical bounds,
comparing synthetic (unit-sphere) vs real data performance.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import matplotlib.pyplot as plt

from turboquant.codebook import compute_theoretical_bounds
from turboquant.quantizer import TurboQuantMSE
from experiments.data_utils import load_glove
from experiments.eval_metrics import mse_distortion


def main():
    dim = 300
    n_vectors = 50_000
    bit_widths = [1, 2, 3, 4]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}, dim={dim}, n_vectors={n_vectors}")

    # Load and normalize GloVe
    words, vectors = load_glove(dim=dim, n=n_vectors, normalize=True)
    vectors = vectors.to(device)

    empirical_mse = []
    upper_bounds = []
    lower_bounds = []

    for b in bit_widths:
        print(f"\n--- b={b} ---")
        tq = TurboQuantMSE(dim, b, seed=42, device=device)

        idx = tq.quantize(vectors)
        x_recon = tq.dequantize(idx)

        mse = mse_distortion(vectors, x_recon)
        empirical_mse.append(mse)

        upper, lower = compute_theoretical_bounds(b)
        upper_bounds.append(upper)
        lower_bounds.append(lower)

        print(f"  Codebook: {tq.codebook.cpu().numpy()}")
        print(f"  Empirical MSE: {mse:.6f}")
        print(f"  Upper bound:   {upper:.6f}")
        print(f"  Lower bound:   {lower:.6f}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    bit_range = np.arange(1, 5)

    ax.semilogy(bit_range, empirical_mse, "o-", color="#6ee7b7", linewidth=2.5,
                markersize=10, label="Empirical MSE (GloVe 300d)", zorder=3)
    ax.semilogy(bit_range, upper_bounds, "s--", color="#f87171", linewidth=1.5,
                markersize=7, label="Upper bound", zorder=2)
    ax.semilogy(bit_range, lower_bounds, "^--", color="#60a5fa", linewidth=1.5,
                markersize=7, label="Lower bound", zorder=2)

    ax.set_xlabel("Bit-width b", fontsize=13)
    ax.set_ylabel("MSE Distortion (log scale)", fontsize=13)
    ax.set_title(
        f"TurboQuant MSE vs Bit-width on GloVe {dim}d (n={n_vectors})",
        fontsize=14, fontweight="bold",
    )
    ax.set_xticks(bit_widths)
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
    out_path = os.path.join(results_dir, "glove_mse_300d.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_path}")

    # Summary
    print("\n=== Summary ===")
    for i, b in enumerate(bit_widths):
        ratio = empirical_mse[i] / lower_bounds[i]
        print(f"b={b}: MSE={empirical_mse[i]:.6f}  "
              f"bounds=[{lower_bounds[i]:.6f}, {upper_bounds[i]:.6f}]  "
              f"ratio_to_optimal={ratio:.2f}x")


if __name__ == "__main__":
    main()
