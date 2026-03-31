"""
Reproduce Figure 3: MSE distortion vs bit-width on synthetic unit-sphere data.

Generates random unit vectors in R^1536, quantizes with TurboQuantMSE at
b=1,2,3,4 bits, and plots empirical MSE against theoretical bounds.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import matplotlib.pyplot as plt

from turboquant.codebook import compute_theoretical_bounds
from turboquant.quantizer import TurboQuantMSE


def sample_unit_vectors(n: int, d: int, device: str = "cuda") -> torch.Tensor:
    """Sample n uniform random unit vectors in R^d."""
    x = torch.randn(n, d, device=device, dtype=torch.float32)
    x /= x.norm(dim=1, keepdim=True)
    return x


def main():
    d = 1536
    n_samples = 10_000
    bit_widths = [1, 2, 3, 4]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}, d={d}, n_samples={n_samples}")

    x = sample_unit_vectors(n_samples, d, device)
    print(f"Sampled {n_samples} unit vectors, mean norm = {x.norm(dim=1).mean():.6f}")

    empirical_mse = []
    upper_bounds = []
    lower_bounds = []

    for b in bit_widths:
        print(f"\n--- b={b} ---")
        tq = TurboQuantMSE(d, b, seed=42, device=device)

        idx = tq.quantize(x)
        x_recon = tq.dequantize(idx)

        # Paper defines distortion as E[||x - x̃||²] (per-vector), not per-element
        mse = ((x - x_recon) ** 2).sum(dim=-1).mean().item()
        empirical_mse.append(mse)

        upper, lower = compute_theoretical_bounds(b)
        upper_bounds.append(upper)
        lower_bounds.append(lower)

        print(f"  Codebook: {tq.codebook.cpu().numpy()}")
        print(f"  Empirical MSE: {mse:.6f}")
        print(f"  Upper bound:   {upper:.6f}")
        print(f"  Lower bound:   {lower:.6f}")
        assert lower <= mse * 1.2, f"MSE {mse} below lower bound {lower}!"
        assert mse <= upper * 1.2, f"MSE {mse} above upper bound {upper}!"

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    bit_range = np.arange(1, 5)

    ax.semilogy(bit_range, empirical_mse, "o-", color="#6ee7b7", linewidth=2.5,
                markersize=10, label="Empirical MSE", zorder=3)
    ax.semilogy(bit_range, upper_bounds, "s--", color="#f87171", linewidth=1.5,
                markersize=7, label="Upper bound (√3π/2)·4⁻ᵇ", zorder=2)
    ax.semilogy(bit_range, lower_bounds, "^--", color="#60a5fa", linewidth=1.5,
                markersize=7, label="Lower bound 4⁻ᵇ", zorder=2)

    ax.set_xlabel("Bit-width b", fontsize=13)
    ax.set_ylabel("MSE Distortion (log scale)", fontsize=13)
    ax.set_title(
        f"TurboQuant MSE vs Bit-width (d={d}, n={n_samples}, synthetic)",
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
    out_path = os.path.join(results_dir, "Fig_3_synthetic.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_path}")

    # Summary
    print("\n=== Summary ===")
    for i, b in enumerate(bit_widths):
        print(f"b={b}: MSE={empirical_mse[i]:.6f}  bounds=[{lower_bounds[i]:.6f}, {upper_bounds[i]:.6f}]")


if __name__ == "__main__":
    main()
