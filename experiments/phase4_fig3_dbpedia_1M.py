"""
Figure 3: Distortion vs bitwidth on DBpedia 1536d 1M (2 panels).

Panel (a): Inner-product distortion D_prod — TurboQuant_mse and TurboQuant_prod + bounds
Panel (b): MSE distortion D_mse — TurboQuant_mse + bounds
Bit-widths b = 1..5, log scale.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import matplotlib.pyplot as plt

from experiments.data_utils import load_dbpedia_1536_1M
from experiments.eval_metrics import mse_distortion, ip_distortion
from turboquant.codebook import compute_theoretical_bounds
from turboquant.quantizer import TurboQuantMSE, TurboQuantProd


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results_dir = os.path.join(os.path.dirname(__file__), "results", "phase4_dbpedia_1536_1M")
    os.makedirs(results_dir, exist_ok=True)

    print("Loading DBpedia 1536d 1M...")
    database, queries = load_dbpedia_1536_1M()
    database = database.to(device)
    queries = queries.to(device)
    d = database.shape[1]

    bit_widths = [1, 2, 3, 4, 5]

    d_mse_vals = []
    d_prod_mse_vals = []
    d_prod_prod_vals = []
    d_prod_prod_b = []  # bit widths where prod is valid (b >= 2)

    for b in bit_widths:
        print(f"\n--- b={b} ---")

        # TurboQuantMSE
        q_mse = TurboQuantMSE(d=d, b=b, device=device)
        idx_mse = q_mse.quantize(database)
        recon_mse = q_mse.dequantize(idx_mse)

        mse_val = mse_distortion(database, recon_mse)
        d_mse_vals.append(mse_val)
        print(f"  TurboQuant_mse: D_mse = {mse_val:.6f}")

        d_prod_mse = ip_distortion(database, recon_mse, queries)
        d_prod_mse_vals.append(d_prod_mse)
        print(f"  TurboQuant_mse: D_prod = {d_prod_mse:.8f}")

        # TurboQuantProd (b >= 2)
        if b >= 2:
            q_prod = TurboQuantProd(d=d, b=b, device=device)
            idx_p, qjl, gamma = q_prod.quantize(database)
            recon_prod = q_prod.dequantize(idx_p, qjl, gamma)

            d_prod_prod = ip_distortion(database, recon_prod, queries)
            d_prod_prod_vals.append(d_prod_prod)
            d_prod_prod_b.append(b)
            print(f"  TurboQuant_prod: D_prod = {d_prod_prod:.8f}")

    # Theoretical bounds
    b_arr = np.array(bit_widths, dtype=float)
    mse_lower = 4.0 ** (-b_arr)
    mse_upper = (np.sqrt(3) * np.pi / 2) * 4.0 ** (-b_arr)

    # D_prod bounds for unit-norm x,y on d-sphere
    prod_lower = (1.0 / d) * 4.0 ** (-b_arr)
    prod_upper = (np.sqrt(3) * np.pi**2 / d) * 4.0 ** (-b_arr)

    # --- Plot ---
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Figure 3: Distortion vs Bit-width (DBpedia 1536d, 1M vectors)",
        fontsize=14, fontweight="bold", color="white",
    )

    # Panel (a): D_prod
    ax_a.semilogy(b_arr, d_prod_mse_vals, "o-", color="#60a5fa", linewidth=2.5,
                  markersize=9, label="TurboQuant_mse", zorder=3)
    if d_prod_prod_vals:
        ax_a.semilogy(d_prod_prod_b, d_prod_prod_vals, "s-", color="#f472b6", linewidth=2.5,
                      markersize=9, label="TurboQuant_prod", zorder=3)
    ax_a.semilogy(b_arr, prod_lower, "^--", color="#6ee7b7", linewidth=1.5,
                  markersize=7, label="Lower bound", zorder=2)
    ax_a.semilogy(b_arr, prod_upper, "v--", color="#f87171", linewidth=1.5,
                  markersize=7, label="Upper bound", zorder=2)
    ax_a.set_xlabel("Bit-width b", fontsize=13)
    ax_a.set_ylabel("D_prod (IP distortion)", fontsize=13)
    ax_a.set_title("(a) Inner-Product Distortion", fontsize=12, color="white")
    ax_a.set_xticks(bit_widths)
    ax_a.legend(fontsize=10, facecolor="#1a1a1a", edgecolor="#333", labelcolor="white")
    ax_a.grid(True, alpha=0.3)

    # Panel (b): D_mse
    ax_b.semilogy(b_arr, d_mse_vals, "o-", color="#60a5fa", linewidth=2.5,
                  markersize=9, label="TurboQuant_mse", zorder=3)
    ax_b.semilogy(b_arr, mse_lower, "^--", color="#6ee7b7", linewidth=1.5,
                  markersize=7, label="Lower bound 4$^{-b}$", zorder=2)
    ax_b.semilogy(b_arr, mse_upper, "v--", color="#f87171", linewidth=1.5,
                  markersize=7, label="Upper bound ($\\sqrt{3}\\pi/2$)$\\cdot$4$^{-b}$", zorder=2)
    ax_b.set_xlabel("Bit-width b", fontsize=13)
    ax_b.set_ylabel("D_mse (reconstruction distortion)", fontsize=13)
    ax_b.set_title("(b) MSE Distortion", fontsize=12, color="white")
    ax_b.set_xticks(bit_widths)
    ax_b.legend(fontsize=10, facecolor="#1a1a1a", edgecolor="#333", labelcolor="white")
    ax_b.grid(True, alpha=0.3)

    for ax in [ax_a, ax_b]:
        fig.axes[0].get_figure().patch.set_facecolor("#0e0e0e")
        ax.set_facecolor("#0e0e0e")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_color("#333")

    plt.tight_layout()
    path = os.path.join(results_dir, "Fig_3_dbpedia_1M.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {path}")

    # Summary table
    print("\n=== Summary ===")
    print(f"{'b':>3} {'D_mse':>12} {'D_prod(mse)':>14} {'D_prod(prod)':>14}")
    for i, b in enumerate(bit_widths):
        prod_val = d_prod_prod_vals[d_prod_prod_b.index(b)] if b in d_prod_prod_b else float("nan")
        print(f"{b:>3} {d_mse_vals[i]:>12.6f} {d_prod_mse_vals[i]:>14.8f} {prod_val:>14.8f}")


if __name__ == "__main__":
    main()
