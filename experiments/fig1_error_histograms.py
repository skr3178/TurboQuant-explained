"""
Figure 1: Inner-product error distribution histograms (2 rows × 4 cols).

- Rows: TurboQuant_prod (top) vs TurboQuant_mse (bottom)
- Cols: bitwidth b = 1, 2, 3, 4
- X-axis: IP distortion [-0.1, 0.1]; Y-axis: Frequency
- Dataset: DBpedia 1536d, 100K database, 1K queries
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import matplotlib.pyplot as plt

from experiments.data_utils import load_dbpedia_1536
from experiments.eval_metrics import inner_product_errors_flat
from turboquant.quantizer import TurboQuantMSE, TurboQuantProd


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    print("Loading DBpedia 1536d...")
    database, queries = load_dbpedia_1536()
    database = database.to(device)
    queries = queries.to(device)
    d = database.shape[1]

    bit_widths = [1, 2, 3, 4]
    methods = [
        ("TurboQuant_prod", TurboQuantProd),
        ("TurboQuant_mse", TurboQuantMSE),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(20, 8), sharey="row")
    fig.suptitle(
        "Figure 1: Inner-Product Error Distribution (DBpedia 1536d, 100K×1K)",
        fontsize=14, fontweight="bold", color="white",
    )

    for col, b in enumerate(bit_widths):
        print(f"\n--- b={b} ---")

        # Compute MSE reconstruction (needed for both methods)
        q_mse = TurboQuantMSE(d=d, b=b, device=device)
        idx_mse = q_mse.quantize(database)
        recon_mse = q_mse.dequantize(idx_mse)

        # Compute Prod reconstruction (requires b >= 2)
        if b >= 2:
            q_prod = TurboQuantProd(d=d, b=b, device=device)
            idx_p, qjl, gamma = q_prod.quantize(database)
            recon_prod = q_prod.dequantize(idx_p, qjl, gamma)

        for row, (name, _) in enumerate(methods):
            if row == 0 and b < 2:
                axes[row, col].text(0.5, 0.5, "N/A (b<2)", transform=axes[row, col].transAxes,
                                    ha="center", va="center", fontsize=12, color="#666")
                axes[row, col].set_title(f"Bitwidth = {b}", fontsize=11, color="white")
                if col == 0:
                    axes[row, col].set_ylabel(name, fontsize=12, fontweight="bold", color="white")
                axes[row, col].set_facecolor("#0e0e0e")
                axes[row, col].tick_params(colors="white")
                for spine in axes[row, col].spines.values():
                    spine.set_color("#333")
                continue
            recon = recon_prod if row == 0 else recon_mse
            print(f"  Computing errors for {name}...")
            errors = inner_product_errors_flat(database, recon, queries)

            ax = axes[row, col]
            ax.hist(errors, bins=200, range=(-0.1, 0.1), color="#6ee7b7", alpha=0.8,
                    edgecolor="none")
            ax.set_title(f"Bitwidth = {b}", fontsize=11, color="white")
            if col == 0:
                ax.set_ylabel(name, fontsize=12, fontweight="bold", color="white")
            ax.set_xlabel("IP distortion", color="white")

            mean_e = errors.mean()
            std_e = errors.std()
            ax.annotate(
                f"μ={mean_e:.4f}\nσ={std_e:.4f}",
                xy=(0.97, 0.95), xycoords="axes fraction",
                ha="right", va="top", fontsize=9, color="white",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1a1a", edgecolor="#444"),
            )

            fig.axes[0].get_figure().patch.set_facecolor("#0e0e0e")
            ax.set_facecolor("#0e0e0e")
            ax.tick_params(colors="white")
            for spine in ax.spines.values():
                spine.set_color("#333")

    plt.tight_layout()
    path = os.path.join(results_dir, "Fig_1_error_histograms.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
