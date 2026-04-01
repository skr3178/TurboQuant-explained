"""
Figure 1: Inner-product error distribution histograms (2 rows × 4 cols).

- Rows: TurboQuant_prod (top) vs TurboQuant_mse (bottom)
- Cols: bitwidth b = 1, 2, 3, 4
- X-axis: IP distortion [-0.1, 0.1]; Y-axis: Frequency
- Dataset: DBpedia 1536d, 1M database, 1K queries
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from experiments.data_utils import load_dbpedia_1536_1M
from experiments.eval_metrics import inner_product_errors_flat
from turboquant.quantizer import TurboQuantMSE, TurboQuantProd

# Per-bitwidth colors matching the paper (teal, blue, orange, dark green)
BITWIDTH_COLORS = {1: "#4DB8C8", 2: "#6B7FD1", 3: "#C87F3C", 4: "#4A7A4A"}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results_dir = os.path.join(os.path.dirname(__file__), "results", "phase4_dbpedia_1536_1M")
    os.makedirs(results_dir, exist_ok=True)

    print("Loading DBpedia 1536d 1M...")
    database, queries = load_dbpedia_1536_1M()
    database = database.to(device)
    queries = queries.to(device)
    d = database.shape[1]

    bit_widths = [1, 2, 3, 4]
    methods = [
        ("TurboQuant_prod", TurboQuantProd),
        ("TurboQuant_mse", TurboQuantMSE),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    fig.patch.set_facecolor("white")

    for col, b in enumerate(bit_widths):
        print(f"\n--- b={b} ---")

        q_mse = TurboQuantMSE(d=d, b=b, device=device)
        idx_mse = q_mse.quantize(database)
        recon_mse = q_mse.dequantize(idx_mse)

        q_prod = TurboQuantProd(d=d, b=b, device=device)
        idx_p, qjl, gamma = q_prod.quantize(database)
        recon_prod = q_prod.dequantize(idx_p, qjl, gamma)

        for row, (name, _) in enumerate(methods):
            recon = recon_prod if row == 0 else recon_mse
            print(f"  Computing errors for {name}...")
            errors = inner_product_errors_flat(database, recon, queries)

            ax = axes[row, col]
            ax.set_facecolor("white")
            ax.hist(errors, bins=200, range=(-0.1, 0.1),
                    color=BITWIDTH_COLORS[b], alpha=0.85, edgecolor="none")
            ax.set_title(f"Bitwidth = {b}", fontsize=10)
            ax.set_xlabel("Inner Product Distortion", fontsize=8)
            ax.set_xlim(-0.1, 0.1)
            ax.tick_params(labelsize=7)
            ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

            if col == 0:
                ax.set_ylabel("Frequency", fontsize=8)
            else:
                ax.set_ylabel("")

            for spine in ax.spines.values():
                spine.set_color("#aaa")
                spine.set_linewidth(0.5)
            ax.tick_params(colors="black", length=3)

    # Row section headers
    row_labels = ["(a)  $\\mathrm{TurboQuant_{prod}}$", "(b)  $\\mathrm{TurboQuant_{mse}}$"]
    for row, label in enumerate(row_labels):
        ax0 = axes[row, 0]
        fig.text(
            ax0.get_position().x0 - 0.07,
            (ax0.get_position().y0 + ax0.get_position().y1) / 2,
            label,
            ha="right", va="center", fontsize=10, rotation=0,
            transform=fig.transFigure,
        )

    plt.tight_layout(rect=[0.07, 0, 1, 1])
    path = os.path.join(results_dir, "Fig_1_error_histograms.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
