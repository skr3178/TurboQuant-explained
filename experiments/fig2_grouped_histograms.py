"""
Figure 2: Error histograms grouped by average inner product (2 rows × 4 cols).

- Rows: TurboQuant_prod (top) vs TurboQuant_mse (bottom)
- Cols: avg IP quartile bins (≈ 0.01, 0.06, 0.10, 0.17)
- Fixed bitwidth b = 2
- Dataset: DBpedia 1536d, 100K database, 1K queries

Key insight: prod histogram width is constant across bins;
            mse width grows with avg IP.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import matplotlib.pyplot as plt

from experiments.data_utils import load_dbpedia_1536
from experiments.eval_metrics import inner_product_errors_flat
from turboquant.quantizer import TurboQuantMSE, TurboQuantProd


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    b = 2

    print("Loading DBpedia 1536d...")
    database, queries = load_dbpedia_1536()
    database = database.to(device)
    queries = queries.to(device)
    d = database.shape[1]

    # Compute avg IP per database vector
    print("Computing avg inner products...")
    avg_ip = []
    chunk = 10_000
    for start in range(0, database.shape[0], chunk):
        end = min(start + chunk, database.shape[0])
        ip = (database[start:end] @ queries.T).mean(dim=1)
        avg_ip.append(ip.cpu().numpy())
    avg_ip = np.concatenate(avg_ip)

    # Split into quartiles
    edges = np.percentile(avg_ip, [0, 25, 50, 75, 100])
    print(f"Avg IP quartile edges: {edges}")

    # Quantize with both methods
    print(f"Quantizing at b={b}...")
    q_mse = TurboQuantMSE(d=d, b=b, device=device)
    idx_mse = q_mse.quantize(database)
    recon_mse = q_mse.dequantize(idx_mse)

    q_prod = TurboQuantProd(d=d, b=b, device=device)
    idx_p, qjl, gamma = q_prod.quantize(database)
    recon_prod = q_prod.dequantize(idx_p, qjl, gamma)

    fig, axes = plt.subplots(2, 4, figsize=(20, 8), sharey="row")
    fig.suptitle(
        f"Figure 2: IP Error by Avg Inner Product (b={b}, DBpedia 1536d)",
        fontsize=14, fontweight="bold", color="white",
    )

    for col in range(4):
        lo, hi = edges[col], edges[col + 1]
        if col == 3:
            mask_np = (avg_ip >= lo) & (avg_ip <= hi)
        else:
            mask_np = (avg_ip >= lo) & (avg_ip < hi)
        mask = torch.from_numpy(mask_np).to(device)
        median_ip = float(np.median(avg_ip[mask_np]))
        n_in_bin = mask_np.sum()
        print(f"  Bin {col}: avg_ip ∈ [{lo:.4f}, {hi:.4f}], median={median_ip:.4f}, n={n_in_bin}")

        db_bin = database[mask]
        recon_mse_bin = recon_mse[mask]
        recon_prod_bin = recon_prod[mask]

        for row, (recon, name) in enumerate([
            (recon_prod_bin, "TurboQuant_prod"),
            (recon_mse_bin, "TurboQuant_mse"),
        ]):
            errors = inner_product_errors_flat(db_bin, recon, queries)

            ax = axes[row, col]
            ax.hist(errors, bins=150, range=(-0.05, 0.05), color="#6ee7b7", alpha=0.8,
                    edgecolor="none")
            ax.set_title(f"Avg IP = {median_ip:.2f}", fontsize=11, color="white")
            if col == 0:
                ax.set_ylabel(name, fontsize=12, fontweight="bold", color="white")
            ax.set_xlabel("IP distortion", color="white")

            std_e = errors.std()
            mean_e = errors.mean()
            ax.annotate(
                f"σ={std_e:.5f}\nμ={mean_e:.5f}",
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
    path = os.path.join(results_dir, "Fig_2_grouped_histograms.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
