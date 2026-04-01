"""
Figure 2: Error histograms grouped by average inner product (2 rows × 4 cols).

- Rows: TurboQuant_prod (top) vs TurboQuant_mse (bottom)
- Cols: quartile bins of the TRUE inner product value <q, x> for each (db, query) pair
- Fixed bitwidth b = 2
- Dataset: DBpedia 1536d, 1M database, 1K queries
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from experiments.data_utils import load_dbpedia_1536_1M
from turboquant.quantizer import TurboQuantMSE, TurboQuantProd

COL_COLORS = ["#4DB8C8", "#6B7FD1", "#C87F3C", "#4A7A4A"]
CHUNK = 5_000


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results_dir = os.path.join(os.path.dirname(__file__), "results", "phase4_dbpedia_1536_1M")
    os.makedirs(results_dir, exist_ok=True)

    b = 2

    print("Loading DBpedia 1536d 1M...")
    database, queries = load_dbpedia_1536_1M()
    database = database.to(device)
    queries = queries.to(device)
    d = database.shape[1]

    # Quantize with both methods
    print(f"Quantizing at b={b}...")
    q_mse = TurboQuantMSE(d=d, b=b, device=device)
    idx_mse = q_mse.quantize(database)
    recon_mse = q_mse.dequantize(idx_mse)

    q_prod = TurboQuantProd(d=d, b=b, device=device)
    idx_p, qjl, gamma = q_prod.quantize(database)
    recon_prod = q_prod.dequantize(idx_p, qjl, gamma)

    # --- Pass 1: sample true IPs to find quartile edges ---
    print("Pass 1: sampling true IPs for quartile edges...")
    sampled_ips = []
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(database.shape[0], size=10_000, replace=False)
    sample_db = database[torch.from_numpy(sample_idx).to(device)]
    ips = (sample_db @ queries.T).cpu().numpy().ravel()
    edges = np.percentile(ips, [0, 25, 50, 75, 100])
    print(f"  True IP range: [{ips.min():.4f}, {ips.max():.4f}]")
    print(f"  Quartile edges: {edges}")

    # --- Pass 2: accumulate errors per bin for each method ---
    print("Pass 2: accumulating errors per IP bin...")
    n_bins = 4
    bins_prod = [[] for _ in range(n_bins)]
    bins_mse  = [[] for _ in range(n_bins)]

    for start in range(0, database.shape[0], CHUNK):
        end = min(start + CHUNK, database.shape[0])
        db_chunk   = database[start:end]
        rprod_chunk = recon_prod[start:end]
        rmse_chunk  = recon_mse[start:end]

        true_ip  = (db_chunk @ queries.T).cpu().numpy()
        err_prod = ((db_chunk - rprod_chunk) @ queries.T).cpu().numpy()
        err_mse  = ((db_chunk - rmse_chunk)  @ queries.T).cpu().numpy()

        for col in range(n_bins):
            lo, hi = edges[col], edges[col + 1]
            mask = (true_ip >= lo) & (true_ip < hi) if col < n_bins - 1 \
                   else (true_ip >= lo) & (true_ip <= hi)
            if mask.any():
                bins_prod[col].append(err_prod[mask].astype(np.float32))
                bins_mse[col].append(err_mse[mask].astype(np.float32))

        if (start // CHUNK) % 50 == 0:
            print(f"  Chunk {start // CHUNK + 1}/{(database.shape[0] + CHUNK - 1) // CHUNK}")

    bins_prod = [np.concatenate(b) for b in bins_prod]
    bins_mse  = [np.concatenate(b) for b in bins_mse]

    for col in range(n_bins):
        lo, hi = edges[col], edges[col + 1]
        med = float(np.median(ips[(ips >= lo) & (ips <= hi)]))
        print(f"  Bin {col}: true_ip in [{lo:.4f}, {hi:.4f}], "
              f"median={med:.4f}, n_prod={len(bins_prod[col]):,}, n_mse={len(bins_mse[col]):,}")

    # --- Plot ---
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    fig.patch.set_facecolor("white")

    for col in range(n_bins):
        lo, hi = edges[col], edges[col + 1]
        med_ip = float(np.median(ips[(ips >= lo) & (ips <= hi)]))

        for row, (errors, name) in enumerate([
            (bins_prod[col], "TurboQuant_prod"),
            (bins_mse[col],  "TurboQuant_mse"),
        ]):
            ax = axes[row, col]
            ax.set_facecolor("white")
            ax.hist(errors, bins=150, range=(-0.05, 0.05),
                    color=COL_COLORS[col], alpha=0.85, edgecolor="none")
            ax.set_title(f"Avg IP = {med_ip:.2f}", fontsize=10)
            ax.set_xlabel("Inner Product Distortion", fontsize=8)
            ax.set_xlim(-0.05, 0.05)
            ax.tick_params(labelsize=7)
            ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            if col == 0:
                ax.set_ylabel("Frequency", fontsize=8)
            for spine in ax.spines.values():
                spine.set_color("#aaa")
                spine.set_linewidth(0.5)

    row_labels = [
        "(a)  $\\mathrm{TurboQuant_{prod}}$",
        "(b)  $\\mathrm{TurboQuant_{mse}}$",
    ]
    for row, label in enumerate(row_labels):
        ax0 = axes[row, 0]
        fig.text(
            ax0.get_position().x0 - 0.07,
            (ax0.get_position().y0 + ax0.get_position().y1) / 2,
            label, ha="right", va="center", fontsize=10,
            transform=fig.transFigure,
        )

    plt.tight_layout(rect=[0.07, 0, 1, 1])
    path = os.path.join(results_dir, "Fig_2_grouped_histograms.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
