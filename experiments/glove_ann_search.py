"""
Reproduce Table 2: ANN search performance on GloVe 300d.

Compares TurboQuantProd (b=4) against brute-force for Recall@k.
Reports timing and recall metrics.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import time

from turboquant.quantizer import TurboQuantMSE, TurboQuantProd
from experiments.data_utils import load_glove, split_train_query
from experiments.eval_metrics import recall_at_k


def main():
    dim = 300
    n_query = 1000
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}, dim={dim}")

    # Load full GloVe 300d
    words, vectors = load_glove(dim=dim, normalize=True)
    database, queries = split_train_query(vectors, n_query=n_query, seed=42)
    database = database.to(device)
    queries = queries.to(device)

    print(f"\nDatabase: {database.shape}, Queries: {queries.shape}")

    # Test TurboQuantMSE at different bit-widths
    print("\n" + "=" * 60)
    print("TurboQuantMSE — Recall@k")
    print("=" * 60)
    for b in [2, 3, 4]:
        tq = TurboQuantMSE(dim, b, seed=42, device=device)
        for k in [1, 10]:
            result = recall_at_k(database, queries, k, tq, device)
            print(f"  b={b}, k={k}: recall={result[f'recall@{k}']:.4f}, "
                  f"quantize={result['quantize_time_s']:.4f}s, "
                  f"search={result['search_time_s']:.4f}s")

    # Test TurboQuantProd
    print("\n" + "=" * 60)
    print("TurboQuantProd — Recall@k")
    print("=" * 60)
    for b in [3, 4]:
        tq = TurboQuantProd(dim, b, seed_mse=42, seed_qjl=123, device=device)
        for k in [1, 10]:
            result = recall_at_k(database, queries, k, tq, device)
            print(f"  b={b}, k={k}: recall={result[f'recall@{k}']:.4f}, "
                  f"quantize={result['quantize_time_s']:.4f}s, "
                  f"search={result['search_time_s']:.4f}s")

    # Brute-force baseline timing
    print("\n" + "=" * 60)
    print("Brute-force baseline")
    print("=" * 60)
    t0 = time.perf_counter()
    ip = queries @ database.T
    _ = ip.topk(10, dim=1)
    t_brute = time.perf_counter() - t0
    print(f"  Brute-force top-10: {t_brute:.4f}s")

    # Save results
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "glove_ann_table2.txt")
    with open(out_path, "w") as f:
        f.write(f"GloVe {dim}d ANN Search Results\n")
        f.write(f"Database: {database.shape[0]}, Queries: {queries.shape[0]}\n")
        f.write(f"Brute-force top-10: {t_brute:.4f}s\n")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
