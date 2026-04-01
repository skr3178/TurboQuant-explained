"""
Data loading utilities for TurboQuant experiments.

Handles GloVe embeddings and DBpedia OpenAI embeddings: download, parse,
L2-normalize, train/query split.
"""

import os
import urllib.request
import zipfile
from pathlib import Path

import torch
import numpy as np


_GLOVE_URL = "https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip"
_GLOVE_VALID_DIMS = {50, 100, 200, 300}


def download_glove(cache_dir: str = "data") -> Path:
    """Download and extract GloVe 6B embeddings if not cached."""
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    zip_path = cache_path / "glove.6B.zip"
    marker = cache_path / ".glove_extracted"

    if marker.exists():
        return cache_path

    if not zip_path.exists():
        print(f"Downloading GloVe 6B (~822 MB)...")
        urllib.request.urlretrieve(_GLOVE_URL, zip_path)
        print(f"  Saved to {zip_path}")

    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(cache_path)
    marker.touch()
    print("  Done.")
    return cache_path


def load_glove(
    dim: int = 300,
    n: int | None = None,
    normalize: bool = True,
    cache_dir: str = "data",
) -> tuple[list[str], torch.Tensor]:
    """
    Load GloVe embeddings as float32 tensor.

    Args:
        dim: embedding dimension (50, 100, 200, 300)
        n: max vectors to load (None = all ~400K)
        normalize: L2-normalize to unit sphere
        cache_dir: where extracted files live

    Returns:
        words: list of vocabulary strings
        vectors: tensor [n, dim], optionally normalized
    """
    assert dim in _GLOVE_VALID_DIMS, f"dim must be one of {_GLOVE_VALID_DIMS}"

    data_dir = download_glove(cache_dir)
    txt_path = data_dir / f"glove.6B.{dim}d.txt"
    assert txt_path.exists(), f"Missing {txt_path} — run download_glove() first"

    words = []
    rows = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if n is not None and i >= n:
                break
            parts = line.rstrip().split(" ")
            words.append(parts[0])
            rows.append([float(v) for v in parts[1:]])

    vectors = torch.tensor(np.array(rows, dtype=np.float32))

    if normalize:
        norms = vectors.norm(dim=1, keepdim=True)
        norms = norms.clamp(min=1e-8)
        vectors /= norms

    print(f"Loaded GloVe {dim}d: {vectors.shape[0]} vectors, "
          f"mean norm = {vectors.norm(dim=1).mean():.6f}")
    return words, vectors


def split_train_query(
    vectors: torch.Tensor,
    n_query: int = 1000,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split vectors into database (train) and query sets."""
    n_total = vectors.shape[0]
    rng = np.random.default_rng(seed)
    query_idx = rng.choice(n_total, size=n_query, replace=False)
    query_mask = np.zeros(n_total, dtype=bool)
    query_mask[query_idx] = True

    database = vectors[~query_mask]
    queries = vectors[query_mask]
    print(f"Split: {database.shape[0]} database, {queries.shape[0]} queries")
    return database, queries


def load_dbpedia_1536_1M(
    n: int = 1_000_000,
    n_query: int = 1_000,
    cache_dir: str = "data",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Load DBpedia 1536d 1M embeddings from parquet and split into database + queries.

    On first call, reads all parquet shards from data/dbpedia_1536_1M/data/,
    extracts embeddings, L2-normalizes, and caches to a .pt file for fast reload.

    Returns:
        database: [n - n_query, 1536] tensor on CPU
        queries:  [n_query, 1536] tensor on CPU
    """
    pt_path = Path(cache_dir) / f"dbpedia_1536_{n // 1000}k.pt"

    if not pt_path.exists():
        import pyarrow.parquet as pq

        parquet_dir = Path(cache_dir) / "dbpedia_1536_1M" / "data"
        shards = sorted(parquet_dir.glob("train-*.parquet"))
        assert shards, f"No parquet files found in {parquet_dir}"

        print(f"Loading {len(shards)} parquet shards (first run — will cache to {pt_path})...")
        rows = []
        for i, shard in enumerate(shards):
            t = pq.read_table(shard, columns=["text-embedding-3-large-1536-embedding"])
            emb_col = t.column("text-embedding-3-large-1536-embedding").to_pylist()
            rows.extend(emb_col)
            print(f"  Shard {i+1}/{len(shards)}: {len(rows):,} vectors loaded")
            if len(rows) >= n:
                break

        vectors = torch.tensor(np.array(rows[:n], dtype=np.float32))
        norms = vectors.norm(dim=1, keepdim=True).clamp(min=1e-8)
        vectors /= norms
        del rows

        print(f"Saving cached tensor {vectors.shape} to {pt_path}...")
        torch.save(vectors, pt_path)
        print("  Done.")

    vectors = torch.load(pt_path, map_location="cpu", weights_only=True)
    print(f"Loaded DBpedia 1536d 1M: {vectors.shape}, mean norm = {vectors.norm(dim=1).mean():.6f}")

    return split_train_query(vectors, n_query=n_query)


def load_dbpedia_1536(
    n: int = 100_000,
    n_query: int = 1_000,
    cache_dir: str = "data",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Load pre-cached DBpedia 1536d embeddings and split into database + queries.

    Expects data/dbpedia_1536_100k.pt (pre-downloaded and L2-normalized).

    Returns:
        database: [n - n_query, 1536] tensor on CPU
        queries:  [n_query, 1536] tensor on CPU
    """
    path = Path(cache_dir) / f"dbpedia_1536_{n // 1000}k.pt"
    assert path.exists(), (
        f"Missing {path}. Download first with:\n"
        f"  python -c \"from datasets import load_dataset; ...\"\n"
        f"Or run the download script."
    )

    vectors = torch.load(path, map_location="cpu", weights_only=True)
    print(f"Loaded DBpedia 1536d: {vectors.shape}, mean norm = {vectors.norm(dim=1).mean():.6f}")

    return split_train_query(vectors, n_query=n_query)
