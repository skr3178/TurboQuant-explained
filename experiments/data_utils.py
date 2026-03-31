"""
Data loading utilities for TurboQuant experiments.

Handles GloVe embeddings: download, parse, L2-normalize, train/query split.
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
