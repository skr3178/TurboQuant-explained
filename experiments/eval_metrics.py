"""
Evaluation metrics for TurboQuant experiments.
"""

import time

import numpy as np
import torch


def mse_distortion(x_orig: torch.Tensor, x_recon: torch.Tensor) -> float:
    """Per-vector MSE: mean(||x - x_tilde||²) over all vectors."""
    return ((x_orig - x_recon) ** 2).sum(dim=-1).mean().item()


def inner_product_error(
    x_orig: torch.Tensor,
    x_recon: torch.Tensor,
    queries: torch.Tensor,
) -> dict:
    """
    Compute inner product error between original and reconstructed vectors.

    Args:
        x_orig: original database vectors [n_db, d]
        x_recon: reconstructed database vectors [n_db, d]
        queries: query vectors [n_q, d]

    Returns:
        dict with mean_abs_error, max_abs_error, correlation
    """
    ip_orig = queries @ x_orig.T     # [n_q, n_db]
    ip_recon = queries @ x_recon.T   # [n_q, n_db]

    diff = ip_orig - ip_recon
    abs_diff = diff.abs()

    correlation = torch.corrcoef(torch.stack([
        ip_orig.flatten(), ip_recon.flatten()
    ]))[0, 1].item()

    return {
        "mean_abs_error": abs_diff.mean().item(),
        "max_abs_error": abs_diff.max().item(),
        "correlation": correlation,
    }


def recall_at_k(
    database: torch.Tensor,
    queries: torch.Tensor,
    k: int,
    quantizer,
    device: str = "cuda",
) -> dict:
    """
    Compute Recall@k for approximate vs brute-force inner product search.

    Args:
        database: database vectors [n_db, d]
        queries: query vectors [n_q, d]
        k: top-k to compare
        quantizer: TurboQuantMSE or TurboQuantProd instance
        device: torch device

    Returns:
        dict with recall@k, quantize_time, search_time
    """
    database = database.to(device)
    queries = queries.to(device)

    # Brute-force ground truth
    ip_gt = queries @ database.T
    gt_topk = ip_gt.topk(k, dim=1).indices  # [n_q, k]

    # Quantize + dequantize
    t0 = time.perf_counter()
    if hasattr(quantizer, 'quantize'):
        # Check if it's TurboQuantProd (returns 3 values) or MSE (returns 1)
        result = quantizer.quantize(database)
        if isinstance(result, tuple):
            idx, qjl, gamma = result
            db_recon = quantizer.dequantize(idx, qjl, gamma)
        else:
            idx = result
            db_recon = quantizer.dequantize(idx)
    t_quant = time.perf_counter() - t0

    # Approximate search
    t0 = time.perf_counter()
    ip_approx = queries @ db_recon.T
    approx_topk = ip_approx.topk(k, dim=1).indices  # [n_q, k]
    t_search = time.perf_counter() - t0

    # Compute recall
    gt_sets = [set(gt_topk[i].tolist()) for i in range(gt_topk.shape[0])]
    approx_sets = [set(approx_topk[i].tolist()) for i in range(approx_topk.shape[0])]
    recall = sum(
        len(gt & approx) / k for gt, approx in zip(gt_sets, approx_sets)
    ) / len(gt_sets)

    return {
        f"recall@{k}": recall,
        "quantize_time_s": t_quant,
        "search_time_s": t_search,
    }


def inner_product_errors_flat(
    x_orig: torch.Tensor,
    x_recon: torch.Tensor,
    queries: torch.Tensor,
    chunk_size: int = 10_000,
) -> np.ndarray:
    """
    Compute flat array of all inner-product errors e(x_i, y_j) = <y_j, x_i> - <y_j, x̃_i>.

    Returns:
        np.ndarray of shape (n_db * n_query,) with dtype float32.
    """
    chunks = []
    for start in range(0, x_orig.shape[0], chunk_size):
        end = min(start + chunk_size, x_orig.shape[0])
        diff = x_orig[start:end] - x_recon[start:end]
        errors = (diff @ queries.T).cpu().numpy()
        chunks.append(errors)
    return np.concatenate(chunks, axis=0).ravel().astype(np.float32)


def ip_distortion(
    x_orig: torch.Tensor,
    x_recon: torch.Tensor,
    queries: torch.Tensor,
    chunk_size: int = 10_000,
) -> float:
    """
    Compute D_prod = E[e²] = mean of squared inner-product errors.

    Accumulates running sum without storing all errors simultaneously.
    """
    sq_sum = 0.0
    count = 0
    for start in range(0, x_orig.shape[0], chunk_size):
        end = min(start + chunk_size, x_orig.shape[0])
        diff = x_orig[start:end] - x_recon[start:end]
        sq_sum += (diff @ queries.T).pow(2).sum().item()
        count += (end - start) * queries.shape[0]
    return sq_sum / count
