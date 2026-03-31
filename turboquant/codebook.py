"""
Codebook construction for TurboQuant via Lloyd-Max on the Beta marginal.

The marginal distribution of a single coordinate of a uniformly random
unit vector in R^d is:

    f_X(x) = Γ(d/2) / (√π · Γ((d-1)/2)) · (1 - x²)^((d-3)/2)

Lloyd-Max finds the optimal scalar quantizer centroids for this distribution.
"""

import os
import numpy as np
import torch
from scipy.special import gammaln
from scipy.integrate import quad


def beta_marginal_pdf(x: np.ndarray, d: int) -> np.ndarray:
    """Marginal PDF of one coordinate of a uniform random unit vector in R^d."""
    log_norm = gammaln(d / 2) - 0.5 * np.log(np.pi) - gammaln((d - 1) / 2)
    log_kernel = ((d - 3) / 2) * np.log(np.maximum(1.0 - x**2, 1e-300))
    return np.exp(log_norm + log_kernel)


def _conditional_mean(a: float, b: float, d: int) -> float:
    """E[X | a <= X <= b] under the Beta marginal."""
    numerator, _ = quad(lambda x: x * beta_marginal_pdf(np.array(x), d), a, b)
    denominator, _ = quad(lambda x: beta_marginal_pdf(np.array(x), d), a, b)
    if denominator < 1e-15:
        return (a + b) / 2.0
    return numerator / denominator


def lloyd_max_1d(n_centroids: int, d: int, n_iter: int = 500) -> np.ndarray:
    """
    Lloyd-Max algorithm for the Beta marginal distribution.

    Returns n_centroids sorted in ascending order, symmetric about zero.
    """
    half = n_centroids // 2

    # Initialize with percentiles of the positive half of the distribution.
    # The half-CDF (0 to 1) integrates to 0.5, so percentiles must be in (0, 0.5).
    percentiles = np.linspace(
        0.5 / (2 * half), 0.5 - 0.5 / (2 * half), half
    )
    positive = np.array([
        _invert_cdf(p, d, lo=0.0, hi=1.0) for p in percentiles
    ])
    centroids = np.concatenate([-positive[::-1], positive])

    for iteration in range(n_iter):
        # Boundaries are midpoints between adjacent centroids
        boundaries = (centroids[:-1] + centroids[1:]) / 2.0
        # Add domain edges
        edges = np.concatenate([[-1.0], boundaries, [1.0]])

        # Update centroids to conditional means within each cell
        new_centroids = np.empty_like(centroids)
        for i in range(n_centroids):
            new_centroids[i] = _conditional_mean(edges[i], edges[i + 1], d)

        # Enforce exact symmetry
        for i in range(half):
            new_centroids[i] = -new_centroids[n_centroids - 1 - i]

        # Check convergence
        if np.allclose(new_centroids, centroids, atol=1e-12):
            break
        centroids = new_centroids

    return centroids


def _invert_cdf(p: float, d: int, lo: float = 0.0, hi: float = 1.0) -> float:
    """Find x such that CDF(x) = p on [lo, hi] via bisection."""
    for _ in range(100):
        mid = (lo + hi) / 2.0
        cdf_val, _ = quad(lambda x: beta_marginal_pdf(np.array(x), d), 0.0, mid)
        if cdf_val < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def get_codebook(b: int, d: int, device: str = "cuda") -> torch.Tensor:
    """
    Return the 2^b centroids as a float32 torch tensor.

    Results are cached to turboquant/cache/codebook_{n}_d{d}.npy.
    """
    n_centroids = 1 << b
    cache_dir = os.path.join(os.path.dirname(__file__), "cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"codebook_{n_centroids}_d{d}.npy")

    if os.path.exists(cache_path):
        centroids = np.load(cache_path)
    else:
        print(f"Computing codebook: b={b}, d={d}, n_centroids={n_centroids}...")
        centroids = lloyd_max_1d(n_centroids, d)
        np.save(cache_path, centroids)
        print(f"  Cached to {cache_path}")
        print(f"  Centroids: {centroids}")

    return torch.tensor(centroids, dtype=torch.float32, device=device)


def compute_theoretical_bounds(b: int) -> tuple[float, float]:
    """
    Return (upper, lower) MSE bounds from the paper.

    Upper: sqrt(3*pi/2) * 4^(-b)
    Lower: 1 / 4^b
    """
    upper = np.sqrt(3 * np.pi / 2) * (4 ** (-b))
    lower = 4 ** (-b)
    return upper, lower
