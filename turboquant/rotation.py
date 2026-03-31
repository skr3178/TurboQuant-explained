"""
Random rotation and QJL projection matrices for TurboQuant.

- Π: orthogonal rotation matrix via QR decomposition of random N(0,1) matrix
- S: raw Gaussian matrix for QJL (Johnson-Lindenstrauss) projection
"""

import os
import torch


def make_rotation_matrix(
    d: int, seed: int | None = None, device: str = "cuda"
) -> torch.Tensor:
    """
    Generate a random orthogonal d×d matrix via QR decomposition.

    The sign ambiguity of QR is resolved by fixing diag(R) > 0.
    Results are cached to turboquant/cache/rotation_d{d}_seed{seed}.pt.
    """
    cache_dir = os.path.join(os.path.dirname(__file__), "cache")
    os.makedirs(cache_dir, exist_ok=True)
    seed_tag = seed if seed is not None else "none"
    cache_path = os.path.join(cache_dir, f"rotation_d{d}_seed{seed_tag}.pt")

    if os.path.exists(cache_path):
        return torch.load(cache_path, map_location=device, weights_only=True)

    generator = torch.Generator(device="cpu")
    if seed is not None:
        generator.manual_seed(seed)

    M = torch.randn(d, d, generator=generator, dtype=torch.float32)
    Q, R = torch.linalg.qr(M)

    # Fix sign ambiguity: ensure diag(R) > 0
    signs = torch.sign(torch.diag(R))
    Q = Q * signs.unsqueeze(0)

    # Verify orthogonality
    assert torch.allclose(Q @ Q.T, torch.eye(d), atol=1e-5), (
        "Rotation matrix is not orthogonal"
    )

    Q = Q.to(device)
    torch.save(Q.cpu(), cache_path)
    return Q


def make_qjl_matrix(
    d: int, seed: int | None = None, device: str = "cuda"
) -> torch.Tensor:
    """
    Generate a raw Gaussian d×d matrix for QJL projection.

    No orthogonalization — just i.i.d. N(0, 1) entries.
    Results are cached to turboquant/cache/qjl_d{d}_seed{seed}.pt.
    """
    cache_dir = os.path.join(os.path.dirname(__file__), "cache")
    os.makedirs(cache_dir, exist_ok=True)
    seed_tag = seed if seed is not None else "none"
    cache_path = os.path.join(cache_dir, f"qjl_d{d}_seed{seed_tag}.pt")

    if os.path.exists(cache_path):
        return torch.load(cache_path, map_location=device, weights_only=True)

    generator = torch.Generator(device="cpu")
    if seed is not None:
        generator.manual_seed(seed)

    S = torch.randn(d, d, generator=generator, dtype=torch.float32)
    S = S.to(device)
    torch.save(S.cpu(), cache_path)
    return S
