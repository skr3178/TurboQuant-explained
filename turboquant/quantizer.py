"""
TurboQuant quantizers: TurboQuantMSE and TurboQuantProd.

Algorithm 1 (MSE-optimal):
  Quant:   y = Π·x, idx_j = argmin_k |y_j - c_k|
  Dequant: ỹ_j = c_{idx_j}, x̃ = Πᵀ·ỹ

Algorithm 2 (inner-product optimal):
  Quant:   idx ← Quant_mse(x), r = x - Dequant_mse(idx),
           qjl = sign(S·r), output (idx, qjl, ‖r‖₂)
  Dequant: x̃_mse + √(π/2)/d · γ · Sᵀ·qjl
"""

import numpy as np
import torch

from turboquant.codebook import get_codebook
from turboquant.rotation import make_qjl_matrix, make_rotation_matrix


class TurboQuantMSE:
    """MSE-optimal vector quantizer (Algorithm 1)."""

    def __init__(
        self, d: int, b: int, seed: int | None = None, device: str = "cuda"
    ):
        self.d = d
        self.b = b
        self.device = device
        self.Pi = make_rotation_matrix(d, seed, device)
        self.codebook = get_codebook(b, d, device)  # shape: [2^b]

    def quantize(self, x: torch.Tensor, chunk_size: int = 8192) -> torch.Tensor:
        """
        Quantize vectors to b-bit indices per coordinate.

        Args:
            x: tensor of shape [n, d]
            chunk_size: process in chunks to avoid OOM
        Returns:
            idx: long tensor of shape [n, d], values in [0, 2^b)
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
            return self.quantize(x, chunk_size).squeeze(0)

        n = x.shape[0]
        indices = []
        for start in range(0, n, chunk_size):
            chunk = x[start:start + chunk_size]
            y = chunk @ self.Pi.T
            distances = torch.abs(y.unsqueeze(-1) - self.codebook)
            indices.append(torch.argmin(distances, dim=-1))
        return torch.cat(indices, dim=0)

    def dequantize(self, idx: torch.Tensor, chunk_size: int = 8192) -> torch.Tensor:
        """
        Reconstruct vectors from indices.

        Args:
            idx: long tensor of shape [n, d]
            chunk_size: process in chunks to avoid OOM
        Returns:
            x_recon: float tensor of shape [n, d]
        """
        if idx.dim() == 1:
            idx = idx.unsqueeze(0)
            return self.dequantize(idx, chunk_size).squeeze(0)

        n = idx.shape[0]
        results = []
        for start in range(0, n, chunk_size):
            chunk = idx[start:start + chunk_size]
            y_recon = self.codebook[chunk]
            results.append(y_recon @ self.Pi)
        return torch.cat(results, dim=0)


class TurboQuantProd:
    """Inner-product-optimal quantizer (Algorithm 2)."""

    def __init__(
        self,
        d: int,
        b: int,
        seed_mse: int | None = None,
        seed_qjl: int | None = None,
        device: str = "cuda",
    ):
        self.d = d
        self.b = b
        self.device = device
        self.mse = TurboQuantMSE(d, b - 1, seed_mse, device)
        self.S = make_qjl_matrix(d, seed_qjl, device)
        self._qjl_coeff = np.sqrt(np.pi / 2) / d

    def quantize(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize vectors to MSE indices + QJL residual.

        Args:
            x: tensor of shape [..., d]
        Returns:
            idx: MSE indices, shape [..., d]
            qjl: QJL signs {-1, +1}, shape [..., d]
            gamma: residual norms, shape [..., 1]
        """
        idx = self.mse.quantize(x)
        x_recon_mse = self.mse.dequantize(idx)
        residual = x - x_recon_mse

        Sr = residual @ self.S.T
        # torch.sign(0) = 0, which breaks QJL — use >= 0 test
        qjl = torch.where(Sr >= 0, 1.0, -1.0)

        gamma = torch.norm(residual, dim=-1, keepdim=True)
        return idx, qjl, gamma

    def dequantize(
        self,
        idx: torch.Tensor,
        qjl: torch.Tensor,
        gamma: torch.Tensor,
    ) -> torch.Tensor:
        """
        Reconstruct vectors from MSE indices + QJL.

        Args:
            idx: MSE indices, shape [..., d]
            qjl: QJL signs, shape [..., d]
            gamma: residual norms, shape [..., 1]
        Returns:
            x_recon: float tensor of shape [..., d]
        """
        x_mse = self.mse.dequantize(idx)
        x_qjl = self._qjl_coeff * gamma * (qjl @ self.S)
        return x_mse + x_qjl
