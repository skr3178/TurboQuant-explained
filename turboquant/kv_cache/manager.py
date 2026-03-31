"""
Quantized KV cache manager.

Stores compressed K and V tensors and reconstructs them on demand.

Design:
  - K uses TurboQuantProd (b bits) — unbiased inner products for attention scores
  - V uses TurboQuantMSE (b bits) — low MSE for weighted value aggregation
  - Each layer/head has independent quantizers sharing the same rotation matrices

Memory per token (vs FP16):
  b=2: ~1/8 of FP16 for indices + small overhead for QJL signs and norms
  b=3: ~3/16 of FP16
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch

from turboquant.quantizer import TurboQuantMSE, TurboQuantProd


@dataclass
class KVCacheConfig:
    """Configuration for quantized KV cache."""
    b_key: int = 2        # bits for key quantization (TurboQuantProd)
    b_value: int = 2      # bits for value quantization (TurboQuantMSE)
    device: str = "cuda"
    seed: int = 42        # for reproducible rotation matrices


class LayerKVStore:
    """
    Per-layer quantized KV storage.

    Accumulates compressed tokens and reconstructs full K/V matrices
    for attention computation.
    """

    def __init__(self, head_dim: int, cfg: KVCacheConfig):
        self.head_dim = head_dim
        self.cfg = cfg

        # One quantizer per layer (shared across heads — heads are processed
        # as separate batch dimension)
        self.q_key = TurboQuantProd(
            d=head_dim,
            b=cfg.b_key,
            seed_mse=cfg.seed,
            seed_qjl=cfg.seed + 1,
            device=cfg.device,
        )
        self.q_val = TurboQuantMSE(
            d=head_dim,
            b=cfg.b_value,
            seed=cfg.seed + 2,
            device=cfg.device,
        )

        # Compressed storage lists: one entry per generation step
        # Key: (idx, qjl, gamma)  Value: idx
        self._k_idx: list[torch.Tensor] = []
        self._k_qjl: list[torch.Tensor] = []
        self._k_gamma: list[torch.Tensor] = []
        self._v_idx: list[torch.Tensor] = []

    def append(self, key: torch.Tensor, value: torch.Tensor) -> None:
        """
        Compress and store one or more new tokens.

        Args:
            key:   [batch, n_heads, seq_new, head_dim]
            value: [batch, n_heads, seq_new, head_dim]
        """
        # Flatten batch × heads × seq into a single batch dimension
        B, H, S, D = key.shape
        k_flat = key.reshape(B * H * S, D)
        v_flat = value.reshape(B * H * S, D)

        idx_k, qjl, gamma = self.q_key.quantize(k_flat)
        idx_v = self.q_val.quantize(v_flat)

        self._k_idx.append(idx_k)
        self._k_qjl.append(qjl)
        self._k_gamma.append(gamma)
        self._v_idx.append(idx_v)

        self._shape = (B, H, D)  # cache for reconstruction

    def reconstruct(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Reconstruct full K and V tensors from compressed storage.

        Returns:
            keys:   [batch, n_heads, seq_total, head_dim]
            values: [batch, n_heads, seq_total, head_dim]
        """
        if not self._k_idx:
            raise RuntimeError("KV cache is empty — call append() first")

        B, H, D = self._shape

        # Concatenate across time steps
        idx_k   = torch.cat(self._k_idx,   dim=0)
        qjl     = torch.cat(self._k_qjl,   dim=0)
        gamma   = torch.cat(self._k_gamma, dim=0)
        idx_v   = torch.cat(self._v_idx,   dim=0)

        total_tokens = idx_k.shape[0]
        S = total_tokens // (B * H)

        keys   = self.q_key.dequantize(idx_k, qjl, gamma).reshape(B, H, S, D)
        values = self.q_val.dequantize(idx_v).reshape(B, H, S, D)
        return keys, values

    def clear(self) -> None:
        self._k_idx.clear()
        self._k_qjl.clear()
        self._k_gamma.clear()
        self._v_idx.clear()

    @property
    def seq_len(self) -> int:
        return sum(t.shape[0] for t in self._k_idx) // (
            math.prod(self._shape[:2]) if hasattr(self, "_shape") else 1
        )


class QuantizedKVCache:
    """
    Full model KV cache: one LayerKVStore per transformer layer.

    Usage:
        cache = QuantizedKVCache(n_layers=32, head_dim=128, cfg=KVCacheConfig(b_key=2))
        # inside attention forward:
        cache[layer_idx].append(key, value)
        k, v = cache[layer_idx].reconstruct()
    """

    def __init__(self, n_layers: int, head_dim: int, cfg: KVCacheConfig):
        self.stores = [LayerKVStore(head_dim, cfg) for _ in range(n_layers)]

    def __getitem__(self, layer_idx: int) -> LayerKVStore:
        return self.stores[layer_idx]

    def clear(self) -> None:
        for store in self.stores:
            store.clear()

    def memory_bytes(self) -> int:
        """Estimate compressed memory usage in bytes."""
        total = 0
        for store in self.stores:
            for t in store._k_idx:
                total += t.element_size() * t.numel()
            for t in store._k_qjl:
                total += t.element_size() * t.numel()
            for t in store._k_gamma:
                total += t.element_size() * t.numel()
            for t in store._v_idx:
                total += t.element_size() * t.numel()
        return total
