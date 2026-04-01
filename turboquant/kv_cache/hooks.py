"""
HuggingFace-compatible KV cache with TurboQuant compression.

Drop-in replacement for DynamicCache. Keys use TurboQuantProd
(unbiased inner products for attention scores); values use
TurboQuantMSE (low MSE for weighted aggregation).

Usage:
    from turboquant.kv_cache import TurboQuantCache

    cache = TurboQuantCache(head_dim=128, b_key=2, b_value=2)
    output = model.generate(input_ids, past_key_values=cache)
"""

from __future__ import annotations

from typing import Optional

import torch

from turboquant.quantizer import TurboQuantMSE, TurboQuantProd

# Inherit from HF Cache if available for isinstance checks in model.forward().
try:
    from transformers.cache_utils import Cache as _HFBase
except ImportError:
    class _HFBase:
        """Fallback base when transformers is not installed."""


class _LayerCache:
    """
    Per-layer quantized KV storage with residual buffer.

    Maintains:
    - Compressed storage: quantized K/V from older tokens
    - Residual buffer: recent ``residual_length`` tokens in full precision

    On each ``update()``:
    1. Append new tokens to residual buffer
    2. If buffer exceeds threshold, quantize the excess
    3. Dequantize compressed + residual -> full K,V for attention
    """

    def __init__(
        self,
        head_dim: int,
        b_key: int,
        b_value: int,
        residual_length: int,
        seed: int,
        device: str,
    ):
        self.head_dim = head_dim
        self.residual_length = residual_length

        self.q_key = TurboQuantProd(
            d=head_dim,
            b=b_key,
            seed_mse=seed,
            seed_qjl=seed + 1,
            device=device,
        )
        self.q_val = TurboQuantMSE(
            d=head_dim,
            b=b_value,
            seed=seed + 2,
            device=device,
        )

        # Compressed storage — one entry per quantization batch
        self._k_idx: list[torch.Tensor] = []
        self._k_qjl: list[torch.Tensor] = []
        self._k_gamma: list[torch.Tensor] = []
        self._v_idx: list[torch.Tensor] = []

        # Residual buffer (full precision, recent tokens)
        self._k_residual: Optional[torch.Tensor] = None
        self._v_residual: Optional[torch.Tensor] = None
        self._shape: Optional[tuple[int, int, int]] = None  # (B, H, D)

    def update(
        self,
        key: torch.Tensor,    # [B, H, S_new, D]
        value: torch.Tensor,  # [B, H, S_new, D]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, H, S_new, D = key.shape

        if self._shape is None:
            self._shape = (B, H, D)

        # Merge new tokens into residual buffer
        if self._k_residual is not None:
            k_buf = torch.cat([self._k_residual, key], dim=2)
            v_buf = torch.cat([self._v_residual, value], dim=2)
        else:
            k_buf, v_buf = key, value

        buf_len = k_buf.shape[2]

        # Quantize excess tokens beyond residual_length
        if buf_len > self.residual_length:
            n_q = buf_len - self.residual_length

            k_flat = k_buf[:, :, :n_q, :].reshape(-1, D)
            v_flat = v_buf[:, :, :n_q, :].reshape(-1, D)

            idx_k, qjl, gamma = self.q_key.quantize(k_flat)
            idx_v = self.q_val.quantize(v_flat)

            self._k_idx.append(idx_k)
            self._k_qjl.append(qjl)
            self._k_gamma.append(gamma)
            self._v_idx.append(idx_v)

            # Keep only residual portion
            self._k_residual = k_buf[:, :, n_q:, :].clone()
            self._v_residual = v_buf[:, :, n_q:, :].clone()
        else:
            self._k_residual = k_buf
            self._v_residual = v_buf

        return self._reconstruct()

    def _reconstruct(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Dequantize compressed storage + residual -> full K,V."""
        if self._shape is None:
            raise RuntimeError("Cache is empty — call update() first")

        B, H, D = self._shape
        k_parts: list[torch.Tensor] = []
        v_parts: list[torch.Tensor] = []

        if self._k_idx:
            idx = torch.cat(self._k_idx, dim=0)
            qjl = torch.cat(self._k_qjl, dim=0)
            gamma = torch.cat(self._k_gamma, dim=0)
            k_dq = self.q_key.dequantize(idx, qjl, gamma)
            k_parts.append(k_dq.reshape(B, H, -1, D))

            v_idx = torch.cat(self._v_idx, dim=0)
            v_dq = self.q_val.dequantize(v_idx)
            v_parts.append(v_dq.reshape(B, H, -1, D))

        if self._k_residual is not None:
            k_parts.append(self._k_residual)
            v_parts.append(self._v_residual)

        if not k_parts:
            dev = self.q_key.mse.codebook.device
            return (
                torch.empty(B, H, 0, D, device=dev),
                torch.empty(B, H, 0, D, device=dev),
            )

        return torch.cat(k_parts, dim=2), torch.cat(v_parts, dim=2)

    @property
    def seq_len(self) -> int:
        if self._shape is None:
            return 0
        B, H, _ = self._shape
        compressed = sum(t.shape[0] for t in self._k_idx) // (B * H)
        residual = self._k_residual.shape[2] if self._k_residual is not None else 0
        return compressed + residual

    def clear(self):
        self._k_idx.clear()
        self._k_qjl.clear()
        self._k_gamma.clear()
        self._v_idx.clear()
        self._k_residual = None
        self._v_residual = None
        self._shape = None


class TurboQuantCache(_HFBase):
    """
    Drop-in replacement for HuggingFace DynamicCache.

    Compresses K/V tensors using TurboQuantProd (keys) and
    TurboQuantMSE (values) with a residual buffer for recent tokens.

    Usage:
        cache = TurboQuantCache(head_dim=128, b_key=2, b_value=2)
        output = model.generate(input_ids, past_key_values=cache)
    """

    def __init__(
        self,
        head_dim: int,
        b_key: int = 2,
        b_value: int = 2,
        residual_length: int = 128,
        device: str = "cuda",
        seed: int = 42,
    ):
        # Handle different transformers versions:
        # - <5.4: Cache.__init__() takes no args
        # - >=5.4: requires layers or layer_class_to_replicate
        # We skip the base __init__ entirely for >=5.4 to avoid property
        # conflicts and manage our own layer state via _layers.
        import inspect

        params = inspect.signature(_HFBase.__init__).parameters
        if "layers" not in params:
            super().__init__()
        if b_key < 2:
            raise ValueError(
                "b_key must be >= 2 (TurboQuantProd uses b-1 bits for MSE "
                "+ 1 bit for QJL)"
            )

        self.head_dim = head_dim
        self.b_key = b_key
        self.b_value = b_value
        self.residual_length = residual_length
        self.device = device
        self.seed = seed

        self._layers: list[Optional[_LayerCache]] = []

    def _ensure_layer(self, layer_idx: int) -> _LayerCache:
        while len(self._layers) <= layer_idx:
            self._layers.append(None)
        if self._layers[layer_idx] is None:
            self._layers[layer_idx] = _LayerCache(
                head_dim=self.head_dim,
                b_key=self.b_key,
                b_value=self.b_value,
                residual_length=self.residual_length,
                seed=self.seed + layer_idx * 3,
                device=self.device,
            )
        return self._layers[layer_idx]

    # --- HuggingFace Cache protocol ---

    def update(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        layer = self._ensure_layer(layer_idx)
        return layer.update(key, value)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx < len(self._layers) and self._layers[layer_idx] is not None:
            return self._layers[layer_idx].seq_len
        return 0

    def get_max_length(self) -> int:
        return -1  # dynamic, no max

    def to_legacy_cache(self):
        """Convert to tuple format for older HF model implementations."""
        legacy = []
        for layer in self._layers:
            if layer is not None:
                k, v = layer._reconstruct()
                legacy.append((k, v))
            else:
                legacy.append((None, None))
        return tuple(legacy)

    # --- Convenience ---

    def clear(self):
        for layer in self._layers:
            if layer is not None:
                layer.clear()
        self._layers.clear()

    @property
    def layer_caches(self):
        return self._layers


def patch_model(
    model,
    b_key: int = 2,
    b_value: int = 2,
    residual_length: int = 128,
    device: str = "cuda",
    seed: int = 42,
) -> TurboQuantCache:
    """
    Create a TurboQuantCache configured for the given model.

    Reads head_dim from model config. Returns the cache — pass it
    as ``past_key_values`` to ``model.generate()``.

    Args:
        model: HuggingFace PreTrainedModel
        b_key: bits per coordinate for keys (min 2)
        b_value: bits per coordinate for values (min 1)
        residual_length: tokens kept in full precision
        device: torch device
        seed: random seed for reproducible rotations

    Returns:
        TurboQuantCache ready for use as past_key_values
    """
    config = model.config
    head_dim = getattr(
        config, "head_dim", config.hidden_size // config.num_attention_heads
    )

    cache = TurboQuantCache(
        head_dim=head_dim,
        b_key=b_key,
        b_value=b_value,
        residual_length=residual_length,
        device=device,
        seed=seed,
    )
    model._tq_cache = cache
    return cache


def unpatch_model(model) -> None:
    """Remove TurboQuant cache reference from model."""
    if hasattr(model, "_tq_cache"):
        del model._tq_cache
