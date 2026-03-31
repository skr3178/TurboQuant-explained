"""
TurboQuant KV Cache compression for transformer models.

Drop-in hooks for HuggingFace transformers that quantize K and V tensors
on-the-fly using TurboQuantProd (unbiased inner-product) for K and
TurboQuantMSE for V.
"""

from turboquant.kv_cache.manager import KVCacheConfig, QuantizedKVCache
from turboquant.kv_cache.hooks import patch_model, unpatch_model

__all__ = ["KVCacheConfig", "QuantizedKVCache", "patch_model", "unpatch_model"]
