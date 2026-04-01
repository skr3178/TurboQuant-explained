# Plan: Add KV Cache Integration for TurboQuant

## Context

TurboQuant's core quantization algorithm is validated (Figures 1-3). To reproduce Figure 4 (Needle-in-Haystack benchmark), we need to integrate TurboQuant into a live LLM's attention mechanism by intercepting and quantizing KV cache during inference.

**Current state:**
- `manager.py` exists with `LayerKVStore` / `QuantizedKVCache` storage classes
- `__init__.py` imports `hooks.py` which doesn't exist
- No HuggingFace transformers integration yet

**Why now:** User wants to complete the KV cache integration to enable LLM inference with quantized KV cache.

---

## Implementation Approach

### Step 1: Create `turboquant/kv_cache/hooks.py`

Implement a HuggingFace-compatible cache that uses TurboQuant for compression.

**Key components:**

1. `TurboQuantLayer(CacheLayerMixin)` — Per-layer quantized cache
   - Uses `TurboQuantProd` for keys (unbiased inner products → accurate attention scores)
   - Uses `TurboQuantMSE` for values (low MSE → accurate value aggregation)
   - Maintains a **residual buffer** (recent tokens in full precision) for quality
   - Implements `update(key_states, value_states, layer_idx)`:
     - Quantize new tokens and append to compressed storage
     - Dequantize full cache and concatenate with new tokens
     - Return to attention mechanism
     - Re-quantize if residual buffer exceeds threshold

2. `TurboQuantCache(Cache)` — Container with one `TurboQuantLayer` per transformer layer

3. `patch_model(model, config)` — Helper to inject cache into model
   - Returns model configured to use `TurboQuantCache` by default

4. `unpatch_model(model)` — Restore original cache behavior

**Design decisions:**
- Residual buffer size: ~128 tokens (keeps recent context in full precision)
- Quantization triggers when buffer exceeds threshold (lazy quantization)
- Compatible with `model.generate(past_key_values=cache)` API

### Step 2: Update `turboquant/kv_cache/__init__.py`

Add `hooks.py` exports that are currently missing.

---

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `turboquant/kv_cache/hooks.py` | **Create** | HuggingFace-compatible cache implementation |
| `turboquant/kv_cache/__init__.py` | Minor fix | Imports are already there, just need the file |

---

## Implementation Details

### TurboQuantLayer.update() signature

```python
def update(
    self,
    key_states: torch.Tensor,      # [batch, heads, seq_new, head_dim]
    value_states: torch.Tensor,    # [batch, heads, seq_new, head_dim]
    layer_idx: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize new tokens, append to compressed storage, dequantize all,
    and return full K,V for attention computation.
    """
```

### Data flow during generation

```
Attention forward:
  key_states, value_states = past_key_values.update(new_k, new_v, layer_idx)
  └─> TurboQuantLayer.update():
      1. Quantize new tokens via q_key.quantize() / q_val.quantize()
      2. Append to compressed storage
      3. Dequantize entire cache
      4. Return (full_cached + new) for attention
```

### Residual buffer pattern

```python
# Keep recent N tokens in full precision, older tokens compressed
residual_length = 128  # configurable
if self.residual_kv.shape[-2] + new_tokens >= residual_length:
    # Quantize accumulated residual and add to compressed storage
    compressed = self.q_key.quantize(self.residual_kv)
    self._k_idx.append(compressed.idx)
    self._k_qjl.append(compressed.qjl)
    self.residual_kv = new_tokens  # reset buffer
else:
    self.residual_kv = torch.cat([self.residual_kv, new_tokens])
```

---

## Reuse Existing Code

| Component | Location | Usage |
|-----------|----------|-------|
| `TurboQuantProd` | `turboquant/quantizer.py` | Quantize keys (unbiased inner products) |
| `TurboQuantMSE` | `turboquant/quantizer.py` | Quantize values (low MSE) |
| `LayerKVStore` | `turboquant/kv_cache/manager.py` | Reference for storage pattern |
| `QuantizedKVCache` | `turboquant/kv_cache/manager.py` | Reference for multi-layer management |

---

## Verification Plan

### Phase 1: Unit tests (no GPU required initially)
1. Create small test that instantiates `TurboQuantCache`
2. Mock attention update calls
3. Verify quantize/dequantize roundtrip preserves shapes

### Phase 2: Integration test (requires GPU)
1. Load small model (e.g., `meta-llama/Llama-3.2-1B-Instruct`)
2. Generate with `TurboQuantCache` vs default cache
3. Compare outputs (should be similar with b=3 or b=4)

### Phase 3: Needle-in-Haystack benchmark (full validation)
1. Run benchmark with quantized cache at different bitwidths
2. Compare recall vs full cache baseline
3. Reproduce Figure 4 from paper

---

## Dependencies

- `transformers` — HuggingFace transformers (already installed)
- `torch` — PyTorch with CUDA support (for LLM inference)
- GPU required for Phases 2-3

---

## Success Criteria

1. `hooks.py` implements `TurboQuantCache` compatible with HuggingFace `Cache` protocol
2. `patch_model(model, config)` returns a model ready for quantized inference
3. Generation produces valid outputs with quantized cache
4. (Stretch) Needle-in-Haystack benchmark runs and produces Figure 4

---

## Notes

- This is a significant but well-scoped task
- HuggingFace's cache abstraction makes integration clean
- Residual buffer is key for maintaining quality while compressing older tokens
- After this, Figure 4 becomes achievable (though still requires GPU and benchmark setup)
