# TurboQuant

Near-optimal KV-cache compression for HuggingFace transformers.
Reduces KV cache memory by ~8x at 2 bits with attention quality within ~2.7x of the Shannon limit.

## Installation

```bash
pip install turboquant-explained
```

> Install PyTorch separately for your hardware first:
> - CPU: `pip install torch`
> - CUDA 12.x: `pip install torch --index-url https://download.pytorch.org/whl/cu121`
> - See [pytorch.org](https://pytorch.org/get-started/locally/) for all variants.

## Quick Start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import turboquant

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

# 2-bit keys + values → ~8x memory reduction
cache = turboquant.patch_model(model, b_key=2, b_value=2)

inputs = tokenizer("Hello, world!", return_tensors="pt").to("cuda")
output = model.generate(**inputs, past_key_values=cache, max_new_tokens=200)
print(tokenizer.decode(output[0]))
```

`patch_model` reads `head_dim` from `model.config` automatically — no manual configuration needed.

### Memory savings at a glance

| Bit-width | Memory vs FP16 | Typical use |
|-----------|---------------|-------------|
| b=2 | ~1/8 | Long contexts, aggressive compression |
| b=3 | ~3/16 | Balanced quality / savings |
| b=4 | ~1/4 | Near-lossless |

---

For implementation details and theory, see the [full README on GitHub](https://github.com/skr3178/TurboQuant-explained).
