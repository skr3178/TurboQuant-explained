import pytest
import torch


def test_top_level_imports():
    import turboquant

    for name in [
        "patch_model",
        "TurboQuantCache",
        "TurboQuantMSE",
        "TurboQuantProd",
        "KVCacheConfig",
        "get_codebook",
    ]:
        assert hasattr(turboquant, name)


def test_codebook_bundled():
    """Cache files are bundled — get_codebook() must return instantly."""
    from turboquant import get_codebook

    cb = get_codebook(b=2, d=1536, device="cpu")
    assert cb.shape == (4,)
    assert cb.dtype == torch.float32


def test_mse_roundtrip():
    from turboquant import TurboQuantMSE

    q = TurboQuantMSE(d=128, b=2, seed=42, device="cpu")
    x = torch.randn(8, 128)
    assert q.dequantize(q.quantize(x)).shape == x.shape


def test_prod_roundtrip():
    from turboquant import TurboQuantProd

    q = TurboQuantProd(d=128, b=2, seed_mse=42, seed_qjl=43, device="cpu")
    x = torch.randn(8, 128)
    idx, qjl, gamma = q.quantize(x)
    assert q.dequantize(idx, qjl, gamma).shape == x.shape


def test_cache_update():
    from turboquant import TurboQuantCache

    cache = TurboQuantCache(head_dim=64, b_key=2, b_value=2, device="cpu")
    k = torch.randn(1, 4, 16, 64)
    v = torch.randn(1, 4, 16, 64)
    k_out, v_out = cache.update(k, v, layer_idx=0)
    assert k_out.shape == (1, 4, 16, 64)
    assert cache.get_seq_length(0) == 16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_cache_cuda():
    from turboquant import TurboQuantCache

    cache = TurboQuantCache(head_dim=64, b_key=2, b_value=2, device="cuda")
    k = torch.randn(1, 2, 8, 64, device="cuda")
    v = torch.randn(1, 2, 8, 64, device="cuda")
    k_out, v_out = cache.update(k, v, layer_idx=0)
    assert k_out.is_cuda
