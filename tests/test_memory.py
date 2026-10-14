import pytest

from gpumix.gpus import load_gpus
from gpumix.memory import memory_fit
from gpumix.models import load_models

GPUS = load_gpus()
QWEN = load_models()["qwen3-8b"]


def test_qwen3_8b_on_h100():
    # usable 80 GB x 0.9 = 72 GB; weights 8.2B x 2 bytes = 16.4 GB; 55.6 GB / 147,456 B
    fit = memory_fit(GPUS["H100-SXM"], QWEN)
    assert fit.fits
    assert fit.weights_gb == pytest.approx(16.4)
    assert fit.kv_cache_gb == pytest.approx(55.6)
    assert fit.kv_tokens == 377_061


def test_qwen3_8b_on_mi300x():
    # usable 192 GB x 0.9 = 172.8 GB; 172.8 - 16.4 = 156.4 GB / 147,456 B
    fit = memory_fit(GPUS["MI300X"], QWEN)
    assert fit.kv_cache_gb == pytest.approx(156.4)
    assert fit.kv_tokens == 1_060_655


def test_weights_bigger_than_usable_memory_does_not_fit():
    # 16.4 GB of weights vs 80 GB x 0.2 = 16 GB usable
    fit = memory_fit(GPUS["H100-SXM"], QWEN, utilization=0.2)
    assert not fit.fits
    assert fit.kv_cache_gb == 0
    assert fit.kv_tokens == 0


def test_fp8_weights_leave_more_room():
    bf16 = memory_fit(GPUS["H100-SXM"], QWEN)
    fp8 = memory_fit(GPUS["H100-SXM"], QWEN, bytes_per_param=1)
    assert fp8.weights_gb == pytest.approx(8.2)
    assert fp8.kv_tokens > bf16.kv_tokens
