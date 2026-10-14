import pytest

from gpumix.gpus import load_gpus
from gpumix.models import load_models
from gpumix.perf import decode_step_seconds, prefill_seconds

GPUS = load_gpus()
H100, MI300X = GPUS["H100-SXM"], GPUS["MI300X"]
QWEN = load_models()["qwen3-8b"]
MS = 1e-3


def test_prefill_1000_tokens_is_compute_bound():
    # 2 x 8.2e9 x 1000 = 1.64e13 operations
    assert prefill_seconds(H100, QWEN, 1000) == pytest.approx(1.64e13 / 989.5e12)  # 16.6 ms
    assert prefill_seconds(MI300X, QWEN, 1000) == pytest.approx(1.64e13 / 1300e12)  # 12.6 ms


def test_short_prefill_is_memory_bound():
    # 10 tokens of math take 0.17 ms, but reading 16.4 GB of weights takes 4.9 ms
    assert prefill_seconds(H100, QWEN, 10) == pytest.approx(16.4e9 / 3.35e12)


def test_decode_one_conversation_is_memory_bound():
    # weights 16.4 GB + 1000 tokens x 147,456 B of KV cache
    bytes_read = 16.4e9 + 1000 * 147_456
    assert decode_step_seconds(H100, QWEN, 1, 1000) == pytest.approx(bytes_read / 3.35e12)  # 4.9 ms
    assert decode_step_seconds(MI300X, QWEN, 1, 1000) == pytest.approx(
        bytes_read / 5.3e12
    )  # 3.1 ms


def test_mi300x_is_faster_than_h100_at_both_phases_on_paper():
    assert decode_step_seconds(MI300X, QWEN, 1, 1000) < decode_step_seconds(H100, QWEN, 1, 1000)
    assert prefill_seconds(MI300X, QWEN, 1000) < prefill_seconds(H100, QWEN, 1000)


def test_big_batch_decode_becomes_compute_bound():
    # Weights are shared, so math overtakes memory past ~295 conversations on H100
    small = decode_step_seconds(H100, QWEN, 256, 0)
    big = decode_step_seconds(H100, QWEN, 512, 0)
    assert small == pytest.approx(16.4e9 / 3.35e12)
    assert big == pytest.approx(2 * 8.2e9 * 512 / 989.5e12)


def test_efficiency_slows_things_down():
    ideal = decode_step_seconds(H100, QWEN, 1, 1000)
    assert decode_step_seconds(H100, QWEN, 1, 1000, memory_efficiency=0.5) == pytest.approx(
        2 * ideal
    )
    assert prefill_seconds(H100, QWEN, 1000, compute_efficiency=0.5) > 30 * MS
