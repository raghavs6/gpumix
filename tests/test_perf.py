import pytest

from gpumix.gpus import load_gpus
from gpumix.links import load_links
from gpumix.models import load_models
from gpumix.perf import decode_step_seconds, prefill_seconds, step_seconds

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


NVLINK = load_links()["nvlink-h100"]
SWAP = 2 * 36  # two all-reduces per layer, 36 layers


def test_tp2_decode_halves_reads_and_adds_swaps():
    # each GPU reads half of (16.4 GB weights + 1000 tokens of KV): about 2.5 ms.
    # 72 swaps of one token's activations (4096 x 2 B = 8 KB): 6.45 us delay each,
    # plus 8 KB over 450 GB/s (a 2-GPU ring sends the whole array once)
    reads = (16.4e9 + 1000 * 147_456) / 2 / 3.35e12
    swaps = SWAP * (6.45e-6 + 8192 / 450e9)
    t = step_seconds(H100, QWEN, 1, 1000, tp=2, tp_link=NVLINK)
    assert t == pytest.approx(reads + swaps)  # about 2.47 + 0.47 = 2.94 ms vs 4.94 ms on one GPU


def test_tp2_prefill_halves_math_and_adds_swaps():
    math = 2 * 8.2e9 * 1000 / 2 / 989.5e12  # 8.3 ms
    swaps = SWAP * (6.45e-6 + 1000 * 8192 / 450e9)  # 0.46 ms delay + 1.31 ms sending
    t = step_seconds(H100, QWEN, 1000, 0, tp=2, tp_link=NVLINK)
    assert t == pytest.approx(math + swaps)  # about 10.1 ms vs 16.6 ms on one GPU


def test_tp_needs_a_measured_delay_for_that_gpu_count():
    with pytest.raises(ValueError, match="no measured all-reduce delay for 3 GPUs"):
        step_seconds(H100, QWEN, 1, 1000, tp=3, tp_link=NVLINK)
    with pytest.raises(ValueError, match="needs a tp_link"):
        step_seconds(H100, QWEN, 1, 1000, tp=2)
