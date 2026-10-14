import pytest

from gpumix.gpus import load_gpus
from gpumix.memory import memory_fit
from gpumix.models import load_models
from gpumix.perf import decode_step_seconds, prefill_seconds
from gpumix.sim import percentile, simulate, summarize
from gpumix.traffic import Request, synth

H100 = load_gpus()["H100-SXM"]
QWEN = load_models()["qwen3-8b"]


def req(arrival, n_in, n_out):
    return Request(arrival_s=arrival, input_tokens=n_in, output_tokens=n_out)


def test_lone_request_ttft_is_its_prefill():
    run = simulate(H100, QWEN, [req(0, 1000, 1)])
    assert run.served[0].ttft == pytest.approx(prefill_seconds(H100, QWEN, 1000))


def test_lone_request_decodes_one_token_per_step():
    # first token comes from prefill; tokens 2..5 each take one decode step,
    # with the cache holding the prompt plus the tokens written so far
    run = simulate(H100, QWEN, [req(0, 1000, 5)])
    s = run.served[0]
    steps = sum(decode_step_seconds(H100, QWEN, 1, 1000 + k) for k in range(1, 5))
    assert s.finish_s - s.first_token_s == pytest.approx(steps)
    assert s.tpot == pytest.approx(steps / 4)  # about 4.9 ms on H100


def test_ttft_counts_from_arrival_after_idle_gap():
    run = simulate(H100, QWEN, [req(10.0, 1000, 1)])
    assert run.served[0].first_token_s == pytest.approx(10.0 + prefill_seconds(H100, QWEN, 1000))
    assert run.served[0].ttft == pytest.approx(prefill_seconds(H100, QWEN, 1000))


def test_second_request_waits_for_first_prefill():
    run = simulate(H100, QWEN, [req(0, 1000, 3), req(0, 1000, 3)])
    a, b = run.served
    pf = prefill_seconds(H100, QWEN, 1000)
    assert a.ttft == pytest.approx(pf)
    assert b.ttft == pytest.approx(2 * pf)  # waited for a's prefill, then its own


def test_two_requests_share_decode_steps():
    # after both prefills, each decode step serves both, so they finish together
    run = simulate(H100, QWEN, [req(0, 1000, 3), req(0, 1000, 3)])
    a, b = run.served
    assert a.finish_s == pytest.approx(b.finish_s)


def test_full_kv_cache_makes_requests_queue():
    cap = memory_fit(H100, QWEN).kv_tokens  # 377,061 tokens
    big = cap // 2 + 1  # two of these can't fit at once
    run = simulate(H100, QWEN, [req(0, big - 1, 1), req(0, big - 1, 1)])
    a, b = run.served
    # b can only start after a finishes and frees its room
    assert b.first_token_s == pytest.approx(a.finish_s + prefill_seconds(H100, QWEN, big - 1))


def test_request_bigger_than_memory_is_rejected():
    with pytest.raises(ValueError, match="KV tokens"):
        simulate(H100, QWEN, [req(0, 400_000, 1)])


def test_every_request_finishes_with_all_its_tokens():
    traffic = synth(300, rate_per_s=5, median_input_tokens=800, median_output_tokens=200, seed=3)
    run = simulate(H100, QWEN, traffic)
    assert all(s.generated == s.request.output_tokens for s in run.served)
    assert all(s.request.arrival_s <= s.first_token_s <= s.finish_s for s in run.served)


def test_percentile_nearest_rank():
    values = [float(v) for v in range(1, 101)]
    assert percentile(values, 50) == 50
    assert percentile(values, 99) == 99
    assert percentile([7.0], 99) == 7


def test_summary_cost_matches_hand_math():
    # one request: 1000 in, 1 out. Billed time = prefill. Cost = $4.41/h x time / 1 token x 1e6
    run = simulate(H100, QWEN, [req(0, 1000, 1)])
    summary = summarize(run, usd_per_gpu_hour=4.41)
    pf = prefill_seconds(H100, QWEN, 1000)
    assert summary.usd_per_million_output_tokens == pytest.approx(4.41 * pf / 3600 * 1e6)
    assert summary.output_tokens_per_s == pytest.approx(1 / pf)
