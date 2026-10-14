import pytest

from gpumix.gpus import load_gpus
from gpumix.memory import memory_fit
from gpumix.models import load_models
from gpumix.perf import decode_step_seconds, prefill_seconds, step_seconds
from gpumix.sim import percentile, simulate, summarize, vllm_defaults
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
    # plain mode: each prompt is its own step
    run = simulate(H100, QWEN, [req(0, 1000, 3), req(0, 1000, 3)], chunked_prefill=False)
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
    run = simulate(H100, QWEN, [req(0, big - 1, 1), req(0, big - 1, 1)], chunked_prefill=False)
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


MI300X = load_gpus()["MI300X"]


def test_vllm_defaults_per_gpu():
    # 80 GB H100 is in the >= 70 GiB tier; 192 GB MI300X is in the >= 160 GiB tier
    assert vllm_defaults(H100) == (8192, 1024)
    assert vllm_defaults(MI300X) == (16384, 1024)


def test_chunked_lone_short_request_matches_plain():
    # a prompt that fits in one budget is one step either way
    for chunked in (True, False):
        run = simulate(H100, QWEN, [req(0, 1000, 5)], chunked_prefill=chunked)
        assert run.served[0].ttft == pytest.approx(prefill_seconds(H100, QWEN, 1000))


def test_chunked_long_prompt_is_split_across_steps():
    # 20,000 tokens with an 8192 budget: chunks of 8192, 8192, 3616.
    # Each later chunk reads the KV the earlier chunks wrote.
    run = simulate(H100, QWEN, [req(0, 20_000, 1)])
    expected = (
        step_seconds(H100, QWEN, 8192, 0)
        + step_seconds(H100, QWEN, 8192, 8192)
        + step_seconds(H100, QWEN, 3616, 16384)
    )
    assert run.served[0].ttft == pytest.approx(expected)


def test_chunked_long_prompt_does_not_stall_decoders():
    # a is writing its answer when a 60k prompt arrives. Plain mode stops a
    # for the whole prompt; chunked mode keeps giving a a token every step.
    traffic = [req(0, 100, 50), req(0.001, 60_000, 1)]
    plain = simulate(H100, QWEN, traffic, chunked_prefill=False).served[0]
    mixed = simulate(H100, QWEN, traffic).served[0]
    assert plain.max_gap_s >= prefill_seconds(H100, QWEN, 60_000)
    # worst chunked step: one decode plus a full chunk on top of earlier chunks
    worst = step_seconds(H100, QWEN, 8192, 100 + 50 + 60_000)
    assert mixed.max_gap_s <= worst * (1 + 1e-9)
    assert mixed.max_gap_s < plain.max_gap_s / 5


def test_max_seqs_makes_requests_queue():
    run = simulate(H100, QWEN, [req(0, 100, 3), req(0, 100, 3)], max_seqs=1)
    a, b = run.served
    assert b.first_token_s > a.finish_s  # b is admitted only after a leaves
