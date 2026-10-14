import statistics

import pytest

from gpumix.traffic import Request, read_traffic, synth, write_traffic


def test_round_trip(tmp_path):
    requests = synth(100, rate_per_s=5, median_input_tokens=800, median_output_tokens=200, seed=1)
    path = tmp_path / "traffic.jsonl"
    write_traffic(path, requests)
    assert read_traffic(path) == requests


def test_blank_lines_are_skipped(tmp_path):
    path = tmp_path / "traffic.jsonl"
    path.write_text('{"arrival_s": 0, "input_tokens": 5, "output_tokens": 3}\n\n')
    assert read_traffic(path) == [Request(arrival_s=0, input_tokens=5, output_tokens=3)]


@pytest.mark.parametrize(
    "bad_line",
    [
        '{"arrival_s": 1, "input_tokens": 0, "output_tokens": 3}',  # zero-length prompt
        '{"arrival_s": 1, "input_tokens": 5}',  # missing field
        '{"arrival_s": 1, "input_tokens": 5, "output_tokens": 3, "user": "x"}',  # unknown field
        "not json",
        '{"arrival_s": 0.5, "input_tokens": 5, "output_tokens": 3}',  # goes backwards
    ],
)
def test_bad_line_is_named(tmp_path, bad_line):
    path = tmp_path / "traffic.jsonl"
    path.write_text('{"arrival_s": 1, "input_tokens": 5, "output_tokens": 3}\n' + bad_line + "\n")
    with pytest.raises(ValueError, match="line 2"):
        read_traffic(path)


def test_same_seed_same_trace_different_seed_different_trace():
    def make(seed):
        return synth(50, rate_per_s=5, median_input_tokens=800, median_output_tokens=200, seed=seed)

    assert make(7) == make(7)
    assert make(7) != make(8)


def test_synth_matches_requested_rate_and_medians():
    requests = synth(
        20_000, rate_per_s=10, median_input_tokens=800, median_output_tokens=200, seed=0
    )
    rate = len(requests) / requests[-1].arrival_s
    assert rate == pytest.approx(10, rel=0.03)
    assert statistics.median(r.input_tokens for r in requests) == pytest.approx(800, rel=0.05)
    assert statistics.median(r.output_tokens for r in requests) == pytest.approx(200, rel=0.05)


def test_synth_has_a_long_tail():
    requests = synth(
        20_000, rate_per_s=10, median_input_tokens=800, median_output_tokens=200, seed=0
    )
    inputs = sorted(r.input_tokens for r in requests)
    p99 = inputs[int(0.99 * len(inputs))]
    assert p99 > 5 * 800  # lognormal, sigma 1: p99 is about e^2.33 = 10x the median
