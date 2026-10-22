import json

import pytest

from gpumix.measured import read_vllm_result, summarize_vllm
from gpumix.traffic import Request

REQUESTS = [
    Request(arrival_s=0, input_tokens=100, output_tokens=3),
    Request(arrival_s=1, input_tokens=200, output_tokens=1),
]


def result(**changes):
    # Shaped like `vllm bench serve --save-result --save-detailed` output.
    base = {
        "duration": 3.6,
        "input_lens": [100, 200],
        "output_lens": [3, 1],
        "ttfts": [0.1, 0.3],
        "latencies": [0.5, 0.3],
        "itls": [[0.1, 0.3], []],
        "errors": ["", ""],
    }
    return base | changes


def write(tmp_path, data):
    path = tmp_path / "result.json"
    path.write_text(json.dumps(data))
    return path


def test_summary_matches_hand_numbers():
    s = summarize_vllm(result(), {"H100-SXM": 1}, {"H100-SXM": 3.6})
    assert s.ttft_p50_s == 0.1 and s.ttft_p99_s == 0.3
    assert s.tpot_p50_s == pytest.approx(0.2)  # (0.5 - 0.1) / 2; the one-token answer has none
    assert s.max_gap_p99_s == 0.3
    assert s.output_tokens_per_s == pytest.approx(4 / 3.6)
    # $3.60/h for 3.6 s is $0.0036, over 4 tokens and 2 requests.
    assert s.usd_per_million_output_tokens == pytest.approx(900)
    assert s.usd_per_1k_requests == pytest.approx(1.8)


def test_reads_a_good_result(tmp_path):
    assert read_vllm_result(write(tmp_path, result()), REQUESTS)["duration"] == 3.6


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"errors": ["", "timeout"]}, "1 requests failed"),
        ({"input_lens": [0, 200]}, "input_lens differ"),  # e.g. a trace without hash_ids
        ({"output_lens": [2, 1]}, "output_lens differ"),
    ],
)
def test_rejects_a_bad_run(tmp_path, changes, message):
    with pytest.raises(ValueError, match=message):
        read_vllm_result(write(tmp_path, result(**changes)), REQUESTS)


def test_needs_detailed_results(tmp_path):
    data = result()
    del data["itls"]
    with pytest.raises(ValueError, match="--save-detailed"):
        read_vllm_result(write(tmp_path, data), REQUESTS)
