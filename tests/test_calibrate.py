import subprocess
import sys
from pathlib import Path

from gpumix.calibrate import (
    DECODE_BATCHES,
    PREFILL_LENGTHS,
    REPEATS,
    decode_sweep,
    flatten,
    prefill_sweep,
)
from gpumix.traffic import read_traffic

SCRIPT = Path(__file__).parent.parent / "scripts" / "make_sweeps.py"


def test_prefill_sweep_is_lone_prompts_with_one_token_answers():
    probes = prefill_sweep(gap_s=10)
    assert [p.size for p in probes] == [n for n in PREFILL_LENGTHS for _ in range(1 + REPEATS)]
    assert [p.warmup for p in probes[:4]] == [True, False, False, False]
    for p in probes:
        (request,) = p.requests
        assert (request.input_tokens, request.output_tokens) == (p.size, 1)
    assert [p.requests[0].arrival_s for p in probes[:3]] == [0, 10, 20]


def test_decode_sweep_batches_arrive_together():
    probes = decode_sweep(gap_s=30)
    assert [p.size for p in probes] == [n for n in DECODE_BATCHES for _ in range(1 + REPEATS)]
    for i, p in enumerate(probes):
        assert len(p.requests) == p.size
        assert {r.arrival_s for r in p.requests} == {i * 30}
    assert len(flatten(probes)) == (1 + REPEATS) * sum(DECODE_BATCHES)


def test_script_writes_both_files(tmp_path):
    result = subprocess.run([sys.executable, SCRIPT, tmp_path], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert len(read_traffic(tmp_path / "prefill.jsonl")) == 12
    assert len(read_traffic(tmp_path / "decode.jsonl")) == 676
