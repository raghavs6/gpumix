import subprocess
import sys
from pathlib import Path

import pytest

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


def test_fit_recovers_the_efficiencies_a_gpu_was_built_with():
    from fake_results import sweeps

    from gpumix.calibrate import fit
    from gpumix.gpus import load_gpus
    from gpumix.models import load_models

    gpu, model = load_gpus()["H100-SXM"], load_models()["qwen3-8b"]
    prefill, decode = sweeps(gpu, model, compute=0.6, memory=0.8)
    f = fit(gpu, model, prefill["ttfts"], decode["itls"])
    assert f.compute_efficiency == pytest.approx(0.6)
    assert f.memory_efficiency == pytest.approx(0.8)
    assert set(f.prefill_points) == set(PREFILL_LENGTHS)
    assert set(f.decode_points) == set(DECODE_BATCHES)


def test_fit_skips_warmups():
    from fake_results import sweeps

    from gpumix.calibrate import fit
    from gpumix.gpus import load_gpus
    from gpumix.models import load_models

    gpu, model = load_gpus()["H100-SXM"], load_models()["qwen3-8b"]
    prefill, decode = sweeps(gpu, model, compute=0.6, memory=0.8)
    prefill["ttfts"][0] *= 100  # a slow first request, as a cold GPU gives
    assert fit(gpu, model, prefill["ttfts"], decode["itls"]).prefill_points[512] == pytest.approx(
        0.6
    )
