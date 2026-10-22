import os
import socket
import subprocess
from pathlib import Path

from gpumix.calibrate import decode_sweep, flatten, prefill_sweep
from gpumix.measured import read_vllm_result
from gpumix.traffic import read_traffic, window

ROOT = Path(__file__).parent.parent
FAKES = Path(__file__).parent / "fakebin"


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def bench(tmp_path, gpu="H100-SXM", **env):
    out = tmp_path / "run"
    result = subprocess.run(
        ["bash", "scripts/bench.sh", gpu],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        env=os.environ
        | {"PATH": f"{FAKES}:{os.environ['PATH']}", "OUT": str(out), "PORT": str(free_port())}
        | env,
    )
    return result, out


def test_runs_every_measurement_with_the_simulators_settings(tmp_path):
    result, out = bench(tmp_path)
    assert result.returncode == 0, result.stderr
    assert "batch_tokens: 8192  max_seqs: 1024" in (out / "env.txt").read_text()
    expected = {
        "prefill": flatten(prefill_sweep()),
        "decode": flatten(decode_sweep()),
        "chat": window(read_traffic(ROOT / "data/traces/azure2023_conv.jsonl"), 600, 600),
        "code": window(read_traffic(ROOT / "data/traces/azure2023_code.jsonl"), 600, 600),
    }
    for run, requests in expected.items():
        read_vllm_result(out / f"{run}.json", requests)  # raises if lengths don't match


def test_refuses_an_h100_that_isnt_sxm(tmp_path):
    result, out = bench(tmp_path, FAKE_GPU_NAME="NVIDIA H100 PCIe")
    assert result.returncode != 0
    assert "isn't the SXM" in result.stderr
    assert not (out / "prefill.json").exists()


def test_unknown_gpu_needs_explicit_limits(tmp_path):
    result, _ = bench(tmp_path, gpu="RTX-3090")
    assert result.returncode != 0
    assert "set BATCH_TOKENS and MAX_SEQS" in result.stderr
    result, out = bench(tmp_path, gpu="RTX-3090", BATCH_TOKENS="2048", MAX_SEQS="256")
    assert result.returncode == 0, result.stderr
