import json
import subprocess
import sys
from pathlib import Path

from fake_results import from_run, sweeps, write

from gpumix.gpus import load_gpus
from gpumix.models import load_models
from gpumix.sim import simulate
from gpumix.traffic import synth

SCRIPT = Path(__file__).parent.parent / "scripts" / "predicted_vs_measured.py"
GPU, MODEL = load_gpus()["H100-SXM"], load_models()["qwen3-8b"]


def test_a_gpu_that_matches_the_model_is_predicted_exactly(tmp_path):
    prefill, decode = sweeps(GPU, MODEL, compute=0.6, memory=0.8)
    write(tmp_path / "prefill.json", prefill)
    write(tmp_path / "decode.json", decode)
    chat = synth(200, rate_per_s=20, median_input_tokens=1000, median_output_tokens=130, seed=3)
    (tmp_path / "traces").mkdir()
    (tmp_path / "traces" / "chat.vllm.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "timestamp": r.arrival_s,
                    "input_length": r.input_tokens,
                    "output_length": r.output_tokens,
                }
            )
            + "\n"
            for r in chat
        )
    )
    run = simulate(GPU, MODEL, chat, compute_efficiency=0.6, memory_efficiency=0.8)
    write(tmp_path / "chat.json", from_run(run))

    result = subprocess.run(
        [sys.executable, SCRIPT, tmp_path, "H100-SXM"], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    table = (tmp_path / "calibration.md").read_text()
    assert "Compute efficiency **0.600**" in table
    assert "memory efficiency **0.800**" in table
    rows = [line for line in table.splitlines() if line.startswith("| TTFT") or "per token" in line]
    assert len(rows) == 4
    assert all(line.endswith("| +0% |") for line in rows)
    assert "## code" not in table  # no code trace in this run
