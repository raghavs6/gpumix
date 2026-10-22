import json
import math
import subprocess
import sys
from pathlib import Path

from gpumix.traffic import Request, write_traffic

SCRIPT = Path(__file__).parent.parent / "scripts" / "to_vllm_trace.py"
AZURE_CHAT = Path(__file__).parent.parent / "data" / "traces" / "azure2023_conv.jsonl"


def run(traffic_path, out_path, *options):
    result = subprocess.run(
        [sys.executable, SCRIPT, traffic_path, out_path, *options], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    return [json.loads(line) for line in out_path.read_text().splitlines()]


def test_maps_fields_and_gives_fresh_ids(tmp_path):
    traffic = tmp_path / "traffic.jsonl"
    write_traffic(
        traffic,
        [
            Request(arrival_s=0, input_tokens=20, output_tokens=5),
            Request(arrival_s=0.5, input_tokens=16, output_tokens=3),
        ],
    )
    assert run(traffic, tmp_path / "out.jsonl") == [
        {"timestamp": 0.0, "input_length": 20, "output_length": 5, "hash_ids": [0, 1]},
        {"timestamp": 0.5, "input_length": 16, "output_length": 3, "hash_ids": [2]},
    ]


def test_azure_chat_trace(tmp_path):
    rows = run(AZURE_CHAT, tmp_path / "out.jsonl")
    assert len(rows) == 19366
    # Enough ids to build each prompt in full, and none shared between prompts.
    assert all(len(r["hash_ids"]) == math.ceil(r["input_length"] / 16) for r in rows)
    ids = [i for r in rows for i in r["hash_ids"]]
    assert len(ids) == len(set(ids))
    timestamps = [r["timestamp"] for r in rows]
    assert timestamps == sorted(timestamps)


def test_azure_chat_window(tmp_path):
    rows = run(AZURE_CHAT, tmp_path / "out.jsonl", "--start-s", "600", "--duration-s", "600")
    assert len(rows) == 3118
    assert 0 <= rows[0]["timestamp"] and rows[-1]["timestamp"] < 600
