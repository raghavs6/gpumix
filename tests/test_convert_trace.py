import subprocess
import sys
from pathlib import Path

import pytest

from gpumix.traffic import Request, read_traffic

SCRIPT = Path(__file__).parent.parent / "scripts" / "convert_trace.py"
HEADER = "TIMESTAMP,ContextTokens,GeneratedTokens\n"


def run(tmp_path, csv_text):
    csv_path = tmp_path / "trace.csv"
    jsonl_path = tmp_path / "trace.jsonl"
    csv_path.write_text(HEADER + csv_text)
    result = subprocess.run(
        [sys.executable, SCRIPT, csv_path, jsonl_path], capture_output=True, text=True
    )
    return result, jsonl_path


def test_converts_azure_rows(tmp_path):
    result, jsonl_path = run(
        tmp_path,
        "2023-11-16 18:17:03.9799600,4808,10\n"
        "2023-11-16 18:17:04.0319600,3180,8\n"
        "2023-11-16 18:18:04.0319600,549,173\n",
    )
    assert result.returncode == 0, result.stderr
    requests = read_traffic(jsonl_path)
    assert requests[0] == Request(arrival_s=0, input_tokens=4808, output_tokens=10)
    assert requests[1].arrival_s == pytest.approx(0.052)
    assert requests[2].arrival_s == pytest.approx(60.052)
    assert [(r.input_tokens, r.output_tokens) for r in requests] == [
        (4808, 10),
        (3180, 8),
        (549, 173),
    ]


@pytest.mark.parametrize(
    "bad_row",
    [
        "2023-11-16 18:17:05.0000000,0,8",  # zero-token prompt
        "2023-11-16 18:17:02.0000000,100,8",  # goes backwards
        "not a time,100,8",
    ],
)
def test_bad_row_is_named(tmp_path, bad_row):
    result, jsonl_path = run(tmp_path, "2023-11-16 18:17:03.9799600,4808,10\n" + bad_row + "\n")
    assert result.returncode != 0
    assert "line 3" in result.stderr
    assert not jsonl_path.exists()
