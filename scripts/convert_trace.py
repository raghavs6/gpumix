"""Convert an Azure LLM inference trace CSV into gpumix traffic JSONL.

Input columns: TIMESTAMP, ContextTokens, GeneratedTokens (Azure 2023 and 2024 traces).
Output: one request per line, arrival_s counted from the first request.

    python scripts/convert_trace.py AzureLLMInferenceTrace_conv.csv data/traces/azure2023_conv.jsonl
"""

import argparse
import csv
from datetime import datetime
from pathlib import Path

from pydantic import ValidationError

from gpumix.traffic import Request, write_traffic


def parse_timestamp(text: str) -> datetime:
    # Azure writes 7 decimal places (e.g. 18:17:03.9799600); datetime keeps 6.
    # Dropping the 7th digit loses 0.1 microseconds.
    date, _, fraction = text.partition(".")
    return datetime.fromisoformat(f"{date}.{fraction[:6]}" if fraction else date)


def convert(csv_path: Path) -> list[Request]:
    """Read an Azure trace CSV. Raises ValueError naming the first bad line."""
    requests: list[Request] = []
    start = None
    with csv_path.open(newline="") as file:
        # Line 1 is the header, so data starts on line 2.
        for number, row in enumerate(csv.DictReader(file), start=2):
            try:
                time = parse_timestamp(row["TIMESTAMP"])
                start = start or time
                request = Request(
                    arrival_s=(time - start).total_seconds(),
                    input_tokens=int(row["ContextTokens"]),
                    output_tokens=int(row["GeneratedTokens"]),
                )
            except (KeyError, ValueError, ValidationError) as error:
                raise ValueError(f"line {number}: {error}") from error
            if requests and request.arrival_s < requests[-1].arrival_s:
                raise ValueError(f"line {number}: TIMESTAMP goes backwards")
            requests.append(request)
    return requests


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("jsonl_path", type=Path)
    args = parser.parse_args()
    requests = convert(args.csv_path)
    write_traffic(args.jsonl_path, requests)
    print(f"wrote {len(requests)} requests to {args.jsonl_path}")


if __name__ == "__main__":
    main()
