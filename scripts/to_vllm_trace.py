"""Convert gpumix traffic JSONL into the trace format `vllm bench serve` replays.

vLLM's `--dataset-name timed_trace` (vLLM 0.32.0 and later) builds each prompt
from `hash_ids`: 16 random tokens per id, glued together. With no ids the
prompt comes out empty, so every request gets ceil(input_length / 16) ids.
Two requests with the same id would share those tokens and vLLM's prefix
cache would skip re-reading them; the model assumes every prompt is read in
full, so ids are never reused.

    python scripts/to_vllm_trace.py data/traces/azure2023_conv.jsonl conv.vllm.jsonl

With --start-s and --duration-s, only that window of the trace is kept, with
times counted from --start-s.
"""

import argparse
import json
import math
from pathlib import Path

from gpumix.traffic import Request, read_traffic, window

TOKENS_PER_HASH_ID = 16  # vLLM's --timed-trace-chunk-hash-size default


def to_vllm(requests: list[Request]) -> list[dict]:
    rows = []
    next_id = 0
    for r in requests:
        count = math.ceil(r.input_tokens / TOKENS_PER_HASH_ID)
        rows.append(
            {
                "timestamp": r.arrival_s,
                "input_length": r.input_tokens,
                "output_length": r.output_tokens,
                "hash_ids": list(range(next_id, next_id + count)),
            }
        )
        next_id += count
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("traffic_path", type=Path)
    parser.add_argument("vllm_trace_path", type=Path)
    parser.add_argument("--start-s", type=float, default=0.0)
    parser.add_argument("--duration-s", type=float, default=math.inf)
    args = parser.parse_args()
    rows = to_vllm(window(read_traffic(args.traffic_path), args.start_s, args.duration_s))
    args.vllm_trace_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    print(f"wrote {len(rows)} requests to {args.vllm_trace_path}")


if __name__ == "__main__":
    main()
