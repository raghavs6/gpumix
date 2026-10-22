"""Measured runs: read a `vllm bench serve` result into the simulator's Summary.

The result file comes from `--save-result --save-detailed`, which lists every
request in trace order: time to first token, the gap before each later chunk
of tokens (`itls`), and total latency, all in seconds from when the request
was sent. Time per output token is computed the way both vLLM and the
simulator do: (latency - time to first token) / (tokens - 1).
"""

import json
import math
from pathlib import Path
from typing import Any

from gpumix.sim import Summary, percentile
from gpumix.traffic import Request


def read_vllm_result(path: Path, requests: list[Request]) -> dict[str, Any]:
    """Load a result file and check it ran `requests` exactly. Raises ValueError if not."""
    result = json.loads(path.read_text())
    if "itls" not in result:
        raise ValueError(f"{path}: no per-request data; run with --save-detailed")
    errors = [e for e in result["errors"] if e]
    if errors:
        raise ValueError(f"{path}: {len(errors)} requests failed, first: {errors[0]}")
    for field, want in (
        ("input_lens", [r.input_tokens for r in requests]),
        ("output_lens", [r.output_tokens for r in requests]),
    ):
        if result[field] != want:
            raise ValueError(f"{path}: {field} differ from the trace")
    return result


def summarize_vllm(
    result: dict[str, Any], gpus: dict[str, int], usd_per_gpu_hour: dict[str, float]
) -> Summary:
    """Same fields as sim.summarize. Billing runs for vLLM's measured `duration`."""
    usd_per_hour = sum(usd_per_gpu_hour[name] * n for name, n in gpus.items())
    rows = list(
        zip(
            result["ttfts"], result["latencies"], result["itls"], result["output_lens"], strict=True
        )
    )
    tpots = [(lat - ttft) / (n - 1) for ttft, lat, _, n in rows if n > 1]
    gaps = [max(itl) for _, _, itl, n in rows if n > 1 and itl]
    tokens = sum(result["output_lens"])
    duration = result["duration"]
    return Summary(
        ttft_p50_s=percentile(result["ttfts"], 50),
        ttft_p99_s=percentile(result["ttfts"], 99),
        tpot_p50_s=percentile(tpots, 50) if tpots else math.nan,
        tpot_p99_s=percentile(tpots, 99) if tpots else math.nan,
        max_gap_p99_s=percentile(gaps, 99) if gaps else math.nan,
        output_tokens_per_s=tokens / duration,
        usd_per_million_output_tokens=usd_per_hour * duration / 3600 / tokens * 1e6,
        usd_per_1k_requests=usd_per_hour * duration / 3600 / len(rows) * 1000,
    )
