"""Calibration: measure how close a real GPU gets to its spec-sheet speeds.

The speed model needs two numbers per GPU. Prefill is limited by math, so
lone prompts of a few lengths give the compute efficiency. Decode is limited
by memory reads, so batches of a few sizes writing long answers give the
memory efficiency. Each point runs once to warm up, then REPEATS times.
Points are spaced `gap_s` apart so one finishes before the next starts.
"""

import statistics
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from gpumix.gpus import GPU
from gpumix.models import Model
from gpumix.traffic import Request

PREFILL_LENGTHS = (512, 2048, 8192)  # prompt tokens
DECODE_BATCHES = (1, 8, 32, 128)  # conversations writing at once
DECODE_PROMPT_TOKENS = 1024
DECODE_ANSWER_TOKENS = 256
REPEATS = 3


@dataclass(frozen=True)
class Probe:
    """Requests that arrive together and are timed as one measurement."""

    size: int  # prompt length (prefill) or batch size (decode)
    warmup: bool
    requests: tuple[Request, ...]


def prefill_sweep(gap_s: float = 10.0) -> list[Probe]:
    """One lone prompt per probe, answered with a single token."""
    probes: list[Probe] = []
    for size in PREFILL_LENGTHS:
        for repeat in range(1 + REPEATS):
            arrival = len(probes) * gap_s
            request = Request(arrival_s=arrival, input_tokens=size, output_tokens=1)
            probes.append(Probe(size=size, warmup=repeat == 0, requests=(request,)))
    return probes


def decode_sweep(gap_s: float = 30.0) -> list[Probe]:
    """`size` conversations arriving at once, each writing a long answer."""
    probes: list[Probe] = []
    for size in DECODE_BATCHES:
        for repeat in range(1 + REPEATS):
            request = Request(
                arrival_s=len(probes) * gap_s,
                input_tokens=DECODE_PROMPT_TOKENS,
                output_tokens=DECODE_ANSWER_TOKENS,
            )
            probes.append(Probe(size=size, warmup=repeat == 0, requests=(request,) * size))
    return probes


def flatten(probes: list[Probe]) -> list[Request]:
    return [r for p in probes for r in p.requests]


@dataclass(frozen=True)
class Fit:
    """Measured share of spec-sheet speed, per sweep point and overall (the median point)."""

    compute_efficiency: float
    memory_efficiency: float
    prefill_points: dict[int, float]  # prompt tokens -> compute efficiency
    decode_points: dict[int, float]  # batch size -> memory efficiency


def _measured(
    probes: list[Probe], per_request: list[Any], pick: Callable[[list[Any]], float]
) -> dict[int, float]:
    """Per probe size, the median over repeats of `pick(that probe's rows)`. Skips warm-ups."""
    values: dict[int, list[float]] = {}
    start = 0
    for p in probes:
        rows = per_request[start : start + len(p.requests)]
        start += len(p.requests)
        if not p.warmup:
            values.setdefault(p.size, []).append(pick(rows))
    return {size: statistics.median(v) for size, v in values.items()}


def fit(gpu: GPU, model: Model, prefill_ttfts: list[float], decode_itls: list[list[float]]) -> Fit:
    """Efficiencies from the two sweeps' measurements, listed in sweep order.

    Prefill: a lone prompt's time to first token, against the math it needs
    (2 operations per parameter per token). Its memory reads and the fixed
    costs of a request (scheduling, HTTP) are left in, so short prompts may
    come out lower; the per-point numbers show whether they do.
    Decode: the median gap between tokens, against the bytes one step reads
    (weights plus every cached token's KV, at the answer's middle length).
    """
    ttft = _measured(prefill_sweep(), prefill_ttfts, lambda rows: rows[0])
    gap = _measured(decode_sweep(), decode_itls, lambda rows: statistics.median(sum(rows, [])))
    compute = {
        n: 2 * model.params_b * 1e9 * n / (gpu.bf16_dense_tflops * 1e12 * t)
        for n, t in ttft.items()
    }
    context = DECODE_PROMPT_TOKENS + DECODE_ANSWER_TOKENS / 2
    memory = {
        b: (model.weight_bytes() + b * context * model.kv_bytes_per_token())
        / (gpu.mem_bandwidth_tbps * 1e12 * t)
        for b, t in gap.items()
    }
    return Fit(
        compute_efficiency=statistics.median(compute.values()),
        memory_efficiency=statistics.median(memory.values()),
        prefill_points=compute,
        decode_points=memory,
    )
