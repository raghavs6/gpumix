"""Calibration: measure how close a real GPU gets to its spec-sheet speeds.

The speed model needs two numbers per GPU. Prefill is limited by math, so
lone prompts of a few lengths give the compute efficiency. Decode is limited
by memory reads, so batches of a few sizes writing long answers give the
memory efficiency. Each point runs once to warm up, then REPEATS times.
Points are spaced `gap_s` apart so one finishes before the next starts.
"""

from dataclasses import dataclass

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
