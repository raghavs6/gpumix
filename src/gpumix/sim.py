"""Simulator: replay traffic through one GPU that does both prefill and decode.

It works like an inference server with continuous batching. Requests wait
in a first-come-first-served line. Whenever the GPU finishes a step it
picks the next one:

- If the first waiting request's KV cache fits, prefill it. Prefill also
  produces the request's first token.
- Otherwise, run one decode step for every conversation in progress, each
  writing one more token.

Each step takes as long as the speed model says, so batching (one weight
read shared by the whole batch) is built in. A request reserves KV room
for its whole prompt and answer when it starts, so nothing is ever
evicted; real servers allocate as they go and can preempt.

Every number here is MODELLED, NOT MEASURED.
"""

import math
from collections import deque
from dataclasses import dataclass

from gpumix.gpus import GPU
from gpumix.memory import memory_fit
from gpumix.models import Model
from gpumix.perf import decode_step_seconds, prefill_seconds
from gpumix.traffic import Request


@dataclass
class Served:
    """What happened to one request."""

    request: Request
    first_token_s: float = math.nan  # clock time the first token came out
    finish_s: float = math.nan  # clock time the last token came out
    generated: int = 0

    @property
    def ttft(self) -> float:
        """Time to first token: waiting in line plus prefill."""
        return self.first_token_s - self.request.arrival_s

    @property
    def tpot(self) -> float:
        """Average time per output token after the first one (nan for one-token answers)."""
        if self.request.output_tokens == 1:
            return math.nan
        return (self.finish_s - self.first_token_s) / (self.request.output_tokens - 1)


@dataclass
class Run:
    served: list[Served]
    end_s: float  # clock time the last request finished


def simulate(gpu: GPU, model: Model, requests: list[Request]) -> Run:
    """Replay `requests` (sorted by arrival) through one GPU.

    Raises ValueError if a single request needs more KV cache than the GPU has.
    """
    capacity = memory_fit(gpu, model).kv_tokens
    for r in requests:
        if r.input_tokens + r.output_tokens > capacity:
            raise ValueError(
                f"request needs {r.input_tokens + r.output_tokens} KV tokens; GPU holds {capacity}"
            )

    served = [Served(r) for r in requests]
    upcoming = deque(served)
    waiting: deque[Served] = deque()
    running: list[Served] = []
    reserved = 0  # KV tokens promised to running requests
    now = 0.0

    while upcoming or waiting or running:
        while upcoming and upcoming[0].request.arrival_s <= now:
            waiting.append(upcoming.popleft())
        if not waiting and not running:
            now = upcoming[0].request.arrival_s  # GPU idles until the next arrival
            continue

        need = waiting[0].request.input_tokens + waiting[0].request.output_tokens if waiting else 0
        if waiting and reserved + need <= capacity:
            s = waiting.popleft()
            reserved += need
            now += prefill_seconds(gpu, model, s.request.input_tokens)
            s.first_token_s = now
            s.generated = 1
            running.append(s)
        else:
            cached = sum(s.request.input_tokens + s.generated for s in running)
            now += decode_step_seconds(gpu, model, len(running), cached)
            for s in running:
                s.generated += 1

        for s in [s for s in running if s.generated == s.request.output_tokens]:
            s.finish_s = now
            reserved -= s.request.input_tokens + s.request.output_tokens
            running.remove(s)

    return Run(served=served, end_s=now)


@dataclass(frozen=True)
class Summary:
    ttft_p50_s: float
    ttft_p99_s: float
    tpot_p50_s: float
    tpot_p99_s: float
    output_tokens_per_s: float
    usd_per_million_output_tokens: float


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile: the smallest value with at least p% of values at or below it."""
    ordered = sorted(values)
    return ordered[max(0, math.ceil(p / 100 * len(ordered)) - 1)]


def summarize(run: Run, usd_per_gpu_hour: float) -> Summary:
    """Latency percentiles, throughput, and cost per million output tokens.

    The GPU is billed from time 0 to the last finish, idle time included.
    """
    ttfts = [s.ttft for s in run.served]
    tpots = [s.tpot for s in run.served if not math.isnan(s.tpot)]
    tokens = sum(s.request.output_tokens for s in run.served)
    return Summary(
        ttft_p50_s=percentile(ttfts, 50),
        ttft_p99_s=percentile(ttfts, 99),
        tpot_p50_s=percentile(tpots, 50) if tpots else math.nan,
        tpot_p99_s=percentile(tpots, 99) if tpots else math.nan,
        output_tokens_per_s=tokens / run.end_s,
        usd_per_million_output_tokens=usd_per_gpu_hour * run.end_s / 3600 / tokens * 1e6,
    )
