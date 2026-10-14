"""Simulator: replay traffic through one GPU that does both prefill and decode.

It works like a vLLM server with continuous batching. Requests wait in a
first-come-first-served line, and the GPU runs one step after another.

With chunked prefill (the default, as in vLLM V1), each step has a token
budget. Every conversation that is already writing its answer gets one
token first. The leftover budget goes to prompts, oldest first, and a long
prompt is cut into chunks across several steps. A request's first token
comes out at the end of the step that finishes its prompt.

Without chunked prefill (kept to compare against), a step is either one
whole prompt or one decode token for every running conversation, and
prompts go first.

A request reserves KV room for its whole prompt and answer when it starts,
so nothing is ever evicted; real servers allocate as they go and can
preempt. Every number here is MODELLED, NOT MEASURED.
"""

import math
from collections import deque
from dataclasses import dataclass, field

from gpumix.gpus import GPU
from gpumix.memory import memory_fit
from gpumix.models import Model
from gpumix.perf import step_seconds
from gpumix.traffic import Request

GIB = 2**30


def vllm_defaults(gpu: GPU) -> tuple[int, int]:
    """vLLM's server defaults for this GPU: (tokens per step, max conversations per batch).

    Copied from vllm/engine/arg_utils.py (main, 2026-10-14), OPENAI_API_SERVER
    context: >= 160 GiB gets 16384 tokens, >= 70 GiB (not A100) gets 8192,
    smaller GPUs get 2048; max_num_seqs is 1024 on the big tiers, 256 otherwise.
    """
    memory = gpu.memory_gb * 1e9
    if memory >= 160 * GIB:
        return 16384, 1024
    if memory >= 70 * GIB:
        return 8192, 1024
    return 2048, 256


@dataclass
class Served:
    """What happened to one request."""

    request: Request
    first_token_s: float = math.nan  # clock time the first token came out
    finish_s: float = math.nan  # clock time the last token came out
    prefilled: int = 0  # prompt tokens processed so far
    generated: int = 0  # answer tokens written so far
    max_gap_s: float = 0.0  # longest wait between two of its tokens
    _last_token_s: float = field(default=0.0, repr=False)

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

    def _emit(self, now: float) -> None:
        if self.generated > 0:
            self.max_gap_s = max(self.max_gap_s, now - self._last_token_s)
        self._last_token_s = now
        self.generated += 1
        if self.generated == 1:
            self.first_token_s = now
        if self.generated == self.request.output_tokens:
            self.finish_s = now


@dataclass
class Run:
    served: list[Served]
    end_s: float  # clock time the last request finished


def simulate(
    gpu: GPU,
    model: Model,
    requests: list[Request],
    chunked_prefill: bool = True,
    token_budget: int | None = None,
    max_seqs: int | None = None,
) -> Run:
    """Replay `requests` (sorted by arrival) through one GPU.

    `token_budget` (tokens per step, chunked mode only) and `max_seqs`
    (conversations in progress at once) default to vLLM's values for this GPU.
    Raises ValueError if a single request needs more KV cache than the GPU has.
    """
    default_budget, default_seqs = vllm_defaults(gpu)
    budget = token_budget or default_budget
    seqs = max_seqs or default_seqs
    capacity = memory_fit(gpu, model).kv_tokens
    for r in requests:
        if r.input_tokens + r.output_tokens > capacity:
            raise ValueError(
                f"request needs {r.input_tokens + r.output_tokens} KV tokens; GPU holds {capacity}"
            )

    served = [Served(r) for r in requests]
    upcoming = deque(served)
    waiting: deque[Served] = deque()
    running: list[Served] = []  # admitted: prompt in progress or answer in progress
    reserved = 0  # KV tokens promised to running requests
    now = 0.0

    def size(s: Served) -> int:
        return s.request.input_tokens + s.request.output_tokens

    def can_admit() -> bool:
        return bool(waiting) and len(running) < seqs and reserved + size(waiting[0]) <= capacity

    while upcoming or waiting or running:
        while upcoming and upcoming[0].request.arrival_s <= now:
            waiting.append(upcoming.popleft())
        if not waiting and not running:
            now = upcoming[0].request.arrival_s  # GPU idles until the next arrival
            continue

        decoding = [s for s in running if s.prefilled == s.request.input_tokens]
        chunks: list[tuple[Served, int]] = []  # (request, prompt tokens this step)
        if chunked_prefill:
            room = budget - len(decoding)
            for s in running:  # finish prompts already started, oldest first
                if room > 0 and s.prefilled < s.request.input_tokens:
                    n = min(room, s.request.input_tokens - s.prefilled)
                    chunks.append((s, n))
                    room -= n
            while room > 0 and can_admit():
                s = waiting.popleft()
                reserved += size(s)
                running.append(s)
                n = min(room, s.request.input_tokens)
                chunks.append((s, n))
                room -= n
        elif can_admit():  # plain: a whole prompt is its own step, and goes first
            s = waiting.popleft()
            reserved += size(s)
            running.append(s)
            chunks, decoding = [(s, s.request.input_tokens)], []

        new = len(decoding) + sum(n for _, n in chunks)
        cached = sum(s.request.input_tokens + s.generated for s in decoding) + sum(
            s.prefilled for s, _ in chunks
        )
        now += step_seconds(gpu, model, new, cached)

        for s in decoding:
            s._emit(now)
        for s, n in chunks:
            s.prefilled += n
            if s.prefilled == s.request.input_tokens:
                s._emit(now)  # prefill's last step also writes the first token
        for s in [s for s in running if s.generated == s.request.output_tokens]:
            reserved -= size(s)
            running.remove(s)

    return Run(served=served, end_s=now)


@dataclass(frozen=True)
class Summary:
    ttft_p50_s: float
    ttft_p99_s: float
    tpot_p50_s: float
    tpot_p99_s: float
    max_gap_p99_s: float  # worst stall between tokens, p99 over requests
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
    gaps = [s.max_gap_s for s in run.served if s.request.output_tokens > 1]
    tokens = sum(s.request.output_tokens for s in run.served)
    return Summary(
        ttft_p50_s=percentile(ttfts, 50),
        ttft_p99_s=percentile(ttfts, 99),
        tpot_p50_s=percentile(tpots, 50) if tpots else math.nan,
        tpot_p99_s=percentile(tpots, 99) if tpots else math.nan,
        max_gap_p99_s=percentile(gaps, 99) if gaps else math.nan,
        output_tokens_per_s=tokens / run.end_s,
        usd_per_million_output_tokens=usd_per_gpu_hour * run.end_s / 3600 / tokens * 1e6,
    )
