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
from dataclasses import dataclass, field, replace
from typing import Any

from gpumix.gpus import GPU
from gpumix.links import Link, transfer_seconds
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
    gpus: dict[str, int]  # GPU name -> how many are billed for the whole run


def simulate(
    gpu: GPU,
    model: Model,
    requests: list[Request],
    chunked_prefill: bool = True,
    token_budget: int | None = None,
    max_seqs: int | None = None,
    prefilled: bool = False,
    tp: int = 1,
    tp_link: Link | None = None,
) -> Run:
    """Replay `requests` (sorted by arrival) through one GPU.

    `token_budget` (tokens per step, chunked mode only) and `max_seqs`
    (conversations in progress at once) default to vLLM's values for this GPU.
    With `prefilled`, prompts arrive already in the KV cache (sent by a
    prefill GPU), so this GPU only decodes: the first token takes one step.
    With `tp` > 1 the "GPU" is a group of `tp` GPUs running one copy of the
    model with tensor parallelism over `tp_link`.
    Raises ValueError if a single request needs more KV cache than the GPU has.
    """
    default_budget, default_seqs = vllm_defaults(gpu)
    budget = token_budget or default_budget
    seqs = max_seqs or default_seqs
    if tp > 1 and (tp_link is None or gpu.vendor not in tp_link.vendors):
        raise ValueError(f"tensor parallelism on {gpu.name} needs a link that joins its GPUs")
    capacity = memory_fit(gpu, model, tp=tp).kv_tokens
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

        while prefilled and can_admit():  # KV already here: join the decodes now
            s = waiting.popleft()
            reserved += size(s)
            running.append(s)
            s.prefilled = s.request.input_tokens

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
        now += step_seconds(gpu, model, new, cached, tp=tp, tp_link=tp_link)

        for s in decoding:
            s._emit(now)
        for s, n in chunks:
            s.prefilled += n
            if s.prefilled == s.request.input_tokens:
                s._emit(now)  # prefill's last step also writes the first token
        for s in [s for s in running if s.generated == s.request.output_tokens]:
            reserved -= size(s)
            running.remove(s)

    return Run(served=served, end_s=now, gpus={gpu.name: tp})


def simulate_replicas(
    gpu: GPU, model: Model, requests: list[Request], count: int, **options: Any
) -> Run:
    """Replay `requests` through `count` identical GPUs that each do both jobs.

    Requests are dealt out in turn (round robin), the way vLLM's own proxy
    routes, so each GPU runs independently. `options` go to `simulate`.
    """
    served: list[Served] = [Served(r) for r in requests]
    end_s = 0.0
    for k in range(count):
        run = simulate(gpu, model, requests[k::count], **options)
        served[k::count] = run.served
        end_s = max(end_s, run.end_s)
    return Run(served=served, end_s=end_s, gpus={gpu.name: count * options.get("tp", 1)})


def simulate_split(
    prefill_gpu: GPU,
    decode_gpu: GPU,
    model: Model,
    requests: list[Request],
    link: Link,
    prefill_count: int = 1,
    decode_count: int = 1,
    **options: Any,
) -> Run:
    """Replay `requests` through separate prefill and decode GPUs.

    Works like vLLM's disaggregated proxy: the prompt runs on a prefill GPU,
    its KV cache crosses `link`, then a decode GPU writes the whole answer,
    so the user's first token comes from the decode GPU. Both pools use
    round robin. Prefill GPUs never wait on decode GPUs, so the two pools
    run as two passes. Each transfer gets the whole link to itself, and the
    prefill GPU frees the prompt's KV as soon as the prompt is done.
    """
    for gpu in (prefill_gpu, decode_gpu):
        if gpu.vendor not in link.vendors:
            raise ValueError(f"{link.name} can't connect {gpu.vendor} GPUs")

    prompts = [r.model_copy(update={"output_tokens": 1}) for r in requests]
    prefill = simulate_replicas(prefill_gpu, model, prompts, prefill_count, **options)
    landed = [
        s.finish_s + transfer_seconds(model, s.request.input_tokens, link) for s in prefill.served
    ]
    order = sorted(range(len(requests)), key=lambda i: landed[i])
    handoffs = [requests[i].model_copy(update={"arrival_s": landed[i]}) for i in order]
    decode = simulate_replicas(decode_gpu, model, handoffs, decode_count, prefilled=True, **options)

    served = [Served(r) for r in requests]
    for i, s in zip(order, decode.served, strict=True):
        served[i] = replace(s, request=requests[i])  # time-to-first-token from the real arrival
    gpus = dict(prefill.gpus)
    for name, n in decode.gpus.items():
        gpus[name] = gpus.get(name, 0) + n
    return Run(served=served, end_s=decode.end_s, gpus=gpus)


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


def summarize(run: Run, usd_per_gpu_hour: dict[str, float]) -> Summary:
    """Latency percentiles, throughput, and cost per million output tokens.

    Every GPU is billed from time 0 to the last finish, idle time included.
    `usd_per_gpu_hour` maps GPU name to price, like a provider's price list.
    """
    usd_per_hour = sum(usd_per_gpu_hour[name] * n for name, n in run.gpus.items())
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
        usd_per_million_output_tokens=usd_per_hour * run.end_s / 3600 / tokens * 1e6,
    )
