"""vllm bench serve results from a GPU that behaves exactly like the model at set efficiencies."""

import json
from pathlib import Path

from gpumix.calibrate import (
    DECODE_ANSWER_TOKENS,
    DECODE_PROMPT_TOKENS,
    decode_sweep,
    flatten,
    prefill_sweep,
)
from gpumix.gpus import GPU
from gpumix.models import Model
from gpumix.sim import Run
from gpumix.traffic import Request


def _result(requests: list[Request], ttfts, latencies, itls, duration) -> dict:
    return {
        "duration": duration,
        "input_lens": [r.input_tokens for r in requests],
        "output_lens": [r.output_tokens for r in requests],
        "ttfts": ttfts,
        "latencies": latencies,
        "itls": itls,
        "errors": [""] * len(requests),
    }


def sweeps(gpu: GPU, model: Model, compute: float, memory: float) -> tuple[dict, dict]:
    prompts = flatten(prefill_sweep())
    ttfts = [
        2 * model.params_b * 1e9 * r.input_tokens / (gpu.bf16_dense_tflops * 1e12 * compute)
        for r in prompts
    ]
    prefill = _result(prompts, ttfts, ttfts, [[] for _ in prompts], 1.0)
    context = DECODE_PROMPT_TOKENS + DECODE_ANSWER_TOKENS / 2
    itls = [
        [
            (model.weight_bytes() + p.size * context * model.kv_bytes_per_token())
            / (gpu.mem_bandwidth_tbps * 1e12 * memory)
        ]
        * (DECODE_ANSWER_TOKENS - 1)
        for p in decode_sweep()
        for _ in p.requests
    ]
    answers = flatten(decode_sweep())
    decode = _result(answers, [0.0] * len(answers), [sum(i) for i in itls], itls, 1.0)
    return prefill, decode


def from_run(run: Run) -> dict:
    served = run.served
    return _result(
        [s.request for s in served],
        [s.ttft for s in served],
        [s.finish_s - s.request.arrival_s for s in served],
        [[s.max_gap_s] if s.request.output_tokens > 1 else [] for s in served],
        run.end_s,
    )


def write(path: Path, result: dict) -> None:
    path.write_text(json.dumps(result))
