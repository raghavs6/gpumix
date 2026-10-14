"""Speed model: how long prefill and one decode step take on one GPU.

Each phase is limited by whichever runs out first: the GPU's math
(compute) or its memory reads (bandwidth). This is the "roofline" model.

Every result here is MODELLED, NOT MEASURED. With the default efficiency
of 1.0 it is the ideal time, a lower bound on what a real GPU will take.
Phase 4 calibrates the efficiency factors from real runs.
"""

from gpumix.gpus import GPU
from gpumix.links import Link, allreduce_seconds
from gpumix.models import Model


def roofline_seconds(
    gpu: GPU,
    flops: float,
    bytes_read: float,
    compute_efficiency: float = 1.0,
    memory_efficiency: float = 1.0,
) -> float:
    """Time for work that needs `flops` of math and `bytes_read` of memory reads."""
    compute_s = flops / (gpu.bf16_dense_tflops * 1e12 * compute_efficiency)
    memory_s = bytes_read / (gpu.mem_bandwidth_tbps * 1e12 * memory_efficiency)
    return max(compute_s, memory_s)


def step_seconds(
    gpu: GPU,
    model: Model,
    new_tokens: int,
    cached_tokens: int,
    compute_efficiency: float = 1.0,
    memory_efficiency: float = 1.0,
    tp: int = 1,
    tp_link: Link | None = None,
) -> float:
    """Time for one forward pass that processes `new_tokens` and reads `cached_tokens` of KV cache.

    Every new token costs about 2 operations per parameter; the weights are
    read once for the whole step, plus every cached token's KV. Prefill,
    decode, and steps that mix both (chunked prefill) are all this one rule.
    Attention's own math (which grows with prompt length squared) is left
    out; it is small for prompts of a few thousand tokens.

    With tensor parallelism (`tp` GPUs joined by `tp_link`), each GPU does
    1/tp of the math and reads 1/tp of the weights and KV, then the GPUs
    swap results twice per layer. Swaps don't overlap the math.
    """
    work = roofline_seconds(
        gpu,
        flops=2 * model.params_b * 1e9 * new_tokens / tp,
        bytes_read=(model.weight_bytes() + cached_tokens * model.kv_bytes_per_token()) / tp,
        compute_efficiency=compute_efficiency,
        memory_efficiency=memory_efficiency,
    )
    if tp == 1:
        return work
    if tp_link is None:
        raise ValueError("tensor parallelism needs a tp_link")
    swap_bytes = new_tokens * model.hidden_size * 2  # BF16 activations
    return work + 2 * model.layers * allreduce_seconds(tp_link, tp, swap_bytes)


def prefill_seconds(
    gpu: GPU,
    model: Model,
    prompt_tokens: int,
    compute_efficiency: float = 1.0,
    memory_efficiency: float = 1.0,
) -> float:
    """Time to read a whole prompt in one step (nothing cached yet)."""
    return step_seconds(gpu, model, prompt_tokens, 0, compute_efficiency, memory_efficiency)


def decode_step_seconds(
    gpu: GPU,
    model: Model,
    batch_size: int,
    context_tokens: int,
    compute_efficiency: float = 1.0,
    memory_efficiency: float = 1.0,
) -> float:
    """Time for one decode step: every conversation in the batch writes one token.

    `context_tokens` is the total number of tokens already in the cache,
    summed over the batch.
    """
    return step_seconds(
        gpu, model, batch_size, context_tokens, compute_efficiency, memory_efficiency
    )
