"""Speed model: how long prefill and one decode step take on one GPU.

Each phase is limited by whichever runs out first: the GPU's math
(compute) or its memory reads (bandwidth). This is the "roofline" model.

Every result here is MODELLED, NOT MEASURED. With the default efficiency
of 1.0 it is the ideal time, a lower bound on what a real GPU will take.
Phase 4 calibrates the efficiency factors from real runs.
"""

from gpumix.gpus import GPU
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


def prefill_seconds(
    gpu: GPU,
    model: Model,
    prompt_tokens: int,
    compute_efficiency: float = 1.0,
    memory_efficiency: float = 1.0,
) -> float:
    """Time to read a prompt: about 2 operations per parameter per token.

    The weights are read once for the whole prompt. Attention's own math
    (which grows with prompt length squared) is left out; it is small for
    prompts of a few thousand tokens.
    """
    return roofline_seconds(
        gpu,
        flops=2 * model.params_b * 1e9 * prompt_tokens,
        bytes_read=model.weight_bytes(),
        compute_efficiency=compute_efficiency,
        memory_efficiency=memory_efficiency,
    )


def decode_step_seconds(
    gpu: GPU,
    model: Model,
    batch_size: int,
    context_tokens: int,
    compute_efficiency: float = 1.0,
    memory_efficiency: float = 1.0,
) -> float:
    """Time for one decode step: every conversation in the batch writes one token.

    Each step reads the weights once (shared by the batch) plus the KV cache
    of every conversation. `context_tokens` is the total number of tokens
    already in the cache, summed over the batch.
    """
    return roofline_seconds(
        gpu,
        flops=2 * model.params_b * 1e9 * batch_size,
        bytes_read=model.weight_bytes() + context_tokens * model.kv_bytes_per_token(),
        compute_efficiency=compute_efficiency,
        memory_efficiency=memory_efficiency,
    )
