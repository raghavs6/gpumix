"""Memory check: does a model fit on a GPU, and how much KV cache is left?"""

from dataclasses import dataclass

from gpumix.gpus import GPU
from gpumix.models import Model

GB = 1e9  # Vendors' "GB". The real chips hold a little more (closer to GiB).
DEFAULT_UTILIZATION = 0.9  # vLLM's default share of GPU memory it will use.


@dataclass(frozen=True)
class MemoryFit:
    weights_gb: float
    kv_cache_gb: float  # room left for KV cache; 0 when the model doesn't fit
    kv_tokens: int  # how many tokens of KV cache fit in that room

    @property
    def fits(self) -> bool:
        return self.kv_tokens > 0


def memory_fit(
    gpu: GPU,
    model: Model,
    utilization: float = DEFAULT_UTILIZATION,
    bytes_per_param: int = 2,
) -> MemoryFit:
    """Split one GPU's usable memory into model weights and KV cache.

    `utilization` is the share of memory the server may use; the rest is
    left for scratch space and the software itself. `bytes_per_param`
    defaults to 2 (BF16).
    """
    usable = gpu.memory_gb * GB * utilization
    weights = model.weight_bytes(bytes_per_param)
    kv_room = max(usable - weights, 0.0)
    return MemoryFit(
        weights_gb=weights / GB,
        kv_cache_gb=kv_room / GB,
        kv_tokens=int(kv_room // model.kv_bytes_per_token()),
    )
