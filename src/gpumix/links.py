"""KV transfer: how long the KV cache takes to move from a prefill GPU to a decode GPU."""

from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import HttpUrl, PositiveFloat

from gpumix.datafile import DATA_DIR, Entry, load_entries
from gpumix.models import Model


class Link(Entry):
    """One kind of connection between GPUs, with its one-way speed."""

    vendors: set[Literal["nvidia", "amd"]]
    gbytes_per_s: PositiveFloat
    source: HttpUrl
    checked: date
    note: str | None = None
    gpu: str | None = None  # in-machine links: the GPU they join; None for networks
    allreduce_latency_us: dict[int, PositiveFloat] = {}  # GPU count -> measured delay
    allreduce_source: HttpUrl | None = None
    allreduce_note: str | None = None


def load_links(path: Path | None = None) -> dict[str, Link]:
    """Read the link file and return the links by name."""
    return load_entries(path or DATA_DIR / "links.yaml", Link)


def transfer_seconds(model: Model, tokens: int, link: Link, efficiency: float = 1.0) -> float:
    """Time to send `tokens` tokens of KV cache over `link`.

    MODELLED, NOT MEASURED: with efficiency 1.0 this is the ideal time,
    ignoring setup cost and protocol overhead.
    """
    return tokens * model.kv_bytes_per_token() / (link.gbytes_per_s * 1e9 * efficiency)


def allreduce_seconds(link: Link, gpus: int, nbytes: float) -> float:
    """Time for `gpus` GPUs to sum an `nbytes` array over `link` (a ring all-reduce).

    A measured fixed delay, plus each GPU sending and receiving
    2 x (gpus - 1) / gpus of the array. The bandwidth part is ideal.
    """
    if gpus not in link.allreduce_latency_us:
        raise ValueError(f"no measured all-reduce delay for {gpus} GPUs on {link.name}")
    latency = link.allreduce_latency_us[gpus] * 1e-6
    return latency + 2 * (gpus - 1) / gpus * nbytes / (link.gbytes_per_s * 1e9)
