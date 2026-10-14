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


def load_links(path: Path | None = None) -> dict[str, Link]:
    """Read the link file and return the links by name."""
    return load_entries(path or DATA_DIR / "links.yaml", Link)


def transfer_seconds(model: Model, tokens: int, link: Link, efficiency: float = 1.0) -> float:
    """Time to send `tokens` tokens of KV cache over `link`.

    MODELLED, NOT MEASURED: with efficiency 1.0 this is the ideal time,
    ignoring setup cost and protocol overhead.
    """
    return tokens * model.kv_bytes_per_token() / (link.gbytes_per_s * 1e9 * efficiency)
