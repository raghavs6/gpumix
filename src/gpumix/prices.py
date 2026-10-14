"""Rental prices: what each cloud provider charges per GPU-hour, on demand."""

from datetime import date
from pathlib import Path

from pydantic import HttpUrl, PositiveFloat

from gpumix.datafile import DATA_DIR, Entry, load_entries


class Provider(Entry):
    """One cloud provider's on-demand prices, keyed by GPU name from gpus.yaml."""

    usd_per_gpu_hour: dict[str, PositiveFloat]
    source: HttpUrl
    checked: date
    note: str | None = None


def load_prices(path: Path | None = None) -> dict[str, Provider]:
    """Read the price file and return the providers by name."""
    return load_entries(path or DATA_DIR / "prices.yaml", Provider)
