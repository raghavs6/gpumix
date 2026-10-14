"""GPU specs: load and check the data file every other module reads from."""

from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import HttpUrl, PositiveFloat

from gpumix.datafile import DATA_DIR, Entry, load_entries


class GPU(Entry):
    """One GPU's spec sheet, with where each number came from."""

    vendor: Literal["nvidia", "amd"]
    bf16_dense_tflops: PositiveFloat
    mem_bandwidth_tbps: PositiveFloat
    memory_gb: PositiveFloat
    source: HttpUrl
    checked: date
    note: str | None = None


def load_gpus(path: Path | None = None) -> dict[str, GPU]:
    """Read the GPU file and return the GPUs by name."""
    return load_entries(path or DATA_DIR / "gpus.yaml", GPU)
