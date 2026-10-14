"""GPU specs: load and check the data file every other module reads from."""

from datetime import date
from importlib.resources import files
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, HttpUrl, PositiveFloat

DEFAULT_PATH = files("gpumix") / "data" / "gpus.yaml"


class GPU(BaseModel):
    """One GPU's spec sheet, with where each number came from."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    vendor: Literal["nvidia", "amd"]
    bf16_dense_tflops: PositiveFloat
    mem_bandwidth_tbps: PositiveFloat
    memory_gb: PositiveFloat
    source: HttpUrl
    checked: date
    note: str | None = None


def load_gpus(path: Path | None = None) -> dict[str, GPU]:
    """Read the GPU file and return the GPUs by name.

    Raises ValueError if two entries share a name, and pydantic's
    ValidationError if any entry has a missing, extra or invalid field.
    """
    text = (path or DEFAULT_PATH).read_text()
    gpus: dict[str, GPU] = {}
    for entry in yaml.safe_load(text):
        gpu = GPU.model_validate(entry)
        if gpu.name in gpus:
            raise ValueError(f"duplicate GPU name: {gpu.name}")
        gpus[gpu.name] = gpu
    return gpus
