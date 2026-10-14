"""Shared loader for the YAML data files (GPUs, models)."""

from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import BaseModel, ConfigDict

DATA_DIR = files("gpumix") / "data"


class Entry(BaseModel):
    """One named entry in a data file. Unknown fields are errors, not ignored."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str


E = TypeVar("E", bound=Entry)


def load_entries(path: Path | Traversable, schema: type[E]) -> dict[str, E]:
    """Read a YAML list of entries, check each against `schema`, return them by name.

    Raises ValueError if two entries share a name, and pydantic's
    ValidationError if any entry has a missing, extra or invalid field.
    """
    entries: dict[str, E] = {}
    for raw in yaml.safe_load(path.read_text()):
        entry = schema.model_validate(raw)
        if entry.name in entries:
            raise ValueError(f"duplicate name: {entry.name}")
        entries[entry.name] = entry
    return entries
