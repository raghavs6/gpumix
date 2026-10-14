"""Model shapes: how big a model is and how much KV cache each token needs."""

from datetime import date
from pathlib import Path

from pydantic import HttpUrl, PositiveFloat, PositiveInt

from gpumix.datafile import DATA_DIR, Entry, load_entries


class Model(Entry):
    """The few numbers about a model that decide its memory and speed."""

    params_b: PositiveFloat
    layers: PositiveInt
    kv_heads: PositiveInt
    head_dim: PositiveInt
    hidden_size: PositiveInt  # width of the activations swapped between GPUs under TP
    source: HttpUrl
    checked: date
    note: str | None = None

    def weight_bytes(self, bytes_per_param: int = 2) -> float:
        """Size of the weights. The default of 2 bytes per parameter is BF16."""
        return self.params_b * 1e9 * bytes_per_param

    def kv_bytes_per_token(self, bytes_per_value: int = 2) -> int:
        """KV cache size for one token: a key and a value, per KV head, per layer.

        The default of 2 bytes is BF16, the dtype these models ship in.
        """
        return 2 * self.layers * self.kv_heads * self.head_dim * bytes_per_value


def load_models(path: Path | None = None) -> dict[str, Model]:
    """Read the model file and return the models by name."""
    return load_entries(path or DATA_DIR / "models.yaml", Model)
