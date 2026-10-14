from pathlib import Path

import pytest
from pydantic import ValidationError

from gpumix.models import load_models

QWEN = """
- name: qwen3-8b
  params_b: 8.2
  layers: 36
  kv_heads: 8
  head_dim: 128
  source: https://huggingface.co/Qwen/Qwen3-8B/blob/main/config.json
  checked: 2026-10-14
"""


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "models.yaml"
    path.write_text(text)
    return path


def test_shipped_file_loads_qwen3_8b():
    assert set(load_models()) == {"qwen3-8b"}


def test_kv_bytes_per_token_matches_hand_calculation():
    # key + value, 36 layers, 8 KV heads, 128 values per head, 2 bytes each (BF16)
    model = load_models()["qwen3-8b"]
    assert model.kv_bytes_per_token() == 2 * 36 * 8 * 128 * 2 == 147_456  # 144 KiB


def test_kv_bytes_per_token_scales_with_value_size():
    model = load_models()["qwen3-8b"]
    assert model.kv_bytes_per_token(bytes_per_value=1) == 73_728  # FP8 cache


@pytest.mark.parametrize("field", ["source", "checked"])
def test_missing_source_or_date_is_rejected(tmp_path, field):
    text = "\n".join(line for line in QWEN.splitlines() if not line.strip().startswith(field))
    with pytest.raises(ValidationError, match=field):
        load_models(write(tmp_path, text))


def test_fractional_layer_count_is_rejected(tmp_path):
    with pytest.raises(ValidationError, match="layers"):
        load_models(write(tmp_path, QWEN.replace("layers: 36", "layers: 36.5")))
