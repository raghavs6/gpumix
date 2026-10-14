from pathlib import Path

import pytest
from pydantic import ValidationError

from gpumix.gpus import load_gpus

H100 = """
- name: H100-SXM
  vendor: nvidia
  bf16_dense_tflops: 989.5
  mem_bandwidth_tbps: 3.35
  memory_gb: 80
  source: https://www.nvidia.com/en-us/data-center/h100/
  checked: 2026-10-14
"""


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "gpus.yaml"
    path.write_text(text)
    return path


def test_shipped_file_loads_all_gpus():
    gpus = load_gpus()
    assert set(gpus) == {"H100-SXM", "H200-SXM", "B200", "L40S", "MI300X", "MI325X", "MI355X"}
    assert gpus["H100-SXM"].vendor == "nvidia"
    assert gpus["MI300X"].memory_gb == 192


def test_h200_is_h100_compute_with_faster_memory():
    # Same Hopper chip, so a halving slip on either sparse number would show here.
    gpus = load_gpus()
    h100, h200 = gpus["H100-SXM"], gpus["H200-SXM"]
    assert h200.bf16_dense_tflops == h100.bf16_dense_tflops
    assert h200.mem_bandwidth_tbps > h100.mem_bandwidth_tbps


def test_valid_entry_loads(tmp_path):
    gpus = load_gpus(write(tmp_path, H100))
    assert gpus["H100-SXM"].mem_bandwidth_tbps == 3.35


@pytest.mark.parametrize("field", ["source", "checked"])
def test_missing_source_or_date_is_rejected(tmp_path, field):
    text = "\n".join(line for line in H100.splitlines() if not line.strip().startswith(field))
    with pytest.raises(ValidationError, match=field):
        load_gpus(write(tmp_path, text))


def test_unknown_field_is_rejected(tmp_path):
    with pytest.raises(ValidationError, match="fp8_tflops"):
        load_gpus(write(tmp_path, H100 + "  fp8_tflops: 1979\n"))


def test_non_positive_number_is_rejected(tmp_path):
    with pytest.raises(ValidationError, match="memory_gb"):
        load_gpus(write(tmp_path, H100.replace("memory_gb: 80", "memory_gb: 0")))


def test_duplicate_name_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="duplicate name: H100-SXM"):
        load_gpus(write(tmp_path, H100 + H100))
