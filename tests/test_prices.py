from pathlib import Path

import pytest
from pydantic import ValidationError

from gpumix.gpus import load_gpus
from gpumix.prices import load_prices

DO = """
- name: digitalocean
  usd_per_gpu_hour:
    H100-SXM: 4.41
    MI300X: 2.59
  source: https://www.digitalocean.com/pricing/gpu-droplets
  checked: 2026-10-14
"""


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "prices.yaml"
    path.write_text(text)
    return path


def test_shipped_file_loads():
    prices = load_prices()
    assert set(prices) == {"digitalocean", "crusoe"}
    assert prices["digitalocean"].usd_per_gpu_hour["MI300X"] == 2.59
    assert prices["digitalocean"].usd_per_gpu_hour["H200-SXM"] == 4.47


def test_every_priced_gpu_is_in_the_gpu_file():
    known = set(load_gpus())
    for provider in load_prices().values():
        assert set(provider.usd_per_gpu_hour) <= known, provider.name


def test_every_provider_rents_both_vendors():
    vendors = {name: gpu.vendor for name, gpu in load_gpus().items()}
    for provider in load_prices().values():
        assert {vendors[g] for g in provider.usd_per_gpu_hour} == {"nvidia", "amd"}, provider.name


@pytest.mark.parametrize("field", ["source", "checked"])
def test_missing_source_or_date_is_rejected(tmp_path, field):
    text = "\n".join(line for line in DO.splitlines() if not line.strip().startswith(field))
    with pytest.raises(ValidationError, match=field):
        load_prices(write(tmp_path, text))


def test_zero_price_is_rejected(tmp_path):
    with pytest.raises(ValidationError, match="usd_per_gpu_hour"):
        load_prices(write(tmp_path, DO.replace("2.59", "0")))
