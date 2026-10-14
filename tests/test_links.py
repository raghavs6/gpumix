import pytest

from gpumix.links import load_links, transfer_seconds
from gpumix.models import load_models

LINKS = load_links()
QWEN = load_models()["qwen3-8b"]


def test_shipped_file_loads():
    assert set(LINKS) == {
        "nvlink-h100",
        "infinity-fabric-mi300x",
        "infiniband-400g",
        "ethernet-200g",
        "ethernet-100g",
    }


def test_in_machine_links_are_single_vendor_and_network_links_are_cross_vendor():
    assert LINKS["nvlink-h100"].vendors == {"nvidia"}
    assert LINKS["infinity-fabric-mi300x"].vendors == {"amd"}
    cross = {name for name, link in LINKS.items() if link.vendors == {"nvidia", "amd"}}
    assert cross == {"infiniband-400g", "ethernet-200g", "ethernet-100g"}


def test_1000_token_prompt_over_400g_and_100g():
    # 1000 tokens x 147,456 B = 147.5 MB of KV cache
    kv_bytes = 1000 * 147_456
    assert transfer_seconds(QWEN, 1000, LINKS["infiniband-400g"]) == pytest.approx(kv_bytes / 50e9)
    assert transfer_seconds(QWEN, 1000, LINKS["ethernet-100g"]) == pytest.approx(kv_bytes / 12.5e9)


def test_nvlink_is_9x_faster_than_400g_network():
    nvlink = transfer_seconds(QWEN, 1000, LINKS["nvlink-h100"])
    network = transfer_seconds(QWEN, 1000, LINKS["infiniband-400g"])
    assert network / nvlink == pytest.approx(9)


def test_efficiency_slows_transfer():
    ideal = transfer_seconds(QWEN, 1000, LINKS["ethernet-100g"])
    assert transfer_seconds(QWEN, 1000, LINKS["ethernet-100g"], efficiency=0.5) == pytest.approx(
        2 * ideal
    )
