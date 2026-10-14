from gpumix.gpus import load_gpus
from gpumix.links import load_links
from gpumix.models import load_models
from gpumix.search import SERVER, Target, pools, search, setups
from gpumix.traffic import synth

GPUS = load_gpus()
LINKS = load_links()
QWEN = load_models()["qwen3-8b"]
H100, MI300X = GPUS["H100-SXM"], GPUS["MI300X"]
LIGHT = synth(50, rate_per_s=0.5, median_input_tokens=200, median_output_tokens=50, seed=1)


def test_h100_can_use_tp_but_mi300x_cannot_yet():
    # NVLink has measured all-reduce delays for 2, 4 and 8 GPUs; Infinity Fabric has none
    assert {(p.copies, p.tp) for p in pools(H100, 4, LINKS)} == {(4, 1), (2, 2), (1, 4)}
    assert {(p.copies, p.tp) for p in pools(MI300X, 4, LINKS)} == {(4, 1)}


def test_setups_up_to_two_gpus():
    found = [str(s) for s in setups([H100, MI300X], 2, LINKS)]
    assert sorted(found) == sorted(
        [
            "1x H100-SXM",
            "2x H100-SXM",
            "1x H100-SXM TP2",
            "1x MI300X",
            "2x MI300X",
            "1x H100-SXM prefill + 1x H100-SXM decode over nvlink-h100",
            "1x H100-SXM prefill + 1x MI300X decode over infiniband-400g",
            "1x MI300X prefill + 1x H100-SXM decode over infiniband-400g",
            "1x MI300X prefill + 1x MI300X decode over infinity-fabric-mi300x",
        ]
    )


def test_impossible_target_finds_nothing():
    assert search(QWEN, LIGHT, Target(ttft_s=1e-6, tpot_s=1e-6), max_gpus=2) == []


def test_light_traffic_picks_the_cheapest_gpu_hour():
    # every setup meets the server target at 0.5 req/s, so the cheapest wins:
    # one MI300X at DigitalOcean, $2.59/h
    plans = search(QWEN, LIGHT, SERVER, max_gpus=2)
    assert (plans[0].setup, plans[0].provider) == ("1x MI300X", "digitalocean")


def test_every_plan_meets_the_target_and_is_sorted_by_cost():
    target = Target(ttft_s=0.05, tpot_s=0.006)
    plans = search(QWEN, LIGHT, target, max_gpus=2)
    assert plans  # some setups pass, some don't
    assert all(p.summary.ttft_p99_s <= 0.05 and p.summary.tpot_p99_s <= 0.006 for p in plans)
    costs = [p.summary.usd_per_million_output_tokens for p in plans]
    assert costs == sorted(costs)
