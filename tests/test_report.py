import xml.etree.ElementTree as ET

from gpumix.models import load_models
from gpumix.report import chart_svg, cheapest_per_setup, family, report_markdown
from gpumix.search import SERVER, Target, evaluate
from gpumix.traffic import synth

QWEN = load_models()["qwen3-8b"]
TRAFFIC = synth(50, rate_per_s=0.5, median_input_tokens=200, median_output_tokens=50, seed=1)
PLANS = evaluate(QWEN, TRAFFIC, max_gpus=2)  # 9 setups, each at both providers
SVG = "{http://www.w3.org/2000/svg}"


def test_family_from_vendors():
    by_setup = {p.setup: family(p) for p in PLANS}
    assert by_setup["1x H100-SXM"] == "nvidia"
    assert by_setup["2x MI300X"] == "amd"
    assert by_setup["1x H100-SXM prefill + 1x MI300X decode over infiniband-400g"] == "mixed"


def test_cheapest_per_setup_keeps_one_provider_each():
    best = cheapest_per_setup(PLANS)
    assert len(best) == 9
    mi300x = next(p for p in best if p.setup == "1x MI300X")
    assert mi300x.provider == "digitalocean"  # $2.59/h vs Crusoe's $3.45/h


def test_chart_is_valid_svg_with_one_dot_per_setup():
    root = ET.fromstring(chart_svg(PLANS, SERVER))
    dots = [c for c in root.iter(f"{SVG}circle") if c.find(f"{SVG}title") is not None]
    assert len(dots) == 9
    assert all(d.get("fill") != "none" for d in dots)  # light traffic: everything passes


def test_chart_draws_failing_setups_hollow():
    tight = Target(ttft_s=0.05, tpot_s=0.004)  # H100 copies (about 5 ms) miss, MI300X (3.2) passes
    root = ET.fromstring(chart_svg(PLANS, tight))
    dots = [c for c in root.iter(f"{SVG}circle") if c.find(f"{SVG}title") is not None]
    fills = {d.get("fill") == "none" for d in dots}
    assert fills == {True, False}  # some pass, some don't


def test_report_has_warning_table_and_chart():
    md = report_markdown(PLANS, SERVER, TRAFFIC, "t.jsonl", "qwen3-8b", "t.svg")
    assert "**MODELLED, NOT MEASURED.**" in md
    assert "(18 of 18 pass)" in md
    assert "| 1 | 1x MI300X | digitalocean |" in md
    assert "![Cost against p99 time per token](t.svg)" in md


def test_report_with_nothing_passing_says_so():
    md = report_markdown(PLANS, Target(1e-6, 1e-6), TRAFFIC, "t.jsonl", "qwen3-8b", "t.svg")
    assert "(0 of 18 pass)" in md
    assert "No setup of up to 8 GPUs meets the target." in md
