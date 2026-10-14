from typer.testing import CliRunner

from gpumix.cli import app, table
from gpumix.traffic import read_traffic

runner = CliRunner()


def test_table_aligns_columns():
    assert table([["a", "bb"], ["ccc", "d"]]) == "a    bb\nccc  d"


def test_gpus_lists_specs_and_prices():
    result = runner.invoke(app, ["gpus"])
    assert result.exit_code == 0
    h100 = next(line for line in result.output.splitlines() if line.startswith("H100-SXM"))
    assert h100.split() == ["H100-SXM", "nvidia", "989.5", "3.35", "80", "4.41", "3.90"]


def test_synth_writes_a_readable_traffic_file(tmp_path):
    out = tmp_path / "t.jsonl"
    result = runner.invoke(app, ["synth", "--out", str(out), "--requests", "30", "--seed", "4"])
    assert result.exit_code == 0
    assert len(read_traffic(out)) == 30


def test_plan_ranks_cheapest_first_and_says_modelled(tmp_path):
    out = tmp_path / "t.jsonl"
    runner.invoke(
        app,
        [
            "synth",
            "--out",
            str(out),
            "--requests",
            "50",
            "--rate",
            "0.5",
            "--input-tokens",
            "200",
            "--output-tokens",
            "50",
            "--seed",
            "1",
        ],
    )
    result = runner.invoke(app, ["plan", str(out), "--top", "3"])
    assert result.exit_code == 0
    lines = result.stdout.splitlines()
    assert lines[0].startswith("MODELLED, NOT MEASURED")
    assert lines[3].split()[1:3] == ["digitalocean", "1x"]
    assert "MI300X" in lines[3]


def test_plan_impossible_limits_says_so(tmp_path):
    out = tmp_path / "t.jsonl"
    runner.invoke(app, ["synth", "--out", str(out), "--requests", "20", "--seed", "1"])
    result = runner.invoke(app, ["plan", str(out), "--ttft-ms", "0.001"])
    assert result.exit_code == 0
    assert "No setup of up to 8 GPUs meets these limits." in result.stdout


def test_plan_unknown_model_is_a_clear_error(tmp_path):
    out = tmp_path / "t.jsonl"
    runner.invoke(app, ["synth", "--out", str(out), "--requests", "5"])
    result = runner.invoke(app, ["plan", str(out), "--model", "gpt-9"])
    assert result.exit_code == 2
    assert "unknown model" in result.output


def test_report_writes_markdown_and_chart(tmp_path):
    traffic = tmp_path / "t.jsonl"
    runner.invoke(
        app,
        [
            "synth",
            "--out",
            str(traffic),
            "--requests",
            "30",
            "--rate",
            "0.5",
            "--input-tokens",
            "200",
            "--output-tokens",
            "50",
            "--seed",
            "1",
        ],
    )
    result = runner.invoke(app, ["report", str(traffic), "--out", str(tmp_path / "r.md")])
    assert result.exit_code == 0
    assert (tmp_path / "r.md").read_text().startswith("# Which GPUs should you rent for `t.jsonl`?")
    assert (tmp_path / "r.svg").read_text().startswith("<svg")
