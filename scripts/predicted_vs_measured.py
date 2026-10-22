"""Fit a GPU's efficiencies from a bench.sh run, then compare the planner with the measurement.

    python scripts/predicted_vs_measured.py results/H100-SXM-20261101T120000Z H100-SXM

Reads prefill.json and decode.json (the sweeps) to fit the two efficiencies,
then for each replayed trace (chat, code) simulates it twice, at spec-sheet
speed and at the fitted speed, and sets both beside the measured run. The
table is written to calibration.md in the same folder.
"""

import argparse
from pathlib import Path

from gpumix.calibrate import Fit, decode_sweep, fit, flatten, prefill_sweep
from gpumix.gpus import GPU, load_gpus
from gpumix.measured import read_vllm_result, read_vllm_trace, summarize_vllm
from gpumix.models import Model, load_models
from gpumix.prices import load_prices
from gpumix.sim import Summary, simulate, summarize

ROWS = [  # label, Summary field, scale to display units
    ("TTFT p50 (ms)", "ttft_p50_s", 1e3),
    ("TTFT p99 (ms)", "ttft_p99_s", 1e3),
    ("time per token p50 (ms)", "tpot_p50_s", 1e3),
    ("time per token p99 (ms)", "tpot_p99_s", 1e3),
    ("worst stall p99 (ms)", "max_gap_p99_s", 1e3),
    ("output tokens/s", "output_tokens_per_s", 1),
    ("$ per million output tokens", "usd_per_million_output_tokens", 1),
]


def fit_section(f: Fit) -> list[str]:
    return [
        f"Compute efficiency **{f.compute_efficiency:.3f}** (median of the prefill points), "
        f"memory efficiency **{f.memory_efficiency:.3f}** (median of the decode points).",
        "",
        "| prompt tokens | compute efficiency |",
        "|--:|--:|",
        *(f"| {n} | {e:.3f} |" for n, e in f.prefill_points.items()),
        "",
        "| batch size | memory efficiency |",
        "|--:|--:|",
        *(f"| {b} | {e:.3f} |" for b, e in f.decode_points.items()),
    ]


def compare_section(name: str, ideal: Summary, calibrated: Summary, measured: Summary) -> list[str]:
    lines = [
        f"## {name}",
        "",
        "| | modelled, spec sheet | modelled, calibrated | measured | calibrated off by |",
        "|--|--:|--:|--:|--:|",
    ]
    for label, field, scale in ROWS:
        i, c, m = (getattr(s, field) * scale for s in (ideal, calibrated, measured))
        off = round((c - m) / m, 2) + 0.0  # + 0.0 turns -0.0 into 0.0
        lines.append(f"| {label} | {i:.2f} | {c:.2f} | {m:.2f} | {off:+.0%} |")
    return lines + [""]


def report(out: Path, gpu: GPU, model: Model, usd_per_gpu_hour: dict[str, float]) -> str:
    prefill = read_vllm_result(out / "prefill.json", flatten(prefill_sweep()))
    decode = read_vllm_result(out / "decode.json", flatten(decode_sweep()))
    f = fit(gpu, model, prefill["ttfts"], decode["itls"])
    lines = [f"# {gpu.name}: predicted vs measured", "", *fit_section(f), ""]
    for name in ("chat", "code"):
        trace = out / "traces" / f"{name}.vllm.jsonl"
        if not trace.exists():
            continue
        requests = read_vllm_trace(trace)
        measured = summarize_vllm(
            read_vllm_result(out / f"{name}.json", requests), {gpu.name: 1}, usd_per_gpu_hour
        )
        ideal = summarize(simulate(gpu, model, requests), usd_per_gpu_hour)
        calibrated = summarize(
            simulate(
                gpu,
                model,
                requests,
                compute_efficiency=f.compute_efficiency,
                memory_efficiency=f.memory_efficiency,
            ),
            usd_per_gpu_hour,
        )
        lines += compare_section(name, ideal, calibrated, measured)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("gpu")
    parser.add_argument("--model", default="qwen3-8b")
    parser.add_argument("--provider", default="digitalocean", help="whose prices to bill at")
    args = parser.parse_args()
    gpu = load_gpus()[args.gpu]
    prices = load_prices()[args.provider].usd_per_gpu_hour
    text = report(args.results_dir, gpu, load_models()[args.model], prices)
    (args.results_dir / "calibration.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
