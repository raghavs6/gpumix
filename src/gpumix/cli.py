"""Command line: `gpumix gpus`, `gpumix synth`, `gpumix plan` and `gpumix report`."""

from pathlib import Path
from typing import Annotated

import typer

from gpumix.gpus import load_gpus
from gpumix.models import load_models
from gpumix.prices import load_prices
from gpumix.report import chart_svg, report_markdown
from gpumix.search import INTERACTIVE, SERVER, Target, evaluate, passing
from gpumix.traffic import read_traffic, synth, write_traffic

app = typer.Typer(help="Find the cheapest GPU setup for your LLM traffic.", no_args_is_help=True)

WARNING = "MODELLED, NOT MEASURED: ideal GPU speeds, not yet checked against real runs."


def table(rows: list[list[str]]) -> str:
    """Rows as left-aligned columns; the first row is the header."""
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    return "\n".join(
        "  ".join(c.ljust(w) for c, w in zip(row, widths, strict=True)).rstrip() for row in rows
    )


def parse_limits(model: str, target: str, ttft_ms: float | None, tpot_ms: float | None) -> Target:
    """Check the model name and turn the target options into latency limits."""
    models = load_models()
    if model not in models:
        raise typer.BadParameter(
            f"unknown model; choose from {', '.join(models)}", param_hint="--model"
        )
    presets = {"server": SERVER, "interactive": INTERACTIVE}
    if target not in presets:
        raise typer.BadParameter("choose server or interactive", param_hint="--target")
    return Target(
        ttft_s=ttft_ms / 1000 if ttft_ms is not None else presets[target].ttft_s,
        tpot_s=tpot_ms / 1000 if tpot_ms is not None else presets[target].tpot_s,
    )


@app.command()
def gpus() -> None:
    """List the GPUs gpumix knows, with specs and on-demand $/GPU-hour per provider."""
    providers = load_prices()
    rows = [["GPU", "vendor", "BF16 TFLOPS", "TB/s", "GB", *providers]]
    for g in load_gpus().values():
        prices = [
            f"{p.usd_per_gpu_hour[g.name]:.2f}" if g.name in p.usd_per_gpu_hour else "-"
            for p in providers.values()
        ]
        rows.append(
            [
                g.name,
                g.vendor,
                f"{g.bf16_dense_tflops:g}",
                f"{g.mem_bandwidth_tbps:g}",
                f"{g.memory_gb:g}",
                *prices,
            ]
        )
    typer.echo(table(rows))


@app.command("synth")
def synth_command(
    out: Annotated[Path, typer.Option(help="JSONL file to write.")],
    requests: Annotated[int, typer.Option(help="Number of requests.")] = 2000,
    rate: Annotated[float, typer.Option(help="Average requests per second.")] = 10.0,
    input_tokens: Annotated[int, typer.Option(help="Median prompt length.")] = 1000,
    output_tokens: Annotated[int, typer.Option(help="Median answer length.")] = 200,
    seed: Annotated[int, typer.Option(help="Random seed; the same seed gives the same file.")] = 0,
) -> None:
    """Write synthetic traffic: Poisson arrivals, lognormal prompt and answer lengths."""
    traffic = synth(requests, rate, input_tokens, output_tokens, seed)
    write_traffic(out, traffic)
    typer.echo(f"wrote {len(traffic)} requests over {traffic[-1].arrival_s:.0f} s to {out}")


@app.command()
def plan(
    traffic: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, help="Traffic JSONL file.")
    ],
    model: Annotated[str, typer.Option(help="Model name from models.yaml.")] = "qwen3-8b",
    target: Annotated[
        str, typer.Option(help="MLPerf latency limits: server or interactive.")
    ] = "server",
    ttft_ms: Annotated[
        float | None, typer.Option(help="Override the p99 time-to-first-token limit.")
    ] = None,
    tpot_ms: Annotated[
        float | None, typer.Option(help="Override the p99 time-per-token limit.")
    ] = None,
    top: Annotated[int, typer.Option(help="How many setups to show.")] = 10,
) -> None:
    """Rank every setup of up to 8 GPUs that meets the latency target, cheapest first."""
    limits = parse_limits(model, target, ttft_ms, tpot_ms)
    requests = read_traffic(traffic)
    typer.echo(
        f"Simulating {len(requests)} requests on every setup; this can take minutes...", err=True
    )
    plans = passing(evaluate(load_models()[model], requests), limits)

    typer.echo(WARNING)
    typer.echo(
        f"{model}, {len(requests)} requests, p99 limits: first token "
        f"{limits.ttft_s * 1000:g} ms, per token {limits.tpot_s * 1000:g} ms"
    )
    if not plans:
        typer.echo("No setup of up to 8 GPUs meets these limits.")
        return
    rows = [["$/M out", "provider", "setup", "p99 first token", "p99 per token"]]
    for p in plans[:top]:
        s = p.summary
        rows.append(
            [
                f"{s.usd_per_million_output_tokens:.3f}",
                p.provider,
                p.setup,
                f"{s.ttft_p99_s * 1000:.0f} ms",
                f"{s.tpot_p99_s * 1000:.1f} ms",
            ]
        )
    typer.echo(table(rows))
    typer.echo(f"{len(plans)} setups pass; showing the cheapest {min(top, len(plans))}.")


@app.command()
def report(
    traffic: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, help="Traffic JSONL file.")
    ],
    out: Annotated[Path, typer.Option(help="Markdown file to write; the chart goes next to it.")],
    model: Annotated[str, typer.Option(help="Model name from models.yaml.")] = "qwen3-8b",
    target: Annotated[
        str, typer.Option(help="MLPerf latency limits: server or interactive.")
    ] = "server",
    ttft_ms: Annotated[
        float | None, typer.Option(help="Override the p99 time-to-first-token limit.")
    ] = None,
    tpot_ms: Annotated[
        float | None, typer.Option(help="Override the p99 time-per-token limit.")
    ] = None,
) -> None:
    """Write a Markdown report with the ranking and a cost-vs-latency chart."""
    limits = parse_limits(model, target, ttft_ms, tpot_ms)
    requests = read_traffic(traffic)
    typer.echo(
        f"Simulating {len(requests)} requests on every setup; this can take minutes...", err=True
    )
    plans = evaluate(load_models()[model], requests)
    chart = out.with_suffix(".svg")
    chart.write_text(chart_svg(plans, limits))
    out.write_text(report_markdown(plans, limits, requests, traffic.name, model, chart.name))
    typer.echo(f"wrote {out} and {chart}")
