"""Report: a Markdown page and a cost-vs-latency chart for one planner run.

The chart is a small hand-written SVG so the report needs no plotting
library. Every number in it is MODELLED, NOT MEASURED.
"""

import math
import statistics
from xml.sax.saxutils import escape

from gpumix.gpus import load_gpus
from gpumix.search import Plan, Target, passing
from gpumix.traffic import Request

COLOURS = {"nvidia": "#2e7d32", "amd": "#c62828", "mixed": "#6a1b9a"}  # green, red, purple
W, H, PAD = 640, 400, 60  # chart size and margin, in pixels


def family(plan: Plan) -> str:
    """'nvidia', 'amd', or 'mixed', from the GPUs a plan rents."""
    gpus = load_gpus()
    vendors = {gpus[name].vendor for name in plan.gpus}
    return vendors.pop() if len(vendors) == 1 else "mixed"


def cheapest_per_setup(plans: list[Plan]) -> list[Plan]:
    """One plan per setup: the provider where it costs least."""
    best: dict[str, Plan] = {}
    for p in plans:
        if (
            p.setup not in best
            or p.summary.usd_per_1k_requests < best[p.setup].summary.usd_per_1k_requests
        ):
            best[p.setup] = p
    return list(best.values())


def chart_svg(plans: list[Plan], target: Target) -> str:
    """Scatter of $ per 1,000 requests against p99 time per token (log scale).

    One dot per setup at its cheapest provider. Filled dots meet both
    latency limits; hollow dots miss at least one. The dashed line is the
    time-per-token limit.
    """
    dots = [p for p in cheapest_per_setup(plans) if not math.isnan(p.summary.tpot_p99_s)]
    tpots_ms = [p.summary.tpot_p99_s * 1000 for p in dots] + [target.tpot_s * 1000]
    lo = 10 ** math.floor(math.log10(min(tpots_ms)))
    hi = 10 ** math.ceil(math.log10(max(tpots_ms)))
    top = max(p.summary.usd_per_1k_requests for p in dots) * 1.1

    def x(ms: float) -> float:
        return PAD + (math.log10(ms) - math.log10(lo)) / (math.log10(hi) - math.log10(lo)) * (
            W - 2 * PAD
        )

    def y(usd: float) -> float:
        return H - PAD - usd / top * (H - 2 * PAD)

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'font-family="sans-serif" font-size="12">',
        f'<rect width="{W}" height="{H}" fill="white"/>',
        f'<line x1="{PAD}" y1="{H - PAD}" x2="{W - PAD}" y2="{H - PAD}" stroke="black"/>',
        f'<line x1="{PAD}" y1="{PAD}" x2="{PAD}" y2="{H - PAD}" stroke="black"/>',
    ]
    tick = lo
    while tick <= hi:
        out.append(
            f'<text x="{x(tick):.1f}" y="{H - PAD + 16}" text-anchor="middle">{tick:g}</text>'
        )
        tick *= 10
    for i in range(5):
        usd = top * i / 4
        out.append(f'<text x="{PAD - 6}" y="{y(usd) + 4:.1f}" text-anchor="end">{usd:.2f}</text>')
    out += [
        f'<text x="{W / 2}" y="{H - 16}" text-anchor="middle">'
        "p99 time per token (ms, log scale)</text>",
        f'<text x="16" y="{H / 2}" text-anchor="middle" transform="rotate(-90 16 {H / 2})">'
        "$ per 1,000 requests</text>",
        f'<line x1="{x(target.tpot_s * 1000):.1f}" y1="{PAD}" x2="{x(target.tpot_s * 1000):.1f}" '
        f'y2="{H - PAD}" stroke="gray" stroke-dasharray="4 4"/>',
        f'<text x="{x(target.tpot_s * 1000) + 4:.1f}" y="{PAD + 12}" fill="gray">limit</text>',
    ]
    for p in dots:
        colour = COLOURS[family(p)]
        fill = colour if p.meets(target) else "none"
        out.append(
            f'<circle cx="{x(p.summary.tpot_p99_s * 1000):.1f}" '
            f'cy="{y(p.summary.usd_per_1k_requests):.1f}" r="4" fill="{fill}" '
            f'stroke="{colour}"><title>{escape(p.setup)}</title></circle>'
        )
    for i, (name, colour) in enumerate(COLOURS.items()):
        out.append(f'<circle cx="{W - PAD - 70}" cy="{PAD + 14 * i}" r="4" fill="{colour}"/>')
        out.append(f'<text x="{W - PAD - 60}" y="{PAD + 14 * i + 4}">{name}</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


def report_markdown(
    plans: list[Plan],
    target: Target,
    requests: list[Request],
    traffic_name: str,
    model_name: str,
    chart_file: str,
    top: int = 10,
) -> str:
    """The report page. `plans` is every evaluated plan, passing or not."""
    ok = passing(plans, target)
    span = requests[-1].arrival_s
    lines = [
        f"# Which GPUs should you rent for `{traffic_name}`?",
        "",
        "> **MODELLED, NOT MEASURED.** These numbers come from gpumix's speed model with ideal",
        "> GPU efficiency. They have not been checked against real GPU runs yet.",
        "",
        "## Traffic",
        "",
        f"- {len(requests):,} requests over {span:,.0f} s ({len(requests) / span:.1f} per second)",
        f"- Median prompt {statistics.median(r.input_tokens for r in requests):,.0f} tokens, "
        f"median answer {statistics.median(r.output_tokens for r in requests):,.0f} tokens",
        f"- Model: `{model_name}`",
        "",
        "## Latency target",
        "",
        f"99% of requests get their first token within {target.ttft_s * 1000:g} ms, and 99% of",
        f"requests average under {target.tpot_s * 1000:g} ms per token after that.",
        "",
        f"## Cheapest setups that meet it ({len(ok)} of {len(plans)} pass)",
        "",
    ]
    if ok:
        lines += [
            "| # | Setup | Provider | $ / 1k requests | $ / M output tokens "
            "| p99 first token | p99 per token |",
            "|--:|--|--|--:|--:|--:|--:|",
        ]
        for i, p in enumerate(ok[:top], start=1):
            s = p.summary
            lines.append(
                f"| {i} | {p.setup} | {p.provider} | {s.usd_per_1k_requests:.2f} | "
                f"{s.usd_per_million_output_tokens:.2f} | {s.ttft_p99_s * 1000:.0f} ms | "
                f"{s.tpot_p99_s * 1000:.1f} ms |"
            )
    else:
        lines.append("No setup of up to 8 GPUs meets the target.")
    lines += [
        "",
        "## Cost against speed",
        "",
        f"![Cost against p99 time per token]({chart_file})",
        "",
        "One dot per setup, at its cheapest provider. Filled dots meet both limits; hollow dots",
        "miss at least one. Green is NVIDIA only, red is AMD only, purple mixes both.",
        "",
        "## What the model assumes",
        "",
        "- **Ideal speed.** Each step takes as long as the GPU's math or memory reads need at full",
        "  spec (vendor dense BF16 TFLOPS and memory bandwidth). Real GPUs are slower.",
        "- **vLLM-style serving.** Continuous batching with chunked prefill, using vLLM's default",
        "  token budget and batch size for each GPU. Requests are routed round robin.",
        "- **Prefill/decode splits** move the KV cache over NVLink or Infinity Fabric inside one",
        "  machine, or 400G InfiniBand between vendors, at full link speed.",
        "- **Tensor parallelism** only on H100, with a measured all-reduce delay from nccl-tests",
        "  (NVIDIA/nccl#1259). No measured MI300X number was found, so MI300X runs without it.",
        "- **Latency limits** are MLPerf's for Llama 3.1 8B (`loadgen/mlperf.conf`).",
        "- **Prices** are on-demand $/GPU-hour from each provider's pricing page; each GPU is",
        "  billed for the whole run, idle time included. Sources and dates: `prices.yaml`.",
        "",
    ]
    return "\n".join(lines)
