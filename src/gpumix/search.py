"""Search: try every setup of up to 8 GPUs, keep those that meet a latency target, rank by cost.

A setup is either copies of the model that each do both jobs, or a split
into a prefill pool and a decode pool. Each copy can use tensor
parallelism (TP) where its in-machine link has a measured all-reduce
delay. Every setup is priced at every provider that rents all its GPUs.
Every number here is MODELLED, NOT MEASURED.
"""

from dataclasses import dataclass
from typing import Any

from gpumix.gpus import GPU, load_gpus
from gpumix.links import Link, load_links
from gpumix.models import Model
from gpumix.prices import load_prices
from gpumix.sim import Run, Summary, simulate_replicas, simulate_split, summarize
from gpumix.traffic import Request

NETWORK = "infiniband-400g"  # joins pools that can't share an in-machine link


@dataclass(frozen=True)
class Target:
    """Latency limits, both checked at p99."""

    ttft_s: float  # time to first token
    tpot_s: float  # time per output token after the first


# MLPerf's limits for Llama 3.1 8B, the closest model it benchmarks:
# loadgen/mlperf.conf in mlcommons/inference (main, 2026-09-02),
# llama3_1-8b(-interactive).Server.{ttft,tpot}_latency, at p99.
SERVER = Target(ttft_s=2.0, tpot_s=0.100)
INTERACTIVE = Target(ttft_s=0.5, tpot_s=0.030)


@dataclass(frozen=True)
class Pool:
    """`copies` copies of the model, each on `tp` GPUs of one type."""

    gpu: GPU
    copies: int
    tp: int
    tp_link: Link | None

    @property
    def gpus(self) -> int:
        return self.copies * self.tp

    def options(self) -> dict[str, Any]:
        return {"tp": self.tp, "tp_link": self.tp_link}

    def __str__(self) -> str:
        tp = f" TP{self.tp}" if self.tp > 1 else ""
        return f"{self.copies}x {self.gpu.name}{tp}"


@dataclass(frozen=True)
class Setup:
    """Copies that each do both jobs (`decode` is None), or a prefill/decode split over `link`."""

    pool: Pool  # the prefill pool, in a split
    decode: Pool | None = None
    link: Link | None = None

    def __str__(self) -> str:
        if self.decode is None or self.link is None:
            return str(self.pool)
        return f"{self.pool} prefill + {self.decode} decode over {self.link.name}"

    def run(self, model: Model, requests: list[Request]) -> Run:
        p, d = self.pool, self.decode
        if d is None or self.link is None:
            return simulate_replicas(p.gpu, model, requests, p.copies, **p.options())
        return simulate_split(
            p.gpu,
            d.gpu,
            model,
            requests,
            self.link,
            prefill_count=p.copies,
            decode_count=d.copies,
            prefill_options=p.options(),
            decode_options=d.options(),
        )


@dataclass(frozen=True)
class Plan:
    setup: str
    provider: str
    summary: Summary


def server_link(gpu: GPU, links: dict[str, Link]) -> Link | None:
    """The in-machine link that joins GPUs of this type, if we have one."""
    return next((link for link in links.values() if link.gpu == gpu.name), None)


def pools(gpu: GPU, n_gpus: int, links: dict[str, Link]) -> list[Pool]:
    """Every way to use exactly `n_gpus` GPUs of this type: TP sizes the link has a delay for."""
    link = server_link(gpu, links)
    sizes = [1] + sorted(link.allreduce_latency_us) if link else [1]
    return [Pool(gpu, n_gpus // tp, tp, link) for tp in sizes if n_gpus % tp == 0]


def setups(gpus: list[GPU], max_gpus: int, links: dict[str, Link]) -> list[Setup]:
    """Every setup of at most `max_gpus` GPUs."""
    found = [Setup(p) for gpu in gpus for n in range(1, max_gpus + 1) for p in pools(gpu, n, links)]
    for pg in gpus:
        for dg in gpus:
            # same GPU type: one machine, in-machine link; otherwise the network
            link = (server_link(pg, links) if pg.name == dg.name else None) or links[NETWORK]
            for n_p in range(1, max_gpus):
                for n_d in range(1, max_gpus - n_p + 1):
                    for p in pools(pg, n_p, links):
                        found += [Setup(p, d, link) for d in pools(dg, n_d, links)]
    return found


def search(
    model: Model,
    requests: list[Request],
    target: Target = SERVER,
    gpu_names: tuple[str, ...] = ("H100-SXM", "MI300X"),
    max_gpus: int = 8,
) -> list[Plan]:
    """Every setup and provider that meets `target`, cheapest first."""
    all_gpus, links, prices = load_gpus(), load_links(), load_prices()
    gpus = [all_gpus[name] for name in gpu_names]
    plans = []
    for setup in setups(gpus, max_gpus, links):
        try:
            run = setup.run(model, requests)
        except ValueError:
            continue  # some request doesn't fit in this setup's memory
        for provider in prices.values():
            if not all(name in provider.usd_per_gpu_hour for name in run.gpus):
                continue
            s = summarize(run, provider.usd_per_gpu_hour)
            if s.ttft_p99_s <= target.ttft_s and s.tpot_p99_s <= target.tpot_s:
                plans.append(Plan(str(setup), provider.name, s))
    return sorted(plans, key=lambda plan: plan.summary.usd_per_million_output_tokens)
