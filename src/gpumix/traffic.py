"""Traffic: the requests a setup has to serve, read from or written to JSONL.

One request per line: {"arrival_s": 0.12, "input_tokens": 850, "output_tokens": 210}
"""

import json
import math
import random
from pathlib import Path

from pydantic import BaseModel, ConfigDict, NonNegativeFloat, PositiveInt, ValidationError


class Request(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    arrival_s: NonNegativeFloat  # seconds since the start of the trace
    input_tokens: PositiveInt  # prompt length
    output_tokens: PositiveInt  # answer length


def read_traffic(path: Path) -> list[Request]:
    """Read a JSONL traffic file. Raises ValueError naming the first bad line."""
    requests: list[Request] = []
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            request = Request.model_validate(json.loads(line))
        except (json.JSONDecodeError, ValidationError) as error:
            raise ValueError(f"line {number}: {error}") from error
        if requests and request.arrival_s < requests[-1].arrival_s:
            raise ValueError(f"line {number}: arrival_s goes backwards")
        requests.append(request)
    return requests


def write_traffic(path: Path, requests: list[Request]) -> None:
    path.write_text("".join(r.model_dump_json() + "\n" for r in requests))


def synth(
    n_requests: int,
    rate_per_s: float,
    median_input_tokens: int,
    median_output_tokens: int,
    seed: int,
    spread: float = 1.0,
) -> list[Request]:
    """Make fake traffic that looks like chat: random arrivals, mostly short requests.

    Arrivals are a Poisson process: gaps between requests are random, averaging
    1 / rate_per_s. Lengths are lognormal around the given medians: most are
    near the median, with a long tail of big ones. `spread` is the lognormal's
    sigma; larger means a longer tail. The same seed always gives the same trace.
    """
    rng = random.Random(seed)
    requests = []
    now = 0.0
    for _ in range(n_requests):
        now += rng.expovariate(rate_per_s)
        requests.append(
            Request(
                arrival_s=now,
                input_tokens=max(
                    1, round(rng.lognormvariate(math.log(median_input_tokens), spread))
                ),
                output_tokens=max(
                    1, round(rng.lognormvariate(math.log(median_output_tokens), spread))
                ),
            )
        )
    return requests


def window(requests: list[Request], start_s: float, duration_s: float) -> list[Request]:
    """Keep the requests arriving in [start_s, start_s + duration_s), with the clock restarted.

    Arrival times are shifted by start_s, not by the first kept arrival, so any
    quiet gap at the start of the window stays in.
    """
    return [
        r.model_copy(update={"arrival_s": r.arrival_s - start_s})
        for r in requests
        if start_s <= r.arrival_s < start_s + duration_s
    ]
