# Build Plan

Each phase ends with something we can show. Don't start the next phase until that exists.

## Phase 1: Planner core (no GPU needed)

- [ ] Data files: GPU specs (H100, H200, B200, L40S, MI300X, MI325X, MI355X), model shapes (8B, 14B, 70B), rental prices with source and date, KV transfer bandwidth (same-vendor vs. cross-vendor).
- [ ] Performance model: time to prefill a prompt and time per decode step, from each GPU's compute and memory bandwidth, with efficiency factors we later calibrate from real runs.
- [ ] Memory check: does the model fit, and how much KV cache is left.
- [ ] Traffic input: JSONL of `arrival_s, input_tokens, output_tokens`, plus a synthetic generator.

**Done when:** unit tests show sensible prefill and decode times for an 8B model on H100 and MI300X.

## Phase 2: Simulator and search

- [ ] Simulator that replays traffic through a setup and reports TTFT and time-per-token (p50/p99), throughput and cost per million tokens.
- [ ] Setups: one GPU type doing both phases, or prefill and decode on separate pools (same or different vendors).
- [ ] Search over GPU type, count and tensor-parallel size; keep only setups that meet the latency target; rank by cost.
- [ ] CLI: `gpumix plan`, `gpumix gpus`, `gpumix synth`.

**Done when:** `gpumix plan` prints a ranked table for a sample traffic file.

## Phase 3: Report and CI

- [ ] Markdown report with a cost-vs-latency chart.
- [ ] GitHub Actions: lint and tests on every push (CPU only).
- [ ] Clear labels: every planner output says "modelled, not measured" until calibrated.

**Done when:** first post: "Which GPUs should you rent for your LLM traffic?" with the planner's ranking on a public trace.

## Phase 4: Real runs (rent GPUs)

- [ ] Benchmark harness: vLLM on NVIDIA (CUDA) and AMD (ROCm), load from the same traffic file, one JSON result per run in `results/`.
- [ ] Single-vendor baselines on one NVIDIA and one AMD GPU (8B model).
- [ ] Cross-vendor prefill/decode via vLLM's NixlConnector, both GPUs at the same provider and region.
- [ ] Calibrate the planner's efficiency factors from the measured runs; report predicted vs. measured.

**Done when:** a public results log with the measured before/after and how close the planner was.

## Phase 5: Upstream and launch

- [ ] Comment on vLLM's P/D roadmap issue (#33702, "validation of hybrid hardware deployments"), then contribute the NVIDIA ↔ AMD fixes and validation.
- [ ] Posts: planner launch → first measured run → vLLM PR. Lead each with one measured number.

## Ground rules

- Never publish a number we didn't measure; label modelled numbers as modelled.
- Every price and spec has a source and a date.
- Keep dependencies small: Python 3.11+, pydantic, PyYAML, Typer.
