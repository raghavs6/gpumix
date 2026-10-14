# gpumix

**Find the cheapest GPU setup for your LLM traffic, including mixed NVIDIA + AMD.**

Every LLM request has two phases. **Prefill** reads the prompt and needs raw compute. **Decode** writes the answer token by token and needs memory bandwidth and capacity. No single GPU vendor wins both, so running prefill on one vendor's GPUs and decode on another's can be cheaper. Today, no open source tool tells you whether it is cheaper for *your* traffic.

gpumix will:

1. **Plan:** take a model, your traffic (prompt lengths, answer lengths, request rate) and a latency target, and rank GPU setups (single-vendor and mixed) by cost per million tokens.
2. **Prove:** run the top setups on real GPUs with vLLM and record the measured numbers, so every claim can be reproduced.

## Try it

You need Python 3.11+ and [uv](https://docs.astral.sh/uv/). No GPU is needed.

```text
$ uv sync
$ uv run gpumix gpus                                       # GPUs, specs and $/GPU-hour per provider
$ uv run gpumix synth --out traffic.jsonl --rate 20        # synthetic traffic
$ uv run gpumix plan data/traces/azure2023_code.jsonl      # rank setups for a real trace
$ uv run gpumix report data/traces/azure2023_code.jsonl --out report.md   # page + chart
```

`gpumix plan` on Azure's public coding-assistant trace (8,819 requests; about 3 minutes):

```text
MODELLED, NOT MEASURED: ideal GPU speeds, not yet checked against real runs.
qwen3-8b, 8819 requests, p99 limits: first token 2000 ms, per token 100 ms
$/M out  provider      setup                                                             p99 first token  p99 per token
20.109   digitalocean  2x MI300X                                                         179 ms           27.3 ms
26.786   crusoe        2x MI300X                                                         179 ms           27.3 ms
30.164   digitalocean  3x MI300X                                                         112 ms           12.4 ms
30.164   digitalocean  2x MI300X prefill + 1x MI300X decode over infinity-fabric-mi300x  188 ms           3.7 ms
30.280   crusoe        1x H100-SXM TP2                                                   709 ms           79.1 ms
...
```

Full reports with charts for both public Azure traces: [chat](reports/azure-chat.md) and
[coding](reports/azure-code.md).

Traffic files are JSONL, one request per line: `{"arrival_s": 0.0, "input_tokens": 1000, "output_tokens": 200}`.
The default latency limits are MLPerf's for Llama 3.1 8B (`--target interactive` for the strict ones).

## Status

The planner works, but every number it prints is **modelled, not measured**: it assumes each
GPU runs at its full spec sheet speed. It covers Qwen3-8B on H100 and MI300X, copies or
prefill/decode splits of up to 8 GPUs, tensor parallelism on H100, and on-demand prices from
DigitalOcean and Crusoe. Next is renting GPUs to measure real runs and calibrate the model.
See [BUILD_PLAN.md](BUILD_PLAN.md).

## Licence

[Apache-2.0](LICENSE)
