# gpumix

**Find the cheapest GPU setup for your LLM traffic, including mixed NVIDIA + AMD.**

Every LLM request has two phases. **Prefill** reads the prompt and needs raw compute. **Decode** writes the answer token by token and needs memory bandwidth and capacity. No single GPU vendor wins both, so running prefill on one vendor's GPUs and decode on another's can be cheaper. Today, no open source tool tells you whether it is cheaper for *your* traffic.

gpumix will:

1. **Plan:** take a model, your traffic (prompt lengths, answer lengths, request rate) and a latency target, and rank GPU setups (single-vendor and mixed) by cost per million tokens.
2. **Prove:** run the top setups on real GPUs with vLLM and record the measured numbers, so every claim can be reproduced.

```text
$ gpumix plan --model qwen3-14b --traffic my-traffic.jsonl --p99-ttft 2s
rank  setup                          $/1M tokens  p99 TTFT
1     1x H100 prefill + 2x MI300X decode   ...        ...
2     4x H100 (no split)                   ...        ...
```

*(Planned interface. Nothing above runs yet.)*

## Status

Early. Only the project skeleton exists. See [BUILD_PLAN.md](BUILD_PLAN.md) for what comes next.

## Licence

[Apache-2.0](LICENSE)
