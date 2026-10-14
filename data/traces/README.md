# Traffic traces

Real request traces in gpumix's traffic format (one JSON request per line:
`arrival_s`, `input_tokens`, `output_tokens`). Read them with `gpumix.traffic.read_traffic`.

## azure2023_code.jsonl, azure2023_conv.jsonl

**Source:** Azure LLM inference trace 2023, from
[Azure/AzurePublicDataset](https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2023.md)
(commit `215becd`), downloaded 2026-10-14.

**Licence:** [CC-BY 4.0](https://github.com/Azure/AzurePublicDataset/blob/master/LICENSE).

**Cite:** Pratyush Patel, Esha Choukse, Chaojie Zhang, Aashaka Shah, Íñigo Goiri,
Saeed Maleki, Ricardo Bianchini. "Splitwise: Efficient generative LLM inference
using phase splitting", ISCA 2024.

| File | Service | Requests | Span | Median prompt / output tokens |
|--|--|--|--|--|
| `azure2023_code.jsonl` | coding assistant | 8,819 | 3,436 s | 1,469 / 13 |
| `azure2023_conv.jsonl` | chat | 19,366 | 3,502 s | 1,020 / 129 |

**What we changed:** columns renamed (`ContextTokens` → `input_tokens`,
`GeneratedTokens` → `output_tokens`); `TIMESTAMP` replaced by `arrival_s`, the
seconds since the file's first request, with the 7th decimal digit dropped
(0.1 µs). No rows were removed or reordered.

The Azure page says the trace was collected on November 11, 2023; the
timestamps in the files say 2023-11-16.

**Regenerate:**

```sh
curl -LO https://raw.githubusercontent.com/Azure/AzurePublicDataset/215becdacba1ce682c7368642ce97d5a332de7a6/data/AzureLLMInferenceTrace_code.csv
curl -LO https://raw.githubusercontent.com/Azure/AzurePublicDataset/215becdacba1ce682c7368642ce97d5a332de7a6/data/AzureLLMInferenceTrace_conv.csv
uv run python scripts/convert_trace.py AzureLLMInferenceTrace_code.csv data/traces/azure2023_code.jsonl
uv run python scripts/convert_trace.py AzureLLMInferenceTrace_conv.csv data/traces/azure2023_conv.jsonl
```

Source SHA-256: code `54e9a6d2…77fb6`, conv `2f1e5b66…c7d0bf8`.
