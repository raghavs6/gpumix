#!/usr/bin/env bash
# Run every phase 4 measurement on this machine's GPU, then stop the server.
#
# Run from the repo root inside vLLM 0.32.0, for example the
# vllm/vllm-openai:v0.32.0 or vllm/vllm-openai-rocm:v0.32.0 image:
#
#   bash scripts/bench.sh H100-SXM
#   BATCH_TOKENS=2048 MAX_SEQS=256 bash scripts/bench.sh RTX-3090   # a GPU gpumix doesn't know
#
# Server settings match what the simulator assumes: BF16, one GPU, prefix
# cache off, and vLLM's default step budget and batch limit for this GPU
# (sim.vllm_defaults), passed explicitly so a vLLM change can't move them.
# Results land in results/<GPU>-<UTC time>/.
set -euo pipefail

GPU=$1
MODEL=Qwen/Qwen3-8B
PORT=${PORT:-8000}
MAX_MODEL_LEN=16384  # longest request in the replayed traces is 14,089 tokens
export PYTHONPATH=src
export PYTHONHASHSEED=0  # vLLM derives prompt tokens from Python's hash(): same prompts every run

OUT=${OUT:-results/$GPU-$(date -u +%Y%m%dT%H%M%SZ)}
mkdir -p "$OUT/traces"

if [[ -z "${BATCH_TOKENS:-}" || -z "${MAX_SEQS:-}" ]]; then
  read -r BATCH_TOKENS MAX_SEQS < <(python3 -c "
import sys
from gpumix.gpus import load_gpus
from gpumix.sim import vllm_defaults
print(*vllm_defaults(load_gpus()[sys.argv[1]]))" "$GPU" 2>/dev/null) || true
fi
[[ -n ${MAX_SEQS:-} ]] || { echo "gpumix doesn't know $GPU; set BATCH_TOKENS and MAX_SEQS" >&2; exit 1; }

# What we ran on. Stop early if a "H100-SXM" is really the slower PCIe card.
{
  echo "gpu: $GPU  batch_tokens: $BATCH_TOKENS  max_seqs: $MAX_SEQS"
  echo "gpumix commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "vllm $(vllm --version)"
  python3 -c "import torch; print('torch', torch.__version__)" 2>/dev/null || echo "torch unknown"
  if command -v nvidia-smi >/dev/null; then
    nvidia-smi --query-gpu=name,memory.total,clocks.max.sm,power.limit --format=csv
  fi
  if command -v rocm-smi >/dev/null; then rocm-smi --showproductname --showmeminfo vram; fi
} | tee "$OUT/env.txt"
if [[ $GPU == H100-SXM ]] && ! grep -q HBM3 "$OUT/env.txt"; then
  echo "This H100 isn't the SXM (HBM3) card the model describes. Stopping." >&2
  exit 1
fi

# The traffic: two calibration sweeps and 10 minutes of each Azure trace.
python3 scripts/make_sweeps.py "$OUT/traces"
for sweep in prefill decode; do
  python3 scripts/to_vllm_trace.py "$OUT/traces/$sweep.jsonl" "$OUT/traces/$sweep.vllm.jsonl"
done
python3 scripts/to_vllm_trace.py data/traces/azure2023_conv.jsonl "$OUT/traces/chat.vllm.jsonl" \
  --start-s 600 --duration-s 600
python3 scripts/to_vllm_trace.py data/traces/azure2023_code.jsonl "$OUT/traces/code.vllm.jsonl" \
  --start-s 600 --duration-s 600

vllm serve "$MODEL" --port "$PORT" --dtype bfloat16 --seed 0 \
  --no-enable-prefix-caching --gpu-memory-utilization 0.9 --max-model-len "$MAX_MODEL_LEN" \
  --max-num-batched-tokens "$BATCH_TOKENS" --max-num-seqs "$MAX_SEQS" \
  >"$OUT/server.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT

# Downloading the model and starting up takes minutes; give up after 30.
for _ in $(seq 900); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && break
  kill -0 $SERVER 2>/dev/null || { echo "vLLM exited; see $OUT/server.log" >&2; exit 1; }
  sleep 2
done
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null || { echo "vLLM never got ready" >&2; exit 1; }

for run in prefill decode chat code; do
  trace=$OUT/traces/$run.vllm.jsonl
  echo "=== $run: $(wc -l <"$trace") requests"
  vllm bench serve --backend vllm --model "$MODEL" --base-url "http://127.0.0.1:$PORT" \
    --dataset-name timed_trace --dataset-path "$trace" --num-prompts "$(wc -l <"$trace")" \
    --save-result --save-detailed --result-dir "$OUT" --result-filename "$run.json"
done

echo "done: $OUT"
