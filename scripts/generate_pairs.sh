#!/usr/bin/env bash
# Generate slopified pairs and real-AI eval answers with vLLM on a single CUDA GPU.
# FP8 Qwen3-30B-A3B needs ~31 GB of VRAM; use a >=40 GB GPU (or swap in smaller generators).
# Resumable: re-running skips chunks already written.
#
#   pip install vllm
#   bash scripts/generate_pairs.sh [OUT_DIR] [JOB_DIR]
#
# Then, locally:  python data/filter.py --inp "OUT_DIR/pairs_*.jsonl" --outdir data/pairs/pilot
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:-data/gen/pilot_out}
JOBS=${2:-data/gen/pilot}
mkdir -p "$OUT"

# model id | tag | max new tokens | gpu memory fraction
GENERATORS=(
  "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8|qwen3-30b-a3b|1024|0.95"
  "RedHatAI/Mistral-Small-3.2-24B-Instruct-2506-FP8|mistral-small-3.2|1024|0.92"
  "openai/gpt-oss-20b|gpt-oss-20b|2048|0.90"   # held-out generator: test split only
)
for g in "${GENERATORS[@]}"; do
  IFS='|' read -r model tag maxtok mem <<<"$g"
  python data/slopify.py --backend vllm --generator "$model" --tag "$tag" \
    --inp "$JOBS/to_$tag.jsonl" --out "$OUT/pairs_$tag.jsonl" \
    --batch 512 --max-tokens "$maxtok" --gpu-mem "$mem" --max-model-len 4096
done

# Real, unedited AI answers for reference-free eval (prompts sharded across models)
ANSWERERS=(
  "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8|qwen3-30b-a3b|0.95"
  "RedHatAI/Mistral-Small-3.2-24B-Instruct-2506-FP8|mistral-small-3.2|0.92"
  "openai/gpt-oss-20b|gpt-oss-20b|0.90"
  "Qwen/Qwen3-4B-Instruct-2507|qwen3-4b|0.90"
)
i=0
for a in "${ANSWERERS[@]}"; do
  IFS='|' read -r model tag mem <<<"$a"
  if [ ! -s "$OUT/real_ai_$tag.jsonl" ]; then
    python data/real_ai.py answer --inp data/gen/real_ai/prompts.jsonl --out "$OUT/real_ai_$tag.jsonl" \
      --generator "$model" --tag "$tag" --shard "$i" --nshards "${#ANSWERERS[@]}" --gpu-mem "$mem"
  fi
  i=$((i + 1))
done
wc -l "$OUT"/*.jsonl
