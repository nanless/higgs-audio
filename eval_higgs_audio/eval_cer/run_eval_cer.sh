#!/bin/bash
# Run CER evaluation on Higgs Audio v3 TTS clone audio.
# Requires: conda env `qwen3-asr` with Qwen3-ASR + jiwer + word2number
# Optional LLM ITN additionally requires openai + vLLM servers.
#
# Default GPU layout:
#   GPUs 0-1: ASR (Qwen3-ASR 1.7B, 2 models loaded in parallel)
#   LLM ITN is disabled by default.
#
# Usage:
#   # Optional Step 1: Start vLLM servers (only when using --enable-llm)
#   bash start_vllm_multi.sh
#
#   # Step 2: Run evaluation
#   bash run_eval_cer.sh                          # ASR + manual ITN only
#   bash run_eval_cer.sh --enable-llm             # ASR + manual ITN + LLM ITN
#   bash run_eval_cer.sh --sample-size 500        # eval 500 random samples
#   bash run_eval_cer.sh --skip-asr               # use cached ASR results
#   bash run_eval_cer.sh --skip-existing          # skip already-evaluated items
#
# Env vars:
#   HIGGS_CLONE_ROOT  - clone output dir (default: production path)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

eval "$(conda shell.bash hook)"
conda activate qwen3-asr

cd "$SCRIPT_DIR"

echo "=== Higgs Audio CER Evaluation ==="
echo "Date: $(date)"
echo "Args: $@"
echo ""

python eval_cer.py "$@"

echo ""
echo "=== Done ==="
