#!/bin/bash
# Run CER evaluation on Higgs Audio v3 TTS clone audio.
# Requires: conda env `qwen3-asr` with Qwen3-ASR model at ~/.cache/huggingface/hub/Qwen3-ASR-1.7B-local
#
# Usage:
#   bash run_eval_cer.sh                          # eval all clones
#   bash run_eval_cer.sh --sample-size 500        # eval 500 random samples
#   bash run_eval_cer.sh --skip-asr               # use cached ASR results
#   bash run_eval_cer.sh --skip-existing          # skip already-evaluated items
#
# Env vars:
#   HIGGS_CLONE_ROOT  - clone output dir (default: production path)
#   CUDA_VISIBLE_DEVICES  - GPU(s) to use

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(dirname "$SCRIPT_DIR")"

# Activate conda env
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
