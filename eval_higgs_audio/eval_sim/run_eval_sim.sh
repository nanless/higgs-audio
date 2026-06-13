#!/bin/bash
# Run speaker similarity evaluation on Higgs Audio v3 TTS clone audio.
# Requires: conda env `omnivoice` (with torch, torchaudio, yaml)
#
# Model: voxblink2_samresnet100_ft (weights at OmniVoice eval_sim/model/)
#
# Usage:
#   bash run_eval_sim.sh                           # eval all pairs
#   bash run_eval_sim.sh --sample-size 500          # eval 500 random samples
#   bash run_eval_sim.sh --workers 4 --gpus 0,1     # multi-GPU
#   bash run_eval_sim.sh --skip-existing            # skip already-evaluated
#
# Env vars:
#   HIGGS_CLONE_ROOT  - clone output dir (default: production path)
#   CUDA_VISIBLE_DEVICES  - GPU(s) to use

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(dirname "$SCRIPT_DIR")"

eval "$(conda shell.bash hook)"
conda activate omnivoice

cd "$SCRIPT_DIR"

echo "=== Higgs Audio Speaker Similarity Evaluation ==="
echo "Date: $(date)"
echo "Args: $@"
echo ""

python eval_sim.py "$@"

echo ""
echo "=== Done ==="
