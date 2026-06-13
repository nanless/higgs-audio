#!/bin/bash
# Run MOS evaluation on Higgs Audio v3 TTS clone audio.
# Requires: conda env `omnivoice` with scoreq, ttsds, utmosv2 packages.
#
# Usage:
#   bash run_eval_mos.sh                          # eval all clones, all metrics
#   bash run_eval_mos.sh --sample-size 500 --metrics UTMOS22Strong,SCOREQ --workers 2
#
# Env vars:
#   HIGGS_CLONE_ROOT  - clone output dir (default: production path)
#   CUDA_VISIBLE_DEVICES  - GPU(s) to use
#   TTS_EVAL_MODEL_DIR - dir containing UTMOS22Strong checkpoint

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(dirname "$SCRIPT_DIR")"

eval "$(conda shell.bash hook)"
conda activate omnivoice

cd "$SCRIPT_DIR"

echo "=== Higgs Audio MOS Evaluation ==="
echo "Date: $(date)"
echo "Args: $@"
echo ""

python eval_mos.py "$@"

echo ""
echo "=== Done ==="
