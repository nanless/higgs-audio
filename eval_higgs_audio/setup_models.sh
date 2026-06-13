#!/bin/bash
# Download / symlink model weights for Higgs Audio evaluation.
#
# Usage:
#   bash setup_models.sh
#
# This script:
#   1. SIM model: creates symlink to voxblink2_samresnet100_ft weights
#   2. UTMOS22Strong: downloads checkpoint from HuggingFace (optional)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIM_MODEL_DIR="${SCRIPT_DIR}/eval_sim/model"

echo "=== Higgs Audio Eval Model Setup ==="

# ── SIM: SamResNet100 speaker encoder weights ─────────────────────────
SIM_WEIGHTS="${SIM_MODEL_DIR}/avg_model.pt"

if [ -f "${SIM_WEIGHTS}" ]; then
    echo "SIM weights already present: ${SIM_WEIGHTS}"
else
    # Known source paths on this server
    POSSIBLE_SOURCES=(
        "/root/workspace/speaker_verification/mix_adult_kid/exp/voxblink2_samresnet100_ft/avg_model.pt"
        "/root/code/github_repos/OmniVoice-fork/batch_generate_text_and_clone/eval_sim/model/avg_model.pt"
    )

    FOUND=""
    for src in "${POSSIBLE_SOURCES[@]}"; do
        if [ -f "${src}" ]; then
            FOUND="${src}"
            break
        fi
    done

    if [ -n "${FOUND}" ]; then
        ln -sf "${FOUND}" "${SIM_WEIGHTS}"
        echo "SIM weights: symlink created → ${FOUND}"
    else
        echo "WARNING: SIM weights (avg_model.pt) not found."
        echo "  Please place voxblink2_samresnet100_ft/avg_model.pt at:"
        echo "  ${SIM_WEIGHTS}"
        echo "  Or symlink from an existing copy."
    fi
fi

# ── UTMOS22Strong checkpoint ──────────────────────────────────────────
UTMOS_CACHE="${HOME}/.cache/higgs_eval"
UTMOS_CKPT="${UTMOS_CACHE}/mos/utmos22_strong_step7459_v1.pt"

if [ -f "${UTMOS_CKPT}" ]; then
    echo "UTMOS checkpoint already present: ${UTMOS_CKPT}"
elif [ -f "/root/code/github_repos/OmniVoice-fork/TTS_eval_models/mos/utmos22_strong_step7459_v1.pt" ]; then
    mkdir -p "$(dirname "${UTMOS_CKPT}")"
    ln -sf "/root/code/github_repos/OmniVoice-fork/TTS_eval_models/mos/utmos22_strong_step7459_v1.pt" "${UTMOS_CKPT}"
    echo "UTMOS checkpoint: symlink created from OmniVoice"
else
    echo "UTMOS checkpoint not found locally. Download:"
    echo "  pip install huggingface_hub"
    echo "  huggingface-cli download --local-dir ${UTMOS_CACHE} k2-fsa/TTS_eval_models mos/utmos22_strong_step7459_v1.pt"
fi

echo ""
echo "=== Setup complete ==="
