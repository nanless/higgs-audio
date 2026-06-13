#!/bin/bash
# Master orchestrator: CER → SIM → MOS evaluation pipeline for Higgs Audio v3 TTS clone.
#
# Usage:
#   bash run_eval_all.sh                              # eval all (CER + SIM + MOS)
#   bash run_eval_all.sh --skip-cer                   # SIM + MOS only
#   bash run_eval_all.sh --skip-sim                   # CER + MOS only
#   bash run_eval_all.sh --skip-mos                   # CER + SIM only
#   bash run_eval_all.sh --sample-size 500             # eval 500 random samples
#
# Env vars:
#   HIGGS_CLONE_ROOT              - clone output dir
#   HIGGS_CER_GPU                 - GPU for CER eval (default: 0)
#   HIGGS_SIM_GPU                 - GPU for SIM eval (default: 0)
#   HIGGS_SIM_WORKERS             - SIM worker count (default: 1)
#   HIGGS_MOS_GPUS                - GPUs for MOS eval (default: 0)
#   HIGGS_MOS_WORKERS             - MOS worker count (default: 1)
#   HIGGS_ASR_BATCH_SIZE          - ASR batch size (default: 16)
#   HIGGS_EVAL_SAMPLE_SIZE        - sample size (default: all)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "$LOG_DIR"

SKIP_CER="${SKIP_CER:-}"
SKIP_SIM="${SKIP_SIM:-}"
SKIP_MOS="${SKIP_MOS:-}"
CER_GPU="${HIGGS_CER_GPU:-0}"
SIM_GPU="${HIGGS_SIM_GPU:-0}"
SIM_WORKERS="${HIGGS_SIM_WORKERS:-1}"
MOS_GPUS="${HIGGS_MOS_GPUS:-0}"
MOS_WORKERS="${HIGGS_MOS_WORKERS:-1}"
ASR_BATCH="${HIGGS_ASR_BATCH_SIZE:-16}"
SAMPLE_SIZE="${HIGGS_EVAL_SAMPLE_SIZE:-}"
CLONE_ROOT="${HIGGS_CLONE_ROOT:-/root/group-shared/voiceprint/data/speech/speaker_diarization/merged_datasets_20250610_vad_segments_mtfaa_enhanced_extend_kid_withclone_addlibrilight_1130/audio_higgs_audio_v3_tts_clone}"

# Parse args
ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-cer) SKIP_CER=1 ;;
        --skip-sim) SKIP_SIM=1 ;;
        --skip-mos) SKIP_MOS=1 ;;
        --sample-size) SAMPLE_SIZE="$2"; shift ;;
        --out-dir) CLONE_ROOT="$2"; shift ;;
        *) ARGS+=("$1") ;;
    esac
    shift
done

echo "============================================================"
echo "  Higgs Audio v3 TTS Clone Evaluation Pipeline"
echo "============================================================"
echo "Clone root:  $CLONE_ROOT"
echo "Sample size: ${SAMPLE_SIZE:-all}"
echo "CER GPU:     $CER_GPU"
echo "SIM GPU:     $SIM_GPU  (workers=$SIM_WORKERS)"
echo "MOS GPUs:    $MOS_GPUS  (workers=$MOS_WORKERS)"
echo "Date:        $(date)"
echo "============================================================"

# ── CER Evaluation ────────────────────────────────────────────────────
if [ -z "$SKIP_CER" ]; then
    echo ""
    echo ">>> Step 1: CER Evaluation <<<"
    echo ""

    CER_CMD="bash ${SCRIPT_DIR}/eval_cer/run_eval_cer.sh \
        --out-dir ${CLONE_ROOT} \
        --batch-size ${ASR_BATCH} \
        --gpu ${CER_GPU}"

    if [ -n "$SAMPLE_SIZE" ]; then
        CER_CMD="$CER_CMD --sample-size ${SAMPLE_SIZE}"
    fi

    echo "Running: $CER_CMD"
    $CER_CMD 2>&1 | tee "${LOG_DIR}/cer_$(date +%Y%m%d_%H%M%S).log"

    echo ""
    echo ">>> CER completed <<<"
else
    echo ""
    echo ">>> Skipping CER <<<"
fi

# ── SIM Evaluation ────────────────────────────────────────────────────
if [ -z "$SKIP_SIM" ]; then
    echo ""
    echo ">>> Step 2: Speaker Similarity Evaluation <<<"
    echo ""

    SIM_CMD="bash ${SCRIPT_DIR}/eval_sim/run_eval_sim.sh \
        --out-dir ${CLONE_ROOT} \
        --gpu ${SIM_GPU} \
        --workers ${SIM_WORKERS}"

    if [ -n "$SAMPLE_SIZE" ]; then
        SIM_CMD="$SIM_CMD --sample-size ${SAMPLE_SIZE}"
    fi

    echo "Running: $SIM_CMD"
    $SIM_CMD 2>&1 | tee "${LOG_DIR}/sim_$(date +%Y%m%d_%H%M%S).log"

    echo ""
    echo ">>> SIM completed <<<"
else
    echo ""
    echo ">>> Skipping SIM <<<"
fi

# ── MOS Evaluation ────────────────────────────────────────────────────
if [ -z "$SKIP_MOS" ]; then
    echo ""
    echo ">>> Step 3: MOS Evaluation <<<"
    echo ""

    MOS_CMD="bash ${SCRIPT_DIR}/eval_mos/run_eval_mos.sh \
        --out-dir ${CLONE_ROOT} \
        --gpus ${MOS_GPUS} \
        --workers ${MOS_WORKERS}"

    if [ -n "$SAMPLE_SIZE" ]; then
        MOS_CMD="$MOS_CMD --sample-size ${SAMPLE_SIZE}"
    fi

    echo "Running: $MOS_CMD"
    $MOS_CMD 2>&1 | tee "${LOG_DIR}/mos_$(date +%Y%m%d_%H%M%S).log"

    echo ""
    echo ">>> MOS completed <<<"
else
    echo ""
    echo ">>> Skipping MOS <<<"
fi

echo ""
echo "============================================================"
echo "  Evaluation complete!"
echo "  Summary files at: ${CLONE_ROOT}/"
echo "    CER: eval_higgs_cer_summary.json"
echo "    SIM: eval_higgs_sim_summary.json"
echo "    MOS: eval_higgs_mos_summary.json"
echo "============================================================"
