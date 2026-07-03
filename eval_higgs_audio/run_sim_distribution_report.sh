#!/bin/bash
# Recompute speaker-similarity distribution per clone directory (raw cosine + mapped).
# Runs in the `omnivoice` conda env (same encoder as eval_sim). Portable to another machine
# as long as: this repo layout + eval_sim/model weights + the clone dirs are present.
#
# Usage:
#   bash run_sim_distribution_report.sh                 # uses default dirs below
#   BASE=/other/path bash run_sim_distribution_report.sh
#   GPUS=0,1 WORKERS=8 bash run_sim_distribution_report.sh
#   SAMPLE_SIZE=1000 bash run_sim_distribution_report.sh  # per-dir sampling
#   bash run_sim_distribution_report.sh --dirs /d1 /d2 --gpus 0 --workers 4   # pass-through
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# conda 环境的 activate.d 脚本在 set -u 下会因未绑定变量报错, 临时关掉
eval "$(conda shell.bash hook)"
set +u
conda activate omnivoice
set -u

cd "$SCRIPT_DIR"

GPUS="${GPUS:-0,1,2,3}"
WORKERS="${WORKERS:-16}"
SCAN_WORKERS="${SCAN_WORKERS:-64}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/sim_dist_report}"
SAMPLE_SIZE="${SAMPLE_SIZE:-}"
SEED="${SEED:-42}"

if [ "$#" -gt 0 ]; then
    # 直接透传所有参数给 python
    python sim_distribution_report.py "$@"
    exit $?
fi

BASE="${BASE:-/root/group-shared/voiceprint/data/speech/speaker_diarization/merged_datasets_20250610_vad_segments_mtfaa_enhanced_extend_kid_withclone_addlibrilight_1130}"
python sim_distribution_report.py \
    --dirs \
        "${BASE}/audio_omnivoice_clone" \
        "${BASE}/audio_higgs_audio_v3_tts_clone" \
        "${BASE}/audio_higgs_audio_v3_tts_clone_2" \
        "${BASE}/audio_higgs_audio_v3_tts_clone_3" \
    --gpus "${GPUS}" \
    --workers "${WORKERS}" \
    --scan-workers "${SCAN_WORKERS}" \
    --output-dir "${OUTPUT_DIR}" \
    ${SAMPLE_SIZE:+--sample-size "${SAMPLE_SIZE}"} \
    --seed "${SEED}"
