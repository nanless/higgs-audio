#!/bin/bash
# Launch full SIM eval in tmux (multi-GPU, multi-process).
#
# Usage:
#   HIGGS_CLONE_ROOT=/path/to/clone_2 bash run_eval_sim_tmux.sh
#   HIGGS_SIM_WORKERS=32 HIGGS_SIM_GPUS=0,1,2,3,4,5,6,7 bash run_eval_sim_tmux.sh

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
CLONE_ROOT="${HIGGS_CLONE_ROOT:-/root/group-shared/voiceprint/data/speech/speaker_diarization/merged_datasets_20250610_vad_segments_mtfaa_enhanced_extend_kid_withclone_addlibrilight_1130/audio_higgs_audio_v3_tts_clone_2}"
GPUS="${HIGGS_SIM_GPUS:-0,1,2,3,4,5,6,7}"
WORKERS="${HIGGS_SIM_WORKERS:-16}"
SESSION="${HIGGS_SIM_SESSION:-higgs_eval_sim_r2}"
LOG_DIR="${REPO}/eval_higgs_audio/logs"

mkdir -p "$LOG_DIR"

eval "$(conda shell.bash hook)"
conda activate omnivoice
set -u

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Killing existing session $SESSION ..."
    tmux kill-session -t "$SESSION"
fi

echo "=== SIM eval (tmux) ==="
echo "Clone root: $CLONE_ROOT"
echo "GPUs:       $GPUS"
echo "Workers:    $WORKERS"
echo "Session:    $SESSION"
echo ""

tmux new-session -d -s "$SESSION" -n sim
LOG="${LOG_DIR}/sim_clone2_$(date +%Y%m%d_%H%M%S).log"
tmux send-keys -t "${SESSION}:sim" \
    "cd ${SCRIPT_DIR} && eval \"\$(conda shell.bash hook)\" && conda activate omnivoice && \
python eval_sim.py \
    --out-dir ${CLONE_ROOT} \
    --gpus ${GPUS} \
    --workers ${WORKERS} \
    --skip-existing \
    2>&1 | tee ${LOG}" C-m

tmux new-window -t "$SESSION" -n progress
tmux send-keys -t "${SESSION}:progress" \
    "while true; do \
        n=\$(find \"${CLONE_ROOT}\" -name 'clone_*.sim.json' 2>/dev/null | wc -l); \
        j=\$(wc -l ${CLONE_ROOT}/eval_higgs_sim_details*.jsonl 2>/dev/null | tail -1 | awk '{print \$1}'); \
        echo \"\$(date -Iseconds) sim_sidecars=\$n details_lines=\$j\"; \
        sleep 120; \
    done | tee -a ${LOG_DIR}/sim_clone2_progress.log" C-m

echo "Started SIM eval."
echo "Attach: tmux attach -t ${SESSION}"
echo "Log:    ${LOG}"
