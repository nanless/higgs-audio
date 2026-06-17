#!/bin/bash
# Run CER eval with N parallel shard processes (one GPU each).
#
# Usage:
#   HIGGS_CLONE_ROOT=/path/to/clone_2 bash run_eval_cer_parallel.sh
#   HIGGS_CLONE_ROOT=/path/to/clone_2 NUM_SHARDS=8 BATCH_SIZE=32 bash run_eval_cer_parallel.sh
#
# Outputs per shard:
#   eval_higgs_cer_details{shard}.jsonl, eval_higgs_cer_progress{shard}.json, etc.
# Merge summaries after all shards finish.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
CLONE_ROOT="${HIGGS_CLONE_ROOT:-/root/group-shared/voiceprint/data/speech/speaker_diarization/merged_datasets_20250610_vad_segments_mtfaa_enhanced_extend_kid_withclone_addlibrilight_1130/audio_higgs_audio_v3_tts_clone_2}"
NUM_SHARDS="${NUM_SHARDS:-8}"
BATCH_SIZE="${HIGGS_ASR_BATCH_SIZE:-64}"
AUDIO_WORKERS="${HIGGS_CER_AUDIO_WORKERS:-8}"
SIDECAR_WRITERS="${HIGGS_CER_SIDECAR_WRITERS:-8}"
WRITE_QUEUE="${HIGGS_CER_WRITE_QUEUE:-16}"
PREFETCH_BATCHES="${HIGGS_CER_PREFETCH:-8}"
SESSION="${HIGGS_CER_SESSION:-higgs_eval_cer_r2}"
LOG_DIR="${REPO}/eval_higgs_audio/logs"

mkdir -p "$LOG_DIR"

eval "$(conda shell.bash hook)"
conda activate qwen3-asr

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Killing existing session $SESSION ..."
    tmux kill-session -t "$SESSION"
fi

echo "=== Parallel CER eval ==="
echo "Clone root:   $CLONE_ROOT"
echo "Shards:       $NUM_SHARDS (1 GPU each)"
echo "Batch size:   $BATCH_SIZE"
echo "Audio workers:$AUDIO_WORKERS"
echo "Sidecar wr:   $SIDECAR_WRITERS"
echo "Write queue:  $WRITE_QUEUE"
echo "Session:      $SESSION"
echo ""

tmux new-session -d -s "$SESSION" -n shard0

for i in $(seq 0 $((NUM_SHARDS - 1))); do
    win="shard${i}"
    if [ "$i" -gt 0 ]; then
        tmux new-window -t "$SESSION" -n "$win"
    fi
    log="${LOG_DIR}/cer_clone2_shard${i}.log"
    tmux send-keys -t "${SESSION}:${win}" \
        "cd ${SCRIPT_DIR} && eval \"\$(conda shell.bash hook)\" && conda activate qwen3-asr && \
CUDA_VISIBLE_DEVICES=${i} python eval_cer.py \
    --out-dir ${CLONE_ROOT} \
    --num-shards ${NUM_SHARDS} \
    --shard-index ${i} \
    --asr-gpus 0 \
    --batch-size ${BATCH_SIZE} \
    --audio-workers ${AUDIO_WORKERS} \
    --prefetch-batches ${PREFETCH_BATCHES} \
    --sidecar-writers ${SIDECAR_WRITERS} \
    --write-queue-depth ${WRITE_QUEUE} \
    --skip-existing \
    2>&1 | tee ${log}" C-m
done

tmux new-window -t "$SESSION" -n progress
tmux send-keys -t "${SESSION}:progress" \
    "while true; do \
        n=\$(find \"${CLONE_ROOT}\" -name 'clone_*.cer.json' 2>/dev/null | wc -l); \
        j=\$(wc -l ${CLONE_ROOT}/eval_higgs_cer_details*.jsonl 2>/dev/null | tail -1 | awk '{print \$1}'); \
        echo \"\$(date -Iseconds) cer_sidecars=\$n details_lines=\$j\"; \
        sleep 120; \
    done | tee ${LOG_DIR}/cer_clone2_progress.log" C-m

echo "Started ${NUM_SHARDS} shard workers."
echo "Attach: tmux attach -t ${SESSION}"
echo "Logs:   ${LOG_DIR}/cer_clone2_shard*.log"
