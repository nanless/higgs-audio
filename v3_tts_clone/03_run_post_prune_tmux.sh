#!/bin/bash
# Launch SGLang servers + post-prune clone client in tmux (round 2 output dir).
#
# Usage:
#   bash v3_tts_clone/03_run_post_prune_tmux.sh
#   bash v3_tts_clone/03_run_post_prune_tmux.sh "0,1,2,3,4,5,6,7"
#
# Attach: tmux attach -t higgs_step3_r2

set -euo pipefail

REPO="/root/code/github_repos/higgs-audio"
GPUS="${1:-0,1,2,3,4,5,6,7}"
MODEL="${MODEL_PATH:-/root/models/higgs-audio-v3-tts-4b}"
BASE_PORT="${BASE_PORT:-8000}"
SESSION="higgs_step3_r2"
WORK="${REPO}/clone_workdir"
CLONE_V1="/root/group-shared/voiceprint/data/speech/speaker_diarization/merged_datasets_20250610_vad_segments_mtfaa_enhanced_extend_kid_withclone_addlibrilight_1130/audio_higgs_audio_v3_tts_clone"
TOTAL_CLONE_HOURS="${TOTAL_CLONE_HOURS:-10000}"
OUTPUT_ROOT="/root/group-shared/voiceprint/data/speech/speaker_diarization/merged_datasets_20250610_vad_segments_mtfaa_enhanced_extend_kid_withclone_addlibrilight_1130/audio_higgs_audio_v3_tts_clone_2"
STATS_CSV="${WORK}/speaker_duration_stats_post_prune_resume.csv"
TEXTS_JSONL="${REPO}/higgs_audio_v3_text_generator/batch_output_v2/generated_texts_final.jsonl"

mkdir -p "$WORK" "$OUTPUT_ROOT"
NUM_SERVERS=$(echo "$GPUS" | awk -F, '{print NF}')

echo "Killing stale sgl-omni servers (if any)..."
pkill -f "sgl-omni serve" 2>/dev/null || true
sleep 5

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Session $SESSION already exists."
    echo "  Attach: tmux attach -t $SESSION"
    echo "  Kill:   tmux kill-session -t $SESSION"
    exit 1
fi

echo "=== tmux session: $SESSION ==="
echo "Output: $OUTPUT_ROOT"
echo "Stats:  $STATS_CSV"
echo "GPUs:   $GPUS"
echo ""

tmux new-session -d -s "$SESSION" -n servers
tmux send-keys -t "$SESSION:servers" \
    "cd $REPO && bash v3_tts_clone/03_launch_servers.sh \"$GPUS\" \"$MODEL\" $BASE_PORT 2>&1 | tee $WORK/step3_r2_tmux_servers.log" C-m

tmux new-window -t "$SESSION" -n client
tmux send-keys -t "$SESSION:client" \
    "cd $REPO && \
wait_health() { \
  local max=\$1; \
  for i in \$(seq 1 \$max); do \
    local ok=1; \
    for p in \$(seq $BASE_PORT $((BASE_PORT + NUM_SERVERS - 1))); do \
      local code=\$(curl -s -o /dev/null -w '%{http_code}' http://localhost:\$p/health 2>/dev/null || echo 000); \
      if [ \"\$code\" != 200 ]; then ok=0; break; fi; \
    done; \
    if [ \"\$ok\" = 1 ]; then return 0; fi; \
    echo \"[\$(date -Iseconds)] waiting for servers... (\$i/\$max)\"; \
    sleep 10; \
  done; \
  return 1; \
}; \
wait_health 90 || { echo 'ERROR: SGLang servers not ready'; exit 1; }; \
python3 v3_tts_clone/04_post_prune_stats.py \
    --stats-csv ${REPO}/clone_workdir/speaker_duration_stats.csv \
    --clone-root $CLONE_V1 \
    --output-dir $WORK \
    --total-clone-hours $TOTAL_CLONE_HOURS; \
wait_health 30 || { echo 'ERROR: SGLang servers not ready after stats'; exit 1; }; \
python3 v3_tts_clone/03_tts_clone.py \
    --stats-csv $STATS_CSV \
    --post-prune \
    --texts-jsonl $TEXTS_JSONL \
    --output-root $OUTPUT_ROOT \
    --base-port $BASE_PORT \
    --num-servers $NUM_SERVERS \
    --workers-per-server 16 \
    --ref-mode random \
    --ref-rotate-every 50 \
    --ref-pool-size 256 \
    --output-sample-rate 16000 \
    --seed 42 \
    2>&1 | tee $WORK/step3_r2_tmux_client.log" C-m

tmux new-window -t "$SESSION" -n progress
tmux send-keys -t "$SESSION:progress" \
    "while true; do \
        n=\$(find \"$OUTPUT_ROOT\" -name 'clone_*.wav' 2>/dev/null | wc -l); \
        sp=\$(find \"$OUTPUT_ROOT\" -mindepth 2 -maxdepth 2 -type d 2>/dev/null | wc -l); \
        echo \"\$(date -Iseconds) clone_wav=\$n speakers_with_dir=\$sp\"; \
        sleep 60; \
    done | tee $WORK/step3_r2_progress_tmux.log" C-m

echo "Started. Attach: tmux attach -t $SESSION"
echo "Logs:"
echo "  $WORK/step3_r2_tmux_servers.log"
echo "  $WORK/step3_r2_tmux_client.log"
echo "  $WORK/step3_r2_progress_tmux.log"
