#!/bin/bash
# Start 1M text generation with optimized Higgs v3 pipeline in tmux session
# batch_size=8 to fit 4096 token context with full tag guide
# Usage: bash run_1m_gen.sh

SESSION="higgs_1m_gen_v2"
OUTDIR="batch_output_v2"

# Check if session already exists
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Session $SESSION already exists. Attach: tmux attach -t $SESSION"
    echo "To kill: tmux kill-session -t $SESSION"
    exit 0
fi

cd /root/code/github_repos/higgs-audio/higgs_audio_v3_text_generator

# Prepare output directory.  Never delete checkpoints here: relaunching this
# script is the documented resume path.
mkdir -p "$OUTDIR"

# Create tmux session with 2 panes: generation + monitoring
tmux new-session -d -s "$SESSION" -n "gen"
tmux send-keys -t "$SESSION:gen" \
  "python3 -u run_parallel_batch.py \
  --total 1000000 --batch-size 8 --workers 8 --num-instances 4 \
  --temperature 0.85 --seed 42 \
  --output-dir $OUTDIR \
  2>&1 | tee /tmp/higgs_1m_gen.log" C-m

# Monitoring pane
tmux split-window -h -t "$SESSION:gen"
tmux send-keys -t "$SESSION:gen.1" \
  "watch -n 30 'echo \"=== Time: \$(date) ===\" && wc -l ${OUTDIR}/.checkpoint_w*.jsonl 2>/dev/null && echo \"---\" && tail -5 /tmp/higgs_1m_gen.log | grep -E \"texts=|Done|error\" && echo \"---\" && nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader'" C-m

echo "Tmux session '$SESSION' started."
echo "  Attach: tmux attach -t $SESSION"
echo "  Detach: Ctrl+B D"
echo "  Kill:   tmux kill-session -t $SESSION"
echo "  Log:    tail -f /tmp/higgs_1m_gen.log"
echo ""
echo "Estimated time for 1M raw texts: ~40 hours (125K batches × bsize=8)"
echo "Checkpoints saved to ${OUTDIR}/.checkpoint_w*.jsonl every 5 batches"
echo "After all workers reach the exact raw target, final dedup + quality filtering runs automatically."
echo ""
echo "To resume after crash:"
echo "  tmux kill-session -t $SESSION"
echo "  bash run_1m_gen.sh  # will auto-resume from checkpoints"
