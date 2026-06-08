#!/bin/bash
# Start 1M text generation in tmux session
# Usage: bash run_1m_gen.sh

SESSION="higgs_1m_gen"

# Check if session already exists
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Session $SESSION already exists. Attach: tmux attach -t $SESSION"
    echo "To kill: tmux kill-session -t $SESSION"
    exit 0
fi

cd /root/code/github_repos/higgs-audio/higgs_audio_v3_text_generator

# Clean old partial outputs
rm -f batch_output/.checkpoint_w*.jsonl batch_output/generated_texts_w*.jsonl

# Create tmux session with 2 panes: generation + monitoring
tmux new-session -d -s "$SESSION" -n "gen"
tmux send-keys -t "$SESSION:gen" \
  "/root/miniforge3/envs/higgs_audio_env/bin/python -u run_parallel_batch.py \
  --total 1000000 --batch-size 16 --workers 8 --num-instances 4 \
  --temperature 0.85 --seed 42 \
  2>&1 | tee /tmp/higgs_1m_gen.log" C-m

# Monitoring pane
tmux split-window -h -t "$SESSION:gen"
tmux send-keys -t "$SESSION:gen.1" \
  "watch -n 30 'echo \"=== Time: \$(date) ===\" && wc -l batch_output/.checkpoint_w*.jsonl 2>/dev/null && echo \"---\" && tail -5 /tmp/higgs_1m_gen.log | grep -E \"texts=|Done|error\" && echo \"---\" && nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader'" C-m

echo "Tmux session '$SESSION' started."
echo "  Attach: tmux attach -t $SESSION"
echo "  Detach: Ctrl+B D"
echo "  Kill:   tmux kill-session -t $SESSION"
echo "  Log:    tail -f /tmp/higgs_1m_gen.log"
echo ""
echo "Estimated time for 1M texts: ~46 hours (~6 texts/s sustained)"
echo "Checkpoints saved to batch_output/.checkpoint_w*.jsonl every 5 batches"
echo ""
echo "To resume after crash:"
echo "  tmux kill-session -t $SESSION"
echo "  bash run_1m_gen.sh  # will auto-resume from checkpoints"
