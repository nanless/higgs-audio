#!/bin/bash
# Launch Qwen3-ASR workers on all GPUs in parallel.
# Each worker loads its own model instance and processes 1/N of speakers.
#
# Usage (positional args):
#     bash 02_asr_launch.sh [GPUS] [STATS_CSV]
#     e.g. bash 02_asr_launch.sh "0,1,2,3,4,5,6,7" ./clone_workdir/speaker_duration_stats.csv

set -euo pipefail

GPUS="${1:-0,1}"
STATS_CSV="${2:-./clone_workdir/speaker_duration_stats.csv}"
LOCAL_MODEL="/root/.cache/huggingface/hub/Qwen3-ASR-1.7B-local"
FILES_PER_BATCH=48
CONDA_PYTHON="/root/miniforge3/envs/qwen3-asr/bin/python3"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

IFS=',' read -ra GPU_ARR <<< "$GPUS"
TOTAL_GPUS=${#GPU_ARR[@]}

echo "=== Higgs Audio v3 TTS Clone — Step 2 ASR ==="
echo "GPUs: ${GPU_ARR[*]}  (total: $TOTAL_GPUS)"
echo "Stats CSV: $STATS_CSV"
echo "Model: $LOCAL_MODEL"
echo ""

# Verify conda env exists
if ! conda env list 2>/dev/null | grep -q 'qwen3-asr'; then
    echo "ERROR: conda env 'qwen3-asr' not found. Please create it first."
    exit 1
fi

# Verify model exists
if [ ! -f "$LOCAL_MODEL/model-00001-of-00002.safetensors" ]; then
    echo "ERROR: Local model not found at $LOCAL_MODEL"
    exit 1
fi

# Kill all children on Ctrl+C
cleanup() {
    echo ""
    echo "Stopping all workers..."
    kill $(jobs -p) 2>/dev/null || true
    wait
    echo "All workers stopped."
    exit 1
}
trap cleanup SIGINT SIGTERM

PIDS=()
for GPU_ID in "${GPU_ARR[@]}"; do
    echo "Launching worker on GPU $GPU_ID..."
    CUDA_VISIBLE_DEVICES=$GPU_ID \
        $CONDA_PYTHON "$SCRIPT_DIR/02_asr_worker.py" \
            --stats-csv "$STATS_CSV" \
            --gpu-id "$GPU_ID" \
            --total-gpus "$TOTAL_GPUS" \
            --local-model "$LOCAL_MODEL" \
            --files-per-batch "$FILES_PER_BATCH" &
    PIDS+=($!)
done

echo ""
echo "All $TOTAL_GPUS workers launched. Waiting for completion..."
echo "Press Ctrl+C to stop all workers."
echo ""

FAILED=0
for PID in "${PIDS[@]}"; do
    wait "$PID" || {
        echo "Worker PID $PID exited with error"
        FAILED=1
    }
done

echo ""
if [ $FAILED -eq 0 ]; then
    echo "=== All workers completed successfully ==="
else
    echo "=== Some workers failed (see output above) ==="
    exit 1
fi
