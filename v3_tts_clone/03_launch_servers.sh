#!/bin/bash
# Launch SGLang-Omni servers for Higgs Audio v3 TTS.
# One server per GPU, each on a different port.
#
# Usage:
#     bash 03_launch_servers.sh [GPU_IDS] [MODEL_PATH] [BASE_PORT]
#
# Default: GPUs 0,1  model /root/models/higgs-audio-v3-tts-4b  ports 8000+

set -euo pipefail

GPUS="${1:-0,1}"
MODEL="${2:-/root/models/higgs-audio-v3-tts-4b}"
BASE_PORT="${3:-8000}"
CONDA_PYTHON="/root/code/github_repos/higgs-audio/higgs_v3_env/bin/python3"
SGL_OMNI="/root/code/github_repos/higgs-audio/higgs_v3_env/bin/sgl-omni"

IFS=',' read -ra GPU_ARR <<< "$GPUS"
TOTAL=${#GPU_ARR[@]}

echo "=== Starting SGLang Higgs v3 TTS Servers ==="
echo "GPUs: ${GPU_ARR[*]}"
echo "Model: $MODEL"
echo "Ports: $((BASE_PORT + GPU_ARR[0])) - $((BASE_PORT + GPU_ARR[TOTAL - 1]))"
echo ""

# Verify model exists
if [ ! -f "$MODEL/model.safetensors" ]; then
    echo "ERROR: model.safetensors not found at $MODEL"
    exit 1
fi

# Verify dependencies (quick check)
if ! $CONDA_PYTHON -c "import sglang_omni" 2>/dev/null; then
    echo "ERROR: sglang_omni is not importable in $CONDA_PYTHON"
    echo "Install it with: $CONDA_PYTHON -m pip install -e /root/code/github_repos/sglang-omni --no-deps"
    exit 1
fi

if [ ! -x "$SGL_OMNI" ]; then
    echo "ERROR: sgl-omni executable not found at $SGL_OMNI"
    exit 1
fi

cleanup() {
    echo ""
    echo "Stopping all servers..."
    kill $(jobs -p) 2>/dev/null || true
    wait
    echo "Done."
    exit 1
}
trap cleanup SIGINT SIGTERM

PIDS=()
for GPU in "${GPU_ARR[@]}"; do
    PORT=$((BASE_PORT + GPU))
    echo "Starting server on GPU $GPU, port $PORT..."
    CUDA_VISIBLE_DEVICES=$GPU \
        "$SGL_OMNI" serve \
            --model-path "$MODEL" \
            --port "$PORT" \
            --host 0.0.0.0 &
    PIDS+=($!)
    echo "  PID: ${PIDS[-1]}"
done

echo ""
echo "All $TOTAL servers launched."
echo "Waiting for servers to be ready..."
sleep 30

# Quick health check
ALL_OK=1
for GPU in "${GPU_ARR[@]}"; do
    PORT=$((BASE_PORT + GPU))
    STATUS=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:$PORT/health" 2>/dev/null || echo "000")
    if [ "$STATUS" = "200" ]; then
        echo "  GPU $GPU :$PORT → OK"
    else
        echo "  GPU $GPU :$PORT → NOT READY (status=$STATUS)"
        ALL_OK=0
    fi
done

echo ""
if [ $ALL_OK -eq 1 ]; then
    echo "=== All servers ready ==="
else
    echo "=== WARNING: some servers not ready yet, waiting 30s more ==="
    sleep 30
fi

echo "Servers running. Press Ctrl+C to stop all."
wait
