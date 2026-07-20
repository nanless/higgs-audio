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
VENV_BIN="/root/code/github_repos/higgs-audio/higgs_v3_env/bin"
CONDA_PYTHON="${VENV_BIN}/python3"
SGL_OMNI="${VENV_BIN}/sgl-omni"
# FlashInfer invokes build helpers such as `ninja` through PATH during its
# first-run JIT compilation. Calling sgl-omni by absolute path alone does not
# expose the rest of the virtualenv executables to child processes.
export PATH="${VENV_BIN}:${PATH}"

IFS=',' read -ra GPU_ARR <<< "$GPUS"
TOTAL=${#GPU_ARR[@]}

echo "=== Starting SGLang Higgs v3 TTS Servers ==="
echo "GPUs: ${GPU_ARR[*]}"
echo "Model: $MODEL"
# Ports are BASE_PORT + list index (0..N-1), NOT physical GPU id — matches 03_tts_clone.py.
echo "Ports: ${BASE_PORT} - $((BASE_PORT + TOTAL - 1))  (index-based; GPUs=${GPU_ARR[*]})"
echo ""

echo "Cleaning stale sgl-omni processes..."
pkill -f "sgl-omni serve" 2>/dev/null || true
sleep 2
# 同时清理 SGLang 的 multiprocessing spawn 引擎子进程 (命令行不含 "sgl-omni serve",
# 单靠上面那句杀不掉, 会残留占满显存导致本轮 OOM)。higgs_v3_env 为 TTS 独占, 可整批杀。
pkill -9 -f "${CONDA_PYTHON%/bin/python3}" 2>/dev/null || true
sleep 3

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

if ! command -v ninja >/dev/null 2>&1; then
    echo "ERROR: ninja executable not found in V3 environment PATH"
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
for i in "${!GPU_ARR[@]}"; do
    GPU="${GPU_ARR[$i]}"
    PORT=$((BASE_PORT + i))
    echo "Starting server on GPU $GPU, port $PORT (index $i)..."
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

# Quick health check (ports = BASE_PORT + index)
ALL_OK=1
for i in "${!GPU_ARR[@]}"; do
    GPU="${GPU_ARR[$i]}"
    PORT=$((BASE_PORT + i))
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
