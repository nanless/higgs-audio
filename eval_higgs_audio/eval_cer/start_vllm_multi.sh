#!/bin/bash
# Optional: start 6 vLLM instances on GPUs 2-7 for LLM ITN.
# CER evaluation skips LLM ITN by default; start these only when using --enable-llm.
# Usage: bash start_vllm_multi.sh
#
# Default: GPUs 2-7, model Qwen3.6-27B-FP8, ports 8002-8007
# Override: ASR_GPUS="0,1" LLM_GPUS="2,3,4,5,6,7" bash start_vllm_multi.sh

set -euo pipefail

MODEL_PATH="${VLLM_MODEL_PATH:-/root/.cache/huggingface/hub/Qwen/Qwen3___6-27B-FP8}"
CONDA_ENV="${VLLM_CONDA_ENV:-qwen3-asr}"
LLM_GPUS="${LLM_GPUS:-2,3,4,5,6,7}"
BASE_PORT="${VLLM_BASE_PORT:-8000}"

IFS=',' read -ra GPU_ARR <<< "$LLM_GPUS"
TOTAL=${#GPU_ARR[@]}

echo "=== Starting vLLM LLM ITN Servers ==="
echo "Model: $MODEL_PATH"
echo "GPUs: ${GPU_ARR[*]} ($TOTAL instances)"
echo "Ports: $((BASE_PORT + GPU_ARR[0])) - $((BASE_PORT + GPU_ARR[TOTAL - 1]))"
echo ""

if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model not found at $MODEL_PATH"
    exit 1
fi

# Register qwen3_5 config
conda run -n "$CONDA_ENV" python3 -c "
try:
    from vllm.transformers_utils.configs.qwen3_5 import Qwen3_5Config
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    try:
        CONFIG_MAPPING.register('qwen3_5', Qwen3_5Config)
    except ValueError:
        pass
    print('qwen3_5 config registered')
except ImportError:
    print('qwen3_5 config not needed (vllm version handles it)')
"

for GPU_ID in "${GPU_ARR[@]}"; do
    PORT=$((BASE_PORT + GPU_ID))
    SESSION="vllm_itn_${GPU_ID}"
    echo "Starting vLLM on GPU $GPU_ID, port $PORT (tmux: $SESSION)..."
    tmux kill-session -t "$SESSION" 2>/dev/null || true
    tmux new-session -d -s "$SESSION" \
        "CUDA_VISIBLE_DEVICES=$GPU_ID conda run -n $CONDA_ENV vllm serve $MODEL_PATH \
        --language-model-only \
        --enable-prefix-caching \
        --port $PORT \
        --gpu-memory-utilization 0.95 \
        --max-model-len 8192 \
        --max-num-seqs 16 \
        --served-model-name qwen3.6-27b \
        --default-chat-template-kwargs '{\"enable_thinking\": false}' \
        2>&1 | tee /tmp/vllm_itn_gpu${GPU_ID}.log"
done

echo ""
echo "All $TOTAL vLLM instances launched in tmux sessions."
echo "Waiting 300s for startup..."
sleep 300

ALL_OK=1
for GPU_ID in "${GPU_ARR[@]}"; do
    PORT=$((BASE_PORT + GPU_ID))
    STATUS=$(curl -s "http://localhost:$PORT/v1/models" 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null || echo "FAILED")
    if [ "$STATUS" != "FAILED" ]; then
        echo "  GPU $GPU_ID :$PORT → OK ($STATUS)"
    else
        echo "  GPU $GPU_ID :$PORT → NOT READY"
        ALL_OK=0
    fi
done

echo ""
if [ $ALL_OK -eq 1 ]; then
    echo "=== All $TOTAL vLLM servers ready ==="
else
    echo "=== WARNING: some servers not ready ==="
fi

echo ""
echo "To stop all: for g in ${GPU_ARR[*]}; do tmux kill-session -t vllm_itn_\$g; done"
