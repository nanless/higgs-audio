#!/bin/bash
# Batch child voice cloning script for Higgs-Audio

export PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "PYTHONPATH set to: $PYTHONPATH"

# Default parameters
MODEL_PATH="bosonai/higgs-audio-v2-generation-3B-base"
AUDIO_TOKENIZER_PATH="bosonai/higgs-audio-v2-tokenizer"
OUTPUT_DIR="./child_voice_clone_output_higgs"
NUM_SAMPLES=100
RANDOM_SEED=42
MODEL_SEED=1988

echo "=========================================="
echo "Higgs-Audio Batch Child Voice Cloning"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Tokenizer: $AUDIO_TOKENIZER_PATH"
echo "Output: $OUTPUT_DIR"
echo "Samples: $NUM_SAMPLES"
echo "=========================================="

# Run the batch cloning script
python3 batch_child_voice_clone_higgs.py \
    --model-path "$MODEL_PATH" \
    --audio-tokenizer-path "$AUDIO_TOKENIZER_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --num-samples $NUM_SAMPLES \
    --random-seed $RANDOM_SEED \
    --seed $MODEL_SEED \
    --max-new-tokens 1024 \
    --temperature 0.3 \
    --top-p 0.95 \
    --top-k 50

echo ""
echo "=========================================="
echo "Batch cloning completed!"
echo "Output directory: $OUTPUT_DIR"
echo "=========================================="

