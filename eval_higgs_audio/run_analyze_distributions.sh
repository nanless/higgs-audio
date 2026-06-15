#!/bin/bash
# Analyze CER/SIM distributions for Higgs Audio v3 TTS clone evaluation.
#
# Usage:
#   bash run_analyze_distributions.sh
#   bash run_analyze_distributions.sh --sample-size 10000
#   HIGGS_CLONE_ROOT=/path/to/clone bash run_analyze_distributions.sh
#
# Outputs (default under eval_higgs_audio/):
#   logs/analyze_distributions_YYYYMMDD_HHMMSS.log
#   eval_distribution_report.json
#   eval_distribution_report.txt

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "$LOG_DIR"

CLONE_ROOT="${HIGGS_CLONE_ROOT:-/root/group-shared/voiceprint/data/speech/speaker_diarization/merged_datasets_20250610_vad_segments_mtfaa_enhanced_extend_kid_withclone_addlibrilight_1130/audio_higgs_audio_v3_tts_clone}"
OUTPUT_JSON="${SCRIPT_DIR}/eval_distribution_report.json"
OUTPUT_TXT="${SCRIPT_DIR}/eval_distribution_report.txt"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --out-dir) CLONE_ROOT="$2"; shift ;;
        --output-json) OUTPUT_JSON="$2"; shift ;;
        --output-txt) OUTPUT_TXT="$2"; shift ;;
        *) EXTRA_ARGS+=("$1") ;;
    esac
    shift
done

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/analyze_distributions_${STAMP}.log"

echo "============================================================"
echo "  Higgs Audio CER/SIM Distribution Analysis"
echo "============================================================"
echo "Clone root:   $CLONE_ROOT"
echo "JSON report:  $OUTPUT_JSON"
echo "Text report:  $OUTPUT_TXT"
echo "Log:          $LOG_FILE"
echo "Date:         $(date)"
echo "============================================================"

cd "$SCRIPT_DIR"

python3 analyze_distributions.py \
    --out-dir "$CLONE_ROOT" \
    --output-json "$OUTPUT_JSON" \
    --output-txt "$OUTPUT_TXT" \
    "${EXTRA_ARGS[@]}" 2>&1 | tee "$LOG_FILE"

echo ""
echo "Done. Reports:"
echo "  $OUTPUT_JSON"
echo "  $OUTPUT_TXT"
echo "  $LOG_FILE"
