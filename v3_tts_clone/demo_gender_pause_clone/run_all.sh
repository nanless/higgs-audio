#!/usr/bin/env bash
# Copyright (c) 2025 Boson AI
#
# End-to-end: sample 20 male / 20 female / 5 child (>5min)
#   -> assign hand-authored Higgs-v3 clone texts (clone_text_corpus.json)
#   -> clone (~20–30s, multi-sentence with long pauses)
#
# Prereq: SGLang-Omni on BASE_URL (default http://localhost:8000)
# Refs: ONLY under AUDIO_ROOT (…/audio), with nonempty *.wav.json transcript.
#
# Usage:
#   bash v3_tts_clone/demo_gender_pause_clone/run_all.sh
#   SKIP_SAMPLE=1 LIMIT=3 bash v3_tts_clone/demo_gender_pause_clone/run_all.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../.." && pwd)"

TRAIN_JSON="${TRAIN_JSON:-/root/data/lists/speech/speaker_diarization/merged_datasets_20250610_vad_segments_mtfaa_enhanced_extend_kid_withclone_addlibrilight_1130_add_child207m_korean_20260625/train.json}"
GENDER_JSON="${GENDER_JSON:-/root/group-shared/voiceprint/data/speech/speaker_diarization/merged_datasets_20250610_vad_segments_mtfaa_enhanced_extend_kid_withclone_addlibrilight_1130/genders_ages_wav2vec2/per_speaker.json}"
AUDIO_ROOT="${AUDIO_ROOT:-/root/group-shared/voiceprint/data/speech/speaker_diarization/merged_datasets_20250610_vad_segments_mtfaa_enhanced_extend_kid_withclone_addlibrilight_1130/audio}"
CORPUS_JSON="${CORPUS_JSON:-${HERE}/clone_text_corpus.json}"
WORKDIR="${WORKDIR:-${REPO}/clone_workdir/demo_gender_pause_clone}"
BASE_URL="${BASE_URL:-http://localhost:8000}"
SEED="${SEED:-42}"
MODE="${MODE:-single}"
WORKERS="${WORKERS:-2}"
LIMIT="${LIMIT:-0}"
SKIP_SAMPLE="${SKIP_SAMPLE:-0}"
SKIP_TEXTS="${SKIP_TEXTS:-0}"
SKIP_CLONE="${SKIP_CLONE:-0}"
RESPLICE="${RESPLICE:-0}"
MIX_BG_NOISE="${MIX_BG_NOISE:-0}"
REVERB="${REVERB:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-768}"
SPEECH_RATE="${SPEECH_RATE:-1.05}"

mkdir -p "${WORKDIR}"
echo "=============================================="
echo " demo_gender_pause_clone"
echo " WORKDIR=${WORKDIR}"
echo " CORPUS=${CORPUS_JSON}"
echo " AUDIO_ROOT=${AUDIO_ROOT}"
echo " SEED=${SEED} MODE=${MODE} LIMIT=${LIMIT}"
echo " BASE_URL=${BASE_URL}"
echo "=============================================="

if [ ! -f "${CORPUS_JSON}" ]; then
  echo "[run] missing corpus: ${CORPUS_JSON}" >&2
  exit 1
fi

if [ "${SKIP_SAMPLE}" != "1" ]; then
  python3 "${HERE}/01_sample_speakers.py" \
    --train-json "${TRAIN_JSON}" \
    --gender-json "${GENDER_JSON}" \
    --audio-root "${AUDIO_ROOT}" \
    --output-dir "${WORKDIR}" \
    --seed "${SEED}" \
    --min-duration-sec 300 \
    --n-male 20 --n-female 20 --n-child 5 \
    --ref-min-sec 8 --ref-max-sec 10 --ref-max-concat 5
else
  echo "[run] skip sample"
fi

if [ "${SKIP_TEXTS}" != "1" ]; then
  python3 "${HERE}/02_assign_texts.py" \
    --speakers-json "${WORKDIR}/selected_speakers.json" \
    --corpus-json "${CORPUS_JSON}" \
    --output-dir "${WORKDIR}" \
    --seed "${SEED}"
else
  echo "[run] skip texts"
fi

if [ "${SKIP_CLONE}" != "1" ]; then
  LIMIT_ARGS=()
  if [ "${LIMIT}" != "0" ]; then
    LIMIT_ARGS+=(--limit "${LIMIT}")
  fi
  REVERB_ARGS=()
  if [ "${REVERB}" = "0" ]; then
    REVERB_ARGS+=(--no-reverb)
  fi
  python3 "${HERE}/03_run_clone.py" \
    --speakers-json "${WORKDIR}/selected_speakers.json" \
    --scripts-json "${WORKDIR}/clone_scripts.json" \
    --output-dir "${WORKDIR}/clones" \
    --base-url "${BASE_URL}" \
    --audio-root "${AUDIO_ROOT}" \
    --mode "${MODE}" \
    --seed "${SEED}" \
    --workers "${WORKERS}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --speech-rate "${SPEECH_RATE}" \
    --skip-existing \
    "${REVERB_ARGS[@]}" \
    "${LIMIT_ARGS[@]}"
else
  echo "[run] skip clone"
fi

if [ "${RESPLICE}" = "1" ]; then
  python3 "${HERE}/04_resplice_variable_pause.py" \
    --clones-dir "${WORKDIR}/clones" \
    --seed "${SEED}"
fi

if [ "${MIX_BG_NOISE}" = "1" ]; then
  python3 "${HERE}/05_mix_bg_noise.py" \
    --clones-dir "${WORKDIR}/clones" \
    --speakers-json "${WORKDIR}/selected_speakers.json" \
    --seed "${SEED}"
fi

echo "[run] done. outputs under ${WORKDIR}"
