#!/bin/bash
# =============================================================================
# Higgs Audio v3 — 从中断轮续跑 topup (默认 v5 Round 2 clone)
#
# 1. source 07_topup_pipeline.env
# 2. 默认 START_ROUND=2 START_STEP=clone (可用环境变量覆盖)
# 3. 清理残留 TTS (higgs_v3_env)
# 4. 跑 08_preflight_resume.py
# 5. exec 07_topup_pipeline.sh (续跑模式跳过预算测算 → 05)
#
# 用法:
#   bash v3_tts_clone/08_resume_topup.sh
#   START_ROUND=2 START_STEP=sim bash v3_tts_clone/08_resume_topup.sh
#   DRY_RUN=1 bash v3_tts_clone/08_resume_topup.sh
#   ADOPT_SUGGEST=1 bash ...   # 采用预检建议的 START_STEP
#   tmux new-session -d -s higgs_v5r2 "bash v3_tts_clone/08_resume_topup.sh"
# =============================================================================

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Capture caller overrides BEFORE sourcing env (07.env sets START_ROUND=1 for fresh runs).
_CALLER_START_ROUND="${START_ROUND-}"
_CALLER_START_STEP="${START_STEP-}"

# shellcheck disable=SC1091
source "${HERE}/07_topup_pipeline.env"

REPO="${REPO:-/root/code/github_repos/higgs-audio}"
# 08 defaults: resume from round 2 clone (unless caller exported START_* before invoke)
export START_ROUND="${_CALLER_START_ROUND:-2}"
export START_STEP="${_CALLER_START_STEP:-clone}"
DRY_RUN="${DRY_RUN:-0}"
ADOPT_SUGGEST="${ADOPT_SUGGEST:-0}"
MIN_FREE_GB="${MIN_FREE_GB:-500}"

echo "=============================================="
echo " 08 resume topup"
echo " START_ROUND=${START_ROUND}  START_STEP=${START_STEP}"
echo " CLONE_ROOT=${CLONE_ROOT}"
echo " PIPELINE_WORKDIR=${PIPELINE_WORKDIR}"
echo "=============================================="

echo "[08] 清理残留 TTS (higgs_v3_env) ..."
pkill -9 -f "${REPO}/higgs_v3_env" 2>/dev/null || true
sleep 2
reap_orphans() {
    local marker="$1"
    ps -eo pid=,ppid=,args= 2>/dev/null | while read -r pid ppid args; do
        if [ "${ppid}" = "1" ] && [[ "${args}" == *"${marker}"* ]] && [[ "${args}" == *multiprocessing* ]]; then
            kill -9 "${pid}" 2>/dev/null || true
        fi
    done
}
reap_orphans "omnivoice/bin/python" || true
reap_orphans "qwen3-asr/bin/python" || true
sleep 1
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader 2>/dev/null | head -20 || true
fi

echo "[08] 预检 ..."
SUGGEST_OUT="$(
    python3 "${HERE}/08_preflight_resume.py" \
        --pipeline-workdir "${PIPELINE_WORKDIR}" \
        --clone-root "${CLONE_ROOT}" \
        --stats-csv "${STATS_CSV}" \
        --texts-jsonl "${TEXTS_JSONL}" \
        --source-dirs "${SOURCE_DIRS}" \
        --start-round "${START_ROUND}" \
        --num-servers "${NUM_SERVERS}" \
        --min-free-gb "${MIN_FREE_GB}" \
        --repo-root "${REPO}"
)"
echo "${SUGGEST_OUT}"
SUGGEST="$(echo "${SUGGEST_OUT}" | awk -F= '/^SUGGEST_START_STEP=/{print $2}' | tail -1)"
if [ "${ADOPT_SUGGEST}" = "1" ] && [ -n "${SUGGEST}" ] && [ "${SUGGEST}" != "${START_STEP}" ]; then
    echo "[08] ADOPT_SUGGEST=1 → START_STEP ${START_STEP} → ${SUGGEST}"
    export START_STEP="${SUGGEST}"
elif [ -n "${SUGGEST}" ] && [ "${SUGGEST}" != "${START_STEP}" ]; then
    echo "[08] 预检建议 START_STEP=${SUGGEST} (当前 ${START_STEP}; 设 ADOPT_SUGGEST=1 可采用)"
fi

if [ "${DRY_RUN}" = "1" ]; then
    echo "[08] DRY_RUN=1: 预检完成, 不启动流水线"
    echo "    将执行: START_ROUND=${START_ROUND} START_STEP=${START_STEP} bash ${HERE}/07_topup_pipeline.sh"
    exit 0
fi

echo "[08] 启动 07_topup_pipeline.sh (续跑) ..."
export START_ROUND START_STEP
exec bash "${HERE}/07_topup_pipeline.sh"
