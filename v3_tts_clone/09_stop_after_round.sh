#!/bin/bash
# =============================================================================
# 监督：指定轮次的 CER+剪枝（整轮评估）结束后，停掉迭代流水线，避免进入下一轮。
#
# 判定「本轮结束」的信号（任一即可）:
#   1. ${PIPELINE_WORKDIR}/round_NN/existing_clones_after.json 出现
#      （写在 CER 剪枝之后的「本轮后统计」里，晚于 Prune(CER)）
#   2. 流水线日志出现: >>> 第 N 轮完成 <<<
#
# 停机动作:
#   - 杀掉 05_iterative_pipeline / 08_resume_topup
#   - 按 higgs_v3_env 路径清理 TTS（含 spawn 子进程）
#   - 杀掉 eval_sim / eval_cer 主进程 + 回收 PPID=1 的 multiprocessing 孤儿
#
# 用法:
#   # 默认停在第 5 轮结束后（当前 v5 topup）
#   bash v3_tts_clone/09_stop_after_round.sh
#   tmux new-session -d -s stop_r5 "bash v3_tts_clone/09_stop_after_round.sh"
#
#   STOP_AFTER_ROUND=5 PIPELINE_WORKDIR=.../iterative_pipeline_v5 \
#     bash v3_tts_clone/09_stop_after_round.sh
#
#   DRY_RUN=1  # 只打印将要执行的停机动作，不真杀
# =============================================================================

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "${HERE}/.." && pwd)}"

# Prefer 07 env for paths if present
if [ -f "${HERE}/07_topup_pipeline.env" ]; then
    # shellcheck disable=SC1091
    source "${HERE}/07_topup_pipeline.env"
fi

STOP_AFTER_ROUND="${STOP_AFTER_ROUND:-5}"
PIPELINE_WORKDIR="${PIPELINE_WORKDIR:-${REPO}/clone_workdir/iterative_pipeline_v5}"
POLL_SEC="${POLL_SEC:-5}"
DRY_RUN="${DRY_RUN:-0}"
LOG_DIR="${PIPELINE_WORKDIR}"
SUPERVISOR_LOG="${PIPELINE_WORKDIR}/stop_after_round_${STOP_AFTER_ROUND}.log"
SUPERVISOR_START_EPOCH="$(date +%s)"

rpad=$(printf "%02d" "${STOP_AFTER_ROUND}")
AFTER_JSON="${PIPELINE_WORKDIR}/round_${rpad}/existing_clones_after.json"
DONE_MARK=">>> 第 ${STOP_AFTER_ROUND} 轮完成 <<<"

mkdir -p "${PIPELINE_WORKDIR}"
exec > >(tee -a "${SUPERVISOR_LOG}") 2>&1

echo "=============================================="
echo " 09 stop-after-round supervisor"
echo " STOP_AFTER_ROUND=${STOP_AFTER_ROUND}"
echo " PIPELINE_WORKDIR=${PIPELINE_WORKDIR}"
echo " wait for: ${AFTER_JSON}"
echo "        or: '${DONE_MARK}' in pipeline_*.log"
echo " DRY_RUN=${DRY_RUN}  POLL_SEC=${POLL_SEC}"
echo " log: ${SUPERVISOR_LOG}"
echo " started: $(date)"
echo "=============================================="

round_done() {
    local artifact_mtime=0
    if [ -f "${AFTER_JSON}" ]; then
        artifact_mtime="$(stat -c %Y "${AFTER_JSON}" 2>/dev/null || echo 0)"
    fi
    if [ "${artifact_mtime}" -ge "${SUPERVISOR_START_EPOCH}" ]; then
        echo "[09] 检测到 ${AFTER_JSON}"
        return 0
    fi
    # Newest pipeline log only.  Both the log and completion marker must have
    # been updated after this supervisor started; stale workdir artifacts must
    # never stop a new run.
    local newest
    newest="$(ls -t "${PIPELINE_WORKDIR}"/pipeline_*.log 2>/dev/null | head -1 || true)"
    local newest_mtime=0
    if [ -n "${newest}" ]; then
        newest_mtime="$(stat -c %Y "${newest}" 2>/dev/null || echo 0)"
    fi
    if [ "${newest_mtime}" -ge "${SUPERVISOR_START_EPOCH}" ] \
        && grep -F -q "${DONE_MARK}" "${newest}" 2>/dev/null; then
        echo "[09] 检测到日志标记: ${DONE_MARK}  (${newest})"
        return 0
    fi
    return 1
}

reap_orphans() {
    local marker="$1"
    ps -eo pid=,ppid=,args= 2>/dev/null | while read -r pid ppid args; do
        if [ "${ppid}" = "1" ] && [[ "${args}" == *"${marker}"* ]] && [[ "${args}" == *multiprocessing* ]]; then
            echo "[09] reap orphan pid=${pid}"
            kill -9 "${pid}" 2>/dev/null || true
        fi
    done
}

stop_pipeline() {
    echo "[09] ===== 开始停机 $(date) ====="
    if [ "${DRY_RUN}" = "1" ]; then
        echo "[09] DRY_RUN=1: 将杀掉 05/08/eval/TTS，但不执行"
        pgrep -af '05_iterative_pipeline|08_resume_topup|eval_sim|eval_cer|sgl-omni serve' || true
        return 0
    fi

    # 1) 停主流水线（先于下一轮启 TTS）
    echo "[09] 杀掉 05_iterative_pipeline / 08_resume_topup ..."
    pkill -TERM -f "${REPO}/v3_tts_clone/05_iterative_pipeline.sh" 2>/dev/null || true
    pkill -TERM -f "${REPO}/v3_tts_clone/08_resume_topup.sh" 2>/dev/null || true
    sleep 2
    pkill -9 -f "${REPO}/v3_tts_clone/05_iterative_pipeline.sh" 2>/dev/null || true
    pkill -9 -f "${REPO}/v3_tts_clone/08_resume_topup.sh" 2>/dev/null || true

    # 2) 停评估（若刚好卡在边界）
    echo "[09] 杀掉 eval_sim / eval_cer ..."
    pkill -TERM -f "eval_sim.py" 2>/dev/null || true
    pkill -TERM -f "eval_cer.py" 2>/dev/null || true
    pkill -TERM -f "run_eval_sim.sh" 2>/dev/null || true
    pkill -TERM -f "run_eval_cer.sh" 2>/dev/null || true
    sleep 2
    pkill -9 -f "eval_sim.py" 2>/dev/null || true
    pkill -9 -f "eval_cer.py" 2>/dev/null || true
    reap_orphans "omnivoice/bin/python" || true
    reap_orphans "qwen3-asr/bin/python" || true

    # 3) 停 TTS（含 spawn 引擎子进程）
    echo "[09] 清理 higgs_v3_env TTS ..."
    pkill -9 -f "${REPO}/higgs_v3_env" 2>/dev/null || true
    sleep 2

    echo "[09] 残留检查:"
    pgrep -af '05_iterative|08_resume|eval_sim|eval_cer|sgl-omni serve|higgs_v3_env' || echo "  (无匹配进程)"
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null | head -8 || true
    fi
    echo "[09] ===== 停机完成 $(date) ====="
    echo "[09] 第 ${STOP_AFTER_ROUND} 轮产物: ${PIPELINE_WORKDIR}/round_${rpad}/"
}

echo "[09] 等待第 ${STOP_AFTER_ROUND} 轮 CER+剪枝（整轮）结束 ..."
while true; do
    if round_done; then
        # 再给「本轮后统计」写盘一点余量（若刚出现日志标记）
        sleep 3
        stop_pipeline
        exit 0
    fi
    # 若流水线已死且目标文件始终未出现 → 报错退出，避免空转
    if ! pgrep -f "${REPO}/v3_tts_clone/05_iterative_pipeline.sh" >/dev/null 2>&1; then
        if round_done; then
            stop_pipeline
            exit 0
        fi
        echo "[09] ⚠️  05_iterative_pipeline 已不在，且未看到第 ${STOP_AFTER_ROUND} 轮完成标记"
        echo "[09] 请人工检查 ${PIPELINE_WORKDIR}"
        exit 2
    fi
    sleep "${POLL_SEC}"
done
