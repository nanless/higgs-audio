#!/bin/bash
# =============================================================================
# Higgs Audio v3 — 补齐(top-up)迭代克隆流水线 启动器
#
# 与 05 的唯一区别: 源目录与统计口径 (见 07_topup_pipeline.env):
#   - 统计"总时长"= 原始 audio + 两个已过滤 clone 目录 (higgs_123 / omnivoice);
#   - 参考音频 (ref) 只用原始 audio;
#   - 只对 总时长 < 半小时 的说话人继续复刻补齐到半小时;
#   - SIM/CER 剪枝阈值与 05 完全一致。
#
# 本启动器做三件事:
#   1. source 07_topup_pipeline.env
#   2. (非续跑且未手动设 TOTAL_CLONE_HOURS 时) 先跑 Step 0 统计 (source + clone-dirs),
#      按 gap 自动测算克隆预算 TOTAL_CLONE_HOURS = ceil(gap_hours / SURVIVAL_EST),
#      避免预算过大导致过度生成 (占满磁盘)。
#   3. 复用 05_iterative_pipeline.sh 的核心逻辑 (它会看到 STATS_CSV 已存在而跳过重复统计)。
#
# 用法:
#   bash v3_tts_clone/07_topup_pipeline.sh
#   tmux new-session -d -s topup "bash v3_tts_clone/07_topup_pipeline.sh"
#
# 可选覆盖:
#   SURVIVAL_EST=0.1       # raw0.8 SIM + CER 联合存活率估计 (预算 = gap/存活率)
#                          # 默认 0.1 (保守, 多生成确保补齐到目标; 可按实测存活率上调以省算力/磁盘)。
#   TOTAL_CLONE_HOURS=...  # 手动指定预算则跳过自动测算
#   START_ROUND / START_STEP  # 续跑 (透传给 05)
# =============================================================================

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Preserve caller START_* before sourcing env (07.env exports START_ROUND=1 for fresh runs).
_CALLER_START_ROUND="${START_ROUND-}"
_CALLER_START_STEP="${START_STEP-}"

PIPELINE_ENV="${PIPELINE_ENV:-${HERE}/07_topup_pipeline.env}"
if [ ! -f "${PIPELINE_ENV}" ]; then
    echo "❌ 配置文件不存在: ${PIPELINE_ENV}" >&2
    exit 1
fi
# shellcheck disable=SC1090
source "${PIPELINE_ENV}"
echo "[07] 配置文件: ${PIPELINE_ENV}"

SURVIVAL_EST="${SURVIVAL_EST:-0.1}"
# Prefer caller override; else env; else defaults
if [ -n "${_CALLER_START_ROUND}" ]; then
    START_ROUND="${_CALLER_START_ROUND}"
else
    START_ROUND="${START_ROUND:-1}"
fi
if [ -n "${_CALLER_START_STEP}" ]; then
    START_STEP="${_CALLER_START_STEP}"
else
    START_STEP="${START_STEP:-clone}"
fi
export START_ROUND START_STEP
FORCE_STATS="${FORCE_STATS:-0}"
SCAN_WORKERS="${SCAN_WORKERS:-64}"
STATS_OUTPUT_DIR="$(dirname "${STATS_CSV}")"

# 续跑模式 (START_ROUND>1 或 START_STEP!=clone): 05 会跳过预算分配并复用已有基准,
# TOTAL_CLONE_HOURS 不会被用到 → 不做自动测算, 直接交给 05。
RESUMING=0
if [ "${START_ROUND}" -gt 1 ] || [ "${START_STEP}" != "clone" ]; then
    RESUMING=1
fi

# A resume reuses the original per-speaker allocation, so its global budget must
# be restored as well.  Otherwise 05 falls back to 10000 and writes a false final
# summary even though generation followed the saved allocation.
RESTORED_BUDGET=0
if [ "${RESUMING}" -eq 1 ] && [ -z "${TOTAL_CLONE_HOURS:-}" ]; then
    ALLOCATION_SUMMARY="${PIPELINE_WORKDIR}/allocation/post_prune_stats_summary.json"
    if [ ! -f "${ALLOCATION_SUMMARY}" ]; then
        echo "❌ 续跑缺少预算记录: ${ALLOCATION_SUMMARY}" >&2
        exit 1
    fi
    TOTAL_CLONE_HOURS="$(
        python3 -c 'import json,sys; v=json.load(open(sys.argv[1]))["total_clone_hours_budget"]; assert float(v)>0; print(v)' \
            "${ALLOCATION_SUMMARY}"
    )" || {
        echo "❌ 无法从 allocation summary 恢复 TOTAL_CLONE_HOURS" >&2
        exit 1
    }
    export TOTAL_CLONE_HOURS
    RESTORED_BUDGET=1
    echo "[07] 续跑恢复 TOTAL_CLONE_HOURS=${TOTAL_CLONE_HOURS} (${ALLOCATION_SUMMARY})"
fi

if [ "${RESUMING}" -eq 0 ] && [ -z "${TOTAL_CLONE_HOURS:-}" ]; then
    SUMMARY_JSON="${STATS_OUTPUT_DIR}/summary.json"
    if [ -f "${STATS_CSV}" ] && [ "${FORCE_STATS}" != "1" ]; then
        if ! STATS_SOURCE_DIRS="${SOURCE_DIRS}" \
            STATS_CLONE_DIRS="${STATS_CLONE_DIRS}" \
            TARGET_SEC="${TARGET_SEC}" \
            SUMMARY_JSON="${SUMMARY_JSON}" \
            python3 - <<'PY'
import json
import os
import sys

try:
    summary = json.load(open(os.environ["SUMMARY_JSON"]))
    want_source = sorted(os.environ["STATS_SOURCE_DIRS"].split())
    want_clone = sorted(os.environ["STATS_CLONE_DIRS"].split())
    have_source = sorted(summary.get("source_dirs") or [])
    have_clone = sorted(summary.get("clone_dirs") or [])
    same_target = float(summary.get("target_sec")) == float(os.environ["TARGET_SEC"])
except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
    sys.exit(1)
sys.exit(0 if want_source == have_source and want_clone == have_clone and same_target else 1)
PY
        then
            echo "[07] 已有统计的 source/clone/target 口径不匹配，强制重算"
            FORCE_STATS=1
        fi
    fi

    if [ ! -f "${STATS_CSV}" ] || [ "${FORCE_STATS}" = "1" ]; then
        echo "[07] Step 0 统计 (source + clone-dirs) 以测算预算 ..."
        echo "     source     : ${SOURCE_DIRS}"
        echo "     clone-dirs : ${STATS_CLONE_DIRS}"
        mkdir -p "${STATS_OUTPUT_DIR}"
        # shellcheck disable=SC2086
        python3 "${HERE}/00_prepare_stats.py" \
            --source-dirs ${SOURCE_DIRS} \
            --clone-dirs ${STATS_CLONE_DIRS} \
            --target-sec "${TARGET_SEC}" \
            --output-dir "${STATS_OUTPUT_DIR}" \
            --workers "${SCAN_WORKERS}"
    else
        echo "[07] STATS_CSV 已存在, 复用: ${STATS_CSV} (FORCE_STATS=1 可强制重算)"
    fi

    GAP_HOURS="$(python3 -c "import json;print(json.load(open('${STATS_OUTPUT_DIR}/summary.json'))['gap_hours'])")"
    if python3 -c "import sys; sys.exit(0 if float('${GAP_HOURS}') <= 0 else 1)"; then
        echo "[07] gap=${GAP_HOURS}h ≤ 0: 无需补齐, 退出"
        exit 0
    fi
    TOTAL_CLONE_HOURS="$(python3 -c "import math;print(int(math.ceil(${GAP_HOURS} / ${SURVIVAL_EST})))")"
    if [ "${TOTAL_CLONE_HOURS}" -eq 0 ]; then
        echo "[07] TOTAL_CLONE_HOURS=0: 无需补齐, 退出"
        exit 0
    fi
    export TOTAL_CLONE_HOURS
    echo "[07] gap=${GAP_HOURS}h  survival_est=${SURVIVAL_EST}  →  TOTAL_CLONE_HOURS=${TOTAL_CLONE_HOURS}"
elif [ "${RESTORED_BUDGET}" -eq 1 ]; then
    echo "[07] 续跑模式: 使用已恢复预算 ${TOTAL_CLONE_HOURS}, 复用 05 已有分配基准"
elif [ -n "${TOTAL_CLONE_HOURS:-}" ]; then
    echo "[07] 使用手动设置 TOTAL_CLONE_HOURS=${TOTAL_CLONE_HOURS}"
else
    echo "[07] 续跑模式: 跳过预算测算 (复用 05 已有分配基准)"
fi

# 07 已经负责统计口径校验及必要的重算。不要把 FORCE_STATS=1 继续传给
# 05，否则同一份四目录统计会无意义地再扫描一遍。
export FORCE_STATS=0

echo "[07] 交给 05_iterative_pipeline.sh 执行核心流程 ..."
exec bash "${HERE}/05_iterative_pipeline.sh"
