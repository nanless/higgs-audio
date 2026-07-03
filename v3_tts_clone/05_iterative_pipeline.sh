#!/bin/bash
# =============================================================================
# Higgs Audio v3 — 10轮迭代克隆流水线
#
# 流程:
#   1. 一次性分配预算 (04_post_prune_stats.py)
#   2. 每轮: 扫描磁盘现有 clone → 限制克隆量为 1/N → 克隆 → SIM评估→SIM剪枝 → CER评估→CER剪枝
#      (先用快的 SIM 删一波, 缩小昂贵的 ASR/CER 工作量)
#   3. 重复 N 轮，不重新分配预算
#
# 用法:
#   bash v3_tts_clone/05_iterative_pipeline.sh
# 或
#   source v3_tts_clone/05_iterative_pipeline.env && bash v3_tts_clone/05_iterative_pipeline.sh
#
# 环境变量 (可选, 有默认值):
#   STATS_CSV          — speaker_duration_stats.csv
#   TEXTS_JSONL        — 文本池 JSONL
#   CLONE_ROOT         — clone 输出根目录
#   TOTAL_CLONE_HOURS  — 全局克隆预算 (默认 10000)
#   TOTAL_ROUNDS       — 总轮数 (默认 10)
#   BASE_PORT          — SGLang 服务端口 (默认 8000)
#   NUM_SERVERS        — SGLang 服务数量 (默认 8)
#   WORKERS_PER_SERVER — 每 server 并发数 (默认 16)
#   START_ROUND        — 从第几轮开始 (默认 1)
#   START_STEP         — 起始轮从哪步开始: clone|sim|cer (默认 clone; SIM 在前, CER 在后)
#
# 续跑示例 (复用前面已跑出的结果):
#   START_ROUND=1 START_STEP=sim   bash 05_iterative_pipeline.sh  # 第1轮从 SIM 评估开始(克隆已完成)
#   START_ROUND=1 START_STEP=cer   bash 05_iterative_pipeline.sh  # 第1轮从 CER 评估开始(SIM也已完成)
#   START_ROUND=3                  bash 05_iterative_pipeline.sh  # 直接从第3轮的克隆开始
# 步骤顺序/级别: clone=1 < sim=2 < cer=3 (SIM 在前, CER 在后)
# (START_ROUND>1 或 START_STEP!=clone 时跳过 ASR + 预算分配, 复用已有分配基准与磁盘 clone)
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
V3_CLONE_DIR="${REPO_ROOT}/v3_tts_clone"
EVAL_CER_DIR="${REPO_ROOT}/eval_higgs_audio/eval_cer"
EVAL_SIM_DIR="${REPO_ROOT}/eval_higgs_audio/eval_sim"
EVAL_DIR="${REPO_ROOT}/eval_higgs_audio"

# ---- 可配置参数 ----
STATS_CSV="${STATS_CSV:-${REPO_ROOT}/clone_workdir/speaker_duration_stats_v2.csv}"
TEXTS_JSONL="${TEXTS_JSONL:-${REPO_ROOT}/higgs_audio_v3_text_generator/batch_output_v2/generated_texts_final.jsonl}"
CLONE_ROOT="${CLONE_ROOT:-/root/group-shared/voiceprint/data/speech/speaker_diarization/merged_datasets_20250610_vad_segments_mtfaa_enhanced_extend_kid_withclone_addlibrilight_1130/audio_higgs_audio_v3_tts_clone}"
PIPELINE_WORKDIR="${PIPELINE_WORKDIR:-${REPO_ROOT}/clone_workdir/iterative_pipeline}"
# 原音频目录 (参考音频选取)
SOURCE_DIRS="${SOURCE_DIRS:-}"
MERGED_SOURCES_DIR="${PIPELINE_WORKDIR}/merged_sources"
TOTAL_CLONE_HOURS="${TOTAL_CLONE_HOURS:-10000}"
TOTAL_ROUNDS="${TOTAL_ROUNDS:-10}"
BASE_PORT="${BASE_PORT:-8000}"
NUM_SERVERS="${NUM_SERVERS:-4}"
WORKERS_PER_SERVER="${WORKERS_PER_SERVER:-16}"
ESTIMATE_CLONE_DURATION="${ESTIMATE_CLONE_DURATION:-10}"
QUALITY_PASS_RATE="${QUALITY_PASS_RATE:-0.5}"
MAX_CER="${MAX_CER:-0.03}"
MIN_SIM="${MIN_SIM:-0.8}"        # raw 余弦阈值 (编码器已改为 raw cos)
TARGET_SEC="${TARGET_SEC:-3600}"  # 每说话人目标时长(秒); 本次生产用 1800 (半小时)
ALL_GPUS="${ALL_GPUS:-0,1,2,3}"
ASR_BACKEND="${ASR_BACKEND:-vllm}"          # vllm (连续批处理, 高 GPU 利用率) | transformers
ASR_BATCH_SIZE="${ASR_BATCH_SIZE:-32}"       # transformers 后端每卡 batch
ASR_VLLM_BATCH="${ASR_VLLM_BATCH:-256}"      # vllm 后端喂给引擎的批大小 (越大越能填满连续批处理)
ASR_GPU_MEM_UTIL="${ASR_GPU_MEM_UTIL:-0.9}"  # vllm gpu_memory_utilization
ASR_AUDIO_WORKERS="${ASR_AUDIO_WORKERS:-16}"
ASR_MAX_NEW_TOKENS="${ASR_MAX_NEW_TOKENS:-512}"
TTS_MAX_NEW_TOKENS="${TTS_MAX_NEW_TOKENS:-1024}"
PRUNE_WORKERS="${PRUNE_WORKERS:-32}"
SCAN_WORKERS="${SCAN_WORKERS:-64}"    # 全盘扫描并行进程数 (128 核, 网络盘 I/O 密集, 用大值猛猛加速)
SIM_WORKERS="${SIM_WORKERS:-16}"      # SIM 评估进程数 (每卡 4 个, 4 卡=16; 逐条计算不变, 填满 GPU)
SEED="${SEED:-42}"
START_ROUND="${START_ROUND:-1}"
START_STEP="${START_STEP:-clone}"

mkdir -p "${PIPELINE_WORKDIR}"

# ---- 日志 ----
LOG_FILE="${PIPELINE_WORKDIR}/pipeline_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "=============================================="
echo " Higgs Audio v3 — 迭代克隆流水线"
echo " 总轮数: ${TOTAL_ROUNDS}"
echo " 克隆预算: ${TOTAL_CLONE_HOURS} 小时"
echo " 开始时间: $(date)"
echo " 日志: ${LOG_FILE}"
echo "=============================================="
echo ""

# ---- 续跑控制: START_ROUND / START_STEP ----
# 步骤级别: clone=1 < sim=2 < cer=3 (SIM 在前, CER 在后); 起始轮只运行 level >= 起始 level 的步骤
case "${START_STEP}" in
    clone) START_STEP_LEVEL=1 ;;
    sim)   START_STEP_LEVEL=2 ;;
    cer)   START_STEP_LEVEL=3 ;;
    *) echo "❌ START_STEP 非法: '${START_STEP}' (可选: clone|sim|cer)"; exit 1 ;;
esac

# 续跑模式: START_ROUND>1 或 起始轮不从 clone 开始 → 跳过 ASR + 预算分配, 复用已有结果
RESUMING=0
if [ "${START_ROUND}" -gt 1 ] || [ "${START_STEP_LEVEL}" -gt 1 ]; then
    RESUMING=1
    echo "▶ 续跑模式: 从第 ${START_ROUND} 轮的 '${START_STEP}' 步开始 (跳过 ASR + 预算分配, 复用已有结果)"
else
    echo "▶ 全新运行: 从第 1 轮 clone 开始"
fi
echo ""

# ---- 工具函数 ----
log_step() {
    echo ""
    echo "============================================================"
    echo "  [$1]  $2  ($(date +%H:%M:%S))"
    echo "============================================================"
}

run_or_fail() {
    local desc="$1"; local cmd="$2"
    log_step "CMD" "${desc}"
    echo "  $ ${cmd}"
    if eval "${cmd}"; then
        echo "  ✅ 成功"
        return 0
    else
        echo "  ❌ 失败 (退出码: $?)"
        return 1
    fi
}
build_merged_sources() {
    local merged_root="$1"
    local csv_path="$2"
    shift 2
    local src_dirs=("$@")

    if [ ${#src_dirs[@]} -eq 0 ]; then
        return 0
    fi

    log_step "MERGE" "合并多个原音频目录 → ${merged_root}"
    echo "  源目录: ${src_dirs[*]}"
    echo "  合并目标: ${merged_root}"

    export _PIPE_MERGED_ROOT="${merged_root}"
    export _PIPE_CSV_PATH="${csv_path}"
    export _PIPE_SRC_DIRS="${src_dirs[*]}"
    python << 'PYEOF'
import os, csv, sys

merged_root = os.environ.get('_PIPE_MERGED_ROOT', '')
csv_path = os.environ.get('_PIPE_CSV_PATH', '')
src_dirs_str = os.environ.get('_PIPE_SRC_DIRS', '')
src_dirs = [d for d in src_dirs_str.split() if d]

AUDIO_EXTS = {'.wav', '.flac', '.mp3'}

# 读取 CSV 获取所有 (dataset, speaker_id) 对
speakers = []
with open(csv_path, newline='') as f:
    for r in csv.DictReader(f):
        speakers.append((r['dataset'], r['speaker_id']))

merged = 0
skipped = 0
missing_all = 0

for dataset, speaker_id in speakers:
    # 检查是否已有合并目录 (含音频文件)
    merged_spk = os.path.join(merged_root, dataset, speaker_id)
    existing = set()
    if os.path.isdir(merged_spk):
        for fname in os.listdir(merged_spk):
            if os.path.splitext(fname)[1].lower() in AUDIO_EXTS:
                existing.add(fname)
    if existing:
        skipped += 1
        continue

    # 从所有源目录收集音频文件
    found = False
    all_files = {}  # fname -> src_path
    for src_root in src_dirs:
        src_spk = os.path.join(src_root, dataset, speaker_id)
        if not os.path.isdir(src_spk):
            continue
        for fname in sorted(os.listdir(src_spk)):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in AUDIO_EXTS:
                continue
            src_path = os.path.join(src_spk, fname)
            if fname not in all_files:
                all_files[fname] = src_path
                found = True

    if not found:
        missing_all += 1
        continue

    # 创建符号链接
    os.makedirs(merged_spk, exist_ok=True)
    for fname, src_path in sorted(all_files.items()):
        dst_path = os.path.join(merged_spk, fname)
        if not os.path.lexists(dst_path):
            os.symlink(os.path.realpath(src_path), dst_path)
            merged += 1

print(f'  合并完成: {merged} 个符号链接, {skipped} 个说话人已有合并目录')
if missing_all:
    print(f'  ⚠️  {missing_all} 个说话人在所有源目录中都未找到音频')
PYEOF
}

# ====================================================================
#  Step 0: 检查并合并原音频目录
# ====================================================================
if [ ! -f "${STATS_CSV}" ]; then
    echo "❌ STATS_CSV 不存在: ${STATS_CSV}"
    echo "   请先运行: python3 v3_tts_clone/00_prepare_stats.py --source-dirs ... --clone-dirs ..."
    exit 1
fi

SRC_COUNT=$(echo ${SOURCE_DIRS} | /usr/bin/wc -w)
if [ "${SRC_COUNT}" -gt 1 ]; then
    build_merged_sources "${MERGED_SOURCES_DIR}" "${STATS_CSV}" ${SOURCE_DIRS}
elif [ -n "${SOURCE_DIRS}" ]; then
    echo "✓ SOURCE_DIRS 仅 1 个目录, speaker_path 会被覆盖为此目录"
else
    echo "⚠️  SOURCE_DIRS 未设置, 将使用 STATS_CSV 中原有的 speaker_path (可能指向 clone 目录)"
fi

# ====================================================================
#  Step 0.5a: ASR 转写 (为参考音频生成 transcript)
# ====================================================================
if [ "${RESUMING}" -eq 1 ]; then
    log_step "ASR" "续跑模式: 跳过 ASR 转写 (复用已有 transcript sidecar)"
else
    log_step "ASR" "ASR 转写源音频 (${ALL_GPUS} GPU)"
    # conda 在非交互 shell 中可能未初始化
    if ! command -v conda &>/dev/null; then
        if [ -f /root/miniforge3/etc/profile.d/conda.sh ]; then
            source /root/miniforge3/etc/profile.d/conda.sh
        elif [ -f /root/anaconda3/etc/profile.d/conda.sh ]; then
            source /root/anaconda3/etc/profile.d/conda.sh
        fi
    fi
    bash "${V3_CLONE_DIR}/02_asr_launch.sh" "${ALL_GPUS}" "${STATS_CSV}" "${TARGET_SEC}"
    echo "  ✅ ASR 转写完成"

    # 释放 GPU 显存: 杀掉 ASR worker 进程, 防止影响后续 TTS
    pkill -f "02_asr_worker" 2>/dev/null || true
    sleep 5
    echo "  ✅ ASR 进程已释放 GPU"
fi

MODEL_PATH="${SGLANG_MODEL_PATH:-/root/models/higgs-audio-v3-tts-4b}"

# ====================================================================
#  Step 1: 合并所有 clone 目录的时长 + 一次性分配预算
# ====================================================================
ALLOC_DIR="${PIPELINE_WORKDIR}/allocation"
mkdir -p "${ALLOC_DIR}"
FULL_RESUME_CSV="${ALLOC_DIR}/speaker_duration_stats_post_prune_resume.csv"
ORIG_CLONES_JSON="${PIPELINE_WORKDIR}/original_clones_needed.json"

if [ "${RESUMING}" -eq 1 ]; then
    log_step "ALLOC" "续跑模式: 跳过预算分配 (复用已有分配基准)"
    if [ ! -f "${FULL_RESUME_CSV}" ] || [ ! -f "${ORIG_CLONES_JSON}" ]; then
        echo "❌ 续跑需要已有的预算分配结果, 但缺少:"
        echo "   ${FULL_RESUME_CSV}"
        echo "   ${ORIG_CLONES_JSON}"
        echo "   请先用 START_ROUND=1 START_STEP=clone 完整跑到分配完成后再续跑。"
        exit 1
    fi
else

# Step 1a: 扫描 CLONE_ROOT 预存 clone (仅用于日志展示已有 clone 时长)
# 注意: 04_post_prune_stats.py 会自己重新扫描 CLONE_ROOT 计入 clone 时长,
#       因此这里生成的 ADJUSTED_STATS_CSV 不再喂给 04 (否则 clone 时长会被双重计入)。
#       旧 clone 目录 (--clone-dirs) 的时长已由 00_prepare_stats.py 纳入 total_duration_sec。
log_step "DUR" "扫描 CLONE_ROOT 预存 clone (信息展示)"
ADJUSTED_STATS_CSV="${ALLOC_DIR}/stats_with_clone_dur.csv"
export _PIPE_STATS_CSV="${STATS_CSV}"
export _PIPE_CLONE_ROOTS="${CLONE_ROOT}"
export _PIPE_OUT_CSV="${ADJUSTED_STATS_CSV}"
python << 'PYEOF2'
import os, csv, struct, re, sys

stats_csv = os.environ.get('_PIPE_STATS_CSV', '')
clone_roots_str = os.environ.get('_PIPE_CLONE_ROOTS', '')
out_csv = os.environ.get('_PIPE_OUT_CSV', '')
clone_roots = [d for d in clone_roots_str.split() if d and os.path.isdir(d)]

CLONE_WAV_RE = re.compile(r'^clone_(\d+)\.wav$')

# 扫描所有 clone root, 计算每个说话人的 clone 总时长
# key: dataset/speaker_id → total_clone_dur_sec
clone_dur_map = {}
for clone_root in clone_roots:
    if not os.path.isdir(clone_root):
        continue
    for dataset in os.listdir(clone_root):
        ds_path = os.path.join(clone_root, dataset)
        if not os.path.isdir(ds_path):
            continue
        for speaker_id in os.listdir(ds_path):
            spk_path = os.path.join(ds_path, speaker_id)
            if not os.path.isdir(spk_path):
                continue
            total_dur = 0.0
            try:
                for fname in os.listdir(spk_path):
                    m = CLONE_WAV_RE.match(fname)
                    if not m:
                        continue
                    fpath = os.path.join(spk_path, fname)
                    try:
                        fsize = os.path.getsize(fpath)
                    except OSError:
                        continue
                    if fsize <= 1000:
                        continue
                    # 快速 WAV 时长计算 (44 bytes header)
                    try:
                        with open(fpath, 'rb') as wf:
                            hdr = wf.read(44)
                        if len(hdr) >= 44 and hdr[:4] == b'RIFF':
                            ch = struct.unpack_from('<H', hdr, 22)[0]
                            sr = struct.unpack_from('<I', hdr, 24)[0]
                            bps = struct.unpack_from('<H', hdr, 34)[0]
                            if sr > 0 and bps > 0 and ch > 0:
                                total_dur += (fsize - 44) / (sr * ch * (bps / 8))
                    except Exception:
                        pass
            except OSError:
                pass
            if total_dur > 0:
                key = f'{dataset}/{speaker_id}'
                clone_dur_map[key] = clone_dur_map.get(key, 0.0) + total_dur

# 读取原始 STATS_CSV, 调整 total_duration_sec
rows = list(csv.DictReader(open(stats_csv, newline='')))
fieldnames = list(rows[0].keys())
adjusted_count = 0
total_added_hours = 0.0
with open(out_csv, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for r in rows:
        key = f'{r["dataset"]}/{r["speaker_id"]}'
        extra_dur = clone_dur_map.get(key, 0.0)
        if extra_dur > 0:
            orig_dur = float(r['total_duration_sec'])
            r['total_duration_sec'] = str(round(orig_dur + extra_dur, 2))
            adjusted_count += 1
            total_added_hours += extra_dur / 3600.0
        w.writerow(r)

print(f'  扫描 clone 目录: {len(clone_roots)} 个 ({clone_roots_str})')
print(f'  找到有 clone 的说话人: {len(clone_dur_map)} 个')
print(f'  total_duration_sec 已调整: {adjusted_count} 个说话人')
print(f'  累计增加 clone 时长: {total_added_hours:,.1f} 小时')
PYEOF2

# Step 1b: 预算分配 (直接用原始 STATS_CSV; 04 自身扫描 --clone-root 计入 clone 时长, 避免双重计入)
log_step "ALLOC" "预算分配 (total_clone_hours=${TOTAL_CLONE_HOURS})"
run_or_fail "04_post_prune_stats" \
    "python ${V3_CLONE_DIR}/04_post_prune_stats.py \
        --stats-csv ${STATS_CSV} \
        --clone-root ${CLONE_ROOT} \
        --output-dir ${ALLOC_DIR} \
        --total-clone-hours ${TOTAL_CLONE_HOURS} \
        --target-duration-sec ${TARGET_SEC} \
        --estimate-clone-duration ${ESTIMATE_CLONE_DURATION}"

if [ ! -f "${FULL_RESUME_CSV}" ]; then
    echo "❌ 预算分配失败: 找不到 ${FULL_RESUME_CSV}"
    exit 1
fi

# Step 1c: 保存原始预算分配
python ${V3_CLONE_DIR}/05_save_orig_allocation.py \
    --resume-csv ${FULL_RESUME_CSV} \
    --out-json ${ORIG_CLONES_JSON}
fi

TOTAL_NEEDED=$(python -c "import json; print(sum(v['clones_needed'] for v in json.load(open('${ORIG_CLONES_JSON}')).values()))")
echo "  每轮克隆约: $(( TOTAL_NEEDED / TOTAL_ROUNDS )) (共 ${TOTAL_ROUNDS} 轮)"

# ====================================================================
#  Step 2: 循环 N 轮
# ====================================================================
for round in $(seq 1 ${TOTAL_ROUNDS}); do
    rpad=$(printf "%02d" ${round})

    # 续跑: 跳过 START_ROUND 之前的轮次
    if [ "${round}" -lt "${START_ROUND}" ]; then
        echo "⏭  跳过第 ${round} 轮 (< START_ROUND=${START_ROUND})"
        continue
    fi
    # 起始轮从 START_STEP 开始, 其余轮从 clone 开始
    if [ "${round}" -eq "${START_ROUND}" ]; then
        cur_start_level=${START_STEP_LEVEL}
    else
        cur_start_level=1
    fi

    echo ""
    echo "################################################################"
    echo "###  第 ${round}/${TOTAL_ROUNDS} 轮 (每轮克隆约 1/${TOTAL_ROUNDS})"
    if [ "${cur_start_level}" -gt 1 ]; then
        echo "###  续跑: 从 '${START_STEP}' 步开始 (跳过前置步骤, 复用磁盘已有结果)"
    fi
    echo "###  $(date)"
    echo "################################################################"

    ROUND_DIR="${PIPELINE_WORKDIR}/round_${rpad}"
    mkdir -p "${ROUND_DIR}"
    ROUND_CSV="${ROUND_DIR}/resume_round.csv"
    EXISTING_CLONES_JSON="${ROUND_DIR}/existing_clones.json"

    # ---- 扫描磁盘，获取每个说话人的实际现有 clone 数量 (始终执行, 作为本轮基线) ----
    log_step "SCAN" "扫描磁盘现有 clone"
    python ${V3_CLONE_DIR}/05_scan_existing_clones.py \
        --clone-root ${CLONE_ROOT} \
        --resume-csv ${FULL_RESUME_CSV} \
        --workers ${SCAN_WORKERS} \
        --out-json ${EXISTING_CLONES_JSON}

    # ================= 2a: 克隆 (clone, level 1) =================
    if [ "${cur_start_level}" -le 1 ]; then
        # ---- 生成本轮 CSV ----
        # 始终用 SOURCE_DIRS 覆盖 speaker_path, 确保参考音频来自原始音频
        # 单目录直接用第一个, 多目录用合并后的符号链接目录
        if [ -n "${SOURCE_DIRS}" ]; then
            SRC_COUNT=$(echo ${SOURCE_DIRS} | /usr/bin/wc -w)
            if [ "${SRC_COUNT}" -gt 1 ]; then
                REF_DIR="${MERGED_SOURCES_DIR}"
            else
                REF_DIR="${SOURCE_DIRS}"
            fi
            MERGED_ARG="--merged-dir ${REF_DIR}"
        else
            MERGED_ARG=""
        fi
        run_or_fail "生成本轮 CSV" \
            "python ${V3_CLONE_DIR}/05_generate_round_csv.py \
                --orig-json ${ORIG_CLONES_JSON} \
                --existing-json ${EXISTING_CLONES_JSON} \
                --resume-csv ${FULL_RESUME_CSV} \
                ${MERGED_ARG} \
                --total-rounds ${TOTAL_ROUNDS} \
                --out-csv ${ROUND_CSV}"

        # 检查是否有待克隆项
        has_clones=$(python -c "import csv; rows=csv.DictReader(open('${ROUND_CSV}','r')); print(sum(int(r.get('clones_needed',0)) for r in rows))") 2>/dev/null || true
        if [ "${has_clones:-0}" = "0" ]; then
            echo "✅ 本轮无需克隆，所有说话人已达预算上限"
            continue
        fi

        # ---- 启动 TTS → 克隆 → 停止 TTS (释放 GPU 给后续评估) ----
        log_step "SGLANG" "启动 TTS 服务 (Round ${round})"
        GPU_LIST=$(echo $(seq 0 $((NUM_SERVERS-1))) | tr ' ' ',')
        nohup bash "${V3_CLONE_DIR}/03_launch_servers.sh" \
            "${GPU_LIST}" "${MODEL_PATH}" "${BASE_PORT}" \
            > "${PIPELINE_WORKDIR}/sglang_r${round}.log" 2>&1 &
        for i in $(seq 1 30); do
            if [ "$(curl -s -o /dev/null -w '%{http_code}' "http://localhost:${BASE_PORT}/health" 2>/dev/null)" = "200" ]; then break; fi
            sleep 10
        done

        # 每轮 seed 不同 (round * 1000000 偏移, 远大于单说话人 clone 数, 避免与 clone_idx 混叠):
        # 即使尾部被剪枝的编号在下一轮被复用, 文本/参考也会不同; 同一轮内仍确定, 支持断点续跑。
        ROUND_SEED=$(( SEED + round * 1000000 ))
        run_or_fail "Round ${round} 克隆 (seed=${ROUND_SEED})" \
            "python ${V3_CLONE_DIR}/03_tts_clone.py \
                --stats-csv ${ROUND_CSV} \
                --texts-jsonl ${TEXTS_JSONL} \
                --output-root ${CLONE_ROOT} \
                --base-port ${BASE_PORT} \
                --num-servers ${NUM_SERVERS} \
                --workers-per-server ${WORKERS_PER_SERVER} \
                --seed ${ROUND_SEED} \
                --post-prune \
                --max-new-tokens ${TTS_MAX_NEW_TOKENS} \
                --estimate-clone-duration ${ESTIMATE_CLONE_DURATION} \
                --quality-pass-rate ${QUALITY_PASS_RATE}"

        pkill -f "sgl-omni serve" 2>/dev/null || true
        sleep 5
        echo "  ✅ TTS 已停止, GPU 释放"
        sleep 5

        # 保存本轮克隆摘要 (clone_summary.json 会被每轮覆盖, 这里归档)
        CLONE_SUMMARY="${CLONE_ROOT}/clone_summary.json"
        if [ -f "${CLONE_SUMMARY}" ]; then
            cp "${CLONE_SUMMARY}" "${ROUND_DIR}/clone_summary.json"
        fi
    else
        echo "  ⏭  跳过克隆 (START_STEP=${START_STEP}), 复用磁盘已有 clone"
    fi

    # ================= 2b: SIM 评估 + SIM 剪枝 (sim, level 2) =================
    # 先跑快的 SIM, 用 SIM 删一波 (缩小后面昂贵的 ASR/CER 工作量)
    if [ "${cur_start_level}" -le 2 ]; then
        # 全部4卡, 每次全新 os.walk 扫描
        run_or_fail "SIM 评估" \
            "bash ${EVAL_SIM_DIR}/run_eval_sim.sh \
                --out-dir ${CLONE_ROOT} \
                --skip-existing \
                --gpus ${ALL_GPUS} \
                --scan-workers ${SCAN_WORKERS} \
                --workers ${SIM_WORKERS}" || echo "⚠️  SIM 评估失败，跳过 SIM 剪枝，继续 CER"

        # 释放 SIM 占用的 GPU, 防止残留 (含 spawn worker) 影响后续 CER/TTS
        pkill -f "eval_sim.py" 2>/dev/null || true
        sleep 5
        echo "  ✅ SIM 评估进程已释放 GPU"

        # SIM 剪枝: 只按 SIM 删 (--max-cer 999 关闭 CER 判定), 缩小 CER 待评估集
        log_step "PRUNE-SIM" "SIM 剪枝 (SIM<${MIN_SIM}, 先删一波再跑 ASR)"
        run_or_fail "Prune(SIM)" \
            "python ${EVAL_DIR}/prune_and_copy.py \
                --out-dir ${CLONE_ROOT} \
                --max-cer 999 \
                --min-sim ${MIN_SIM} \
                --eval-source sidecar \
                --eval-workers ${PRUNE_WORKERS} \
                --scan-workers ${SCAN_WORKERS} \
                --workers ${PRUNE_WORKERS}" || echo "⚠️  SIM 剪枝失败，继续 CER"
    else
        echo "  ⏭  跳过 SIM 评估/剪枝 (START_STEP=${START_STEP}), 复用已有 .sim.json"
    fi

    # ================= 2c: CER 评估 + CER 剪枝 (cer, level 3) =================
    if [ "${cur_start_level}" -le 3 ]; then
        # 全部4卡, 强制刷新扫描缓存 (SIM 剪枝已删文件, 缓存必失效); 只评估 SIM 存活的 clone
        # vllm 后端用连续批处理 (TP=卡数) 填满 GPU; batch 用较大的 ASR_VLLM_BATCH
        # 语言预分组对 transformers 有用(防子批变小), 对 vllm 收益小且要预扫全部 json → vllm 关掉
        if [ "${ASR_BACKEND}" = "vllm" ]; then
            CER_BATCH="${ASR_VLLM_BATCH}"
            LANG_GROUP_ARG="--no-group-by-language"
        else
            CER_BATCH="${ASR_BATCH_SIZE}"
            LANG_GROUP_ARG="--group-by-language"
        fi
        run_or_fail "CER 评估" \
            "bash ${EVAL_CER_DIR}/run_eval_cer.sh \
                --out-dir ${CLONE_ROOT} \
                --skip-existing \
                --refresh-scan \
                --scan-workers ${SCAN_WORKERS} \
                --asr-backend ${ASR_BACKEND} \
                --asr-gpus ${ALL_GPUS} \
                --asr-gpu-mem-util ${ASR_GPU_MEM_UTIL} \
                --batch-size ${CER_BATCH} \
                --audio-workers ${ASR_AUDIO_WORKERS} \
                --asr-max-new-tokens ${ASR_MAX_NEW_TOKENS} \
                ${LANG_GROUP_ARG}" || echo "⚠️  CER 评估失败，跳过 CER 剪枝，继续下一轮"

        # 释放 CER (ASR) 占用的 GPU
        pkill -f "eval_cer.py" 2>/dev/null || true
        sleep 5
        echo "  ✅ CER 评估进程已释放 GPU"

        # CER 剪枝: 完整阈值 (SIM 存活里再删 CER>${MAX_CER}; SIM 已合格)
        log_step "PRUNE-CER" "CER 剪枝 (CER>${MAX_CER} 或 SIM<${MIN_SIM})"
        run_or_fail "Prune(CER)" \
            "python ${EVAL_DIR}/prune_and_copy.py \
                --out-dir ${CLONE_ROOT} \
                --max-cer ${MAX_CER} \
                --min-sim ${MIN_SIM} \
                --eval-source sidecar \
                --eval-workers ${PRUNE_WORKERS} \
                --scan-workers ${SCAN_WORKERS} \
                --workers ${PRUNE_WORKERS}" || echo "⚠️  CER 剪枝失败，继续下一轮"
    else
        echo "  ⏭  跳过 CER 评估/剪枝 (START_STEP=${START_STEP}), 复用已有 .cer.json"
    fi

    # 剪枝后统计本轮保留多少
    log_step "STATS" "本轮后统计"
    python ${V3_CLONE_DIR}/05_scan_existing_clones.py \
        --clone-root ${CLONE_ROOT} \
        --resume-csv ${FULL_RESUME_CSV} \
        --workers ${SCAN_WORKERS} \
        --out-json ${ROUND_DIR}/existing_clones_after.json
    python -c "
import json
b=json.load(open('${EXISTING_CLONES_JSON}')); a=json.load(open('${ROUND_DIR}/existing_clones_after.json'))
tb=sum(b.values()); ta=sum(a.values()); d=ta-tb
print(f'  克隆前: {tb:,} 条 | 克隆+剪枝后: {ta:,} 条 | 净{(\"+\" if d>=0 else \"\")}{d}')
" 2>/dev/null || true

    echo "  >>> 第 ${round} 轮完成 <<<"
done

# ====================================================================
#  最终: 汇总
# ====================================================================
echo ""
echo "=============================================="
echo "  流水线完成"
echo "  结束时间: $(date)"
echo "  日志: ${LOG_FILE}"
echo "=============================================="

log_step "FINAL" "最终克隆统计"
python ${V3_CLONE_DIR}/04_post_prune_stats.py \
    --stats-csv ${STATS_CSV} \
    --clone-root ${CLONE_ROOT} \
    --output-dir ${PIPELINE_WORKDIR}/final \
    --total-clone-hours ${TOTAL_CLONE_HOURS} \
    --target-duration-sec ${TARGET_SEC} \
    --estimate-clone-duration ${ESTIMATE_CLONE_DURATION}

log_step "FINAL" "最终质量验证"
python ${EVAL_DIR}/verify_kept_clones.py --out-dir ${CLONE_ROOT} \
    --min-sim ${MIN_SIM} --max-cer ${MAX_CER} \
    --eval-source sidecar --eval-workers ${PRUNE_WORKERS} || true

# 汇总各轮统计
echo ""
echo "=============================================="
echo "  各轮统计汇总"
echo "=============================================="
for rpad in $(seq -w 1 ${TOTAL_ROUNDS}); do
    round_dir="${PIPELINE_WORKDIR}/round_${rpad}"
    after_json="${round_dir}/existing_clones_after.json"
    if [ -f "${after_json}" ]; then
        count=$(python -c "import json;print(sum(json.load(open('${after_json}')).values()))")
        echo "  第 ${rpad} 轮后: ${count} 条 clone"
    fi
done
