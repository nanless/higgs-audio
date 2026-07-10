# Higgs Audio v3 — 迭代克隆流水线

## 概述

一键运行：统计说话人 → ASR 转写 → 预算分配 → 10 轮迭代（每轮：启动 TTS → 克隆 1/10 → 停止 TTS → SIM → CER → 剪枝）。

## 脚本

```
00_prepare_stats.py            生成全量说话人统计 (Step 0, 被 05 自动调用)
02_asr_launch.sh / 02_asr_worker.py  ASR 转写 (被 05 调用)
03_launch_servers.sh           SGLang TTS 启动 (被 05 调用, 每轮启停)
03_tts_clone.py                TTS 克隆 (被 05 调用)
04_post_prune_stats.py         预算分配 (被 05 调用)
05_scan_existing_clones.py     扫描已有 clone 数
05_generate_round_csv.py       每轮生成受限 CSV
05_save_orig_allocation.py     保存分配基准
05_iterative_pipeline.sh       主流水线 (包含 ASR + TTS + 10 轮迭代)
05_iterative_pipeline.env      配置
06_filter_copy_by_sim.py       按 raw SIM 阈值过滤并拷贝老 clone 目录 (合并/去重名, 见下文)
07_topup_pipeline.sh           补齐(top-up)流水线启动器: 把已有过滤 clone 计入 baseline (见下文)
07_topup_pipeline.env          07 的配置 (源=audio, 统计口径含已过滤 clone 目录)
08_resume_topup.sh             从中断轮续跑 topup (默认 START_ROUND=2 START_STEP=clone)
08_preflight_resume.py         续跑前预检 (allocation / 磁盘 / GPU / 建议 START_STEP)
```

---

## 准备

### 1. 统计说话人 (Step 0, 流水线自动完成)

**不需要手动跑**——`05_iterative_pipeline.sh` 的 Step 0 会自动运行 `00_prepare_stats.py`，
统计 `SOURCE_DIRS`(audio 目录) 下**所有数据集/说话人**，生成 **source-only** 的
`${STATS_OUTPUT_DIR}/all_speakers.csv`（默认写到 `STATS_CSV` 所在目录）。
若 `STATS_CSV` 已存在则跳过；`FORCE_STATS=1` 可强制重算。

> 只在需要把**以前的克隆时长**也计入统计时才手动跑并带 `--clone-dirs`（此时 STATS 不再是 source-only，
> 与迭代流水线的 source-only 约定冲突，一般不用）：
> ```bash
> python3 v3_tts_clone/00_prepare_stats.py \
>     --source-dirs /root/group-shared/.../audio \
>     --target-sec 1800 \
>     --output-dir clone_workdir/stats_source_only --workers 64
> ```
> - `--source-dirs`：原始音频，`speaker_path` 取自此。输出固定为 `{output-dir}/all_speakers.csv`。

### 2. 编辑配置

`v3_tts_clone/05_iterative_pipeline.env`：

```bash
REPO="/root/code/github_repos/higgs-audio"
BASE="/root/group-shared/..."

# STATS_CSV: Step 0 自动生成的 source-only all_speakers.csv (已存在则复用)
export STATS_CSV="${REPO}/clone_workdir/stats_source_only/all_speakers.csv"
# TEXTS_JSONL = 克隆用的文本池 (每行一条 JSON, 含 text/clean_text)
export TEXTS_JSONL="${REPO}/higgs_audio_v3_text_generator/batch_output_v2/generated_texts_final.jsonl"
export SOURCE_DIRS="${BASE}/audio"                 # Step 0 统计此目录; 也是参考音频来源
export CLONE_ROOT="${BASE}/audio_higgs_audio_v3_tts_clone_4"
export TOTAL_CLONE_HOURS=60000
export TARGET_SEC=1800
export TOTAL_ROUNDS=10
export NUM_SERVERS=8                                # 八卡机
export ALL_GPUS="0,1,2,3,4,5,6,7"                  # ASR/评估用
export SIM_WORKERS=32                               # 每卡 4 × 8 卡
# 评估提速 / 长度控制
export ASR_BATCH_SIZE=32
export ASR_AUDIO_WORKERS=16
export ASR_MAX_NEW_TOKENS=512
export TTS_MAX_NEW_TOKENS=1024   # ↑ 提高可让长文本完整发声(更耗算力), 详见下文
```

### 3. 运行

```bash
source v3_tts_clone/05_iterative_pipeline.env
tmux new-session -d -s higgs "bash v3_tts_clone/05_iterative_pipeline.sh"
```

查看：`tmux attach -t higgs` 或 `tail -f clone_workdir/iterative_pipeline/pipeline_*.log`

---

## 内部流程

```
Step 0    生成说话人统计 (00_prepare_stats.py, source-only; 已存在则跳过, FORCE_STATS=1 强制)
Step 0b   检查 STATS_CSV, 合并参考音频目录
Step 0.5a ASR 转写源音频 → 释放 GPU
Step 1a   扫描 CLONE_ROOT 预存 clone
Step 1b   预算分配 (04_post_prune_stats.py)
Step 1c   保存分配基准 (05_save_orig_allocation.py)

Step 2 × 10:
  ├ 启动 TTS → 健康检查
  ├ 扫描已有 clone (05_scan_existing_clones.py)
  ├ 生成本轮 CSV (05_generate_round_csv.py): per_round = ceil(原始/10)
  ├ 克隆 (03_tts_clone.py --post-prune)
  ├ 停止 TTS → 释放 GPU
  ├ SIM 评估 (--skip-existing) → pkill eval_sim.py 释放 GPU
  ├ SIM 剪枝 (只按 SIM<0.8 删一波, --max-cer 999 关闭 CER 判定; 缩小 CER 待评集)
  ├ CER 评估 (SIM 存活集; --skip-existing --refresh-scan, batch/audio-workers/语言分组) → pkill eval_cer.py 释放 GPU
  ├ CER 剪枝 (CER>0.03 或 SIM<0.8, 读 per-clone sidecar)
  └ 统计 (归档本轮 existing_clones_after.json)

最终  04 最终统计 + verify_kept_clones.py 质量验证 + 各轮汇总
```

> **评估顺序 = SIM 先, CER 后**: SIM 快, 先删掉说话人相似度不达标的一批, 再对存活集跑昂贵的 ASR/CER, 显著减少 ASR 量。`START_STEP` 顺序: `clone(1) < sim(2) < cer(3)`。

> Step 1a 仅扫描 `CLONE_ROOT` 用于日志展示；`04` 会自己扫描 `--clone-root` 计入 clone 时长，两者不叠加。

---

## 关键设计

| 设计 | 原因 |
|------|------|
| 每轮启停 TTS | 克隆后释放 GPU 给 CER/SIM 评估 |
| ASR 后释放 GPU | `pkill asr_worker` 防止抢占 TTS 显存 |
| 预算只分配一次 | 避免每轮重新分配导致漂移 |
| `ceil/10` + `still_need` 防超量 | 原始 5 条第 6 轮自动停止 |
| 磁盘扫描替代内存 tracker | 自动反映剪枝删除和 API 失败 |
| 每轮 seed 不同 (`SEED + round*1000000`) | 尾部剪枝编号被复用时文本/参考仍不同; 同轮内确定, 可续跑 |
| `speaker_path` 覆盖为 SOURCE_DIRS | 参考音频只用原始音频 |
| `--refresh-scan` for CER | 防止剪枝后 pickle 缓存失效 |
| 剪枝/验证读 per-clone sidecar (`--eval-source sidecar`, 多进程) | 聚合 jsonl「首条胜出」去重会用旧记录, 复用编号时误判; sidecar 剪枝删、重评重建, 永远新鲜 |
| SIM = **raw 余弦** (阈值 0.8) | `speaker_encoder` 已去掉 `(cos+1)/2` 映射; `MIN_SIM=0.8` 是 raw 口径 (旧 0.85 mapped = raw 0.70) |
| 目标时长可配 `TARGET_SEC` | 生产用 1800 (半小时); 传给 `00/04/02` |
| 统计只算源音频 | 重跑 `00_prepare_stats.py --source-dirs {audio} --target-sec 1800`(**不带 `--clone-dirs`**), 新 CLONE_ROOT 从 0 |

## 老 clone 目录筛选 (`eval_higgs_audio/prune_prev_clones.py`)

把以前的 clone 目录 (omnivoice/_1/_2/_3) 按**新质量线** (raw<0.8 或 cer>0.03) 筛一遍。读缓存 `.sim.json`(mapped)→`raw=2·sim-1`、`.cer.json`，**GPU-free、无需参考文件**（`_2` 参考已删也能筛）。默认 dry-run：
```bash
python eval_higgs_audio/prune_prev_clones.py --dirs /…/audio_omnivoice_clone /…/_clone /…/_clone_2 /…/_clone_3   # 预览
python eval_higgs_audio/prune_prev_clones.py --dirs … --execute                                                  # 真删
```

## 老 clone 目录过滤+拷贝 (`06_filter_copy_by_sim.py`)

与 `prune_prev_clones.py`「原地删除」不同，此工具**不动源目录**，把满足 `raw 余弦 > 0.8`
的 clone **拷贝**到新目录（wav + 全部 sidecar），用于产出一份干净的高相似度子集。

- **过滤**：`raw = 2·mapped − 1 > 0.8`（老目录 `.sim.json` 存的是 mapped=(cos+1)/2；缺 `.sim.json`/`similarity` 的直接跳过）。GPU-free，无需参考文件。
- **合并去重名**：多个源目录若有相同 `{dataset}/{speaker}/clone_NNNN` 名，给每个源加**自己的文件名前缀**（默认 `c1_/c2_/c3_`）拷进**同一个**目标目录，互不覆盖；单一源（omnivoice）不加前缀、保留原名。
- **保留子目录结构**：目标为 `{dest}/{dataset}/{speaker}/{前缀}{原文件名}.{wav,json,sim.json,cer.json,...}`，与源结构一致。
- **`.sim.json` 追加 raw**：拷贝时写入 `similarity_raw`(原始余弦)、`similarity_mapped`(=原 `similarity`)、`similarity_scale_note`；原 `similarity` 字段保持不变。sidecar 内部的 `cloned_audio`/`wav_path` 等路径**不改写**（仍指向源）。
- 默认 **dry-run**（只统计不拷）；`--execute` 真拷。多进程（`--workers`，默认 32）按 speaker 单元并行。报告写到 `v3_tts_clone/filter_copy_report/`。

```bash
# 预览（默认 jobs 已写死为 higgs _clone/_2/_3 → _123_sim0.8_filtered, omnivoice → _sim0.8_filtered）
python v3_tts_clone/06_filter_copy_by_sim.py --workers 48
# 真拷
python v3_tts_clone/06_filter_copy_by_sim.py --workers 48 --execute
# 自定义 (SRC:DEST:PREFIX 每个 job, prefix 可省)
python v3_tts_clone/06_filter_copy_by_sim.py --jobs /a:/merged:a_ /b:/merged:b_ --min-sim-raw 0.8 --execute
```

> 大规模建议放 tmux：`tmux new-session -d -s filtercopy "python v3_tts_clone/06_filter_copy_by_sim.py --workers 48 --execute > v3_tts_clone/filter_copy_report/execute.log 2>&1"`

## 补齐(top-up)流水线 (`07_topup_pipeline.sh` + `.env`)

在**已有一批过滤后 clone** 的基础上继续把说话人补齐到目标时长的场景。与 `05` 的**唯一区别是源目录与统计口径**——它复用 `05` 的全部核心逻辑 (ASR → 预算分配 → 10 轮 克隆→SIM→CER→剪枝)。

**核心差异 (口径)**：
- **统计"总时长"** = 原始 `audio` + 若干**已过滤 clone 目录** (如 `audio_higgs_audio_v3_tts_clone_123_sim0.8_filtered`、`audio_omnivoice_clone_sim0.8_filtered`)。靠 `00_prepare_stats.py --clone-dirs` 把这些目录时长计入 `total_duration_sec`。
- **参考音频 (ref) 只用 `audio`**：`speaker_path` 由 `--source-dirs` 决定 (只 audio)，每轮 `05_generate_round_csv.py --merged-dir` 也把 `speaker_path` 覆盖为 `SOURCE_DIRS=audio`。
- **只补齐 total < 目标 的说话人**，新 clone 只写入**全新的** `CLONE_ROOT` (如 `..._clone_5`)，不动作为 baseline 的已过滤目录 (它们不是 `--clone-root`，`04` 不会重复计入)。
- **SIM/CER 剪枝阈值与 05 完全一致** (`MAX_CER=0.03`, `MIN_SIM=0.8` raw)。

**实现要点**：
- `05_iterative_pipeline.sh` Step 0 新增可选环境变量 `STATS_CLONE_DIRS`（留空 = source-only，即 05 默认行为；设置后额外 `--clone-dirs`）。完全向后兼容。
- `07_topup_pipeline.sh` 启动器：先跑 Step 0 统计 (source + clone-dirs) 测出 `gap_hours`，按 `TOTAL_CLONE_HOURS = ceil(gap_hours / SURVIVAL_EST)` 自动定预算 (默认 `SURVIVAL_EST=0.1` 保守；实测存活率更高可上调以省算力/磁盘)，再 `exec` `05_iterative_pipeline.sh` (它见 `STATS_CSV` 已存在会跳过重复统计)。手动设 `TOTAL_CLONE_HOURS` 则跳过自动测算；续跑模式 (`START_ROUND>1`/`START_STEP!=clone`) 跳过测算并复用已有分配基准。

```bash
# 编辑 07_topup_pipeline.env (源目录 / 已过滤 clone 目录 / CLONE_ROOT / 预算等), 然后:
tmux new-session -d -s higgs_v5 "bash v3_tts_clone/07_topup_pipeline.sh"
tail -f clone_workdir/iterative_pipeline_v5/pipeline_*.log

# 省算力: 按实测存活率把预算调高存活率 (少生成)
SURVIVAL_EST=0.4 bash v3_tts_clone/07_topup_pipeline.sh
# 或直接手动指定预算 (跳过自动测算)
TOTAL_CLONE_HOURS=12000 bash v3_tts_clone/07_topup_pipeline.sh
```

## 从中断轮续跑 (`08_resume_topup.sh`)

中断后不要手敲一长串变量。`08` 会 source `07_topup_pipeline.env`、清理残留 TTS、跑预检，再以续跑模式启动 `07→05`。

默认：**v5 topup 从第 2 轮 `clone` 接上**（`03` 对已有 wav+json 幂等跳过）。

```bash
# 预检 only
DRY_RUN=1 bash v3_tts_clone/08_resume_topup.sh

# 正式续跑 (tmux)
tmux new-session -d -s higgs_v5r2 "bash v3_tts_clone/08_resume_topup.sh"
tail -f clone_workdir/iterative_pipeline_v5/pipeline_*.log

# 覆盖步: 第2轮从 SIM 开始
START_ROUND=2 START_STEP=sim bash v3_tts_clone/08_resume_topup.sh
# 采用预检建议的 START_STEP
ADOPT_SUGGEST=1 bash v3_tts_clone/08_resume_topup.sh
```

---

## 评估提速 (CER)

- **`ASR_BACKEND=vllm`（默认，推荐）**：CER 用 vLLM 后端（`Qwen3ASRModel.LLM`，`tensor_parallel_size`=卡数），连续批处理把 GPU 吃满。旧的 `transformers` 后端用「单进程多线程」驱动多卡受 GIL 限制、`model.generate` 自回归解码利用率低。已验证 vLLM 与 transformers 转写结果逐条一致（CER 等价）。
- `ASR_VLLM_BATCH=256`（喂给 vLLM 引擎的批, 越大越能填满）、`ASR_GPU_MEM_UTIL=0.9`。
- 通用: `ASR_AUDIO_WORKERS=16`（音频解码并行, 与 GPU 重叠）、`ASR_MAX_NEW_TOKENS=512`（原 256 会截断长音频转写）。
- 语言预分组: transformers 用 `--group-by-language`（防子批变小）；**vllm 用 `--no-group-by-language`**（对混合语言不敏感, 省掉全量 json 预扫）。
- `ASR_BATCH_SIZE=32` 仅在 `ASR_BACKEND=transformers` 时用。
- **全盘扫描全部多进程加速（`SCAN_WORKERS=64`）**：`05_scan_existing_clones`（→7s）、CER 扫描（`list_clone_items` 无-meta, 79s→3s）、prune 扫描（0.5s）、SIM 扫描（20s, 需读 ref）。每轮多次全树扫描不再是瓶颈。
- vLLM 驱动用单独后台线程做 manual ITN, 与下一批 transcribe 重叠 (GPU 不空转)。

## ⚠️ 长文本被截断 → 过度剪枝 (排查结论)

TTS 音频受 `TTS_MAX_NEW_TOKENS`（SGLang audio-token 上限）限制: **1024 tokens ≈ 40.7s @ 25fps**。约 **31.7%** 的 clone 命中该上限, 长文本(>~250字)音频被从中间截断 → ASR 只听到前半 → CER 虚高(300+字桶删除率 25.6%) → 长clone被过度剪枝。默认保持 `1024`；需要长文本完整发声时上调 `TTS_MAX_NEW_TOKENS`（如 2048/3072, 每条长clone更耗算力）。此为 `03_tts_clone.py --max-new-tokens`。

---

## 续跑

**A. 无脑重跑（断点续跑）**：直接重跑整条流水线。每轮克隆前跳过已有文件，ASR 已有 `.json` sidecar 的也跳过，天然幂等。

**B. 从指定轮/指定步开始**（复用前面已跑出的结果）：用 `START_ROUND` + `START_STEP` 控制。

```bash
# 第 1 轮从 CER 评估开始 (复用已生成的 clone)
START_ROUND=1 START_STEP=cer bash v3_tts_clone/05_iterative_pipeline.sh

# 直接从第 3 轮的克隆开始
START_ROUND=3 bash v3_tts_clone/05_iterative_pipeline.sh

# 第 3 轮从 SIM 评估开始
START_ROUND=3 START_STEP=sim bash v3_tts_clone/05_iterative_pipeline.sh
```

- `START_STEP` 取值：`clone`(默认) < `sim` < `cer`。起始轮只跑 `>=` 该步的部分，之后的轮次都从 `clone` 完整跑。
- 一旦 `START_ROUND>1` 或 `START_STEP!=clone`，即进入**续跑模式**：跳过 ASR 转写与预算分配，复用已有的 `allocation/` 基准（`speaker_duration_stats_post_prune_resume.csv` + `original_clones_needed.json`）——若这两个文件缺失会直接报错，需先完整跑一次到分配完成。
- 复用来源：磁盘已有 clone（`03_tts_clone.py` 跳过已存在文件）、已有 `.cer.json` / `.sim.json`（评估 `--skip-existing`）。
