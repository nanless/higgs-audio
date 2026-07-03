# Higgs Audio v3 — 迭代克隆流水线

## 概述

一键运行：ASR 转写 → 预算分配 → 10 轮迭代（每轮：启动 TTS → 克隆 1/10 → 停止 TTS → CER → SIM → 剪枝）。

## 脚本

```
00_prepare_stats.py            生成全量说话人统计 (手动运行一次)
02_asr_launch.sh / 02_asr_worker.py  ASR 转写 (被 05 调用)
03_launch_servers.sh           SGLang TTS 启动 (被 05 调用, 每轮启停)
03_tts_clone.py                TTS 克隆 (被 05 调用)
04_post_prune_stats.py         预算分配 (被 05 调用)
05_scan_existing_clones.py     扫描已有 clone 数
05_generate_round_csv.py       每轮生成受限 CSV
05_save_orig_allocation.py     保存分配基准
05_iterative_pipeline.sh       主流水线 (包含 ASR + TTS + 10 轮迭代)
05_iterative_pipeline.env      配置
```

---

## 准备

### 1. 生成统计 (只跑一次)

```bash
python3 v3_tts_clone/00_prepare_stats.py \
    --source-dirs /root/group-shared/.../audio \
    --clone-dirs \
        /root/group-shared/.../audio_higgs_audio_v3_tts_clone \
        /root/group-shared/.../audio_higgs_audio_v3_tts_clone_2 \
        /root/group-shared/.../audio_omnivoice_clone \
    --output-dir clone_workdir/stats_output \
    --workers 32
```

- `--source-dirs`：原始音频。`speaker_path` 取自此。
- `--clone-dirs`：复刻音频。时长计入 `clone_duration_sec`，不用于 `speaker_path`。
- **输出文件名固定为 `{output-dir}/all_speakers.csv`**。下一步的 `STATS_CSV` 必须指向该文件（示例里的 `speaker_duration_stats_v2.csv` 是重命名后的路径——按需改名，或直接把 `STATS_CSV` 指向 `all_speakers.csv`）。

### 2. 编辑配置

`v3_tts_clone/05_iterative_pipeline.env`：

```bash
REPO="/root/code/github_repos/higgs-audio"
BASE="/root/group-shared/..."

# STATS_CSV = 上一步 00 生成的 all_speakers.csv (或其重命名)
export STATS_CSV="${REPO}/clone_workdir/speaker_duration_stats_v2.csv"
# TEXTS_JSONL = 克隆用的文本池 (每行一条 JSON, 含 text/clean_text)
export TEXTS_JSONL="${REPO}/higgs_audio_v3_text_generator/batch_output_v2/generated_texts_final.jsonl"
export SOURCE_DIRS="${BASE}/audio"
export CLONE_ROOT="${BASE}/audio_higgs_audio_v3_tts_clone_3"
export TOTAL_CLONE_HOURS=10000
export TOTAL_ROUNDS=10
export NUM_SERVERS=4
export ALL_GPUS="0,1,2,3"
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
Step 0   检查 STATS_CSV, 合并参考音频目录
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
  ├ SIM 剪枝 (只按 SIM<0.85 删一波, --max-cer 999 关闭 CER 判定; 缩小 CER 待评集)
  ├ CER 评估 (SIM 存活集; --skip-existing --refresh-scan, batch/audio-workers/语言分组) → pkill eval_cer.py 释放 GPU
  ├ CER 剪枝 (CER>0.03 或 SIM<0.85, 读 per-clone sidecar)
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

- `START_STEP` 取值：`clone`(默认) < `cer` < `sim` < `prune`。起始轮只跑 `>=` 该步的部分，之后的轮次都从 `clone` 完整跑。
- 一旦 `START_ROUND>1` 或 `START_STEP!=clone`，即进入**续跑模式**：跳过 ASR 转写与预算分配，复用已有的 `allocation/` 基准（`speaker_duration_stats_post_prune_resume.csv` + `original_clones_needed.json`）——若这两个文件缺失会直接报错，需先完整跑一次到分配完成。
- 复用来源：磁盘已有 clone（`03_tts_clone.py` 跳过已存在文件）、已有 `.cer.json` / `.sim.json`（评估 `--skip-existing`）。
