# Higgs Audio v3 TTS 声音复刻流水线

将总时长不足 1 小时（3600s）且音频 ≥20 条的说话人复刻到 1 小时水平。

## 流水线

```
Step 1 → speaker_duration_stats.csv
Step 2 → ASR 转写所有 <1h 说话人的全部音频 (1,045,442 文件)
Step 3 → SGLang-Omni 本地部署 TTS 声音复刻
```

## 环境

| Step | 环境 | 说明 |
|------|------|------|
| Step 1 | 系统 Python | `pip install librosa soundfile` |
| Step 2 | `qwen3-asr` conda env | Qwen3-ASR 1.7B + transformers 4.57.6 |
| Step 3 服务端 | `higgs_v3_env` conda env | SGLang-Omni，每 GPU 一个进程 |
| Step 3 客户端 | 系统 Python | 仅需 requests、soundfile、numpy |

## 被排除的数据集

`childmandarin`, `child207m-korean-filtered`, `chineseenglishchildren`, `king-asr-725`, `kingasr612`, `speechocean762`

## Step 1: 说话人时长统计

```bash
python v3_tts_clone/01_stats_speakers.py \
    --audio-root /root/group-shared/.../audio \
    --output-dir ./clone_workdir \
    --workers 8
```

→ `clone_workdir/speaker_duration_stats.csv`

## Step 2: 批量 ASR 转写

```bash
bash v3_tts_clone/02_asr_launch.sh "0,1"
```

→ `{音频路径}.json` （与音频同目录，含 transcript、language）

## Step 3: TTS 声音复刻

当前生产配置：

- 8 个 SGLang-Omni 服务：GPU `0-7` → 端口 `8000-8007`
- 客户端并发：`--workers-per-server 16`（总并发约 128）
- 文本选择：不按 dataset 或 speaker 语言过滤；所有说话人都从完整文本池随机抽样
- 请求模式：非流式 `/v1/audio/speech`，每条音频完整返回后写入 `clone_XXXX.wav`
- 后台运行：使用 `tmux` session `higgs_step3`

### SGLang-Omni 安装

`sgl-omni` 入口来自 `/root/code/github_repos/sglang-omni` 的 editable 安装：

```bash
/root/code/github_repos/higgs-audio/higgs_v3_env/bin/python3 \
    -m pip install -e /root/code/github_repos/sglang-omni --no-deps
```

不建议让 pip 重新解析依赖；当前环境依赖已固定，`--no-deps` 用于避免升级 torch/transformers 等大包。

### 手动运行

终端 1 — 启动 SGLang 服务（一直跑着）：

```bash
bash v3_tts_clone/03_launch_servers.sh "0,1,2,3,4,5,6,7" /root/models/higgs-audio-v3-tts-4b 8000
```

终端 2 — 运行克隆客户端（可 Ctrl+C 中断，重跑续传）：

```bash
python v3_tts_clone/03_tts_clone.py \
    --stats-csv ./clone_workdir/speaker_duration_stats.csv \
    --texts-jsonl higgs_audio_v3_text_generator/batch_output_v2/generated_texts_final.jsonl \
    --output-root /root/group-shared/.../audio_higgs_audio_v3_tts_clone \
    --base-port 8000 \
    --num-servers 8 \
    --workers-per-server 16
```

首次测试可用 `--max-speakers 5 --max-clones-per-speaker 1` 限制范围。

文本选择不按 dataset 或 speaker 的语言做偏好过滤；每个说话人都从完整文本池随机抽样，可混合中文、英文和中英混合文本。

### tmux 后台生产

当前生产使用 `tmux`，退出 Cursor 或断开终端后任务仍会继续：

```bash
tmux attach -t higgs_step3
```

窗口约定：

| Window | 用途 |
|--------|------|
| `servers` | 8 个 SGLang-Omni 服务 |
| `client` | 全量复刻客户端 |
| `progress` | 每分钟写进度 |

常用日志：

| 文件 | 说明 |
|------|------|
| `clone_workdir/step3_tmux_servers.log` | 服务端启动与运行日志 |
| `clone_workdir/step3_tmux_client.log` | 客户端运行日志 |
| `clone_workdir/step3_progress_tmux.log` | 每分钟进度统计 |

查看进度：

```bash
tail -f clone_workdir/step3_progress_tmux.log
```

进度行示例：

```text
2026-06-13T18:05:52 clone_wav=1179 clone_json=1180 ref_wav=128 speakers_with_clones=128
```

其中 `clone_wav` / `clone_json` 是已生成音频和元数据数量，`speakers_with_clones` 是已有 clone 输出的说话人数。

### 重跑与断点续跑

客户端会跳过已存在且有效的 `clone_XXXX.wav` + `clone_XXXX.json`，因此同一输出目录可断点续跑。

如果需要**完全重跑**，先把旧输出目录移动到备份目录，再用空的 `audio_higgs_audio_v3_tts_clone/` 启动客户端；不要直接删除旧结果，便于回滚和抽查。

## 输出目录结构

```
audio_higgs_audio_v3_tts_clone/
└── {dataset}/
    └── {speaker_id}/
        ├── ref/
        │   ├── ref_pool.json          # 候选池配置 + 全部候选元数据
        │   └── ref_{hash}.wav         # 按需物化的参考（random 模式多条）
        ├── clone_0000.wav             # 克隆音频
        ├── clone_0000.json            # 含本条实际 ref_audio_path
        └── ...
```

每条 clone 的 `clone_XXXX.json` 记录**当次使用的** `ref_audio_path`（指向 `ref/ref_{hash}.wav`），不再写统一的 `ref_audio.wav`。

## JSON 元数据

### 参考候选池 (`ref/ref_pool.json`)

```json
{"uid": "...", "ref_mode": "random", "ref_pool_size": 256,
 "candidates": [{"type": "concat", "duration_sec": 8.3, "source_files": [...]}]}
```

### 克隆音频 (`clone_NNNN.json`)
```json
{"clone_idx": 0, "uid": "...", "text": "...", "clean_text": "...",
 "ref_audio_path": ".../ref/ref_{hash}.wav", "ref_audio_type": "concat",
 "ref_transcript": "...", "ref_mode": "random", "audio_format": "wav"}
```

## Step 4: 质量过滤后重新评估时长

质量过滤（CER/SIM prune）后，用 **源音频时长 + 保留 clone 时长** 重新统计，找出仍不足 1h 的说话人：

```bash
python v3_tts_clone/04_post_prune_stats.py \
    --stats-csv ./clone_workdir/speaker_duration_stats.csv \
    --clone-root /root/group-shared/.../audio_higgs_audio_v3_tts_clone \
    --output-dir ./clone_workdir \
    --total-clone-hours 10000
```

输出：

| 文件 | 说明 |
|------|------|
| `speaker_duration_stats_post_prune.csv` | 全部说话人（含 OK / NEED_CLONE） |
| `speaker_duration_stats_post_prune_resume.csv` | 仅 NEED_CLONE，供 Step 3 续跑 |
| `post_prune_stats_summary.json` / `.txt` | 汇总报告 |

**分配策略**：在 NEED_CLONE 说话人之间，按各自 `gap_sec`（距 3600s 的缺口）占比，分配 `--total-clone-hours`（默认 **10000** 小时）的生成预算。每人 `clones_needed = max(1, int(allocated_sec / estimate_clone_duration) + 1)`，不再按「每人补满 1h + quality_pass_rate buffer」估算。

| 参数 | 默认 | 说明 |
|------|------|------|
| `--total-clone-hours` | `10000` | 全局 clone 生成小时预算（按缺口比例分配） |
| `--estimate-clone-duration` | `10` | 估算每条 clone 秒数，用于换算条数 |

### Step 3 续跑（post-prune）

```bash
python v3_tts_clone/03_tts_clone.py \
    --stats-csv ./clone_workdir/speaker_duration_stats_post_prune_resume.csv \
    --post-prune \
    --texts-jsonl higgs_audio_v3_text_generator/batch_output_v2/generated_texts_final.jsonl \
    --output-root /root/group-shared/.../audio_higgs_audio_v3_tts_clone \
    --base-port 8000 --num-servers 8 --workers-per-server 16
```

续跑会从已有 `clone_XXXX` 最大编号 +1 开始生成，不会覆盖已保留的 clone。

### 参考/文本随机性（`03_tts_clone.py`）

| 参数 | 默认 | 说明 |
|------|------|------|
| `--ref-mode` | `random` | `fixed` 每人固定一条 ref；`rotate` 每 N 条换 ref；`random` 每条 clone 独立 ref |
| `--ref-rotate-every` | `50` | `rotate` 模式下每 N 条 clone 换一次参考 |
| `--ref-pool-size` | `256` | 每说话人从合法单条/拼接组合中最多采样 256 个候选 ref |
| `--seed` | `42` | 全局 seed（ref 与文本选取可复现、可断点续跑） |

- **拼接参考**：在合法 7–20s 组合中随机/穷举采样，不再只取第一个命中组合  
- **文本**：每条 clone 独立随机抽取（`pick_text(uid, clone_idx, seed)`），不再一次 shuffle 顺序取  

## 参数速查

| 参数 | 默认值 | 当前生产值 | 说明 |
|------|--------|------------|------|
| `--base-port` | 8000 | 8000 | SGLang 起始端口 |
| `--num-servers` | 2 | 8 | SGLang 实例数（=GPU数）|
| `--workers-per-server` | 16 | 16 | 每服务器并发线程数 |
| `--total-clone-hours` | 10000 | 10000 | Step 4 全局 clone 预算（按缺口比例分配） |
| `--estimate-clone-duration` | 10 | 10 | 平均每条克隆音频秒数 |
| `--max-speakers` | None | None | 限制人数（测试用）|
| `--max-clones-per-speaker` | None | None | 限制每个说话人生成条数（测试用）|
