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

终端 1 — 启动 SGLang 服务（一直跑着）：

```bash
bash v3_tts_clone/03_launch_servers.sh "0,1" /root/models/higgs-audio-v3-tts-4b 8000
```

终端 2 — 运行克隆客户端（可 Ctrl+C 中断，重跑续传）：

```bash
python v3_tts_clone/03_tts_clone.py \
    --stats-csv ./clone_workdir/speaker_duration_stats.csv \
    --texts-jsonl higgs_audio_v3_text_generator/batch_output_v2/generated_texts_final.jsonl \
    --output-root /root/group-shared/.../audio_higgs_audio_v3_tts_clone \
    --base-port 8000 \
    --num-servers 2 \
    --workers-per-server 8
```

首次测试可用 `--max-speakers 5 --max-clones-per-speaker 1` 限制范围。

文本选择不按 dataset 或 speaker 的语言做偏好过滤；每个说话人都从完整文本池随机抽样，可混合中文、英文和中英混合文本。

## 输出目录结构

```
audio_higgs_audio_v3_tts_clone/
└── {dataset}/
    └── {speaker_id}/
        ├── ref_audio.wav          # 参考音频
        ├── ref_audio.json         # 参考音频元数据
        ├── clone_0000.wav         # 克隆音频
        ├── clone_0000.json        # 克隆元数据
        └── ...
```

## JSON 元数据

### 参考音频 (`ref_audio.json`)
```json
{"uid": "...", "ref_audio_path": "...", "ref_before_duration_sec": 12.5,
 "ref_audio_type": "single|concat", "ref_transcript": "...",
 "source_files": [...], "num_concat_clips": 1}
```

### 克隆音频 (`clone_NNNN.json`)
```json
{"clone_idx": 0, "uid": "...", "text": "...", "clean_text": "...",
 "emotion": "...", "scenario": "...", "tags_used": [...],
 "ref_audio_source": "...", "ref_transcript": "...",
 "ref_audio_duration_sec": 12.5, "audio_format": "wav"}
```

## 参数速查

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--base-port` | 8000 | SGLang 起始端口 |
| `--num-servers` | 2 | SGLang 实例数（=GPU数）|
| `--workers-per-server` | 8 | 每服务器并发线程数 |
| `--quality-pass-rate` | 0.5 | 质量通过率（决定 2x buffer）|
| `--estimate-clone-duration` | 10 | 平均每条克隆音频秒数 |
| `--max-speakers` | None | 限制人数（测试用）|
| `--max-clones-per-speaker` | None | 限制每个说话人生成条数（测试用）|
