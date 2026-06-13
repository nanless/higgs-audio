# Higgs Audio v3 TTS Clone — 评估流水线

从 OmniVoice `batch_generate_text_and_clone/` 的 eval 代码适配而来，
用于评估 Higgs Audio v3 TTS 的 clone 音频质量（CER + SIM + MOS）。

## 目录结构

```
eval_higgs_audio/
├── README.md                           # 本文档
├── __init__.py                         # 空文件
├── eval_common.py                      # 共享工具（扫描、分片、累加器、I/O）
├── run_eval_all.sh                     # 一键评估总控（CER → SIM → MOS）
├── logs/                               # 运行日志
│
├── eval_cer/                           # 字错率评估
│   ├── eval_cer.py                     # CER 评估主脚本（ASR + Manual ITN）
│   ├── .env                            # LLM ITN 配置（可选）
│   └── run_eval_cer.sh                 # 启动脚本
│
├── eval_sim/                           # 说话人相似度评估
│   ├── eval_sim.py                     # SIM 评估主脚本（SamResNet100 余弦相似度）
│   ├── speaker_encoder.py              # Speaker encoder（fbank + SamResNet100ASP）
│   ├── speaker_similarity.py           # 相似度计算工具
│   ├── models/samresnet.py             # SimAM_ResNet100 + ASP 模型定义
│   └── run_eval_sim.sh                 # 启动脚本
│
└── eval_mos/                           # 音质评估
    ├── eval_mos.py                     # MOS 评估主脚本（多指标，多进程）
    ├── scorers.py                      # 4 种评分的封装（UTMOS22Strong/SCOREQ/TTSDS2/UTMOSv2）
    └── run_eval_mos.sh                 # 启动脚本
```

## 与 OmniVoice 源版本的关键差异

| 维度 | OmniVoice | Higgs Audio (本适配版) |
|------|-----------|------------------------|
| **音频格式** | 16 kHz mono WAV | **24 kHz** mono WAV（评估时内部 resample 到 16 kHz） |
| **文件命名** | `text_NNN.wav` / `text_NNN.json` | `clone_NNNN.wav` / `clone_NNNN.json` |
| **目录结构** | `{dataset}/{speaker}/{utt_id}/text_NNN.wav` | `{dataset}/{speaker_id}/clone_NNNN.wav` |
| **参考文本** | `gen_text` 字段 | `clean_text` 字段（fallback `text`） |
| **标签格式** | `[sigh]`, `[laughter]` 方括号 | `<\|emotion:joy\|>`, `<\|prosody:speed_slow\|>` 管道格式 |
| **ITN 标签匹配** | 有专门的 `[sigh]→哎` 映射 | 直接 strip `<\|...\|>` 标签 |
| **LLM ITN** | 支持（多 vLLM endpoint） | **暂不支持**（仅 Manual ITN） |
| **参考音频** | sidecar 中的 `ref_audio` 路径 | 同目录下的 `ref_audio.wav` |

## 依赖环境

| 评估步骤 | Conda 环境 | 额外依赖 |
|----------|-----------|----------|
| **CER** | `qwen3-asr` | Qwen3-ASR 1.7B、`jiwer`, `soundfile`, `torchaudio`, `word2number` |
| **SIM** | `omnivoice` | `torch`, `torchaudio`, `yaml`、SamResNet100 权重（来自 OmniVoice） |
| **MOS** | `omnivoice` | `scoreq`, `ttsds`, `utmosv2`、OmniVoice 源码 + UTMOS22Strong 模型 |

## 快速开始

### 1. CER 评估

```bash
cd eval_higgs_audio/eval_cer

# 评估所有 clone（需要 conda qwen3-asr 环境）
bash run_eval_cer.sh --out-dir /path/to/audio_higgs_audio_v3_tts_clone

# 评估 500 个随机样本
bash run_eval_cer.sh --sample-size 500 --seed 42

# 使用缓存的 ASR 结果（跳过 ASR 推理）
bash run_eval_cer.sh --skip-asr --sample-size 500

# 跳过已评估的（已有 .eval.json 的跳过）
bash run_eval_cer.sh --skip-existing

# 直接调用 Python
python eval_cer.py --out-dir /path/to/clone_output --sample-size 200 --gpu 0
```

### 2. SIM 评估（说话人相似度）

```bash
cd eval_higgs_audio/eval_sim

# 评估所有 pair（需要 conda omnivoice 环境）
bash run_eval_sim.sh --out-dir /path/to/audio_higgs_audio_v3_tts_clone

# 采样 200 条，4 进程并行
bash run_eval_sim.sh --sample-size 200 --workers 4 --gpu 0

# 跳过已评估的（已有 .sim.json 的跳过）
bash run_eval_sim.sh --skip-existing
```

### 3. MOS 评估

```bash
cd eval_higgs_audio/eval_mos

# 评估所有 clone（需要 conda omnivoice 环境）
bash run_eval_mos.sh --out-dir /path/to/audio_higgs_audio_v3_tts_clone

# 仅 UTMOS22Strong，2 进程并行
bash run_eval_mos.sh --metrics UTMOS22Strong --workers 2 --gpus 0,1

# 采样 200 条
bash run_eval_mos.sh --sample-size 200 --metrics UTMOS22Strong,SCOREQ --skip-existing
```

### 3. 一键全流程

```bash
# CER + MOS 全流程（所有 clone）
bash eval_higgs_audio/run_eval_all.sh

# 仅 CER
SKIP_MOS=1 bash eval_higgs_audio/run_eval_all.sh --sample-size 500

# 仅 MOS
SKIP_CER=1 bash eval_higgs_audio/run_eval_all.sh --sample-size 1000

# 自定义 clone 目录
HIGGS_CLONE_ROOT=/your/path bash eval_higgs_audio/run_eval_all.sh --sample-size 500
```

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `HIGGS_CLONE_ROOT` | 生产路径 | Clone 音频输出根目录 |
| `CUDA_VISIBLE_DEVICES` | `0` | 使用的 GPU |
| `HIGGS_CER_GPU` | `0` | CER ASR 使用的 GPU |
| `HIGGS_MOS_GPUS` | `0` | MOS 使用的 GPU（逗号分隔） |
| `HIGGS_MOS_WORKERS` | `1` | MOS 并行 worker 数 |
| `HIGGS_ASR_BATCH_SIZE` | `16` | ASR 批大小 |
| `HIGGS_EVAL_SAMPLE_SIZE` | 全部 | 随机采样数量 |

## 输出格式

### CER 输出（每个 clone）

```json
// clone_NNNN.eval.json（与 clone_NNNN.json 同目录）
{
  "wav_path": ".../clone_0000.wav",
  "gen_text": "Well, that package...",
  "asr_hypo": "well that package...",
  "ref_manual": "well that package...",
  "hypo_manual": "well that package...",
  "manual_cer": 0.0,
  "substitutions": 0,
  "insertions": 0,
  "deletions": 0,
  "chars": 19,
  "evaluated_at": "2026-06-14T12:00:00",
  "dataset": "libritts",
  "speaker_id": "libritts_636"
}
```

### SIM 输出（每个 clone）

```json
// clone_NNNN.sim.json（与 clone_NNNN.json 同目录）
{
  "cloned_audio": ".../clone_0000.wav",
  "ref_audio": ".../ref_audio.wav",
  "similarity": 0.8472,
  "dataset": "libritts",
  "speaker_id": "libritts_636",
  "evaluated_at": "2026-06-14T12:00:00"
}
```

### MOS 输出（每个 clone）

```json
// clone_NNNN.mos.json（与 clone_NNNN.json 同目录）
{
  "cloned_audio": ".../clone_0000.wav",
  "dataset": "libritts",
  "utmos22strong": 3.85,
  "evaluated_at": "2026-06-14T12:00:00"
}
```

### 汇总文件（在 clone 根目录）

- `eval_higgs_cer_summary.json` — CER 汇总（含 per-dataset 分解）
- `eval_higgs_cer_details.jsonl` — CER 逐条明细
- `eval_higgs_sim_summary.json` — SIM 汇总（含 per-dataset 分解）
- `eval_higgs_sim_details.jsonl` — SIM 逐条明细
- `eval_higgs_mos_summary.json` — MOS 汇总（含 per-dataset 分解）
- `eval_higgs_mos_details.jsonl` — MOS 逐条明细

## CER 评估流程详解

```
1. 扫描 clone 目录 → 找到所有 clone_NNNN.wav + clone_NNNN.json 配对
2. Qwen3-ASR 转写 wav（batch_size=16，24 kHz → 16 kHz 内部重采样）
3. Manual ITN：
   a. 去除 Higgs 标签（<|emotion:X|>, <|prosody:X|>, <|sfx:X|>, <|style:X|>）
   b. 数字归一化（中文数词→阿拉伯、英文单词→数字、百分数等）
   c. 去除标点符号（保留数字间的小数点）
   d. 全小写 + 空格规范化
4. jiwer.process_characters() 计算字符级 CER
5. 结果写入 clone_NNNN.eval.json 和 eval_higgs_cer_details.jsonl
```

## SIM 评估流程详解

```
1. 扫描 clone 目录 → 找到所有 (clone_NNNN.wav, ref_audio.wav, clone_NNNN.json) 三元素
2. SamResNet100ASP 提取 clone 和 ref 语音的 256 维 speaker embedding
3. 计算余弦相似度 → 映射到 [0, 1] 范围
4. 结果写入 clone_NNNN.sim.json 和 eval_higgs_sim_details.jsonl
```

## 4 种 MOS 指标

| 指标 | 来源 | 范围 | 类型 |
|------|------|------|------|
| **UTMOS22Strong** | OmniVoice 内置模型 | 1-5 | 参考感知质量（需 OmniVoice 源码 + checkpoint） |
| **SCOREQ** | pip `scoreq` | NR | 无参考质量（ONNX 推理） |
| **TTSDS2** | pip `ttsds` | 0-100 | TTS 综合评分（WavLM + Whisper + pitch） |
| **UTMOSv2** | git `utmosv2` | 1-5 | SSL+spectrogram MOS 预测 |

## UTMOS22Strong 模型准备

UTMOS22Strong 需要 OmniVoice 仓库源码和模型权重：

1. OmniVoice 源码路径（硬编码）：`/root/code/github_repos/OmniVoice-fork`
2. 模型权重：需要在 `TTS_eval_models/mos/utmos22_strong_step7459_v1.pt` 或 `~/.cache/`
3. 下载命令：
```bash
huggingface-cli download --local-dir /root/code/github_repos/OmniVoice-fork/TTS_eval_models \
    k2-fsa/TTS_eval_models mos/utmos22_strong_step7459_v1.pt
```
