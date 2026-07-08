<div align="center">

# Higgs Audio

**富有表现力的音频生成基础模型 · Redefining Expressiveness in Audio Generation**

<a href="https://boson.ai/blog/higgs-audio-v2"><img src='https://img.shields.io/badge/🚀-V2%20Blogpost-228B22'></a>
<a href="https://www.boson.ai/blog/higgs-audio-v2.5"><img src='https://img.shields.io/badge/🚀-V2.5%20Blogpost-228B22'></a>
<a href="https://huggingface.co/bosonai/higgs-audio-v3-tts-4b"><img src="https://img.shields.io/badge/🤗-v3%20Weights-ED5A22.svg"></a>
<a href="https://docs.boson.ai/models/higgs-audio-tts/overview"><img src="https://img.shields.io/badge/📖-v3%20API%20Docs-9C276A"></a>
<a href="https://boson.ai/demo/tts"><img src="https://img.shields.io/badge/🕹️-Playground-8A2BE2"></a>

</div>

---

## 这个仓库是什么

一句话：**本仓库是 Higgs Audio v2 / v2.5 的模型与推理代码库，并在其上生长出一整套面向 Higgs Audio v3 的生产工具链**（造训练数据、大规模声音复刻、质量评估）。

> [!IMPORTANT]
> **Higgs Audio v3 是独立发布，不依赖本仓库代码。** 若你只想用最新的 v3 模型，直接用 [Hugging Face 权重](https://huggingface.co/bosonai/higgs-audio-v3-tts-4b) 或 [Boson AI 云端 API](https://docs.boson.ai/models/higgs-audio-tts/overview)（见 [第 2 节](#2-使用-higgs-audio-v3)）。本仓库中与 v3 相关的部分（`v3_tts_clone/`、`eval_higgs_audio/`、`higgs_audio_v3_text_generator/`）是**围绕 v3 的数据/复刻/评估工程**，不是模型本体。

安装出来的 Python 包名是 **`boson_multimodal`**（不是 `higgs_audio`）。

---

## 目录

1. [仓库全景与快速导航](#1-仓库全景与快速导航)
2. [使用 Higgs Audio v3](#2-使用-higgs-audio-v3)（API / SGLang-Omni / HF 权重）
   - [v3 模型架构](#21-v3-模型架构)
   - [方式一：Boson AI 云端 API](#22-方式一boson-ai-云端-api无需-gpu)
   - [方式二：SGLang-Omni 本地部署](#23-方式二sglang-omni-本地部署高吞吐)
   - [完整标签参考](#24-完整标签参考emotion--style--sfx--prosody)
   - [使用规则与限制](#25-核心使用规则与限制)
   - [v3 评估基准](#26-v3-评估基准)
3. [使用 Higgs Audio v2 / v2.5](#3-使用-higgs-audio-v2--v25)（本仓库代码）
   - [安装](#31-安装)
   - [快速上手](#32-快速上手)
   - [命令行示例](#33-命令行示例examplesgenerationpy)
   - [vLLM 高吞吐部署](#34-vllm-高吞吐部署)
   - [技术细节与评估](#35-技术细节与评估)
4. [核心包架构 `boson_multimodal/`](#4-核心包架构-boson_multimodal)
5. [仓库内 v3 工具链](#5-仓库内-v3-工具链)
   - [v3 文本数据生成](#51-v3-文本数据生成-higgs_audio_v3_text_generator)
   - [v3 声音复刻流水线](#52-v3-声音复刻流水线-v3_tts_clone)
   - [评估流水线](#53-评估流水线-eval_higgs_audio)
   - [童声批量复刻](#54-童声批量复刻v2--v3)
6. [开发与贡献规范](#6-开发与贡献规范)
7. [许可证、引用与链接](#7-许可证引用与链接)

---

## 1. 仓库全景与快速导航

```
higgs-audio/
├── boson_multimodal/              # ① v2/v2.5 核心多模态模型 + 推理引擎（安装为 Python 包）
├── examples/                      # v2 推理示例：CLI、ServeEngine、vLLM、17 个预置音色
│
├── higgs_audio_v3_text_generator/   # ② v3 训练用文本数据批量生成（独立子项目，走外部 vLLM）
├── v3_tts_clone/                  # ③ v3 大规模声音复刻流水线（SGLang-Omni 后端）
├── eval_higgs_audio/              # ④ 复刻质量评估：CER / SIM / MOS
│
├── batch_child_voice_clone_higgs.py / batch_clone_v3.py   # 童声复刻工具（v2 / v3）
├── tech_blogs/                    # DualFFN 架构 & 25fps tokenizer 技术博客
├── figures/                       # README/博客配图
│
├── setup.cfg / setup.py / pyproject.toml / requirements.txt
├── .github/workflows/test.yml     # CI：仅 `ruff format --check .`
└── README.md                      # 本文件
```

按你的目标选择入口：

| 我想…… | 去这里 |
|--------|--------|
| 直接用最新 v3 模型（不写代码 / 不要 GPU） | [§2.2 云端 API](#22-方式一boson-ai-云端-api无需-gpu) |
| 本地自托管 v3 高吞吐服务 | [§2.3 SGLang-Omni](#23-方式二sglang-omni-本地部署高吞吐) |
| 用本仓库跑 v2/v2.5 推理 | [§3 v2/v2.5](#3-使用-higgs-audio-v2--v25) |
| 理解模型内部实现 | [§4 核心包架构](#4-核心包架构-boson_multimodal) |
| 为 v3 训练造文本数据 | [§5.1 文本生成](#51-v3-文本数据生成-higgs_audio_v3_text_generator) |
| 大规模复刻说话人补齐时长 | [§5.2 复刻流水线](#52-v3-声音复刻流水线-v3_tts_clone) |
| 评估复刻音频质量 | [§5.3 评估流水线](#53-评估流水线-eval_higgs_audio) |

---

## 2. 使用 Higgs Audio v3

Higgs Audio v3 TTS 是 Boson AI 的 4B 参数自回归 TTS 模型，输出 **24 kHz** 语音，支持 **100+ 种语言**、零样本声音克隆，以及通过内联标签精细控制情绪 / 风格 / 音效 / 韵律。

| 方式 | 适用场景 | 硬件 | 延迟 | 特点 |
|------|----------|------|------|------|
| **Boson AI API** | 快速集成、原型 | 无需 GPU | 低 | 免费公测，直接 curl |
| **SGLang-Omni 本地** | 生产部署、高吞吐 | 1× H100(80GB) | 极低 | CUDA Graph 加速、批量 |
| **HuggingFace 权重** | 研究、自定义推理 | GPU ≥24GB | 中 | 完全自控 |

### 2.1 v3 模型架构

```
Backbone : 4B 参数自回归解码器（36 层 Transformer，hidden 2560，GQA 32 头 / 8 KV 头，8192 上下文）
Audio    : Higgs Tokenizer（8 层 codebook × 1026 token，25 fps / 40ms 帧，24 kHz，delay pattern）
Input    : 交错式 text + audio token
Output   : 24 kHz 单声道语音
```

- 文本与音频 token 交错消费，8 层 codebook 通过 delay pattern 交错排列
- 多 codebook 融合 embedding 映射到 backbone 隐藏维度；输出经融合 head，de-delay 后解码为波形
- 多轮生成交错 `<|text|>...<|audio|>...` 块，每块基于参考与先前块

### 2.2 方式一：Boson AI 云端 API（无需 GPU）

在 [boson.ai/workspace](https://boson.ai/workspace) 获取免费 API Key（公测限流），OpenAI 兼容。

```bash
export BOSON_API_KEY=bai-xxxx

curl https://api.boson.ai/v1/audio/speech \
  -H "Authorization: Bearer $BOSON_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "higgs-audio-v3-tts", "input": "你好，这是一个测试。"}' \
  --output out.mp3
```

**预设声音**（`voice` 参数）：

| Voice | 风格 | Voice | 风格 |
|-------|------|-------|------|
| `chloe` | 友善清晰美式女声 | `marcus` | 热情自信、教授风男声 |
| `eleanor` | 冷静专业美式女声 | `nora` | 叙事风美式女声 |
| `jake` | 活力戏剧性男声 | `oliver` | 深思熟虑美式男声 |

**零样本声音克隆**（提供参考音频 + 逐字稿）：

```bash
curl https://api.boson.ai/v1/audio/speech \
  -H "Authorization: Bearer $BOSON_API_KEY" -H "Content-Type: application/json" \
  -d '{
    "model": "higgs-audio-v3-tts",
    "input": "你好，欢迎来到我们的团队！",
    "ref_audio": "https://example.com/your-voice-sample.mp3",
    "ref_text": "这是我的声音样本，用于声音克隆测试。"
  }' --output out.mp3
```

**自定义声音（可复用）**：先 `POST /v1/voices` 创建拿到 `voice_id`，之后用 `"voice": "voice_abc123..."` 复用，无需每次上传参考。

**流式传输**：加 `"stream": true`，**强制 `response_format: "pcm"`**（16-bit / 24kHz / mono 裸 PCM）。curl 加 `-N` 禁用缓冲。

**API 完整参数**：

```jsonc
{
  "model": "higgs-audio-v3-tts",  // 必需，固定
  "input": "要合成的文本。",        // 必需，1-5000 字符，可含内联标签
  "voice": "default",             // 预设声音名 或 自定义 voice_id
  "response_format": "mp3",       // mp3/wav/opus/flac/aac/pcm
  "stream": false,                // 流式（仅 pcm）
  "ref_audio": "URL | base64",    // 参考音频（克隆用，base64 ≤10MB）
  "ref_text": "参考音频的逐字稿。",  // 建议提供
  "temperature": 1.0,
  "top_p": null, "top_k": null,
  "seed": null                    // 复现用
}
```

完整参考：[docs.boson.ai/api-reference/text-to-speech/create-speech](https://docs.boson.ai/api-reference/text-to-speech/create-speech)。语言**自动检测**，共 102 种（85 种生产级 WER/CER<5），完整列表见 [languages 文档](https://docs.boson.ai/models/higgs-audio-tts/languages)。

### 2.3 方式二：SGLang-Omni 本地部署（高吞吐）

权重：[bosonai/higgs-audio-v3-tts-4b](https://huggingface.co/bosonai/higgs-audio-v3-tts-4b)，推荐用 [SGLang-Omni](https://github.com/sgl-project/sglang-omni) 服务。

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxx
hf download bosonai/higgs-audio-v3-tts-4b

sgl-omni serve --model-path bosonai/higgs-audio-v3-tts-4b --port 8000
```

**基础生成 / 声音克隆**：

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Have a nice day and enjoy the sunshine!",
    "references": [{"audio_path": "/path/to/voice.wav", "text": "Reference transcript here..."}],
    "temperature": 0.8, "top_k": 50, "max_new_tokens": 1024
  }' --output output.wav
```

**流式 + 裸 PCM（最低延迟）**：加 `"stream": true, "stream_format": "audio", "response_format": "pcm", "initial_codec_chunk_frames": 1`，返回裸 `audio/pcm` 字节流（无 SSE JSON 包装）。默认流式为 SSE：`data: {"object":"audio.speech.chunk","audio":{"data":"<base64 WAV>",...}}` 逐块，末尾 `data: [DONE]`。

**性能基准**（1× H100，Seed-TTS EN N=1088，bf16，CUDA Graph）：

| 并发 | 吞吐 req/s | 平均延迟 | RTF | 音频 s/s |
|------|-----------|---------|-----|----------|
| 1  | 1.62  | 617 ms  | 0.147 | 6.89 |
| 4  | 5.45  | 733 ms  | 0.177 | 22.84 |
| 8  | 8.91  | 898 ms  | 0.217 | 37.38 |
| 16 | 14.74 | 1079 ms | 0.262 | 61.84 |

> 若要在本仓库内**批量复刻低时长说话人**（统计 → ASR → 多卡 SGLang-Omni TTS clone → 评估剪枝），见 [§5.2](#52-v3-声音复刻流水线-v3_tts_clone) 与 [`v3_tts_clone/README.md`](v3_tts_clone/README.md)。

### 2.4 完整标签参考（emotion / style / sfx / prosody）

所有标签格式：`<|category:value|>`。示例：

```
<|emotion:enthusiasm|><|prosody:expressive_high|>欢迎来到节目！<|prosody:long_pause|>我们开始吧！
```

**情绪（21 种）**：`elation` 兴高采烈 · `amusement` 逗乐 · `enthusiasm` 热情 · `determination` 坚定 · `pride` 自豪 · `contentment` 满足 · `affection` 亲昵 · `relief` 松口气 · `awe` 敬畏 · `longing` 渴望 · `contemplation` 沉思 · `confusion` 困惑 · `surprise` 惊讶 · `arousal` 激动 · `anger` 愤怒 · `fear` 恐惧 · `disgust` 厌恶 · `bitterness` 苦涩 · `sadness` 悲伤 · `shame` 羞耻 · `helplessness` 无助。用法：`<|emotion:sadness|>...`

**风格（3 种）**：`<|style:singing|>` 唱歌（需含歌词）· `<|style:shouting|>` 喊叫（内容宜全大写）· `<|style:whispering|>` 耳语。

**音效（9 种，需紧跟拟声词）**：

| Tag | 拟声词 | Tag | 拟声词 | Tag | 拟声词 |
|-----|--------|-----|--------|-----|--------|
| `<\|sfx:cough\|>` | Ahem | `<\|sfx:laughter\|>` | Haha | `<\|sfx:crying\|>` | Boohoo |
| `<\|sfx:screaming\|>` | Ahh | `<\|sfx:burping\|>` | Burp | `<\|sfx:humming\|>` | Hmm |
| `<\|sfx:sigh\|>` | Uh | `<\|sfx:sniff\|>` | Sff | `<\|sfx:sneeze\|>` | Achoo |

> 音效在**说话人声音中生成**，不是混入的外部音轨；除 `sniff` 外每个 SFX 标签后须在 ~10 字符内紧跟拟声词。

**韵律（10 种）**：`speed_very_slow`(~0.65×) / `speed_slow`(~0.85×) / `speed_fast`(~1.2×) / `speed_very_fast`(~1.4×) · `pitch_low`(~−3 半音) / `pitch_high`(~+2.5 半音) · `pause`(~400–700ms) / `long_pause`(~700–1500ms) · `expressive_high` / `expressive_low`。

**互斥约束**：speed 之间互斥、pitch 之间互斥、`shouting`↔`whispering` 互斥、`laughter`↔`crying` 互斥、`pause`↔`long_pause`、`expressive_high`↔`expressive_low`。

### 2.5 核心使用规则与限制

1. **交付类标签放开头**：emotion、style、prosody 的 speed/pitch/expressive 决定整个 turn 的交付方式，**必须放在 `input` 最开头、任何文本之前**。
   - ✅ `<|emotion:enthusiasm|>欢迎！`   ❌ `欢迎！<|emotion:enthusiasm|>`
2. **音效配拟声词**：每个 `<|sfx:...|>` 后紧贴拟声词。 ✅ `<|sfx:laughter|>Haha，太好笑了！`
3. **位置型标签是例外**：`<|prosody:pause|>` / `long_pause` 与各 `<|sfx:...|>` 放在触发位置，而非开头。

| 限制 | 值 | 限制 | 值 |
|------|-----|------|-----|
| `input` 最大 | 5000 字符 | 采样率 | 24 kHz（固定） |
| `ref_audio` 内联 | ≤10MB | 声道 / 位深 | 单声道 / 16-bit |
| 流式格式 | 必须 `pcm` | 音频格式 | mp3/wav/opus/flac/aac/pcm |

**声音克隆注意**：参考音频 5–30s、干净无 BGM/多说话人；`ref_text` 须为精确逐字稿（含语气词）；输入与参考语言匹配最佳。⚠️ **必须拥有被克隆声音的合法权益。**

### 2.6 v3 评估基准

**多语种声音克隆**：Seed-TTS(2 语) 1.11 · CV3(9) 4.41 · MiniMax-Multilingual(23) 2.74 · Higgs-Multilingual(111) 3.61（WER/CER↓）。

**Emergent TTS 胜率**（vs 基线）：Overall 53.65% · Emotions 53.75% · Paralinguistics 68.57% · Questions 61.43% · Syntactic Complexity 60.71%。

---

## 3. 使用 Higgs Audio v2 / v2.5

> [!NOTE]
> 以下是**本仓库承载的 v2 / v2.5 模型**的用法。v2 在 1000 万+ 小时音频上预训练，无需后训练即擅长富有表现力的音频生成；v2.5 通过 GRPO 对齐将架构压缩到 1B 参数并超越 3B 版本速度与准确度（见 [v2.5 博客](https://www.boson.ai/blog/higgs-audio-v2.5)）。

<p align="center"><img src="figures/emergent-tts-emotions-win-rate.png" width="820"></p>

### 3.1 安装

推荐 NVIDIA 深度学习容器（已验证 `nvcr.io/nvidia/pytorch:25.02-py3` 或 `25.01-py3`）：

```bash
docker run --gpus all --ipc=host --net=host --ulimit memlock=-1 \
  --ulimit stack=67108864 -it --rm nvcr.io/nvidia/pytorch:25.02-py3 bash
```

```bash
git clone https://github.com/boson-ai/higgs-audio.git && cd higgs-audio
pip install -r requirements.txt
pip install -e .
```

也支持 venv / conda(python 3.10) / uv 安装。GPU 推理建议 **≥24GB 显存**。

### 3.2 快速上手

最小示例（另见根目录 [`quick_start.py`](quick_start.py)）：

```python
from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine, HiggsAudioResponse
from boson_multimodal.data_types import ChatMLSample, Message
import torch, torchaudio

MODEL_PATH = "bosonai/higgs-audio-v2-generation-3B-base"
AUDIO_TOKENIZER_PATH = "bosonai/higgs-audio-v2-tokenizer"

system_prompt = (
    "Generate audio following instruction.\n\n"
    "<|scene_desc_start|>\nAudio is recorded from a quiet room.\n<|scene_desc_end|>"
)
messages = [
    Message(role="system", content=system_prompt),
    Message(role="user", content="The sun rises in the east and sets in the west."),
]

engine = HiggsAudioServeEngine(MODEL_PATH, AUDIO_TOKENIZER_PATH, device="cuda")
output: HiggsAudioResponse = engine.generate(
    chat_ml_sample=ChatMLSample(messages=messages),
    max_new_tokens=1024, temperature=0.3, top_p=0.95, top_k=50,
    stop_strings=["<|end_of_text|>", "<|eot_id|>"],
)
torchaudio.save("output.wav", torch.from_numpy(output.audio)[None, :], output.sampling_rate)
```

### 3.3 命令行示例（`examples/generation.py`）

```bash
# 零样本声音克隆（参考音色见 examples/voice_prompts/，可自行添加）
python3 examples/generation.py \
  --transcript "The sun rises in the east and sets in the west." \
  --ref_audio belinda --temperature 0.3 --out_path generation.wav

# 智能音色（不指定参考，模型自行决定声音）
python3 examples/generation.py --transcript "..." --temperature 0.3 --out_path generation.wav

# 多说话人对话 + 双声音克隆
python3 examples/generation.py \
  --transcript examples/transcript/multi_speaker/en_argument.txt \
  --ref_audio belinda,broom_salesman --ref_audio_in_system_message \
  --chunk_method speaker --seed 12345 --out_path generation.wav
```

内置 17 个预置音色（`belinda`、`broom_salesman`、`en_man/woman`、`shrek_*` 等）+ `profile.yaml` 纯文本 profile（`--ref_audio profile:male_en_british`）。`--chunk_method word` 用 `langid`+`jieba` 处理长文本；MPS 上音频 tokenizer 自动放 CPU 并禁用 StaticCache/CUDA graph。

### 3.4 vLLM 高吞吐部署

`examples/vllm/` 提供 OpenAI 兼容 API（`/v1/audio/speech`、`/v1/chat/completions`）：

```bash
docker run --gpus all --ipc=host --shm-size=20gb --network=host \
  bosonai/higgs-audio-vllm:latest \
  --served-model-name "higgs-audio-v2-generation-3B-base" \
  --model "bosonai/higgs-audio-v2-generation-3B-base" \
  --audio-tokenizer-type "bosonai/higgs-audio-v2-tokenizer" \
  --limit-mm-per-prompt audio=50 --max-model-len 8192 \
  --port 8000 --gpu-memory-utilization 0.8 --disable-mm-preprocessor-cache

python examples/vllm/run_chat_completion.py --api-base http://localhost:8000/v1 --task voice_clone
```

吞吐参考：A100 40GB ~1500 audio tokens/s（~60s 音频/s），RTX 4090 24GB ~600 tokens/s。

### 3.5 技术细节与评估

<p align="center"><img src="figures/higgs_audio_v2_architecture_combined.png" width="820"></p>

v2 的三项关键创新：
- **AudioVerse 数据**：多 ASR + 音效分类 + 自研理解模型清洗标注的 1000 万小时音频。
- **统一音频 tokenizer**：从零训练，兼顾语义与声学特征（评估集 [AudioTokenBench](https://huggingface.co/datasets/bosonai/AudioTokenBench)，见 [tokenizer 博客](./tech_blogs/TOKENIZER_BLOG.md)）。
- **DualFFN 架构**：以极小算力开销增强 LLM 对声学 token 的建模（见 [架构博客](./tech_blogs/ARCHITECTURE_BLOG.md)）。

评估要点：Seed-TTS Eval SIM **67.70**（SOTA）、ESD SIM(emo2vec) **86.13**；EmergentTTS-Eval Emotions 胜率 **75.71%**、Questions **55.71%**（vs gpt-4o-mini-tts）；多说话人对话 WER/相似度亦领先 MoonCast、Dia-1.6B。

---

## 4. 核心包架构 `boson_multimodal/`

```
boson_multimodal/
├── serve/serve_engine.py             # HiggsAudioServeEngine（唯一简洁推理入口）
├── model/higgs_audio/
│   ├── modeling_higgs_audio.py       # HiggsAudioModel（~2290 行，自定义 generate/_sample）
│   ├── configuration_higgs_audio.py  # 配置：audio_adapter_type 等
│   ├── audio_head.py                 # text lm_head + audio lm_head
│   ├── cuda_graph_runner.py          # CUDA graph 捕获/回放
│   ├── custom_modules.py             # PartiallyFrozenEmbedding/Linear（训练用）
│   └── utils.py                      # delay pattern、merge、DeepSpeed Ulysses
├── audio_processing/                 # 音频 tokenizer（DAC + 语义 + RVQ；部分源自 xcodec）
├── dataset/chatml_dataset.py         # prepare_chatml_sample()
├── data_collator/higgs_audio_collator.py
└── data_types.py / constants.py      # Message/ChatMLSample、特殊 token
```

**关键点（也是易踩坑处）**：

- **推理入口** `HiggsAudioServeEngine`：构造时自动下载模型/tokenizer/audio-tokenizer，建多个 KV cache bucket（默认 `[1024,4096,8192]`），CUDA 上 `capture_model()` 捕获 CUDA graph；`generate()` 返回 `HiggsAudioResponse`（`audio`/`sampling_rate`/`generated_text`），`generate_delta_stream()` 异步流式。
- **三种 audio adapter**（`audio_adapter_type`）：`stack`（backbone 后堆 Llama 层）/ `dual_ffn`（指定层双路 FFN）/ `dual_ffn_fast_forward`（非指定层音频 hidden 透传）。
- **自定义 `generate()/_sample()`**：完全覆盖 HF 流程，三状态机 `TEXT / AUDIO_INIT / AUDIO_IN_PROGRESS`，每步据 `input_ids[-1]` 判断；音频模式维护 delay pattern + RAS 重复采样（`ras_win_len=7`）。**仅支持 batch_size=1**（硬 assert）。
- **音频 tokenizer**：默认 16kHz、~50Hz 帧率；decode 前必须 `revert_delay_pattern()` 再剪掉首尾 stream token。⚠️ 三个「size」易混：config `audio_codebook_size=1024`、模型内部 `+2=1026`、tokenizer `.codebook_size` 返回 `quantizer_dim`。
- 特殊 token：`<|AUDIO|>`(128015 输入占位) / `<|AUDIO_OUT|>`(128016 输出占位) / `<|audio_out_bos|>`(128013 触发音频生成)。

---

## 5. 仓库内 v3 工具链

> 以下三个子系统是本仓库围绕 v3 的**工程管线**，彼此独立于核心包。生成物目录（`clone_workdir/`、`*.wav`、`*.jsonl`、`higgs_v3_env/` 等）均在 `.gitignore`，请勿提交。

### 5.1 v3 文本数据生成 `higgs_audio_v3_text_generator/`

为 v3 TTS 训练批量生成多样化、带标签的文本。**与核心包完全独立，HTTP 只用 urllib（零第三方）**，LLM 推理走外部 vLLM（OpenAI 兼容，默认 Qwen3.6-27B，~80GB GPU/实例）。

**数据流**：`GenConfig` → `task_generator`（加权随机场景/情绪/长度/语言 + 多样性保护）→ `compact_prompt`（10 轴多样性注入 + 43 标签指南）→ `llm_client`（180s 超时，去 Qwen3 `<think>` 块）→ 在线 MD5 去重 → checkpoint（每 5 task）→ 后处理（精确去重 → Jaccard 语义去重 0.88 → 质量过滤）→ JSONL。

**运行**：

```bash
cd higgs_audio_v3_text_generator          # 必须在子项目目录下运行（sys.path 注入）

# 交互式单批测试
python generate_single.py --scenario daily_chat --emotion enthusiasm \
  --length medium --lang pure_cn --count 16 --output sample.jsonl

# 单实例批量
python run_batch_generation.py --total 10000 --batch-size 16 --workers 8 \
  --output generated_texts.jsonl --checkpoint checkpoint.jsonl

# 4 实例并行（需 4 个 vLLM 在 8000-8003）→ 合并去重
python run_parallel_batch.py --total 1000000 --batch-size 8 --workers 8 --num-instances 4 --output-dir ./batch_output
python postprocess_merge.py --input-dir ./batch_output --output generated_texts_final.jsonl

# 一键 100 万（tmux）
bash run_1m_gen.sh
```

**标签系统**：21 情绪 + 3 风格 + 9 SFX + 10 韵律 = 43 标签（`tags.py`），14 对互斥约束（`tag_guide.py`）。**JSONL 字段**：`text`、`clean_text`、`scenario`、`subscene`、`emotion`、`length_type`、`lang_type`、`language`、`tags_used`、`tag_count`、`char_count`、`task_id`。环境变量：`LLM_MODEL` / `LLM_BASE_URL` / `LLM_API_KEY`。

> 注意：`compact_prompt.py`（生产）与 `prompt_builder.py`+`diversity.py`（旧版，仍导出但不再调用）两套多样性池独立；`run_parallel_batch.py`/`postprocess_merge.py` 端口与实例数硬编码为 4。详见子项目内文档。

### 5.2 v3 声音复刻流水线 `v3_tts_clone/`

把「样本数 ≥20 但总时长不足 `TARGET_SEC`（生产 1800s）」的说话人，用 v3 TTS（SGLang-Omni 后端）克隆补齐，并按 SIM/CER 多轮剪枝。

**主入口**：[`v3_tts_clone/05_iterative_pipeline.sh`](v3_tts_clone/05_iterative_pipeline.sh) + `05_iterative_pipeline.env`（详细文档见 [`v3_tts_clone/README.md`](v3_tts_clone/README.md)）。

```bash
source v3_tts_clone/05_iterative_pipeline.env
tmux new-session -d -s higgs "bash v3_tts_clone/05_iterative_pipeline.sh"
tail -f clone_workdir/iterative_pipeline*/pipeline_*.log
```

**流程**：Step 0 统计（`00_prepare_stats.py`，source-only）→ Step 0.5 ASR 转写（`02_asr_*`，Qwen3-ASR）→ Step 1 gap 加权预算分配（`04_post_prune_stats.py`，只分一次并冻结）→ Step 2 × N 轮：`03_tts_clone.py` 克隆 → SIM 评估+剪枝（快，先删）→ CER 评估+剪枝（贵，后删）。

**三大设计支柱**：
1. **预算只分一次**（冻结 JSON），每轮切 `ceil(orig/N)`，原始 5 条第 6 轮自动停。
2. **磁盘扫描替代内存 tracker** → 天然幂等/断点续跑（剪枝删除自动反映）。
3. **GPU 时间片**：TTS / SIM / CER 从不同时占同一批卡，配 `stop_tts` / `reap_orphans` / `trap EXIT` 三重清理（SGLang spawn 引擎子进程会变孤儿吃显存）。

**环境隔离**：统计→系统 Python；ASR/CER→`qwen3-asr` env；TTS 服务→`higgs_v3_env`（SGLang editable 装，`--no-deps`）；TTS 客户端→系统 Python；SIM/MOS→`omnivoice` env。续跑用 `START_ROUND` / `START_STEP`（`clone=1<sim=2<cer=3`）。

> ⚠️ **当前最大质量隐患**：`TTS_MAX_NEW_TOKENS=1024`（≈40.7s @25fps）会截断长文本音频 → ASR 只听到前半 → CER 虚高 → 长 clone 被过度剪枝。根因在 TTS 侧，需要时上调到 2048/3072。

### 5.3 评估流水线 `eval_higgs_audio/`

从 OmniVoice 适配，三维度评估复刻质量。布局 `{CLONE_ROOT}/{dataset}/{speaker}/clone_NNNN.{wav,json}` + sidecar `.cer.json / .sim.json / .mos.json`。

| 维度 | 脚本 | conda env | 说明 |
|------|------|-----------|------|
| **CER** 字错率 | `eval_cer/eval_cer.py` | `qwen3-asr` | Qwen3-ASR（`vllm`/`transformers` 后端）+ 手工 ITN + jiwer 字符级 |
| **SIM** 说话人相似度 | `eval_sim/eval_sim.py` | `omnivoice` | SamResNet100ASP 256维 embedding，**raw 余弦** [-1,1] |
| **MOS** 音质 | `eval_mos/eval_mos.py` | `omnivoice` | UTMOS22Strong / SCOREQ / TTSDS2 / UTMOSv2 |

```bash
# 一键全流程（CER→SIM→MOS）
bash eval_higgs_audio/run_eval_all.sh
# 仅某一维度：--skip-sim --skip-mos / --skip-cer ...
HIGGS_CLONE_ROOT=/your/path bash eval_higgs_audio/run_eval_all.sh --sample-size 500

# 按阈值剪枝（默认 CER>0.03 或 SIM<0.8 删除）
python eval_higgs_audio/prune_and_copy.py --dry-run     # 预览
python eval_higgs_audio/prune_and_copy.py --workers 32  # 执行
```

**默认阈值**：`CER ≤ 0.03` 且 `SIM(raw) ≥ 0.8`（`postprocess_common.py`）。

> ⚠️ 生产剪枝务必用 **`--eval-source sidecar`（默认）**：直接读每条 `.cer.json/.sim.json`，剪枝时删、重评估时重建，永远新鲜；`jsonl` 是 append-only 聚合按 wav 保留首条，剪枝后编号复用时旧失败记录会压过新记录 → 误判。

### 5.4 童声批量复刻（v2 & v3）

从 BAAI-ChildMandarin 采样做童声克隆并与 SoulX-Podcast 对比（见 `CHILD_VOICE_CLONE_README.md`、`COMPARISON_WITH_SOULX.md`）。

```bash
./run_child_voice_clone_higgs.sh            # v2：HiggsAudioServeEngine
python3 batch_clone_v3.py                   # v3：先启动 SGLang-Omni，再 POST /v1/audio/speech
```

---

## 6. 开发与贡献规范

- **CI 仅一步**：`ruff format --check .`（`.github/workflows/test.yml`，push/PR 到 `main` 触发）。提交前务必本地跑 `ruff format .`。
- **版本锁定**：`ruff==0.12.2`、`transformers>=4.45.1,<4.47.0`（硬上限，勿升级）、`boto3==1.35.36`；Python 3.10+。
- **Ruff 规则**：行宽 119、双引号、py310；启用 import 排序(I)、pyupgrade(UP)、版权头(CPY)；**禁止 `os.getenv`/`os.putenv`/`os.unsetenv`，必须用 `os.environ`**；`__init__.py` 豁免 F401。
- 无 pytest / mypy；`setup.cfg` 打包排除 `tests*`、`training*`。
- 贡献与支持指南见 [SUPPORT_GUIDELINES.md](SUPPORT_GUIDELINES.md)。

---

## 7. 许可证、引用与链接

- **v3** 采用 **Boson Higgs Audio v3 Research and Non-Commercial License**；生产 / 托管 / 盈利用途需单独商业许可。
- **本仓库代码（v2/v2.5）** 见根目录 [`LICENSE`](LICENSE)（Apache-2.0）。
- **第三方**：`boson_multimodal/audio_processing/` 含源自 [xcodec](https://github.com/zhenye234/xcodec) 的代码，见该目录内 [`LICENSE`](boson_multimodal/audio_processing/LICENSE)。

```bibtex
@misc{bosonai_higgs_audio_tts_v3_2026,
  title  = {Higgs Audio v3 TTS: Conversational Speech for Voice AI from Boson AI},
  author = {Boson AI},
  year   = {2026},
  howpublished = {https://huggingface.co/bosonai/higgs-audio-v3-tts-4b},
}

@misc{higgsaudio2025,
  author = {{Boson AI}},
  title  = {{Higgs Audio V2: Redefining Expressiveness in Audio Generation}},
  year   = {2025},
  howpublished = {\url{https://github.com/boson-ai/higgs-audio}},
}
```

**相关链接**：[v3 权重](https://huggingface.co/bosonai/higgs-audio-v3-tts-4b) · [v3 API 文档](https://docs.boson.ai/models/higgs-audio-tts/overview) · [SGLang-Omni Cookbook](https://sgl-project.github.io/sglang-omni/cookbook/higgs_tts.html) · [v2 权重](https://huggingface.co/bosonai/higgs-audio-v2-generation-3B-base) · [Playground](https://boson.ai/demo/tts) · [开发者控制台](https://boson.ai/workspace)

**We are hiring！** 对多模态 AI / 语音音频模型 / 大规模系统感兴趣，欢迎投递 [Boson AI Careers](https://jobs.lever.co/bosonai)。
