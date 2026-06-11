# Higgs Audio v3 TTS

> Chat 原生文本转语音：**流式传输、102 种语言、即时声音克隆、内联情绪与风格控制**。

Higgs Audio v3 TTS 是 Boson AI 最新的 4B 参数自回归 TTS 模型，生成 **24 kHz** 语音，支持超过 100 种语言，零样本声音克隆，以及通过内联控制标签对情绪、风格、音效和韵律进行精细化控制。

---

## 目录

1. [模型架构](#模型架构)
2. [3 种使用方式总览](#3-种使用方式总览)
3. [方式一：Boson AI 云端 API（推荐，无需 GPU）](#方式一boson-ai-云端-api推荐无需-gpu)
    - [快速开始](#快速开始)
    - [预设声音](#预设声音)
    - [声音克隆](#声音克隆)
    - [自定义声音（可复用）](#自定义声音可复用)
    - [流式传输](#流式传输)
    - [内联控制标签](#内联控制标签)
    - [API 完整参数参考](#api-完整参数参考)
    - [支持的语言](#支持的语言)
4. [方式二：SGLang-Omni 本地部署（高吞吐）](#方式二sglang-omni-本地部署高吞吐)
    - [安装与启动](#安装与启动)
    - [基础生成](#基础生成)
    - [声音克隆与流式](#声音克隆与流式)
    - [标签控制示例](#标签控制示例)
    - [SSE 流式格式详解](#sse-流式格式详解)
    - [性能基准](#性能基准)
5. [方式三：HuggingFace 开源权重](#方式三huggingface-开源权重)
6. [声音选择对比](#声音选择对比)
7. [完整标签参考](#完整标签参考)
8. [核心使用规则与注意事项](#核心使用规则与注意事项)
9. [许可证与引用](#许可证与引用)

---

## 模型架构

```
┌──────────────────────────────────────────────────────────┐
│                   Higgs Audio v3 TTS                      │
│                                                          │
│  Backbone:  4B 参数自回归解码器                           │
│             - 36 层 Transformer                           │
│             - hidden_size = 2560                          │
│             - GQA 32 头 / 8 KV 头                         │
│             - 8,192 token 上下文窗口                      │
│                                                          │
│  Audio:     Higgs Tokenizer                              │
│             - 8 层 codebook，每层 1026 个 token            │
│             - 25 fps 帧率（40ms/帧）                       │
│             - 24 kHz 采样率                               │
│             - delay pattern 编码                           │
│                                                          │
│  Input:    交错式 text + audio token                      │
│  Output:   24 kHz 单声道语音                              │
└──────────────────────────────────────────────────────────┘
```

**架构亮点：**
- 输入文本和音频 token 交错消费，通过 delay pattern 交错排列 8 层 codebook
- 多 codebook 融合 embedding 将音频 token 映射到 backbone 隐藏维度
- 输出经过多 codebook 融合 head，de-delay 后解码为波形
- 多轮生成交错 `<|text|>...<|audio|>...` 块，每块基于参考和先前块

---

## 3 种使用方式总览

| 方式 | 适用场景 | 硬件要求 | 延迟 | 特点 |
|------|----------|----------|------|------|
| **Boson AI API** | 快速集成、原型开发 | 无需 GPU | 低 | 免费公测，直接 curl |
| **SGLang-Omni 本地** | 生产部署、高吞吐 | 1× H100 (80GB) | 极低 | CUDA Graph 加速，支持批量 |
| **HuggingFace 权重** | 研究、自定义推理 | GPU (≥24GB) | 中等 | 完全自控 |

---

## 方式一：Boson AI 云端 API（推荐，无需 GPU）

### 快速开始

1. 在 [boson.ai/workspace](https://boson.ai/workspace) 获取免费 API Key
2. 设置环境变量：

```bash
export BOSON_API_KEY=bai-xxxx
```

3. 最简单的调用：

```bash
curl https://api.boson.ai/v1/audio/speech \
  -H "Authorization: Bearer $BOSON_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "higgs-audio-v3-tts",
    "input": "你好，这是一个测试。"
  }' \
  --output out.mp3
```

**Python 版本：**

```python
import os
import requests

resp = requests.post(
    "https://api.boson.ai/v1/audio/speech",
    headers={"Authorization": f"Bearer {os.environ['BOSON_API_KEY']}"},
    json={
        "model": "higgs-audio-v3-tts",
        "input": "你好，这是一个测试。",
    },
)
resp.raise_for_status()
with open("out.mp3", "wb") as f:
    f.write(resp.content)
```

**TypeScript 版本：**

```ts
import { writeFile } from "node:fs/promises";

const res = await fetch("https://api.boson.ai/v1/audio/speech", {
  method: "POST",
  headers: {
    Authorization: `Bearer ${process.env.BOSON_API_KEY}`,
    "Content-Type": "application/json",
  },
  body: JSON.stringify({
    model: "higgs-audio-v3-tts",
    input: "Hello, this is a test.",
  }),
});
await writeFile("out.mp3", Buffer.from(await res.arrayBuffer()));
```

### 预设声音

API 提供 6 种预设声音，通过 `voice` 参数选择：

```bash
curl https://api.boson.ai/v1/audio/speech \
  -H "Authorization: Bearer $BOSON_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "higgs-audio-v3-tts",
    "input": "欢迎来到我的频道！今天我们来聊聊AI的未来。",
    "voice": "chloe"
  }' \
  --output out.mp3
```

| Voice | 风格 | 适用场景 |
|-------|------|----------|
| `chloe` | 友善清晰的美式女声，具有吸引力和信息传递感 | 教程、产品介绍 |
| `eleanor` | 冷静专业的美式女声，适合教育内容 | 教学、纪录片 |
| `jake` | 充满活力和戏剧性的男声，对体育充满激情 | 运动评论、播客 |
| `marcus` | 热情自信、略带教授风范的美式男声 | 学术演讲、创意分享 |
| `nora` | 冷静清晰、叙事风格的美式女声 | 有声书、故事讲述 |
| `oliver` | 冷静清晰、深思熟虑的美式男声 | 反思内容、冥想引导 |

### 声音克隆

通过参考音频（`ref_audio`）和参考文本（`ref_text`）实现零样本声音克隆：

```bash
curl https://api.boson.ai/v1/audio/speech \
  -H "Authorization: Bearer $BOSON_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "higgs-audio-v3-tts",
    "input": "你好，欢迎来到我们的团队！",
    "ref_audio": "https://example.com/your-voice-sample.mp3",
    "ref_text": "这是我的声音样本，用于声音克隆测试。"
  }' \
  --output out.mp3
```

**声音克隆最佳实践：**
- 参考音频 5-30 秒，清晰干净，无背景音乐或多说话人
- `ref_text` 必须是与 `ref_audio` 内容完全一致的逐字稿，包含语气词
- 支持格式：wav、mp3、opus、pcm、flac
- `ref_audio` 可以为 URL、data URI 或 base64 编码（base64 最大 10MB）
- 克隆后匹配输入文本与参考音频的语言可获得最佳效果

> ⚠️ **你必须拥有被克隆声音的合法权利。**

### 自定义声音（可复用）

创建一次后通过 `voice_id` 复用，无需每次上传参考音频：

**Step 1：创建自定义声音**

```bash
# 返回 voice_id
curl https://api.boson.ai/v1/voices \
  -H "Authorization: Bearer $BOSON_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "ref_audio": "https://example.com/voice-sample.mp3",
    "ref_text": "这是我的声音样本，用于声音克隆测试。",
    "title": "我的专属声音"
  }'
# Response: {"voice_id": "voice_abc123...", "title": "我的专属声音", "created_at": "..."}
```

**Step 2：复用声音**

```bash
curl https://api.boson.ai/v1/audio/speech \
  -H "Authorization: Bearer $BOSON_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "higgs-audio-v3-tts",
    "input": "使用我的专属声音说话。",
    "voice": "voice_abc123..."
  }' \
  --output out.mp3
```

> 重复使用相同参考音频会返回相同 `voice_id`，因此可安全重复调用。

### 流式传输

实时流式接收音频，降低首包延迟。**流式模式强制使用 `response_format: "pcm"`**：

**cURL：**

```bash
curl -N https://api.boson.ai/v1/audio/speech \
  -H "Authorization: Bearer $BOSON_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "higgs-audio-v3-tts",
    "input": "这是一个流式传输的测试文本。",
    "response_format": "pcm",
    "stream": true
  }'
# 响应体为原始 16-bit / 24kHz / mono PCM 字节流
```

> `-N` 禁用 curl 输出缓冲，使 SSE 事件即时输出。

**Python（采集 PCM 块）：**

```python
import os
import requests

BASE_URL = "https://api.boson.ai/v1"
API_KEY = os.environ["BOSON_API_KEY"]

payload = {
    "model": "higgs-audio-v3-tts",
    "input": "这是一个流式传输的测试文本。",
    "voice": "default",
    "response_format": "pcm",
    "stream": True,
}

pcm = bytearray()
with requests.post(
    f"{BASE_URL}/audio/speech",
    headers={
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    },
    json=payload,
    stream=True,
    timeout=180,
) as r:
    r.raise_for_status()
    for chunk in r.iter_content(chunk_size=4096):
        if chunk:  # 第一个非空 chunk = time-to-first-audio
            pcm.extend(chunk)

with open("out.pcm", "wb") as f:
    f.write(pcm)

# 可以用 pydub / ffmpeg 转换为 wav 播放
# ffmpeg -f s16le -ar 24000 -ac 1 -i out.pcm out.wav
```

### 内联控制标签

所有标签格式：`<|category:value|>`。以下是一个完整示例：

```bash
curl https://api.boson.ai/v1/audio/speech \
  -H "Authorization: Bearer $BOSON_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "higgs-audio-v3-tts",
    "input": "<|emotion:enthusiasm|><|prosody:expressive_high|>欢迎来到节目！<|prosody:long_pause|>我们开始吧！"
  }' \
  --output out.mp3
```

#### 情绪标签（21 种）

| Tag | 效果 | 示例 |
|-----|------|------|
| `<\|emotion:elation\|>` | 兴高采烈 | "这是最好的消息，我太激动了！" |
| `<\|emotion:amusement\|>` | 逗乐/调侃 | "哈哈哈，这太好笑了，我笑个不停。" |
| `<\|emotion:enthusiasm\|>` | 热情/兴奋 | "我们走！我等不及了！" |
| `<\|emotion:determination\|>` | 决心/坚定 | "无论发生什么，我都不会放弃。" |
| `<\|emotion:pride\|>` | 自豪/自信 | "看看我走了多远，我从未怀疑过。" |
| `<\|emotion:contentment\|>` | 平静满足 | "一切都刚刚好，我内心平静。" |
| `<\|emotion:affection\|>` | 温暖/亲昵 | "有你在我生命中，真的很感激。" |
| `<\|emotion:relief\|>` | 松了口气 | "谢天谢地，终于结束了。" |
| `<\|emotion:contemplation\|>` | 沉思/反省 | "有时我就坐着想，这一切到底意味着什么。" |
| `<\|emotion:confusion\|>` | 困惑 | "等等，我不明白，这怎么回事？" |
| `<\|emotion:surprise\|>` | 惊讶 | "什么？不可能！我完全没料到！" |
| `<\|emotion:awe\|>` | 敬畏/惊叹 | "哇，看看那，我从没见过这么令人叹为观止的。" |
| `<\|emotion:longing\|>` | 渴望/思念 | "我非常想你，请回到我身边。" |
| `<\|emotion:anger\|>` | 愤怒 | "你竟敢这样对我！这完全不可接受！" |
| `<\|emotion:fear\|>` | 恐惧 | "你听到了吗？有东西在外面，我真的很害怕。" |
| `<\|emotion:disgust\|>` | 厌恶 | "呃，太恶心了，我看都不想看。" |
| `<\|emotion:bitterness\|>` | 苦涩 | "我为他们做了这么多，他们就这样回报我。" |
| `<\|emotion:sadness\|>` | 悲伤 | "我真的以为会不一样，好难过。" |
| `<\|emotion:shame\|>` | 羞耻 | "我不敢相信我那么做了，太尴尬了。" |
| `<\|emotion:helplessness\|>` | 无助 | "我什么也做不了了，感觉好无力。" |

#### 风格标签（3 种）

| Tag | 效果 |
|-----|------|
| `<\|style:singing\|>` | 唱歌模式 |
| `<\|style:shouting\|>` | 喊叫/高声 |
| `<\|style:whispering\|>` | 耳语 |

#### 音效标签（9 种，需配拟声词）

| Tag | 效果 | 建议拟声词 |
|-----|------|------------|
| `<\|sfx:cough\|>` | 咳嗽 | Ahem |
| `<\|sfx:laughter\|>` | 笑声 | Haha / Hehe |
| `<\|sfx:crying\|>` | 哭泣 | Boohoo / Sob |
| `<\|sfx:screaming\|>` | 尖叫 | Ahh / Aaah |
| `<\|sfx:burping\|>` | 打嗝 | Burp |
| `<\|sfx:humming\|>` | 哼唱 | Hmm / Mmm |
| `<\|sfx:sigh\|>` | 叹气 | Uh / Ahh |
| `<\|sfx:sniff\|>` | 抽鼻子 | Sff |
| `<\|sfx:sneeze\|>` | 喷嚏 | Achoo |

> 音效是**在说话人声音中生成的**，不是混入的外部音轨。每个音效标签后必须紧贴对应的拟声词。

#### 韵律标签（10 种）

| Tag | 效果 |
|-----|------|
| `<\|prosody:speed_very_slow\|>` | ~0.65× 语速 |
| `<\|prosody:speed_slow\|>` | ~0.85× 语速 |
| `<\|prosody:speed_fast\|>` | ~1.2× 语速 |
| `<\|prosody:speed_very_fast\|>` | ~1.4× 语速 |
| `<\|prosody:pitch_low\|>` | ~−3 半音 |
| `<\|prosody:pitch_high\|>` | ~+2.5 半音 |
| `<\|prosody:pause\|>` | ~400–700 ms 停顿 |
| `<\|prosody:long_pause\|>` | ~700–1500 ms 停顿 |
| `<\|prosody:expressive_high\|>` | 更高表现力 |
| `<\|prosody:expressive_low\|>` | 更平淡表达 |

### API 完整参数参考

```jsonc
{
  "model": "higgs-audio-v3-tts",     // 必需: 模型名（固定）
  "input": "要合成的文本。",           // 必需: 1-5000 字符，可含内联标签
  "voice": "default",                // 可选: 预设声音名 或 自定义 voice_id
  "response_format": "mp3",          // 可选: mp3/wav/opus/flac/aac/pcm
  "stream": false,                   // 可选: 流式模式(仅 pcm)
  "ref_audio": "URL | base64",       // 可选: 参考音频(克隆用)
  "ref_text": "参考音频的文本。",      // 可选: 建议提供
  "temperature": 1.0,                // 可选: 采样温度
  "top_p": null,                     // 可选: top-p 采样
  "top_k": null,                     // 可选: top-k 采样
  "seed": null                       // 可选: 随机种子(保证可复现)
}
```

详细 API 文档：[docs.boson.ai/api-reference/text-to-speech/create-speech](https://docs.boson.ai/api-reference/text-to-speech/create-speech)

### 支持的语言

语言**自动检测**，无需指定。共支持 102 种语言：

**WER/CER < 5（85 种，生产级质量）：**

中文、英文、日文、韩文、法文、德文、西班牙文、意大利文、葡萄牙文、俄文、阿拉伯文、印地文、孟加拉文、荷兰文、波兰文、土耳其文、越南文、泰文、印尼文、马来文、瑞典文、丹麦文、挪威文、芬兰文、捷克文、匈牙利文、罗马尼亚文、希腊文、希伯来文、乌克兰文、保加利亚文、克罗地亚文、斯洛伐克文、斯洛维尼亚文、爱沙尼亚文、拉脱维亚文、立陶宛文、加泰罗尼亚文、巴斯克文、加利西亚文、格鲁吉亚文、亚美尼亚文、阿塞拜疆文、哈萨克文、乌兹别克文、蒙古文、尼泊尔文、僧伽罗文、泰米尔文、泰卢固文、卡纳达文、马拉雅拉姆文、马拉地文、古吉拉特文、菲律宾文、爪哇文、斯瓦希里文、祖鲁文、科萨文、南非荷兰文、豪萨文、伊博文、约鲁巴文、林加拉文等

**WER/CER 5-10（17 种，可用）：**

阿尔巴尼亚文、冰岛文、爱尔兰文、威尔士文、卢森堡文、拉丁文、索马里文、奥罗莫文、普什图文等

> 语言质量由 WER/CER（词/字错误率）衡量，越低越好。完整列表见 [docs.boson.ai/models/higgs-audio-tts/languages](https://docs.boson.ai/models/higgs-audio-tts/languages)

---

## 方式二：SGLang-Omni 本地部署（高吞吐）

### 安装与启动

```bash
# 安装 sglang-omni（跟随官方安装指南）
# https://sgl-project.github.io/sglang-omni/get_started/installation.html

# 下载模型权重
export HF_TOKEN=hf_xxxxxxxxxxxxxxxx
hf download bosonai/higgs-audio-v3-tts-4b

# 启动服务
sgl-omni serve \
  --model-path bosonai/higgs-audio-v3-tts-4b \
  --port 8000
```

### 基础生成

**cURL：**

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "你好，最近怎么样？"}' \
  --output output.wav
```

**Python：**

```python
import requests

resp = requests.post(
    "http://localhost:8000/v1/audio/speech",
    json={"input": "Hello, how are you?"},
)
resp.raise_for_status()
with open("output.wav", "wb") as f:
    f.write(resp.content)
```

### 声音克隆与流式

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Have a nice day and enjoy the sunshine!",
    "references": [{
      "audio_path": "/path/to/voice.wav",
      "text": "Hey, Adam here. Let's create something that feels real."
    }],
    "temperature": 0.8,
    "top_k": 50,
    "max_new_tokens": 1024
  }' \
  --output output.wav
```

**流式传输（SSE）：**

```bash
curl -N -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Get the trust fund to the bank early.",
    "references": [{
      "audio_path": "/path/to/voice.wav",
      "text": "Reference transcript here..."
    }],
    "stream": true
  }'
```

**流式 + 原始 PCM（最低延迟）：**

```bash
curl -N -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Get the trust fund to the bank early.",
    "references": [{
      "audio_path": "https://example.com/voice.wav",
      "text": "Reference transcript here..."
    }],
    "stream": true,
    "stream_format": "audio",
    "response_format": "pcm",
    "initial_codec_chunk_frames": 1
  }' \
  --output output.pcm
```

> `stream_format="audio"` + `response_format="pcm"`：返回裸 `audio/pcm` 16-bit mono PCM 字节流，无 SSE JSON 包装，无 `[DONE]` 哨兵。`initial_codec_chunk_frames=1` 设置更低的首次音频延迟。

### 标签控制示例

#### 情绪：逗乐 + 笑声

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "<|emotion:amusement|><|prosody:expressive_high|>等等等等，那太搞笑了。<|sfx:laughter|>嘿嘿，不，说真的，我还没准备好呢。",
    "temperature": 0.8,
    "top_k": 50,
    "max_new_tokens": 1024
  }' \
  --output output.wav
```

#### 情绪：愤怒 + 喊叫

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "<|emotion:anger|><|style:shouting|>不，这不行！我们不能发布听起来坏了、延迟且不自然的东西。",
    "temperature": 0.8,
    "top_k": 50,
    "max_new_tokens": 1024
  }' \
  --output output.wav
```

#### 情绪：悲伤 + 哭泣 + 抽鼻子

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "<|emotion:sadness|><|sfx:crying|>我……对不起。<|sfx:sniff|>Sff，我们真的尽力了。那么多深夜加班之后，我以为一切都失败了。",
    "references": [{
      "audio_path": "/path/to/voice.wav",
      "text": "It was the night before my birthday. Hooray!"
    }],
    "temperature": 0.8,
    "top_k": 50,
    "max_new_tokens": 1024
  }' \
  --output output.wav
```

#### 情绪：惊讶 + 尖叫

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "<|emotion:surprise|><|prosody:pitch_high|><|sfx:screaming|>啊！等等，我差点忘了！Higgs Audio v3 还支持一百多种语言。",
    "references": [{
      "audio_path": "/path/to/voice.wav",
      "text": "It was the night before my birthday. Hooray!"
    }],
    "temperature": 0.8,
    "top_k": 50,
    "max_new_tokens": 1024
  }' \
  --output output.wav
```

#### 多轮对话示例

使用两个不同声音克隆模拟对话：

```bash
# 第一轮 — 她询问
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "<|emotion:contemplation|>Hi David, 我今天感冒了，错过了生物课。<|sfx:cough|>Ahem！抱歉，你能告诉我老师讲了什么吗？",
    "references": [{"audio_path": "female-voice.wav", "text": "By repeating what students say..."}],
    "temperature": 0.8, "top_k": 50, "max_new_tokens": 1024
  }' --output part1.wav

# 第二轮 — 他回答
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "<|emotion:enthusiasm|>没问题！我们学了植物如何通过光合作用制造食物，还有<|prosody:long_pause|>这周五有小测验。",
    "references": [{"audio_path": "male-voice.wav", "text": "Hey, Adam here..."}],
    "temperature": 0.8, "top_k": 50, "max_new_tokens": 1024
  }' --output part2.wav

# 合成对话（插入 0.6s 间隔）
ffmpeg -y \
  -i part1.wav -f lavfi -t 0.6 -i anullsrc=r=24000:cl=mono \
  -i part2.wav \
  -filter_complex "[0:a][1:a][2:a]concat=n=3:v=0:a=1" \
  dialogue.wav
```

### SSE 流式格式详解

SGLang-Omni 的默认流式响应格式（SSE）：

```
data: {"id":"speech-...","object":"audio.speech.chunk","index":0,"audio":{"data":"<base64 WAV>","format":"wav"},"finish_reason":null}
data: {"id":"speech-...","object":"audio.speech.chunk","index":1,"audio":{"data":"<base64 WAV>","format":"wav"},"finish_reason":null}
...
data: {"id":"speech-...","object":"audio.speech.chunk","index":N,"audio":null,"finish_reason":"stop","usage":{...}}
data: [DONE]
```

**Python 流式消费：**

```python
import requests
import base64
import json

REFERENCE_AUDIO = "docs/_static/audio/male-voice.wav"
REFERENCE_TEXT = "Hey, Adam here. Let's create something that feels real."
SPEECH_INPUT = "Get the trust fund to the bank early."

with requests.post(
    "http://localhost:8000/v1/audio/speech",
    json={
        "input": SPEECH_INPUT,
        "references": [{"audio_path": REFERENCE_AUDIO, "text": REFERENCE_TEXT}],
        "stream": True,
    },
    stream=True,
) as resp:
    resp.raise_for_status()
    with open("output_streaming.wav", "wb") as f:
        for line in resp.iter_lines():
            if not line or line == b"data: [DONE]":
                continue
            if not line.startswith(b"data: "):
                continue
            event = json.loads(line[len(b"data: "):])
            if event.get("finish_reason") == "stop":
                break
            audio_data = event.get("audio") or {}
            if audio_data.get("data"):
                chunk = base64.b64decode(audio_data["data"])
                f.write(chunk)
                # 实际应用中直接将 chunk 喂给音频播放器
```

### 性能基准

SGLang-Omni 在 1× H100 上的性能数据（Seed-TTS EN 全集 N=1088，bf16，CUDA Graph 开启）：

| 并发数 | 吞吐 (req/s) | 平均延迟 | RTF | 音频 s/s |
|--------|-------------|---------|-----|----------|
| 1 | 1.62 | 617 ms | 0.147 | 6.89 |
| 2 | 2.70 | 742 ms | 0.180 | 11.37 |
| 4 | 5.45 | 733 ms | 0.177 | 22.84 |
| 8 | 8.91 | 898 ms | 0.217 | 37.38 |
| 16 | 14.74 | 1079 ms | 0.262 | 61.84 |

- **并发数** — 最大同时请求数
- **RTF** — 处理时间 / 音频时长比值（<1 即比实时快）
- **音频 s/s** — 每秒生成的音频秒数

---

## 方式三：HuggingFace 开源权重

权重地址：[bosonai/higgs-audio-v3-tts-4b](https://huggingface.co/bosonai/higgs-audio-v3-tts-4b)

> **注意**：Higgs Audio v3 **不依赖本仓库代码**（本仓库是 v2/v2.5 代码）。v3 推荐使用 SGLang-Omni 提供服务。

```bash
# 下载权重
export HF_TOKEN=hf_xxxxxxxxxxxxxxxx
hf download bosonai/higgs-audio-v3-tts-4b

# 使用 SGLang-Omni 服务
sgl-omni serve --model-path bosonai/higgs-audio-v3-tts-4b --port 8000
```

---

## 核心使用规则与注意事项

### 标签使用两大规则

1. **交付类标签放在开头。** Emotion、Style、以及韵律的速度（speed）、音高（pitch）和表现力（expressive）标签决定整个 turn 的交付方式，**必须放在 `input` 文本的最开头，任何文本之前**。

   正确：`<|emotion:enthusiasm|><|prosody:expressive_high|>欢迎来到节目！`
   错误：`欢迎来到节目！<|emotion:enthusiasm|>`

2. **音效必须配拟声词。** 每个 `<|sfx:...|>` 标签后必须紧贴对应拟声词。

   正确：`<|sfx:laughter|>Haha, 太好笑了！`
   错误：`<|sfx:laughter|>太好笑了！`

### 位置型标签是例外

- `<|prosody:pause|>` 和 `<|prosody:long_pause|>` **放在需要停顿的位置**，而非开头
- 每个 `<|sfx:...|>` 放在触发该音效的位置

### 常见限制

| 限制项 | 值 |
|--------|-----|
| `input` 最大长度 | 5000 字符 |
| `ref_audio` 内联编码最大 | 10MB |
| 流式模式要求 | `response_format` 必须为 `pcm` |
| API 调用频率 | 公测期间有限流 |
| 支持音频格式 | mp3, wav, opus, flac, aac, pcm |
| 采样率 | 24 kHz（固定） |
| 声道 | 单声道（mono） |
| 位深 | 16-bit |

### 声音克隆注意事项

- 音频需 5-30 秒，**干净语音，无背景音乐或多说话人**
- **`ref_text` 必须是准确的逐字稿**（包括填充词如 "uh", "well" 等）
- 输入文本和参考音频的语言应匹配
- 克隆质量因声音特征、录音质量和语言而异
- ⚠️ **法律合规：必须拥有被克隆声音的合法权益**

### 错误处理

```json
// 400 错误示例
{
  "error": {
    "message": "input_too_long",
    "type": "invalid_request_error"
  }
}
```

常见错误类型：
- `input_too_long` — 输入超过 5000 字符
- `401` — 缺少或无效的 API Key
- `400` — 无效的请求参数

---

## 评估基准

### 多语种声音克隆

| 基准 | 语言数 | WER/CER ↓ |
|------|--------|-----------|
| Seed-TTS | 2 | 1.11 |
| CV3 | 9 | 4.41 |
| MiniMax-Multilingual | 23 | 2.74 |
| Higgs-Multilingual | 111 | 3.61 |

### Emergent TTS（胜率 vs 基线）

| 类别 | 胜率 ↑ |
|------|--------|
| Overall | 53.65% |
| Emotions | 53.75% |
| Foreign Words | 48.75% |
| Paralinguistics | 68.57% |
| Complex Pronunciation | 25.10% |
| Questions | 61.43% |
| Syntactic Complexity | 60.71% |

---

## 许可证与引用

### 许可证

Higgs Audio v3 TTS 使用 **Boson Higgs Audio v3 Research and Non-Commercial License**。生产/托管/盈利用途需单独获取商业许可。

### 引用

```bibtex
@misc{bosonai_higgs_audio_tts_v3_2026,
  title  = {Higgs Audio v3 TTS: Conversational Speech for Voice AI from Boson AI},
  author = {Boson AI},
  year   = {2026},
  howpublished = {https://huggingface.co/bosonai/higgs-audio-v3-tts-4b},
}
```

### 相关链接

- [HuggingFace 模型权重](https://huggingface.co/bosonai/higgs-audio-v3-tts-4b)
- [Boson AI API 文档](https://docs.boson.ai/models/higgs-audio-tts/overview)
- [SGLang-Omni Cookbook](https://sgl-project.github.io/sglang-omni/cookbook/higgs_tts.html)
- [Boson AI 开发者控制台](https://boson.ai/workspace)
- [v2/v2.5 仓库（当前）](https://github.com/boson-ai/higgs-audio)
