# Higgs Audio v3 Text Generator

> 为 Higgs Audio v3 TTS 训练生成带标注标签的多样化多语种文本数据集。

核心思路：用 LLM（Qwen3.6-27B）批量生成带 Higgs v3 标签的对话文本，经过多层去重和质量过滤，产出可直接用于 TTS 训练的高质量 `<场景, 情绪, 长度, 语言>` 标注语料。

---

## 目录

1. [架构概览](#架构概览)
2. [环境准备](#环境准备)
3. [快速开始](#快速开始)
4. [配置系统](#配置系统)
5. [标签系统](#标签系统)
6. [场景系统](#场景系统)
7. [去重与质量过滤](#去重与质量过滤)
8. [输出格式](#输出格式)
9. [入口脚本详解](#入口脚本详解)
10. [API 参考](#api-参考)
11. [生产环境大规模生成](#生产环境大规模生成)
12. [故障排查](#故障排查)

---

## 架构概览

```
┌─────────────────────┐
│   Task Generator     │  根据配置生成任务列表（场景/情绪/长度/语言组合）
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│    Prompt Builder    │  构建紧凑 prompt，注入 10 轴多样性（人设/话题/开头/焦点/感官/地点/时间/语体/话轮/句式）
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   vLLM API Client    │  调用 Qwen3.6-27B 批量生成文本（OpenAI 兼容 API）
│  (port 8000-8003)    │  支持多实例并行（最多 4 个 vLLM）
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│    去重引擎          │  精确去重 (MD5) + 语义去重 (Jaccard, 3-gram shingle)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│    质量过滤器        │  标签格式校验、SFX 拟声词配对、完整性检查、敏感词过滤
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│    JSONL 输出        │  每行: {text, clean_text, scenario, emotion, tags_used, ...}
└─────────────────────┘
```

### 多样性控制

Prompt 中注入了 **10 个多样性维度的随机轴**，确保生成文本不重复：

| 维度 | 选项数 | 示例 |
|------|--------|------|
| 人设 (`_PERSONA_PROFILES`) | 12 | 急性子打工人、温柔妈妈、挑剔顾客、兴奋小孩、疲惫上班族等 |
| 话题 (`_TOPIC_SEEDS`) | 每场景 10-15 | 丢了东西/找到了、收到意外礼物、被放鸽子等 |
| 开头类型 (`_OPENING_TYPES`) | 8 | 感叹词开头、填充词开头、问句开头、动作动词开头等 |
| 焦点类型 (`_FOCUS_TYPES`) | 10 | 描述事件、表达感受、提问、给建议、比较、回忆过去等 |
| 感官类型 (`_SENSORY_TYPES`) | 6 | 视觉、听觉、味觉、触觉、嗅觉、身体感觉 |
| 地点 (`_PLACES`) | 20 | 咖啡馆、医院、图书馆、健身房、厨房、办公室等 |
| 时间 (`_TIMES`) | 10 | 早上刚醒、中午午饭、深夜加班、下雨天等 |
| 语体 (`_REGISTER_TYPES`) | 8 | 随意口语、半正式、正式、急躁、亲密、幽默等 |
| 对话状态 (`_DIALOGUE_STATES`) | 9 | 独白、对朋友说话、发语音消息、内心独白等 |
| 句法结构 | 18 | 句式(8种)+语序(5种)+节奏(5种) |

- 情绪混合策略：50% 主情绪 + ~30% 次要情绪 + ~20% 无情绪标签
- 标签数量分配：30% 无标签 + 40% 单标签 + 25% 双标签 + 5% 三标签（基于 `RECOMMENDED_COMBINATIONS` 推荐搭配）
- 多标签组合自动检测互斥冲突（如 speed_very_slow/speed_very_fast、shouting/whispering 不会同条出现）
- 每个多样性轴采用基于 MD5 的确定性随机打乱，无连续重复
- 温度抖动：每个 batch 的温度在 `config.temperature ± 0.15` 范围内随机，限制在 `[0.65, 1.0]`
- **标签多样性压制**：双重窗口（近期 300 条 + 全局 5000 条）检测 SFX/style/prosody 标签缺漏，缺 ≥5 个 SFX/≥2 个 style/≥6 个 prosody 时自动注入提示

---

## 环境准备

### 依赖

```txt
# requirements.txt
requests
tqdm
loguru
```

### 环境变量

| 变量 | 用途 | 默认值 |
|------|------|--------|
| `LLM_MODEL` | vLLM 模型名 | `qwen3.6-27b` |
| `LLM_API_KEY` | vLLM API Key | `EMPTY` |
| `LLM_BASE_URL` | vLLM 服务地址 | `http://localhost:8000` |
| `VLLM_API_KEY` | 备选 API Key（第二优先级） | - |
| `OPENAI_API_KEY` | 备选 API Key（第三优先级） | - |
| `BOSON_API_KEY` | Boson TTS API Key（仅 tts_client.py 使用） | - |

### vLLM 服务器要求

确保 Qwen3.6-27B 已在 vLLM 中启动（单实例或 4 实例并行）：

```bash
# 单实例模式
vllm serve Qwen/Qwen3.6-27B --port 8000

# 4 实例并行模式（每个实例一个端口）
vllm serve Qwen/Qwen3.6-27B --port 8000 &
vllm serve Qwen/Qwen3.6-27B --port 8001 &
vllm serve Qwen/Qwen3.6-27B --port 8002 &
vllm serve Qwen/Qwen3.6-27B --port 8003 &
```

---

## 快速开始

### 单 batch 交互式生成

生成一小批文本，查看效果：

```bash
python generate_single.py \
  --scenario daily_chat \
  --emotion amusement \
  --length short \
  --lang pure_cn \
  --count 16 \
  --temperature 0.85 \
  --output sample.jsonl
```

### 单实例批量生成

```bash
python run_batch_generation.py \
  --total 10000 \
  --batch-size 16 \
  --workers 8 \
  --temperature 0.85 \
  --output generated_texts.jsonl \
  --checkpoint checkpoint.jsonl
```

### 4 实例并行生成

```bash
python run_parallel_batch.py \
  --total 1000000 \
  --batch-size 16 \
  --workers 8 \
  --num-instances 4 \
  --temperature 0.85 \
  --seed 42 \
  --output-dir ./batch_output
```

### 后处理合并

合并 4 个 worker 的输出并进行去重和质量过滤：

```bash
python postprocess_merge.py \
  --input-dir ./batch_output \
  --output generated_texts.jsonl \
  --semantic-threshold 0.88 \
  --max-tags 5
```

---

## 配置系统

所有配置通过 `GenConfig` dataclass 管理（`higgs_text_gen/config.py`）：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `total_target` | `int` | `10000` | 目标生成文本总数 |
| `batch_size` | `int` | `16` | 每次 LLM 调用生成的文本数量 |
| `max_workers` | `int` | `1` | 线程池并发数（建议 8） |
| `scenario_distribution` | `Dict[str,float]` | 加权分布 | 10 个场景的相对权重 |
| `length_distribution` | `Dict[str,float]` | 5 级分布 | ultra_short:0.15, short:0.30, medium:0.30, long:0.18, very_long:0.07 |
| `lang_mix_distribution` | `Dict[str,float]` | 4 种分布 | pure_cn:0.45, pure_en:0.35, cn_main:0.12, en_main:0.08 |
| `stress_test_ratio` | `float` | `0.10` | 压力测试场景占比（asr_stress） |
| `semantic_dedup_threshold` | `float` | `0.88` | 语义去重的 Jaccard 阈值 |
| `same_context_dup_threshold` | `float` | `0.52` | 同上下文去重的 SequenceMatcher 阈值 |
| `suppression_window_size` | `int` | `500` | 前缀抑制回看窗口（防重复） |
| `model` | `str` | `qwen3.6-27b` | LLM 模型名 |
| `base_url` | `str` | `http://localhost:8000` | vLLM API 端点 |
| `temperature` | `float` | `0.85` | LLM 采样温度 |
| `max_tokens` | `int` | `2048` | LLM 最大输出 token 数 |
| `max_retries` | `int` | `3` | LLM 调用失败重试次数 |
| `retry_base_delay` | `float` | `1.0` | 指数退避基础延迟（秒） |
| `max_tags_per_text` | `int` | `5` | 单文本最大标签数（超出则丢弃） |
| `max_same_tag_repeat` | `int` | `2` | 同一标签最大连续重复次数 |
| `seed` | `int` | `42` | 随机种子 |
| `generate_clean_text` | `bool` | `True` | 是否剥离标签生成 `clean_text` 字段 |

---

## 标签系统

> 43 个标签，分 4 个类别。所有标签格式：`<|category:name|>`

### Emotion（21 种情绪）

| Tag | 效果 |
|-----|------|
| `<|emotion:elation|>` | 兴高采烈 / 喜悦 |
| `<|emotion:amusement|>` | 逗乐 / 调侃 |
| `<|emotion:enthusiasm|>` | 热情 / 兴奋 |
| `<|emotion:determination|>` | 决心 / 坚定 |
| `<|emotion:pride|>` | 自豪 / 自信 |
| `<|emotion:contentment|>` | 平静满足 |
| `<|emotion:affection|>` | 温暖 / 亲昵 |
| `<|emotion:relief|>` | 松了口气 |
| `<|emotion:contemplation|>` | 沉思 / 反省 |
| `<|emotion:confusion|>` | 困惑 |
| `<|emotion:surprise|>` | 惊讶 |
| `<|emotion:awe|>` | 敬畏 / 惊叹 |
| `<|emotion:longing|>` | 渴望 / 思念 |
| `<|emotion:arousal|>` | 高度渴望 |
| `<|emotion:anger|>` | 愤怒 |
| `<|emotion:fear|>` | 恐惧 |
| `<|emotion:disgust|>` | 厌恶 |
| `<|emotion:bitterness|>` | 苦涩 |
| `<|emotion:sadness|>` | 悲伤 |
| `<|emotion:shame|>` | 羞耻 |
| `<|emotion:helplessness|>` | 无助 |

### Style（3 种风格）

| Tag | 效果 |
|-----|------|
| `<|style:singing|>` | 唱歌 |
| `<|style:shouting|>` | 喊叫 / 高声 |
| `<|style:whispering|>` | 耳语 |

### Sound Effects（9 种音效，需配对拟声词）

| Tag | 效果 | 拟声词 |
|-----|------|--------|
| `<|sfx:laughter|>` | 笑声 | Haha / Hehe |
| `<|sfx:sigh|>` | 叹气 | Uh / Ahh |
| `<|sfx:cough|>` | 咳嗽 | Ahem |
| `<|sfx:crying|>` | 哭泣 | Boohoo / Sob |
| `<|sfx:screaming|>` | 尖叫 | Ahh / Aaah |
| `<|sfx:humming|>` | 哼唱 | Hmm / Mmm |
| `<|sfx:sniff|>` | 抽鼻子 | Sff |
| `<|sfx:sneeze|>` | 打喷嚏 | Achoo |
| `<|sfx:burping|>` | 打嗝 | Burp |

### Prosody（10 种韵律控制）

| Tag | 效果 |
|-----|------|
| `<|prosody:speed_very_slow|>` | ~0.65× 语速 |
| `<|prosody:speed_slow|>` | ~0.85× 语速 |
| `<|prosody:speed_fast|>` | ~1.2× 语速 |
| `<|prosody:speed_very_fast|>` | ~1.4× 语速 |
| `<|prosody:pitch_low|>` | ~−3 半音 |
| `<|prosody:pitch_high|>` | ~+2.5 半音 |
| `<|prosody:pause|>` | ~400–700 ms 停顿 |
| `<|prosody:long_pause|>` | ~700–1500 ms 停顿 |
| `<|prosody:expressive_high|>` | 更高表现力 |
| `<|prosody:expressive_low|>` | 更平淡表达 |

### 重要约束

- **SFX 必须配对拟声词**：`<|sfx:laughter|>` 后必须紧跟 "Haha/Hehe/嘿嘿/哈哈" 之一
- **SFX 在 8 个中严格要求**（`sniff` 除外）——质量过滤器会丢弃不符合的文本
- **标签数量上限**：每文本最多 5 个标签（`max_tags_per_text`）
- **标签不得连续重复**：同一标签最多连用 2 次（`max_same_tag_repeat`）

---

## 场景系统

10 个场景，各有权重分布、子场景和典型情绪：

| 场景 Key | 中文名 | 简介 | 子场景数 |
|----------|--------|------|----------|
| `daily_chat` | 日常聊天 | 朋友、家人、同事之间的闲聊 | 11 |
| `business` | 商务 | 会议、谈判、汇报、邮件 | 8 |
| `education` | 教育 | 教学、讲座、辅导、学术讨论 | 9 |
| `emotional` | 情感 | 告白、争吵、和解、安慰 | 10 |
| `entertainment` | 娱乐 | 游戏、影视评论、搞笑吐槽 | 7 |
| `narration` | 叙述 | 讲故事、新闻播报、纪录片旁白 | 6 |
| `social_media` | 社交平台 | 直播带货、短视频、vlog | 8 |
| `service` | 服务 | 客服、点餐、酒店、医院 | 9 |
| `creative_writing` | 创作 | 诗歌、剧本、小说、广告文案 | 5 |
| `asr_stress` | ASR 压力测试 | 含噪音、口语化、方言混合 | 8 |

### 场景分布（默认）

```
daily_chat:      0.18
business:        0.10
education:       0.10
emotional:       0.10
entertainment:   0.12
narration:       0.08
social_media:    0.10
service:         0.10
creative_writing:0.05
asr_stress:      0.07
```

### 长度规格（5 级）

| 类型 | 中文字数 | 英文词数 |
|------|---------|---------|
| `ultra_short` | 5-15 | 3-10 |
| `short` | 15-40 | 10-25 |
| `medium` | 40-100 | 25-60 |
| `long` | 100-250 | 60-150 |
| `very_long` | 250-500 | 150-300 |

### 语言混合规格（4 种）

| 类型 | 说明 |
|------|------|
| `pure_cn` | 纯中文 |
| `pure_en` | 纯英文 |
| `cn_main` | 中文为主，含 1-2 个英文词 |
| `en_main` | 英文为主，含 1-2 个中文词 |

---

## 去重与质量过滤

### 去重流程（3 层）

```
┌──────────────┐
│  1. 在线去重  │  batch 内生成时立即检查 MD5 哈希（raw text + normalized text）
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  2. 精确去重  │  全量 MD5 哈希比对（去掉标签、替换数字为 <NUM>、去标点、小写）
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  3. 语义去重  │  按 (scenario, subscene, emotion, length_type, lang_type) 分组
│               │  Jaccard 相似度 (3-gram shingle), 阈值 0.88
│               │  同上下文 SequenceMatcher 阈值 0.52
└──────────────┘
```

### 质量控制规则

| 规则 | 说明 |
|------|------|
| 最小长度 | 文本去除标签后至少 2 字符 |
| 敏感词黑名单 | 股票、投资、政治、战争、sex、kill、die、porn、脏话等 |
| 标签格式校验 | 所有 `<|category:name|>` 中的 category 和 name 必须合法 |
| SFX 拟声词校验 | SFX 标签后 10 字符内必须出现预期拟声词 |
| 完整性检查 | 文本不以逗号、分号、破折号、省略号、"and"、"but" 等结尾 |
| 标签数量限制 | 单文本标签数 ≤ `max_tags_per_text`（默认 5） |
| 同标签重复限制 | 同一标签连续出现 ≤ `max_same_tag_repeat`（默认 2） |
| 长度范围校验 | 去除标签后的字符数必须在 `LENGTH_BOUNDS * 1.5` 范围内 |

---

## 输出格式

每行一个 JSON 对象：

```json
{
  "text": "<|emotion:amusement|>没闻到那股廉价香水味吗？<|sfx:laughter|>哈哈，快回家！",
  "clean_text": "没闻到那股廉价香水味吗？哈哈，快回家！",
  "scenario": "entertainment",
  "subscene": "搞笑吐槽",
  "emotion": "amusement",
  "length_type": "ultra_short",
  "lang_type": "pure_cn",
  "language": "zh",
  "tags_used": ["emotion:amusement", "sfx:laughter"],
  "tag_count": 2,
  "char_count": 16,
  "task_id": 2
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `text` | `str` | 原始文本（含 Higgs v3 标签） |
| `clean_text` | `str` | 剥离标签后的纯净文本 |
| `scenario` | `str` | 场景标识（10 个场景之一） |
| `subscene` | `str` | 子场景名称（如 "搞笑吐槽"） |
| `emotion` | `str` | 情绪标签（21 个情绪之一） |
| `length_type` | `str` | 长度类型（ultra_short/short/medium/long/very_long） |
| `lang_type` | `str` | 语言混合类型（pure_cn/pure_en/cn_main/en_main） |
| `language` | `str` | ISO 语言代码（zh/en） |
| `tags_used` | `list[str]` | 实际使用的标签列表 |
| `tag_count` | `int` | 标签数量 |
| `char_count` | `int` | 字符数（去除标签后） |
| `task_id` | `int` | 任务 ID（用于回溯） |

---

## 入口脚本详解

### 1. `generate_single.py` — 交互式单 batch 生成

快速测试 prompt 效果的小工具。

```bash
python generate_single.py \
  --scenario <场景key> \       # 必填: daily_chat / emotional / ...
  --emotion <情绪key> \        # 必填: amusement / anger / ...
  --length <长度type> \        # 必填: ultra_short / short / ...
  --lang <语言type> \          # 必填: pure_cn / pure_en / ...
  --count <数量> \             # 生成数量, 默认 16
  --temperature <温度> \       # 采样温度, 默认 0.85
  --model <模型名> \           # LLM 模型名
  --base-url <vLLM地址> \      # vLLM API 地址
  --api-key <API密钥> \        # API 密钥
  --output <输出路径>          # 输出 JSON 路径, 默认 print stdout
```

### 2. `run_batch_generation.py` — 单实例批量管道

核心批量生成脚本，使用线程池并发。

```bash
python run_batch_generation.py \
  --total 10000 \              # 目标生成数量
  --batch-size 16 \            # 每次 LLM 调用的 batch 大小
  --workers 8 \               # 线程池并发数
  --model qwen3.6-27b \        # LLM 模型
  --base-url http://localhost:8000 \
  --api-key EMPTY \
  --temperature 0.85 \
  --output generated_texts.jsonl \
  --checkpoint checkpoint.jsonl \
  --resume \                   # 从 checkpoint 恢复
  --no-postprocess             # 跳过去重和质量过滤（用于多实例模式）
```

**关键流程：**
1. 创建任务列表（`generate_task_list()`）
2. 从 checkpoint 恢复已完成的任务
3. 线程池并发处理任务
4. 每 5 个任务保存一次 checkpoint
5. 在线去重（batch 内去重）
6. 最终去重 + 质量过滤 + 保存

**前缀抑制机制：**
程序会追踪最近 500 条文本的前 4 个字符前缀，如果某前缀出现频率 ≥1.25%，会将抑制提示注入 prompt 中，告诉 LLM 避免以该前缀开头。

**标签多样性压制机制：**
双重窗口检测：近期窗口（300 条）检测当前是否还在使用全部标签类别，全局窗口（5000 条）检测长期累积使用是否严重偏斜。缺漏 ≥5 个 SFX / ≥2 个 style / ≥6 个 prosody 标签时自动注入提示引导 LLM 轮换。

### 3. `run_parallel_batch.py` — 4 实例并行管道

适用于大规模生产（如百万级文本生成）。

```bash
python run_parallel_batch.py \
  --total 1000000 \            # 总目标（自动均分到 4 个 worker）
  --batch-size 16 \
  --workers 8 \
  --num-instances 4 \          # vLLM 实例数（端口 8000-8003）
  --temperature 0.85 \
  --seed 42 \
  --output-dir ./batch_output
```

**工作原理：**
1. 将 `total / 4` 分配给 4 个子进程
2. 每个子进程调用 `run_batch_generation.py --no-postprocess`，禁用内部去重
3. 每个 worker 使用不同 seed（`seed + i * 1000`）
4. 全部完成后调用 `postprocess_merge.py` 统一去重和过滤

### 4. `postprocess_merge.py` — 后处理合并

独立的后处理脚本，支持从多 worker 目录合并。

```bash
python postprocess_merge.py \
  --input-dir ./batch_output \ # 输入目录（含 worker 输出文件）
  --output generated_texts.jsonl \  # 最终输出路径
  --semantic-threshold 0.88 \  # 语义去重阈值
  --max-tags 5                 # 单文本最大标签数
```

### 5. `run_1m_gen.sh` — 百万级生产脚本

一键启动 4 实例并行生成 100 万条文本，使用 tmux 管理会话。

```bash
bash run_1m_gen.sh
```

### 6. `tts_client.py` — 可选 TTS 客户端

将生成的文本转为实际音频。

```bash
# 生成单条语音
python tts_client.py generate \
  --input "Hello, this is a test." \
  --voice default \
  --output output.wav

# 批量生成
python tts_client.py batch \
  --input generated_texts.jsonl \
  --output-dir ./audio_output

# 创建自定义声音
python tts_client.py create-voice \
  --ref-audio /path/to/ref.wav \
  --ref-text "Reference transcript" \
  --title "My Voice"
```

---

## API 参考

### `call_llm()` (`higgs_text_gen/llm_client.py`)

核心 LLM 调用函数。

```python
from higgs_text_gen.llm_client import call_llm

results = call_llm(
    prompt="<你的 prompt>",
    model="qwen3.6-27b",
    api_key="EMPTY",
    base_url="http://localhost:8000",
    max_retries=3,
    retry_base_delay=1.0,
    max_tokens=4096,
    temperature=0.85,
)
# 返回: List[Dict] — 解析后的 JSON 对象列表
```

**JSON 解析逻辑：**
1. 去除 `</think>` 内容（Qwen3.6 思考模式）
2. 去除 markdown 代码块（```json```）
3. 找到第一个 `[` 和最后一个 `]` 提取 JSON 数组
4. 退回正则匹配单个 `{"text": "..."}` 对象
5. 失败时自动重试（最多 3 次，指数退避）

### `build_compact_prompt()` (`higgs_text_gen/compact_prompt.py`)

构建紧凑版 prompt（batch pipeline 使用）。

```python
from higgs_text_gen.compact_prompt import build_compact_prompt

prompt = build_compact_prompt(
    scenario_key="daily_chat",
    subscene="问候寒暄",
    length_key="short",
    lang_key="pure_cn",
    emotion="amusement",
    batch_size=16,
    suppression_hint="不要以'你好'开头",  # 可选
    task_id=1,
)
```

### `GenConfig` (`higgs_text_gen/config.py`)

```python
from higgs_text_gen.config import GenConfig

config = GenConfig(
    total_target=100000,
    batch_size=16,
    max_workers=8,
    temperature=0.85,
    seed=42,
)
```

---

## 生产环境大规模生成

### 硬件要求

- 每个 vLLM 实例：~80GB GPU 内存（推荐 H100/A100）
- 4 实例并行：4× 80GB GPU
- 磁盘空间：每 100 万条文本约需 1-2GB

### 性能估算

| 规模 | vLLM 实例数 | 预计耗时 | 推荐 workers |
|------|-------------|----------|-------------|
| 1 万 | 1 | ~10 分钟 | 8 |
| 10 万 | 1 | ~2 小时 | 8 |
| 100 万 | 4 | ~6-8 小时 | 8 |
| 1000 万 | 4 | ~3 天 | 8 per worker |

### 最佳实践

1. **先小批量测试**：用 `generate_single.py` 验证 prompt 效果
2. **梯度增加：** 先跑 1000 条验证管道完整，再放大
3. **使用 --resume**：大规模运行务必开启 checkpoint 恢复
4. **监控去重率**：如果去重率 >60%，降低 `temperature` 或增加多样性轴
5. **定期检查输出质量**：随机采样 JSON 行，验证标签格式和文本自然度

### 输出统计样例

```python
from higgs_text_gen.output import print_statistics

# 打印分布统计
print_statistics(data_list)
# 输出：
#   Distribution of length_type:
#     ultra_short: 14.8%
#     short: 30.1%
#     medium: 30.3%
#     long: 17.9%
#     very_long: 6.9%
#   Distribution of lang_type:
#     pure_cn: 44.9%
#     pure_en: 35.1%
#     ...
```

---

## 故障排查

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| LLM 返回空列表 | Qwen3.6 思考模式输出格式异常 | 检查 `_extract_json()` 日志，调整 `max_tokens` |
| 去重率过高 | 多样性不足 | 增加 `temperature`，降低 `suppression_window_size` |
| 质量过滤器丢弃过多 | SFX 标签后无拟声词 | 检查 prompt 中的标签使用指南 |
| vLLM 连接超时 | 实例负载过高 | 减少 `max_workers`，增加 `retry_base_delay` |
| 内存不足（本地 HF 模式） | 模型太大无法载入 | 使用 vLLM 模式，不要直接 `call_llm_local()` |
| JSON 解析失败 | LLM 输出格式不规范 | 已内置正则 fallback，如需增强参考 `_extract_json()` |
| 语言检测不准 | 跨语言文本混合 | `lang_type` 为 `cn_main`/`en_main` 时正常 |
