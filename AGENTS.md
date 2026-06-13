# AGENTS.md — Higgs Audio

## 环境与安装

- Python 3.10+
- 安装：`pip install -r requirements.txt && pip install -e .`
- Docker 镜像（推荐）：`nvcr.io/nvidia/pytorch:25.02-py3` 或 `25.01-py3`
- 包名为 `boson_multimodal`（非 `higgs_audio`），定义在 `setup.cfg`
- GPU 推理需要至少 24GB 显存
- `higgs_v3_env/`（repo 根目录）：conda 虚拟环境，专用于 v3 TTS 声音复刻流水线的 SGLang-Omni 服务端（已加入 `.gitignore`，不提交）

## 构建与质量检查

- **Lint/Format**：`ruff format --check .`（唯一 CI 步骤）
- CI 工作流：`.github/workflows/test.yml`，在 push/PR 到 `main` 分支时执行，使用 sysmtem python 安装 `ruff==0.12.2` 后检查格式
- ruff 版本锁定 `0.12.2`，配置在 `pyproject.toml`（行宽 119、双引号、py310 目标）
- ruff 额外规则：import 排序（`I`）、pyupgrade（`UP`）、banned API（`os.getenv`/`os.putenv`/`os.unsetenv` 禁止，必须用 `os.environ`）、copyright 检查（`CPY`）
- **禁止** `os.getenv`、`os.putenv`、`os.unsetenv`（banned API），必须用 `os.environ` 访问
- `__init__.py` 中 F401（unused import）被忽略（用于 re-export）
- **无 mypy/typecheck 配置，无 pytest 目录，无任何 `test_*.py` 文件**
- `pyproject.toml` 中 `extend-select` 额外启用了 `B009`（static getattr）、`B010`（static setattr）
- `pyproject.toml` 中 ignore 了 `E501`（行宽由 ruff-format 处理）、`E741`（歧义变量名）、`W605`（非法转义序列）、`UP007`（X | Y 类型注解）
- `pyproject.toml` 中 isort 配置：`lines-after-imports = 2`、`known-first-party = ["character_tuning"]`

## 仓库定位

- **本仓库是 v2 / v2.5 代码**。v3 不依赖此仓库，权重在 HuggingFace `bosonai/higgs-audio-v3-tts-4b`
- 核心模块在 `boson_multimodal/` 下，安装后即为顶级 Python 包
- `setup.cfg` 排除 `tests*` 和 `training*` 目录
- `boson_multimodal/__init__.py` 是**空文件**
- 根目录 `README.md` 现在是 v3 入口页面，v2/v2.5 文档已移至 `README_V2.md`，v3 详细文档在 `README_V3.md`
- 贡献和支持指南：`SUPPORT_GUIDELINES.md`
- 技术博客：`tech_blogs/ARCHITECTURE_BLOG.md`（DualFFN 架构）、`tech_blogs/TOKENIZER_BLOG.md`（25fps audio tokenizer）
- **重要 `.gitignore` 模式**：`clone_workdir/`（v3 声音复刻工作目录）、`*.wav`（生成的音频）、`*.jsonl`（数据集文件）、`higgs_audio_v3_text_generator/batch_output/`、`child_voice_clone_output_higgs`——这些不被跟踪，不要提交

## v3 文本生成子项目（`higgs_audio_v3_text_generator/`）

独立于 v2/v2.5 核心代码的子项目，用于为 Higgs Audio v3 TTS 训练批量生成多样化、打标的文本数据。

### 定位与依赖

- **与 `boson_multimodal/` 完全独立**，不 import 主项目代码
- 自己独立的 `requirements.txt`（仅 3 个依赖：`requests`, `tqdm`, `loguru`），无需 torch/transformers
- LLM 推理通过外部 vLLM 服务调用（OpenAI 兼容 API，`/v1/chat/completions`），不在本进程中加载模型
- 默认使用 Qwen3.6-27B 作为文本生成模型
- 硬件需求：约 80GB GPU per vLLM instance（H100/A100 推荐）

### 目录结构

```
higgs_audio_v3_text_generator/
├── .env.example                        # 3 个 vLLM 环境变量
├── requirements.txt                    # requests, tqdm, loguru
├── README.md                           # 654 行中文完整文档
├── generate_single.py                  # 交互式单批次测试入口
├── run_batch_generation.py             # 单实例批量 pipeline（核心）
├── run_parallel_batch.py               # 4 实例并行 pipeline
├── postprocess_merge.py                # 最终去重 + 质量过滤合并
├── run_1m_gen.sh                       # tmux 一键 100 万文本生产
└── higgs_text_gen/                     # 核心 Python 子包
    ├── __init__.py                     # Re-export 所有公开 API
    ├── config.py                       # GenConfig dataclass（中心配置）
    ├── scenarios.py                    # 10 个场景、21 种情绪、长度/语言规格
    ├── tags.py                         # 43 个标签（情绪/风格/SFX/韵律），标签格式验证
    ├── tag_guide.py                    # 标签指南构建器（注入 prompt），combo 验证
    ├── compact_prompt.py               # 紧凑型 prompt builder（10 轴多样性，生产使用）
    ├── prompt_builder.py               # 长格式 prompt builder（旧版）
    ├── diversity.py                    # 多样性轴池 + 指令构建（prompt_builder.py 使用）
    ├── llm_client.py                   # vLLM API 客户端（仅用 urllib，无第三方）
    ├── llm_local.py                    # 直接 HF 加载模型（Qwen3.6-27B-FP8，未被生产使用）
    ├── task_generator.py               # 从 GenConfig 生成任务列表
    ├── worker.py                       # Worker 函数
    ├── dedup.py                        # 3 层去重：MD5 + Jaccard + SequenceMatcher
    ├── quality_filter.py               # 质量过滤（标签格式、SFX 配对、完整性、敏感词、长度、标签计数）
    ├── text_clean.py                   # 去除标签 + 附加 clean_text
    ├── output.py                       # JSONL 格式化 + 统计打印
    ├── checkpoint.py                   # 断点续跑（每 5 个 task 保存一次）
    └── tts_client.py                   # 可选 TTS 客户端（Boson API，独立于 pipeline）
```

### 数据流（整体 pipeline）

```
GenConfig → TaskGenerator (加权随机场景/情绪/长度/语言, 多样性保护)
    → CompactPromptBuilder (10 轴多样性注入 + 标签指南 + 抑制提示)
    → vLLM (OpenAI /v1/chat/completions, ThreadPool 并发)
    → 在线 MD5 去重 (每批内部 + 跨所有已生成文本)
    → Checkpoint (每 5 task)
    → 后处理: 精确去重 → Jaccard 语义去重 (3-gram shingle, 阈值 0.88)
    → 质量过滤 (标签格式/SFX 拟声词配对/完整性/敏感词/长度/标签计数)
    → JSONL 输出 + 分布统计
```

### 关键配置项（`GenConfig`）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `total_target` | 10000 | 目标生成数量 |
| `batch_size` | 8 | 每次 LLM 调用的批大小 |
| `max_workers` | 1 | ThreadPool 并发数 |
| `temperature` | 0.85 | LLM 温度 |
| `max_tokens` | 1536 | LLM 输出 token 上限 |
| `scenario_distribution` | 相对权重 | 10 个场景的分布 |
| `length_distribution` | 相对权重 | 5 种长度分布 |
| `lang_mix_distribution` | 相对权重 | 4 种语言混合分布 |
| `stress_test_ratio` | 0.10 | ASR 压力测试占比 |
| `semantic_dedup_threshold` | 0.88 | Jaccard 语义去重阈值 |
| `max_tags_per_text` | 5 | 每条文本最大标签数 |
| `max_same_tag_repeat` | 2 | 同标签最大重复次数 |

### 标签系统（43 tags）

- **情绪标签 (21)**：amusement, anger, anxiety, awe, compassion, contentment, determination, disappointment, disgust, embarrassment, enthusiasm, fear, gratitude, joy, love, melancholy, neutral, pride, relief, sadness, surprise
- **风格标签 (3)**：singing（需包含歌词/旋律文本），shouting（必须全大写），whispering
- **SFX 标签 (9)**：laughter, sigh, cough, crying, screaming, humming, sniff, sneeze, burping——其中 8 个（除 sniff）**必须在标签后 10 字符内出现对应的拟声词**
- **韵律标签 (10)**：speed（4 级）、pitch（2 级）、pause（2 级）、expressive（2 级）
- 互斥约束：speed 之间互斥、pitch 之间互斥、shouting 与 whispering 互斥、laughter 与 crying 互斥

### 10 轴多样性体系（`compact_prompt.py`）

每条批处理项注入 10 个维度的多样性：
- **Persona profiles** (12)、**Topic seeds** (每场景 10-15)、**Opening types** (8)、**Focus types** (10)、**Sensory types** (6)、**Places** (20)、**Times** (10)、**Register types** (8)、**Dialogue states** (9)、**Sentence structures** (8) + Clause patterns (5) + Utterance rhythms (5)

### 抑制机制

- **前缀抑制**：跟踪最近 500 条文本的前 4 字符前缀，出现频率 ≥1.25% 时注入抑制提示
- **标签多样性抑制**：双窗口（最近 300 + 全局 5000）跟踪 SFX/style/prosody 标签使用率，缺失过多时注入提示
- **温度 jitter**：每批 temperature 在 `config.temperature ± 0.15` 内随机抖动，夹紧到 `[0.65, 1.0]`

### 运行方式

```bash
# 交互式单批测试
python generate_single.py --scenario daily_chat --emotion enthusiasm --length medium \
    --lang pure_cn --count 16 --temperature 0.85 --output sample.jsonl

# 小规模批量
python run_batch_generation.py --total 10000 --batch-size 16 --workers 8 --temperature 0.85 \
    --output generated_texts.jsonl --checkpoint checkpoint.jsonl

# 大规模并行（需 4 个 vLLM 实例分别运行在 8000-8003 端口）
python run_parallel_batch.py --total 1000000 --batch-size 8 --workers 8 --num-instances 4 \
    --temperature 0.85 --seed 42 --output-dir ./batch_output

# 并行后合并去重
python postprocess_merge.py --input-dir ./batch_output --output generated_texts_final.jsonl \
    --semantic-threshold 0.88 --max-tags 5

# 一键 100 万生产（tmux）
bash run_1m_gen.sh
```

### JSONL 输出格式

每行一条 JSON，包含字段：`text`（原始标签文本）、`clean_text`（去标签）、`scenario`、`subscene`、`emotion`、`length_type`、`lang_type`、`language`、`tags_used`、`tag_count`、`char_count`、`task_id`

### tts_client.py（独立可选的 TTS 客户端）

- 将生成的文本发送到 Boson Higgs Audio v3 TTS API 进行语音合成
- 支持 `generate`、`batch`、`create-voice` 子命令
- 需要 `BOSON_API_KEY` 环境变量

## v3 TTS 声音复刻流水线（`v3_tts_clone/`）

用于把总时长不足 1 小时且音频不少于 20 条的说话人，通过 Higgs Audio v3 TTS 批量复刻到 1 小时水平。

### 核心文件

```
v3_tts_clone/
├── README.md                  # 174 行详细文档
├── 01_stats_speakers.py       # Step 1: 统计说话人时长 → speaker_duration_stats.csv
├── 02_asr_launch.sh           # Step 2: 启动多 GPU ASR 服务
├── 02_asr_worker.py           # Step 2: ASR 转写 worker（Qwen3-ASR 1.7B）
├── 03_launch_servers.sh       # Step 3: 启动多 GPU SGLang-Omni TTS 服务
└── 03_tts_clone.py            # Step 3: TTS 克隆客户端
```

### 环境隔离

| Step | 环境 | 说明 |
|------|------|------|
| Step 1 | 系统 Python | 仅需 librosa、soundfile |
| Step 2 | `qwen3-asr` conda env | Qwen3-ASR 1.7B + transformers 4.57.6 |
| Step 3 服务端 | `higgs_v3_env` conda env | SGLang-Omni editable 安装，每 GPU 一个进程 |
| Step 3 客户端 | 系统 Python | 仅需 requests、soundfile、numpy |

### 排除的数据集

`childmandarin`, `child207m-korean-filtered`, `chineseenglishchildren`, `king-asr-725`, `kingasr612`, `speechocean762`

### 当前生产配置

- 输出工作目录：`clone_workdir/`（已加入 `.gitignore`，不要提交）
- 正式输出目录：`/root/group-shared/voiceprint/data/speech/speaker_diarization/merged_datasets_20250610_vad_segments_mtfaa_enhanced_extend_kid_withclone_addlibrilight_1130/audio_higgs_audio_v3_tts_clone`
- 后台任务使用 `tmux` session `higgs_step3`
- 8 个 SGLang-Omni 服务：GPU `0-7` → 端口 `8000-8007`
- 客户端并发：`--workers-per-server 16`（总并发约 128）
- 请求模式：非流式 `/v1/audio/speech`；每条完整返回后写入 `clone_XXXX.wav`
- 文本选择：不按 dataset 或 speaker 语言过滤；所有 speaker 从完整文本池随机抽样，覆盖中文、英文、中英混合
- 进度日志：`clone_workdir/step3_progress_tmux.log`
- 客户端日志：`clone_workdir/step3_tmux_client.log`
- 服务端日志：`clone_workdir/step3_tmux_servers.log`

### SGLang-Omni 本地安装

- 源码固定在 `/root/code/github_repos/sglang-omni`
- `higgs_v3_env` 使用 editable 安装，**必须加 `--no-deps`**：

```bash
/root/code/github_repos/higgs-audio/higgs_v3_env/bin/python3 \
    -m pip install -e /root/code/github_repos/sglang-omni --no-deps
```

- 不要让 pip 重新解析依赖，避免升级 torch/transformers 等大包。

### 常用命令

```bash
# 查看后台任务
tmux attach -t higgs_step3

# 查看进度
tail -f clone_workdir/step3_progress_tmux.log

# 启动 8 卡服务
bash v3_tts_clone/03_launch_servers.sh "0,1,2,3,4,5,6,7" /root/models/higgs-audio-v3-tts-4b 8000

# 全量客户端（当前生产参数）
python v3_tts_clone/03_tts_clone.py \
    --stats-csv ./clone_workdir/speaker_duration_stats.csv \
    --texts-jsonl higgs_audio_v3_text_generator/batch_output_v2/generated_texts_final.jsonl \
    --output-root /root/group-shared/voiceprint/data/speech/speaker_diarization/merged_datasets_20250610_vad_segments_mtfaa_enhanced_extend_kid_withclone_addlibrilight_1130/audio_higgs_audio_v3_tts_clone \
    --base-port 8000 \
    --num-servers 8 \
    --workers-per-server 16
```

## 童声批量复刻流水线（v2，独立工具）

基于 v2 模型（`HiggsAudioServeEngine`）的童声批量复刻，与 `SoulX-Podcast` 对比。

### 核心文件

- `batch_child_voice_clone_higgs.py`：批处理脚本，从 BAAI-ChildMandarin 数据集随机采样 100 个样本进行语音克隆
- `run_child_voice_clone_higgs.sh`：启动脚本
- `CHILD_VOICE_CLONE_README.md`：148 行详细文档
- `COMPARISON_WITH_SOULX.md`：与 SoulX-Podcast 的对比分析（模型架构、API 差异、输出采样率等）
- 输出目录：`child_voice_clone_output_higgs/`（已加入 `.gitignore`）

```bash
# 一键启动
./run_child_voice_clone_higgs.sh

# 或直接调用 Python
python3 batch_child_voice_clone_higgs.py \
    --model-path "bosonai/higgs-audio-v2-generation-3B-base" \
    --audio-tokenizer-path "bosonai/higgs-audio-v2-tokenizer" \
    --output-dir "./child_voice_clone_output_higgs" \
    --num-samples 100 --random-seed 42 --seed 1988
```

## 核心架构

```
boson_multimodal/
  __init__.py                         # 空文件
  constants.py                        # AUDIO_IN_TOKEN ("<|AUDIO|>"), AUDIO_OUT_TOKEN ("<|AUDIO_OUT|>"), EOS_TOKEN ("<|end_of_text|>")
  data_types.py                       # Message, AudioContent, TextContent, ChatMLSample (dataclass)
  model/higgs_audio/
    __init__.py                       # 注册 AutoConfig("higgs_audio"), AutoModel (import 时自动执行)
    configuration_higgs_audio.py      # HiggsAudioConfig, HiggsAudioEncoderConfig
    modeling_higgs_audio.py           # HiggsAudioModel (核心模型，2289 行，继承 GenerationMixin)
    common.py                         # HiggsAudioPreTrainedModel 基类 (继承 PreTrainedModel)
    audio_head.py                     # HiggsAudioDecoderProjector (text lm_head + audio lm_head)
    custom_modules.py                 # PartiallyFrozenEmbedding, PartiallyFrozenLinear (训练时拆分冻结/可训练部分)
    cuda_graph_runner.py              # CUDAGraphRunner (捕获 CUDA graph 用于推理加速)
    utils.py                          # revert_delay_pattern, build_delay_pattern_mask, merge_input_ids_with_audio_features, DeepSpeed Ulysses 工具
  audio_processing/                   # 音频 tokenizer（部分代码来自 xcodec，MIT License）
    higgs_audio_tokenizer.py          # HiggsAudioTokenizer, load_higgs_audio_tokenizer()
    semantic_module.py                # Encoder/Decoder for semantic features
    descriptaudiocodec/               # DAC encoder/decoder（第三方 from descript-audio-codec）
    quantization/                     # ResidualVectorQuantizer（第三方）
    LICENSE                           # 第三方代码 license
  serve/
    serve_engine.py                   # HiggsAudioServeEngine（推理入口，唯一公开 API）
    utils.py                          # pcm/format 转换、文本预处理、split_interleaved_delayed_audios 等
  dataset/chatml_dataset.py           # ChatMLDatasetSample, prepare_chatml_sample(), prepare_chatml_dataframe()
  data_collator/higgs_audio_collator.py # HiggsAudioSampleCollator (whisper 编码、delay pattern、padding)
```

## 音频适配器架构（audio_adapter_type）

模型支持 3 种音频适配器架构，由 `HiggsAudioConfig.audio_adapter_type` 控制：

- **`stack`**：在 LLM backbone 之后堆叠额外的 Transformer 层，使用 `LlamaDecoderLayer`
- **`dual_ffn`**：在 LLM backbone 指定层将 text FFN 替换为双路 FFN（text FFN + audio FFN），通过 `audio_dual_ffn_layers` 指定
- **`dual_ffn_fast_forward`**：类似 dual_ffn，但非 dual_ffn 层的 audio hidden states 直接 fast-forward 跳过该层，减少计算开销

dual_ffn* 时使用 `HiggsAudioDualFFNDecoderLayer`；stack 时使用 `LlamaDecoderLayer`。
`HiggsAudioDualFFNDecoderLayer` 中 text/audio hidden states 先一起过共享 attention，再分开过各自的 FFN，最后 reorder 回原位。

## 关键入口

### HiggsAudioServeEngine（推理唯一入口）

```python
from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine, HiggsAudioResponse
from boson_multimodal.data_types import ChatMLSample, Message

serve_engine = HiggsAudioServeEngine(
    model_name_or_path,      # 比如 "bosonai/higgs-audio-v2-generation-3B-base"
    audio_tokenizer_name_or_path,  # 比如 "bosonai/higgs-audio-v2-tokenizer"
    tokenizer_name_or_path=None,   # 默认同 model_name_or_path
    device="cuda",
    torch_dtype="auto",
    kv_cache_lengths=[1024, 4096, 8192],
)
```

- 构造函数自动下载模型、tokenizer、audio tokenizer（从 HuggingFace Hub）；创建多个 bucket 的 `StaticCache`；若 device=cuda 则自动 `capture_model()`
- `serve_engine.generate(chat_ml_sample=..., max_new_tokens=..., temperature=..., top_p=..., ...)` → 同步返回 `HiggsAudioResponse`
- `serve_engine.generate_delta_stream(...)` → 异步返回 `AsyncGenerator[HiggsAudioStreamerDelta]`
- `HiggsAudioResponse` 含 `audio`（np.ndarray）、`sampling_rate`、`generated_text`、`generated_audio_tokens`、`usage`
- CLI 示例：`python3 examples/generation.py --transcript ... --ref_audio belinda --out_path out.wav`
- 快速上手：`quick_start.py`（单文件最小示例，38 行）
- vLLM 部署：`examples/vllm/` 提供 OpenAI 兼容 API（`/v1/audio/speech`、`/v1/chat/completions`）

### HiggsAudioModel（HuggingFace 模型）

- `model.set_audio_special_tokens(tokenizer)` — **必须调用**，注册 `<|audio_out_bos|>` 和 `<|audio_eos|>` 的 token ID
- `model.generate()` — 自定义 override，**不走标准 `GenerationMixin.generate()` 流程**
- `model.capture_model(kv_caches)` — CUDA graph 预捕获，每个 kv_cache_length × 2 个 graph（text decode + audio decode）

## 自定义 generate() 流程（关键陷阱）

`HiggsAudioModel.generate()` **完全覆盖**了 `GenerationMixin.generate()`，核心在自定义的 `_sample()` 方法中：

### 三种生成模式（GenerationMode enum，定义于 modeling_higgs_audio.py:45）

```
TEXT              → 生成普通文本 token
AUDIO_INIT        → 遇到 <|audio_out_bos|>，开始声频生成模式
AUDIO_IN_PROGRESS → 正在生成声频 token
```

### 生成循环（`_sample()` 方法，约 1624-1960 行）

1. 检查 `input_ids[0][-1]` 判断当前 mode
2. 文本模式下：从 `outputs.logits` 采样文本 token，若遇到 `audio_out_bos_token_id` 则输出 `<|AUDIO_OUT|>` + `audio_stream_bos_id` tokens 触发声频生成
3. 声频模式下：从 `outputs.audio_logits` 采样声频 token（shape `(num_codebooks, audio_codebook_size)`）
4. 声频模式下每个 step 也生成一个 `<|AUDIO_OUT|>` 作为文本 token
5. `audio_out_ids` 逐步累积声频 token 序列
6. **仅支持 batch_size=1**（代码中有 assert）
7. **不使用 HF 的 LogitsWarper、StoppingCriteria 等标准组件**，而是在 `_sample_audio_tokens()` / `_sample_text_tokens()` 中手动处理
8. 文本 token 采样自定义 temperature、top_k、top_p，与 HF 标准行为**不完全一致**

### 声频生成的 delay pattern 处理

声频生成时在 `_sample_audio_tokens()` 中同时处理 delay pattern：
- `num_delay` 跟踪已延迟的 codebook 数
- `num_remaining_delays` 跟踪还需等待关闭的 codebook 数
- 当所有 codebook 都生成 `audio_stream_eos_id` 后，输出 `audio_eos_token_id` 结束声频段

### 声频生成的 RAS（Repetition Aware Sampling）

仅声频模式下生效：
- `ras_win_len`（默认 7）：回溯窗口检查重复
- `ras_win_max_num_repeat`（默认 2）：超过此次数则 resample（无 temperature）

## 特殊 token 与 token ID

### ChatML 格式 tokens
- `boson_multimodal/constants.py`: `AUDIO_IN_TOKEN="<|AUDIO|>"`, `AUDIO_OUT_TOKEN="<|AUDIO_OUT|>"`, `EOS_TOKEN="<|end_of_text|>"`

### Llama-3.1-8B-Instruct reserved special tokens 映射
| Token | ID | 用途 |
|-------|-----|------|
| `<\|audio_bos\|>` | 128011 | 声频输入起始 |
| `<\|audio_eos\|>` | 128012 | 声频输入/输出结束 |
| `<\|audio_out_bos\|>` | 128013 | 声频输出起始（触发声频生成） |
| `<\|AUDIO\|>` | 128015 | 声频输入占位符（替换为 whisper features） |
| `<\|AUDIO_OUT\|>` | 128016 | 声频输出占位符（替换为声频 codebook tokens） |
| `<\|audio_out_bos\|>` | 通过 `set_audio_special_tokens()` 动态注册 | 同上 |
| `<\|audio_eos\|>` | 通过 `set_audio_special_tokens()` 动态注册 | 同上 |
| `pad_token_id` | 128001 | padding token |

### Codebook 维度 tokens
- `audio_stream_bos_id=1024`（codebook 维度标记声频流开始）
- `audio_stream_eos_id=1025`（codebook 维度标记声频流结束）
- `audio_codebook_size` **实际值为 `config.audio_codebook_size + 2`**（因为有 stream_bos/stream_eos）

## 音频 tokenizer（HiggsAudioTokenizer）

### 关键属性
- 内部结构：DAC Encoder/Decoder + Semantic Encoder/Decoder（Hubert teacher）+ RVQ/ResidualFSQ quantizer
- 默认路径：`bosonai/higgs-audio-v2-tokenizer`
- `tokenizer.sampling_rate`：采样率（如 16000）
- `tokenizer.tps`（tokens per second）：`frame_rate`（如 50 Hz）
- `tokenizer.num_codebooks`：codebook 数量（如 12）
- `tokenizer.codebook_size`：返回 `quantizer_dim`（**注意不是实际 codebook size，实际 size 需 +2**）

### encode/decode 签名
```python
# encode: 输入 numpy wav 或文件路径，输出 shape (num_codebooks, seq_len)
vq_code = tokenizer.encode(audio_path_or_wv, sr=None)
# decode: 输入 shape (batch, num_codebooks, seq_len)，输出 numpy shape (batch, channels, samples)
wv = tokenizer.decode(vq_code.unsqueeze(0))[0, 0]
```
- encode 内部用 librosa 做 resample（不是 torchaudio）
- encode 返回的 code 不含 stream_bos/stream_eos
- decode 前必须先 `revert_delay_pattern()`（如果使用了 delay pattern）

### MPS（Apple Silicon）陷阱
- **MPS 上必须将 audio tokenizer 放在 CPU**：量化层的 embedding 操作在 MPS 上受限
- MPS 不支持 StaticCache / CUDA graph，需要禁用
- 相关处理见 `examples/generation.py:672-677`

## delay pattern

- 模型配置 `use_delay_pattern` 控制（论文 "Simple and Controllable Music Generation"）
- **encode 方向**：`build_delay_pattern_mask()` 在 collator 中将 codebook 序列做 delay 偏移，每行前插 BOS token、末补 PAD token
- **decode 方向**：**必须调用 `revert_delay_pattern()` 恢复原始顺序**，然后 `[:, 1:-1]` 剪掉首尾
- 参考：`serve_engine.py:401`、`examples/generation.py:356`、`modeling_higgs_audio.py:1502-1575`
- `_sample_audio_tokens()` 中声频生成时实时维护 `num_delay` 和 `num_remaining_delays`

## KV Cache

- 使用 `StaticCache`（`transformers.cache_utils`），需手动创建及 `reset()`
- 多 bucket 大小（默认 `[1024, 4096, 8192]`），运行时自动将小 cache 复制到大 cache（`_copy_kv_cache()`，在 `_update_model_kwargs_for_generation()` 中实现）
- CUDA 设备上执行 `model.capture_model()` 捕获 CUDA graph：每个 kv_cache_length × 2（text decode + audio decode 各一个）
- CUDA graph 在 `_forward_core`（即 layer loop）级别捕获，不包含 `audio_decoder_proj`（head 仍走正常 forward）
- MPS 不支持 StaticCache / CUDA graph

## 前向传播数据流

### `HiggsAudioModel.forward()` 流程（约 1142-1417 行）

1. `embed_tokens(input_ids)` → text embeddings
2. `_apply_audio_tower(audio_features)` → whisper encoder + projector → audio feature embeddings
3. `_embed_audio_ids(audio_in_ids)` / `_embed_audio_ids(audio_out_ids)` → audio codebook embeddings
4. `merge_input_ids_with_audio_features()` 将 text embedding + audio feature embedding + audio codebook embedding 合并
5. 如果合并后 seq_len 超过当前 KV cache，自动切换更大 bucket
6. 生成 causal mask（`_update_causal_mask`）+ audio_discrete_codes_mask
7. 若使用 static cache 则预计算 `fast_forward_attention_mask` 和 `audio_attention_mask`
8. layer loop → `_forward_core()`（或被 CUDA graph runner 替代）
9. `self.norm(hidden_states)` → final norm
10. `audio_decoder_proj(hidden_states, audio_out_mask)` → text logits + audio logits

### `merge_input_ids_with_audio_features()` 关键行为
- 找到 `input_ids` 中所有 `<|AUDIO|>` 和 `<|AUDIO_OUT|>` token
- 将每个 `<|AUDIO|>` 替换为 whisper features（可能 + 离散 audio code 如果 `encode_audio_in_tokens=True`）
- 将每个 `<|AUDIO_OUT|>` 替换为 audio codebook embeddings
- 重新计算 position_ids 和 attention_mask
- 自动左 padding 并用 `round_to` 对齐（训练时 round_to=8，推理时 round_to=1）

### `prepare_chatml_sample()` 约定
- 每条消息按 ChatML 格式编码：
  - `system/user` 消息格式：`<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>`
  - `assistant` 消息格式：`<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>`
  - 连续的 assistant 消息用 `<|eom_id|>` 分隔
- audio-in（user/system 消息中的音频）：生成 `<|audio_bos|><|AUDIO|><|audio_eos|>` token 序列
- audio-out（assistant 消息中的音频）：生成 `<|audio_out_bos|><|AUDIO_OUT|><|audio_eos|>` token 序列
- `label_ids`：assistant 消息和 system 消息（当 `start_index` 匹配时）正常 label，其余为 -100
- `speaker_id` 从 `sample.speaker` 或 `sample.misc["speaker"]` 中提取

### `HiggsAudioSampleCollator()` 关键行为
- 若 `encode_whisper_embed=True`：对长音频做 chunk（默认 30s），自动复制 `<|audio_bos|><|AUDIO|><|audio_eos|>` token 序列
- 为每个 audio-in 的 codebook 序列首尾插入 `audio_stream_bos_id` 和 `audio_stream_eos_id`
- 若 `use_delay_pattern=True`：调用 `build_delay_pattern_mask()` 做 delay 偏移
- 若 `return_audio_in_tokens=False`（推理时）：audio_in_ids 置为 None（只用 whisper features）
- padding 默认 `pad_left=False`（推理和训练均用左 padding），单样本时 `left_padding=False`

## Whisper 编码器

- `encode_whisper_embed` 配置控制是否使用 whisper encoder 编码音频为 mel 特征
- whisper 模型：`openai/whisper-large-v3-turbo`
- whisper forward 被 monkey-patch 以支持 zero-shape tensor（`_whisper_encoder_zero_shape_forward`，`modeling_higgs_audio.py:53`），因为原始 whisper encoder 的 `_shape` 方法在 bsz=0 时有 bug
- **monkey-patch 在每次 `_apply_audio_tower()` 调用时动态应用和恢复**，不是一次性全局 patch（见 `modeling_higgs_audio.py:62` 注释及 `:947` 调用点）
- whisper encoder 不支持 flash_attention_2，强制使用 sdpa

## 依赖版本约束

- `transformers>=4.45.1,<4.47.0`（**注意上限 <4.47**，不能升级）
- `ruff==0.12.2`（精确锁定）
- `boto3==1.35.36`（精确锁定）
- `torch`、`torchaudio`、`torchvision` 无版本约束，随 Docker 镜像提供
- 音频处理依赖 `vector_quantize_pytorch`、`descript-audio-codec`、`librosa`
- 其他：`dacite`、`s3fs`、`json_repair`、`pandas`、`pydantic`、`loguru`、`pydub`、`omegaconf`、`click`、`langid`、`jieba`、`accelerate>=0.26.0`

## 训练相关模块

- `PartiallyFrozenEmbedding` / `PartiallyFrozenLinear`：将 embedding/linear 层拆分为冻结部分和可训练部分
- `model.freeze_llm(freeze_embed=True, freeze_embed_until_idx=None)`：冻结 LLM backbone（含 self_attn、mlp、layernorm）
- `model.freeze_text_head(freeze_text_head_until_idx=None)`：冻结 text lm_head
- `model.freeze_audio_tower()` / `model.freeze_audio_encoder_proj()`：冻结音频塔
- `model.merge_weights_from_checkpoint(checkpoint_dir, merged_output_dir)`：训练后将 PartiallyFrozen 模块 merge 回普通 Embedding/Linear
- `support_deepspeed_ulysses` 装饰器：为 module 添加 `sp_size`/`sp_rank`/`sp_group` 属性
- 数据并行工具：`drop_tokens`/`gather_tokens`/`sequence_chunking_per_rank`

## 流式生成（`generate_delta_stream`）

- 在后台线程运行 `model.generate()`，通过 `AsyncHiggsAudioStreamer` 流式输出
- `AsyncHiggsAudioStreamer` 继承 `BaseStreamer`，用 `asyncio.Queue` 在线程间传递 delta
- 文本 delta：`HiggsAudioStreamerDelta(text=..., text_tokens=...)`
- 声频 delta：`HiggsAudioStreamerDelta(audio_tokens=...)`，shape 为 `(audio_num_codebooks,)`

## 第三方代码

- `boson_multimodal/audio_processing/` 包含来自 xcodec 的第三方代码（LICENSE 文件在该目录内）
- 修改此目录时注意保留原始 license 声明
- DAC 编解码器来自 `descript-audio-codec`（`descriptaudiocodec/` 目录）
- `quantization/` 目录：ResidualVectorQuantizer

## Ruff 禁止事项速查

- `os.getenv` / `os.putenv` / `os.unsetenv` → 改用 `os.environ`
- `__init__.py` 中 unused import → F401 豁免（用于 re-export）
- 所有源文件需要包含 Copyright 头（ruff CPY 规则检查），不要删除或修改已有版权声明

## 常见操作速查

```bash
# Lint 检查（唯一 CI 步骤 / 提交前必跑）
ruff format --check .

# 自动修复
ruff format .

# 安装
pip install -r requirements.txt && pip install -e .

# 运行推理示例（默认 cuda:0）
python3 examples/generation.py --transcript "Hello world." --out_path out.wav

# 带参考语音
python3 examples/generation.py --transcript "..." --ref_audio belinda --out_path out.wav

# 多说话人
python3 examples/generation.py --transcript examples/transcript/multi_speaker/en_argument.txt \
    --ref_audio belinda,broom_salesman --ref_audio_in_system_message --chunk_method speaker --out_path out.wav
```
