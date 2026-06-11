# AGENTS.md — Higgs Audio

## 环境与安装

- Python 3.10+
- 安装：`pip install -r requirements.txt && pip install -e .`
- Docker 镜像（推荐）：`nvcr.io/nvidia/pytorch:25.02-py3` 或 `25.01-py3`
- 包名为 `boson_multimodal`（非 `higgs_audio`），安装在 `setup.cfg` 中定义
- GPU 推理需要至少 24GB 显存

## 构建与质量检查

- **Lint/Format**：`ruff format --check .`（CI 只检查这一项，无单元测试）
- ruff 版本锁定为 `0.12.2`，配置在 `pyproject.toml`（行宽 119、双引号、3.10 目标）
- 无 mypy/typecheck 配置，无 pytest 目录，无任何 `test_*.py` 文件
- ruff 额外规则：import 排序（`I`）、pyupgrade（`UP`）、banned API（`os.getenv` 等禁止）、copyright 检查（`CPY`）

## 仓库定位

- **本仓库是 v2 / v2.5 代码**。v3 不依赖此仓库，权重在 HuggingFace `bosonai/higgs-audio-v3-tts-4b`
- v3 独立子项目：`higgs_audio_v3_text_generator/`（文本生成相关，有独立 requirements.txt）
- 核心模块在 `boson_multimodal/` 下，安装后即为顶级 Python 包
- `setup.cfg` 排除 `tests*` 和 `training*` 目录

## 核心架构

```
boson_multimodal/
  __init__.py                          # 空文件
  constants.py                         # AUDIO_IN_TOKEN, AUDIO_OUT_TOKEN, EOS_TOKEN
  data_types.py                        # Message, AudioContent, TextContent, ChatMLSample
  model/higgs_audio/
    __init__.py                        # 注册 AutoConfig("higgs_audio"), AutoModel
    configuration_higgs_audio.py       # HiggsAudioConfig, HiggsAudioEncoderConfig
    modeling_higgs_audio.py            # HiggsAudioModel (HuggingFace 风格, 继承 GenerationMixin)
    common.py                          # HiggsAudioPreTrainedModel 基类
    audio_head.py                      # HiggsAudioDecoderProjector (text lm_head + audio lm_head)
    custom_modules.py                  # PartiallyFrozenEmbedding, PartiallyFrozenLinear (训练用)
    cuda_graph_runner.py               # CUDAGraphRunner (捕获 CUDA graph 用于推理加速)
    utils.py                           # revert_delay_pattern, build_delay_pattern_mask, merge_input_ids_with_audio_features 等
  audio_processing/                    # 音频 tokenizer（部分代码来自 xcodec，MIT License）
    higgs_audio_tokenizer.py           # HiggsAudioTokenizer, load_higgs_audio_tokenizer()
    semantic_module.py                 # Encoder/Decoder for semantic features
    descriptaudiocodec/                # DAC encoder/decoder（第三方）
    quantization/                      # ResidualVectorQuantizer（第三方）
  serve/
    serve_engine.py                    # HiggsAudioServeEngine（推理入口）
    utils.py                           # pcm/format 转换等工具
  dataset/chatml_dataset.py            # ChatMLDatasetSample, prepare_chatml_sample()
  data_collator/higgs_audio_collator.py # HiggsAudioSampleCollator (含 whisper 编码、delay pattern 等)
```

## 音频适配器架构（audio_adapter_type）

模型支持 3 种音频适配器架构，由 `HiggsAudioConfig.audio_adapter_type` 控制：

- **`stack`**：在 LLM backbone 之后堆叠额外的 Transformer 层处理音频生成
- **`dual_ffn`**：在 LLM backbone 的指定层将 text FFN 替换为双路 FFN（text FFN + audio FFN），通过 `audio_dual_ffn_layers` 指定哪些层插入
- **`dual_ffn_fast_forward`**：类似 dual_ffn，但非 dual_ffn 层的 audio hidden states 直接 fast-forward 到下一层，减少计算开销

配置为 `stack` 时使用 `LlamaDecoderLayer`；`dual_ffn*` 时使用 `HiggsAudioDualFFNDecoderLayer`。

## 关键入口

- **推理**：`boson_multimodal.serve.serve_engine.HiggsAudioServeEngine`
  - 构造函数参数：`model_name_or_path`, `audio_tokenizer_name_or_path`，可选 `tokenizer_name_or_path`（默认同 `model_name_or_path`）、`device`、`torch_dtype`、`kv_cache_lengths`
  - 非流式：`serve_engine.generate(chat_ml_sample=..., max_new_tokens=..., temperature=..., top_p=..., ras_win_len=..., ...)`
  - 流式：`serve_engine.generate_delta_stream(...)`，返回 `AsyncGenerator[HiggsAudioStreamerDelta]`
  - 返回类型：`HiggsAudioResponse`（含 `audio`, `sampling_rate`, `generated_text`, `generated_audio_tokens`, `usage` 等字段）
- **CLI 示例**：`python3 examples/generation.py --transcript ... --ref_audio belinda --out_path out.wav`
- **快速上手**：`quick_start.py`（单文件最小示例）
- **vLLM 部署**：`examples/vllm/` 目录提供 OpenAI 兼容 API 服务
- **v3 文本生成**：`higgs_audio_v3_text_generator/` 是独立的 v3 子项目

## 重要陷阱与约定

### 音频 tokenizer
- **MPS (Apple Silicon) 上必须将 audio tokenizer 放在 CPU**：量化层中的 embedding 操作在 MPS 上受限
  - 参考：`examples/generation.py:672` 和 `serve_engine.py:223` 中 `load_higgs_audio_tokenizer` 的 device 参数
- tokenizer 默认路径：`bosonai/higgs-audio-v2-tokenizer`
- tokenizer 内部结构：DAC encoder/decoder + Semantic Encoder/Decoder + ResidualFSQ / ResidualVectorQuantizer

### delay pattern
- 模型配置 `use_delay_pattern` 决定音频 token 的编解码方式（论文 "Simple and Controllable Music Generation"）
- 解码音频 token 前**必须调用** `revert_delay_pattern()` 恢复原始顺序，再裁剪首尾（`[:, 1:-1]`）
  - 参考：`serve_engine.py:401` 和 `examples/generation.py:356`
- `build_delay_pattern_mask()` 在 collator 中用于构建带延迟模式的输入，将 BOS/PAD token 插入 codebook 序列

### KV Cache
- 使用 `StaticCache`（来自 `transformers.cache_utils`），需手动创建及 `reset()`
- 支持多 bucket 大小（默认 `[1024, 4096, 8192]`），实际生成时自动选择合适大小
- CUDA 设备上会执行 `model.capture_model()` 来捕获 CUDA graph：每个 kv_cache_length × 2（text decode + audio decode 各一个 graph）
- MPS 不支持 static KV cache / CUDA graph

### generate() 限制
- **`model.generate()` 仅支持 `batch_size=1`**（代码中有 assert）
- 生成使用自定义的 `generate()` 覆盖了 HuggingFace `GenerationMixin.generate()`，不走标准 `GenerationMixin` 流程
- 音频生成模式由 `<|audio_out_bos|>` token 触发，`force_audio_gen=True` 时自动插入此前缀

### 特殊 token 与 token ID
- `boson_multimodal/constants.py` 定义了 `AUDIO_IN_TOKEN`（`<|AUDIO|>`）、`AUDIO_OUT_TOKEN`（`<|AUDIO_OUT|>`）、`EOS_TOKEN`（`<|end_of_text|>`）
- 模型通过 `set_audio_special_tokens(tokenizer)` 注册 `<|audio_out_bos|>` 和 `<|audio_eos|>` 的 token ID
- 默认使用 Llama-3.1-8B-Instruct 的 reserved special tokens：128011（`<|audio_bos|>`）、128012（`<|audio_eos|>`）、128013（`<|audio_out_bos|>`）、128015（`<|AUDIO|>`）、128016（`<|AUDIO_OUT|>`）
- `audio_stream_bos_id=1024`、`audio_stream_eos_id=1025` 在 codebook 维度标记音频流的起止
- `audio_codebook_size` 实际被设为 `config.audio_codebook_size + 2`（因为有 stream_bos/stream_eos）

### Whisper 编码器
- `encode_whisper_embed` 配置控制是否通过 whisper encoder 编码音频为 mel 特征
- whisper forward 被 monkey-patch 以支持 zero-shape tensor（`_whisper_encoder_zero_shape_forward`），因为原始 whisper encoder 的 `_shape` 方法在 bsz=0 时有 bug
- whisper encoder 不支持 flash_attention_2，强制使用 sdpa

### 依赖版本约束
- `transformers>=4.45.1,<4.47.0`（注意上限）
- `ruff==0.12.2`（精确锁定）
- `torch`、`torchaudio`、`torchvision` 无版本约束，随 Docker 镜像提供
- 音频处理依赖 `vector_quantize_pytorch`、`descript-audio-codec`、`librosa`

### RAS（Repetition Aware Sampling）
- `ras_win_len`（默认 7）：防止重复生成的回看窗口长度，设为 `None` 或 `<=0` 可禁用
- `ras_win_max_num_repeat`（默认 2）：允许的最大重复次数

### 训练相关模块
- `PartiallyFrozenEmbedding` 和 `PartiallyFrozenLinear`：将 embedding/linear 层拆分为冻结部分和可训练部分，用于训练时只更新新增 token 的权重
- `support_deepspeed_ulysses` 装饰器：为 sequence parallelism 添加 `sp_size`/`sp_rank`/`sp_group` 属性
- 数据并行工具：`drop_tokens`/`gather_tokens`/`sequence_chunking_per_rank` 等

### 第三方代码
- `boson_multimodal/audio_processing/` 包含来自 xcodec 的第三方代码（LICENSE 文件在该目录内）
- 修改此目录时注意保留原始 license 声明
- DAC 编解码器来自 `descript-audio-codec`（`descriptaudiocodec/` 目录）

### 禁止事项
- Ruff 配置中 `os.getenv`、`os.putenv`、`os.unsetenv` 被标记为 banned API，必须使用 `os.environ` 代替
- `__init__.py` 文件中的未使用 import 被排除 F401 规则（用于 re-export）
