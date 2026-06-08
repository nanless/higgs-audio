# AGENTS.md — Higgs Audio

## 环境与安装

- Python 3.10+
- 安装：`pip install -r requirements.txt && pip install -e .`
- Docker 镜像（推荐）：`nvcr.io/nvidia/pytorch:25.02-py3` 或 `25.01-py3`
- 包名为 `boson_multimodal`（非 `higgs_audio`）

## 构建与质量检查

- **Lint/Format**：`ruff format --check .`（CI 只检查这一项，无单元测试）
- ruff 版本锁定为 `0.12.0+`，配置在 `pyproject.toml`（行宽 119、双引号、3.10 目标）
- 无 mypy/typecheck 配置，无 pytest 目录

## 仓库定位

- **本仓库是 v2 / v2.5 代码**。v3 不依赖此仓库，权重在 HuggingFace `bosonai/higgs-audio-v3-tts-4b`。
- 核心模块在 `boson_multimodal/` 下，安装后即为顶级 Python 包。

## 核心架构

```
boson_multimodal/
  model/higgs_audio/          # HiggsAudioModel (HuggingFace 风格)
  serve/serve_engine.py       # HiggsAudioServeEngine (推理入口)
  audio_processing/           # 音频 tokenizer（部分代码来自 xcodec，MIT License）
    higgs_audio_tokenizer.py  # load_higgs_audio_tokenizer()
  dataset/chatml_dataset.py   # ChatMLSample, prepare_chatml_sample()
  data_collator/higgs_audio_collator.py  # HiggsAudioSampleCollator
  data_types.py               # Message, AudioContent, TextContent
  constants.py                # 音频特殊 token 常量
```

## 关键入口

- **推理**：`boson_multimodal.serve.serve_engine.HiggsAudioServeEngine`
  - 需传入 `model_name_or_path` 和 `audio_tokenizer_name_or_path`
  - 生成调用 `serve_engine.generate(chat_ml_sample=..., max_new_tokens=..., ...)`
  - 流式：`serve_engine.generate_delta_stream(...)`，返回异步 delta 迭代器
- **CLI 示例**：`python3 examples/generation.py --transcript ... --ref_audio belinda --out_path out.wav`

## 重要陷阱与约定

### 音频 tokenizer
- **MPS (Apple Silicon) 上必须将 audio tokenizer 放在 CPU**：量化层中的 embedding 操作在 MPS 上受限
  - 参考：`examples/generation.py:672` 和 `serve_engine.py` 中 `load_higgs_audio_tokenizer` 的 device 参数
- tokenizer 路径默认为 `bosonai/higgs-audio-v2-tokenizer`

### delay pattern
- 模型配置 `use_delay_pattern` 决定音频 token 的编解码方式
- 解码音频 token 前**必须调用** `revert_delay_pattern()` 恢复原始顺序，再裁剪首尾（`[:, 1:-1]`）
- 参考 `serve_engine.py:401` 和 `examples/generation.py:356`

### KV Cache
- `StaticCache` 来自 `transformers.cache_utils`，需手动创建及 `reset()`
- 支持多 bucket 大小（如 `[1024, 4096, 8192]`）
- CUDA 设备上会执行 `model.capture_model()` 来捕获 CUDA graph
- MPS 不支持 static KV cache / CUDA graph

### 依赖版本约束
- `transformers>=4.45.1,<4.47.0`（注意上限）
- `ruff==0.12.2`（精确锁定）

### 特殊 token
- `boson_multimodal/constants.py` 定义了 `AUDIO_IN_TOKEN`、`AUDIO_OUT_TOKEN`、`EOS_TOKEN`
- 模型通过 `set_audio_special_tokens(tokenizer)` 注册音频特殊 token

### 第三方代码
- `boson_multimodal/audio_processing/` 包含来自 xcodec 的第三方代码（LICENSE 文件在该目录内）
- 修改此目录时注意保留原始 license 声明
