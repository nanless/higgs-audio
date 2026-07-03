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
- 自己独立的 `requirements.txt`（列了 `requests`, `tqdm`, `loguru`），无需 torch/transformers
- **注意**：`tqdm` 和 `loguru` 虽然在 `requirements.txt` 中，但实际源码中**从未被 import**。`requests` 仅在 `tts_client.py` 中使用
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
- API 端点：`https://api.boson.ai/v1/audio/speech`、`https://api.boson.ai/v1/voices`
- `batch_generate` 固定输出 wav 格式，忽略 `response_format` 参数

### 环境变量（文本生成子项目）

| 变量 | 读取位置 | 默认值 | 说明 |
|------|---------|--------|------|
| `LLM_MODEL` | `config.py`, `llm_client.py` | `qwen3.6-27b` | vLLM 模型名 |
| `LLM_BASE_URL` | `config.py`, `llm_client.py` | `http://localhost:8000` | vLLM API 地址 |
| `LLM_API_KEY` | `llm_client.py` | `EMPTY` | API Key（也检查 `VLLM_API_KEY`、`OPENAI_API_KEY`） |
| `BOSON_API_KEY` | `tts_client.py` | — | Boson TTS API Key |

### 代码级注意事项与陷阱

#### llm_client.py（核心 HTTP 客户端）
- **仅用 `urllib`**，不依赖 `requests` 库（零第三方依赖）
- HTTP 超时 **180 秒**
- JSON 提取 `_extract_json()` 会先去除 Qwen3 的 `<think>...</think>` 思考块（找最后一个 `</think>` 并取其后内容）
- 去除 markdown 代码围栏后提取 `[...]` JSON 数组
- 兜底正则提取单个 `{"text":"..."}` 对象——**比较脆弱**
- 重试指数退避：`wait = retry_base_delay * (2 ** attempt)`
- 总失败时返回空列表（不抛异常）

#### llm_local.py（本地 HF 推理，未被生产使用）
- `_extract_json` 与 `llm_client.py` 中的**代码重复**
- 使用 `tokenizer.apply_chat_template(enable_thinking=False)` 禁用 Qwen3 思考模式
- 固定 `top_p=0.95`

#### compact_prompt.py vs prompt_builder.py
- **生产代码使用 `compact_prompt.py`**（通过 `worker.py` 导入 `build_compact_prompt`）
- `prompt_builder.py` + `diversity.py` 是**旧版本**，仍从 `__init__.py` 导出但不再被调用
- `compact_prompt.py` 有自己内联的 10 轴多样性池，与 `diversity.py` 的池**完全独立**

#### run_batch_generation.py 代码重复
- 内部有自己的 `worker_fn`——与 `worker.py` 几乎相同但添加了抑制提示注入
- 重复了 `tags.py`、`text_clean.py` 中的正则模式
- 重复了 `dedup.py` 中的 `_normalize_for_dedup` 逻辑

#### run_parallel_batch.py 硬编码限制
- 端口列表**硬编码为 4 个**：`[8000, 8001, 8002, 8003]`
- `--num-instances` 参数超过 4 时**不会生效**
- 合并阶段无去重，仅简单拼接 JSONL

#### postprocess_merge.py 硬编码限制
- `for w in range(4)` 硬编码读取 4 个 worker 输出文件

#### quality_filter.py 陷阱
- `_validate_emotion_position()` 是**死代码**——始终返回 True，从不拒绝任何文本
- `_validate_length_match()` 使用 `bounds[1] * 1.5` 上界容差——即 "short" 类型最多可到 75 字符（50 × 1.5）
- `reject_severe_length_mismatch` 配置字段**已声明但从未被检查**——长度不匹配只产生警告不拒绝
- `_is_complete_utterance()` 会将以 `"..."` 或 `"…"` 结尾的文本判为不完整——可能误拒意图留白的文本
- `BAD_MARKERS` 含 13 个敏感词：股票/投资/政治/战争/sex/kill/die/porn/no cap/fuck/shit/damn/cunt

#### tag_guide.py 遗留代码
- `_PROHIBITED_SAME_CATEGORY = frozenset({"prosody"})` 已声明但**从未被使用**——被显式的 `_PROHIBITED_PAIRS`（14 对互斥约束）取代

#### tags.py 常量
- `LENGTH_BOUNDS`（字符数范围）：`ultra_short=(3,20)`, `short=(10,50)`, `medium=(30,120)`, `long=(80,300)`, `very_long=(150,600)`
- `_LABEL_DENSITIES`（标签密度分布）：`[(0, 0.30), (1, 0.40), (2, 0.25), (3, 0.05)]`——30% 无标签、40% 1标签、25% 2标签、5% 3标签
- `SFX_REQUIRES_ONO` 包含 8 个（不含 `sniff`），检查窗口为标签后 **10 个字符**

#### dedup.py 算法细节
- 精确去重：对原始文本和归一化文本分别做 MD5 哈希（双重哈希）
- 增量去重 `filter_incremental_duplicates()`：
  - 按 `scenario+subscene+emotion+length+lang` 分组（context key）
  - 长度差 > 55% 跳过比较
  - 3-gram shingle 重叠率 < 12% 跳过
  - 最终用 `SequenceMatcher.ratio()` 检查 `same_context_dup_threshold`（默认 0.52）
- 语义去重 `semantic_deduplicate()`：分组内 Jaccard 3-gram 字符级相似度（非词级）

#### 所有入口脚本的 sys.path 注入
- `generate_single.py`、`run_batch_generation.py`、`postprocess_merge.py` 均使用 `sys.path.insert(0, ...)` + 绝对导入 `from higgs_text_gen.xxx`
- **从其他目录运行会失败**——必须在 `higgs_audio_v3_text_generator/` 目录下执行，或确保 `sys.path` 正确

## v3 TTS 声音复刻流水线（`v3_tts_clone/`）

把总时长不足目标（`TARGET_SEC`，默认 3600s，**生产用 1800s** 半小时）且音频 ≥20 条的说话人，通过 Higgs Audio v3 TTS 复刻补齐。**当前主入口是 `05_iterative_pipeline.sh` 的 10 轮迭代流水线**（统计 → ASR → 预算分配 → 每轮 克隆→**SIM→CER**→剪枝，SIM 在前 CER 在后）；旧的单次 `01/03/04` 顺序步骤已被整合为被 05 调用的子步骤。

> **注意**：`01_stats_speakers.py` 与 `03_run_post_prune_tmux.sh` 已删除，被 `00_prepare_stats.py` 和 `05_*` 系列取代。

### 核心文件

```
v3_tts_clone/
├── README.md                       # 迭代克隆流水线详细文档
├── 00_prepare_stats.py             # 生成全量说话人统计 (Step 0 自动跑, source-only) → all_speakers.csv
├── 02_asr_launch.sh                # ASR 启动脚本 (被 05 调用)
├── 02_asr_worker.py                # ASR 转写 worker (Qwen3-ASR 1.7B, 每 GPU 一个)
├── 03_launch_servers.sh            # SGLang-Omni TTS 启动 (被 05 每轮启停)
├── 03_tts_clone.py                 # TTS 克隆客户端 (ref-pool 设计, 被 05 调用)
├── 04_post_prune_stats.py          # 预算分配 (gap 加权分配全局克隆预算)
├── 05_iterative_pipeline.sh        # 主流水线 (ASR + 预算分配 + 10 轮迭代)
├── 05_iterative_pipeline.env       # 主流水线配置 (source 后运行)
├── 05_scan_existing_clones.py      # 扫描磁盘现有 clone 数 → {dataset/spk: count}
├── 05_generate_round_csv.py        # 每轮生成受限 CSV (限制本轮克隆量为 1/N)
└── 05_save_orig_allocation.py      # 保存首轮预算基准 (避免每轮重新分配漂移)
```

### 环境隔离

| 角色 | 环境 | 说明 |
|------|------|------|
| 统计 (`00`) | 系统 Python | 仅需 soundfile |
| ASR (`02`) | `qwen3-asr` conda env | Qwen3-ASR 1.7B + transformers 4.57.6 |
| TTS 服务端 (`03_launch`) | `higgs_v3_env` conda env | SGLang-Omni editable 安装，每 GPU 一个进程 |
| TTS 客户端 (`03_tts_clone`) | 系统 Python | 仅需 requests、soundfile、numpy、librosa |
| CER/SIM 评估 | `qwen3-asr` / `omnivoice` | 见 `eval_higgs_audio/` |

`05_iterative_pipeline.sh` 会在非交互 shell 中自动 `source .../conda.sh`（尝试 miniforge3 / anaconda3）。

### 运行主流水线

```bash
# 1. 编辑配置 (STATS_CSV / SOURCE_DIRS / CLONE_ROOT / NUM_SERVERS=8 / ALL_GPUS=0..7 等)
#    见 v3_tts_clone/05_iterative_pipeline.env
#    注意: 统计（00_prepare_stats.py）已作为 Step 0 自动运行, 无需手动跑
#    （仅当想把旧 clone 时长也计入时才手动带 --clone-dirs, 但那会破坏 source-only 约定）

# 2. 后台运行 (tmux session 名自定，README 用 higgs)
source v3_tts_clone/05_iterative_pipeline.env
tmux new-session -d -s higgs "bash v3_tts_clone/05_iterative_pipeline.sh"
tail -f clone_workdir/iterative_pipeline*/pipeline_*.log
```

### 迭代流水线内部流程（`05_iterative_pipeline.sh`）

1. **Step 0（统计，新增）**：非续跑模式下自动运行 `00_prepare_stats.py --source-dirs $STATS_SOURCE_DIRS(默认=SOURCE_DIRS) --target-sec $TARGET_SEC --output-dir $STATS_OUTPUT_DIR(默认=STATS_CSV 所在目录) --workers $SCAN_WORKERS`，统计 audio 下所有数据集/说话人生成 **source-only** 的 `all_speakers.csv`，并把 `STATS_CSV` 指向它。**已存在则跳过**（`FORCE_STATS=1` 强制重算）；续跑模式跳过并要求 `STATS_CSV` 存在
2. **Step 0b**：校验 `STATS_CSV` 存在；若 `SOURCE_DIRS` 有多个目录，用符号链接合并到 `merged_sources/`（参考音频只用原始音频，不用 clone）
3. **Step 0.5a**：`02_asr_launch.sh` 转写源音频生成 transcript，然后 **`pkill -f 02_asr_worker` 释放 GPU** 给 TTS
4. **Step 1a**：内联 Python 扫描 `CLONE_ROOT` 已有 clone 时长写入 `stats_with_clone_dur.csv`——**仅作信息展示**（04 会自己扫描 `CLONE_ROOT`，为避免 clone 时长双重计入，此 CSV 不再喂给 04）
5. **Step 1b**：`04_post_prune_stats.py` 用**原始 `STATS_CSV`** + 自身扫描 `--clone-root` 的 clone 时长，按 gap 加权分配全局 `TOTAL_CLONE_HOURS` 预算 → `speaker_duration_stats_post_prune_resume.csv`
6. **Step 1c**：`05_save_orig_allocation.py` 保存首轮 `clones_needed` / `start_clone_idx` 基准 JSON（**预算只分配一次，避免每轮漂移**）
7. **Step 2 × N 轮**（默认 10）：
   - 启动 TTS 服务（GPU `0..NUM_SERVERS-1`）+ 30 次 ×10s 健康检查
   - `05_scan_existing_clones.py` 扫描磁盘现有 clone 数
   - `05_generate_round_csv.py` 生成本轮 CSV：`clones_needed = min(ceil(原始/N), 原始-已有)`，`start_clone_idx = 基数+已有`
   - `03_tts_clone.py --post-prune --seed $((SEED + round*1000000))` 克隆（**每轮 seed 不同**，保证尾部被剪枝编号被复用时文本/参考仍不同；同轮内确定可续跑）
   - **`pkill -f "sgl-omni serve"` 停止 TTS 释放 GPU** 给评估
   - **评估顺序 SIM 先、CER 后**（SIM 快，先删一波缩小昂贵的 ASR 量）：SIM 评估 → **`pkill -f eval_sim.py` 释放 GPU** → SIM 剪枝（`--max-cer 999` 只按 SIM<`MIN_SIM`(0.8 raw 余弦) 删）→ CER 评估（SIM 存活集；`--refresh-scan` 防缓存失效；`--batch-size 32 --audio-workers 16 --group-by-language --asr-max-new-tokens 512` 提速）→ **`pkill -f eval_cer.py` 释放 GPU** → CER 剪枝（CER>0.03 或 SIM<0.8）。两次剪枝均 `--eval-source sidecar`。`START_STEP` 级别：`clone=1<sim=2<cer=3`
   - 失败的评估/剪枝只 warning，继续下一轮
8. **最终**：`04_post_prune_stats.py` 最终统计 + `verify_kept_clones.py` 质量验证 + 各轮汇总

### 关键设计（README「关键设计」表）

- 每轮启停 TTS：克隆后释放 GPU 给 CER/SIM 评估；CER/SIM 各自结束后也 `pkill` 释放 GPU
- 预算只分配一次：`05_save_orig_allocation.py` 存基准，`ceil(原始/N)` + `still_need` 防超量（原始 5 条第 6 轮自动停止）
- 磁盘扫描替代内存 tracker：自动反映剪枝删除和 API 失败，天然支持断点续跑
- `speaker_path` 每轮被 `05_generate_round_csv.py --merged-dir` 覆盖为 `SOURCE_DIRS`（保证参考音频只用原始音频）
- 每轮 seed 不同（`SEED + round*1000000`）：尾部剪枝编号被复用时文本/参考仍不同

### 续跑 / 从指定轮·步开始（`START_ROUND` / `START_STEP`）

- `START_ROUND`（默认 1）+ `START_STEP`（默认 `clone`，可选 `clone|sim|cer`）。步骤级别 `clone=1<sim=2<cer=3`（SIM 在前、CER 在后）
- 起始轮只跑 `>=` 该 level 的步骤，之后每轮都从 `clone` 完整跑；`round < START_ROUND` 的轮次整个跳过
- **续跑模式**（`START_ROUND>1` 或 `START_STEP!=clone`）：跳过 ASR + 预算分配，复用已有 `allocation/speaker_duration_stats_post_prune_resume.csv` + `original_clones_needed.json`（缺失则报错退出）
- 复用来源：磁盘已有 clone（`03_tts_clone.py` 跳过已存在文件）、已有 `.cer.json`/`.sim.json`（评估 `--skip-existing`）
- 每轮的磁盘 SCAN、统计始终执行（不受 `START_STEP` 影响），保证基线与汇总正确
- 例：`START_ROUND=1 START_STEP=sim bash 05_iterative_pipeline.sh`（第1轮从 SIM 开始，克隆已完成）；`START_ROUND=3 bash ...`（从第3轮克隆开始）

### `00_prepare_stats.py` 细节

- **由 `05_iterative_pipeline.sh` 的 Step 0 自动调用**（非续跑且 `STATS_CSV` 缺失/`FORCE_STATS=1` 时）；也可手动跑
- `--source-dirs`：原始音频，`speaker_path` 取自此；`--clone-dirs`：复刻音频，时长计入 `clone_duration_sec` 但**不**用于 `speaker_path`
- 遍历 `{root}/{dataset}/{speaker_id}/`；WAV 用 44 字节头快速算时长，非 WAV 用 `soundfile.info()`；`fsize <= 1000` 视为无效跳过
- `_list_files()`：speaker 目录有子目录时改为 `os.walk` 递归；`SKIP_DIRS = {logs, __pycache__, ref, eval_sim_embedding_cache}`
- 输出 `all_speakers.csv` 字段：`dataset, speaker_id, num_files, has_7to20s, has_source_audio, source_duration_sec, clone_duration_sec, total_duration_sec, gap_sec, status, speaker_path`
- `status`：`OK`(≥target) / `NEED`(<target 且 num_files≥20) / `LOW`(num_files<20)；**目标时长可配 `--target-sec`（默认 3600，生产用 1800）**
- **此脚本不再排除童声数据集**（旧 `01_stats_speakers.py` 的 `CHILD_DATASETS` 逻辑已移除）

### `02_asr_worker.py` 细节与 Bug

- **文件级 batching**（非 speaker 级），按语言分组批处理最大化 GPU 吞吐（`--files-per-batch 48`）
- 只处理 `total_duration_sec < target` 的 speaker（`--target-sec`，默认 3600）；worker 分片 `i % total_gpus == gpu_id`（**GPU ID 必须是 0..N-1 连续整数**）
- 硬编码 `DATASET_LANG`（17 个 dataset → Chinese/English/Japanese），不在表中的用 `language=None` 自动检测
- **跳过已完成**：`{audio_path}.json` 已存在则跳过；输出含 `transcript, language, dataset, speaker_id`
- **失败处理（已修复原 json.dump bug）**：batch 失败时**不写 sidecar**，只 print 日志并计数 → 失败文件下次运行自动重试（显式的隐式重试机制，适合 ASR 瞬时失败）

### `03_tts_clone.py` 参考音频池（ref-pool）设计

- **不再是单条 `ref_audio.wav`**：`build_ref_pool()` 为每个 speaker 构建候选池（单条 7-20s + 短片拼接组合，`max_pool=256`），随机打乱
- 拼接插入 **300ms 静音**，统一 16000 Hz；`_legacy_concat_fallback()` 在无 in-range 组合时取最接近中心 13.5s 的组合（**可能超出 [7,20]s**）
- `--ref-mode`：`fixed`(每 speaker 一条) / `rotate`(每 N 条换一次) / `random`(每条独立，默认)
- 每条 clone 用到的参考物化为 `ref/ref_{md5[:12]}.wav`，池元数据写 `ref/ref_pool.json`
- transcript 来自源音频 sidecar `{audio_path}.json` 的 `transcript` 字段作为 `ref_text`
- **`--post-prune`**：读 resume CSV，用预算分配的 `clones_needed` / `start_clone_idx`，跳过 `status==OK` 与 `clones_needed<=0`。**注意**：`post_prune` 实际是 `args.post_prune or ("clone_duration_sec" in CSV 表头)`——round CSV 带该列时会自动进入 post-prune 分支，即使不传 `--post-prune`
- ⚠️ **`start_clone_idx` 对输出文件名实际无效**：`clone_speaker()` 里 `file_start, _text_base = _resolve_clone_indices(out_dir, start_clone_idx)`，`file_start` 取自 `_next_clone_file_idx()`（磁盘现有最大 `clone_NNNN` +1），`_text_base` 未被使用。即输出编号纯粹磁盘续接，`start_clone_idx` 只影响那个被丢弃的 text_base——这正是「磁盘扫描替代 tracker」幂等的根源
- **非 post-prune 路径**（旧单次模式）：`clones_net = int(gap/estimate)+1`，再 `clones_total = int(clones_net/quality_pass_rate)+1`（默认 2× buffer 补偿 50% 通过率）
- 种子偏移：`pick_text` 用 `make_seed(uid, global_seed + clone_idx + 100003)`；`random` ref 用 `+ clone_idx + 200007`；`rotate` 用 `+ slot`；`fixed` 用 `+0`。每轮 `global_seed = SEED + round*1000000`
- 生成的 clone WAV 默认从 TTS 的 24kHz **降采样到 16000 Hz**（`--output-sample-rate`，soxr_vhq）
- 断点续跑：`clone_XXXX.wav` 存在且 `>1000 bytes` 且 `clone_XXXX.json` 存在 → 跳过
- API 响应 <100 bytes 视为失败；启动时对每个 server 逐一健康检查，任一不可达即 `sys.exit(1)`（05 pipeline 内的健康检查只轮询 `base_port` 首个 server，30×10s）
- 种子确定性 `make_seed(uid, base) = int(md5(uid),16) % 100000 + base`；`--seed` 作为 `global_seed`

#### 输出格式（新）
```
{CLONE_ROOT}/{dataset}/{speaker_id}/
├── ref/
│   ├── ref_pool.json          # 候选池配置 + 全部候选元数据
│   └── ref_{hash}.wav         # 按需物化的参考 (random 模式可多条)
├── clone_0000.wav             # 克隆音频 (16kHz)
├── clone_0000.json            # 含本条实际 ref_audio_path / ref_audio_source / ref_transcript
└── ...
```
- 运行结束在 `CLONE_ROOT/` 下写 `clone_summary.json`（每轮会被覆盖，05 会归档到 `round_NN/`）

#### SGLang TTS API
- `POST http://localhost:{base_port+i}/v1/audio/speech`
- Payload：`{"input": text, "references": [{"audio_path": ref, "text": ref_text}], "temperature": 0.8, "top_k": 50, "max_new_tokens": MAX_NEW_TOKENS}`（`--max-new-tokens`，默认 1024）
- **`--max-new-tokens`（默认 1024）= SGLang audio-token 上限，1024 ≈ 40.7s @ 25fps**。长文本(>~250字)音频会被从中间截断 → ASR 只听到前半 → CER 虚高 → 长 clone 被过度剪枝（实测 300+字桶删除率 25.6%，约 31.7% clone 命中上限）。需长文本完整发声时上调（如 2048/3072，每条长 clone 更耗算力）。模块级 `MAX_NEW_TOKENS` 全局在 `main()` 由 `--max-new-tokens` 设置（ThreadPool 同进程，安全）
- 重试最多 3 次，5xx 指数退避，timeout 300s，非流式
- 并发：speaker round-robin 到 N server；每 server `ThreadPoolExecutor(workers_per_server)`；外层再并行所有 server

### `04_post_prune_stats.py` 预算分配

- 结合原始音频 + 已有 clone 时长，按每个 speaker 到 target 的 gap 比例分配全局 `--total-clone-hours`（`--target-duration-sec` 默认 3600，生产 1800；`--total-clone-hours` 生产 60000）
- `_process_row` 用 CSV 的 `total_duration_sec` 列当作 source 时长（所以 STATS_CSV 必须 source-only），再自身扫描 `--clone-root` 加上 clone 时长；`num_files<20` 直接返回 None（丢弃，不出现在输出）
- `clones_needed = max(1, int(allocated_clone_sec / estimate_clone_duration) + 1)`（`estimate_clone_duration` 默认每条 10s）
- `status`：只有 `OK`（gap≤0）/ `NEED_CLONE`（gap>0）两种；`num_files<20` 不产出行
- 输出：`speaker_duration_stats_post_prune.csv`（全部）+ `speaker_duration_stats_post_prune_resume.csv`（仅 NEED_CLONE，作 `03_tts_clone.py --stats-csv --post-prune` 输入）+ summary json/txt
- 快速 44 字节 WAV 头算时长

### `03_launch_servers.sh` 端口计算

- 端口公式 `PORT = BASE_PORT + GPU_ID`（GPU 3 → 8003）；**与 `03_tts_clone.py` 的 `base_port + i`（i=0..N-1 序号）在非连续 GPU 时不一致**
- 启动前验证 `sglang_omni` 可 import 且 `sgl-omni` 存在；每 GPU 独占：`CUDA_VISIBLE_DEVICES=$GPU sgl-omni serve --model-path $MODEL --port $PORT --host 0.0.0.0`
- 启动后健康检查 `GET /health` 返回 200

### 当前生产配置（`05_iterative_pipeline.env`）

- `STATS_CSV`：`clone_workdir/stats_source_only/all_speakers.csv`（**source-only**，用 `00_prepare_stats.py --source-dirs {audio} --target-sec 1800` 生成，不带 `--clone-dirs`）
- `CLONE_ROOT`：`.../audio_higgs_audio_v3_tts_clone_4`（全新目录从 0 生成）；`PIPELINE_WORKDIR`：`clone_workdir/iterative_pipeline_v4`（`clone_workdir/` 与 `*.wav` 已在 `.gitignore`，不要提交）
- **八卡生产**：`NUM_SERVERS=8`，`ALL_GPUS=0..7`，`WORKERS_PER_SERVER=16`，`TOTAL_ROUNDS=10`，`TOTAL_CLONE_HOURS=60000`（raw0.8 存活 ~10% + 目标 0.5h → gross≈58.5k），`SCAN_WORKERS=64`，`SIM_WORKERS=32`（每卡 4×8 卡），`PRUNE_WORKERS=32`
- `MAX_CER=0.03`，`MIN_SIM=0.8`（**raw 余弦**剪枝阈值），`TARGET_SEC=1800`（每说话人目标半小时，可配），`SEED=42`，`ESTIMATE_CLONE_DURATION=10`，`QUALITY_PASS_RATE=0.5`
- **TTS 用 GPU `seq 0..NUM_SERVERS-1`**（与用于 ASR/评估的 `ALL_GPUS` 相互独立，可不同）
- CER 提速：`ASR_BACKEND=vllm`（默认，vLLM 连续批处理、TP=卡数、GPU 利用率高；`transformers` 为回退）、`ASR_VLLM_BATCH=256`、`ASR_GPU_MEM_UTIL=0.9`、`ASR_AUDIO_WORKERS=16`、`ASR_MAX_NEW_TOKENS=512`、`ASR_BATCH_SIZE=32`（仅 transformers 用）。透传给 `run_eval_cer.sh` 的 `--asr-backend/--asr-gpu-mem-util/--batch-size/--audio-workers/--asr-max-new-tokens`。语言预分组：transformers 用 `--group-by-language`，**vllm 用 `--no-group-by-language`**（vllm 对混合语言不敏感，省掉全量 json 预扫）
- **全盘扫描全部多进程（`SCAN_WORKERS=64`）**：`05_scan_existing_clones.py --workers`（重写为多进程，367k→~7s）、`eval_cer --scan-workers`（`list_clone_items` 改无-meta 并行扫，79s→~3s）、`prune_and_copy --scan-workers`（`scan_clone_wavs` 改按 speaker 分片，~0.5s）、`run_eval_sim.sh --scan-workers`（~20s，需读每条 json 取 ref）
- `TTS_MAX_NEW_TOKENS=1024`（默认保持；即 `03_tts_clone.py --max-new-tokens`，SGLang audio-token 上限 ≈40.7s，长文本截断根因，需要时上调）
- `set -u` 兼容：流水线脚本使用 `set -u`（未绑定变量报错），conda activate 前需要 `set +u` 绕过 binutils activate.d 的 `ADDR2LINE` 未绑定 bug（`run_eval_cer.sh`/`run_eval_sim.sh` 内部也各自 `set +u`/`set -u` 包裹 `conda activate`）
- 日志：`clone_workdir/iterative_pipeline*/pipeline_*.log`；各轮工件在 `round_NN/`

### SGLang-Omni 本地安装

- 源码固定在 `/root/code/github_repos/sglang-omni`
- `higgs_v3_env` 使用 editable 安装，**必须加 `--no-deps`**（避免升级 torch/transformers 等大包）：

```bash
/root/code/github_repos/higgs-audio/higgs_v3_env/bin/python3 \
    -m pip install -e /root/code/github_repos/sglang-omni --no-deps
```

## 评估流水线（`eval_higgs_audio/`）

从 OmniVoice `batch_generate_text_and_clone/` 的 eval 代码适配而来，评估 Higgs Audio v3 TTS 克隆音频质量。三个维度：**CER**（字错率）、**SIM**（说话人相似度）、**MOS**（音质评分）。

### 目录结构

```
eval_higgs_audio/
├── __init__.py                         # 空文件
├── eval_common.py                      # 共享工具（扫描、分片、累加器、I/O）
├── run_eval_all.sh                     # 一键评估总控（CER → SIM → MOS）
├── setup_models.sh                     # 模型权重下载/符号链接
├── eval_cer/                           # 字错率评估
│   ├── eval_cer.py                     # CER 评估主脚本（ASR + Manual ITN）
│   ├── .env                            # LLM ITN 配置（可选，当前未启用）
│   ├── run_eval_cer.sh                 # 启动脚本（自动激活 qwen3-asr conda env）
│   └── start_vllm_multi.sh            # vLLM 多 GPU 启动脚本（LLM ITN 用）
├── eval_sim/                           # 说话人相似度评估
│   ├── eval_sim.py                     # SIM 评估主脚本（多进程）
│   ├── speaker_encoder.py              # SpeakerEncoder 类（fbank + SamResNet100ASP）
│   ├── speaker_similarity.py           # 相似度计算工具（封装 encoder）
│   ├── models/samresnet.py             # SimAM_ResNet100 + ASP 模型定义（vendored，无 wespeaker 依赖）
│   ├── model/avg_model.pt              # 模型权重（symlink，由 setup_models.sh 创建）
│   ├── model/config.yaml               # 模型配置
│   ├── run_eval_sim.sh                 # 启动脚本（自动激活 omnivoice conda env）
│   └── run_eval_sim_tmux.sh            # tmux 多 GPU 分布式 SIM 评估
├── eval_mos/                           # 音质评估
│   ├── eval_mos.py                     # MOS 评估主脚本（多指标，多进程）
│   ├── scorers.py                      # 4 种评分器（UTMOS22Strong/SCOREQ/TTSDS2/UTMOSv2）
│   ├── utmos_model.py                  # UTMOS22Strong 模型定义（Wav2Vec2 架构）
│   ├── audio_utils.py                  # 音频加载工具（resample + mono 转换）
│   └── run_eval_mos.sh                 # 启动脚本（自动激活 omnivoice conda env）
├── postprocess_common.py               # 共享后处理工具（阈值、加载器、分类器）
├── prune_and_copy.py                   # 按 CER/SIM 阈值剪枝低质量 clone
├── analyze_distributions.py            # CER/SIM 分布分析（含剪枝预览、阈值矩阵）
├── verify_eval_consistency.py          # JSONL 汇总与 sidecar 一致性验证
├── verify_kept_clones.py               # 验证保留 clone 均满足 KEEP 规则
├── run_analyze_distributions.sh        # 一键运行分布分析
├── sim_distribution_report.py          # 跨目录 SIM 分布重算（全量重算，不读 .sim.json）
└── run_sim_distribution_report.sh      # sim_distribution_report.py 启动脚本
```

### 与 OmniVoice 源版本的关键差异

| 维度 | OmniVoice | Higgs Audio（本适配版） |
|------|-----------|------------------------|
| **音频格式** | 16 kHz mono WAV | **24 kHz** mono WAV（评估时内部 resample 到 16 kHz） |
| **文件命名** | `text_NNN.wav` / `text_NNN.json` | `clone_NNNN.wav` / `clone_NNNN.json` |
| **目录结构** | `{dataset}/{speaker}/{utt_id}/text_NNN.wav` | `{dataset}/{speaker_id}/clone_NNNN.wav` |
| **参考文本** | `gen_text` 字段 | `clean_text` 字段（fallback `text`） |
| **标签格式** | `[sigh]`, `[laughter]` 方括号 | `<\|emotion:joy\|>`, `<\|prosody:speed_slow\|>` 管道格式 |
| **ITN** | 有 LLM ITN | 仅 Manual ITN（LLM ITN 暂未启用） |
| **参考音频** | sidecar 中的 `ref_audio` 路径 | 每条 clone sidecar 的 `ref_audio_path`（指向 `ref/ref_{hash}.wav`） |

### 共享工具（`eval_common.py`）

- **扫描逻辑**：`iter_clone_records()` 递归查找 `clone_NNNN.json` + `.wav` 配对，多进程并行（ProcessPoolExecutor）
- **sidecar 判定**：匹配 `clone_\d+\.json` 正则，排除 `.eval.json` / `.mos.json` / `.sim.json` 后缀
- **跳过目录**：`logs`, `__pycache__`, `eval_sim_embedding_cache`
- `list_clone_items()`：返回 `(wav_path, json_path)` 列表（CER/MOS 使用）
- `list_clone_pairs()`：返回 `(clone_wav, ref_audio, json_path)` 三元组（SIM 使用）；`_resolve_ref_audio()` 从每条 clone sidecar 的 `ref_audio_path` 字段解析参考（不再是固定的 `ref_audio.wav`）
- `CerAccumulator`：加权 CER 累加器（按字符数加权）
- `split_shards()`：round-robin 分片，用于多进程分工

### 环境依赖

| 评估步骤 | Conda 环境 | 额外依赖 |
|----------|-----------|----------|
| **CER** | `qwen3-asr` | `qwen_asr`（来自 `/root/code/github_repos/Qwen3-ASR`）、`jiwer`、`word2number`、`soundfile`、`torchaudio`、`tqdm` |
| **SIM** | `omnivoice` | `torch`、`torchaudio`、`yaml`、SamResNet100 权重 |
| **MOS** | `omnivoice` | `scoreq`（pip）、`ttsds`（pip）、`utmosv2`（git）、UTMOS22Strong 权重 |

### CER 评估详解（`eval_cer/eval_cer.py`）

#### 流程
1. 扫描 clone 目录 → 找到所有 `clone_NNNN.wav` + `clone_NNNN.json` 配对
2. （可选）`--group-by-language`（默认开）：并行推断每条语言后按语言排序，使每个 batch 尽量单语言、跑满（避免 `transcribe_asr_batch` 内按语言拆子批变小）
3. Qwen3-ASR 转写 wav。**`--asr-backend`（默认 `transformers`，流水线用 `vllm`）**：
   - `vllm`：`Qwen3ASRModel.LLM(tensor_parallel_size=len(asr_gpus), gpu_memory_utilization=--asr-gpu-mem-util)` 单引擎连续批处理，GPU 利用率高；`main()` 里在 CUDA 初始化前 `os.environ["CUDA_VISIBLE_DEVICES"]=--asr-gpus`；`--batch-size` 用较大值（流水线 `ASR_VLLM_BATCH=256`）。已验证与 transformers 转写逐条一致（CER 等价）。
   - `transformers`：每 GPU 一个 HF 模型、`ThreadPoolExecutor(len(gpus))` 多线程驱动（受 GIL 限制 + `model.generate` 自回归 → 利用率低）；`--batch-size` 流水线用 32。
   - 通用：`--audio-workers`（默认 0，流水线 16，音频预取重叠）；`--asr-max-new-tokens` 默认 **512**（原 256 截断长音频）；24 kHz → 16 kHz 内部重采样
4. Manual ITN（对 ref 和 hypo 同时执行）
5. `jiwer.process_characters()` 计算字符级 CER
6. 结果写入 sidecar + JSONL

#### Manual ITN 流程（`manual_itn()`）
- **去除 Higgs 标签**：正则 `<\|[^|>]+\|[^>]*>` 匹配并删除
- **数字归一化**：
  - 百分数：`百分之三十` → `30`
  - 分数：`三分之一` → `3分之1`
  - 中文金额：`三块五` → `3块5`
  - 单位：`kilometers per hour` → `kmh`
  - 英文数词：`twenty-three` → `23`（依赖 `word2number` 库）
  - 中文数词：`三百六十五` → `365`
  - 中文数字串：`一二三` → `123`
- **去除标点**：保留数字间的小数点（如 `3.14`）
- **全小写** + **空格规范化**

#### ASR 模型
- 模型路径：`/root/.cache/huggingface/hub/Qwen3-ASR-1.7B-local`
- 依赖 `qwen_asr` 包：通过 `sys.path.insert(0, "/root/code/github_repos/Qwen3-ASR")` 加载
- 转写语言**自动检测**：`infer_asr_language()` 根据 CJK/Latin 字符比例判断——`latin > cjk*2` → English，`cjk > latin*2` → Chinese，否则 "Unknown"（传 `None` 让 Qwen3 自动检测）
- ASR 结果缓存：`eval_higgs_asr_cache.json`（避免重复推理）
- 支持多 GPU ASR：按 round-robin 分配 batch 到多块 GPU，每块 GPU 在独立线程中运行

#### 输出
- Sidecar：`clone_NNNN.cer.json`（含 `wav_path`, `gen_text`, `asr_hypo`, `ref_manual`, `hypo_manual`, `manual_cer`, `substitutions`, `insertions`, `deletions`, `chars`）
- 汇总：`eval_higgs_cer_summary.json`（含 overall + per-dataset 分解：`weighted_cer`, `avg_cer`, `median_cer`, `p10_cer`, `p90_cer`）
- 明细：`eval_higgs_cer_details.jsonl`

#### 命令

```bash
# 需要 conda qwen3-asr 环境
conda activate qwen3-asr

# 评估所有 clone
python eval_cer.py --out-dir /path/to/clone_output

# 随机采样 500 条
python eval_cer.py --out-dir /path/to/clone_output --sample-size 500 --seed 42

# 使用缓存 ASR 结果（跳过推理）
python eval_cer.py --out-dir /path/to/clone_output --skip-asr

# 跳过已评估的
python eval_cer.py --out-dir /path/to/clone_output --skip-existing

# 或直接用 shell 脚本
bash run_eval_cer.sh --out-dir /path/to/clone_output --sample-size 500
```

### SIM 评估详解（`eval_sim/eval_sim.py`）

#### 流程
1. 扫描 clone 目录 → 找到 `(clone_NNNN.wav, ref, clone_NNNN.json)` 三元组（ref 从 sidecar `ref_audio_path` 解析）
2. SamResNet100ASP 提取 256 维 speaker embedding
3. **raw 余弦相似度**（`dot/(‖‖‖‖)` ∈[-1,1]，`speaker_encoder.py` 已改为返回 raw，不再做 `(cos+1)/2` 映射；mapped 可由 `(raw+1)/2` 反推）
4. 结果写入 sidecar + JSONL

#### Speaker Encoder（`speaker_encoder.py`）
- 模型：`SimAMResNet100ASP`（`models/samresnet.py`），vendored 实现（无 wespeaker 外部依赖）
- 权重：`eval_sim/model/avg_model.pt`（voxblink2_samresnet100_ft），通过 `setup_models.sh` 创建符号链接
- 前处理：与 wespeaker `Speaker` 类一致：`torchaudio.load(normalize=False)` → `float pcm` → `Resample(16kHz)` → `fbank * (1<<15)` → `CMN`
- fbank 参数：`num_mel_bins=80`, `frame_length=25`, `frame_shift=10`, `window_type=hamming`
- 支持单条 `extract_embedding()` 和批量 `extract_embeddings_batch()`（pad fbank 到 max frame）

#### 多进程
- `mp.get_context("spawn")` 创建进程
- GPU round-robin 分配：`gpu_list[i % len(gpu_list)]`
- 每个 worker 限制线程数：`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1` 等
- 多进程结果通过 JSONL part 文件汇总（`eval_higgs_sim_details.w{i}.jsonl` → merge → `eval_higgs_sim_details.jsonl`）

#### 输出
- Sidecar：`clone_NNNN.sim.json`（含 `cloned_audio`, `ref_audio`, `similarity`, `dataset`）
- 汇总：`eval_higgs_sim_summary.json`（含 overall + per-dataset：`mean`, `min`, `max`, `p10`, `p50`, `p90`）

#### 命令

```bash
# 需要 conda omnivoice 环境
conda activate omnivoice

# 评估所有 pair
python eval_sim.py --out-dir /path/to/clone_output

# 采样 200 条，4 进程
python eval_sim.py --out-dir /path/to/clone_output --sample-size 200 --workers 4 --gpus 0,1

# 跳过已评估的
python eval_sim.py --out-dir /path/to/clone_output --skip-existing

# 或直接用 shell 脚本
bash run_eval_sim.sh --out-dir /path/to/clone_output --sample-size 200

# 多 GPU 分布式 SIM 评估（tmux）
HIGGS_SIM_WORKERS=32 HIGGS_SIM_GPUS=0,1,2,3,4,5,6,7 bash run_eval_sim_tmux.sh
```

### MOS 评估详解（`eval_mos/eval_mos.py` + `scorers.py`）

#### 4 种 MOS 指标

| 指标 | 来源 | 范围 | 类型 | 批量评分 |
|------|------|------|------|---------|
| **UTMOS22Strong** | 自定义 PyTorch 模型（`utmos_model.py`） | 1-5 | 参考感知质量 | 支持（动态 batch，max 8 per batch，30s 截断） |
| **SCOREQ** | pip `scoreq` | NR | ONNX 无参考质量 | 不支持（逐条） |
| **TTSDS2** | pip `ttsds` | 0-100 | WavLM+Whisper+pitch 综合 | 支持（ThreadPool 8 线程） |
| **UTMOSv2** | git `utmosv2` | 1-5 | SSL+spectrogram MOS | 支持（numpy batch，max 30s，batch_size=64） |

#### UTMOS22Strong 模型
- 架构：自定义 Wav2Vec2（7 层 ConvFeatureExtractor + 12 层 Transformer）→ domain_emb + judge_emb → BLSTM → Projection → 时间平均 → `score * 2 + 3`
- Checkpoint 搜索顺序：
  1. `--model-dir` / `mos/utmos22_strong_step7459_v1.pt`
  2. `/root/code/github_repos/OmniVoice-fork/TTS_eval_models/mos/utmos22_strong_step7459_v1.pt`
  3. `~/.cache/higgs_eval/utmos22_strong_step7459_v1.pt`
- 输入音频统一 resample 到 16 kHz

#### TTSDS2 评分（`scorers.py`）
- 主模式：加载 ttsds 包中的 WavLM / Whisper / Pitch benchmark，计算 embedding 与噪声 embedding 的距离
- Fallback 模式：如果 benchmark 加载失败，使用 librosa 的 spectral_flatness + RMS + SNR 代理指标
- 不依赖外部模型权重下载

#### 多进程
- 与 SIM 相同的 spawn 多进程 + GPU round-robin 模式
- 支持 `--no-sidecar` 不写 sidecar JSON

#### 输出
- Sidecar：`clone_NNNN.mos.json`（含 `cloned_audio`, `dataset`, `utmos22strong`/`scoreq`/`ttsds2`/`utmosv2` 各指标得分）
- 汇总：`eval_higgs_mos_summary.json`（含 overall + per-dataset + per-language 分解）

#### 命令

```bash
# 需要 conda omnivoice 环境
conda activate omnivoice

# 评估所有 clone（全部指标）
python eval_mos.py --out-dir /path/to/clone_output

# 仅 UTMOS22Strong + SCOREQ，2 进程
python eval_mos.py --out-dir /path/to/clone_output --metrics UTMOS22Strong,SCOREQ --workers 2 --gpus 0,1

# 采样 200 条
python eval_mos.py --out-dir /path/to/clone_output --sample-size 200 --skip-existing

# 或直接用 shell 脚本
bash run_eval_mos.sh --out-dir /path/to/clone_output --sample-size 200
```

### 一键全流程评估（`run_eval_all.sh`）

```bash
# CER + SIM + MOS 全流程
bash eval_higgs_audio/run_eval_all.sh

# 仅 CER
bash eval_higgs_audio/run_eval_all.sh --skip-sim --skip-mos --sample-size 500

# 仅 SIM
bash eval_higgs_audio/run_eval_all.sh --skip-cer --skip-mos --sample-size 500

# 仅 MOS
bash eval_higgs_audio/run_eval_all.sh --skip-cer --skip-sim --sample-size 1000

# 自定义 clone 目录
HIGGS_CLONE_ROOT=/your/path bash eval_higgs_audio/run_eval_all.sh --sample-size 500
```

### 评估环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `HIGGS_CLONE_ROOT` | 生产路径 | Clone 音频输出根目录 |
| `HIGGS_CER_GPU` | `0` | CER ASR 使用的 GPU |
| `HIGGS_SIM_GPU` | `0` | SIM 使用的 GPU |
| `HIGGS_SIM_WORKERS` | `1` | SIM 并行 worker 数 |
| `HIGGS_MOS_GPUS` | `0` | MOS 使用的 GPU（逗号分隔） |
| `HIGGS_MOS_WORKERS` | `1` | MOS 并行 worker 数 |
| `HIGGS_ASR_BATCH_SIZE` | `16` | ASR 批大小 |
| `HIGGS_EVAL_SAMPLE_SIZE` | 全部 | 随机采样数量 |

### 模型权重准备（`setup_models.sh`）

```bash
# 一键准备所有评估模型权重
bash eval_higgs_audio/setup_models.sh
```

- **SIM 权重**（`avg_model.pt`）：自动从已知路径创建符号链接（`/root/workspace/speaker_verification/...` 或 OmniVoice-fork）
- **UTMOS checkpoint**（`utmos22_strong_step7459_v1.pt`）：自动从 OmniVoice-fork 创建符号链接，或提示手动下载：
  ```bash
  huggingface-cli download --local-dir ~/.cache/higgs_eval k2-fsa/TTS_eval_models mos/utmos22_strong_step7459_v1.pt
  ```

### 评估输出文件汇总

所有汇总文件写在 clone 根目录下：

| 文件 | 说明 |
|------|------|
| `eval_higgs_cer_summary.json` | CER 汇总（overall + per-dataset） |
| `eval_higgs_cer_details.jsonl` | CER 逐条明细 |
| `eval_higgs_asr_cache.json` | ASR 推理缓存（避免重复转写） |
| `eval_higgs_sim_summary.json` | SIM 汇总（overall + per-dataset） |
| `eval_higgs_sim_details.jsonl` | SIM 逐条明细 |
| `eval_higgs_mos_summary.json` | MOS 汇总（overall + per-dataset + per-language） |
| `eval_higgs_mos_details.jsonl` | MOS 逐条明细 |

Sidecar 文件写在每个 clone 音频旁边：`clone_NNNN.cer.json`, `clone_NNNN.sim.json`, `clone_NNNN.mos.json`

### 代码级注意事项与陷阱

#### eval_common.py 通用工具
- `write_json()` 使用**原子写入**：先写 `.tmp` 文件再 `os.rename()`，避免进程崩溃时产生半写文件
- `iter_clone_records()` 支持一级（`{dataset}/clone_*.json`）和二级（`{dataset}/{speaker}/clone_*.json`）目录结构
- `_is_clone_sidecar()` 双重过滤：先匹配 `clone_\d+\.json`，再排除 `.eval.json`/`.cer.json`/`.mos.json`/`.sim.json` 后缀
- **跳过目录**：`logs`、`__pycache__`、`eval_sim_embedding_cache`
- `CerAccumulator` 使用**字符数加权** CER 而非简单平均

#### eval_cer.py（CER 评估，~1700 行）
- **LLM ITN 默认关闭**——需要 `--enable-llm` 才会启用 LLM 反向文本规范化
- `load_env_file()` 在**模块导入时**即执行——`.env` 值通过 `os.environ.setdefault` 注入，不会覆盖已存在的环境变量
- `get_truth_text()` 使用 `@lru_cache(maxsize=200_000)`——大规模评估时可能消耗显著内存
- **Pickle 扫描缓存**：`eval_higgs_scan_cache.pkl` 写入 clone 根目录，多进程/多次评估可能冲突
- **分片支持**：`--num-shards N --shard-index I` 用于分布式评估（`i % N == I` 确定性分片）
- **`--asr-max-new-tokens`（默认 512，原硬编码 256）**：256 会截断长音频转写；解码遇 EOS 即停，加大对短音频无额外开销
- **`--group-by-language`（默认开）**：切 batch 前并行推断语言并排序，使 batch 单语言跑满（`transcribe_asr_batch` 会按语言拆子批，混合语言 batch 会变小）
- **高删除率根因不在 ASR**：实测 clone 音频被 TTS `max_new_tokens=1024`（≈40.7s）截断，长文本（300+字桶删除率 25.6%）ASR 只听到前半 → CER 虚高 → 长 clone 过度剪枝。修复在 `03_tts_clone.py --max-new-tokens`，非 ASR 侧
- Manual ITN 的 `_parse_chinese_number()` 支持大写数字（壹佰→100），`_CN_NUM` 含 21 个映射
- Manual ITN 有 34 个英文缩写展开模式（`_CONTRACTION_REPLACEMENTS`）、12 个 SFX 规范化模式（`_SFX_REPLACEMENTS`）
- `_extract_json_array` 会去除 Qwen3 的 `<|thinker|>...</|thinker|>` 和 `<|assistant|>` 标记
- LLM ITN 生产者-消费者模式：ASR 线程生产 → 有界队列 → `--llm-concurrency`（默认 24）个消费者线程并发处理
- LLM ITN 后处理：对比 LLM CER 与 manual CER，仅在 LLM 结果**更优**时才采用

#### eval_sim.py 和 speaker_encoder.py
- SIM 现在写 **raw 余弦** `[-1,1]`（`speaker_encoder.cosine_similarity` 已去掉 `(cos+1)/2` 映射；与 wespeaker CLI 的 mapped 口径不同，与 `wespeaker/examples/.../v2_organized`(sklearn) 的 raw 口径一致）
- **fbank 缩放关键**：`torchaudio.load(normalize=False)` 返回 int16 值，再 `× (1<<15)` 匹配 wespeaker 约定——**这对正确的 speaker embedding 至关重要**
- `yaml.FullLoader` 加载模型配置（`config.yaml`）——若配置文件不可信存在安全风险
- 多进程使用 `mp.get_context("spawn")`（非 fork），每个 worker 设 `torch.set_num_threads(1)`

#### eval_mos.py 和 scorers.py
- UTMOS22Strong 分数公式：`raw_score * 2 + 3`，映射到 [1, 5] MOS 范围
- UTMOS22Strong 批量推理：最大 8 条/batch，30s 截断（`MAX_AUDIO_LEN = 30 * 16000`）
- `utmos_model.py` 中 `MultiheadAttention.forward` **硬编码 `training=False`**——训练模式下行为不正确
- SCOREQ 尝试加载 `CUDAExecutionProvider` 的 ONNX session，失败时**静默回退到 CPU**
- TTSDS2 benchmark 初始化**深度脆弱**——动态加载 ttsds 内部 5 个模块，任一失败则静默退回 librosa 启发式指标（spectral_flatness + RMS + SNR），分数含义完全不同
- UTMOSv2 批量推理：`batch_size=64`，`num_workers=4`，30s 截断
- `scorers.py` 使用 `importlib.util.spec_from_file_location` 动态加载 `utmos_model.py` 和 `audio_utils.py`——避免导入顺序问题
- UTMOS22Strong checkpoint 搜索顺序：`--model-dir/mos/...` → OmniVoice-fork 路径 → `~/.cache/higgs_eval/...`

#### start_vllm_multi.sh（LLM ITN 用）
- 默认使用 GPU `2,3,4,5,6,7`（6 卡），端口 `BASE_PORT + GPU_ID`
- 每个实例：`vllm serve --language-model-only --enable-prefix-caching --gpu-memory-utilization 0.95 --max-model-len 8192 --max-num-seqs 16`
- 启动后等待 **300 秒** 再健康检查

### 评估后处理工具

基于 CER/SIM 评估结果的后续处理工具，用于质量剪枝、分布分析和一致性验证。

| 文件 | 功能 |
|------|------|
| `postprocess_common.py` | 共享加载器/分类器/统计函数，被 `prune_and_copy.py`、`analyze_distributions.py`、`verify_*` 共用 |
| `prune_and_copy.py` | 按阈值剪枝低质量 clone：**删除 CER > 0.03 或 SIM(raw) < 0.8** 的音频及所有 sidecar 文件 |
| `prune_prev_clones.py` | **独立**筛选老 clone 目录到新质量线：读缓存 `.sim.json`(mapped)→`raw=2·sim-1`、`.cer.json`，删 `raw<0.8 或 cer>0.03`；GPU-free、默认 dry-run(`--execute` 才删) |
| `analyze_distributions.py` | 读取 CER/SIM JSONL 明细，计算整体/按 dataset/按语言统计，CER-SIM 联合分布，剪枝 rule 预览，阈值矩阵 |
| `verify_eval_consistency.py` | 验证 JSONL 汇总与 per-audio sidecar JSON 是否一致（多进程并行校验） |
| `verify_kept_clones.py` | 验证磁盘上保留的 clone 均满足 KEEP 规则（CER ≤ `--max-cer`(默认 0.03) 且 SIM ≥ `--min-sim`(默认 0.8 raw)），汇总总时长；退出码非 0 表示有违规或磁盘数 > eval 预期 KEEP × 1.01 |
| `run_analyze_distributions.sh` | 一键运行分布分析脚本 |

默认阈值（`postprocess_common.py`）：`DEFAULT_MAX_CER = 0.03`、`DEFAULT_MIN_SIM = 0.8`（**raw 余弦**；旧 0.85 是 mapped=raw0.70）

**eval 数据源（`--eval-source`，`prune_and_copy.py` / `verify_kept_clones.py` 默认 `sidecar`）**：
- `postprocess_common.py` 新增 `load_cer_map_sidecars()` / `load_sim_map_sidecars()`：多进程（`--eval-workers`，默认 32，按 `dataset/speaker` 目录 round-robin 分片）直接读 per-clone `.cer.json`(`wav_path`+`manual_cer`) / `.sim.json`(`cloned_audio`+`similarity`)，返回 `{wav: 值}`。
- **为什么需要**：旧的 `load_cer_data/load_sim_data` 读 append-only 聚合 jsonl 且按 wav **保留首条**；jsonl 从不清理 → 编号被复用（尾部剪枝后重生成到同路径）时旧的失败记录会压过新记录 → 误判。sidecar 剪枝时删、重评估时重建，**永远新鲜**。
- `jsonl` 模式仍保留（`--eval-source jsonl`）作回退；`analyze_distributions.py` 仍用 jsonl（报表用途）。

剪枝删除时会同时删除对应的 `.wav`、`.json`、`.cer.json`、`.sim.json`、`.mos.json` 等所有 sidecar。

```bash
# 预览剪枝（dry run）
python eval_higgs_audio/prune_and_copy.py --dry-run

# 执行剪枝
python eval_higgs_audio/prune_and_copy.py --workers 32

# 分布分析
python eval_higgs_audio/analyze_distributions.py \
    --out-dir /path/to/clone_root \
    --output-json eval_higgs_audio/eval_distribution_report.json \
    --output-txt eval_higgs_audio/eval_distribution_report.txt

# 一致性验证
python eval_higgs_audio/verify_eval_consistency.py --out-dir /path/to/clone_root

# 验证保留 clone 质量
python eval_higgs_audio/verify_kept_clones.py --out-dir /path/to/clone_root
```

## 童声批量复刻流水线（v2，独立工具）

基于 v2 模型（`HiggsAudioServeEngine`）的童声批量复刻，与 `SoulX-Podcast` 对比。

### 核心文件

- `batch_child_voice_clone_higgs.py`：批处理脚本，从 BAAI-ChildMandarin 数据集随机采样 100 个样本进行语音克隆
- `run_child_voice_clone_higgs.sh`：启动脚本
- `CHILD_VOICE_CLONE_README.md`：148 行详细文档
- `COMPARISON_WITH_SOULX.md`：与 SoulX-Podcast 的对比分析（模型架构、API 差异、输出采样率等）
- `batch_clone_v3.py`：使用 v3 SGLang-Omni API 复刻童声的简单脚本（POST `/v1/audio/speech`，遍历 `child_voice_clone_output_higgs/sample_*`）
- 输出目录：`child_voice_clone_output_higgs/`（已加入 `.gitignore`）

```bash
# 一键启动（v2）
./run_child_voice_clone_higgs.sh

# 或直接调用 Python
python3 batch_child_voice_clone_higgs.py \
    --model-path "bosonai/higgs-audio-v2-generation-3B-base" \
    --audio-tokenizer-path "bosonai/higgs-audio-v2-tokenizer" \
    --output-dir "./child_voice_clone_output_higgs" \
    --num-samples 100 --random-seed 42 --seed 1988

# v3 童声复刻（需要先启动 SGLang-Omni 服务）
python3 batch_clone_v3.py
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
