# Higgs-Audio 童声批量复刻流程

## 简介

这个批处理脚本可以从 BAAI-ChildMandarin 数据集中随机选择 100 个童声样本，使用随机英文句子进行语音克隆。

## 文件说明

- `batch_child_voice_clone_higgs.py`: Python 批处理脚本
- `run_child_voice_clone_higgs.sh`: Shell 启动脚本
- `CHILD_VOICE_CLONE_README.md`: 本说明文档

## 核心功能

1. **自动加载数据集**：从 Kaldi 格式文件（wav.scp 和 text.tn）加载童声数据
2. **随机采样**：随机选择 100 个样本进行处理
3. **语音克隆**：使用 Higgs-Audio 模型克隆童声
4. **随机文本生成**：使用 100 个预定义的英文句子随机生成
5. **结果保存**：每个样本保存在独立的子目录中

## 输出目录结构

```
child_voice_clone_output_higgs/
├── clone_results.csv                    # 汇总结果
├── sample_0001_<utt_id>/
│   ├── prompt_audio.wav                 # 原始音频（参考音频）
│   ├── cloned_audio.wav                 # 克隆生成的音频
│   ├── prompt_text.txt                  # 原始文本（中文）
│   └── cloned_text.txt                  # 生成文本（英文）
├── sample_0002_<utt_id>/
│   └── ...
└── sample_0100_<utt_id>/
    └── ...
```

## 使用方法

### 方法一：使用 Shell 脚本（推荐）

```bash
cd /root/code/github_repos/higgs-audio
./run_child_voice_clone_higgs.sh
```

### 方法二：直接运行 Python 脚本

```bash
cd /root/code/github_repos/higgs-audio
python3 batch_child_voice_clone_higgs.py \
    --model-path "bosonai/higgs-audio-v2-generation-3B-base" \
    --audio-tokenizer-path "bosonai/higgs-audio-v2-tokenizer" \
    --output-dir "./child_voice_clone_output_higgs" \
    --num-samples 100 \
    --random-seed 42 \
    --seed 1988
```

## 参数说明

### 数据路径参数
- `--wav-scp`: wav.scp 文件路径（默认：BAAI 数据集路径）
- `--text-tn`: text.tn 文件路径（默认：BAAI 数据集路径）

### 模型参数
- `--model-path`: Higgs-Audio 模型路径（默认：bosonai/higgs-audio-v2-generation-3B-base）
- `--audio-tokenizer-path`: 音频分词器路径（默认：bosonai/higgs-audio-v2-tokenizer）
- `--output-dir`: 输出目录（默认：./child_voice_clone_output_higgs）

### 采样参数
- `--num-samples`: 随机选择的样本数量（默认：100）
- `--random-seed`: 随机种子，用于可重现性（默认：42）

### 生成参数
- `--max-new-tokens`: 最大生成 token 数（默认：1024）
- `--temperature`: 采样温度（默认：0.3）
- `--top-p`: Nucleus 采样参数（默认：0.95）
- `--top-k`: Top-k 采样参数（默认：50）
- `--seed`: 模型随机种子（默认：1988）
- `--device`: 运行设备（默认：cuda 或 cpu）

## 技术细节

### 语音克隆流程

脚本使用 Higgs-Audio 的 few-shot voice cloning 能力：

1. **系统提示词**：设置安静房间的场景描述
2. **参考音频+文本对**：提供原始童声音频和对应的中文文本
3. **目标文本**：提供要生成的英文文本
4. **生成音频**：模型使用参考音频的声音特征生成新的语音

### 对比 SoulX-Podcast 版本

| 特性 | Higgs-Audio 版本 | SoulX-Podcast 版本 |
|------|------------------|-------------------|
| 模型接口 | HiggsAudioServeEngine | SoulX-Podcast Model |
| 消息格式 | ChatMLSample + Message | process_single_input |
| 音频分词器 | HiggsAudioTokenizer | 内置 |
| 参考音频格式 | AudioContent in messages | prompt_wav_list |
| 生成方式 | serve_engine.generate() | model.forward_longform() |
| 输出格式 | HiggsAudioResponse.audio | generated_wavs |

## 示例输出

运行成功后，你会看到类似以下的日志：

```
========================================
Processing 1/100: M8_N10_S7_R1_005
========================================
Original text: 我最喜欢的颜色是蓝色
Generated text: My favorite color is blue, what's yours?
Audio path: /path/to/audio.wav
[INFO] Starting inference...
  → Prompt audio: sample_0001_M8_N10_S7_R1_005/prompt_audio.wav
  → Cloned audio: sample_0001_M8_N10_S7_R1_005/cloned_audio.wav
  → Prompt text: sample_0001_M8_N10_S7_R1_005/prompt_text.txt
  → Cloned text: sample_0001_M8_N10_S7_R1_005/cloned_text.txt
✓ Successfully cloned and saved to: sample_0001_M8_N10_S7_R1_005
```

## 注意事项

1. 确保有足够的 GPU 内存（推荐至少 16GB）
2. 首次运行会下载模型，可能需要一些时间
3. 100 个样本大约需要 1-2 小时完成（取决于硬件）
4. 可以修改 `--num-samples` 参数来调整样本数量
5. 如果遇到内存不足，可以减少 `--max-new-tokens` 参数

## 常见问题

### Q: 如何修改生成的英文句子？
A: 编辑 `batch_child_voice_clone_higgs.py` 文件中的 `ENGLISH_SENTENCES` 和 `EXTENDED_SENTENCES` 列表。

### Q: 如何使用不同的数据集？
A: 使用 `--wav-scp` 和 `--text-tn` 参数指定你的数据集路径。

### Q: 如何调整音频质量？
A: 可以调整 `--temperature`（降低可以提高稳定性）和 `--max-new-tokens`（增加可以生成更长的音频）参数。

### Q: 生成的音频采样率是多少？
A: Higgs-Audio 默认输出 24kHz 采样率的音频。

## 许可证

本脚本遵循 Higgs-Audio 项目的许可证。

