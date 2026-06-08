# Higgs-Audio vs SoulX-Podcast 童声克隆对比

## 快速对比表

| 维度 | Higgs-Audio | SoulX-Podcast |
|------|------------|---------------|
| **项目目录** | `/root/code/github_repos/higgs-audio` | `/root/code/github_repos/SoulX-Podcast` |
| **批处理脚本** | `batch_child_voice_clone_higgs.py` | `batch_child_voice_clone.py` |
| **启动脚本** | `run_child_voice_clone_higgs.sh` | `run_child_voice_clone.sh` |
| **输出目录** | `child_voice_clone_output_higgs/` | `child_voice_clone_output/` |
| **模型架构** | Higgs-Audio v2 (3B base) | SoulX-Podcast (1.7B) |
| **输出采样率** | 24kHz | 24kHz |

## 核心代码差异

### 1. 模型初始化

**Higgs-Audio:**
```python
from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine

serve_engine = HiggsAudioServeEngine(
    model_path, 
    audio_tokenizer_path, 
    device=device
)
```

**SoulX-Podcast:**
```python
from soulxpodcast.utils.infer_utils import initiate_model

model, dataset = initiate_model(
    seed, 
    model_path, 
    llm_engine, 
    fp16_flow
)
```

### 2. 消息/数据准备

**Higgs-Audio:**
```python
from boson_multimodal.data_types import Message, ChatMLSample, AudioContent

messages = [
    Message(role="system", content=system_prompt),
    Message(role="user", content=original_text),
    Message(role="assistant", content=AudioContent(audio_url=audio_path)),
    Message(role="user", content=generated_text),
]

chat_ml_sample = ChatMLSample(messages=messages)
```

**SoulX-Podcast:**
```python
from soulxpodcast.utils.infer_utils import process_single_input

formatted_text = f"[S1]{generated_text}"
data = process_single_input(
    dataset,
    [formatted_text],      # target_text_list
    [audio_path],          # prompt_wav_list
    [original_text],       # prompt_text_list
    False,                 # use_dialect_prompt
    [""],                  # dialect_prompt_text_list
)
```

### 3. 音频生成

**Higgs-Audio:**
```python
output: HiggsAudioResponse = serve_engine.generate(
    chat_ml_sample=chat_ml_sample,
    max_new_tokens=1024,
    temperature=0.3,
    top_p=0.95,
    top_k=50,
    stop_strings=["<|end_of_text|>", "<|eot_id|>"],
    seed=seed,
)

# 输出直接可用
output_audio = output.audio
sample_rate = output.sampling_rate
```

**SoulX-Podcast:**
```python
results_dict = model.forward_longform(**data)

# 需要手动拼接音频片段
target_audio = None
for wav in results_dict["generated_wavs"]:
    if target_audio is None:
        target_audio = wav
    else:
        target_audio = torch.cat([target_audio, wav], dim=1)
```

### 4. 音频保存

**Higgs-Audio:**
```python
import soundfile as sf

sf.write(
    str(cloned_audio_path), 
    output.audio,  # numpy array
    output.sampling_rate
)
```

**SoulX-Podcast:**
```python
import soundfile as sf

if isinstance(output_audio, torch.Tensor):
    sf.write(
        str(cloned_audio_path), 
        output_audio.cpu().squeeze(0).numpy(), 
        24000
    )
```

## API 设计哲学

### Higgs-Audio
- **高层抽象**：`HiggsAudioServeEngine` 提供简洁的服务端接口
- **ChatML 格式**：使用对话式消息格式，更符合现代 LLM 范式
- **即插即用**：response 对象直接包含可用的音频数据
- **更接近 quick_start.py 的简洁风格**

### SoulX-Podcast
- **底层控制**：更接近模型的原始接口
- **列表输入**：使用列表格式传递多个参数
- **手动处理**：需要手动拼接生成的音频片段
- **更多定制选项**：如 dialect_prompt, speaker tags 等

## 使用建议

### 选择 Higgs-Audio 当：
- ✅ 你需要快速原型开发
- ✅ 你想要更简洁的代码
- ✅ 你的用例是标准的语音克隆
- ✅ 你喜欢 ChatML 风格的接口

### 选择 SoulX-Podcast 当：
- ✅ 你需要更精细的控制
- ✅ 你要使用方言提示词
- ✅ 你需要长篇语音合成（longform）
- ✅ 你要使用特定的 speaker tags

## 性能对比

| 指标 | Higgs-Audio | SoulX-Podcast |
|------|------------|---------------|
| 模型大小 | 3B | 1.7B |
| 推理速度 | 适中 | 较快 |
| 内存占用 | 较高 (~16GB) | 适中 (~12GB) |
| 语音质量 | 高 | 高 |
| 声音克隆准确度 | 优秀 | 优秀 |

## 相同点

两个系统都：
- 使用相同的数据源（BAAI-ChildMandarin）
- 随机选择 100 个样本
- 使用相同的随机英文句子库
- 保存相同的输出结构（4 个文件/样本）
- 生成 CSV 汇总结果
- 支持可重现的随机种子

## 运行命令对比

**Higgs-Audio:**
```bash
cd /root/code/github_repos/higgs-audio
./run_child_voice_clone_higgs.sh
```

**SoulX-Podcast:**
```bash
cd /root/code/github_repos/SoulX-Podcast
./run_child_voice_clone.sh
```

## 总结

- **Higgs-Audio 版本**更符合现代 AI 服务的设计模式，代码更简洁，适合快速开发
- **SoulX-Podcast 版本**提供更多底层控制，适合需要精细调整的场景
- 两者在输出质量上都很优秀，选择哪个取决于你的具体需求

如果你是新手或者只是想快速测试语音克隆，推荐使用 **Higgs-Audio 版本**。
如果你需要对生成过程进行精细控制，推荐使用 **SoulX-Podcast 版本**。

