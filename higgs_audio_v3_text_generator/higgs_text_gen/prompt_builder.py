"""
Dynamic prompt builder for Higgs Audio v3 text generation.
Assembles a multi-section prompt for Qwen 3.6 27B.
"""

import hashlib
import random
from typing import Optional

from .scenarios import (
    SCENARIOS, EMOTION_PROFILES, TAG_DENSITY_MAP,
    LENGTH_SPECS, LANG_MIX_SPECS, EMOTIONS,
)
from .tags import (
    HIGGS_V3_TAGS, RECOMMENDED_COMBINATIONS, SFX_TAGS,
    get_all_tags, get_sfx_onomatopoeia, SFX_REQUIRES_ONO,
)
from .diversity import build_diversity_instructions


def _build_lang_override(lang_key: str) -> str:
    if lang_key == "pure_en":
        return """
[CRITICAL LANGUAGE RULE: All output must be NATURAL ENGLISH. No Chinese characters allowed.
Use spoken English patterns: contractions (I'm, don't, can't), natural fillers (well, actually, I mean),
self-repairs ("I was going to, wait, never mind"), and authentic rhythm. 
DO NOT use formal written English. Sound like a real person talking, not an essay.]"""
    elif lang_key == "en_main":
        return """
[CRITICAL LANGUAGE RULE: Mostly English. May embed 1-2 Chinese words naturally.
Use spoken English patterns: contractions, natural fillers, self-repairs.
The embedded Chinese words should feel natural in context.]"""
    elif lang_key == "cn_main":
        return """
[CRITICAL LANGUAGE RULE: 中文为主。可自然夹入1-2个常见英文词。
使用自然口语中文：语气词(呢、嘛、吧)、填充词(那个、就是)、口语化表达。
不要写作文腔。]"""
    elif lang_key == "pure_cn":
        return """
[CRITICAL LANGUAGE RULE: 纯中文。不使用任何英文单词。
使用自然口语中文：语气词、填充词、口语化表达。不要写作文腔。]"""
    return ""


def _build_length_constraint(length_key: str, lang_key: str) -> str:
    spec = LENGTH_SPECS.get(length_key, LENGTH_SPECS["medium"])
    is_cn = lang_key in ("pure_cn", "cn_main")

    tag_prob_map = {
        "ultra_short": 10,
        "short": 15,
        "medium": 20,
        "long": 25,
        "very_long": 30,
    }
    prob = tag_prob_map.get(length_key, 20)

    return f"""
⚠️ 长度约束 ⚠️
当前任务: {spec['name']} ({length_key})
{spec['cn'] if is_cn else spec['en']}
例: {spec['cn_example'] if is_cn else spec['en_example']}
标签注入概率: 约{prob}%的句子带标签，{100-prob}%的句子为纯文本。"""


def _build_tag_guide(emotion: str) -> str:
    profile = EMOTION_PROFILES.get(emotion, EMOTION_PROFILES["contentment"])
    primary = ", ".join(f"<|{t}|>" for t in profile["primary_tags"])
    secondary = ", ".join(f"<|{t}|>" for t in profile["secondary_tags"]) if profile["secondary_tags"] else "无"
    density_min, density_max = TAG_DENSITY_MAP.get(profile["tag_density"], (0, 2))

    rec_combo = RECOMMENDED_COMBINATIONS.get(emotion, [])
    rec_combo_str = ", ".join(f"<|{c}|>" for c in rec_combo) if rec_combo else "无特定推荐"

    emotion_list = []
    for name, info in HIGGS_V3_TAGS["emotion"].items():
        emotion_list.append(f"  <|emotion:{name}|> — {info['cn']}")
    emotion_table = "\n".join(emotion_list[:25])

    style_list = []
    for name, info in HIGGS_V3_TAGS["style"].items():
        style_list.append(f"  <|style:{name}|> — {info['cn']}: {info['rule']}")
    style_table = "\n".join(style_list)

    sfx_list = []
    for name, info in SFX_TAGS.items():
        ono = "、".join(info["onomatopoeia_cn"][:3])
        sfx_list.append(f"  <|sfx:{name}|> — {info['cn']} (拟声词: {ono})")
    sfx_table = "\n".join(sfx_list)

    prosody_list = []
    for name, info in HIGGS_V3_TAGS["prosody"].items():
        prosody_list.append(f"  <|prosody:{name}|> — {info['cn']} (放{info['placement']})")
    prosody_table = "\n".join(prosody_list)

    return f"""
=== Higgs Audio v3 标签使用指南（极其重要，必须严格遵守） ===

Higgs Audio v3 TTS 支持4类内联控制标签来控制语音交付方式。

**核心规则1: 情感/风格/韵律(非pause)标签放在 input 开头，统领整句**
✅ "<|emotion:enthusiasm|>太棒了！我们成功了！"
✅ "<|emotion:sadness|><|sfx:sigh|>唉，又输了..."
✅ "<|style:whispering|><|emotion:affection|>我喜欢你"

**核心规则2: prosody:pause 和 long_pause 插入句中需要停顿的位置**
✅ "我想了很久<|prosody:pause|>最后还是决定离开"
✅ "有一件事我要告诉你<|prosody:long_pause|>我要走了"

**核心规则3: SFX标签后面必须紧跟拟声词（强制要求！）**
✅ "<|sfx:laughter|>哈哈，太好笑了"
✅ "<|sfx:cough|>咳咳，不好意思"
✅ "<|sfx:sneeze|>阿嚏！谁在想我？"
✅ "<|sfx:sigh|>唉，又要加班了"
❌ "<|sfx:laughter|>今天天气真好"  (缺少拟声词！这是错误的)

**核心规则4: 多种标签可以组合，情绪标签放前面**
✅ "<|emotion:fear|><|sfx:screaming|>啊！有老鼠！"
✅ "<|emotion:enthusiasm|><|prosody:expressive_high|>这太不可思议了！"

**核心规则5: 每条文本建议0-3个标签，不要过多。约70-80%句子不需要任何标签。**

当前情绪: {emotion}
主标签: {primary}
次标签: {secondary}
每条文本标签数: {density_min}-{density_max} 个
推荐搭配: {rec_combo_str}
位置倾向: {profile['position_bias']}

**标签列表:**

情感标签 (放置规则: 句首):
{emotion_table}

风格标签 (放置规则: 句首):
{style_table}

音效标签 (放置规则: 紧跟拟声词):
{sfx_table}

韵律标签:
{prosody_table}

注意: 音效标签 `sfx` 后面必须紧跟对应的拟声词文本，否则模型无法正确生成音效。"""


def _build_naturalness_rules(lang_key: str) -> str:
    cn_rules = """
**中文口语自然度规则:**
- 优先使用填充词: 嗯、那个、就是、呃、啊
- 允许自我修复: "我要去，不对，我不想去了"、"我-我-我想要那个"
- 允许句子重叠/重复: "那个那个"、"就是就是"
- 允许拖长音: "好——的——"、"知——道——了——"
- 可以用身体动作描述: "这样这样"、"那边那边"
- 不要用省略号表示未完成
- 结尾必须自然收束，不能以逗号、顿号结尾
"""

    en_rules = """
**English naturalness rules:**
- Use natural fillers: "well", "actually", "I mean", "you know"
- Allow self-repairs: "I was going to, wait, never mind"
- Allow repetitions: "very very", "wait wait", "no no no"
- Use contractions: I'm, don't, can't, won't, it's, that's
- DO NOT use: "um", "uh" (these are often not in ASR vocab)
- End naturally, not with comma or trailing conjunction
"""

    if lang_key == "pure_cn":
        return cn_rules
    elif lang_key == "pure_en":
        return en_rules
    elif lang_key in ("cn_main", "en_main"):
        return cn_rules + "\n" + en_rules
    return cn_rules


def _build_output_format(batch_size: int, scenario_key: str, subscene: str,
                         emotion: str, length_key: str, lang_key: str) -> str:
    examples = [
        '{"text": "<|emotion:enthusiasm|>太棒了！我们终于成功了！这个项目真的不容易。", "length_type": "short", "lang_type": "pure_cn", "scenario": "daily_chat", "subscene": "分享趣事", "emotion": "enthusiasm", "language": "zh"}',
        '{"text": "其实我觉得<|prosody:pause|>这个方案还有一个问题需要解决", "length_type": "short", "lang_type": "pure_cn", "scenario": "business", "subscene": "项目讨论协作", "emotion": "contemplation", "language": "zh"}',
        '{"text": "<|emotion:sadness|><|sfx:sigh|>唉，又输了这场比赛，明明就差那么一点点", "length_type": "short", "lang_type": "pure_cn", "scenario": "emotional", "subscene": "委屈抱怨不满", "emotion": "sadness", "language": "zh"}',
        '{"text": "Well, I was going to say something, wait, never mind, it\'s not important anymore.", "length_type": "medium", "lang_type": "pure_en", "scenario": "daily_chat", "subscene": "日常寒暄问候", "emotion": "contentment", "language": "en"}',
        '{"text": "<|emotion:surprise|>真的假的? That\'s absolutely incredible! <|sfx:laughter|>Haha, I can\'t believe it!", "length_type": "short", "lang_type": "cn_main", "scenario": "social_media", "subscene": "vlog日常生活", "emotion": "surprise", "language": "zh"}',
        '{"text": "我今天去那家新开的咖啡店试了一下<|prosody:pause|>环境确实不错，咖啡也很好喝，下次可以约朋友一起去。", "length_type": "medium", "lang_type": "pure_cn", "scenario": "daily_chat", "subscene": "讨论美食吃饭", "emotion": "contentment", "language": "zh"}',
    ]

    return f"""
=== 输出格式 ===

只返回 JSON 数组。每个元素必须有这些字段:
- "text": 带 Higgs v3 标签的自然口语文本 (TTS input)
- "length_type": "{length_key}"
- "lang_type": "{lang_key}"
- "scenario": "{scenario_key}"
- "subscene": "{subscene}"
- "emotion": "{emotion}"
- "language": 中文为主的文本填 "zh"，英文为主的填 "en"

好例子 (大部分无标签，少数带标签):
[
  {','.join(examples[:4])}
]

生成恰好 {batch_size} 条。不要在 JSON 前后加任何文字。"""


def build_prompt(
    scenario_key: str,
    subscene: str,
    length_key: str,
    lang_key: str,
    emotion: str,
    batch_size: int,
    suppression_hint: str = "",
    task_id: Optional[int] = None,
) -> str:
    scenario = SCENARIOS.get(scenario_key, SCENARIOS["daily_chat"])
    length_spec = LENGTH_SPECS.get(length_key, LENGTH_SPECS["medium"])
    lang_spec = LANG_MIX_SPECS.get(lang_key, LANG_MIX_SPECS["pure_cn"])
    emotion_profile = EMOTION_PROFILES.get(emotion, EMOTION_PROFILES["contentment"])
    is_stress = scenario.get("is_stress_test", False)

    lang_override = _build_lang_override(lang_key)
    length_strict = _build_length_constraint(length_key, lang_key)
    tag_guide = _build_tag_guide(emotion)
    naturalness = _build_naturalness_rules(lang_key)
    output_fmt = _build_output_format(batch_size, scenario_key, subscene,
                                      emotion, length_key, lang_key)

    stress_instruction = ""
    if is_stress:
        stress_instruction = f"""
=== 压力测试特殊要求 ===
当前为压力测试场景: {scenario['name']}
请生成满足极端条件的文本。确保文本自然但测试性强。
"""

    diversity_instruction = build_diversity_instructions(
        scenario_key, subscene, emotion, lang_key, batch_size, task_id
    )

    typical_tags = ", ".join(f"<|{t}|>" for t in scenario.get("typical_tags", [])[:8])
    typical_emotion_names = ", ".join(
        f"{e}({w:.1f})" for e, w in scenario.get("typical_emotions", {}).items()
    )

    prompt = f"""你是专业的TTS语音合成文本数据生成专家。请生成 {batch_size} 条自然口语文本，用于 Higgs Audio v3 TTS 模型的语音合成训练。{lang_override}

{length_strict}

Higgs Audio v3 是一款先进的文本到语音模型，支持内联标签来控制情感、风格、音效和韵律表现。

=== 生成任务说明 ===
场景: {scenario['name']} — {subscene}
场景描述: {scenario.get('description', '')}
情绪: {emotion} (标签密度: {emotion_profile['tag_density']}，位置倾向: {emotion_profile['position_bias']})
长度: {length_spec['cn'] if lang_key in ('pure_cn','cn_main') else length_spec['en']}
语言: {lang_spec}
{stress_instruction}
{diversity_instruction}
{suppression_hint}
{tag_guide}
=== 场景标签参考 ===
本场景典型标签: {typical_tags}
本场景典型情绪: {typical_emotion_names}

{naturalness}
=== 通用禁忌 ===
- 不要用括号解释 (如 "我(小明)")
- 不要在 text 中写拼音、注音、声调符号
- 不要涉及政治敏感、暴力、色情内容
- 内容必须适合所有年龄段 (PG级)
- 不要直接复制示例文本生成相似内容

{output_fmt}"""
    return prompt
