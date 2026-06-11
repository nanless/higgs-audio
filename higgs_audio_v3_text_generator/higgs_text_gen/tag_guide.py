"""
Shared tag guide builder for Higgs Audio v3.
Used by both compact_prompt.py and prompt_builder.py.
"""

from typing import Optional

from .scenarios import EMOTION_PROFILES
from .tags import (
    EMOTION_TAGS,
    PROSODY_TAGS,
    RECOMMENDED_COMBINATIONS,
    SFX_TAGS,
    STYLE_TAGS,
)


_PROHIBITED_SAME_CATEGORY = frozenset(
    {
        "prosody",
    }
)

_PROHIBITED_PAIRS = frozenset(
    {
        ("prosody:speed_very_slow", "prosody:speed_very_fast"),
        ("prosody:speed_slow", "prosody:speed_very_fast"),
        ("prosody:speed_fast", "prosody:speed_very_slow"),
        ("prosody:speed_very_fast", "prosody:speed_very_slow"),
        ("prosody:speed_very_fast", "prosody:speed_slow"),
        ("prosody:pitch_low", "prosody:pitch_high"),
        ("prosody:pitch_high", "prosody:pitch_low"),
        ("prosody:expressive_high", "prosody:expressive_low"),
        ("prosody:expressive_low", "prosody:expressive_high"),
        ("prosody:long_pause", "prosody:pause"),
        ("style:shouting", "style:whispering"),
        ("style:whispering", "style:shouting"),
        ("sfx:laughter", "sfx:crying"),
        ("sfx:crying", "sfx:laughter"),
    }
)


def validate_tag_combo(tags):
    """Check if a list of tag strings contains mutually exclusive pairs.

    Returns (is_valid, conflict_description).
    """
    if len(tags) <= 1:
        return True, None
    seen = set()
    for tag in tags:
        for other in seen:
            pair = (tag, other)
            if pair in _PROHIBITED_PAIRS:
                return False, f"{tag} 与 {other} 互斥"
        seen.add(tag)
    return True, None


def build_tag_guide(emotion: str, is_cn: bool = True) -> str:
    """Build a comprehensive Higgs v3 tag usage guide for the prompt.

    Args:
        emotion: Primary emotion key (e.g. 'enthusiasm').
        is_cn: Whether the output language is Chinese (affects onomatopoeia display).

    Returns:
        A formatted tag guide string for injection into the LLM prompt.
    """
    profile = EMOTION_PROFILES.get(emotion, EMOTION_PROFILES["enthusiasm"])
    primary = ", ".join(f"<|{t}|>" for t in profile["primary_tags"])
    secondary = ", ".join(f"<|{t}|>" for t in profile["secondary_tags"][:2]) if profile["secondary_tags"] else "无"
    rec_combo = RECOMMENDED_COMBINATIONS.get(emotion, [])
    rec_combo_str = " + ".join(f"<|{c}|>" for c in rec_combo) if rec_combo else "无特定搭配"
    density = profile.get("tag_density", "medium")

    emotion_list = []
    for name, info in EMOTION_TAGS.items():
        emotion_list.append(f"  {name} — {info['cn']}")
    emotion_table = "\n".join(emotion_list)

    style_list = []
    for name, info in STYLE_TAGS.items():
        style_list.append(f"  {name} — {info['cn']} ({info['rule']})")
    style_table = "\n".join(style_list)

    sfx_list = []
    for name, info in SFX_TAGS.items():
        ono = "、".join(info["onomatopoeia_cn"][:3] if is_cn else info["onomatopoeia_en"][:2])
        sfx_list.append(f"  {name} → {ono}")
    sfx_table = "\n".join(sfx_list)

    prosody_list = []
    for name, info in PROSODY_TAGS.items():
        prosody_list.append(f"  {name} — {info['cn']} ({info['placement']})")
    prosody_table = "\n".join(prosody_list)

    return f"""
=== Higgs Audio v3 标签系统 ===
当前情绪:{emotion}  主标签:{primary}
推荐搭配:{rec_combo_str}

规则(极其重要):
0. 标签格式严格为 <|category:name|> — 必须包含category冒号,如 <|emotion:anger|> 而非 <|anger|>
1. emotion/style/prosody(speed/pitch/expressive)放句首,统领整句
2. prosody:pause 和 long_pause 插入句中需要停顿的位置
3. SFX标签后面必须紧跟对应的拟声词,否则模型无法生成音效
4. 多种标签可组合,情绪标签放最前,音效标签紧跟拟声词
5. 互斥标签不能用在一起: speed_very_slow/speed_very_fast、pitch_low/pitch_high、shouting/whispering、laughter/crying 不能同条出现

情绪标签(句首):
{emotion_table}

风格标签(句首):
{style_table}

音效标签(紧跟拟声词):
{sfx_table}

韵律标签:
{prosody_table}

标签数量建议(当前情绪密度: {density}):
- ~30%纯文本无标签, ~40%带1个, ~25%带2个, ~5%带3个"""
