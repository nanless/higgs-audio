"""
Compact prompt for Higgs Audio v3 text generation.
Optimized for speed with minimal token overhead.
"""

from .scenarios import SCENARIOS, EMOTION_PROFILES, LENGTH_SPECS, LANG_MIX_SPECS


def build_compact_prompt(
    scenario_key: str,
    subscene: str,
    length_key: str,
    lang_key: str,
    emotion: str,
    batch_size: int,
    task_id: int = 0,
) -> str:
    scenario = SCENARIOS.get(scenario_key, SCENARIOS["daily_chat"])
    profile = EMOTION_PROFILES.get(emotion, EMOTION_PROFILES["enthusiasm"])
    length_spec = LENGTH_SPECS[length_key]
    lang_spec = LANG_MIX_SPECS[lang_key]
    is_cn = lang_key in ("pure_cn", "cn_main")

    prompt = f"""生成{batch_size}条自然口语文本用于TTS训练。

场景:{scenario['name']}-{subscene} 情绪:{emotion}
长度:{length_spec['cn'] if is_cn else length_spec['en']}
语言:{lang_spec}

Higgs v3标签用法(每条约0-1个):
<|emotion:{emotion}|>放句首 <|sfx:laughter|>哈哈(拟声词紧跟) <|prosody:pause|>放句中

输出纯JSON数组:
[{{"text":"带标签的文本","length_type":"{length_key}","lang_type":"{lang_key}","scenario":"{scenario_key}","subscene":"{subscene}","emotion":"{emotion}","language":"{'zh' if is_cn else 'en'}"}}]

直接输出JSON。"""

    return prompt
