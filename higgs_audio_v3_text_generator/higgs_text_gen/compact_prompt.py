"""
Diversity-enforced compact prompt for Higgs Audio v3.
Balances speed (minimal tokens) with variety (batch-level constraints).
"""

import random
from typing import Optional

from .scenarios import SCENARIOS, EMOTION_PROFILES, LENGTH_SPECS, LANG_MIX_SPECS

_OPENING_TYPES = [
    "感叹词开头(哇/天呐/哎呀)", "填充词开头(嗯/那个/就是)", "疑问词开头(什么/怎么/为什么)",
    "时间开头(今天/刚才/昨天)", "动作动词开头(快看/走/吃)",
    "称呼开头(妈妈/老师/老板)", "否定词开头(不/别/没)", "程度词开头(太/好/真)",
]

_FOCUS_TYPES = [
    "描述一个事件", "表达个人感受", "提出问题", "给出建议",
    "对比两个事物", "回忆过去", "展望未来", "描述场景",
]

_SENSORY_TYPES = [
    "视觉细节", "听觉/声音", "味觉/食物", "触觉/温度",
    "嗅觉/气味", "身体感受",
]

_PLACES = ["家里", "公司", "咖啡店", "地铁", "公园", "超市", "餐厅", "学校", "医院", "健身房",
           "图书馆", "商场", "路上", "车里", "电影院", "海边", "山上", "朋友家"]

_TIMES = ["早上", "中午", "下午", "晚上", "深夜", "周末", "周一", "假期", "下雨天", "晴天"]


def _pick_axis(pool, batch_size, seed_str, suffix):
    rng = random.Random(hash(f"{seed_str}|{suffix}") & 0xFFFFFFFF)
    shuffled = list(pool)
    rng.shuffle(shuffled)
    result = []
    for i in range(batch_size):
        if i < len(shuffled):
            result.append(shuffled[i])
        else:
            result.append(rng.choice(pool))
    return result


def build_compact_prompt(
    scenario_key: str,
    subscene: str,
    length_key: str,
    lang_key: str,
    emotion: str,
    batch_size: int,
    suppression_hint: str = "",
    task_id: int = 0,
) -> str:
    scenario = SCENARIOS.get(scenario_key, SCENARIOS["daily_chat"])
    profile = EMOTION_PROFILES.get(emotion, EMOTION_PROFILES["enthusiasm"])
    length_spec = LENGTH_SPECS[length_key]
    lang_spec = LANG_MIX_SPECS[lang_key]
    is_cn = lang_key in ("pure_cn", "cn_main")
    seed_str = f"{scenario_key}|{subscene}|{emotion}|{task_id}"

    openings = _pick_axis(_OPENING_TYPES, batch_size, seed_str, "open")
    focuses = _pick_axis(_FOCUS_TYPES, batch_size, seed_str, "focus")
    sensories = _pick_axis(_SENSORY_TYPES, batch_size, seed_str, "sense")
    places = _pick_axis(_PLACES, batch_size, seed_str, "place")
    times = _pick_axis(_TIMES, batch_size, seed_str, "time")

    rows = []
    for i in range(batch_size):
        rows.append(
            f"  {i+1}. 开头={openings[i]} 关注={focuses[i]} 感官={sensories[i]} "
            f"地点={places[i]} 时间={times[i]}"
        )

    diversity_block = f"""
批内多样性(每条走不同方向):
{chr(10).join(rows)}
- 任意两条开头方式/地点/主语不重复"""

    suppress_block = ""
    if suppression_hint:
        suppress_block = f"\n{suppression_hint}"

    prompt = f"""生成{batch_size}条自然口语文本用于TTS。

场景:{scenario['name']}-{subscene} 情绪:{emotion}
长度:{length_spec['cn'] if is_cn else length_spec['en']}
语言:{lang_spec}{diversity_block}{suppress_block}

Higgs v3标签用法(每条约0-1个):
<|emotion:{emotion}|>放句首 <|sfx:laughter|>哈哈(拟声词紧跟) <|prosody:pause|>放句中

输出纯JSON数组:
[{{"text":"带标签的文本","length_type":"{length_key}","lang_type":"{lang_key}","scenario":"{scenario_key}","subscene":"{subscene}","emotion":"{emotion}","language":"{'zh' if is_cn else 'en'}"}}]

直接输出JSON。"""

    return prompt
