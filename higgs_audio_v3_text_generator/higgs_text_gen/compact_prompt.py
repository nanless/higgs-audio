"""
Diversity-enforced compact prompt for Higgs Audio v3.
7-axis diversity + multi-emotion + length mixing per batch.
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

_REGISTER_TYPES = [
    "随意口语(好友聊天)", "半正式(同事/熟人)", "正式(演讲/客户)", "急切(催促/紧急)",
    "亲昵(家人/恋人)", "幽默调侃", "严肃警告", "低语秘密",
]

_DIALOGUE_STATES = [
    "独白/自说自话", "对朋友说话", "对陌生人说话", "对领导/长辈", "对晚辈/孩子",
    "自言自语", "发语音消息", "对一群人讲话", "内心独白反思",
]

_EMO_INTENSITY = [
    "强烈爆发", "中等强度", "轻微流露", "压抑克制",
]


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

    # Secondary emotions for mixing (3 closest emotions)
    scenario_emos = scenario.get("typical_emotions", {})
    secondary_emotions = [e for e in sorted(scenario_emos, key=scenario_emos.get, reverse=True)
                          if e != emotion][:3]
    if not secondary_emotions:
        secondary_emotions = ["amusement", "surprise", "contentment"]

    # Assign per-item params
    openings = _pick_axis(_OPENING_TYPES, batch_size, seed_str, "open")
    focuses = _pick_axis(_FOCUS_TYPES, batch_size, seed_str, "focus")
    sensories = _pick_axis(_SENSORY_TYPES, batch_size, seed_str, "sense")
    places = _pick_axis(_PLACES, batch_size, seed_str, "place")
    times = _pick_axis(_TIMES, batch_size, seed_str, "time")
    registers = _pick_axis(_REGISTER_TYPES, batch_size, seed_str, "register")
    dialogue_states = _pick_axis(_DIALOGUE_STATES, batch_size, seed_str, "dialogue")
    intensities = _pick_axis(_EMO_INTENSITY, batch_size, seed_str, "intensity")

    # Length mixing: vary length per item within the batch
    length_variants = [length_key] * (batch_size // 2)
    alt_lengths = [k for k in LENGTH_SPECS.keys() if k != length_key]
    rng = random.Random(hash(f"{seed_str}|length") & 0xFFFFFFFF)
    for _ in range(batch_size - len(length_variants)):
        length_variants.append(rng.choice(alt_lengths))
    rng.shuffle(length_variants)
    length_variants = length_variants[:batch_size]

    # Emotion mixing: 50% primary, 30% secondary, 20% no emotion tag
    emo_assignments = [emotion] * (batch_size // 2)
    rng2 = random.Random(hash(f"{seed_str}|emo") & 0xFFFFFFFF)
    for _ in range(max(0, batch_size // 3)):
        if secondary_emotions:
            emo_assignments.append(rng2.choice(secondary_emotions))
    rem = batch_size - len(emo_assignments)
    emo_assignments.extend(emotion for _ in range(max(0, rem)))
    rng2.shuffle(emo_assignments)
    emo_assignments = emo_assignments[:batch_size]

    rows = []
    for i in range(batch_size):
        li = length_variants[i]
        ei = emo_assignments[i]
        ls = LENGTH_SPECS[li]
        rows.append(
            f"  {i+1}. 开头={openings[i]} 关注={focuses[i]} 感官={sensories[i]} "
            f"地点={places[i]} 时间={times[i]} 语体={registers[i]} 话轮={dialogue_states[i]} "
            f"情绪强度={intensities[i]} 长度={ls['name']}({li}) 本条情绪={ei}"
        )

    diversity_block = f"""
批内多样性(每条严格执行):
{chr(10).join(rows)}
- 任意两条开头方式/语体/地点/话轮状态不重复
- 不同条用不同情绪({emotion}为主/也可{','.join(secondary_emotions[:2])})
- 约20%条不用emotion标签(纯文本) 约30%用次情绪标签
- 长度混合: 不同条用不同长度({length_key}为主,穿插其他)"""

    suppress_block = ""
    if suppression_hint:
        suppress_block = f"\n{suppression_hint}"

    prompt = f"""生成{batch_size}条自然口语文本用于TTS。

场景:{scenario['name']}-{subscene}
默认情绪:{emotion}  默认长度:{length_spec['cn'] if is_cn else length_spec['en']}
语言:{lang_spec}{diversity_block}{suppress_block}

Higgs v3标签(每条约0-1个):
<|emotion:xxx|>放句首(仅当本条情绪=xxx时用)
<|sfx:laughter|>哈哈(拟声词紧跟) <|prosody:pause|>放句中

输出纯JSON数组:
[{{"text":"带标签的文本","length_type":"对应本条长度","lang_type":"{lang_key}","scenario":"{scenario_key}","subscene":"{subscene}","emotion":"对应本条情绪","language":"{'zh' if is_cn else 'en'}"}}]

直接输出JSON。"""

    return prompt
