"""
Ultra-diverse compact prompt for Higgs Audio v3.
7-axis + topic seeds + narrative structure + persona + causality.
"""

import random
from typing import Optional

from .scenarios import SCENARIOS, EMOTION_PROFILES, LENGTH_SPECS, LANG_MIX_SPECS, EMOTIONS

_OPENING_TYPES = [
    "感叹词开头(哇/天呐/哎呀)", "填充词开头(嗯/那个/就是)", "疑问词开头(什么/怎么/为什么)",
    "时间开头(今天/刚才/昨天)", "动作动词开头(快看/走/吃)",
    "称呼开头(妈妈/老师/老板)", "否定词开头(不/别/没)", "程度词开头(太/好/真)",
]

_FOCUS_TYPES = [
    "描述一个事件过程", "表达个人主观感受", "提出一个疑问或困惑",
    "给出建议或劝告", "对比两个事物/状态", "回忆过去经历",
    "展望未来计划", "描述场景画面", "表达惊喜/意外", "表达遗憾/惋惜",
]

_SENSORY_TYPES = [
    "视觉细节", "听觉/声音", "味觉/食物", "触觉/温度",
    "嗅觉/气味", "身体感受(累/痛/饿)",
]

_PLACES = ["家里", "公司", "咖啡店", "地铁", "公园", "超市", "餐厅", "学校", "医院", "健身房",
           "图书馆", "商场", "路上", "车里", "电影院", "海边", "山上", "朋友家", "酒店", "机场"]

_TIMES = ["早上刚醒", "中午吃饭", "下午茶时间", "下班路上", "深夜加班",
          "周末清晨", "周一早晨", "假期第一天", "下雨天", "晴天傍晚"]

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

_PERSONA_PROFILES = [
    "急性子打工人(说话快、直接)", "温柔妈妈(说话软、带关心)", "挑剔顾客(话里带刺)",
    "兴奋的小孩(充满好奇)", "疲惫上班族(语气低沉)", "话痨朋友(絮絮叨叨)",
    "严肃老板(简短有力)", "害羞内向(欲言又止)", "自信达人(语气上扬)",
    "焦虑患者(反复确认)", "佛系青年(无所谓态度)", "八卦闺蜜(神秘兮兮)",
]

# Topic seeds per scenario - concrete mini-events that the text should be about
_TOPIC_SEEDS = {
    "daily_chat": [
        "丢了东西/找到了", "收到意外礼物", "被放鸽子", "做饭翻车", "偶遇老同学",
        "手机坏了", "减肥失败/成功", "看到有趣视频", "宠物捣乱", "天气突变",
        "忘带钥匙", "快递到了", "睡过头", "被种草/拔草", "小区装修噪音",
    ],
    "business": [
        "项目延期/提前完成", "客户突然改需求", "预算被砍", "升职/被表扬",
        "方案被否", "临时会议", "同事离职", "新工具上线", "数据不对",
        "竞品分析", "季度汇报", "团建活动", "远程办公吐槽", "甲方傻逼",
    ],
    "education": [
        "解不开题", "突然顿悟", "背不下来的公式", "老师提问被点名",
        "考试前夜", "同桌借笔记", "实验做错了", "课外书发现知识",
        "在线课卡顿", "发成绩时刻", "请教同学", "准备presentation",
    ],
    "emotional": [
        "冷战和解", "收到分手信", "暗恋被发现", "久别重逢", "被误解",
        "突然想家", "听到一首老歌", "翻到旧照片", "生日被遗忘",
        "默默付出被看到", "真心话大冒险", "酒后吐真言", "深夜emo",
    ],
    "entertainment": [
        "游戏通关/翻车", "追剧到凌晨", "KTV跑调", "密室逃脱", "桌游吵架",
        "刷到好笑视频", "打牌输了", "看演唱会", "剧本杀反转", "烟花/灯光秀",
    ],
    "narration": [
        "老街拆迁", "手艺人故事", "古镇清晨", "雪山/沙漠风景", "城市变迁",
        "一棵老树", "菜市场烟火气", "站台离别", "工厂车间", "图书馆一角",
    ],
    "social_media": [
        "开箱vlog", "新店探店踩雷/惊艳", "美妆翻车", "穿搭翻车/成功",
        "健身前后对比", "做饭教程", "宠物卖萌", "旅行攻略", "避雷指南",
    ],
    "service": [
        "退换货被拒", "投诉后反转", "预约不上", "系统故障", "账单多扣",
        "排队太久", "客服态度差/好", "物流丢件", "维修拖沓", "换套餐纠结",
    ],
    "creative_writing": [
        "第一场雪", "雨中等候", "黄昏的窗台", "空无一人的街道",
        "风筝断线", "一封未寄出的信", "深夜便利店", "最后一次见面",
    ],
    "asr_stress": [
        "报电话号码", "绕口令", "情绪过山车", "极快数数", "悄悄话",
        "大声求救", "含数字日期", "重复三遍", "中英切换急刹车",
    ],
}


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

    scenario_emos = scenario.get("typical_emotions", {})
    secondary_emotions = [e for e in sorted(scenario_emos, key=scenario_emos.get, reverse=True)
                          if e != emotion][:3]
    if not secondary_emotions:
        secondary_emotions = ["amusement", "surprise", "contentment"]

    # All per-item axes
    openings = _pick_axis(_OPENING_TYPES, batch_size, seed_str, "open")
    focuses = _pick_axis(_FOCUS_TYPES, batch_size, seed_str, "focus")
    sensories = _pick_axis(_SENSORY_TYPES, batch_size, seed_str, "sense")
    places = _pick_axis(_PLACES, batch_size, seed_str, "place")
    times = _pick_axis(_TIMES, batch_size, seed_str, "time")
    registers = _pick_axis(_REGISTER_TYPES, batch_size, seed_str, "register")
    dialogue_states = _pick_axis(_DIALOGUE_STATES, batch_size, seed_str, "dialogue")
    intensities = _pick_axis(_EMO_INTENSITY, batch_size, seed_str, "intensity")
    personas = _pick_axis(_PERSONA_PROFILES, batch_size, seed_str, "persona")
    topics = _pick_axis(
        _TOPIC_SEEDS.get(scenario_key, _TOPIC_SEEDS["daily_chat"]),
        batch_size, seed_str, "topic"
    )

    # Length mixing
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
            f"  {i+1}. 话题={topics[i]} 人设={personas[i]} 情绪={ei}({intensities[i]}) "
            f"开头={openings[i]} 关注={focuses[i]} 感官={sensories[i]} "
            f"地点={places[i]} 时间={times[i]} 语体={registers[i]} 话轮={dialogue_states[i]} "
            f"长度={ls['name']}({li})"
        )

    diversity_block = f"""
批内多样性(每条严格按分配生成):
{chr(10).join(rows)}
- 每条必须围绕分配的"话题"生成具体事件,不能是泛泛而谈
- "人设"决定该条的语气/用词/节奏:
  急性子=短句快节奏, 温柔=语气词多+关心, 疲惫=低沉简约, 兴奋=感叹号+叠词
  挑剔=反问+对比, 害羞=省略号+犹豫, 自信=感叹+高表现力标签
- 不同条用不同情绪({emotion}为主/也穿插{','.join(secondary_emotions[:2])})
- 约20%条不用emotion标签(纯文本) 约30%用次情绪标签
- 长度严格按分配,不能越界
- 任意两条话题/人设/语体/开头方式不重复

逻辑要求(每条都要完整):
- 原因→事件→反应: 说清楚发生了什么、什么感受、什么结果
- 有具体细节(数字/名字/颜色/声音/味道),不能空洞
- 好例: "昨天在咖啡店等了一小时对方没来<|prosody:pause|>气死我了，电话也不接"
- 坏例: "今天天气真好心情也不错" (泛泛空洞,无具体事件无逻辑链条)"""

    suppress_block = ""
    if suppression_hint:
        suppress_block = f"\n{suppression_hint}"

    prompt = f"""生成{batch_size}条自然口语文本用于TTS。

场景:{scenario['name']}-{subscene}
默认情绪:{emotion}  默认长度:{length_spec['cn'] if is_cn else length_spec['en']}
语言:{lang_spec}{diversity_block}{suppress_block}

Higgs v3标签(每条约0-1个):
<|emotion:xxx|>放句首(用于强烈情绪) <|sfx:laughter|>哈哈(拟声词紧跟)
<|sfx:sigh|>唉(拟声词紧跟) <|prosody:pause|>放句中表示停顿

输出纯JSON:
[{{"text":"带标签的文本","length_type":"对应本条长度","lang_type":"{lang_key}","scenario":"{scenario_key}","subscene":"{subscene}","emotion":"对应本条情绪","language":"{'zh' if is_cn else 'en'}"}}]

直接输出JSON。"""

    return prompt
