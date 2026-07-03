"""
Diversity enforcement for Higgs Audio v3 text generation batches.
Simplified from OmniVoice — no children-specific axes.
"""

import hashlib
import random
from typing import List, Optional, Tuple


_SENTENCE_TYPE_CN = [
    "陈述句直叙",
    "疑问句",
    "感叹句",
    "祈使/命令句",
    "把字句",
    "被字句",
    "转折复句",
    "因果复句",
    "条件假设句",
    "并列分句",
    "让步句",
    "设问句",
    "排比列举",
    "比喻描写",
    "对比句",
    "递进句",
    "省略主语句",
    "倒装句",
    "反问句",
    "无主语句",
]

_SENTENCE_TYPE_EN = [
    "declarative statement",
    "yes/no question",
    "wh- question",
    "exclamatory",
    "imperative/command",
    "conditional if/when",
    "contrastive but/however",
    "causal because/so",
    "compound and/then",
    "comparative as/than",
    "passive voice",
    "present continuous action",
    "present perfect experience",
    "future intention going to",
    "short response (1-3 words)",
    "rhetorical question",
]

_SUBJECT_STYLE_CN = [
    "我(第一人称)",
    "你/您(第二人称)",
    "他/她/它(第三人称)",
    "我们(复数)",
    "时间主语",
    "处所主语",
    "事物主语(抽象)",
    "事件主语",
    "省略主语",
    "疑问词主语",
]

_SUBJECT_STYLE_EN = [
    "I/first person",
    "you/second person",
    "he/she/it/third person singular",
    "we/plural first",
    "they/plural third",
    "it/dummy subject",
    "there+be existential",
    "question word subject",
    "implied/elided subject",
]

_OBJECT_STYLE_CN = [
    "具体物品",
    "人",
    "处所/地点",
    "小句/从句",
    "无宾语(不及物)",
    "双宾语",
    "抽象概念",
    "数量",
    "感觉/情绪",
    "方位",
]

_OBJECT_STYLE_EN = [
    "concrete object",
    "person",
    "place/location",
    "clause/that-clause",
    "no object (intransitive)",
    "indirect+direct object",
    "abstract concept",
    "quantity/measurement",
    "feeling/emotion",
]

_VERB_FOCUS_CN = [
    "动作(跑、拿、打)",
    "心理(想、怕、爱)",
    "感官(看、听、闻)",
    "言语(说、问、喊)",
    "存在/状态(在、有、是)",
    "趋向(来、去、进)",
    "使令(让、叫、使)",
    "能愿(能、会、可以)",
    "判断/关系(是、属于)",
    "变化(变成、成为)",
    "互动(帮、陪、告诉)",
]

_VERB_FOCUS_EN = [
    "action (run, take, make)",
    "mental (think, believe, want)",
    "sensory (see, hear, feel)",
    "communication (say, tell, ask)",
    "state (be, have, exist)",
    "motion (go, come, move)",
    "causative (let, make, have)",
    "modal (can, must, should)",
    "change (become, turn, get)",
    "interaction (help, show, give)",
]

_CLAUSE_CONNECTOR_CN = [
    "然后/接着(顺承)",
    "但是/可是(转折)",
    "因为/所以(因果)",
    "如果/要是(假设)",
    "虽然(让步)",
    "而且/并且(递进)",
    "单一小句无连接",
    "动词短语串联",
]

_CLAUSE_CONNECTOR_EN = [
    "and/then (sequence)",
    "but/however (contrast)",
    "because/so (cause)",
    "if/when (condition)",
    "single clause",
    "which/who (relative clause)",
    "I think/feel that...",
    "comma splice (casual)",
]

_OPENING_STYLES = [
    "以感叹词/语气词开头",
    "以填充词开头(嗯、那个、well)",
    "以称呼开头",
    "以疑问词开头",
    "以时间开头",
    "以地点开头",
    "以条件开头",
    "以否定词开头",
    "以动词开头(祈使)",
    "以形容词开头",
    "以比较开头",
    "以话题标记开头",
    "以引语开头",
    "以数字开头",
    "以不确定开头(也许、可能)",
    "以回应词开头(对、是啊)",
]

_FOCUS_ANGLES = [
    "描述一个事件过程",
    "表达个人感受",
    "提出问题",
    "给出建议/意见",
    "讲述一个发现",
    "比较两个选项",
    "回忆过去经历",
    "展望未来计划",
    "解释原因",
    "描述场景/画面",
    "表达惊讶/意外",
    "表达遗憾/惋惜",
    "表达感谢/赞美",
    "请求帮助",
    "分享趣事",
]

_SPEECH_PATTERNS = [
    "直接表达，单刀直入",
    "先铺垫再核心",
    "先核心后补充",
    "自我修正型",
    "重复强调型",
    "犹豫试探型",
    "层层递进型",
    "对比反差型",
    "设问自答型",
    "列举分说型",
    "倒叙转折型",
    "首尾呼应型",
]

_PLACE_ANCHOR_POOL = [
    "家里客厅",
    "咖啡店",
    "办公室",
    "地铁上",
    "公园里",
    "超市",
    "餐厅",
    "学校",
    "医院",
    "健身房",
    "图书馆",
    "商场",
    "机场",
    "海边",
    "山上",
    "车里",
    "电影院",
    "博物馆",
    "酒店",
    "朋友家",
]

_TIME_ANCHOR_POOL = [
    "早上刚醒",
    "午饭时间",
    "下午茶",
    "下班路上",
    "晚上睡前",
    "周末清晨",
    "周一早晨",
    "深夜加班",
    "假期第一天",
    "下雨天",
    "晴朗午后",
    "节日当天",
    "生日那天",
]

_PROP_ANCHOR_POOL = [
    "手机",
    "电脑",
    "咖啡杯",
    "书",
    "雨伞",
    "钥匙",
    "耳机",
    "笔记本",
    "水杯",
    "零食",
    "遥控器",
    "背包",
    "眼镜",
    "衣服",
    "宠物",
    "照片",
]

_CREATIVE_TWIST_POOL = [
    "无转折，平铺直叙",
    "意外发现新信息",
    "当事人改变主意",
    "突然想起某事",
    "被别人提醒后反应",
    "从正面变负面情绪",
    "从负面变正面情绪",
    "误会消除",
    "计划被打乱",
]

_SENSORY_DETAIL_POOL = [
    "视觉细节",
    "听觉细节",
    "触觉/温度",
    "味觉/食物",
    "嗅觉/气味",
    "身体感受",
    "无特殊感官细节",
]

_DIALOGUE_STATES = [
    "自说自话/独白",
    "对朋友说话",
    "对陌生人说话",
    "对上级/长辈说话",
    "对下属/晚辈说话",
    "对一群人说话",
    "内心独白",
    "电话/语音消息",
    "直播对观众说话",
    "自言自语反思",
    "对宠物说话",
]


def _syntactic_pools_for_lang(lang_key: str) -> Tuple[List[str], List[str], List[str], List[str], List[str]]:
    use_en = lang_key in ("pure_en", "en_main")
    use_cn = lang_key in ("pure_cn", "cn_main")
    if use_en and not use_cn:
        return (
            list(_SENTENCE_TYPE_EN),
            list(_SUBJECT_STYLE_EN),
            list(_OBJECT_STYLE_EN),
            list(_VERB_FOCUS_EN),
            list(_CLAUSE_CONNECTOR_EN),
        )
    if use_cn and not use_en:
        return (
            list(_SENTENCE_TYPE_CN),
            list(_SUBJECT_STYLE_CN),
            list(_OBJECT_STYLE_CN),
            list(_VERB_FOCUS_CN),
            list(_CLAUSE_CONNECTOR_CN),
        )
    return (
        list(_SENTENCE_TYPE_CN) + list(_SENTENCE_TYPE_EN),
        list(_SUBJECT_STYLE_CN) + list(_SUBJECT_STYLE_EN),
        list(_OBJECT_STYLE_CN) + list(_OBJECT_STYLE_EN),
        list(_VERB_FOCUS_CN) + list(_VERB_FOCUS_EN),
        list(_CLAUSE_CONNECTOR_CN) + list(_CLAUSE_CONNECTOR_EN),
    )


def _pick_shuffled_pool(pool: List[str], batch_size: int, seed_text: str, seed_suffix: str = "pool") -> List[str]:
    if len(pool) >= batch_size:
        seed = f"{seed_text}|{seed_suffix}"
        rng = random.Random(hashlib.md5(seed.encode()).hexdigest())
        shuffled = list(pool)
        rng.shuffle(shuffled)
        return shuffled[:batch_size]
    rng = random.Random(hashlib.md5(f"{seed_text}|{seed_suffix}".encode()).hexdigest())
    result = []
    for i in range(batch_size):
        result.append(rng.choice(pool))
    return result


def _draw_diverse_axis_list(options: List[str], batch_size: int, rng: random.Random) -> List[str]:
    if len(options) <= 1:
        return [options[0]] * batch_size if options else [""] * batch_size
    pool = list(options)
    rng.shuffle(pool)
    result = []
    for i in range(batch_size):
        if not pool:
            pool = list(options)
            rng.shuffle(pool)
        chosen = pool.pop(0)
        if i > 0 and len(options) > 1:
            for _ in range(min(3, len(options))):
                if chosen == result[-1] and pool:
                    pool.append(chosen)
                    chosen = pool.pop(0)
                else:
                    break
        result.append(chosen)
    return result


def draw_syntactic_axes(lang_key: str, batch_size: int, seed_text: str) -> Tuple[List[str], ...]:
    st, su, ob, vb, cc = _syntactic_pools_for_lang(lang_key)
    return (
        _pick_shuffled_pool(st, batch_size, seed_text, "syn_sent"),
        _pick_shuffled_pool(su, batch_size, seed_text, "syn_subj"),
        _pick_shuffled_pool(ob, batch_size, seed_text, "syn_obj"),
        _pick_shuffled_pool(vb, batch_size, seed_text, "syn_verb"),
        _draw_diverse_axis_list(
            cc,
            batch_size,
            random.Random(hashlib.md5(f"syn_conn|{seed_text}".encode()).hexdigest()),
        ),
    )


def build_diversity_instructions(
    scenario_key: str,
    subscene: str,
    emotion: str,
    lang_key: str,
    batch_size: int,
    task_id: Optional[int] = None,
) -> str:
    seed_text = f"{scenario_key}|{subscene}|{emotion}|{lang_key}|task={task_id}"
    rng = random.Random(hashlib.md5(seed_text.encode()).hexdigest())

    openings = _draw_diverse_axis_list(list(_OPENING_STYLES), batch_size, rng)
    focuses = _draw_diverse_axis_list(list(_FOCUS_ANGLES), batch_size, rng)
    patterns = _draw_diverse_axis_list(list(_SPEECH_PATTERNS), batch_size, rng)
    places = _pick_shuffled_pool(_PLACE_ANCHOR_POOL, batch_size, seed_text, "places")
    props = _pick_shuffled_pool(_PROP_ANCHOR_POOL, batch_size, seed_text, "props")
    times = _pick_shuffled_pool(_TIME_ANCHOR_POOL, batch_size, seed_text, "times")
    twists = _draw_diverse_axis_list(_CREATIVE_TWIST_POOL, batch_size, rng)
    sensories = _draw_diverse_axis_list(_SENSORY_DETAIL_POOL, batch_size, rng)
    dialogues = _draw_diverse_axis_list(list(_DIALOGUE_STATES), batch_size, rng)

    syn_sent, syn_subj, syn_obj, syn_verb, syn_conn = draw_syntactic_axes(lang_key, batch_size, seed_text)

    min_sent = min(7, batch_size)
    min_subj = min(5, batch_size)
    min_verb = min(5, batch_size)
    min_open = min(8, batch_size)

    rows = []
    for i in range(batch_size):
        prev_note = ""
        if i > 0:
            prev_note = f"; 与第{i}条在开头/句式/主语至少3项不同"
        rows.append(
            f"{i + 1}. 句式={syn_sent[i]}; 主语={syn_subj[i]}; 宾语={syn_obj[i]}; "
            f"动词={syn_verb[i]}; 连接={syn_conn[i]}; "
            f"开头方式={openings[i]}; 关注点={focuses[i]}; "
            f"地点={places[i]}; 物品={props[i]}; 时间={times[i]}; "
            f"说话对象={dialogues[i]}; 感官细节={sensories[i]}; "
            f"情节转折={twists[i]}; 口语模式={patterns[i]}"
            f"{prev_note}"
        )

    return f"""
=== 批内多样性要求（必须遵守） ===
这 {batch_size} 条不能只是同一意思换说法。任意两条至少3项维度不同：开头方式、句式、主语类型、核心物品、地点。

- 至少 {min_open} 种不同开头方式
- 至少 {min_sent} 种不同句式，至少 {min_subj} 种不同主语类型
- 至少 {min_verb} 种不同动词类型
- 以"我"开头的句子本 batch 最多2条
- 同一条内语言要一致，但 batch 内可以有不同的主语和场景
- 每条必须是完整可朗读的自然口语
- 禁止任何两条共享相同开头前4字

逐条差异计划（逐条执行）：
{chr(10).join(rows)}
"""
