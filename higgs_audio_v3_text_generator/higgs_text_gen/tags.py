"""
Higgs Audio v3 complete tag system.
43 tags across 4 categories: emotion, style, sfx, prosody.
"""

import re

EMOTION_TAGS = {
    "elation":       {"cn": "喜悦/兴高采烈",     "intensity": "high",   "valence": "positive"},
    "amusement":     {"cn": "被逗乐/轻松笑",     "intensity": "medium", "valence": "positive"},
    "enthusiasm":    {"cn": "热情/兴奋",          "intensity": "high",   "valence": "positive"},
    "determination": {"cn": "坚定/决心",          "intensity": "medium", "valence": "positive"},
    "pride":         {"cn": "自豪/自信",          "intensity": "medium", "valence": "positive"},
    "contentment":   {"cn": "平静满足",           "intensity": "low",    "valence": "positive"},
    "affection":     {"cn": "温暖/爱意",          "intensity": "low",    "valence": "positive"},
    "relief":        {"cn": "如释重负",           "intensity": "medium", "valence": "positive"},
    "awe":           {"cn": "惊叹/崇敬",          "intensity": "high",   "valence": "positive"},
    "longing":       {"cn": "渴望/思念",          "intensity": "medium", "valence": "positive"},
    "contemplation": {"cn": "沉思/反思",          "intensity": "low",    "valence": "neutral"},
    "confusion":     {"cn": "困惑/不解",          "intensity": "medium", "valence": "neutral"},
    "surprise":      {"cn": "惊讶/意外",          "intensity": "high",   "valence": "neutral"},
    "arousal":       {"cn": "激动/紧张期待",      "intensity": "high",   "valence": "neutral"},
    "anger":         {"cn": "愤怒",               "intensity": "high",   "valence": "negative"},
    "fear":          {"cn": "恐惧/害怕",          "intensity": "high",   "valence": "negative"},
    "disgust":       {"cn": "厌恶/反感",          "intensity": "medium", "valence": "negative"},
    "bitterness":    {"cn": "苦涩/怨愤",          "intensity": "medium", "valence": "negative"},
    "sadness":       {"cn": "悲伤/难过",          "intensity": "medium", "valence": "negative"},
    "shame":         {"cn": "羞耻/尴尬",          "intensity": "medium", "valence": "negative"},
    "helplessness":  {"cn": "无助/无力",          "intensity": "low",    "valence": "negative"},
}

STYLE_TAGS = {
    "singing":    {"cn": "唱歌",       "placement": "句首", "rule": "后接歌词内容"},
    "shouting":   {"cn": "大声/喊叫",  "placement": "句首", "rule": "后接强调内容"},
    "whispering": {"cn": "耳语/悄悄话", "placement": "句首", "rule": "后接轻声内容"},
}

SFX_TAGS = {
    "laughter":  {"cn": "笑声",   "onomatopoeia_cn": ["哈哈", "嘿嘿", "呵呵", "嘻嘻"],
                 "onomatopoeia_en": ["Haha", "Hehe", "Hoho"]},
    "sigh":      {"cn": "叹气",   "onomatopoeia_cn": ["唉", "哎", "呼", "嗯"],
                 "onomatopoeia_en": ["Ahh", "Ugh", "Sigh", "Oh"]},
    "cough":     {"cn": "咳嗽",   "onomatopoeia_cn": ["咳咳", "咳"],
                 "onomatopoeia_en": ["Ahem", "Cough"]},
    "crying":    {"cn": "哭泣",   "onomatopoeia_cn": ["呜呜", "呜", "嘤嘤"],
                 "onomatopoeia_en": ["Sob", "Boo hoo"]},
    "screaming": {"cn": "尖叫",   "onomatopoeia_cn": ["啊", "呀"],
                 "onomatopoeia_en": ["Ahh", "Noo", "Aaa"]},
    "humming":   {"cn": "哼唱/沉吟", "onomatopoeia_cn": ["嗯", "哼"],
                 "onomatopoeia_en": ["Hmm", "Mmm"]},
    "sniff":     {"cn": "抽鼻子", "onomatopoeia_cn": ["吸", "嘶"],
                 "onomatopoeia_en": ["Sniff", "Snf"]},
    "sneeze":    {"cn": "打喷嚏", "onomatopoeia_cn": ["阿嚏", "哈啾"],
                 "onomatopoeia_en": ["Achoo", "Atchoo"]},
    "burping":   {"cn": "打嗝",   "onomatopoeia_cn": ["嗝", "呃"],
                 "onomatopoeia_en": ["Burp", "Ugh"]},
}

SFX_REQUIRES_ONO = frozenset({
    "laughter", "sigh", "cough", "crying", "screaming", "humming", "sneeze", "burping",
})

PROSODY_TAGS = {
    "speed_very_slow":  {"cn": "极慢 ~0.65x",     "placement": "句首"},
    "speed_slow":       {"cn": "慢 ~0.85x",       "placement": "句首"},
    "speed_fast":       {"cn": "快 ~1.2x",        "placement": "句首"},
    "speed_very_fast":  {"cn": "极快 ~1.4x",      "placement": "句首"},
    "pitch_low":        {"cn": "低音 ~-3半音",    "placement": "句首"},
    "pitch_high":       {"cn": "高音 ~+2.5半音",  "placement": "句首"},
    "pause":            {"cn": "短暂停 ~400-700ms","placement": "句中任意位置"},
    "long_pause":       {"cn": "长暂停 ~700-1500ms","placement": "句中任意位置"},
    "expressive_high":  {"cn": "高表现力/夸张",    "placement": "句首"},
    "expressive_low":   {"cn": "低表现力/平淡",    "placement": "句首"},
}

HIGGS_V3_TAGS = {
    "emotion": EMOTION_TAGS,
    "style": STYLE_TAGS,
    "sfx": SFX_TAGS,
    "prosody": PROSODY_TAGS,
}

VALID_EMOTIONS = frozenset(EMOTION_TAGS.keys())
VALID_STYLES = frozenset(STYLE_TAGS.keys())
VALID_SFX = frozenset(SFX_TAGS.keys())
VALID_PROSODY = frozenset(PROSODY_TAGS.keys())

HIGGS_TAG_RE = re.compile(r"<\|(emotion|style|sfx|prosody):([a-z_]+)\|>")
HIGGS_TAG_CLEAN_RE = re.compile(r"<\|(emotion|style|sfx|prosody):[a-z_]+\|>")


def get_all_tags():
    result = []
    for category, tag_dict in HIGGS_V3_TAGS.items():
        for name, info in tag_dict.items():
            result.append((category, name, info))
    return result


def validate_tag(category, name):
    if category == "emotion":
        return name in VALID_EMOTIONS
    if category == "style":
        return name in VALID_STYLES
    if category == "sfx":
        return name in VALID_SFX
    if category == "prosody":
        return name in VALID_PROSODY
    return False


def count_tags(text):
    matches = HIGGS_TAG_RE.findall(text)
    return len(matches), matches


def get_sfx_onomatopoeia(sfx_name):
    sfx_info = SFX_TAGS.get(sfx_name, {})
    return sfx_info.get("onomatopoeia_cn", []) + sfx_info.get("onomatopoeia_en", [])


RECOMMENDED_COMBINATIONS = {
    "enthusiasm": ["prosody:expressive_high", "prosody:speed_fast"],
    "sadness": ["prosody:speed_slow", "prosody:expressive_low", "sfx:sigh"],
    "anger": ["style:shouting", "prosody:expressive_high"],
    "fear": ["prosody:speed_fast", "sfx:screaming"],
    "contentment": ["prosody:speed_slow", "prosody:expressive_low"],
    "amusement": ["sfx:laughter", "prosody:expressive_high"],
    "contemplation": ["prosody:speed_slow", "prosody:pause"],
    "surprise": ["sfx:screaming", "prosody:pitch_high"],
    "affection": ["prosody:speed_slow", "style:whispering"],
    "determination": ["prosody:expressive_high", "prosody:speed_fast"],
    "awe": ["prosody:speed_slow", "prosody:pause"],
    "longing": ["prosody:speed_slow", "prosody:expressive_low"],
    "confusion": ["prosody:pause", "sfx:humming"],
    "bitterness": ["prosody:speed_slow", "sfx:sigh"],
    "shame": ["prosody:speed_slow", "prosody:expressive_low"],
    "helplessness": ["prosody:speed_slow", "sfx:sigh"],
    "elation": ["prosody:expressive_high", "sfx:laughter", "prosody:speed_fast"],
    "pride": ["prosody:expressive_high", "prosody:pause"],
    "relief": ["sfx:sigh", "prosody:speed_slow"],
    "disgust": ["prosody:expressive_high", "sfx:cough"],
    "arousal": ["prosody:expressive_high", "prosody:pitch_high"],
}
