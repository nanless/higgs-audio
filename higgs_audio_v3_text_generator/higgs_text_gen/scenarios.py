"""
Scenario system for Higgs Audio v3 text generation.
10 scenarios covering general-purpose TTS use cases.
"""

from typing import Dict, List


SCENARIOS: Dict[str, dict] = {
    "daily_chat": {
        "name": "日常对话",
        "description": "日常生活场景中的自然口语对话",
        "subscenes": [
            "日常寒暄问候", "分享生活趣事", "吐槽抱怨琐事", "询问帮忙求助",
            "闲聊天气季节", "讨论美食吃饭", "交流兴趣爱好", "约会聊天",
            "家庭日常对话", "朋友聚会聊天", "邻里偶遇交谈",
        ],
        "typical_emotions": {
            "enthusiasm": 2.0, "amusement": 2.0, "contentment": 1.5,
            "confusion": 0.8, "affection": 0.8, "surprise": 0.5, "longing": 0.3,
        },
        "typical_tags": [
            "emotion:enthusiasm", "emotion:amusement", "emotion:contentment",
            "sfx:laughter", "prosody:pause", "emotion:surprise",
        ],
    },
    "business": {
        "name": "商务/专业",
        "description": "商务会议、产品介绍、工作汇报等专业场景",
        "subscenes": [
            "会议开场发言", "产品功能介绍", "客户沟通谈判", "工作进度汇报",
            "面试回答问题", "项目讨论协作", "演讲致辞", "商务谈判",
            "团队激励动员", "邮件口述回复", "技术方案讲解",
        ],
        "typical_emotions": {
            "determination": 2.0, "pride": 1.5, "enthusiasm": 1.2,
            "contemplation": 1.0, "contentment": 0.8, "relief": 0.5,
        },
        "typical_tags": [
            "emotion:determination", "emotion:pride", "prosody:speed_slow",
            "prosody:pause", "prosody:expressive_low", "emotion:enthusiasm",
        ],
    },
    "education": {
        "name": "教育/讲解",
        "description": "课程教学、知识科普、读书分享等教育场景",
        "subscenes": [
            "课程知识讲解", "科学原理科普", "读书笔记分享", "语言学习方法",
            "技能操作指导", "历史故事讲述", "答疑解惑", "学习经验分享",
            "考试辅导", "启蒙教育", "成人在线课程",
        ],
        "typical_emotions": {
            "enthusiasm": 2.0, "contemplation": 1.5, "awe": 1.0,
            "contentment": 1.0, "surprise": 0.5, "determination": 0.5,
        },
        "typical_tags": [
            "emotion:enthusiasm", "emotion:contemplation", "prosody:speed_slow",
            "prosody:pause", "emotion:awe",
        ],
    },
    "emotional": {
        "name": "情感表达",
        "description": "告白、安慰、道歉等情感丰富的表达场景",
        "subscenes": [
            "深情告白表白", "温暖安慰鼓励", "真诚道歉认错", "思念远方的人",
            "离别不舍告别", "感恩感谢表达", "委屈抱怨不满", "孤独失落倾诉",
            "后悔遗憾自责", "感动落泪瞬间",
        ],
        "typical_emotions": {
            "affection": 2.5, "sadness": 1.5, "longing": 1.5,
            "relief": 1.0, "bitterness": 0.8, "shame": 0.8, "helplessness": 0.5,
        },
        "typical_tags": [
            "emotion:affection", "emotion:sadness", "emotion:longing",
            "sfx:sigh", "sfx:crying", "prosody:speed_slow", "prosody:expressive_low",
        ],
    },
    "entertainment": {
        "name": "娱乐/创意",
        "description": "讲故事、角色配音、游戏解说等娱乐场景",
        "subscenes": [
            "生动讲故事", "角色配音表演", "游戏实况解说", "脱口秀段子",
            "模仿配音", "搞笑短剧", "童话故事朗读", "动漫角色配音",
            "广告创意配音", "综艺旁白", "搞笑吐槽",
        ],
        "typical_emotions": {
            "enthusiasm": 2.5, "amusement": 2.5, "surprise": 1.5,
            "elation": 1.5, "fear": 0.8, "anger": 0.5,
        },
        "typical_tags": [
            "emotion:enthusiasm", "emotion:surprise", "style:shouting",
            "style:whispering", "sfx:laughter", "sfx:screaming",
            "prosody:expressive_high", "prosody:speed_fast",
        ],
    },
    "narration": {
        "name": "叙述/旁白",
        "description": "纪录片、宣传片、有声书等叙述旁白场景",
        "subscenes": [
            "纪录片旁白解说", "企业宣传片配音", "旅游景点介绍", "新闻资讯播报",
            "有声书籍朗读", "品牌故事讲述", "展览导览解说", "电影解说旁白",
            "商业广告画外音", "自然风光解说",
        ],
        "typical_emotions": {
            "contentment": 2.0, "contemplation": 1.5, "awe": 1.5,
            "enthusiasm": 1.0, "pride": 0.8, "determination": 0.5,
        },
        "typical_tags": [
            "emotion:contentment", "emotion:contemplation", "emotion:awe",
            "prosody:speed_slow", "prosody:pause", "prosody:expressive_low",
        ],
    },
    "social_media": {
        "name": "社交媒体/短视频",
        "description": "短视频、直播、vlog等新媒体场景",
        "subscenes": [
            "vlog日常生活", "开箱测评体验", "穿搭美妆分享", "美食探店吃播",
            "直播带货推荐", "观点分享评论", "旅行vlog记录", "健身运动打卡",
            "宠物日常分享", "数码产品评测", "影视剧reaction",
        ],
        "typical_emotions": {
            "enthusiasm": 2.5, "elation": 2.0, "amusement": 1.5,
            "surprise": 1.2, "determination": 0.8, "pride": 0.5,
        },
        "typical_tags": [
            "emotion:enthusiasm", "emotion:elation", "emotion:surprise",
            "sfx:laughter", "prosody:speed_fast", "prosody:expressive_high",
        ],
    },
    "service": {
        "name": "客服/服务",
        "description": "电话客服、语音导航等标准化服务场景",
        "subscenes": [
            "电话客服应答", "在线客服回复", "语音导航提示", "自动通知播报",
            "预约确认提醒", "账单说明解释", "活动通知推送", "售后问题处理",
            "投诉建议回应", "业务办理指引",
        ],
        "typical_emotions": {
            "contentment": 2.5, "determination": 1.5, "pride": 1.0,
            "affection": 0.8, "relief": 0.5,
        },
        "typical_tags": [
            "emotion:contentment", "prosody:expressive_low",
            "prosody:speed_slow", "prosody:pause",
        ],
    },
    "creative_writing": {
        "name": "创意写作/文学",
        "description": "诗歌、散文、小说等文学性朗读场景",
        "subscenes": [
            "古诗词朗诵", "现代散文朗读", "小说片段演绎", "戏剧对白朗读",
            "歌词深情演绎", "书信朗读", "微小说创作", "童话故事创作",
            "文学赏析评论", "经典名篇朗读",
        ],
        "typical_emotions": {
            "awe": 2.0, "longing": 1.5, "sadness": 1.2,
            "elation": 1.0, "contemplation": 1.5, "affection": 1.0,
        },
        "typical_tags": [
            "emotion:awe", "emotion:longing", "emotion:contemplation",
            "prosody:speed_slow", "prosody:pause", "prosody:expressive_high",
            "prosody:long_pause",
        ],
    },
    "asr_stress": {
        "name": "压力/极限测试",
        "description": "ASR极限测试场景，覆盖极端语音条件",
        "subscenes": [
            "极快速朗读", "极慢速朗读", "高音朗读", "低音朗读",
            "情绪剧烈切换", "耳语极小声", "大声喊叫", "超长句连续朗读",
            "极短句片段", "中英混合切换", "数字串朗读",
        ],
        "typical_emotions": {
            "fear": 1.5, "anger": 1.5, "elation": 1.5,
            "sadness": 1.0, "surprise": 1.5, "enthusiasm": 1.0,
        },
        "typical_tags": [
            "style:shouting", "style:whispering",
            "prosody:speed_very_fast", "prosody:speed_very_slow",
            "prosody:pitch_high", "prosody:pitch_low",
            "prosody:expressive_high", "prosody:long_pause",
        ],
        "is_stress_test": True,
    },
}


EMOTION_PROFILES = {
    "elation":       {"primary_tags": ["emotion:elation"], "secondary_tags": ["sfx:laughter", "prosody:expressive_high"], "tag_density": "high", "position_bias": "start_weighted"},
    "amusement":     {"primary_tags": ["emotion:amusement"], "secondary_tags": ["sfx:laughter", "prosody:pause"], "tag_density": "medium", "position_bias": "end_weighted"},
    "enthusiasm":    {"primary_tags": ["emotion:enthusiasm"], "secondary_tags": ["prosody:expressive_high", "prosody:speed_fast"], "tag_density": "high", "position_bias": "start_weighted"},
    "determination": {"primary_tags": ["emotion:determination"], "secondary_tags": ["prosody:expressive_high", "prosody:pause"], "tag_density": "medium", "position_bias": "start_weighted"},
    "pride":         {"primary_tags": ["emotion:pride"], "secondary_tags": ["prosody:expressive_high", "prosody:pause"], "tag_density": "medium", "position_bias": "start_weighted"},
    "contentment":   {"primary_tags": ["emotion:contentment"], "secondary_tags": ["prosody:speed_slow", "prosody:expressive_low"], "tag_density": "low", "position_bias": "start_weighted"},
    "affection":     {"primary_tags": ["emotion:affection"], "secondary_tags": ["prosody:speed_slow", "style:whispering"], "tag_density": "low", "position_bias": "start_weighted"},
    "relief":        {"primary_tags": ["emotion:relief"], "secondary_tags": ["sfx:sigh", "prosody:speed_slow"], "tag_density": "medium", "position_bias": "start_weighted"},
    "awe":           {"primary_tags": ["emotion:awe"], "secondary_tags": ["prosody:speed_slow", "prosody:pause"], "tag_density": "medium", "position_bias": "start_weighted"},
    "longing":       {"primary_tags": ["emotion:longing"], "secondary_tags": ["sfx:sigh", "prosody:speed_slow"], "tag_density": "medium", "position_bias": "start_weighted"},
    "contemplation": {"primary_tags": ["emotion:contemplation"], "secondary_tags": ["prosody:pause", "prosody:speed_slow"], "tag_density": "low", "position_bias": "mixed"},
    "confusion":     {"primary_tags": ["emotion:confusion"], "secondary_tags": ["sfx:humming", "prosody:pause"], "tag_density": "medium", "position_bias": "start_weighted"},
    "surprise":      {"primary_tags": ["emotion:surprise"], "secondary_tags": ["sfx:screaming", "prosody:pitch_high"], "tag_density": "high", "position_bias": "start_weighted"},
    "arousal":       {"primary_tags": ["emotion:arousal"], "secondary_tags": ["prosody:expressive_high", "prosody:pitch_high"], "tag_density": "medium", "position_bias": "start_weighted"},
    "anger":         {"primary_tags": ["emotion:anger"], "secondary_tags": ["style:shouting", "prosody:expressive_high"], "tag_density": "high", "position_bias": "start_weighted"},
    "fear":          {"primary_tags": ["emotion:fear"], "secondary_tags": ["sfx:screaming", "prosody:speed_fast"], "tag_density": "high", "position_bias": "start_weighted"},
    "disgust":       {"primary_tags": ["emotion:disgust"], "secondary_tags": ["sfx:cough", "prosody:expressive_high"], "tag_density": "medium", "position_bias": "start_weighted"},
    "bitterness":    {"primary_tags": ["emotion:bitterness"], "secondary_tags": ["sfx:sigh", "prosody:expressive_low"], "tag_density": "medium", "position_bias": "start_weighted"},
    "sadness":       {"primary_tags": ["emotion:sadness"], "secondary_tags": ["sfx:crying", "sfx:sigh", "prosody:speed_slow"], "tag_density": "medium", "position_bias": "start_weighted"},
    "shame":         {"primary_tags": ["emotion:shame"], "secondary_tags": ["sfx:sigh", "prosody:speed_slow"], "tag_density": "low", "position_bias": "start_weighted"},
    "helplessness":  {"primary_tags": ["emotion:helplessness"], "secondary_tags": ["sfx:sigh", "prosody:expressive_low"], "tag_density": "low", "position_bias": "start_weighted"},
}

EMOTIONS = list(EMOTION_PROFILES.keys())

TAG_DENSITY_MAP = {
    "very_high": (2, 4),
    "high": (1, 3),
    "medium": (0, 2),
    "low": (0, 1),
}

LENGTH_SPECS = {
    "ultra_short": {
        "name": "极短",
        "cn": "中文5-15字，简短直接",
        "en": "English 3-10 words, short and direct",
        "cn_example": "今天天气真好",
        "en_example": "What a beautiful day",
    },
    "short": {
        "name": "短句",
        "cn": "中文15-40字，表达一个完整意思",
        "en": "English 10-25 words, one complete thought",
        "cn_example": "我今天去超市买了很多东西，有水果、蔬菜，还有牛奶和面包。",
        "en_example": "I went to the supermarket today and picked up some fruits, vegetables, milk and bread.",
    },
    "medium": {
        "name": "中等",
        "cn": "中文40-100字，可以包含转折或细节描述",
        "en": "English 25-60 words, with some detail or a turn of thought",
        "cn_example": "",
        "en_example": "",
    },
    "long": {
        "name": "长句",
        "cn": "中文100-250字，可包含多个句子",
        "en": "English 60-150 words, multiple sentences",
        "cn_example": "",
        "en_example": "",
    },
    "very_long": {
        "name": "很长",
        "cn": "中文250-500字，完整段落或短篇",
        "en": "English 150-300 words, full paragraph or short passage",
        "cn_example": "",
        "en_example": "",
    },
}

LANG_MIX_SPECS = {
    "pure_cn": "纯中文文本，不使用任何英文单词。所有内容均为中文表达。",
    "pure_en": "Pure English text. No Chinese characters at all. Use natural spoken English.",
    "cn_main": "以中文为主，可自然夹入1-2个常见英文词（如OK、cool、app、email等），但整体必须是中文语境。",
    "en_main": "Mostly English with 1-2 natural Chinese words embedded (like chengyu, food names, etc).",
}

LENGTH_BOUNDS = {
    "ultra_short": (3, 20),
    "short": (10, 50),
    "medium": (30, 120),
    "long": (80, 300),
    "very_long": (150, 600),
}
