#!/usr/bin/env python3
# Copyright (c) 2025 Boson AI
"""One-shot builder: hand-authored spoken scripts -> clone_text_corpus.json.

Pause control is post-TTS: generate continuous speech (no long_pause tags),
then VAD-split speech spans and insert fixed silence (pause_sec).

Do NOT put <|prosody:long_pause|> in the TTS text — even one-per-sentence
burns audio tokens, hits max_new_tokens (~81s), and leaves only a few
seconds of real speech for VAD to keep.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUT = HERE / "clone_text_corpus.json"
# Manual pause after VAD splice: each gap sampled from [min, max] (not fixed).
PAUSE_SEC_MIN = 1.0
PAUSE_SEC_MAX = 3.5
PAUSE_SEC_MEAN = (PAUSE_SEC_MIN + PAUSE_SEC_MAX) / 2.0
# Join sentences directly; punctuation already ends each sentence.
GAP = ""
TAG_RE = re.compile(r"<\|(?:emotion|style|sfx|prosody):[a-z_]+\|>")
# Strip any leftover pause tags if present in authored text.
PAUSE_TAG_RE = re.compile(r"<\|prosody:long_pause\|>")
# Measured on good clones: CN ~7 c/s speech, EN ~3 w/s (~15 c/s). Target ~24s speech.
CN_CHARS_PER_SEC = 7.0
EN_WORDS_PER_SEC = 3.0
TARGET_SPEECH_SEC = 24.0


def _hdr(emotion: str, prosody: str, style: str | None, sfx: str | None, ono: str | None) -> str:
    parts = [f"<|emotion:{emotion}|>"]
    if style:
        parts.append(f"<|style:{style}|>")
    if prosody:
        parts.append(f"<|prosody:{prosody}|>")
    if sfx and ono:
        parts.append(f"<|sfx:{sfx}|>{ono}")
    return "".join(parts)


def _est_speech(lang: str, clean_sents: list[str]) -> float:
    text = "".join(clean_sents)
    if lang == "pure_en":
        return max(8.0, len(text.split()) / EN_WORDS_PER_SEC)
    if lang == "mixed":
        # rough: treat as CN-char dominant with some English tokens
        return max(8.0, len(TAG_RE.sub("", text)) / 6.0)
    return max(8.0, len(TAG_RE.sub("", text)) / CN_CHARS_PER_SEC)


def pack(
    audience: str,
    idx: int,
    lang: str,
    emotion: str,
    prosody: str,
    style: str | None,
    sfx: str | None,
    ono: str | None,
    sents: list[str],
) -> dict:
    first = sents[0]
    hdr = _hdr(emotion, prosody, style, sfx, ono)
    if sfx and ono and first.lower().startswith(ono.lower()):
        rest = first[len(ono) :].lstrip(" ，,.—-")
        tagged_first = hdr + rest
    else:
        tagged_first = hdr + first
    tagged_sents = [PAUSE_TAG_RE.sub("", s) for s in ([tagged_first] + sents[1:])]
    text = GAP.join(tagged_sents)
    clean_sents = list(sents)
    speech = round(_est_speech(lang, clean_sents), 2)
    gaps = max(0, len(sents) - 1) * PAUSE_SEC_MEAN
    return {
        "id": f"{audience}_{idx:04d}",
        "audience": audience,
        "lang": lang,
        "emotion": emotion,
        "prosody": prosody,
        "style": style,
        "sfx": sfx,
        "text": text,
        "clean_text": " ".join(clean_sents),
        "sentences": tagged_sents,
        "clean_sentences": clean_sents,
        "num_sentences": len(sents),
        "pause_sec_min": PAUSE_SEC_MIN,
        "pause_sec_max": PAUSE_SEC_MAX,
        "pause_tag": "none_in_tts",
        "pause_postprocess": "vad_splice",
        "est_speech_sec": speech,
        "est_total_sec": round(speech + gaps, 2),
    }


ADULT_CN: list[tuple] = [
    ("pure_cn", "contentment", "speed_slow", None, "sigh", "唉", ["嗯……周末下午吧，其实也没什么安排。", "阳光正好铺在地板上，我泡了杯茶，把窗开了一条缝。", "书翻了两页又放下，听外面有人练琴。", "偶尔这样什么都不赶，反而觉得日子完整了一点。"]),
    ("pure_cn", "longing", "pitch_low", None, None, None, ["又路过那家旧书店，门还是半掩着。", "灰尘味和纸张味混在一起，一下子把我拉回去好几年。", "那时我们常在角落吵架，又很快和好。", "我站在门口看了会儿……最终还是没有进去。", "有些地方适合想念，不适合重返。"]),
    ("pure_cn", "pride", "expressive_high", None, None, None, ["这份报告——我熬了三个通宵改出来的！", "客户当场点头，说这版终于抓住重点了。", "我没表现得很夸张，心里却像放了烟花。", "走出会议室，我给自己买了杯最贵的咖啡。"]),
    ("pure_cn", "amusement", "expressive_high", None, "laughter", "哈哈", ["哈哈，我家猫刚才把遥控器按成了静音。", "全家找了一圈，才发现它趴在沙发上装无辜。", "我指着电视跟它对视三秒，它居然打了个哈欠。", "算了，今晚就当给自己放个无声电影吧。"]),
    ("pure_cn", "contemplation", "speed_slow", None, None, None, ["地铁里忽然想起一件很小的事。", "很多年前有人问我，长大后想变成什么样的人。", "我当时随口答了个答案，现在却发现答错了重点。", "重要的也许不是变成谁，而是别丢掉好奇心。", "车门开了，我跟着人群走出去，心里反而安静。"]),
    ("pure_cn", "sadness", "speed_slow", None, "sigh", "唉", ["唉……今天整理旧手机，翻到一段语音。", "是你离开前一天说的，声音还那么轻。", "我听了两遍，又不敢听第三遍。", "把它存进一个很深的文件夹，像藏起一块还没愈合的伤口。"]),
    ("pure_cn", "enthusiasm", "speed_fast", None, None, None, ["你知道吗！今晚那家新开的面馆居然不用排队！", "汤底浓得像熬了一整天，辣椒油香得我头皮发麻。", "老板还额外送了卤蛋，说看我吃得太开心。", "我现在就想再去第二碗，胃已经在抗议了。"]),
    ("pure_cn", "anger", "expressive_high", "shouting", None, None, ["我真的受不了了！约好九点，你九点四十才回消息！", "电话不接，定位也不开，把人晾在路边吹风。", "下次要是再这样，我就直接走，不等了。", "尊重别人的时间，真的有那么难吗？"]),
    ("pure_cn", "affection", "speed_slow", "whispering", None, None, ["过来一点……我跟你说小声的。", "其实你今天穿这件外套，比你自己以为的好看很多。", "别紧张，我不是在夸你夸张，是真的。", "晚上回家记得热牛奶，天气凉了。"]),
    ("pure_cn", "surprise", "pitch_high", None, None, None, ["等一下——你刚才说什么？升职了？！", "我还以为你在开玩笑，结果邮件截图都发过来了。", "天哪，你明明上周还在说自己不行。", "走，今晚必须好好庆祝一下！"]),
    ("pure_cn", "relief", "speed_slow", None, "sigh", "呼", ["呼……体检报告出来了，指标都正常。", "这一个月我每天都睡不踏实。", "现在整个人像卸了块石头，肩膀都轻了。", "今晚想吃点好的，不节食了。"]),
    ("pure_cn", "fear", "speed_fast", None, None, None, ["别关灯！我刚才明明听见走廊有脚步声。", "不是错觉，真的一下一下很清晰。", "你先别笑我，我手心全是汗。", "我们一起去看看，好吗？别让我一个人。"]),
    ("pure_cn", "determination", "expressive_high", None, None, None, ["这次我不退了。简历投出去，面试我也去。", "失败可以，但我不想再因为害怕而停住。", "从明天开始早起背题，一点一点来。", "我要让自己相信：我配得上更好的机会。"]),
    ("pure_cn", "confusion", "expressive_low", None, "humming", "嗯", ["嗯……他说的到底是这个意思，还是我听岔了？", "聊天记录翻了三遍，语气怎么都像在生气。", "可当面他又笑着说没事。", "我真的搞不懂，要不要直接问清楚。"]),
    ("pure_cn", "bitterness", "speed_slow", None, "sigh", "唉", ["唉，当年一起创业的人，最后都不回消息了。", "不是怪谁，就是觉得有点空。", "努力过的痕迹还在，关系却淡成了礼貌。", "有时候成长，就是学会一个人把故事讲完。"]),
    ("pure_cn", "elation", "expressive_high", None, "laughter", "哈哈", ["哈哈！中了！演唱会票抢到了！", "内场第三排，我盯着页面手都在抖。", "马上跟你说，周末别安排别的。", "我们要一起疯一整个晚上！"]),
    ("pure_cn", "helplessness", "speed_slow", None, "sigh", "唉", ["唉……房租又涨了，这个月只够吃饭。", "简历石沉大海，电话一个都没有。", "我也想努力，可有时候真的不知道下一步踩哪。", "你要是有空，能不能听我说两句。"]),
    ("pure_cn", "shame", "expressive_low", None, None, None, ["我……今天会议上说错了一个关键数字。", "所有人都看向我，我耳朵烫得不行。", "散会后我立刻发了更正，可还是很丢脸。", "下次我一定先核对两遍，再开口。"]),
    ("pure_cn", "awe", "speed_slow", None, None, None, ["你看那边的山，云把整座峰托起来了。", "风一过，光线一下子全变了。", "我突然说不出话，只想多站一会儿。", "原来有些风景，不需要滤镜也够震撼。"]),
    ("pure_cn", "disgust", "expressive_high", None, None, None, ["这牛奶——你闻一下，明显酸了吧？", "包装日期还是前天的，也太离谱了。", "我现在只想把舌头洗干净。", "退款，必须退，还要投诉。"]),
    ("pure_cn", "arousal", "pitch_high", None, None, None, ["心跳好快……再过十分钟就轮到我上场了。", "手心全是汗，台词却一句都忘不掉。", "深呼吸，深呼吸——我能行。", "帷幕拉开的那一秒，我会把所有紧张变成声音。"]),
    ("pure_cn", "amusement", "expressive_high", None, "laughter", "嘿嘿", ["嘿嘿，同事把我名字写成了谐音梗。", "全公司群里刷屏，我又气又想笑。", "后来他自己也中招，被改成了更离谱的。", "办公室这点快乐，真的很廉价又很珍贵。"]),
    ("pure_cn", "sadness", "speed_slow", None, "crying", "呜", ["呜……小狗走的那天，项圈还挂在门把手上。", "我每次开门都下意识要喊它的名字。", "空荡荡的碗，我到现在都没敢收起来。", "原来家里少了一个脚步声，会这么安静。"]),
    ("pure_cn", "enthusiasm", "expressive_high", None, None, None, ["这个周末去海边吧！我查好了火车。", "早上出发，傍晚就能踩到沙子。", "带上你那件最花的衬衫，我们拍照！", "别找借口加班，生活也要有浪花。"]),
    ("pure_cn", "contentment", "expressive_low", None, None, None, ["雨停了，屋檐还在滴水。", "我坐在阳台，吃刚出锅的饺子。", "邻居在楼下散步，收音机放着老歌。", "这种普通的晚上，我居然觉得特别幸福。"]),
    ("pure_cn", "longing", "speed_slow", None, None, None, ["外婆做的酱茄子，我在外面再也没吃到过。", "每次回老家，第一件事就是开冰箱找那一小罐。", "她总说下次给你多做点，下次却越来越少。", "我想她了，想得心口发酸。"]),
    ("pure_cn", "determination", "speed_fast", None, None, None, ["截止日期就在周五，别再刷手机了。", "把任务拆开，一块一块啃掉。", "累了就站起来走两圈，但绝不放弃。", "做完那天，我会好好睡一觉。"]),
    ("pure_cn", "surprise", "expressive_high", None, None, None, ["箱子里怎么会有这张旧照片？！", "那是我们毕业旅行，你还扎着马尾。", "我以为早就丢了，结果藏在书的夹层里。", "时间真的很会捉弄人。"]),
    ("pure_cn", "affection", "expressive_low", None, None, None, ["谢谢你今天特意为我留了座位。", "其实你不说我也看得到，你总是先想到别人。", "这份细心，让我一整天都暖暖的。", "晚上想吃什么，我请你。"]),
    ("pure_cn", "anger", "expressive_high", None, None, None, ["快递又被扔在门口淋雨！贵重物品啊！", "我打了三通电话，客服只会道歉。", "下次再这样，我直接拒收并投诉到平台。", "把别人的东西当垃圾，太过分了。"]),
    ("pure_cn", "relief", "expressive_low", None, "sigh", "唉", ["唉，终于交稿了。光标闪了两小时我都不敢动。", "编辑回了个收到，我差点站起来鼓掌。", "现在只想点一份炸鸡，什么都不想。", "明天的事，明天再说。"]),
    ("pure_cn", "contemplation", "speed_slow", None, None, None, ["人到了某个年纪，会开始珍惜沉默。", "不是无话可说，是知道有些话不必说满。", "咖啡馆窗外的人来人往，各自赶路。", "我忽然觉得，慢一点也没什么不好。"]),
    ("pure_cn", "fear", "expressive_high", None, "screaming", "啊", ["啊——电梯突然晃了一下！", "我抓紧扶手，腿都软了。", "旁边的人也脸色发白。", "到楼层的时候我几乎是逃出去的。"]),
    ("pure_cn", "pride", "speed_slow", None, None, None, ["孩子第一次独立完成手工作业。", "胶水粘得到处都是，可他眼睛亮亮的。", "我没有插手，只在旁边鼓掌。", "那一刻我比拿奖还骄傲。"]),
    ("pure_cn", "confusion", "expressive_low", None, None, None, ["导航让我左转，可前面明明是单行道。", "我停在路口，后面车开始按喇叭。", "手机信号也卡了一下，地图转圈。", "算了，先靠边，问一下路人吧。"]),
    ("pure_cn", "elation", "speed_fast", None, None, None, ["签证下来了！绿色的通过字样我看了十遍。", "机票可以定了，行程单我已经草稿好了。", "背包、插头、防晒霜——列成了清单。", "终于可以去看那片海了！"]),
    ("pure_cn", "bitterness", "expressive_low", None, None, None, ["聚会上大家谈升职，我只能笑着听。", "不是羡慕不起，是提不起劲去解释。", "回家的地铁上，耳机里放着很吵的歌。", "有些失落，说出来反而更尴尬。"]),
    ("pure_cn", "helplessness", "expressive_low", None, None, None, ["医院走廊的灯白得刺眼。", "家属们坐成一排，有人哭有人沉默。", "我握着号码牌，不知道该祈什么。", "时间走得好慢，慢得不像话。"]),
    ("pure_cn", "awe", "expressive_high", None, None, None, ["烟花炸开的时候，整条江都亮了。", "人群齐声哇，连空气都在震。", "我仰着头，忘了拍照。", "原来有些美，只能用眼睛收藏。"]),
    ("pure_cn", "disgust", "speed_slow", None, "cough", "咳", ["咳，这公厕也太难闻了吧。", "地面黏黏的，我脚都不敢挪。", "赶紧出去洗手，还要洗两遍。", "下次打死都不进这层。"]),
    ("pure_cn", "shame", "speed_slow", None, None, None, ["我把客户的名字叫错了，还叫了两次。", "对方愣了一下，然后很礼貌地纠正我。", "我道歉到声音发虚。", "事后只想找个地缝钻进去。"]),
    ("pure_cn", "arousal", "expressive_high", None, None, None, ["倒计时三十秒，全场开始跺脚。", "鼓点一下比一下猛，我嗓子发干。", "灯束打到脸上，热得发烫。", "开口的瞬间，世界只剩我和麦克风。"]),
    ("pure_cn", "contentment", "speed_slow", None, None, None, ["下班路上买了烤红薯，烫手又香。", "撕开皮，甜气扑到脸上。", "一边走一边吃，也不在乎形象。", "冬天就该有这种小确幸。"]),
    ("pure_cn", "longing", "expressive_low", None, None, None, ["夜班结束后，城市像被按了静音。", "我想给你打电话，又怕吵醒你。", "只好发了条语音，说今晚的月亮很圆。", "不知道你醒来时，会不会先点开。"]),
    ("pure_cn", "enthusiasm", "pitch_high", None, None, None, ["新剧更新了！男女主终于对上戏了！", "弹幕刷得飞起，我跟着尖叫。", "停——这一段我要回看三遍。", "明天上班肯定顶着黑眼圈，值！"]),
    ("pure_cn", "sadness", "expressive_low", None, "sigh", "哎", ["哎，搬家的纸箱贴满胶带。", "有些东西舍不得扔，又不知道该放哪。", "旧信、旧票根，忽然都变得很重。", "原来离开一个地方，是一点一点撕开的。"]),
    ("pure_cn", "determination", "expressive_high", None, None, None, ["戒糖第七天，抽屉里的巧克力还在。", "我打开看了一眼，又关上。", "不是我不爱吃，是我更想对自己守信。", "再坚持三天，就过最难的那一关。"]),
    ("pure_cn", "amusement", "speed_fast", None, "laughter", "哈哈", ["哈哈，我把盐当糖放进了咖啡。", "喝第一口差点喷出来。", "室友笑到拍桌子，说这是黑暗料理。", "行吧，今天的笑话份额我包了。"]),
    ("pure_cn", "affection", "speed_slow", None, None, None, ["你发烧了怎么不早说？", "我煮了粥，还有你爱吃的小菜。", "药放在床头，闹钟也设好了。", "好好睡，我在客厅，有事喊我。"]),
    ("pure_cn", "surprise", "speed_fast", None, None, None, ["钱包找到了！夹在沙发缝里！", "证件、卡、现金，一样不少。", "我刚才差点把派出所电话拨出去。", "虚惊一场，腿还在发软。"]),
    ("pure_cn", "anger", "speed_fast", None, None, None, ["谁把我的车位占了？还贴张纸条？", "临时停一下，临时停了一下午！", "我绕了三圈才找到街边的位置。", "下次再占，我一定贴条上去。"]),
    ("pure_cn", "relief", "speed_slow", None, None, None, ["航班延误两小时，终于登机了。", "安全带扣上的那一刻，我才真正坐下。", "耳机一戴，世界安静许多。", "只要能飞走，晚点也认了。"]),
    ("pure_cn", "contemplation", "expressive_low", None, None, None, ["老照片里的我，笑得那么用力。", "那时候觉得未来无限长。", "现在回头看，原来最长的是当下。", "我轻轻合上相册，去把灯打开。"]),
    ("pure_cn", "fear", "speed_slow", None, None, None, ["体检前一晚，我反复查症状。", "越查越像自己，心越跳越快。", "明明知道可能是焦虑，还是控制不住。", "只想快点到明天，好或不好都有个答案。"]),
    ("pure_cn", "pride", "expressive_high", None, None, None, ["马拉松，我跑完了！", "膝盖在叫，肺在烧，可计时器是诚实的。", "奖牌挂在脖子上，沉甸甸的。", "证明给自己看：说到就能做到。"]),
    ("pure_cn", "confusion", "speed_slow", None, "humming", "嗯", ["嗯，合同这条款怎么读都别扭。", "律师说风险可控，我心里还是打鼓。", "要不要再问第三个人？", "签字笔握着，迟迟落不下去。"]),
    ("pure_cn", "elation", "expressive_high", None, "laughter", "哈哈", ["哈哈，奖金到账了！数字跳出来我愣了两秒。", "立刻还了一部分信用卡。", "剩下的，给自己买张去看海的票。", "努力这么久，也该被生活回一次抱。"]),
    ("pure_cn", "bitterness", "speed_slow", None, None, None, ["团建合影里，我站在最边上。", "不是没人叫我，是气氛里缺一块。", "回宿舍把照片裁掉了自己。", "有些不合群，不必硬解释成性格。"]),
    ("pure_cn", "helplessness", "speed_slow", None, "sigh", "唉", ["唉，系统又崩了，客户还在催。", "我能做的只有刷新，再刷新。", "解释听起来像借口，可我真的没办法。", "今晚大概又要通宵守着进度条。"]),
    ("pure_cn", "awe", "speed_slow", None, None, None, ["博物馆里那尊佛像，眼神像活的。", "我站了很久，不敢出声。", "耳麦里的讲解突然也变得多余。", "美有时候会按停时间。"]),
    ("pure_cn", "disgust", "expressive_high", None, None, None, ["谁在地铁里剪指甲啊？！", "声音清脆得像打击乐，我头皮发麻。", "周围人表情都很精彩。", "请把个人习惯留在家里，求求了。"]),
    ("pure_cn", "shame", "expressive_low", None, None, None, ["发言时我把关键词说岔了。", "全场安静一秒，然后有人轻咳。", "我装作镇定继续往下讲。", "散场后才敢捂脸又笑又崩溃。"]),
    ("pure_cn", "arousal", "speed_fast", None, None, None, ["比赛最后一球！全场站起来了！", "哨声响起前，我屏住呼吸。", "进了——！喇叭炸开，我跟着跳。", "嗓子哑了也值，这夜不会忘。"]),
    ("pure_cn", "contentment", "speed_slow", None, None, None, ["洗衣晾在阳台，风一吹就飘。", "肥皂香混着阳光，简单得不像话。", "我靠着门框发呆了好一会儿。", "原来平凡也可以很满。"]),
    ("pure_cn", "longing", "pitch_low", None, None, None, ["列车广播报着站名，越来越近家乡。", "窗外的田换成了熟悉的山。", "我把额头贴在玻璃上，像小时候那样。", "到站的时候，希望有人还在出口等。"]),
    ("pure_cn", "enthusiasm", "expressive_high", None, None, None, ["新开的展览我蹲了半个月！", "票根捏在手里，还温热。", "第一展厅的装置就让我愣住。", "走，我们慢慢逛，别赶时间。"]),
    ("pure_cn", "sadness", "speed_slow", None, None, None, ["毕业歌一响，礼堂里有人开始哭。", "帽子抛起来，又轻轻落下。", "我们说着保持联系，心里都清楚。", "青春退场的声音，其实很轻。"]),
]

ADULT_EN: list[tuple] = [
    ("pure_en", "contentment", "speed_slow", None, "sigh", "Ahh", ["Well… I finally sat down with nowhere to be.", "The window’s open, and the evening air feels kind of soft.", "I poured tea, then forgot about it on purpose.", "Nights like this don’t fix anything—they just make room to breathe."]),
    ("pure_en", "longing", "pitch_low", None, None, None, ["I walked past our old cafe without meaning to.", "Same chipped table by the window, same noisy espresso machine.", "I almost texted you, then put the phone away.", "Some places still know my name, even when I don’t say it."]),
    ("pure_en", "pride", "expressive_high", None, None, None, ["I finished the deck at 2 a.m.—and it actually held together.", "In the meeting they nodded like they’d been waiting for that version.", "I didn’t cheer out loud, but my hands were shaking a little.", "On the way out I bought the expensive coffee. I earned it."]),
    ("pure_en", "amusement", "expressive_high", None, "laughter", "Haha", ["Haha—my dog just stole a sock and negotiated with his eyes.", "I chased him around the couch like an idiot.", "He dropped it only after I offered a treat. Blackmail, basically.", "Okay, fine. He wins. I laugh anyway."]),
    ("pure_en", "contemplation", "speed_slow", None, None, None, ["On the train I kept thinking about a question someone asked years ago.", "What do you want to become? I answered too quickly then.", "Now I think the better answer is what I refuse to lose.", "Curiosity, maybe. Softness. The habit of noticing small things."]),
    ("pure_en", "sadness", "speed_slow", None, "sigh", "Uh", ["Uh… I found an old voice note while cleaning my phone.", "You sounded tired, but kind, like you always did.", "I played it twice and couldn’t press play a third time.", "I buried it in a folder and pretended that counts as moving on."]),
    ("pure_en", "enthusiasm", "speed_fast", None, None, None, ["Okay you have to hear this—there was no line at that new noodle place!", "The broth was ridiculous, like someone simmered patience all day.", "They even threw in an extra egg because I looked too happy.", "I’m already planning round two. My stomach can complain later."]),
    ("pure_en", "anger", "expressive_high", "shouting", None, None, ["I AM DONE WAITING IN THE COLD WHILE YOU IGNORE MY TEXTS!", "We said nine. You reply at nine-forty like that’s normal.", "Next time I’m leaving. No debate.", "Respecting someone’s time shouldn’t be this hard."]),
    ("pure_en", "affection", "speed_slow", "whispering", None, None, ["Come closer… I’m saying this quietly.", "That jacket looks better on you than you think.", "I’m not exaggerating. I mean it.", "Warm some milk when you get home—it’s gotten chilly."]),
    ("pure_en", "surprise", "pitch_high", None, None, None, ["Wait—say that again. You got the promotion?!", "I thought you were joking until the screenshot landed.", "Last week you were sure you weren’t ready.", "We’re celebrating tonight. Non-negotiable."]),
    ("pure_en", "relief", "speed_slow", None, "sigh", "Phew", ["Phew… the lab results are normal.", "I’ve been sleeping like a broken machine for weeks.", "My shoulders just dropped about two inches.", "Tonight I’m eating something ridiculous and not apologizing."]),
    ("pure_en", "fear", "speed_fast", None, None, None, ["Don’t turn the light off—I heard footsteps in the hall.", "Not imagining it. Clear. One after another.", "Stop laughing. My palms are soaked.", "Come with me. Please don’t make me check alone."]),
    ("pure_en", "determination", "expressive_high", None, None, None, ["I’m not backing down this time.", "I’ll send the applications. I’ll walk into the interviews.", "Failing is allowed. Freezing because I’m scared is not.", "Tomorrow I start early, one ugly step at a time."]),
    ("pure_en", "confusion", "expressive_low", None, "humming", "Hmm", ["Hmm… was that sarcasm, or am I overreading again?", "I scrolled the chat three times and still can’t tell.", "In person he smiled and said it was fine.", "Do I ask straight, or keep guessing myself sick?"]),
    ("pure_en", "bitterness", "speed_slow", None, "sigh", "Ahh", ["Ahh… the people I built with don’t even reply anymore.", "I’m not blaming anyone. It just feels hollow.", "The work scars are still there; the friendship turned polite.", "Growing up sometimes means finishing the story alone."]),
    ("pure_en", "elation", "expressive_high", None, "laughter", "Haha", ["Haha! I got the tickets—third row!", "My hands were shaking so hard I almost mistapped.", "Clear your weekend. I’m serious.", "We’re going to lose our voices and love every second."]),
    ("pure_en", "helplessness", "speed_slow", None, "sigh", "Uh", ["Uh… rent went up again. This month is basically food money.", "My resume vanishes into silence. No calls.", "I want to try, but I can’t see the next foothold.", "If you have ten minutes, can you just listen?"]),
    ("pure_en", "shame", "expressive_low", None, None, None, ["I… said the wrong number in the meeting.", "Everyone looked at me and my ears went hot.", "I sent a correction right after, but it still stung.", "Next time I check twice before I open my mouth."]),
    ("pure_en", "awe", "speed_slow", None, None, None, ["Look at that ridge—clouds are holding the peak up.", "One gust and the whole light changes.", "I went quiet without deciding to.", "Some views don’t need a filter. They just stop you."]),
    ("pure_en", "disgust", "expressive_high", None, None, None, ["Smell this milk—it’s gone, right?", "The date says yesterday. That’s absurd.", "I need to rinse my tongue somehow.", "Refund. Now. And a complaint."]),
    ("pure_en", "arousal", "pitch_high", None, None, None, ["My heart’s sprinting—ten minutes until I’m on.", "Sweaty hands. Lines somehow still stuck in my head.", "Breathe. Breathe. I can do this.", "When the curtain lifts, nerves become sound."]),
    ("pure_en", "amusement", "expressive_high", None, "laughter", "Hehe", ["Hehe—someone turned my name into a pun in the group chat.", "The whole office piled on. I wanted to disappear and also stay.", "Then the joke bounced back on him, even worse.", "Cheap joy, maybe. Still counts."]),
    ("pure_en", "sadness", "speed_slow", None, "crying", "Boohoo", ["Boohoo… the day we said goodbye to the dog, his collar stayed on the door.", "I still almost call his name when I walk in.", "His bowl is empty and I can’t put it away yet.", "A missing set of paws makes a house too quiet."]),
    ("pure_en", "enthusiasm", "expressive_high", None, None, None, ["Beach this weekend—I’ve already checked the trains!", "Leave morning, toes in sand by evening.", "Bring that ridiculous shirt. Photos are mandatory.", "No overtime excuses. Life needs a little splash."]),
    ("pure_en", "contentment", "expressive_low", None, None, None, ["Rain stopped. The eaves are still dripping.", "I’m on the balcony with fresh dumplings.", "Someone downstairs walks with an old radio song.", "Ordinary nights can feel strangely complete."]),
    ("pure_en", "longing", "speed_slow", None, None, None, ["Grandma’s eggplant sauce—I’ve never found it anywhere else.", "Home trips start with opening that fridge door.", "She always promised more next time. Next times got fewer.", "I miss her in a way that sits heavy in the chest."]),
    ("pure_en", "determination", "speed_fast", None, None, None, ["Deadline’s Friday. Phone down.", "Break the work into pieces and chew.", "Tired? Walk two laps. Then continue.", "When it’s done, I’ll sleep like I mean it."]),
    ("pure_en", "surprise", "expressive_high", None, None, None, ["Why is this old photo in the box?!", "Graduation trip. Your ponytail. That dumb grin.", "I thought it was gone—tucked in a book the whole time.", "Time loves little traps."]),
    ("pure_en", "affection", "expressive_low", None, None, None, ["Thanks for saving me a seat today.", "You don’t announce it, but you notice people first.", "That small kindness warmed the whole day.", "Dinner’s on me. Tell me what you want."]),
    ("pure_en", "anger", "expressive_high", None, None, None, ["The package sat in the rain again—fragile stuff!", "Three calls, and support only knew how to apologize.", "Next time I refuse delivery and escalate.", "Treating someone’s things like trash is not okay."]),
    ("pure_en", "relief", "expressive_low", None, "sigh", "Ahh", ["Ahh—manuscript submitted. The cursor blinked at me for hours.", "Editor replied ‘got it’ and I nearly stood up clapping.", "Now I want fried chicken and zero thoughts.", "Tomorrow can wait until tomorrow."]),
    ("pure_en", "contemplation", "speed_slow", None, None, None, ["At some age you start valuing silence.", "Not because there’s nothing to say—because not everything needs filling.", "Outside the cafe, people hurry in their own weather.", "Going slower suddenly doesn’t feel like losing."]),
    ("pure_en", "fear", "expressive_high", None, "screaming", "Ahh", ["Ahh—the elevator jolted!", "I grabbed the rail; my knees went soft.", "Everyone’s face said the same thing.", "I basically fled when the doors opened."]),
    ("pure_en", "pride", "speed_slow", None, None, None, ["First time the kid finished the craft alone.", "Glue everywhere, eyes bright.", "I didn’t fix it. I just clapped.", "Proud hits different when you stay out of the way."]),
    ("pure_en", "confusion", "expressive_low", None, None, None, ["Nav says left, but that’s a one-way street.", "I freeze; horns start.", "Signal stutters; map spins.", "Pull over. Ask a human. Done."]),
    ("pure_en", "elation", "speed_fast", None, None, None, ["Visa approved! I read the green word ten times.", "Flights can be booked. Draft itinerary already exists.", "Bag, adapter, sunscreen—listed.", "That ocean is finally getting closer."]),
    ("pure_en", "bitterness", "expressive_low", None, None, None, ["At the party everyone talked promotions; I smiled and nodded.", "Not jealousy exactly—more like no energy to explain.", "On the train home I blasted something loud.", "Some letdowns feel dumber when spoken aloud."]),
    ("pure_en", "helplessness", "expressive_low", None, None, None, ["Hospital hallway light is too white.", "Families sit in a row—some crying, some silent.", "I hold a number ticket and don’t know what to pray for.", "Time crawls in a way that feels unfair."]),
    ("pure_en", "awe", "expressive_high", None, None, None, ["When the fireworks opened, the whole river lit up.", "A shared wow moved through the crowd.", "I stared up and forgot to take a photo.", "Some beauty only fits in your eyes."]),
    ("pure_en", "disgust", "speed_slow", None, "cough", "Ahem", ["Ahem—this restroom smells criminal.", "Floor sticky; I barely want to move.", "Wash hands twice. Maybe a third.", "Never again on this floor."]),
    ("pure_en", "shame", "speed_slow", None, None, None, ["I mispronounced the client’s name. Twice.", "They paused, then corrected me politely.", "My apology came out thin.", "Afterwards I wanted the floor to open."]),
    ("pure_en", "arousal", "expressive_high", None, None, None, ["Thirty seconds. The crowd stamps.", "Drums hit harder; my throat goes dry.", "Lights on my face feel like heat.", "First note—and it’s just me and the mic."]),
    ("pure_en", "contentment", "speed_slow", None, None, None, ["Bought a hot sweet potato on the walk home.", "Peel splits; sugar-steam hits my face.", "I eat while walking, dignity optional.", "Winter should always have this kind of small win."]),
    ("pure_en", "longing", "expressive_low", None, None, None, ["After night shift the city feels muted.", "I want to call you and also don’t want to wake you.", "So I send a voice note about the round moon.", "Wonder if you’ll open that first."]),
    ("pure_en", "enthusiasm", "pitch_high", None, None, None, ["New episode dropped—and they finally shared a scene!", "Comments exploding; I’m yelling at the screen.", "Pause—replay that beat three times.", "Dark circles tomorrow. Worth it."]),
    ("pure_en", "sadness", "expressive_low", None, "sigh", "Ahh", ["Ahh… moving boxes sealed with too much tape.", "Things I can’t trash and can’t place.", "Old letters get heavy for no reason.", "Leaving a place is a slow tear, not a clean cut."]),
    ("pure_en", "determination", "expressive_high", None, None, None, ["Day seven without sugar. Chocolate still in the drawer.", "I opened it, looked, closed it.", "Not because I don’t want it—because I want my word more.", "Three more days and the hardest ridge is behind me."]),
    ("pure_en", "amusement", "speed_fast", None, "laughter", "Haha", ["Haha—I put salt in my coffee thinking it was sugar.", "First sip almost ended on the wall.", "Roommate slammed the table laughing—‘dark cuisine.’", "Fine. Today’s joke quota is on me."]),
    ("pure_en", "affection", "speed_slow", None, None, None, ["You’re feverish and you didn’t say?", "I made porridge and the side you like.", "Meds on the nightstand; alarm set.", "Sleep. I’m in the living room. Call if you need me."]),
    ("pure_en", "surprise", "speed_fast", None, None, None, ["Found the wallet—in the couch crack!", "Cards, cash, ID—all there.", "I nearly called the police station.", "False alarm. Legs still jelly."]),
    ("pure_en", "anger", "speed_fast", None, None, None, ["Who took my parking spot and left a note?", "‘Just a minute’ turned into an afternoon.", "I circled three times for street parking.", "Next time that note gets answered."]),
    ("pure_en", "relief", "speed_slow", None, None, None, ["Two-hour delay, finally boarding.", "Seatbelt click—and I actually sit.", "Headphones on; the world shrinks.", "As long as we leave, late is fine."]),
    ("pure_en", "contemplation", "expressive_low", None, None, None, ["Old-photo me smiled too hard.", "Back then the future felt endless.", "Looking back, the longest thing is right now.", "I close the album and turn a light on."]),
    ("pure_en", "fear", "speed_slow", None, None, None, ["Night before the checkup I spiral through symptoms.", "Everything starts sounding like me; heart races.", "I know it’s anxiety and still can’t stop.", "I just want morning—any answer."]),
    ("pure_en", "pride", "expressive_high", None, None, None, ["I finished the marathon!", "Knees screaming, lungs burning, clock honest.", "Medal heavy on my neck.", "Proof I can do what I said."]),
    ("pure_en", "confusion", "speed_slow", None, "humming", "Hmm", ["Hmm, this contract clause reads wrong both ways.", "Lawyer says risk is manageable; my gut disagrees.", "Ask a third person?", "Pen ready. Hand won’t drop."]),
    ("pure_en", "elation", "expressive_high", None, "laughter", "Haha", ["Haha—bonus hit! I stared at the number for two seconds.", "Paid down part of the card immediately.", "What’s left buys a ticket to the sea.", "After all that work, life can hug back once."]),
    ("pure_en", "bitterness", "speed_slow", None, None, None, ["In the team photo I’m at the edge.", "Not uninvited—just missing from the weather of it.", "I cropped myself out later.", "Some not-fitting doesn’t need a personality speech."]),
    ("pure_en", "helplessness", "speed_slow", None, "sigh", "Uh", ["Uh… system crashed again and the client’s still pinging.", "All I can do is refresh. Refresh.", "Explanations sound like excuses even when true.", "Looks like another night watching progress bars."]),
    ("pure_en", "awe", "speed_slow", None, None, None, ["That museum statue’s eyes feel alive.", "I stood too long and went quiet.", "The audio guide suddenly felt extra.", "Beauty can pause a clock."]),
    ("pure_en", "disgust", "expressive_high", None, None, None, ["Someone’s clipping nails on the subway?!", "That sound is a crime against my scalp.", "Faces around me are a whole movie.", "Please leave personal hygiene at home."]),
    ("pure_en", "shame", "expressive_low", None, None, None, ["I swapped a key word mid-speech.", "One second of silence, then a polite cough.", "I kept going like nothing happened.", "Afterwards: face-cover, laugh-crash combo."]),
    ("pure_en", "arousal", "speed_fast", None, None, None, ["Last play of the game—whole stadium up!", "Before the whistle I forget to breathe.", "It goes in—speakers explode; I jump.", "Voice gone. Night stays."]),
    ("pure_en", "contentment", "speed_slow", None, None, None, ["Laundry on the balcony lifts with the wind.", "Soap and sun, almost embarrassingly simple.", "I lean on the doorframe and zone out.", "Ordinary can still feel full."]),
    ("pure_en", "longing", "pitch_low", None, None, None, ["Station names get closer to home.", "Fields turn into the mountain I know.", "Forehead on the glass, kid habit returning.", "Hope someone’s still waiting at the exit."]),
    ("pure_en", "enthusiasm", "expressive_high", None, None, None, ["I waited a month for this exhibit!", "Ticket stub still warm in my hand.", "First room already stopped me cold.", "Let’s walk slow. No rushing."]),
    ("pure_en", "sadness", "speed_slow", None, None, None, ["Graduation song starts; someone cries in the hall.", "Caps rise, then fall soft.", "We promise to keep in touch, knowing the math.", "Youth exits quietly, not with a slam."]),
]

# 外企-style mixed: CN base + natural EN inserts
ADULT_MX: list[tuple] = [
    ("mixed", "contentment", "speed_slow", None, "sigh", "唉", ["唉，今天的 standup 居然准时结束了。", "我回 desk 倒了杯咖啡，耳机一戴世界安静。", "Slack 红点先不点开，给自己五分钟。", "偶尔这样不 rush，反而更像活人。"]),
    ("mixed", "longing", "pitch_low", None, None, None, ["又路过以前常去的 coffee shop。", "角落那张桌还在，像我们的 unofficial meeting room。", "我差点发消息，又把手机扣下去。", "有些 place，适合 miss，不适合 revisit。"]),
    ("mixed", "pride", "expressive_high", None, None, None, ["这版 proposal 我改到凌晨两点！", "客户在 call 上直接说 this is what we needed。", "我表面很 calm，手指却在抖。", "散会后我给自己买了杯 oat latte，算奖励。"]),
    ("mixed", "amusement", "expressive_high", None, "laughter", "哈哈", ["哈哈，谁把我名字在 group chat 里改成了 pun。", "全办公室跟风，我又羞又想笑。", "后来 joke 反噬到始作俑者，更离谱。", "这种廉价快乐，居然挺 heal 的。"]),
    ("mixed", "contemplation", "speed_slow", None, None, None, ["地铁上突然想起一个 old question。", "你想成为什么样的人？我当年答得太快。", "现在觉得更重要的是 don’t lose curiosity。", "车门开了，我走进人潮，心里反而 quiet。"]),
    ("mixed", "sadness", "speed_slow", None, "sigh", "唉", ["唉……整理手机翻到一段 voice note。", "你走前一天录的，声音轻得像怕吵醒谁。", "我听了两遍，不敢第三遍。", "存进很深的 folder，像藏一块未愈的伤。"]),
    ("mixed", "enthusiasm", "speed_fast", None, None, None, ["你知道吗！那家新开的面馆 tonight 居然 no line！", "汤底浓到像 simmered all day，辣油香疯了。", "老板看我吃得开心，extra 送了卤蛋。", "我想再来一碗，stomach 已经在抗议。"]),
    ("mixed", "anger", "expressive_high", None, None, None, ["约好 nine，你九点四十才回 message！", "电话不接，location 也不开，让我在路边吹风。", "下次再这样，我直接 leave，不等了。", "Respect 别人的时间，很难吗？"]),
    ("mixed", "affection", "speed_slow", "whispering", None, None, ["过来一点……我小声说。", "你今天这件 jacket，真的比你以为的好看。", "我不是在 overpraise，是认真的。", "回家记得热牛奶，天气 cool down 了。"]),
    ("mixed", "surprise", "pitch_high", None, None, None, ["等一下——你说 promotion？认真的？！", "我以为在开玩笑，结果邮件 screenshot 甩过来。", "上周你还在说 I’m not ready。", "走，tonight 必须 celebrate！"]),
    ("mixed", "relief", "speed_slow", None, "sigh", "呼", ["呼……体检 report 出来了，指标都 normal。", "这一个月我每天 sleep poorly。", "现在肩膀像卸了 block，整个人轻了。", "今晚吃点好的，diet 暂停。"]),
    ("mixed", "fear", "speed_fast", None, None, None, ["别关灯！我听见走廊有 footsteps。", "不是错觉，一下一下很 clear。", "你别笑，我 palms 全是汗。", "一起去 check，好吗？别让我一个人。"]),
    ("mixed", "determination", "expressive_high", None, None, None, ["这次我不退了。简历投出去，interview 我也去。", "失败可以，但我不想 freeze because of fear。", "明天开始早起，一点一点来。", "我要相信：我配得上更好的 chance。"]),
    ("mixed", "confusion", "expressive_low", None, "humming", "嗯", ["嗯……他那句到底是 sarcasm，还是我 overthink？", "聊天记录翻了三遍，语气像在生气。", "当面他又笑着说 it’s fine。", "要不要直接 ask clear？我真的搞不懂。"]),
    ("mixed", "bitterness", "speed_slow", None, "sigh", "唉", ["唉，一起创业的人，现在都不回 message 了。", "不是怪谁，就是 hollow。", "努力痕迹还在，关系淡成了 polite。", "成长有时是学会独自把 story 讲完。"]),
    ("mixed", "elation", "expressive_high", None, "laughter", "哈哈", ["哈哈！票抢到了！third row！", "页面刷新时手在抖，差点 mistap。", "周末 clear 一下，别安排别的。", "我们要疯一整个 night！"]),
    ("mixed", "helplessness", "speed_slow", None, "sigh", "唉", ["唉……房租又涨，这个月 basically 只够吃饭。", "简历石沉大海，no calls。", "我也想努力，可看不到 next step。", "你要是有空，能不能 just listen？"]),
    ("mixed", "shame", "expressive_low", None, None, None, ["我在会议上说错了一个 key number。", "所有人看过来，我耳朵 hot 得不行。", "散会立刻发了 correction，还是丢脸。", "下次 check twice，再开口。"]),
    ("mixed", "awe", "speed_slow", None, None, None, ["你看那座山，云像把峰托起来。", "风一过，光线整个 shift。", "我突然 quiet，只想多站一会儿。", "有些 view，不需要 filter。"]),
    ("mixed", "disgust", "expressive_high", None, None, None, ["这牛奶——smell 一下，明显坏了吧？", "日期还是昨天，太 absurd。", "我舌头都觉得 dirty。", "Refund，必须，还要 complain。"]),
    ("mixed", "arousal", "pitch_high", None, None, None, ["心跳好快……十分钟后轮到我 on stage。", "手心全是汗，台词却还在。", "Deep breath——我能行。", "帷幕拉开那秒，nerves 会变成声音。"]),
    ("mixed", "amusement", "speed_fast", None, "laughter", "嘿嘿", ["嘿嘿，我把 salt 当糖放进咖啡。", "第一口差点喷——dark cuisine。", "室友拍桌子笑，说这是我的 signature。", "行，今天的 joke quota 我包了。"]),
    ("mixed", "sadness", "speed_slow", None, "crying", "呜", ["呜……小狗走的那天，collar 还挂在门上。", "每次开门都差点叫它的名字。", "空碗到现在不敢收，太 quiet。", "家里少一个脚步声，会空成这样。"]),
    ("mixed", "enthusiasm", "expressive_high", None, None, None, ["Weekend 去海边吧！火车我查好了。", "早上出发，傍晚就能 touch sand。", "带上你那件花衬衫，photos 必须有。", "别用 overtime 当借口，生活也要有浪。"]),
    ("mixed", "contentment", "expressive_low", None, None, None, ["雨停了，屋檐还在 drip。", "阳台吃刚出锅的饺子，simple 得不像话。", "楼下有人散步，收音机放 old song。", "普通的晚上，居然很 complete。"]),
    ("mixed", "longing", "speed_slow", None, None, None, ["外婆的酱茄子，外面再也没吃到。", "回老家第一件事就是 open fridge 找那一罐。", "她总说 next time 多做点，next time 越来越少。", "我想她了，想得胸口发酸。"]),
    ("mixed", "determination", "speed_fast", None, None, None, ["Deadline 就在周五，别再刷手机。", "任务拆开，一块一块 chew。", "累了走两圈，但不 quit。", "做完那天，我要 sleep hard。"]),
    ("mixed", "surprise", "expressive_high", None, None, None, ["箱子里怎么会有这张旧 photo？！", "毕业旅行，你还扎着 ponytail。", "我以为丢了，结果藏在 book 夹层。", "Time 真的很会捉弄人。"]),
    ("mixed", "affection", "expressive_low", None, None, None, ["谢谢你今天为我留 seat。", "你不说我也看得到，你总是 notice 别人先。", "这份细心，暖了我一整天。", "晚饭我请，tell me what you want。"]),
    ("mixed", "anger", "expressive_high", None, None, None, ["快递又扔门口淋雨！fragile 的啊！", "打了三通电话，support 只会 apologize。", "下次拒收，并 escalate 到平台。", "把别人东西当 trash，过分。"]),
    ("mixed", "relief", "expressive_low", None, "sigh", "唉", ["唉，终于 submit 了。光标闪了两小时。", "编辑回了个 got it，我差点鼓掌。", "现在只想 fried chicken，零思考。", "Tomorrow 的事，tomorrow 再说。"]),
    ("mixed", "contemplation", "speed_slow", None, None, None, ["到了某个年纪，会开始珍惜 silence。", "不是无话，是知道不必 fill every gap。", "窗外人来人往，各自 rush。", "慢一点，突然不觉得是 lose。"]),
    ("mixed", "fear", "expressive_high", None, "screaming", "啊", ["啊——电梯突然 jolt！", "我抓扶手，膝盖 soft。", "旁边人脸也白了。", "到层我几乎是 flee 出去的。"]),
    ("mixed", "pride", "speed_slow", None, None, None, ["孩子第一次独立做完 craft。", "胶水到处都是，眼睛却 bright。", "我没插手，只在旁边 clap。", "那一刻比拿奖还 proud。"]),
    ("mixed", "confusion", "expressive_low", None, None, None, ["导航说左转，前面却是 one-way。", "我停住，后面开始按喇叭。", "信号卡了，map 转圈。", "靠边，ask a human 吧。"]),
    ("mixed", "elation", "speed_fast", None, None, None, ["签证下来了！绿色 approved 我看了十遍。", "机票可以 book，行程草稿已有。", "背包、adapter、防晒——listed。", "那片海终于近了！"]),
    ("mixed", "bitterness", "expressive_low", None, None, None, ["聚会上大家谈 promotion，我只能笑着听。", "不是羡慕不起，是没 energy 解释。", "地铁上耳机开很吵的歌。", "有些失落，说出来更 awkward。"]),
    ("mixed", "helplessness", "expressive_low", None, None, None, ["医院走廊灯白得刺眼。", "家属坐成一排，有人哭有人 silent。", "我握着号码牌，不知 pray 什么。", "时间慢得 unfair。"]),
    ("mixed", "awe", "expressive_high", None, None, None, ["烟花炸开，整条江 lit up。", "人群齐声 wow，空气都在震。", "我仰头，忘了拍照。", "有些美，只能用眼睛 keep。"]),
    ("mixed", "disgust", "speed_slow", None, "cough", "咳", ["咳，这 restroom 也太难闻。", "地 sticky，脚都不敢挪。", "出去洗手两遍。", "这层 never again。"]),
    ("mixed", "shame", "speed_slow", None, None, None, ["客户名字我叫错两次。", "对方 pause 一下，礼貌纠正。", "我道歉到声音 thin。", "事后只想钻地缝。"]),
    ("mixed", "arousal", "expressive_high", None, None, None, ["倒计时三十秒，全场 stamp。", "鼓点更猛，嗓子 dry。", "灯打在脸上像 heat。", "开口瞬间，只剩我和 mic。"]),
    ("mixed", "contentment", "speed_slow", None, None, None, ["下班买了烤红薯，烫手又香。", "撕开皮，甜气 hit 脸。", "边走边吃，形象 optional。", "冬天就该有这种 small win。"]),
    ("mixed", "longing", "expressive_low", None, None, None, ["夜班结束，城市像 mute。", "想打电话，又怕 wake you。", "发了条语音，说月亮很 round。", "不知道你醒来会不会 first open。"]),
    ("mixed", "enthusiasm", "pitch_high", None, None, None, ["新剧更新了！终于有对戏！", "弹幕爆炸，我对着屏幕 yell。", "停——这段 replay 三遍。", "明天黑眼圈，worth it。"]),
    ("mixed", "sadness", "expressive_low", None, "sigh", "哎", ["哎，搬家纸箱贴满胶带。", "有些东西 can’t trash，又不知放哪。", "旧信忽然很 heavy。", "离开一个地方，是 slow tear。"]),
    ("mixed", "determination", "expressive_high", None, None, None, ["戒糖第七天，抽屉巧克力还在。", "打开看一眼，又关上。", "不是不想吃，是更想 keep my word。", "再三天，最难的 ridge 就过了。"]),
    ("mixed", "amusement", "expressive_high", None, "laughter", "哈哈", ["哈哈，会议室 camera 一直开着我还在整理头发。", "有人在 chat 里打了个 wink emoji。", "我假装镇定，继续讲 slide。", "散会后才允许自己 facepalm。"]),
    ("mixed", "affection", "speed_slow", None, None, None, ["你发烧怎么不早说？", "粥煮好了，还有你爱的 side。", "药在床头，alarm 设好。", "睡吧，我在客厅，need me 就喊。"]),
    ("mixed", "surprise", "speed_fast", None, None, None, ["钱包找到了！夹在 couch 缝里！", "证件现金都在，一样不少。", "我差点打给 police station。", "虚惊一场，腿还 jelly。"]),
    ("mixed", "anger", "speed_fast", None, None, None, ["谁占我车位还留 note？", "临时一下，停了一下午！", "我绕了三圈找 street parking。", "下次再占，我一定回应那张 note。"]),
    ("mixed", "relief", "speed_slow", None, None, None, ["航班 delay 两小时，终于 boarding。", "安全带扣上，我才真正 sit。", "耳机一戴，世界 shrink。", "只要能飞，late 也认了。"]),
    ("mixed", "contemplation", "expressive_low", None, None, None, ["老照片里的我，笑得太用力。", "那时觉得 future endless。", "回头看，最长的是 right now。", "合上相册，把灯 turn on。"]),
    ("mixed", "fear", "speed_slow", None, None, None, ["体检前一晚，我反复搜 symptoms。", "越搜越像自己，心跳 race。", "知道是 anxiety，还是停不了。", "只想快点到 morning，要个答案。"]),
    ("mixed", "pride", "expressive_high", None, None, None, ["马拉松，我跑完了！", "膝盖叫，肺在烧，clock 很诚实。", "奖牌挂脖上，沉甸甸。", "证明给自己：说到就能做到。"]),
    ("mixed", "confusion", "speed_slow", None, "humming", "嗯", ["嗯，这合同条款怎么读都别扭。", "律师说 risk manageable，我心里打鼓。", "要不要再 ask 第三个人？", "笔握着，迟迟落不下去。"]),
    ("mixed", "elation", "expressive_high", None, "laughter", "哈哈", ["哈哈，奖金到账！数字跳出来我愣两秒。", "立刻还了一部分 card。", "剩下的买张去海边的票。", "努力这么久，也该被生活 hug back。"]),
    ("mixed", "bitterness", "speed_slow", None, None, None, ["团建合影，我站在 edge。", "不是没人叫，是气氛缺一块。", "回宿舍把照片 crop 掉自己。", "有些不合群，不必讲成 personality。"]),
    ("mixed", "helplessness", "speed_slow", None, "sigh", "唉", ["唉，系统又 crash，客户还在催。", "我只能 refresh，再 refresh。", "解释听起来像 excuse，可真的没办法。", "今晚大概又要守 progress bar。"]),
    ("mixed", "awe", "speed_slow", None, None, None, ["博物馆那尊像，眼神像 alive。", "我站很久，不敢出声。", "耳麦讲解忽然变得 extra。", "美有时会 pause 时间。"]),
    ("mixed", "disgust", "expressive_high", None, None, None, ["谁在地铁剪指甲？！", "声音 crisp 得像打击乐，头皮发麻。", "周围人表情是整部 movie。", "个人习惯请 leave at home。"]),
    ("mixed", "shame", "expressive_low", None, None, None, ["发言时我把关键词说岔。", "全场 quiet 一秒，有人轻咳。", "我假装镇定继续讲。", "散场后才敢 facepalm 兼崩溃。"]),
    ("mixed", "arousal", "speed_fast", None, None, None, ["最后一球！全场站起来！", "哨响前我屏住呼吸。", "进了——喇叭 explode，我跟着跳。", "嗓子哑了也值，这夜 stays。"]),
    ("mixed", "contentment", "speed_slow", None, None, None, ["洗衣在阳台被风吹起。", "肥皂香混阳光，simple 得不像话。", "我靠门框发呆一会儿。", "平凡也可以很 full。"]),
    ("mixed", "longing", "pitch_low", None, None, None, ["列车报站，越来越近 hometown。", "窗外田换成熟悉的山。", "额头贴玻璃，像小时候。", "到站时，希望有人还在 exit 等。"]),
    ("mixed", "enthusiasm", "expressive_high", None, None, None, ["这个展我蹲了半个月！", "票根还 warm。", "第一展厅就让我 stop。", "慢慢逛，别 rush。"]),
]

CHILD_CN: list[tuple] = [
    ("pure_cn", "enthusiasm", "expressive_high", None, None, None, ["今天幼儿园发了星星贴纸！", "老师说我把玩具都收好了。", "我把贴纸贴在书包上，走路都想跑。", "回家第一件事就是给妈妈看！"]),
    ("pure_cn", "amusement", "expressive_high", None, "laughter", "哈哈", ["哈哈，小鸭子跟着我呱呱叫。", "我一跑它也跑，像在玩游戏。", "后来它撞到水盆，湿答答的。", "我笑得好大声，它还看着我。"]),
    ("pure_cn", "affection", "speed_slow", None, None, None, ["妈妈，今晚可以再讲一个故事吗？", "我喜欢那只勇敢的小兔子。", "你声音轻轻的，我就不怕黑了。", "讲完我要抱抱你，再睡觉。"]),
    ("pure_cn", "surprise", "pitch_high", None, None, None, ["哇！蛋糕上有我的名字！", "蜡烛亮亮的，像小星星。", "我要先许愿，再吹气。", "愿望是保密的，嘿嘿。"]),
    ("pure_cn", "sadness", "speed_slow", None, "sigh", "唉", ["唉，气球飞走了。", "我举得好高，它还是跑了。", "天空那么大，我找不到它。", "妈妈说可以再买一个，可我还是有点难受。"]),
    ("pure_cn", "contentment", "speed_slow", None, None, None, ["雨停了，地上有小水坑。", "我穿着雨靴去踩水，啪啪响。", "裤脚湿了一点点，也不要紧。", "这种声音好好听。"]),
    ("pure_cn", "fear", "speed_slow", None, None, None, ["打雷了，我把被子盖住耳朵。", "灯一闪一闪，我有点怕。", "爸爸坐在床边，握住我的手。", "有他在，雷声就不那么凶了。"]),
    ("pure_cn", "pride", "expressive_high", None, None, None, ["我自己把鞋带系好了！", "老师表扬我，还拍了拍手。", "以前总要妈妈帮忙的。", "今天我也可以帮弟弟！"]),
    ("pure_cn", "confusion", "expressive_low", None, "humming", "嗯", ["嗯……这块积木该放哪里？", "红色的好像不稳。", "换蓝色试试——哎，倒了。", "再来一次，我一定能搭高。"]),
    ("pure_cn", "elation", "speed_fast", None, "laughter", "耶", ["耶！秋千荡得好高！", "风吹在脸上凉凉的。", "再推一次，再推一次！", "我像一只小鸟！"]),
    ("pure_cn", "longing", "speed_slow", None, None, None, ["我想奶奶了。", "她总会给我剥橘子。", "电话里听她声音，就想过去。", "周末我们去看她，好不好？"]),
    ("pure_cn", "determination", "expressive_high", None, None, None, ["这次骑车我不要求扶了。", "摔一下也没关系，我自己爬起来。", "看，脚踩稳，把手抓好。", "我能骑到那棵树那儿！"]),
    ("pure_cn", "relief", "speed_slow", None, "sigh", "呼", ["呼……找不到的橡皮在抽屉里。", "我找了好久，差点哭出来。", "原来它躲在彩笔旁边。", "下次我要放回原位。"]),
    ("pure_cn", "awe", "speed_slow", None, None, None, ["天上的云像一只大绵羊。", "它慢慢走，我不眨眼。", "太阳从云后面钻出来，金金的。", "好漂亮，我想画下来。"]),
    ("pure_cn", "disgust", "expressive_high", None, None, None, ["这个青菜好苦！", "我嚼两下就想吐。", "能不能少吃一口？求求了。", "下次我吃胡萝卜，好不好？"]),
    ("pure_cn", "amusement", "expressive_high", None, "laughter", "嘻嘻", ["嘻嘻，猫咪追自己的尾巴。", "转呀转，转得头晕。", "它倒下装睡，一只眼还睁着。", "小猫好笨，又好可爱。"]),
    ("pure_cn", "enthusiasm", "pitch_high", None, None, None, ["明天春游要去公园！", "我要带小水壶和帽子。", "还可以在草地上打滚。", "晚上我一定睡不着！"]),
    ("pure_cn", "sadness", "expressive_low", None, None, None, ["最好的朋友搬家了。", "我们拉钩说会写信。", "可操场少了一个人，空空的。", "我会想他，每天想一点。"]),
    ("pure_cn", "affection", "speed_slow", "whispering", None, None, ["哥哥，小声说哦。", "你的手工飞机飞得好远。", "我好喜欢，能不能教我？", "学会了，我们一起飞。"]),
    ("pure_cn", "surprise", "expressive_high", None, None, None, ["书包里怎么有糖？！", "是妈妈偷偷放的吗？", "甜甜的，柠檬味。", "谢谢妈妈，我好开心！"]),
    ("pure_cn", "contentment", "expressive_low", None, None, None, ["洗澡水暖暖的。", "泡沫堆成小胡子，我照镜子笑。", "洗完穿上软软的睡衣。", "今天也是好日。"]),
    ("pure_cn", "fear", "expressive_low", None, None, None, ["黑夜里衣柜好像会动。", "我不敢看，把灯打开一点点。", "原来是衣服架子晃了一下。", "虚惊一场，我还是抱抱枕头。"]),
    ("pure_cn", "pride", "speed_slow", None, None, None, ["画画得了小红花。", "我画了我家和太阳。", "老师说颜色很大胆。", "我要把画贴在冰箱上。"]),
    ("pure_cn", "helplessness", "speed_slow", None, "sigh", "唉", ["唉，积木塔又倒了。", "我搭了好多层，一下子没了。", "想哭，可眼泪忍住了。", "你能不能帮我扶一下底部？"]),
    ("pure_cn", "elation", "expressive_high", None, None, None, ["雪！真的下雪了！", "我伸出舌头接雪花。", "凉凉的，一下子化掉。", "我们堆个胖胖的雪人吧！"]),
]

CHILD_EN: list[tuple] = [
    ("pure_en", "enthusiasm", "expressive_high", None, None, None, ["I got a star sticker at school today!", "Teacher said I cleaned up all the toys.", "I stuck it on my backpack and almost ran home.", "Mom has to see it first thing!"]),
    ("pure_en", "amusement", "expressive_high", None, "laughter", "Haha", ["Haha—the duckling followed me quacking!", "I ran and it ran, like a game.", "Then it bumped the water bowl. Soaked.", "I laughed so loud and it just stared."]),
    ("pure_en", "affection", "speed_slow", None, None, None, ["Mom, one more story tonight?", "I like the brave little rabbit.", "Your soft voice makes the dark okay.", "After the story I need a hug, then sleep."]),
    ("pure_en", "surprise", "pitch_high", None, None, None, ["Wow! My name is on the cake!", "The candles look like tiny stars.", "I wish first, then blow.", "The wish is secret. Hehe."]),
    ("pure_en", "sadness", "speed_slow", None, "sigh", "Ahh", ["Ahh… my balloon flew away.", "I held it high and it still ran off.", "The sky is too big to find it.", "Mom says we can buy another, but I still feel sad."]),
    ("pure_en", "contentment", "speed_slow", None, None, None, ["Rain stopped. Puddles everywhere.", "I stomp in boots—splash splash.", "Pants a little wet. That’s okay.", "I love that sound."]),
    ("pure_en", "fear", "speed_slow", None, None, None, ["Thunder. I cover my ears with the blanket.", "Lights flicker and I get scared.", "Dad sits by the bed and holds my hand.", "With him here, thunder isn’t so mean."]),
    ("pure_en", "pride", "expressive_high", None, None, None, ["I tied my shoes by myself!", "Teacher clapped for me.", "Mom used to help every time.", "Today I can help my little brother too!"]),
    ("pure_en", "confusion", "expressive_low", None, "humming", "Hmm", ["Hmm… where does this block go?", "The red one feels wobbly.", "Try blue—oh no, it fell.", "One more try. I can build it tall."]),
    ("pure_en", "elation", "speed_fast", None, "laughter", "Yay", ["Yay! The swing goes so high!", "Wind on my face feels cool.", "Push again, push again!", "I’m like a little bird!"]),
    ("pure_en", "longing", "speed_slow", None, None, None, ["I miss Grandma.", "She always peels oranges for me.", "Hearing her on the phone makes me want to visit.", "Can we go this weekend?"]),
    ("pure_en", "determination", "expressive_high", None, None, None, ["No holding the bike this time.", "If I fall, I get up myself.", "Feet steady, hands tight.", "I can ride to that tree!"]),
    ("pure_en", "relief", "speed_slow", None, "sigh", "Phew", ["Phew… the eraser was in the drawer.", "I looked forever and almost cried.", "It was hiding by the crayons.", "Next time I put it back."]),
    ("pure_en", "awe", "speed_slow", None, None, None, ["That cloud looks like a big sheep.", "It walks slow. I don’t blink.", "Sun peeks out all golden.", "So pretty. I want to draw it."]),
    ("pure_en", "disgust", "expressive_high", None, None, None, ["This green veggie tastes bitter!", "Two chews and I want to spit.", "Can I have a tiny bite? Please?", "Next time carrots, okay?"]),
    ("pure_en", "amusement", "expressive_high", None, "laughter", "Hehe", ["Hehe—kitty chases its own tail.", "Spin spin, then dizzy.", "It flops and pretends to sleep—one eye open.", "Silly cat. So cute."]),
    ("pure_en", "enthusiasm", "pitch_high", None, None, None, ["Field trip to the park tomorrow!", "I’m packing my bottle and hat.", "We can roll on the grass!", "I won’t sleep tonight!"]),
    ("pure_en", "sadness", "expressive_low", None, None, None, ["My best friend moved away.", "We pinkie-promised to write.", "The playground feels empty now.", "I miss him a little every day."]),
    ("pure_en", "affection", "speed_slow", "whispering", None, None, ["Brother, whisper okay?", "Your paper plane flew so far.", "I love it—can you teach me?", "Then we fly together."]),
    ("pure_en", "surprise", "expressive_high", None, None, None, ["There’s candy in my backpack?!", "Did Mom hide it?", "Lemon sweet. Yum.", "Thank you Mom! I’m so happy!"]),
    ("pure_en", "contentment", "expressive_low", None, None, None, ["Bath water is warm.", "Bubbles make a mustache—I laugh in the mirror.", "Then soft pajamas.", "Today was a good day."]),
    ("pure_en", "fear", "expressive_low", None, None, None, ["At night the closet seems to move.", "I peek with a tiny light.", "Just a hanger swinging.", "Still hugging my pillow though."]),
    ("pure_en", "pride", "speed_slow", None, None, None, ["I got a red flower for my drawing.", "I drew our house and the sun.", "Teacher said my colors are brave.", "I’m sticking it on the fridge."]),
    ("pure_en", "helplessness", "speed_slow", None, "sigh", "Ahh", ["Ahh… my block tower fell again.", "So many layers—gone.", "I want to cry but I hold it.", "Can you help hold the bottom?"]),
    ("pure_en", "elation", "expressive_high", None, None, None, ["Snow! Real snow!", "I catch flakes on my tongue.", "Cold, then gone.", "Let’s build a chubby snowman!"]),
]


# Extra spoken beats appended so clean text reaches ~20–30s speech
# (measured ~7 CN chars/s, ~3 EN words/s on good clones).
EXTRA_CN = [
    "我把杯子放回桌上，指尖还留着一点温度。",
    "窗外有车驶过，灯光在墙上晃了一下。",
    "说真的，这种细节平时根本注意不到。",
    "我深吸一口气，决定把心里话慢慢讲清楚。",
    "后来我想了很久，还是觉得那一步走得对。",
    "你要是当时在场，大概也会愣住两秒。",
    "空气里有点潮，像要下雨又没下下来。",
    "我摸了摸口袋，钥匙还在，心却跳得有点快。",
    "这件事听起来不大，可它卡在我胸口好几天。",
    "我试着笑一笑，发现笑比我预想的更难。",
    "街对面的灯牌闪了闪，像在催我往前走。",
    "我把手机屏幕按灭，忽然觉得世界安静了。",
    "有些话到了嘴边，又被我生生咽回去。",
    "再过一会儿，夜色会更沉，人也更诚实。",
    "我听见自己的呼吸，一下一下很清楚。",
    "那一刻我忽然明白，原来我一直在等一个答案。",
    "风从楼道灌进来，带着一点灰尘味。",
    "我抬起头，天花板的裂缝像一条细细的河。",
    "别急着打断我，让我把后半段说完。",
    "其实我也怕说错，可沉默更让人难受。",
    "我把外套拉链拉上，像给自己加了一层壳。",
    "远处有人喊名字，不是在喊我，我却回了一下头。",
    "记忆里的声音忽然清晰，像被重新按下播放。",
    "我在原地站了几秒，才迈出下一步。",
    "这个问题没有标准答案，可我还是想试着答。",
    "茶凉了也没关系，凉的刚好能喝。",
    "我把椅子挪近一点，膝盖碰到桌沿。",
    "你听，外面的雨点开始变密了。",
    "我把那句话又在心里重复了一遍。",
    "有时候勇敢不是冲出去，是留下来把话说开。",
    "灯忽明忽暗，我的影子在墙上晃。",
    "我攥紧拳头，又慢慢松开手指。",
    "这件事过后，我和世界的距离好像近了一点。",
    "我笑出声，自己也觉得有点不好意思。",
    "时间过得好慢，慢到我能数清心跳。",
    "我把杯子转了半圈，水痕在桌面上画圆。",
    "说到底，我只是想被认真听完。",
    "走廊尽头的门开了一条缝，光漏进来。",
    "我把耳机摘下，周围的声音一下子涌回来。",
    "那一刻我没有哭，只是鼻子有点酸。",
    "我把计划又改了一版，写得更具体了。",
    "你放心，这次我不会再半途而废。",
    "我抬脚跨过水洼，鞋底溅起一点泥。",
    "心里那块石头，好像终于松动了一点。",
    "我把窗帘拉开一半，城市还醒着。",
    "有些温柔来得很晚，但来了就够了。",
    "我把故事讲到这里，剩下的交给沉默。",
    "夜风贴着窗户走，像有人轻轻敲门。",
    "我点点头，像是给自己一个确认。",
    "这一段路不长，可足够我想清楚很多事。",
]

EXTRA_EN = [
    "I set the cup down and felt the warmth leave my fingertips.",
    "A car passed outside, and light slid across the wall for a second.",
    "Honestly, I usually miss details like that.",
    "I took a slow breath and decided to say it properly.",
    "I thought about it for a long time and still think that step was right.",
    "If you had been there, you probably would have frozen too.",
    "The air felt damp, like rain that never quite arrives.",
    "I checked my pocket—keys still there—but my heart was racing.",
    "It sounds small, yet it sat on my chest for days.",
    "I tried to smile and realized it was harder than I expected.",
    "The sign across the street flickered, like it was pushing me forward.",
    "I turned the phone screen off and the room went quiet.",
    "Some words rose to my mouth, then I swallowed them again.",
    "In a little while the night will get heavier, and people get more honest.",
    "I could hear my own breathing, clear and steady.",
    "That was when I understood I had been waiting for an answer.",
    "Wind poured through the stairwell with a dusty smell.",
    "I looked up; a crack in the ceiling looked like a thin river.",
    "Don’t cut me off yet—let me finish the second half.",
    "I’m afraid of saying it wrong, but silence feels worse.",
    "I zipped my jacket like I was adding another layer of armor.",
    "Someone called a name in the distance—not mine—but I still turned.",
    "A voice from memory came back sharp, like someone hit play again.",
    "I stood still for a few seconds before taking the next step.",
    "There’s no perfect answer, but I still want to try.",
    "The tea went cold; cold was fine—I could drink it anyway.",
    "I pulled the chair closer and my knee bumped the table edge.",
    "Listen—the rain is getting denser outside.",
    "I repeated that sentence once more in my head.",
    "Sometimes bravery isn’t running out; it’s staying to talk it through.",
    "The lamp flickered and my shadow shook on the wall.",
    "I clenched my fist, then slowly opened my fingers.",
    "After that, the distance between me and the world felt smaller.",
    "I laughed out loud and immediately felt a little shy.",
    "Time moved so slowly I could count my heartbeat.",
    "I turned the cup halfway; a water ring drew a circle on the table.",
    "In the end I just wanted to be heard all the way through.",
    "A door at the corridor end cracked open and light leaked in.",
    "I took the earphones off and the world rushed back in.",
    "I didn’t cry then—just a sting in my nose.",
    "I revised the plan again and made every line more concrete.",
    "Trust me—this time I won’t quit halfway.",
    "I stepped over a puddle and mud flicked onto my shoe.",
    "That heavy stone in my chest finally loosened a little.",
    "I pulled the curtain halfway; the city was still awake.",
    "Some kindness arrives late, but late can still be enough.",
    "I’ll leave the story here and let the silence hold the rest.",
    "Night wind slid along the window like a soft knock.",
    "I nodded, mostly to confirm it for myself.",
    "The road wasn’t long, but it was long enough to clear my head.",
]

EXTRA_MX = [
    "我把 laptop 合上，决定先不看 unread messages。",
    "走廊里空调很响，像 background noise 一直在。",
    "说真的，这点 detail 平时我根本 ignore。",
    "我深呼吸一次，准备把心里话讲 clear。",
    "后来我想很久，还是觉得那个 decision 是对的。",
    "你要是当时在场，大概也会 pause 两秒。",
    "空气有点 humid，像要下雨又没下。",
    "我摸口袋，钥匙还在，心跳却有点 fast。",
    "事不大，可它卡在我胸口好几天。",
    "我试着笑，发现比预期更 hard。",
    "街对面 neon 闪了一下，像在催我往前。",
    "我把屏幕按灭，世界突然 quiet。",
    "有些话到嘴边，又被我 swallow 回去。",
    "再过一会儿 night 更沉，人也更诚实。",
    "我听见自己的呼吸，一下一下很 clear。",
    "那一刻我明白，我一直在等一个 answer。",
    "风从楼道灌进来，带着一点 dust。",
    "别急着 interrupt，让我把后半段说完。",
    "其实我也怕说错，可 silence 更难受。",
    "我把外套拉链拉上，像给自己加一层 shell。",
    "远处有人喊名字，不是我，我却 turn 了一下。",
    "记忆里的声音忽然 sharp，像重新按下 play。",
    "我在原地站几秒，才迈出 next step。",
    "这问题没有标准答案，可我还是想 try。",
    "茶凉了也没关系，凉的刚好能喝。",
    "我把椅子挪近，膝盖碰到桌沿。",
    "你听，外面的雨开始 denser 了。",
    "我把那句话在心里 repeat 了一遍。",
    "有时候 brave 不是冲出去，是留下来谈开。",
    "灯忽明忽暗，影子在墙上 shake。",
    "我攥紧拳头，又慢慢松开。",
    "这件事过后，我和世界的距离近了一点。",
    "我笑出声，自己也有点 shy。",
    "时间慢到我能 count 心跳。",
    "说到底，我只是想被认真听完。",
    "走廊尽头的门开一条缝，光 leak 进来。",
    "我摘下耳机，周围声音一下子回来。",
    "那一刻我没哭，只是鼻子有点酸。",
    "我把计划又改一版，写得更 concrete。",
    "你放心，这次我不会 halfway quit。",
    "我跨过水洼，鞋底溅一点泥。",
    "心里那块石头，终于 loosen 一点。",
    "我拉开窗帘一半，城市还 awake。",
    "有些温柔来得晚，但来了就够。",
    "故事讲到这里，剩下交给 silence。",
    "夜风贴着窗户走，像轻轻 knock。",
    "我点点头，像给自己一个 confirm。",
    "这段路不长，却够我想 clear 很多事。",
    "我把通知 mute 掉，先把眼前这杯喝完。",
    "到站的时候，我会把这句话再说一次。",
]

EXTRA_CHILD_CN = [
    "我把书包放好，拉链拉得紧紧的。",
    "窗外有小鸟叫，我听了一会儿。",
    "妈妈说慢慢讲，不要着急。",
    "我想把后面的事情也告诉你。",
    "今天的云像棉花糖，软软的。",
    "我数了三下，才敢往前走。",
    "鞋子有点湿，可我还是很开心。",
    "你等我一下，我把话说完整。",
    "月亮好像在对我眨眼。",
    "我把愿望藏在手心里。",
    "弟弟也笑了，我们一起拍手。",
    "草地上有小虫子，我蹲下来看。",
    "回家路上我会再想一遍。",
    "明天还想继续玩这个游戏。",
    "我把贴纸贴得整整齐齐。",
    "风吹过来，头发痒痒的。",
    "老师表扬我的时候，我脸红了。",
    "我抱紧玩偶，就不怕黑了。",
    "雨停以后，我们还能出去吗？",
    "我说完啦，你听懂了吗？",
]

EXTRA_CHILD_EN = [
    "I put my backpack down and zipped it tight.",
    "A little bird sang outside, and I listened.",
    "Mom said speak slowly—no rushing.",
    "I want to tell you the next part too.",
    "The clouds look like cotton candy today.",
    "I counted to three, then I stepped forward.",
    "My shoes got a bit wet, but I was still happy.",
    "Wait for me—I want to finish my sentence.",
    "The moon looks like it’s winking at me.",
    "I kept my wish inside my hands.",
    "My brother laughed and we clapped together.",
    "There was a tiny bug in the grass; I squatted to see.",
    "On the way home I’ll think about it again.",
    "Tomorrow I want to play this game more.",
    "I stuck the stickers on very neatly.",
    "The wind made my hair tickle.",
    "When teacher praised me, my face got warm.",
    "I hugged my toy and the dark felt okay.",
    "After the rain, can we go outside again?",
    "That’s all—did you understand me?",
]


def _lengthen(row: tuple, idx: int, bank: list[str], n_extra: int) -> tuple:
    lang, emotion, prosody, style, sfx, ono, sents = row
    extras = [bank[(idx * n_extra + k) % len(bank)] for k in range(n_extra)]
    return (lang, emotion, prosody, style, sfx, ono, list(sents) + extras)


def build_list(raws: list[tuple], audience: str, bank: list[str], n_extra: int) -> list[dict]:
    out = []
    for i, row in enumerate(raws):
        row = _lengthen(row, i, bank, n_extra=n_extra)
        lang, emotion, prosody, style, sfx, ono, sents = row
        out.append(pack(audience, i, lang, emotion, prosody, style, sfx, ono, sents))
    return out


def main() -> None:
    adult = (
        build_list(ADULT_CN, "adult", EXTRA_CN, 4)
        + build_list(ADULT_EN, "adult", EXTRA_EN, 4)
        + build_list(ADULT_MX, "adult", EXTRA_MX, 4)
    )
    for i, item in enumerate(adult):
        item["id"] = f"adult_{i:04d}"
    child = build_list(CHILD_CN, "child", EXTRA_CHILD_CN, 3) + build_list(
        CHILD_EN, "child", EXTRA_CHILD_EN, 3
    )
    for i, item in enumerate(child):
        item["id"] = f"child_{i:04d}"

    from collections import Counter

    assert len(ADULT_CN) == 67, len(ADULT_CN)
    assert len(ADULT_EN) == 67, len(ADULT_EN)
    assert len(ADULT_MX) == 66, len(ADULT_MX)
    assert len(CHILD_CN) == 25, len(CHILD_CN)
    assert len(CHILD_EN) == 25, len(CHILD_EN)

    speeches = [x["est_speech_sec"] for x in adult + child]
    doc = {
        "seed": None,
        "target_sec": [20, 30],
        "n_adult": len(adult),
        "n_child": len(child),
        "mode_intent": "single",
        "pause_strategy": (
            "no <|prosody:long_pause|> in TTS text; "
            f"postprocess VAD speech spans + insert random silence "
            f"in [{PAUSE_SEC_MIN},{PAUSE_SEC_MAX}]s per gap"
        ),
        "lang_mix": {
            "adult": dict(Counter(x["lang"] for x in adult)),
            "child": dict(Counter(x["lang"] for x in child)),
        },
        "est_speech_sec": {
            "mean": round(sum(speeches) / len(speeches), 2),
            "min": round(min(speeches), 2),
            "max": round(max(speeches), 2),
        },
        "note": (
            "Hand-authored spoken scripts for Higgs v3, lengthened for ~20–30s speech. "
            "Delivery tags at start; sentences joined without long_pause tags; "
            "clone pipeline VAD-splices speech and inserts fixed silence. "
            "Adult mixed = 外企-style code-switching. Child no mixed."
        ),
        "adult": adult,
        "child": child,
    }
    OUT.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT} adult={len(adult)} child={len(child)}")
    print("adult langs", doc["lang_mix"]["adult"])
    print("est_speech", doc["est_speech_sec"])
    print("sample:", adult[0]["text"][:220])
    print("sample est_speech", adult[0]["est_speech_sec"], "n_sent", adult[0]["num_sentences"])


if __name__ == "__main__":
    main()
