#!/usr/bin/env python3
# Copyright (c) 2025 Boson AI
"""Build coherent multi-sentence clone corpus with designed per-gap pauses.

Each script is one continuous monologue (setup → develop → turn → close).
pause_secs[i] sits between sentence i and i+1:
  - every pause >= 1.0s
  - mean(pause_secs) >= 2.0s
  - mix of short / mid / long for natural rhythm
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUT = HERE / "clone_text_corpus.json"
TAG_RE = re.compile(r"<\|(?:emotion|style|sfx|prosody):[a-z_]+\|>")
CN_CHARS_PER_SEC = 7.0
EN_WORDS_PER_SEC = 3.0


def design_pause_secs(n_gaps: int, rng: random.Random) -> list[float]:
    """Design varied pauses: min 1.0s, mean >= 2.0s."""
    if n_gaps <= 0:
        return []
    pauses: list[float] = []
    # Force at least one clearly long pause for drama.
    long_slot = rng.randrange(n_gaps)
    for i in range(n_gaps):
        if i == long_slot:
            pauses.append(round(rng.uniform(3.0, 4.5), 2))
            continue
        r = rng.random()
        if r < 0.28:
            pauses.append(round(rng.uniform(1.0, 1.45), 2))
        elif r < 0.62:
            pauses.append(round(rng.uniform(1.6, 2.3), 2))
        else:
            pauses.append(round(rng.uniform(2.4, 3.4), 2))
    # Lift mean to >= 2.0 without shrinking any pause below 1.0.
    while sum(pauses) / len(pauses) < 2.0 - 1e-9:
        j = min(range(len(pauses)), key=lambda k: pauses[k])
        pauses[j] = round(min(4.5, pauses[j] + 0.25), 2)
    return [max(1.0, float(p)) for p in pauses]


def _hdr(emotion: str, prosody: str, style: str | None) -> str:
    parts = [f"<|emotion:{emotion}|>"]
    if style:
        parts.append(f"<|style:{style}|>")
    if prosody:
        parts.append(f"<|prosody:{prosody}|>")
    return "".join(parts)


def _strip_leading_ono(sent: str, ono: str | None) -> str:
    if not ono:
        return sent
    s = sent or ""
    if s.lower().startswith(ono.lower()):
        return s[len(ono) :].lstrip(" ，,.—-")
    return s


def _insert_sfx_flexible(sent: str, sfx: str, ono: str, rng: random.Random) -> str:
    """Insert <|sfx:…|> + onomatopoeia at a varied in-sentence position (not always head)."""
    tag = f"<|sfx:{sfx}|>{ono}"
    s = (sent or "").strip()
    if not s:
        return tag
    # Only cut after clause punctuation — never mid-glyph / mid-word.
    cuts: list[int] = []
    for i, ch in enumerate(s):
        if ch in "，,、；;—" and 2 <= i <= len(s) - 3:
            cuts.append(i + 1)

    modes = ["prefix", "mid", "before_end"]
    weights = [0.25, 0.45, 0.30] if cuts else [0.45, 0.0, 0.55]
    mode = rng.choices(modes, weights=weights, k=1)[0]
    if mode == "mid" and cuts:
        pos = rng.choice(cuts)
        return s[:pos] + tag + s[pos:]
    if mode == "before_end" and len(s) > 6:
        cut = len(s)
        while cut > 0 and s[cut - 1] in "。.！!？?…\"'”’":
            cut -= 1
        if cut >= 3:
            return s[:cut] + tag + s[cut:]
    return tag + s


def _est_speech(lang: str, clean_sents: list[str]) -> float:
    text = "".join(clean_sents)
    if lang == "pure_en":
        return max(8.0, len(text.split()) / EN_WORDS_PER_SEC)
    if lang == "mixed":
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
    rng: random.Random,
) -> dict:
    # Single-shot TTS: emotion/prosody/style once at the start.
    # SFX is placed flexibly inside a randomly chosen sentence (often mid-clause),
    # not fixed at the script head.
    # Chance to add an ambient SFX even when the template had none.
    if (not sfx or not ono) and rng.random() < 0.35:
        if lang == "pure_en":
            sfx, ono = rng.choice(
                [("sigh", "Ah"), ("laughter", "Haha"), ("cough", "cough")]
            )
        else:
            sfx, ono = rng.choice(
                [("sigh", "唉"), ("laughter", "哈哈"), ("cough", "咳"), ("sniff", "嗯")]
            )
    clean_sents = [_strip_leading_ono(s, ono) for s in sents]
    hdr = _hdr(emotion, prosody, style)
    tagged_sents = list(clean_sents)
    sfx_sentence_idx: int | None = None
    sfx_mode = None
    if sfx and ono and tagged_sents:
        sfx_sentence_idx = rng.randrange(len(tagged_sents))
        before = tagged_sents[sfx_sentence_idx]
        after = _insert_sfx_flexible(before, sfx, ono, rng)
        tagged_sents[sfx_sentence_idx] = after
        if after.startswith(f"<|sfx:{sfx}|>"):
            sfx_mode = "sent_prefix"
        elif after.endswith(ono) or f"{ono}。" in after or f"{ono}." in after:
            sfx_mode = "before_end"
        else:
            sfx_mode = "mid"
    tagged_sents[0] = hdr + tagged_sents[0]
    pause_secs = design_pause_secs(max(0, len(sents) - 1), rng)
    speech = round(_est_speech(lang, clean_sents), 2)
    pause_sum = round(sum(pause_secs), 2)
    return {
        "id": f"{audience}_{idx:04d}",
        "audience": audience,
        "lang": lang,
        "emotion": emotion,
        "prosody": prosody,
        "style": style,
        "sfx": sfx,
        "sfx_sentence_idx": sfx_sentence_idx,
        "sfx_placement": sfx_mode,
        "text": "".join(tagged_sents) if lang != "pure_en" else " ".join(tagged_sents),
        "clean_text": " ".join(clean_sents),
        "sentences": tagged_sents,
        "clean_sentences": clean_sents,
        "num_sentences": len(sents),
        "pause_secs": pause_secs,
        "pause_sec_min": min(pause_secs) if pause_secs else 0.0,
        "pause_sec_max": max(pause_secs) if pause_secs else 0.0,
        "pause_sec_mean": round(sum(pause_secs) / len(pause_secs), 3) if pause_secs else 0.0,
        "pause_tag": "designed_per_gap",
        "pause_postprocess": "vad_splice",
        "est_speech_sec": speech,
        "est_total_sec": round(speech + pause_sum, 2),
    }


# ---------------------------------------------------------------------------
# Coherent monologues: each entry is one continuous arc (not loosely related lines).
# ---------------------------------------------------------------------------

ADULT_CN: list[tuple] = [
    ("pure_cn", "contentment", "", None, "sigh", "唉", [
        "唉，周末下午其实没什么安排，阳光刚好铺在地板上。",
        "我泡了杯茶，把窗开了一条缝，书翻了两页又放下。",
        "外面有人在练琴，断断续续的，却不让人烦。",
        "我忽然觉得，偶尔什么都不赶，日子反而完整了一点。",
        "茶杯慢慢凉了，我还是坐着，没有去开灯。",
        "就让这个下午停在这里吧，不用证明给谁看。",
    ]),
    ("pure_cn", "longing", "pitch_low", None, None, None, [
        "又路过那家旧书店，门还是半掩着，灰尘味混着纸张味。",
        "好几年前我们常在角落吵架，又很快和好，像排练过一样。",
        "我站在门口看了会儿，最终还是没有进去。",
        "有些地方适合想念，不适合重返，进去就会把记忆拆开。",
        "街灯亮了，我把帽檐压低，继续往车站走。",
        "车票还在钱包夹层里，我到现在也没扔掉。",
    ]),
    ("pure_cn", "pride", "expressive_high", None, None, None, [
        "这份报告我熬了三个通宵改出来，每一页都标过版本。",
        "客户当场点头，说这版终于抓住重点了，会议室里安静了一秒。",
        "我没表现得很夸张，心里却像放了一串烟花。",
        "走出门，我给自己买了杯最贵的咖啡，甜得有点过分。",
        "电梯下行的时候，我又把结论默背了一遍，确认没有漏。",
        "今晚可以好好睡了，至少这一仗，我打赢了。",
    ]),
    ("pure_cn", "amusement", "expressive_high", None, "laughter", "哈哈", [
        "哈哈，我家猫刚才把遥控器按成了静音，全家找了一圈。",
        "结果它趴在沙发上装无辜，眼睛都不抬一下。",
        "我指着电视跟它对视三秒，它居然打了个哈欠。",
        "算了，今晚就当给自己放个无声电影吧。",
        "后来我把遥控器藏高了，它又去啃我的充电线。",
        "真是服了，这个家里谁说了算，一目了然。",
    ]),
    ("pure_cn", "contemplation", "", None, None, None, [
        "地铁里忽然想起一件很小的事，很多年前有人问我想变成什么样的人。",
        "我当时随口答了个答案，现在却发现答错了重点。",
        "重要的也许不是变成谁，而是别丢掉好奇心。",
        "车门开了，我跟着人群走出去，心里反而安静。",
        "站台上风有点冷，我把围巾裹紧，继续往出口走。",
        "今晚回家，我要把这件事写进备忘录，免得又忘。",
    ]),
    ("pure_cn", "sadness", "", None, "sigh", "唉", [
        "唉，今天整理旧手机，翻到一段你离开前一天的语音。",
        "声音还那么轻，像怕吵醒谁似的，我听了两遍就不敢再听。",
        "我把它存进一个很深的文件夹，像藏起一块还没愈合的伤口。",
        "屏幕暗下去，房间里只剩冰箱的低鸣。",
        "我倒了杯水，却一直端着，没有喝。",
        "有些告别不是门关上的那一刻，是你再也打不开那段语音。",
    ]),
    ("pure_cn", "enthusiasm", "speed_fast", None, None, None, [
        "你知道吗，今晚那家新开的面馆居然不用排队！",
        "汤底浓得像熬了一整天，辣椒油香得我头皮发麻。",
        "老板还额外送了卤蛋，说看我吃得太开心。",
        "我现在就想再去第二碗，胃已经在抗议了。",
        "回去路上我还在回味，差点走错路口。",
        "明天还去，你要不要一起，我请客前两碗。",
    ]),
    ("pure_cn", "anger", "expressive_high", None, None, None, [
        "快递又被扔在门口淋雨，里面是贵重物品啊！",
        "我打了三通电话，客服只会道歉，一问具体就转移话题。",
        "下次再这样，我直接拒收并投诉到平台。",
        "把别人的东西当垃圾，太过分了。",
        "我把外箱拍照存证，手都在抖。",
        "今晚先不睡觉，也要把这件事跟到底。",
    ]),
    ("pure_cn", "fear", "", None, None, None, [
        "楼道灯又坏了，我摸着墙往上走，每一步都听得见自己的呼吸。",
        "身后好像有脚步声，我停，它也停，我不敢回头。",
        "钥匙在口袋里翻了半天，才对准锁孔。",
        "门一开，屋里灯还亮着，我才觉得膝盖发软。",
        "我把门反锁两道，还是去看了看阳台。",
        "明天一定要报修，这种黑，我再也不想走第二次。",
    ]),
    ("pure_cn", "affection", "pitch_low", None, None, None, [
        "你睡着的时候眉心还皱着，我伸手轻轻揉开。",
        "窗外雨声很细，像有人在远处讲悄悄话。",
        "我把毯子拉高一点，你嘟囔了一句，又沉下去。",
        "这样的夜晚不需要计划，只需要你在旁边。",
        "手机亮了一下，我没点开，不想打断这份安静。",
        "明天的事明天再说，今晚我就守着这点温度。",
    ]),
    ("pure_cn", "bitterness", "", None, "sigh", "唉", [
        "唉，一起创业的人，现在连消息都不回了。",
        "不是怪谁，就是空，努力的痕迹还在，关系淡成了客套。",
        "合同、聊天记录、那张合影，都还在云盘里躺着。",
        "成长有时是学会独自把故事讲完。",
        "我删掉了置顶，却没有删对话，挺可笑的。",
        "算了，先把灯关掉，明天还要上班。",
    ]),
    ("pure_cn", "relief", "", None, "sigh", "呼", [
        "呼，体检报告终于出来了，各项指标都正常。",
        "这两周我每天刷手机，像等判决书。",
        "护士叫到名字时，我手心全是汗。",
        "走出医院大门，风居然有点香，我这才发现自己一直绷着。",
        "给家里打了电话，声音都发飘。",
        "今晚想吃顿好的，然后早睡，把担心从身体里清出去。",
    ]),
    ("pure_cn", "surprise", "expressive_high", None, None, None, [
        "抽屉里怎么会有这张旧照片？毕业旅行，你的马尾，那个傻笑。",
        "我以为它早丢了，原来一直夹在书页里。",
        "时间真会藏陷阱，翻到的瞬间胸口一紧。",
        "我对着照片看了很久，才想起那天海边的风有多大。",
        "后来我把它重新夹好，不敢再随手乱翻。",
        "有些东西不是你想忘就能忘，它会自己回来找你。",
    ]),
    ("pure_cn", "determination", "expressive_high", None, None, None, [
        "这次比赛我准备了四个月，动作都拆成节拍练。",
        "上场前手是冰的，但我告诉自己，怕也要完整做完。",
        "音乐起的时候，世界忽然变窄，只剩呼吸和脚步。",
        "结束鞠躬，我才听见观众席的声音涌回来。",
        "分数怎么样先不管，至少我没有逃。",
        "回更衣室我把护腕解开，决定下周从弱项重新加练。",
    ]),
    ("pure_cn", "shame", "", None, None, None, [
        "客户的名字我叫错了两次，对方停顿一下，礼貌地纠正我。",
        "我道歉的声音薄得不像自己，耳根一路热到脖子。",
        "会后我在卫生间洗了很久的手，还是觉得尴尬贴在脸上。",
        "回去把名单又背了三遍，写在手心又擦掉。",
        "有些错不大，但会反复在夜里回放。",
        "明天再见他，我要先叫对，再谈别的。",
    ]),
    ("pure_cn", "awe", "", None, None, None, [
        "博物馆那尊像，眼神像活的，我站了很久不敢出声。",
        "耳麦讲解忽然变得多余，美有时会把时间按暂停。",
        "旁边有个孩子问妈妈，石头怎么会有表情。",
        "我听着，忽然也想问同样的问题。",
        "离开展厅时，我回头又看了一眼，灯光正好落在眉骨上。",
        "回家路上我一直在想，人为什么需要被什么东西震住。",
    ]),
    ("pure_cn", "confusion", "", None, None, None, [
        "他说的话我听懂了每个字，连起来却不知道他想怎样。",
        "是拒绝，还是在留余地，我反复听那条语音。",
        "屏幕上的输入框亮了又灭，我删掉重写了四遍。",
        "最后还是只回了一个好，像把问题踢回去。",
        "夜里两点我还在想，是不是我多心了。",
        "算了，明天当面问清楚，猜来猜去最耗人。",
    ]),
    ("pure_cn", "elation", "expressive_high", None, "laughter", "哈哈", [
        "哈哈，演唱会票抢到了，内场第三排，我盯着页面手都在抖。",
        "马上跟你说，周末别安排别的，我们要一起疯一整个晚上。",
        "街对面的灯牌闪了闪，像在催我往前走。",
        "我把手机屏幕按灭，忽然觉得世界安静了一秒，又喧闹起来。",
        "歌单已经在脑子里循环了，连睡觉都带着鼓点。",
        "到时候见，迟到的人请喝奶茶，说定了。",
    ]),
    ("pure_cn", "helplessness", "", None, "sigh", "唉", [
        "唉，系统又崩了，客户在群里刷屏，我重启了三次还是一样。",
        "文档写着标准流程，现实却处处对不上。",
        "我盯着进度条转圈，像盯着一个不肯开门的人。",
        "领导问原因，我只能说还在查，声音发干。",
        "有些问题不是不努力，是你手里的工具根本不够。",
        "先把现状同步出去吧，至少别让大家空等。",
    ]),
    ("pure_cn", "disgust", "expressive_high", None, None, None, [
        "冰箱底层那盒东西已经鼓包了，一打开味道直冲上来。",
        "我戴着手套清理，还是觉得恶心贴在舌头上。",
        "谁放的，什么时候放的，标签早糊成一团。",
        "清完我把窗户开到最大，喷了两遍消毒水。",
        "以后过期的东西当天扔，别再心软留着。",
        "现在闻见酸奶味我都要躲一下，太可怕了。",
    ]),
]

# Expand CN with more coherent arcs built from theme chains
_CN_MORE = [
    ("pure_cn", "arousal", "speed_fast", None, None, None, [
        "闹钟没响，我是被心跳吵醒的，一看表还来得及。",
        "牙膏都没挤匀就冲出门，电梯刚好到。",
        "地铁门要关的瞬间我侧身挤进去，背后一片抱怨。",
        "到公司第一件事是打开邮箱，关键邮件居然还没来。",
        "我把外套挂好，手还在抖，才发现自己一直屏着气。",
        "今天从这一秒开始，把节奏抢回来。",
    ]),
    ("pure_cn", "contemplation", "pitch_low", None, None, None, [
        "雨停之后路面反光，像一面碎掉又拼起来的镜子。",
        "我走得很慢，想把白天没说完的话在脑子里排好队。",
        "有人从我身边跑过，留下一声短促的笑。",
        "我停在便利店门口，买了瓶水，却站着喝完才走。",
        "回家的路其实不长，只是今晚显得格外远。",
        "推开门之前我深吸一口气，把外面的潮湿留在门外。",
    ]),
]


def _dup_variants(base: list[tuple], n: int, seed: int) -> list[tuple]:
    """Create coherent lengthened variants by appending closing beats (same arc)."""
    rng = random.Random(seed)
    closers_cn = [
        "我说完这些，才发觉声音有点哑。",
        "夜色压下来，我把话题轻轻收住。",
        "先走到这里吧，剩下的明天再想。",
        "空气里还留着刚才的情绪，像没散尽的雾。",
    ]
    closers_en = [
        "I let the last sentence hang there for a second.",
        "That is where I stop for tonight.",
        "I tuck the thought away and keep walking.",
        "The rest can wait until morning.",
    ]
    closers_mx = [
        "我说到这里先 pause，剩下的明天再讲。",
        "Anyway，今晚就到这，别把情绪拖太长。",
        "我把这句话收住，像 save 了一个草稿。",
        "先停在这里，别让 story 失控。",
    ]
    out = list(base)
    i = 0
    while len(out) < n:
        lang, emotion, prosody, style, sfx, ono, sents = base[i % len(base)]
        sents = list(sents)
        if lang == "pure_en":
            extra = closers_en[i % len(closers_en)]
        elif lang == "mixed":
            extra = closers_mx[i % len(closers_mx)]
        else:
            extra = closers_cn[i % len(closers_cn)]
        # slight shuffle of middle sentences to reduce exact dupes while keeping arc
        mid = sents[1:-1]
        if len(mid) >= 2 and rng.random() < 0.5:
            a, b = rng.sample(range(len(mid)), 2)
            mid[a], mid[b] = mid[b], mid[a]
        new_sents = [sents[0]] + mid + [sents[-1], extra]
        out.append((lang, emotion, prosody, style, sfx, ono, new_sents))
        i += 1
    return out[:n]


ADULT_EN: list[tuple] = [
    ("pure_en", "contentment", "", None, "sigh", "Ah", [
        "Ah, the afternoon had no agenda, just sunlight on the floorboards.",
        "I made tea, cracked the window, and let a half-read book rest on my lap.",
        "Someone practiced piano outside, uneven notes that somehow fit the quiet.",
        "I realized how rare it is to stop without explaining yourself.",
        "The cup cooled in my hands and I still did not turn on the light.",
        "I decided to keep the hour exactly as it was, unfinished and enough.",
    ]),
    ("pure_en", "longing", "pitch_low", None, None, None, [
        "I walked past our old cafe without meaning to.",
        "Same chipped table by the window, same noisy espresso machine.",
        "For a second I almost went in, then I pictured us arguing over nothing.",
        "Some doors are better left as memory, not as a revisit.",
        "I pulled my collar up and kept walking toward the station.",
        "The ticket stub is still in my wallet; I still have not thrown it away.",
    ]),
    ("pure_en", "pride", "expressive_high", None, None, None, [
        "I rewrote that report through three sleepless nights, version by version.",
        "In the meeting the client nodded and said this draft finally hit the point.",
        "I stayed calm on the outside while something bright went off in my chest.",
        "Afterward I bought myself the most expensive coffee on the block.",
        "In the elevator I rehearsed the conclusion once more, just to be sure.",
        "Tonight I can sleep; at least this round, I won.",
    ]),
    ("pure_en", "amusement", "expressive_high", None, "laughter", "Ha", [
        "Ha, the cat muted the TV and the whole house went hunting for the remote.",
        "She was on the sofa pretending innocence, not even blinking.",
        "I stared at her for three seconds; she yawned in my face.",
        "Fine, silent movie night it is.",
        "I hid the remote higher, and she started chewing the charger instead.",
        "In this apartment, the hierarchy is painfully clear.",
    ]),
    ("pure_en", "sadness", "", None, "sigh", "Ah", [
        "Ah, I found an old voice note while cleaning my phone.",
        "You sounded tired but kind, like you always did the day before you left.",
        "I played it twice and could not press play a third time.",
        "I buried it in a deep folder and pretended that counted as moving on.",
        "The room went quiet except for the fridge humming.",
        "Some goodbyes are not the closed door; they are the file you never open again.",
    ]),
    ("pure_en", "anger", "expressive_high", None, None, None, [
        "The package was left in the rain again, and it was fragile on purpose.",
        "I called three times; every agent apologized and changed the subject.",
        "Next time I will refuse delivery and escalate it all the way up.",
        "Treating someone else's things like trash is not a small mistake.",
        "I photographed the wet box with shaking hands for evidence.",
        "I am not sleeping until this is written down and filed.",
    ]),
    ("pure_en", "fear", "", None, None, None, [
        "The hallway light was out again, so I climbed by touch and breath.",
        "Footsteps seemed to mirror mine; when I stopped, they stopped.",
        "My key fought the lock for what felt like a minute.",
        "Inside, the lamp was still on, and my knees finally loosened.",
        "I locked both bolts and still checked the balcony.",
        "Tomorrow I am calling maintenance; I will not walk that dark twice.",
    ]),
    ("pure_en", "affection", "pitch_low", None, None, None, [
        "You frowned in your sleep, so I smoothed the line between your brows.",
        "Rain tapped the window like a soft conversation we did not need to join.",
        "I pulled the blanket higher; you mumbled and sank again.",
        "Nights like this do not need plans, only the fact of you nearby.",
        "My phone lit up once; I ignored it to keep the quiet intact.",
        "Tomorrow can wait; tonight I am guarding this small warmth.",
    ]),
    ("pure_en", "determination", "expressive_high", None, None, None, [
        "I trained for four months and broke every move into counts.",
        "Backstage my hands were ice, but I told myself fear could still finish the routine.",
        "When the music started, the room narrowed to breath and footwork.",
        "After the bow, the audience sound rushed back in all at once.",
        "Score aside, I did not run from the moment.",
        "In the locker room I untied the wraps and scheduled extra drills on the weak parts.",
    ]),
    ("pure_en", "surprise", "expressive_high", None, None, None, [
        "Why was this old photo in the box? Graduation trip, your ponytail, that dumb grin.",
        "I thought it was gone, tucked in a book the whole time.",
        "Time loves little traps; my chest tightened before I could laugh.",
        "I stared until I remembered how hard the sea wind hit that day.",
        "I slid it back carefully, afraid of another accidental find.",
        "Some things do not leave when you ask; they return on their own schedule.",
    ]),
    ("pure_en", "relief", "", None, "sigh", "Phew", [
        "Phew, the lab results finally came back normal across the board.",
        "For two weeks I refreshed my phone like it was a verdict.",
        "When they called my name, my palms were soaked.",
        "Outside the hospital the air smelled oddly sweet, and I noticed I had been clenched.",
        "I called home with a voice that would not stay steady.",
        "Tonight I want a real meal and early sleep, to flush the worry out.",
    ]),
    ("pure_en", "bitterness", "", None, "sigh", "Ah", [
        "Ah, the people I built with do not even reply anymore.",
        "I am not blaming anyone; it is just hollow where the effort used to sit.",
        "Contracts and chat logs still live in the cloud like fossils.",
        "Growing up sometimes means finishing the story alone.",
        "I unmuted the thread, then muted it again, which felt ridiculous.",
        "Lights off; work still starts in the morning.",
    ]),
    ("pure_en", "awe", "", None, None, None, [
        "That museum statue's eyes felt alive; I stood too long and went quiet.",
        "The audio guide suddenly felt extra, like beauty had paused the clock.",
        "A kid asked why stone could have an expression, and I wanted the same answer.",
        "Leaving the hall, I looked back once; light caught the brow just right.",
        "On the way home I kept wondering why people need to be shaken awake by art.",
        "I wrote one line in my notes and then put the phone away.",
    ]),
    ("pure_en", "confusion", "", None, None, None, [
        "I understood every word he said, but not what he wanted from me.",
        "Was it a no, or a soft maybe? I replayed the voice note twice.",
        "The text box blinked while I deleted and rewrote four drafts.",
        "I finally sent okay, which kicked the question back to him.",
        "At two in the morning I still wondered if I was overreading.",
        "Tomorrow I will ask face to face; guessing is the most expensive habit.",
    ]),
    ("pure_en", "elation", "expressive_high", None, "laughter", "Yes", [
        "Yes! I got the concert tickets, third row, hands shaking on the checkout page.",
        "Clear the weekend; we are staying loud until the lights come up.",
        "The sign across the street flickered like it was pushing me forward.",
        "I locked the phone and the world went quiet for one second, then roared back.",
        "The setlist is already looping in my head, even in sleep.",
        "See you there; whoever is late buys drinks, deal sealed.",
    ]),
    ("pure_en", "shame", "", None, None, None, [
        "I mispronounced the client's name twice; they paused, then corrected me politely.",
        "My apology came out thin, heat climbing from my ears to my neck.",
        "I washed my hands too long afterward, as if soap could erase the moment.",
        "Back at the desk I relearned the roster three times.",
        "Some mistakes are small and still replay at night on loop.",
        "Next time I see him, I will say it right before anything else.",
    ]),
    ("pure_en", "helplessness", "", None, "sigh", "Ah", [
        "Ah, the system crashed again while the client group kept refreshing.",
        "I rebooted three times; the progress wheel just kept spinning.",
        "The runbook said one thing; reality refused to match any step.",
        "When leadership asked why, all I had was still checking, throat dry.",
        "Sometimes it is not effort; it is tools that cannot reach the problem.",
        "I will post the status now so nobody waits in the dark.",
    ]),
    ("pure_en", "disgust", "expressive_high", None, None, None, [
        "The container at the back of the fridge had ballooned; the smell hit instantly.",
        "I cleaned it with gloves on and still felt it on my tongue.",
        "Whoever left it there left no label worth reading.",
        "I opened every window and sprayed disinfectant twice.",
        "From now on, expired food leaves the same day; no soft heart.",
        "Even yogurt perfume would make me step back after that.",
    ]),
    ("pure_en", "arousal", "speed_fast", None, None, None, [
        "The alarm never rang; my heartbeat woke me, and the clock said I could still make it.",
        "Toothpaste half-spread, I ran for the elevator that somehow waited.",
        "I slid through the train doors as they closed, catching a chorus of complaints.",
        "At the office the urgent email still had not arrived, which somehow made it worse.",
        "I hung my coat and noticed I had been holding my breath the whole way.",
        "From this second, I take the day back.",
    ]),
    ("pure_en", "contemplation", "pitch_low", None, None, None, [
        "After the rain the street reflected like a mirror that had been broken and reset.",
        "I walked slowly, lining up the sentences I failed to say earlier.",
        "Someone jogged past with a short laugh that vanished into traffic.",
        "I bought water at the corner store and finished it standing still.",
        "The route home is short; tonight it felt longer on purpose.",
        "Before opening the door I breathed once and left the damp outside.",
    ]),
]

ADULT_MX: list[tuple] = [
    ("mixed", "contentment", "", None, "sigh", "唉", [
        "唉，周末下午没什么 plan，阳光刚好铺在地板上。",
        "我泡了茶，开了一条窗缝，书翻两页又放下。",
        "外面有人练琴，断断续续，居然很适合发呆。",
        "偶尔什么都不赶，日子反而 complete 一点。",
        "茶杯凉了我也没开灯，就让这个 afternoon 停着。",
        "不用证明给谁看，这种安静本身就够。",
    ]),
    ("mixed", "longing", "pitch_low", None, None, None, [
        "又路过那家旧 cafe，门半掩着，灰尘味混着纸味。",
        "好几年前我们在角落吵架又和好，像排练过。",
        "我站在门口看了会儿，最终还是没有进去。",
        "有些地方适合 miss，不适合 revisit。",
        "街灯亮了，我把帽檐压低，继续往车站走。",
        "那张 ticket stub 还在钱包里，我一直没扔。",
    ]),
    ("mixed", "pride", "expressive_high", None, None, None, [
        "这份 report 我熬了三个通宵，每一页都标过 version。",
        "客户当场点头，说这版终于抓住 key point。",
        "我表面很稳，心里却像放了 fireworks。",
        "出门我买了杯最贵的咖啡，甜得有点过分。",
        "电梯下行时我又默背了一遍 conclusion。",
        "今晚可以睡了，至少这一仗我赢了。",
    ]),
    ("mixed", "amusement", "expressive_high", None, "laughter", "哈哈", [
        "哈哈，猫把遥控器按成 mute，全家找了一圈。",
        "它趴在沙发上装无辜，眼睛都不抬。",
        "我对视三秒，它居然打了个哈欠。",
        "算了，今晚就 silent movie 吧。",
        "我把遥控藏高，它又去啃 charger。",
        "这个家谁说了算，一目了然。",
    ]),
    ("mixed", "sadness", "", None, "sigh", "唉", [
        "唉，整理手机翻到你离开前一天的 voice note。",
        "声音很轻，我听两遍就不敢再听。",
        "我把它丢进很深的 folder，像藏伤口。",
        "屏幕暗下去，只剩冰箱在响。",
        "我倒了水却一直端着，没有喝。",
        "有些 goodbye 不是关门，是你再也打不开那段音频。",
    ]),
    ("mixed", "anger", "expressive_high", None, None, None, [
        "快递又被扔在雨里，里面是 fragile 物品啊。",
        "我打了三通电话，客服只会 apologize。",
        "下次再这样，我拒收并 escalate 到平台。",
        "把别人的东西当垃圾，太过分了。",
        "我拍照取证，手都在抖。",
        "今晚不睡也要把这事跟到底。",
    ]),
    ("mixed", "fear", "", None, None, None, [
        "楼道灯又坏了，我摸墙往上走，呼吸声特别响。",
        "身后像有脚步，我停它也停，我不敢回头。",
        "钥匙对了半天锁孔才打开。",
        "屋里灯还亮着，我膝盖才软下来。",
        "我反锁两道，还是去看了阳台。",
        "明天一定报修，这种 dark 我不要第二次。",
    ]),
    ("mixed", "affection", "pitch_low", None, None, None, [
        "你睡着时眉心皱着，我伸手轻轻揉开。",
        "窗外雨声很细，像远处的 whisper。",
        "我把毯子拉高，你嘟囔一句又沉下去。",
        "这种夜晚不需要 plan，只需要你在旁边。",
        "手机亮了一下，我没点开，不想打断 quiet。",
        "明天的事明天说，今晚我守着这点温度。",
    ]),
    ("mixed", "bitterness", "", None, "sigh", "唉", [
        "唉，一起创业的人现在都不回 message 了。",
        "不是怪谁，就是 hollow，努力痕迹还在，关系淡成 polite。",
        "合同和聊天记录都还在云盘里躺着。",
        "成长有时是学会独自把 story 讲完。",
        "我取消置顶，却没删对话，挺可笑。",
        "关灯吧，明天还要上班。",
    ]),
    ("mixed", "relief", "", None, "sigh", "呼", [
        "呼，体检报告出来了，指标都 normal。",
        "这两周我天天刷手机，像等 verdict。",
        "叫到名字时手心全是汗。",
        "走出医院，风居然有点香，我才发现自己一直绷着。",
        "给家里打电话，声音都发飘。",
        "今晚吃顿好的，早睡，把担心清出去。",
    ]),
    ("mixed", "surprise", "expressive_high", None, None, None, [
        "抽屉里怎么有这张旧 photo？毕业旅行，你的马尾，那个傻笑。",
        "我以为丢了，原来一直夹在书里。",
        "时间真会藏 trap，胸口一下子紧了。",
        "我看了很久，才想起那天海边风有多大。",
        "后来我重新夹好，不敢再随手翻。",
        "有些东西不是你想忘就能忘，它会自己回来。",
    ]),
    ("mixed", "determination", "expressive_high", None, None, None, [
        "这次比赛我准备了四个月，动作拆成 beat 练。",
        "上场前手是冰的，但我告诉自己怕也要做完。",
        "音乐起时世界变窄，只剩呼吸和脚步。",
        "鞠躬后我才听见观众席的声音涌回来。",
        "分数先不管，至少我没有 run away。",
        "回更衣室我解开护腕，决定从弱项加练。",
    ]),
    ("mixed", "shame", "", None, None, None, [
        "客户名字我叫错两次，对方 pause 一下礼貌纠正。",
        "我道歉到声音 thin，耳根热到脖子。",
        "会后在卫生间洗手很久，尴尬还贴在脸上。",
        "回去把名单又背三遍，写在手心又擦掉。",
        "有些错不大，却会在夜里 loop。",
        "明天再见他，我要先叫对再谈别的。",
    ]),
    ("mixed", "awe", "", None, None, None, [
        "博物馆那尊像眼神像 alive，我站很久不敢出声。",
        "耳麦讲解忽然变得 extra，美有时会 pause 时间。",
        "有个孩子问石头怎么会有表情，我也想问。",
        "离开展厅我回头一眼，灯光正好落在眉骨。",
        "回家路上一直在想，人为什么需要被震住。",
        "我在备忘录写了一行，然后把手机收起。",
    ]),
    ("mixed", "confusion", "", None, None, None, [
        "他的话每个字都懂，连起来却不知道他想怎样。",
        "是拒绝还是留余地，我反复听那条 voice note。",
        "输入框亮了又灭，我删掉重写四遍。",
        "最后只回了一个好，像把问题踢回去。",
        "夜里两点还在想是不是我 overthink 了。",
        "明天当面问清楚，猜来猜去最耗人。",
    ]),
    ("mixed", "elation", "expressive_high", None, "laughter", "哈哈", [
        "哈哈，演唱会票抢到了，内场第三排，手都在抖。",
        "周末别安排，我们要一起疯一整个晚上。",
        "街对面灯牌闪了闪，像在催我往前走。",
        "我把手机按灭，世界安静一秒又喧闹起来。",
        "歌单已经在脑子里 loop，睡觉都带着鼓点。",
        "到时候见，迟到的请喝奶茶，说定了。",
    ]),
    ("mixed", "helplessness", "", None, "sigh", "唉", [
        "唉，系统又崩了，客户在群里刷屏，我重启三次还一样。",
        "文档写着标准流程，现实处处对不上。",
        "我盯着进度条转圈，像盯着不肯开门的人。",
        "领导问原因，我只能说还在查，声音发干。",
        "有些问题不是不努力，是工具不够。",
        "先把 status 同步出去，别让大家空等。",
    ]),
    ("mixed", "disgust", "expressive_high", None, None, None, [
        "冰箱底层那盒东西鼓包了，一打开味道直冲上来。",
        "我戴手套清理，还是觉得恶心贴在舌头上。",
        "标签早糊了，谁放的完全 unknown。",
        "清完开窗喷消毒水两遍。",
        "以后过期当天扔，别再心软。",
        "现在闻见酸奶味我都要躲一下。",
    ]),
    ("mixed", "arousal", "speed_fast", None, None, None, [
        "闹钟没响，我是被心跳吵醒的，一看表还来得及。",
        "牙膏没挤匀就冲出门，电梯刚好到。",
        "地铁门要关我侧身挤进去，背后一片抱怨。",
        "到公司打开邮箱，urgent 邮件居然还没来。",
        "挂好外套才发现自己一直屏着气。",
        "今天从这一秒开始，把节奏抢回来。",
    ]),
    ("mixed", "contemplation", "pitch_low", None, None, None, [
        "雨停后路面反光，像碎掉又拼起来的镜子。",
        "我走得很慢，想把白天没说完的话排好队。",
        "有人跑过，留下一声短促的笑。",
        "我在便利店买了水，站着喝完才走。",
        "回家路不长，今晚却显得格外远。",
        "推门前深吸一口气，把潮湿留在门外。",
    ]),
]

CHILD_CN: list[tuple] = [
    ("pure_cn", "affection", "", None, None, None, [
        "妈妈，今晚可以再讲一个故事吗？",
        "我喜欢那只勇敢的小兔子，它下雨也不怕。",
        "你声音轻轻的，我就不怕黑了。",
        "讲完我要抱抱你，再睡觉。",
        "枕头有点凉，你可以再靠近一点吗？",
        "明天早上我想吃鸡蛋饼，好不好？",
    ]),
    ("pure_cn", "elation", "expressive_high", None, "laughter", "哈哈", [
        "哈哈，今天体育课我跑得最快！",
        "老师吹哨的时候我已经冲出去了。",
        "同学给我竖大拇指，我开心到跳起来。",
        "回家第一件事就是告诉你。",
        "明天我还要跑，你会来看吗？",
        "奖状我放书包最上层了，你拆开就能看见。",
    ]),
    ("pure_cn", "fear", "", None, None, None, [
        "刚才打雷好响，我把耳朵捂住了。",
        "窗外星星都不见了，只剩雨声。",
        "你可以坐在床边吗，我数到十就不怕。",
        "小夜灯开着，我才能把眼睛闭上。",
        "如果再响，你就拍拍我的背。",
        "我抓紧你的袖子，这样就不孤单。",
    ]),
    ("pure_cn", "amusement", "expressive_high", None, "laughter", "嘻嘻", [
        "嘻嘻，橡皮被我画成小猪了。",
        "同桌笑到桌子都抖，老师看了我们一眼。",
        "我赶紧坐好，可是还是想笑。",
        "下课我又画了一只戴帽子的。",
        "回家给你看，保证你也会笑。",
        "下次我们比赛谁画得更像。",
    ]),
    ("pure_cn", "sadness", "", None, "sigh", "呜", [
        "呜，今天积木塔倒了，倒得好响。",
        "我搭了很久，最上面那块是红色的。",
        "老师说可以再搭，可我还是难受。",
        "你抱抱我，我就重新开始。",
        "这次我要把底座搭得更宽。",
        "搭好了一定先给你看。",
    ]),
    ("pure_cn", "pride", "expressive_high", None, None, None, [
        "今天我自己把鞋带系好了，没有求助。",
        "老师说很棒，还摸了摸我的头。",
        "我走得直直的，生怕它又松开。",
        "到家门口我专门跳了一下给你看。",
        "明天我要系得更快一点。",
        "你会为我骄傲吗，我已经很努力了。",
    ]),
    ("pure_cn", "curiosity", "", None, None, None, [
        "为什么月亮有时候弯，有时候圆？",
        "它是不是也要睡觉，所以会换形状。",
        "你讲的时候我都能听懂，讲慢一点就好。",
        "我想把它画下来，贴在床头。",
        "明天晚上我们一起看，好不好？",
        "如果下雨看不见，我们就看照片。",
    ]),
    ("pure_cn", "enthusiasm", "speed_fast", None, None, None, [
        "动物园的熊猫在爬树，我差点喊出声。",
        "还有小企鹅走路一摇一摆，太好玩了。",
        "我拍了很多张，有的糊了也没关系。",
        "回家我们一张一张看。",
        "下次我想喂胡萝卜，可以预约吗？",
        "今天是最开心的一天，真的。",
    ]),
    ("pure_cn", "shame", "", None, None, None, [
        "今天上课我走神了，老师叫到我名字。",
        "我站起来，答案却卡在嘴巴里。",
        "同学轻轻笑了一下，我脸好烫。",
        "放学我把那一课又读了两遍。",
        "明天我会举手，把丢的分找回来。",
        "你不要生气，我会改的。",
    ]),
    ("pure_cn", "determination", "expressive_high", None, None, None, [
        "跳绳我要连续跳过五十下。",
        "今天跳到四十二就绊倒了，可是我不哭。",
        "我休息十秒，再来一次。",
        "脚有点酸，我还是咬牙继续。",
        "等我成功了，你要鼓掌。",
        "我会记在小本本上，一天比一天多。",
    ]),
]

CHILD_EN: list[tuple] = [
    ("pure_en", "affection", "", None, None, None, [
        "Mom, can you read one more story tonight?",
        "I like the brave little rabbit that is not afraid of rain.",
        "When your voice is soft, the dark feels smaller.",
        "After the story I want a hug, then sleep.",
        "My pillow is cold; can you sit closer?",
        "In the morning I want egg pancakes, okay?",
    ]),
    ("pure_en", "elation", "expressive_high", None, "laughter", "Yay", [
        "Yay, I was the fastest in gym class today!",
        "When the whistle blew I was already gone.",
        "My friend gave me a thumbs-up and I jumped.",
        "Coming home, the first thing is telling you.",
        "Will you watch me run again tomorrow?",
        "I put the certificate on top of my bag so you see it first.",
    ]),
    ("pure_en", "fear", "", None, None, None, [
        "The thunder was so loud I covered my ears.",
        "Outside the stars are gone; there is only rain.",
        "Can you sit by the bed while I count to ten?",
        "I need the night light on before I close my eyes.",
        "If it booms again, just pat my back.",
        "I am holding your sleeve so I am not alone.",
    ]),
    ("pure_en", "amusement", "expressive_high", None, "laughter", "Hee", [
        "Hee, I drew a pig face on my eraser.",
        "My desk partner laughed until the table shook, and the teacher looked over.",
        "I sat up straight but I still wanted to giggle.",
        "At recess I drew another one with a hat.",
        "I will show you at home; you will laugh too.",
        "Next time we can race who draws better.",
    ]),
    ("pure_en", "sadness", "", None, "sigh", "Oh", [
        "Oh, my block tower fell with a big crash.",
        "I built it for a long time; the top brick was red.",
        "Teacher said I can rebuild, but it still hurts.",
        "If you hug me, I will start again.",
        "This time I will make the base wider.",
        "When it stands, you get the first look.",
    ]),
    ("pure_en", "pride", "expressive_high", None, None, None, [
        "Today I tied my shoes by myself, no help.",
        "Teacher said great job and patted my head.",
        "I walked super straight so the bows would stay.",
        "At the door I jumped once just to show you.",
        "Tomorrow I want to tie them even faster.",
        "Are you proud? I tried really hard.",
    ]),
    ("pure_en", "curiosity", "", None, None, None, [
        "Why is the moon sometimes curved and sometimes round?",
        "Does it sleep too, and change shape when it dreams?",
        "If you explain slowly, I can understand.",
        "I want to draw it and stick it by my bed.",
        "Can we watch it together tomorrow night?",
        "If rain hides it, we can look at photos instead.",
    ]),
    ("pure_en", "enthusiasm", "speed_fast", None, None, None, [
        "The panda climbed a tree and I almost yelled.",
        "The penguins walked like toys and it was so funny.",
        "I took lots of pictures; blurry ones still count.",
        "At home we will look at them one by one.",
        "Next time can we book carrot feeding?",
        "Today was the happiest day, for real.",
    ]),
    ("pure_en", "shame", "", None, None, None, [
        "I daydreamed in class and teacher called my name.",
        "I stood up, but the answer stuck in my mouth.",
        "Someone giggled softly and my face got hot.",
        "After school I read that lesson twice.",
        "Tomorrow I will raise my hand and fix it.",
        "Please do not be mad; I will change.",
    ]),
    ("pure_en", "determination", "expressive_high", None, None, None, [
        "I want fifty jump-rope skips in a row.",
        "Today I fell at forty-two, but I did not cry.",
        "I rest ten seconds, then I try again.",
        "My legs are sore and I still keep going.",
        "When I make it, you have to clap.",
        "I will write the number in my little notebook every day.",
    ]),
]


def _form(raws: list[tuple], audience: str, start_idx: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for i, row in enumerate(raws):
        lang, emotion, prosody, style, sfx, ono, sents = row
        # curiosity mapped for child tags that may not exist in emotion set — remap
        if emotion == "curiosity":
            emotion = "awe"
        item = pack(audience, start_idx + i, lang, emotion, prosody, style, sfx, ono, sents, rng)
        out.append(item)
    return out


def main() -> None:
    # Larger pools so we can assign ~hundreds of speakers without heavy reuse.
    adult_cn = _dup_variants(ADULT_CN + _CN_MORE, 134, seed=11)
    adult_en = _dup_variants(ADULT_EN, 134, seed=22)
    adult_mx = _dup_variants(ADULT_MX, 132, seed=33)
    child_cn = _dup_variants(CHILD_CN, 50, seed=44)
    child_en = _dup_variants(CHILD_EN, 50, seed=55)

    adult = (
        _form(adult_cn, "adult", 0, seed=101)
        + _form(adult_en, "adult", 0, seed=102)
        + _form(adult_mx, "adult", 0, seed=103)
    )
    for i, item in enumerate(adult):
        item["id"] = f"adult_{i:04d}"
    child = _form(child_cn, "child", 0, seed=201) + _form(child_en, "child", 0, seed=202)
    for i, item in enumerate(child):
        item["id"] = f"child_{i:04d}"

    # Validate pause constraints.
    for item in adult + child:
        ps = item["pause_secs"]
        assert all(p >= 1.0 for p in ps), item["id"]
        assert sum(ps) / len(ps) >= 2.0 - 1e-6, (item["id"], item["pause_sec_mean"])

    speeches = [x["est_speech_sec"] for x in adult + child]
    means = [x["pause_sec_mean"] for x in adult + child]
    doc = {
        "seed": 42,
        "target_sec": [20, 35],
        "n_adult": len(adult),
        "n_child": len(child),
        "mode_intent": "single",
        "pause_strategy": (
            "one-shot TTS (no long_pause tags); VAD-split speech spans; "
            "insert designed pause_secs (min>=1.0s, mean>=2.0s) between segments"
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
        "pause_sec_mean_corpus": {
            "mean": round(sum(means) / len(means), 3),
            "min": round(min(means), 3),
            "max": round(max(means), 3),
        },
        "note": (
            "Coherent monologues (setup→develop→close). "
            "Delivery tags: emotion/prosody/style at script start; "
            "SFX inserted flexibly mid-text (random sentence, often mid-clause). "
            "Clone via single-shot TTS + VAD splice with designed silence; "
            "optional pyroomacoustics reverb after merge."
        ),
        "adult": adult,
        "child": child,
    }
    OUT.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT} adult={len(adult)} child={len(child)}")
    print("adult langs", doc["lang_mix"]["adult"])
    print("est_speech", doc["est_speech_sec"])
    print("pause_mean", doc["pause_sec_mean_corpus"])
    print("sample pauses", adult[0]["pause_secs"], "mean", adult[0]["pause_sec_mean"])
    print("sample text:", adult[0]["clean_text"][:180])


if __name__ == "__main__":
    main()
