#!/usr/bin/env python3
"""CER evaluation for Higgs Audio v3 TTS clone audio.

Workflow:
  1. Scan clone output for clone_*.wav + clone_*.json pairs
  2. ASR: Qwen3-ASR transcribes wav (batch, 24→16 kHz resample)
  3. Manual ITN (rule-based normalization)
  4. CER via jiwer.process_characters
  5. Optional LLM ITN (disabled by default; use --enable-llm)

Usage:
    conda activate qwen3-asr

    # Default: ASR + manual ITN only
    python eval_cer.py --out-dir /path/to/clone_output

    # Optional LLM ITN pipeline (2 ASR GPUs + 6 LLM GPUs)
    python eval_cer.py --out-dir /path/to/clone_output --enable-llm --asr-gpus 0,1 --llm-concurrency 24

    # Sample mode
    python eval_cer.py --out-dir /path/to/clone_output --sample-size 500 --seed 42
"""

import argparse
import json
import multiprocessing as mp
import os
import pickle
import queue
import re
import string
import sys
import threading
import time
import traceback
import unicodedata
from collections import defaultdict
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from tqdm import tqdm


EVAL_DIR = Path(__file__).resolve().parent
PARENT_DIR = EVAL_DIR.parent
sys.path.insert(0, str(PARENT_DIR))

from eval_common import CerAccumulator, append_jsonl, list_clone_items, write_json  # noqa: E402


QWEN3_ASR_LOCAL = "/root/.cache/huggingface/hub/Qwen3-ASR-1.7B-local"
ENV_FILE = EVAL_DIR / ".env"
DEFAULT_OUT_DIR = Path(
    os.environ.get(
        "HIGGS_CLONE_ROOT",
        "/root/group-shared/voiceprint/data/speech/speaker_diarization/"
        "merged_datasets_20250610_vad_segments_mtfaa_enhanced_extend_kid_withclone_addlibrilight_1130/"
        "audio_higgs_audio_v3_tts_clone",
    )
)
DEFAULT_BATCH_SIZE = 16
DEFAULT_AUDIO_WORKERS = 0

HIGGS_TAG_RE = re.compile(r"<\|[^|>]+\|[^>]*>|<\s*(?:emotion|style|sfx|prosody):[^>]+>", re.IGNORECASE)
_CJK_CHAR = r"[\u4e00-\u9fff]"

PUNCTUATION = set(
    string.punctuation
    + "，。！？；：\"\"''（）【】《》、·～｀＠＃￥％＆＊—＋｜＜＞？／"
    + "\u2018\u2019\u201c\u201d\u2026"
)

ITN_SYSTEM_PROMPT = """你是ASR评测ITN专家。对ref/hyp文本对做逆文本归一化+交叉对齐，消除虚假字错。

## 核心规则
1. **等价统一**：读音/语义相同仅写法不同→统一为ref侧形式，hyp跟随
2. **真差异不对齐**：不同名词/动词/数字/语义保留
3. **输出**：JSON数组 [{"id":0,"ref_final":"...","hypo_final":"..."}]，小写无标点保留空格

## 拟声词/语气词同音对齐
文本中可能有TTS音效对应的拟声词（唉、哈哈、咳咳等），ASR可能转写为同音字，应做同音对齐：
- 唉/哎/诶 视为等价（叹气声）
- 哈哈/嘿嘿/呵呵/嘻嘻 视为等价（笑声）
- 咳/咳咳 视为等价（咳嗽声）
- 呜呜/呜/嘤嘤 视为等价（哭泣声）
- 嗯/唔/哼 视为等价（哼声）
- 啊/呀/哇 视为等价（惊叫声）
- 阿嚏/哈啾 视为等价（喷嚏声）
- Haha/Hehe/Hoho 视为等价 | Sob/Boo hoo 视为等价 | Achoo/Atchoo 视为等价

## 数字/单位
- 中文数词→阿拉伯：一百二十→120 | 逐字读数：三零二→302 | 英文：forty-five→45
- 百分数：百分之八十/80percent→80 | 分数：三分之七→3分之7
- 单位：km/h→kmh，centimeters→cm，3minutes→3min
- 保留小数点：12.5≠125 | 一样/一般/一起不转1

## 同音对齐
- 人名：芳芳/方方 | 词语：刻舟/勾舟 | 拼音↔汉字：hongjun→红军
- 公式：a²+b²=c²↔a块大c色块 | 英文变体：question1/questionone

## 示例
ref: `80和120块wait` hyp: `八百分之和一百二十块weight` → `80和120块weight`
ref: `芳芳的blue shirt` hyp: `方方的blue shirt` → `芳芳的blue shirt`
ref: `call 8877` hyp: `call eight eight seven seven` → `call 8877`
ref: `哈哈太棒了` hyp: `嘿嘿太棒了` → hyp改为 `哈哈太棒了`
ref: `慢点慢点唉算了` hyp: `慢点慢点哎算了` → hyp改为 `慢点慢点唉算了`
"""


def load_env_file(path: Path):
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip())


load_env_file(ENV_FILE)

_LLM_ENDPOINTS: list[str] = []
_LLM_ENDPOINT_LOCK = threading.Lock()
_LLM_ENDPOINT_IDX = 0


def _init_llm_endpoints():
    global _LLM_ENDPOINTS
    urls_str = os.environ.get("ITN_LLM_BASE_URLS", "")
    if urls_str:
        _LLM_ENDPOINTS = [u.strip().rstrip("/") for u in urls_str.split(",") if u.strip()]
    else:
        single = os.environ.get("ITN_LLM_BASE_URL", "")
        if single:
            _LLM_ENDPOINTS = [single.strip().rstrip("/")]


def _next_llm_endpoint() -> str:
    global _LLM_ENDPOINT_IDX
    with _LLM_ENDPOINT_LOCK:
        endpoint = _LLM_ENDPOINTS[_LLM_ENDPOINT_IDX % len(_LLM_ENDPOINTS)]
        _LLM_ENDPOINT_IDX += 1
    return endpoint


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=200_000)
def get_truth_text(json_path: Path) -> str:
    meta = load_json(json_path)
    return meta.get("clean_text", "") or meta.get("text", "")


def infer_asr_language(text: str) -> str:
    cjk = len(re.findall(_CJK_CHAR, text or ""))
    latin = len(re.findall(r"[A-Za-z]", text or ""))
    if latin and not cjk:
        return "English"
    if cjk and not latin:
        return "Chinese"
    # Only force a prompt when one language clearly dominates. Balanced code-switching
    # should let Qwen3-ASR infer the language instead of biasing it toward either side.
    if latin > cjk * 2:
        return "English"
    if cjk > latin * 2:
        return "Chinese"
    return "Unknown"


def infer_asr_language_for_json(json_path: Path) -> str:
    return infer_asr_language(get_truth_text(json_path))


def get_cached_asr_text(asr_results: dict, wav_path: Path, language: str) -> str | None:
    entry = asr_results.get(str(wav_path))
    if entry is None:
        return None
    if isinstance(entry, dict):
        if entry.get("language") == language:
            return entry.get("text") or None
        return None
    # Legacy cache entries were produced with the old fixed Chinese prompt.
    if language == "Chinese":
        text = str(entry)
        return text or None
    return None


def set_cached_asr_text(asr_results: dict, wav_path: Path, language: str, text: str) -> None:
    asr_results[str(wav_path)] = {"text": text, "language": language}


def has_cached_asr_text(asr_results: dict, wav_path: Path, json_path: Path) -> bool:
    language = infer_asr_language_for_json(json_path)
    return get_cached_asr_text(asr_results, wav_path, language) is not None


def extract_speech(wav_path: Path, target_sr: int = 16000):
    wav, sr = sf.read(str(wav_path), dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != target_sr:
        wav = torchaudio.functional.resample(torch.from_numpy(wav), orig_freq=sr, new_freq=target_sr).numpy()
    return wav


def load_asr_audio_input(wav_path: str) -> tuple[np.ndarray, int]:
    return extract_speech(Path(wav_path), target_sr=16000), 16000


def asr_language_arg(language: str) -> str | None:
    return None if language == "Unknown" else language


def submit_audio_prefetch(batch_pairs: list, audio_pool: ProcessPoolExecutor | None = None) -> list:
    prefetched = []
    for wav_path, json_path in batch_pairs:
        language = infer_asr_language_for_json(json_path)
        if audio_pool is None:
            prefetched.append((wav_path, language, None))
        else:
            prefetched.append((wav_path, language, audio_pool.submit(load_asr_audio_input, str(wav_path))))
    return prefetched


def strip_higgs_tags(text: str) -> str:
    return HIGGS_TAG_RE.sub("", text).strip()


def strip_punctuation(text: str) -> str:
    chars = list(text)
    out = []
    for i, ch in enumerate(chars):
        if ch == "." and i > 0 and i + 1 < len(chars) and chars[i - 1].isdigit() and chars[i + 1].isdigit():
            out.append(ch)
        elif ch not in PUNCTUATION:
            out.append(ch)
    return "".join(out)


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def finalize_cer_text(text: str) -> str:
    text = normalize_spaces(text)
    text = re.sub(rf"({_CJK_CHAR})\s+([a-zA-Z0-9])", r"\1\2", text)
    text = re.sub(rf"([a-zA-Z0-9])\s+({_CJK_CHAR})", r"\1\2", text)
    text = re.sub(
        r"\b(question|number|no|room|page|chapter|section|episode|item|part|version)\s+(\d+)\b",
        r"\1\2",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\b([a-zA-Z])\s+(\d+)(?=\b)", r"\1\2", text)
    text = re.sub(r"\b(\d+)\s+([a-zA-Z]+)\b", r"\1\2", text)
    return text


_CN_DIGIT = {
    "零": "0",
    "〇": "0",
    "一": "1",
    "二": "2",
    "两": "2",
    "三": "3",
    "四": "4",
    "五": "5",
    "六": "6",
    "七": "7",
    "八": "8",
    "九": "9",
}
_CN_NUM = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "壹": 1,
    "二": 2,
    "两": 2,
    "贰": 2,
    "三": 3,
    "叁": 3,
    "四": 4,
    "肆": 4,
    "五": 5,
    "伍": 5,
    "六": 6,
    "陆": 6,
    "七": 7,
    "柒": 7,
    "八": 8,
    "捌": 8,
    "九": 9,
    "玖": 9,
    "十": 10,
    "拾": 10,
    "百": 100,
    "佰": 100,
    "千": 1000,
    "仟": 1000,
    "万": 10000,
}
_CN_SUFFIXES = (
    "单元",
    "号楼",
    "毫米",
    "厘米",
    "公里",
    "小时",
    "分钟",
    "块",
    "毛",
    "元",
    "个",
    "岁",
    "分",
    "秒",
    "年",
    "月",
    "日",
    "号",
    "层",
    "页",
    "名",
    "倍",
    "点",
    "支",
    "次",
    "遍",
    "顿",
    "米",
)

_EN_NUM_WORDS = {
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
    "hundred",
    "thousand",
    "million",
    "billion",
    "and",
    "point",
}
_EN_DIGIT_WORDS = {
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
}
_EN_NUM_WORDS_LIST = sorted(_EN_NUM_WORDS - {"and", "point"}, key=len, reverse=True)
_EN_DIGIT_SEQ_RE = re.compile(
    r"\b(?:" + "|".join(_EN_DIGIT_WORDS) + r")(?:[\s,\-]+(?:" + "|".join(_EN_DIGIT_WORDS) + r"))+\b",
    re.IGNORECASE,
)


def _parse_chinese_number(s: str):
    if not s:
        return None
    if s.isdigit():
        return int(s)
    if len(s) == 1 and s in _CN_NUM:
        v = _CN_NUM[s]
        return v if v < 10 else None
    total = 0
    current = 0
    for ch in s:
        if ch not in _CN_NUM:
            return None
        v = _CN_NUM[ch]
        if v >= 10:
            if current == 0:
                current = 1
            total += current * v
            current = 0
        else:
            current = current * 10 + v if current else v
    total += current
    return total


def _cn_to_str(s: str) -> str:
    if s.isdigit():
        return s
    val = _parse_chinese_number(s)
    return str(val) if val is not None else s


def normalize_percent(text: str) -> str:
    text = re.sub(r"([零〇一二三四五六七八])百分之", lambda m: _cn_to_str(m.group(1)) + "0", text)
    text = re.sub(r"百分之([零〇一二两三四五六七八九十百千万亿]+)", lambda m: _cn_to_str(m.group(1)), text)
    text = re.sub(r"百分之(\d+(?:\.\d+)?)", r"\1", text)
    text = re.sub(r"(\d+(?:\.\d+)?)\s*%", r"\1", text)
    text = text.replace("百分之", "")
    text = re.sub(r"\bpercent\b", "", text, flags=re.IGNORECASE)
    return text


def normalize_fractions(text: str) -> str:
    return re.sub(
        r"([零〇一二两三四五六七八九十百千万\d]+)分之([零〇一二两三四五六七八九十百千万\d]+)",
        lambda m: f"{_cn_to_str(m.group(1))}分之{_cn_to_str(m.group(2))}",
        text,
    )


def normalize_chinese_money(text: str) -> str:
    return re.sub(
        r"([零〇一二两三四五六七八九十\d])块([零〇一二两三四五六七八九十\d])",
        lambda m: f"{_cn_to_str(m.group(1))}块{_cn_to_str(m.group(2))}",
        text,
    )


def normalize_numeric_format(text: str) -> str:
    text = re.sub(r"(?<=\d)[,，](?=\d{3}\b)", "", text)
    text = re.sub(r"[$＄]\s*(\d+(?:\.\d+)?)", r"\1dollar", text)
    text = re.sub(r"[¥￥]\s*(\d+(?:\.\d+)?)", r"\1yuan", text)
    text = re.sub(r"[€]\s*(\d+(?:\.\d+)?)", r"\1euro", text)
    text = re.sub(r"[£]\s*(\d+(?:\.\d+)?)", r"\1pound", text)
    return text


def normalize_units(text: str) -> str:
    text = re.sub(r"\b(?:kilometers?|kms?)\s*(?:per|/)\s*(?:hour|hr|h)\b", "kmh", text, flags=re.IGNORECASE)
    text = re.sub(r"\bkm/h\b", "kmh", text, flags=re.IGNORECASE)
    text = re.sub(r"\bcentimeters?\b", "cm", text, flags=re.IGNORECASE)
    text = re.sub(r"\bcentimetres?\b", "cm", text, flags=re.IGNORECASE)
    text = re.sub(r"\bmillimeters?\b", "mm", text, flags=re.IGNORECASE)
    text = re.sub(r"\bmillimetres?\b", "mm", text, flags=re.IGNORECASE)
    text = re.sub(r"\bmeters?\b", "m", text, flags=re.IGNORECASE)
    text = re.sub(r"\bmetres?\b", "m", text, flags=re.IGNORECASE)
    text = re.sub(r"\bminutes?\b", "min", text, flags=re.IGNORECASE)
    text = re.sub(r"\bseconds?\b", "s", text, flags=re.IGNORECASE)
    text = re.sub(r"\bhours?\b", "h", text, flags=re.IGNORECASE)
    text = re.sub(r"\bdollars?\b", "dollar", text, flags=re.IGNORECASE)
    text = re.sub(r"\byuan\b", "yuan", text, flags=re.IGNORECASE)
    text = re.sub(r"\beuros?\b", "euro", text, flags=re.IGNORECASE)
    text = re.sub(r"\bpounds?\b", "pound", text, flags=re.IGNORECASE)
    text = re.sub(r"(\d)\s+kmh\b", r"\1kmh", text, flags=re.IGNORECASE)
    text = re.sub(r"(\d)\s+(cm|mm|m|min|s|h|dollar|yuan|euro|pound)\b", r"\1\2", text, flags=re.IGNORECASE)
    return text


def normalize_english_hundred_money_words(text: str) -> str:
    word_pattern = r"(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)(?:[\s-]+(?:one|two|three|four|five|six|seven|eight|nine))*"
    pattern = re.compile(rf"\b({word_pattern})\s+hundred\s+(dollars?|yuan|euros?|pounds?)\b", re.IGNORECASE)
    values = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "eleven": 11,
        "twelve": 12,
        "thirteen": 13,
        "fourteen": 14,
        "fifteen": 15,
        "sixteen": 16,
        "seventeen": 17,
        "eighteen": 18,
        "nineteen": 19,
        "twenty": 20,
        "thirty": 30,
        "forty": 40,
        "fifty": 50,
        "sixty": 60,
        "seventy": 70,
        "eighty": 80,
        "ninety": 90,
    }

    def _repl(m):
        value = 0
        for word in re.split(r"[\s-]+", m.group(1).lower()):
            if word not in values:
                return m.group(0)
            value += values[word]
        if not value:
            return m.group(0)
        unit = normalize_units(m.group(2))
        return f"{value * 100}{unit}"

    return pattern.sub(_repl, text)


def normalize_english_numbers(text: str) -> str:
    from word2number import w2n

    def _repl_digit_seq(m):
        digits = []
        for w in re.split(r"[\s,\-]+", m.group(0)):
            w = w.lower()
            if w in _EN_DIGIT_WORDS:
                digits.append(str(w2n.word_to_num(w)))
            else:
                return m.group(0)
        return "".join(digits)

    text = _EN_DIGIT_SEQ_RE.sub(_repl_digit_seq, text)

    _EN_COMPOUND_RE = re.compile(
        r"\b(?:"
        + "|".join(_EN_NUM_WORDS_LIST)
        + r")(?:(?:[\s\-]+(?:and[\s\-]+)?(?:"
        + "|".join(_EN_NUM_WORDS_LIST)
        + r"))+)(?![a-zA-Z])",
        re.IGNORECASE,
    )
    _EN_SINGLE_RE = re.compile(r"\b(" + "|".join(_EN_NUM_WORDS_LIST) + r")\b", re.IGNORECASE)

    def _convert_english(phrase):
        try:
            return str(w2n.word_to_num(phrase.replace("-", " ").lower()))
        except Exception:
            return phrase

    for _ in range(4):
        prev = text
        text = _EN_COMPOUND_RE.sub(lambda m: _convert_english(m.group(0)), text)
        text = _EN_SINGLE_RE.sub(lambda m: _convert_english(m.group(1)), text)
        if text == prev:
            break
    return text


def normalize_english_hundreds(text: str) -> str:
    return re.sub(
        r"\b(\d+)\s+hundred\s+(dollar|yuan|euro|pound)\b",
        lambda m: f"{int(m.group(1)) * 100}{m.group(2)}",
        text,
        flags=re.IGNORECASE,
    )


_CN_SUFFIX_PATTERN = "|".join(re.escape(s) for s in _CN_SUFFIXES)
_CN_COMPOUND_RE = re.compile(
    r"[零〇一二两贰三四五六七八九十拾百千仟万]{2,}(?:" + _CN_SUFFIX_PATTERN + r")?"
    r"|[一二两三四五六七八九](?:" + _CN_SUFFIX_PATTERN + r")"
)
_CN_DIGIT_RUN_RE = re.compile(r"[零〇一二三四五六七八九]{2,}")


def _split_cn_num_suffix(full: str):
    for suffix in sorted(_CN_SUFFIXES, key=len, reverse=True):
        if full.endswith(suffix):
            return full[: -len(suffix)], suffix
    return full, ""


def normalize_chinese_numbers(text: str) -> str:
    def _repl(m):
        full = m.group(0)
        num_str, unit = _split_cn_num_suffix(full)
        val = _parse_chinese_number(num_str)
        return (str(val) + unit) if val is not None else m.group(0)

    return _CN_COMPOUND_RE.sub(_repl, text)


def normalize_chinese_digit_runs(text: str) -> str:
    def _repl(m):
        s = m.group(0)
        if any(c in "十百千万亿" for c in s):
            return s
        return "".join(_CN_DIGIT.get(c, c) for c in s)

    return _CN_DIGIT_RUN_RE.sub(_repl, text)


_CONTRACTION_REPLACEMENTS = (
    (r"\bcan(?:'|’)?t\b", "can not"),
    (r"\bdon(?:'|’)?t\b", "do not"),
    (r"\bdidn(?:'|’)?t\b", "did not"),
    (r"\bisn(?:'|’)?t\b", "is not"),
    (r"\baren(?:'|’)?t\b", "are not"),
    (r"\bwasn(?:'|’)?t\b", "was not"),
    (r"\bweren(?:'|’)?t\b", "were not"),
    (r"\bwon(?:'|’)?t\b", "will not"),
    (r"\bi(?:'|’)?m\b", "i am"),
    (r"\byou(?:'|’)?re\b", "you are"),
    (r"\bwe(?:'|’)?re\b", "we are"),
    (r"\bthey(?:'|’)?re\b", "they are"),
    (r"\bi(?:'|’)?ve\b", "i have"),
    (r"\byou(?:'|’)?ve\b", "you have"),
    (r"\bwe(?:'|’)?ve\b", "we have"),
    (r"\bi(?:'|’)?ll\b", "i will"),
    (r"\byou(?:'|’)?ll\b", "you will"),
    (r"\bwe(?:'|’)?ll\b", "we will"),
    (r"\bthat(?:'|’)?s\b", "that is"),
    (r"\bwhat(?:'|’)?s\b", "what is"),
    (r"\bit(?:'|’)?s\b", "it is"),
    (r"\bthere(?:'|’)?s\b", "there is"),
    (r"\bhere(?:'|’)?s\b", "here is"),
    (r"\blet(?:'|’)?s\b", "let us"),
    (r"\bi(?:'|’)?d\b", "i would"),
    (r"\byou(?:'|’)?d\b", "you would"),
    (r"\bhe(?:'|’)?d\b", "he would"),
    (r"\bshe(?:'|’)?d\b", "she would"),
    (r"\bwe(?:'|’)?d\b", "we would"),
    (r"\bthey(?:'|’)?d\b", "they would"),
    (r"\bshould(?:'|’)?ve\b", "should have"),
    (r"\bwould(?:'|’)?ve\b", "would have"),
    (r"\bcould(?:'|’)?ve\b", "could have"),
)


def normalize_english_contractions(text: str) -> str:
    for pattern, repl in _CONTRACTION_REPLACEMENTS:
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
    return text


def normalize_symbols(text: str) -> str:
    text = re.sub(r"\s*&\s*", " and ", text)
    text = re.sub(r"\s*@\s*", " at ", text)
    return text


_SFX_REPLACEMENTS = (
    (r"哎|诶|欸", "唉"),
    (r"嘿嘿|呵呵|嘻嘻|哈哈哈+", "哈哈"),
    (r"咳咳+", "咳"),
    (r"呜呜+|呜|嘤嘤+", "呜"),
    (r"唔|哼", "嗯"),
    (r"呀|哇", "啊"),
    (r"哈啾", "阿嚏"),
    (r"\b(?:ha\s*){2,}\b|\bhe\s*he\b|\bho\s*ho\b", "哈哈"),
    (r"\bboo\s*hoo\b|\bsobs?\b", "呜"),
    (r"\b(?:a|ha)?choo\b|\b(?:a|ha)?tchoo\b", "阿嚏"),
    (r"\b(?:ah+|aah+|oh+)\b", "啊"),
    (r"\b(?:um+|uh+|erm|hmm+|mmm+)\b", "嗯"),
)


def normalize_sfx_words(text: str) -> str:
    for pattern, repl in _SFX_REPLACEMENTS:
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
    return text


_SPELLED_LETTER_SEQ_RE = re.compile(r"\b[a-zA-Z](?:\s+[a-zA-Z]){1,5}\b")


def normalize_spelled_letters(text: str) -> str:
    def _repl(m):
        letters = m.group(0).split()
        if len(letters) < 2 or not all(len(ch) == 1 for ch in letters):
            return m.group(0)
        return "".join(letters)

    return _SPELLED_LETTER_SEQ_RE.sub(_repl, text)


_ERHUA_RE = re.compile(r"(?<=[\u4e00-\u9fff])(味|会|点|事|声|劲|样|块|片|阵|下|天|地|门|人|孩)儿")


def normalize_chinese_particles(text: str) -> str:
    text = _ERHUA_RE.sub(r"\1", text)
    text = re.sub(r"(?<=[\u4e00-\u9fff])[得地](?=[像很真太特更不没一二三四五六七八九十])", "的", text)
    return text


def normalize_numbers(text: str) -> str:
    text = normalize_numeric_format(text)
    text = normalize_percent(text)
    text = normalize_fractions(text)
    text = normalize_chinese_money(text)
    text = normalize_units(text)
    text = normalize_english_hundred_money_words(text)
    text = normalize_english_numbers(text)
    text = normalize_english_hundreds(text)
    text = normalize_chinese_numbers(text)
    text = normalize_chinese_digit_runs(text)
    text = normalize_units(text)
    return text


def manual_itn_preprocess(text: str, keep_tag: bool = False) -> str:
    processed = normalize_unicode(text)
    if not keep_tag:
        processed = strip_higgs_tags(processed)
    processed = normalize_symbols(processed)
    processed = normalize_english_contractions(processed)
    processed = normalize_sfx_words(processed)
    processed = normalize_numbers(processed)
    processed = normalize_chinese_particles(processed)
    processed = normalize_spelled_letters(processed)
    return finalize_cer_text(strip_punctuation(processed).lower())


def manual_itn(text: str) -> str:
    return manual_itn_preprocess(text, keep_tag=False)


def cleanup_llm_spacing(hyp: str) -> str:
    if not hyp:
        return hyp
    hyp = re.sub(rf"({_CJK_CHAR})\s+([a-zA-Z0-9])", r"\1\2", hyp)
    hyp = re.sub(rf"([a-zA-Z0-9])\s+({_CJK_CHAR})", r"\1\2", hyp)
    return hyp


def calc_cer(ref: str, hyp: str):
    from jiwer import process_characters

    try:
        cm = process_characters(ref, hyp)
        return cm.cer, cm.substitutions, cm.insertions, cm.deletions, len(ref)
    except Exception:
        return 1.0, 0, 0, 0, len(ref)


def llm_itn_postprocess(ref_llm: str, hyp_llm: str, ref_manual: str, hyp_manual: str) -> tuple[str, str]:
    ref_f = ref_manual
    if not hyp_llm:
        return ref_f, hyp_manual
    hyp_llm_clean = cleanup_llm_spacing(finalize_cer_text(hyp_llm))
    if calc_cer(ref_f, hyp_llm_clean)[0] <= calc_cer(ref_f, hyp_manual)[0] + 1e-9:
        return ref_f, hyp_llm_clean
    return ref_f, hyp_manual


def _extract_json_array(raw_text: str) -> list:
    text = raw_text.strip()
    text = re.sub(r"<\|thinker\|>.*?<\|/thinker\|>", "", text, flags=re.DOTALL)
    text = re.sub(r"<\|assistant\|>", "", text)
    text = re.sub(r"<\|endoftext\|>", "", text)
    for marker in ("```json", "```"):
        if marker in text:
            text = text.split(marker, 1)[1]
            if "```" in text:
                text = text.split("```", 1)[0].strip()
            break
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]
    decoder = json.JSONDecoder()
    try:
        data, _ = decoder.raw_decode(text)
    except json.JSONDecodeError:
        data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError("Expected JSON array")
    return data


def call_llm_itn_batch(batch_items: list[dict], max_retries: int = 6, endpoint: str | None = None) -> list[dict]:
    from openai import OpenAI

    api_key = os.environ.get("ITN_LLM_API_KEY") or os.environ.get("LLM_API_KEY", "EMPTY")
    model = os.environ.get("ITN_LLM_MODEL") or os.environ.get("LLM_MODEL", "qwen3.6-27b")
    if endpoint is None:
        endpoint = _next_llm_endpoint()

    client = OpenAI(api_key=api_key, base_url=endpoint.rstrip("/"))

    user_lines = [
        "以下 pairs 为第一阶段手工 ITN 结果（已去标签/部分数字/去标点/小写，**保留空格**）。",
        "手工 ITN 不完备，请继续 ITN + ref/hyp 交叉对齐，返回 JSON 数组：",
    ]
    for item in batch_items:
        user_lines.extend([f"\n[id={item['id']}]", f"ref: {item['ref']}", f"hypo: {item['hypo']}"])

    last_err = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": ITN_SYSTEM_PROMPT},
                    {"role": "user", "content": "\n".join(user_lines)},
                ],
                max_tokens=4096,
                temperature=0,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            content = response.choices[0].message.content
            return _extract_json_array(content)
        except Exception as e:
            last_err = e
            if attempt + 1 >= max_retries:
                break
            if "429" in str(e):
                time.sleep(min(60, 5 * (2**attempt)))
            else:
                time.sleep(2**attempt)
    raise RuntimeError(f"LLM ITN failed after {max_retries} retries: {last_err}")


def llm_itn_batch_fetch(batch: list[dict]) -> dict[str, dict]:
    if not batch:
        return {}
    batch_input = [{"id": j, "ref": it["ref_manual"], "hypo": it["hypo_manual"]} for j, it in enumerate(batch)]
    raw = call_llm_itn_batch(batch_input)
    by_id = {r["id"]: r for r in raw}
    entries = {}
    for j, it in enumerate(batch):
        r = by_id.get(j, {})
        ref_f, hyp_f = llm_itn_postprocess(
            r.get("ref_final", ""),
            r.get("hypo_final", ""),
            it["ref_manual"],
            it["hypo_manual"],
        )
        entries[it["wav"]] = {"ref_final": ref_f, "hypo_final": hyp_f}
    return entries


def load_asr_model(batch_size: int = 16, gpu_id: int = 0):
    print(f"Loading Qwen3-ASR model on GPU {gpu_id}...", flush=True)
    sys.path.insert(0, "/root/code/github_repos/Qwen3-ASR")
    from qwen_asr import Qwen3ASRModel

    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    asr = Qwen3ASRModel.from_pretrained(
        QWEN3_ASR_LOCAL,
        dtype=torch.bfloat16,
        device_map=device,
        max_inference_batch_size=batch_size,
        max_new_tokens=256,
    )
    print(f"ASR loaded (batch_size={batch_size}, gpu={gpu_id}).\n", flush=True)
    return asr


def transcribe_asr_batch(
    asr,
    batch_pairs: list,
    results: dict,
    audio_pool: ProcessPoolExecutor | None = None,
    prefetched: list | None = None,
) -> None:
    if prefetched is None:
        prefetched = submit_audio_prefetch(batch_pairs, audio_pool=audio_pool)

    by_language: dict[str, list[tuple[Path, tuple]]] = defaultdict(list)
    for wav_path, language, audio_future in prefetched:
        try:
            audio_input = audio_future.result() if audio_future is not None else str(wav_path)
            by_language[language].append((wav_path, audio_input))
        except Exception as e:
            print(f"Error loading {wav_path}: {e}", flush=True)
            set_cached_asr_text(results, wav_path, language, "")

    for language, items in by_language.items():
        if not items:
            continue
        valid_paths = [wav_path for wav_path, _ in items]
        audio_inputs = [audio for _, audio in items]
        try:
            hypos = asr.transcribe(audio=audio_inputs, language=asr_language_arg(language), return_time_stamps=False)
            for wav_path, h in zip(valid_paths, hypos):
                set_cached_asr_text(results, wav_path, language, h.text)
        except Exception as e:
            print(f"ASR batch error (language={language}): {e}", flush=True)
            for wav_path in valid_paths:
                set_cached_asr_text(results, wav_path, language, "")


def build_eval_item(wav_path: Path, json_path: Path, hypo_raw: str) -> dict:
    meta = load_json(json_path)
    truth_raw = meta.get("clean_text", "") or meta.get("text", "")
    asr_language = infer_asr_language(truth_raw)

    ref_manual = manual_itn(truth_raw)
    hyp_manual = manual_itn(hypo_raw)
    manual_cer, csub, cins, cdel, cnum = calc_cer(ref_manual, hyp_manual)

    return {
        "wav": str(wav_path),
        "json": str(json_path),
        "name": wav_path.name,
        "ref_start": truth_raw,
        "hypo_start": hypo_raw,
        "ref_manual": ref_manual,
        "hypo_manual": hyp_manual,
        "manual_cer": manual_cer,
        "substitutions": csub,
        "insertions": cins,
        "deletions": cdel,
        "chars": cnum,
        "dataset": meta.get("dataset", "unknown"),
        "speaker_id": meta.get("speaker_id", "unknown"),
        "uid": meta.get("uid", "unknown"),
        "asr_language": asr_language,
    }


def apply_llm_to_item(item: dict, llm: dict, model_name: str) -> dict:
    item["ref_llm"], item["hypo_llm"] = llm_itn_postprocess(
        llm.get("ref_final", ""),
        llm.get("hypo_final", ""),
        item["ref_manual"],
        item["hypo_manual"],
    )
    item["llm_cer"], item["llm_sub"], item["llm_ins"], item["llm_del"], _ = calc_cer(item["ref_llm"], item["hypo_llm"])
    item["llm_model"] = model_name
    return item


def build_eval_record(item: dict, evaluated_at: str, llm: bool = False, model_name=None) -> dict:
    record = {
        "wav_path": item["wav"],
        "gen_text": item["ref_start"],
        "asr_hypo": item["hypo_start"],
        "asr_language": item.get("asr_language", "unknown"),
        "ref_manual": item["ref_manual"],
        "hypo_manual": item["hypo_manual"],
        "manual_cer": item["manual_cer"],
        "substitutions": item["substitutions"],
        "insertions": item["insertions"],
        "deletions": item["deletions"],
        "chars": item["chars"],
        "evaluated_at": evaluated_at,
        "dataset": item["dataset"],
        "speaker_id": item["speaker_id"],
        "stage": "complete" if llm else "manual",
    }
    if llm:
        record.update(
            {
                "ref_llm": item.get("ref_llm", ""),
                "hypo_llm": item.get("hypo_llm", ""),
                "llm_cer": item.get("llm_cer", 0),
                "llm_model": model_name,
            }
        )
    return record


def write_eval_json(json_path: Path, record: dict):
    eval_path = json_path.with_suffix(".cer.json")
    eval_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def eval_paths(out_dir: Path) -> dict:
    return {
        "asr_cache": out_dir / "eval_higgs_asr_cache.json",
        "llm_cache": out_dir / "eval_higgs_llm_itn_cache.json",
        "summary": out_dir / "eval_higgs_cer_summary.json",
        "summary_progress": out_dir / "eval_higgs_cer_progress.json",
        "details_jsonl": out_dir / "eval_higgs_cer_details.jsonl",
    }


def summarize_cer(details: list, ref_key: str = "ref_manual", hyp_key: str = "hypo_manual") -> dict:
    total_sub = total_ins = total_del = total_chars = 0
    cers = []
    for item in details:
        cer, sub, ins, dele, chars = calc_cer(item[ref_key], item[hyp_key])
        total_sub += sub
        total_ins += ins
        total_del += dele
        total_chars += chars
        cers.append(cer)
    weighted = (total_sub + total_ins + total_del) / total_chars * 100 if total_chars else 0
    return {
        "weighted_cer": weighted,
        "avg_cer": float(np.mean(cers) * 100) if cers else 0,
        "median_cer": float(np.median(cers) * 100) if cers else 0,
        "min_cer": float(min(cers) * 100) if cers else 0,
        "max_cer": float(max(cers) * 100) if cers else 0,
        "p10_cer": float(np.percentile(cers, 10) * 100) if cers else 0,
        "p90_cer": float(np.percentile(cers, 90) * 100) if cers else 0,
        "total_chars": total_chars,
        "total_ins": total_ins,
        "total_del": total_del,
        "total_sub": total_sub,
        "count": len(cers),
    }


CACHE_FLUSH_INTERVAL = 50


def load_scan_cache(out_dir: Path, refresh: bool = False) -> list[tuple[Path, Path]]:
    cache_path = out_dir / "eval_higgs_scan_cache.pkl"
    if not refresh and cache_path.exists():
        pairs = pickle.loads(cache_path.read_bytes())
        ok = all(w.exists() and j.exists() for w, j in pairs[:20])
        if ok and pairs:
            print(f"[scan-cache] loaded {len(pairs)} pairs from {cache_path.name}", flush=True)
            return pairs
        print("[scan-cache] stale cache, re-scanning...", flush=True)

    pairs = list_clone_items(out_dir, label="cer-scan")
    if pairs:
        cache_path.write_bytes(pickle.dumps(pairs))
        print(f"[scan-cache] saved {len(pairs)} pairs to {cache_path.name}", flush=True)
    return pairs


def flush_json_cache(cache_path: Path, data: dict):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(cache_path)


def apply_shard_paths(paths: dict, shard_index: int | None) -> dict:
    if shard_index is None:
        return paths
    suffix = f".shard{shard_index:02d}"
    out = dict(paths)
    out["asr_cache"] = paths["asr_cache"].with_name(f"eval_higgs_asr_cache{suffix}.json")
    out["llm_cache"] = paths["llm_cache"].with_name(f"eval_higgs_llm_itn_cache{suffix}.json")
    out["summary"] = paths["summary"].with_name(f"eval_higgs_cer_summary{suffix}.json")
    out["summary_progress"] = paths["summary_progress"].with_name(f"eval_higgs_cer_progress{suffix}.json")
    out["details_jsonl"] = paths["details_jsonl"].with_name(f"eval_higgs_cer_details{suffix}.jsonl")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-asr", action="store_true", help="Use cached ASR results")
    parser.add_argument("--skip-llm", action="store_true", help="Deprecated: LLM ITN is skipped by default")
    parser.add_argument("--enable-llm", action="store_true", help="Enable optional LLM ITN after manual ITN")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--asr-gpus", type=str, default="0,1", help="Comma-separated GPU ids for ASR (default: 0,1)")
    parser.add_argument(
        "--audio-workers", type=int, default=DEFAULT_AUDIO_WORKERS, help="CPU processes for audio loading/resampling"
    )
    parser.add_argument("--prefetch-batches", type=int, default=2, help="ASR batches to prefetch per GPU worker")
    parser.add_argument(
        "--refresh-asr-cache", action="store_true", help="Ignore existing ASR cache and transcribe all selected items"
    )
    parser.add_argument("--num-shards", type=int, default=1, help="Split work into N deterministic shards")
    parser.add_argument("--shard-index", type=int, default=0, help="Shard index for this process, in [0, num-shards)")
    parser.add_argument(
        "--llm-concurrency", type=int, default=24, help="Parallel LLM ITN workers (default: 24 = 6 GPUs x 4)"
    )
    parser.add_argument("--refresh-llm-cache", action="store_true")
    parser.add_argument("--refresh-scan", action="store_true", help="Force re-scan (ignore scan cache)")
    args = parser.parse_args()
    args.skip_llm = not args.enable_llm or args.skip_llm

    import random

    random.seed(args.seed)

    asr_gpu_ids = [int(g.strip()) for g in args.asr_gpus.split(",") if g.strip()]
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must be in [0, num-shards)")
    shard_suffix = args.shard_index if args.num_shards > 1 else None

    pairs = load_scan_cache(args.out_dir, refresh=args.refresh_scan)
    if not pairs:
        print("No clone audio found.", flush=True)
        return

    if args.num_shards > 1:
        before = len(pairs)
        pairs = [pair for i, pair in enumerate(pairs) if i % args.num_shards == args.shard_index]
        print(f"Shard {args.shard_index}/{args.num_shards}: {len(pairs)} of {before} clones", flush=True)

    if args.skip_existing:
        before = len(pairs)
        if args.skip_llm:
            pairs = [(w, j) for w, j in pairs if not j.with_suffix(".cer.json").exists()]
        else:
            filtered = []
            for w, j in pairs:
                ev = j.with_suffix(".cer.json")
                if not ev.exists():
                    filtered.append((w, j))
                    continue
                try:
                    meta = json.loads(ev.read_text(encoding="utf-8"))
                    if meta.get("stage") == "complete" or meta.get("llm_cer") is not None:
                        continue
                except (json.JSONDecodeError, OSError):
                    pass
                filtered.append((w, j))
            pairs = filtered
        print(f"Skip-existing: {before - len(pairs)} done, {len(pairs)} remaining", flush=True)

    if args.sample_size and args.sample_size < len(pairs):
        pairs = random.sample(pairs, args.sample_size)
        print(f"Sampled {len(pairs)} clones (seed={args.seed})", flush=True)

    if not pairs:
        print("No pairs to evaluate.", flush=True)
        return

    paths = apply_shard_paths(eval_paths(args.out_dir), shard_suffix)
    if not args.skip_existing and paths["details_jsonl"].exists():
        paths["details_jsonl"].unlink()

    evaluated_at = datetime.now().isoformat()
    detailed: list = []
    manual_acc = CerAccumulator()
    llm_acc = CerAccumulator()
    model_name = os.environ.get("ITN_LLM_MODEL", "qwen3.6-27b")
    use_llm_cache = not args.refresh_llm_cache
    batch_size = args.batch_size

    asr_results = (
        {} if args.refresh_asr_cache else load_json(paths["asr_cache"]) if paths["asr_cache"].exists() else {}
    )
    if asr_results:
        print(f"Loaded ASR cache: {len(asr_results)} entries", flush=True)
    elif args.refresh_asr_cache:
        print("Refresh ASR cache: existing ASR cache will be ignored", flush=True)

    llm_cache: dict = {}
    if not args.skip_llm:
        if not _LLM_ENDPOINTS:
            _init_llm_endpoints()
        if not _LLM_ENDPOINTS:
            print("ERROR: No LLM endpoints configured. Set ITN_LLM_BASE_URLS in .env or use --skip-llm", flush=True)
            sys.exit(1)
        if use_llm_cache and paths["llm_cache"].exists():
            llm_cache = load_json(paths["llm_cache"])
            print(f"Loaded LLM ITN cache: {len(llm_cache)} entries", flush=True)

    write_lock = threading.Lock()
    llm_cache_lock = threading.Lock()
    asr_cache_lock = threading.Lock()
    producer_error: list = []

    def flush_progress(stage: str):
        prog = {
            "out_dir": str(args.out_dir),
            "stage": stage,
            "items_done": len(detailed),
            "items_total": len(pairs),
            "batch_size": batch_size,
            "asr_gpus": asr_gpu_ids,
            "audio_workers": args.audio_workers,
            "prefetch_batches": args.prefetch_batches,
            "llm_concurrency": args.llm_concurrency,
            "manual": manual_acc.to_dict(),
            "evaluated_at": datetime.now().isoformat(),
        }
        if llm_acc.count:
            prog["llm"] = llm_acc.to_dict()
        write_json(paths["summary_progress"], prog)

    def log_line(msg: str):
        tqdm.write(msg, file=sys.stderr)

    def save_item(item: dict, llm: bool = False):
        write_eval_json(Path(item["json"]), build_eval_record(item, evaluated_at, llm=llm, model_name=model_name))
        row = {
            "wav": item["wav"],
            "manual_cer": item["manual_cer"],
            "dataset": item["dataset"],
            "speaker_id": item["speaker_id"],
            "asr_language": item.get("asr_language", "unknown"),
            "stage": "complete" if llm else "manual",
        }
        if llm:
            row["llm_cer"] = item.get("llm_cer", 0)
        append_jsonl(paths["details_jsonl"], row)

    def build_batch_items(batch_pairs):
        items = []
        for wav_path, json_path in batch_pairs:
            language = infer_asr_language_for_json(json_path)
            with asr_cache_lock:
                hypo_raw = get_cached_asr_text(asr_results, wav_path, language) or ""
            item = build_eval_item(wav_path, json_path, hypo_raw)
            items.append(item)
            manual_acc.add(item["substitutions"], item["insertions"], item["deletions"], item["chars"])
        return items

    def save_manual_batch(batch_pairs, stage: str = "asr"):
        with write_lock:
            batch_items = build_batch_items(batch_pairs)
            for item in batch_items:
                detailed.append(item)
                save_item(item, llm=False)
            if len(detailed) % (CACHE_FLUSH_INTERVAL * batch_size) < len(batch_pairs):
                flush_progress(stage)

    itn_batch_counter = [0]

    def process_itn_batch(batch_items: list, worker_id: int = 0):
        with llm_cache_lock:
            pending = [it for it in batch_items if not (use_llm_cache and it["wav"] in llm_cache)]
        n_pending = len(pending)
        log_line(f"  [ITN start w{worker_id}] batch={len(batch_items)} llm_call={n_pending}")
        if pending:
            new_entries = llm_itn_batch_fetch(pending)
            with llm_cache_lock:
                llm_cache.update(new_entries)
                itn_batch_counter[0] += 1
                if itn_batch_counter[0] % CACHE_FLUSH_INTERVAL == 0:
                    flush_json_cache(paths["llm_cache"], llm_cache)

        with write_lock:
            for item in batch_items:
                with llm_cache_lock:
                    llm = llm_cache.get(item["wav"], {})
                apply_llm_to_item(item, llm, model_name)
                llm_acc.add(item.get("llm_sub", 0), item.get("llm_ins", 0), item.get("llm_del", 0), item["chars"])
                detailed.append(item)
                save_item(item, llm=True)
            if len(detailed) % (CACHE_FLUSH_INTERVAL * batch_size) < batch_size:
                flush_progress("itn")
        log_line(f"  [ITN done w{worker_id}] +{len(batch_items)} total={len(detailed)}")

    num_batches = (len(pairs) + batch_size - 1) // batch_size
    print(
        f"\nEvaluating {len(pairs)} clones | batch={batch_size} | "
        f"asr_gpus={asr_gpu_ids} | llm={'skip' if args.skip_llm else f'{args.llm_concurrency} workers'}\n",
        flush=True,
    )

    if args.skip_llm:
        asr_models = {}
        need_asr = not args.skip_asr and any(not has_cached_asr_text(asr_results, w, j) for w, j in pairs)
        audio_pool = None
        if need_asr and args.audio_workers > 0:
            audio_pool = ProcessPoolExecutor(max_workers=args.audio_workers, mp_context=mp.get_context("spawn"))
            print(
                f"Audio prefetch enabled: workers={args.audio_workers}, prefetch_batches={args.prefetch_batches}",
                flush=True,
            )

        try:
            if need_asr:
                for gpu_id in asr_gpu_ids:
                    asr_models[gpu_id] = load_asr_model(batch_size, gpu_id=gpu_id)

            if asr_models:
                from concurrent.futures import ThreadPoolExecutor as _TPE

                def _asr_only(gpu_id, batch_indices, pbar):
                    pending = deque()
                    batch_iter = iter(batch_indices)
                    prefetch_depth = max(1, args.prefetch_batches)

                    def _enqueue_next_batch():
                        bi = next(batch_iter)
                        batch = pairs[bi * batch_size : (bi + 1) * batch_size]
                        need = [(w, j) for w, j in batch if not has_cached_asr_text(asr_results, w, j)]
                        prefetched = submit_audio_prefetch(need, audio_pool=audio_pool) if need else []
                        pending.append((batch, need, prefetched))

                    for _ in range(prefetch_depth):
                        try:
                            _enqueue_next_batch()
                        except StopIteration:
                            break

                    while pending:
                        batch, need, prefetched = pending.popleft()
                        try:
                            _enqueue_next_batch()
                        except StopIteration:
                            pass
                        if need:
                            new_asr_results = {}
                            transcribe_asr_batch(
                                asr_models[gpu_id],
                                need,
                                new_asr_results,
                                audio_pool=audio_pool,
                                prefetched=prefetched,
                            )
                            with asr_cache_lock:
                                asr_results.update(new_asr_results)
                        save_manual_batch(batch, stage="asr")
                        pbar.update(1)

                gpu_batches = {g: [] for g in asr_gpu_ids}
                for bi in range(num_batches):
                    gpu_batches[asr_gpu_ids[bi % len(asr_gpu_ids)]].append(bi)

                pbar = tqdm(total=num_batches, desc=f"ASR+Manual ({len(asr_gpu_ids)} GPUs)")
                with _TPE(max_workers=len(asr_gpu_ids)) as pool:
                    futs = [pool.submit(_asr_only, g, idxs, pbar) for g, idxs in gpu_batches.items()]
                    for f in futs:
                        f.result()
                pbar.close()
                with asr_cache_lock:
                    flush_json_cache(paths["asr_cache"], asr_results)
                flush_progress("asr")
            else:
                for bi in tqdm(range(num_batches), desc="Manual ITN"):
                    batch = pairs[bi * batch_size : (bi + 1) * batch_size]
                    save_manual_batch(batch, stage="asr")
                flush_progress("asr")
        finally:
            if audio_pool is not None:
                audio_pool.shutdown(wait=True, cancel_futures=True)
            for asr in asr_models.values():
                del asr
            torch.cuda.empty_cache()
    else:
        batch_queue: queue.Queue = queue.Queue(maxsize=args.llm_concurrency * 6)

        def itn_worker(worker_id: int):
            while True:
                batch_items = batch_queue.get()
                try:
                    if batch_items is None:
                        break
                    process_itn_batch(batch_items, worker_id)
                except Exception as e:
                    producer_error.append(e)
                    tb = traceback.format_exc()
                    print(f"[itn-w{worker_id}] ERROR: {e}\n{tb}", file=sys.stderr, flush=True)
                finally:
                    batch_queue.task_done()

        workers = [threading.Thread(target=itn_worker, args=(i,), daemon=False) for i in range(args.llm_concurrency)]
        for t in workers:
            t.start()

        asr_models = {}
        if not args.skip_asr and any(not has_cached_asr_text(asr_results, w, j) for w, j in pairs):
            for gpu_id in asr_gpu_ids:
                asr_models[gpu_id] = load_asr_model(batch_size, gpu_id=gpu_id)

        num_asr_gpus = len(asr_gpu_ids)
        print(
            f"\n[Pipeline] {num_asr_gpus} ASR producers → queue → {args.llm_concurrency} ITN workers...",
            flush=True,
        )

        def _asr_gpu_worker(gpu_id: int, batch_indices: list[int], pbar):
            for bi in batch_indices:
                batch = pairs[bi * batch_size : (bi + 1) * batch_size]
                if not args.skip_asr and gpu_id in asr_models:
                    need = [(w, j) for w, j in batch if not has_cached_asr_text(asr_results, w, j)]
                    if need:
                        new_asr_results = {}
                        transcribe_asr_batch(asr_models[gpu_id], need, new_asr_results)
                        with asr_cache_lock:
                            asr_results.update(new_asr_results)
                with write_lock:
                    batch_items = build_batch_items(batch)
                batch_queue.put(batch_items)
                pbar.update(1)

        def asr_producer():
            try:
                gpu_batches: dict[int, list[int]] = {g: [] for g in asr_gpu_ids}
                for bi in range(num_batches):
                    gpu_batches[asr_gpu_ids[bi % num_asr_gpus]].append(bi)

                pbar = tqdm(total=num_batches, desc="ASR → queue")
                if num_asr_gpus > 1 and not args.skip_asr and asr_models:
                    from concurrent.futures import ThreadPoolExecutor as _TPE

                    with _TPE(max_workers=num_asr_gpus) as pool:
                        futs = [pool.submit(_asr_gpu_worker, g, idxs, pbar) for g, idxs in gpu_batches.items()]
                        for f in futs:
                            f.result()
                else:
                    _asr_gpu_worker(asr_gpu_ids[0], list(range(num_batches)), pbar)
                pbar.close()

                with asr_cache_lock:
                    flush_json_cache(paths["asr_cache"], asr_results)
            except Exception as e:
                producer_error.append(e)
                print(f"[ASR producer] error: {e}\n{traceback.format_exc()}", file=sys.stderr, flush=True)
            finally:
                for _ in range(args.llm_concurrency):
                    batch_queue.put(None)

        producer = threading.Thread(target=asr_producer, daemon=False)
        producer.start()
        producer.join()
        batch_queue.join()
        for t in workers:
            t.join(timeout=30)

        if llm_cache:
            flush_json_cache(paths["llm_cache"], llm_cache)
        flush_progress("complete")

        for asr in asr_models.values():
            del asr
        torch.cuda.empty_cache()

        if producer_error:
            print(f"\n[ERROR] {len(producer_error)} errors during pipeline:", file=sys.stderr, flush=True)
            for i, e in enumerate(producer_error):
                print(f"  [{i}] {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            raise producer_error[0]

    manual_summary = summarize_cer(detailed, "ref_manual", "hypo_manual")
    llm_summary = None
    if not args.skip_llm:
        llm_summary = summarize_cer(detailed, "ref_llm", "hypo_llm")

    by_dataset = defaultdict(list)
    for d in detailed:
        by_dataset[d["dataset"]].append(d)

    summary = {
        "out_dir": str(args.out_dir),
        "sample_size": len(detailed),
        "batch_size": batch_size,
        "asr_gpus": asr_gpu_ids,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "seed": args.seed,
        "evaluated_at": evaluated_at,
        "manual": manual_summary,
        "manual_by_dataset": {
            ds: summarize_cer(items, "ref_manual", "hypo_manual") for ds, items in sorted(by_dataset.items())
        },
    }
    if llm_summary:
        summary["llm"] = llm_summary
        summary["llm_model"] = model_name
        summary["llm_concurrency"] = args.llm_concurrency
        summary["llm_by_dataset"] = {
            ds: summarize_cer(items, "ref_llm", "hypo_llm") for ds, items in sorted(by_dataset.items())
        }

    write_json(paths["summary"], summary)

    print(f"\n{'=' * 60}", flush=True)
    sd = manual_summary
    print(
        f"Manual  Weighted CER: {sd['weighted_cer']:.2f}% | Avg: {sd['avg_cer']:.2f}% | Median: {sd['median_cer']:.2f}%",
        flush=True,
    )
    if llm_summary:
        ld = llm_summary
        print(
            f"LLM    Weighted CER: {ld['weighted_cer']:.2f}% | Avg: {ld['avg_cer']:.2f}% | Median: {ld['median_cer']:.2f}%",
            flush=True,
        )
        print(f"Delta: {ld['weighted_cer'] - sd['weighted_cer']:+.2f}%", flush=True)
    print(f"\nSummary: {paths['summary']}", flush=True)
    print(f"Details: {paths['details_jsonl']}", flush=True)


if __name__ == "__main__":
    main()
