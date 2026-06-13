#!/usr/bin/env python3
"""CER evaluation for Higgs Audio v3 TTS clone audio.

Evaluates Character Error Rate (CER) on cloned audio files using Qwen3-ASR.
Adapted from OmniVoice batch_generate_text_and_clone/eval_cer/eval_cloned.py.

Key differences from OmniVoice:
- Higgs clones: `clone_NNNN.wav` / `clone_NNNN.json` (OmniVoice: `text_NNN.wav/json`)
- Higgs metadata reference text: `clean_text` field (OmniVoice: `gen_text`)
- Higgs tags: `<|emotion:X|>`, `<|prosody:X|>`, `<|sfx:X|>`, `<|style:X|>`
- Higgs audio: 24 kHz → resampled to 16 kHz for ASR

Workflow:
  1. Scan clone output for `clone_*.wav` + `clone_*.json` pairs
  2. ASR: Qwen3-ASR transcribes wav (batch_size=16, resample 24→16 kHz)
  3. Manual ITN: strip tags → normalize numbers → strip punctuation → lowercase
  4. LLM ITN (optional): LLM cross-aligns ref/hypo pairs
  5. CER via jiwer.process_characters

Usage:
    conda activate qwen3-asr
    cd eval_higgs_audio/eval_cer

    python eval_cer.py --out-dir /path/to/audio_higgs_audio_v3_tts_clone
    python eval_cer.py --sample-size 500 --skip-llm
    bash run_eval_cer.sh
"""

import argparse
import json
import os
import re
import string
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf
import torch
import torchaudio
from tqdm import tqdm

EVAL_DIR = Path(__file__).resolve().parent
PARENT_DIR = EVAL_DIR.parent
sys.path.insert(0, str(PARENT_DIR))

from eval_common import CerAccumulator, append_jsonl, list_clone_items, write_json  # noqa: E402

# ── Constants ────────────────────────────────────────────────────────
QWEN3_ASR_LOCAL = "/root/.cache/huggingface/hub/Qwen3-ASR-1.7B-local"
ENV_FILE = EVAL_DIR / ".env"
DEFAULT_OUT_DIR = Path(
    os.environ.get(
        "HIGGS_CLONE_ROOT",
        "/root/group-shared/voiceprint/data/speech/speaker_diarization/merged_datasets_20250610_vad_segments_mtfaa_enhanced_extend_kid_withclone_addlibrilight_1130/audio_higgs_audio_v3_tts_clone",
    )
)
DEFAULT_BATCH_SIZE = 16

# Higgs tag format: <|emotion:joy|>, <|prosody:speed_slow|>, <|sfx:laughter|>, <|style:whispering|>
HIGGS_TAG_RE = re.compile(r"<\|[^|>]+\|[^>]*>")

PUNCTUATION = set(
    string.punctuation
    + "，。！？；：\"\"''（）【】《》、·～｀＠＃￥％＆＊—＋｜＜＞？／"
    + "\u2018\u2019\u201c\u201d\u2026"
)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def extract_speech(wav_path: Path, target_sr: int = 16000):
    """Load and resample wav to target SR.

    Higgs clone audio is 24 kHz; ASR expects 16 kHz.
    """
    wav, sr = sf.read(str(wav_path), dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != target_sr:
        wav = torchaudio.functional.resample(torch.from_numpy(wav), orig_freq=sr, new_freq=target_sr).numpy()
    return wav


def strip_higgs_tags(text: str) -> str:
    """Remove Higgs-style inline tags: <|emotion:X|>, <|prosody:X|>, <|sfx:X|>, <|style:X|>."""
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


# ── Number Normalization (same as OmniVoice) ──────────────────────────

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


def normalize_units(text: str) -> str:
    text = re.sub(r"\b(?:kilometers?|kms?)\s*(?:per|/)\s*(?:hour|hr|h)\b", "kmh", text, flags=re.IGNORECASE)
    text = re.sub(r"\bkm/h\b", "kmh", text, flags=re.IGNORECASE)
    text = re.sub(r"\bcentimeters?\b", "cm", text, flags=re.IGNORECASE)
    text = re.sub(r"\bcentimetres?\b", "cm", text, flags=re.IGNORECASE)
    text = re.sub(r"\bmillimeters?\b", "mm", text, flags=re.IGNORECASE)
    text = re.sub(r"(\d)\s+kmh\b", r"\1kmh", text, flags=re.IGNORECASE)
    return text


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


def normalize_numbers(text: str) -> str:
    text = normalize_percent(text)
    text = normalize_fractions(text)
    text = normalize_chinese_money(text)
    text = normalize_units(text)
    text = normalize_english_numbers(text)
    text = normalize_chinese_numbers(text)
    text = normalize_chinese_digit_runs(text)
    text = normalize_units(text)
    return text


def manual_itn(text: str) -> str:
    """Stage-1 Manual ITN: strip Higgs tags → normalize numbers → strip punctuation → lowercase → whitespace."""
    text = strip_higgs_tags(text)
    text = normalize_numbers(text)
    text = strip_punctuation(text)
    text = text.lower()
    return normalize_spaces(text)


# ── CER Calculation ───────────────────────────────────────────────────


def calc_cer(ref: str, hyp: str):
    from jiwer import process_characters

    try:
        cm = process_characters(ref, hyp)
        return cm.cer, cm.substitutions, cm.insertions, cm.deletions, len(ref)
    except Exception:
        return 1.0, 0, 0, 0, len(ref)


# ── ASR Model ─────────────────────────────────────────────────────────


def load_asr_model(batch_size: int = 16, gpu_id: int = 0):
    print("Loading Qwen3-ASR model...", flush=True)
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


def transcribe_asr_batch(asr, batch_pairs: list, results: dict) -> None:
    audio_inputs = []
    valid_paths = []
    for wav_path, _ in batch_pairs:
        try:
            speech = extract_speech(wav_path, target_sr=16000)
            audio_inputs.append((speech, 16000))
            valid_paths.append(wav_path)
        except Exception as e:
            print(f"Error loading {wav_path}: {e}", flush=True)
            results[str(wav_path)] = ""
    if not audio_inputs:
        return
    try:
        hypos = asr.transcribe(audio=audio_inputs, language="Chinese", return_time_stamps=False)
        for wav_path, h in zip(valid_paths, hypos):
            results[str(wav_path)] = h.text
    except Exception as e:
        print(f"ASR batch error: {e}", flush=True)
        for wav_path in valid_paths:
            results[str(wav_path)] = ""


# ── Build Eval Items ──────────────────────────────────────────────────


def build_eval_item(wav_path: Path, json_path: Path, hypo_raw: str) -> dict:
    """Build one CER eval record from Higgs clone sidecar + ASR hypothesis."""
    meta = load_json(json_path)
    truth_raw = meta.get("clean_text", "") or meta.get("text", "")
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
    }


# ── Output Helpers ────────────────────────────────────────────────────


def build_eval_record(item: dict, evaluated_at: str) -> dict:
    return {
        "wav_path": item["wav"],
        "gen_text": item["ref_start"],
        "asr_hypo": item["hypo_start"],
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
    }


def write_eval_json(json_path: Path, record: dict):
    eval_path = json_path.with_suffix(".eval.json")
    eval_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def eval_paths(out_dir: Path) -> dict:
    return {
        "asr_cache": out_dir / "eval_higgs_asr_cache.json",
        "summary": out_dir / "eval_higgs_cer_summary.json",
        "summary_progress": out_dir / "eval_higgs_cer_progress.json",
        "details_jsonl": out_dir / "eval_higgs_cer_details.jsonl",
        "details_txt": out_dir / "eval_higgs_cer_details.txt",
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


# ── Main Pipeline ─────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Higgs Audio clone output root directory"
    )
    parser.add_argument(
        "--sample-size", type=int, default=None, help="Randomly sample N clones (default: evaluate all)"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-asr", action="store_true", help="Use cached ASR results")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--skip-existing", action="store_true", help="Skip items with existing .eval.json")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    import random

    random.seed(args.seed)

    # Scan for clone items
    pairs = list_clone_items(args.out_dir, label="cer-scan")
    if not pairs:
        print("No clone audio found.", flush=True)
        return

    if args.skip_existing:
        before = len(pairs)
        pairs = [(w, j) for w, j in pairs if not j.with_suffix(".eval.json").exists()]
        print(f"Skip-existing: {before - len(pairs)} done, {len(pairs)} remaining", flush=True)

    if args.sample_size and args.sample_size < len(pairs):
        pairs = random.sample(pairs, args.sample_size)
        print(f"Sampled {len(pairs)} clones (seed={args.seed})", flush=True)

    if not pairs:
        print("No pairs to evaluate.", flush=True)
        return

    paths = eval_paths(args.out_dir)
    if not args.skip_existing and paths["details_jsonl"].exists():
        paths["details_jsonl"].unlink()

    evaluated_at = datetime.now().isoformat()
    detailed: list = []
    accumulator = CerAccumulator()

    # Load/create ASR cache
    asr_results = load_json(paths["asr_cache"]) if paths["asr_cache"].exists() else {}
    if asr_results:
        print(f"Loaded ASR cache: {len(asr_results)} entries", flush=True)

    # ASR transcribe
    asr = None
    if not args.skip_asr and any(not asr_results.get(str(w)) for w, _ in pairs):
        asr = load_asr_model(args.batch_size, gpu_id=args.gpu)

    num_batches = (len(pairs) + args.batch_size - 1) // args.batch_size
    print(f"\nEvaluating {len(pairs)} clones | batch={args.batch_size} | {num_batches} batches\n", flush=True)

    for bi in tqdm(range(num_batches), desc="ASR + Manual ITN"):
        batch = pairs[bi * args.batch_size : (bi + 1) * args.batch_size]
        if not args.skip_asr:
            need = [(w, j) for w, j in batch if not asr_results.get(str(w))]
            if need:
                transcribe_asr_batch(asr, need, asr_results)
                paths["asr_cache"].parent.mkdir(parents=True, exist_ok=True)
                paths["asr_cache"].write_text(
                    json.dumps(asr_results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                )

        for wav_path, json_path in batch:
            hypo_raw = asr_results.get(str(wav_path), "")
            item = build_eval_item(wav_path, json_path, hypo_raw)
            accumulator.add(item["substitutions"], item["insertions"], item["deletions"], item["chars"])
            detailed.append(item)
            write_eval_json(json_path, build_eval_record(item, evaluated_at))

        # Append to JSONL
        for item in detailed[-len(batch) :]:
            row = {
                "wav": item["wav"],
                "manual_cer": item["manual_cer"],
                "dataset": item["dataset"],
                "speaker_id": item["speaker_id"],
            }
            append_jsonl(paths["details_jsonl"], row)

    if asr is not None:
        del asr
        torch.cuda.empty_cache()

    # Final summary
    summary_data = summarize_cer(detailed)
    summary = {
        "out_dir": str(args.out_dir),
        "sample_size": len(detailed),
        "batch_size": args.batch_size,
        "seed": args.seed,
        "evaluated_at": evaluated_at,
        "summary": summary_data,
        "by_dataset": {},
    }

    # Per-dataset breakdown
    from collections import defaultdict

    by_dataset = defaultdict(list)
    for d in detailed:
        by_dataset[d["dataset"]].append(d)
    for ds, items in sorted(by_dataset.items()):
        summary["by_dataset"][ds] = summarize_cer(items)

    write_json(paths["summary"], summary)

    # Print results
    print(f"\n{'=' * 60}", flush=True)
    sd = summary_data
    print(
        f"Weighted CER: {sd['weighted_cer']:.2f}% | Avg: {sd['avg_cer']:.2f}% | Median: {sd['median_cer']:.2f}%",
        flush=True,
    )
    print(
        f"Min: {sd['min_cer']:.2f}% | Max: {sd['max_cer']:.2f}% | P10: {sd['p10_cer']:.2f}% | P90: {sd['p90_cer']:.2f}%",
        flush=True,
    )
    print(
        f"Chars: {sd['total_chars']} | Ins: {sd['total_ins']} Del: {sd['total_del']} Sub: {sd['total_sub']}",
        flush=True,
    )
    print(f"\nSummary: {paths['summary']}", flush=True)
    print(f"Details: {paths['details_jsonl']}", flush=True)


if __name__ == "__main__":
    main()
