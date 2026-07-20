#!/usr/bin/env python3
# Copyright (c) 2025 Boson AI
#
# Assign hand-authored Higgs-v3 clone scripts to sampled speakers.
# Adult: round-robin pure_cn / pure_en / mixed
# Child: round-robin pure_cn / pure_en (no mixed)
"""
Usage:
  python 02_assign_texts.py \
    --speakers-json ./workdir/selected_speakers.json \
    --corpus-json ./clone_text_corpus.json \
    --output-dir ./workdir \
    --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from typing import Any


ADULT_LANG_ORDER = ("pure_cn", "pure_en", "mixed")
CHILD_LANG_ORDER = ("pure_cn", "pure_en")


def _pools_by_lang(
    items: list[dict[str, Any]], rng: random.Random, lang_order: tuple[str, ...]
) -> dict[str, list[dict[str, Any]]]:
    pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        lang = item.get("lang") or "pure_cn"
        if lang not in lang_order:
            continue
        pools[lang].append(item)
    for lang in lang_order:
        rng.shuffle(pools[lang])
    return pools


def _next_item(
    pools: dict[str, list[dict[str, Any]]],
    cursor: dict[str, int],
    turn: int,
    lang_order: tuple[str, ...],
) -> dict[str, Any]:
    for offset in range(len(lang_order)):
        lang = lang_order[(turn + offset) % len(lang_order)]
        bucket = pools.get(lang) or []
        if not bucket:
            continue
        idx = cursor[lang] % len(bucket)
        cursor[lang] += 1
        return bucket[idx]
    for bucket in pools.values():
        if bucket:
            return bucket[0]
    raise RuntimeError("empty corpus pools")


def main() -> None:
    ap = argparse.ArgumentParser(description="Assign hand-authored v3 clone texts to speakers")
    ap.add_argument("--speakers-json", required=True)
    ap.add_argument("--corpus-json", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    with open(args.speakers_json, encoding="utf-8") as f:
        speakers_doc = json.load(f)
    speakers = speakers_doc.get("speakers") or []
    if not speakers:
        raise SystemExit("[02] no speakers")

    with open(args.corpus_json, encoding="utf-8") as f:
        corpus = json.load(f)
    adult_pool = list(corpus.get("adult") or [])
    child_pool = list(corpus.get("child") or [])
    if len(adult_pool) < 1 or len(child_pool) < 1:
        raise SystemExit("[02] corpus missing adult/child texts")

    rng = random.Random(args.seed)
    adult_by_lang = _pools_by_lang(adult_pool, rng, ADULT_LANG_ORDER)
    child_by_lang = _pools_by_lang(child_pool, rng, CHILD_LANG_ORDER)
    adult_cursor = {k: 0 for k in ADULT_LANG_ORDER}
    child_cursor = {k: 0 for k in CHILD_LANG_ORDER}
    adult_turn = 0
    child_turn = 0

    scripts: list[dict[str, Any]] = []
    for spk in speakers:
        uid = spk.get("uid") or f"{spk['dataset']}/{spk['speaker_id']}"
        gender = spk.get("gender_consensus") or "male"
        if gender == "child":
            item = _next_item(child_by_lang, child_cursor, child_turn, CHILD_LANG_ORDER)
            child_turn += 1
            audience = "child"
        else:
            item = _next_item(adult_by_lang, adult_cursor, adult_turn, ADULT_LANG_ORDER)
            adult_turn += 1
            audience = "adult"

        scripts.append(
            {
                "uid": uid,
                "dataset": spk["dataset"],
                "speaker_id": spk["speaker_id"],
                "gender_consensus": gender,
                "audience": audience,
                "corpus_id": item.get("id"),
                "lang": item.get("lang"),
                "text": item["text"],
                "text_tagged": item["text"],
                "clean_text": item.get("clean_text"),
                "sentences": item["sentences"],
                "clean_sentences": item.get("clean_sentences"),
                "num_sentences": item.get("num_sentences"),
                "pause_secs": item.get("pause_secs") or [],
                "pause_sec_min": item.get("pause_sec_min"),
                "pause_sec_max": item.get("pause_sec_max"),
                "pause_sec_mean": item.get("pause_sec_mean"),
                "pause_postprocess": item.get("pause_postprocess"),
                "est_speech_sec": item.get("est_speech_sec"),
                "est_total_sec": item.get("est_total_sec"),
                "emotion": item.get("emotion"),
                "prosody": item.get("prosody"),
            }
        )

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "clone_scripts.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "seed": args.seed,
                "corpus_json": os.path.abspath(args.corpus_json),
                "n_adult_corpus": len(adult_pool),
                "n_child_corpus": len(child_pool),
                "lang_counts": dict(Counter(s.get("lang") for s in scripts)),
                "scripts": scripts,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    est = [s["est_total_sec"] for s in scripts if s.get("est_total_sec") is not None]
    by_aud = Counter(s["audience"] for s in scripts)
    by_lang = Counter(s.get("lang") for s in scripts)
    print(
        f"[02] wrote {out_path} n={len(scripts)} audience={dict(by_aud)} lang={dict(by_lang)} "
        f"est_total mean={sum(est) / len(est):.1f} min={min(est):.1f} max={max(est):.1f}",
        flush=True,
    )
    for lang in ("pure_cn", "pure_en", "mixed"):
        sample = next((s for s in scripts if s.get("lang") == lang), None)
        if sample:
            print(f"[02] sample {lang}: {sample['text'][:140]}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
