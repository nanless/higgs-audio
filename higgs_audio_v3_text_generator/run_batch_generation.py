#!/usr/bin/env python3
"""
Batch text generation pipeline for Higgs Audio v3 TTS via vLLM API.
Uses ThreadPoolExecutor for concurrent requests + diversity compact prompt.
"""

import argparse
import json
import os
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from higgs_text_gen.checkpoint import load_checkpoint, save_checkpoint
from higgs_text_gen.compact_prompt import build_compact_prompt
from higgs_text_gen.config import GenConfig
from higgs_text_gen.dedup import build_duplicate_index, deduplicate, semantic_deduplicate
from higgs_text_gen.llm_client import call_llm
from higgs_text_gen.output import print_statistics, save_jsonl
from higgs_text_gen.quality_filter import quality_filter
from higgs_text_gen.task_generator import generate_task_list
from higgs_text_gen.text_clean import attach_clean_text_batch


HIGGS_TAG_CLEAN_RE = re.compile(r"<\|(emotion|style|sfx|prosody):[a-z_]+\|>")
HIGGS_TAG_EXTRACT_RE = re.compile(r"<\|(emotion|style|sfx|prosody):([a-z_]+)\|>")

_TAG_POOL_STYLE = {"singing", "shouting", "whispering"}
_TAG_POOL_SFX = {"laughter", "sigh", "cough", "crying", "screaming", "humming", "sniff", "sneeze", "burping"}
_TAG_POOL_PROSODY = {
    "speed_very_slow",
    "speed_slow",
    "speed_fast",
    "speed_very_fast",
    "pitch_low",
    "pitch_high",
    "pause",
    "long_pause",
    "expressive_high",
    "expressive_low",
}


def _normalize_for_dedup(text):
    t = HIGGS_TAG_CLEAN_RE.sub("", text)
    t = re.sub(r"\d+", "<NUM>", t)
    t = re.sub(r"[^\w\s]", "", t.lower().strip())
    return re.sub(r"\s+", " ", t).strip()


def build_suppression_hint(texts, window_size=500):
    if len(texts) < 40:
        return ""
    recent = texts[-window_size:]
    char_counter = Counter()
    for item in recent:
        text = item.get("clean_text", item.get("text", ""))
        if len(text) >= 4:
            char_counter[text[:4]] += 1
    overused = [k for k, v in char_counter.most_common(10) if v >= max(4, len(recent) // 80)]
    if not overused:
        return ""
    hints = ["避免以下重复开头:"]
    for i, o in enumerate(overused[:5]):
        hints.append(f"  避免'{o}'开头")
    return "\n".join(hints)


def build_tag_diversity_hint(texts, window_size=300, global_window_size=5000):
    """Track tag usage across two windows and hint LLM to use under-utilized tag categories.

    - Short window (default 300): detects if current generation has stopped using certain tags.
    - Global window (default 5000): detects long-term imbalance in tag coverage.
    """
    if len(texts) < 50:
        return ""

    hints = []

    def _count_tags(slice_end, wsize):
        if len(texts) < wsize:
            start_idx = 0
        else:
            start_idx = max(0, slice_end - wsize)
        window = texts[start_idx:slice_end]

        sfx_counter = Counter()
        style_counter = Counter()
        prosody_counter = Counter()
        for item in window:
            text = item.get("text", "")
            for m in HIGGS_TAG_EXTRACT_RE.finditer(text):
                cat, name = m.group(1), m.group(2)
                if cat == "sfx":
                    sfx_counter[name] += 1
                elif cat == "style":
                    style_counter[name] += 1
                elif cat == "prosody":
                    prosody_counter[name] += 1
        return sfx_counter, style_counter, prosody_counter

    # Recent window check
    recent_sfx, recent_style, recent_prosody = _count_tags(len(texts), window_size)

    used_sfx = set(recent_sfx.keys())
    missing_sfx = _TAG_POOL_SFX - used_sfx
    if len(missing_sfx) >= 5:
        hints.append(
            f"近{window_size}条中SFX音效标签只用过{len(used_sfx)}种,严重缺:{','.join(sorted(missing_sfx)[:4])}等"
        )

    used_style = set(recent_style.keys())
    missing_style = _TAG_POOL_STYLE - used_style
    if len(missing_style) >= 2:
        hints.append(f"近{window_size}条中风格标签只用过{len(used_style)}种,请多用:{','.join(sorted(missing_style))}")

    used_prosody = set(recent_prosody.keys())
    missing_prosody = _TAG_POOL_PROSODY - used_prosody
    if len(missing_prosody) >= 6:
        hints.append(
            f"近{window_size}条中韵律标签只用过{len(used_prosody)}种,"
            f"请多用speed_fast/slow/pitch_high/low/long_pause/expressive等: {','.join(sorted(missing_prosody)[:6])}"
        )

    # Global imbalance check
    if len(texts) >= global_window_size:
        global_sfx, global_style, global_prosody = _count_tags(len(texts), global_window_size)

        for label, counter, pool, threshold in [
            ("SFX", global_sfx, _TAG_POOL_SFX, 5),
            ("风格", global_style, _TAG_POOL_STYLE, 2),
            ("韵律", global_prosody, _TAG_POOL_PROSODY, 6),
        ]:
            total_uses = sum(counter.values())
            zero_used = [t for t in pool if counter.get(t, 0) == 0]
            if total_uses > 50 and len(zero_used) >= threshold:
                hints.append(
                    f"全局{global_window_size}条内{label}标签'{','.join(zero_used[:4])}'完全没用过,"
                    f"当前已生成{len(texts)}条,请速补这些标签类型"
                )

    if not hints:
        return ""
    return "=== 标签多样性压制 ===\n" + "\n".join(hints) + "\n请看上方标签参考,生成包含缺失标签类的文本。"


def worker_fn(task, config):
    import random

    task_id = task.get("task_id")
    scenario_key = task["scenario_key"]
    emotion = task.get("emotion", "contentment")

    prompt = build_compact_prompt(
        scenario_key=scenario_key,
        subscene=task.get("subscene", ""),
        length_key=task.get("length_key", "medium"),
        lang_key=task.get("lang_key", "pure_cn"),
        emotion=emotion,
        batch_size=config.batch_size,
        suppression_hint=task.get("suppression_hint", ""),
        task_id=task_id or 0,
    )

    seed = hash(f"{task_id}|{scenario_key}|{emotion}") & 0xFFFFFFFF
    rng = random.Random(seed)
    temp = min(1.0, max(0.65, config.temperature + rng.uniform(-0.15, 0.15)))

    results = call_llm(
        prompt=prompt,
        model=config.model,
        api_key=config.api_key,
        base_url=config.base_url,
        max_retries=config.max_retries,
        retry_base_delay=config.retry_base_delay,
        max_tokens=config.max_tokens,
        temperature=temp,
    )

    if not results:
        return None

    for item in results:
        item["task_id"] = task_id
        item["scenario"] = item.get("scenario", scenario_key)
        item["subscene"] = item.get("subscene", task.get("subscene", ""))
        item["emotion"] = item.get("emotion", emotion)
        item["length_type"] = item.get("length_type", task.get("length_key", "medium"))
        item["lang_type"] = item.get("lang_type", task.get("lang_key", "pure_cn"))
        lt = item.get("lang_type", "")
        item["language"] = item.get("language", "zh" if "cn" in lt else "en")

    if config.generate_clean_text:
        attach_clean_text_batch(results)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--output", type=str, default="batch_output/generated_texts.jsonl")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--checkpoint", type=str, default="batch_output/.checkpoint.jsonl")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-postprocess", action="store_true", help="Skip final dedup/quality for speed")
    args = parser.parse_args()

    config = GenConfig(
        total_target=args.total,
        batch_size=args.batch_size,
        max_workers=args.workers,
        model=args.model or os.environ.get("LLM_MODEL", "qwen3.6-27b"),
        base_url=args.base_url or os.environ.get("LLM_BASE_URL", "http://localhost:8000"),
        api_key=args.api_key or os.environ.get("LLM_API_KEY", "EMPTY"),
        temperature=args.temperature,
        seed=args.seed or int(time.time()),
        output_dir=os.path.dirname(args.output) or "batch_output",
    )

    print(
        f"Config: total={config.total_target} batch={config.batch_size} "
        f"workers={config.max_workers} seed={config.seed}"
    )
    print(f"Model: {config.model}  Base: {config.base_url}")

    all_texts = load_checkpoint(args.checkpoint) if args.resume else []
    print(f"Loaded {len(all_texts)} from checkpoint")

    tasks = generate_task_list(config)
    completed_ids = {t.get("task_id", -1) for t in all_texts}
    pending = [t for t in tasks if t["task_id"] not in completed_ids]
    print(f"Tasks: {len(tasks)} total, {len(pending)} pending")

    if not pending:
        print("All done!")
        return

    lock = threading.Lock()
    seen_norm = set()
    for item in all_texts:
        seen_norm.add(_normalize_for_dedup(item.get("text", "")))

    completed = 0
    failed = 0
    skipped = 0
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        future_to_idx = {}
        next_idx = 0

        def submit():
            nonlocal next_idx, all_texts
            if next_idx >= len(pending):
                return
            task = dict(pending[next_idx])
            prefix_hint = build_suppression_hint(all_texts, config.suppression_window_size)
            tag_hint = build_tag_diversity_hint(all_texts, config.suppression_window_size)
            combined_hint = "\n".join(filter(None, [prefix_hint, tag_hint]))
            task["suppression_hint"] = combined_hint
            future = executor.submit(worker_fn, task, config)
            future_to_idx[future] = next_idx
            next_idx += 1

        for _ in range(min(config.max_workers, len(pending))):
            submit()

        while future_to_idx:
            for future in as_completed(list(future_to_idx)):
                idx = future_to_idx.pop(future)
                break
            try:
                results = future.result(timeout=180)
                if results:
                    with lock:
                        new_items = []
                        for item in results:
                            norm = _normalize_for_dedup(item.get("text", ""))
                            if norm in seen_norm:
                                skipped += 1
                                continue
                            seen_norm.add(norm)
                            new_items.append(item)
                        all_texts.extend(new_items)
                        completed += 1
                else:
                    with lock:
                        failed += 1
            except Exception as e:
                print(f"Task {idx} err: {e}")
                with lock:
                    failed += 1

            with lock:
                if completed > 0 and completed % 5 == 0:
                    elapsed = time.time() - start_time
                    rate = len(all_texts) / max(1, elapsed)
                    print(
                        f"[{completed}/{len(pending)}] ok={completed} fail={failed} "
                        f"texts={len(all_texts)} skip={skipped} rate={rate:.1f} t/s"
                    )
                    save_checkpoint(all_texts, args.checkpoint)

            submit()

    save_checkpoint(all_texts, args.checkpoint)
    elapsed = time.time() - start_time
    print(
        f"\nDone: {completed} ok {failed} fail {len(all_texts)} texts in {elapsed:.1f}s "
        f"({len(all_texts) / max(1, elapsed):.1f} t/s)"
    )

    if not args.no_postprocess:
        print(f"\nPost-processing...")
        all_texts = deduplicate(all_texts)
        print(f"After exact dedup: {len(all_texts)}")
        all_texts = semantic_deduplicate(all_texts, threshold=config.semantic_dedup_threshold)
        print(f"After semantic dedup: {len(all_texts)}")
        all_texts = quality_filter(
            all_texts, max_tags_per_text=config.max_tags_per_text, max_same_tag_repeat=config.max_same_tag_repeat
        )
        print(f"After quality: {len(all_texts)}")

    save_jsonl(all_texts, args.output)
    print(f"Saved {len(all_texts)} -> {args.output}")
    print_statistics(all_texts)


if __name__ == "__main__":
    main()
