#!/usr/bin/env python3
"""
Batch text generation pipeline for Higgs Audio v3 TTS.
Orchestrates concurrent LLM calls, deduplication, quality filtering,
and final output.

Usage:
    python run_batch_generation.py --total 10000 --workers 10
    python run_batch_generation.py --total 50000 --workers 16 --output my_texts.jsonl
"""

import argparse
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from higgs_text_gen.config import GenConfig
from higgs_text_gen.task_generator import generate_task_list
from higgs_text_gen.worker import worker
from higgs_text_gen.dedup import (
    deduplicate, semantic_deduplicate, build_duplicate_index,
    filter_incremental_duplicates,
)
from higgs_text_gen.quality_filter import quality_filter
from higgs_text_gen.checkpoint import save_checkpoint, load_checkpoint
from higgs_text_gen.output import save_jsonl, print_statistics


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

    hints = ["\n=== 频率抑制提示 ==="]
    hints.append("最近生成中以下开头被过度使用，请避免:")
    for i, opening in enumerate(overused[:5]):
        hints.append(f"  {i+1}. 避免以 '{opening}' 开头的句子")
    return "\n".join(hints)


def main():
    parser = argparse.ArgumentParser(description="Batch text generation for Higgs Audio v3 TTS")
    parser.add_argument("--total", type=int, default=10000, help="Total target texts")
    parser.add_argument("--batch-size", type=int, default=8, help="Texts per LLM call")
    parser.add_argument("--workers", type=int, default=10, help="Concurrent LLM workers")
    parser.add_argument("--model", type=str, default=None, help="LLM model name")
    parser.add_argument("--base-url", type=str, default=None, help="LLM API base URL")
    parser.add_argument("--api-key", type=str, default=None, help="LLM API key")
    parser.add_argument("--temperature", type=float, default=0.85, help="LLM temperature")
    parser.add_argument("--output", type=str, default="batch_output/generated_texts.jsonl",
                        help="Output JSONL path")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--stress-ratio", type=float, default=0.10,
                        help="Stress test ratio")
    parser.add_argument("--checkpoint", type=str, default="batch_output/.checkpoint.jsonl",
                        help="Checkpoint path")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--no-quality-filter", action="store_true",
                        help="Skip quality filter (raw output)")

    args = parser.parse_args()

    config = GenConfig(
        total_target=args.total,
        batch_size=args.batch_size,
        max_workers=args.workers,
        model=args.model or os.environ.get("LLM_MODEL", "qwen3.6-27b"),
        base_url=args.base_url or os.environ.get("LLM_BASE_URL", "http://localhost:8000"),
        api_key=args.api_key or os.environ.get("LLM_API_KEY"),
        temperature=args.temperature,
        stress_test_ratio=args.stress_ratio,
        seed=args.seed or int(time.time()),
        output_dir=os.path.dirname(args.output) or "batch_output",
    )

    print(f"Config: target={config.total_target}, batch_size={config.batch_size}, "
          f"workers={config.max_workers}, seed={config.seed}")
    print(f"Model: {config.model}")
    print(f"Output: {args.output}")
    print(f"Checkpoint: {args.checkpoint}")
    print()

    output_jsonl = args.output
    checkpoint_path = args.checkpoint

    if args.resume:
        all_texts = load_checkpoint(checkpoint_path)
        print(f"Resumed {len(all_texts)} texts from checkpoint")
    else:
        all_texts = []

    seen_normalized, duplicate_context_index = build_duplicate_index(all_texts)

    tasks = generate_task_list(config)
    total_tasks = len(tasks)
    print(f"Total tasks: {total_tasks} (batch_size={config.batch_size}, target={config.total_target})")

    completed_task_ids = {t.get("task_id", -1) for t in all_texts}
    pending_tasks = [t for t in tasks if t["task_id"] not in completed_task_ids]
    print(f"Pending tasks: {len(pending_tasks)}")

    if not pending_tasks:
        print("All tasks completed!")
    else:
        completed = 0
        failed = 0
        skipped_duplicates = 0
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
            future_to_task = {}
            next_task_idx = 0

            def submit_next():
                nonlocal next_task_idx
                if next_task_idx >= len(pending_tasks):
                    return
                task = dict(pending_tasks[next_task_idx])
                task["suppression_hint"] = build_suppression_hint(
                    all_texts, config.suppression_window_size
                )
                future = executor.submit(worker, task, config)
                future_to_task[future] = task
                next_task_idx += 1

            for _ in range(min(config.max_workers, len(pending_tasks))):
                submit_next()

            while future_to_task:
                for future in as_completed(list(future_to_task)):
                    task = future_to_task.pop(future)
                    break
                try:
                    results = future.result(timeout=180)
                    if results:
                        results, skipped = filter_incremental_duplicates(
                            results, seen_normalized, duplicate_context_index,
                            same_context_threshold=config.same_context_dup_threshold,
                        )
                        skipped_duplicates += skipped
                        all_texts.extend(results)
                        completed += 1
                    else:
                        failed += 1
                except Exception as e:
                    print(f"Task {task.get('task_id', '?')} failed: {e}")
                    failed += 1

                if (completed + failed) % 10 == 0:
                    elapsed = time.time() - start_time
                    rate = len(all_texts) / max(1, elapsed)
                    print(
                        f"[{completed+failed}/{len(pending_tasks)}] "
                        f"ok={completed} fail={failed} total_texts={len(all_texts)} "
                        f"skipped={skipped_duplicates} rate={rate:.1f} texts/s"
                    )
                    save_checkpoint(all_texts, checkpoint_path)

                submit_next()

        save_checkpoint(all_texts, checkpoint_path)
        elapsed = time.time() - start_time
        print(f"\nGeneration complete: {completed} ok, {failed} failed, "
              f"{len(all_texts)} texts in {elapsed:.1f}s")

    print(f"\nPost-processing {len(all_texts)} texts...")

    all_texts = deduplicate(all_texts)
    print(f"After exact dedup: {len(all_texts)}")

    all_texts = semantic_deduplicate(all_texts, threshold=config.semantic_dedup_threshold)
    print(f"After semantic dedup: {len(all_texts)}")

    raw_snapshot_path = args.output.replace(".jsonl", ".raw.jsonl")
    save_jsonl(all_texts, raw_snapshot_path)
    print(f"Raw snapshot saved to {raw_snapshot_path}")

    if not args.no_quality_filter:
        all_texts = quality_filter(
            all_texts,
            max_tags_per_text=config.max_tags_per_text,
            max_same_tag_repeat=config.max_same_tag_repeat,
        )
        print(f"After quality filter: {len(all_texts)}")

    save_jsonl(all_texts, output_jsonl)
    print(f"\nFinal output saved to {output_jsonl}")

    print_statistics(all_texts)


if __name__ == "__main__":
    main()
