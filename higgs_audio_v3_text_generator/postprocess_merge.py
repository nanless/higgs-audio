#!/usr/bin/env python3
"""
Final merge + postprocess for large-scale batch generation.
Reads raw outputs from 4 workers, deduplicates, quality filters, saves final JSONL.
"""

import json
import os
import re
import sys
from glob import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from higgs_text_gen.dedup import deduplicate, semantic_deduplicate
from higgs_text_gen.quality_filter import quality_filter
from higgs_text_gen.output import save_jsonl, print_statistics


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="batch_output")
    parser.add_argument("--output", default="batch_output/generated_texts_final.jsonl")
    parser.add_argument("--semantic-threshold", type=float, default=0.88)
    parser.add_argument("--max-tags", type=int, default=5)
    parser.add_argument("--target-count", type=int, default=None, help="Truncate to this count; fail if fewer survive")
    parser.add_argument("--num-workers", type=int, default=None, help="Read exactly worker IDs 0..N-1")
    args = parser.parse_args()

    all_texts = []
    if args.num_workers is not None:
        paths = [os.path.join(args.input_dir, f"generated_texts_w{i}.jsonl") for i in range(args.num_workers)]
        missing = [path for path in paths if not os.path.exists(path)]
        if missing:
            print(f"ERROR: missing worker outputs: {missing}", file=sys.stderr)
            sys.exit(1)
    else:
        paths = glob(os.path.join(args.input_dir, "generated_texts_w*.jsonl"))
        paths.sort(key=lambda p: int(re.search(r"_w(\d+)\.jsonl$", p).group(1)))
    if not paths:
        print(f"ERROR: no worker outputs under {args.input_dir}", file=sys.stderr)
        sys.exit(1)
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                if line.strip():
                    try:
                        all_texts.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
        print(f"Loaded {path}")

    print(f"Total raw: {len(all_texts)}")

    all_texts = deduplicate(all_texts)
    print(f"Exact dedup: {len(all_texts)}")

    all_texts = semantic_deduplicate(all_texts, threshold=args.semantic_threshold)
    print(f"Semantic dedup: {len(all_texts)}")

    all_texts = quality_filter(all_texts, max_tags_per_text=args.max_tags, max_same_tag_repeat=2)
    print(f"Quality filter: {len(all_texts)}")

    below_target = args.target_count is not None and len(all_texts) < args.target_count
    if args.target_count is not None and len(all_texts) >= args.target_count:
        all_texts = all_texts[: args.target_count]
        print(f"Exact target: {len(all_texts)}")

    save_jsonl(all_texts, args.output)
    print(f"\nSaved {len(all_texts)} -> {args.output}")
    print_statistics(all_texts)
    if below_target:
        print(
            f"ERROR: only {len(all_texts)} texts survived postprocessing; target is {args.target_count}",
            file=sys.stderr,
        )
        sys.exit(2)


if __name__ == "__main__":
    main()
