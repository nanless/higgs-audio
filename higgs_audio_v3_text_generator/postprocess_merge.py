#!/usr/bin/env python3
"""
Final merge + postprocess for large-scale batch generation.
Reads raw outputs from 4 workers, deduplicates, quality filters, saves final JSONL.
"""

import json
import os
import sys

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
    args = parser.parse_args()

    all_texts = []
    for w in range(4):
        path = f"{args.input_dir}/generated_texts_w{w}.jsonl"
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    if line.strip():
                        all_texts.append(json.loads(line))
            print(f"Loaded {path}")

    print(f"Total raw: {len(all_texts)}")

    all_texts = deduplicate(all_texts)
    print(f"Exact dedup: {len(all_texts)}")

    all_texts = semantic_deduplicate(all_texts, threshold=args.semantic_threshold)
    print(f"Semantic dedup: {len(all_texts)}")

    all_texts = quality_filter(all_texts, max_tags_per_text=args.max_tags, max_same_tag_repeat=2)
    print(f"Quality filter: {len(all_texts)}")

    save_jsonl(all_texts, args.output)
    print(f"\nSaved {len(all_texts)} -> {args.output}")
    print_statistics(all_texts)


if __name__ == "__main__":
    main()
