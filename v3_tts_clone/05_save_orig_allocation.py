#!/usr/bin/env python3
"""
从 04_post_prune_stats.py 的输出 CSV 保存原始 clones_needed 和 start_clone_idx.

Usage:
  python 05_save_orig_allocation.py --resume-csv /path/resume.csv --out-json /path/orig.json
"""

import argparse, csv, json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--resume-csv", required=True)
    p.add_argument("--out-json", required=True)
    args = p.parse_args()

    rows = list(csv.DictReader(open(args.resume_csv, newline="")))
    orig = {}
    for r in rows:
        uid = f"{r['dataset']}__{r['speaker_id']}"
        orig[uid] = {
            "clones_needed": int(r.get("clones_needed", 0)),
            "start_clone_idx": int(r.get("start_clone_idx", 0)),
        }

    with open(args.out_json, "w") as f:
        json.dump(orig, f, ensure_ascii=False, indent=2)

    total = sum(v["clones_needed"] for v in orig.values())
    print(f"  NEED_CLONE 说话人数: {len(orig)}")
    print(f"  全量克隆总数: {total}")


if __name__ == "__main__":
    main()
