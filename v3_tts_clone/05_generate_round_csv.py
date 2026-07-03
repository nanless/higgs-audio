#!/usr/bin/env python3
"""
每轮克隆前生成受限的 CSV:
  clones_needed = min(ceil(原始/总轮数), 原始 - 已有)
  start_clone_idx = 原始基数 + 已有
  speaker_path 覆盖为合并目录 (如果设置)

Usage:
  python 05_generate_round_csv.py \
      --orig-json   clone_workdir/iterative_pipeline/original_clones_needed.json \
      --existing-json round_NN/existing_clones.json \
      --resume-csv   allocation/speaker_duration_stats_post_prune_resume.csv \
      --merged-dir   clone_workdir/iterative_pipeline/merged_sources \
      --total-rounds 10 \
      --out-csv      round_NN/resume_round.csv
"""

import argparse, csv, json, math, os


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--orig-json", required=True, help="01 分配的原始 clones_needed")
    p.add_argument("--existing-json", required=True, help="scan_existing_clones 的输出")
    p.add_argument("--resume-csv", required=True, help="04_post_prune_stats 生成的 resume CSV")
    p.add_argument("--merged-dir", default="", help="合并后的原音频目录 (为空则不覆盖 speaker_path)")
    p.add_argument("--total-rounds", type=int, required=True)
    p.add_argument("--out-csv", required=True)
    args = p.parse_args()

    orig = json.load(open(args.orig_json))

    existing = {}
    if os.path.isfile(args.existing_json) and os.path.getsize(args.existing_json) > 2:
        raw = json.load(open(args.existing_json))
        for path_key, cnt in raw.items():
            existing[path_key.replace("/", "__")] = cnt

    reader = csv.DictReader(open(args.resume_csv, newline=""))
    full_rows = list(reader)
    if not full_rows:
        # resume CSV 只有表头(零 NEED_CLONE 行): 写出空的本轮 CSV(带表头)并返回,
        # 让流水线的 has_clones 检查判定为 0 → 本轮无克隆, 不崩溃。
        fieldnames = reader.fieldnames or []
        with open(args.out_csv, "w", newline="") as f:
            if fieldnames:
                csv.DictWriter(f, fieldnames=fieldnames).writeheader()
        print("resume CSV 无数据行, 本轮无克隆任务")
        print(f"输出: {args.out_csv}")
        return
    fieldnames = list(full_rows[0].keys())

    use_merged = bool(args.merged_dir)
    total = 0
    stats = []

    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in full_rows:
            uid = f"{r['dataset']}__{r['speaker_id']}"
            if use_merged:
                r["speaker_path"] = f"{args.merged_dir}/{r['dataset']}/{r['speaker_id']}"
            if uid not in orig:
                r["clones_needed"] = "0"
                w.writerow(r)
                continue
            full_needed = orig[uid]["clones_needed"]
            already = existing.get(uid, 0)
            still_need = max(0, full_needed - already)
            per_round = max(1, math.ceil(full_needed / args.total_rounds))
            this_round = min(per_round, still_need)
            r["clones_needed"] = str(this_round)
            r["start_clone_idx"] = str(orig[uid]["start_clone_idx"] + already)
            total += this_round
            w.writerow(r)
            if this_round > 0:
                stats.append(f"    {r['dataset']}/{r['speaker_id']}: 已有{already} 仍需{still_need} 本轮{this_round}")

    for line in stats[:20]:
        print(line)
    if len(stats) > 20:
        print(f"    ... 共 {len(stats)} 个说话人本次克隆")
    print(f"本轮计划克隆: {total} 条")
    print(f"输出: {args.out_csv}")


if __name__ == "__main__":
    main()
