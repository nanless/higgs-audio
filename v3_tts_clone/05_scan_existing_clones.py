#!/usr/bin/env python3
"""
扫描 CLONE_ROOT 下每个说话人的已有 clone 数量.
输出 JSON: {"dataset/speaker_id": count}

Usage:
  python 05_scan_existing_clones.py --clone-root /path --resume-csv /path/resume.csv
"""

import argparse, csv, json, os, re

CLONE_WAV_RE = re.compile(r"^clone_(\d+)\.wav$")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clone-root", required=True)
    p.add_argument("--resume-csv", required=True)
    p.add_argument("--out-json", default=None, help="输出文件 (默认 stdout)")
    args = p.parse_args()

    counts = {}
    with open(args.resume_csv, newline="") as f:
        for r in csv.DictReader(f):
            dataset = r.get("dataset", "")
            speaker_id = r.get("speaker_id", "")
            if not dataset or not speaker_id:
                continue
            spk_dir = os.path.join(args.clone_root, dataset, speaker_id)
            if not os.path.isdir(spk_dir):
                continue
            cnt = 0
            try:
                for fname in os.listdir(spk_dir):
                    if not CLONE_WAV_RE.match(fname):
                        continue
                    try:
                        if os.path.getsize(os.path.join(spk_dir, fname)) > 1000:
                            cnt += 1
                    except OSError:
                        pass
            except OSError:
                pass
            if cnt > 0:
                counts[f"{dataset}/{speaker_id}"] = cnt

    out = json.dumps(counts, ensure_ascii=False)
    if args.out_json:
        with open(args.out_json, "w") as f:
            f.write(out + "\n")
    else:
        print(out)


if __name__ == "__main__":
    main()
