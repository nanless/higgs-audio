#!/usr/bin/env python3
"""
扫描 CLONE_ROOT 下每个说话人的已有 clone 数量 (多进程加速).
输出 JSON: {"dataset/speaker_id": count}

Usage:
  python 05_scan_existing_clones.py --clone-root /path --resume-csv /path/resume.csv [--workers 64]
"""

import argparse, csv, json, os, re
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count

CLONE_WAV_RE = re.compile(r"^clone_(\d+)\.wav$")


def _count_dir(args: tuple) -> list:
    """args = (clone_root, [(dataset, speaker_id), ...]) → [(key, count), ...]"""
    clone_root, rows = args
    out = []
    for dataset, speaker_id in rows:
        spk_dir = os.path.join(clone_root, dataset, speaker_id)
        if not os.path.isdir(spk_dir):
            continue
        cnt = 0
        try:
            with os.scandir(spk_dir) as it:
                for e in it:
                    if not CLONE_WAV_RE.match(e.name):
                        continue
                    try:
                        if e.stat().st_size > 1000:
                            cnt += 1
                    except OSError:
                        pass
        except OSError:
            pass
        if cnt > 0:
            out.append((f"{dataset}/{speaker_id}", cnt))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clone-root", required=True)
    p.add_argument("--resume-csv", required=True)
    p.add_argument("--out-json", default=None, help="输出文件 (默认 stdout)")
    p.add_argument("--workers", type=int, default=min(cpu_count(), 64))
    args = p.parse_args()

    speakers = []
    with open(args.resume_csv, newline="") as f:
        for r in csv.DictReader(f):
            dataset = r.get("dataset", "")
            speaker_id = r.get("speaker_id", "")
            if dataset and speaker_id:
                speakers.append((dataset, speaker_id))

    counts = {}
    workers = max(1, args.workers)
    if len(speakers) > 1 and workers > 1:
        # round-robin shard speakers across workers
        shards = [speakers[i::workers] for i in range(workers)]
        tasks = [(args.clone_root, s) for s in shards if s]
        with ProcessPoolExecutor(max_workers=len(tasks)) as ex:
            for batch in ex.map(_count_dir, tasks):
                for key, cnt in batch:
                    counts[key] = cnt
    else:
        for key, cnt in _count_dir((args.clone_root, speakers)):
            counts[key] = cnt

    out = json.dumps(counts, ensure_ascii=False)
    if args.out_json:
        with open(args.out_json, "w") as f:
            f.write(out + "\n")
    else:
        print(out)


if __name__ == "__main__":
    main()
