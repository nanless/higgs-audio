#!/usr/bin/env python3
"""Preflight checks before resuming the v3 topup / iterative clone pipeline.

Exit 0 if safe to resume; non-zero otherwise.
Prints a one-line suggestion: SUGGEST_START_STEP=clone|sim|cer

Uses 05_scan_existing_clones (multiprocess, resume-csv speakers only) instead of
a full tree walk.

Usage:
  python 08_preflight_resume.py \\
    --pipeline-workdir clone_workdir/iterative_pipeline_v5 \\
    --clone-root /path/to/clone_5 \\
    --stats-csv clone_workdir/stats_topup_v5/all_speakers.csv \\
    --texts-jsonl /path/texts.jsonl \\
    --source-dirs /path/audio \\
    --start-round 2 \\
    --num-servers 8
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pipeline-workdir", required=True)
    p.add_argument("--clone-root", required=True)
    p.add_argument("--stats-csv", required=True)
    p.add_argument("--texts-jsonl", required=True)
    p.add_argument("--source-dirs", required=True, help="space-separated; first dir checked")
    p.add_argument("--start-round", type=int, default=2)
    p.add_argument("--num-servers", type=int, default=8)
    p.add_argument("--min-free-gb", type=float, default=500.0)
    p.add_argument("--scan-workers", type=int, default=64)
    p.add_argument("--repo-root", default="")
    args = p.parse_args()

    ok = True
    here = os.path.dirname(os.path.abspath(__file__))
    wd = args.pipeline_workdir
    resume_csv = os.path.join(wd, "allocation", "speaker_duration_stats_post_prune_resume.csv")
    orig_json = os.path.join(wd, "original_clones_needed.json")

    checks = [
        ("STATS_CSV", args.stats_csv),
        ("resume CSV", resume_csv),
        ("original_clones_needed.json", orig_json),
        ("TEXTS_JSONL", args.texts_jsonl),
        ("CLONE_ROOT", args.clone_root),
    ]
    for label, path in checks:
        if label == "CLONE_ROOT":
            if not os.path.isdir(path):
                print(f"❌ {label} 不是目录: {path}")
                ok = False
            else:
                print(f"✓ {label}: {path}")
        elif not os.path.isfile(path):
            print(f"❌ 缺少 {label}: {path}")
            ok = False
        elif label == "TEXTS_JSONL" and os.path.getsize(path) == 0:
            print(f"❌ TEXTS_JSONL 为空: {path}")
            ok = False
        else:
            print(f"✓ {label}: {path}")

    src0 = args.source_dirs.split()[0] if args.source_dirs.strip() else ""
    if not src0 or not os.path.isdir(src0):
        print(f"❌ SOURCE_DIRS 首目录无效: {src0!r}")
        ok = False
    else:
        print(f"✓ SOURCE_DIRS[0]: {src0}")

    total_needed = 0
    if os.path.isfile(orig_json):
        try:
            orig = json.load(open(orig_json))
            total_needed = sum(int(v.get("clones_needed", 0)) for v in orig.values())
            print(f"  original clones_needed 合计: {total_needed:,}")
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as e:
            print(f"❌ 无法读 original_clones_needed.json: {e}")
            ok = False

    rpad = f"{args.start_round:02d}"
    round_dir = os.path.join(wd, f"round_{rpad}")
    print(f"  round dir: {round_dir} ({'存在' if os.path.isdir(round_dir) else '不存在'})")
    has_summary = os.path.isfile(os.path.join(round_dir, "clone_summary.json"))
    has_after = os.path.isfile(os.path.join(round_dir, "existing_clones_after.json"))
    print(f"  round_{rpad} clone_summary.json: {'是' if has_summary else '否'}")
    print(f"  round_{rpad} existing_clones_after.json: {'是' if has_after else '否'}")

    pairs = 0
    if os.path.isfile(resume_csv):
        print(f"  扫描 CLONE_ROOT (05_scan, workers={args.scan_workers}) ...", flush=True)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
            out_json = tf.name
        try:
            cmd = [
                sys.executable,
                os.path.join(here, "05_scan_existing_clones.py"),
                "--clone-root",
                args.clone_root,
                "--resume-csv",
                resume_csv,
                "--workers",
                str(args.scan_workers),
                "--out-json",
                out_json,
            ]
            subprocess.check_call(cmd)
            counts = json.load(open(out_json))
            pairs = int(sum(counts.values()))
            print(f"  磁盘 clone(wav+json)={pairs:,}  speakers_with_clones={len(counts):,}")
        except (OSError, subprocess.CalledProcessError, json.JSONDecodeError, TypeError, ValueError) as e:
            print(f"⚠️  扫描失败: {e}")
        finally:
            try:
                os.unlink(out_json)
            except OSError:
                pass

    suggest = "clone"
    if has_summary and not has_after:
        suggest = "sim"
    print(f"SUGGEST_START_STEP={suggest}")

    try:
        out = subprocess.check_output(["nvidia-smi", "-L"], text=True, stderr=subprocess.DEVNULL)
        gpu_n = len([ln for ln in out.splitlines() if ln.strip()])
        if gpu_n < args.num_servers:
            print(f"❌ 可见 GPU={gpu_n} < NUM_SERVERS={args.num_servers}")
            ok = False
        else:
            print(f"✓ GPU 可见 {gpu_n} 张")
    except (OSError, subprocess.CalledProcessError):
        print("⚠️  nvidia-smi 不可用, 跳过 GPU 检查")

    repo = args.repo_root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    marker = os.path.join(repo, "higgs_v3_env")
    try:
        ps = subprocess.check_output(["ps", "-eo", "pid,cmd"], text=True, stderr=subprocess.DEVNULL)
        stale = [ln for ln in ps.splitlines() if marker in ln and "08_preflight" not in ln]
        if stale:
            print(f"⚠️  仍有 higgs_v3_env 相关进程 {len(stale)} 个 (08_resume 会尝试清理)")
        else:
            print("✓ 无残留 higgs_v3_env 进程")
    except (OSError, subprocess.CalledProcessError):
        pass

    try:
        usage = shutil.disk_usage(args.clone_root)
        free_gb = usage.free / (1024**3)
        if free_gb < args.min_free_gb:
            print(f"❌ 磁盘可用 {free_gb:.0f}GB < MIN_FREE_GB={args.min_free_gb}")
            ok = False
        else:
            print(f"✓ 磁盘可用约 {free_gb:.0f}GB")
    except OSError as e:
        print(f"⚠️  无法检查磁盘: {e}")

    if not ok:
        print("❌ 预检失败")
        return 1
    print("✓ 预检通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
