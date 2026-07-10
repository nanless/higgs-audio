#!/usr/bin/env python3
# Copyright (c) 2025 Boson AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Filter previously-cloned dirs by raw cosine SIM and COPY the survivors elsewhere.

Reuses the ALREADY-computed sidecars (no GPU, no re-embedding, no reference files):
  - .sim.json `similarity`  (produced with the (cos+1)/2 mapping on the old dirs)

Since mapped = (raw + 1) / 2 EXACTLY, raw cosine is recovered losslessly as
  raw = 2 * similarity - 1        (for --sim-scale mapped, the default for old dirs)

KEEP + COPY a clone (wav + every sidecar) if
      raw_cos >= --min-sim-raw    (default 0.8; matches prune sim < min)
A clone with no .sim.json (or missing `similarity`) is SKIPPED (cannot judge).

The destination mirrors the source's {dataset}/{speaker}/ subdir structure exactly.
Because several source dirs can share identical {dataset}/{speaker}/clone_NNNN names,
each source may be given its own filename PREFIX (e.g. c1_/c2_/c3_) so files copied
into the SAME destination never collide. The single omnivoice source keeps its
original filenames (empty prefix). The copied .sim.json gets the raw cosine written
in alongside the original mapped value:
      similarity_raw   = 2 * similarity - 1
      similarity_mapped = <original similarity>   (unchanged 'similarity' also kept)

Default is DRY-RUN. Pass --execute to actually copy.

Usage (defaults already match the requested layout, just add --execute):
    python v3_tts_clone/06_filter_copy_by_sim.py                 # dry-run preview
    python v3_tts_clone/06_filter_copy_by_sim.py --execute       # do the copy

    # override sources/dests/prefixes (SRC:DEST:PREFIX per job, prefix optional):
    python v3_tts_clone/06_filter_copy_by_sim.py \
        --jobs /path/src_a:/path/dst_merged:a_ /path/src_b:/path/dst_merged:b_ \
        --min-sim-raw 0.8 --execute
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


SKIP_DIRS = frozenset({"logs", "__pycache__", "ref", "eval_sim_embedding_cache"})
# sidecar suffixes copied verbatim (the .sim.json is handled separately so raw can be injected)
COPY_SUFFIXES = (".json", ".cer.json", ".mos.json", ".eval.json")

_BASE = (
    "/root/group-shared/voiceprint/data/speech/speaker_diarization/"
    "merged_datasets_20250610_vad_segments_mtfaa_enhanced_extend_kid_withclone_addlibrilight_1130"
)

# (source_root, dest_root, filename_prefix)
DEFAULT_JOBS = [
    (f"{_BASE}/audio_higgs_audio_v3_tts_clone", f"{_BASE}/audio_higgs_audio_v3_tts_clone_123_sim0.8_filtered", "c1_"),
    (
        f"{_BASE}/audio_higgs_audio_v3_tts_clone_2",
        f"{_BASE}/audio_higgs_audio_v3_tts_clone_123_sim0.8_filtered",
        "c2_",
    ),
    (
        f"{_BASE}/audio_higgs_audio_v3_tts_clone_3",
        f"{_BASE}/audio_higgs_audio_v3_tts_clone_123_sim0.8_filtered",
        "c3_",
    ),
    (f"{_BASE}/audio_omnivoice_clone", f"{_BASE}/audio_omnivoice_clone_sim0.8_filtered", ""),
]


def _is_main_sidecar(name: str) -> bool:
    return (
        name.endswith(".json")
        and not name.endswith(".cer.json")
        and not name.endswith(".mos.json")
        and not name.endswith(".sim.json")
        and not name.endswith(".eval.json")
    )


def _wav_dur(path: str) -> float:
    try:
        with open(path, "rb") as f:
            h = f.read(44)
        if len(h) >= 44 and h[:4] == b"RIFF" and h[8:12] == b"WAVE":
            ch = struct.unpack_from("<H", h, 22)[0]
            sr = struct.unpack_from("<I", h, 24)[0]
            bps = struct.unpack_from("<H", h, 34)[0]
            if sr > 0 and bps > 0 and ch > 0:
                return (os.path.getsize(path) - 44) / (sr * ch * (bps / 8))
    except OSError:
        pass
    return 0.0


def _process_unit(task: tuple) -> dict:
    """Scan one speaker/dataset subtree; optionally copy survivors. Return counters."""
    root, unit, dest_root, prefix, min_sim_raw, sim_scale, execute = task
    c = {
        "total": 0,
        "kept": 0,
        "below": 0,
        "missing_sim": 0,
        "copied_files": 0,
        "copy_fail": 0,
        "dur_total": 0.0,
        "dur_kept": 0.0,
    }
    by_ds = defaultdict(lambda: [0, 0])  # dataset -> [total_with_sim, kept]
    for dirpath, dirs, files in os.walk(unit):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in files:
            if not _is_main_sidecar(name):
                continue
            base = name[:-5]  # strip .json -> stem filename (e.g. clone_0005)
            stem = os.path.join(dirpath, base)
            wav = stem + ".wav"
            if not os.path.isfile(wav):
                continue
            sim_path = stem + ".sim.json"
            if not os.path.isfile(sim_path):
                c["missing_sim"] += 1
                continue
            try:
                with open(sim_path, encoding="utf-8") as f:
                    sim_obj = json.load(f)
                sim = sim_obj.get("similarity")
            except (OSError, json.JSONDecodeError, TypeError):
                sim = None
            if sim is None:
                c["missing_sim"] += 1
                continue
            raw = float(sim) if sim_scale == "raw" else (2.0 * float(sim) - 1.0)
            dur = _wav_dur(wav)
            dataset = os.path.relpath(wav, root).split(os.sep)[0]
            c["total"] += 1
            c["dur_total"] += dur
            by_ds[dataset][0] += 1
            if raw < min_sim_raw:
                c["below"] += 1
                continue
            c["kept"] += 1
            c["dur_kept"] += dur
            by_ds[dataset][1] += 1
            if not execute:
                continue
            reldir = os.path.relpath(dirpath, root)
            ddir = os.path.join(dest_root, reldir)
            try:
                os.makedirs(ddir, exist_ok=True)
                shutil.copy2(wav, os.path.join(ddir, prefix + base + ".wav"))
                c["copied_files"] += 1
                for suf in COPY_SUFFIXES:
                    src = stem + suf
                    if os.path.isfile(src):
                        shutil.copy2(src, os.path.join(ddir, prefix + base + suf))
                        c["copied_files"] += 1
                # sim.json with raw cosine injected (mapped kept intact)
                sim_obj["similarity_raw"] = raw
                sim_obj["similarity_mapped"] = float(sim)
                sim_obj["similarity_scale_note"] = (
                    "'similarity' is mapped (cos+1)/2 as originally stored; "
                    "'similarity_raw' is the raw cosine = 2*mapped-1"
                )
                dst_sim = os.path.join(ddir, prefix + base + ".sim.json")
                tmp = dst_sim + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(sim_obj, f, ensure_ascii=False)
                os.replace(tmp, dst_sim)
                c["copied_files"] += 1
            except OSError as e:
                c["copy_fail"] += 1
                print(f"  [COPY FAIL] {wav}: {e}", file=sys.stderr)
    c["by_dataset"] = {ds: v for ds, v in by_ds.items()}
    return c


def _units(root: str) -> list:
    units = []
    for ds in sorted(os.listdir(root)):
        ds_path = os.path.join(root, ds)
        if not os.path.isdir(ds_path) or ds in SKIP_DIRS:
            continue
        spk = [
            os.path.join(ds_path, s)
            for s in os.listdir(ds_path)
            if os.path.isdir(os.path.join(ds_path, s)) and s not in SKIP_DIRS
        ]
        units.extend(spk if spk else [ds_path])
    return units


def process_job(root: str, dest_root: str, prefix: str, args) -> dict:
    name = os.path.basename(os.path.normpath(root))
    print(f"\n########## {name}  (prefix='{prefix}')  ->  {os.path.basename(dest_root)} ##########", flush=True)
    if not os.path.isdir(root):
        print(f"  跳过: 不存在 {root}", flush=True)
        return {"name": name, "error": "missing"}

    units = _units(root)
    t0 = time.time()
    agg = {
        "total": 0,
        "kept": 0,
        "below": 0,
        "missing_sim": 0,
        "copied_files": 0,
        "copy_fail": 0,
        "dur_total": 0.0,
        "dur_kept": 0.0,
    }
    by_ds = defaultdict(lambda: [0, 0])
    tasks = [(root, u, dest_root, prefix, args.min_sim_raw, args.sim_scale, args.execute) for u in units]
    if len(tasks) > 1 and args.workers > 1:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as ex:
            results = list(ex.map(_process_unit, tasks, chunksize=4))
    else:
        results = [_process_unit(t) for t in tasks]
    for c in results:
        for k in agg:
            agg[k] += c[k]
        for ds, (t, kp) in c["by_dataset"].items():
            by_ds[ds][0] += t
            by_ds[ds][1] += kp

    rep = {
        "name": name,
        "source": root,
        "dest": dest_root,
        "prefix": prefix,
        "units": len(units),
        "total": agg["total"],
        "kept": agg["kept"],
        "below": agg["below"],
        "missing_sim": agg["missing_sim"],
        "copied_files": agg["copied_files"],
        "copy_fail": agg["copy_fail"],
        "min_sim_raw": args.min_sim_raw,
        "sim_scale": args.sim_scale,
        "dur_total_hours": round(agg["dur_total"] / 3600.0, 2),
        "dur_kept_hours": round(agg["dur_kept"] / 3600.0, 2),
        "executed": bool(args.execute),
        "seconds": round(time.time() - t0, 1),
        "by_dataset": {
            ds: {"total": t, "kept": kp} for ds, (t, kp) in sorted(by_ds.items(), key=lambda kv: -kv[1][1])
        },
    }
    return rep


def report_text(rep: dict) -> str:
    if rep.get("error"):
        return f"{rep['name']}: {rep['error']}"
    tag = "已拷贝" if rep["executed"] else "DRY-RUN(未拷)"
    L = [
        "=" * 92,
        f"  {rep['name']}   [{tag}]   规则: raw>{rep['min_sim_raw']}  (sim_scale={rep['sim_scale']}, prefix='{rep['prefix']}')",
        f"  -> {rep['dest']}",
        "=" * 92,
        f"  有sim {rep['total']:,}  →  保留(拷) {rep['kept']:,}  未达阈值 {rep['below']:,}   缺sim.json {rep['missing_sim']:,}",
        f"    时长: 有sim总 {rep['dur_total_hours']:,}h   保留 {rep['dur_kept_hours']:,}h",
        f"    单元 {rep['units']:,}   耗时 {rep['seconds']}s",
    ]
    if rep["executed"]:
        L.append(f"    实拷文件 {rep['copied_files']:,}   失败 {rep['copy_fail']:,}")
    L.append("    保留最多的 dataset (top 10):")
    for ds, v in list(rep["by_dataset"].items())[:10]:
        if v["kept"]:
            L.append(f"      {ds:<34} 保留 {v['kept']:,}/{v['total']:,}")
    if not rep["executed"]:
        L.append("  * DRY-RUN: 未拷贝任何文件。确认无误后加 --execute 真正拷贝。")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--jobs",
        nargs="+",
        default=None,
        help="Override default jobs, each as SRC:DEST:PREFIX (prefix optional). Colons in paths not supported.",
    )
    ap.add_argument("--min-sim-raw", type=float, default=0.8, help="Keep+copy if raw cosine >= this (default 0.8)")
    ap.add_argument(
        "--sim-scale",
        choices=("mapped", "raw"),
        default="mapped",
        help="Scale of 'similarity' in .sim.json: mapped=(cos+1)/2 (old dirs, default) -> raw=2*sim-1; raw=已是原始余弦",
    )
    ap.add_argument("--execute", action="store_true", help="Actually copy (default: dry-run preview only)")
    ap.add_argument("--workers", type=int, default=32, help="Parallel scan/copy workers")
    ap.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "filter_copy_report")
    args = ap.parse_args()

    if args.jobs:
        jobs = []
        for spec in args.jobs:
            parts = spec.split(":")
            src, dest = parts[0], parts[1]
            prefix = parts[2] if len(parts) > 2 else ""
            jobs.append((src, dest, prefix))
    else:
        jobs = DEFAULT_JOBS

    args.output_dir.mkdir(parents=True, exist_ok=True)
    mode = "EXECUTE(真拷)" if args.execute else "DRY-RUN(预览)"
    print(f"模式={mode}  规则: raw>{args.min_sim_raw}  sim_scale={args.sim_scale}  jobs={len(jobs)}", flush=True)

    reports = []
    for root, dest_root, prefix in jobs:
        rep = process_job(root, dest_root, prefix, args)
        reports.append(rep)
        print("\n" + report_text(rep), flush=True)
        (args.output_dir / f"{rep['name']}.json").write_text(
            json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    (args.output_dir / "filter_copy_summary.json").write_text(
        json.dumps({"execute": args.execute, "reports": reports}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    tot = sum(r.get("total", 0) for r in reports)
    kept = sum(r.get("kept", 0) for r in reports)
    print(f"\n合计: 有sim {tot:,} clones, 保留 {kept:,} ({'已拷贝' if args.execute else 'DRY-RUN'})", flush=True)
    print(f"报告: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
