#!/usr/bin/env python3
"""Step 4: Re-evaluate speaker duration after quality prune.

Combines original source audio duration + remaining kept clone duration.
Allocates a global clone generation budget (default 10,000 hours) across
NEED_CLONE speakers proportional to each speaker's gap to the target (--target-duration-sec, default 3600).

Usage:
    python 04_post_prune_stats.py \\
        --stats-csv ./clone_workdir/speaker_duration_stats.csv \\
        --clone-root /path/to/audio_higgs_audio_v3_tts_clone \\
        --output-dir ./clone_workdir \\
        --total-clone-hours 10000
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import struct
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

TARGET_DURATION_SEC = 3600.0
MIN_SOURCE_FILES = 20
DEFAULT_TOTAL_CLONE_HOURS = 10_000.0
CLONE_WAV_RE = re.compile(r"^clone_(\d+)\.wav$")

CSV_FIELDS = [
    "dataset",
    "speaker_id",
    "num_files",
    "source_duration_sec",
    "clone_duration_sec",
    "total_duration_sec",
    "gap_sec",
    "gap_weight",
    "allocated_clone_hours",
    "existing_clones",
    "start_clone_idx",
    "clones_needed",
    "has_7to20s",
    "speaker_path",
    "status",
]


def _wav_duration_sec(path: str) -> float:
    try:
        with open(path, "rb") as f:
            header = f.read(44)
        if len(header) >= 44 and header[:4] == b"RIFF" and header[8:12] == b"WAVE":
            channels = struct.unpack_from("<H", header, 22)[0]
            sr = struct.unpack_from("<I", header, 24)[0]
            bps = struct.unpack_from("<H", header, 34)[0]
            if sr > 0 and bps > 0 and channels > 0:
                data_bytes = os.path.getsize(path) - 44
                if data_bytes > 0:
                    return data_bytes / (sr * (bps / 8) * channels)
    except OSError:
        pass
    return 0.0


def _scan_speaker_clone_dir(clone_dir: str) -> tuple[float, int, int]:
    """Return (clone_duration_sec, clone_count, start_clone_idx)."""
    if not os.path.isdir(clone_dir):
        return 0.0, 0, 0

    total = 0.0
    count = 0
    max_idx = -1
    try:
        for name in os.listdir(clone_dir):
            m = CLONE_WAV_RE.match(name)
            if not m:
                continue
            path = os.path.join(clone_dir, name)
            if os.path.getsize(path) <= 1000:
                continue
            dur = _wav_duration_sec(path)
            if dur > 0:
                total += dur
                count += 1
                max_idx = max(max_idx, int(m.group(1)))
    except OSError:
        pass
    return total, count, max_idx + 1


def _process_row(row: dict, clone_root: str, target_sec: float = TARGET_DURATION_SEC) -> dict | None:
    num_files = int(row["num_files"])
    if num_files < MIN_SOURCE_FILES:
        return None

    source_dur = float(row["total_duration_sec"])
    dataset = row["dataset"]
    speaker_id = row["speaker_id"]
    clone_dir = os.path.join(clone_root, dataset, speaker_id)
    clone_dur, clone_count, start_idx = _scan_speaker_clone_dir(clone_dir)
    combined = source_dur + clone_dur
    gap = max(0.0, target_sec - combined)

    base = {
        "dataset": dataset,
        "speaker_id": speaker_id,
        "speaker_path": row["speaker_path"],
        "num_files": num_files,
        "source_duration_sec": round(source_dur, 2),
        "clone_duration_sec": round(clone_dur, 2),
        "total_duration_sec": round(combined, 2),
        "gap_sec": round(gap, 2),
        "gap_weight": 0.0,
        "allocated_clone_hours": 0.0,
        "existing_clones": clone_count,
        "start_clone_idx": start_idx,
        "clones_needed": 0,
        "has_7to20s": row.get("has_7to20s", ""),
        "status": "OK" if gap <= 0 else "NEED_CLONE",
    }
    return base


def _process_chunk(rows_chunk: list[dict], clone_root: str, target_sec: float) -> list[dict]:
    """Process a batch of rows in one worker. Chunking avoids the ProcessPoolExecutor
    deadlock/overhead from submitting tens of thousands of tiny per-row tasks."""
    out: list[dict] = []
    for row in rows_chunk:
        r = _process_row(row, clone_root, target_sec)
        if r is not None:
            out.append(r)
    return out


def allocate_clone_budget(
    results: list[dict],
    total_clone_hours: float,
    estimate_clone_duration: float,
) -> None:
    """Split total_clone_hours across NEED_CLONE speakers by gap_sec proportion."""
    need = [r for r in results if r["status"] == "NEED_CLONE"]
    total_gap_sec = sum(float(r["gap_sec"]) for r in need)
    if total_gap_sec <= 0 or total_clone_hours <= 0:
        return

    total_budget_sec = total_clone_hours * 3600.0
    allocated_sum_sec = 0.0
    for r in need:
        weight = float(r["gap_sec"]) / total_gap_sec
        alloc_sec = total_budget_sec * weight
        allocated_sum_sec += alloc_sec
        r["gap_weight"] = round(weight, 8)
        r["allocated_clone_hours"] = round(alloc_sec / 3600.0, 4)
        r["clones_needed"] = max(1, int(alloc_sec / estimate_clone_duration) + 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats-csv", required=True, help="Original Step 1 speaker_duration_stats.csv")
    parser.add_argument("--clone-root", required=True, help="Existing clone output root (for kept clone duration)")
    parser.add_argument("--output-dir", default="./clone_workdir")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--estimate-clone-duration", type=float, default=10.0)
    parser.add_argument(
        "--total-clone-hours",
        type=float,
        default=DEFAULT_TOTAL_CLONE_HOURS,
        help="Total clone generation hours to allocate across NEED_CLONE speakers by gap",
    )
    parser.add_argument(
        "--target-duration-sec",
        type=float,
        default=TARGET_DURATION_SEC,
        help="Per-speaker target duration in seconds (default 3600)",
    )
    args = parser.parse_args()
    target_sec = args.target_duration_sec

    os.makedirs(args.output_dir, exist_ok=True)
    with open(args.stats_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    print(f"Loaded {len(rows):,} speakers from {args.stats_csv}", flush=True)

    t0 = time.time()
    results: list[dict] = []
    chunk_size = 200
    chunks = [rows[i : i + chunk_size] for i in range(0, len(rows), chunk_size)]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_process_chunk, c, args.clone_root, target_sec) for c in chunks]
        for fut in as_completed(futs):
            results.extend(fut.result())

    allocate_clone_budget(results, args.total_clone_hours, args.estimate_clone_duration)
    results.sort(key=lambda x: (x["status"] != "NEED_CLONE", x["total_duration_sec"]))
    elapsed = time.time() - t0

    need = [r for r in results if r["status"] == "NEED_CLONE"]
    ok = [r for r in results if r["status"] == "OK"]
    total_clones = sum(r["clones_needed"] for r in need)
    total_gap_hours = sum(float(r["gap_sec"]) for r in need) / 3600.0
    total_allocated_hours = sum(float(r["allocated_clone_hours"]) for r in need)

    out_csv = os.path.join(args.output_dir, "speaker_duration_stats_post_prune.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(results)

    resume_csv = os.path.join(args.output_dir, "speaker_duration_stats_post_prune_resume.csv")
    with open(resume_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(need)

    summary = {
        "target_duration_sec": target_sec,
        "min_source_files": MIN_SOURCE_FILES,
        "total_clone_hours_budget": args.total_clone_hours,
        "allocation": "gap_proportional",
        "estimate_clone_duration": args.estimate_clone_duration,
        "elapsed_sec": elapsed,
        "speakers_ge20_source": len(results),
        "speakers_ok": len(ok),
        "speakers_need_clone": len(need),
        "total_gap_hours": round(total_gap_hours, 2),
        "allocated_clone_hours_sum": round(total_allocated_hours, 2),
        "additional_clones_to_generate": total_clones,
        "estimated_gross_clone_hours": round(total_clones * args.estimate_clone_duration / 3600.0, 2),
        "by_dataset": {},
    }
    by_ds: dict[str, dict] = {}
    for r in need:
        ds = r["dataset"]
        if ds not in by_ds:
            by_ds[ds] = {
                "need_clone": 0,
                "clones_needed": 0,
                "gap_hours": 0.0,
                "allocated_clone_hours": 0.0,
            }
        by_ds[ds]["need_clone"] += 1
        by_ds[ds]["clones_needed"] += r["clones_needed"]
        by_ds[ds]["gap_hours"] += float(r["gap_sec"]) / 3600.0
        by_ds[ds]["allocated_clone_hours"] += float(r["allocated_clone_hours"])
    summary["by_dataset"] = {k: dict(v) for k, v in sorted(by_ds.items())}

    json_path = os.path.join(args.output_dir, "post_prune_stats_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    txt_path = os.path.join(args.output_dir, "post_prune_stats_summary.txt")
    lines = [
        "Post-Prune Speaker Duration Re-evaluation",
        "=" * 72,
        f"Per-speaker target: {target_sec:.0f}s (source + kept clones)",
        f"Clone budget: {args.total_clone_hours:,.0f} hours (allocated by gap proportion)",
        f"Avg clone duration for sizing: {args.estimate_clone_duration:.0f}s",
        f"Elapsed: {elapsed:.1f}s",
        "",
        f"Speakers (>= {MIN_SOURCE_FILES} source clips): {len(results):,}",
        f"  OK (>= target combined):          {len(ok):,}",
        f"  NEED_CLONE (< target combined):   {len(need):,}",
        f"  Total gap (to {target_sec:.0f}s each):        {total_gap_hours:,.1f} hours",
        f"  Allocated clone hours (sum):      {total_allocated_hours:,.1f} hours",
        f"  Clones to generate:               {total_clones:,}",
        f"  Est. gross clone hours:           {summary['estimated_gross_clone_hours']:,.1f} hours",
        "",
        "By dataset (NEED_CLONE):",
        f"  {'Dataset':<28s} {'Spk':>6s} {'Alloc(h)':>10s} {'Clones':>12s} {'Gap(h)':>10s}",
    ]
    for ds, v in summary["by_dataset"].items():
        lines.append(
            f"  {ds:<28s} {v['need_clone']:>6,} {v['allocated_clone_hours']:>10.1f} "
            f"{v['clones_needed']:>12,} {v['gap_hours']:>10.1f}"
        )
    lines.append("=" * 72)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\n" + "\n".join(lines), flush=True)
    print(f"\nSaved: {out_csv}", flush=True)
    print(f"Saved: {resume_csv}  (NEED_CLONE only, for Step 3 resume)", flush=True)
    print(f"Saved: {json_path}", flush=True)
    print(f"Saved: {txt_path}", flush=True)


if __name__ == "__main__":
    main()
