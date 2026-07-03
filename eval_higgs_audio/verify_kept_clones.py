#!/usr/bin/env python3
"""Verify remaining clone wavs on disk meet KEEP rules and sum total duration.

KEEP: CER <= max_cer AND SIM >= min_sim
DELETE: CER > max_cer OR SIM < min_sim
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time
import wave
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from postprocess_common import (
    DEFAULT_CLONE_ROOT,
    DEFAULT_MAX_CER,
    DEFAULT_MIN_SIM,
    classify,
    load_cer_data,
    load_cer_map_sidecars,
    load_sim_data,
    load_sim_map_sidecars,
    scan_clone_wavs,
)


def _wav_duration_sec(path: str) -> float | None:
    """Fast duration from WAV header; fallback to wave module."""
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
    try:
        with wave.open(path, "rb") as wf:
            return wf.getnframes() / float(wf.getframerate())
    except (wave.Error, OSError):
        return None


def _duration_batch(paths: list[str]) -> tuple[float, int, int]:
    total = 0.0
    ok = 0
    fail = 0
    for p in paths:
        d = _wav_duration_sec(p)
        if d is None:
            fail += 1
        else:
            total += d
            ok += 1
    return total, ok, fail


def sum_durations(wavs: list[str], workers: int = 16) -> dict:
    t0 = time.time()
    if not wavs:
        return {"total_sec": 0.0, "total_hours": 0.0, "count_ok": 0, "count_fail": 0}
    chunk = max(500, len(wavs) // (workers * 4))
    chunks = [wavs[i : i + chunk] for i in range(0, len(wavs), chunk)]
    total_sec = 0.0
    ok = fail = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_duration_batch, c) for c in chunks]
        for fut in as_completed(futs):
            sec, o, f = fut.result()
            total_sec += sec
            ok += o
            fail += f
    print(f"[duration] {ok:,} wavs summed in {time.time() - t0:.1f}s", file=sys.stderr)
    return {
        "total_sec": total_sec,
        "total_hours": total_sec / 3600.0,
        "count_ok": ok,
        "count_fail": fail,
    }


def verify(wavs: list[str], cer_map: dict, sim_map: dict, max_cer: float, min_sim: float) -> dict:
    violations: list[dict] = []
    counts = Counter()
    by_dataset: dict[str, Counter] = defaultdict(Counter)
    missing_eval = 0

    for wav in wavs:
        cer = cer_map.get(wav)
        sim = sim_map.get(wav)
        if cer is None and sim is None:
            missing_eval += 1
            counts["missing_eval"] += 1
        action = classify(cer, sim, max_cer=max_cer, min_sim=min_sim)
        counts[action] += 1
        rel = wav.split(os.sep)
        ds = rel[-3] if len(rel) >= 3 else "unknown"
        by_dataset[ds][action] += 1
        if action == "DELETE":
            reason = []
            if cer is not None and cer > max_cer:
                reason.append(f"cer={cer:.4f}")
            if sim is not None and sim < min_sim:
                reason.append(f"sim={sim:.4f}")
            if len(violations) < 30:
                violations.append({"wav": wav, "cer": cer, "sim": sim, "reason": ",".join(reason) or "unknown"})

    return {
        "total_on_disk": len(wavs),
        "counts": dict(counts),
        "missing_eval": missing_eval,
        "violations_sample": violations,
        "violation_count": counts.get("DELETE", 0),
        "keep_count": counts.get("KEEP", 0),
        "by_dataset": {k: dict(v) for k, v in sorted(by_dataset.items())},
        "rules": f"KEEP: CER <= {max_cer} AND SIM >= {min_sim}",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_CLONE_ROOT)
    parser.add_argument("--max-cer", type=float, default=DEFAULT_MAX_CER)
    parser.add_argument("--min-sim", type=float, default=DEFAULT_MIN_SIM)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument(
        "--eval-source",
        choices=("sidecar", "jsonl"),
        default="sidecar",
        help="sidecar=read per-clone .cer.json/.sim.json (authoritative, fresh); jsonl=aggregate details",
    )
    parser.add_argument("--eval-workers", type=int, default=32, help="Parallel workers for reading eval sidecars")
    args = parser.parse_args()

    if not args.out_dir.is_dir():
        print(f"ERROR: {args.out_dir} not found", file=sys.stderr)
        sys.exit(1)

    print("=== Loading eval maps ===", file=sys.stderr)
    if args.eval_source == "sidecar":
        cer_map = load_cer_map_sidecars(args.out_dir, workers=args.eval_workers)
        sim_map = load_sim_map_sidecars(args.out_dir, workers=args.eval_workers)
    else:
        cer_records = load_cer_data(args.out_dir)
        sim_records = load_sim_data(args.out_dir)
        cer_map = {r["wav"]: r.get("manual_cer") for r in cer_records}
        sim_map = {r["wav"]: r.get("similarity") for r in sim_records}

    print("=== Scanning disk ===", file=sys.stderr)
    wavs = scan_clone_wavs(args.out_dir, workers=args.workers)

    print("=== Verifying KEEP rules ===", file=sys.stderr)
    report = verify(wavs, cer_map, sim_map, args.max_cer, args.min_sim)

    print("=== Summing duration ===", file=sys.stderr)
    dur = sum_durations(wavs, workers=args.workers)
    report["duration"] = dur

    # Expected keep from eval (may differ if prune incomplete)
    expected_keep = sum(
        1
        for wav in set(cer_map.keys()) | set(sim_map.keys())
        if classify(cer_map.get(wav), sim_map.get(wav), args.max_cer, args.min_sim) == "KEEP"
    )
    report["expected_keep_from_eval"] = expected_keep
    report["prune_complete_guess"] = len(wavs) <= expected_keep * 1.01

    print("\n" + "=" * 72)
    print("  Remaining Clone Verification")
    print("=" * 72)
    print(f"  Rules: {report['rules']}")
    print(f"  Clone wavs on disk:     {report['total_on_disk']:>12,}")
    print(f"  Expected KEEP (eval):   {expected_keep:>12,}")
    print(f"  PASS (KEEP on disk):    {report['keep_count']:>12,}")
    print(f"  FAIL (should delete):   {report['violation_count']:>12,}")
    print(f"  Missing eval record:    {report['missing_eval']:>12,}")
    print(f"  Total duration:         {dur['total_hours']:>12,.2f} hours  ({dur['total_sec']:,.0f} sec)")
    print(f"  Avg per clone:          {(dur['total_sec'] / dur['count_ok'] if dur['count_ok'] else 0):.2f} sec")
    fully_ok = report["violation_count"] == 0 and report["missing_eval"] == 0
    print(f"  Fully compliant:        {fully_ok}")

    if report["violation_count"]:
        print("\n  Sample violations (up to 10):")
        for v in report["violations_sample"][:10]:
            print(f"    {v['wav']}")
            print(f"      cer={v['cer']} sim={v['sim']} ({v['reason']})")

    if report["total_on_disk"] > expected_keep * 1.05:
        print(
            f"\n  NOTE: disk count ({report['total_on_disk']:,}) >> expected KEEP ({expected_keep:,}). "
            "Prune may be incomplete."
        )

    print("\n  By dataset (on disk):")
    print(f"  {'Dataset':<28s} {'KEEP':>10s} {'DELETE':>10s} {'miss':>8s}")
    for ds, c in report["by_dataset"].items():
        print(f"  {ds:<28s} {c.get('KEEP', 0):>10,} {c.get('DELETE', 0):>10,} {c.get('missing_eval', 0):>8,}")
    print("=" * 72)

    if args.output_json:
        args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[Wrote] {args.output_json}", file=sys.stderr)

    sys.exit(0 if fully_ok and report["total_on_disk"] <= expected_keep * 1.01 else 1)


if __name__ == "__main__":
    main()
