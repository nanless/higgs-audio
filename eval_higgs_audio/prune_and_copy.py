#!/usr/bin/env python3
"""Delete low-quality Higgs Audio v3 clone audios by CER and SIM thresholds.

Rules:
  DELETE: CER > 0.03  OR  SIM < 0.85
  KEEP:   otherwise (stay in clone dir)

Fast path (default --source disk):
  1. Scan existing clone_*.wav on disk (~3s for ~1M files)
  2. Classify only those paths against eval maps
  3. Parallel delete violations (no stat on missing eval-only paths)

Usage:
    python prune_and_copy.py --dry-run
    python prune_and_copy.py --workers 32
    python prune_and_copy.py --source eval   # slow: iterate all eval DELETE paths
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from postprocess_common import (
    CLONE_SIDECAR_SUFFIXES,
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

_DELETE_DRY = False


def _paths_for_clone(wav_path: str) -> list[str]:
    base = wav_path[: -len(".wav")] if wav_path.endswith(".wav") else str(Path(wav_path).with_suffix(""))
    return [wav_path, *[base + suffix for suffix in CLONE_SIDECAR_SUFFIXES]]


def _delete_one(wav_path: str) -> tuple[int, int]:
    ok = fail = 0
    for fp in _paths_for_clone(wav_path):
        if _DELETE_DRY:
            ok += 1
            continue
        try:
            os.remove(fp)
            ok += 1
        except FileNotFoundError:
            pass
        except OSError as exc:
            print(f"  [DELETE FAIL] {fp}: {exc}", file=sys.stderr)
            fail += 1
    return ok, fail


def _delete_batch(wavs: list[str]) -> tuple[int, int]:
    ok = fail = 0
    for wav in wavs:
        o, f = _delete_one(wav)
        ok += o
        fail += f
    return ok, fail


def _delete_parallel(delete_list: list[str], workers: int, dry: bool) -> tuple[int, int]:
    global _DELETE_DRY
    _DELETE_DRY = dry
    if not delete_list:
        return 0, 0
    if dry or workers <= 1 or len(delete_list) < 200:
        return _delete_batch(delete_list)

    chunk = max(50, len(delete_list) // (workers * 8))
    chunks = [delete_list[i : i + chunk] for i in range(0, len(delete_list), chunk)]
    ok = fail = 0
    done = 0
    report_every = max(1, len(delete_list) // 20)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_delete_batch, c): len(c) for c in chunks}
        for fut in as_completed(futs):
            o, f = fut.result()
            ok += o
            fail += f
            done += futs[fut]
            if done % report_every < futs[fut] or done == len(delete_list):
                pct = 100.0 * done / len(delete_list)
                print(
                    f"  [DELETE] {done:,}/{len(delete_list):,} ({pct:.0f}%) ok={ok:,} fail={fail:,}",
                    file=sys.stderr,
                    flush=True,
                )
    return ok, fail


def _build_maps(
    out_dir: Path, source: str = "sidecar", workers: int = 32
) -> tuple[dict[str, float | None], dict[str, float | None]]:
    """Build {wav: cer} and {wav: sim} maps.

    source="sidecar" (default): read per-clone .cer.json/.sim.json (authoritative, always
    fresh even after prune+regenerate). source="jsonl": legacy aggregate JSONL loaders.
    """
    if source == "sidecar":
        cer_map = load_cer_map_sidecars(out_dir, workers=workers)
        sim_map = load_sim_map_sidecars(out_dir, workers=workers)
        return cer_map, sim_map
    cer_records = load_cer_data(out_dir)
    sim_records = load_sim_data(out_dir)
    cer_map = {r["wav"]: r.get("manual_cer") for r in cer_records}
    sim_map = {r["wav"]: r.get("similarity") for r in sim_records}
    return cer_map, sim_map


def _classify_all(
    wavs: list[str],
    cer_map: dict,
    sim_map: dict,
    max_cer: float,
    min_sim: float,
) -> tuple[list[str], int, Counter, Counter]:
    delete_list: list[str] = []
    keep_count = 0
    stats: Counter = Counter()
    delete_reasons: Counter = Counter()

    for wav in wavs:
        cer = cer_map.get(wav)
        sim = sim_map.get(wav)
        action = classify(cer, sim, max_cer=max_cer, min_sim=min_sim)
        stats[action] += 1
        if action == "DELETE":
            delete_list.append(wav)
            if cer is not None and cer > max_cer:
                delete_reasons["cer"] += 1
            if sim is not None and sim < min_sim:
                delete_reasons["sim"] += 1
            if cer is not None and cer > max_cer and sim is not None and sim < min_sim:
                delete_reasons["both"] += 1
        else:
            keep_count += 1
    return delete_list, keep_count, stats, delete_reasons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_CLONE_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-cer", type=float, default=DEFAULT_MAX_CER)
    parser.add_argument("--min-sim", type=float, default=DEFAULT_MIN_SIM)
    parser.add_argument(
        "--source",
        choices=("disk", "eval"),
        default="disk",
        help="disk=scan existing wavs only (fast); eval=all eval DELETE paths (slow)",
    )
    parser.add_argument("--workers", type=int, default=32, help="Parallel delete workers")
    parser.add_argument("--scan-workers", type=int, default=16, help="Parallel scan workers")
    parser.add_argument(
        "--eval-source",
        choices=("sidecar", "jsonl"),
        default="sidecar",
        help="sidecar=read per-clone .cer.json/.sim.json (authoritative, fresh); jsonl=aggregate details",
    )
    parser.add_argument("--eval-workers", type=int, default=32, help="Parallel workers for reading eval sidecars")
    parser.add_argument("--log", type=Path, default=None)
    args = parser.parse_args()

    if not args.out_dir.is_dir():
        print(f"ERROR: clone root not found: {args.out_dir}", file=sys.stderr)
        sys.exit(1)

    rules = f"DELETE: CER > {args.max_cer} OR SIM < {args.min_sim}; KEEP: otherwise"
    t0 = time.time()
    cer_map, sim_map = _build_maps(args.out_dir, source=args.eval_source, workers=args.eval_workers)
    if not cer_map and not sim_map:
        print("ERROR: no CER/SIM records found", file=sys.stderr)
        sys.exit(1)

    print(f"\n[classify] {rules}", file=sys.stderr, flush=True)
    print(f"[mode] source={args.source} workers={args.workers}", file=sys.stderr)

    if args.source == "disk":
        disk_wavs = scan_clone_wavs(args.out_dir, workers=args.scan_workers)
        delete_list, keep_on_disk, stats, delete_reasons = _classify_all(
            disk_wavs, cer_map, sim_map, args.max_cer, args.min_sim
        )
        keep_expected = sum(
            1
            for wav in set(cer_map) | set(sim_map)
            if classify(cer_map.get(wav), sim_map.get(wav), args.max_cer, args.min_sim) == "KEEP"
        )
        print(
            f"[classify] on disk: DELETE={len(delete_list):,}  KEEP={keep_on_disk:,}  "
            f"(eval expected KEEP={keep_expected:,})",
            file=sys.stderr,
        )
        keep_count = keep_on_disk
    else:
        all_wavs = sorted(set(cer_map.keys()) | set(sim_map.keys()))
        delete_list, keep_count, stats, delete_reasons = _classify_all(
            all_wavs, cer_map, sim_map, args.max_cer, args.min_sim
        )
        delete_list = [w for w in delete_list if os.path.isfile(w)]
        print(
            f"[classify] eval DELETE={stats.get('DELETE', 0):,}  "
            f"existing on disk={len(delete_list):,}  KEEP={keep_count:,}",
            file=sys.stderr,
        )

    tag = "[DRY-RUN] " if args.dry_run else ""
    print(f"\n{tag}DELETE {len(delete_list):,} clones on disk", file=sys.stderr)
    del_ok, del_fail = _delete_parallel(delete_list, workers=args.workers, dry=args.dry_run)
    print(f"  [{tag}DELETE] DONE: ok={del_ok:,} fail={del_fail:,}", file=sys.stderr)

    elapsed = time.time() - t0
    print(f"\nSUMMARY (elapsed {elapsed:.0f}s)", file=sys.stderr)
    print(f"  Rules: {rules}", file=sys.stderr)
    print(f"  Source: {args.source}", file=sys.stderr)
    print(f"  Deleted: {del_ok:,} files ({del_fail} failed)", file=sys.stderr)
    print(f"  Kept on disk: {keep_count:,} clones", file=sys.stderr)
    print(f"  Dry run: {args.dry_run}", file=sys.stderr)

    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        args.log.write_text(
            json.dumps(
                {
                    "dry_run": args.dry_run,
                    "elapsed_s": elapsed,
                    "source": args.source,
                    "workers": args.workers,
                    "deleted_ok": del_ok,
                    "deleted_fail": del_fail,
                    "kept_on_disk": keep_count,
                    "delete_candidates": len(delete_list),
                    "rules": rules,
                    "max_cer": args.max_cer,
                    "min_sim": args.min_sim,
                    "classify": dict(stats),
                    "delete_reasons": dict(delete_reasons),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"[Wrote log] {args.log}", file=sys.stderr)


if __name__ == "__main__":
    main()
