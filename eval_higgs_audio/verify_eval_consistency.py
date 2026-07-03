#!/usr/bin/env python3
"""Verify CER/SIM root jsonl aggregates match per-audio sidecar JSON files."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from postprocess_common import DEFAULT_CLONE_ROOT, iter_jsonl

SKIP_DIRS = {"logs", "__pycache__", "eval_sim_embedding_cache"}

CER_JSONL_FIELDS = ("manual_cer", "dataset", "speaker_id", "asr_language")
CER_SIDECAR_FIELDS = ("manual_cer", "dataset", "speaker_id", "asr_language")
SIM_JSONL_FIELDS = ("similarity", "dataset", "speaker_id", "ref_audio")
SIM_SIDECAR_FIELDS = ("similarity", "dataset", "speaker_id", "ref_audio")


def _norm_path(p: str) -> str:
    return os.path.normpath(p)


def _float_eq(a, b, tol: float = 1e-9) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return abs(float(a) - float(b)) <= tol


def _str_eq(a, b) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return str(a) == str(b)


def load_cer_jsonl(out_dir: Path) -> dict[str, dict]:
    files = sorted(out_dir.glob("eval_higgs_cer_details*.jsonl"))
    data: dict[str, dict] = {}
    dupes = 0
    t0 = time.time()
    for fp in files:
        for r in iter_jsonl(fp):
            wav = _norm_path(r.get("wav") or r.get("wav_path") or "")
            if not wav:
                continue
            if wav in data:
                dupes += 1
                old = data[wav]
                for key in CER_JSONL_FIELDS:
                    if key == "manual_cer":
                        if not _float_eq(old.get(key), r.get(key)):
                            data[wav]["_jsonl_internal_conflict"] = True
                    elif old.get(key) != r.get(key):
                        data[wav]["_jsonl_internal_conflict"] = True
            row = {k: r.get(k) for k in CER_JSONL_FIELDS}
            row["_source_file"] = fp.name
            data[wav] = row
    print(
        f"[CER jsonl] {len(data):,} unique wavs from {len(files)} files "
        f"(dup lines={dupes}) in {time.time() - t0:.1f}s",
        file=sys.stderr,
    )
    return data


def load_sim_jsonl(out_dir: Path) -> dict[str, dict]:
    files = sorted(out_dir.glob("eval_higgs_sim_details*.jsonl"))
    data: dict[str, dict] = {}
    dupes = 0
    t0 = time.time()
    for fp in files:
        for r in iter_jsonl(fp):
            wav = _norm_path(r.get("cloned_audio") or "")
            if not wav:
                continue
            if wav in data:
                dupes += 1
            row = {k: r.get(k) for k in SIM_JSONL_FIELDS}
            row["_source_file"] = fp.name
            data[wav] = row
    print(
        f"[SIM jsonl] {len(data):,} unique wavs from {len(files)} files "
        f"(dup lines={dupes}) in {time.time() - t0:.1f}s",
        file=sys.stderr,
    )
    return data


def _scan_cer_dir(root: str) -> tuple[list[tuple[str, dict]], list[str]]:
    out: list[tuple[str, dict]] = []
    errs: list[str] = []
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in files:
            if not name.endswith(".cer.json"):
                continue
            fp = os.path.join(dirpath, name)
            try:
                with open(fp, encoding="utf-8") as fh:
                    r = json.load(fh)
            except (json.JSONDecodeError, OSError) as exc:
                errs.append(f"bad cer {fp}: {exc}")
                continue
            wav = _norm_path(r.get("wav_path") or r.get("wav") or "")
            if not wav:
                errs.append(f"cer missing wav_path: {fp}")
                continue
            out.append((wav, {k: r.get(k) for k in CER_SIDECAR_FIELDS}))
    return out, errs


def _scan_sim_dir(root: str) -> tuple[list[tuple[str, dict]], list[str]]:
    out: list[tuple[str, dict]] = []
    errs: list[str] = []
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in files:
            if not name.endswith(".sim.json"):
                continue
            fp = os.path.join(dirpath, name)
            try:
                with open(fp, encoding="utf-8") as fh:
                    r = json.load(fh)
            except (json.JSONDecodeError, OSError) as exc:
                errs.append(f"bad sim {fp}: {exc}")
                continue
            wav = _norm_path(r.get("cloned_audio") or "")
            if not wav:
                errs.append(f"sim missing cloned_audio: {fp}")
                continue
            ref = r.get("ref_audio")
            out.append(
                (
                    wav,
                    {k: r.get(k) for k in SIM_SIDECAR_FIELDS} | {"ref_audio": _norm_path(ref) if ref else ref},
                )
            )
    return out, errs


def scan_cer_sidecars(out_dir: Path, workers: int = 8) -> tuple[dict[str, dict], list[str]]:
    subdirs = [str(p) for p in out_dir.iterdir() if p.is_dir() and p.name not in SKIP_DIRS]
    t0 = time.time()
    data: dict[str, dict] = {}
    errors: list[str] = []
    dupes = 0

    if len(subdirs) > 1 and workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_scan_cer_dir, sd) for sd in subdirs]
            for fut in as_completed(futs):
                batch, errs = fut.result()
                errors.extend(errs)
                for wav, row in batch:
                    if wav in data:
                        dupes += 1
                    data[wav] = row
    else:
        for sd in subdirs:
            batch, errs = _scan_cer_dir(sd)
            errors.extend(errs)
            for wav, row in batch:
                if wav in data:
                    dupes += 1
                data[wav] = row

    print(
        f"[CER sidecar] {len(data):,} records (dup={dupes}) in {time.time() - t0:.1f}s",
        file=sys.stderr,
    )
    return data, errors


def scan_sim_sidecars(out_dir: Path, workers: int = 8) -> tuple[dict[str, dict], list[str]]:
    subdirs = [str(p) for p in out_dir.iterdir() if p.is_dir() and p.name not in SKIP_DIRS]
    t0 = time.time()
    data: dict[str, dict] = {}
    errors: list[str] = []
    dupes = 0

    if len(subdirs) > 1 and workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_scan_sim_dir, sd) for sd in subdirs]
            for fut in as_completed(futs):
                batch, errs = fut.result()
                errors.extend(errs)
                for wav, row in batch:
                    if wav in data:
                        dupes += 1
                    data[wav] = row
    else:
        for sd in subdirs:
            batch, errs = _scan_sim_dir(sd)
            errors.extend(errs)
            for wav, row in batch:
                if wav in data:
                    dupes += 1
                data[wav] = row

    print(
        f"[SIM sidecar] {len(data):,} records (dup={dupes}) in {time.time() - t0:.1f}s",
        file=sys.stderr,
    )
    return data, errors


def compare_maps(
    name: str,
    jsonl_map: dict[str, dict],
    sidecar_map: dict[str, dict],
    fields: tuple[str, ...],
    float_fields: set[str],
    max_examples: int = 20,
) -> dict:
    only_jsonl = sorted(set(jsonl_map) - set(sidecar_map))
    only_sidecar = sorted(set(sidecar_map) - set(jsonl_map))
    common = set(jsonl_map) & set(sidecar_map)

    mismatches: list[dict] = []
    field_mismatch_counts: Counter = Counter()
    value_mismatch_count = 0

    for wav in common:
        j = jsonl_map[wav]
        s = sidecar_map[wav]
        diff_fields = []
        for key in fields:
            jv, sv = j.get(key), s.get(key)
            if key in float_fields:
                if not _float_eq(jv, sv):
                    diff_fields.append(key)
            elif key == "ref_audio":
                if _norm_path(str(jv or "")) != _norm_path(str(sv or "")):
                    diff_fields.append(key)
            elif not _str_eq(jv, sv):
                diff_fields.append(key)
        if diff_fields:
            value_mismatch_count += 1
            for f in diff_fields:
                field_mismatch_counts[f] += 1
            if len(mismatches) < max_examples:
                mismatches.append(
                    {
                        "wav": wav,
                        "fields": diff_fields,
                        "jsonl": {k: j.get(k) for k in diff_fields},
                        "sidecar": {k: s.get(k) for k in diff_fields},
                    }
                )

    return {
        "metric": name,
        "jsonl_count": len(jsonl_map),
        "sidecar_count": len(sidecar_map),
        "common_count": len(common),
        "only_in_jsonl": len(only_jsonl),
        "only_in_sidecar": len(only_sidecar),
        "value_mismatches": value_mismatch_count,
        "field_mismatch_counts": dict(field_mismatch_counts),
        "examples_only_jsonl": only_jsonl[:max_examples],
        "examples_only_sidecar": only_sidecar[:max_examples],
        "examples_value_mismatch": mismatches,
        "fully_consistent": (len(only_jsonl) == 0 and len(only_sidecar) == 0 and not field_mismatch_counts),
    }


def print_report(report: dict) -> None:
    print("\n" + "=" * 72)
    print(f"  {report['metric']} Consistency Report")
    print("=" * 72)
    print(f"  jsonl records:     {report['jsonl_count']:>12,}")
    print(f"  sidecar records:   {report['sidecar_count']:>12,}")
    print(f"  in both:           {report['common_count']:>12,}")
    print(f"  only in jsonl:     {report['only_in_jsonl']:>12,}")
    print(f"  only in sidecar:   {report['only_in_sidecar']:>12,}")
    print(f"  value mismatches:  {report['value_mismatches']:>12,}")
    print(f"  fully consistent:  {report['fully_consistent']}")
    if report["field_mismatch_counts"]:
        print("  field mismatch breakdown:")
        for k, v in sorted(report["field_mismatch_counts"].items()):
            print(f"    {k}: {v:,}")
    if report["examples_only_jsonl"]:
        print("\n  examples only in jsonl (up to 5):")
        for p in report["examples_only_jsonl"][:5]:
            print(f"    {p}")
    if report["examples_only_sidecar"]:
        print("\n  examples only in sidecar (up to 5):")
        for p in report["examples_only_sidecar"][:5]:
            print(f"    {p}")
    if report["examples_value_mismatch"]:
        print("\n  examples value mismatch (up to 5):")
        for ex in report["examples_value_mismatch"][:5]:
            print(f"    wav: {ex['wav']}")
            for f in ex["fields"]:
                print(f"      {f}: jsonl={ex['jsonl'][f]!r} sidecar={ex['sidecar'][f]!r}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_CLONE_ROOT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cer-only", action="store_true")
    parser.add_argument("--sim-only", action="store_true")
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    out = args.out_dir
    if not out.is_dir():
        print(f"ERROR: {out} not found", file=sys.stderr)
        sys.exit(1)

    results = {}
    t0 = time.time()

    if not args.sim_only:
        print("=== CER ===", file=sys.stderr)
        cer_jsonl = load_cer_jsonl(out)
        cer_sidecar, cer_scan_errors = scan_cer_sidecars(out, workers=args.workers)
        cer_report = compare_maps(
            "CER",
            cer_jsonl,
            cer_sidecar,
            CER_JSONL_FIELDS,
            float_fields={"manual_cer"},
        )
        cer_report["sidecar_scan_errors"] = len(cer_scan_errors)
        cer_report["sidecar_scan_error_examples"] = cer_scan_errors[:10]
        print_report(cer_report)
        results["cer"] = cer_report

    if not args.cer_only:
        print("\n=== SIM ===", file=sys.stderr)
        sim_jsonl = load_sim_jsonl(out)
        # normalize ref_audio in jsonl
        for wav, row in sim_jsonl.items():
            ref = row.get("ref_audio")
            if ref:
                row["ref_audio"] = _norm_path(ref)
        sim_sidecar, sim_scan_errors = scan_sim_sidecars(out, workers=args.workers)
        sim_report = compare_maps(
            "SIM",
            sim_jsonl,
            sim_sidecar,
            SIM_JSONL_FIELDS,
            float_fields={"similarity"},
        )
        sim_report["sidecar_scan_errors"] = len(sim_scan_errors)
        sim_report["sidecar_scan_error_examples"] = sim_scan_errors[:10]
        print_report(sim_report)
        results["sim"] = sim_report

    elapsed = time.time() - t0
    all_ok = all(r.get("fully_consistent") for r in results.values())
    print(f"\nTotal elapsed: {elapsed:.1f}s")
    print(f"Overall: {'PASS — fully consistent' if all_ok else 'FAIL — differences found'}")

    if args.output_json:
        args.output_json.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[Wrote] {args.output_json}")

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
