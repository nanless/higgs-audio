#!/usr/bin/env python3
"""Distribution analysis for Higgs Audio v3 TTS clone CER and SIM evaluations.

Reads eval_higgs_cer_details*.jsonl and eval_higgs_sim_details*.jsonl, computes:
  - Overall / by-dataset / by-language statistics
  - CER vs SIM correlation on paired records
  - Prune-rule preview (DELETE / KEEP counts, reasons, quadrants)
  - Threshold matrix for manual tuning

Outputs:
  - Text report to stdout
  - Optional JSON summary via --output-json
  - Optional text file via --output-txt

Usage:
    python analyze_distributions.py
    python analyze_distributions.py --out-dir /path/to/clone_root \\
        --output-json ./eval_distribution_report.json \\
        --output-txt ./eval_distribution_report.txt
"""

from __future__ import annotations

import argparse
import io
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from postprocess_common import (
    DEFAULT_CLONE_ROOT,
    DEFAULT_MAX_CER,
    DEFAULT_MIN_SIM,
    PRUNE_RULES_TEXT,
    analyze_prune_breakdown,
    build_joined_table,
    classify_records,
    compute_stats,
    histogram_bins,
    load_cer_data,
    load_sim_data,
    threshold_matrix,
)


def format_stats_section(label: str, stats: dict) -> str:
    if not stats.get("count"):
        return f"  {label}: (no data)\n"
    lines = [f"  {label}: n={stats['count']}"]
    for key in ["mean", "std", "min", "max", "p10", "p25", "p50", "p75", "p90", "p95", "p99"]:
        val = stats.get(key)
        if val is not None:
            lines.append(f"    {key}: {val:.4f}")
    return "\n".join(lines) + "\n"


def _group_metric(records: list[dict], metric_key: str, group_key: str) -> dict[str, list[float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in records:
        val = row.get(metric_key)
        if val is not None:
            grouped[row.get(group_key, "unknown")].append(val)
    return grouped


def print_text_report(
    cer_records: list[dict],
    sim_records: list[dict],
    table: list[dict],
    out_dir: str,
) -> str:
    buf = io.StringIO()
    buf.write("=" * 72 + "\n")
    buf.write("  Higgs Audio v3 TTS Clone — CER / SIM Distribution Report\n")
    buf.write(f"  Clone root: {out_dir}\n")
    buf.write("=" * 72 + "\n\n")

    # ── CER ──
    buf.write("━" * 72 + "\n")
    buf.write(" 1. Character Error Rate (CER)\n")
    buf.write("━" * 72 + "\n\n")

    manual_vals = [r["manual_cer"] for r in cer_records if r["manual_cer"] is not None]
    llm_vals = [r["llm_cer"] for r in cer_records if r.get("llm_cer") is not None]

    buf.write(f"  Total CER records: {len(cer_records):,}\n")
    buf.write(f"  With manual CER:   {len(manual_vals):,}\n")
    if llm_vals:
        buf.write(f"  With LLM CER:      {len(llm_vals):,}\n")
    buf.write("\n")

    buf.write("  ── Manual CER ──\n")
    buf.write(format_stats_section("Overall", compute_stats(manual_vals)))

    cer_by_ds = _group_metric(cer_records, "manual_cer", "dataset")
    cer_by_lang = _group_metric(cer_records, "manual_cer", "language")
    if cer_by_ds:
        buf.write("\n  ── Manual CER by Dataset ──\n")
        for ds in sorted(cer_by_ds.keys()):
            buf.write(format_stats_section(ds, compute_stats(cer_by_ds[ds])))
    if cer_by_lang:
        buf.write("\n  ── Manual CER by Language ──\n")
        for lang in sorted(cer_by_lang.keys()):
            if len(cer_by_lang[lang]) >= 100:
                buf.write(format_stats_section(lang, compute_stats(cer_by_lang[lang])))

    # ── SIM ──
    buf.write("\n" + "━" * 72 + "\n")
    buf.write(" 2. Speaker Similarity (SIM)\n")
    buf.write("━" * 72 + "\n\n")

    sim_vals = [r["similarity"] for r in sim_records if r["similarity"] is not None]
    buf.write(f"  Total SIM records: {len(sim_records):,}\n")
    buf.write(f"  With similarity:   {len(sim_vals):,}\n\n")
    buf.write(format_stats_section("Overall", compute_stats(sim_vals)))

    sim_by_ds = _group_metric(sim_records, "similarity", "dataset")
    sim_by_lang = _group_metric(sim_records, "similarity", "language")
    if sim_by_ds:
        buf.write("\n  ── Similarity by Dataset ──\n")
        for ds in sorted(sim_by_ds.keys()):
            buf.write(format_stats_section(ds, compute_stats(sim_by_ds[ds])))
    if sim_by_lang:
        buf.write("\n  ── Similarity by Language ──\n")
        for lang in sorted(sim_by_lang.keys()):
            if len(sim_by_lang[lang]) >= 100:
                buf.write(format_stats_section(lang, compute_stats(sim_by_lang[lang])))

    # ── Correlation ──
    buf.write("\n" + "━" * 72 + "\n")
    buf.write(" 3. CER vs SIM Correlation\n")
    buf.write("━" * 72 + "\n\n")

    paired = [r for r in table if r.get("manual_cer") is not None and r.get("similarity") is not None]
    buf.write(f"  Paired records (CER + SIM): {len(paired):,}\n")
    buf.write(
        f"  CER-only records:           {sum(1 for r in table if r.get('manual_cer') is not None and r.get('similarity') is None):,}\n"
    )
    buf.write(
        f"  SIM-only records:           {sum(1 for r in table if r.get('similarity') is not None and r.get('manual_cer') is None):,}\n\n"
    )

    if len(paired) >= 3:
        cer_arr = np.array([r["manual_cer"] for r in paired])
        sim_arr = np.array([r["similarity"] for r in paired])
        r_val = float(np.corrcoef(cer_arr, sim_arr)[0, 1])
        buf.write(f"  Pearson r (CER vs SIM): {r_val:.4f}\n")
        buf.write("  (negative r means higher SIM tends to co-occur with lower CER)\n")

    # ── Prune preview ──
    buf.write("\n" + "━" * 72 + "\n")
    buf.write(f" 4. Prune Rule Analysis (CER > {DEFAULT_MAX_CER} OR SIM < {DEFAULT_MIN_SIM} → DELETE)\n")
    buf.write("━" * 72 + "\n\n")
    buf.write(f"  Rules: {PRUNE_RULES_TEXT}\n\n")

    breakdown = analyze_prune_breakdown(table)
    buf.write(f"  Total clones:  {breakdown['total']:>12,}\n")
    buf.write(f"  DELETE:        {breakdown['delete']:>12,}  ({breakdown['delete_pct']:.2f}%)\n")
    buf.write(f"  KEEP:          {breakdown['keep']:>12,}  ({breakdown['keep_pct']:.2f}%)\n\n")

    reasons = breakdown["delete_reasons"]
    buf.write("  ── DELETE trigger breakdown (paired records) ──\n")
    buf.write(f"    CER > {DEFAULT_MAX_CER} only:  {reasons['cer_only']:>10,}\n")
    buf.write(f"    SIM < {DEFAULT_MIN_SIM} only:  {reasons['sim_only']:>10,}\n")
    buf.write(f"    Both failed:      {reasons['both']:>10,}\n")
    buf.write(f"    ({reasons['note']})\n\n")

    quad = breakdown["quadrants"]
    buf.write("  ── Quadrant view (records with both CER + SIM) ──\n")
    buf.write(
        f"    KEEP zone  (CER≤{DEFAULT_MAX_CER} & SIM≥{DEFAULT_MIN_SIM}): {quad['keep_ok']:>10,}  "
        f"({100 * quad['keep_ok'] / len(paired):.2f}% of paired)\n"
        if paired
        else f"    KEEP zone  (CER≤{DEFAULT_MAX_CER} & SIM≥{DEFAULT_MIN_SIM}): {quad['keep_ok']:>10,}\n"
    )
    buf.write(f"    DELETE / CER only:                  {quad['delete_cer_only']:>10,}\n")
    buf.write(f"    DELETE / SIM only:                  {quad['delete_sim_only']:>10,}\n")
    buf.write(f"    DELETE / both:                      {quad['delete_both']:>10,}\n\n")

    buf.write("  ── Metric stats on DELETE vs KEEP subsets ──\n")
    d_cer = breakdown["delete_subset"]["cer"]
    d_sim = breakdown["delete_subset"]["sim"]
    k_cer = breakdown["keep_subset"]["cer"]
    k_sim = breakdown["keep_subset"]["sim"]
    buf.write(
        f"    DELETE  CER mean={d_cer['mean']:.4f}  p50={d_cer['p50']:.4f}  "
        f"SIM mean={d_sim['mean']:.4f}  p50={d_sim['p50']:.4f}\n"
    )
    buf.write(
        f"    KEEP    CER mean={k_cer['mean']:.4f}  p50={k_cer['p50']:.4f}  "
        f"SIM mean={k_sim['mean']:.4f}  p50={k_sim['p50']:.4f}\n\n"
    )

    preview = classify_records(table)
    buf.write(f"  Missing CER: {preview['missing_cer']:,}  Missing SIM: {preview['missing_sim']:,}\n\n")

    buf.write("  ── By Dataset (DELETE / KEEP / rate) ──\n")
    buf.write(
        f"  {'Dataset':<24s} {'Total':>10s} {'DELETE':>10s} {'KEEP':>10s} "
        f"{'Del%':>7s} {'CER↓':>8s} {'SIM↓':>8s} {'Both':>8s}\n"
    )
    buf.write(f"  {'─' * 24} {'─' * 10} {'─' * 10} {'─' * 10} {'─' * 7} {'─' * 8} {'─' * 8} {'─' * 8}\n")
    for row in breakdown["by_dataset"]:
        buf.write(
            f"  {row['dataset']:<24s} {row['total']:>10,} {row['delete']:>10,} "
            f"{row['keep']:>10,} {row['delete_pct']:>6.1f}% "
            f"{row['delete_cer_only']:>8,} {row['delete_sim_only']:>8,} "
            f"{row['delete_both']:>8,}\n"
        )

    if breakdown["by_language"]:
        buf.write("\n  ── By Language (DELETE rate, n≥100) ──\n")
        buf.write(f"  {'Language':<12s} {'Total':>10s} {'DELETE':>10s} {'KEEP':>10s} {'Del%':>7s}\n")
        buf.write(f"  {'─' * 12} {'─' * 10} {'─' * 10} {'─' * 10} {'─' * 7}\n")
        for row in breakdown["by_language"]:
            buf.write(
                f"  {row['language']:<12s} {row['total']:>10,} {row['delete']:>10,} "
                f"{row['keep']:>10,} {row['delete_pct']:>6.1f}%\n"
            )

    # ── Threshold matrix ──
    buf.write("\n" + "━" * 72 + "\n")
    buf.write(" 5. Threshold Matrix (CER <= X AND SIM > Y)\n")
    buf.write("━" * 72 + "\n\n")
    matrix = threshold_matrix(table)
    buf.write(f"  {'CER<=':>7s} {'SIM>':>6s} {'Count':>12s} {'Pct':>8s}\n")
    buf.write(f"  {'─' * 7} {'─' * 6} {'─' * 12} {'─' * 8}\n")
    for row in matrix:
        buf.write(f"  {row['cer_max']:>7.2f} {row['sim_min']:>6.2f} {row['count']:>12,} {row['pct']:>7.2f}%\n")

    # ── Summary table ──
    buf.write("\n" + "━" * 72 + "\n")
    buf.write(" 6. Per-Dataset Summary\n")
    buf.write("━" * 72 + "\n\n")

    all_ds = set(cer_by_ds.keys()) | set(sim_by_ds.keys())
    max_name = max((len(ds) for ds in all_ds), default=10)
    header = (
        f"  {'Dataset':<{max_name}s} {'CER_n':>8s} {'CER_mean':>9s} "
        f"{'SIM_n':>8s} {'SIM_mean':>9s} {'DELETE':>8s} {'KEEP':>8s}"
    )
    buf.write(header + "\n")
    buf.write(f"  {'─' * max_name} {'─' * 8} {'─' * 9} {'─' * 8} {'─' * 9} {'─' * 8} {'─' * 8}\n")

    for ds in sorted(all_ds):
        c_stats = compute_stats(cer_by_ds.get(ds, []))
        s_stats = compute_stats(sim_by_ds.get(ds, []))
        ds_break = next((r for r in breakdown["by_dataset"] if r["dataset"] == ds), {})

        def _fmt(stats: dict, key: str) -> str:
            val = stats.get(key) if stats else None
            return f"{val:>9.4f}" if val is not None else f"{'-':>9s}"

        buf.write(
            f"  {ds:<{max_name}s} "
            f"{c_stats.get('count', 0):>8,d} {_fmt(c_stats, 'mean')}  "
            f"{s_stats.get('count', 0):>8,d} {_fmt(s_stats, 'mean')}  "
            f"{ds_break.get('delete', 0):>8,d} {ds_break.get('keep', 0):>8,d}\n"
        )

    buf.write("\n" + "=" * 72 + "\n")
    buf.write("  Report Complete\n")
    buf.write("=" * 72 + "\n")
    return buf.getvalue()


def build_json_report(cer_records: list[dict], sim_records: list[dict], table: list[dict], out_dir: str) -> dict:
    def _metric_report(records: list[dict], metric_key: str) -> dict:
        values = [r[metric_key] for r in records if r.get(metric_key) is not None]
        overall = compute_stats(values)
        overall["histogram"] = histogram_bins(values)
        by_dataset = {
            ds: compute_stats(vals) for ds, vals in sorted(_group_metric(records, metric_key, "dataset").items())
        }
        by_language = {
            lang: compute_stats(vals)
            for lang, vals in sorted(_group_metric(records, metric_key, "language").items())
            if len(vals) >= 100
        }
        return {"overall": overall, "by_dataset": by_dataset, "by_language": by_language}

    report: dict = {
        "clone_root": out_dir,
        "cer": _metric_report(cer_records, "manual_cer"),
        "sim": _metric_report(sim_records, "similarity"),
        "prune_preview": classify_records(table),
        "prune_breakdown": analyze_prune_breakdown(table),
        "threshold_matrix": threshold_matrix(table),
        "counts": {
            "total_cer": len(cer_records),
            "total_sim": len(sim_records),
            "total_joined": len(table),
        },
    }

    llm_vals = [r["llm_cer"] for r in cer_records if r.get("llm_cer") is not None]
    if llm_vals:
        report["cer_llm"] = _metric_report(
            [r for r in cer_records if r.get("llm_cer") is not None],
            "llm_cer",
        )

    paired = [r for r in table if r.get("manual_cer") is not None and r.get("similarity") is not None]
    if len(paired) >= 3:
        cer_arr = np.array([r["manual_cer"] for r in paired])
        sim_arr = np.array([r["similarity"] for r in paired])
        report["correlations"] = {
            "cer_vs_similarity": float(np.corrcoef(cer_arr, sim_arr)[0, 1]),
            "paired_count": len(paired),
        }

    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_CLONE_ROOT)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-txt", type=Path, default=None)
    parser.add_argument("--sample-size", type=int, default=None, help="Random sample for quick test")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = args.out_dir
    if not out.is_dir():
        print(f"ERROR: clone root not found: {out}", file=sys.stderr)
        sys.exit(1)

    print("=== Loading CER data ===", file=sys.stderr)
    cer_records = load_cer_data(out)

    print("=== Loading SIM data ===", file=sys.stderr)
    sim_records = load_sim_data(out)

    if args.sample_size:
        rng = random.Random(args.seed)
        if args.sample_size < len(cer_records):
            cer_records = rng.sample(cer_records, args.sample_size)
        if args.sample_size < len(sim_records):
            sim_records = rng.sample(sim_records, args.sample_size)

    print("=== Building joined table ===", file=sys.stderr)
    table = build_joined_table(cer_records, sim_records)

    print("=== Generating reports ===", file=sys.stderr)
    text_report = print_text_report(cer_records, sim_records, table, str(out))
    print(text_report)

    if args.output_txt:
        args.output_txt.parent.mkdir(parents=True, exist_ok=True)
        args.output_txt.write_text(text_report, encoding="utf-8")
        print(f"[Wrote] {args.output_txt}", file=sys.stderr)

    if args.output_json:
        json_report = build_json_report(cer_records, sim_records, table, str(out))
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(json_report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"[Wrote] {args.output_json}", file=sys.stderr)


if __name__ == "__main__":
    main()
