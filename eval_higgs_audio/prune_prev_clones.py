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
#!/usr/bin/env python3
"""Filter previously-cloned directories to the NEW quality bar (raw cosine + CER).

Uses the ALREADY-computed sidecars (no GPU, no re-embedding, no reference files needed):
  - .sim.json `similarity`  (these were produced with the (cos+1)/2 mapping)
  - .cer.json `manual_cer`

Since mapped = (raw + 1) / 2 EXACTLY, raw cosine is recovered losslessly as
  raw = 2 * similarity - 1        (for --sim-scale mapped, the default for old dirs)

DELETE a clone (wav + all sidecars) if:
      raw_cos < --min-sim-raw   (default 0.8)
   OR manual_cer > --max-cer    (default 0.03, when --apply-cer)
A metric that is missing does not trigger deletion (kept + counted).

Default is DRY-RUN (no deletion). Pass --execute to actually delete. Each directory is
scanned and reported independently. Handles both naming schemes (clone_NNNN.wav and
omnivoice {utt}_clone_text_NNN.wav) by enumerating the main .json sidecars.

Usage:
    # preview (no deletion)
    python prune_prev_clones.py --dirs /path/audio_omnivoice_clone /path/audio_higgs_audio_v3_tts_clone \
                                       /path/audio_higgs_audio_v3_tts_clone_2 /path/audio_higgs_audio_v3_tts_clone_3
    # actually delete
    python prune_prev_clones.py --dirs ... --execute
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

SKIP_DIRS = frozenset({"logs", "__pycache__", "ref", "eval_sim_embedding_cache"})
DELETE_SUFFIXES = (".wav", ".json", ".cer.json", ".sim.json", ".mos.json", ".eval.json")
PERCENTILES = [1, 5, 10, 25, 50, 75, 90, 95, 99]


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


def _read_val(json_path: str, key: str):
    try:
        with open(json_path, encoding="utf-8") as f:
            return json.load(f).get(key)
    except (OSError, json.JSONDecodeError, TypeError):
        return None


def _scan_unit(task: tuple) -> list:
    """Return list of (wav, stem, raw_sim|None, cer|None, dataset, dur) for one subtree."""
    root, unit, sim_scale = task
    out = []
    for dirpath, dirs, files in os.walk(unit):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in files:
            if not _is_main_sidecar(name):
                continue
            stem = os.path.join(dirpath, name[:-5])
            wav = stem + ".wav"
            if not os.path.isfile(wav):
                continue
            sim = _read_val(stem + ".sim.json", "similarity") if os.path.isfile(stem + ".sim.json") else None
            if sim is not None:
                raw = float(sim) if sim_scale == "raw" else (2.0 * float(sim) - 1.0)
            else:
                raw = None
            cer = _read_val(stem + ".cer.json", "manual_cer") if os.path.isfile(stem + ".cer.json") else None
            cer = float(cer) if cer is not None else None
            dataset = os.path.relpath(wav, root).split(os.sep)[0]
            out.append((wav, stem, raw, cer, dataset, _wav_dur(wav)))
    return out


def scan_dir(root: str, sim_scale: str, scan_workers: int) -> list:
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
    t0 = time.time()
    recs: list = []
    if len(units) > 1 and scan_workers > 1:
        with ProcessPoolExecutor(max_workers=min(scan_workers, len(units))) as ex:
            for batch in ex.map(_scan_unit, [(root, u, sim_scale) for u in units], chunksize=8):
                recs.extend(batch)
    else:
        for u in units:
            recs.extend(_scan_unit((root, u, sim_scale)))
    print(
        f"[scan] {os.path.basename(root)}: {len(recs):,} clones, {len(units):,} units in {time.time() - t0:.1f}s",
        flush=True,
    )
    return recs


def _delete_batch(stems: list) -> tuple:
    ok = fail = 0
    for stem in stems:
        for suf in DELETE_SUFFIXES:
            fp = stem + suf
            try:
                os.remove(fp)
                ok += 1
            except FileNotFoundError:
                pass
            except OSError as e:
                print(f"  [DEL FAIL] {fp}: {e}", file=sys.stderr)
                fail += 1
    return ok, fail


def _stats(a: np.ndarray) -> dict:
    if a.size == 0:
        return {"count": 0}
    return {
        "count": int(a.size),
        "mean": float(a.mean()),
        "min": float(a.min()),
        "max": float(a.max()),
        "percentiles": {str(p): float(np.percentile(a, p)) for p in PERCENTILES},
    }


def process_dir(root: str, args) -> dict:
    name = os.path.basename(os.path.normpath(root))
    print(f"\n########## {name} ##########", flush=True)
    if not os.path.isdir(root):
        print(f"  跳过: 不存在 {root}", flush=True)
        return {"name": name, "error": "missing"}

    recs = scan_dir(root, args.sim_scale, args.scan_workers)
    total = len(recs)
    del_stems: list = []
    n_del_sim = n_del_cer = n_del_both = 0
    n_no_sim = n_no_cer = 0
    raw_all, raw_kept = [], []
    dur_total = dur_del = 0.0
    by_ds = defaultdict(lambda: [0, 0])  # dataset -> [total, deleted]

    for wav, stem, raw, cer, ds, dur in recs:
        dur_total += dur
        by_ds[ds][0] += 1
        if raw is None:
            n_no_sim += 1
        else:
            raw_all.append(raw)
        if cer is None:
            n_no_cer += 1
        bad_sim = raw is not None and raw < args.min_sim_raw
        bad_cer = args.apply_cer and cer is not None and cer > args.max_cer
        if bad_sim or bad_cer:
            del_stems.append(stem)
            dur_del += dur
            by_ds[ds][1] += 1
            n_del_sim += int(bad_sim)
            n_del_cer += int(bad_cer)
            n_del_both += int(bad_sim and bad_cer)
        else:
            if raw is not None:
                raw_kept.append(raw)

    rep = {
        "name": name,
        "total": total,
        "to_delete": len(del_stems),
        "to_keep": total - len(del_stems),
        "deleted_by_sim": n_del_sim,
        "deleted_by_cer": n_del_cer,
        "deleted_by_both": n_del_both,
        "missing_sim": n_no_sim,
        "missing_cer": n_no_cer,
        "min_sim_raw": args.min_sim_raw,
        "max_cer": args.max_cer if args.apply_cer else None,
        "sim_scale": args.sim_scale,
        "dur_total_hours": round(dur_total / 3600.0, 2),
        "dur_delete_hours": round(dur_del / 3600.0, 2),
        "dur_keep_hours": round((dur_total - dur_del) / 3600.0, 2),
        "raw_all_stats": _stats(np.asarray(raw_all, dtype=np.float64)),
        "raw_kept_stats": _stats(np.asarray(raw_kept, dtype=np.float64)),
        "executed": bool(args.execute),
    }

    # top deleted datasets
    ds_sorted = sorted(by_ds.items(), key=lambda kv: -kv[1][1])
    rep["by_dataset"] = {ds: {"total": t, "deleted": d} for ds, (t, d) in ds_sorted}

    if args.execute and del_stems:
        t0 = time.time()
        chunk = max(50, len(del_stems) // (args.workers * 8))
        chunks = [del_stems[i : i + chunk] for i in range(0, len(del_stems), chunk)]
        ok = fail = 0
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for o, f in ex.map(_delete_batch, chunks):
                ok += o
                fail += f
        rep["deleted_files_ok"] = ok
        rep["deleted_files_fail"] = fail
        print(f"  [DELETE] removed {ok:,} files ({fail} fail) in {time.time() - t0:.1f}s", flush=True)
    return rep


def report_text(rep: dict) -> str:
    if rep.get("error"):
        return f"{rep['name']}: {rep['error']}"
    tag = "已删除" if rep["executed"] else "DRY-RUN(未删)"
    L = [
        "=" * 88,
        f"  {rep['name']}   [{tag}]   规则: raw<{rep['min_sim_raw']} 或 cer>{rep['max_cer']}  (sim_scale={rep['sim_scale']})",
        "=" * 88,
        f"  总计 {rep['total']:,}  →  删 {rep['to_delete']:,}  保留 {rep['to_keep']:,}",
        f"    因 SIM 删: {rep['deleted_by_sim']:,}   因 CER 删: {rep['deleted_by_cer']:,}   两者都超: {rep['deleted_by_both']:,}",
        f"    缺 sim.json: {rep['missing_sim']:,}   缺 cer.json: {rep['missing_cer']:,}",
        f"    时长: 总 {rep['dur_total_hours']:,}h  删 {rep['dur_delete_hours']:,}h  保留 {rep['dur_keep_hours']:,}h",
    ]
    ra = rep["raw_all_stats"]
    rk = rep["raw_kept_stats"]
    if ra.get("count"):
        p = ra["percentiles"]
        L.append(
            f"    raw(全部): n={ra['count']:,} mean={ra['mean']:.4f} p10={p['10']:.4f} p50={p['50']:.4f} p90={p['90']:.4f}"
        )
    if rk.get("count"):
        p = rk["percentiles"]
        L.append(
            f"    raw(保留): n={rk['count']:,} mean={rk['mean']:.4f} p10={p['10']:.4f} p50={p['50']:.4f} p90={p['90']:.4f}"
        )
    L.append("    删除最多的 dataset (top 10):")
    for ds, v in list(rep["by_dataset"].items())[:10]:
        if v["deleted"]:
            L.append(f"      {ds:<34} 删 {v['deleted']:,}/{v['total']:,}")
    if not rep["executed"]:
        L.append("  * DRY-RUN: 未删除任何文件。确认无误后加 --execute 真正删除。")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dirs", nargs="+", required=True, help="Previously-cloned root dirs (each processed separately)")
    ap.add_argument("--min-sim-raw", type=float, default=0.8, help="Delete if raw cosine < this (default 0.8)")
    ap.add_argument("--max-cer", type=float, default=0.03, help="Delete if manual_cer > this (default 0.03)")
    ap.add_argument(
        "--apply-cer", dest="apply_cer", action="store_true", default=True, help="Also filter by CER (default on)"
    )
    ap.add_argument("--no-apply-cer", dest="apply_cer", action="store_false", help="Filter by SIM only")
    ap.add_argument(
        "--sim-scale",
        choices=("mapped", "raw"),
        default="mapped",
        help="Scale of similarity in .sim.json: mapped=(cos+1)/2 (old dirs, default) -> raw=2*sim-1; raw=已是原始余弦",
    )
    ap.add_argument("--execute", action="store_true", help="Actually delete (default: dry-run preview only)")
    ap.add_argument("--scan-workers", type=int, default=64)
    ap.add_argument("--workers", type=int, default=32, help="Parallel delete workers")
    ap.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "prune_prev_report")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    mode = "EXECUTE(真删)" if args.execute else "DRY-RUN(预览)"
    print(
        f"模式={mode}  规则: raw<{args.min_sim_raw}"
        + (f" 或 cer>{args.max_cer}" if args.apply_cer else " (仅SIM)")
        + f"  sim_scale={args.sim_scale}  dirs={len(args.dirs)}",
        flush=True,
    )

    reports = []
    for root in args.dirs:
        rep = process_dir(root, args)
        reports.append(rep)
        txt = report_text(rep)
        print("\n" + txt, flush=True)
        (args.output_dir / f"{rep['name']}.json").write_text(
            json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    (args.output_dir / "prune_prev_summary.json").write_text(
        json.dumps({"execute": args.execute, "reports": reports}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    tot_del = sum(r.get("to_delete", 0) for r in reports)
    tot = sum(r.get("total", 0) for r in reports)
    print(f"\n合计: {tot:,} clones, 计划删除 {tot_del:,} ({'已执行' if args.execute else 'DRY-RUN'})", flush=True)
    print(f"报告: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
