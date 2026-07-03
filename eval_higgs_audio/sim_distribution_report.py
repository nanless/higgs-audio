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
"""Recompute speaker-similarity distribution per clone directory (standalone).

For every clone under each given directory, recompute the similarity between the
clone audio and *its own reference audio* using the CURRENT speaker encoder
(SamResNet100 fbank embeddings, identical to eval_sim). Reports BOTH:
  - raw cosine     = dot(e_clone, e_ref) / (||e_clone|| * ||e_ref||)   in [-1, 1]
  - mapped         = (raw + 1) / 2                                      in [0, 1]   (wespeaker/OmniVoice convention)

Design:
  - Each directory is scanned and reported INDEPENDENTLY.
  - Reference resolution per clone (first existing file wins):
        ref_audio_path (main json) -> ref_audio (.sim.json) -> ref_audio (main json) -> ref_audio_source (main json)
    Clones whose reference file cannot be found are SKIPPED and counted.
  - Reads whatever clones are on disk (already-pruned dirs => survivor distribution).
  - Full recompute (no reuse of cached .sim.json). Multi-GPU, multi-process.
  - Does NOT write sidecars or modify the clone directories. Only writes the report.

Run (in the `omnivoice` conda env, same one eval_sim uses):
    python sim_distribution_report.py \
        --dirs /path/audio_omnivoice_clone \
               /path/audio_higgs_audio_v3_tts_clone \
               /path/audio_higgs_audio_v3_tts_clone_2 \
               /path/audio_higgs_audio_v3_tts_clone_3 \
        --gpus 0,1,2,3 --workers 16 --scan-workers 64 \
        --output-dir ./sim_dist_report
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np

EVAL_DIR = Path(__file__).resolve().parent
EVAL_SIM_DIR = EVAL_DIR / "eval_sim"
# Make speaker_encoder + models importable (no torch import happens here).
sys.path.insert(0, str(EVAL_SIM_DIR))

DEFAULT_MODEL_DIR = EVAL_SIM_DIR / "model"
SKIP_DIRS = frozenset({"logs", "__pycache__", "ref", "eval_sim_embedding_cache"})

# Report thresholds (survival = fraction with score >= threshold).
RAW_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
MAPPED_THRESHOLDS = [0.70, 0.75, 0.80, 0.85, 0.90]
PERCENTILES = [1, 5, 10, 25, 50, 75, 90, 95, 99]


# --------------------------------------------------------------------------- #
#  Scan: enumerate (clone_wav, ref_wav, dataset) triples per directory
# --------------------------------------------------------------------------- #
def _is_main_sidecar(name: str) -> bool:
    return (
        name.endswith(".json")
        and not name.endswith(".cer.json")
        and not name.endswith(".mos.json")
        and not name.endswith(".sim.json")
        and not name.endswith(".eval.json")
    )


def _resolve_ref(stem: str, meta: dict) -> str | None:
    """First existing reference file, in priority order."""
    # 1) ref_audio_path (higgs v3 ref-pool)
    v = meta.get("ref_audio_path")
    if v and os.path.isfile(v):
        return v
    # 2) ref_audio in the .sim.json sidecar (omnivoice; old higgs ref_audio.wav)
    sj = stem + ".sim.json"
    if os.path.isfile(sj):
        try:
            with open(sj, encoding="utf-8") as f:
                sref = json.load(f).get("ref_audio")
            if sref and os.path.isfile(sref):
                return sref
        except (OSError, json.JSONDecodeError, TypeError):
            pass
    # 3) ref_audio in the main json
    v = meta.get("ref_audio")
    if v and os.path.isfile(v):
        return v
    # 4) ref_audio_source (original source audio; old higgs fallback)
    v = meta.get("ref_audio_source")
    if v and os.path.isfile(v):
        return v
    return None


def _scan_unit(task: tuple) -> tuple:
    """Scan one dataset/speaker subtree.

    Returns (triples, skipped_no_ref, field_counts) where
    triples = [(clone_wav, ref_wav, dataset), ...].
    """
    root, unit = task
    triples: list = []
    skipped = 0
    field_counts: dict = defaultdict(int)
    for dirpath, dirs, files in os.walk(unit):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in files:
            if not _is_main_sidecar(name):
                continue
            stem = os.path.join(dirpath, name[:-5])
            wav = stem + ".wav"
            if not os.path.isfile(wav):
                continue
            try:
                with open(stem + ".json", encoding="utf-8") as f:
                    meta = json.load(f)
            except (OSError, json.JSONDecodeError):
                meta = {}
            ref = _resolve_ref(stem, meta)
            if ref is None:
                skipped += 1
                continue
            # which field matched (for reporting)
            if meta.get("ref_audio_path") == ref:
                field_counts["ref_audio_path"] += 1
            elif meta.get("ref_audio") == ref:
                field_counts["ref_audio"] += 1
            elif meta.get("ref_audio_source") == ref:
                field_counts["ref_audio_source"] += 1
            else:
                field_counts["sim.ref_audio"] += 1
            dataset = os.path.relpath(wav, root).split(os.sep)[0]
            triples.append((wav, ref, dataset))
    return triples, skipped, dict(field_counts)


def scan_dir(root: str, scan_workers: int) -> tuple:
    """Return (triples, skipped_no_ref, field_counts) for one clone root."""
    root = str(root)
    units: list = []
    for ds in sorted(os.listdir(root)):
        ds_path = os.path.join(root, ds)
        if not os.path.isdir(ds_path) or ds in SKIP_DIRS:
            continue
        spk_dirs = [
            os.path.join(ds_path, s)
            for s in os.listdir(ds_path)
            if os.path.isdir(os.path.join(ds_path, s)) and s not in SKIP_DIRS
        ]
        units.extend(spk_dirs if spk_dirs else [ds_path])

    triples: list = []
    skipped = 0
    field_counts: dict = defaultdict(int)
    t0 = time.time()
    if len(units) > 1 and scan_workers > 1:
        with ProcessPoolExecutor(max_workers=min(scan_workers, len(units))) as ex:
            for tr, sk, fc in ex.map(_scan_unit, [(root, u) for u in units], chunksize=8):
                triples.extend(tr)
                skipped += sk
                for k, v in fc.items():
                    field_counts[k] += v
    else:
        for u in units:
            tr, sk, fc = _scan_unit((root, u))
            triples.extend(tr)
            skipped += sk
            for k, v in fc.items():
                field_counts[k] += v
    print(
        f"[scan] {os.path.basename(root)}: {len(triples):,} computable, "
        f"{skipped:,} skipped(no ref), {len(units):,} units in {time.time() - t0:.1f}s "
        f"({scan_workers}p) fields={dict(field_counts)}",
        flush=True,
    )
    return triples, skipped, dict(field_counts)


# --------------------------------------------------------------------------- #
#  Compute worker (one per (gpu, process); writes a TSV part: "dataset\traw")
# --------------------------------------------------------------------------- #
def _compute_worker(rank: int, gpu: str, shard: list, model_dir: str, part_path: str):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[var] = "1"
    import torch

    torch.set_num_threads(1)
    from speaker_encoder import SpeakerEncoder

    encoder = SpeakerEncoder(model_dir=model_dir, device="cuda:0")
    print(f"[sim-w{rank}] gpu={gpu} items={len(shard)}", flush=True)

    # sort by ref for cache locality; canonicalize paths so cache hits are robust
    shard.sort(key=lambda x: x[1])
    ref_cache: dict = {}
    _MISS = object()
    done = 0
    failed = 0
    with open(part_path, "w", encoding="utf-8") as out:
        for clone_wav, ref_wav, dataset in shard:
            ref_key = os.path.realpath(ref_wav)
            try:
                cached = ref_cache.get(ref_key, _MISS)
                if cached is _MISS:
                    r = encoder.extract_embedding(ref_wav)
                    ref_cache[ref_key] = r
                else:
                    r = cached
                c = encoder.extract_embedding(clone_wav)
            except Exception:
                r = c = None
            if r is None or c is None:
                failed += 1
                continue
            r32 = r.detach().cpu().float()
            c32 = c.detach().cpu().float()
            denom = (torch.norm(r32) * torch.norm(c32)).item()
            if denom <= 0:
                failed += 1
                continue
            raw = float(torch.dot(r32, c32).item() / denom)
            out.write(f"{dataset}\t{raw:.6f}\n")
            done += 1
            if done % 20000 == 0:
                print(f"[sim-w{rank}] {done:,}/{len(shard):,} done, {failed} failed", flush=True)
    print(f"[sim-w{rank}] DONE {done:,} computed, {failed} failed", flush=True)


# --------------------------------------------------------------------------- #
#  Stats
# --------------------------------------------------------------------------- #
def _arr_stats(a: np.ndarray) -> dict:
    if a.size == 0:
        return {"count": 0}
    return {
        "count": int(a.size),
        "mean": float(a.mean()),
        "std": float(a.std()),
        "min": float(a.min()),
        "max": float(a.max()),
        "percentiles": {str(p): float(np.percentile(a, p)) for p in PERCENTILES},
    }


def _histogram(a: np.ndarray, lo: float, hi: float, step: float) -> list:
    if a.size == 0:
        return []
    edges = np.arange(lo, hi + step / 2, step)
    counts, _ = np.histogram(a, bins=edges)
    return [
        {"lo": round(float(edges[i]), 4), "hi": round(float(edges[i + 1]), 4), "count": int(counts[i])}
        for i in range(len(counts))
    ]


def _survival(a: np.ndarray, thresholds: list) -> dict:
    n = a.size
    return {
        str(t): {
            "ge_count": int((a >= t).sum()),
            "ge_frac": (float((a >= t).sum()) / n) if n else 0.0,
            "lt_count": int((a < t).sum()),
            "lt_frac": (float((a < t).sum()) / n) if n else 0.0,
        }
        for t in thresholds
    }


def build_report(name: str, raw: np.ndarray, datasets: np.ndarray, scan_info: dict) -> dict:
    mapped = (raw + 1.0) / 2.0
    rep: dict = {
        "name": name,
        "scan": scan_info,
        "evaluated": int(raw.size),
        "raw_cosine": {
            **_arr_stats(raw),
            "histogram": _histogram(raw, -1.0, 1.0, 0.05),
            "survival": _survival(raw, RAW_THRESHOLDS),
        },
        "mapped_(cos+1)/2": {
            **_arr_stats(mapped),
            "histogram": _histogram(mapped, 0.0, 1.0, 0.025),
            "survival": _survival(mapped, MAPPED_THRESHOLDS),
        },
        "by_dataset": {},
    }
    for ds in sorted(set(datasets.tolist())):
        m = datasets == ds
        ra = raw[m]
        rep["by_dataset"][ds] = {
            "count": int(ra.size),
            "raw": _arr_stats(ra),
            "mapped": _arr_stats((ra + 1.0) / 2.0),
        }
    return rep


def _fmt_stats(s: dict) -> str:
    if not s.get("count"):
        return "count=0"
    p = s["percentiles"]
    return (
        f"n={s['count']:,} mean={s['mean']:.4f} std={s['std']:.4f} "
        f"min={s['min']:.4f} p10={p['10']:.4f} p50={p['50']:.4f} p90={p['90']:.4f} max={s['max']:.4f}"
    )


def report_to_text(rep: dict) -> str:
    L = []
    L.append("=" * 100)
    L.append(f"  目录: {rep['name']}")
    L.append("=" * 100)
    sc = rep["scan"]
    L.append(
        f"  扫描: computable={sc['computable']:,}  skipped_no_ref={sc['skipped_no_ref']:,}  "
        f"failed={sc.get('failed', 0):,}  evaluated={rep['evaluated']:,}"
    )
    L.append(f"  参考字段: {sc.get('field_counts', {})}")
    L.append("")
    L.append(f"  [raw 余弦]    {_fmt_stats(rep['raw_cosine'])}")
    L.append(f"  [mapped 映射] {_fmt_stats(rep['mapped_(cos+1)/2'])}")
    L.append("")
    L.append("  raw 余弦 存活率 (score >= 阈值):")
    for t, v in rep["raw_cosine"]["survival"].items():
        L.append(
            f"      raw>={t}: keep {v['ge_count']:,} ({v['ge_frac'] * 100:.1f}%)  |  <{t}: {v['lt_count']:,} ({v['lt_frac'] * 100:.1f}%)"
        )
    L.append("  mapped 存活率 (score >= 阈值):")
    for t, v in rep["mapped_(cos+1)/2"]["survival"].items():
        L.append(
            f"      map>={t}: keep {v['ge_count']:,} ({v['ge_frac'] * 100:.1f}%)  |  <{t}: {v['lt_count']:,} ({v['lt_frac'] * 100:.1f}%)"
        )
    L.append("")
    L.append("  raw 余弦 直方图:")
    for b in rep["raw_cosine"]["histogram"]:
        bar = "#" * int(60 * b["count"] / max(1, rep["evaluated"]))
        L.append(f"      [{b['lo']:+.2f},{b['hi']:+.2f})  {b['count']:>9,}  {bar}")
    L.append("")
    L.append(f"  按 dataset ({len(rep['by_dataset'])} 个), 按 raw 均值升序:")
    L.append(f"      {'dataset':<34}{'n':>9}{'raw_mean':>10}{'raw_p50':>10}{'map_mean':>10}{'map_p50':>10}")
    items = sorted(
        rep["by_dataset"].items(), key=lambda kv: kv[1]["raw"].get("mean", 0) if kv[1]["raw"].get("count") else 0
    )
    for ds, dv in items:
        r = dv["raw"]
        mp_ = dv["mapped"]
        if not r.get("count"):
            continue
        L.append(
            f"      {ds:<34}{r['count']:>9,}{r['mean']:>10.4f}{r['percentiles']['50']:>10.4f}"
            f"{mp_['mean']:>10.4f}{mp_['percentiles']['50']:>10.4f}"
        )
    L.append("")
    return "\n".join(L)


def comparison_text(reports: list) -> str:
    L = []
    L.append("=" * 100)
    L.append("  跨目录对比 (raw 余弦 / mapped)")
    L.append("=" * 100)
    L.append(
        f"  {'目录':<40}{'n':>9}{'raw_mean':>9}{'raw_p50':>9}{'raw≥0.5':>9}{'map_mean':>9}{'map_p50':>9}{'map≥0.85':>9}"
    )
    for rep in reports:
        rc = rep["raw_cosine"]
        mc = rep["mapped_(cos+1)/2"]
        if not rc.get("count"):
            L.append(f"  {rep['name']:<40}{'(0)':>9}")
            continue
        raw_ge05 = rc["survival"]["0.5"]["ge_frac"] * 100
        map_ge085 = mc["survival"]["0.85"]["ge_frac"] * 100
        L.append(
            f"  {rep['name']:<40}{rc['count']:>9,}{rc['mean']:>9.3f}{rc['percentiles']['50']:>9.3f}"
            f"{raw_ge05:>8.1f}%{mc['mean']:>9.3f}{mc['percentiles']['50']:>9.3f}{map_ge085:>8.1f}%"
        )
    L.append("=" * 100)
    return "\n".join(L)


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
def process_one_dir(root: str, args, out_dir: Path) -> dict:
    name = os.path.basename(os.path.normpath(root))
    print(f"\n########## {name} ##########", flush=True)
    if not os.path.isdir(root):
        print(f"  跳过: 目录不存在 {root}", flush=True)
        return {"name": name, "error": "missing", "raw_cosine": {"count": 0}, "mapped_(cos+1)/2": {"count": 0}}

    triples, skipped, field_counts = scan_dir(root, args.scan_workers)
    if args.sample_size and args.sample_size < len(triples):
        import random

        triples = random.Random(args.seed).sample(triples, args.sample_size)
        print(f"  采样 {len(triples):,} 条 (seed={args.seed})", flush=True)

    if not triples:
        rep = {
            "name": name,
            "scan": {"computable": 0, "skipped_no_ref": skipped, "failed": 0, "field_counts": field_counts},
            "evaluated": 0,
            "raw_cosine": {"count": 0},
            "mapped_(cos+1)/2": {"count": 0},
            "by_dataset": {},
        }
        return rep

    gpu_list = [g.strip() for g in args.gpus.split(",") if g.strip()] or ["0"]
    workers = max(1, args.workers)
    shards: list = [[] for _ in range(workers)]
    for i, t in enumerate(triples):
        shards[i % workers].append(t)

    part_paths = [str(out_dir / f".{name}.w{i}.tsv") for i in range(workers)]
    ctx = mp.get_context("spawn")
    procs = []
    t0 = time.time()
    for i in range(workers):
        if not shards[i]:
            continue
        gpu = gpu_list[i % len(gpu_list)]
        p = ctx.Process(target=_compute_worker, args=(i, gpu, shards[i], str(args.model_dir), part_paths[i]))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
        if p.exitcode != 0:
            raise SystemExit(f"[{name}] compute worker exited with code {p.exitcode}")

    # merge parts
    raws: list = []
    dsets: list = []
    for pp in part_paths:
        if not os.path.isfile(pp):
            continue
        with open(pp, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                ds, rv = line.rsplit("\t", 1)
                dsets.append(ds)
                raws.append(float(rv))
        os.remove(pp)

    raw = np.asarray(raws, dtype=np.float64)
    datasets = np.asarray(dsets)
    failed = len(triples) - raw.size
    print(f"  [{name}] evaluated={raw.size:,} failed={failed:,} in {time.time() - t0:.1f}s", flush=True)

    rep = build_report(name, raw, datasets, scan_info={})
    rep["scan"] = {
        "computable": len(triples),
        "skipped_no_ref": skipped,
        "failed": int(failed),
        "field_counts": field_counts,
    }
    return rep


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dirs", nargs="+", required=True, help="Clone root directories (each reported separately)")
    ap.add_argument("--gpus", type=str, default="0,1,2,3", help="Comma GPU ids")
    ap.add_argument("--workers", type=int, default=16, help="Total compute processes (round-robin over --gpus)")
    ap.add_argument("--scan-workers", type=int, default=64, help="Parallel workers for the directory scan")
    ap.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="SpeakerEncoder model dir (config.yaml + avg_model.pt)",
    )
    ap.add_argument("--output-dir", type=Path, default=EVAL_DIR / "sim_dist_report")
    ap.add_argument("--sample-size", type=int, default=None, help="Optional: per-dir random sample (default: full)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    if not Path(args.model_dir, "avg_model.pt").is_file():
        print(f"WARNING: model weights not found at {args.model_dir}/avg_model.pt", file=sys.stderr)

    print(
        f"dirs={len(args.dirs)} gpus={args.gpus} workers={args.workers} scan_workers={args.scan_workers} "
        f"model_dir={args.model_dir} sample_size={args.sample_size}",
        flush=True,
    )

    reports = []
    t_all = time.time()
    for root in args.dirs:
        rep = process_one_dir(root, args, out_dir)
        reports.append(rep)
        # per-dir outputs
        (out_dir / f"{rep['name']}.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
        txt = report_to_text(rep)
        (out_dir / f"{rep['name']}.txt").write_text(txt + "\n", encoding="utf-8")
        print("\n" + txt, flush=True)

    comp = comparison_text(reports)
    print("\n" + comp, flush=True)
    (out_dir / "comparison.txt").write_text(comp + "\n", encoding="utf-8")
    (out_dir / "all_reports.json").write_text(
        json.dumps(
            {"generated_at": datetime.now().isoformat(), "dirs": args.dirs, "reports": reports},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n报告已写入: {out_dir}  (每目录 .txt/.json + comparison.txt + all_reports.json)", flush=True)
    print(f"总耗时: {time.time() - t_all:.0f}s", flush=True)


if __name__ == "__main__":
    main()
