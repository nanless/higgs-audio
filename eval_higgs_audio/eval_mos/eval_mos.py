#!/usr/bin/env python3
"""MOS (Mean Opinion Score) evaluation for Higgs Audio v3 TTS clone audio.

Evaluates cloned audio quality with multiple metrics:
  UTMOS22Strong, SCOREQ, TTSDS2, UTMOSv2

Multi-process: splits items across --workers processes, each on its own GPU.
Adapted from OmniVoice batch_generate_text_and_clone/eval_mos/eval_clone_mos.py.

Usage:
    conda activate omnivoice   # (or env with scoreq, ttsds, utmosv2)
    cd eval_higgs_audio/eval_mos

    python eval_mos.py --out-dir /path/to/audio_higgs_audio_v3_tts_clone
    python eval_mos.py --metrics UTMOS22Strong,SCOREQ --workers 4 --gpus 0,1
    bash run_eval_mos.sh
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

EVAL_DIR = Path(__file__).resolve().parent
PARENT_DIR = EVAL_DIR.parent
sys.path.insert(0, str(PARENT_DIR))
sys.path.insert(0, str(EVAL_DIR))

from scorers import AVAILABLE_METRICS, BaseScorer, create_scorer, try_create_scorer  # noqa: E402
from eval_common import (  # noqa: E402
    append_jsonl,
    list_clone_items,
    merge_jsonl_parts,
    parse_gpu_list,
    split_shards,
    write_json,
)

DEFAULT_OUT_DIR = Path(
    os.environ.get(
        "HIGGS_CLONE_ROOT",
        "/root/group-shared/voiceprint/data/speech/speaker_diarization/merged_datasets_20250610_vad_segments_mtfaa_enhanced_extend_kid_withclone_addlibrilight_1130/audio_higgs_audio_v3_tts_clone",
    )
)

ALL_METRICS = AVAILABLE_METRICS


def find_cloned_wavs(out_dir: Path, scan_workers: int = 8) -> List[Tuple[Path, Path]]:
    return list_clone_items(out_dir, label="mos-scan", scan_workers=scan_workers)


def write_eval_json(json_path: Path, record: dict):
    json_path.with_suffix(".mos.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def summarize(results: list, metrics: List[str]) -> dict:
    def stats(vals):
        if not vals:
            return {"count": 0, "mean": None, "min": None, "max": None, "p50": None, "p10": None, "p90": None}
        a = np.array(vals, dtype=np.float64)
        return {
            "count": int(len(a)),
            "mean": float(a.mean()),
            "min": float(a.min()),
            "max": float(a.max()),
            "p10": float(np.percentile(a, 10)),
            "p50": float(np.percentile(a, 50)),
            "p90": float(np.percentile(a, 90)),
        }

    summary = {"total_count": len(results)}
    for metric in metrics:
        key = metric.lower()
        scores = [r[key] for r in results if r.get(key) is not None]
        by_dataset = defaultdict(list)
        by_language = defaultdict(list)
        for r in results:
            if r.get(key) is None:
                continue
            by_dataset[r.get("dataset", "unknown")].append(r[key])
            by_language[r.get("language", "unknown")].append(r[key])
        summary[metric] = {
            "overall": stats(scores),
            "by_dataset": {k: stats(v) for k, v in sorted(by_dataset.items())},
            "by_language": {k: stats(v) for k, v in sorted(by_language.items())},
            "failed_count": sum(1 for r in results if r.get(key) is None),
        }
    return summary


def _load_results_from_jsonl(path: Path) -> list:
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _has_batch_score(scorer: BaseScorer) -> bool:
    return type(scorer).score_files is not BaseScorer.score_files


def _worker(rank, gpu, shard, out_dir, model_dir, metrics, details_path, no_sidecar):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    out = Path(out_dir)
    details = Path(details_path)

    scorers: Dict[str, Optional[BaseScorer]] = {}
    for metric in metrics:
        scorers[metric] = try_create_scorer(metric, device="cuda:0", model_dir=model_dir)

    active = {k: v for k, v in scorers.items() if v is not None}
    if not active:
        print(f"[w{rank}] No metrics available, exiting.", flush=True)
        return

    print(f"[w{rank}] gpu={gpu} items={len(shard)} metrics={list(active.keys())}", flush=True)

    records = []
    wav_paths = []
    json_paths = []
    for wav_s, json_s in shard:
        wav_path = Path(wav_s)
        json_path = Path(json_s)
        rel = wav_path.relative_to(out)
        dataset = rel.parts[0] if rel.parts else "unknown"
        record = {
            "cloned_audio": str(wav_path),
            "dataset": dataset,
            "sidecar_json": str(json_path),
            "worker": rank,
            "evaluated_at": datetime.now().isoformat(),
        }
        try:
            meta = json.loads(json_path.read_text(encoding="utf-8"))
            record.update(
                ref_audio_source=meta.get("ref_audio_source"),
                gen_text=meta.get("clean_text") or meta.get("text"),
                emotion=meta.get("emotion"),
                scenario=meta.get("scenario"),
                speaker_id=meta.get("speaker_id"),
            )
        except (json.JSONDecodeError, OSError):
            pass
        records.append(record)
        wav_paths.append(wav_path)
        json_paths.append(json_path)

    for metric_name, scorer in active.items():
        key = metric_name.lower()
        if _has_batch_score(scorer):
            print(f"[w{rank}] {metric_name} batch-scoring {len(wav_paths)} files ...", flush=True)
            try:
                scores = scorer.score_files(wav_paths)
                for rec, score in zip(records, scores):
                    rec[key] = score
            except Exception as e:
                print(f"[w{rank}] {metric_name} batch failed ({e}), falling back.", flush=True)
                for rec, wp in zip(records, wav_paths):
                    try:
                        rec[key] = scorer.score_file(wp)
                    except Exception as e2:
                        rec[key] = None
                        rec[f"{key}_error"] = str(e2)
        else:
            print(f"[w{rank}] {metric_name} loop-scoring {len(wav_paths)} files ...", flush=True)
            for rec, wp in tqdm(zip(records, wav_paths), total=len(records), desc=f"w{rank}-{metric_name}"):
                try:
                    rec[key] = scorer.score_file(wp)
                except Exception as e:
                    rec[key] = None
                    rec[f"{key}_error"] = str(e)

    for rec, jp in zip(records, json_paths):
        if not no_sidecar:
            write_eval_json(jp, rec)
        append_jsonl(details, rec)


def _run_single_process(items, args, metrics, summary_path, details_path):
    gpu_list = parse_gpu_list(args.gpus, args.gpu)
    shard = [(str(w), str(j)) for w, j in items]
    _worker(0, gpu_list[0], shard, str(args.out_dir), args.model_dir, metrics, str(details_path), args.no_sidecar)
    results = _load_results_from_jsonl(details_path)
    summary = summarize(results, metrics)
    summary.update(
        metrics=metrics,
        model_dir=str(args.model_dir),
        evaluated_at=datetime.now().isoformat(),
        out_dir=str(args.out_dir),
        sample_size=args.sample_size,
        seed=args.seed if args.sample_size else None,
        workers=1,
        gpus=gpu_list,
        items_done=len(results),
        items_total=len(items),
    )
    write_json(summary_path, summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--gpus", type=str, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--metrics", type=str, default=None, help=f"Comma-separated metrics. Available: {','.join(ALL_METRICS)}"
    )
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-sidecar", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    if args.metrics:
        metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
        for m in metrics:
            if m not in ALL_METRICS:
                parser.error(f"Unknown metric '{m}'. Available: {','.join(ALL_METRICS)}")
    else:
        metrics = list(ALL_METRICS)

    gpu_list = parse_gpu_list(args.gpus, args.gpu if args.gpu is not None else 0)
    workers = max(1, args.workers)

    items = find_cloned_wavs(args.out_dir, scan_workers=max(8, args.workers * 2))
    if not items:
        print("No cloned audio found.", flush=True)
        return

    if args.skip_existing:
        before = len(items)
        items = [(w, j) for w, j in items if not j.with_suffix(".mos.json").exists()]
        print(f"Skip-existing: {before - len(items)} done, {len(items)} remaining", flush=True)
        if not items:
            print("All already evaluated.", flush=True)
            return

    if args.sample_size is not None and args.sample_size < len(items):
        items = random.Random(args.seed).sample(items, args.sample_size)
        print(f"Sampled {len(items)} files (seed={args.seed})", flush=True)

    tag = f"_n{args.sample_size}" if args.sample_size else ""
    summary_path = args.out_dir / f"eval_higgs_mos_summary{tag}.json"
    details_path = args.out_dir / f"eval_higgs_mos_details{tag}.jsonl"
    if not args.skip_existing:
        if details_path.exists():
            details_path.unlink()
        for p in args.out_dir.glob(f"eval_higgs_mos_details{tag}.w*.jsonl"):
            p.unlink()

    metric_str = ",".join(metrics)
    print(f"Found {len(items)} wavs | workers={workers} gpus={gpu_list} | metrics={metric_str}", flush=True)

    if workers == 1:
        summary = _run_single_process(items, args, metrics, summary_path, details_path)
    else:
        shards = split_shards(items, workers)
        shard_strs = [[(str(w), str(j)) for w, j in s] for s in shards]
        part_paths = [args.out_dir / f"eval_higgs_mos_details{tag}.w{i}.jsonl" for i in range(workers)]
        ctx = mp.get_context("spawn")
        procs = []
        for i in range(workers):
            if not shard_strs[i]:
                continue
            gpu = gpu_list[i % len(gpu_list)]
            p = ctx.Process(
                target=_worker,
                args=(
                    i,
                    gpu,
                    shard_strs[i],
                    str(args.out_dir),
                    args.model_dir,
                    metrics,
                    str(part_paths[i]),
                    args.no_sidecar,
                ),
            )
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
            if p.exitcode != 0:
                raise SystemExit(f"Worker exited with code {p.exitcode}")

        merge_jsonl_parts(part_paths, details_path)
        for p in part_paths:
            p.unlink(missing_ok=True)

        results = _load_results_from_jsonl(details_path)
        summary = summarize(results, metrics)
        summary.update(
            metrics=metrics,
            model_dir=args.model_dir,
            evaluated_at=datetime.now().isoformat(),
            out_dir=str(args.out_dir),
            sample_size=args.sample_size,
            seed=args.seed if args.sample_size else None,
            workers=workers,
            gpus=gpu_list,
            items_done=len(results),
            items_total=len(items),
        )
        write_json(summary_path, summary)

    print(f"\n{'=' * 60}", flush=True)
    for metric in metrics:
        if metric not in summary:
            continue
        ms = summary[metric]
        ov = ms["overall"]
        failed = ms["failed_count"]
        print(f"{metric} (n={ov['count']}, failed={failed})", flush=True)
        if ov["count"]:
            print(
                f"  mean={ov['mean']:.3f}  p50={ov['p50']:.3f}  p10={ov['p10']:.3f}  p90={ov['p90']:.3f}", flush=True
            )
    print(f"\nSummary: {summary_path}", flush=True)
    print(f"Details: {details_path}", flush=True)


if __name__ == "__main__":
    main()
