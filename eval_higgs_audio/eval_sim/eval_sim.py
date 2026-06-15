#!/usr/bin/env python3
"""Speaker similarity evaluation for Higgs Audio v3 TTS clone audio.

Compares cloned audio with reference audio using cosine similarity
of SamResNet100 speaker embeddings.

Multi-process: splits pairs across --workers processes.
Adapted from OmniVoice batch_generate_text_and_clone/eval_sim/eval_clone_similarity.py.

Key differences from OmniVoice:
- Higgs reference audio: per-clone ref_audio_path in clone sidecar JSON
- Higgs metadata: clean_text field (OmniVoice: gen_text)
- Output: clone_NNNN.sim.json (OmniVoice: text_NNN.sim.json)

Usage:
    conda activate omnivoice
    cd eval_higgs_audio/eval_sim

    python eval_sim.py --out-dir /path/to/audio_higgs_audio_v3_tts_clone
    python eval_sim.py --sample-size 500 --workers 4 --gpus 0
    bash run_eval_sim.sh
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import re
import random
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
from tqdm import tqdm

EVAL_DIR = Path(__file__).resolve().parent
PARENT_DIR = EVAL_DIR.parent
sys.path.insert(0, str(PARENT_DIR))
sys.path.insert(0, str(EVAL_DIR))

import speaker_similarity as sim  # noqa: E402
from eval_common import (  # noqa: E402
    append_jsonl,
    merge_jsonl_parts,
    parse_gpu_list,
    split_shards,
    write_json,
)

# Default model dir (voxblink2_samresnet100_ft weights)
# Use local eval_sim/model/ (setup via setup_models.sh)
LOCAL_SIM_MODEL = Path(__file__).resolve().parent / "model"

DEFAULT_OUT_DIR = Path(
    os.environ.get(
        "HIGGS_CLONE_ROOT",
        "/root/group-shared/voiceprint/data/speech/speaker_diarization/merged_datasets_20250610_vad_segments_mtfaa_enhanced_extend_kid_withclone_addlibrilight_1130/audio_higgs_audio_v3_tts_clone",
    )
)


_CLONE_WAV_RE = re.compile(r"^clone_\d+\.wav$")
_SKIP_DIRS = frozenset({"logs", "__pycache__", "eval_sim_embedding_cache"})


def _fast_scan_speaker(speaker_dir: str) -> List[Tuple[str, str, str]]:
    results: list = []
    for entry in os.scandir(speaker_dir):
        if not entry.is_file() or not _CLONE_WAV_RE.match(entry.name):
            continue
        json_path = entry.path[:-4] + ".json"
        if not os.path.isfile(json_path):
            continue
        try:
            with open(json_path, encoding="utf-8") as jf:
                ref = json.load(jf).get("ref_audio_path", "")
            if not ref or not os.path.isfile(ref):
                continue
        except (OSError, json.JSONDecodeError, TypeError):
            continue
        results.append((entry.path, ref, json_path))
    return results


def find_clone_pairs(out_dir: Path, scan_workers: int = 16) -> List[Tuple[Path, Path, Path]]:
    """Fast scan for (cloned_wav, ref_audio, sidecar_json) triples.

    Only stats files/dirs — does NOT read JSON content during scan.
    """
    out_dir = Path(out_dir)
    t0 = time.time()

    speaker_dirs: list[str] = []
    for ds_entry in os.scandir(str(out_dir)):
        if not ds_entry.is_dir() or ds_entry.name in _SKIP_DIRS:
            continue
        has_sub = False
        for spk_entry in os.scandir(ds_entry.path):
            if spk_entry.is_dir() and spk_entry.name not in _SKIP_DIRS:
                speaker_dirs.append(spk_entry.path)
                has_sub = True
        if not has_sub:
            speaker_dirs.append(ds_entry.path)

    raw: list[tuple] = []
    if len(speaker_dirs) > 1 and scan_workers > 1:
        with ProcessPoolExecutor(max_workers=scan_workers) as executor:
            futs = {executor.submit(_fast_scan_speaker, sd): sd for sd in speaker_dirs}
            done = 0
            for fut in as_completed(futs):
                batch = fut.result()
                raw.extend(batch)
                done += 1
                if done % 2000 == 0:
                    print(f"[sim-scan] scanned {done}/{len(speaker_dirs)} speakers, {len(raw)} pairs ...", flush=True)
    else:
        for sd in speaker_dirs:
            raw.extend(_fast_scan_speaker(sd))

    pairs = [(Path(c), Path(r), Path(j)) for c, r, j in raw]
    print(f"[sim-scan] {len(pairs)} pairs from {len(speaker_dirs)} speakers in {time.time() - t0:.1f}s", flush=True)
    return pairs


def write_sim_json(json_path: Path, record: dict):
    json_path.with_suffix(".sim.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def summarize(results: list) -> dict:
    scores = [r["similarity"] for r in results if r.get("similarity") is not None]
    by_dataset = defaultdict(list)
    for r in results:
        if r.get("similarity") is not None:
            by_dataset[r.get("dataset", "unknown")].append(r["similarity"])

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

    return {
        "overall": stats(scores),
        "by_dataset": {k: stats(v) for k, v in sorted(by_dataset.items())},
        "failed_count": sum(1 for r in results if r.get("similarity") is None),
        "total_count": len(results),
    }


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


def _sim_worker(rank: int, gpu: str, shard: list, out_dir: str, model_dir: Path, details_path: str, no_sidecar: bool):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    out = Path(out_dir)
    details = Path(details_path)

    encoder = sim.load_encoder(str(model_dir), device="cuda:0")
    print(f"[sim-w{rank}] gpu={gpu} items={len(shard)}", flush=True)

    shard.sort(key=lambda x: x[1])

    ref_emb_cache: dict[str, object] = {}
    _SENTINEL = object()

    for cloned_s, ref_s, json_s in tqdm(shard, desc=f"sim-w{rank}", position=rank):
        cloned_wav = Path(cloned_s)
        ref_audio = Path(ref_s)
        json_path = Path(json_s)
        rel = cloned_wav.relative_to(out)
        dataset = rel.parts[0] if rel.parts else "unknown"
        meta = {}
        try:
            meta = json.loads(json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass

        ref_key = str(ref_audio)
        cached = ref_emb_cache.get(ref_key, _SENTINEL)
        if cached is _SENTINEL:
            ref_emb = encoder.extract_embedding(ref_key)
            ref_emb_cache[ref_key] = ref_emb
        else:
            ref_emb = cached
        clone_emb = encoder.extract_embedding(str(cloned_wav))
        if ref_emb is not None and clone_emb is not None:
            score = sim.cosine_similarity(ref_emb, clone_emb)
        else:
            score = None
        record = {
            "cloned_audio": str(cloned_wav),
            "ref_audio": str(ref_audio),
            "similarity": score,
            "dataset": dataset,
            "sidecar_json": str(json_path),
            "model_dir": str(model_dir),
            "worker": rank,
            "evaluated_at": datetime.now().isoformat(),
        }
        if meta:
            record.update(
                gen_text=meta.get("clean_text") or meta.get("text"),
                ref_transcript=meta.get("ref_transcript"),
                emotion=meta.get("emotion"),
                scenario=meta.get("scenario"),
                speaker_id=meta.get("speaker_id"),
            )
        if not no_sidecar:
            write_sim_json(json_path, record)
        append_jsonl(details, record)


def _run_single_process(pairs, args, summary_path, details_path):
    gpu_list = parse_gpu_list(args.gpus, args.gpu)
    shard = [(str(c), str(r), str(j)) for c, r, j in pairs]
    model_dir = args.model_dir or LOCAL_SIM_MODEL
    _sim_worker(0, gpu_list[0], shard, str(args.out_dir), model_dir, str(details_path), args.no_sidecar)
    results = _load_results_from_jsonl(details_path)
    summary = summarize(results)
    summary.update(
        model_dir=str(model_dir),
        evaluated_at=datetime.now().isoformat(),
        out_dir=str(args.out_dir),
        sample_size=args.sample_size,
        seed=args.seed if args.sample_size else None,
        workers=1,
        gpus=gpu_list,
        items_done=len(results),
        items_total=len(pairs),
    )
    write_json(summary_path, summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help=f"Model dir containing config.yaml + avg_model.pt (default: {LOCAL_SIM_MODEL})",
    )
    parser.add_argument("--gpu", type=int, default=None, help="Single GPU id")
    parser.add_argument("--gpus", type=str, default=None, help="Comma GPU ids, e.g. 0 or 0,1")
    parser.add_argument("--workers", type=int, default=16, help="Process count (default 16, 4 per GPU)")
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-sidecar", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    if args.gpus is None and args.gpu is None:
        gpu_list = parse_gpu_list("0,1,2,3")
    else:
        gpu_list = parse_gpu_list(args.gpus, args.gpu)
    workers = max(1, args.workers)

    pairs = find_clone_pairs(args.out_dir)
    if not pairs:
        print("No cloned audio pairs found.", flush=True)
        return

    if args.skip_existing:
        before = len(pairs)
        pairs = [(c, r, j) for c, r, j in pairs if not j.with_suffix(".sim.json").exists()]
        print(f"Skip-existing: {before - len(pairs)} already done, {len(pairs)} remaining", flush=True)
        if not pairs:
            print("All pairs already evaluated.", flush=True)
            return

    if args.sample_size is not None and args.sample_size < len(pairs):
        pairs = random.Random(args.seed).sample(pairs, args.sample_size)
        print(f"Using sample of {len(pairs)} pairs (seed={args.seed})", flush=True)

    model_dir = args.model_dir or LOCAL_SIM_MODEL
    tag = f"_n{args.sample_size}" if args.sample_size else ""
    summary_path = args.out_dir / f"eval_higgs_sim_summary{tag}.json"
    details_path = args.out_dir / f"eval_higgs_sim_details{tag}.jsonl"
    if not args.skip_existing:
        if details_path.exists():
            details_path.unlink()
        for p in args.out_dir.glob(f"eval_higgs_sim_details{tag}.w*.jsonl"):
            p.unlink()

    print(f"Found {len(pairs)} pairs | workers={workers} gpus={gpu_list} | model={model_dir}", flush=True)

    if workers == 1:
        summary = _run_single_process(pairs, args, summary_path, details_path)
    else:
        shards = split_shards(pairs, workers)
        shard_strs = [[(str(c), str(r), str(j)) for c, r, j in s] for s in shards]
        part_paths = [args.out_dir / f"eval_higgs_sim_details{tag}.w{i}.jsonl" for i in range(workers)]
        ctx = mp.get_context("spawn")
        procs = []
        for i in range(workers):
            if not shard_strs[i]:
                continue
            gpu = gpu_list[i % len(gpu_list)]
            p = ctx.Process(
                target=_sim_worker,
                args=(
                    i,
                    gpu,
                    shard_strs[i],
                    str(args.out_dir),
                    model_dir,
                    str(part_paths[i]),
                    args.no_sidecar,
                ),
            )
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
            if p.exitcode != 0:
                raise SystemExit(f"SIM worker exited with code {p.exitcode}")

        merge_jsonl_parts(part_paths, details_path)
        for p in part_paths:
            p.unlink(missing_ok=True)

        results = _load_results_from_jsonl(details_path)
        summary = summarize(results)
        summary.update(
            model_dir=str(model_dir),
            evaluated_at=datetime.now().isoformat(),
            out_dir=str(args.out_dir),
            sample_size=args.sample_size,
            seed=args.seed if args.sample_size else None,
            workers=workers,
            gpus=gpu_list,
            items_done=len(results),
            items_total=len(pairs),
        )
        write_json(summary_path, summary)

    ov = summary["overall"]
    print(f"\n{'=' * 60}", flush=True)
    print(f"Overall similarity (n={ov['count']}, failed={summary['failed_count']})", flush=True)
    if ov["count"]:
        print(f"  mean={ov['mean']:.4f}  p50={ov['p50']:.4f}  p10={ov['p10']:.4f}  p90={ov['p90']:.4f}", flush=True)
    print(f"Summary: {summary_path}", flush=True)
    print(f"Details: {details_path}", flush=True)


if __name__ == "__main__":
    main()
