"""
Step 2: ASR Worker (per GPU, file-batching instead of speaker-batching).

Pre-collects all pending audio files, groups by language, then processes
in large batches for maximum GPU throughput.

Usage (one per GPU):
    CUDA_VISIBLE_DEVICES=0 python 02_asr_worker.py \
        --stats-csv ./clone_workdir/speaker_duration_stats.csv \
        --shard-index 0 --total-gpus 8 \
        --files-per-batch 48
"""

import argparse
import csv
import json
import os
import time
from collections import defaultdict

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3"}

DATASET_LANG = {
    "aishell-3": "Chinese",
    "aishell1": "Chinese",
    "casia": "Chinese",
    "dailytalk": "Chinese",
    "didispeech": "Chinese",
    "ears": "Chinese",
    "esd": "Chinese",
    "hq-conversations": "Chinese",
    "copy_reports": "Chinese",
    "emov-db": "English",
    "expresso": "English",
    "librilight_medium_small": "English",
    "libritts": "English",
    "ravdess": "English",
    "vctk": "English",
    "hifi-tts": "English",
    "jvs": "Japanese",
}


def scan_audio(speaker_path: str) -> list:
    files = []
    try:
        for entry in os.scandir(speaker_path):
            if entry.is_file() and os.path.splitext(entry.name)[1].lower() in AUDIO_EXTENSIONS:
                files.append(entry.path)
            elif entry.is_dir():
                for root, _dirs, filenames in os.walk(entry.path):
                    for fname in filenames:
                        if os.path.splitext(fname)[1].lower() in AUDIO_EXTENSIONS:
                            files.append(os.path.join(root, fname))
    except OSError:
        pass
    return files


def main():
    parser = argparse.ArgumentParser(description="Qwen3-ASR per-GPU worker (file-batching)")
    parser.add_argument("--stats-csv", required=True)
    parser.add_argument(
        "--shard-index",
        type=int,
        default=None,
        help="0..N-1 worker shard (preferred). Falls back to --gpu-id for old callers.",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=None,
        help="Deprecated: physical GPU id used as shard when --shard-index omitted (breaks non-contiguous lists).",
    )
    parser.add_argument("--total-gpus", type=int, required=True)
    parser.add_argument("--local-model", default="/root/.cache/huggingface/hub/Qwen3-ASR-1.7B-local")
    parser.add_argument("--files-per-batch", type=int, default=48)
    parser.add_argument("--target-sec", type=float, default=3600.0, help="只转写 total_duration_sec < target 的说话人")
    args = parser.parse_args()

    if args.shard_index is not None:
        shard_index = args.shard_index
    elif args.gpu_id is not None:
        shard_index = args.gpu_id
    else:
        parser.error("require --shard-index (or deprecated --gpu-id)")
    total_gpus = args.total_gpus
    batch_size = args.files_per_batch

    # ---- Phase 1: Collect all speakers assigned to this shard ----
    all_speakers = []
    with open(args.stats_csv) as f:
        for i, r in enumerate(csv.DictReader(f)):
            if float(r["total_duration_sec"]) >= args.target_sec:
                continue
            # Skip LOW speakers (num_files<20): 04 drops them from clone budget too
            try:
                num_files = int(float(r.get("num_files") or 0))
            except (TypeError, ValueError):
                num_files = 0
            status = (r.get("status") or "").strip().upper()
            if status == "LOW" or (num_files > 0 and num_files < 20):
                continue
            if i % total_gpus != shard_index:
                continue
            all_speakers.append(
                {
                    "dataset": r["dataset"],
                    "speaker_id": r["speaker_id"],
                    "speaker_path": r["speaker_path"],
                    "lang_hint": DATASET_LANG.get(r["dataset"]),
                }
            )

    print(f"shard {shard_index}/{total_gpus}: {len(all_speakers)} speakers assigned", flush=True)

    # ---- Phase 2: Pre-collect all pending files, grouped by language ----
    lang_files = defaultdict(list)  # lang_hint -> [(audio_path, dataset, speaker_id)]
    total_scanned = 0
    total_already = 0

    for spk in all_speakers:
        audio_paths = scan_audio(spk["speaker_path"])
        for p in audio_paths:
            total_scanned += 1
            if os.path.exists(p + ".json"):
                total_already += 1
                continue
            lang = spk["lang_hint"]
            lang_files[lang].append((p, spk["dataset"], spk["speaker_id"]))

    total_pending = sum(len(v) for v in lang_files.values())
    print(
        f"shard {shard_index}: scanned={total_scanned}, already_done={total_already}, pending={total_pending}",
        flush=True,
    )

    if total_pending == 0:
        print(f"shard {shard_index}: nothing to do, exiting", flush=True)
        return

    for lang, files in sorted(lang_files.items(), key=lambda x: (x[0] is None, x[0] or "")):
        print(f"shard {shard_index}:   lang={lang or 'auto'}: {len(files)} files", flush=True)

    # ---- Phase 3: Load model ----
    import torch
    from qwen_asr import Qwen3ASRModel

    print(f"shard {shard_index}: loading model...", flush=True)
    model = Qwen3ASRModel.from_pretrained(
        args.local_model,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        max_inference_batch_size=max(batch_size, 48),
    )
    print(f"shard {shard_index}: model loaded (device={model.device})", flush=True)

    # ---- Phase 4: Batch process by language ----
    processed = 0
    errors = 0
    t0 = time.time()

    for lang, files in sorted(lang_files.items(), key=lambda x: (x[0] is None, x[0] or "")):
        lang_label = lang or "auto"
        n = len(files)
        print(f"shard {shard_index}: processing lang={lang_label}, {n} files", flush=True)

        for i in range(0, n, batch_size):
            batch = files[i : i + batch_size]
            paths = [p for p, ds, spk in batch]
            metadatas = [(ds, spk) for p, ds, spk in batch]

            try:
                results = model.transcribe(audio=paths, language=lang)

                for idx, result in enumerate(results):
                    path = paths[idx]
                    ds, spk = metadatas[idx]
                    meta = {
                        "audio_path": path,
                        "dataset": ds,
                        "speaker_id": spk,
                        "transcript": (result.text or "").strip(),
                        "language": result.language or "",
                    }
                    with open(path + ".json", "w") as jf:
                        json.dump(meta, jf, ensure_ascii=False)
                    processed += 1

            except Exception as e:
                errors += len(batch)
                # Do not write a sidecar on failure: leaving {path}.json absent means
                # these files are retried on the next run (implicit retry for transient errors).
                print(
                    f"[shard {shard_index}] batch failed ({len(batch)} files), will retry next run: {str(e)[:200]}",
                    flush=True,
                )

            # Progress
            total_done = processed + errors
            elapsed = time.time() - t0
            rate = f"{processed / elapsed:.1f}" if elapsed > 0 else "?"
            pct = total_done / total_pending * 100 if total_pending else 100
            print(
                f"[shard {shard_index}] {total_done}/{total_pending} ({pct:.1f}%) elapsed={elapsed:.0f}s rate={rate} fs/s",
                flush=True,
            )

    elapsed = time.time() - t0
    print(
        f"\n[shard {shard_index}] DONE: processed={processed} errors={errors} "
        f"elapsed={elapsed:.0f}s ({processed / elapsed:.1f} fs/s)"
        if elapsed > 0
        else "",
        flush=True,
    )


if __name__ == "__main__":
    main()
