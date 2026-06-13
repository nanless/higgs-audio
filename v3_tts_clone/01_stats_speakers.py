"""
Step 1: Speaker Duration Statistics.

Walk all non-child datasets in parallel, compute total audio duration per speaker.
WAV files: duration from file size / bytes_per_sec (instant, no header read per file).
Other formats: soundfile.info().

Excluded child datasets:
    childmandarin, child207m-korean-filtered, chineseenglishchildren,
    king-asr-725, kingasr612, speakocean762

Usage:
    python 01_stats_speakers.py \
        --audio-root /root/group-shared/.../audio \
        --output-dir ./clone_workdir \
        --workers 8
"""

import argparse
import csv
import os
import struct
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed


CHILD_DATASETS = {
    "childmandarin",
    "child207m-korean-filtered",
    "chineseenglishchildren",
    "king-asr-725",
    "kingasr612",
    "speechocean762",
}

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3"}


def _read_wav_format(filepath: str) -> float:
    """Read a WAV header, return bytes_per_sec. Returns 0 on failure."""
    try:
        with open(filepath, "rb") as f:
            header = f.read(44)
            if len(header) < 44 or header[:4] != b"RIFF":
                return 0.0
            sr = struct.unpack_from("<I", header, 24)[0]
            bps = struct.unpack_from("<H", header, 34)[0]
            ch = struct.unpack_from("<H", header, 22)[0]
            if sr == 0 or bps == 0 or ch == 0:
                return 0.0
            return sr * ch * (bps // 8)
    except Exception:
        return 0.0


def _list_audio(dirpath: str) -> list:
    """List audio files: flat scan first, recursive fallback."""
    files = []
    has_subdirs = False
    try:
        for entry in os.scandir(dirpath):
            if entry.is_dir():
                has_subdirs = True
            elif os.path.splitext(entry.name)[1].lower() in AUDIO_EXTENSIONS:
                files.append(entry.path)
    except OSError:
        return []
    if has_subdirs:
        files = []
        for root, _dirs, filenames in os.walk(dirpath):
            for fname in filenames:
                if os.path.splitext(fname)[1].lower() in AUDIO_EXTENSIONS:
                    files.append(os.path.join(root, fname))
    return files


def process_dataset(args):
    """Process one dataset. Runs in subprocess."""
    ds_name, ds_path = args
    try:
        return _process_dataset_impl(ds_name, ds_path)
    except Exception:
        sys.stderr.write(f"[{ds_name}] ERROR:\n{traceback.format_exc()}\n")
        sys.stderr.flush()
        return ds_name, [], 0, str(traceback.format_exc())


def _process_dataset_impl(ds_name, ds_path):
    t0 = time.time()
    stats = []

    try:
        speaker_names = sorted(d for d in os.listdir(ds_path) if os.path.isdir(os.path.join(ds_path, d)))
    except OSError as e:
        return ds_name, stats, 0, str(e)

    file_count = 0

    for speaker_id in speaker_names:
        speaker_path = os.path.join(ds_path, speaker_id)
        audio_paths = _list_audio(speaker_path)
        if not audio_paths:
            continue

        file_count += len(audio_paths)

        # Detect WAV format once per speaker dir
        bytes_per_sec = 0.0
        first_wav = next((p for p in audio_paths if p.lower().endswith(".wav")), None)
        if first_wav:
            bytes_per_sec = _read_wav_format(first_wav)

        durations = []
        if bytes_per_sec > 0:
            # Fast path: file size / bytes_per_sec for all WAVs
            for p in audio_paths:
                if p.lower().endswith(".wav"):
                    fsize = os.path.getsize(p)
                    durations.append(fsize / bytes_per_sec)
                else:
                    try:
                        import soundfile as sf

                        durations.append(sf.info(p).duration)
                    except Exception:
                        durations.append(0.0)
        else:
            # All files are non-WAV
            for p in audio_paths:
                try:
                    import soundfile as sf

                    durations.append(sf.info(p).duration)
                except Exception:
                    durations.append(0.0)

        total_dur = sum(durations)
        n = len(durations)
        stats.append(
            {
                "dataset": ds_name,
                "speaker_id": speaker_id,
                "speaker_path": speaker_path,
                "num_files": n,
                "total_duration_sec": round(total_dur, 2),
                "avg_duration_sec": round(total_dur / n, 2) if n else 0,
                "max_duration_sec": round(max(durations), 2) if durations else 0,
                "min_duration_sec": round(min(durations), 2) if durations else 0,
                "has_7to20s": any(7 <= d <= 20 for d in durations),
            }
        )

    elapsed = time.time() - t0
    rate = f"{file_count / elapsed:.0f}" if elapsed > 0 else "?"
    return ds_name, stats, file_count, None


def main():
    parser = argparse.ArgumentParser(description="Speaker duration statistics")
    parser.add_argument("--audio-root", required=True)
    parser.add_argument("--output-dir", default="./clone_workdir")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    audio_root = args.audio_root
    dataset_names = sorted(
        d for d in os.listdir(audio_root) if os.path.isdir(os.path.join(audio_root, d)) and d not in CHILD_DATASETS
    )

    print(f"{len(dataset_names)} non-child datasets, {args.workers} workers", flush=True)
    t_start = time.time()

    work_items = [(name, os.path.join(audio_root, name)) for name in dataset_names]
    all_stats = []
    total_files = 0
    errors = []

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_dataset, w): w[0] for w in work_items}
        remaining = len(futures)
        for future in as_completed(futures):
            ds_name = futures[future]
            remaining -= 1
            ds_name2, stats, file_count, error = future.result()
            if error:
                errors.append((ds_name2, error))
                print(f"[{ds_name2}] ERROR: {error[:200]}", flush=True)
            else:
                print(f"[{ds_name2}] {len(stats)} speakers, {file_count} files ({remaining} remaining)", flush=True)
            all_stats.extend(stats)
            total_files += file_count
            # Save incrementally so we don't lose progress
            _save_progress(all_stats, args.output_dir)

    if errors:
        print(f"\n{len(errors)} datasets had errors:")
        for name, err in errors:
            print(f"  [{name}] {err[:200]}")

    all_stats.sort(key=lambda x: x["total_duration_sec"])
    total_time = time.time() - t_start
    rate_str = f"({total_files / total_time:.0f} files/s)" if total_time > 0 else ""
    print(f"\nTotal: {len(all_stats)} speakers, {total_files} files, {total_time:.1f}s {rate_str}")

    # Save CSV
    csv_path = os.path.join(args.output_dir, "speaker_duration_stats.csv")
    csv_fields = [
        "dataset",
        "speaker_id",
        "num_files",
        "total_duration_sec",
        "avg_duration_sec",
        "max_duration_sec",
        "min_duration_sec",
        "has_7to20s",
        "speaker_path",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(all_stats)
    print(f"Saved: {csv_path}")

    durations = [s["total_duration_sec"] for s in all_stats]
    if durations:
        print(f"\nDuration distribution:")
        print(f"  min={min(durations):.1f}s  p10={_pct(durations, 10):.1f}s  p25={_pct(durations, 25):.1f}s")
        print(f"  p50={_pct(durations, 50):.1f}s  p75={_pct(durations, 75):.1f}s  p90={_pct(durations, 90):.1f}s")
        print(f"  max={max(durations):.1f}s")
        for thresh in [30, 60, 120, 300]:
            print(f"  <{thresh}s: {sum(1 for d in durations if d < thresh)}")


def _pct(data, p):
    k = (len(data) - 1) * p / 100.0
    f = int(k)
    c = k - f
    if f + 1 < len(data):
        return data[f] * (1 - c) + data[f + 1] * c
    return data[f]


def _save_progress(all_stats, output_dir):
    """Incrementally save CSV as datasets complete (so we don't lose work)."""
    all_stats.sort(key=lambda x: x["total_duration_sec"])
    csv_path = os.path.join(output_dir, "speaker_duration_stats.csv")
    csv_fields = [
        "dataset",
        "speaker_id",
        "num_files",
        "total_duration_sec",
        "avg_duration_sec",
        "max_duration_sec",
        "min_duration_sec",
        "has_7to20s",
        "speaker_path",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(all_stats)


if __name__ == "__main__":
    main()
