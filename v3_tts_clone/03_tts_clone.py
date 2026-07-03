"""
Step 3: TTS Voice Cloning via SGLang-Omni Local Servers.

For each low-duration speaker (< 3600s, >= 20 audio files):
1. Select reference audio (7-20s, with ASR transcript from step 2)
2. Calculate clones needed to reach 3600s (with 2x buffer for 50% quality pass rate)
3. Generate clones via local SGLang Higgs v3 TTS servers (one per GPU)
4. Save audio + JSON metadata under audio_higgs_audio_v3_tts_clone/{dataset}/{speaker_id}/

Architecture:
    - N SGLang servers on ports 8000..8000+N-1 (one per GPU)
    - Speakers round-robin assigned to servers
    - ThreadPoolExecutor per server for concurrent requests
    - No base64 encoding (pass file paths directly)

Usage:
    python 03_tts_clone.py \
        --stats-csv ./clone_workdir/speaker_duration_stats.csv \
        --texts-jsonl higgs_audio_v3_text_generator/batch_output_v2/generated_texts_final.jsonl \
        --output-root /root/group-shared/.../audio_higgs_audio_v3_tts_clone \
        --base-port 8000 \
        --num-servers 8 \
        --workers-per-server 16
"""

import argparse
import csv
import hashlib
import json
import os
import random
import re
import shutil
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
from typing import List, Optional

import numpy as np
import requests


AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3"}
CLONE_WAV_RE = re.compile(r"^clone_(\d+)\.wav$")
REF_SUBDIR = "ref"
DEFAULT_REF_MODE = "random"
DEFAULT_REF_ROTATE_EVERY = 50
DEFAULT_REF_POOL_SIZE = 256
DEFAULT_OUTPUT_SAMPLE_RATE = 16000
# SGLang audio-token budget. 1024 tokens ~= 40.7s @ 25fps, which truncates long texts
# (>~250 chars) mid-utterance -> ASR sees short audio -> inflated CER -> over-pruning.
# Raise for fuller long clones (costs more compute/memory per long clone).
DEFAULT_MAX_NEW_TOKENS = 1024
MAX_NEW_TOKENS = DEFAULT_MAX_NEW_TOKENS  # set from --max-new-tokens in main()


def _next_clone_file_idx(out_dir: str) -> int:
    """Next clone file index in output dir (for resume within this root)."""
    max_idx = -1
    if not os.path.isdir(out_dir):
        return 0
    try:
        for name in os.listdir(out_dir):
            m = CLONE_WAV_RE.match(name)
            if not m:
                continue
            path = os.path.join(out_dir, name)
            if os.path.getsize(path) > 1000:
                max_idx = max(max_idx, int(m.group(1)))
    except OSError:
        pass
    return max_idx + 1


def _resolve_clone_indices(out_dir: str, csv_start_idx: int) -> tuple[int, int]:
    """Return (file_start, text_offset) for clone filenames vs text pool offset."""
    local_start = _next_clone_file_idx(out_dir)
    if local_start < csv_start_idx:
        return local_start, csv_start_idx + local_start
    return local_start, local_start


# ---- SGLang TTS Client ----


class SGLangTTSClient:
    """Client for a local SGLang-Omni Higgs Audio v3 TTS server."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers["Content-Type"] = "application/json"
        self._server_ok = False

    def check_health(self) -> bool:
        try:
            resp = self.session.get(f"{self.base_url}/health", timeout=10)
            self._server_ok = resp.status_code == 200
            return self._server_ok
        except Exception:
            self._server_ok = False
            return False

    def generate_speech(
        self,
        text: str,
        ref_audio_path: str,
        ref_text: str = "",
        temperature: float = 0.8,
        timeout: int = 300,
        max_retries: int = 3,
    ) -> Optional[bytes]:
        payload = {
            "input": text,
            "references": [{"audio_path": ref_audio_path, "text": ref_text}],
            "temperature": temperature,
            "top_k": 50,
            "max_new_tokens": MAX_NEW_TOKENS,
        }
        for attempt in range(max_retries):
            try:
                resp = self.session.post(
                    f"{self.base_url}/v1/audio/speech",
                    json=payload,
                    timeout=timeout,
                )
                if resp.status_code == 200:
                    return resp.content
                elif resp.status_code >= 500:
                    time.sleep((2**attempt) * 2)
                else:
                    print(f"  API error {resp.status_code}: {resp.text[:200]}")
                    return None
            except requests.exceptions.ConnectionError:
                time.sleep(5)
            except requests.exceptions.Timeout:
                time.sleep(5)
            except Exception as e:
                print(f"  Unexpected error: {e}")
                time.sleep(5)
        return None


# ---- Text Selection ----


def load_texts(jsonl_path: str) -> List[dict]:
    texts = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    item = json.loads(line)
                    if item.get("clean_text") or item.get("text"):
                        texts.append(item)
                except json.JSONDecodeError:
                    continue
    print(f"Loaded {len(texts)} texts", flush=True)
    return texts


def make_seed(uid: str, base: int) -> int:
    """Deterministic seed from uid string (stable across Python runs)."""
    return int(hashlib.md5(uid.encode()).hexdigest(), 16) % 100000 + base


def pick_text(texts: List[dict], uid: str, clone_idx: int, global_seed: int) -> dict:
    """Pick one text independently per clone (deterministic for resume)."""
    rng = random.Random(make_seed(uid, global_seed + clone_idx + 100003))
    return rng.choice(texts)


def _combo_to_ref(combo: list, ref_type: str) -> dict:
    silence = 0.3 * (len(combo) - 1) if len(combo) > 1 else 0.0
    total = sum(c[1] for c in combo) + silence
    return {
        "type": ref_type,
        "ref_audio_path": combo[0][0] if ref_type == "single" else None,
        "duration_sec": round(total, 2),
        "transcript": combo[0][2] if ref_type == "single" else " ".join(c[2] for c in combo if c[2]),
        "source_files": [c[0] for c in combo],
        "source_durations": [round(c[1], 2) for c in combo],
        "num_concat_clips": len(combo),
    }


def _sample_concat_combos(
    short: list,
    min_dur: float,
    max_dur: float,
    max_concat: int,
    rng: random.Random,
    max_combos: int,
) -> list[list]:
    """Collect valid concat combinations via random sampling + small exhaustive pass."""
    valid: list[list] = []
    seen: set[tuple[str, ...]] = set()
    max_attempts = max(800, max_combos * 12)
    attempts = 0
    while len(valid) < max_combos and attempts < max_attempts:
        attempts += 1
        n = rng.randint(2, min(max_concat, len(short)))
        combo = rng.sample(short, n)
        key = tuple(c[0] for c in combo)
        if key in seen:
            continue
        seen.add(key)
        total = sum(c[1] for c in combo) + 0.3 * (n - 1)
        if min_dur <= total <= max_dur:
            valid.append(combo)

    if len(short) <= 24:
        for n in range(2, min(max_concat + 1, len(short) + 1)):
            silence = 0.3 * (n - 1)
            for combo in combinations(short, n):
                key = tuple(c[0] for c in combo)
                if key in seen:
                    continue
                seen.add(key)
                total = sum(c[1] for c in combo) + silence
                if min_dur <= total <= max_dur:
                    valid.append(list(combo))
                    if len(valid) >= max_combos:
                        return valid
    return valid


def build_ref_pool(
    speaker_path: str,
    seed: int,
    min_dur: float = 7.0,
    max_dur: float = 20.0,
    max_concat: int = 5,
    max_pool: int = DEFAULT_REF_POOL_SIZE,
) -> list[dict]:
    """Build a pool of valid single/concat reference candidates, then shuffle."""
    rng = random.Random(seed)
    audio_files = _list_audio(speaker_path)
    if not audio_files:
        return []

    clips = []
    for p in audio_files:
        dur = _get_duration(p)
        if dur > 0.5:
            transcript = ""
            json_path = p + ".json"
            if os.path.exists(json_path):
                try:
                    with open(json_path) as jf:
                        transcript = json.load(jf).get("transcript", "")
                except Exception:
                    pass
            clips.append((p, dur, transcript))
    if not clips:
        return []

    pool: list[dict] = []
    singles = [c for c in clips if min_dur <= c[1] <= max_dur]
    for c in singles:
        pool.append(_combo_to_ref([c], "single"))

    short = [c for c in clips if c[1] < max_dur]
    if len(short) >= 2:
        combos = _sample_concat_combos(short, min_dur, max_dur, max_concat, rng, max_pool)
        for combo in combos:
            pool.append(_combo_to_ref(combo, "concat"))

    if not pool and len(short) >= 2:
        legacy = _legacy_concat_fallback(clips, min_dur, max_dur, max_concat, seed)
        if legacy:
            pool.append(legacy)

    rng.shuffle(pool)
    return pool[:max_pool]


def pick_ref_candidate(
    pool: list[dict],
    uid: str,
    clone_idx: int,
    ref_mode: str,
    ref_rotate_every: int,
    global_seed: int,
) -> dict:
    if not pool:
        raise ValueError("empty ref pool")
    if ref_mode == "fixed":
        rng = random.Random(make_seed(uid, global_seed))
        return rng.choice(pool)
    if ref_mode == "rotate":
        slot = clone_idx // ref_rotate_every
        rng = random.Random(make_seed(uid, global_seed + slot))
        return rng.choice(pool)
    rng = random.Random(make_seed(uid, global_seed + clone_idx + 200007))
    return rng.choice(pool)


def _list_audio(speaker_dir: str) -> list:
    files = []
    has_subdirs = False
    try:
        for entry in os.scandir(speaker_dir):
            if entry.is_file() and os.path.splitext(entry.name)[1].lower() in AUDIO_EXTENSIONS:
                files.append(entry.path)
            elif entry.is_dir():
                has_subdirs = True
    except OSError:
        return []
    if has_subdirs:
        files = []
        for root, _dirs, filenames in os.walk(speaker_dir):
            for fname in filenames:
                if os.path.splitext(fname)[1].lower() in AUDIO_EXTENSIONS:
                    files.append(os.path.join(root, fname))
    return files


def _get_duration(path: str) -> float:
    try:
        import soundfile as sf

        return sf.info(path).duration
    except Exception:
        return 0.0


def _legacy_concat_fallback(
    clips: list,
    min_dur: float,
    max_dur: float,
    max_concat: int,
    seed: int,
) -> Optional[dict]:
    """Fallback when no in-range combo found: pick closest concat to target center."""
    rng = random.Random(seed)
    short = sorted([c for c in clips if c[1] < max_dur], key=lambda x: -x[1])
    if len(short) < 2:
        return None
    best = None
    best_dist = float("inf")
    target_center = (min_dur + max_dur) / 2
    for n in range(2, min(max_concat + 1, len(short) + 1)):
        silence = 0.3 * (n - 1)
        for _ in range(min(20, len(short) * 2)):
            rng.shuffle(short)
            combo = short[:n]
            total = sum(c[1] for c in combo) + silence
            dist = abs(total - target_center)
            if min_dur <= total <= max_dur:
                return _combo_to_ref(combo, "concat")
            if dist < best_dist:
                best_dist = dist
                best = _combo_to_ref(combo, "concat")
    return best


def select_ref_audio(
    speaker_path: str, min_dur: float = 7.0, max_dur: float = 20.0, max_concat: int = 5, seed: int = 42
) -> Optional[dict]:
    """Pick one reference (backward-compatible wrapper)."""
    pool = build_ref_pool(speaker_path, seed, min_dur, max_dur, max_concat, max_pool=DEFAULT_REF_POOL_SIZE)
    if pool:
        rng = random.Random(seed)
        return rng.choice(pool)
    return None


def _ref_cache_key(ref_info: dict) -> str:
    return "|".join(ref_info.get("source_files", []))


def speaker_ref_dir(out_dir: str) -> str:
    return os.path.join(out_dir, REF_SUBDIR)


def materialize_ref(ref_info: dict, ref_dir: str, cache_key: str) -> tuple[str, str, str]:
    """Write ref wav under ref_dir; return (wav_path, src_ref, transcript)."""
    os.makedirs(ref_dir, exist_ok=True)
    safe_key = hashlib.md5(cache_key.encode()).hexdigest()[:12]
    dst_ref = os.path.join(ref_dir, f"ref_{safe_key}.wav")
    if ref_info["type"] == "concat":
        if not (os.path.isfile(dst_ref) and os.path.getsize(dst_ref) > 1000):
            ok, _ = concat_audio(ref_info["source_files"], dst_ref)
            if not ok:
                raise RuntimeError("Failed to concatenate reference audio")
        return dst_ref, dst_ref, ref_info.get("transcript", "")

    src_ref = ref_info["ref_audio_path"]
    if not (os.path.isfile(dst_ref) and os.path.getsize(dst_ref) > 1000):
        try:
            shutil.copy2(src_ref, dst_ref)
        except OSError as exc:
            raise RuntimeError(f"Failed to copy reference audio: {exc}") from exc
    return dst_ref, src_ref, ref_info.get("transcript", "")


def concat_audio(file_paths: List[str], output_path: str, sample_rate: int = 16000):
    import soundfile as sf
    import librosa

    segments = []
    for fp in file_paths:
        try:
            audio, _ = librosa.load(fp, sr=sample_rate, mono=True)
            segments.append(audio)
        except Exception as e:
            print(f"  Warning: failed to load {fp}: {e}", flush=True)
            return False, 0.0
    if not segments:
        return False, 0.0
    silence_dur = 0.3 * (len(segments) - 1) if len(segments) > 1 else 0.0
    if len(segments) > 1:
        silence = np.zeros(int(0.3 * sample_rate))
        parts = []
        for i, seg in enumerate(segments):
            parts.append(seg)
            if i < len(segments) - 1:
                parts.append(silence)
        combined = np.concatenate(parts)
    else:
        combined = segments[0]
    sf.write(output_path, combined, sample_rate)
    return True, silence_dur


def downsample_wav_file(path: str, target_sr: int = DEFAULT_OUTPUT_SAMPLE_RATE) -> tuple[int, int]:
    """Resample wav in place to target_sr. Returns (original_sr, target_sr)."""
    import librosa
    import soundfile as sf

    audio, sr = librosa.load(path, sr=None, mono=True)
    if sr == target_sr:
        return sr, target_sr
    resampled = librosa.resample(audio, orig_sr=sr, target_sr=target_sr, res_type="soxr_vhq")
    sf.write(path, resampled, target_sr)
    return sr, target_sr


# ---- Clone One Speaker ----


def clone_speaker(
    server_url: str,
    task: dict,
    texts: List[dict],
    output_root: str,
    clones_needed: int,
    start_clone_idx: int = 0,
    ref_mode: str = DEFAULT_REF_MODE,
    ref_rotate_every: int = DEFAULT_REF_ROTATE_EVERY,
    ref_pool_size: int = DEFAULT_REF_POOL_SIZE,
    global_seed: int = 42,
    output_sample_rate: int = DEFAULT_OUTPUT_SAMPLE_RATE,
) -> dict:
    client = SGLangTTSClient(server_url)
    dataset = task["dataset"]
    speaker_id = task["speaker_id"]
    uid = f"{dataset}__{speaker_id}"

    out_dir = os.path.join(output_root, dataset, speaker_id)
    ref_dir = speaker_ref_dir(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(ref_dir, exist_ok=True)

    result = {
        "uid": uid,
        "dataset": dataset,
        "speaker_id": speaker_id,
        "clones_needed": clones_needed,
        "start_clone_idx": start_clone_idx,
        "clones_done": 0,
        "clones_failed": 0,
        "server": server_url,
        "ref_mode": ref_mode,
    }

    ref_pool = build_ref_pool(task["speaker_path"], make_seed(uid, global_seed), max_pool=ref_pool_size)
    if not ref_pool:
        result["error"] = "No suitable reference audio found"
        return result

    pool_meta = {
        "uid": uid,
        "ref_mode": ref_mode,
        "ref_rotate_every": ref_rotate_every,
        "ref_pool_size": len(ref_pool),
        "global_seed": global_seed,
        "candidates": [
            {
                "type": r["type"],
                "duration_sec": r["duration_sec"],
                "source_files": r.get("source_files", []),
                "source_durations": r.get("source_durations", []),
                "num_concat_clips": r.get("num_concat_clips", 1),
            }
            for r in ref_pool
        ],
    }
    with open(os.path.join(ref_dir, "ref_pool.json"), "w") as jf:
        json.dump(pool_meta, jf, ensure_ascii=False, indent=2)

    file_start, _text_base = _resolve_clone_indices(out_dir, start_clone_idx)
    ref_cache: dict[str, tuple[str, str, str, dict]] = {}

    for j in range(clones_needed):
        i = file_start + j
        clone_wav = os.path.join(out_dir, f"clone_{i:04d}.wav")
        clone_json = os.path.join(out_dir, f"clone_{i:04d}.json")

        if os.path.exists(clone_wav) and os.path.getsize(clone_wav) > 1000 and os.path.exists(clone_json):
            result["clones_done"] += 1
            continue

        try:
            ref_info = pick_ref_candidate(ref_pool, uid, i, ref_mode, ref_rotate_every, global_seed)
            if ref_mode == "rotate":
                slot = i // ref_rotate_every
                cache_key = f"slot_{slot}|{_ref_cache_key(ref_info)}"
            elif ref_mode == "fixed":
                cache_key = f"fixed|{_ref_cache_key(ref_info)}"
            else:
                cache_key = f"clone_{i}|{_ref_cache_key(ref_info)}"

            if cache_key not in ref_cache:
                dst_ref, src_ref, ref_transcript = materialize_ref(ref_info, ref_dir, cache_key)
                ref_cache[cache_key] = (dst_ref, src_ref, ref_transcript, ref_info)
            else:
                dst_ref, src_ref, ref_transcript, ref_info = ref_cache[cache_key]

            text_item = pick_text(texts, uid, i, global_seed)
            text = text_item.get("text", text_item.get("clean_text", ""))
            if not text:
                result["clones_failed"] += 1
                continue

            audio_bytes = client.generate_speech(
                text=text,
                ref_audio_path=dst_ref,
                ref_text=ref_transcript,
            )
            if audio_bytes and len(audio_bytes) > 100:
                with open(clone_wav, "wb") as wf:
                    wf.write(audio_bytes)
                generated_sr, sample_rate = downsample_wav_file(clone_wav, output_sample_rate)
                meta = {
                    "clone_idx": i,
                    "uid": uid,
                    "dataset": dataset,
                    "speaker_id": speaker_id,
                    "text": text,
                    "clean_text": text_item.get("clean_text", ""),
                    "emotion": text_item.get("emotion", ""),
                    "scenario": text_item.get("scenario", ""),
                    "tags_used": text_item.get("tags_used", []),
                    "ref_audio_path": dst_ref,
                    "ref_audio_source": src_ref,
                    "ref_transcript": ref_transcript,
                    "ref_audio_duration_sec": ref_info["duration_sec"],
                    "ref_audio_type": ref_info["type"],
                    "ref_mode": ref_mode,
                    "ref_pool_size": len(ref_pool),
                    "audio_format": "wav",
                    "generated_sample_rate": generated_sr,
                    "sample_rate": sample_rate,
                }
                with open(clone_json, "w") as jf:
                    json.dump(meta, jf, ensure_ascii=False, indent=2)
                result["clones_done"] += 1
            else:
                result["clones_failed"] += 1
        except Exception as e:
            result["clones_failed"] += 1
            print(f"  [{i}] ERROR: {e}", flush=True)

    return result


# ---- Worker ----


def clone_worker(args_tuple):
    (
        server_url,
        task,
        texts,
        output_root,
        clones_needed,
        start_clone_idx,
        ref_mode,
        ref_rotate_every,
        ref_pool_size,
        global_seed,
        output_sample_rate,
    ) = args_tuple
    try:
        return clone_speaker(
            server_url,
            task,
            texts,
            output_root,
            clones_needed,
            start_clone_idx,
            ref_mode,
            ref_rotate_every,
            ref_pool_size,
            global_seed,
            output_sample_rate,
        )
    except Exception:
        return {"uid": f"{task['dataset']}__{task['speaker_id']}", "error": traceback.format_exc()}


# ---- Main ----


def main():
    parser = argparse.ArgumentParser(description="TTS Voice Cloning via SGLang Local Servers")
    parser.add_argument("--stats-csv", required=True)
    parser.add_argument("--texts-jsonl", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--base-port", type=int, default=8000)
    parser.add_argument("--num-servers", type=int, default=2)
    parser.add_argument("--quality-pass-rate", type=float, default=0.5)
    parser.add_argument("--workers-per-server", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-speakers", type=int, default=None)
    parser.add_argument("--max-clones-per-speaker", type=int, default=None)
    parser.add_argument("--estimate-clone-duration", type=float, default=10.0)
    parser.add_argument(
        "--post-prune",
        action="store_true",
        help="Use post-prune stats CSV (combined duration + precomputed clones_needed/start_clone_idx)",
    )
    parser.add_argument(
        "--ref-mode",
        choices=("fixed", "rotate", "random"),
        default=DEFAULT_REF_MODE,
        help="Reference selection: fixed=one ref/speaker; rotate=every N clones; random=per clone",
    )
    parser.add_argument(
        "--ref-rotate-every",
        type=int,
        default=DEFAULT_REF_ROTATE_EVERY,
        help="When --ref-mode=rotate, switch reference every N clones",
    )
    parser.add_argument(
        "--ref-pool-size",
        type=int,
        default=DEFAULT_REF_POOL_SIZE,
        help="Max candidate references (single + concat combos) per speaker",
    )
    parser.add_argument(
        "--output-sample-rate",
        type=int,
        default=DEFAULT_OUTPUT_SAMPLE_RATE,
        help="Downsample generated clone wav to this sample rate (TTS outputs 24kHz)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help="SGLang audio-token cap (1024~=40.7s; raise so long texts aren't truncated mid-utterance)",
    )
    args = parser.parse_args()

    global MAX_NEW_TOKENS
    MAX_NEW_TOKENS = args.max_new_tokens
    print(f"max_new_tokens (audio budget) = {MAX_NEW_TOKENS} (~{MAX_NEW_TOKENS / 25.14:.0f}s cap)", flush=True)

    # Servers
    servers = [f"http://localhost:{args.base_port + i}" for i in range(args.num_servers)]
    print(f"Servers: {servers}", flush=True)

    # Health check
    for url in servers:
        c = SGLangTTSClient(url)
        if c.check_health():
            print(f"  {url} → OK", flush=True)
        else:
            print(f"  {url} → NOT REACHABLE (make sure SGLang servers are running)", flush=True)
            sys.exit(1)

    # Load tasks — filter: <3600s AND >= 20 audio files (or post-prune resume CSV)
    tasks = []
    all_low = 0
    filtered_out = 0
    with open(args.stats_csv) as f:
        reader = csv.DictReader(f)
        post_prune = args.post_prune or ("clone_duration_sec" in (reader.fieldnames or []))
        f.seek(0)
        reader = csv.DictReader(f)
        for r in reader:
            num_files = int(r["num_files"])
            if post_prune:
                if r.get("status") == "OK":
                    continue
                clones_needed = int(r.get("clones_needed") or 0)
                if clones_needed <= 0:
                    continue
                if num_files < 20:
                    filtered_out += 1
                    continue
                tasks.append(
                    {
                        "dataset": r["dataset"],
                        "speaker_id": r["speaker_id"],
                        "speaker_path": r["speaker_path"],
                        "current_dur": float(r["total_duration_sec"]),
                        "num_files": num_files,
                        "clones_needed": clones_needed,
                        "start_clone_idx": int(r.get("start_clone_idx") or 0),
                    }
                )
                continue

            dur = float(r["total_duration_sec"])
            if dur < 3600:
                all_low += 1
                if num_files < 20:
                    filtered_out += 1
                    continue
                gap = 3600 - dur
                clones_net = int(gap / args.estimate_clone_duration) + 1
                clones_total = int(clones_net / args.quality_pass_rate) + 1
                tasks.append(
                    {
                        "dataset": r["dataset"],
                        "speaker_id": r["speaker_id"],
                        "speaker_path": r["speaker_path"],
                        "current_dur": dur,
                        "num_files": num_files,
                        "clones_needed": clones_total,
                        "start_clone_idx": 0,
                    }
                )

    if post_prune:
        print(f"Post-prune resume: {len(tasks)} speakers need more clones", flush=True)
    else:
        print(f"Speakers <1h: {all_low}, discarded (<20 clips): {filtered_out}, to clone: {len(tasks)}", flush=True)
    total_clones = sum(t["clones_needed"] for t in tasks)
    print(f"Total clones to generate: {total_clones:,}", flush=True)
    print(f"Estimated added duration: {total_clones * args.estimate_clone_duration / 3600:.0f}h", flush=True)
    print(
        f"(quality pass rate: {args.quality_pass_rate * 100:.0f}%, avg clone: {args.estimate_clone_duration:.0f}s)",
        flush=True,
    )

    if args.max_speakers:
        tasks = tasks[: args.max_speakers]
        print(f"Limited to {len(tasks)} speakers (--max-speakers)", flush=True)

    if args.max_clones_per_speaker:
        for task in tasks:
            task["clones_needed"] = min(task["clones_needed"], args.max_clones_per_speaker)
        total_clones = sum(t["clones_needed"] for t in tasks)
        print(
            f"Limited to {args.max_clones_per_speaker} clones per speaker "
            f"(--max-clones-per-speaker), total clones: {total_clones:,}",
            flush=True,
        )

    print(
        f"Randomization: ref_mode={args.ref_mode}, ref_rotate_every={args.ref_rotate_every}, "
        f"ref_pool_size={args.ref_pool_size}, seed={args.seed}, "
        f"output_sample_rate={args.output_sample_rate}",
        flush=True,
    )

    # Load texts
    texts = load_texts(args.texts_jsonl)
    if not texts:
        print("ERROR: No texts found", flush=True)
        sys.exit(1)

    os.makedirs(args.output_root, exist_ok=True)
    t_start = time.time()

    # Assign speakers to servers round-robin
    server_tasks = {url: [] for url in servers}
    for i, task in enumerate(tasks):
        url = servers[i % len(servers)]
        server_tasks[url].append(task)

    all_results = []

    # Process each server concurrently
    with ThreadPoolExecutor(max_workers=len(servers)) as server_pool:
        server_futures = {}
        for url, stasks in server_tasks.items():
            if not stasks:
                continue
            future = server_pool.submit(
                _process_server_tasks,
                url,
                stasks,
                texts,
                args.output_root,
                args.workers_per_server,
                args.seed,
                args.ref_mode,
                args.ref_rotate_every,
                args.ref_pool_size,
                args.output_sample_rate,
            )
            server_futures[future] = url

        for future in as_completed(server_futures):
            url = server_futures[future]
            try:
                results = future.result()
                all_results.extend(results)
                server_done = sum(r.get("clones_done", 0) for r in results)
                server_failed = sum(r.get("clones_failed", 0) for r in results)
                print(
                    f"[{url}] DONE: {len(results)} speakers, {server_done} clones, {server_failed} failed", flush=True
                )
            except Exception as e:
                print(f"[{url}] ERROR: {e}", flush=True)

    elapsed = time.time() - t_start
    total_done = sum(r.get("clones_done", 0) for r in all_results)
    total_failed = sum(r.get("clones_failed", 0) for r in all_results)

    # Save summary
    summary_path = os.path.join(args.output_root, "clone_summary.json")
    with open(summary_path, "w") as f:
        json.dump(
            {
                "speakers_processed": len(all_results),
                "clones_done": total_done,
                "clones_failed": total_failed,
                "elapsed_sec": elapsed,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\n=== Complete ===", flush=True)
    print(f"Speakers: {len(all_results)}", flush=True)
    print(f"Clones done: {total_done}", flush=True)
    print(f"Clones failed: {total_failed}", flush=True)
    print(f"Time: {elapsed / 3600:.1f}h", flush=True)


def _process_server_tasks(
    server_url,
    tasks,
    texts,
    output_root,
    workers,
    global_seed,
    ref_mode,
    ref_rotate_every,
    ref_pool_size,
    output_sample_rate,
):
    """Process all tasks assigned to one server with concurrent workers."""
    results = []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        work_items = [
            (
                server_url,
                t,
                texts,
                output_root,
                t["clones_needed"],
                t.get("start_clone_idx", 0),
                ref_mode,
                ref_rotate_every,
                ref_pool_size,
                global_seed,
                output_sample_rate,
            )
            for t in tasks
        ]
        futures = {pool.submit(clone_worker, wi): wi[1]["speaker_id"] for wi in work_items}

        for future in as_completed(futures):
            try:
                r = future.result()
                results.append(r)
                print(f"[{r.get('uid', '?')}] done={r.get('clones_done', 0)}/{r.get('clones_needed', 0)}", flush=True)
            except Exception as e:
                print(f"[{server_url}] ERROR: {e}", flush=True)

    return results


if __name__ == "__main__":
    main()
