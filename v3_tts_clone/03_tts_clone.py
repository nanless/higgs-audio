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
        --num-servers 2 \
        --workers-per-server 8
"""

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

import numpy as np
import requests


AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3"}


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
            "max_new_tokens": 1024,
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


def pick_texts(texts: List[dict], n: int, seed: int) -> List[dict]:
    rng = random.Random(seed)
    candidates = list(texts)
    rng.shuffle(candidates)
    return candidates[:n]


# ---- Reference Audio Selection ----


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


def select_ref_audio(
    speaker_path: str, min_dur: float = 7.0, max_dur: float = 20.0, max_concat: int = 5, seed: int = 42
) -> Optional[dict]:
    """Select a 7-20s audio clip. Single clip preferred; concat if needed."""
    rng = random.Random(seed)
    audio_files = _list_audio(speaker_path)
    if not audio_files:
        return None

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
        return None

    candidates = [(p, d, t) for p, d, t in clips if min_dur <= d <= max_dur]
    if candidates:
        sel = rng.choice(candidates)
        return {
            "type": "single",
            "ref_audio_path": sel[0],
            "duration_sec": round(sel[1], 2),
            "transcript": sel[2],
            "source_files": [sel[0]],
            "source_durations": [round(sel[1], 2)],
        }

    short = sorted([c for c in clips if c[1] < max_dur], key=lambda x: -x[1])
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
                return {
                    "type": "concat",
                    "ref_audio_path": None,
                    "duration_sec": round(total, 2),
                    "transcript": " ".join(c[2] for c in combo if c[2]),
                    "source_files": [c[0] for c in combo],
                    "source_durations": [round(c[1], 2) for c in combo],
                    "num_concat_clips": n,
                }
            if dist < best_dist:
                best_dist = dist
                best = {
                    "type": "concat",
                    "ref_audio_path": None,
                    "duration_sec": round(total, 2),
                    "transcript": " ".join(c[2] for c in combo if c[2]),
                    "source_files": [c[0] for c in combo],
                    "source_durations": [round(c[1], 2) for c in combo],
                    "num_concat_clips": n,
                }
    return best


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


# ---- Clone One Speaker ----


def clone_speaker(server_url: str, task: dict, texts: List[dict], output_root: str, clones_needed: int) -> dict:
    client = SGLangTTSClient(server_url)
    dataset = task["dataset"]
    speaker_id = task["speaker_id"]
    uid = f"{dataset}__{speaker_id}"

    out_dir = os.path.join(output_root, dataset, speaker_id)
    os.makedirs(out_dir, exist_ok=True)

    result = {
        "uid": uid,
        "dataset": dataset,
        "speaker_id": speaker_id,
        "clones_needed": clones_needed,
        "clones_done": 0,
        "clones_failed": 0,
        "server": server_url,
    }

    # Select reference audio (deterministic seed)
    ref_info = select_ref_audio(task["speaker_path"], seed=make_seed(uid, 0))
    if ref_info is None:
        result["error"] = "No suitable reference audio found"
        return result

    # Resolve ref audio path
    if ref_info["type"] == "concat":
        dst_ref = os.path.join(out_dir, "ref_audio.wav")
        ok, _ = concat_audio(ref_info["source_files"], dst_ref)
        if not ok:
            result["error"] = "Failed to concatenate reference audio"
            return result
        src_ref = dst_ref
    else:
        src_ref = ref_info["ref_audio_path"]
        dst_ref = os.path.join(out_dir, "ref_audio.wav")
        try:
            shutil.copy2(src_ref, dst_ref)
        except OSError:
            pass

    ref_transcript = ref_info.get("transcript", "")

    # Save ref metadata
    ref_meta = {
        "uid": uid,
        "dataset": dataset,
        "speaker_id": speaker_id,
        "ref_audio_path": dst_ref,
        "ref_audio_source": src_ref,
        "ref_audio_duration_sec": ref_info["duration_sec"],
        "ref_audio_type": ref_info["type"],
        "ref_transcript": ref_transcript,
        "source_files": ref_info.get("source_files", []),
        "source_durations": ref_info.get("source_durations", []),
        "num_concat_clips": ref_info.get("num_concat_clips", 1),
    }
    with open(os.path.join(out_dir, "ref_audio.json"), "w") as jf:
        json.dump(ref_meta, jf, ensure_ascii=False, indent=2)

    # Pick texts for this speaker
    seed = make_seed(uid, 0)
    texts_picked = pick_texts(texts, clones_needed, seed)
    while len(texts_picked) < clones_needed:
        texts_picked.append({"text": "Hello, this is a test.", "clean_text": "Hello, this is a test."})

    # Generate clones — no sleep between requests for local server
    for i, text_item in enumerate(texts_picked):
        text = text_item.get("text", text_item.get("clean_text", ""))
        if not text:
            continue

        clone_wav = os.path.join(out_dir, f"clone_{i:04d}.wav")
        clone_json = os.path.join(out_dir, f"clone_{i:04d}.json")

        # Resume: skip if already generated
        if os.path.exists(clone_wav) and os.path.getsize(clone_wav) > 1000 and os.path.exists(clone_json):
            result["clones_done"] += 1
            continue

        try:
            audio_bytes = client.generate_speech(
                text=text,
                ref_audio_path=dst_ref,
                ref_text=ref_transcript,
            )
            if audio_bytes and len(audio_bytes) > 100:
                with open(clone_wav, "wb") as wf:
                    wf.write(audio_bytes)
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
                    "ref_audio_source": src_ref,
                    "ref_transcript": ref_transcript,
                    "ref_audio_duration_sec": ref_info["duration_sec"],
                    "audio_format": "wav",
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
    (server_url, task, texts, output_root, clones_needed) = args_tuple
    try:
        return clone_speaker(server_url, task, texts, output_root, clones_needed)
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
    parser.add_argument("--workers-per-server", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-speakers", type=int, default=None)
    parser.add_argument("--max-clones-per-speaker", type=int, default=None)
    parser.add_argument("--estimate-clone-duration", type=float, default=10.0)
    args = parser.parse_args()

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

    # Load tasks — filter: <3600s AND >= 20 audio files
    tasks = []
    all_low = 0
    filtered_out = 0
    with open(args.stats_csv) as f:
        for r in csv.DictReader(f):
            dur = float(r["total_duration_sec"])
            num_files = int(r["num_files"])
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
                    }
                )

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
                _process_server_tasks, url, stasks, texts, args.output_root, args.workers_per_server, args.seed
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


def _process_server_tasks(server_url, tasks, texts, output_root, workers, seed):
    """Process all tasks assigned to one server with concurrent workers."""
    results = []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        work_items = [(server_url, t, texts, output_root, t["clones_needed"]) for t in tasks]
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
