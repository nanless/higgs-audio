#!/usr/bin/env python3
# Copyright (c) 2025 Boson AI
#
# Clone multi-sentence scripts via SGLang-Omni.
# Per-speaker layout:
#   {out}/{dataset}/{speaker}/ref/{ref.wav,ref.json}
#   {out}/{dataset}/{speaker}/clone/{clone.wav,clone.json}
#   {out}/{dataset}/{speaker}/speaker.json
"""
Usage:
  python 03_run_clone.py \
    --speakers-json ./workdir/selected_speakers.json \
    --scripts-json ./workdir/clone_scripts.json \
    --output-dir ./workdir/clones \
    --base-url http://localhost:8000 \
    --mode single
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import re
import sys
import threading
import time
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np
import requests

from stable_seed import stable_int

DEFAULT_AUDIO_ROOT = (
    "/root/group-shared/voiceprint/data/speech/speaker_diarization/"
    "merged_datasets_20250610_vad_segments_mtfaa_enhanced_extend_kid_withclone_addlibrilight_1130/audio"
)

_VAD_MODEL = None
_VAD_LOCK = threading.Lock()


def under_audio_root(path: str, audio_root: str) -> bool:
    if not path or not audio_root:
        return False
    root = os.path.abspath(audio_root)
    ap = os.path.abspath(path)
    return ap == root or ap.startswith(root + os.sep)


def _get_vad_model():
    global _VAD_MODEL
    if _VAD_MODEL is None:
        with _VAD_LOCK:
            if _VAD_MODEL is None:
                from silero_vad import load_silero_vad

                _VAD_MODEL = load_silero_vad()
    return _VAD_MODEL


def _energy_vad_segments(
    audio: np.ndarray,
    sr: int,
    *,
    frame_ms: float = 20.0,
    thresh: float = 0.01,
    min_speech_ms: float = 200.0,
    merge_gap_ms: float = 250.0,
) -> list[tuple[int, int]]:
    """Simple energy VAD fallback (sample indices)."""
    frame = max(1, int(sr * frame_ms / 1000.0))
    n = len(audio) // frame
    if n <= 0:
        return [(0, len(audio))]
    ener = np.array(
        [float(np.sqrt(np.mean(audio[i * frame : (i + 1) * frame] ** 2) + 1e-12)) for i in range(n)],
        dtype=np.float32,
    )
    speech = ener > thresh
    min_frames = max(1, int(min_speech_ms / frame_ms))
    merge_frames = max(1, int(merge_gap_ms / frame_ms))
    segs: list[tuple[int, int]] = []
    i = 0
    while i < n:
        if not speech[i]:
            i += 1
            continue
        j = i
        while j < n and speech[j]:
            j += 1
        if j - i >= min_frames:
            segs.append((i * frame, min(len(audio), j * frame)))
        i = j
    if not segs:
        return [(0, len(audio))]
    # merge close segments
    merged = [segs[0]]
    for a, b in segs[1:]:
        pa, pb = merged[-1]
        if a - pb <= merge_frames * frame:
            merged[-1] = (pa, b)
        else:
            merged.append((a, b))
    return merged


_PAUSE_TAG_RE = re.compile(r"<\|prosody:long_pause\|>")
_EMOTION_TAG_RE = re.compile(r"<\|emotion:[a-z_]+\|>")
_DELIVERY_TAG_RE = re.compile(r"<\|(?:emotion|style|prosody|sfx):[^|>]+\|>")


def strip_pause_tags(text: str) -> str:
    """Remove model pause tags; pause length is controlled by VAD splice only."""
    return _PAUSE_TAG_RE.sub("", text or "")


def ensure_sentence_emotion_tags(
    sentences: list[str],
    emotion: str | None,
    prosody: str | None = None,
    style: str | None = None,
) -> list[str]:
    """Prefix every sentence with the same emotion/prosody/style for concat TTS.

    Concat mode synthesizes each sentence independently; without per-sentence
    emotion tags the model drifts to neutral on later lines.
    """
    emotion = (emotion or "").strip()
    if not emotion:
        return list(sentences)
    parts = [f"<|emotion:{emotion}|>"]
    if style:
        parts.append(f"<|style:{style}|>")
    if prosody:
        parts.append(f"<|prosody:{prosody}|>")
    hdr = "".join(parts)
    out: list[str] = []
    for sent in sentences:
        s = sent or ""
        if _EMOTION_TAG_RE.search(s):
            out.append(s)
            continue
        # Keep any leading SFX/style already present; strip only if bare.
        out.append(hdr + s)
    return out


def sentences_have_consistent_emotion(sentences: list[str], emotion: str | None) -> bool:
    emotion = (emotion or "").strip()
    if not emotion or not sentences:
        return False
    needle = f"<|emotion:{emotion}|>"
    return all(needle in (s or "") for s in sentences)


def vad_splice_with_pauses(
    audio: np.ndarray,
    sr: int,
    pause_sec: float | None = None,
    *,
    pause_sec_min: float | None = None,
    pause_sec_max: float | None = None,
    pause_secs_designed: list[float] | None = None,
    rng: random.Random | None = None,
    min_speech_ms: int = 200,
    min_silence_ms: int = 100,
) -> tuple[np.ndarray, list[dict[str, float]], list[float]]:
    """Detect speech spans, then re-concat with silence gaps.

    Prefer ``pause_secs_designed`` (in order) for each gap. Extra VAD gaps
    fall back to uniform sampling in [pause_sec_min, pause_sec_max].
    Returns (audio, speech_segment_meta, pause_secs_between_segments).
    """
    if audio.ndim > 1:
        audio = audio.mean(axis=0)
    audio = np.asarray(audio, dtype=np.float32)
    rng = rng or random.Random()
    designed = [max(1.0, float(p)) for p in (pause_secs_designed or [])]

    if pause_sec_min is None or pause_sec_max is None:
        if designed:
            pause_sec_min = min(designed) if pause_sec_min is None else pause_sec_min
            pause_sec_max = max(designed) if pause_sec_max is None else pause_sec_max
        else:
            if pause_sec is None:
                pause_sec = 2.0
            lo = pause_sec * 0.6
            hi = pause_sec * 1.4
            pause_sec_min = lo if pause_sec_min is None else pause_sec_min
            pause_sec_max = hi if pause_sec_max is None else pause_sec_max
    pause_sec_min = float(pause_sec_min)
    pause_sec_max = float(pause_sec_max)
    if pause_sec_max < pause_sec_min:
        pause_sec_min, pause_sec_max = pause_sec_max, pause_sec_min

    spans: list[tuple[int, int]] = []
    try:
        import torch
        from silero_vad import get_speech_timestamps

        vad_sr = 16000
        if sr != vad_sr:
            n_new = int(round(len(audio) * vad_sr / sr))
            x_old = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
            x_new = np.linspace(0.0, 1.0, num=n_new, endpoint=False)
            wav_16k = np.interp(x_new, x_old, audio).astype(np.float32)
            scale = sr / float(vad_sr)
        else:
            wav_16k = audio
            scale = 1.0
        model = _get_vad_model()
        with _VAD_LOCK:
            ts = get_speech_timestamps(
                torch.from_numpy(wav_16k),
                model,
                sampling_rate=vad_sr,
                min_speech_duration_ms=min_speech_ms,
                min_silence_duration_ms=min_silence_ms,
            )
        for t in ts:
            start = int(round(t["start"] * scale))
            end = int(round(t["end"] * scale))
            start = max(0, min(start, len(audio)))
            end = max(start, min(end, len(audio)))
            if end - start >= int(0.05 * sr):
                spans.append((start, end))
    except Exception:
        spans = []

    if not spans:
        spans = _energy_vad_segments(audio, sr, min_speech_ms=float(min_speech_ms))

    speech_chunks: list[tuple[np.ndarray, dict[str, float]]] = []
    for start, end in spans:
        chunk = audio[start:end]
        if len(chunk) < int(0.05 * sr):
            continue
        speech_chunks.append(
            (
                chunk,
                {
                    "index": len(speech_chunks),
                    "start_sec": round(start / sr, 3),
                    "end_sec": round(end / sr, 3),
                    "dur_sec": round((end - start) / sr, 3),
                },
            )
        )

    if not speech_chunks:
        return (
            audio,
            [{"start_sec": 0.0, "end_sec": round(len(audio) / sr, 3), "note": "vad_empty_fallback"}],
            [],
        )

    segs: list[np.ndarray] = []
    meta: list[dict[str, float]] = []
    pause_secs: list[float] = []
    for i, (chunk, m) in enumerate(speech_chunks):
        segs.append(chunk)
        meta.append(m)
        if i < len(speech_chunks) - 1:
            if i < len(designed):
                p = round(float(designed[i]), 3)
            else:
                p = round(rng.uniform(pause_sec_min, pause_sec_max), 3)
            p = max(1.0, p)
            pause_secs.append(p)
            segs.append(np.zeros(int(round(p * sr)), dtype=np.float32))

    out = np.concatenate(segs)
    return out.astype(np.float32), meta, pause_secs


def read_ref_transcript(wav_path: str) -> str:
    sidecar = wav_path + ".json"
    if not os.path.isfile(sidecar):
        return ""
    try:
        with open(sidecar, encoding="utf-8") as f:
            obj = json.load(f)
        return (obj.get("transcript") or obj.get("text") or "").strip()
    except Exception:
        return ""


def wav_bytes_to_float_mono(data: bytes) -> tuple[np.ndarray, int]:
    """Decode WAV bytes -> float32 mono, sample rate."""
    with wave.open(io.BytesIO(data), "rb") as wf:
        sr = wf.getframerate()
        nch = wf.getnchannels()
        sw = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())
    if sw == 2:
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 4:
        audio = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sw}")
    if nch > 1:
        audio = audio.reshape(-1, nch).mean(axis=1)
    return audio, sr


def write_wav(path: str, audio: np.ndarray, sr: int) -> None:
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767.0).astype(np.int16)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


class TTSClient:
    def __init__(self, base_url: str, max_new_tokens: int = 1024):
        self.base_url = base_url.rstrip("/")
        self.max_new_tokens = max_new_tokens
        self.session = requests.Session()
        self.session.headers["Content-Type"] = "application/json"

    def health(self) -> bool:
        try:
            r = self.session.get(f"{self.base_url}/health", timeout=10)
            return r.status_code == 200
        except Exception:
            return False

    def generate(
        self,
        text: str,
        ref_audio_path: str,
        ref_text: str = "",
        temperature: float = 0.8,
        timeout: int = 300,
        max_retries: int = 3,
    ) -> bytes | None:
        payload = {
            "input": text,
            "references": [{"audio_path": ref_audio_path, "text": ref_text}],
            "temperature": temperature,
            "top_k": 50,
            "max_new_tokens": self.max_new_tokens,
        }
        for attempt in range(max_retries):
            try:
                resp = self.session.post(
                    f"{self.base_url}/v1/audio/speech",
                    json=payload,
                    timeout=timeout,
                )
                if resp.status_code == 200 and len(resp.content) > 100:
                    return resp.content
                if resp.status_code >= 500:
                    time.sleep((2**attempt) * 2)
                    continue
                print(f"  API {resp.status_code}: {resp.text[:200]}", flush=True)
                return None
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                time.sleep(3 + attempt)
            except Exception as e:
                print(f"  error: {e}", flush=True)
                time.sleep(2)
        return None


def load_audio_mono(path: str, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    """Load audio as float32 mono at target_sr."""
    try:
        import librosa

        audio, _ = librosa.load(path, sr=target_sr, mono=True)
        return audio.astype(np.float32), target_sr
    except Exception:
        pass
    try:
        import soundfile as sf

        audio, sr = sf.read(path, always_2d=False)
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=-1)
        if sr != target_sr and len(audio) > 1:
            x_old = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
            n_new = int(round(len(audio) * target_sr / sr))
            x_new = np.linspace(0.0, 1.0, num=max(1, n_new), endpoint=False)
            audio = np.interp(x_new, x_old, audio).astype(np.float32)
        return audio, target_sr
    except Exception:
        pass
    # WAV-only fallback
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        nch = wf.getnchannels()
        sw = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())
    if sw == 2:
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 4:
        audio = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width in {path}: {sw}")
    if nch > 1:
        audio = audio.reshape(-1, nch).mean(axis=1)
    if sr != target_sr and len(audio) > 1:
        x_old = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
        n_new = int(round(len(audio) * target_sr / sr))
        x_new = np.linspace(0.0, 1.0, num=max(1, n_new), endpoint=False)
        audio = np.interp(x_new, x_old, audio).astype(np.float32)
    return audio, target_sr


def concat_ref_files(
    source_files: list[str],
    output_path: str,
    gap_sec: float = 0.3,
    sample_rate: int = 16000,
) -> float:
    """Concatenate source wavs with silence gaps; return total duration seconds."""
    parts: list[np.ndarray] = []
    silence = np.zeros(int(gap_sec * sample_rate), dtype=np.float32)
    for i, fp in enumerate(source_files):
        audio, _ = load_audio_mono(fp, target_sr=sample_rate)
        parts.append(audio)
        if i < len(source_files) - 1:
            parts.append(silence)
    if not parts:
        raise RuntimeError("No source files to concatenate")
    merged = np.concatenate(parts)
    write_wav(output_path, merged, sample_rate)
    return float(len(merged) / sample_rate)


def wav_file_duration(path: str) -> float | None:
    try:
        with wave.open(path, "rb") as wf:
            return round(wf.getnframes() / float(wf.getframerate()), 3)
    except Exception:
        return None


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def speaker_dirs(out_dir: str, dataset: str, speaker_id: str) -> dict[str, str]:
    root = os.path.join(out_dir, dataset, speaker_id)
    ref_dir = os.path.join(root, "ref")
    clone_dir = os.path.join(root, "clone")
    os.makedirs(ref_dir, exist_ok=True)
    os.makedirs(clone_dir, exist_ok=True)
    return {
        "root": root,
        "ref_dir": ref_dir,
        "clone_dir": clone_dir,
        "ref_wav": os.path.join(ref_dir, "ref.wav"),
        "ref_json": os.path.join(ref_dir, "ref.json"),
        "clone_wav": os.path.join(clone_dir, "clone.wav"),
        "clone_json": os.path.join(clone_dir, "clone.json"),
        "speaker_json": os.path.join(root, "speaker.json"),
    }


def materialize_ref(
    ref: dict[str, Any],
    paths: dict[str, str],
    spk: dict[str, Any],
    seed: int,
    audio_root: str = DEFAULT_AUDIO_ROOT,
) -> tuple[str, str, dict[str, Any]]:
    """Materialize reference into ref/ref.wav + detailed ref/ref.json (~8–10s)."""
    import shutil

    source_files = list(ref.get("source_files") or ([] if not ref.get("path") else [ref["path"]]))
    if not source_files:
        raise ValueError("ref has no source_files")
    bad = [fp for fp in source_files if not under_audio_root(fp, audio_root)]
    if bad:
        raise ValueError(f"ref sources outside audio_root: {bad[:2]}")
    gap = float(ref.get("gap_sec") or 0.3)
    ref_type = ref.get("type") or ("single" if len(source_files) == 1 else "concat")
    utt_ids = list(ref.get("utterance_ids") or [None] * len(source_files))

    per_source = []
    for i, fp in enumerate(source_files):
        t = read_ref_transcript(fp)
        dur = None
        if ref.get("source_durations") and i < len(ref["source_durations"]):
            dur = float(ref["source_durations"][i])
        else:
            try:
                audio, sr = load_audio_mono(fp, target_sr=16000)
                dur = round(len(audio) / sr, 3)
            except Exception:
                dur = None
        per_source.append(
            {
                "index": i,
                "path": fp,
                "exists": os.path.isfile(fp),
                "bytes": os.path.getsize(fp) if os.path.isfile(fp) else 0,
                "duration_sec": dur,
                "transcript": t,
                "utterance_id": utt_ids[i] if i < len(utt_ids) else None,
            }
        )

    transcript = (ref.get("transcript") or "").strip()
    if not transcript:
        transcript = " ".join(p["transcript"] for p in per_source if p.get("transcript")).strip()
    if not transcript:
        raise ValueError(f"empty ref transcript for {spk.get('uid')}; sidecar missing text under {audio_root}")

    dst = paths["ref_wav"]
    sample_rate = 16000
    # Always rematerialize from source so we never keep a stale clone-dir ref.wav
    if ref_type == "single" and len(source_files) == 1:
        src = source_files[0]
        try:
            shutil.copy2(src, dst)
        except OSError:
            audio, sample_rate = load_audio_mono(src, target_sr=16000)
            write_wav(dst, audio, sample_rate)
        dur = wav_file_duration(dst) or float(ref.get("duration") or 0.0)
    else:
        dur = concat_ref_files(source_files, dst, gap_sec=gap, sample_rate=sample_rate)

    meta = {
        "uid": spk.get("uid") or f"{spk.get('dataset')}/{spk.get('speaker_id')}",
        "dataset": spk.get("dataset"),
        "speaker_id": spk.get("speaker_id"),
        "gender_consensus": spk.get("gender_consensus"),
        "gender_confidence": spk.get("gender_confidence"),
        "age_mean": spk.get("age_mean"),
        "age_median": spk.get("age_median"),
        "speaker_total_duration_sec": spk.get("total_duration_sec"),
        "speaker_path": spk.get("speaker_path"),
        "audio_root": audio_root,
        "type": ref_type,
        "target_duration_sec": [8.0, 10.0],
        "gap_sec": gap if ref_type == "concat" else 0.0,
        "num_concat_clips": len(source_files),
        "source_files": source_files,
        "source_durations": ref.get("source_durations") or [p.get("duration_sec") for p in per_source],
        "sources": per_source,
        "transcript": transcript,
        "ref_audio_path": dst,
        "ref_audio_relpath": "ref/ref.wav",
        "duration_sec": round(float(dur), 3),
        "sample_rate": sample_rate,
        "bytes": os.path.getsize(dst) if os.path.isfile(dst) else 0,
        "seed": seed,
        "in_target_band": 8.0 <= float(dur) <= 10.0,
    }
    write_json(paths["ref_json"], meta)
    return dst, transcript, meta


def pick_ref(
    spk: dict[str, Any],
    paths: dict[str, str],
    seed: int,
    audio_root: str = DEFAULT_AUDIO_ROOT,
) -> tuple[str, str, dict[str, Any]]:
    refs = list(spk.get("ref_candidates") or [])
    # Prefer candidates that are under audio_root and already have transcript
    usable = []
    for r in refs:
        files = list(r.get("source_files") or ([] if not r.get("path") else [r["path"]]))
        if not files or any(not under_audio_root(fp, audio_root) for fp in files):
            continue
        tx = (r.get("transcript") or "").strip()
        if not tx:
            tx = " ".join(read_ref_transcript(fp) for fp in files).strip()
        if not tx:
            continue
        rr = dict(r)
        rr["transcript"] = tx
        usable.append(rr)
    if not usable:
        raise ValueError(f"No refs under audio_root with nonempty sidecar transcript for {spk.get('uid')}")
    rng = random.Random(seed)
    in_band = [r for r in usable if 8.0 <= float(r.get("duration") or 0.0) <= 10.0]
    ref = rng.choice(in_band or usable)
    return materialize_ref(ref, paths, spk, seed, audio_root=audio_root)


def _write_speaker_overview(
    paths: dict[str, str],
    spk: dict[str, Any],
    script: dict[str, Any],
    ref_meta: dict[str, Any],
    clone_meta: dict[str, Any],
) -> None:
    overview = {
        "uid": script.get("uid"),
        "dataset": script.get("dataset"),
        "speaker_id": script.get("speaker_id"),
        "gender_consensus": script.get("gender_consensus") or spk.get("gender_consensus"),
        "audience": script.get("audience"),
        "lang": script.get("lang"),
        "layout": {
            "root": paths["root"],
            "ref_dir": "ref/",
            "clone_dir": "clone/",
            "ref_wav": "ref/ref.wav",
            "ref_json": "ref/ref.json",
            "clone_wav": "clone/clone.wav",
            "clone_json": "clone/clone.json",
        },
        "speaker": {
            "total_duration_sec": spk.get("total_duration_sec"),
            "total_utterances": spk.get("total_utterances") or spk.get("num_utterances_listed"),
            "gender_confidence": spk.get("gender_confidence"),
            "age_mean": spk.get("age_mean"),
            "age_median": spk.get("age_median"),
            "speaker_path": spk.get("speaker_path"),
            "num_ref_candidates": len(spk.get("ref_candidates") or []),
        },
        "ref": {
            "type": ref_meta.get("type"),
            "duration_sec": ref_meta.get("duration_sec"),
            "num_concat_clips": ref_meta.get("num_concat_clips"),
            "in_target_band": ref_meta.get("in_target_band"),
            "transcript": ref_meta.get("transcript"),
            "path": "ref/ref.wav",
            "json": "ref/ref.json",
        },
        "clone": {
            "mode": clone_meta.get("mode"),
            "lang": clone_meta.get("lang"),
            "corpus_id": clone_meta.get("corpus_id"),
            "duration_sec": clone_meta.get("duration_sec"),
            "est_total_sec": clone_meta.get("est_total_sec"),
            "emotion": clone_meta.get("emotion"),
            "path": "clone/clone.wav",
            "json": "clone/clone.json",
            "text": clone_meta.get("text") or clone_meta.get("text_tagged"),
        },
    }
    write_json(paths["speaker_json"], overview)


def apply_speech_rate(
    audio: np.ndarray,
    sr: int,
    rate: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Time-stretch without intended pitch shift (librosa phase vocoder)."""
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    meta: dict[str, Any] = {"enabled": False, "rate": float(rate)}
    if rate <= 0 or abs(rate - 1.0) < 1e-3:
        return x, meta
    try:
        import librosa

        y = librosa.effects.time_stretch(x, rate=float(rate)).astype(np.float32)
        peak = float(np.max(np.abs(y))) if y.size else 0.0
        if peak > 0.99:
            y = y * (0.99 / peak)
        meta["enabled"] = True
        meta["ok"] = True
        meta["in_sec"] = round(len(x) / sr, 3)
        meta["out_sec"] = round(len(y) / sr, 3)
        return y, meta
    except Exception as e:
        meta["ok"] = False
        meta["error"] = str(e)
        return x, meta


def apply_random_reverb(
    audio: np.ndarray,
    sr: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply a mild random room impulse via pyroomacoustics."""
    rng = random.Random(seed)
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    # Random small-to-medium room; keep wet mix moderate for speech clarity.
    room_dim = [
        round(rng.uniform(3.5, 8.0), 2),
        round(rng.uniform(3.0, 6.5), 2),
        round(rng.uniform(2.4, 3.6), 2),
    ]
    absorption = round(rng.uniform(0.15, 0.45), 3)
    max_order = int(rng.choice([8, 10, 12, 14]))
    mic_loc = [
        round(rng.uniform(0.5, room_dim[0] - 0.5), 2),
        round(rng.uniform(0.5, room_dim[1] - 0.5), 2),
        round(rng.uniform(1.1, 1.8), 2),
    ]
    src_loc = [
        round(min(room_dim[0] - 0.5, max(0.5, mic_loc[0] + rng.uniform(-1.2, 1.2))), 2),
        round(min(room_dim[1] - 0.5, max(0.5, mic_loc[1] + rng.uniform(-1.2, 1.2))), 2),
        round(rng.uniform(1.2, 1.9), 2),
    ]
    wet = round(rng.uniform(0.18, 0.42), 3)
    meta: dict[str, Any] = {
        "enabled": True,
        "backend": "pyroomacoustics",
        "room_dim_m": room_dim,
        "absorption": absorption,
        "max_order": max_order,
        "mic_loc_m": mic_loc,
        "src_loc_m": src_loc,
        "wet": wet,
        "seed": seed,
    }
    try:
        import pyroomacoustics as pra

        room = pra.ShoeBox(
            room_dim,
            fs=sr,
            absorption=absorption,
            max_order=max_order,
        )
        room.add_source(src_loc, signal=x)
        room.add_microphone_array(np.array(mic_loc).reshape(3, 1))
        room.simulate()
        y = np.asarray(room.mic_array.signals[0], dtype=np.float32)
        # Match length / mix dry+wet
        n = len(x)
        if len(y) < n:
            y = np.pad(y, (0, n - len(y)))
        elif len(y) > n:
            y = y[:n]
        out = (1.0 - wet) * x + wet * y
        peak = float(np.max(np.abs(out))) if out.size else 0.0
        if peak > 0.99:
            out = out * (0.99 / peak)
        meta["ok"] = True
        return out.astype(np.float32), meta
    except Exception as e:
        meta["ok"] = False
        meta["error"] = str(e)
        return x, meta


def clone_one_concat(
    client: TTSClient,
    spk: dict[str, Any],
    script: dict[str, Any],
    out_dir: str,
    seed: int,
    temperature: float,
    skip_existing: bool,
    audio_root: str = DEFAULT_AUDIO_ROOT,
    apply_reverb: bool = True,
    speech_rate: float = 1.12,
) -> dict[str, Any]:
    uid = script["uid"]
    ds = script["dataset"]
    sid = script["speaker_id"]
    paths = speaker_dirs(out_dir, ds, sid)
    out_wav, out_json = paths["clone_wav"], paths["clone_json"]

    if (
        skip_existing
        and os.path.isfile(out_wav)
        and os.path.getsize(out_wav) > 1000
        and os.path.isfile(out_json)
        and os.path.isfile(paths["ref_wav"])
        and os.path.isfile(paths["ref_json"])
    ):
        try:
            with open(paths["ref_json"], encoding="utf-8") as f:
                old_ref = json.load(f)
            if (old_ref.get("transcript") or "").strip() and under_audio_root(
                (old_ref.get("source_files") or [""])[0], audio_root
            ):
                try:
                    with open(out_json, encoding="utf-8") as f:
                        old_clone = json.load(f)
                    old_rate = float((old_clone.get("speech_rate") or {}).get("rate") or 0.0)
                    if (
                        old_clone.get("pause_postprocess") == "sentence_concat"
                        and isinstance(old_clone.get("pause_secs"), list)
                        and abs(old_rate - float(speech_rate)) < 1e-3
                        and (not apply_reverb or (old_clone.get("reverb") or {}).get("ok"))
                        and "<|prosody:speed_slow|>" not in (old_clone.get("text_tagged") or "")
                        and "<|prosody:speed_slow|>" not in " ".join(old_clone.get("sentences") or [])
                        and sentences_have_consistent_emotion(
                            old_clone.get("sentences") or [],
                            old_clone.get("emotion") or script.get("emotion"),
                        )
                    ):
                        return {
                            "uid": uid,
                            "status": "skipped",
                            "out_wav": out_wav,
                            "ref_wav": paths["ref_wav"],
                        }
                except Exception:
                    pass
                # fall through to regenerate when metadata is stale
        except Exception:
            pass

    ref_path, ref_text, ref_meta = pick_ref(spk, paths, seed, audio_root=audio_root)
    sentences = script.get("sentences") or script.get("clean_sentences") or []
    if not sentences and (script.get("text") or script.get("text_tagged")):
        sentences = [script.get("text") or script.get("text_tagged")]
    sentences = ensure_sentence_emotion_tags(
        list(sentences),
        script.get("emotion"),
        script.get("prosody"),
        script.get("style"),
    )
    pause_secs = script.get("pause_secs")
    if not isinstance(pause_secs, list) or len(pause_secs) != max(0, len(sentences) - 1):
        # Fallback: design on the fly with same constraints.
        pause_secs = []
        prng = random.Random(seed + 4242 + stable_int(uid))
        n_gaps = max(0, len(sentences) - 1)
        for i in range(n_gaps):
            pause_secs.append(round(prng.uniform(1.0, 3.5), 2))
        while n_gaps and sum(pause_secs) / n_gaps < 2.0:
            j = min(range(n_gaps), key=lambda k: pause_secs[k])
            pause_secs[j] = round(min(4.5, pause_secs[j] + 0.25), 2)
    pause_secs = [max(1.0, float(p)) for p in pause_secs]
    clean_sents = script.get("clean_sentences") or []

    pieces: list[np.ndarray] = []
    sr = None
    sent_meta = []
    t0 = time.time()
    rate_meta: dict[str, Any] = {"enabled": abs(float(speech_rate) - 1.0) >= 1e-3, "rate": float(speech_rate)}
    for i, sent in enumerate(sentences):
        audio_bytes = client.generate(sent, ref_path, ref_text=ref_text, temperature=temperature)
        if not audio_bytes:
            return {"uid": uid, "status": "fail", "reason": f"tts_failed_sentence_{i}"}
        audio, this_sr = wav_bytes_to_float_mono(audio_bytes)
        if sr is None:
            sr = this_sr
        elif this_sr != sr:
            x_old = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
            n_new = int(round(len(audio) * sr / this_sr))
            x_new = np.linspace(0.0, 1.0, num=n_new, endpoint=False)
            audio = np.interp(x_new, x_old, audio).astype(np.float32)
        # Speed up speech only; keep designed pause lengths absolute.
        if abs(float(speech_rate) - 1.0) >= 1e-3:
            audio, sm = apply_speech_rate(audio, sr, float(speech_rate))
            if not sm.get("ok", True):
                rate_meta["ok"] = False
                rate_meta["error"] = sm.get("error")
            else:
                rate_meta["ok"] = True
        pieces.append(audio)
        sent_meta.append(
            {
                "index": i,
                "text": sent,
                "clean_text": clean_sents[i] if i < len(clean_sents) else None,
                "duration_sec": round(len(audio) / sr, 3),
            }
        )
        if i < len(sentences) - 1:
            pieces.append(np.zeros(int(round(pause_secs[i] * sr)), dtype=np.float32))

    assert sr is not None
    merged = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)

    # Optional random room reverb (pyroomacoustics).
    reverb_meta: dict[str, Any] = {"enabled": False}
    if apply_reverb:
        merged, reverb_meta = apply_random_reverb(merged, sr, seed=seed + 7771 + stable_int(uid))

    write_wav(out_wav, merged, sr)
    elapsed = round(time.time() - t0, 2)
    speech_sec = round(sum(float(s["duration_sec"]) for s in sent_meta), 3)
    meta = {
        "uid": uid,
        "dataset": ds,
        "speaker_id": sid,
        "gender_consensus": script.get("gender_consensus"),
        "audience": script.get("audience"),
        "lang": script.get("lang"),
        "corpus_id": script.get("corpus_id"),
        "emotion": script.get("emotion"),
        "prosody": script.get("prosody"),
        "mode": "concat",
        "temperature": temperature,
        "seed": seed,
        "pause_secs": pause_secs,
        "pause_sec_min": round(min(pause_secs), 3) if pause_secs else 0.0,
        "pause_sec_max": round(max(pause_secs), 3) if pause_secs else 0.0,
        "pause_sec_mean": round(sum(pause_secs) / len(pause_secs), 3) if pause_secs else 0.0,
        "pause_postprocess": "sentence_concat",
        "speech_rate": rate_meta,
        "num_sentences": len(sentences),
        "sentences": sentences,
        "clean_sentences": clean_sents,
        "sentence_meta": sent_meta,
        "text_tagged": script.get("text") or script.get("text_tagged"),
        "clean_text": script.get("clean_text"),
        "est_speech_sec": script.get("est_speech_sec"),
        "est_total_sec": script.get("est_total_sec"),
        "speech_duration_sec": speech_sec,
        "duration_sec": round(len(merged) / sr, 3),
        "sample_rate": sr,
        "bytes": os.path.getsize(out_wav),
        "elapsed_sec": elapsed,
        "reverb": reverb_meta,
        "clone_audio_path": out_wav,
        "clone_audio_relpath": "clone/clone.wav",
        "ref_audio_path": ref_path,
        "ref_audio_relpath": "ref/ref.wav",
        "ref_text": ref_text,
        "ref_meta_path": "ref/ref.json",
        "ref_duration_sec": ref_meta.get("duration_sec"),
        "ref_type": ref_meta.get("type"),
    }
    write_json(out_json, meta)
    _write_speaker_overview(paths, spk, script, ref_meta, meta)
    return {
        "uid": uid,
        "status": "ok",
        "duration_sec": meta["duration_sec"],
        "out_wav": out_wav,
        "ref_wav": ref_path,
        "lang": meta.get("lang"),
    }


def clone_one_single(
    client: TTSClient,
    spk: dict[str, Any],
    script: dict[str, Any],
    out_dir: str,
    seed: int,
    temperature: float,
    skip_existing: bool,
    audio_root: str = DEFAULT_AUDIO_ROOT,
    apply_reverb: bool = True,
    speech_rate: float = 1.05,
) -> dict[str, Any]:
    """One-shot TTS of the full script, then VAD-split and insert designed pauses."""
    uid = script["uid"]
    ds = script["dataset"]
    sid = script["speaker_id"]
    paths = speaker_dirs(out_dir, ds, sid)
    out_wav, out_json = paths["clone_wav"], paths["clone_json"]

    if (
        skip_existing
        and os.path.isfile(out_wav)
        and os.path.getsize(out_wav) > 1000
        and os.path.isfile(out_json)
        and os.path.isfile(paths["ref_wav"])
        and os.path.isfile(paths["ref_json"])
    ):
        try:
            with open(paths["ref_json"], encoding="utf-8") as f:
                old_ref = json.load(f)
            if (
                (old_ref.get("transcript") or "").strip()
                and under_audio_root((old_ref.get("source_files") or [""])[0], audio_root)
                and os.path.isfile(out_json)
            ):
                try:
                    with open(out_json, encoding="utf-8") as f:
                        old_clone = json.load(f)
                    old_rate = float((old_clone.get("speech_rate") or {}).get("rate") or 0.0)
                    if (
                        old_clone.get("mode") == "single"
                        and old_clone.get("pause_postprocess") == "vad_splice"
                        and "<|prosody:long_pause|>" not in (old_clone.get("text") or "")
                        and isinstance(old_clone.get("pause_secs"), list)
                        and abs(old_rate - float(speech_rate)) < 1e-3
                        and (not apply_reverb or (old_clone.get("reverb") or {}).get("ok"))
                    ):
                        return {
                            "uid": uid,
                            "status": "skipped",
                            "out_wav": out_wav,
                            "ref_wav": paths["ref_wav"],
                        }
                except Exception:
                    pass
        except Exception:
            pass

    ref_path, ref_text, ref_meta = pick_ref(spk, paths, seed, audio_root=audio_root)
    text = script.get("text") or script.get("text_tagged") or (" ".join(script.get("sentences") or []))
    text = strip_pause_tags(text)
    if not text:
        return {"uid": uid, "status": "fail", "reason": "empty_text"}

    designed = script.get("pause_secs")
    if not isinstance(designed, list):
        designed = []
    designed = [max(1.0, float(p)) for p in designed]
    pause_sec_min = float(script.get("pause_sec_min") or (min(designed) if designed else 1.0))
    pause_sec_max = float(script.get("pause_sec_max") or (max(designed) if designed else 3.5))

    t0 = time.time()
    audio_bytes = client.generate(text, ref_path, ref_text=ref_text, temperature=temperature)
    if not audio_bytes:
        return {"uid": uid, "status": "fail", "reason": "tts_failed"}
    audio_raw, sr = wav_bytes_to_float_mono(audio_bytes)

    rate_meta: dict[str, Any] = {
        "enabled": abs(float(speech_rate) - 1.0) >= 1e-3,
        "rate": float(speech_rate),
    }
    if abs(float(speech_rate) - 1.0) >= 1e-3:
        audio_raw, sm = apply_speech_rate(audio_raw, sr, float(speech_rate))
        rate_meta.update(sm)

    splice_rng = random.Random(seed + 9173 + stable_int(uid))
    audio, vad_segs, pause_secs = vad_splice_with_pauses(
        audio_raw,
        sr,
        pause_sec_min=pause_sec_min,
        pause_sec_max=pause_sec_max,
        pause_secs_designed=designed,
        rng=splice_rng,
    )
    speech_sec = round(sum(float(s.get("dur_sec") or 0.0) for s in vad_segs), 3)
    est = float(script.get("est_speech_sec") or 0.0)
    if speech_sec < 8.0 or (est > 0 and speech_sec < 0.45 * est):
        # Silero sometimes under-segments soft speech; fall back to energy VAD.
        energy_spans = _energy_vad_segments(
            audio_raw, sr, frame_ms=20.0, thresh=0.005, min_speech_ms=80.0, merge_gap_ms=200.0
        )
        chunks: list[np.ndarray] = []
        meta2: list[dict[str, float]] = []
        for a, b in energy_spans:
            chunk = audio_raw[a:b]
            if len(chunk) < int(0.05 * sr):
                continue
            chunks.append(chunk)
            meta2.append(
                {
                    "index": len(meta2),
                    "start_sec": round(a / sr, 3),
                    "end_sec": round(b / sr, 3),
                    "dur_sec": round((b - a) / sr, 3),
                    "note": "energy_vad_fallback",
                }
            )
        if chunks:
            pieces: list[np.ndarray] = []
            pause_secs = []
            for i, chunk in enumerate(chunks):
                pieces.append(chunk)
                if i < len(chunks) - 1:
                    if i < len(designed):
                        p = max(1.0, float(designed[i]))
                    else:
                        p = max(1.0, float(splice_rng.uniform(pause_sec_min, pause_sec_max)))
                    pause_secs.append(round(p, 3))
                    pieces.append(np.zeros(int(round(p * sr)), dtype=np.float32))
            audio = np.concatenate(pieces).astype(np.float32)
            vad_segs = meta2
            speech_sec = round(sum(float(s.get("dur_sec") or 0.0) for s in vad_segs), 3)

    if speech_sec < 8.0 or (est > 0 and speech_sec < 0.45 * est):
        return {
            "uid": uid,
            "status": "fail",
            "reason": f"too_little_speech speech={speech_sec} est={est} raw={round(len(audio_raw) / sr, 3)}",
            "out_wav": out_wav,
        }

    reverb_meta: dict[str, Any] = {"enabled": bool(apply_reverb)}
    if apply_reverb:
        audio, reverb_meta = apply_random_reverb(audio, sr, seed=seed + 5151)

    write_wav(out_wav, audio, sr)
    elapsed = round(time.time() - t0, 2)
    meta = {
        "uid": uid,
        "dataset": ds,
        "speaker_id": sid,
        "gender_consensus": script.get("gender_consensus"),
        "audience": script.get("audience"),
        "lang": script.get("lang"),
        "corpus_id": script.get("corpus_id"),
        "emotion": script.get("emotion"),
        "prosody": script.get("prosody"),
        "mode": "single",
        "pause_postprocess": "vad_splice",
        "temperature": temperature,
        "seed": seed,
        "pause_sec_min": round(pause_sec_min, 3),
        "pause_sec_max": round(pause_sec_max, 3),
        "pause_secs": pause_secs,
        "pause_secs_designed": designed,
        "pause_sec_mean": round(sum(pause_secs) / len(pause_secs), 3) if pause_secs else 0.0,
        "speech_rate": rate_meta,
        "num_sentences": script.get("num_sentences") or len(script.get("sentences") or []),
        "text": text,
        "text_tagged": text,
        "clean_text": script.get("clean_text"),
        "sentences": script.get("sentences"),
        "clean_sentences": script.get("clean_sentences"),
        "est_speech_sec": script.get("est_speech_sec"),
        "est_total_sec": script.get("est_total_sec"),
        "raw_duration_sec": round(len(audio_raw) / sr, 3),
        "speech_duration_sec": speech_sec,
        "vad_num_segments": len(vad_segs),
        "vad_segments": vad_segs,
        "duration_sec": round(len(audio) / sr, 3),
        "sample_rate": sr,
        "bytes": os.path.getsize(out_wav),
        "elapsed_sec": elapsed,
        "reverb": reverb_meta,
        "clone_audio_path": out_wav,
        "clone_audio_relpath": "clone/clone.wav",
        "ref_audio_path": ref_path,
        "ref_audio_relpath": "ref/ref.wav",
        "ref_text": ref_text,
        "ref_meta_path": "ref/ref.json",
        "ref_duration_sec": ref_meta.get("duration_sec"),
        "ref_type": ref_meta.get("type"),
    }
    write_json(out_json, meta)
    _write_speaker_overview(paths, spk, script, ref_meta, meta)
    return {
        "uid": uid,
        "status": "ok",
        "duration_sec": meta["duration_sec"],
        "out_wav": out_wav,
        "ref_wav": ref_path,
        "lang": meta.get("lang"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run multi-sentence pause cloning via SGLang")
    ap.add_argument("--speakers-json", required=True)
    ap.add_argument("--scripts-json", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--mode", choices=["concat", "single"], default="single")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="Only clone first N (debug)")
    ap.add_argument(
        "--audio-root",
        default=DEFAULT_AUDIO_ROOT,
        help="Only accept ref wavs under this …/audio tree (not audio_higgs_* clone dirs)",
    )
    ap.add_argument("--no-reverb", action="store_true", help="Disable pyroomacoustics reverb")
    ap.add_argument(
        "--speech-rate",
        type=float,
        default=1.05,
        help="Time-stretch factor for each sentence (>1 faster). Pauses stay absolute.",
    )
    args = ap.parse_args()
    apply_reverb = not bool(args.no_reverb)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(args.speakers_json, encoding="utf-8") as f:
        speakers = {(s.get("uid") or f"{s['dataset']}/{s['speaker_id']}"): s for s in json.load(f)["speakers"]}
    with open(args.scripts_json, encoding="utf-8") as f:
        scripts = json.load(f)["scripts"]
    if args.limit > 0:
        scripts = scripts[: args.limit]

    probe = TTSClient(args.base_url, max_new_tokens=args.max_new_tokens)
    if not probe.health():
        raise SystemExit(
            f"[03] TTS server not healthy at {args.base_url}/health. "
            "Start SGLang-Omni first (see v3_tts_clone/03_launch_servers.sh)."
        )

    clone_fn = clone_one_concat if args.mode == "concat" else clone_one_single
    if args.mode == "single":
        # Preload VAD once in main thread (avoid concurrent init segfaults).
        try:
            _get_vad_model()
            print("[03] Silero VAD loaded (locked for thread-safe inference)", flush=True)
        except Exception as e:
            print(f"[03] WARNING: Silero VAD unavailable ({e}); will use energy VAD", flush=True)
    results = []
    print(
        f"[03] Cloning {len(scripts)} speakers mode={args.mode} workers={args.workers} "
        f"reverb={apply_reverb} audio_root={args.audio_root} "
        f"(layout: {{dataset}}/{{speaker}}/ref|clone)",
        flush=True,
    )

    def _job(idx_script: tuple[int, dict[str, Any]]) -> dict[str, Any]:
        idx, script = idx_script
        uid = script["uid"]
        spk = speakers.get(uid)
        if not spk:
            return {"uid": uid, "status": "fail", "reason": "speaker_missing"}
        client = TTSClient(args.base_url, max_new_tokens=args.max_new_tokens)
        t0 = time.time()
        try:
            kwargs: dict[str, Any] = dict(
                client=client,
                spk=spk,
                script=script,
                out_dir=args.output_dir,
                seed=args.seed + idx * 97,
                temperature=args.temperature,
                skip_existing=args.skip_existing,
                audio_root=args.audio_root,
            )
            kwargs["apply_reverb"] = apply_reverb
            kwargs["speech_rate"] = float(args.speech_rate)
            res = clone_fn(**kwargs)
        except Exception as e:
            res = {"uid": uid, "status": "fail", "reason": str(e)}
        res["elapsed_sec"] = round(time.time() - t0, 2)
        return res

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = [ex.submit(_job, (i, s)) for i, s in enumerate(scripts)]
        for i, fut in enumerate(as_completed(futs), 1):
            res = fut.result()
            results.append(res)
            print(
                f"[{i}/{len(scripts)}] {res.get('uid')} {res.get('status')} "
                f"dur={res.get('duration_sec')} lang={res.get('lang')} "
                f"elapsed={res.get('elapsed_sec')}s",
                flush=True,
            )

    summary = {
        "mode": args.mode,
        "base_url": args.base_url,
        "output_dir": args.output_dir,
        "layout": "{output_dir}/{dataset}/{speaker_id}/{ref|clone}/",
        "files_per_speaker": [
            "ref/ref.wav",
            "ref/ref.json",
            "clone/clone.wav",
            "clone/clone.json",
            "speaker.json",
        ],
        "total": len(results),
        "ok": sum(1 for r in results if r.get("status") == "ok"),
        "skipped": sum(1 for r in results if r.get("status") == "skipped"),
        "fail": sum(1 for r in results if r.get("status") == "fail"),
        "results": sorted(results, key=lambda r: r.get("uid") or ""),
    }
    summary_path = os.path.join(args.output_dir, "clone_summary.json")
    write_json(summary_path, summary)
    print(
        f"[03] Done ok={summary['ok']} skipped={summary['skipped']} fail={summary['fail']} -> {summary_path}",
        flush=True,
    )
    if summary["fail"] and summary["ok"] == 0:
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
