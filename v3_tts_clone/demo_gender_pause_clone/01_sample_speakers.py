#!/usr/bin/env python3
# Copyright (c) 2025 Boson AI
#
# Sample speakers with total_duration >= 5 minutes:
# 20 male / 20 female / 5 child (from wav2vec2 gender consensus),
# then attach reference utterance candidates from train.json.
"""
Usage:
  python 01_sample_speakers.py \
    --train-json /path/to/train.json \
    --gender-json /path/to/per_speaker.json \
    --output-dir ./workdir \
    --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
import wave
from collections import defaultdict
from typing import Any

from stable_seed import stable_int


SPEAKER_HEAD_RE = re.compile(
    rb'\{\s*"speaker_id"\s*:\s*"(?P<sid>[^"]+)"\s*,\s*'
    rb'"dataset_name"\s*:\s*"(?P<ds>[^"]+)"\s*,\s*'
    rb'"total_utterances"\s*:\s*(?P<nu>\d+)\s*,\s*'
    rb'"total_duration"\s*:\s*(?P<td>[0-9.]+)'
)

# Source refs MUST come from this tree only (not audio_higgs_* / omnivoice clone dirs).
DEFAULT_AUDIO_ROOT = (
    "/root/group-shared/voiceprint/data/speech/speaker_diarization/"
    "merged_datasets_20250610_vad_segments_mtfaa_enhanced_extend_kid_withclone_addlibrilight_1130/audio"
)


def under_audio_root(path: str, audio_root: str) -> bool:
    """True iff path is under audio_root/ (avoids matching audio_higgs_* prefixes)."""
    if not path or not audio_root:
        return False
    root = os.path.abspath(audio_root)
    ap = os.path.abspath(path)
    return ap == root or ap.startswith(root + os.sep)


def speaker_audio_dir(audio_root: str, dataset: str, speaker_id: str) -> str:
    return os.path.join(audio_root, dataset, speaker_id)


def wav_duration_sec(path: str) -> float:
    try:
        with wave.open(path, "rb") as wf:
            rate = float(wf.getframerate() or 0)
            if rate <= 0:
                return 0.0
            return wf.getnframes() / rate
    except Exception:
        try:
            import soundfile as sf

            return float(sf.info(path).duration)
        except Exception:
            return 0.0


def load_gender_map(path: str) -> dict[str, dict[str, Any]]:
    """key = dataset/speaker_id -> summary fields."""
    print(f"[01] Loading gender map: {path}", flush=True)
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    out: dict[str, dict[str, Any]] = {}
    for key, val in raw.items():
        summary = (val or {}).get("summary") or {}
        consensus = summary.get("gender_consensus")
        if not consensus:
            continue
        age_stats = summary.get("age_stats") or {}
        out[key] = {
            "gender_consensus": consensus,
            "gender_confidence": float(summary.get("gender_confidence") or 0.0),
            "age_mean": age_stats.get("mean"),
            "age_median": age_stats.get("median"),
        }
    print(f"[01] Gender speakers: {len(out)}", flush=True)
    return out


def scan_long_speakers(train_json: str, min_duration_sec: float) -> list[dict[str, Any]]:
    """Fast regex scan: only speaker header fields (skip utterance bodies)."""
    print(
        f"[01] Scanning speakers with duration >= {min_duration_sec:.0f}s: {train_json}",
        flush=True,
    )
    t0 = time.time()
    rows: list[dict[str, Any]] = []
    with open(train_json, "rb") as f:
        leftover = b""
        while True:
            chunk = f.read(8 * 1024 * 1024)
            if not chunk:
                break
            data = leftover + chunk
            for m in SPEAKER_HEAD_RE.finditer(data):
                td = float(m.group("td"))
                if td < min_duration_sec:
                    continue
                rows.append(
                    {
                        "speaker_id": m.group("sid").decode("utf-8"),
                        "dataset": m.group("ds").decode("utf-8"),
                        "total_utterances": int(m.group("nu")),
                        "total_duration_sec": td,
                    }
                )
            leftover = data[-2048:]
    print(f"[01] Candidates: {len(rows)} ({time.time() - t0:.1f}s)", flush=True)
    return rows


def sample_by_gender(
    rows: list[dict[str, Any]],
    gender_map: dict[str, dict[str, Any]],
    n_male: int,
    n_female: int,
    n_child: int,
    seed: int,
    min_gender_confidence: float,
) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    missing = 0
    for row in rows:
        key = f"{row['dataset']}/{row['speaker_id']}"
        ginfo = gender_map.get(key)
        if not ginfo:
            missing += 1
            continue
        if ginfo["gender_confidence"] < min_gender_confidence:
            continue
        item = dict(row)
        item.update(ginfo)
        item["uid"] = key
        buckets[ginfo["gender_consensus"]].append(item)

    print(
        f"[01] Bucket sizes (conf>={min_gender_confidence}): "
        f"male={len(buckets['male'])} female={len(buckets['female'])} "
        f"child={len(buckets['child'])} missing_gender={missing}",
        flush=True,
    )

    rng = random.Random(seed)
    need = {"male": n_male, "female": n_female, "child": n_child}
    selected: list[dict[str, Any]] = []
    for label, n in need.items():
        pool = buckets[label]
        if len(pool) < n:
            raise SystemExit(f"[01] Not enough '{label}' speakers: have {len(pool)}, need {n}")
        chosen = rng.sample(pool, n)
        chosen.sort(key=lambda x: x["uid"])
        selected.extend(chosen)
        print(f"[01] Sampled {n} {label}", flush=True)

    rng.shuffle(selected)
    return selected


def _read_transcript(wav_path: str) -> str:
    sidecar = wav_path + ".json"
    if not os.path.isfile(sidecar):
        return ""
    try:
        with open(sidecar, encoding="utf-8") as f:
            obj = json.load(f)
        return (obj.get("transcript") or obj.get("text") or "").strip()
    except Exception:
        return ""


def list_source_clips(
    speaker_dir: str,
    audio_root: str,
    *,
    require_transcript: bool = True,
) -> list[dict[str, Any]]:
    """List wav clips under speaker_dir that live in audio_root and have sidecar text."""
    if not speaker_dir or not os.path.isdir(speaker_dir):
        return []
    if not under_audio_root(speaker_dir, audio_root):
        return []
    out: list[dict[str, Any]] = []
    try:
        names = os.listdir(speaker_dir)
    except OSError:
        return []
    for name in names:
        if not name.endswith(".wav"):
            continue
        path = os.path.join(speaker_dir, name)
        if not under_audio_root(path, audio_root):
            continue
        if not os.path.isfile(path) or os.path.getsize(path) <= 1000:
            continue
        transcript = _read_transcript(path)
        if require_transcript and not transcript:
            continue
        dur = wav_duration_sec(path)
        if dur < 0.4:
            continue
        out.append(
            {
                "path": path,
                "duration": dur,
                "utterance_id": os.path.splitext(name)[0],
                "transcript": transcript,
            }
        )
    return out


def _combo_duration(durs: list[float], gap_sec: float = 0.3) -> float:
    if not durs:
        return 0.0
    return sum(durs) + gap_sec * max(0, len(durs) - 1)


def _clips_to_ref(clips: list[dict[str, Any]], gap_sec: float = 0.3) -> dict[str, Any]:
    """clips: [{path, duration, utterance_id, transcript}]"""
    total = _combo_duration([c["duration"] for c in clips], gap_sec)
    ref_type = "single" if len(clips) == 1 else "concat"
    return {
        "type": ref_type,
        "path": clips[0]["path"] if ref_type == "single" else None,
        "source_files": [c["path"] for c in clips],
        "source_durations": [round(c["duration"], 3) for c in clips],
        "duration": round(total, 3),
        "gap_sec": gap_sec,
        "num_concat_clips": len(clips),
        "transcript": " ".join(c["transcript"] for c in clips if c.get("transcript")).strip(),
        "utterance_ids": [c.get("utterance_id") for c in clips],
    }


def _greedy_pack_to_target(
    clips: list[dict[str, Any]],
    rng: random.Random,
    min_sec: float,
    max_sec: float,
    max_concat: int,
    gap_sec: float = 0.3,
) -> list[dict[str, Any]] | None:
    """Pack short clips toward ~center of [min_sec, max_sec]."""
    if not clips:
        return None
    target = (min_sec + max_sec) / 2.0
    ordered = list(clips)
    rng.shuffle(ordered)
    # Prefer starting from medium-length pieces
    ordered.sort(key=lambda c: abs(c["duration"] - 3.0))
    best: list[dict[str, Any]] | None = None
    best_err = 1e9
    for start in range(min(12, len(ordered))):
        pack: list[dict[str, Any]] = []
        for c in ordered[start:] + ordered[:start]:
            if len(pack) >= max_concat:
                break
            trial = pack + [c]
            tot = _combo_duration([x["duration"] for x in trial], gap_sec)
            if tot > max_sec and pack:
                break
            pack = trial
            if min_sec <= tot <= max_sec:
                err = abs(tot - target)
                if err < best_err:
                    best_err = err
                    best = list(pack)
                break
        if pack and best is None:
            tot = _combo_duration([x["duration"] for x in pack], gap_sec)
            err = abs(tot - target)
            if err < best_err:
                best_err = err
                best = list(pack)
    return best


def _pick_refs(
    utterances: list[dict[str, Any]],
    rng: random.Random,
    max_refs: int,
    ref_min_sec: float = 8.0,
    ref_max_sec: float = 10.0,
    max_concat: int = 5,
    gap_sec: float = 0.3,
    audio_root: str | None = None,
) -> list[dict[str, Any]]:
    """Build reference candidates with total duration ≈ 8–10s (concat if needed).

    Only keeps clips with non-empty transcript, optionally restricted to audio_root/.
    """
    valid: list[dict[str, Any]] = []
    for u in utterances:
        path = u.get("path") or ""
        dur = float(u.get("duration") or 0.0)
        transcript = (u.get("transcript") or "").strip() or _read_transcript(path)
        if audio_root and not under_audio_root(path, audio_root):
            continue
        if not path or not os.path.isfile(path):
            continue
        if os.path.getsize(path) <= 1000 or dur < 0.4:
            continue
        if not transcript:
            continue
        valid.append(
            {
                "path": path,
                "duration": dur,
                "utterance_id": u.get("utterance_id"),
                "transcript": transcript,
            }
        )
    if not valid:
        return []

    pool: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()

    def _add(clips: list[dict[str, Any]]) -> None:
        key = tuple(c["path"] for c in clips)
        if key in seen:
            return
        # every clip must have transcript
        if any(not (c.get("transcript") or "").strip() for c in clips):
            return
        tot = _combo_duration([c["duration"] for c in clips], gap_sec)
        if not (ref_min_sec <= tot <= ref_max_sec):
            return
        seen.add(key)
        pool.append(_clips_to_ref(clips, gap_sec))

    # 1) singles already in range
    singles = [v for v in valid if ref_min_sec <= v["duration"] <= ref_max_sec]
    rng.shuffle(singles)
    for c in singles:
        _add([c])
        if len(pool) >= max_refs:
            return pool[:max_refs]

    # 2) random concat of shorter clips
    short = [v for v in valid if v["duration"] < ref_max_sec]
    attempts = 0
    max_attempts = max(600, max_refs * 40)
    while len(pool) < max_refs and attempts < max_attempts and len(short) >= 2:
        attempts += 1
        n = rng.randint(2, min(max_concat, len(short)))
        combo = rng.sample(short, n)
        _add(combo)

    # 3) greedy pack fallback (may slightly miss band; still closest)
    if not pool:
        packed = _greedy_pack_to_target(short or valid, rng, ref_min_sec, ref_max_sec, max_concat, gap_sec)
        if packed and all((c.get("transcript") or "").strip() for c in packed):
            tot = _combo_duration([c["duration"] for c in packed], gap_sec)
            if tot < ref_min_sec and len(packed) < max_concat:
                rest = [c for c in (short or valid) if c["path"] not in {x["path"] for x in packed}]
                rng.shuffle(rest)
                for c in rest:
                    if len(packed) >= max_concat:
                        break
                    trial = packed + [c]
                    t2 = _combo_duration([x["duration"] for x in trial], gap_sec)
                    if t2 > ref_max_sec + 1.5:
                        continue
                    packed = trial
                    if t2 >= ref_min_sec:
                        break
            tot = _combo_duration([c["duration"] for c in packed], gap_sec)
            if ref_min_sec <= tot <= ref_max_sec + 1.5:
                pool.append(_clips_to_ref(packed, gap_sec))

    # Do NOT fall back to no-transcript / wrong-root clips.
    rng.shuffle(pool)
    return pool[:max_refs]


def attach_refs_from_audio_root(
    selected: list[dict[str, Any]],
    audio_root: str,
    seed: int,
    max_refs: int,
    ref_min_sec: float = 8.0,
    ref_max_sec: float = 10.0,
    max_concat: int = 5,
) -> list[dict[str, Any]]:
    """Attach refs by scanning {audio_root}/{dataset}/{speaker_id}/*.wav + *.wav.json sidecars."""
    print(
        f"[01] Building refs from AUDIO_ROOT={audio_root} "
        f"(require sidecar transcript, target {ref_min_sec}-{ref_max_sec}s)...",
        flush=True,
    )
    t0 = time.time()
    out: list[dict[str, Any]] = []
    missing: list[str] = []
    for i, s in enumerate(selected):
        ds = s["dataset"]
        sid = s["speaker_id"]
        sp_dir = speaker_audio_dir(audio_root, ds, sid)
        clips = list_source_clips(sp_dir, audio_root, require_transcript=True)
        rng = random.Random(seed + stable_int(ds, sid))
        refs = _pick_refs(
            clips,
            rng,
            max_refs,
            ref_min_sec=ref_min_sec,
            ref_max_sec=ref_max_sec,
            max_concat=max_concat,
            audio_root=audio_root,
        )
        s = dict(s)
        s["speaker_path"] = sp_dir
        s["ref_candidates"] = refs
        s["num_utterances_listed"] = len(clips)
        s["audio_root"] = audio_root
        if not refs:
            missing.append(f"{ds}/{sid}")
        out.append(s)
        if (i + 1) % 10 == 0:
            print(f"[01]   scanned {i + 1}/{len(selected)}", flush=True)
    if missing:
        print(f"[01] WARNING: no usable ref (audio/+transcript) for {len(missing)}:", flush=True)
        for k in missing[:15]:
            print(f"       - {k}", flush=True)
    print(f"[01] Ref scan done ({time.time() - t0:.1f}s)", flush=True)
    return out


def sample_with_valid_refs(
    rows: list[dict[str, Any]],
    gender_map: dict[str, dict[str, Any]],
    *,
    audio_root: str,
    n_male: int,
    n_female: int,
    n_child: int,
    seed: int,
    min_gender_confidence: float,
    max_refs: int,
    ref_min_sec: float,
    ref_max_sec: float,
    max_concat: int,
) -> list[dict[str, Any]]:
    """Sample speakers and keep only those with audio/-only refs + nonempty transcripts."""
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        key = f"{r['dataset']}/{r['speaker_id']}"
        g = gender_map.get(key)
        if not g:
            continue
        if float(g.get("gender_confidence") or 0.0) < min_gender_confidence:
            continue
        cons = g["gender_consensus"]
        if cons not in ("male", "female", "child"):
            continue
        item = dict(r)
        item.update(g)
        item["uid"] = key
        buckets[cons].append(item)

    rng = random.Random(seed)
    for b in buckets.values():
        rng.shuffle(b)

    need = {"male": n_male, "female": n_female, "child": n_child}
    selected: list[dict[str, Any]] = []
    skipped = {"male": 0, "female": 0, "child": 0}

    for label, n_want in need.items():
        pool = buckets.get(label) or []
        got = 0
        for cand in pool:
            if got >= n_want:
                break
            sp_dir = speaker_audio_dir(audio_root, cand["dataset"], cand["speaker_id"])
            clips = list_source_clips(sp_dir, audio_root, require_transcript=True)
            if not clips:
                skipped[label] += 1
                continue
            local_rng = random.Random(seed + stable_int(cand["uid"]))
            refs = _pick_refs(
                clips,
                local_rng,
                max_refs,
                ref_min_sec=ref_min_sec,
                ref_max_sec=ref_max_sec,
                max_concat=max_concat,
                audio_root=audio_root,
            )
            if not refs:
                skipped[label] += 1
                continue
            item = dict(cand)
            item["speaker_path"] = sp_dir
            item["ref_candidates"] = refs
            item["num_utterances_listed"] = len(clips)
            item["audio_root"] = audio_root
            selected.append(item)
            got += 1
        print(
            f"[01] Sampled {got}/{n_want} {label} (skipped {skipped[label]} without audio/+transcript refs)",
            flush=True,
        )
        if got < n_want:
            print(f"[01] WARNING: only {got}/{n_want} {label} available", flush=True)

    rng.shuffle(selected)
    return selected


def attach_refs_ijson(
    train_json: str,
    selected: list[dict[str, Any]],
    seed: int,
    max_refs: int,
    ref_min_sec: float = 8.0,
    ref_max_sec: float = 10.0,
    max_concat: int = 5,
    audio_root: str = DEFAULT_AUDIO_ROOT,
) -> list[dict[str, Any]]:
    """Deprecated path: prefer attach_refs_from_audio_root. Kept for compatibility."""
    _ = train_json
    return attach_refs_from_audio_root(
        selected,
        audio_root=audio_root,
        seed=seed,
        max_refs=max_refs,
        ref_min_sec=ref_min_sec,
        ref_max_sec=ref_max_sec,
        max_concat=max_concat,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Sample male/female/child speakers for pause-clone demo")
    ap.add_argument(
        "--train-json",
        required=True,
        help="Merged train.json (speakers[].total_duration)",
    )
    ap.add_argument(
        "--gender-json",
        required=True,
        help="genders_ages_wav2vec2/per_speaker.json",
    )
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-duration-sec", type=float, default=300.0, help="Default 5 minutes")
    ap.add_argument("--n-male", type=int, default=20)
    ap.add_argument("--n-female", type=int, default=20)
    ap.add_argument("--n-child", type=int, default=5)
    ap.add_argument("--min-gender-confidence", type=float, default=0.6)
    ap.add_argument("--max-refs-per-speaker", type=int, default=8)
    ap.add_argument("--ref-min-sec", type=float, default=8.0, help="Reference audio min total duration")
    ap.add_argument("--ref-max-sec", type=float, default=10.0, help="Reference audio max total duration")
    ap.add_argument("--ref-max-concat", type=int, default=5, help="Max clips to concat into one ref")
    ap.add_argument(
        "--audio-root",
        default=DEFAULT_AUDIO_ROOT,
        help="Only use reference wavs under this directory (…/audio), never clone dirs",
    )
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    gender_map = load_gender_map(args.gender_json)
    rows = scan_long_speakers(args.train_json, args.min_duration_sec)
    selected = sample_with_valid_refs(
        rows,
        gender_map,
        audio_root=args.audio_root,
        n_male=args.n_male,
        n_female=args.n_female,
        n_child=args.n_child,
        seed=args.seed,
        min_gender_confidence=args.min_gender_confidence,
        max_refs=args.max_refs_per_speaker,
        ref_min_sec=args.ref_min_sec,
        ref_max_sec=args.ref_max_sec,
        max_concat=args.ref_max_concat,
    )

    usable = [s for s in selected if s.get("ref_candidates")]
    if not usable:
        raise SystemExit("[01] No usable speakers with reference audio + transcript under audio/")

    out_path = os.path.join(args.output_dir, "selected_speakers.json")
    # Ref duration stats
    ref_durs = []
    ref_types = {"single": 0, "concat": 0}
    empty_tx = 0
    for s in usable:
        for r in s.get("ref_candidates") or []:
            ref_durs.append(float(r.get("duration") or 0.0))
            ref_types[r.get("type") or "single"] = ref_types.get(r.get("type") or "single", 0) + 1
            if not (r.get("transcript") or "").strip():
                empty_tx += 1

    payload = {
        "seed": args.seed,
        "audio_root": args.audio_root,
        "min_duration_sec": args.min_duration_sec,
        "ref_target_sec": [args.ref_min_sec, args.ref_max_sec],
        "ref_max_concat": args.ref_max_concat,
        "require_ref_transcript": True,
        "counts": {
            "male": sum(1 for s in usable if s["gender_consensus"] == "male"),
            "female": sum(1 for s in usable if s["gender_consensus"] == "female"),
            "child": sum(1 for s in usable if s["gender_consensus"] == "child"),
            "total": len(usable),
        },
        "ref_stats": {
            "n_candidates": len(ref_durs),
            "single": ref_types.get("single", 0),
            "concat": ref_types.get("concat", 0),
            "empty_transcript": empty_tx,
            "dur_mean": round(sum(ref_durs) / len(ref_durs), 3) if ref_durs else None,
            "dur_min": round(min(ref_durs), 3) if ref_durs else None,
            "dur_max": round(max(ref_durs), 3) if ref_durs else None,
            "in_8_10": sum(1 for d in ref_durs if 8.0 <= d <= 10.0) if ref_durs else 0,
        },
        "speakers": usable,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[01] Wrote {out_path} ({payload['counts']}) ref_stats={payload['ref_stats']}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
