#!/usr/bin/env python3
# Copyright (c) 2025 Boson AI
"""Mix background crowd/human babble onto clones → fixed 30s @ 16 kHz outputs.

Requirements:
  - noise must be background noisy human voices (cafe / cafeteria / restaurant /
    meeting / public square / CHiME cafe&pedestrian), NOT traffic/nature/machines
  - noise clips must be >30s, then crop a 30s window
  - final mixed wav is exactly 30.0s at 16 kHz (mono)
  - SNR uniform in [5, 20] dB
  - every clone is mixed; dataset 20 male / 20 female / 5 child
"""

from __future__ import annotations

import argparse
import json
import os
import random
import wave
from pathlib import Path

import numpy as np

from stable_seed import stable_int


NOISE_ROOT = Path("/root/group-shared/voiceprint/data/noise")
TARGET_SEC = 30.0
OUTPUT_SR = 16000

# Only scenes that are predominantly human chatter / crowd ambience.
CATEGORY_DIRS: dict[str, list[Path]] = {
    "cafe": [
        NOISE_ROOT / "DEMAND-noise/SCAFE",
        NOISE_ROOT / "DEMAND-noise/PCAFETER",
        NOISE_ROOT / "chime4noise/backgrounds_wav",  # filtered to *CAF* below
        NOISE_ROOT / "freesound_downloaded_noises/freesound_download_audio_10to60s_filtered_20220819/Cafe,",
    ],
    "restaurant": [
        NOISE_ROOT / "DEMAND-noise/PRESTO",  # restaurant
        NOISE_ROOT / "freesound_downloaded_noises/freesound_download_audio_10to60s_filtered_20220819/Restaurant,",
    ],
    "meeting": [
        NOISE_ROOT / "DEMAND-noise/OMEETING",
        NOISE_ROOT / "DEMAND-noise/OOFFICE",
    ],
    "crowd": [
        NOISE_ROOT / "DEMAND-noise/SPSQUARE",  # public square people
        NOISE_ROOT / "chime4noise/backgrounds_wav",  # filtered to *PED* below
    ],
}

# Freesound filenames must match at least one of these (human ambient).
_HUMAN_ALLOW = (
    "cafe",
    "cafeteria",
    "restaurant",
    "ambience",
    "ambiance",
    "crowd",
    "people",
    "chatter",
    "party",
    "dinner",
    "dining",
    "pub",
    "bar",
    "busy",
    "voices",
    "talk",
    "conversation",
    "murmur",
    "population",
    "parisian",
    "joeys",
    "dimsum",
    "saladbar",
    "surfers",
)

# Hard exclude non-human / machine / nature false positives.
_HUMAN_DENY = (
    "frog",
    "brook",
    "creek",
    "stream",
    "thunder",
    "rain",
    "bird",
    "seagull",
    "wind",
    "ocean",
    "wave",
    "insect",
    "cricket",
    "sewer",
    "water",
    "jackhammer",
    "airplane",
    "baby",
    "vauva",
    "kitchen",
    "bell",
    "whoisit",
    "smelly",
    "fan",
    "lift",
    "tram",
    "car",
    "traffic",
    "extractor",
    "door",
    "chair",
    "fridge",
    "wc",
    "bathroom",
    "construction",
    "highway",
    "residential",
    "closed",
    "solo",
    "steamer",
    "spout",
    "garden",
    "autopista",
    "nightime",
    "metro",
    "subway",
    "bus",
    "station",
)


def read_wav_mono(path: Path) -> tuple[np.ndarray, int]:
    # Prefer soundfile (handles 24-bit / odd formats).
    try:
        import soundfile as sf

        x, sr = sf.read(str(path), always_2d=True, dtype="float32")
        x = x.mean(axis=1).astype(np.float32)
        return x, int(sr)
    except Exception:
        pass

    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        nch = wf.getnchannels()
        sw = wf.getsampwidth()
        nframes = wf.getnframes()
        frames = wf.readframes(nframes)
    if sw == 2:
        x = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 3:
        # 24-bit little-endian PCM
        a = np.frombuffer(frames, dtype=np.uint8).astype(np.int32)
        if len(a) < 3:
            raise ValueError(f"empty 24-bit wav: {path}")
        a = a[: (len(a) // 3) * 3].reshape(-1, 3)
        vals = a[:, 0] | (a[:, 1] << 8) | (a[:, 2] << 16)
        vals = np.where(vals >= 0x800000, vals - 0x1000000, vals)
        x = vals.astype(np.float32) / 8388608.0
    elif sw == 4:
        x = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"unsupported sampwidth {sw} for {path}")
    if nch > 1:
        x = x.reshape(-1, nch).mean(axis=1)
    return x.astype(np.float32), sr


def wav_duration_sec(path: Path) -> float | None:
    try:
        import soundfile as sf

        return float(sf.info(str(path)).duration)
    except Exception:
        pass
    try:
        with wave.open(str(path), "rb") as wf:
            return wf.getnframes() / float(wf.getframerate())
    except Exception:
        return None


def _accept_noise_file(path: Path, category: str) -> bool:
    """Keep only background noisy human-voice ambience."""
    name = path.name
    low = name.lower()
    parent = str(path.parent)

    # CHiME-4: cafe → CAF, crowd → PED only (drop BUS/STR vehicle/street).
    if "chime4noise" in parent.replace("\\", "/"):
        if category == "cafe" and "_CAF." in name:
            return True
        if category == "crowd" and "_PED." in name:
            return True
        return False

    # DEMAND dirs are curated human scenes; accept all channels.
    if "DEMAND-noise" in parent.replace("\\", "/"):
        return True

    # Freesound: require human-ambient allow keywords and deny non-human.
    if any(k in low for k in _HUMAN_DENY):
        return False
    return any(k in low for k in _HUMAN_ALLOW)


def write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    audio = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    pcm = (audio * 32767.0).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def resample_linear(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x.astype(np.float32)
    n_out = int(round(len(x) * sr_out / sr_in))
    if n_out <= 1 or len(x) <= 1:
        return np.zeros(max(1, n_out), dtype=np.float32)
    t_old = np.linspace(0.0, 1.0, num=len(x), endpoint=False)
    t_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(t_new, t_old, x).astype(np.float32)


def crop_exact(x: np.ndarray, n: int, rng: random.Random) -> np.ndarray:
    """Return exactly n samples; pad with zeros if short, random crop if long."""
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if len(x) == n:
        return x
    if len(x) > n:
        start = rng.randrange(0, len(x) - n + 1)
        return x[start : start + n]
    out = np.zeros(n, dtype=np.float32)
    out[: len(x)] = x
    return out


def crop_from_start(x: np.ndarray, n: int) -> np.ndarray:
    """Take first n samples; pad with zeros if shorter than n."""
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if len(x) == n:
        return x
    if len(x) > n:
        return x[:n]
    out = np.zeros(n, dtype=np.float32)
    out[: len(x)] = x
    return out


def read_wav_mono_window(path: Path, n_out: int, out_sr: int, rng: random.Random) -> np.ndarray:
    """Read a random contiguous window and resample to exactly n_out @ out_sr.

    Avoids loading multi-minute CHiME backgrounds in full.
    Source duration must already be verified > target_sec.
    """
    import soundfile as sf

    info = sf.info(str(path))
    sr_in = int(info.samplerate)
    n_in_total = int(info.frames)
    # Need enough input samples to cover n_out after resample.
    need_in = int(np.ceil(n_out * sr_in / out_sr)) + 8
    if n_in_total <= need_in:
        x, _ = read_wav_mono(path)
        x = resample_linear(x, sr_in, out_sr)
        return crop_exact(x, n_out, rng)
    start = rng.randrange(0, n_in_total - need_in + 1)
    x, _ = sf.read(str(path), start=start, stop=start + need_in, always_2d=True, dtype="float32")
    x = x.mean(axis=1).astype(np.float32)
    x = resample_linear(x, sr_in, out_sr)
    return crop_exact(x, n_out, rng)


def collect_noise_pool(min_sec: float) -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = {}
    for cat, dirs in CATEGORY_DIRS.items():
        files: list[Path] = []
        seen: set[str] = set()
        for d in dirs:
            if not d.is_dir():
                continue
            for f in d.rglob("*.wav"):
                if not _accept_noise_file(f, cat):
                    continue
                key = str(f.resolve())
                if key in seen:
                    continue
                du = wav_duration_sec(f)
                if du is not None and du > min_sec:
                    seen.add(key)
                    files.append(f)
        out[cat] = files
    return out


def mix_at_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    ps = float(np.mean(speech**2) + 1e-12)
    pn = float(np.mean(noise**2) + 1e-12)
    target_pn = ps / (10 ** (snr_db / 10.0))
    scale = (target_pn / pn) ** 0.5
    mixed = speech + noise * scale
    peak = float(np.max(np.abs(mixed))) if mixed.size else 0.0
    if peak > 0.99:
        mixed = mixed * (0.99 / peak)
    return mixed.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--clones-dir",
        action="append",
        default=None,
        help="Clone root containing */clone/clone.json (repeatable).",
    )
    ap.add_argument(
        "--workdir",
        default="",
        help="If set, mix ALL */clone/clone.json under this dir "
        "(e.g. demo_gender_pause_clone: clones + clones_bak_*).",
    )
    ap.add_argument("--speakers-json", default="")
    ap.add_argument("--target-sec", type=float, default=TARGET_SEC)
    ap.add_argument("--output-sr", type=int, default=OUTPUT_SR)
    ap.add_argument("--snr-min", type=float, default=5.0)
    ap.add_argument("--snr-max", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    assert args.snr_max >= args.snr_min
    target_sec = float(args.target_sec)
    out_sr = int(args.output_sr)
    if out_sr <= 0:
        raise SystemExit("--output-sr must be > 0")

    pool = collect_noise_pool(min_sec=target_sec)
    print("[05] noise pool (duration > {:.0f}s):".format(target_sec))
    for cat, files in pool.items():
        print(f"  {cat}: {len(files)}")
    cats = [c for c, fs in pool.items() if fs]
    if not cats:
        raise SystemExit("no noise files longer than target_sec")

    # Optional gender check
    if args.speakers_json and os.path.isfile(args.speakers_json):
        with open(args.speakers_json, encoding="utf-8") as f:
            speakers = json.load(f).get("speakers") or []
        from collections import Counter

        g = Counter(s.get("gender_consensus") for s in speakers)
        print(f"[05] speakers gender={dict(g)}")
        if g.get("male", 0) != 20 or g.get("female", 0) != 20 or g.get("child", 0) != 5:
            print("[05] WARNING: expected 20/20/5 male/female/child")

    roots: list[Path] = []
    if args.workdir:
        wd = Path(args.workdir)
        if not wd.is_dir():
            raise SystemExit(f"--workdir not a directory: {wd}")
        roots = [wd]
    elif args.clones_dir:
        roots = [Path(p) for p in args.clones_dir]
    else:
        roots = [Path("/root/code/github_repos/higgs-audio/clone_workdir/demo_gender_pause_clone/clones")]

    jsons: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        for jp in root.rglob("clone/clone.json"):
            # Skip ref-only trees / non-clone sidecars.
            if jp.name != "clone.json":
                continue
            key = str(jp.resolve())
            if key in seen:
                continue
            seen.add(key)
            jsons.append(jp)
    jsons = sorted(jsons)
    print(f"[05] found {len(jsons)} clone.json under {len(roots)} root(s)")

    ok = skip = fail = 0
    snrs: list[float] = []
    cats_used: list[str] = []

    for jp in jsons:
        wav = jp.with_suffix(".wav")
        if not wav.is_file():
            fail += 1
            continue
        meta = json.loads(jp.read_text(encoding="utf-8"))
        bg = meta.get("bg_noise") or {}
        if (
            bg.get("ok")
            and bg.get("noise_kind") == "human_crowd_babble"
            and bg.get("speech_crop_mode") == "head"
            and abs(float(bg.get("target_sec") or 0) - target_sec) < 1e-6
            and int(bg.get("sample_rate") or meta.get("sample_rate") or 0) == out_sr
            and not args.overwrite
        ):
            skip += 1
            continue

        uid = meta.get("uid") or f"{meta.get('dataset')}/{meta.get('speaker_id')}"
        # Disambiguate same uid across bak trees so noise crop differs.
        seed_key = f"{jp.parent.parent.parent.name}/{uid}"
        rng = random.Random(args.seed + stable_int(seed_key))
        cat = rng.choice(cats)
        noise_path = rng.choice(pool[cat])
        snr = round(rng.uniform(args.snr_min, args.snr_max), 2)

        try:
            dry_path = wav.with_name("clone_dry.wav")
            # Prefer never-mixed dry source (keep dry at original TTS sr).
            if dry_path.is_file():
                speech, sr = read_wav_mono(dry_path)
            else:
                speech, sr = read_wav_mono(wav)
                if not bg.get("ok"):
                    write_wav(dry_path, speech, sr)

            speech = resample_linear(speech, sr, out_sr)
            n_target = int(round(target_sec * out_sr))
            # Speech: always take the head 30s (pad if shorter).
            speech_30 = crop_from_start(speech, n_target)

            # Noise: random window from a >30s human-ambient clip.
            noise_30 = read_wav_mono_window(noise_path, n_target, out_sr, rng)

            mixed = mix_at_snr(speech_30, noise_30, snr)
            assert len(mixed) == n_target
            write_wav(wav, mixed, out_sr)

            meta["bg_noise"] = {
                "ok": True,
                "category": cat,
                "path": str(noise_path),
                "snr_db": snr,
                "snr_range_db": [args.snr_min, args.snr_max],
                "target_sec": target_sec,
                "speech_crop_sec": target_sec,
                "speech_crop_mode": "head",
                "sample_rate": out_sr,
                "noise_kind": "human_crowd_babble",
                "noise_src": "DEMAND_cafe_cafeteria_meeting_presto_square_CHiME_CAF_PED_Freesound_cafe_restaurant",
                "dry_wav": "clone/clone_dry.wav",
                "output_wav": "clone/clone.wav",
            }
            meta["duration_sec"] = round(len(mixed) / out_sr, 3)
            meta["bytes"] = os.path.getsize(wav)
            meta["sample_rate"] = out_sr
            jp.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            ok += 1
            snrs.append(snr)
            cats_used.append(cat)
            print(
                f"ok {seed_key} gender={meta.get('gender_consensus')} "
                f"cat={cat} snr={snr} dur={meta['duration_sec']:.1f}s "
                f"sr={out_sr} noise={noise_path.name}"
            )
        except Exception as e:
            fail += 1
            print(f"fail {seed_key}: {e}")

    print(f"[05] done ok={ok} skip={skip} fail={fail} total={len(jsons)}")
    if snrs:
        print(f"[05] snr min={min(snrs):.1f} max={max(snrs):.1f} mean={sum(snrs) / len(snrs):.1f}")
        from collections import Counter

        print("[05] categories", dict(Counter(cats_used)))


if __name__ == "__main__":
    main()
