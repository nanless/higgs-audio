#!/usr/bin/env python3
# Copyright (c) 2025 Boson AI
"""Re-VAD existing clones and re-insert variable pause gaps (no TTS)."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path

from stable_seed import stable_int


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

_mod = SourceFileLoader("run_clone", str(HERE / "03_run_clone.py")).load_module()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--clones-dir",
        default=str(HERE.parent.parent / "clone_workdir" / "demo_gender_pause_clone" / "clones"),
    )
    ap.add_argument("--pause-sec-min", type=float, default=1.0)
    ap.add_argument("--pause-sec-max", type=float, default=3.5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(args.clones_dir)
    jsons = sorted(root.rglob("clone/clone.json"))
    if not jsons:
        raise SystemExit(f"no clone.json under {root}")

    _mod._get_vad_model()
    ok = fail = 0
    pause_all: list[float] = []
    for jp in jsons:
        wav = jp.with_suffix(".wav")
        if not wav.is_file():
            fail += 1
            continue
        meta = json.loads(jp.read_text(encoding="utf-8"))
        with open(wav, "rb") as f:
            data = f.read()
        audio, sr = _mod.wav_bytes_to_float_mono(data)
        uid = meta.get("uid") or f"{meta.get('dataset')}/{meta.get('speaker_id')}"
        rng = random.Random(args.seed + 9173 + stable_int(uid))
        out, vad_segs, pause_secs = _mod.vad_splice_with_pauses(
            audio,
            sr,
            pause_sec_min=args.pause_sec_min,
            pause_sec_max=args.pause_sec_max,
            rng=rng,
        )
        speech_sec = round(sum(float(s.get("dur_sec") or 0.0) for s in vad_segs), 3)
        if speech_sec < 5.0:
            print(f"fail {uid} speech={speech_sec}")
            fail += 1
            continue
        _mod.write_wav(str(wav), out, sr)
        meta["pause_postprocess"] = "vad_splice"
        meta["pause_sec_min"] = args.pause_sec_min
        meta["pause_sec_max"] = args.pause_sec_max
        meta["pause_secs"] = pause_secs
        meta["pause_sec_mean"] = round(sum(pause_secs) / len(pause_secs), 3) if pause_secs else 0.0
        meta.pop("pause_sec", None)
        meta["speech_duration_sec"] = speech_sec
        meta["vad_num_segments"] = len(vad_segs)
        meta["vad_segments"] = vad_segs
        meta["duration_sec"] = round(len(out) / sr, 3)
        meta["bytes"] = os.path.getsize(wav)
        meta["respliced_variable_pause"] = True
        jp.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        pause_all.extend(pause_secs)
        ok += 1
        print(
            f"ok {uid} dur={meta['duration_sec']} speech={speech_sec} "
            f"pauses={pause_secs[:6]}{'...' if len(pause_secs) > 6 else ''}"
        )

    print(f"done ok={ok} fail={fail}")
    if pause_all:
        print(
            f"pause_secs n={len(pause_all)} "
            f"min={min(pause_all):.2f} max={max(pause_all):.2f} "
            f"mean={sum(pause_all) / len(pause_all):.2f}"
        )


if __name__ == "__main__":
    main()
