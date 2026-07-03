#!/usr/bin/env python3
"""Shared loaders and helpers for Higgs Audio CER/SIM post-processing.

Adapted from OmniVoice batch_generate_text_and_clone/analyze_distributions.py
and prune_and_copy.py.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_CLONE_ROOT = Path(
    os.environ.get(
        "HIGGS_CLONE_ROOT",
        "/root/group-shared/voiceprint/data/speech/speaker_diarization/"
        "merged_datasets_20250610_vad_segments_mtfaa_enhanced_extend_kid_withclone_addlibrilight_1130/"
        "audio_higgs_audio_v3_tts_clone",
    )
)

DEFAULT_TARGET_ROOT = Path(
    os.environ.get(
        "HIGGS_AUDIO_ROOT",
        "/root/group-shared/voiceprint/data/speech/speaker_diarization/"
        "merged_datasets_20250610_vad_segments_mtfaa_enhanced_extend_kid_withclone_addlibrilight_1130/audio",
    )
)

CLONE_SIDECAR_SUFFIXES = [".json", ".eval.json", ".cer.json", ".sim.json", ".mos.json"]
SIDECAR_RENAME = {".eval.json": ".cer.json"}

DEFAULT_MAX_CER = 0.03
DEFAULT_MIN_SIM = 0.8  # raw cosine (encoder now returns raw cos, not (cos+1)/2)

PRUNE_RULES_TEXT = f"DELETE: CER > {DEFAULT_MAX_CER} OR SIM < {DEFAULT_MIN_SIM}; KEEP: otherwise"

SKIP_DIR_NAMES = frozenset({"logs", "__pycache__", "eval_sim_embedding_cache"})


def extract_dataset(audio_path: str, out_dir: str) -> str:
    try:
        rel = os.path.relpath(audio_path, out_dir)
        return rel.split(os.sep)[0]
    except (ValueError, IndexError):
        return "unknown"


def parse_language(raw_lang: Any) -> str:
    if not raw_lang:
        return "unknown"
    lang = str(raw_lang).strip()
    lang_map = {
        "Chinese": "zh",
        "English": "en",
        "Japanese": "ja",
        "Unknown": "unknown",
        "en_mostly": "en",
        "cn_mostly": "zh",
        "frequent_mix": "mix",
    }
    return lang_map.get(lang, lang.lower())


def compute_stats(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    arr = np.array(values, dtype=np.float64)
    return {
        "count": int(len(arr)),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "p1": float(np.percentile(arr, 1)),
        "p5": float(np.percentile(arr, 5)),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }


def histogram_bins(values: list[float], num_bins: int = 20) -> dict:
    if not values:
        return {"bins": [], "counts": [], "edges": []}
    counts, edges = np.histogram(values, bins=num_bins)
    return {
        "bins": [float((edges[i] + edges[i + 1]) / 2) for i in range(len(counts))],
        "counts": [int(c) for c in counts],
        "edges": [float(e) for e in edges],
    }


def iter_jsonl(path: Path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def load_cer_data(out_dir: Path) -> list[dict]:
    """Load CER records from eval_higgs_cer_details*.jsonl (supports shard files)."""
    files = sorted(out_dir.glob("eval_higgs_cer_details*.jsonl"))
    if not files:
        print(f"WARN: no eval_higgs_cer_details*.jsonl under {out_dir}", file=sys.stderr)
        return []

    t0 = time.time()
    seen: set[str] = set()
    records: list[dict] = []
    out_str = str(out_dir)

    for fp in files:
        for r in iter_jsonl(fp):
            wav = r.get("wav") or r.get("wav_path") or ""
            if not wav or wav in seen:
                continue
            seen.add(wav)
            records.append(
                {
                    "wav": wav,
                    "dataset": r.get("dataset") or extract_dataset(wav, out_str),
                    "speaker_id": r.get("speaker_id", "unknown"),
                    "language": parse_language(r.get("asr_language")),
                    "manual_cer": r.get("manual_cer"),
                    "llm_cer": r.get("llm_cer"),
                }
            )

    print(
        f"[CER] Loaded {len(records):,} records (deduped from {len(files)} files) in {time.time() - t0:.1f}s",
        file=sys.stderr,
    )
    return records


def load_sim_data(out_dir: Path) -> list[dict]:
    """Load SIM records from eval_higgs_sim_details*.jsonl."""
    files = sorted(out_dir.glob("eval_higgs_sim_details*.jsonl"))
    if not files:
        print(f"WARN: no eval_higgs_sim_details*.jsonl under {out_dir}", file=sys.stderr)
        return []

    t0 = time.time()
    seen: set[str] = set()
    records: list[dict] = []
    out_str = str(out_dir)

    for fp in files:
        for r in iter_jsonl(fp):
            wav = r.get("cloned_audio", "")
            if not wav or wav in seen:
                continue
            seen.add(wav)
            records.append(
                {
                    "wav": wav,
                    "dataset": r.get("dataset") or extract_dataset(wav, out_str),
                    "speaker_id": r.get("speaker_id", "unknown"),
                    "language": parse_language(r.get("language")),
                    "similarity": r.get("similarity"),
                    "ref_audio": r.get("ref_audio", ""),
                }
            )

    print(
        f"[SIM] Loaded {len(records):,} records (deduped from {len(files)} files) in {time.time() - t0:.1f}s",
        file=sys.stderr,
    )
    return records


def build_joined_table(cer_data: list[dict], sim_data: list[dict]) -> list[dict]:
    """Join CER and SIM by wav path."""
    t0 = time.time()
    cer_map = {r["wav"]: r for r in cer_data}
    sim_map = {r["wav"]: r for r in sim_data}
    all_wavs = set(cer_map.keys()) | set(sim_map.keys())

    table = []
    for wav in all_wavs:
        r_cer = cer_map.get(wav, {})
        r_sim = sim_map.get(wav, {})
        table.append(
            {
                "wav": wav,
                "dataset": r_cer.get("dataset") or r_sim.get("dataset") or "unknown",
                "speaker_id": r_cer.get("speaker_id") or r_sim.get("speaker_id") or "unknown",
                "language": r_cer.get("language") or r_sim.get("language") or "unknown",
                "manual_cer": r_cer.get("manual_cer"),
                "llm_cer": r_cer.get("llm_cer"),
                "similarity": r_sim.get("similarity"),
                "ref_audio": r_sim.get("ref_audio", ""),
            }
        )

    print(f"[JOIN] {len(table):,} unified records in {time.time() - t0:.1f}s", file=sys.stderr)
    return table


def classify(
    cer: float | None,
    sim: float | None,
    max_cer: float = DEFAULT_MAX_CER,
    min_sim: float = DEFAULT_MIN_SIM,
) -> str:
    """Return DELETE | KEEP."""
    if cer is not None and cer > max_cer:
        return "DELETE"
    if sim is not None and sim < min_sim:
        return "DELETE"
    return "KEEP"


def classify_records(
    table: list[dict],
    max_cer: float = DEFAULT_MAX_CER,
    min_sim: float = DEFAULT_MIN_SIM,
) -> dict[str, Any]:
    """Summarize DELETE/KEEP counts overall and by dataset."""
    overall = Counter()
    by_dataset: dict[str, Counter] = defaultdict(Counter)
    missing_sim = 0
    missing_cer = 0

    for row in table:
        cer = row.get("manual_cer")
        sim = row.get("similarity")
        if cer is None:
            missing_cer += 1
        if sim is None:
            missing_sim += 1
        action = classify(cer, sim, max_cer=max_cer, min_sim=min_sim)
        overall[action] += 1
        ds = row.get("dataset", "unknown")
        by_dataset[ds][action] += 1

    return {
        "overall": dict(overall),
        "by_dataset": {k: dict(v) for k, v in sorted(by_dataset.items())},
        "missing_cer": missing_cer,
        "missing_sim": missing_sim,
        "max_cer": max_cer,
        "min_sim": min_sim,
        "rules": f"DELETE: CER > {max_cer} OR SIM < {min_sim}; KEEP: otherwise",
    }


def _scan_dataset_wavs(root: str) -> list[str]:
    wavs: list[str] = []
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in SKIP_DIR_NAMES]
        for name in files:
            if name.startswith("clone_") and name.endswith(".wav"):
                wavs.append(os.path.join(dirpath, name))
    return wavs


def scan_clone_wavs(out_dir: Path, workers: int = 32) -> list[str]:
    """Scan clone root for existing clone_*.wav files (parallel over speaker dirs)."""
    from concurrent.futures import ProcessPoolExecutor

    out_dir = Path(out_dir)
    subdirs = _list_speaker_dirs(out_dir)  # dataset/speaker level -> better balanced parallelism
    t0 = time.time()
    wavs: list[str] = []
    if len(subdirs) > 1 and workers > 1:
        with ProcessPoolExecutor(max_workers=min(workers, len(subdirs))) as ex:
            for batch in ex.map(_scan_dataset_wavs, subdirs, chunksize=16):
                wavs.extend(batch)
    else:
        for sd in subdirs:
            wavs.extend(_scan_dataset_wavs(sd))
    print(f"[scan] {len(wavs):,} clone wavs on disk in {time.time() - t0:.1f}s ({workers}p)", file=sys.stderr)
    return wavs


# ---- Authoritative per-clone eval maps from sidecars (parallel) ----
# Unlike load_cer_data/load_sim_data (which read the append-only aggregate JSONL and
# keep the FIRST record per wav), these read the per-clone .cer.json / .sim.json
# sidecars directly. Sidecars are deleted on prune and regenerated on re-eval, so they
# are always current -- immune to the stale-first-record problem when a clone index is
# reused across rounds.


def _list_speaker_dirs(out_dir: Path) -> list[str]:
    """All dataset/speaker directories under a clone root."""
    dirs: list[str] = []
    for ds in out_dir.iterdir():
        if not ds.is_dir() or ds.name in SKIP_DIR_NAMES:
            continue
        try:
            for spk in ds.iterdir():
                if spk.is_dir() and spk.name not in SKIP_DIR_NAMES:
                    dirs.append(str(spk))
        except OSError:
            continue
    return dirs


def _scan_sidecars(task: tuple) -> list[tuple[str, float | None]]:
    """Worker: read `suffix` sidecars under the given speaker dirs -> (wav_path, value)."""
    dirs, suffix, wav_key, val_key = task
    out: list[tuple[str, float | None]] = []
    for d in dirs:
        try:
            names = os.listdir(d)
        except OSError:
            continue
        for name in names:
            if not name.endswith(suffix):
                continue
            fp = os.path.join(d, name)
            try:
                with open(fp, encoding="utf-8") as f:
                    rec = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            wav = rec.get(wav_key) or (fp[: -len(suffix)] + ".wav")
            out.append((wav, rec.get(val_key)))
    return out


def _load_sidecar_map(
    out_dir: Path, suffix: str, wav_key: str, val_key: str, workers: int, label: str
) -> dict[str, float | None]:
    from concurrent.futures import ProcessPoolExecutor, as_completed

    out_dir = Path(out_dir)
    spk_dirs = _list_speaker_dirs(out_dir)
    t0 = time.time()
    result: dict[str, float | None] = {}
    if not spk_dirs:
        print(f"WARN: no speaker dirs under {out_dir}", file=sys.stderr)
        return result
    if workers > 1 and len(spk_dirs) > 1:
        shards = [spk_dirs[i::workers] for i in range(workers)]
        tasks = [(s, suffix, wav_key, val_key) for s in shards if s]
        with ProcessPoolExecutor(max_workers=len(tasks)) as ex:
            futs = [ex.submit(_scan_sidecars, t) for t in tasks]
            for fut in as_completed(futs):
                for wav, val in fut.result():
                    result[wav] = val
    else:
        for wav, val in _scan_sidecars((spk_dirs, suffix, wav_key, val_key)):
            result[wav] = val
    print(
        f"[{label}] {len(result):,} sidecars from {len(spk_dirs):,} speakers ({workers}p) in {time.time() - t0:.1f}s",
        file=sys.stderr,
    )
    return result


def load_cer_map_sidecars(out_dir: Path, workers: int = 32) -> dict[str, float | None]:
    """Authoritative {wav_path: manual_cer} from per-clone .cer.json (parallel)."""
    return _load_sidecar_map(out_dir, ".cer.json", "wav_path", "manual_cer", workers, "CER-sidecar")


def load_sim_map_sidecars(out_dir: Path, workers: int = 32) -> dict[str, float | None]:
    """Authoritative {wav_path: similarity} from per-clone .sim.json (parallel)."""
    return _load_sidecar_map(out_dir, ".sim.json", "cloned_audio", "similarity", workers, "SIM-sidecar")


def parse_higgs_target(sim_rec: dict, cloned_path: str, out_dir: Path) -> tuple[str, str, str] | None:
    """Map a clone to (target_dataset, target_speaker, ref_stem) for copy."""
    dataset = sim_rec.get("dataset")
    speaker_id = sim_rec.get("speaker_id")
    ref_audio = sim_rec.get("ref_audio", "")
    ref_stem = Path(ref_audio).stem if ref_audio else "ref_audio"

    if dataset and speaker_id:
        return dataset, speaker_id, ref_stem

    try:
        rel = Path(cloned_path).relative_to(out_dir)
        if len(rel.parts) >= 2:
            return rel.parts[0], rel.parts[1], ref_stem
    except ValueError:
        pass
    return None


def analyze_prune_breakdown(
    table: list[dict],
    max_cer: float = DEFAULT_MAX_CER,
    min_sim: float = DEFAULT_MIN_SIM,
) -> dict[str, Any]:
    """Detailed DELETE/KEEP analysis for current prune thresholds."""
    total = len(table)
    delete_n = keep_n = 0
    reason_cer_only = reason_sim_only = reason_both = 0
    delete_cers: list[float] = []
    delete_sims: list[float] = []
    keep_cers: list[float] = []
    keep_sims: list[float] = []

    quadrants = {
        "keep_ok": 0,
        "delete_cer_only": 0,
        "delete_sim_only": 0,
        "delete_both": 0,
    }

    by_dataset: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"total": 0, "delete": 0, "keep": 0, "cer_only": 0, "sim_only": 0, "both": 0}
    )
    by_language: dict[str, dict[str, Any]] = defaultdict(lambda: {"total": 0, "delete": 0, "keep": 0})

    for row in table:
        cer = row.get("manual_cer")
        sim = row.get("similarity")
        ds = row.get("dataset", "unknown")
        lang = row.get("language", "unknown")
        action = classify(cer, sim, max_cer=max_cer, min_sim=min_sim)

        by_dataset[ds]["total"] += 1
        by_language[lang]["total"] += 1

        cer_bad = cer is not None and cer > max_cer
        sim_bad = sim is not None and sim < min_sim

        if cer is not None and sim is not None:
            if not cer_bad and not sim_bad:
                quadrants["keep_ok"] += 1
            elif cer_bad and sim_bad:
                quadrants["delete_both"] += 1
            elif cer_bad:
                quadrants["delete_cer_only"] += 1
            else:
                quadrants["delete_sim_only"] += 1

        if action == "DELETE":
            delete_n += 1
            by_dataset[ds]["delete"] += 1
            by_language[lang]["delete"] += 1
            if cer is not None:
                delete_cers.append(cer)
            if sim is not None:
                delete_sims.append(sim)
            if cer_bad and sim_bad:
                reason_both += 1
                by_dataset[ds]["both"] += 1
            elif cer_bad:
                reason_cer_only += 1
                by_dataset[ds]["cer_only"] += 1
            elif sim_bad:
                reason_sim_only += 1
                by_dataset[ds]["sim_only"] += 1
        else:
            keep_n += 1
            by_dataset[ds]["keep"] += 1
            by_language[lang]["keep"] += 1
            if cer is not None:
                keep_cers.append(cer)
            if sim is not None:
                keep_sims.append(sim)

    ds_rows = []
    for ds, info in sorted(by_dataset.items()):
        t = info["total"]
        d = info["delete"]
        ds_rows.append(
            {
                "dataset": ds,
                "total": t,
                "delete": d,
                "keep": info["keep"],
                "delete_pct": round(100.0 * d / t, 2) if t else 0.0,
                "keep_pct": round(100.0 * info["keep"] / t, 2) if t else 0.0,
                "delete_cer_only": info["cer_only"],
                "delete_sim_only": info["sim_only"],
                "delete_both": info["both"],
            }
        )

    lang_rows = []
    for lang, info in sorted(by_language.items()):
        if info["total"] < 100:
            continue
        t = info["total"]
        d = info["delete"]
        lang_rows.append(
            {
                "language": lang,
                "total": t,
                "delete": d,
                "keep": info["keep"],
                "delete_pct": round(100.0 * d / t, 2) if t else 0.0,
            }
        )

    return {
        "rules": f"DELETE: CER > {max_cer} OR SIM < {min_sim}; KEEP: otherwise",
        "max_cer": max_cer,
        "min_sim": min_sim,
        "total": total,
        "delete": delete_n,
        "keep": keep_n,
        "delete_pct": round(100.0 * delete_n / total, 2) if total else 0.0,
        "keep_pct": round(100.0 * keep_n / total, 2) if total else 0.0,
        "delete_reasons": {
            "cer_only": reason_cer_only,
            "sim_only": reason_sim_only,
            "both": reason_both,
            "note": "Reasons overlap when both thresholds fail; sums exceed DELETE count.",
        },
        "quadrants": quadrants,
        "delete_subset": {
            "cer": compute_stats(delete_cers),
            "sim": compute_stats(delete_sims),
        },
        "keep_subset": {
            "cer": compute_stats(keep_cers),
            "sim": compute_stats(keep_sims),
        },
        "by_dataset": ds_rows,
        "by_language": lang_rows,
    }


def threshold_matrix(table: list[dict]) -> list[dict]:
    """Count records meeting CER/SIM threshold pairs (for analysis)."""
    cer_thresholds = [0.03, 0.05, 0.08, 0.10, 0.15, 0.20]
    sim_thresholds = [0.60, 0.70, 0.75, 0.80, 0.85]  # raw cosine scale
    rows = []
    paired = [r for r in table if r.get("manual_cer") is not None and r.get("similarity") is not None]
    total = len(paired)
    for cer_max in cer_thresholds:
        for sim_min in sim_thresholds:
            n = sum(1 for r in paired if r["manual_cer"] <= cer_max and r["similarity"] > sim_min)
            rows.append(
                {
                    "cer_max": cer_max,
                    "sim_min": sim_min,
                    "count": n,
                    "pct": round(100.0 * n / total, 2) if total else 0.0,
                }
            )
    return rows
