#!/usr/bin/env python3
"""Shared helpers for Higgs Audio clone evaluation (scan + incremental I/O).

Adapted from OmniVoice batch_generate_text_and_clone/eval_common.py.

Key differences from OmniVoice:
- Higgs clones use `clone_NNNN.wav` / `clone_NNNN.json` instead of `text_NNN.wav/json`
- Higgs clone audio is 24 kHz (OmniVoice is 16 kHz)
- Higgs metadata uses `clean_text` as reference, OmniVoice uses `gen_text`
"""

from __future__ import annotations

import json
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

SIDECAR_SUFFIXES = (".eval.json", ".cer.json", ".mos.json", ".sim.json")
SKIP_DIRS = {"logs", "__pycache__", "eval_sim_embedding_cache"}


def write_json(path: Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def append_jsonl(path: Path, record: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_jsonl_lines(path: Path, records: list[dict]) -> None:
    if not records:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_gpu_list(gpus: str | None = None, gpu: int | None = None) -> List[str]:
    if gpus:
        return [g.strip() for g in gpus.split(",") if g.strip()]
    if gpu is not None:
        return [str(gpu)]
    env = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    return [g.strip() for g in env.split(",") if g.strip()] or ["0"]


def split_shards(items: list, num_workers: int) -> List[list]:
    if num_workers <= 1:
        return [items]
    shards: List[list] = [[] for _ in range(num_workers)]
    for i, item in enumerate(items):
        shards[i % num_workers].append(item)
    return shards


def merge_jsonl_parts(parts: List[Path], out: Path) -> int:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out, "w", encoding="utf-8") as dst:
        for part in sorted(parts):
            if not part.exists():
                continue
            with open(part, encoding="utf-8") as src:
                for line in src:
                    dst.write(line)
                    n += 1
    return n


class CerAccumulator:
    def __init__(self):
        self.sub = self.ins = self.del_ = self.chars = self.count = 0

    def add(self, substitutions: int, insertions: int, deletions: int, chars: int):
        self.sub += substitutions
        self.ins += insertions
        self.del_ += deletions
        self.chars += chars
        self.count += 1

    def to_dict(self) -> dict:
        weighted = (self.sub + self.ins + self.del_) / self.chars * 100 if self.chars else 0.0
        return {
            "count": self.count,
            "weighted_cer": weighted,
            "total_substitutions": self.sub,
            "total_insertions": self.ins,
            "total_deletions": self.del_,
            "total_chars": self.chars,
        }


CLONE_SIDECAR_RE = re.compile(r"^clone_\d+\.json$")


def _is_clone_sidecar(name: str) -> bool:
    """Match Higgs clone sidecar: clone_NNNN.json (digits only, not eval/mos/sim sidecars)."""
    if not CLONE_SIDECAR_RE.match(name):
        return False
    return not any(name.endswith(s) for s in SIDECAR_SUFFIXES)


def _scan_dir(root: str) -> List[Tuple[str, str, dict]]:
    """Scan a single directory tree for Higgs clone records.

    Returns (wav_path, json_path, meta_dict).
    """
    results = []
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in files:
            if not _is_clone_sidecar(name):
                continue
            json_path = Path(dirpath) / name
            wav_path = json_path.with_suffix(".wav")
            if not wav_path.is_file():
                continue
            try:
                meta = json.loads(json_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            results.append((str(wav_path), str(json_path), meta))
    return results


def iter_clone_records(out_dir: Path, workers: int = 8) -> Iterator[Tuple[Path, Path, Dict[str, Any]]]:
    """Yield (wav_path, sidecar_json, meta) for Higgs clone files."""
    out_dir = Path(out_dir)
    t0 = time.time()

    subdirs: list[str] = []
    for p in out_dir.iterdir():
        if p.is_dir() and p.name not in SKIP_DIRS:
            # Handle two-level: {dataset}/ or {dataset}/{speaker_id}/
            speaker_dirs = [str(sd) for sd in p.iterdir() if sd.is_dir() and sd.name not in SKIP_DIRS]
            if speaker_dirs:
                subdirs.extend(speaker_dirs)
            else:
                subdirs.append(str(p))

    total = 0
    if len(subdirs) > 1 and workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_scan_dir, sd): sd for sd in subdirs}
            for future in as_completed(futures):
                batch = future.result()
                total += len(batch)
                if total % 50000 < len(batch):
                    print(f"[scan] {total} items ... {time.time() - t0:.1f}s", flush=True)
                for wav_s, json_s, meta in batch:
                    yield Path(wav_s), Path(json_s), meta
    else:
        for sd in subdirs:
            for wav_s, json_s, meta in _scan_dir(sd):
                yield Path(wav_s), Path(json_s), meta


def _clone_speaker_subdirs(out_dir: Path) -> List[str]:
    """All dataset/speaker dirs (two-level) or dataset dirs (one-level) under a clone root."""
    subdirs: List[str] = []
    for p in out_dir.iterdir():
        if p.is_dir() and p.name not in SKIP_DIRS:
            try:
                spk = [str(sd) for sd in p.iterdir() if sd.is_dir() and sd.name not in SKIP_DIRS]
            except OSError:
                spk = []
            subdirs.extend(spk if spk else [str(p)])
    return subdirs


def _scan_dir_pairs(root: str) -> List[Tuple[str, str]]:
    """Fast: list (wav, json) clone pairs WITHOUT reading json meta (stat only)."""
    out: List[Tuple[str, str]] = []
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in files:
            if not _is_clone_sidecar(name):
                continue
            jp = os.path.join(dirpath, name)
            wp = jp[:-5] + ".wav"  # ".json" -> ".wav"
            if os.path.isfile(wp):
                out.append((wp, jp))
    return out


def list_clone_items(out_dir: Path, label: str = "scan", scan_workers: int = 32) -> List[Tuple[Path, Path]]:
    """Fast (wav, json) scan: parallel over speaker dirs, no json meta read."""
    out_dir = Path(out_dir)
    t0 = time.time()
    subdirs = _clone_speaker_subdirs(out_dir)
    items: List[Tuple[Path, Path]] = []
    if len(subdirs) > 1 and scan_workers > 1:
        # 线程池 (非进程池): 扫描纯 I/O (os.walk + isfile)。调用方 eval_cer.py 在 import 时
        # 已加载 torch/vllm → fork 进程池会继承其线程锁, 概率性 fork-after-threads 死锁。
        # 线程池无 fork, I/O 密集下并行度不减 (GIL 在系统调用时释放)。
        with ThreadPoolExecutor(max_workers=min(scan_workers, len(subdirs))) as ex:
            for batch in ex.map(_scan_dir_pairs, subdirs, chunksize=16):
                items.extend((Path(w), Path(j)) for w, j in batch)
    else:
        for sd in subdirs:
            items.extend((Path(w), Path(j)) for w, j in _scan_dir_pairs(sd))
    print(f"[{label}] {len(items)} clones in {time.time() - t0:.1f}s ({scan_workers}p, no-meta)", flush=True)
    return items


def list_clone_pairs(out_dir: Path, label: str = "scan", scan_workers: int = 8) -> List[Tuple[Path, Path, Path]]:
    """Yield (cloned_wav, ref_audio, sidecar_json) for similarity evaluation."""
    t0 = time.time()
    pairs = []
    for cloned, json_path, _meta in iter_clone_records(out_dir, workers=scan_workers):
        ref = _resolve_ref_audio(cloned, json_path)
        if ref is not None:
            pairs.append((cloned, ref, json_path))
    print(f"[{label}] {len(pairs)} sim pairs in {time.time() - t0:.1f}s", flush=True)
    return pairs


def _resolve_ref_audio(cloned: Path, json_path: Path) -> Path | None:
    """Per-clone reference from clone sidecar ref_audio_path."""
    try:
        meta = json.loads(json_path.read_text(encoding="utf-8"))
        ref_path = Path(meta.get("ref_audio_path", ""))
        if ref_path.is_file():
            return ref_path
    except (OSError, json.JSONDecodeError, TypeError):
        pass
    return None
