# Copyright (c) 2025 Boson AI
"""Content-identity checks for the Higgs Audio v3 ASR cache."""

from __future__ import annotations

from pathlib import Path
from typing import Any


CACHE_VERSION = 2


def wav_fingerprint(wav_path: Path) -> dict[str, int]:
    """Build a cheap identity that changes when a clone path is reused."""
    stat = wav_path.stat()
    return {
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
        "inode": stat.st_ino,
    }


def get_cached_asr_text(cache: dict[str, Any], wav_path: Path, language: str) -> str | None:
    entry = cache.get(str(wav_path))
    if not isinstance(entry, dict) or entry.get("cache_version") != CACHE_VERSION:
        return None
    if entry.get("language") != language:
        return None
    try:
        current = wav_fingerprint(wav_path)
    except OSError:
        return None
    if entry.get("wav_fingerprint") != current:
        return None
    text = entry.get("text")
    return text if isinstance(text, str) and text else None


def set_cached_asr_text(cache: dict[str, Any], wav_path: Path, language: str, text: str) -> None:
    cache[str(wav_path)] = {
        "cache_version": CACHE_VERSION,
        "text": text,
        "language": language,
        "wav_fingerprint": wav_fingerprint(wav_path),
    }
