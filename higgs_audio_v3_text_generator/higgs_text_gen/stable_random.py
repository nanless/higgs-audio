# Copyright (c) 2025 Boson AI
"""Stable random seed helpers shared by Higgs Audio v3 data tools."""

from __future__ import annotations

import hashlib


def stable_int(*parts: object, bits: int = 64) -> int:
    """Return a deterministic unsigned integer for arbitrary seed parts.

    Python's built-in ``hash`` is salted per process.  SHA-256 keeps generated
    prompts, references, pauses, and noise choices reproducible across hosts.
    """
    if bits <= 0 or bits > 256 or bits % 8:
        raise ValueError("bits must be a positive multiple of 8 up to 256")
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[: bits // 8], "big")
