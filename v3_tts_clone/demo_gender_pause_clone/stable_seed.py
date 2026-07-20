# Copyright (c) 2025 Boson AI
"""Stable seed helper for the gender/pause clone demo."""

from __future__ import annotations

import hashlib


def stable_int(*parts: object, modulo: int = 100000) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return value % modulo if modulo else value
