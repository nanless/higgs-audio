"""
Checkpoint/resume logic for batch text generation.
"""

import json
import os
import tempfile
from typing import Dict, List


def save_checkpoint(texts: List[Dict], path: str):
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=f".{os.path.basename(path)}.", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for item in texts:
                clean = {k: v for k, v in item.items() if not k.startswith("_")}
                f.write(json.dumps(clean, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    finally:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass


def load_checkpoint(path: str) -> List[Dict]:
    if not os.path.exists(path):
        return []
    texts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    texts.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return texts
