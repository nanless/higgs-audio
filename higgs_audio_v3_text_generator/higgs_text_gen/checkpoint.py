"""
Checkpoint/resume logic for batch text generation.
"""

import json
import os
from typing import Dict, List


def save_checkpoint(texts: List[Dict], path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in texts:
            clean = {k: v for k, v in item.items() if not k.startswith("_")}
            f.write(json.dumps(clean, ensure_ascii=False) + "\n")


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
