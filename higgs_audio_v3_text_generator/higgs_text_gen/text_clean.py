"""
Strip Higgs Audio v3 tags from text to produce clean reference text.
"""

import re
from typing import Dict, List

HIGGS_TAG_CLEAN_RE = re.compile(r"<\|(emotion|style|sfx|prosody):[a-z_]+\|>")


def strip_higgs_tags(text: str) -> str:
    cleaned = HIGGS_TAG_CLEAN_RE.sub("", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def attach_clean_text(item: Dict) -> Dict:
    text = (item.get("text") or "").strip()
    if not text:
        item["clean_text"] = ""
        return item
    item["clean_text"] = strip_higgs_tags(text)
    item["char_count"] = len(item["clean_text"])
    return item


def attach_clean_text_batch(items: List[Dict]) -> List[Dict]:
    for item in items:
        attach_clean_text(item)
    return items
