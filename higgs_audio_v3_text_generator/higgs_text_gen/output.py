"""
Output formatting and statistics.
"""

import json
import os
from collections import Counter
from typing import Dict, List


def format_jsonl_record(item: Dict) -> Dict:
    record = {
        "text": item.get("text", ""),
        "clean_text": item.get("clean_text", ""),
        "scenario": item.get("scenario", ""),
        "subscene": item.get("subscene", ""),
        "emotion": item.get("emotion", ""),
        "length_type": item.get("length_type", ""),
        "lang_type": item.get("lang_type", ""),
        "language": item.get("language", ""),
        "tags_used": item.get("_tags_used", []),
        "tag_count": item.get("_tag_count", 0),
        "char_count": item.get("char_count", len(item.get("clean_text", ""))),
        "task_id": item.get("task_id"),
    }
    return record


def save_jsonl(texts: List[Dict], path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in texts:
            record = format_jsonl_record(item)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def print_statistics(texts: List[Dict]):
    print("\n" + "=" * 60)
    print(f"Total texts: {len(texts)}")
    print(f"Length distribution: {dict(Counter(t.get('length_type') for t in texts))}")
    print(f"Language distribution: {dict(Counter(t.get('lang_type') for t in texts))}")
    print(f"Scenario distribution: {dict(Counter(t.get('scenario') for t in texts))}")

    tag_counter = Counter()
    for t in texts:
        for tag in t.get("_tags_used", []):
            tag_counter[tag] += 1
    print(f"Top 10 tags: {tag_counter.most_common(10)}")

    emotion_counter = Counter(t.get("emotion") for t in texts)
    print(f"Emotion distribution: {emotion_counter.most_common(15)}")

    tagged_count = sum(1 for t in texts if t.get("_tag_count", 0) > 0)
    untagged_count = len(texts) - tagged_count
    print(f"Tagged texts: {tagged_count} ({tagged_count/max(1,len(texts))*100:.1f}%)")
    print(f"Untagged texts: {untagged_count} ({untagged_count/max(1,len(texts))*100:.1f}%)")

    avg_chars = sum(t.get("char_count", 0) for t in texts) / max(1, len(texts))
    print(f"Average chars per text: {avg_chars:.1f}")
    print("=" * 60)
