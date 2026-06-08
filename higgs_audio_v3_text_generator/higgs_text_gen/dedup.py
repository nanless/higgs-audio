"""
Deduplication logic: exact MD5 + semantic Jaccard similarity.
"""

import hashlib
import difflib
from typing import Dict, List, Set, Tuple


def _normalize_for_dup_check(text: str) -> str:
    import re
    t = re.sub(r"<\|(emotion|style|sfx|prosody):[a-z_]+\|>", "", text)
    t = re.sub(r"\d+", "<NUM>", t)
    t = t.lower().strip()
    t = re.sub(r"[^\w\s]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _shingle_set(text: str, n: int = 3) -> Set[str]:
    return set(text[i:i + n] for i in range(len(text) - n + 1))


def _semantic_similarity(text_a: str, text_b: str) -> float:
    a_norm = _normalize_for_dup_check(text_a)
    b_norm = _normalize_for_dup_check(text_b)
    if not a_norm or not b_norm:
        return 0.0
    a_shingles = _shingle_set(a_norm, 3)
    b_shingles = _shingle_set(b_norm, 3)
    if not a_shingles or not b_shingles:
        return 0.0
    intersection = a_shingles & b_shingles
    union = a_shingles | b_shingles
    return len(intersection) / len(union)


def deduplicate(texts: List[Dict]) -> List[Dict]:
    seen_hashes = set()
    result = []
    for item in texts:
        text = item.get("text", "")
        h1 = hashlib.md5(text.encode("utf-8")).hexdigest()
        norm = _normalize_for_dup_check(text)
        h2 = hashlib.md5(norm.encode("utf-8")).hexdigest()
        if h1 in seen_hashes or h2 in seen_hashes:
            continue
        seen_hashes.add(h1)
        seen_hashes.add(h2)
        result.append(item)
    return result


def _duplicate_context_key(item: Dict) -> Tuple:
    return (
        item.get("scenario", ""),
        item.get("subscene", ""),
        item.get("emotion", ""),
        item.get("length_type", ""),
        item.get("lang_type", ""),
    )


def build_duplicate_index(texts: List[Dict]) -> Tuple[Set[str], Dict]:
    seen_normalized = set()
    context_index = {}
    for item in texts:
        text = item.get("text", "")
        norm = _normalize_for_dup_check(text)
        seen_normalized.add(norm)
        ctx_key = _duplicate_context_key(item)
        if ctx_key not in context_index:
            context_index[ctx_key] = []
        context_index[ctx_key].append((norm, _shingle_set(norm, 3)))
    return seen_normalized, context_index


def filter_incremental_duplicates(
    results: List[Dict],
    seen_normalized: Set[str],
    context_index: Dict,
    same_context_threshold: float = 0.52,
) -> Tuple[List[Dict], int]:
    filtered = []
    skipped = 0
    for item in results:
        text = item.get("text", "")
        norm = _normalize_for_dup_check(text)
        if norm in seen_normalized:
            skipped += 1
            continue
        ctx_key = _duplicate_context_key(item)
        ctx_items = context_index.get(ctx_key, [])
        dup = False
        for prev_norm, prev_shingles in ctx_items:
            if abs(len(norm) - len(prev_norm)) / max(1, max(len(norm), len(prev_norm))) > 0.55:
                continue
            curr_shingles = _shingle_set(norm, 3)
            if not curr_shingles or not prev_shingles:
                continue
            overlap = len(curr_shingles & prev_shingles) / max(1, min(len(curr_shingles), len(prev_shingles)))
            if overlap < 0.12:
                continue
            ratio = difflib.SequenceMatcher(None, norm, prev_norm).ratio()
            if ratio >= same_context_threshold:
                dup = True
                break
        if dup:
            skipped += 1
            continue
        seen_normalized.add(norm)
        if ctx_key not in context_index:
            context_index[ctx_key] = []
        context_index[ctx_key].append((norm, _shingle_set(norm, 3)))
        filtered.append(item)
    return filtered, skipped


def semantic_deduplicate(texts: List[Dict], threshold: float = 0.88) -> List[Dict]:
    by_context = {}
    for item in texts:
        ctx_key = _duplicate_context_key(item)
        by_context.setdefault(ctx_key, []).append(item)

    result = []
    for ctx_items in by_context.values():
        kept = []
        for item in ctx_items:
            dup = False
            for kept_item in kept:
                sim = _semantic_similarity(item.get("text", ""), kept_item.get("text", ""))
                if sim >= threshold:
                    dup = True
                    break
            if not dup:
                kept.append(item)
        result.extend(kept)
    return result
