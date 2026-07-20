"""
Quality filter for Higgs Audio v3 generated texts.
Validates tag format, SFX+onomatopoeia pairing, completeness, length.
"""

import re
from typing import Dict, List

from .tag_guide import validate_tag_combo
from .tags import (
    HIGGS_TAG_RE,
    VALID_EMOTIONS,
    VALID_STYLES,
    VALID_SFX,
    VALID_PROSODY,
    SFX_TAGS,
    SFX_REQUIRES_ONO,
    count_tags,
)
from .text_clean import attach_clean_text


BAD_MARKERS = [
    "股票",
    "投资",
    "政治",
    "战争",
    "sex",
    "kill",
    "die",
    "porn",
    "no cap",
    "fuck",
    "shit",
    "damn",
    "cunt",
]


def _is_complete_utterance(text: str) -> bool:
    clean = re.sub(r"<\|(emotion|style|sfx|prosody):[a-z_]+\|>", "", text).strip()
    if not clean:
        return False
    incomplete_endings = [
        ",",
        ";",
        ":",
        "，",
        "；",
        "：",
        "-",
        "—",
        "...",
        "…",
        "and",
        "or",
        "but",
        "还有",
        "可是",
        "因为",
        "而且",
    ]
    for end in incomplete_endings:
        if clean.endswith(end):
            return False
    return True


_MALFORMED_TAG_RE = re.compile(r"<\|([a-zA-Z_]+)\|>")

_VALID_CATEGORIES = {"emotion", "style", "sfx", "prosody"}

_ANY_TAG_RE = re.compile(r"<\|([a-zA-Z_]+):([a-zA-Z_]+)\|>")
_TAG_LIKE_RE = re.compile(r"<\|[^>]*\|>")


def _contains_bad_marker(text: str) -> bool:
    lower = text.lower()
    for marker in BAD_MARKERS:
        if marker.isascii():
            if re.search(rf"(?<![a-z]){re.escape(marker)}(?![a-z])", lower):
                return True
        elif marker in lower:
            return True
    return False


def _validate_higgs_tag_format(text: str) -> bool:
    # Every tag-like token must be fully understood.  Invalid punctuation or
    # names must not survive simply because the narrower valid regex skipped it.
    for token in _TAG_LIKE_RE.findall(text):
        match = HIGGS_TAG_RE.fullmatch(token)
        if match is None:
            return False
    remaining = HIGGS_TAG_RE.sub("", text)
    if "<|" in remaining or "|>" in remaining:
        return False
        if not (
            (match.group(1) == "emotion" and match.group(2) in VALID_EMOTIONS)
            or (match.group(1) == "style" and match.group(2) in VALID_STYLES)
            or (match.group(1) == "sfx" and match.group(2) in VALID_SFX)
            or (match.group(1) == "prosody" and match.group(2) in VALID_PROSODY)
        ):
            return False
    # Check for malformed tags without category prefix (e.g. <|whispering|>)
    for match in _MALFORMED_TAG_RE.finditer(text):
        bare_name = match.group(1).lower()
        # A bare name that matches a valid tag name is almost certainly a malformed tag
        if (
            bare_name in VALID_EMOTIONS
            or bare_name in VALID_STYLES
            or bare_name in VALID_SFX
            or bare_name in VALID_PROSODY
        ):
            return False
    # Check for properly formatted tags with invalid content
    for match in _ANY_TAG_RE.finditer(text):
        category, name = match.group(1), match.group(2)
        category = category.lower()
        name = name.lower()
        if category not in _VALID_CATEGORIES:
            return False
        if category == "emotion" and name not in VALID_EMOTIONS:
            return False
        if category == "style" and name not in VALID_STYLES:
            return False
        if category == "sfx" and name not in VALID_SFX:
            return False
        if category == "prosody" and name not in VALID_PROSODY:
            return False
    return True


def _validate_sfx_onomatopoeia(text: str) -> bool:
    for match in HIGGS_TAG_RE.finditer(text):
        category, name = match.group(1), match.group(2)
        if category == "sfx" and name in SFX_REQUIRES_ONO:
            end_pos = match.end()
            following = text[end_pos : end_pos + 10]
            info = SFX_TAGS.get(name, {})
            ono_cn = info.get("onomatopoeia_cn", [])
            ono_en = info.get("onomatopoeia_en", [])
            all_ono = ono_cn + ono_en
            following_lower = following.lower()
            if not any(o.lower() in following_lower for o in all_ono):
                return False
    return True


def _validate_emotion_position(text: str) -> bool:
    matches = list(HIGGS_TAG_RE.finditer(text))
    allowed_first = {"emotion", "style", "prosody"}
    for m in matches:
        category = m.group(1)
        if category not in allowed_first:
            pos = m.start()
            tag_region_end = 0
            for pm in matches:
                if pm.group(1) in allowed_first and pm.end() > tag_region_end:
                    tag_region_end = pm.end()
            if pos > tag_region_end + len(text) * 0.2 and category == "sfx":
                continue
    return True


def _validate_length_match(text: str, length_type: str) -> bool:
    from .scenarios import LENGTH_BOUNDS

    text_stripped = re.sub(r"<\|[^|]+\|>", "", text)
    char_count = len(text_stripped)
    bounds = LENGTH_BOUNDS.get(length_type, (10, 200))
    return bounds[0] <= char_count <= bounds[1] * 1.5


def _validate_tag_combinations(text: str) -> List[str]:
    issues = []
    tags = list(HIGGS_TAG_RE.finditer(text))
    tag_names = [f"{m.group(1)}:{m.group(2)}" for m in tags]
    if len(set(tag_names)) < len(tag_names):
        issues.append("repeated_tags")
    valid, conflict = validate_tag_combo(tag_names)
    if not valid:
        issues.append(conflict or "mutually_exclusive_tags")
    return issues


def _validate_style_semantics(text: str) -> bool:
    tags = {(m.group(1), m.group(2)) for m in HIGGS_TAG_RE.finditer(text)}
    if ("style", "shouting") in tags:
        clean = HIGGS_TAG_RE.sub("", text)
        letters = "".join(re.findall(r"[A-Za-z]", clean))
        if len(letters) >= 3 and letters != letters.upper():
            return False
    return True


def quality_filter(
    texts: List[Dict],
    reject_severe_length_mismatch: bool = True,
    max_tags_per_text: int = 5,
    max_same_tag_repeat: int = 2,
) -> List[Dict]:
    filtered = []
    for item in texts:
        text = item.get("text", "").strip()
        if not text or len(text) < 2:
            continue

        if _contains_bad_marker(text):
            continue

        if not _validate_higgs_tag_format(text):
            continue

        if not _validate_sfx_onomatopoeia(text):
            continue

        if not _is_complete_utterance(text):
            continue

        if not _validate_style_semantics(text):
            continue

        tag_count, tags_list = count_tags(text)
        if tag_count > max_tags_per_text:
            continue

        tag_name_counter = {}
        for m in HIGGS_TAG_RE.finditer(text):
            tn = f"{m.group(1)}:{m.group(2)}"
            tag_name_counter[tn] = tag_name_counter.get(tn, 0) + 1
        if any(c > max_same_tag_repeat for c in tag_name_counter.values()):
            continue

        length_type = item.get("length_type", "medium")
        if not _validate_length_match(text, length_type):
            if reject_severe_length_mismatch:
                continue
            item["_length_warning"] = True

        combo_issues = _validate_tag_combinations(text)
        if combo_issues:
            continue

        attach_clean_text(item)
        if not item.get("clean_text"):
            continue

        item["_tag_count"] = tag_count
        item["_tags_used"] = [f"{m.group(1)}:{m.group(2)}" for m in HIGGS_TAG_RE.finditer(text)]
        filtered.append(item)

    return filtered
