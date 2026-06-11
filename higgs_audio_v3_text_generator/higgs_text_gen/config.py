import os
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class GenConfig:
    total_target: int = 10000
    batch_size: int = 8
    max_workers: int = 1

    scenario_distribution: Dict[str, float] = field(default_factory=lambda: {
        "daily_chat": 2.0, "business": 1.0, "education": 1.5,
        "emotional": 1.5, "entertainment": 1.8, "narration": 1.2,
        "social_media": 1.8, "service": 0.8, "creative_writing": 1.2,
        "asr_stress": 1.0,
    })

    length_distribution: Dict[str, float] = field(default_factory=lambda: {
        "ultra_short": 0.15, "short": 0.30, "medium": 0.30,
        "long": 0.18, "very_long": 0.07,
    })

    lang_mix_distribution: Dict[str, float] = field(default_factory=lambda: {
        "pure_cn": 0.45, "pure_en": 0.35, "cn_main": 0.12, "en_main": 0.08,
    })

    stress_test_ratio: float = 0.10

    semantic_dedup_threshold: float = 0.88
    same_context_dup_threshold: float = 0.52
    suppression_window_size: int = 500

    model: str = os.environ.get("LLM_MODEL", "qwen3.6-27b")
    api_key: Optional[str] = None
    base_url: str = os.environ.get("LLM_BASE_URL", "http://localhost:8000")
    temperature: float = 0.85
    max_tokens: int = 1536
    max_retries: int = 3
    retry_base_delay: float = 1.0

    reject_severe_length_mismatch: bool = True
    max_tags_per_text: int = 5
    max_same_tag_repeat: int = 2

    output_dir: str = "batch_output"
    seed: int = 42

    generate_clean_text: bool = True
