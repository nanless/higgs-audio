"""
Worker for vLLM API text generation (optimized).
Uses compact prompt + OpenAI-compatible vLLM API.
"""

import random
from typing import Dict, List

from .config import GenConfig
from .compact_prompt import build_compact_prompt
from .llm_client import call_llm
from .text_clean import attach_clean_text_batch


def worker(task: Dict, config: GenConfig) -> List[Dict]:
    task_id = task.get("task_id")
    scenario_key = task["scenario_key"]
    emotion = task.get("emotion", "contentment")

    prompt = build_compact_prompt(
        scenario_key=scenario_key,
        subscene=task.get("subscene", ""),
        length_key=task.get("length_key", "medium"),
        lang_key=task.get("lang_key", "pure_cn"),
        emotion=emotion,
        batch_size=config.batch_size,
        task_id=task_id or 0,
    )

    seed = hash(f"{task_id}|{scenario_key}|{emotion}") & 0xFFFFFFFF
    rng = random.Random(seed)
    batch_temperature = min(1.0, max(0.7, config.temperature + rng.uniform(-0.08, 0.12)))

    results = call_llm(
        prompt=prompt,
        model=config.model,
        api_key=config.api_key,
        base_url=config.base_url,
        max_retries=config.max_retries,
        retry_base_delay=config.retry_base_delay,
        max_tokens=config.max_tokens,
        temperature=batch_temperature,
    )

    if not results:
        return []

    for item in results:
        item["task_id"] = task_id
        item["scenario"] = item.get("scenario", scenario_key)
        item["subscene"] = item.get("subscene", task.get("subscene", ""))
        item["emotion"] = item.get("emotion", emotion)
        item["length_type"] = item.get("length_type", task.get("length_key", "medium"))
        item["lang_type"] = item.get("lang_type", task.get("lang_key", "pure_cn"))
        lang_t = item.get("lang_type", "")
        item["language"] = item.get("language", "zh" if "cn" in lang_t else "en")

    if config.generate_clean_text:
        attach_clean_text_batch(results)

    return results
