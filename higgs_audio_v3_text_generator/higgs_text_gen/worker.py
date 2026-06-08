"""
Concurrent worker for LLM text generation.
Supports both vLLM API and local HuggingFace inference.
"""

import os
import random
from typing import Dict, List, Optional

from .config import GenConfig
from .prompt_builder import build_prompt
from .text_clean import attach_clean_text_batch


def _get_llm_caller():
    local_model_path = os.environ.get("LLM_LOCAL_MODEL_PATH", "")
    if local_model_path:
        from .llm_local import call_llm_local
        return lambda **kw: call_llm_local(model_path=local_model_path, **kw)
    else:
        from .llm_client import call_llm
        return call_llm


def worker(task: Dict, config: GenConfig) -> List[Dict]:
    task_id = task.get("task_id")
    scenario_key = task["scenario_key"]
    emotion = task.get("emotion", "contentment")

    prompt = build_prompt(
        scenario_key=scenario_key,
        subscene=task.get("subscene", ""),
        length_key=task.get("length_key", "medium"),
        lang_key=task.get("lang_key", "pure_cn"),
        emotion=emotion,
        batch_size=config.batch_size,
        suppression_hint=task.get("suppression_hint", ""),
        task_id=task_id,
    )

    seed = hash(f"{task_id}|{scenario_key}|{emotion}") & 0xFFFFFFFF
    rng = random.Random(seed)
    batch_temperature = min(1.0, max(0.7, config.temperature + rng.uniform(-0.08, 0.12)))

    llm_caller = _get_llm_caller()
    results = llm_caller(
        prompt=prompt,
        max_tokens=config.max_tokens,
        temperature=batch_temperature,
        max_retries=config.max_retries,
        retry_base_delay=config.retry_base_delay,
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
        item["language"] = item.get("language", "zh" if item.get("lang_type", "").startswith("pure_cn") or item.get("lang_type", "").startswith("cn") else "en")

    if config.generate_clean_text:
        attach_clean_text_batch(results)

    return results
