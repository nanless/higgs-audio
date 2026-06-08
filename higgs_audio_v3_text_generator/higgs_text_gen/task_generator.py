"""
Task generation via stratified random sampling.
"""

import random
from typing import Dict, List

from .scenarios import SCENARIOS, EMOTIONS, LENGTH_SPECS, LANG_MIX_SPECS
from .config import GenConfig


def generate_task_list(config: GenConfig) -> List[Dict]:
    rng = random.Random(config.seed)

    total_batches = max(1, config.total_target // config.batch_size)

    regular_scenarios = {k: v for k, v in SCENARIOS.items() if not v.get("is_stress_test")}
    stress_scenarios = {k: v for k, v in SCENARIOS.items() if v.get("is_stress_test")}

    scenario_keys = list(regular_scenarios.keys())
    stress_keys = list(stress_scenarios.keys())

    tasks = []

    num_stress = int(total_batches * config.stress_test_ratio)
    num_regular = total_batches - num_stress

    scenario_weights = []
    for k in scenario_keys:
        scenario_weights.append(config.scenario_distribution.get(k, 1.0))
    total_w = sum(scenario_weights)
    scenario_probs = [w / total_w for w in scenario_weights]

    length_keys = list(config.length_distribution.keys())
    length_weights = list(config.length_distribution.values())
    length_total = sum(length_weights)
    length_probs = [w / length_total for w in length_weights]

    lang_keys = list(config.lang_mix_distribution.keys())
    lang_weights = list(config.lang_mix_distribution.values())
    lang_total = sum(lang_weights)
    lang_probs = [w / lang_total for w in lang_weights]

    subscene_indices = {}
    for key in scenario_keys:
        subscene_indices[key] = 0

    for _ in range(num_regular):
        scenario_key = rng.choices(scenario_keys, weights=scenario_probs, k=1)[0]
        scenario = regular_scenarios[scenario_key]

        subscenes = scenario.get("subscenes", [scenario["name"]])
        subscene = subscenes[subscene_indices[scenario_key] % len(subscenes)]
        subscene_indices[scenario_key] += 1

        emotion_weights_dict = scenario.get("typical_emotions", {})
        emotion_list = list(emotion_weights_dict.keys())
        emotion_weights_val = list(emotion_weights_dict.values())
        if not emotion_list:
            emotion_list = EMOTIONS[:]
            emotion_weights_val = [1.0] * len(emotion_list)
        emotion = rng.choices(emotion_list, weights=emotion_weights_val, k=1)[0]

        length_key = rng.choices(length_keys, weights=length_probs, k=1)[0]
        lang_key = rng.choices(lang_keys, weights=lang_probs, k=1)[0]

        tasks.append({
            "scenario_key": scenario_key,
            "subscene": subscene,
            "emotion": emotion,
            "length_key": length_key,
            "lang_key": lang_key,
            "is_stress_test": False,
        })

    for i in range(num_stress):
        sk = stress_keys[i % len(stress_keys)] if stress_keys else scenario_keys[0]
        scenario = SCENARIOS.get(sk, regular_scenarios[scenario_keys[0]])
        subscenes = scenario.get("subscenes", [scenario["name"]])
        subscene = subscenes[i % len(subscenes)]

        emotion = rng.choice(list(scenario.get("typical_emotions", {"surprise": 1.0}).keys()))
        length_key = rng.choices(length_keys, weights=length_probs, k=1)[0]
        lang_key = rng.choices(lang_keys, weights=lang_probs, k=1)[0]

        tasks.append({
            "scenario_key": sk,
            "subscene": subscene,
            "emotion": emotion,
            "length_key": length_key,
            "lang_key": lang_key,
            "is_stress_test": True,
        })

    rng.shuffle(tasks)
    for idx, task in enumerate(tasks):
        task["task_id"] = idx

    return tasks
