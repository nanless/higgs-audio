#!/usr/bin/env python3
"""
Interactively generate expressive texts for Higgs Audio v3 TTS.
Supports scene/emotion/length/language specification or free-form text profiles.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from higgs_text_gen.config import GenConfig
from higgs_text_gen.prompt_builder import build_prompt
from higgs_text_gen.llm_client import call_llm
from higgs_text_gen.text_clean import attach_clean_text_batch
from higgs_text_gen.scenarios import SCENARIOS, EMOTIONS, LENGTH_SPECS, LANG_MIX_SPECS
from higgs_text_gen.tags import RECOMMENDED_COMBINATIONS


def main():
    parser = argparse.ArgumentParser(description="Generate expressive texts for Higgs Audio v3 TTS")
    parser.add_argument("--scenario", type=str, default="daily_chat",
                        choices=list(SCENARIOS.keys()),
                        help="Scenario key")
    parser.add_argument("--subscene", type=str, default=None,
                        help="Specific subscene (auto-picked if not set)")
    parser.add_argument("--emotion", type=str, default="enthusiasm",
                        choices=EMOTIONS, help="Emotion to convey")
    parser.add_argument("--length", type=str, default="medium",
                        choices=list(LENGTH_SPECS.keys()), help="Text length")
    parser.add_argument("--lang", type=str, default="pure_cn",
                        choices=list(LANG_MIX_SPECS.keys()), help="Language type")
    parser.add_argument("--count", type=int, default=5, help="Number of texts to generate")
    parser.add_argument("--temperature", type=float, default=0.85, help="LLM temperature")
    parser.add_argument("--model", type=str, default=None, help="LLM model name")
    parser.add_argument("--base-url", type=str, default=None, help="LLM API base URL")
    parser.add_argument("--api-key", type=str, default=None, help="LLM API key")
    parser.add_argument("--output", type=str, default=None, help="Output JSONL file path")
    parser.add_argument("--list-scenarios", action="store_true", help="List available scenarios")
    parser.add_argument("--list-emotions", action="store_true", help="List available emotions")
    parser.add_argument("--list-tags", action="store_true", help="List available tags")
    parser.add_argument("--list-combinations", action="store_true", help="List recommended tag combinations")

    args = parser.parse_args()

    if args.list_scenarios:
        print("\nAvailable scenarios:")
        for key, s in SCENARIOS.items():
            print(f"  {key}: {s['name']} — {s.get('description','')}")
        return

    if args.list_emotions:
        print("\nAvailable emotions:")
        for e in EMOTIONS:
            combo = RECOMMENDED_COMBINATIONS.get(e, [])
            combo_str = " + ".join(combo[:3])
            print(f"  {e}" + (f"  (推荐搭配: {combo_str})" if combo_str else ""))
        return

    if args.list_tags:
        print("\nAll 43 Higgs Audio v3 tags:")
        from higgs_text_gen.tags import HIGGS_V3_TAGS
        for cat, tags in HIGGS_V3_TAGS.items():
            print(f"\n  [{cat}]")
            for name, info in tags.items():
                print(f"    <|{cat}:{name}|> — {info['cn']}")
        return

    if args.list_combinations:
        print("\nRecommended tag combinations:")
        for emotion, combos in RECOMMENDED_COMBINATIONS.items():
            combo_str = " + ".join(f"<|{c.replace(':','|>')}|>" for c in combos)
            print(f"  {emotion}: {combo_str}")
        return

    config = GenConfig(
        model=args.model or os.environ.get("LLM_MODEL", "qwen3.6-27b"),
        base_url=args.base_url or os.environ.get("LLM_BASE_URL", "http://localhost:8000"),
        api_key=args.api_key or os.environ.get("LLM_API_KEY"),
        temperature=args.temperature,
        batch_size=args.count,
    )

    scenario = SCENARIOS.get(args.scenario, SCENARIOS["daily_chat"])
    subscene = args.subscene or scenario["subscenes"][0]

    print(f"\nGenerating {args.count} texts...")
    print(f"  Scenario: {scenario['name']}")
    print(f"  Subscene: {subscene}")
    print(f"  Emotion: {args.emotion}")
    print(f"  Length: {args.length}")
    print(f"  Language: {args.lang}")
    print()

    prompt = build_prompt(
        scenario_key=args.scenario,
        subscene=subscene,
        length_key=args.length,
        lang_key=args.lang,
        emotion=args.emotion,
        batch_size=args.count,
        suppression_hint="",
        task_id=0,
    )

    results = call_llm(
        prompt=prompt,
        model=config.model,
        api_key=config.api_key,
        base_url=config.base_url,
        max_retries=config.max_retries,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )

    if not results:
        print("No results generated.")
        return

    for item in results:
        item.setdefault("scenario", args.scenario)
        item.setdefault("subscene", subscene)
        item.setdefault("emotion", args.emotion)
        item.setdefault("length_type", args.length)
        item.setdefault("lang_type", args.lang)
        item.setdefault("language", "zh" if args.lang.startswith("pure_cn") or args.lang.startswith("cn") else "en")

    attach_clean_text_batch(results)

    for i, item in enumerate(results):
        text = item.get("text", "")
        clean = item.get("clean_text", "")
        print(f"[{i+1}] {text}")
        if clean != text:
            print(f"    Clean: {clean}")
        print()

    if args.output:
        from higgs_text_gen.output import save_jsonl
        save_jsonl(results, args.output)
        print(f"Saved to {args.output}")

    print(f"\nUse these 'text' fields directly as Higgs Audio v3 API 'input' parameter:")
    print("  curl https://api.boson.ai/v1/audio/speech \\")
    print("    -H 'Authorization: Bearer $BOSON_API_KEY' \\")
    print(f"    -d '{{\"model\":\"higgs-audio-v3-tts\",\"input\":\"{results[0].get('text','')[:50]}...\"}}'")


if __name__ == "__main__":
    main()
