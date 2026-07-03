"""
Direct HuggingFace model loader for Qwen3.6-27B-FP8.
Uses device_map='auto' for multi-GPU inference, replacing vLLM.
"""

import json
import re
import time
import torch
from typing import Dict, List, Optional


_MODEL = None
_TOKENIZER = None
_MODEL_PATH = None


def load_model(model_path: str):
    global _MODEL, _TOKENIZER, _MODEL_PATH
    if _MODEL is not None and _MODEL_PATH == model_path:
        return _MODEL, _TOKENIZER

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model from {model_path}...")
    _TOKENIZER = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    _MODEL = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    _MODEL.eval()
    _MODEL_PATH = model_path
    print(f"Model loaded. Devices: {set(p.device for p in _MODEL.parameters())}")
    return _MODEL, _TOKENIZER


def _extract_json(raw_text: str) -> List[Dict]:
    text = raw_text.strip()

    # Qwen3.6 thinking mode: strip think content before JSON
    think_end = text.rfind("</think>")
    if think_end != -1:
        text = text[think_end + len("</think>") :].strip()

    for marker in ("```json", "```"):
        if marker in text:
            text = text.split(marker, 1)[1]
            if "```" in text:
                text = text.split("```", 1)[0].strip()
            break
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass
    objects = []
    pattern = r'\{\s*"text"\s*:\s*"((?:[^"\\]|\\.)*)"(?:\s*,\s*"(\w+)"\s*:\s*("(?:\\.|[^"\\])*"|[^"}]*))*\s*\}'
    for m in re.finditer(pattern, text, re.DOTALL):
        try:
            obj = json.loads(m.group(0))
            if "text" in obj:
                objects.append(obj)
        except json.JSONDecodeError:
            pass
    return objects


@torch.inference_mode()
def call_llm_local(
    prompt: str,
    model_path: Optional[str] = None,
    max_tokens: int = 4096,
    temperature: float = 0.85,
    max_retries: int = 2,
    retry_base_delay: float = 1.0,
    **kwargs,
) -> List[Dict]:
    model, tokenizer = load_model(model_path)

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    for attempt in range(max_retries):
        try:
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else 1.0,
                do_sample=temperature > 0,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id,
            )
            generated_ids = outputs[0][inputs.input_ids.shape[1] :]
            raw = tokenizer.decode(generated_ids, skip_special_tokens=True)

            results = _extract_json(raw)
            if results:
                return results

            if attempt < max_retries - 1:
                time.sleep(retry_base_delay * (2**attempt))
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Generation attempt {attempt + 1} failed: {e}")
                time.sleep(retry_base_delay * (2**attempt))
            else:
                print(f"Generation failed after {max_retries} attempts: {e}")

    return []
