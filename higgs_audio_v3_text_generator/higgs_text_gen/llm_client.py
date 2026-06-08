"""
Qwen 3.6 27B vLLM client (OpenAI-compatible API).
"""

import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Dict, List, Optional


def call_llm(
    prompt: str,
    model: str = "",
    api_key: Optional[str] = None,
    base_url: str = "",
    max_retries: int = 3,
    retry_base_delay: float = 1.0,
    max_tokens: int = 4096,
    temperature: float = 0.85,
) -> List[Dict]:
    resolved_api_key = (
        api_key
        or os.environ.get("LLM_API_KEY")
        or os.environ.get("VLLM_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or "EMPTY"
    )
    resolved_model = model or os.environ.get("LLM_MODEL", "qwen3.6-27b")
    resolved_base_url = (
        base_url.rstrip("/")
        or os.environ.get("LLM_BASE_URL", "http://localhost:8000").rstrip("/")
    )

    def _extract_json(raw_text: str) -> List[Dict]:
        text = raw_text.strip()
        # Qwen3.6 thinking mode: strip think content
        think_end = text.rfind("</think>")
        if think_end != -1:
            text = text[think_end + len("</think>"):].strip()
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

    def _call_openai_compatible() -> str:
        endpoint = resolved_base_url + "/v1/chat/completions"
        payload = {
            "model": resolved_model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            endpoint,
            data=data,
            headers={
                "Authorization": f"Bearer {resolved_api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                body = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {e.code}: {err_body[:500]}")
        resp_json = json.loads(body)
        return resp_json["choices"][0]["message"]["content"]

    last_error = None
    for attempt in range(max_retries):
        try:
            raw = _call_openai_compatible()
            results = _extract_json(raw)
            if results:
                return results
            last_error = f"Empty JSON parse result from: {raw[:200]}"
        except Exception as e:
            last_error = str(e)
        if attempt < max_retries - 1:
            wait = retry_base_delay * (2 ** attempt)
            time.sleep(wait)

    print(f"LLM call failed after {max_retries} retries: {last_error}")
    return []
