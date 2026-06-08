"""批量使用 Higgs Audio v3 复刻童声。"""
import os
import sys
import time
import glob
import requests
from pathlib import Path

BASE_DIR = "/root/code/github_repos/higgs-audio/child_voice_clone_output_higgs"
API_URL = "http://localhost:8000/v1/audio/speech"

sample_dirs = sorted(glob.glob(os.path.join(BASE_DIR, "sample_*")))
print(f"找到 {len(sample_dirs)} 个样本")

success_count = 0
fail_count = 0

for i, sample_dir in enumerate(sample_dirs):
    sample_name = os.path.basename(sample_dir)
    prompt_audio = os.path.join(sample_dir, "prompt_audio.wav")
    prompt_text_file = os.path.join(sample_dir, "prompt_text.txt")
    cloned_text_file = os.path.join(sample_dir, "cloned_text.txt")
    output_audio = os.path.join(sample_dir, "cloned_audio.wav")

    if not os.path.exists(prompt_audio):
        print(f"[{i+1}/{len(sample_dirs)}] {sample_name}: SKIP - 缺少 prompt_audio.wav")
        continue
    if not os.path.exists(cloned_text_file):
        print(f"[{i+1}/{len(sample_dirs)}] {sample_name}: SKIP - 缺少 cloned_text.txt")
        continue

    with open(cloned_text_file, "r", encoding="utf-8") as f:
        input_text = f.read().strip()
    ref_text = ""
    if os.path.exists(prompt_text_file):
        with open(prompt_text_file, "r", encoding="utf-8") as f:
            ref_text = f.read().strip()

    payload = {
        "input": input_text,
        "references": [{"audio_path": prompt_audio, "text": ref_text}],
        "temperature": 0.8,
        "top_k": 50,
    }

    try:
        print(f"[{i+1}/{len(sample_dirs)}] {sample_name}: 生成中...", end=" ", flush=True)
        start = time.time()
        resp = requests.post(API_URL, json=payload, timeout=120)
        elapsed = time.time() - start

        if resp.status_code == 200:
            with open(output_audio, "wb") as f:
                f.write(resp.content)
            size_kb = len(resp.content) / 1024
            print(f"OK ({size_kb:.0f}KB, {elapsed:.1f}s)")
            success_count += 1
        else:
            print(f"FAIL (HTTP {resp.status_code}: {resp.text[:200]})")
            fail_count += 1
    except Exception as e:
        print(f"FAIL ({e})")
        fail_count += 1

    if (i + 1) % 10 == 0:
        print(f"  进度: {i+1}/{len(sample_dirs)}, 成功 {success_count}, 失败 {fail_count}")

print(f"\n完成! 成功 {success_count}, 失败 {fail_count}")
