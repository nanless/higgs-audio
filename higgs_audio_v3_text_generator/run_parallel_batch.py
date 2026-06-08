#!/usr/bin/env python3
"""
Multi-process parallel batch text generation for Higgs Audio v3.
Each process uses dedicated GPUs for maximum utilization.
"""

import argparse
import os
import sys
import subprocess
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="batch_output")
    args = parser.parse_args()

    model_path = os.environ.get("LLM_LOCAL_MODEL_PATH",
                                 "/root/.cache/modelscope/Qwen/Qwen3___6-27B-FP8")

    texts_per_worker = args.total // args.num_workers
    tasks_per_worker = texts_per_worker // args.batch_size

    print(f"Launching {args.num_workers} workers:")
    print(f"  Model: {model_path}")
    print(f"  Total: {args.total} texts ({texts_per_worker} per worker)")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Tasks per worker: {tasks_per_worker}")

    gpu_pairs = [(0, 1), (2, 3)]  # 2 GPUs per worker

    processes = []
    for i in range(args.num_workers):
        gpus = gpu_pairs[i]
        gpu_str = ",".join(str(g) for g in gpus)
        output = f"{args.output_dir}/generated_texts_w{i}.jsonl"
        checkpoint = f"{args.output_dir}/.checkpoint_w{i}.jsonl"
        seed = args.seed + i * 1000

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_str
        env["LLM_LOCAL_MODEL_PATH"] = model_path

        cmd = [
            sys.executable, "-u", os.path.join(os.path.dirname(__file__), "run_batch_generation.py"),
            "--total", str(texts_per_worker),
            "--batch-size", str(args.batch_size),
            "--workers", "1",
            "--temperature", str(args.temperature),
            "--seed", str(seed),
            "--output", output,
            "--checkpoint", checkpoint,
        ]

        print(f"\nWorker {i}: CUDA_VISIBLE_DEVICES={gpu_str}")
        print(f"  Output: {output}")
        print(f"  Cmd: {' '.join(cmd)}")

        proc = subprocess.Popen(cmd, env=env)
        processes.append((i, proc, output))

    print(f"\nAll {args.num_workers} workers started. Waiting for completion...")

    for i, proc, output in processes:
        ret = proc.wait()
        print(f"Worker {i} exited with code {ret}, output: {output}")

    print("\nAll workers done. Merging outputs...")
    all_texts = []
    for i, _, output in processes:
        if os.path.exists(output):
            with open(output) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        all_texts.append(line)

    merged_path = f"{args.output_dir}/generated_texts.jsonl"
    with open(merged_path, "w") as f:
        for line in all_texts:
            f.write(line + "\n")

    print(f"Merged {len(all_texts)} texts to {merged_path}")


if __name__ == "__main__":
    main()
