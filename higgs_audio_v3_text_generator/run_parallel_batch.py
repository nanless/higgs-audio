#!/usr/bin/env python3
"""
Parallel batch text generation for Higgs Audio v3 using vLLM.
4 workers each call a dedicated vLLM instance on ports 8000-8003.
"""

import argparse
import os
import sys
import subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="batch_output")
    args = parser.parse_args()

    texts_per_worker = args.total // args.num_workers

    print(f"Launching {args.num_workers} workers via vLLM API:")
    print(f"  Total: {args.total} texts ({texts_per_worker} per worker)")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Concurrent requests per vLLM: 8")

    vllm_ports = [8000, 8001, 8002, 8003]

    processes = []
    for i in range(args.num_workers):
        port = vllm_ports[i]
        output = f"{args.output_dir}/generated_texts_w{i}.jsonl"
        checkpoint = f"{args.output_dir}/.checkpoint_w{i}.jsonl"
        seed = args.seed + i * 1000

        env = os.environ.copy()
        env["LLM_BASE_URL"] = f"http://localhost:{port}"
        env["LLM_MODEL"] = "qwen3.6-27b"
        env["LLM_API_KEY"] = "EMPTY"

        cmd = [
            sys.executable, "-u",
            os.path.join(os.path.dirname(__file__), "run_batch_generation.py"),
            "--total", str(texts_per_worker),
            "--batch-size", str(args.batch_size),
            "--workers", "8",
            "--temperature", str(args.temperature),
            "--seed", str(seed),
            "--output", output,
            "--checkpoint", checkpoint,
        ]

        print(f"\nWorker {i}: port {port} -> {output}")
        proc = subprocess.Popen(cmd, env=env)
        processes.append((i, proc, output))

    print(f"\nAll {args.num_workers} workers started. Waiting...")

    for i, proc, output in processes:
        ret = proc.wait()
        print(f"Worker {i} exited with code {ret}, output: {output}")

    print("\nMerging outputs...")
    all_texts = []
    for i, _, output in processes:
        if os.path.exists(output):
            with open(output) as f:
                for line in f:
                    if line.strip():
                        all_texts.append(line.strip())

    merged_path = f"{args.output_dir}/generated_texts.jsonl"
    with open(merged_path, "w") as f:
        for line in all_texts:
            f.write(line + "\n")

    print(f"Merged {len(all_texts)} texts to {merged_path}")


if __name__ == "__main__":
    main()
