#!/usr/bin/env python3
"""
Parallel batch text generation via 4 vLLM instances (ports 8000-8003).
Each worker uses ThreadPoolExecutor for concurrent requests to its vLLM.
"""

import argparse
import os
import sys
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--num-instances", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="batch_output")
    args = parser.parse_args()

    texts_per_worker = args.total // args.num_instances
    vllm_ports = [8000, 8001, 8002, 8003]

    print(f"4 vLLM instances × {args.workers} threads each")
    print(f"Total: {args.total} texts ({texts_per_worker}/instance)")
    print(f"Batch size: {args.batch_size}")

    processes = []
    for i in range(args.num_instances):
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
            "--workers", str(args.workers),
            "--temperature", str(args.temperature),
            "--seed", str(seed),
            "--output", output,
            "--checkpoint", checkpoint,
            "--resume",
            "--no-postprocess",
        ]

        print(f"Instance {i}: port {port} -> {output}")
        proc = subprocess.Popen(cmd, env=env)
        processes.append((i, proc, output))

    print(f"\nAll {args.num_instances} instances started. Waiting...")

    for i, proc, output in processes:
        ret = proc.wait()
        print(f"Instance {i} done (code {ret}): {output}")

    print("\nMerging outputs...")
    all_lines = []
    for i, _, output in processes:
        if os.path.exists(output):
            with open(output) as f:
                for line in f:
                    if line.strip():
                        all_lines.append(line.strip())

    merged = f"{args.output_dir}/generated_texts.jsonl"
    with open(merged, "w") as f:
        for line in all_lines:
            f.write(line + "\n")

    print(f"Merged {len(all_lines)} texts -> {merged}")


if __name__ == "__main__":
    main()
