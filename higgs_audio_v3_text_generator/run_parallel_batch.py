#!/usr/bin/env python3
"""
Parallel batch text generation via 4 vLLM instances (ports 8000-8003).
Each worker uses ThreadPoolExecutor for concurrent requests to its vLLM.
"""

import argparse
import os
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--num-instances", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="batch_output")
    parser.add_argument("--base-port", type=int, default=8000)
    parser.add_argument("--no-postprocess", action="store_true", help="Only produce and merge exact raw output")
    args = parser.parse_args()

    if args.total <= 0:
        parser.error("--total must be positive")
    if args.num_instances <= 0:
        parser.error("--num-instances must be positive")
    if args.num_instances > args.total:
        parser.error("--num-instances cannot exceed --total")
    os.makedirs(args.output_dir, exist_ok=True)
    base, remainder = divmod(args.total, args.num_instances)
    worker_totals = [base + (1 if i < remainder else 0) for i in range(args.num_instances)]
    vllm_ports = [args.base_port + i for i in range(args.num_instances)]

    print(f"{args.num_instances} vLLM instances × {args.workers} threads each")
    print(f"Total raw target: {args.total} texts (per instance: {worker_totals})")
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
            sys.executable,
            "-u",
            os.path.join(os.path.dirname(__file__), "run_batch_generation.py"),
            "--total",
            str(worker_totals[i]),
            "--batch-size",
            str(args.batch_size),
            "--workers",
            str(args.workers),
            "--temperature",
            str(args.temperature),
            "--seed",
            str(seed),
            "--output",
            output,
            "--checkpoint",
            checkpoint,
            "--resume",
        ]

        print(f"Instance {i}: port {port} -> {output}")
        proc = subprocess.Popen(cmd, env=env)
        processes.append((i, proc, output))

    print(f"\nAll {args.num_instances} instances started. Waiting...")

    failures = []
    for i, proc, output in processes:
        ret = proc.wait()
        print(f"Instance {i} done (code {ret}): {output}")
        if ret != 0:
            failures.append((i, ret))

    if failures:
        print(f"ERROR: worker failures: {failures}; existing checkpoints were preserved", file=sys.stderr)
        sys.exit(1)

    print("\nMerging outputs...")
    merged = f"{args.output_dir}/generated_texts.jsonl"
    fd, tmp_path = tempfile.mkstemp(prefix=".generated_texts.", suffix=".tmp", dir=args.output_dir)
    merged_count = 0
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as dst:
            for _, _, output in processes:
                if not os.path.exists(output):
                    print(f"ERROR: missing worker output: {output}", file=sys.stderr)
                    sys.exit(1)
                with open(output, encoding="utf-8") as src:
                    for line in src:
                        if line.strip():
                            dst.write(line if line.endswith("\n") else line + "\n")
                            merged_count += 1
            dst.flush()
            os.fsync(dst.fileno())
        os.replace(tmp_path, merged)
    finally:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass

    if merged_count != args.total:
        print(f"ERROR: merged raw count {merged_count} != requested {args.total}", file=sys.stderr)
        sys.exit(2)
    print(f"Merged exact raw target {merged_count} -> {merged}")

    if not args.no_postprocess:
        final_output = f"{args.output_dir}/generated_texts_final.jsonl"
        cmd = [
            sys.executable,
            os.path.join(os.path.dirname(__file__), "postprocess_merge.py"),
            "--input-dir",
            args.output_dir,
            "--output",
            final_output,
            "--num-workers",
            str(args.num_instances),
            "--target-count",
            str(args.total),
        ]
        print(f"Postprocessing -> {final_output}")
        ret = subprocess.run(cmd, check=False).returncode
        if ret != 0:
            sys.exit(ret)


if __name__ == "__main__":
    main()
