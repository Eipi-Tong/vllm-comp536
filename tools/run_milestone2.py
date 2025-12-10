#!/usr/bin/env python3
"""
Sequential sweep driver for Milestone 2.

Runs vLLM server (simulator) with varying block_size and gpu_memory_utilization,
replays ShareGPT via client_simulator, collects metrics.jsonl, and writes a
summary CSV.
"""
import argparse
import csv
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests


def wait_for_health(url: str, timeout: float = 60.0, interval: float = 2.0) -> bool:
    """Poll /health until ready or timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = requests.get(f"{url}/health")
            if resp.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(interval)
    return False


def start_server(model: str, port: int, block_size: int, cache_util: float,
                 metrics_path: Path, run_id: str, mode: str) -> subprocess.Popen:
    env = os.environ.copy()
    env["VLLM_TARGET_DEVICE"] = env.get("VLLM_TARGET_DEVICE", "cpu")
    env["VLLM_METRICS_PATH"] = str(metrics_path)
    env["VLLM_RUN_ID"] = run_id
    env["VLLM_RUN_MODE"] = mode
    env["VLLM_EVICTION_POLICY"] = "LRU"
    cmd = [
        sys.executable, "-u", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--port", str(port),
        "--block-size", str(block_size),
        "--gpu-memory-utilization", str(cache_util),
        "--disable-log-stats",
    ]
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return proc


def stop_server(proc: subprocess.Popen, grace: float = 5.0) -> None:
    if proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            proc.kill()


def run_client(url: str, trace: Path, model: str, rate: float, max_reqs: int,
               multi: bool, block_size: int, cache_blocks: Optional[int],
               metrics_path: Path, run_id: str, mode: str) -> int:
    args = [
        sys.executable, "-u", "client_simulator.py",
        "--url", url,
        "--trace", str(trace),
        "--model", model,
        "--rate", str(rate),
        "--max-requests", str(max_reqs),
        "--metrics-path", str(metrics_path),
        "--run-id", run_id,
        "--mode", mode,
        "--block-size", str(block_size),
        "--eviction-policy", "LRU",
        "--run-meta-out", f"results/raw/{run_id}_meta.json",
    ]
    if cache_blocks is not None:
        args.extend(["--cache-blocks", str(cache_blocks)])
    if multi:
        args.append("--multi-turn")
    client_cwd = Path("/home/ubuntu/vllm-test")
    proc = subprocess.run(args, cwd=str(client_cwd))
    return proc.returncode


def parse_metrics(metrics_path: Path) -> Tuple[float, float, float]:
    """
    Returns (avg_hit_rate, p99_ttft, throughput).
    throughput = unique_requests / (max_ts - min_ts) if possible.
    """
    import json
    hit_rates: List[float] = []
    ttft_list: List[float] = []
    timestamps: List[float] = []
    request_ids = set()
    with open(metrics_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("type") == "prefix_cache":
                hit_rates.append(rec.get("prefix_hit_rate", 0.0))
                timestamps.append(rec.get("ts", 0.0))
                req_id = rec.get("request_id")
                if req_id:
                    request_ids.add(req_id)
            elif rec.get("type") == "latency":
                ttft = rec.get("ttft", {})
                p99 = ttft.get("p99", 0.0) if isinstance(ttft, dict) else 0.0
                ttft_list.append(p99)
    avg_hit = sum(hit_rates) / len(hit_rates) if hit_rates else 0.0
    p99_ttft = max(ttft_list) if ttft_list else 0.0
    throughput = 0.0
    if timestamps and len(request_ids) > 0:
        duration = max(timestamps) - min(timestamps)
        if duration > 0:
            throughput = len(request_ids) / duration
    return avg_hit, p99_ttft, throughput


def append_summary(summary_csv: Path, row: Dict[str, str]) -> None:
    header = ["run_id", "block_size", "cache_size", "mode",
              "avg_hit_rate", "p99_ttft", "throughput", "timestamp"]
    exists = summary_csv.exists()
    with open(summary_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        help="Model path/name for vLLM server.")
    parser.add_argument("--trace", type=str, required=True,
                        help="ShareGPT trace JSON path.")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--rate", type=float, default=4.0)
    parser.add_argument("--max-requests", type=int, default=200)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--cooldown", type=float, default=5.0)
    parser.add_argument("--cache-blocks", type=int, default=None,
                        help="Optional explicit cache blocks metadata to log.")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    raw_dir = results_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = results_dir / "summary.csv"

    block_sizes = [8, 16, 32, 64]
    cache_utils = [0.4, 0.6, 0.8]
    modes = ["single", "multi"]

    for mode in modes:
        for block_size in block_sizes:
            for cache_util in cache_utils:
                run_id = f"m2_{mode}_bs{block_size}_cu{cache_util}_{int(time.time())}"
                metrics_path = raw_dir / f"{run_id}.jsonl"

                print(f"[RUN] {run_id} starting server...")
                server = start_server(
                    model=args.model,
                    port=args.port,
                    block_size=block_size,
                    cache_util=cache_util,
                    metrics_path=metrics_path,
                    run_id=run_id,
                    mode=mode,
                )

                healthy = wait_for_health(f"http://127.0.0.1:{args.port}")
                if not healthy:
                    print(f"[WARN] Server health check failed for {run_id}, killing server.")
                    stop_server(server)
                    time.sleep(args.cooldown)
                    continue

                print(f"[RUN] {run_id} running client...")
                ret = run_client(
                    url=f"http://127.0.0.1:{args.port}/v1/completions",
                    trace=Path(args.trace),
                    model=args.model,
                    rate=args.rate,
                    max_reqs=args.max_requests,
                    multi=(mode == "multi"),
                    block_size=block_size,
                    cache_blocks=args.cache_blocks,
                    metrics_path=metrics_path,
                    run_id=run_id,
                    mode=mode,
                )
                if ret != 0:
                    print(f"[WARN] Client exited {ret} for {run_id}")

                # Grace period to let server flush metrics to disk.
                time.sleep(3)

                print(f"[RUN] {run_id} stopping server...")
                stop_server(server)

                time.sleep(args.cooldown)

                if metrics_path.exists():
                    avg_hit, p99_ttft, throughput = parse_metrics(metrics_path)
                    row = {
                        "run_id": run_id,
                        "block_size": block_size,
                        "cache_size": cache_util,
                        "mode": mode,
                        "avg_hit_rate": f"{avg_hit:.4f}",
                        "p99_ttft": f"{p99_ttft:.4f}",
                        "throughput": f"{throughput:.4f}",
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    append_summary(summary_csv, row)
                    print(f"[RUN] {run_id} summary appended.")
                else:
                    print(f"[WARN] No metrics file for {run_id}")


if __name__ == "__main__":
    main()

