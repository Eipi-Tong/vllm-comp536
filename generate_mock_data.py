import csv
import json
import random
import time
from pathlib import Path
import numpy as np

# --- Configuration ---
results_dir = Path("results")
raw_dir = results_dir / "raw"
summary_csv = results_dir / "summary.csv"

block_sizes = [8, 16, 32, 64]
cache_utils = [0.4, 0.6, 0.8]
modes = ["single", "multi"]

# CSV Header
header = ["run_id", "block_size", "cache_size", "mode",
          "avg_hit_rate", "p99_ttft", "throughput", "timestamp"]

print("Generating mock experiment data...")

with open(summary_csv, "w", newline="", encoding="utf-8") as f_sum:
    writer = csv.DictWriter(f_sum, fieldnames=header)
    writer.writeheader()

    for mode in modes:
        for bs in block_sizes:
            for cu in cache_utils:
                run_id = f"m2_{mode}_bs{bs}_cu{cu}_{int(time.time())}"
                
                # --- Logic for Realistic Data ---
                
                # 1. Hit Rate Base: Multi-turn is naturally higher due to history reuse
                base_hit = 0.55 if mode == "multi" else 0.15
                
                # 2. Cache Size Impact: Larger cache = slightly better hit rate
                base_hit += (cu - 0.4) * 0.1 
                
                # 3. Block Size Impact (Sweet Spot Analysis)
                # BS=8: High overhead, slightly lower efficiency
                # BS=16/32: Sweet spot, optimal reuse
                # BS=64: Internal fragmentation reduces hit rate
                if bs == 8: base_hit -= 0.02
                if bs == 16: base_hit += 0.03
                if bs == 32: base_hit += 0.02
                if bs == 64: base_hit -= 0.05
                
                # Add random noise
                avg_hit = min(0.98, max(0.05, base_hit + random.uniform(-0.01, 0.01)))
                
                # TTFT (Time To First Token): Smaller blocks = more metadata overhead = higher latency
                ttft = 0.015 + (16/bs) * 0.002 + random.uniform(0, 0.002)
                
                # Throughput
                thp = 4.5 + random.uniform(-0.2, 0.2)

                # --- Write Summary Row ---
                writer.writerow({
                    "run_id": run_id,
                    "block_size": bs,
                    "cache_size": cu,
                    "mode": mode,
                    "avg_hit_rate": f"{avg_hit:.4f}",
                    "p99_ttft": f"{ttft:.4f}",
                    "throughput": f"{thp:.4f}",
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                })

                # --- Generate Raw JSONL (For CDF Plots) ---
                jsonl_path = raw_dir / f"{run_id}.jsonl"
                with open(jsonl_path, "w") as f_raw:
                    # Generate 200 requests per run
                    for i in range(200):
                        # Use Normal Distribution for realistic curves
                        req_hit = min(1.0, max(0.0, np.random.normal(avg_hit, 0.15)))
                        
                        # Simulate Cold Starts (some requests have 0 hit rate)
                        if random.random() < 0.15: req_hit = 0.0
                        
                        rec = {
                            "ts": time.time() + i*0.05,
                            "type": "prefix_cache",
                            "request_id": f"conv_{i}",
                            "prefix_hit_rate": req_hit,
                            "reused_block_count": int(req_hit * 100),
                            "total_blocks": 100,
                            # Simulate Reuse Interval (Temporal Locality)
                            "reuse_intervals": [random.expovariate(1.0/10.0) for _ in range(5)]
                        }
                        f_raw.write(json.dumps(rec) + "\n")

print(f"Done! Mock data generated in {results_dir}")
