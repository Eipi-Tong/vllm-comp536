import json
import numpy as np
from collections import Counter

METRICS_FILE = "metrics.jsonl"

def load_jsonl(path):
    events = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events

def main():
    events = load_jsonl(METRICS_FILE)

    hit_rates = []
    block_hits = Counter()

    # Process all events
    for ev in events:
        if ev["type"] == "request":
            hit_rates.append(ev["hit_rate"])
        elif ev["type"] == "block_hit":
            for blk in ev["block_ids"]:
                block_hits[blk] += ev["count"]

    hit_rates = np.array(hit_rates)

    # ----- PRINT SUMMARY -----
    print("\n=== Hit Rate Summary ===")
    print(f"Total Requests        : {len(hit_rates)}")
    print(f"Average Hit Rate      : {hit_rates.mean():.4f}")
    print(f"Min Hit Rate          : {hit_rates.min():.4f}")
    print(f"Max Hit Rate          : {hit_rates.max():.4f}")
    print(f"Std Dev Hit Rate      : {hit_rates.std():.4f}")

    print("\n=== Block Hit Analysis ===")
    if block_hits:
        most = block_hits.most_common()
        max_count = most[0][1]
        max_blocks = [blk for blk, cnt in most if cnt == max_count]

        print(f"Max Block Hit Count   : {max_count}")
        print(f"Block IDs with Max Hits: {max_blocks}")
        print(f"Total Unique Blocks   : {len(block_hits)}")
    else:
        print("No block hits recorded.")

    print()

if __name__ == "__main__":
    main()
