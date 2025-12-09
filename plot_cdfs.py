import json
import numpy as np
import matplotlib.pyplot as plt

def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def cdf(data):
    data = np.sort(np.array(data))
    y = np.arange(1, len(data) + 1) / len(data)
    return data, y

def plot_cdf(data, label, xlabel, outfile):
    x, y = cdf(data)
    plt.figure()
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel("CDF")
    plt.title(label)
    plt.grid(True)
    plt.savefig(outfile, bbox_inches="tight")
    print(f"Saved {outfile}")

def main():
    # 1) Prefix-sharing ratio (hit_rate)
    events = load_jsonl("metrics.jsonl")
    hit_rates = [e["hit_rate"] for e in events if e.get("type") == "request"]
    if hit_rates:
        plot_cdf(hit_rates,
                 label="Prefix Sharing Ratio (AgentBank apps)",
                 xlabel="Prefix Hit Rate",
                 outfile="cdf_hit_rate_agentbank_apps.png")

    # 2) Block hit counts (how skewed reuse is)
    block_hits = {}
    for e in events:
        if e.get("type") == "block_hit":
            for blk in e["block_ids"]:
                block_hits[blk] = block_hits.get(blk, 0) + e["count"]
    if block_hits:
        counts = list(block_hits.values())
        plot_cdf(counts,
                 label="Block Hit Count Distribution",
                 xlabel="#Hits per Block",
                 outfile="cdf_block_hits_agentbank_apps.png")

    # 3) TTFT / latency from client
    client_rows = load_jsonl("client_metrics_agentbank_apps_multi.jsonl")
    latencies = [r["latency"] for r in client_rows]
    if latencies:
        plot_cdf(latencies,
                 label="TTFT (approx) AgentBank apps multi-turn",
                 xlabel="Latency (s)",
                 outfile="cdf_ttft_agentbank_apps_multi.png")

if __name__ == "__main__":
    main()