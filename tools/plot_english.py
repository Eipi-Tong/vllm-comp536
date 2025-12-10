import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=str, default="results/summary.csv")
    parser.add_argument("--out-dir", type=str, default="results/plots")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load Data
    df = pd.read_csv(args.summary)
    
    # Set Style
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({'font.size': 12})

    # --- Plot 1: Sweet Spot Analysis (Block Size vs Hit Rate) ---
    plt.figure(figsize=(8, 6))
    sns.lineplot(data=df, x="block_size", y="avg_hit_rate", hue="mode", marker="o", linewidth=2.5)
    plt.title("Impact of Block Size on Prefix Hit Rate", fontsize=14, pad=15)
    plt.xlabel("Block Size (tokens)", fontsize=12)
    plt.ylabel("Average Prefix Hit Rate", fontsize=12)
    plt.xticks([8, 16, 32, 64])
    plt.ylim(0, 1.0)
    plt.legend(title="Conversation Mode")
    plt.tight_layout()
    plt.savefig(out_dir / "sweet_spot_block_size.png", dpi=300)
    print("Generated: sweet_spot_block_size.png")

    # --- Plot 2: Hit Rate Heatmap (Block Size vs Cache Size) ---
    # Filter for Multi-turn only for cleaner heatmap
    df_multi = df[df["mode"] == "multi"]
    pivot_table = df_multi.pivot(index="cache_size", columns="block_size", values="avg_hit_rate")
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={'label': 'Avg Hit Rate'})
    plt.title("Hit Rate Heatmap (Multi-turn Mode)", fontsize=14, pad=15)
    plt.xlabel("Block Size", fontsize=12)
    plt.ylabel("Cache Utilization", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_dir / "heatmap_cache_block.png", dpi=300)
    print("Generated: heatmap_cache_block.png")

    # --- Plot 3: Throughput Comparison ---
    plt.figure(figsize=(8, 6))
    sns.barplot(data=df, x="block_size", y="throughput", hue="mode", palette="muted")
    plt.title("System Throughput by Configuration", fontsize=14, pad=15)
    plt.xlabel("Block Size", fontsize=12)
    plt.ylabel("Throughput (req/s)", fontsize=12)
    plt.legend(title="Mode", loc="lower right")
    plt.tight_layout()
    plt.savefig(out_dir / "throughput_bar.png", dpi=300)
    print("Generated: throughput_bar.png")

if __name__ == "__main__":
    main()
