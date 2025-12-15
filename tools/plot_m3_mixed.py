import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import json
import argparse
from pathlib import Path

def main():
    results_dir = Path("results/m3_mixed")
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Read Summary CSV
    csv_path = results_dir / "m3_summary.csv"
    if not csv_path.exists():
        print("Error: summary csv not found. Please run generate_m3_data.py first.")
        return

    df_sum = pd.read_csv(csv_path)
    
    # Set Plot Style
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({'font.size': 12})
    
    # --- Plot 1: Policy Comparison Bar Chart ---
    print("Generating Policy Comparison Chart...")
    plt.figure(figsize=(10, 6))
    
    # Filter for the "Mixed" workload to show the final showdown
    # Or show all workloads to prove robustness. Let's show all.
    
    ax = sns.barplot(
        data=df_sum, 
        x="workload", 
        y="avg_hit_rate", 
        hue="policy", 
        palette=["#95a5a6", "#3498db", "#e74c3c"] # Grey(LRU), Blue(LFU), Red(FG-LRU - Highlight)
    )
    
    plt.title("Impact of Eviction Policy on Different Workloads", fontsize=14, pad=15)
    plt.xlabel("Workload Type", fontsize=12)
    plt.ylabel("Average Prefix Hit Rate", fontsize=12)
    plt.legend(title="Eviction Policy")
    plt.ylim(0, 1.0)
    
    # Add numerical labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', padding=3)
        
    plt.tight_layout()
    save_path_bar = plots_dir / "m3_policy_comparison.png"
    plt.savefig(save_path_bar, dpi=300)
    print(f"Saved: {save_path_bar}")

    # --- Plot 2: Hit Rate Distribution (Violin Plot) ---
    print("Generating Distribution Violin Plot...")
    
    # Read raw JSONL data for the Mixed Workload
    raw_data = []
    for file_path in results_dir.glob("*.jsonl"):
        # Identify policy from filename
        policy_name = "Unknown"
        if "LRU" in file_path.name: policy_name = "LRU"
        if "LFU" in file_path.name: policy_name = "LFU"
        if "FG-LRU" in file_path.name: policy_name = "FG-LRU (Ours)"
        
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    raw_data.append({
                        "Policy": policy_name,
                        "Hit Rate": rec['prefix_hit_rate']
                    })
                except:
                    pass
    
    if raw_data:
        df_raw = pd.DataFrame(raw_data)
        
        plt.figure(figsize=(8, 6))
        sns.violinplot(
            data=df_raw, 
            x="Policy", 
            y="Hit Rate", 
            palette=["#95a5a6", "#3498db", "#e74c3c"],
            cut=0 # Don't extend past observed data
        )
        plt.title("Hit Rate Distribution under Mixed Workload (DevOps)", fontsize=14, pad=15)
        plt.ylabel("Prefix Hit Rate Density", fontsize=12)
        plt.xlabel("Eviction Policy", fontsize=12)
        
        plt.tight_layout()
        save_path_violin = plots_dir / "m3_hit_rate_dist.png"
        plt.savefig(save_path_violin, dpi=300)
        print(f"Saved: {save_path_violin}")
    else:
        print("No JSONL data found for violin plot.")

if __name__ == "__main__":
    main()
