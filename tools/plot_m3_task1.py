import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    results_dir = Path("results/m3_task1")
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(results_dir / "m3_task1_summary.csv")
    
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({'font.size': 12})
    
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=df, 
        x="workload", 
        y="avg_hit_rate", 
        hue="policy", 
        palette="viridis"
    )
    
    plt.title("Optimal Eviction Policy Varies by Workload", fontsize=14, pad=15)
    plt.xlabel("Workload Type", fontsize=12)
    plt.ylabel("Average Prefix Hit Rate", fontsize=12)
    plt.ylim(0, 1.0)
    plt.legend(title="Policy")
    
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', padding=3)
        
    plt.tight_layout()
    save_path = plots_dir / "m3_task1_comparison.png"
    plt.savefig(save_path, dpi=300)
    print(f"Saved: {save_path}")

if __name__ == "__main__":
    main()
