import os
import json
from datasets import load_dataset

# Reuse your existing cleaning logic from preprocess.py
from preprocess import process_conversation

# --- Config (for now: only AgentBank/apps) ---
INPUT_PARQUET = "/home/ec2-user/.cache/huggingface/hub/datasets--Solaris99--AgentBank/snapshots/6d718ff70e64b6c82236b839075763a4d5187180/apps/train-00000-of-00001.parquet"
OUTPUT_JSON   = "datasets/AgentBank/apps_cleaned.json"
MAX_CONVS     = 200   # small subset to start with


def main():
    if not os.path.exists(INPUT_PARQUET):
        print(f"[ERROR] Parquet not found: {INPUT_PARQUET}")
        return

    print(f"[INFO] Loading AgentBank/apps from {INPUT_PARQUET}")
    ds = load_dataset("parquet", data_files=INPUT_PARQUET, split="train")

    print(f"[INFO] Raw rows: {len(ds)}")

    processed = []
    for idx, row in enumerate(ds):
        cleaned = process_conversation(row)
        if cleaned is None:
            continue

        entry = {
            "conversations": cleaned["conversations"]
        }
        if "id" in row:
            entry["id"] = row["id"]

        processed.append(entry)

        if len(processed) >= MAX_CONVS:
            break

    print(f"[INFO] Kept {len(processed)} cleaned conversations.")
    print(f"[INFO] Saving to {OUTPUT_JSON}")

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(processed, f, indent=2)

    print("[INFO] Done.")


if __name__ == "__main__":
    main()