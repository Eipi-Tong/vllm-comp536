import os
from datasets import load_dataset
import json

# --- Configuration ---
# 1. Point this to the specific JSON file shown in your screenshot
INPUT_FILE_PATH = "datasets/ShareGPT_Vicuna_unfiltered/ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json"
OUTPUT_FILE_PATH = "datasets/ShareGPT_Vicuna_unfiltered/cleaned_dataset_no_imsorry.json"

SYSTEM_MESSAGES_TO_REMOVE = [
    "This chat conversation is shared from TypingMind.com"
]


# all roles: {'gpt', 'system', 'bing', 'bard', 'chatgpt', 'user', 'human'}


def normalize_role(role):
    """Normalize raw roles to standard 'user', 'assistant', 'system'."""
    role = role.lower().strip()
    if role in ['human', 'user']:
        return 'user'
    if role in ['gpt', 'chatgpt', 'bing', 'bard', 'assistant']:
        return 'assistant'
    if role in ['system']:
        return 'system'
    return 'user' # Fallback default, though we filter unknown roles usually

def process_conversation(example):
    """
    Cleaning logic:
    1. Normalize roles.
    2. Remove system messages & specific spam.
    3. Remove leading assistant messages.
    4. Validate structure (Min 2 turns, Strict Alternation).
    """
    # Adjust key access based on your specific JSON structure (usually 'conversations' or 'items')
    # If your JSON is a list of objects, HF datasets puts the whole object in `example`.
    raw_conv = example.get('conversations', [])
    
    cleaned_conv = []
    
    # --- Step 1 & 2: Normalize and Filter content ---
    for turn in raw_conv:
        # Support multiple value keys common in ShareGPT formats
        content = turn.get('value') or turn.get('text') or turn.get('content') or ""
        
        # Skip empty messages
        if not content or not content.strip():
            continue
            
        # Remove specific system content (e.g. TypingMind)
        if any(sys_msg in content for sys_msg in SYSTEM_MESSAGES_TO_REMOVE):
            continue

        role = normalize_role(turn.get('from', ''))
        
        # Remove 'system' roles entirely
        if role == 'system':
            continue
            
        cleaned_conv.append({
            "from": role,
            "value": content.strip()
        })

    # --- Step 3: Remove leading 'assistant' messages ---
    while cleaned_conv and cleaned_conv[0]['from'] == 'assistant':
        cleaned_conv.pop(0)

    # --- Step 4: Strict Validation ---
    
    # (1) Must contain at least one user-assistant pair (min 2 turns)
    if len(cleaned_conv) < 2:
        return None 
    
    # (2) Roles must be user/assistant (handled by normalize)
    # (3) No empty messages (handled by strip check above)
    
    # (4) Turns must alternate strictly: user -> assistant -> user...
    # We check if any two adjacent roles are the same.
    for i in range(len(cleaned_conv) - 1):
        if cleaned_conv[i]['from'] == cleaned_conv[i+1]['from']:
            return None # Drop conversation if alternation is broken
            
    return {"conversations": cleaned_conv}

def main():
    # check if file exists
    if not os.path.exists(INPUT_FILE_PATH):
        print(f"Error: File not found at {INPUT_FILE_PATH}")
        print("Please check the path matches your folder structure.")
        return

    print(f"Loading local dataset from: {INPUT_FILE_PATH}")
    
    # Load local JSON file
    # If the JSON is a list of objects, split="train" loads it all
    dataset = load_dataset("json", data_files=INPUT_FILE_PATH, split="train")
    
    print(f"Original count: {len(dataset)}")
    
    # Process
    print("Cleaning conversations...")
    processed_data = []
    
    # Iterating is often safer for complex filtering than .map() when dropping rows
    for entry in dataset:
        cleaned = process_conversation(entry)
        if cleaned:
            # Create new entry, preserving an ID if available, else just convs
            new_entry = cleaned
            if 'id' in entry:
                new_entry['id'] = entry['id']
            processed_data.append(new_entry)

    # Convert back to HF Dataset
    from datasets import Dataset
    final_dataset = Dataset.from_list(processed_data)
    
    print(f"Cleaned count:  {len(final_dataset)}")
    print(f"Removed {len(dataset) - len(final_dataset)} invalid conversations.")

    # Save to JSON
    print(f"Saving to {OUTPUT_FILE_PATH}...")
    # final_dataset.to_json(OUTPUT_FILE_PATH, indent=2)

    with open(OUTPUT_FILE_PATH, "w") as f:
        json.dump(processed_data, f, indent=2)
    print("Done!")

if __name__ == "__main__":
    main()