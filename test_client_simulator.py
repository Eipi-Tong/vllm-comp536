import requests
import time
import json

# Configuration - Adjust to your server URL
API_URL = "http://localhost:8000/v1/chat/completions"
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct" # Replace with your model name

# 1. Create a Long Shared Prefix (enough to fill multiple blocks)
# A repeating string simulates a long system prompt or document context.
# SHARED_PREFIX = "You are a helpful assistant. " * 500  # Approx 2500 tokens

def send_request(prompt_suffix, request_id_label):
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": ""},
            {"role": "user", "content": prompt_suffix}
        ],
        "temperature": 0,
        "max_tokens": 10  # We only need a short generation to test the prompt processing
    }

    print(f"--- Sending {request_id_label} ---")
    start_time = time.time()
    
    response = requests.post(API_URL, json=payload)
    
    end_time = time.time()
    latency = end_time - start_time
    
    if response.status_code == 200:
        data = response.json()
        # Some servers return usage stats like prompt_tokens here
        usage = data.get('usage', {})
        print(f"Status: Success | Total Latency: {latency:.4f}s")
        print(f"Prompt Tokens: {usage.get('prompt_tokens', 'N/A')}")
        return data
    else:
        print(f"Error: {response.text}")
        return None

def main():
    # 2. Craft Query 1 (The Cache Warmer)
    print("Step 1: Sending Initial Request (Cold Start)...")
    res1 = send_request("What is 2+2?", "Request A")
    
    # Small sleep to ensure logs separate clearly (optional)
    time.sleep(1)

    # 3. Craft Query 2 (The Cache User)
    print("\nStep 2: Sending Follow-up Request (Prefix Sharing)...")
    res2 = send_request("What is 3+3?", "Request B")

    print("\nCheck your server logs now for 'Hit Rate' or 'Block Hits'.")

if __name__ == "__main__":
    main()