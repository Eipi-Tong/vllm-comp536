# from datasets import load_dataset

# ds = load_dataset(
#     "json", 
#     data_files="datasets/ShareGPT_Vicuna_unfiltered/ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json"
# )["train"]

# # print(ds.keys())

# for i in range(1):
#     print(f"--- Example {i} ---")
#     print(ds[i]["id"])
#     # print(ds[i]["conversations"])
#     for msg in ds[i]["conversations"]:
#         print(msg)
#         print()
#         # break
#     print()



import argparse
import asyncio
import json
import time
import numpy as np
import aiohttp
from typing import List, Dict, Any
from transformers import AutoTokenizer

ROLE_MAPPING = {
    "human": "user",
    "gpt": "assistant",
    "system": "system"
}

class ShareGPTClient:
    def __init__(self, api_url, model_name, tokenizer_name, rate):
        self.api_url = api_url
        self.model_name = model_name
        self.rate = rate
        print(f"Loading tokenizer: {tokenizer_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def apply_template(self, messages: List[Dict[str, str]]) -> str:
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    async def send_request(self, session, prompt, request_id):
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": 1024,
            "temperature": 0.7,
        }
        try:
            start_time = time.time()
            async with session.post(self.api_url, json=payload) as response:
                if response.status == 200:
                    resp_json = await response.json()
                    latency = time.time() - start_time
                    # Return the generated text so we can append it to history
                    output_text = resp_json['choices'][0]['text']
                    return output_text, latency
                else:
                    print(f"[Error] Req {request_id} failed: {response.status}")
                    return None, 0
        except Exception as e:
            print(f"[Exception] Req {request_id}: {str(e)}")
            return None, 0

    async def process_conversation(self, session, conversation_id, raw_conv, multi_turn):
        history = [
            # Force a long shared prefix for every single request
            # {"role": "system", "content": "You are a helpful assistant. " * 50} 
        ]
        
        # Iterate through turns
        # ShareGPT is [User, Assistant, User, Assistant...]
        for i in range(0, len(raw_conv) - 1, 2):
            user_turn = raw_conv[i]
            
            if user_turn['from'] not in ['human', 'user']:
                continue

            # 1. Add User Input
            history.append({"role": "user", "content": user_turn['value']})
            
            # 2. Build Prompt
            full_prompt = self.apply_template(history)
            
            # 3. Send Request
            req_id = f"conv_{conversation_id}_turn_{i//2}"
            output_text, latency = await self.send_request(session, full_prompt, req_id)
            
            if output_text:
                print(f"[Success] {req_id} ({len(full_prompt)} chars) -> {latency:.2f}s")
                # 4. Add Assistant Output to History
                history.append({"role": "assistant", "content": output_text})
            else:
                break # Stop conversation on error

            if not multi_turn:
                break # Stop after first turn if not multi-turn mode

    async def run_trace(self, trace_path, max_requests=None, multi_turn=False):
        print(f"Loading trace from {trace_path}...")
        with open(trace_path, 'r') as f:
            dataset = json.load(f)
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            req_count = 0
            
            start_time = time.time()

            for i, item in enumerate(dataset):
                if max_requests and req_count >= max_requests:
                    break
                
                raw_conv = item.get('conversations', [])
                if not raw_conv: continue

                # Schedule the conversation
                task = asyncio.create_task(self.process_conversation(session, i, raw_conv, multi_turn))
                tasks.append(task)
                req_count += 1
                
                # Poisson Arrival
                await asyncio.sleep(np.random.exponential(1.0 / self.rate))

            print("All conversations scheduled. Waiting...")
            await asyncio.gather(*tasks)
            print(f"Finished {req_count} conversations.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, default="http://localhost:8000/v1/completions")
    parser.add_argument("--trace", type=str, required=True)
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--rate", type=float, default=1.0)
    parser.add_argument("--max-requests", type=int, default=100)
    parser.add_argument("--multi-turn", action="store_true", help="Enable multi-turn conversation replay")
    
    args = parser.parse_args()

    client = ShareGPTClient(args.url, args.model, args.model, args.rate)
    asyncio.run(client.run_trace(args.trace, args.max_requests, args.multi_turn))


"""

### 3. Usage

1.  **Download ShareGPT:**
    You need the `ShareGPT_V3_unfiltered_cleaned_split.json` (or similar). If you don't have it, you can download a sample or the full set from HuggingFace.

2.  **Start your vLLM Server (Simulator):**
    Ensure your server (from Milestone 1) is running:
    ```bash
    python -m vllm.entrypoints.openai.api_server \
      --model meta-llama/Llama-3.2-1B-Instruct \
      --device cpu \
      --port 8000
    ```

3.  **Run the Client:**
    ```bash
    python client_simulator.py \
      --trace sharegpt.json \
      --model meta-llama/Llama-3.2-1B-Instruct \
      --rate 2.0

"""