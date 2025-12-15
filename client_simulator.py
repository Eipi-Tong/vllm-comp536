import argparse
import asyncio
import json
import os
import time
from typing import Optional
from datetime import datetime

import aiohttp
import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer


ROLE_MAPPING = {"human": "user", "gpt": "assistant", "system": "system"}


class Client:
    def __init__(self, api_url: str, model_name: str, tokenizer_name: str, rate: float):
        self.api_url = api_url
        self.model_name = model_name
        self.rate = rate
        print(f"Loading tokenizer: {tokenizer_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.metrics_path = os.getenv("CLIENT_METRICS_PATH", "client_metrics.jsonl")
        os.truncate(self.metrics_path, 0)

    def apply_template(self, messages: list[dict[str, str]]) -> str:
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    async def send_request(self, session: aiohttp.ClientSession, prompt: str, request_id: str, timestamp: Optional[float] = None):
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": 1024,
            "temperature": 0.7,
        }
        try:
            start_time = time.time()
            async with session.post(self.api_url, json=payload) as response:
                latency = time.time() - start_time

                if response.status == 200:
                    resp_json = await response.json()
                    output_text = resp_json["choices"][0]["text"]

                    record = {
                        "request_id": request_id,
                        "latency": latency,  # approx TTFT since we don’t stream
                        "prompt_len_chars": len(prompt),
                        "output_len_chars": len(output_text),
                        "workload": os.getenv("WORKLOAD_NAME", "agentbank-apps"),
                        "multi_turn": self.rate is not None,  # crude flag; or pass in explicitly
                        "timestamp": time.time() if timestamp is None else timestamp,
                    }
                    with open(self.metrics_path, "a") as f:
                        f.write(json.dumps(record) + "\n")

                    return output_text, latency
                else:
                    print(f"[Error] Req {request_id} failed: {response.status}")
                    return None, 0
        except Exception as e:
            print(f"[Exception] Req {request_id}: {str(e)}")
            return None, 0

    async def _process_conversation(self, session: aiohttp.ClientSession, conversation_id: int, item: dict, multi_turn: bool):
        raise NotImplementedError("Subclasses should implement this method.")

    async def run_trace(self, trace_path: str, max_requests: Optional[int] = None, multi_turn=False):
        print(f"Loading trace from {trace_path}...")
        with open(trace_path, "rb") as f:
            if trace_path.endswith(".json"):
                dataset = json.load(f)
            elif trace_path.endswith(".jsonl"):
                dataset = []
                for line in f:
                    line = line.strip()
                    if line:
                        dataset.append(json.loads(line))
            elif trace_path.endswith(".parquet"):
                table = pq.read_table(f)
                dataset = table.to_pylist()
            else:
                return

        async with aiohttp.ClientSession() as session:
            tasks = []
            req_count = 0

            start_time = time.time()

            for i, item in enumerate(dataset):
                if max_requests and req_count >= max_requests:
                    break

                # Schedule the conversation
                task = asyncio.create_task(self._process_conversation(session, i, item, multi_turn))
                tasks.append(task)
                req_count += 1

                # Poisson Arrival
                if self.rate > 1e-6:
                    await asyncio.sleep(np.random.exponential(1.0 / self.rate))

            print("All conversations scheduled. Waiting...")
            await asyncio.gather(*tasks)
            print(f"Finished {req_count} conversations.")


class ShareGPTClient(Client):
    async def _process_conversation(self, session, conversation_id, item, multi_turn):
        raw_conv = item.get("conversations")
        if not raw_conv:
            return

        history = [
            # Force a long shared prefix for every single request
            # {"role": "system", "content": "You are a helpful assistant. " * 50}
        ]

        # Iterate through turns
        # ShareGPT is [User, Assistant, User, Assistant...]
        for i in range(0, len(raw_conv) - 1, 2):
            user_turn = raw_conv[i]

            if user_turn["from"] not in ["human", "user"]:
                continue

            # 1. Add User Input
            history.append({"role": "user", "content": user_turn["value"]})

            # 2. Build Prompt
            full_prompt = self.apply_template(history)

            # 3. Send Request
            req_id = f"conv_{conversation_id}_turn_{i // 2}"
            output_text, latency = await self.send_request(session, full_prompt, req_id)

            if output_text:
                print(f"[Success] {req_id} ({len(full_prompt)} chars) -> {latency:.2f}s")
                # 4. Add Assistant Output to History
                history.append({"role": "assistant", "content": output_text})
            else:
                break  # Stop conversation on error

            if not multi_turn:
                break  # Stop after first turn if not multi-turn mode


class CCBenchTrajectoriesClient(Client):
    async def _process_conversation(self, session, conversation_id, item, multi_turn):
        if "trajectory" not in item:
            return

        history = []
        turn = 0

        for step in json.loads(item["trajectory"]):
            if not "message" in step or step["message"]["role"] != "user":
                continue

            # 1. Add User Input
            history.append({"role": "user", "content": step["message"]["content"]})

            # 2. Build Prompt
            full_prompt = self.apply_template(history)

            # 3. Send Request
            req_id = f"conv_{conversation_id}_turn_{turn}"
            turn += 1
            timestamp = datetime.strptime(step["timestamp"], "%Y-%m-%dT%H:%M:%S.%f%z").timestamp()
            output_text, latency = await self.send_request(session, full_prompt, req_id, timestamp)

            if output_text:
                print(f"[Success] {req_id} ({len(full_prompt)} chars) -> {latency:.2f}s")
                # 4. Add Assistant Output to History
                history.append({"role": "assistant", "content": output_text})
            else:
                break  # Stop conversation on error

            if not multi_turn:
                break  # Stop after first turn if not multi-turn mode


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, default="http://localhost:8000/v1/completions")
    parser.add_argument("--template", type=str, choices=["sharegpt", "ccbench_trajectories"], default="sharegpt", help="Prompt template")
    parser.add_argument("--trace", type=lambda path: path if path.endswith((".json", ".jsonl", ".parquet")) else argparse.ArgumentTypeError("Trace file must be .json/.jsonl/.parquet"), required=True, help="Path to the trace file (.json/.jsonl/.parquet)")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="Model name to send to the vLLM OpenAI server")
    parser.add_argument("--tokenizer", type=str, default="models/Llama-3.2-1B-Instruct", help="Local tokenizer path or HF model id")
    parser.add_argument("--rate", type=float, default=1.0, help="Parameter for Poisson distribution of request sending time (0 if the request sending time is provided in the trace)")
    parser.add_argument("--max-requests", type=int, default=100)
    parser.add_argument("--multi-turn", action="store_true", help="Enable multi-turn conversation replay")

    args = parser.parse_args()

    if args.template == "sharegpt":
        ClientType = ShareGPTClient
    elif args.template == "ccbench_trajectories":
        ClientType = CCBenchTrajectoriesClient
    else:
        ClientType = Client
    client = ClientType(args.url, args.model, args.tokenizer, args.rate)
    asyncio.run(client.run_trace(args.trace, args.max_requests, args.multi_turn))
