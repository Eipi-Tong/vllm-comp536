import os
import json
import logging
import traceback
import sys
from typing import List, Dict, Tuple, Optional
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.sequence import CompletionSequenceGroupOutput, SequenceOutput, Logprob
from vllm.core.scheduler import SchedulerOutputs

logger = logging.getLogger(__name__)

"""
A simulator for model execution that bypasses GPU computation by replaying
pre-recorded traces of token outputs.
This is useful for testing and benchmarking the scheduling and sampling
components of vLLM without the overhead of actual model inference.
"""
class Simulator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
        # 1. Safely extract EOS token
        self.eos_token_id = 2 # Default fallback
        try:
            # Check raw tokenizer first if wrapped
            if hasattr(tokenizer, "get_lora_tokenizer"):
                raw = tokenizer.get_lora_tokenizer(None)
                if hasattr(raw, "eos_token_id") and raw.eos_token_id is not None:
                    self.eos_token_id = raw.eos_token_id
            # Check wrapper/direct
            elif hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
                self.eos_token_id = tokenizer.eos_token_id
        except Exception as e:
            logger.warning(f"Simulator: EOS extraction failed ({e}). Defaulting to {self.eos_token_id}")

        self.trace_data: Dict[Tuple[int, ...], List[int]] = {} 
        self.current_indices: Dict[str, int] = {} 
        
        # 2. Robust Path Resolution
        base_path = os.getcwd() 
        # Default path based on your snippet
        relative_path = "datasets/ShareGPT_Vicuna_unfiltered/cleaned_dataset_no_imsorry.json"
        
        # Try ENV, then relative, then absolute, then user-provided hardcoded path
        possible_paths = []
        if os.getenv("SIMULATOR_TRACE_PATH"):
            possible_paths.append(os.getenv("SIMULATOR_TRACE_PATH"))
        
        possible_paths.extend([
            relative_path,
            os.path.abspath(relative_path),
            os.path.join(base_path, relative_path),
            "../../datasets/ShareGPT_Vicuna_unfiltered/cleaned_dataset_no_imsorry.json"
        ])

        loaded = False
        for path in possible_paths:
            if path and os.path.exists(path):
                self._load_sharegpt_trace(path)
                loaded = True
                break
        
        if not loaded:
            logger.error(f"Simulator: CRITICAL - Trace file not found. Searched: {possible_paths}")
            logger.warning("Simulator: Falling back to DUMMY DATA mode.")

    def _load_sharegpt_trace(self, path: str):
        logger.info(f"Simulator: Loading traces from {path}...")
        sys.stdout.flush() # Ensure log is printed before potentially crashing
        
        try:
            with open(path, 'r') as f:
                dataset = json.load(f)
            
            raw_tok = self.tokenizer
            if hasattr(self.tokenizer, "get_lora_tokenizer"):
                raw_tok = self.tokenizer.get_lora_tokenizer(None)

            count = 0
            # LIMIT THE TRACES TO PREVENT OOM / TIMEOUT
            # ShareGPT is large; loading 100k conversations crashes small instances
            MAX_TRACES = 2000 
            
            for item in dataset:
                if count >= MAX_TRACES:
                    break

                conversations = item.get("conversations", [])
                if len(conversations) < 2:
                    continue
                
                history = []
                # Iterate pairs: User -> Assistant
                for i in range(0, len(conversations) - 1, 2):
                    user_turn = conversations[i]
                    assistant_turn = conversations[i+1]
                    
                    if user_turn['from'] not in ['human', 'user']:
                        continue
                    
                    # Add User Input
                    history.append({"role": "user", "content": user_turn['value']})
                    
                    # Apply Template
                    # Note: We tokenize=False to get string, then encode. 
                    prompt_text = raw_tok.apply_chat_template(
                        history, tokenize=False, add_generation_prompt=True
                    )
                    
                    prompt_ids = tuple(raw_tok.encode(prompt_text))
                    response_ids = raw_tok.encode(assistant_turn['value'])
                    
                    self.trace_data[prompt_ids] = response_ids
                    
                    # Add Assistant Output for next turn
                    history.append({"role": "assistant", "content": assistant_turn['value']})
                    count += 1
                
            logger.info(f"Simulator: Successfully loaded {count} multi-turn prompts.")
            sys.stdout.flush()
            
        except Exception as e:
            logger.error(f"Simulator: Error loading trace: {e}")
            logger.error(traceback.format_exc())
            sys.stdout.flush()
    
    # def simulate_step(self) -> List[SamplerOutput]:
    #     """
    #     Simulates a single step of model execution by returning the next token
    #     from the pre-recorded traces for each scheduled sequence group.
    #     If a trace is exhausted, returns EOS token.
    #     """
    #     completion_seq_group_outputs = []
    #     output = CompletionSequenceGroupOutput(
    #         samples=[
    #             SequenceOutput(
    #                 parent_seq_id=request_id,
    #                 output_token=new_token,
    #                 logprobs={
    #                     new_token: Logprob(
    #                     logprob=float("inf"), 
    #                     rank=None, 
    #                     decoded_token=None)
    #                 }
    #             )
    #         ],
    #         prompt_logprobs=None
    #     )
    #     completion_seq_group_outputs.append(output)
    #     outputs = [
    #         SamplerOutput(outputs=completion_seq_group_outputs, 
    #                       sampled_token_probs=None,
    #                       sampled_token_ids=None,
    #                       spec_decode_worker_metrics=None)
    #     ]
    #     return outputs


    def step(self, scheduler_outputs: SchedulerOutputs) -> List[SamplerOutput]:
        try:
            scheduled_groups = scheduler_outputs.scheduled_seq_groups
            group_outputs = []

            for scheduled_group in scheduled_groups:
                seq_group = scheduled_group.seq_group
                request_id = seq_group.request_id
                
                # Safely get prompt_token_ids
                first_seq = seq_group.get_seqs()[0]
                if hasattr(first_seq, "data"):
                    prompt_ids = tuple(first_seq.data.prompt_token_ids)
                else:
                    prompt_ids = tuple(first_seq.prompt_token_ids)

                if request_id not in self.current_indices:
                    self.current_indices[request_id] = 0
                    if prompt_ids not in self.trace_data:
                        # Optional: Enable debug log only if needed to reduce noise
                        # logger.debug(f"Simulator: Cache Miss! Len {len(prompt_ids)}")
                        self.trace_data[prompt_ids] = [i + 1 for i in range(20)] 
                
                response_tokens = self.trace_data[prompt_ids]
                current_idx = self.current_indices[request_id]

                if current_idx < len(response_tokens):
                    output_token = response_tokens[current_idx]
                    self.current_indices[request_id] += 1
                else:
                    output_token = self.eos_token_id
                
                seq_outputs = []
                for seq in seq_group.get_seqs():
                    # Create Logprob object safely
                    logprobs = {output_token: Logprob(0.0)}
                    seq_outputs.append(SequenceOutput(
                        parent_seq_id=seq.seq_id,
                        output_token=output_token,
                        logprobs=logprobs
                    ))

                group_outputs.append(CompletionSequenceGroupOutput(
                    samples=seq_outputs,
                    prompt_logprobs=None
                ))

            return [SamplerOutput(
                outputs=group_outputs,
                sampled_token_probs=None,
                logprobs=None,
                sampled_token_ids=None,
            )]
        except Exception as e:
            # Catch crashes to print traceback to log
            logger.critical(f"Simulator: CRASH IN STEP: {e}")
            logger.critical(traceback.format_exc())
            sys.stdout.flush()
            raise e