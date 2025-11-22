from typing import List, Dict, Optional
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.sequence import CompletionSequenceGroupOutput, SequenceOutput, Logprob
from vllm.core.scheduler import SchedulerOutputs

"""
A simulator for model execution that bypasses GPU computation by replaying
pre-recorded traces of token outputs.
This is useful for testing and benchmarking the scheduling and sampling
components of vLLM without the overhead of actual model inference.
"""
class Simulator:

    def __init__(self):
        # request_id -> list of token ids (The trace)
        self.trace_data: Dict[str, List[int]] = {} 
        # request_id -> current generated token index
        self.current_indices: Dict[str, int] = {} 
        # print("Simulator initialized: GPU execution will be bypassed.")

    def add_trace(self, request_id: str, token_ids: List[int]):
        """
        Register a trace for a specific request. 
        In later milestones, you will load these from a file.
        """
        self.trace_data[request_id] = token_ids

    def step(self, scheduler_outputs: SchedulerOutputs) -> List[SamplerOutput]:
        """
        Mimics the model_executor.execute_model behavior.
        Returns a list of SamplerOutput (one per pipeline stage, usually just 1).
        """
        scheduled_groups = scheduler_outputs.scheduled_seq_groups
        group_outputs = []

        for scheduled_group in scheduled_groups:
            seq_group = scheduled_group.seq_group
            request_id = seq_group.request_id
            
            # 1. Initialize state for new requests
            if request_id not in self.current_indices:
                self.current_indices[request_id] = 0
                # If trace is missing, generate a dummy trace (e.g., 10 tokens)
                if request_id not in self.trace_data:
                    # Mock: Output tokens 10, 11, 12, ...
                    self.trace_data[request_id] = [10 + i for i in range(20)]

            # 2. Retrieve the next token from the trace
            current_idx = self.current_indices[request_id]
            trace_tokens = self.trace_data[request_id]

            if current_idx < len(trace_tokens):
                output_token = trace_tokens[current_idx]
                self.current_indices[request_id] += 1
            else:
                # End of trace: repeat last token or handle EOS. 
                # The StopChecker in vLLM will eventually kill it if we emit EOS.
                output_token = trace_tokens[-1] 
            
            # 3. Create SequenceOutput for each sequence in the group
            # (Assuming greedy decoding / single sequence per group for simplicity)
            seq_outputs = []
            for seq in seq_group.get_seqs():
                 # Dummy logprob (required by SequenceOutput)
                 # We can set it to 0.0 (probability 1.0)
                 logprobs = {output_token: Logprob(0.0)}
                 
                 seq_outputs.append(SequenceOutput(
                     parent_seq_id=seq.seq_id,
                     output_token=output_token,
                     logprobs=logprobs
                 ))

            # 4. Create CompletionSequenceGroupOutput
            # prompt_logprobs is None unless specifically requested
            group_outputs.append(CompletionSequenceGroupOutput(
                samples=seq_outputs,
                prompt_logprobs=None
            ))

        # 5. Wrap in SamplerOutput
        # The engine expects a list of SamplerOutputs.
        sampler_output = SamplerOutput(
            outputs=group_outputs,
            sampled_token_probs=None,
            logprobs=None,
            sampled_token_ids=None,
        )

        return [sampler_output]