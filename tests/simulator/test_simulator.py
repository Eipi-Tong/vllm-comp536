import pytest
import random
from unittest.mock import MagicMock
from typing import List, Dict, Optional, Tuple

from vllm.core.scheduler import SchedulerOutputs, ScheduledSequenceGroup
from vllm.sequence import Sequence, SequenceGroup, SequenceStatus, CompletionSequenceGroupOutput
from vllm.simulator.simulator import Simulator

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _prepare_simulator(traces: Dict[str, List[int]]) -> Simulator:
    """Creates a simulator and pre-loads it with specific traces."""
    sim = Simulator()
    for req_id, tokens in traces.items():
        sim.add_trace(req_id, tokens)
    return sim

def _create_scheduler_outputs(request_ids: List[str]) -> SchedulerOutputs:
    """
    Constructs the mock SchedulerOutputs object that the LLMEngine 
    passes to the Simulator.step() method.
    """
    scheduled_seq_groups = []
    
    for req_id in request_ids:
        # Mock Sequence (The individual sequence inside a group)
        seq = MagicMock(spec=Sequence)
        seq.seq_id = 0
        seq.status = SequenceStatus.RUNNING
        
        # Mock SequenceGroup (The parent request object)
        seq_group = MagicMock(spec=SequenceGroup)
        seq_group.request_id = req_id
        seq_group.get_seqs.return_value = [seq]
        # Allow checking if it's prefill or decode, though Simulator might ignore it
        seq_group.is_prefill.return_value = False 

        # Mock ScheduledSequenceGroup (The wrapper used by Scheduler)
        ssg = MagicMock(spec=ScheduledSequenceGroup)
        ssg.seq_group = seq_group
        
        scheduled_seq_groups.append(ssg)
        
    scheduler_outputs = MagicMock(spec=SchedulerOutputs)
    scheduler_outputs.scheduled_seq_groups = scheduled_seq_groups
    scheduler_outputs.is_empty.return_value = (len(request_ids) == 0)
    
    return scheduler_outputs

def _do_step(simulator: Simulator, request_ids: List[str]) -> List[CompletionSequenceGroupOutput]:
    """
    Helper to perform one simulation step and return the flat list of 
    sequence group outputs for easier assertion.
    """
    scheduler_outputs = _create_scheduler_outputs(request_ids)
    
    # The simulator returns a list of SamplerOutput (usually one per pipeline stage)
    # We take the first one and extract its 'outputs' list.
    sampler_outputs = simulator.step(scheduler_outputs)
    
    if not sampler_outputs:
        return []
        
    # vLLM SamplerOutput contains a list of CompletionSequenceGroupOutput
    return sampler_outputs[0].outputs

# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("trace_len", [1, 10, 100])
def test_simulator_trace_correctness(trace_len: int):
    """
    Verifies that the simulator returns tokens exactly as defined in the trace
    sequentially for a single request.
    """
    req_id = "req_test_1"
    # Generate a random trace
    expected_tokens = [random.randint(0, 32000) for _ in range(trace_len)]
    
    simulator = _prepare_simulator({req_id: expected_tokens})
    
    # Step through the simulator one token at a time
    for i in range(trace_len):
        outputs = _do_step(simulator, [req_id])
        
        assert len(outputs) == 1
        # Drill down: CompletionSequenceGroupOutput -> SequenceOutput -> output_token
        actual_token = outputs[0].samples[0].output_token
        
        assert actual_token == expected_tokens[i], \
            f"Step {i}: Expected token {expected_tokens[i]}, got {actual_token}"

@pytest.mark.parametrize("batch_size", [1, 4, 16])
def test_simulator_batch_interleaving(batch_size: int):
    """
    Verifies that the simulator correctly maintains state for multiple 
    concurrent requests (batching).
    """
    # Create traces for multiple requests
    traces = {}
    request_ids = []
    
    # Example:
    # Req 0: [0, 10, 20]
    # Req 1: [1, 11, 21]
    for i in range(batch_size):
        req_id = f"req_{i}"
        request_ids.append(req_id)
        # Create distinct traces to verify we don't mix them up
        traces[req_id] = [i + (10 * step) for step in range(5)]
        
    simulator = _prepare_simulator(traces)
    
    # Run 5 steps
    for step_idx in range(5):
        outputs = _do_step(simulator, request_ids)
        assert len(outputs) == batch_size
        
        # Verify every request in the batch got the correct token for this step
        for i, output in enumerate(outputs):
            actual_token = output.samples[0].output_token
            expected_token = traces[request_ids[i]][step_idx]
            
            assert actual_token == expected_token, \
                f"Req {request_ids[i]} Step {step_idx}: Expected {expected_token}, got {actual_token}"

def test_simulator_missing_trace_fallback():
    """
    Verifies behavior when a request is scheduled but has no trace loaded.
    Should fallback to dummy generation (or error, depending on implementation).
    """
    req_id = "req_unknown"
    simulator = Simulator() # No traces loaded
    
    outputs = _do_step(simulator, [req_id])
    
    assert len(outputs) == 1
    token = outputs[0].samples[0].output_token
    assert isinstance(token, int)
    # Assuming implementation creates dummy trace starting at 10 or similar
    assert token >= 0 

def test_simulator_trace_exhaustion():
    """
    Verifies behavior when the simulator steps BEYOND the length of the trace.
    Standard behavior is often to repeat the last token or emit EOS.
    """
    req_id = "req_short"
    trace = [101, 102] # Only 2 tokens
    simulator = _prepare_simulator({req_id: trace})
    
    # Step 1
    out1 = _do_step(simulator, [req_id])[0].samples[0].output_token
    assert out1 == 101
    
    # Step 2
    out2 = _do_step(simulator, [req_id])[0].samples[0].output_token
    assert out2 == 102
    
    # Step 3 (Exhausted) - Expect repetition of last token (based on previous implementation logic)
    out3 = _do_step(simulator, [req_id])[0].samples[0].output_token
    assert out3 == 102 

def test_simulator_dynamic_batching():
    """
    Verifies simulator handles requests entering and leaving the batch dynamically.
    """
    req_1 = "req_1" # Trace: [10, 11, 12]
    req_2 = "req_2" # Trace: [20, 21, 22]
    
    simulator = _prepare_simulator({
        req_1: [10, 11, 12],
        req_2: [20, 21, 22]
    })
    
    # Step 1: Only Req 1
    out = _do_step(simulator, [req_1])
    assert out[0].samples[0].output_token == 10
    
    # Step 2: Req 1 and Req 2
    out = _do_step(simulator, [req_1, req_2])
    assert out[0].samples[0].output_token == 11 # Req 1 (2nd token)
    assert out[1].samples[0].output_token == 20 # Req 2 (1st token)
    
    # Step 3: Only Req 2
    out = _do_step(simulator, [req_2])
    assert out[0].samples[0].output_token == 21 # Req 2 (2nd token)
    
    # Step 4: Req 1 returns
    out = _do_step(simulator, [req_1])
    assert out[0].samples[0].output_token == 12 # Req 1 (3rd token, resumed state)