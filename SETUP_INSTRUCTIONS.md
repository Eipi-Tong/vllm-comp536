# Set Up Instructions

## 1. Set up the EC2 Instance (m6i.xlarge + 50G)

## 2. Download vLLM and checkout to right branch

```bash
git clone https://github.com/vllm-project/vllm.git vllm-test
cd vllm-test
git checkout 94d545a1a18e20ea8763a6760194589b8a3c9065
```

## 3. Install python venv

```bash
sudo apt update
sudo apt -y install python3.12-venv
```

## 4. Create new virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 5. Maybe problems ()

CMake Error at CMakeLists.txt:14 (project):
No CMAKE_CXX_COMPILER could be found.

```bash
sudo apt install -y build-essential cmake ninja-build python3-dev g++ gcc libnuma-dev
```

lower cmake version (in venv)

```bash
pip uninstall cmake -y
pip install cmake==3.30
```

## 6. Set up vLLM for CPUs

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Remote unnecesary dependency. We have manually installed them above.
sed -i '/^torch==2\.5\.1$/d' requirements-build.txt
sed -i '/^torch/d; /^torchvision/d' requirements-cpu.txt

# Install other dependency
pip install -r requirements-build.txt -r requirements-common.txt -r requirements-cpu.txt

# Install vLLM for CPUs
VLLM_TARGET_DEVICE=cpu pip install -e . --no-build-isolation

# Uninstall trition which is only needed for GPU execution
pip uninstall -y triton triton_pre_mlir
```

## 7. download llama and decompress

```bash
pip install gdown
gdown 1-ho7BOqPmpf7Bdub62kFj-F66x0bEJkA
tar -xzf Llama-3.2-1B-Instruct.tar.gz
```

数据集位置在 datasets/ShareGPT_Vicuna_unfiltered

## 8. Online test

```bash
python -m vllm.entrypoints.openai.api_server \
  --model models/Llama-3.2-1B-Instruct \
  --tokenizer models/Llama-3.2-1B-Instruct \
  --served-model-name meta-llama/Llama-3.2-1B-Instruct \
  --device cpu \
  --max-model-len 4096 \
  --max-num-seqs 4 \
  --block-size 16 \
  --host localhost \
  --port 8000 \
  --enable-prefix-caching
```

## 9. Set Up Client Simulator

```bash
python client_simulator.py \
	--trace datasets/ShareGPT_Vicuna_unfiltered/ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json \
  --model meta-llama/Llama-3.2-1B-Instruct \
	--rate 2.0 \
	--multi-turn
```

```bash
python client_simulator.py \
	--trace datasets/ShareGPT_Vicuna_unfiltered/cleaned_dataset_no_imsorry.json \
  --model meta-llama/Llama-3.2-1B-Instruct \
	--rate 2.0 \
	--multi-turn
```
## 10. Eviction Policy: How It Works and How to Test It

We added support for **selectable KV-cache eviction policies**:

* **LRU** (Least Recently Used) — default
* **FIFO** (First In, First Out)
* **LFU** (Least Frequently Used)

You can switch policies via an environment variable before launching the server.

### Available values:

| Policy   | Env Value |
| -------- | --------- |
| **LRU**  | `LRU`     |
| **FIFO** | `FIFO`    |
| **LFU**  | `LFU`     |


### How To Select an Eviction Policy

Before starting the vLLM server, export:

```bash
export VLLM_EVICTION_POLICY=LFU
```

(Change `LFU` to `FIFO` or `LRU` as needed.)


### My Testing setup (might be wrong / biased / just in case)

```bash
python -m vllm.entrypoints.openai.api_server \
  --model models/Llama-3.2-1B-Instruct \
  --tokenizer models/Llama-3.2-1B-Instruct \
  --served-model-name meta-llama/Llama-3.2-1B-Instruct \
  --device cpu \
  --max-model-len 4096 \
  --max-num-seqs 4 \
  --block-size 8 \
  --host localhost \
  --port 8000 \
  --enable-prefix-caching
```
* `--block-size 8` (smaller block → fewer total blocks → easier to force eviction)

```bash
python client_simulator.py \
  --url http://localhost:8000/v1/completions \
  --trace datasets/AgentBank/apps_cleaned.json \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --tokenizer models/Llama-3.2-1B-Instruct \
  --rate 50 \
  --max-requests 4000 \
  --multi-turn
```

### Where Eviction Debug Logs Go

The eviction policies write debug messages to:

```
evict.temp
```

Every time an eviction occurs, you will see entries like:

```
[LRUEvictor] Evicting block 42
[FIFOEvictor] Evicting block 17
[LFUEvictor] Evicting block 91
```

These logs help verify:

* which policy is active
* when eviction happens
* ordering differences between LRU/FIFO/LFU

### Where the Debug Logging Code Lives

If you need to modify or remove the debug logs, they are located in:

```
vllm-comp536/vllm/core/evictor.py
```

Added `_log("...")` calls in the following methods:

* `LRUEvictor.evict()`
* `FIFOEvictor.evict()`
* `LFUEvictor.evict()`
