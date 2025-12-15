# vLLM Prefix Sharing Project (Team 3 / COMP 536)

This repo is a **CPU-only, trace-driven vLLM simulator** for studying **prefix sharing (KV-cache reuse)** under different workloads and cache settings (block size, cache capacity/utilization) and **eviction policies** (LRU/FIFO/LFU).

Repo: https://github.com/Eipi-Tong/vllm-comp536

---

## 0) Requirements

- Ubuntu on EC2 recommended (tested on **m6i.xlarge + 50GB**)
- Python **3.12**
- Git
- (CPU-only) PyTorch wheels

---

## 1) Provision EC2 (Suggested)

- Instance: `m6i.xlarge`
- Storage: `50GB`
- OS: Ubuntu (22.04+ recommended)

---

## 2) Clone This Repo

```bash
git clone https://github.com/Eipi-Tong/vllm-comp536.git
cd vllm-comp536
````

> If you are required to start from upstream vLLM commit `94d545a...`, follow the instructor flow instead:
>
> * `git clone https://github.com/vllm-project/vllm.git vllm-test`
> * `git checkout 94d545a1a18e20ea8763a6760194589b8a3c9065`
> * then apply our changes (see `git log` / branches in this repo).

---

## 3) Create Python Virtual Env

```bash
sudo apt update
sudo apt -y install python3.12-venv

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
```

---

## 4) System Dependencies (if build fails)

If you see errors like **No CMAKE_CXX_COMPILER could be found**:

```bash
sudo apt install -y build-essential cmake ninja-build python3-dev g++ gcc libnuma-dev
```

If your cmake version is incompatible (rare), pin inside venv:

```bash
pip uninstall -y cmake
pip install cmake==3.30
```

---

## 5) Install vLLM (CPU Mode)

### 5.1 Install CPU PyTorch

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 5.2 Remove GPU-only dependency pins (repo expectation)

```bash
sed -i '/^torch==2\.5\.1$/d' requirements-build.txt
sed -i '/^torch/d; /^torchvision/d' requirements-cpu.txt
```

### 5.3 Install vLLM requirements + editable install

```bash
pip install -r requirements-build.txt -r requirements-common.txt -r requirements-cpu.txt
VLLM_TARGET_DEVICE=cpu pip install -e . --no-build-isolation
```

### 5.4 Remove Triton (GPU-only)

```bash
pip uninstall -y triton triton_pre_mlir
```

---

## 6) Download Model (Llama-3.2-1B-Instruct)

We use a prepackaged tarball (for course convenience):

```bash
pip install gdown
gdown 1-ho7BOqPmpf7Bdub62kFj-F66x0bEJkA
tar -xzf Llama-3.2-1B-Instruct.tar.gz
```

This should create:

```
models/Llama-3.2-1B-Instruct
```

---

## 7) Datasets

ShareGPT dataset path (expected):

```
datasets/ShareGPT_Vicuna_unfiltered/
```

Example cleaned traces referenced in our scripts:

* `datasets/ShareGPT_Vicuna_unfiltered/ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json`
* `datasets/ShareGPT_Vicuna_unfiltered/cleaned_dataset_no_imsorry.json`

Agent workload example:

* `datasets/AgentBank/apps_cleaned.json`

---

## 8) Run vLLM Server (Online Mode)

Start the OpenAI-compatible server:

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

---

## 9) Run Client Simulator (Replay Traces)

### Multi-turn ShareGPT

```bash
python client_simulator.py \
  --trace datasets/ShareGPT_Vicuna_unfiltered/ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --rate 2.0 \
  --multi-turn
```

Alternative trace:

```bash
python client_simulator.py \
  --trace datasets/ShareGPT_Vicuna_unfiltered/cleaned_dataset_no_imsorry.json \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --rate 2.0 \
  --multi-turn
```

> Tip: Use `--rate` to control load (Poisson arrivals).

---

## 10) KV-Cache Eviction Policies (LRU / FIFO / LFU)

We added **selectable eviction policies** via environment variable:

| Policy | Env Value |
| -----: | --------- |
|    LRU | `LRU`     |
|   FIFO | `FIFO`    |
|    LFU | `LFU`     |

Select policy before launching the server:

```bash
export VLLM_EVICTION_POLICY=LFU
```

Then start server as usual.

### Stress Test Setup (forces evictions more easily)

Server (smaller block size → more blocks / more churn):

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

Client (high rate + many requests):

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

---

## 11) Eviction Debug Logs

Eviction debug logs are written to:

```
evict.temp
```

Example lines:

```
[LRUEvictor] Evicting block 42
[FIFOEvictor] Evicting block 17
[LFUEvictor] Evicting block 91
```

Eviction logging code lives in:

```
vllm/core/evictor.py
```

---

## 12) Troubleshooting

### “Address already in use” (port 8000 busy)

```bash
lsof -i :8000
kill -9 <PID>
```

### “Your local changes would be overwritten by merge”

Stash only the file you want to keep, pull, then pop:

```bash
git stash push -m "keep local client_simulator" -- client_simulator.py
git pull
git stash pop
```

---

## 13) What’s Special About Our vLLM Fork

* **CPU-only simulator mode**: bypasses GPU execution while preserving vLLM’s scheduler + KV-cache behavior.
* **Trace replay**: returns tokens from pre-collected traces to emulate end-to-end serving.
* **Eviction policies**: LRU / FIFO / LFU selectable via `VLLM_EVICTION_POLICY`.
* **Client simulator**: async replay of multi-turn traces with controllable request rate.
