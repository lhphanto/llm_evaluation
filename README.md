# LLM evaluation

Reproducing published benchmark results with
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) (lm-eval).

## Benchmarks

| Benchmark | Setting | Doc | Script |
|---|---|---|---|
| MMLU | 5-shot | [mmlu.md](mmlu.md) | `run_mmlu.sh` |
| GPQA | 0-shot | planned | |

## Models

| Model | Notes |
|---|---|
| [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) | Default. Gated: accept the license first (see below). ~16 GB of bf16 weights. |
| [HF1BitLLM/Llama3-8B-1.58-100B-tokens](https://huggingface.co/HF1BitLLM/Llama3-8B-1.58-100B-tokens) | BitNet 1.58-bit fine-tune of Llama-3-8B-Instruct (100B FineWeb-edu tokens). ~3.8 GB. See [BitNet notes](#bitnet-model-notes). |

Pick the model with `MODEL=<hf id or local path>`.

## Files

| File | Purpose |
|---|---|
| `run_mmlu.sh` | MMLU 5-shot runner (see [mmlu.md](mmlu.md)) |
| `summarize.py` | Reads an MMLU results JSON and prints macro/micro accuracy plus the diff from the model card |
| `requirements.txt` | Pinned Python dependencies |

## Setup (on the evaluation machine)

### 1. Hardware

- A CUDA GPU with **≥ 24 GB** memory (e.g. RTX 3090/4090, L4, A10G, A100, H100).
  The bf16 Llama-3-8B weights alone are ~16 GB.
- ~25 GB of free disk for models and datasets in the Hugging Face cache.
  Set `HF_HOME=/path/with/space` to move the cache.
- An 8 GB laptop (e.g. an M2 MacBook) can only run the tiny-model smoke test below.

### 2. Python environment

Tested with Python 3.12, `lm_eval==0.4.13`, `transformers==5.17.0`, `torch==2.14.0`.

```bash
cd llm_evaluation
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt       # lm_eval[hf]==0.4.13 (pulls torch + transformers)
```

If the default torch wheel doesn't match your CUDA driver, install the matching one
first from [pytorch.org](https://pytorch.org/get-started/locally/), then run the command above.

**vLLM (optional, recommended on CUDA).** It is much faster for generative tasks, but
it is not in `requirements.txt` because it only installs on Linux + CUDA. `BACKEND=vllm`
fails unless it is installed in the same environment. Install both extras in one
command, so pip picks `torch`/`transformers` versions that satisfy lm-eval and vLLM together:

```bash
.venv/bin/pip install "lm_eval[hf,vllm]==0.4.13"
.venv/bin/python -c "import vllm; print(vllm.__version__)"    # check
```

Any environment name works (the commands above use `.venv`). The scripts pick the Python
interpreter in this order:

1. `$PYTHON`, if set (e.g. `PYTHON=~/envs/llm_eval/bin/python ./run_mmlu.sh llama`)
2. the currently activated venv or conda env (conda `base` is ignored)
3. `./.venv/bin/python`, if it exists
4. `python3` on `PATH`

They exit early with a clear message if `lm_eval` isn't installed in the chosen interpreter.

### 3. Hugging Face access

Llama 3 is gated:

1. On the [model page](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct),
   accept the Llama 3 license with your HF account.
2. Log in on the evaluation machine:

```bash
.venv/bin/hf auth login          # or: export HF_TOKEN=hf_xxx
```

3. Check access:

```bash
.venv/bin/python -c "from huggingface_hub import HfApi; HfApi().model_info('meta-llama/Meta-Llama-3-8B-Instruct'); print('ok')"
```

## Common options

The run scripts take these environment variables:

| Variable | Default | |
|---|---|---|
| `MODEL` | `meta-llama/Meta-Llama-3-8B-Instruct` | any HF model id or local path |
| `BACKEND` | `hf` | `hf` or `vllm` |
| `DEVICE` | auto (`cuda:0` → `mps` → `cpu`) | |
| `BATCH_SIZE` | `auto` (`8` on Apple `mps`) | also `auto:N`, or a fixed number if you hit OOM |
| `OUT_DIR` | `results` | results go to `$OUT_DIR/<mode>/<model>/results_<timestamp>.json` |
| `TASKS` | per benchmark | override the lm-eval task list, e.g. a single subject |
| `PYTHON` | see [Python environment](#2-python-environment) | interpreter to use |
| `LAUNCHER` | none | job-submit prefix, e.g. `vbatch -P h100-1s` (see [Velda](#running-on-velda)) |

Extra arguments are passed through to `lm_eval run`, e.g. `--limit 5` or
`--max_batch_size 64`. Full runs also save per-question outputs (`--log_samples`),
which helps debug wrong answers. lm-eval can't log samples together with `--limit`,
so limited runs skip them.

## Running on Velda

Set `LAUNCHER` to the submit command, and the script submits **itself** as a batch job:

```bash
source llm_eval/bin/activate          # or set PYTHON=/path/to/llm_eval/bin/python
LAUNCHER="vbatch -P h100-1s" ./run_mmlu.sh llama
LAUNCHER="vbatch -P h100-1s" MODEL=HF1BitLLM/Llama3-8B-1.58-100B-tokens ./run_mmlu.sh llama
```

It wraps the whole script rather than only the `python -m lm_eval` line, because
`vbatch` (`velda run --batch`) queues the job and returns immediately:

- GPU detection has to run on the H100 node, not on the machine you submit from.
- `summarize.py` has to run after the eval finishes, so it runs inside the same job.

Inside the job the script is re-run as
`env PYTHON=<abs path> MODEL=... BACKEND=... OUT_DIR=... [DEVICE/BATCH_SIZE/TASKS/HF_HOME] bash run_mmlu.sh <mode> <args>`.
Settings are passed explicitly because a batch job may not inherit your shell's
environment. The job uses the instance's filesystem, so the venv, the HF cache and
login token (`~/.cache/huggingface/token`), and `results/` are shared with your
session. `HF_TOKEN` is deliberately not forwarded, to keep it out of the job's command
line, so log in with `hf auth login` instead of relying on that variable.

Follow the job with `velda task log <task-id>` / `velda task watch <task-id>`.
The summary is printed at the end of the job log, and `.venv/bin/python summarize.py results/llama`
(or with your env's python) re-prints it from your session.

## BitNet model notes

`HF1BitLLM/Llama3-8B-1.58-100B-tokens` stores ternary weights packed 4 per byte
(`"quantization_config": {"quant_method": "bitnet"}`).

- **No special transformers build needed.** The model card says to install a
  transformers pull request from 2024. BitNet support has since been merged, and the
  pinned `transformers` loads it as is (`BitNetHfQuantizer`, which needs `accelerate`,
  already installed by lm-eval).
- **Tokenizer:** the repo includes the Llama 3 tokenizer and chat template, so no
  separate `tokenizer=` argument is needed, and it isn't gated.
- **Use `BACKEND=hf`.** vLLM is not expected to load the `bitnet` quantization format.
- **Use a GPU.** Weights are unpacked on the fly. transformers supports it on CPU
  but warns that inference is slow there.
- **Speed:** transformers' BitNet layers unpack to bf16 for the matmul. They save
  memory but aren't faster than the bf16 Llama; the fast 1-bit kernels live in
  [microsoft/BitNet](https://github.com/microsoft/BitNet) (bitnet.cpp), which lm-eval
  doesn't drive.

```bash
MODEL=HF1BitLLM/Llama3-8B-1.58-100B-tokens ./run_mmlu.sh llama
MODEL=HF1BitLLM/Llama3-8B-1.58-100B-tokens BATCH_SIZE=auto:4 ./run_mmlu.sh standard
```

## Smoke test (any machine, a few minutes)

This checks the pipeline end to end with a 135M model and 2 questions per subject.
The scores mean nothing.

```bash
MODEL=HuggingFaceTB/SmolLM2-135M-Instruct OUT_DIR=smoke BATCH_SIZE=8 ./run_mmlu.sh llama --limit 2
```
