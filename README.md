# LLM evaluation: Llama-3-8B-Instruct on MMLU (5-shot)

Goal: reproduce the MMLU number on the
[Meta-Llama-3-8B-Instruct model card](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
with [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) on
[MMLU](https://huggingface.co/datasets/cais/mmlu).

| Model card (Meta internal eval) | MMLU 5-shot |
|---|---|
| Llama 3 8B (base) | 66.6 |
| **Llama 3 8B Instruct** | **68.4** (macro avg; micro avg = 67.4) |

## Files

| File | Purpose |
|---|---|
| `run_mmlu.sh` | Runs lm-eval MMLU 5-shot, in `llama` mode (Meta protocol) or `standard` mode |
| `summarize.py` | Reads the lm-eval results JSON and prints macro/micro accuracy plus the diff from the model card |
| `requirements.txt` | Pinned Python dependencies |

## How Meta evaluated it, and why there are two modes

According to Meta's [eval_details.md](https://github.com/meta-llama/llama3/blob/main/eval_details.md),
the **instruct** models were evaluated differently from the base models:

- the 5 few-shot examples are given as a **user/assistant dialogue** (chat template),
- the model **generates** the answer letter (no loglikelihood scoring),
- the reported number is the **macro average** over the 57 subjects.

lm-eval's default `mmlu` task does something else: a plain-text prompt, the model
picks among A/B/C/D by loglikelihood, and the score is a micro average. So:

| Mode | lm-eval task | Prompting | Scoring | Compare to |
|---|---|---|---|---|
| `llama` (default) | `mmlu_llama` | chat template, few-shot as multi-turn | generate letter, exact match | **68.4 macro / 67.4 micro** |
| `standard` | `mmlu` | plain text, no chat template | loglikelihood over A/B/C/D | community/leaderboard numbers; expect a few points below the card |

`summarize.py` reports both averages. lm-eval's own group score is the **micro**
average (weighted by the number of questions per subject). The **macro** average
(unweighted mean of the 57 subjects) is the one that matches the model card.

Datasets: the `standard` task reads `cais/mmlu`. The `mmlu_llama` task reads
`hails/mmlu_no_train`, which is `cais/mmlu` with the large `auxiliary_train` split
removed. Test questions and 5-shot `dev` examples are the same in both. Both download
automatically on first run.

## Setup (on the evaluation machine)

### 1. Hardware

- A CUDA GPU with **≥ 24 GB** memory (e.g. RTX 3090/4090, L4, A10G, A100, H100).
  The bf16 weights alone are ~16 GB.
- ~20 GB of free disk for the model and datasets in the Hugging Face cache.
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

Optional, but much faster for the generative `llama` mode:

```bash
.venv/bin/pip install "lm_eval[vllm]==0.4.13"
```

`run_mmlu.sh` uses `.venv/bin/python` when it exists and falls back to `python3` on
`PATH`, so a conda env with the same packages also works.

### 3. Hugging Face access (the model is gated)

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

## Running

```bash
# Main result, Meta protocol (HF transformers backend)
./run_mmlu.sh llama

# Same with vLLM (recommended if installed)
BACKEND=vllm ./run_mmlu.sh llama

# Standard lm-eval MMLU (loglikelihood, no chat template)
./run_mmlu.sh standard
```

Environment overrides:

| Variable | Default | |
|---|---|---|
| `MODEL` | `meta-llama/Meta-Llama-3-8B-Instruct` | any HF model id or local path |
| `BACKEND` | `hf` | `hf` or `vllm` |
| `DEVICE` | auto (`cuda:0` → `mps` → `cpu`) | |
| `BATCH_SIZE` | `auto` | lower it (e.g. `8`) if you hit OOM |
| `OUT_DIR` | `results` | results go to `$OUT_DIR/<mode>/<model>/results_<timestamp>.json` |
| `TASKS` | `mmlu_llama` / `mmlu` | override, e.g. a single subject such as `mmlu_llama_anatomy` |

Extra arguments are passed through to `lm_eval run`, e.g. `./run_mmlu.sh llama --limit 5`.
Full runs also save per-question outputs (`--log_samples`), which helps debug wrong
answers. lm-eval can't log samples together with `--limit`, so limited runs skip them.

Fixed settings: `--num_fewshot 5`, `dtype=bfloat16`, `--seed 1234`, greedy decoding
(from the task config), few-shot examples = the first 5 `dev` questions of each subject.

To re-print the summary of an existing run:

```bash
.venv/bin/python summarize.py results/llama
```

## Smoke test (any machine, a few minutes)

This checks the pipeline end to end with a 135M model and 2 questions per subject.
The scores mean nothing.

```bash
MODEL=HuggingFaceTB/SmolLM2-135M-Instruct OUT_DIR=smoke BATCH_SIZE=8 ./run_mmlu.sh llama --limit 2
```

## Notes on matching the card

- Expect the `llama` mode to land close to the card but maybe not exactly on it.
  Meta used an internal library, and small prompt and tokenization differences can
  move MMLU by about ±1 point.
- `exact_match` uses the `strict_match` filter: the model must produce the letter
  right after `The best answer is`. If the score is unexpectedly low, check the
  `samples_*.jsonl` files for formatting failures.
- The chat template comes from the model's tokenizer. The Llama 3 template inserts no
  system prompt by default; `--system_instruction "..."` adds one.
