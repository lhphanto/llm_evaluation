# LLM evaluation

Reproducing published benchmark results with
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) (lm-eval).

## Benchmarks

| Benchmark | Setting | Doc | Script |
|---|---|---|---|
| MMLU | 5-shot | [mmlu.md](mmlu.md) | `./run_mmlu.sh llama\|standard` |
| GSM8K | 8-shot, CoT | [gsm8k.md](gsm8k.md) | `./run_mmlu.sh gsm8k` |
| MATH | 4-shot, CoT | [math.md](math.md) | `./run_mmlu.sh math` |
| GPQA | 0-shot | [gpqa.md](gpqa.md) | `./run_mmlu.sh gpqa` |
| HumanEval | 0-shot | [humaneval.md](humaneval.md) | `./run_mmlu.sh humaneval` |

## Results: HF1BitLLM/Llama3-8B-1.58-100B-tokens

1.58-bit (ternary) BitNet fine-tune of Llama-3-8B-Instruct, continued-pretrained on
100B tokens of a FineWeb-edu subset (see [BitNet model notes](#bitnet-model-notes)).
Full test sets throughout; per-benchmark detail (raw generations, failure-mode
analysis, results file paths) is in the linked doc.

| Benchmark | Setting | Ours | Card | Diff | Doc |
|---|---|---|---|---|---|
| MMLU | 5-shot, `standard` | 35.99 (macro) | 68.40 | -32.41 | [mmlu.md](mmlu.md) |
| MMLU | 5-shot, `llama` | 28.75 (macro) | 68.40 | -39.65 | [mmlu.md](mmlu.md) |
| MMLU | 0-shot, `standard` | 37.24 (macro) | 41.60\* | -4.36 | [mmlu.md](mmlu.md) |
| GSM8K | 8-shot, CoT | 1.74 | 79.60 | -77.86 | [gsm8k.md](gsm8k.md) |
| MATH | 4-shot, CoT | 1.13 | 30.00 | -28.87 | [math.md](math.md) |
| GPQA | 0-shot | 24.78 | 34.20 | -9.42 | [gpqa.md](gpqa.md) |
| HumanEval | 0-shot, pass@1 | 0.00 | 62.20 | -62.20 | [humaneval.md](humaneval.md) |

"Card" is Meta's Llama-3-8B-**Instruct** model card, 5-shot (68.4 for MMLU, etc.) for
every row except the one marked \*: **this checkpoint's own authors publish a
different MMLU comparison** (0-shot, against **base** — not Instruct — Llama models,
via LightEval on a nanotron checkpoint): 49.3 → 41.6. We reran MMLU at 0-shot
ourselves to check, and landed within ~4-5 points of their number (37.24 vs. 41.6) —
much closer than the ~32-40 point gaps above. That confirms the huge gaps elsewhere in
this table are overwhelmingly a **protocol/reference-model artifact** (5-shot
Instruct-chat vs. 0-shot base), not evidence that this checkpoint or our evaluation of
it is broken — see
[mmlu.md](mmlu.md#a-second-different-reference-the-bitnet-authors-own-numbers) for the
full reference table and the 0-shot reproduction. It doesn't change the GSM8K/MATH/
HumanEval findings below, which come from inspecting actual generated text, not from
picking a reference number.

## Known issues with the 1.58-bit checkpoint

Investigating why the scores above are so far below the card surfaced several
distinct, confirmed problems, not just "quantization makes it worse" — each is
detailed with sample generations in its benchmark's doc:

- **Two legitimate but very different MMLU reference points — confirmed, not just
  argued.** Comparing this checkpoint to the Instruct card at 5-shot (what this repo
  does throughout) gives a ~32-40 point gap; the model's own authors instead compare
  it to base (non-instruct) Llama models at 0-shot and find only an ~8-point gap (49.3
  → 41.6). We didn't just note the discrepancy — we reran MMLU ourselves at 0-shot and
  landed within ~4-5 points of their 41.6 (37.24 vs. 41.6), confirming the gap is
  mostly about which protocol/reference you pick, not a sign that our evaluation setup
  or this checkpoint's weights are somehow different from what the authors tested.
- **Lost instruction-following, not just lost knowledge.** The base model is
  Llama-3-8B-**Instruct**, but the continued pretraining was on 100B tokens of plain
  FineWeb-edu text (no chat-formatted data). That plausibly eroded chat-template/
  instruction-following behavior on top of the raw precision loss from ternary
  weights. It shows up concretely as: MMLU's `llama` mode (which depends on the model
  reliably following "The best answer is [X]") scoring *lower* than `standard` mode
  (loglikelihood ranking, format-insensitive) — the reverse of the usual pattern — and
  HumanEval, where the model never once attempts to write code (see below).
- **GSM8K: runaway generation.** `gsm8k_llama` sets no stop sequence, relying on the
  model's own end-of-turn token. In 80.9% of responses this model doesn't stop after
  answering — it hallucinates a new, unrelated question and answers that instead,
  which the scoring regex (last-match) grades in place of the real answer. Re-scoring
  on the first answer instead of the last only recovers ~1.6 points, so this explains
  a small slice of the gap, not most of it. See [gsm8k.md](gsm8k.md).
- **MATH: degenerate repetition loops.** ~80% of responses get stuck repeating the
  same line verbatim until the token budget runs out, and 0 of 1,324 responses ever
  produce a `\boxed{}` answer at all. See [math.md](math.md).
- **HumanEval: the model never attempts the task.** 100% of extracted completions are
  byte-identical to the bare prompt (no function body added, 0% contain `return`);
  instead of code, generation is repetitive chat-style filler like "The function has
  passed the tests." See [humaneval.md](humaneval.md).
- **GPQA is a poor discriminator here.** Scored by loglikelihood (not generation), so
  none of the above pathologies apply — but both this model (24.78%) and the
  full-precision card (34.2%) sit close to the 25% random-guess floor for 4-choice
  questions, so it doesn't show much regardless of quantization. See [gpqa.md](gpqa.md).

Net read: taken against the Instruct 5-shot card, MMLU, GSM8K, and MATH all point to
severe capability loss from 1.58-bit quantization + only 100B tokens of (non-chat)
recovery training, rather than the FineWeb-edu domain choice being the main cause —
GSM8K in particular only needs simple arithmetic, which should be largely
domain-agnostic, and still collapses to ~3%. Judged instead against the authors' own
0-shot/base-model comparison, the picture is milder (~8 points on MMLU) — but that
gentler framing doesn't extend to GSM8K, MATH, or HumanEval, where we didn't just
compute a lower aggregate score, we found the model failing to produce a usable answer
at all in the majority of cases (runaway generation, repetition loops, or no attempt),
confirmed by hand from the raw generations rather than inferred from any score.

## Models

| Model | Notes |
|---|---|
| [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) | Default. Gated: accept the license first (see below). ~16 GB of bf16 weights. |
| [HF1BitLLM/Llama3-8B-1.58-100B-tokens](https://huggingface.co/HF1BitLLM/Llama3-8B-1.58-100B-tokens) | BitNet 1.58-bit fine-tune of Llama-3-8B-Instruct (100B FineWeb-edu tokens). ~3.8 GB. See [BitNet notes](#bitnet-model-notes). |

Pick the model with `MODEL=<hf id or local path>`.

## Files

| File | Purpose |
|---|---|
| `run_mmlu.sh` | Runner for all benchmarks below (see [mmlu.md](mmlu.md), [gsm8k.md](gsm8k.md), [math.md](math.md), [gpqa.md](gpqa.md), [humaneval.md](humaneval.md)) |
| `summarize.py` | Reads a results JSON and prints its score(s) plus the diff from the model card (macro/micro + category breakdown for MMLU, a single score for the others) |
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
.venv/bin/pip install -r requirements.txt       # lm_eval[hf,math]==0.4.13 (pulls torch + transformers)
```

If the default torch wheel doesn't match your CUDA driver, install the matching one
first from [pytorch.org](https://pytorch.org/get-started/locally/), then run the command above.

**vLLM (optional, recommended on CUDA).** It is much faster for generative tasks, but
it is not in `requirements.txt` because it only installs on Linux + CUDA. `BACKEND=vllm`
fails unless it is installed in the same environment. Install all extras in one
command, so pip picks `torch`/`transformers` versions that satisfy lm-eval and vLLM together:

```bash
.venv/bin/pip install "lm_eval[hf,vllm,math]==0.4.13"
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
| `DEVICE` | auto (`cuda:0` → `mps` → `cpu`); `cuda:0` when `LAUNCHER` is set | |
| `BATCH_SIZE` | `auto` (`8` on Apple `mps`) | also `auto:N`, or a fixed number if you hit OOM |
| `OUT_DIR` | `results` | results go to `$OUT_DIR/<benchmark>/<model>/results_<timestamp>.json` |
| `TASKS` | per benchmark | override the lm-eval task list, e.g. a single subject |
| `NUM_FEWSHOT` | per benchmark (5 for MMLU, 8 GSM8K, 4 MATH, 0 GPQA/HumanEval) | override the shot count, e.g. `NUM_FEWSHOT=0` for a 0-shot MMLU run |
| `PYTHON` | see [Python environment](#2-python-environment) | interpreter to use |
| `LAUNCHER` | none | job-submit prefix, e.g. `vbatch -P h100-1s` (see [Velda](#running-on-velda)) |

Extra arguments are passed through to `lm_eval run`, e.g. `--limit 5` or
`--max_batch_size 64`. Full runs also save per-question outputs (`--log_samples`),
which helps debug wrong answers. lm-eval can't log samples together with `--limit`,
so limited runs skip them.

**`./run_mmlu.sh humaneval` executes the model's generated code** on the evaluation
machine to check it against the tests (that's what `--confirm_run_unsafe_code` opts
into). Only run it against models you trust, ideally in a sandbox/VM.

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
  With a `LAUNCHER`, `DEVICE` defaults to `cuda:0`, so a job that can't see the GPU
  fails instead of silently running on CPU (set `DEVICE=...` to override).
- `summarize.py` has to run after the eval finishes, so it runs inside the same job.

Inside the job the script is re-run as
`env PYTHON=<abs path> MODEL=... BACKEND=... OUT_DIR=... DEVICE=cuda:0 [BATCH_SIZE/TASKS/HF_HOME] bash run_mmlu.sh <mode> <args>`.
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
MODEL=HF1BitLLM/Llama3-8B-1.58-100B-tokens ./run_mmlu.sh gsm8k
```

## Smoke test (any machine, a few minutes)

This checks the pipeline end to end with a 135M model and 2 questions per subject.
The scores mean nothing.

```bash
MODEL=HuggingFaceTB/SmolLM2-135M-Instruct OUT_DIR=smoke BATCH_SIZE=8 ./run_mmlu.sh llama --limit 2
```
