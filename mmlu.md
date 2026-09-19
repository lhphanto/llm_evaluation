# MMLU (5-shot)

Dataset: [cais/mmlu](https://huggingface.co/datasets/cais/mmlu), 57 subjects, 14,042 test questions.
Setup, installation and common options are in [README.md](README.md).

## Reference numbers

From the [Meta-Llama-3-8B-Instruct model card](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
(Meta's internal eval library):

| Model | MMLU 5-shot |
|---|---|
| Llama 3 8B (base) | 66.6 |
| **Llama 3 8B Instruct** | **68.4** (macro avg; micro avg = 67.4) |

### A second, different reference: the BitNet authors' own numbers

If you're evaluating `HF1BitLLM/Llama3-8B-1.58-100B-tokens` (or a sibling checkpoint),
its own model card/[blog post](https://huggingface.co/blog/1_58_llm_extreme_quantization)
publishes a **different MMLU comparison**, evaluated 0-shot with
[LightEval](https://github.com/huggingface/lighteval) on a
[nanotron](https://github.com/huggingface/nanotron)-format checkpoint (HF's internal
training/eval stack — see [README.md](README.md) for whether you need it, short
answer: no):

| Model | MMLU (0-shot, LightEval) |
|---|---|
| Llama 7B (base) | 35.1 |
| Llama2 7B (base) | 45.3 |
| Llama3 8B (base) | 49.3 |
| **Llama3-8B-1.58-100B-tokens** | **41.6** |

Note every row here is a **base** model (no "-Instruct" anywhere) at **0-shot** — a
different, harder setting than Meta's 5-shot chat protocol above, and a different
reference model (base Llama3 8B, 49.3) than the Instruct card (68.4) this doc otherwise
compares to. The two tables are not measuring the same thing, even though both are
called "MMLU": this one puts the BitNet checkpoint's gap at a modest ~8 points, ours
(below) finds a ~32-40 point gap against the Instruct 5-shot card. See the Results
section below for why that isn't a contradiction, and which comparison is more
appropriate for this checkpoint (it was fine-tuned from Llama-3-8B-**Instruct**, so
neither reference is a perfect match).

## How Meta evaluated it, and why there are two modes

According to Meta's [eval_details.md](https://github.com/meta-llama/llama3/blob/main/eval_details.md),
the **instruct** models were evaluated differently from the base models:

- the 5 few-shot examples are given as a **user/assistant dialogue** (chat template),
- the model **generates** the answer letter (no loglikelihood scoring),
- the reported number is the **macro average** over the 57 subjects.

lm-eval's default `mmlu` task does something else: a plain-text prompt, the model
picks among A/B/C/D by loglikelihood, and the score is a micro average. So `run_mmlu.sh`
has two modes:

| Mode | lm-eval task | Prompting | Scoring | Compare to |
|---|---|---|---|---|
| `llama` (default) | `mmlu_llama` | chat template, few-shot as multi-turn | generate letter, exact match | **68.4 macro / 67.4 micro** |
| `standard` | `mmlu` | plain text, no chat template | loglikelihood over A/B/C/D | community/leaderboard numbers; expect a few points below the card |

In `llama` mode each question looks like this. The few-shot examples come first as
earlier user/assistant turns, and the final assistant turn is pre-filled:

```
<user>      Given the following question and four candidate answers (A, B, C and D), choose the best answer.
            Question: ...
            A. ...  B. ...  C. ...  D. ...
            Your response should end with "The best answer is [the_answer_letter]" ...
<assistant> The best answer is
```

The model then generates up to the next `.`, and that output is matched against the gold letter.

Datasets: the `standard` task reads `cais/mmlu`. The `mmlu_llama` task reads
`hails/mmlu_no_train`, which is `cais/mmlu` with the large `auxiliary_train` split
removed. Test questions and 5-shot `dev` examples are the same in both. Both download
automatically on first run.

## Running

```bash
# Main result, Meta protocol
BACKEND=vllm ./run_mmlu.sh llama     # or BACKEND=hf (default)

# Standard lm-eval MMLU (loglikelihood, no chat template)
BATCH_SIZE=auto:4 ./run_mmlu.sh standard

# Another model, e.g. the 1.58-bit BitNet fine-tune (HF backend only, see README)
MODEL=HF1BitLLM/Llama3-8B-1.58-100B-tokens ./run_mmlu.sh llama

# On Velda: submit any of the above as a batch job on an H100 (see README)
LAUNCHER="vbatch -P h100-1s" BACKEND=vllm ./run_mmlu.sh llama
```

Results go to `results/<mode>/<model>/results_<timestamp>.json`, plus per-question
`samples_*.jsonl` on full runs. Each run ends by printing a summary. To re-print it later:

```bash
.venv/bin/python summarize.py results/llama
```

Fixed settings: `--num_fewshot 5`, `dtype=bfloat16`, `--seed 1234`, greedy decoding
(from the task config), few-shot examples = the first 5 `dev` questions of each subject.

## Batch size

- **`llama` mode with vLLM:** keep `auto`. vLLM groups requests itself.
- **`llama` mode with HF:** `auto` works. Each answer is at most 10 tokens, so a
  fixed value such as `64` is also reasonable on an 80 GB GPU.
- **`standard` mode:** use `auto:4`. Memory use is dominated by the logits over the
  full prompt (vocabulary of 128k tokens), and 5-shot prompt lengths range from a few
  hundred to a few thousand tokens. lm-eval runs the longest prompts first, and
  `auto:4` re-measures the batch size 4 times as the prompts get shorter. Add
  `--max_batch_size 64` if auto-detection is unstable.

## Reading the summary

`summarize.py` reports two averages:

- **Macro:** the unweighted mean over the 57 subjects. This is what the model card reports (68.4).
- **Micro:** weighted by the number of questions per subject. This is lm-eval's own
  group score (card equivalent 67.4).

It also prints per-category (STEM, humanities, social sciences, other) micro averages,
and flags partial runs (`--limit` or a subset of subjects) as not comparable to the card.

## Notes on matching the card

- Expect `llama` mode to land close to the card but maybe not exactly on it.
  Meta used an internal library, and small prompt and tokenization differences can
  move MMLU by about ±1 point.
- `exact_match` uses the `strict_match` filter: the model must produce the letter
  right after `The best answer is`. If the score is unexpectedly low, check the
  `samples_*.jsonl` files for formatting failures. This matters more for models that
  follow the format less reliably, such as heavily quantized fine-tunes.
- The chat template comes from the model's tokenizer. The Llama 3 template inserts no
  system prompt by default; `--system_instruction "..."` adds one.

## Results

### HF1BitLLM/Llama3-8B-1.58-100B-tokens

1.58-bit (ternary) BitNet fine-tune of Llama 3 8B. Full 57-subject runs, HF backend, bfloat16.

| Mode | macro | micro | vs. card (macro 68.4) |
|---|---|---|---|
| `llama` | 28.75 | 29.22 | -39.65 |
| `standard` | 35.99 | 35.07 | -32.41 |

Category breakdown (micro):

| Mode | stem | humanities | social_sciences | other |
|---|---|---|---|---|
| `llama` | 29.69 | 26.82 | 32.27 | 29.35 |
| `standard` | 33.11 | 32.07 | 38.64 | 38.08 |

Both modes land far below the Llama 3 8B Instruct card, and not far above the 25%
random-guess baseline for 4-choice questions. `llama` mode scores lower than `standard`
here, the opposite of the usual pattern (see "Notes on matching the card" above) —
likely this quantized model follows the "The best answer is [X]" generation format
less reliably than it picks among A/B/C/D by loglikelihood, so it's worth checking the
`samples_mmlu_llama_*.jsonl` files for `exact_match` failures before trusting the gap.

Results files:
- `results/llama/HF1BitLLM__Llama3-8B-1.58-100B-tokens/results_2026-09-18T10-58-16.525698.json`
- `results/standard/HF1BitLLM__Llama3-8B-1.58-100B-tokens/results_2026-09-18T11-44-42.654947.json`

**Reconciling with the authors' own 41.6 number (see above).** Our -32 to -40 point
gap looks far more damning than the ~8-point gap (49.3 → 41.6) the model's own authors
report, but these are different measurements, not a discrepancy in either result:

- Their 41.6 is 0-shot, scored against **base** Llama models via LightEval on a
  nanotron checkpoint. Ours is 5-shot, scored against the **Instruct** card, via
  lm-eval on the standard `transformers`-loaded checkpoint (`BitNetHfQuantizer`) —
  same published weights, different harness and protocol.
- Their own baseline (base Llama3 8B, 49.3 at 0-shot) is itself far below Meta's
  official 5-shot numbers (66.6 base / 68.4 Instruct) — so their whole table lives on
  a lower absolute scale than the one this doc otherwise compares to. "41.6 vs 68.4"
  is not a fair read of either table; "49.3 → 41.6" (their comparison) and "68.4 →
  28.75/35.99" (ours) are each internally consistent but answer different questions.
- Neither reference is a perfect fit for this specific checkpoint: it was fine-tuned
  from Llama-3-8B-**Instruct**, but the authors' own table compares it only to
  **base** (non-instruct) Llama models, not to Instruct at a chat/instruct protocol —
  arguably as much of a mismatch as our choice to compare it to the Instruct card at
  full 5-shot fidelity.
- This does **not** retract the concrete, independently-verified problems found by
  inspecting raw generations elsewhere in this benchmark suite (GSM8K's runaway
  generation, MATH's repetition collapse, HumanEval's total non-engagement — see
  [gsm8k.md](gsm8k.md), [math.md](math.md), [humaneval.md](humaneval.md)). Those were
  confirmed from the actual generated text, independent of which reference number the
  aggregate score is measured against.
