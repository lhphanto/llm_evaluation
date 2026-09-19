# HumanEval (0-shot)

Setup, installation and common options are in [README.md](README.md).

## Reference number

From the [Meta-Llama-3-8B-Instruct model card](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct):
**62.2** (0-shot, pass@1).

## Task

`humaneval_instruct`: 0-shot, instruction-style prompt (no chat template), the model
generates code, scored by `pass@1` (does the generated function pass the problem's
unit tests). Standard lm-eval task — no Meta-branded HumanEval task exists.

**Warning — this executes untrusted code.** `humaneval_instruct` checks each generated
solution by actually running it. Two separate libraries gate this, and
`./run_mmlu.sh humaneval` opts into both: lm-eval's own `--confirm_run_unsafe_code`,
and HF `evaluate`'s `code_eval` metric, which refuses to run at all unless
`HF_ALLOW_CODE_EVAL=1` is set (the script exports it for this mode only). Neither flag
sandboxes the execution — it just runs in the eval process with no isolation. Only
evaluate models you trust with this task, ideally in a sandbox or disposable VM.

## Running

```bash
./run_mmlu.sh humaneval        # runs model-generated code locally, see warning above

# Another model
MODEL=HF1BitLLM/Llama3-8B-1.58-100B-tokens ./run_mmlu.sh humaneval

# On Velda: submit as a batch job on an H100 (see README)
LAUNCHER="vbatch -P h100-1s" BACKEND=vllm ./run_mmlu.sh humaneval
```

Results go to `results/humaneval/<model>/results_<timestamp>.json`, plus per-question
`samples_humaneval_instruct_*.jsonl` on full runs. Each run ends by printing a summary.
To re-print it later:

```bash
.venv/bin/python summarize.py results/humaneval
```

## Reading the summary

Single task, no macro/micro split — `summarize.py` prints one score (`pass@1`) against
the 62.2 card reference. It flags `--limit` runs as not comparable to the card.

## Notes

HumanEval needs exact, runnable syntax, which tends to be one of the more fragile
capabilities under aggressive quantization or narrow continued-pretraining corpora
that contain little or no code — see the Results section below for what that looks
like when it goes all the way to zero.

## Results

### HF1BitLLM/Llama3-8B-1.58-100B-tokens

1.58-bit (ternary) BitNet fine-tune of Llama-3-8B-Instruct. Full 164-problem set.

| | ours | card | diff |
|---|---|---|---|
| pass@1 | 0.00 | 62.20 | -62.20 |

Results file: `results/humaneval/HF1BitLLM__Llama3-8B-1.58-100B-tokens/results_2026-09-18T16-29-14.319738.json`

**Total collapse, and a third distinct failure mode.** Unlike GSM8K (runaway
generation past a missing stop sequence) and MATH (repetition-loop, never reaching an
answer), here the model doesn't attempt the task at all. Across all 164 problems
(`samples_humaneval_instruct_*.jsonl`):

- **100%** of the extracted completions are byte-identical to the bare prompt/docstring
  stub — no function body was added.
- **0%** contain a `return` statement anywhere.

Instead of writing code, its generation is chatty acknowledgment filler that gets
stripped out by the code-extraction filter, e.g.:

```
The function has passed the tests.The function has passed the tests.
The function has passed the tests.
The function has passed the tests.
```

(repeated until cut off), or in a handful of cases just an immediate closing
` ``` ` with nothing inside.

So this isn't "wrong code" (as a capability-only failure would show) — the model
treats the prompt as something to acknowledge rather than something to complete,
declaring a function that was never written "complete" and "passing." That's
consistent with the pattern across all four benchmarks: whatever instruction-following
behavior made the base Instruct checkpoint actually do the task it's given appears to
have eroded during the FineWeb-edu-only (non-chat) continued pretraining, compounding
the raw 1.58-bit precision loss. Combined with the repetitive filler text, it also
echoes MATH's decoding-collapse pattern rather than adding new information about
coding ability specifically — there's no code being generated to judge.
