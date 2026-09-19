# GPQA (0-shot)

Setup, installation and common options are in [README.md](README.md).

## Reference number

From the [Meta-Llama-3-8B-Instruct model card](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct):
**34.2** (0-shot).

## Task

There's no Meta-branded GPQA task in lm-eval, so this uses `leaderboard_gpqa_main`
(Open LLM Leaderboard v2's protocol): plain text (no chat template), 0-shot, multiple
choice scored by loglikelihood over the 4 answer choices (`acc_norm`). Only ~448
questions (the "main" split), so run-to-run noise is a few points.

Because it's scored by loglikelihood rather than free-form generation, GPQA is immune
to the decoding pathologies (runaway generation, repetition loops) that can affect
GSM8K/MATH/HumanEval — see their docs for examples of those.

## Running

```bash
./run_mmlu.sh gpqa

# Another model
MODEL=HF1BitLLM/Llama3-8B-1.58-100B-tokens ./run_mmlu.sh gpqa

# On Velda: submit as a batch job on an H100 (see README)
LAUNCHER="vbatch -P h100-1s" BACKEND=vllm ./run_mmlu.sh gpqa
```

Results go to `results/gpqa/<model>/results_<timestamp>.json`, plus per-question
`samples_leaderboard_gpqa_main_*.jsonl` on full runs. Each run ends by printing a
summary. To re-print it later:

```bash
.venv/bin/python summarize.py results/gpqa
```

## Reading the summary

Single task, no macro/micro split — `summarize.py` prints one score (`acc_norm`)
against the 34.2 card reference. It flags `--limit` runs as not comparable to the card.

## Notes

GPQA is 0-shot and graduate-level; most 8B-class models (even at full precision) score
close to the 25% random-guess floor for its 4-choice questions, so it doesn't
discriminate well at this model scale — see the Results section below.

## Results

### HF1BitLLM/Llama3-8B-1.58-100B-tokens

1.58-bit (ternary) BitNet fine-tune of Llama-3-8B-Instruct. Full 448-question main split.

| | ours | card | diff |
|---|---|---|---|
| acc_norm | 24.78 (±2.04 stderr) | 34.20 | -9.42 |

Results file: `results/gpqa/HF1BitLLM__Llama3-8B-1.58-100B-tokens/results_2026-09-18T16-02-22.790957.json`

**Clean number, smallest gap of the four newer benchmarks, and the least informative.**
`leaderboard_gpqa_main` is scored by loglikelihood, so it's immune to the decoding
pathologies found in GSM8K (runaway generation past a missing stop sequence) and MATH
(repetition-loop collapse) — no per-sample bug-hunting needed here.

24.78% is statistically indistinguishable from the 25% random-guess floor for
4-choice questions (well within the ±2.04 stderr). But so is the card's own 34.2 —
GPQA is graduate-level science, deliberately hard for 8B-class models, so even the
full-precision Instruct model is only modestly above chance. That floor effect means
this benchmark doesn't discriminate well at this model scale regardless of
quantization: the 1.58-bit model did lose real ground (from "modestly better than
chance" to "at chance"), but MMLU, GSM8K, and MATH make a much stronger case for the
size of the capability loss than GPQA does.
