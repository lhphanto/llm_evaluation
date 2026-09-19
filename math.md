# MATH (4-shot, CoT)

Setup, installation and common options are in [README.md](README.md).

## Reference number

From the [Meta-Llama-3-8B-Instruct model card](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct):
**30.0** (4-shot, chain-of-thought).

## Task

There's no Meta-branded MATH task in lm-eval, so this uses `leaderboard_math_hard`
(Open LLM Leaderboard v2's protocol): plain text (no chat template), 4-shot worked
solutions, `generate_until`, answer extracted from `\boxed{}`. It's a group of 7
subject tasks (algebra, counting & probability, geometry, intermediate algebra, number
theory, prealgebra, precalculus).

This is the 1,324-question "hard" (Level 5) subset, **not** the full 5,000-question
MATH test set — treat scores as directionally comparable, not an exact reproduction of
the card's protocol.

## Running

```bash
./run_mmlu.sh math

# Another model
MODEL=HF1BitLLM/Llama3-8B-1.58-100B-tokens ./run_mmlu.sh math

# On Velda: submit as a batch job on an H100 (see README)
LAUNCHER="vbatch -P h100-1s" BACKEND=vllm ./run_mmlu.sh math
```

Results go to `results/math/<model>/results_<timestamp>.json`, plus per-question
`samples_leaderboard_math_*_hard_*.jsonl` (one file per subject) on full runs. Each
run ends by printing a summary. To re-print it later:

```bash
.venv/bin/python summarize.py results/math
```

## Reading the summary

`leaderboard_math_hard` is a group task: lm-eval already computes its own
size-weighted aggregate across the 7 subject subtasks (matching how the group's score
is normally reported), so `summarize.py` reads that aggregate directly rather than
recomputing a macro/micro split, and sums the effective question count across the 7
subtasks. It flags `--limit` runs as not comparable to the card.

## Notes

MATH is chain-of-thought, like GSM8K: the few-shot examples are worked solutions, and
`exact_match` only scores the final answer, extracted from `\boxed{...}`. A low score
can mean either wrong reasoning or a model that never produces a `\boxed{}` answer at
all — check `samples_leaderboard_math_*.jsonl` before concluding it's the former,
especially for heavily quantized or otherwise fine-tuned models (see the Results
section below for a concrete example of the latter).

## Results

### HF1BitLLM/Llama3-8B-1.58-100B-tokens

1.58-bit (ternary) BitNet fine-tune of Llama-3-8B-Instruct. Full 1,324-question "hard"
subset (7 subject groups).

| | ours | card | diff |
|---|---|---|---|
| exact_match | 1.13 | 30.00 | -28.87 |
| exact_match_original (strict, requires `\boxed{}`) | ~0 (nonzero in 1 of 7 subject groups) | — | — |

Results file: `results/math/HF1BitLLM__Llama3-8B-1.58-100B-tokens/results_2026-09-18T18-54-33.057029.json`

**Different failure mode than GSM8K, and more severe.** There's no stop-token bug here
(`leaderboard_math_hard`'s `until: ["Problem:"]` works — 0% of responses run on into a
hallucinated next problem). Instead, the model falls into a **degenerate repetition
loop**: it restates the same line verbatim over and over until `max_gen_toks` (1024)
cuts it off. This happens in **~80% of the 1,324 responses**
(`samples_leaderboard_math_*.jsonl`), and **none of the 1,324 responses ever contain
`\boxed{}`** — the format every few-shot example uses for its final answer. Example
(correct answer: 4):

```
We can multiply the first equation by $-\frac[1][x^3 + y^3]$ to obtain
$$4x - 4y^2 = -\frac[1][x^3 + y^3]$$
4x - 4y^2 = -\frac[1][x^3 + y^3]$$
4x - 4y^2 = -\frac[1][x^3 + y^3]$$
... (repeats until cut off, never reaches an answer)
```

The 1.13% "exact_match" figure is likely mostly or entirely noise rather than genuine
solutions: with no `\boxed{}` anywhere, the lenient `math-verify`-based checker falls
back to loosely extracted numbers/expressions from the response text, and both
"correct" examples inspected by hand had the target digit appear incidentally inside
the repeated garbage (e.g. target `4` matching a stray `4x - 4y^2` in the loop) rather
than as a deliberate final answer. The stricter `exact_match_original` metric (which
expects `\boxed{}`) is ~0 across all 7 subject groups, consistent with that read.

Net effect: on MATH, this checkpoint doesn't get to "wrong answer" — it doesn't
produce an answer at all. This is consistent with the GSM8K result (severe capability
loss, not just a formatting mismatch) but shows a distinct decoding pathology
(repetition collapse under greedy decoding) rather than a runaway-generation/stop-token
issue. Repetition loops like this are a known symptom of a badly degraded next-token
distribution — plausible here given the combination of 1.58-bit ternary weights and
only 100B tokens of recovery training.
