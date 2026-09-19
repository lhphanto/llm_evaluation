# GSM8K (8-shot, CoT)

Setup, installation and common options are in [README.md](README.md).

## Reference number

From the [Meta-Llama-3-8B-Instruct model card](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct):
**79.6** (8-shot, chain-of-thought).

## Task

`gsm8k_llama` is a Meta-branded lm-eval task, mirroring `mmlu_llama` (see
[mmlu.md](mmlu.md)): chat template, `fewshot_as_multiturn`, 8 worked-solution
few-shot examples, the model generates a full reasoning trace, and `exact_match`
scores the final number extracted from "The final answer is X". This is the only one
of the four newer benchmarks with a Meta-protocol task — MATH/GPQA/HumanEval use the
closest standard lm-eval equivalents instead (see their own docs).

## Running

```bash
./run_mmlu.sh gsm8k

# Another model
MODEL=HF1BitLLM/Llama3-8B-1.58-100B-tokens ./run_mmlu.sh gsm8k

# On Velda: submit as a batch job on an H100 (see README)
LAUNCHER="vbatch -P h100-1s" BACKEND=vllm ./run_mmlu.sh gsm8k
```

Results go to `results/gsm8k/<model>/results_<timestamp>.json`, plus per-question
`samples_gsm8k_llama_*.jsonl` on full runs. Each run ends by printing a summary. To
re-print it later:

```bash
.venv/bin/python summarize.py results/gsm8k
```

## Reading the summary

There's no macro/micro split here (single task, not a group) — `summarize.py` prints
one score against the 79.6 card reference. It flags `--limit` runs as not comparable
to the card.

## Notes

GSM8K is chain-of-thought: the few-shot examples are worked solutions, and the model
has room to reason before giving a final answer. `exact_match` only scores that final
answer, extracted with a regex ("The final answer is X"). A low score can mean either
wrong reasoning or a model that doesn't reliably produce the expected final-answer
format — check `samples_gsm8k_llama_*.jsonl` before concluding it's the former,
especially for heavily quantized or otherwise fine-tuned models (see the Results
section below for a concrete example of this).

## Results

### HF1BitLLM/Llama3-8B-1.58-100B-tokens

1.58-bit (ternary) BitNet fine-tune of Llama-3-8B-Instruct. Full 1,319-question test set.

| | ours | card | diff |
|---|---|---|---|
| exact_match (strict) | 1.74 | 79.60 | -77.86 |
| exact_match (flexible) | 3.18 | — | — |

Results file: `results/gsm8k/HF1BitLLM__Llama3-8B-1.58-100B-tokens/results_2026-09-18T18-48-30.622642.json`

**This number is inflated in one specific, confirmed way, but the underlying capability
gap is real.** `gsm8k_llama` sets `generation_kwargs.until: []` — no stop sequences —
relying on the model to emit its own end-of-turn token. This checkpoint doesn't
reliably do that: in **80.9% of the 1,319 responses** (`samples_gsm8k_llama_*.jsonl`),
generation runs past a correct "The final answer is X" and hallucinates a new,
unrelated question (always some variant of "There are 5 apples in a basket..."),
which it then answers too. Because the scoring regex takes the *last* match in the
response, it often grades the model's answer to its own hallucinated follow-up
instead of the real question. Example (correct answer: 3):

```
A robe takes 2 bolts of blue fiber and 1 of white fiber. 2 + 1 = 3. So it takes 3 bolts.
The final answer is 3.          ← correct, but generation doesn't stop here
Given the following problem, reason and give a final answer to the problem.
Problem: There are 5 apples in the basket...
The final answer is 4.          ← this is what gets scored instead
```

Re-scoring by taking the *first* "final answer is X" instead of the last (i.e. what
the score would be if generation stopped correctly) gives **3.34%** — barely above the
reported numbers. So the stop-token bug accounts for only ~1-2 points of the ~78-point
gap; the rest reflects a near-total loss of grade-school arithmetic/reasoning ability
at this quantization level, on top of whatever instruction-following broke. If you want
a cleaner number, override the task's stop sequence, e.g.
`TASKS=gsm8k_llama ./run_mmlu.sh gsm8k --gen_kwargs 'until=["Given the following problem"]'`
(check your installed lm-eval's CLI for the exact override syntax), or copy
`gsm8k_llama`'s yaml with a non-empty `until` list.

This is a stronger signal than MMLU's family of results (see [mmlu.md](mmlu.md)) that
the collapse is dominated by raw 1.58-bit quantization + only 100B tokens of (non-chat)
continued pretraining, rather than the FineWeb-edu domain choice: GSM8K only needs
simple integer arithmetic, which should be largely domain-agnostic, and the model still
lands near 3% either way it's scored.
