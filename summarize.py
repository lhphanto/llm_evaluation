"""Summarize an lm-eval run and compare it with the Llama-3-8B-Instruct model card.

For MMLU, lm-eval's group score is a *micro* average (weighted by #questions per
subject), while Meta's model card reports the *macro* average over the 57 subjects
(see github.com/meta-llama/llama3/blob/main/eval_details.md), so both are shown.
Other benchmarks (GSM8K, MATH, GPQA, HumanEval) report a single score.

Usage: python summarize.py <results dir or results_*.json>
"""

import glob
import json
import os
import sys

# Meta-Llama-3-8B-Instruct card numbers (see README.md for per-benchmark protocol notes).
MMLU_REFERENCE = {"macro": 68.4, "micro": 67.4}  # macro: model card; micro: eval_details.md
CARD_REFERENCE = {
    "gsm8k_llama": 79.6,  # GSM8K, 8-shot CoT
    "leaderboard_math_hard": 30.0,  # MATH, 4-shot CoT (Level-5/"hard" subset)
    "leaderboard_gpqa_main": 34.2,  # GPQA, 0-shot
    "humaneval": 62.2,  # HumanEval, 0-shot pass@1
    "humaneval_instruct": 62.2,
}
CATEGORIES = ["stem", "humanities", "social_sciences", "other"]

# Preferred metric name, then preferred filter/variant, when a task entry has more
# than one (e.g. MATH reports both exact_match and exact_match_original).
METRIC_PRIORITY = ["exact_match", "acc_norm", "acc", "pass@1"]
FILTER_PRIORITY = ["strict_match", "strict-match", "none", "flexible-extract", "flexible_extract"]


def find_results_file(path):
    if os.path.isfile(path):
        return path
    files = glob.glob(os.path.join(path, "**", "results_*.json"), recursive=True)
    if not files:
        sys.exit(f"No results_*.json under {path}")
    return max(files, key=os.path.getmtime)


def main_metric(entry):
    """Return (key, value) of the accuracy-like metric of a task entry."""
    for metric in METRIC_PRIORITY:
        candidates = {k: v for k, v in entry.items() if "," in k and k.split(",")[0] == metric}
        if not candidates:
            continue
        for filt in FILTER_PRIORITY:
            key = f"{metric},{filt}"
            if key in candidates:
                return key, candidates[key]
        return next(iter(candidates.items()))
    raise KeyError(f"No known metric in {sorted(entry)}")


def find_root(results, groups):
    """The top-level task/group name: one not listed as a subtask of another group."""
    root = next((g for g in groups if not any(g in subs for subs in groups.values())), None)
    return root or next(iter(results))


def summarize_mmlu(data, fname, root):
    results, groups = data["results"], data.get("group_subtasks", {})
    n_samples = data.get("n-samples", {})

    subjects = {t: r for t, r in results.items() if t not in groups}
    metric_key, _ = main_metric(next(iter(subjects.values())))

    n = {t: n_samples.get(t, {}).get("effective", 1) for t in subjects}
    n_docs = sum(n.values())
    macro = 100 * sum(main_metric(r)[1] for r in subjects.values()) / len(subjects)
    micro = 100 * sum(main_metric(r)[1] * n[t] for t, r in subjects.items()) / n_docs

    cfg = data.get("config", {})
    print(f"\nResults file : {fname}")
    print(f"Model        : {cfg.get('model_args')}")
    print(f"Task / metric: {root} / {metric_key}")
    print(f"Chat template: {data.get('chat_template') is not None}, "
          f"fewshot_as_multiturn: {data.get('fewshot_as_multiturn')}")
    print(f"Subjects     : {len(subjects)}   questions evaluated: {n_docs}")
    if cfg.get("limit") is not None:
        print(f"  NOTE: partial run (--limit {cfg['limit']}), NOT comparable to the model card.")
    elif n_docs < 14042:
        print("  NOTE: partial run (subset of subjects), NOT comparable to the model card.")

    print("\nCategory (micro)")
    for cat in CATEGORIES:
        g = next((g for g in groups if g.endswith(cat)), None)
        if g in results:
            print(f"  {cat:<16} {100 * main_metric(results[g])[1]:6.2f}")

    print(f"\n{'':16} {'ours':>7} {'card':>7} {'diff':>7}")
    for name, ours in (("macro (card)", macro), ("micro", micro)):
        ref = MMLU_REFERENCE[name.split()[0]]
        print(f"  {name:<14} {ours:7.2f} {ref:7.2f} {ours - ref:+7.2f}")


def summarize_generic(data, fname, root):
    results, groups = data["results"], data.get("group_subtasks", {})
    n_samples = data.get("n-samples", {})

    metric_key, value = main_metric(results[root])
    leaves = groups.get(root) or [root]
    n_docs = sum(n_samples.get(t, {}).get("effective", 1) for t in leaves)

    cfg = data.get("config", {})
    print(f"\nResults file : {fname}")
    print(f"Model        : {cfg.get('model_args')}")
    print(f"Task / metric: {root} / {metric_key}")
    print(f"Chat template: {data.get('chat_template') is not None}, "
          f"fewshot_as_multiturn: {data.get('fewshot_as_multiturn')}")
    print(f"Questions evaluated: {n_docs}" + (f"   ({len(leaves)} subtasks)" if len(leaves) > 1 else ""))
    if cfg.get("limit") is not None:
        print(f"  NOTE: partial run (--limit {cfg['limit']}), NOT comparable to the model card.")

    score = 100 * value
    ref = CARD_REFERENCE.get(root)
    print(f"\n{'':22} {'ours':>7} {'card':>7} {'diff':>7}")
    if ref is not None:
        print(f"  {root:<20} {score:7.2f} {ref:7.2f} {score - ref:+7.2f}")
    else:
        print(f"  {root:<20} {score:7.2f}   (no card reference for this task)")


def main(path):
    fname = find_results_file(path)
    with open(fname) as f:
        data = json.load(f)
    results, groups = data["results"], data.get("group_subtasks", {})
    if not results:
        sys.exit("No results found")

    root = find_root(results, groups)
    if root.startswith("mmlu"):
        summarize_mmlu(data, fname, root)
    else:
        summarize_generic(data, fname, root)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
