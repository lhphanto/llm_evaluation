"""Summarize an lm-eval MMLU run and compare it with the Llama 3 model card.

lm-eval's group score for MMLU is a *micro* average (weighted by #questions per
subject). Meta's model card reports the *macro* average over the 57 subjects
(see github.com/meta-llama/llama3/blob/main/eval_details.md), so both are shown.

Usage: python summarize.py <results dir or results_*.json>
"""

import glob
import json
import os
import sys

# Meta-Llama-3-8B-Instruct, MMLU 5-shot (model card: macro; eval_details.md: micro).
REFERENCE = {"macro": 68.4, "micro": 67.4}
CATEGORIES = ["stem", "humanities", "social_sciences", "other"]


def find_results_file(path):
    if os.path.isfile(path):
        return path
    files = glob.glob(os.path.join(path, "**", "results_*.json"), recursive=True)
    if not files:
        sys.exit(f"No results_*.json under {path}")
    return max(files, key=os.path.getmtime)


def main_metric(entry):
    """Return (key, value) of the accuracy-like metric of a task entry."""
    for key in ("exact_match,strict_match", "acc,none"):
        if key in entry:
            return key, entry[key]
    raise KeyError(f"No known metric in {sorted(entry)}")


def main(path):
    fname = find_results_file(path)
    with open(fname) as f:
        data = json.load(f)
    results, groups = data["results"], data.get("group_subtasks", {})
    n_samples = data.get("n-samples", {})

    subjects = {t: r for t, r in results.items() if t not in groups}
    if not subjects:
        sys.exit("No per-subject results found")
    metric_key, _ = main_metric(next(iter(subjects.values())))

    root = next(g for g in groups if not any(g in subs for subs in groups.values()))
    macro = 100 * sum(main_metric(r)[1] for r in subjects.values()) / len(subjects)
    micro = 100 * main_metric(results[root])[1]
    n_docs = sum(n_samples.get(t, {}).get("effective", 0) for t in subjects)

    cfg = data.get("config", {})
    print(f"\nResults file : {fname}")
    print(f"Model        : {cfg.get('model_args')}")
    print(f"Task / metric: {root} / {metric_key}")
    print(f"Chat template: {data.get('chat_template') is not None}, "
          f"fewshot_as_multiturn: {data.get('fewshot_as_multiturn')}")
    print(f"Subjects     : {len(subjects)}   questions evaluated: {n_docs}")
    if n_docs and n_docs < 14042:
        print("  NOTE: partial run (--limit), numbers are NOT comparable to the model card.")

    print("\nCategory (micro)")
    for cat in CATEGORIES:
        g = next((g for g in groups if g.endswith(cat)), None)
        if g in results:
            print(f"  {cat:<16} {100 * main_metric(results[g])[1]:6.2f}")

    print(f"\n{'':16} {'ours':>7} {'card':>7} {'diff':>7}")
    for name, ours in (("macro (card)", macro), ("micro", micro)):
        ref = REFERENCE[name.split()[0]]
        print(f"  {name:<14} {ours:7.2f} {ref:7.2f} {ours - ref:+7.2f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
