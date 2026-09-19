#!/usr/bin/env bash
# Llama-3-8B-Instruct card benchmarks with lm-evaluation-harness.
#
# Usage: ./run_mmlu.sh [llama|standard|gsm8k|math|gpqa|humaneval] [extra lm_eval args...]
#
#   llama      MMLU, Meta's instruct protocol (what the model card's 68.4 uses):
#              5-shot as multi-turn user/assistant chat, the model *generates* the
#              answer letter, exact match. lm-eval task `mmlu_llama`.
#   standard   MMLU, classic lm-eval: plain-text 5-shot prompt, loglikelihood over
#              A/B/C/D, no chat template. lm-eval task `mmlu` (Open LLM Leaderboard style).
#   gsm8k      GSM8K, 8-shot chain-of-thought, Meta's instruct protocol (chat template,
#              generates a full reasoning trace, exact match on the final number).
#              lm-eval task `gsm8k_llama`. Card: 79.6.
#   math       MATH, 4-shot chain-of-thought (Open LLM Leaderboard v2's "hard"/Level-5
#              subset, `leaderboard_math_hard`, no chat template). Card: 30.0. This is a
#              1,324-question subset, not the full 5,000-question MATH test set, so
#              treat it as directionally comparable rather than an exact reproduction.
#   gpqa       GPQA (main split), 0-shot, multiple choice by loglikelihood, no chat
#              template. lm-eval task `leaderboard_gpqa_main`. Card: 34.2. Only ~448
#              questions, so run-to-run noise is a few points.
#   humaneval  HumanEval, 0-shot code generation, pass@1. lm-eval task
#              `humaneval_instruct`. Card: 62.2. Runs model-generated code locally to
#              check it (`--confirm_run_unsafe_code`) — only run this against models
#              you trust, ideally in a sandbox/VM.
#
# Only `gsm8k` has a Meta-branded lm-eval task; math/gpqa/humaneval use the closest
# standard lm-eval equivalents, so expect more drift from the card than MMLU/GSM8K.
#
# Env overrides:
#   MODEL=meta-llama/Meta-Llama-3-8B-Instruct  BACKEND=hf|vllm  DEVICE=cuda:0
#   BATCH_SIZE=auto  OUT_DIR=results  TASKS=<override task list>
#   PYTHON=<interpreter>  LAUNCHER=<job submit prefix, e.g. "vbatch -P h100-1s">
#
# Examples:
#   ./run_mmlu.sh llama
#   BACKEND=vllm ./run_mmlu.sh llama
#   ./run_mmlu.sh standard
#   ./run_mmlu.sh gsm8k
#   ./run_mmlu.sh math
#   ./run_mmlu.sh gpqa
#   ./run_mmlu.sh humaneval
#   ./run_mmlu.sh llama --limit 5        # quick smoke test (5 docs per subject)
#   LAUNCHER="vbatch -P h100-1s" ./run_mmlu.sh llama   # submit as a Velda batch job
set -euo pipefail
cd "$(dirname "$0")"

MODE="${1:-llama}"; shift || true
MODEL="${MODEL:-meta-llama/Meta-Llama-3-8B-Instruct}"
BACKEND="${BACKEND:-hf}"
OUT_DIR="${OUT_DIR:-results}"
# Python to use: $PYTHON if set, else the activated venv/conda env (not conda `base`),
# else ./.venv, else python3 on PATH.
if [ -n "${PYTHON:-}" ]; then PY="$PYTHON"
elif [ -n "${VIRTUAL_ENV:-}" ] || { [ -n "${CONDA_DEFAULT_ENV:-}" ] && [ "$CONDA_DEFAULT_ENV" != base ]; }; then PY=python
elif [ -x .venv/bin/python ]; then PY=.venv/bin/python
else PY=python3; fi
"$PY" -c "import lm_eval" 2>/dev/null || { echo "lm_eval is not installed for $("$PY" -c 'import sys; print(sys.executable)'); see README setup." >&2; exit 1; }

# Optional job launcher, e.g. LAUNCHER="vbatch -P h100-1s" on Velda. The whole script is
# re-run inside the job, so device detection, the eval and the summary all happen on the
# GPU node. Settings are passed explicitly with `env`, since a batch job may not inherit
# this shell's environment. Assumes the job sees the same filesystem (true on Velda).
if [ -n "${LAUNCHER:-}" ] && [ -z "${_IN_LAUNCHER:-}" ]; then
  # Launchers target GPU pools: pin cuda:0 so a job that can't see CUDA fails loudly
  # instead of silently falling back to CPU. Override with DEVICE=... if needed.
  DEVICE="${DEVICE:-cuda:0}"
  FWD=(_IN_LAUNCHER=1 "PYTHON=$("$PY" -c 'import sys; print(sys.executable)')"
       "MODEL=$MODEL" "BACKEND=$BACKEND" "OUT_DIR=$OUT_DIR")
  for v in DEVICE BATCH_SIZE TASKS HF_HOME; do
    [ -n "${!v:-}" ] && FWD+=("$v=${!v}")
  done
  set -x
  exec $LAUNCHER env "${FWD[@]}" bash "$PWD/run_mmlu.sh" "$MODE" "$@"
fi

if [ -z "${DEVICE:-}" ]; then
  DEVICE=$("$PY" -c "import torch; print('cuda:0' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')")
fi
# `auto` batch-size probing can loop on Metal OOM errors, so use a fixed size on Apple GPUs.
if [ "$DEVICE" = "mps" ]; then BATCH_SIZE="${BATCH_SIZE:-8}"; else BATCH_SIZE="${BATCH_SIZE:-auto}"; fi

case "$MODE" in
  llama)
    TASKS="${TASKS:-mmlu_llama}"; NUM_FEWSHOT=5
    MODE_ARGS=(--apply_chat_template --fewshot_as_multiturn)
    ;;
  standard)
    TASKS="${TASKS:-mmlu}"; NUM_FEWSHOT=5
    MODE_ARGS=()
    ;;
  gsm8k)
    TASKS="${TASKS:-gsm8k_llama}"; NUM_FEWSHOT=8
    MODE_ARGS=(--apply_chat_template --fewshot_as_multiturn)
    ;;
  math)
    TASKS="${TASKS:-leaderboard_math_hard}"; NUM_FEWSHOT=4
    MODE_ARGS=()
    ;;
  gpqa)
    TASKS="${TASKS:-leaderboard_gpqa_main}"; NUM_FEWSHOT=0
    MODE_ARGS=()
    ;;
  humaneval)
    TASKS="${TASKS:-humaneval_instruct}"; NUM_FEWSHOT=0
    # Executes the model's generated code locally to check it against the tests.
    # Two separate libraries gate this, so both opt-ins are needed: lm-eval's own
    # --confirm_run_unsafe_code, and HF `evaluate`'s code_eval metric, which refuses
    # to run at all without HF_ALLOW_CODE_EVAL=1.
    MODE_ARGS=(--confirm_run_unsafe_code)
    export HF_ALLOW_CODE_EVAL=1
    ;;
  *) echo "Unknown mode '$MODE' (use llama|standard|gsm8k|math|gpqa|humaneval)"; exit 1 ;;
esac

case "$BACKEND" in
  hf)   MODEL_ARGS="pretrained=${MODEL},dtype=bfloat16" ;;
  vllm) MODEL_ARGS="pretrained=${MODEL},dtype=bfloat16,gpu_memory_utilization=0.85,max_model_len=8192" ;;
  *) echo "Unknown BACKEND '$BACKEND' (use hf|vllm)"; exit 1 ;;
esac

# --log_samples is incompatible with --limit, so only keep per-sample logs on full runs.
LOG_ARGS=(--log_samples)
for a in "$@"; do case "$a" in --limit|-L|--limit=*) LOG_ARGS=() ;; esac; done

RUN_OUT="${OUT_DIR}/${MODE}"
mkdir -p "$RUN_OUT"
set -x
"$PY" -m lm_eval run \
  --model "$BACKEND" \
  --model_args "$MODEL_ARGS" \
  --tasks "$TASKS" \
  --num_fewshot "$NUM_FEWSHOT" \
  --device "$DEVICE" \
  --batch_size "$BATCH_SIZE" \
  --seed 1234 \
  --output_path "$RUN_OUT" \
  ${MODE_ARGS[@]+"${MODE_ARGS[@]}"} ${LOG_ARGS[@]+"${LOG_ARGS[@]}"} "$@"
set +x

"$PY" summarize.py "$RUN_OUT"
