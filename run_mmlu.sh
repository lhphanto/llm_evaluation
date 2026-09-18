#!/usr/bin/env bash
# MMLU 5-shot with lm-evaluation-harness.
#
# Usage: ./run_mmlu.sh [llama|standard] [extra lm_eval args...]
#
#   llama     Meta's instruct protocol (what the model card's 68.4 uses):
#             5-shot as multi-turn user/assistant chat, the model *generates* the
#             answer letter, exact match. lm-eval task `mmlu_llama`.
#   standard  Classic lm-eval MMLU: plain-text 5-shot prompt, loglikelihood over
#             A/B/C/D, no chat template. lm-eval task `mmlu` (Open LLM Leaderboard style).
#
# Env overrides:
#   MODEL=meta-llama/Meta-Llama-3-8B-Instruct  BACKEND=hf|vllm  DEVICE=cuda:0
#   BATCH_SIZE=auto  OUT_DIR=results  TASKS=<override task list>
#
# Examples:
#   ./run_mmlu.sh llama
#   BACKEND=vllm ./run_mmlu.sh llama
#   ./run_mmlu.sh standard
#   ./run_mmlu.sh llama --limit 5        # quick smoke test (5 docs per subject)
set -euo pipefail
cd "$(dirname "$0")"

MODE="${1:-llama}"; shift || true
MODEL="${MODEL:-meta-llama/Meta-Llama-3-8B-Instruct}"
BACKEND="${BACKEND:-hf}"
OUT_DIR="${OUT_DIR:-results}"
# Prefer the local venv (see README); fall back to whatever is on PATH.
if [ -x .venv/bin/python ]; then PY=.venv/bin/python; else PY=python3; fi

if [ -z "${DEVICE:-}" ]; then
  DEVICE=$("$PY" -c "import torch; print('cuda:0' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')")
fi
# `auto` batch-size probing can loop on Metal OOM errors, so use a fixed size on Apple GPUs.
if [ "$DEVICE" = "mps" ]; then BATCH_SIZE="${BATCH_SIZE:-8}"; else BATCH_SIZE="${BATCH_SIZE:-auto}"; fi

case "$MODE" in
  llama)
    TASKS="${TASKS:-mmlu_llama}"
    MODE_ARGS=(--apply_chat_template --fewshot_as_multiturn)
    ;;
  standard)
    TASKS="${TASKS:-mmlu}"
    MODE_ARGS=()
    ;;
  *) echo "Unknown mode '$MODE' (use llama|standard)"; exit 1 ;;
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
  --num_fewshot 5 \
  --device "$DEVICE" \
  --batch_size "$BATCH_SIZE" \
  --seed 1234 \
  --output_path "$RUN_OUT" \
  ${MODE_ARGS[@]+"${MODE_ARGS[@]}"} ${LOG_ARGS[@]+"${LOG_ARGS[@]}"} "$@"
set +x

"$PY" summarize.py "$RUN_OUT"
