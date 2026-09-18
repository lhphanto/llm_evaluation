#!/usr/bin/env bash
# Create a local venv with lm-evaluation-harness.
#   ./setup.sh            # HF transformers backend only
#   WITH_VLLM=1 ./setup.sh  # also install vLLM (CUDA machines only)
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"
[ -d .venv ] || "$PYTHON" -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
if [ "${WITH_VLLM:-0}" = "1" ]; then
  .venv/bin/pip install "lm_eval[vllm]==0.4.13"
fi

# Llama 3 is gated: the token must belong to an account that accepted the license.
if ! .venv/bin/python -c "from huggingface_hub import whoami; whoami()" >/dev/null 2>&1; then
  echo "Not logged in to Hugging Face. Run: .venv/bin/hf auth login  (or export HF_TOKEN=...)"
fi
echo "Setup done."
