#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

cd "${SGLANG_REPO_ROOT}"

export HF_HOME
export HF_HUB_CACHE

exec "${PYTHON_BIN}" -m sglang.launch_server \
  --model-path "${SGLANG_MODEL_PATH}" \
  --host "${SGLANG_HOST}" \
  --port "${SGLANG_PORT}" \
  --attention-backend "${SGLANG_ATTENTION_BACKEND}" \
  --sampling-backend "${SGLANG_SAMPLING_BACKEND}" \
  ${SGLANG_SERVE_EXTRA_ARGS}
