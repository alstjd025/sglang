#!/usr/bin/env bash
set -euo pipefail

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "Please source this helper from another script." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

cd "${SGLANG_REPO_ROOT}"

export HF_HOME
export HF_HUB_CACHE

run_sglang_server() {
  exec "${PYTHON_BIN}" -m sglang.launch_server \
    --model-path "${SGLANG_MODEL_PATH}" \
    --host "${SGLANG_HOST}" \
    --port "${SGLANG_PORT}" \
    --attention-backend "${SGLANG_ATTENTION_BACKEND}" \
    --sampling-backend "${SGLANG_SAMPLING_BACKEND}" \
    --mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}" \
    "$@" \
    ${SGLANG_SERVE_EXTRA_ARGS}
}
