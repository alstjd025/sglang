#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

args=(
  -m sglang.launch_server
  --model-path "${SGLANG_MODEL_PATH}"
  --host "${SGLANG_HOST}"
  --port "${SGLANG_PD_DECODE_PORT}"
  --attention-backend "${SGLANG_ATTENTION_BACKEND}"
  --sampling-backend "${SGLANG_SAMPLING_BACKEND}"
  --mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}"
  --disaggregation-mode decode
  --base-gpu-id "${SGLANG_PD_DECODE_BASE_GPU_ID}"
  --tp-size 1
)

if [[ -n "${SGLANG_PD_IB_DEVICE}" ]]; then
  args+=(--disaggregation-ib-device "${SGLANG_PD_IB_DEVICE}")
fi

if [[ -n "${SGLANG_PD_DECODE_EXTRA_ARGS}" ]]; then
  # shellcheck disable=SC2206
  extra_args=( ${SGLANG_PD_DECODE_EXTRA_ARGS} )
  args+=("${extra_args[@]}")
fi

echo "[decode_pd] launching decode:"
printf ' %q' "${PYTHON_BIN}" "${args[@]}"
echo

exec "${PYTHON_BIN}" "${args[@]}"