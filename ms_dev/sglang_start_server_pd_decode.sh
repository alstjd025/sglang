#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/lib_sglang_server.sh"

args=(
  --disaggregation-mode decode
  --port "${SGLANG_PD_DECODE_PORT}"
  --base-gpu-id "${SGLANG_PD_DECODE_BASE_GPU_ID}"
)

if [[ -n "${SGLANG_PD_IB_DEVICE}" ]]; then
  args+=(--disaggregation-ib-device "${SGLANG_PD_IB_DEVICE}")
fi

if [[ -n "${SGLANG_PD_DECODE_EXTRA_ARGS}" ]]; then
  # shellcheck disable=SC2206
  extra_args=( ${SGLANG_PD_DECODE_EXTRA_ARGS} )
  args+=("${extra_args[@]}")
fi

run_sglang_server "${args[@]}"
