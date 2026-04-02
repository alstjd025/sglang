#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/lib_sglang_server.sh"

args=(
  --enable-metrics
)

if [[ "${SGLANG_ENABLE_METRICS_FOR_ALL_SCHEDULERS:-0}" == "1" ]]; then
  args+=(--enable-metrics-for-all-schedulers)
fi

run_sglang_server "${args[@]}"
