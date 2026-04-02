#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/lib_sglang_server.sh"

export SGLANG_LOG_SCHEDULER_STATUS_TARGET="${SGLANG_LOG_SCHEDULER_STATUS_TARGET:-stdout}"

args=(
  --enable-metrics
  --log-requests
  --log-requests-level "${SGLANG_LOG_REQUESTS_LEVEL}"
  --log-requests-format "${SGLANG_LOG_REQUESTS_FORMAT}"
  --log-requests-target stdout "${SGLANG_REQUEST_LOG_DIR}"
  --enable-request-time-stats-logging
  --export-metrics-to-file
  --export-metrics-to-file-dir "${SGLANG_REQUEST_METRICS_DIR}"
  --crash-dump-folder "${SGLANG_CRASH_DUMP_DIR}"
)

if [[ "${SGLANG_ENABLE_METRICS_FOR_ALL_SCHEDULERS:-0}" == "1" ]]; then
  args+=(--enable-metrics-for-all-schedulers)
fi

run_sglang_server "${args[@]}"
