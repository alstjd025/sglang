#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/lib_sglang_server.sh"

if ! "${PYTHON_BIN}" -c "import opentelemetry.sdk" >/dev/null 2>&1; then
  echo "OpenTelemetry deps are missing. Run: ${PIP_BIN} install -e \"python[tracing]\"" >&2
  exit 1
fi

export SGLANG_OTLP_EXPORTER_SCHEDULE_DELAY_MILLIS
export SGLANG_OTLP_EXPORTER_MAX_EXPORT_BATCH_SIZE

args=(
  --enable-trace
  --otlp-traces-endpoint "${SGLANG_OTLP_TRACES_ENDPOINT}"
)

run_sglang_server "${args[@]}"
