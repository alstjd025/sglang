#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

cd "${SGLANG_REPO_ROOT}"

SGLANG_PYTHON_EXTRAS="${SGLANG_PYTHON_EXTRAS:-tracing}"
if [[ -n "${SGLANG_PYTHON_EXTRAS}" ]]; then
  SGLANG_PYTHON_EDITABLE_TARGET="python[${SGLANG_PYTHON_EXTRAS}]"
else
  SGLANG_PYTHON_EDITABLE_TARGET="python"
fi

"${PYTHON_BIN}" --version
"${PIP_BIN}" --version

echo "[install_dev] upgrading pip..."
"${PIP_BIN}" install --upgrade pip

echo "[install_dev] installing editable package: ${SGLANG_PYTHON_EDITABLE_TARGET}"
"${PIP_BIN}" install -e "${SGLANG_PYTHON_EDITABLE_TARGET}"
"${PIP_BIN}" install -U huggingface_hub

echo "[install_dev] verifying runtime imports..."
"${PYTHON_BIN}" - <<'PY'
modules = [
    ('sglang', 'sglang core'),
    ('prometheus_client', 'prometheus client'),
    ('opentelemetry.sdk', 'opentelemetry sdk'),
    ('opentelemetry.exporter.otlp.proto.grpc.trace_exporter', 'opentelemetry otlp grpc exporter'),
]
for mod, label in modules:
    __import__(mod)
    print(f'[install_dev] ok: {label}')
PY

echo "[install_dev] checking sglang CLI..."
"${PYTHON_BIN}" -m sglang.launch_server --help >/dev/null

echo "[install_dev] done"
