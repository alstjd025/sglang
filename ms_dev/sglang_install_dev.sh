#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

if [[ -f "$HOME/.cargo/env" ]]; then
  # Prefer a rustup-managed toolchain when it exists.
  source "$HOME/.cargo/env"
fi

cd "${SGLANG_REPO_ROOT}"

SGLANG_PYTHON_EXTRAS="${SGLANG_PYTHON_EXTRAS:-tracing}"
if [[ -n "${SGLANG_PYTHON_EXTRAS}" ]]; then
  SGLANG_PYTHON_EDITABLE_TARGET="python[${SGLANG_PYTHON_EXTRAS}]"
else
  SGLANG_PYTHON_EDITABLE_TARGET="python"
fi

ROUTER_BINDINGS_DIR="${SGLANG_REPO_ROOT}/sgl-model-gateway/bindings/python"

"${PYTHON_BIN}" --version
"${PIP_BIN}" --version

echo "[install_dev] upgrading pip..."
"${PIP_BIN}" install --upgrade pip

echo "[install_dev] installing router build dependencies..."
apt-get update
apt-get install -y --no-install-recommends \
  cargo \
  rustc \
  pkg-config \
  libssl-dev \
  protobuf-compiler \
  libprotobuf-dev

echo "[install_dev] ensuring maturin is installed..."
"${PIP_BIN}" install -U maturin

echo "[install_dev] installing editable package: ${SGLANG_PYTHON_EDITABLE_TARGET}"
"${PIP_BIN}" install -e "${SGLANG_PYTHON_EDITABLE_TARGET}"
"${PIP_BIN}" install -U huggingface_hub

echo "[install_dev] building and installing sglang_router..."
(
  cd "${ROUTER_BINDINGS_DIR}"
  "${PYTHON_BIN}" -m maturin build --release --out dist --features vendored-openssl
  "${PIP_BIN}" install dist/sglang_router-*.whl
)

echo "[install_dev] verifying runtime imports..."
"${PYTHON_BIN}" - <<'PY'
modules = [
    ('sglang', 'sglang core'),
    ('prometheus_client', 'prometheus client'),
    ('opentelemetry.sdk', 'opentelemetry sdk'),
    ('opentelemetry.exporter.otlp.proto.grpc.trace_exporter', 'opentelemetry otlp grpc exporter'),
    ('sglang_router.launch_router', 'sglang router'),
]
for mod, label in modules:
    __import__(mod)
    print(f'[install_dev] ok: {label}')
PY

echo "[install_dev] checking CLIs..."
"${PYTHON_BIN}" -m sglang.launch_server --help >/dev/null
"${PYTHON_BIN}" -m sglang_router.launch_router --help >/dev/null

echo "[install_dev] done"
