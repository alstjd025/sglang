#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

if ! "${PYTHON_BIN}" -c "import sglang_router.launch_router" >/dev/null 2>&1; then
  cat >&2 <<'MSG'
sglang_router is not installed in this environment.
To use the PD router, build and install the Python binding first:

  cd /workspace/sglang/sgl-model-gateway/bindings/python
  maturin build --release --out dist --features vendored-openssl
  /venv/main/bin/pip install dist/sglang_router-*.whl
MSG
  exit 1
fi

exec "${PYTHON_BIN}" -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill "http://${SGLANG_PD_BACKEND_HOST}:${SGLANG_PD_PREFILL_PORT}" \
  --decode "http://${SGLANG_PD_BACKEND_HOST}:${SGLANG_PD_DECODE_PORT}" \
  --host "${SGLANG_PD_ROUTER_HOST}" \
  --port "${SGLANG_PD_ROUTER_PORT}" \
  ${SGLANG_PD_ROUTER_EXTRA_ARGS}
