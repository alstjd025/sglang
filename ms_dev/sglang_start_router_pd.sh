#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

if [[ -f "$HOME/.cargo/env" ]]; then
  source "$HOME/.cargo/env"
fi

ensure_router_installed() {
  if "${PYTHON_BIN}" -c "import sglang_router.launch_router" >/dev/null 2>&1; then
    return 0
  fi

  echo "[router_pd] sglang_router is missing; attempting automatic build/install..."
  apt-get update
  apt-get install -y --no-install-recommends \
    cargo \
    rustc \
    pkg-config \
    libssl-dev \
    protobuf-compiler \
    libprotobuf-dev

  "${PIP_BIN}" install -U maturin

  (
    cd "${SGLANG_REPO_ROOT}/sgl-model-gateway/bindings/python"
    "${PYTHON_BIN}" -m maturin build --release --out dist --features vendored-openssl
    "${PIP_BIN}" install dist/sglang_router-*.whl
  )

  "${PYTHON_BIN}" -c "import sglang_router.launch_router" >/dev/null 2>&1
}

ensure_router_installed

args=(
  --pd-disaggregation
  --prefill "http://${SGLANG_PD_BACKEND_HOST}:${SGLANG_PD_PREFILL_PORT}"
  --decode "http://${SGLANG_PD_BACKEND_HOST}:${SGLANG_PD_DECODE_PORT}"
  --host "${SGLANG_PD_ROUTER_HOST}"
  --port "${SGLANG_PD_ROUTER_PORT}"
  --log-dir "${SGLANG_ROUTER_LOG_DIR}"
  --json-log
  --prometheus-port "${SGLANG_ROUTER_PROMETHEUS_PORT}"
  --enable-trace
  --otlp-traces-endpoint "${SGLANG_OTLP_TRACES_ENDPOINT}"
)

if [[ -n "${SGLANG_PD_ROUTER_EXTRA_ARGS}" ]]; then
  # shellcheck disable=SC2206
  extra_args=( ${SGLANG_PD_ROUTER_EXTRA_ARGS} )
  args+=("${extra_args[@]}")
fi

echo "[router_pd] launching router:"
printf ' %q' "${PYTHON_BIN}" -m sglang_router.launch_router "${args[@]}"
echo

exec "${PYTHON_BIN}" -m sglang_router.launch_router "${args[@]}"
