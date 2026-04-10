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

build_single_hicache_args() {
  local -n _out="$1"
  _out=()

  if [[ "${SGLANG_SERVER_HICACHE_ENABLE:-0}" != "1" ]]; then
    return
  fi

  _out+=(
    --enable-hierarchical-cache
    --hicache-write-policy "${SGLANG_SERVER_HICACHE_WRITE_POLICY}"
    --hicache-io-backend "${SGLANG_SERVER_HICACHE_IO_BACKEND}"
    --hicache-mem-layout "${SGLANG_SERVER_HICACHE_MEM_LAYOUT}"
    --hicache-ratio "${SGLANG_SERVER_HICACHE_RATIO}"
    --hicache-size "${SGLANG_SERVER_HICACHE_SIZE}"
  )

  if [[ -n "${SGLANG_SERVER_HICACHE_STORAGE_BACKEND:-}" ]]; then
    _out+=(
      --hicache-storage-backend "${SGLANG_SERVER_HICACHE_STORAGE_BACKEND}"
      --hicache-storage-prefetch-policy "${SGLANG_SERVER_HICACHE_PREFETCH_POLICY}"
    )
  fi

  if [[ "${SGLANG_SERVER_ENABLE_CACHE_REPORT:-0}" == "1" ]]; then
    _out+=(--enable-cache-report)
  fi
}

run_sglang_server() {
  local extra_args=()
  local hicache_args=()

  if [[ -n "${SGLANG_SERVE_EXTRA_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    extra_args=( ${SGLANG_SERVE_EXTRA_ARGS} )
  fi

  build_single_hicache_args hicache_args

  local launch_cmd=(
    "${PYTHON_BIN}" -m sglang.launch_server
    --model-path "${SGLANG_MODEL_PATH}"
    --host "${SGLANG_HOST}"
    --port "${SGLANG_PORT}"
    --attention-backend "${SGLANG_ATTENTION_BACKEND}"
    --sampling-backend "${SGLANG_SAMPLING_BACKEND}"
    --mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}"
    "$@"
    "${hicache_args[@]}"
    "${extra_args[@]}"
  )

  echo "[single_server] launching:"
  printf ' %q' "${launch_cmd[@]}"
  echo

  exec "${launch_cmd[@]}"
}
