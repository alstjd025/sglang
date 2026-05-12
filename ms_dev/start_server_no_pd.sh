#!/usr/bin/env bash
# Single-server entry point (no PD disaggregation).
#
# Observability flags (all off by default):
#   --metrics / --no-metrics
#   --request-logs / --no-request-logs
#   --trace / --no-trace
#   --obs=full|metrics|request|trace|none  (preset)
# Any unrecognized flags pass through to sglang.launch_server.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export SGLANG_ENV_PROFILE="${SGLANG_ENV_PROFILE:-single}"
source "${SCRIPT_DIR}/env.sh"
source "${SCRIPT_DIR}/lib_server.sh"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[start_server_no_pd] missing venv python at ${PYTHON_BIN}" >&2
  echo "[start_server_no_pd] run ${SGLANG_MS_DEV_DIR}/sglang_install_dev.sh first" >&2
  exit 1
fi

export HF_HOME
export HF_HUB_CACHE

# Defaults: no observability (user opts in)
OBS_METRICS=0
OBS_REQUEST_LOGS=0
OBS_TRACE=0
parse_obs_flags "$@"

obs_args=()
append_obs_args obs_args \
  "${SGLANG_REQUEST_LOG_DIR}" \
  "${SGLANG_REQUEST_METRICS_DIR}" \
  "${SGLANG_CRASH_DUMP_DIR}"

hicache_args=()
if [[ "${SGLANG_SERVER_HICACHE_ENABLE:-0}" == "1" ]]; then
  hicache_args+=(
    --enable-hierarchical-cache
    --hicache-write-policy "${SGLANG_SERVER_HICACHE_WRITE_POLICY}"
    --hicache-io-backend "${SGLANG_SERVER_HICACHE_IO_BACKEND}"
    --hicache-mem-layout "${SGLANG_SERVER_HICACHE_MEM_LAYOUT}"
    --hicache-ratio "${SGLANG_SERVER_HICACHE_RATIO}"
    --hicache-size "${SGLANG_SERVER_HICACHE_SIZE}"
  )
  if [[ -n "${SGLANG_SERVER_HICACHE_STORAGE_BACKEND:-}" ]]; then
    hicache_args+=(
      --hicache-storage-backend "${SGLANG_SERVER_HICACHE_STORAGE_BACKEND}"
      --hicache-storage-prefetch-policy "${SGLANG_SERVER_HICACHE_PREFETCH_POLICY}"
    )
  fi
  if [[ "${SGLANG_SERVER_ENABLE_CACHE_REPORT:-0}" == "1" ]]; then
    hicache_args+=(--enable-cache-report)
  fi
fi

extra_args=()
if [[ -n "${SGLANG_SERVE_EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  extra_args=( ${SGLANG_SERVE_EXTRA_ARGS} )
fi

# Admission control (Phase A: NULL/single only).
# See ms_dev/CLAUDE.md and managers/admission_control/CLAUDE.md.
admission_args=()
append_admission_args admission_args "single"

# HALO: Project Halo Phase 1 — job-level slowdown tracking.
# Off by default. See managers/halo/CLAUDE.md and ms_dev/halo_dev/CLAUDE.md.
halo_args=()
append_halo_args halo_args "single"

cd "${SGLANG_REPO_ROOT}"

launch_server \
  "${PYTHON_BIN}" -m sglang.launch_server \
  --model-path "${SGLANG_MODEL_PATH}" \
  --host "${SGLANG_HOST}" \
  --port "${SGLANG_PORT}" \
  --attention-backend "${SGLANG_ATTENTION_BACKEND}" \
  --sampling-backend "${SGLANG_SAMPLING_BACKEND}" \
  --mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}" \
  --tp-size "${SGLANG_TP_SIZE}" \
  --random-seed 42 \
  "${obs_args[@]}" \
  "${hicache_args[@]}" \
  "${admission_args[@]}" \
  "${halo_args[@]}" \
  "${extra_args[@]}" \
  "${OBS_REMAINING_ARGS[@]}"
