#!/usr/bin/env bash
# PD disaggregation — prefill role.
#
# Observability flags (defaults: metrics + request-logs ON, trace OFF):
#   --metrics / --no-metrics
#   --request-logs / --no-request-logs
#   --trace / --no-trace
#   --obs=full|metrics|request|trace|none  (preset)
# Any unrecognized flags pass through to sglang.launch_server.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export SGLANG_ENV_PROFILE="${SGLANG_ENV_PROFILE:-pd}"
source "${SCRIPT_DIR}/env.sh"
source "${SCRIPT_DIR}/lib_server.sh"

cd "${SGLANG_REPO_ROOT}"

# Defaults match prior behavior of sglang_start_server_pd_prefill.sh
OBS_METRICS=1
OBS_REQUEST_LOGS=1
OBS_TRACE=0
parse_obs_flags "$@"

obs_args=()
append_obs_args obs_args \
  "${SGLANG_PD_PREFILL_REQUEST_LOG_DIR}" \
  "${SGLANG_PD_PREFILL_REQUEST_METRICS_DIR}" \
  "${SGLANG_PD_PREFILL_CRASH_DUMP_DIR}"

args=(
  -m sglang.launch_server
  --model-path "${SGLANG_MODEL_PATH}"
  --host "${SGLANG_HOST}"
  --port "${SGLANG_PD_PREFILL_PORT}"
  --attention-backend "${SGLANG_ATTENTION_BACKEND}"
  --sampling-backend "${SGLANG_SAMPLING_BACKEND}"
  --mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}"
  --random-seed 42
  --disaggregation-mode prefill
  --base-gpu-id "${SGLANG_PD_PREFILL_BASE_GPU_ID}"
  --tp-size 1
)

args+=("${obs_args[@]}")

if [[ -n "${SGLANG_PD_IB_DEVICE}" ]]; then
  args+=(--disaggregation-ib-device "${SGLANG_PD_IB_DEVICE}")
fi

# Prefill-side HiCache (hierarchical cache) can be safely enabled in PD mode.
if [[ "${SGLANG_PD_PREFILL_HICACHE_ENABLE:-0}" == "1" ]]; then
  args+=(
    --enable-hierarchical-cache
    --hicache-write-policy "${SGLANG_PD_PREFILL_HICACHE_WRITE_POLICY}"
    --hicache-io-backend "${SGLANG_PD_PREFILL_HICACHE_IO_BACKEND}"
    --hicache-mem-layout "${SGLANG_PD_PREFILL_HICACHE_MEM_LAYOUT}"
    --hicache-ratio "${SGLANG_PD_PREFILL_HICACHE_RATIO}"
    --hicache-size "${SGLANG_PD_PREFILL_HICACHE_SIZE}"
  )
  if [[ -n "${SGLANG_PD_PREFILL_HICACHE_STORAGE_BACKEND:-}" ]]; then
    args+=(
      --hicache-storage-backend "${SGLANG_PD_PREFILL_HICACHE_STORAGE_BACKEND}"
      --hicache-storage-prefetch-policy "${SGLANG_PD_PREFILL_HICACHE_PREFETCH_POLICY}"
    )
  fi
  if [[ "${SGLANG_PD_PREFILL_ENABLE_CACHE_REPORT:-0}" == "1" ]]; then
    args+=(--enable-cache-report)
  fi
fi

# Admission control flags propagate but the controller is bypassed in PD mode
# (Phase A scope) — append_admission_args emits a warning if any SLO is set.
admission_args=()
append_admission_args admission_args "pd-prefill"
args+=("${admission_args[@]}")

if [[ -n "${SGLANG_PD_PREFILL_EXTRA_ARGS}" ]]; then
  # shellcheck disable=SC2206
  raw_extra=( ${SGLANG_PD_PREFILL_EXTRA_ARGS} )
  filtered_extra=()
  dropped_decode_only=0
  for token in "${raw_extra[@]}"; do
    if [[ "${token}" == "--disaggregation-decode-enable-offload-kvcache" || "${token}" == --disaggregation-decode-enable-offload-kvcache=* ]]; then
      dropped_decode_only=1
      continue
    fi
    filtered_extra+=("${token}")
  done
  if [[ "${dropped_decode_only}" == "1" ]]; then
    echo "[start_server_pd_prefill] WARN: removed --disaggregation-decode-enable-offload-kvcache from SGLANG_PD_PREFILL_EXTRA_ARGS (decode-only flag)." >&2
  fi
  args+=("${filtered_extra[@]}")
fi

args+=("${OBS_REMAINING_ARGS[@]}")

launch_server "${PYTHON_BIN}" "${args[@]}"
