#!/usr/bin/env bash
# PD disaggregation — decode role.
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

OBS_METRICS=1
OBS_REQUEST_LOGS=1
OBS_TRACE=0
parse_obs_flags "$@"

obs_args=()
append_obs_args obs_args \
  "${SGLANG_PD_DECODE_REQUEST_LOG_DIR}" \
  "${SGLANG_PD_DECODE_REQUEST_METRICS_DIR}" \
  "${SGLANG_PD_DECODE_CRASH_DUMP_DIR}"

args=(
  -m sglang.launch_server
  --model-path "${SGLANG_MODEL_PATH}"
  --host "${SGLANG_HOST}"
  --port "${SGLANG_PD_DECODE_PORT}"
  --attention-backend "${SGLANG_ATTENTION_BACKEND}"
  --sampling-backend "${SGLANG_SAMPLING_BACKEND}"
  --mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}"
  --random-seed 42
  --disaggregation-mode decode
  --base-gpu-id "${SGLANG_PD_DECODE_BASE_GPU_ID}"
  --tp-size 1
)

args+=("${obs_args[@]}")

if [[ -n "${SGLANG_PD_IB_DEVICE}" ]]; then
  args+=(--disaggregation-ib-device "${SGLANG_PD_IB_DEVICE}")
fi

# Current SGLang forces disable_radix_cache=True in decode mode, which
# collides with --enable-hierarchical-cache. Skip L2 HiCache here.
if [[ "${SGLANG_PD_DECODE_HICACHE_ENABLE:-0}" == "1" ]]; then
  echo "[start_server_pd_decode] WARN: decode L2 HiCache is not compatible with PD decode mode (radix cache is forced off); ignoring --enable-hierarchical-cache." >&2
fi

# Optional decode-side async KV offload (requires L3 storage backend).
if [[ "${SGLANG_PD_DECODE_OFFLOAD_ENABLE:-0}" == "1" ]]; then
  if [[ -z "${SGLANG_PD_DECODE_HICACHE_STORAGE_BACKEND:-}" ]]; then
    echo "[start_server_pd_decode] ERROR: SGLANG_PD_DECODE_HICACHE_STORAGE_BACKEND is required when decode offload is enabled." >&2
    exit 1
  fi
  args+=(
    --hicache-storage-backend "${SGLANG_PD_DECODE_HICACHE_STORAGE_BACKEND}"
    --hicache-storage-prefetch-policy "${SGLANG_PD_DECODE_HICACHE_PREFETCH_POLICY}"
    --hicache-write-policy "${SGLANG_PD_DECODE_HICACHE_WRITE_POLICY}"
    --hicache-io-backend "${SGLANG_PD_DECODE_HICACHE_IO_BACKEND}"
    --hicache-mem-layout "${SGLANG_PD_DECODE_HICACHE_MEM_LAYOUT}"
    --hicache-ratio "${SGLANG_PD_DECODE_HICACHE_RATIO}"
    --hicache-size "${SGLANG_PD_DECODE_HICACHE_SIZE}"
    --disaggregation-decode-enable-offload-kvcache
  )
  if [[ "${SGLANG_PD_DECODE_ENABLE_CACHE_REPORT:-0}" == "1" ]]; then
    args+=(--enable-cache-report)
  fi
fi

if [[ -n "${SGLANG_PD_DECODE_EXTRA_ARGS}" ]]; then
  # shellcheck disable=SC2206
  raw_extra=( ${SGLANG_PD_DECODE_EXTRA_ARGS} )
  filtered_extra=()
  dropped_hicache=0
  for token in "${raw_extra[@]}"; do
    if [[ "${token}" == "--enable-hierarchical-cache" || "${token}" == --enable-hierarchical-cache=* ]]; then
      dropped_hicache=1
      continue
    fi
    filtered_extra+=("${token}")
  done
  if [[ "${dropped_hicache}" == "1" ]]; then
    echo "[start_server_pd_decode] WARN: removed --enable-hierarchical-cache from SGLANG_PD_DECODE_EXTRA_ARGS (conflicts with PD decode mode)." >&2
  fi
  args+=("${filtered_extra[@]}")
fi

args+=("${OBS_REMAINING_ARGS[@]}")

launch_server "${PYTHON_BIN}" "${args[@]}"
