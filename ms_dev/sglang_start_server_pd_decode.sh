#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

export SGLANG_LOG_SCHEDULER_STATUS_TARGET="${SGLANG_LOG_SCHEDULER_STATUS_TARGET:-stdout}"
export SGLANG_OTLP_EXPORTER_SCHEDULE_DELAY_MILLIS
export SGLANG_OTLP_EXPORTER_MAX_EXPORT_BATCH_SIZE

args=(
  -m sglang.launch_server
  --model-path "${SGLANG_MODEL_PATH}"
  --host "${SGLANG_HOST}"
  --port "${SGLANG_PD_DECODE_PORT}"
  --attention-backend "${SGLANG_ATTENTION_BACKEND}"
  --sampling-backend "${SGLANG_SAMPLING_BACKEND}"
  --mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}"
  --disaggregation-mode decode
  --base-gpu-id "${SGLANG_PD_DECODE_BASE_GPU_ID}"
  --tp-size 1
  --enable-metrics
  --log-requests
  --log-requests-level "${SGLANG_LOG_REQUESTS_LEVEL}"
  --log-requests-format "${SGLANG_LOG_REQUESTS_FORMAT}"
  --log-requests-target stdout "${SGLANG_PD_DECODE_REQUEST_LOG_DIR}"
  --enable-request-time-stats-logging
  --export-metrics-to-file
  --export-metrics-to-file-dir "${SGLANG_PD_DECODE_REQUEST_METRICS_DIR}"
  --crash-dump-folder "${SGLANG_PD_DECODE_CRASH_DUMP_DIR}"
  # --enable-trace
  # --otlp-traces-endpoint "${SGLANG_OTLP_TRACES_ENDPOINT}"
)

if [[ "${SGLANG_ENABLE_METRICS_FOR_ALL_SCHEDULERS:-0}" == "1" ]]; then
  args+=(--enable-metrics-for-all-schedulers)
fi

if [[ -n "${SGLANG_PD_IB_DEVICE}" ]]; then
  args+=(--disaggregation-ib-device "${SGLANG_PD_IB_DEVICE}")
fi

# NOTE:
# In current SGLang, PD decode mode forcibly sets disable_radix_cache=True internally.
# Therefore --enable-hierarchical-cache conflicts with decode mode and will raise:
#   "enable-hierarchical-cache and disable-radix-cache are mutually exclusive"
# Keep decode L2 HiCache disabled here to avoid startup failure.
if [[ "${SGLANG_PD_DECODE_HICACHE_ENABLE:-0}" == "1" ]]; then
  echo "[decode_pd] WARN: decode L2 HiCache is not compatible with current PD decode mode (radix cache forced disabled). Skipping --enable-hierarchical-cache." >&2
fi

# Optional decode-side async KV offload (L3 backend required)
if [[ "${SGLANG_PD_DECODE_OFFLOAD_ENABLE:-0}" == "1" ]]; then
  if [[ -z "${SGLANG_PD_DECODE_HICACHE_STORAGE_BACKEND:-}" ]]; then
    echo "[decode_pd] ERROR: SGLANG_PD_DECODE_HICACHE_STORAGE_BACKEND is required when L3 hicache is enabled." >&2
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
  extra_args=( ${SGLANG_PD_DECODE_EXTRA_ARGS} )
  filtered_extra_args=()
  dropped_hierarchical_cache_arg=0
  for token in "${extra_args[@]}"; do
    if [[ "${token}" == "--enable-hierarchical-cache" || "${token}" == --enable-hierarchical-cache=* ]]; then
      dropped_hierarchical_cache_arg=1
      continue
    fi
    filtered_extra_args+=("${token}")
  done

  if [[ "${dropped_hierarchical_cache_arg}" == "1" ]]; then
    echo "[decode_pd] WARN: removed --enable-hierarchical-cache from SGLANG_PD_DECODE_EXTRA_ARGS because it conflicts with PD decode mode." >&2
  fi

  args+=("${filtered_extra_args[@]}")
fi

echo "[decode_pd] launching decode:"
printf ' %q' "${PYTHON_BIN}" "${args[@]}"
echo

exec "${PYTHON_BIN}" "${args[@]}"
