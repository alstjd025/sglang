#!/usr/bin/env bash

# Shared helpers for ms_dev server launchers.
# Expected to be sourced, not executed directly.

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "lib_server.sh is a library; source it from a launcher." >&2
  exit 1
fi

# parse_obs_flags
#
# Parses observability toggles from script args. Unknown args are preserved
# in OBS_REMAINING_ARGS so launchers can forward them to sglang.launch_server.
#
# Globals set:
#   OBS_METRICS, OBS_REQUEST_LOGS, OBS_TRACE  (0 or 1)
#   OBS_REMAINING_ARGS                        (passthrough args)
#
# Callers decide defaults BEFORE invoking this by pre-setting OBS_METRICS etc.
parse_obs_flags() {
  OBS_REMAINING_ARGS=()
  while (( $# )); do
    case "$1" in
      --metrics)          OBS_METRICS=1 ;;
      --no-metrics)       OBS_METRICS=0 ;;
      --request-logs)     OBS_REQUEST_LOGS=1 ;;
      --no-request-logs)  OBS_REQUEST_LOGS=0 ;;
      --trace)            OBS_TRACE=1 ;;
      --no-trace)         OBS_TRACE=0 ;;
      --obs=full|--obs=all)
        OBS_METRICS=1; OBS_REQUEST_LOGS=1; OBS_TRACE=1 ;;
      --obs=metrics)
        OBS_METRICS=1; OBS_REQUEST_LOGS=0; OBS_TRACE=0 ;;
      --obs=request|--obs=request-logs)
        OBS_METRICS=1; OBS_REQUEST_LOGS=1; OBS_TRACE=0 ;;
      --obs=trace)
        OBS_METRICS=0; OBS_REQUEST_LOGS=0; OBS_TRACE=1 ;;
      --obs=none)
        OBS_METRICS=0; OBS_REQUEST_LOGS=0; OBS_TRACE=0 ;;
      --obs=*)
        echo "[lib_server] unknown --obs preset: ${1#--obs=}" >&2
        return 1 ;;
      *)
        OBS_REMAINING_ARGS+=("$1") ;;
    esac
    shift
  done
}

# append_obs_args <out-array-name> <request-log-dir> <metrics-dir> <crash-dir>
#
# Appends CLI args for enabled observability toggles to the named array.
# Relies on globals set by parse_obs_flags and on env vars from env.common.sh.
append_obs_args() {
  local -n _obs_out="$1"
  local log_dir="$2"
  local metrics_dir="$3"
  local crash_dir="$4"

  if [[ "${OBS_METRICS:-0}" == "1" ]]; then
    # sglang requires --enable-metrics when SGLANG_LOG_SCHEDULER_STATUS_TARGET
    # is set, so only export it here (not in env.common.sh) to avoid breaking
    # metrics-off runs.
    export SGLANG_LOG_SCHEDULER_STATUS_TARGET="${SGLANG_LOG_SCHEDULER_STATUS_TARGET:-stdout}"
    _obs_out+=(
      --enable-metrics
      --export-metrics-to-file
      --export-metrics-to-file-dir "${metrics_dir}"
    )
    if [[ "${SGLANG_ENABLE_METRICS_FOR_ALL_SCHEDULERS:-0}" == "1" ]]; then
      _obs_out+=(--enable-metrics-for-all-schedulers)
    fi
  else
    # Proactively clear it in case the shell inherited it from a prior session.
    unset SGLANG_LOG_SCHEDULER_STATUS_TARGET
  fi

  if [[ "${OBS_REQUEST_LOGS:-0}" == "1" ]]; then
    _obs_out+=(
      --log-requests
      --log-requests-level "${SGLANG_LOG_REQUESTS_LEVEL}"
      --log-requests-format "${SGLANG_LOG_REQUESTS_FORMAT}"
      --log-requests-target stdout "${log_dir}"
      --enable-request-time-stats-logging
      --crash-dump-folder "${crash_dir}"
    )
  fi

  if [[ "${OBS_TRACE:-0}" == "1" ]]; then
    if ! "${PYTHON_BIN}" -c "import opentelemetry.sdk" >/dev/null 2>&1; then
      echo "[lib_server] --trace requested but opentelemetry.sdk is missing. Install with: ${PIP_BIN} install -e 'python[tracing]'" >&2
      return 1
    fi
    _obs_out+=(
      --enable-trace
      --otlp-traces-endpoint "${SGLANG_OTLP_TRACES_ENDPOINT}"
    )
  fi
}

# append_admission_args <out-array-name> [mode]
#
# Appends --admission-* CLI args to the named array based on
# SGLANG_ADMISSION_* env vars (set in env.common.sh). Each env var is only
# forwarded if explicitly non-empty; unset vars fall back to the SGLang
# CLI defaults. Setting either TTFT or TBT SLO turns admission control on.
#
# Arguments:
#   $1 — array variable name (passed by reference, like append_obs_args).
#   $2 — optional launcher mode: "single" (default), "pd-prefill", "pd-decode".
#        For PD modes a one-line warning is emitted if any SLO is configured,
#        because Phase A admission control is bypassed in PD mode.
#
# See managers/admission_control/CLAUDE.md and ms_dev/CLAUDE.md.
append_admission_args() {
  local -n _adm_out="$1"
  local mode="${2:-single}"

  local has_slo=0
  if [[ -n "${SGLANG_ADMISSION_TTFT_SLO_MS:-}" \
     || -n "${SGLANG_ADMISSION_TBT_SLO_MS:-}" \
     || -n "${SGLANG_ADMISSION_TTFT_SLO_RATIO:-}" \
     || -n "${SGLANG_ADMISSION_TBT_SLO_RATIO:-}" ]]; then
    has_slo=1
  fi

  if (( has_slo )) && [[ "${mode}" != "single" ]]; then
    echo "[lib_server] admission control flags will be passed but the controller is bypassed in PD mode (Phase A scope)" >&2
  fi

  if [[ -n "${SGLANG_ADMISSION_TTFT_SLO_MS:-}" ]]; then
    _adm_out+=(--admission-ttft-slo-ms "${SGLANG_ADMISSION_TTFT_SLO_MS}")
  fi
  if [[ -n "${SGLANG_ADMISSION_TBT_SLO_MS:-}" ]]; then
    _adm_out+=(--admission-tbt-slo-ms "${SGLANG_ADMISSION_TBT_SLO_MS}")
  fi
  if [[ -n "${SGLANG_ADMISSION_TTFT_SLO_RATIO:-}" ]]; then
    _adm_out+=(--admission-ttft-slo-ratio "${SGLANG_ADMISSION_TTFT_SLO_RATIO}")
  fi
  if [[ -n "${SGLANG_ADMISSION_TBT_SLO_RATIO:-}" ]]; then
    _adm_out+=(--admission-tbt-slo-ratio "${SGLANG_ADMISSION_TBT_SLO_RATIO}")
  fi
  if [[ -n "${SGLANG_ADMISSION_PREFILL_COST_MODEL:-}" ]]; then
    _adm_out+=(--admission-prefill-cost-model-path "${SGLANG_ADMISSION_PREFILL_COST_MODEL}")
  fi
  if [[ -n "${SGLANG_ADMISSION_TBT_COST_MODEL:-}" ]]; then
    _adm_out+=(--admission-tbt-cost-model-path "${SGLANG_ADMISSION_TBT_COST_MODEL}")
  fi
  if [[ -n "${SGLANG_ADMISSION_TBT_EWMA_ALPHA:-}" ]]; then
    _adm_out+=(--admission-tbt-ewma-alpha "${SGLANG_ADMISSION_TBT_EWMA_ALPHA}")
  fi
  if [[ -n "${SGLANG_ADMISSION_TBT_REACTIVE_RATIO:-}" ]]; then
    _adm_out+=(--admission-tbt-reactive-ratio "${SGLANG_ADMISSION_TBT_REACTIVE_RATIO}")
  fi
  if [[ "${SGLANG_ADMISSION_DRY_RUN:-0}" == "1" ]]; then
    _adm_out+=(--admission-dry-run)
  fi
  if [[ -n "${SGLANG_ADMISSION_DECISION_LOG:-}" ]]; then
    _adm_out+=(--admission-decision-log "${SGLANG_ADMISSION_DECISION_LOG}")
  fi
}

# append_halo_args <out-array-name> [mode]
#
# HALO: Translates SGLANG_HALO_* env vars (set in env.common.sh) into --halo-*
# CLI args. Off-by-default (SGLANG_HALO_ENABLED=0); non-zero values turn it on.
# Each optional knob (default SLO, tick interval, cost-model paths, job log)
# only forwards when explicitly set, falling back to sglang CLI defaults.
#
# Arguments:
#   $1 — array variable name (passed by reference).
#   $2 — optional launcher mode: "single" (default), "pd-prefill", "pd-decode".
#        Phase 1 only supports NULL disaggregation; in PD modes we emit a
#        one-line warning and still pass the flag so the rejection path is
#        visible (HaloController itself is a no-op outside NULL disagg).
#
# See managers/halo/CLAUDE.md and ms_dev/halo_dev/CLAUDE.md.
append_halo_args() {
  local -n _halo_out="$1"
  local mode="${2:-single}"

  # The cost-model sampler is INDEPENDENT of --halo-enabled — fit-data
  # collection works with Halo off. Forward those two flags even when the
  # Halo controller is disabled. expctl/run_experiment.py already routes
  # the sample-log path into the session dir.
  if [[ -n "${SGLANG_HALO_COST_MODEL_SAMPLE_LOG:-}" ]]; then
    _halo_out+=(--halo-cost-model-sample-log "${SGLANG_HALO_COST_MODEL_SAMPLE_LOG}")
  fi
  if [[ -n "${SGLANG_HALO_COST_MODEL_SAMPLE_EVERY:-}" ]]; then
    _halo_out+=(--halo-cost-model-sample-every "${SGLANG_HALO_COST_MODEL_SAMPLE_EVERY}")
  fi

  if [[ "${SGLANG_HALO_ENABLED:-0}" != "1" ]]; then
    return 0
  fi

  if [[ "${mode}" != "single" ]]; then
    echo "[lib_server] HALO Phase 1 is single-instance only — flag will be passed but the controller is a no-op outside NULL disaggregation" >&2
  fi

  _halo_out+=(--halo-enabled)
  if [[ -n "${SGLANG_HALO_DEFAULT_SLO:-}" ]]; then
    _halo_out+=(--halo-default-slo "${SGLANG_HALO_DEFAULT_SLO}")
  fi
  if [[ -n "${SGLANG_HALO_TICK_INTERVAL_MS:-}" ]]; then
    _halo_out+=(--halo-tick-interval-ms "${SGLANG_HALO_TICK_INTERVAL_MS}")
  fi
  if [[ -n "${SGLANG_HALO_AGGREGATOR:-}" ]]; then
    _halo_out+=(--halo-aggregator "${SGLANG_HALO_AGGREGATOR}")
  fi
  if [[ -n "${SGLANG_HALO_JOB_LOG:-}" ]]; then
    _halo_out+=(--halo-job-log "${SGLANG_HALO_JOB_LOG}")
  fi
  if [[ -n "${SGLANG_HALO_PREFILL_COST_MODEL:-}" ]]; then
    _halo_out+=(--halo-prefill-cost-model-path "${SGLANG_HALO_PREFILL_COST_MODEL}")
  fi
  if [[ -n "${SGLANG_HALO_TBT_COST_MODEL:-}" ]]; then
    _halo_out+=(--halo-tbt-cost-model-path "${SGLANG_HALO_TBT_COST_MODEL}")
  fi
  # Step cost model — supersedes the legacy two-model pair when set.
  if [[ -n "${SGLANG_HALO_STEP_COST_MODEL:-}" ]]; then
    _halo_out+=(--halo-step-cost-model-path "${SGLANG_HALO_STEP_COST_MODEL}")
  fi
  if [[ -n "${SGLANG_HALO_JOB_LOG_INTERVAL_SECONDS:-}" ]]; then
    _halo_out+=(--halo-job-log-interval-seconds "${SGLANG_HALO_JOB_LOG_INTERVAL_SECONDS}")
  fi
  if [[ -n "${SGLANG_HALO_QUIESCENT_TIMEOUT_SECONDS:-}" ]]; then
    _halo_out+=(--halo-quiescent-timeout-seconds "${SGLANG_HALO_QUIESCENT_TIMEOUT_SECONDS}")
  fi
  # Phase 2 admission control.
  if [[ -n "${SGLANG_HALO_ADMISSION_MODE:-}" && "${SGLANG_HALO_ADMISSION_MODE}" != "off" ]]; then
    _halo_out+=(--halo-admission-mode "${SGLANG_HALO_ADMISSION_MODE}")
  fi
  if [[ -n "${SGLANG_HALO_ADMISSION_VIOLATION_THRESHOLD:-}" ]]; then
    _halo_out+=(--halo-admission-violation-threshold "${SGLANG_HALO_ADMISSION_VIOLATION_THRESHOLD}")
  fi
  if [[ -n "${SGLANG_HALO_ADMISSION_LOOKAHEAD_HORIZON_SEC:-}" ]]; then
    _halo_out+=(--halo-admission-lookahead-horizon-sec "${SGLANG_HALO_ADMISSION_LOOKAHEAD_HORIZON_SEC}")
  fi
  if [[ "${SGLANG_HALO_ADMISSION_DRY_RUN:-0}" == "1" ]]; then
    _halo_out+=(--halo-admission-dry-run)
  fi
  if [[ -n "${SGLANG_HALO_ADMISSION_DECISION_LOG:-}" ]]; then
    _halo_out+=(--halo-admission-decision-log "${SGLANG_HALO_ADMISSION_DECISION_LOG}")
  fi
}

# launch_server <cmd...>
#
# Prints a quoted command line and execs it (replacing the shell).
launch_server() {
  local caller="${BASH_SOURCE[1]##*/}"
  echo "[${caller%.sh}] launching:"
  printf ' %q' "$@"
  echo
  # The server's HTTP port is already passed via --port. Do not leak
  # SGLANG_PORT into Python because sglang.srt.utils.network.get_open_port()
  # treats it as a preferred internal IPC port, which can make the server
  # bind its own HTTP port before uvicorn starts.
  unset SGLANG_PORT
  exec "$@"
}
