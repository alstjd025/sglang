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
