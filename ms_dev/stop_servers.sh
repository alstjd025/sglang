#!/usr/bin/env bash
# Stop sglang servers / routers and free their known ports.
#
# Targets:
#   1. processes whose cmdline matches `sglang.launch_server` or
#      `sglang_router.launch_router`
#   2. anyone listening on the ports declared in env.sh
#      (SGLANG_PORT, SGLANG_PD_*, SGLANG_ROUTER_PROMETHEUS_PORT, bootstrap)
#
# Usage:
#   bash ms_dev/stop_servers.sh
#   bash ms_dev/stop_servers.sh --port 31000   # extra port (repeatable)
#   bash ms_dev/stop_servers.sh --dry-run
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# Load BOTH profiles so every known port is covered regardless of what was running.
SGLANG_ENV_PROFILE=all source "${SCRIPT_DIR}/env.sh" >/dev/null 2>&1 || true

default_ports=(
  "${SGLANG_PORT:-}"
  "${SGLANG_PD_PREFILL_PORT:-}"
  "${SGLANG_PD_DECODE_PORT:-}"
  "${SGLANG_PD_ROUTER_PORT:-}"
  "${SGLANG_ROUTER_PROMETHEUS_PORT:-}"
  "${SGLANG_DISAGGREGATION_BOOTSTRAP_PORT:-8998}"
)

extra_ports=()
timeout_s=8
dry_run=0
escalate=0

while (( $# )); do
  case "$1" in
    -p|--port)     extra_ports+=("$2"); shift 2 ;;
    -t|--timeout)  timeout_s="$2"; shift 2 ;;
    -n|--dry-run)  dry_run=1; shift ;;
    -e|--escalate) escalate=1; shift ;;
    -h|--help)
      cat <<EOF
Usage: $(basename "$0") [options]
  -p, --port N       Additional port to free (repeatable)
  -t, --timeout SEC  Wait this long between SIGTERM and SIGKILL (default 8)
  -n, --dry-run      Show what would be killed, do not kill
  -e, --escalate     For ports held by another user, retry via gcsudo fuser -k
EOF
      exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

# Dedupe / drop empties while preserving order
mapfile -t all_ports < <(
  printf '%s\n' "${default_ports[@]}" "${extra_ports[@]-}" \
    | awk 'NF && !seen[$0]++'
)

patterns=( 'sglang\.launch_server' 'sglang_router\.launch_router' )

send_signal() {
  local sig="$1" pid="$2" note="${3-}"
  local cmd
  cmd=$(ps -o cmd= -p "$pid" 2>/dev/null | head -c 100 || true)
  echo "  [${sig}] pid=${pid} ${note} ${cmd}"
  if (( dry_run )); then
    return
  fi
  kill "-${sig}" "$pid" 2>/dev/null || true
}

collect_pattern_pids() {
  local pat="$1"
  pgrep -f "$pat" 2>/dev/null || true
}

collect_port_pids() {
  local port="$1"
  ss -ltnpH "sport = :${port}" 2>/dev/null \
    | grep -oP 'pid=\K[0-9]+' \
    | sort -u
}

echo "[stop_servers] SIGTERM matching processes:"
any_term=0
for pat in "${patterns[@]}"; do
  while IFS= read -r pid; do
    [[ -z "$pid" ]] && continue
    send_signal TERM "$pid" "(pattern: $pat)"
    any_term=1
  done < <(collect_pattern_pids "$pat")
done

echo "[stop_servers] SIGTERM port holders:"
for port in "${all_ports[@]}"; do
  while IFS= read -r pid; do
    [[ -z "$pid" ]] && continue
    send_signal TERM "$pid" "(port: $port)"
    any_term=1
  done < <(collect_port_pids "$port")
done

if (( any_term )) && ! (( dry_run )); then
  echo "[stop_servers] waiting up to ${timeout_s}s for graceful exit..."
  deadline=$(( $(date +%s) + timeout_s ))
  while (( $(date +%s) < deadline )); do
    alive=0
    for pat in "${patterns[@]}"; do
      if pgrep -f "$pat" >/dev/null 2>&1; then alive=1; break; fi
    done
    if ! (( alive )); then
      for port in "${all_ports[@]}"; do
        if ss -ltnH "sport = :${port}" 2>/dev/null | grep -q .; then
          alive=1; break
        fi
      done
    fi
    if ! (( alive )); then break; fi
    sleep 0.5
  done

  echo "[stop_servers] SIGKILL survivors:"
  for pat in "${patterns[@]}"; do
    while IFS= read -r pid; do
      [[ -z "$pid" ]] && continue
      send_signal KILL "$pid" "(survivor pattern: $pat)"
    done < <(collect_pattern_pids "$pat")
  done
  for port in "${all_ports[@]}"; do
    while IFS= read -r pid; do
      [[ -z "$pid" ]] && continue
      send_signal KILL "$pid" "(survivor port: $port)"
    done < <(collect_port_pids "$port")
  done
fi

echo "[stop_servers] final port status:"
held_without_pid_ports=()
for port in "${all_ports[@]}"; do
  if ss -ltnH "sport = :${port}" 2>/dev/null | grep -q .; then
    if ss -ltnpH "sport = :${port}" 2>/dev/null | grep -q 'pid='; then
      echo "  :${port}  IN USE (pid visible — kill may have failed)"
    else
      echo "  :${port}  IN USE (owner pid hidden — likely another user or netns)"
      held_without_pid_ports+=("$port")
    fi
  else
    echo "  :${port}  free"
  fi
done

if (( ${#held_without_pid_ports[@]} > 0 )) && (( escalate )) && ! (( dry_run )); then
  if ! command -v gcsudo >/dev/null 2>&1; then
    echo "[stop_servers] --escalate requested but gcsudo not found on PATH." >&2
  else
    echo "[stop_servers] escalating via gcsudo fuser -k ..."
    for port in "${held_without_pid_ports[@]}"; do
      echo "  gcsudo fuser -k -TERM ${port}/tcp"
      gcsudo fuser -k -TERM "${port}/tcp" 2>/dev/null || true
    done
    sleep 2
    for port in "${held_without_pid_ports[@]}"; do
      if ss -ltnH "sport = :${port}" 2>/dev/null | grep -q .; then
        echo "  gcsudo fuser -k -KILL ${port}/tcp"
        gcsudo fuser -k -KILL "${port}/tcp" 2>/dev/null || true
      fi
    done

    echo "[stop_servers] post-escalation port status:"
    for port in "${held_without_pid_ports[@]}"; do
      if ss -ltnH "sport = :${port}" 2>/dev/null | grep -q .; then
        echo "  :${port}  STILL IN USE"
      else
        echo "  :${port}  free"
      fi
    done
  fi
elif (( ${#held_without_pid_ports[@]} > 0 )); then
  cat >&2 <<'EOF'

[stop_servers] Some ports are held by processes this user cannot see/kill.
  Likely causes:
    - owned by another UID (try:  gcsudo fuser -k <port>/tcp)
    - inside another container / network namespace
    - stuck TIME_WAIT (usually resolves in ~60s; use SO_REUSEADDR or a different port)
  To retry with privilege escalation in one go:
    bash ms_dev/stop_servers.sh --escalate
EOF
fi

