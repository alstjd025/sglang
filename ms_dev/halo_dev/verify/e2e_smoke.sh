#!/usr/bin/env bash
# HALO E2E smoke — exercises POST /halo/programs + chat.completions + /server_info
# against a running sglang server. Validates:
#   - register_program HTTP RTT (verification #3, E2E side)
#   - multi-call job: slowdown_max/mean aggregate across N requests (#2)
#   - admission gate behaviors: HTTP 200 / 400 / 409 paths
#
# Assumes the server is already up. Start it in another terminal with:
#   source ms_dev/experiments/halo_observe_only.sh
#   bash ms_dev/start_server_no_pd.sh
#
# Usage:
#   bash ms_dev/halo_dev/verify/e2e_smoke.sh [BASE_URL]   # default http://127.0.0.1:31000

set -u

BASE_URL="${1:-http://127.0.0.1:${SGLANG_PORT:-31000}}"
JOB_ID="halo-smoke-$(date +%s)"
SLO=5.0
N_CALLS=4
MODEL_NAME="${SGLANG_MODEL_PATH:-meta-llama/Llama-3.3-70B-Instruct}"

c_ok()   { printf '\033[32m%s\033[0m\n' "$1"; }
c_fail() { printf '\033[31m%s\033[0m\n' "$1"; }
c_info() { printf '\033[36m%s\033[0m\n' "$1"; }
h2()     { printf '\n\033[1m== %s ==\033[0m\n' "$1"; }

curl_status() { # echoes the HTTP code only, body to stderr
  curl -sS -o /tmp/halo_smoke_body.$$ -w '%{http_code}' "$@" 2>/dev/null
  echo
}

require_status() { # require_status <expected> <actual> <label>
  if [[ "$2" == "$1" ]]; then
    c_ok "  PASS  $3 → HTTP $2"
  else
    c_fail "  FAIL  $3 → HTTP $2 (expected $1)"
    echo "        body:"
    sed 's/^/        | /' /tmp/halo_smoke_body.$$ | head -5
  fi
}

cleanup() { rm -f /tmp/halo_smoke_body.$$ /tmp/halo_smoke_meta.$$ 2>/dev/null || true; }
trap cleanup EXIT

h2 "Pre-flight — server reachability"
hc=$(curl_status "$BASE_URL/health")
require_status "200" "$hc" "/health"

h2 "POST /halo/programs (fresh) — should be 200"
t0=$(date +%s%N)
hc=$(curl_status -X POST "$BASE_URL/halo/programs" \
    -H 'Content-Type: application/json' \
    -d "{\"job_id\":\"$JOB_ID\",\"slo\":$SLO,\"total_calls\":$N_CALLS,
         \"stage_sequence\":[\"UNDERSTAND\",\"LOCATE\",\"PLAN\",\"VERIFY\"]}")
t1=$(date +%s%N)
register_us=$(( (t1 - t0) / 1000 ))
require_status "200" "$hc" "register fresh job_id=$JOB_ID"
c_info "        register HTTP RTT ≈ ${register_us} µs"
cat /tmp/halo_smoke_body.$$

h2 "POST /halo/programs (duplicate) — should be 409"
hc=$(curl_status -X POST "$BASE_URL/halo/programs" \
    -H 'Content-Type: application/json' \
    -d "{\"job_id\":\"$JOB_ID\",\"slo\":$SLO}")
require_status "409" "$hc" "duplicate register"

h2 "POST /halo/programs (missing slo) — should be 400 MISSING_FIELDS"
hc=$(curl_status -X POST "$BASE_URL/halo/programs" \
    -H 'Content-Type: application/json' -d "{\"job_id\":\"only-id\"}")
require_status "400" "$hc" "missing-slo register"

h2 "chat.completions WITHOUT halo_job_id — should reject 400 HALO_NO_JOB_ID"
hc=$(curl_status -X POST "$BASE_URL/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],
         \"max_tokens\":4,\"stream\":false}")
require_status "400" "$hc" "missing halo_job_id"

h2 "chat.completions with UNREGISTERED halo_job_id — should reject 400 HALO_PROGRAM_NOT_REGISTERED"
hc=$(curl_status -X POST "$BASE_URL/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],
         \"max_tokens\":4,\"stream\":false,
         \"halo_job_id\":\"ghost-$RANDOM\",\"halo_slo\":3.0}")
require_status "400" "$hc" "unregistered job_id"

h2 "chat.completions REGISTERED — $N_CALLS sequential calls"
for i in $(seq 1 $N_CALLS); do
  t0=$(date +%s%N)
  hc=$(curl_status -X POST "$BASE_URL/v1/chat/completions" \
      -H 'Content-Type: application/json' \
      -d "{\"model\":\"$MODEL_NAME\",
           \"messages\":[{\"role\":\"user\",\"content\":\"Call $i: write one short sentence.\"}],
           \"max_tokens\":32,\"stream\":false,
           \"halo_job_id\":\"$JOB_ID\",\"halo_slo\":$SLO}")
  t1=$(date +%s%N)
  call_ms=$(( (t1 - t0) / 1000000 ))
  if [[ "$hc" == "200" ]]; then
    c_ok "  PASS  call $i/$N_CALLS → HTTP 200  (${call_ms} ms)"
  else
    c_fail "  FAIL  call $i/$N_CALLS → HTTP $hc"
    sed 's/^/        | /' /tmp/halo_smoke_body.$$ | head -3
  fi
done

h2 "GET /server_info — halo_state snapshot"
curl -sS "$BASE_URL/server_info" > /tmp/halo_smoke_meta.$$ 2>/dev/null
if command -v jq >/dev/null 2>&1; then
  jq '.internal_states[0].halo_state // .halo_state' /tmp/halo_smoke_meta.$$ 2>/dev/null \
    || jq '.halo_state' /tmp/halo_smoke_meta.$$
else
  python3 -c "
import json, sys
d = json.load(open('/tmp/halo_smoke_meta.$$'))
state = d.get('halo_state') or (d.get('internal_states') or [{}])[0].get('halo_state')
print(json.dumps(state, indent=2, ensure_ascii=False))
"
fi

h2 "Summary"
echo "  job_id=$JOB_ID  expected total_request_number ≥ $N_CALLS"
echo "  look at the halo_state snapshot above:"
echo "    - jobs[*].job_id == \"$JOB_ID\""
echo "    - jobs[*].total_request_number == $N_CALLS  (multi-call aggregation, verification #2)"
echo "    - jobs[*].slowdown_max / slowdown_mean updated by the 100ms sweep"
echo "    - jobs[*].from_program == true  (Option A pre-registration confirmed)"
