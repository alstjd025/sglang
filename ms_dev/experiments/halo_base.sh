#!/usr/bin/env bash
# HALO: Project Halo base configuration (R1 slowdown tracking + step cost
# model). Strict mode (Q7+Q12) only — no Phase 2 admission decisions.
#
# Renamed 2026-05-15 from halo_observe_only.sh. The previous name is kept
# as a symlink for backward compatibility with old --sglang-start-cmd
# values still pointing to it.
#
# This wrapper sets up the *base* Halo state:
#   - SGLANG_HALO_ENABLED=1
#   - SGLANG_HALO_DEFAULT_SLO=5.0
#   - SGLANG_HALO_TICK_INTERVAL_MS=100
#   - SGLANG_HALO_STEP_COST_MODEL=<halo_step_split JSON>
#       (legacy two-model paths kept as fallback)
#
# It does NOT touch SGLANG_HALO_ADMISSION_MODE. To enable Phase 2 job-level
# predictive admission, prepend SGLANG_HALO_ADMISSION_MODE=level0 to the
# server-launch command after sourcing this script. Example:
#
#   source ms_dev/experiments/halo_base.sh
#   SGLANG_HALO_ADMISSION_MODE=level0 \
#       python3 ms_dev/expctl/server_run_experiment.py --mode single
#
# What Halo Phase 1 (R1) does once SGLANG_HALO_ENABLED=1:
#   - tags each request with halo_job_id + halo_slo (sent by the client)
#   - registers/groups them into Job records (Linux task_struct analog)
#   - every 100ms sweeps the running/waiting set and computes per-job
#     slowdown_max + slowdown_mean vs the solo-run cost-model baseline
#   - exposes the table via /server_info and an optional JSONL log
#
# STRICT MODE (Phase 1 Option A + B, decisions Q7 + Q12):
#   * Every LLM request MUST carry halo_job_id in its body — missing field
#     → HTTP 400 (HALO_NO_JOB_ID).
#   * The job_id MUST already be pre-registered via POST /halo/programs
#     before the first LLM call — missing pre-registration → HTTP 400
#     (HALO_PROGRAM_NOT_REGISTERED). Lazy-create fallback is removed.
#
# The JSONL log is auto-routed to <session_dir>/halo_jobs.jsonl by
# ms_dev/expctl/server_run_experiment.py.
#
# Client integration recipe (per job, before any LLM call):
#   curl -X POST $BASE_URL/halo/programs -H 'Content-Type: application/json' \
#       -d '{"job_id":"agent-42", "slo":5.0, "total_calls":12,
#            "stage_sequence":["UNDERSTAND","LOCATE",...]}'
# Then issue chat.completions with body fields halo_job_id="agent-42",
# halo_slo=5.0.

# Resolve repo root from this script's own location so cwd doesn't matter.
_self="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SGLANG_REPO_ROOT="${SGLANG_REPO_ROOT:-$(cd "$_self/../.." && pwd)}"

export SGLANG_HALO_ENABLED=1
export SGLANG_HALO_DEFAULT_SLO="${SGLANG_HALO_DEFAULT_SLO:-5.0}"
export SGLANG_HALO_TICK_INTERVAL_MS="${SGLANG_HALO_TICK_INTERVAL_MS:-100}"

# Default cost model: Halo Step Cost Model in SPLIT form. Validated as the
# best-balanced model (closest tail/p90 to ground truth; cliff resolved) —
# see ms_dev/halo_dev/prediction_model.md §17. When this var is set, the
# server logs INFO and ignores SGLANG_HALO_PREFILL_COST_MODEL /
# SGLANG_HALO_TBT_COST_MODEL. To opt back into the legacy two-model pair,
# explicitly set SGLANG_HALO_STEP_COST_MODEL="" before sourcing.
export SGLANG_HALO_STEP_COST_MODEL="${SGLANG_HALO_STEP_COST_MODEL-${SGLANG_REPO_ROOT}/ms_dev/runtime/cost_models/halo_step_split_llama3-70b_b200x4.json}"

# Legacy fallback. Used only when SGLANG_HALO_STEP_COST_MODEL is empty;
# otherwise ignored by the server.
export SGLANG_HALO_PREFILL_COST_MODEL="${SGLANG_HALO_PREFILL_COST_MODEL:-${SGLANG_REPO_ROOT}/ms_dev/runtime/cost_models/prefill_llama3-70b_b200x4.json}"
export SGLANG_HALO_TBT_COST_MODEL="${SGLANG_HALO_TBT_COST_MODEL:-${SGLANG_REPO_ROOT}/ms_dev/runtime/cost_models/tbt_llama3-70b_b200x4.json}"

# Job log path is auto-routed to <session_dir>/halo_jobs.jsonl by
# ms_dev/expctl/server_run_experiment.py; leave unset here.
unset SGLANG_HALO_JOB_LOG

if [[ -n "${SGLANG_HALO_STEP_COST_MODEL}" ]]; then
  echo "[experiments/halo_base] halo_enabled=${SGLANG_HALO_ENABLED} default_slo=${SGLANG_HALO_DEFAULT_SLO} tick_ms=${SGLANG_HALO_TICK_INTERVAL_MS} step_cost=${SGLANG_HALO_STEP_COST_MODEL##*/}"
else
  echo "[experiments/halo_base] halo_enabled=${SGLANG_HALO_ENABLED} default_slo=${SGLANG_HALO_DEFAULT_SLO} tick_ms=${SGLANG_HALO_TICK_INTERVAL_MS} prefill_cost=${SGLANG_HALO_PREFILL_COST_MODEL##*/} tbt_cost=${SGLANG_HALO_TBT_COST_MODEL##*/} (legacy fallback)"
fi
