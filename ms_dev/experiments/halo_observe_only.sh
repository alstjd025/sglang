#!/usr/bin/env bash
# HALO: Project Halo Phase 1 — observe-only configuration.
#
# Phase 1 does NOT make admission or scheduling decisions based on jobs.
# It only:
#   - tags each request with halo_job_id + halo_slo (sent by the client)
#   - registers/groups them into Job records (Linux task_struct analog)
#   - every 100ms sweeps the running/waiting set and computes per-job
#     slowdown_max + slowdown_mean vs the solo-run cost-model baseline
#   - exposes the table via /server_info and an optional JSONL log
#
# Reuses the *admission_control* cost models (same files, same schema). The
# JSONL log is auto-routed to <session_dir>/halo_jobs.jsonl by
# ms_dev/expctl/run_experiment.py.
#
# STRICT MODE (Phase 1 Option A + B, decisions Q7 + Q12):
#   * Every LLM request MUST carry halo_job_id in its body — missing field
#     → HTTP 400 (HALO_NO_JOB_ID).
#   * The job_id MUST already be pre-registered via POST /halo/programs
#     before the first LLM call — missing pre-registration → HTTP 400
#     (HALO_PROGRAM_NOT_REGISTERED). Lazy-create fallback is removed.
#
# Client integration recipe (per job, before any LLM call):
#   curl -X POST $BASE_URL/halo/programs -H 'Content-Type: application/json' \
#       -d '{"job_id":"agent-42", "slo":5.0, "total_calls":12,
#            "stage_sequence":["UNDERSTAND","LOCATE",...]}'
# Then issue chat.completions with body fields halo_job_id="agent-42",
# halo_slo=5.0.
#
# Usage:
#   source ms_dev/experiments/halo_observe_only.sh
#   python3 ms_dev/expctl/run_experiment.py --mode single

# Resolve repo root from this script's own location so cwd doesn't matter.
_self="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SGLANG_REPO_ROOT="${SGLANG_REPO_ROOT:-$(cd "$_self/../.." && pwd)}"

export SGLANG_HALO_ENABLED=1
export SGLANG_HALO_DEFAULT_SLO="${SGLANG_HALO_DEFAULT_SLO:-5.0}"
export SGLANG_HALO_TICK_INTERVAL_MS="${SGLANG_HALO_TICK_INTERVAL_MS:-100}"

# Reuse admission_control cost models — same schema, same JSON files.
# Halo's slowdown computation needs both; if either is missing, the sweep
# still runs but contributes no math (job counters / state still work).
export SGLANG_HALO_PREFILL_COST_MODEL="${SGLANG_HALO_PREFILL_COST_MODEL:-${SGLANG_REPO_ROOT}/ms_dev/runtime/cost_models/prefill_llama3-70b_b200x4.json}"
export SGLANG_HALO_TBT_COST_MODEL="${SGLANG_HALO_TBT_COST_MODEL:-${SGLANG_REPO_ROOT}/ms_dev/runtime/cost_models/tbt_llama3-70b_b200x4.json}"

# Job log path is auto-routed to <session_dir>/halo_jobs.jsonl by
# ms_dev/expctl/run_experiment.py; leave unset here.
unset SGLANG_HALO_JOB_LOG

echo "[experiments/halo_observe_only] halo_enabled=${SGLANG_HALO_ENABLED} default_slo=${SGLANG_HALO_DEFAULT_SLO} tick_ms=${SGLANG_HALO_TICK_INTERVAL_MS} prefill_cost=${SGLANG_HALO_PREFILL_COST_MODEL##*/} tbt_cost=${SGLANG_HALO_TBT_COST_MODEL##*/}"
