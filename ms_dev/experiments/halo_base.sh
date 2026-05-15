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

# Auto-detect hardware tag for cost-model filename lookup. Cost-model
# coefficients are fit per-(GPU type × TP size), so the same JSON cannot
# be reused across nodes with different GPU counts or GPU types — moving
# from B200x4 to B200x2 means refitting. Override by exporting
# SGLANG_HALO_HW_TAG before sourcing (e.g. "b200x2", "h100x8").
# Detection: nvidia-smi → "<gpu_short>x<count>", e.g. "b200x4". Falls
# back to "b200x4" if nvidia-smi is unavailable or unparseable.
_halo_detect_hw_tag() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "b200x4"; return
  fi
  local count first_name short
  count="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
  first_name="$(nvidia-smi -L 2>/dev/null | head -1)"
  if [[ -z "$count" || "$count" -le 0 ]]; then
    echo "b200x4"; return
  fi
  if   [[ "$first_name" == *"B200"* ]]; then short="b200"
  elif [[ "$first_name" == *"H200"* ]]; then short="h200"
  elif [[ "$first_name" == *"H100"* ]]; then short="h100"
  elif [[ "$first_name" == *"A100"* ]]; then short="a100"
  elif [[ "$first_name" == *"L40"*  ]]; then short="l40s"
  else short="gpu"
  fi
  echo "${short}x${count}"
}
export SGLANG_HALO_HW_TAG="${SGLANG_HALO_HW_TAG:-$(_halo_detect_hw_tag)}"

# Default cost model: Halo Step Cost Model in SPLIT form. Validated as the
# best-balanced model (closest tail/p90 to ground truth; cliff resolved) —
# see ms_dev/halo_dev/prediction_model.md §17. When this var is set, the
# server logs INFO and ignores SGLANG_HALO_PREFILL_COST_MODEL /
# SGLANG_HALO_TBT_COST_MODEL. To opt back into the legacy two-model pair,
# explicitly set SGLANG_HALO_STEP_COST_MODEL="" before sourcing.
export SGLANG_HALO_STEP_COST_MODEL="${SGLANG_HALO_STEP_COST_MODEL-${SGLANG_REPO_ROOT}/ms_dev/runtime/cost_models/halo_step_split_llama3-70b_${SGLANG_HALO_HW_TAG}.json}"

# Legacy fallback. Used only when SGLANG_HALO_STEP_COST_MODEL is empty;
# otherwise ignored by the server.
export SGLANG_HALO_PREFILL_COST_MODEL="${SGLANG_HALO_PREFILL_COST_MODEL:-${SGLANG_REPO_ROOT}/ms_dev/runtime/cost_models/prefill_llama3-70b_${SGLANG_HALO_HW_TAG}.json}"
export SGLANG_HALO_TBT_COST_MODEL="${SGLANG_HALO_TBT_COST_MODEL:-${SGLANG_REPO_ROOT}/ms_dev/runtime/cost_models/tbt_llama3-70b_${SGLANG_HALO_HW_TAG}.json}"

# Job log path is auto-routed to <session_dir>/halo_jobs.jsonl by
# ms_dev/expctl/server_run_experiment.py; leave unset here.
unset SGLANG_HALO_JOB_LOG

# Warn loudly if the auto-selected step cost model doesn't exist on this
# host — most likely cause is "haven't fit on this hardware yet". Server
# would die at startup with a less obvious error; surface it earlier.
if [[ -n "${SGLANG_HALO_STEP_COST_MODEL}" && ! -f "${SGLANG_HALO_STEP_COST_MODEL}" ]]; then
  echo "[experiments/halo_base] WARN: cost model not found for hw_tag=${SGLANG_HALO_HW_TAG}:" >&2
  echo "[experiments/halo_base]       ${SGLANG_HALO_STEP_COST_MODEL}" >&2
  echo "[experiments/halo_base]       Fit one with tools/halo/fit_halo_cost_model.py, or override SGLANG_HALO_HW_TAG / SGLANG_HALO_STEP_COST_MODEL." >&2
fi

if [[ -n "${SGLANG_HALO_STEP_COST_MODEL}" ]]; then
  echo "[experiments/halo_base] halo_enabled=${SGLANG_HALO_ENABLED} default_slo=${SGLANG_HALO_DEFAULT_SLO} tick_ms=${SGLANG_HALO_TICK_INTERVAL_MS} hw=${SGLANG_HALO_HW_TAG} step_cost=${SGLANG_HALO_STEP_COST_MODEL##*/}"
else
  echo "[experiments/halo_base] halo_enabled=${SGLANG_HALO_ENABLED} default_slo=${SGLANG_HALO_DEFAULT_SLO} tick_ms=${SGLANG_HALO_TICK_INTERVAL_MS} hw=${SGLANG_HALO_HW_TAG} prefill_cost=${SGLANG_HALO_PREFILL_COST_MODEL##*/} tbt_cost=${SGLANG_HALO_TBT_COST_MODEL##*/} (legacy fallback)"
fi
