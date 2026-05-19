#!/usr/bin/env bash
# HALO: Project Halo base configuration — request-level admission + tracking.
#
# Sets up the *base* Halo state:
#   - SGLANG_HALO_ENABLED=1
#   - SGLANG_HALO_TICK_INTERVAL_MS=100
#   - SGLANG_HALO_STEP_COST_MODEL=<halo_step_split JSON for this hardware>
#       (legacy prefill/tbt pair kept as fallback)
#
# It does NOT set an admission policy. Halo with no policy only *tracks*
# per-request TTFT / TBT / e2e + slowdown and applies the KV cap (if set).
# To enable an admission policy, prepend it to the server-launch command
# after sourcing this script:
#
#   source ms_dev/experiments/halo_base.sh
#   SGLANG_HALO_ADMISSION_POLICY=mooncake \
#       python3 ms_dev/expctl/server_run_experiment.py --mode single
#
# Policies: mooncake | vss | reactive. SGLANG_HALO_SLO_MODE ∈ {ratio, absolute}
# sets how halo_ttft_slo / halo_tbt_slo are interpreted.
#
# Per-request SLOs (halo_ttft_slo / halo_tbt_slo / halo_e2e_slo) travel in
# each request body — the client sets them. There is no job pre-registration
# and no halo_job_id; admission is fully per-request.
#
# The admission decision log is auto-routed to
# <session_dir>/admission_decisions.jsonl by server_run_experiment.py.

# Resolve repo root from this script's own location so cwd doesn't matter.
_self="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SGLANG_REPO_ROOT="${SGLANG_REPO_ROOT:-$(cd "$_self/../.." && pwd)}"

export SGLANG_HALO_ENABLED=1
export SGLANG_HALO_TICK_INTERVAL_MS="${SGLANG_HALO_TICK_INTERVAL_MS:-100}"

# Auto-detect hardware tag for cost-model filename lookup. Cost-model
# coefficients are fit per-(GPU type × TP size), so the same JSON cannot
# be reused across nodes with different GPU counts or GPU types — moving
# from B200x4 to B200x2 means refitting. Override by exporting
# SGLANG_HALO_HW_TAG before sourcing (e.g. "b200x2", "h100x8").
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

# Default cost model: Halo Step Cost Model in SPLIT form. When set, the
# server ignores SGLANG_HALO_PREFILL_COST_MODEL / SGLANG_HALO_TBT_COST_MODEL.
# To opt back into the legacy two-model pair, set SGLANG_HALO_STEP_COST_MODEL=""
# before sourcing.
export SGLANG_HALO_STEP_COST_MODEL="${SGLANG_HALO_STEP_COST_MODEL-${SGLANG_REPO_ROOT}/ms_dev/runtime/cost_models/halo_step_split_llama3-70b_${SGLANG_HALO_HW_TAG}.json}"

# Legacy fallback — used only when SGLANG_HALO_STEP_COST_MODEL is empty.
export SGLANG_HALO_PREFILL_COST_MODEL="${SGLANG_HALO_PREFILL_COST_MODEL:-${SGLANG_REPO_ROOT}/ms_dev/runtime/cost_models/prefill_llama3-70b_${SGLANG_HALO_HW_TAG}.json}"
export SGLANG_HALO_TBT_COST_MODEL="${SGLANG_HALO_TBT_COST_MODEL:-${SGLANG_REPO_ROOT}/ms_dev/runtime/cost_models/tbt_llama3-70b_${SGLANG_HALO_HW_TAG}.json}"

# Warn loudly if the auto-selected step cost model doesn't exist on this
# host — most likely cause is "haven't fit on this hardware yet".
if [[ -n "${SGLANG_HALO_STEP_COST_MODEL}" && ! -f "${SGLANG_HALO_STEP_COST_MODEL}" ]]; then
  echo "[experiments/halo_base] WARN: cost model not found for hw_tag=${SGLANG_HALO_HW_TAG}:" >&2
  echo "[experiments/halo_base]       ${SGLANG_HALO_STEP_COST_MODEL}" >&2
  echo "[experiments/halo_base]       Fit one with tools/halo/fit_halo_cost_model.py, or override SGLANG_HALO_HW_TAG / SGLANG_HALO_STEP_COST_MODEL." >&2
fi

if [[ -n "${SGLANG_HALO_STEP_COST_MODEL}" ]]; then
  echo "[experiments/halo_base] halo_enabled=${SGLANG_HALO_ENABLED} tick_ms=${SGLANG_HALO_TICK_INTERVAL_MS} hw=${SGLANG_HALO_HW_TAG} step_cost=${SGLANG_HALO_STEP_COST_MODEL##*/} (set SGLANG_HALO_ADMISSION_POLICY to enable a policy)"
else
  echo "[experiments/halo_base] halo_enabled=${SGLANG_HALO_ENABLED} tick_ms=${SGLANG_HALO_TICK_INTERVAL_MS} hw=${SGLANG_HALO_HW_TAG} prefill_cost=${SGLANG_HALO_PREFILL_COST_MODEL##*/} tbt_cost=${SGLANG_HALO_TBT_COST_MODEL##*/} (legacy fallback)"
fi
