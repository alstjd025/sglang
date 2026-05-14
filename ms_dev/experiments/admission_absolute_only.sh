#!/usr/bin/env bash
# Admission control with absolute (ms) SLOs only.
#
# All 3 stages active: TTFT predicted / TBT predicted / TBT reactive EWMA.
# Use when you have a fixed user-facing SLA expressed in milliseconds.
#
# Usage:
#   source ms_dev/experiments/admission_absolute_only.sh
#   python3 ms_dev/expctl/server_run_experiment.py --mode single

# Resolve repo root from this script's own location so cwd doesn't matter.
_self="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SGLANG_REPO_ROOT="${SGLANG_REPO_ROOT:-$(cd "$_self/../.." && pwd)}"

export SGLANG_ADMISSION_PREFILL_COST_MODEL="${SGLANG_REPO_ROOT}/ms_dev/runtime/cost_models/prefill_llama3-70b_b200x4.json"
export SGLANG_ADMISSION_TBT_COST_MODEL="${SGLANG_REPO_ROOT}/ms_dev/runtime/cost_models/tbt_llama3-70b_b200x4.json"

export SGLANG_ADMISSION_TTFT_SLO_MS=5000
export SGLANG_ADMISSION_TBT_SLO_MS=200
unset SGLANG_ADMISSION_TTFT_SLO_RATIO
unset SGLANG_ADMISSION_TBT_SLO_RATIO

export SGLANG_ADMISSION_DRY_RUN=0
unset SGLANG_ADMISSION_DECISION_LOG

echo "[experiments/admission_absolute_only] ttft_slo_ms=${SGLANG_ADMISSION_TTFT_SLO_MS:-unset} ttft_ratio=${SGLANG_ADMISSION_TTFT_SLO_RATIO:-unset} tbt_slo_ms=${SGLANG_ADMISSION_TBT_SLO_MS:-unset} tbt_ratio=${SGLANG_ADMISSION_TBT_SLO_RATIO:-unset} dry_run=${SGLANG_ADMISSION_DRY_RUN:-0}"
