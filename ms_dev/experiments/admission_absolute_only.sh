#!/usr/bin/env bash
# Admission control with absolute (ms) SLOs only.
#
# All 3 stages active: TTFT predicted / TBT predicted / TBT reactive EWMA.
# Use when you have a fixed user-facing SLA expressed in milliseconds.
#
# Usage:
#   source ms_dev/experiments/admission_absolute_only.sh
#   python3 ms_dev/expctl/run_experiment.py --mode single

export SGLANG_ADMISSION_PREFILL_COST_MODEL="${SGLANG_REPO_ROOT:-$(pwd)}/ms_dev/runtime/cost_models/prefill_llama3-70b_b200x4.json"
export SGLANG_ADMISSION_TBT_COST_MODEL="${SGLANG_REPO_ROOT:-$(pwd)}/ms_dev/runtime/cost_models/tbt_llama3-70b_b200x4.json"

export SGLANG_ADMISSION_TTFT_SLO_MS=5000
export SGLANG_ADMISSION_TBT_SLO_MS=200
unset SGLANG_ADMISSION_TTFT_SLO_RATIO
unset SGLANG_ADMISSION_TBT_SLO_RATIO

export SGLANG_ADMISSION_DRY_RUN=0
unset SGLANG_ADMISSION_DECISION_LOG

echo "[experiments/admission_absolute_only] configured: ttft=5s tbt=200ms"
