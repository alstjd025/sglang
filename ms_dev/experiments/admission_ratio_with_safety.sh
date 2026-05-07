#!/usr/bin/env bash
# Admission control with RATIO SLOs + loose absolute caps (★ recommended).
#
# Ratio SLOs do the day-to-day work (per-request fairness, adapts to sizes).
# Absolute caps are loose enough that they only fire under catastrophic drift
# of the cost model — Stage 3 reactive EWMA is active and serves as the
# safety net.
#
# Usage:
#   source ms_dev/experiments/admission_ratio_with_safety.sh
#   python3 ms_dev/expctl/run_experiment.py --mode single

export SGLANG_ADMISSION_PREFILL_COST_MODEL="${SGLANG_REPO_ROOT:-$(pwd)}/ms_dev/runtime/cost_models/prefill_llama3-70b_b200x4.json"
export SGLANG_ADMISSION_TBT_COST_MODEL="${SGLANG_REPO_ROOT:-$(pwd)}/ms_dev/runtime/cost_models/tbt_llama3-70b_b200x4.json"

# Primary policy: ratio.
export SGLANG_ADMISSION_TTFT_SLO_RATIO=2.0
export SGLANG_ADMISSION_TBT_SLO_RATIO=3.0

# Loose hard caps as catastrophic-prevention safety net (also activates Stage 3).
export SGLANG_ADMISSION_TTFT_SLO_MS=30000
export SGLANG_ADMISSION_TBT_SLO_MS=500

export SGLANG_ADMISSION_DRY_RUN=0
unset SGLANG_ADMISSION_DECISION_LOG

echo "[experiments/admission_ratio_with_safety] configured: ttft_ratio=2.0+30s tbt_ratio=3.0+500ms"
