#!/usr/bin/env bash
# Hybrid: TTFT ratio + absolute TBT SLO with reactive EWMA safety net.
#
# Why: empirical analysis (parallel_tool_lambda_0p2 trace) showed TBT_RATIO
# behaves as a binary classifier on this workload because the TBT cost model's
# prediction is tightly clustered at ~55ms regardless of batch composition.
# Absolute TBT SLO + reactive EWMA bypasses cost-model brittleness for TBT.
#
# Usage:
#   source ms_dev/experiments/admission_ratio_ttft_only.sh
#   python3 ms_dev/expctl/server_run_experiment.py --mode single

# Resolve repo root from this script's own location so cwd doesn't matter.
_self="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SGLANG_REPO_ROOT="${SGLANG_REPO_ROOT:-$(cd "$_self/../.." && pwd)}"

export SGLANG_ADMISSION_PREFILL_COST_MODEL="${SGLANG_REPO_ROOT}/ms_dev/runtime/cost_models/prefill_llama3-70b_b200x4.json"
export SGLANG_ADMISSION_TBT_COST_MODEL="${SGLANG_REPO_ROOT}/ms_dev/runtime/cost_models/tbt_llama3-70b_b200x4.json"

# TTFT: ratio policy (well-behaved on this workload, ~1.5% reject rate).
unset SGLANG_ADMISSION_TTFT_SLO_MS
export SGLANG_ADMISSION_TTFT_SLO_RATIO=3.0

# TBT: absolute SLO instead of ratio. Picked above measured p90 (~100 ms)
# so cost-model TBT_PREDICTED rarely fires on its own; the real protection
# comes from the reactive EWMA safety net (trips at 200 * 0.9 = 180 ms).
export SGLANG_ADMISSION_TBT_SLO_MS=200
unset SGLANG_ADMISSION_TBT_SLO_RATIO

export SGLANG_ADMISSION_DRY_RUN=0
unset SGLANG_ADMISSION_DECISION_LOG

echo "[experiments/admission_ratio_ttft_only] ttft_slo_ms=${SGLANG_ADMISSION_TTFT_SLO_MS:-unset} ttft_ratio=${SGLANG_ADMISSION_TTFT_SLO_RATIO:-unset} tbt_slo_ms=${SGLANG_ADMISSION_TBT_SLO_MS:-unset} tbt_ratio=${SGLANG_ADMISSION_TBT_SLO_RATIO:-unset} dry_run=${SGLANG_ADMISSION_DRY_RUN:-0}"
