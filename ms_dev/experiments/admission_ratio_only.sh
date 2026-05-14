#!/usr/bin/env bash
# Admission control with RATIO-only SLOs (no absolute ms caps).
#
# Each request is admitted only if it would not be slowed down beyond Nx
# its solo-run baseline. Adapts to heterogeneous request sizes.
#
# Trade-off: Stage 3 (reactive EWMA safety net) is disabled — there is no
# absolute ms threshold for it to compare against. If cost-model accuracy
# drifts, admission may become too lenient. For production-like runs use
# admission_ratio_with_safety.sh instead.
#
# Usage:
#   source ms_dev/experiments/admission_ratio_only.sh
#   python3 ms_dev/expctl/server_run_experiment.py --mode single

# Cost models (regenerate via tools/admission_control/fit_cost_model.py if
# the model / GPU / TP / quantization / kernel cache changes).
# Resolve repo root from this script's own location so cwd doesn't matter.
_self="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SGLANG_REPO_ROOT="${SGLANG_REPO_ROOT:-$(cd "$_self/../.." && pwd)}"

export SGLANG_ADMISSION_PREFILL_COST_MODEL="${SGLANG_REPO_ROOT}/ms_dev/runtime/cost_models/prefill_llama3-70b_b200x4.json"
export SGLANG_ADMISSION_TBT_COST_MODEL="${SGLANG_REPO_ROOT}/ms_dev/runtime/cost_models/tbt_llama3-70b_b200x4.json"

# Ratio SLOs only — no absolute caps.
unset SGLANG_ADMISSION_TTFT_SLO_MS
unset SGLANG_ADMISSION_TBT_SLO_MS
export SGLANG_ADMISSION_TTFT_SLO_RATIO=5.0
export SGLANG_ADMISSION_TBT_SLO_RATIO=5.0

# Dry-run off — actually reject. Switch to 1 when tuning.
export SGLANG_ADMISSION_DRY_RUN=0

# Decision log path is auto-set to <session_dir>/admission_decisions.jsonl
# by run_experiment.py; leave SGLANG_ADMISSION_DECISION_LOG unset.
unset SGLANG_ADMISSION_DECISION_LOG

echo "[experiments/admission_ratio_only] ttft_slo_ms=${SGLANG_ADMISSION_TTFT_SLO_MS:-unset} ttft_ratio=${SGLANG_ADMISSION_TTFT_SLO_RATIO:-unset} tbt_slo_ms=${SGLANG_ADMISSION_TBT_SLO_MS:-unset} tbt_ratio=${SGLANG_ADMISSION_TBT_SLO_RATIO:-unset} dry_run=${SGLANG_ADMISSION_DRY_RUN:-0}"
