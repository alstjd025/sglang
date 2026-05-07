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
#   python3 ms_dev/expctl/run_experiment.py --mode single

# Cost models (regenerate via tools/admission_control/fit_cost_model.py if
# the model / GPU / TP / quantization / kernel cache changes).
export SGLANG_ADMISSION_PREFILL_COST_MODEL="${SGLANG_REPO_ROOT:-$(pwd)}/ms_dev/runtime/cost_models/prefill_llama3-70b_b200x4.json"
export SGLANG_ADMISSION_TBT_COST_MODEL="${SGLANG_REPO_ROOT:-$(pwd)}/ms_dev/runtime/cost_models/tbt_llama3-70b_b200x4.json"

# Ratio SLOs only — no absolute caps.
unset SGLANG_ADMISSION_TTFT_SLO_MS
unset SGLANG_ADMISSION_TBT_SLO_MS
export SGLANG_ADMISSION_TTFT_SLO_RATIO=2.0
export SGLANG_ADMISSION_TBT_SLO_RATIO=3.0

# Dry-run off — actually reject. Switch to 1 when tuning.
export SGLANG_ADMISSION_DRY_RUN=0

# Decision log path is auto-set to <session_dir>/admission_decisions.jsonl
# by run_experiment.py; leave SGLANG_ADMISSION_DECISION_LOG unset.
unset SGLANG_ADMISSION_DECISION_LOG

echo "[experiments/admission_ratio_only] configured: ttft_ratio=2.0 tbt_ratio=3.0 dry_run=0"
