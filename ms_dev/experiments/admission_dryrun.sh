#!/usr/bin/env bash
# Same configuration as admission_ratio_with_safety.sh but with DRY-RUN ON.
#
# Use this for SLO tuning against real production traffic: every decision
# is logged to <session_dir>/admission_decisions.jsonl, but the controller
# always admits. Replay the log with tools/admission_control/replay_admission.py
# to compute "what would have been rejected" at any candidate SLO combo.
#
# Usage:
#   source ms_dev/experiments/admission_dryrun.sh
#   python3 ms_dev/expctl/run_experiment.py --mode single
#
# Then after the session:
#   python tools/admission_control/replay_admission.py \
#       --decision-log ms_dev/runtime/sessions/<sess>/admission_decisions.jsonl \
#       --ttft-slo 0 --tbt-slo 0 --ttft-slo-ratio 1.5 2.0 3.0 --tbt-slo-ratio 2.0 3.0 5.0

export SGLANG_ADMISSION_PREFILL_COST_MODEL="${SGLANG_REPO_ROOT:-$(pwd)}/ms_dev/runtime/cost_models/prefill_llama3-70b_b200x4.json"
export SGLANG_ADMISSION_TBT_COST_MODEL="${SGLANG_REPO_ROOT:-$(pwd)}/ms_dev/runtime/cost_models/tbt_llama3-70b_b200x4.json"

# Loose initial guesses — replay will find the right ones.
export SGLANG_ADMISSION_TTFT_SLO_RATIO=2.0
export SGLANG_ADMISSION_TBT_SLO_RATIO=3.0
export SGLANG_ADMISSION_TTFT_SLO_MS=30000
export SGLANG_ADMISSION_TBT_SLO_MS=500

# ★ Always admit, log everything.
export SGLANG_ADMISSION_DRY_RUN=1

# decision_log path auto-routed by run_experiment.
unset SGLANG_ADMISSION_DECISION_LOG

echo "[experiments/admission_dryrun] DRY-RUN: decisions logged but never enforced"
