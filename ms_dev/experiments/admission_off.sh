#!/usr/bin/env bash
# Disable admission control by unsetting every SGLANG_ADMISSION_* env var.
#
# Why: lib_server.sh::append_admission_args adds no --admission-* CLI flags
# when none of the four SLO env vars
# (SGLANG_ADMISSION_TTFT_SLO_MS / TBT_SLO_MS / TTFT_SLO_RATIO / TBT_SLO_RATIO)
# are set, and the controller is then never instantiated. Unsetting all
# SGLANG_ADMISSION_* (not just the four SLOs) also clears the cost-model
# paths and DRY_RUN flag so a leftover combination cannot accidentally
# re-arm Stage 3 or change behavior.
#
# Usage:
#   source ms_dev/experiments/admission_off.sh
#   python3 ms_dev/expctl/run_experiment.py --mode single

_vars="$(compgen -v 2>/dev/null | grep '^SGLANG_ADMISSION_' || true)"
if [[ -n "$_vars" ]]; then
  unset $_vars
fi
unset _vars

echo "[experiments/admission_off] ttft_slo_ms=${SGLANG_ADMISSION_TTFT_SLO_MS:-unset} ttft_ratio=${SGLANG_ADMISSION_TTFT_SLO_RATIO:-unset} tbt_slo_ms=${SGLANG_ADMISSION_TBT_SLO_MS:-unset} tbt_ratio=${SGLANG_ADMISSION_TBT_SLO_RATIO:-unset} dry_run=${SGLANG_ADMISSION_DRY_RUN:-unset}"
