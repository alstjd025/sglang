#!/usr/bin/env bash

# Dispatcher for ms_dev environment variables.
#
# Profiles:
#   single : single-server (no PD disaggregation)        -> env.common.sh + env.single.sh
#   pd     : PD disaggregated (prefill / decode / router) -> env.common.sh + env.pd.sh
#   all    : loads both profiles (backward-compat default)
#
# Usage:
#   SGLANG_ENV_PROFILE=single source env.sh
#   SGLANG_ENV_PROFILE=pd     source env.sh
#   source env.sh                 # defaults to "all"
#
# Each launcher script under ms_dev/ sets SGLANG_ENV_PROFILE to the right value
# before sourcing this file, so end-users usually do not need to set it manually.

ENV_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "${ENV_SCRIPT_DIR}/env.common.sh"

case "${SGLANG_ENV_PROFILE:-all}" in
  single)
    source "${ENV_SCRIPT_DIR}/env.single.sh"
    ;;
  pd)
    source "${ENV_SCRIPT_DIR}/env.pd.sh"
    ;;
  all)
    source "${ENV_SCRIPT_DIR}/env.single.sh"
    source "${ENV_SCRIPT_DIR}/env.pd.sh"
    ;;
  *)
    echo "[env.sh] unknown SGLANG_ENV_PROFILE='${SGLANG_ENV_PROFILE}' (valid: single, pd, all)" >&2
    return 1 2>/dev/null || exit 1
    ;;
esac

echo "SGLang env setup done. (profile=${SGLANG_ENV_PROFILE:-all})"
