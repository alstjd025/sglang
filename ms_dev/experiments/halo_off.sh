#!/usr/bin/env bash
# HALO: Disable Project Halo (request-level admission control + tracking).
#
# Resets every SGLANG_HALO_* env var to its inert default. After sourcing
# this, lib_server.sh::append_halo_args adds no --halo-* CLI flags and
# the HaloController is never instantiated.
#
# Usage:
#   source ms_dev/experiments/halo_off.sh
#   python3 ms_dev/expctl/server_run_experiment.py --mode single

_vars="$(compgen -v 2>/dev/null | grep '^SGLANG_HALO_' || true)"
if [[ -n "$_vars" ]]; then
  unset $_vars
fi
unset _vars

# Set the master flag explicitly to 0 so the launcher prints a clear state.
export SGLANG_HALO_ENABLED=0

echo "[experiments/halo_off] halo_enabled=${SGLANG_HALO_ENABLED} (all SGLANG_HALO_* cleared)"
