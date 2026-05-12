#!/usr/bin/env bash

# Variables shared across all profiles (single + PD).
# Do not put mode-specific knobs here.

ENV_COMMON_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Paths
export SGLANG_REPO_ROOT="${SGLANG_REPO_ROOT:-$(cd "${ENV_COMMON_SCRIPT_DIR}/.." && pwd)}"
export SGLANG_MS_DEV_DIR="${SGLANG_MS_DEV_DIR:-${SGLANG_REPO_ROOT}/ms_dev}"
export SGLANG_RUNTIME_DIR="${SGLANG_RUNTIME_DIR:-${SGLANG_MS_DEV_DIR}/runtime}"
export SGLANG_VENV_DIR="${SGLANG_VENV_DIR:-${SGLANG_REPO_ROOT}/.venv}"

# Python / venv
export PYTHON_BIN="${PYTHON_BIN:-${SGLANG_VENV_DIR}/bin/python3}"
export PIP_BIN="${PIP_BIN:-${SGLANG_VENV_DIR}/bin/pip}"

# Hugging Face cache
export HF_HOME="${HF_HOME:-${SGLANG_REPO_ROOT}/.hf_home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"

# Model / server config shared by single and each PD role
export SGLANG_MODEL_PATH="${SGLANG_MODEL_PATH:-meta-llama/Llama-3.3-70B-Instruct}"
# export SGLANG_MODEL_PATH="${SGLANG_MODEL_PATH:-meta-llama/Meta-Llama-3.1-70B-Instruct}"
export SGLANG_HOST="${SGLANG_HOST:-0.0.0.0}"
export SGLANG_PORT="${SGLANG_PORT:-31000}"
export SGLANG_ATTENTION_BACKEND="${SGLANG_ATTENTION_BACKEND:-triton}"
export SGLANG_SAMPLING_BACKEND="${SGLANG_SAMPLING_BACKEND:-pytorch}"
export SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.85}"
# export SGLANG_SERVE_EXTRA_ARGS="${SGLANG_SERVE_EXTRA_ARGS:-}"
# export SGLANG_SERVE_EXTRA_ARGS="${SGLANG_SERVE_EXTRA_ARGS:-"--chunked-prefill-size 4096 --enable-mixed-chunk"}"

# 아래 옵션은 런타임에 cuda graph recompile이 너무 자주 발생함. 연산이 멈춤.
# export SGLANG_SERVE_EXTRA_ARGS="${SGLANG_SERVE_EXTRA_ARGS:-"--enable-mixed-chunk --disable-piecewise-cuda-graph"}"

# Logging / tracing
export SGLANG_LOG_REQUESTS_LEVEL="${SGLANG_LOG_REQUESTS_LEVEL:-1}"
export SGLANG_LOG_REQUESTS_FORMAT="${SGLANG_LOG_REQUESTS_FORMAT:-json}"
export SGLANG_OTLP_TRACES_ENDPOINT="${SGLANG_OTLP_TRACES_ENDPOINT:-localhost:4317}"
export SGLANG_OTLP_EXPORTER_SCHEDULE_DELAY_MILLIS="${SGLANG_OTLP_EXPORTER_SCHEDULE_DELAY_MILLIS:-500}"
export SGLANG_OTLP_EXPORTER_MAX_EXPORT_BATCH_SIZE="${SGLANG_OTLP_EXPORTER_MAX_EXPORT_BATCH_SIZE:-64}"

# SGLANG_LOG_SCHEDULER_STATUS_TARGET must NOT be exported unconditionally:
# sglang errors out if it is set without --enable-metrics. It is exported
# inside lib_server.sh::append_obs_args only when metrics is enabled.
export SGLANG_ENABLE_METRICS_FOR_ALL_SCHEDULERS="${SGLANG_ENABLE_METRICS_FOR_ALL_SCHEDULERS:-0}"

# HF token
# export HF_TOKEN="${HF_TOKEN:-hf_xxx}"
export HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-${HF_TOKEN:-}}"

# Mooncake TCP
export MC_TCP_ENABLE_CONNECTION_POOL="${MC_TCP_ENABLE_CONNECTION_POOL:-1}"

# Admission control (Mooncake-style predictive SLO admission).
# Off by default — set TTFT/TBT SLO to enable. See ms_dev/CLAUDE.md.
# These env vars are translated to --admission-* CLI flags by
# lib_server.sh::append_admission_args.
export SGLANG_ADMISSION_TTFT_SLO_MS="${SGLANG_ADMISSION_TTFT_SLO_MS:-}"
export SGLANG_ADMISSION_TBT_SLO_MS="${SGLANG_ADMISSION_TBT_SLO_MS:-}"
# Ratio SLOs — per-request slowdown bound vs solo-run baseline (e.g. 2.0 = "no
# more than 2x slower than running alone"). Either or both can be set together
# with the absolute SLOs above; first violation rejects.
export SGLANG_ADMISSION_TTFT_SLO_RATIO="${SGLANG_ADMISSION_TTFT_SLO_RATIO:-}"
export SGLANG_ADMISSION_TBT_SLO_RATIO="${SGLANG_ADMISSION_TBT_SLO_RATIO:-}"
export SGLANG_ADMISSION_PREFILL_COST_MODEL="${SGLANG_ADMISSION_PREFILL_COST_MODEL:-}"
export SGLANG_ADMISSION_TBT_COST_MODEL="${SGLANG_ADMISSION_TBT_COST_MODEL:-}"
export SGLANG_ADMISSION_TBT_EWMA_ALPHA="${SGLANG_ADMISSION_TBT_EWMA_ALPHA:-}"
export SGLANG_ADMISSION_TBT_REACTIVE_RATIO="${SGLANG_ADMISSION_TBT_REACTIVE_RATIO:-}"
export SGLANG_ADMISSION_DRY_RUN="${SGLANG_ADMISSION_DRY_RUN:-0}"
export SGLANG_ADMISSION_DECISION_LOG="${SGLANG_ADMISSION_DECISION_LOG:-}"

# HALO: Project Halo Phase 1 — job-level slowdown tracking.
# Off by default. See managers/halo/CLAUDE.md and ms_dev/halo_dev/CLAUDE.md.
# Translated to --halo-* CLI flags by lib_server.sh::append_halo_args.
# To turn on for an experiment, set SGLANG_HALO_ENABLED=1 (one-off) or
# source ms_dev/experiments/halo_observe_only.sh.
export SGLANG_HALO_ENABLED="${SGLANG_HALO_ENABLED:-0}"
export SGLANG_HALO_DEFAULT_SLO="${SGLANG_HALO_DEFAULT_SLO:-}"
export SGLANG_HALO_TICK_INTERVAL_MS="${SGLANG_HALO_TICK_INTERVAL_MS:-}"
export SGLANG_HALO_AGGREGATOR="${SGLANG_HALO_AGGREGATOR:-}"
export SGLANG_HALO_JOB_LOG="${SGLANG_HALO_JOB_LOG:-}"
export SGLANG_HALO_PREFILL_COST_MODEL="${SGLANG_HALO_PREFILL_COST_MODEL:-}"
export SGLANG_HALO_TBT_COST_MODEL="${SGLANG_HALO_TBT_COST_MODEL:-}"

# Some environments mount /tmp with noexec, which breaks Triton /
# torchinductor when they try to mmap compiled .so files. Redirect their
# caches under the runtime dir (which lives on the repo FS and allows exec).
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${SGLANG_RUNTIME_DIR}/cache/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${SGLANG_RUNTIME_DIR}/cache/torchinductor}"

# sglang's numa-bind v2 (default True since 0.5.x) writes a numactl wrapper
# script into /tmp and points multiprocessing.spawn at it. With /tmp mounted
# noexec, every spawned scheduler/detokenizer dies silently with exit 255 and
# no traceback. Disable v2 to fall back to in-process libnuma binding.
export SGLANG_NUMA_BIND_V2="${SGLANG_NUMA_BIND_V2:-0}"

mkdir -p "${SGLANG_RUNTIME_DIR}"
mkdir -p "${HF_HOME}"
mkdir -p "${HF_HUB_CACHE}"
mkdir -p "${TRITON_CACHE_DIR}"
mkdir -p "${TORCHINDUCTOR_CACHE_DIR}"
