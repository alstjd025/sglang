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

# Some environments mount /tmp with noexec, which breaks Triton /
# torchinductor when they try to mmap compiled .so files. Redirect their
# caches under the runtime dir (which lives on the repo FS and allows exec).
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${SGLANG_RUNTIME_DIR}/cache/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${SGLANG_RUNTIME_DIR}/cache/torchinductor}"

mkdir -p "${SGLANG_RUNTIME_DIR}"
mkdir -p "${HF_HOME}"
mkdir -p "${HF_HUB_CACHE}"
mkdir -p "${TRITON_CACHE_DIR}"
mkdir -p "${TORCHINDUCTOR_CACHE_DIR}"
