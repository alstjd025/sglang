#!/usr/bin/env bash

# Paths
export SGLANG_REPO_ROOT="/workspace/sglang"
export SGLANG_MS_DEV_DIR="${SGLANG_REPO_ROOT}/ms_dev"
export SGLANG_RUNTIME_DIR="${SGLANG_MS_DEV_DIR}/runtime"

# Python / venv
export PYTHON_BIN="/venv/main/bin/python3"
export PIP_BIN="/venv/main/bin/pip"

# Hugging Face cache
export HF_HOME="/workspace/.hf_home"
export HF_HUB_CACHE="${HF_HOME}/hub"

# Runtime dirs
export SGLANG_REQUEST_LOG_DIR="${SGLANG_RUNTIME_DIR}/request_logs"
export SGLANG_REQUEST_METRICS_DIR="${SGLANG_RUNTIME_DIR}/request_metrics"
export SGLANG_CRASH_DUMP_DIR="${SGLANG_RUNTIME_DIR}/crash_dump"

# Model / server config
# export SGLANG_MODEL_PATH="meta-llama/Meta-Llama-3.1-8B-Instruct"
export SGLANG_MODEL_PATH="meta-llama/Meta-Llama-3-8B-Instruct"
export SGLANG_HOST="0.0.0.0"
export SGLANG_ATTENTION_BACKEND="triton"
export SGLANG_SAMPLING_BACKEND="pytorch"
export SGLANG_MEM_FRACTION_STATIC="0.85"

# Optional logging / tracing
export SGLANG_LOG_REQUESTS_LEVEL="1"
export SGLANG_LOG_REQUESTS_FORMAT="json"
export SGLANG_OTLP_TRACES_ENDPOINT="localhost:4317"
export SGLANG_OTLP_EXPORTER_SCHEDULE_DELAY_MILLIS="500"
export SGLANG_OTLP_EXPORTER_MAX_EXPORT_BATCH_SIZE="64"

# PD disaggregation for 1 node / 2 GPUs
export SGLANG_PD_ROUTER_PORT="30000"
export SGLANG_PD_DECODE_PORT="30001"
export SGLANG_PD_PREFILL_PORT="30002"

export SGLANG_PD_PREFILL_BASE_GPU_ID="0"
export SGLANG_PD_DECODE_BASE_GPU_ID="1"

# Router/backend host
# host network 전제면 127.0.0.1 유지해도 충분
export SGLANG_PD_BACKEND_HOST="127.0.0.1"
export SGLANG_PD_ROUTER_HOST="0.0.0.0"

# Optional PD settings
export SGLANG_PD_IB_DEVICE="${SGLANG_PD_IB_DEVICE:-}"
export SGLANG_PD_PREFILL_EXTRA_ARGS="${SGLANG_PD_PREFILL_EXTRA_ARGS:-}"
export SGLANG_PD_DECODE_EXTRA_ARGS="${SGLANG_PD_DECODE_EXTRA_ARGS:-}"
export SGLANG_PD_ROUTER_EXTRA_ARGS="${SGLANG_PD_ROUTER_EXTRA_ARGS:-}"

# Optional: disaggregation timeout 완화
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT="${SGLANG_DISAGGREGATION_WAITING_TIMEOUT:-600}"

# HF token
export HF_TOKEN="${HF_TOKEN:-hf_xxx}"
export HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-$HF_TOKEN}"

mkdir -p "${SGLANG_RUNTIME_DIR}"
mkdir -p "${HF_HOME}"
mkdir -p "${HF_HUB_CACHE}"
mkdir -p "${SGLANG_REQUEST_LOG_DIR}"
mkdir -p "${SGLANG_REQUEST_METRICS_DIR}"
mkdir -p "${SGLANG_CRASH_DUMP_DIR}"