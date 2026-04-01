#!/usr/bin/env bash

# Paths
export SGLANG_REPO_ROOT="${SGLANG_REPO_ROOT:-/workspace/sglang}"
export SGLANG_MS_DEV_DIR="${SGLANG_MS_DEV_DIR:-${SGLANG_REPO_ROOT}/ms_dev}"
export SGLANG_RUNTIME_DIR="${SGLANG_RUNTIME_DIR:-${SGLANG_MS_DEV_DIR}/runtime}"

# Python / venv
export PYTHON_BIN="${PYTHON_BIN:-/venv/main/bin/python3}"
export PIP_BIN="${PIP_BIN:-/venv/main/bin/pip}"

# Hugging Face cache
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"

# Server runtime files
export SGLANG_SERVER_LOG="${SGLANG_SERVER_LOG:-${SGLANG_RUNTIME_DIR}/server.log}"
export SGLANG_SERVER_PID_FILE="${SGLANG_SERVER_PID_FILE:-${SGLANG_RUNTIME_DIR}/server.pid}"

# Model / server config
export SGLANG_MODEL_PATH="${SGLANG_MODEL_PATH:-meta-llama/Llama-3.1-8B-Instruct}"
export SGLANG_HOST="${SGLANG_HOST:-0.0.0.0}"
export SGLANG_PORT="${SGLANG_PORT:-30000}"
export SGLANG_ATTENTION_BACKEND="${SGLANG_ATTENTION_BACKEND:-triton}"
export SGLANG_SAMPLING_BACKEND="${SGLANG_SAMPLING_BACKEND:-pytorch}"

# Extra args for sglang
export SGLANG_SERVE_EXTRA_ARGS="${SGLANG_SERVE_EXTRA_ARGS:-}"

mkdir -p "${SGLANG_RUNTIME_DIR}"
mkdir -p "${HF_HOME}"
mkdir -p "${HF_HUB_CACHE}"
