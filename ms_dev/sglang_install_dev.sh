#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

cd "${SGLANG_REPO_ROOT}"

"${PYTHON_BIN}" --version
"${PIP_BIN}" --version

"${PIP_BIN}" install --upgrade pip
"${PIP_BIN}" install -e "python"
"${PIP_BIN}" install -U huggingface_hub

echo "[install_dev] done"
echo "[install_dev] checking sglang..."
"${PYTHON_BIN}" -m sglang.launch_server --help >/dev/null
echo "[install_dev] sglang import ok"
