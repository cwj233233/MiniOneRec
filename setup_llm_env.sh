#!/usr/bin/env bash
export PIP_ROOT_USER_ACTION=ignore
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQUIREMENTS_FILE="${PROJECT_ROOT}/requirements.txt"
MODEL_DIR="${PROJECT_ROOT}/models/Qwen2.5-Instruct-0.5b"
MODEL_ID="Qwen/Qwen2.5-0.5B-Instruct"

echo "[1/3] Check python command..."
if ! command -v python >/dev/null 2>&1; then
  echo "ERROR: python not found in current environment."
  exit 1
fi

echo "[2/3] Install project dependencies in current environment..."
python -m pip install --upgrade pip

if [[ ! -f "${REQUIREMENTS_FILE}" ]]; then
  echo "ERROR: requirements file not found at ${REQUIREMENTS_FILE}"
  exit 1
fi
python -m pip install -r "${REQUIREMENTS_FILE}"

echo "[3/3] Check model directory..."
if [[ -d "${MODEL_DIR}" ]] && [[ -n "$(ls -A "${MODEL_DIR}" 2>/dev/null || true)" ]]; then
  echo "Model already exists at ${MODEL_DIR}, skip download."
else
  echo "Model not found. Downloading ${MODEL_ID} to ${MODEL_DIR} ..."
  mkdir -p "${MODEL_DIR}"
  python - <<PY
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="${MODEL_ID}",
    local_dir="${MODEL_DIR}",
    local_dir_use_symlinks=False,
)
print("Model download completed.")
PY
fi

echo "All done."
