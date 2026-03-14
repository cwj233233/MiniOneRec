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
if [[ -f "${MODEL_DIR}/config.json" ]] && ls "${MODEL_DIR}"/*.safetensors >/dev/null 2>&1; then
  echo "Model already exists at ${MODEL_DIR}, skip download."
  if [[ -f "${LEGACY_MODEL_DIR}/config.json" ]] && ls "${LEGACY_MODEL_DIR}"/*.safetensors >/dev/null 2>&1; then
    echo "Removing duplicate legacy model cache at ${LEGACY_MODEL_DIR} ..."
    rm -rf "${LEGACY_MODEL_DIR}"
    rmdir --ignore-fail-on-non-empty "$(dirname "${LEGACY_MODEL_DIR}")" 2>/dev/null || true
  fi
else
  echo "Model not found. Downloading ${MODEL_ID} from ModelScope to ${MODEL_DIR} ..."
  python - <<PY
import os

try:
    from modelscope import snapshot_download
except ImportError as exc:
    raise SystemExit(
        "ERROR: modelscope is not installed in current environment. "
        "Please re-run step 2 or install requirements.txt first."
    ) from exc

model_id = "${MODEL_ID}"
target_dir = os.path.abspath("${MODEL_DIR}")
os.makedirs(os.path.dirname(target_dir), exist_ok=True)
snapshot_download(model_id=model_id, local_dir=target_dir)
print(f"Model download completed: {target_dir}")
PY
fi

echo "All done."
