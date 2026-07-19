#!/usr/bin/env bash
# Stable executable wrapper for the pinned official codonGPT HF checkpoint.
set -euo pipefail

ROOT="${ROOT:-/home/cunyuliu/mrna_editflow_goal/mrna_editflow}"
ENV_DIR="${CODONGPT_ENV_DIR:-${ROOT}/external_tools/envs/codongpt}"
PYTHON_BIN="${CODONGPT_PYTHON:-${ENV_DIR}/bin/python}"
MODEL_DIR="${CODONGPT_MODEL_DIR:-${ROOT}/external_tools/codonGPT_hf_ee7017c4}"
MANIFEST="${MODEL_DIR}/model_manifest.json"

if [[ "${1:-}" == "--version" ]]; then
  if [[ ! -x "${PYTHON_BIN}" || ! -f "${MANIFEST}" ]]; then
    echo "codonGPT official HF runtime unavailable" >&2
    exit 2
  fi
  MODEL_DIR="${MODEL_DIR}" "${PYTHON_BIN}" - <<'PY'
import json
import os

root = os.environ["MODEL_DIR"]
manifest = json.load(open(os.path.join(root, "model_manifest.json")))
weights = next(
    row["sha256"]
    for row in manifest["files"]
    if row["path"] == "pytorch_model.bin"
)
print(
    "codonGPT "
    f"hf_repo={manifest['hf_repo']} "
    f"revision={manifest['hf_revision']} "
    f"weights_sha256={weights} "
    f"license={manifest['license']}"
)
PY
  exit 0
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "codonGPT Python 环境不存在：${PYTHON_BIN}" >&2
  exit 2
fi
if [[ ! -f "${MANIFEST}" ]]; then
  echo "codonGPT 模型清单不存在：${MANIFEST}" >&2
  exit 2
fi

export PYTHONPATH="$(dirname "${ROOT}")${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PYTHON_BIN}" -m mrna_editflow.baselines.external_codongpt_adapter \
  --model-dir "${MODEL_DIR}" "$@"
