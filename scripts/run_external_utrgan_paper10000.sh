#!/usr/bin/env bash
# Run the official UTRGAN paper-default 10000-step protocol without overwrite.
set -euo pipefail

ROOT="${ROOT:-/home/cunyuliu/mrna_editflow_goal/mrna_editflow}"
PYTHON_BIN="${PYTHON_BIN:-/home/cunyuliu/miniconda3/envs/editflow/bin/python3.10}"
INPUT_PACK_SUMMARY="${INPUT_PACK_SUMMARY:-${ROOT}/benchmark/external_sota/input_pack_t5_head1024/summary.json}"
EXECUTABLE="${UTRGAN_BIN:-${ROOT}/scripts/external_utrgan.sh}"
TOOL_ROOT="${UTRGAN_ROOT:-${ROOT}/external_tools/UTRGAN}"
OUT_DIR="${OUT_DIR:-${ROOT}/benchmark/external_sota/real_runs_t5_head1024/UTRGAN_paper10000}"
STATUS_JSON="${STATUS_JSON:-${ROOT}/benchmark/external_sota/real_runs_t5_head1024/UTRGAN_paper10000.status.json}"
LOG_PATH="${LOG_PATH:-${ROOT}/logs/external_utrgan_paper10000.log}"
GPU="${UTRGAN_GPU:-4}"
TIMEOUT_S="${TIMEOUT_S:-21600}"

if [[ "${1:-}" == "--dry-run" ]]; then
  echo "EXTERNAL UTRGAN PAPER10000"
  echo "INPUT_PACK_SUMMARY=${INPUT_PACK_SUMMARY}"
  echo "EXECUTABLE=${EXECUTABLE}"
  echo "OUT_DIR=${OUT_DIR}"
  echo "GPU=${GPU} STEPS=10000 TIMEOUT_S=${TIMEOUT_S}"
  echo "STATUS_JSON=${STATUS_JSON}"
  echo "LOG_PATH=${LOG_PATH}"
  exit 0
fi

write_status() {
  local status="$1"
  local message="$2"
  STATUS="${status}" MESSAGE="${message}" STATUS_JSON="${STATUS_JSON}" \
    GPU="${GPU}" "${PYTHON_BIN}" - <<'PY'
import json
import os
import time

payload = {
    "artifact_kind": "external_utrgan_paper10000_status",
    "time": time.time(),
    "status": os.environ["STATUS"],
    "message": os.environ["MESSAGE"],
    "gpu": os.environ["GPU"],
    "steps": 10000,
}
path = os.environ["STATUS_JSON"]
os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
with open(path, "w", encoding="utf-8") as fh:
    json.dump(payload, fh, indent=2, sort_keys=True)
PY
}

mkdir -p "${OUT_DIR}" "$(dirname "${LOG_PATH}")"
export PYTHONPATH="$(dirname "${ROOT}")${PYTHONPATH:+:${PYTHONPATH}}"
on_error() {
  local code=$?
  write_status "failed" "UTRGAN paper-default run failed with exit ${code}"
  exit "${code}"
}
trap on_error ERR
write_status "running" "official UTRGAN paper-default 10000-step run is active"

UTRGAN_CUDA_VISIBLE_DEVICES="${GPU}" \
"${PYTHON_BIN}" -m mrna_editflow.baselines.external_utrgan_adapter \
  --input-pack-summary "${INPUT_PACK_SUMMARY}" \
  --executable "${EXECUTABLE}" \
  --tool-root "${TOOL_ROOT}" \
  --out-dir "${OUT_DIR}" \
  --steps 10000 \
  --learning-rate-exponent 5 \
  --timeout-s "${TIMEOUT_S}" > "${LOG_PATH}"

write_status "complete" "official UTRGAN paper-default 10000-step run completed"
trap - ERR
