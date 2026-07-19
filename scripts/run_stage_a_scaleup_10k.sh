#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/cunyuliu/mrna_editflow_goal/mrna_editflow}"
PYTHON_BIN="${PYTHON_BIN:-/home/cunyuliu/miniconda3/envs/editflow/bin/python3.10}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"
STEPS="${STEPS:-10000}"
SEED="${SEED:-0}"
SPLIT_NAME="${SPLIT_NAME:-public_full_clean}"
RUN_NAME="${RUN_NAME:-stage_a_public_full_10k_bs8ga4_seed${SEED}}"
CONFIG="${CONFIG:-${ROOT}/configs/stage_a_full_bs8_gradaccum4.json}"
RECORDS_JSONL="${RECORDS_JSONL:-${ROOT}/data/processed/gencode_human_transcripts.records.jsonl}"
SAVE_DIR="${SAVE_DIR:-${ROOT}/ckpts/${RUN_NAME}}"
PROFILE_PATH="${PROFILE_PATH:-${ROOT}/logs/${RUN_NAME}.profile.jsonl}"
LOG_PATH="${LOG_PATH:-${ROOT}/logs/${RUN_NAME}.train.log}"
METADATA_JSON="${METADATA_JSON:-${ROOT}/logs/${RUN_NAME}.metadata.json}"
MIN_FREE_MEM_MB="${MIN_FREE_MEM_MB:-20000}"
SKIP_GPU_CHECK="${SKIP_GPU_CHECK:-0}"
ALLOW_EXISTING="${ALLOW_EXISTING:-0}"

usage() {
  cat <<'EOF'
Usage:
  run_stage_a_scaleup_10k.sh [--dry-run]

Purpose:
  Launch the Stage A full-corpus 10k-step scale-up with reproducibility guards.
  The script records dataset SHA256, split, seed, record count, runtime target,
  GPU selection and host hardware before training starts.

Environment overrides:
  ROOT, PYTHON_BIN, CUDA_VISIBLE_DEVICES, STEPS, SEED, SPLIT_NAME, RUN_NAME,
  CONFIG, RECORDS_JSONL, SAVE_DIR, PROFILE_PATH, LOG_PATH, METADATA_JSON,
  MIN_FREE_MEM_MB, SKIP_GPU_CHECK, ALLOW_EXISTING
EOF
}

print_plan() {
  cat <<EOF
ROOT=${ROOT}
PYTHON_BIN=${PYTHON_BIN}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
STEPS=${STEPS}
SEED=${SEED}
SPLIT_NAME=${SPLIT_NAME}
RUN_NAME=${RUN_NAME}
CONFIG=${CONFIG}
RECORDS_JSONL=${RECORDS_JSONL}
SAVE_DIR=${SAVE_DIR}
PROFILE_PATH=${PROFILE_PATH}
LOG_PATH=${LOG_PATH}
METADATA_JSON=${METADATA_JSON}
MIN_FREE_MEM_MB=${MIN_FREE_MEM_MB}

Command:
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" PYTHONPATH="\$(dirname "${ROOT}")" "${PYTHON_BIN}" -m mrna_editflow.train_backbone \\
    --config "${CONFIG}" \\
    --records-jsonl "${RECORDS_JSONL}" \\
    --steps "${STEPS}" \\
    --save-dir "${SAVE_DIR}" \\
    --profile-path "${PROFILE_PATH}" \\
    --device cuda \\
    --seed "${SEED}"
EOF
}

write_metadata() {
  "${PYTHON_BIN}" - <<PY
import hashlib
import json
import os
import platform
import socket
import time

records = "${RECORDS_JSONL}"
digest = hashlib.sha256()
record_count = 0
with open(records, "rb") as fh:
    for chunk in iter(lambda: fh.read(1024 * 1024), b""):
        digest.update(chunk)
with open(records, "r", encoding="utf-8") as fh:
    record_count = sum(1 for line in fh if line.strip())

payload = {
    "status": "launch_metadata_ready",
    "time": time.time(),
    "run_name": "${RUN_NAME}",
    "stage": "stage_a_scaleup_10k",
    "steps": int("${STEPS}"),
    "seed": int("${SEED}"),
    "split_name": "${SPLIT_NAME}",
    "records_jsonl": records,
    "dataset_sha256": digest.hexdigest(),
    "record_count": record_count,
    "config": "${CONFIG}",
    "save_dir": "${SAVE_DIR}",
    "profile_path": "${PROFILE_PATH}",
    "log_path": "${LOG_PATH}",
    "cuda_visible_devices": "${CUDA_VISIBLE_DEVICES}",
    "hostname": socket.gethostname(),
    "machine": platform.machine(),
    "platform": platform.platform(),
}
os.makedirs(os.path.dirname("${METADATA_JSON}"), exist_ok=True)
with open("${METADATA_JSON}", "w", encoding="utf-8") as fh:
    json.dump(payload, fh, indent=2, sort_keys=True)
print(json.dumps(payload, sort_keys=True))
PY
}

check_gpu_memory() {
  if [[ "${SKIP_GPU_CHECK}" == "1" ]]; then
    return 0
  fi
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return 0
  fi
  local first_gpu
  first_gpu="${CUDA_VISIBLE_DEVICES%%,*}"
  local gpu_row
  gpu_row="$(nvidia-smi -i "${first_gpu}" --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
  local used total free
  used="${gpu_row%,*}"
  total="${gpu_row#*,}"
  free=$((total - used))
  if (( free < MIN_FREE_MEM_MB )); then
    echo "GPU ${first_gpu} has only ${free} MiB free; require ${MIN_FREE_MEM_MB} MiB. Set SKIP_GPU_CHECK=1 to override." >&2
    exit 1
  fi
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

if [[ "${1:-}" == "--dry-run" ]]; then
  print_plan
  exit 0
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing executable PYTHON_BIN: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ ! -s "${CONFIG}" ]]; then
  echo "Missing config: ${CONFIG}" >&2
  exit 1
fi
if [[ ! -s "${RECORDS_JSONL}" ]]; then
  echo "Missing records JSONL: ${RECORDS_JSONL}" >&2
  exit 1
fi
if [[ "${ALLOW_EXISTING}" != "1" ]]; then
  if [[ -e "${SAVE_DIR}/stage_a_best.pt" || -e "${PROFILE_PATH}" || -e "${LOG_PATH}" ]]; then
    echo "Refusing to overwrite existing Stage A scale-up artifacts for RUN_NAME=${RUN_NAME}." >&2
    echo "Set ALLOW_EXISTING=1 or choose a new RUN_NAME." >&2
    exit 1
  fi
fi

check_gpu_memory
mkdir -p "${SAVE_DIR}" "$(dirname "${PROFILE_PATH}")" "$(dirname "${LOG_PATH}")"
write_metadata

export CUDA_VISIBLE_DEVICES
export PYTHONPATH="$(dirname "${ROOT}")${PYTHONPATH:+:${PYTHONPATH}}"

"${PYTHON_BIN}" -m mrna_editflow.train_backbone \
  --run-mode development \
  --config "${CONFIG}" \
  --records-jsonl "${RECORDS_JSONL}" \
  --steps "${STEPS}" \
  --save-dir "${SAVE_DIR}" \
  --profile-path "${PROFILE_PATH}" \
  --device cuda \
  --seed "${SEED}" \
  > "${LOG_PATH}" 2>&1
