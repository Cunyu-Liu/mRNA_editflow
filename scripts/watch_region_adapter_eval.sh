#!/usr/bin/env bash
# Wait for region-adapter training, then run the downstream multiseed eval.
#
# This is a resource-safe bridge from run_region_adapter_ablation.sh to
# eval_region_adapter_ablation.sh. It does not train anything itself.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-/home/cunyuliu/mrna_editflow_goal/mrna_editflow}"
PYTHON_BIN="${PYTHON_BIN:-/home/cunyuliu/miniconda3/envs/editflow/bin/python3.10}"
SLICE="${SLICE:-head256}"
TASK_ID="${TASK_ID:-T5}"
TASK_ID_LC="$(printf '%s' "${TASK_ID}" | tr '[:upper:]' '[:lower:]')"
REGION_MODES="${REGION_MODES:-utr5 cds utr3 all}"
CKPT_ROOT="${CKPT_ROOT:-${ROOT}/ckpts}"
EVAL_SCRIPT="${EVAL_SCRIPT:-${SCRIPT_DIR}/eval_region_adapter_ablation.sh}"
WAIT_PID="${WAIT_PID:-}"
POLL_SECONDS="${POLL_SECONDS:-120}"
WAIT_MAX_SECONDS="${WAIT_MAX_SECONDS:-86400}"
CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
MAX_LOADAVG="${MAX_LOADAVG:-0}"
LOAD_MAX_WAIT_SECONDS="${LOAD_MAX_WAIT_SECONDS:-86400}"
MAX_GPU_UTIL="${MAX_GPU_UTIL:-0}"
MAX_GPU_MEM_USED_MB="${MAX_GPU_MEM_USED_MB:-0}"
GPU_MAX_WAIT_SECONDS="${GPU_MAX_WAIT_SECONDS:-86400}"

usage() {
  cat <<'EOF'
Usage:
  watch_region_adapter_eval.sh [--dry-run]

Waits for a region-adapter training PID to exit, verifies all requested region
adapter checkpoints exist, waits for optional load/GPU resource gates, then runs
eval_region_adapter_ablation.sh.

Environment overrides:
  ROOT, PYTHON_BIN, SLICE, TASK_ID, REGION_MODES, CKPT_ROOT, EVAL_SCRIPT,
  WAIT_PID, POLL_SECONDS, WAIT_MAX_SECONDS, CUDA_VISIBLE_DEVICES,
  MAX_LOADAVG, LOAD_MAX_WAIT_SECONDS, MAX_GPU_UTIL, MAX_GPU_MEM_USED_MB,
  GPU_MAX_WAIT_SECONDS, plus variables accepted by eval_region_adapter_ablation.sh
EOF
}

ckpt_for_mode() {
  echo "${CKPT_ROOT}/region_adapter_${TASK_ID_LC}_$1_${SLICE}/stage_b_region_${TASK_ID_LC}_best.pt"
}

print_plan() {
  echo "ROOT=${ROOT}"
  echo "SLICE=${SLICE}  TASK_ID=${TASK_ID}  REGION_MODES=${REGION_MODES}"
  echo "WAIT_PID=${WAIT_PID:-<none>}  POLL_SECONDS=${POLL_SECONDS}  WAIT_MAX_SECONDS=${WAIT_MAX_SECONDS}"
  echo "CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER}  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  echo "MAX_LOADAVG=${MAX_LOADAVG}  LOAD_MAX_WAIT_SECONDS=${LOAD_MAX_WAIT_SECONDS}"
  echo "MAX_GPU_UTIL=${MAX_GPU_UTIL}  MAX_GPU_MEM_USED_MB=${MAX_GPU_MEM_USED_MB}  GPU_MAX_WAIT_SECONDS=${GPU_MAX_WAIT_SECONDS}"
  echo "EVAL_SCRIPT=${EVAL_SCRIPT}"
  for mode in ${REGION_MODES}; do
    echo "--- mode=${mode}"
    echo "    ckpt -> $(ckpt_for_mode "${mode}")"
  done
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then usage; exit 0; fi
if [[ "${1:-}" == "--dry-run" ]]; then print_plan; exit 0; fi

if [[ -n "${WAIT_PID}" ]]; then
  waited=0
  echo "[$(date -Iseconds)] waiting for region-adapter training PID ${WAIT_PID}"
  while kill -0 "${WAIT_PID}" 2>/dev/null; do
    if (( waited >= WAIT_MAX_SECONDS )); then
      echo "[$(date -Iseconds)] training wait timed out after ${waited}s for PID ${WAIT_PID}" >&2
      exit 2
    fi
    sleep "${POLL_SECONDS}"
    waited=$(( waited + POLL_SECONDS ))
  done
  echo "[$(date -Iseconds)] training PID ${WAIT_PID} exited"
fi

missing=0
for mode in ${REGION_MODES}; do
  ckpt="$(ckpt_for_mode "${mode}")"
  if [[ ! -f "${ckpt}" ]]; then
    echo "Missing region-adapter checkpoint for mode=${mode}: ${ckpt}" >&2
    missing=1
  fi
done
if (( missing != 0 )); then
  exit 2
fi

read_loadavg1() {
  if [[ -r /proc/loadavg ]]; then
    cut -d' ' -f1 /proc/loadavg
  else
    uptime | awk -F'load average[s]?: ' '{print $2}' | awk -F',' '{gsub(/ /, "", $1); print $1}'
  fi
}

if awk "BEGIN{exit !(${MAX_LOADAVG} > 0)}"; then
  waited=0
  while :; do
    load1="$(read_loadavg1)"
    if awk "BEGIN{exit !(${load1} < ${MAX_LOADAVG})}"; then
      echo "[$(date -Iseconds)] loadavg ${load1} < ${MAX_LOADAVG}; proceeding"
      break
    fi
    if (( waited >= LOAD_MAX_WAIT_SECONDS )); then
      echo "[$(date -Iseconds)] loadavg gate timed out after ${waited}s (load ${load1} >= ${MAX_LOADAVG})" >&2
      exit 2
    fi
    echo "[$(date -Iseconds)] loadavg ${load1} >= ${MAX_LOADAVG}; waiting ${POLL_SECONDS}s (waited ${waited}s)"
    sleep "${POLL_SECONDS}"
    waited=$(( waited + POLL_SECONDS ))
  done
fi

read_gpu_state() {
  gpu_index="${CUDA_VISIBLE_DEVICES%%,*}"
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
    | awk -F',' -v gpu="${gpu_index}" '
        {
          gsub(/ /, "", $1); gsub(/ /, "", $2); gsub(/ /, "", $3);
          if ($1 == gpu) {print $2, $3; found=1}
        }
        END {if (!found) exit 1}
      '
}

gpu_within_limits() {
  mem_used="$1"
  gpu_util="$2"
  if awk "BEGIN{exit !(${MAX_GPU_MEM_USED_MB} > 0 && ${mem_used} >= ${MAX_GPU_MEM_USED_MB})}"; then
    return 1
  fi
  if awk "BEGIN{exit !(${MAX_GPU_UTIL} > 0)}"; then
    if ! awk -v value="${gpu_util}" 'BEGIN{exit !(value ~ /^[0-9.]+$/)}'; then
      return 1
    fi
    if awk "BEGIN{exit !(${gpu_util} >= ${MAX_GPU_UTIL})}"; then
      return 1
    fi
  fi
  return 0
}

if awk "BEGIN{exit !(${MAX_GPU_UTIL} > 0 || ${MAX_GPU_MEM_USED_MB} > 0)}"; then
  waited=0
  while :; do
    if ! state="$(read_gpu_state)"; then
      echo "[$(date -Iseconds)] unable to read GPU ${CUDA_VISIBLE_DEVICES%%,*} state" >&2
      exit 2
    fi
    read -r mem_used gpu_util <<< "${state}"
    if gpu_within_limits "${mem_used}" "${gpu_util}"; then
      echo "[$(date -Iseconds)] GPU ${CUDA_VISIBLE_DEVICES%%,*} mem=${mem_used}MiB util=${gpu_util}% within limits; proceeding"
      break
    fi
    if (( waited >= GPU_MAX_WAIT_SECONDS )); then
      echo "[$(date -Iseconds)] GPU gate timed out after ${waited}s (gpu ${CUDA_VISIBLE_DEVICES%%,*} mem=${mem_used}MiB util=${gpu_util}%)" >&2
      exit 2
    fi
    echo "[$(date -Iseconds)] GPU ${CUDA_VISIBLE_DEVICES%%,*} busy (mem=${mem_used}MiB util=${gpu_util}%); waiting ${POLL_SECONDS}s (waited ${waited}s)"
    sleep "${POLL_SECONDS}"
    waited=$(( waited + POLL_SECONDS ))
  done
fi

export ROOT PYTHON_BIN SLICE TASK_ID REGION_MODES CKPT_ROOT CUDA_DEVICE_ORDER CUDA_VISIBLE_DEVICES
echo "[$(date -Iseconds)] launching region-adapter eval"
bash "${EVAL_SCRIPT}"
