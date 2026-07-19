#!/usr/bin/env bash
# Region-specialized adapter ablation chain (roadmap architecture upgrade #2).
#
# Trains lightweight per-region Stage B adapters over a frozen Stage A MEF head.
# This script intentionally only trains adapters; downstream head256/head1024
# decoding comparisons should be launched separately once checkpoints are ready.
set -euo pipefail

ROOT="${ROOT:-/home/cunyuliu/mrna_editflow_goal/mrna_editflow}"
PYTHON_BIN="${PYTHON_BIN:-/home/cunyuliu/miniconda3/envs/editflow/bin/python3.10}"
SLICE="${SLICE:-head256}"
CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
DEVICE="${DEVICE:-cuda}"
TASK_ID="${TASK_ID:-T5}"
TASK_ID_LC="$(printf '%s' "${TASK_ID}" | tr '[:upper:]' '[:lower:]')"
STEPS="${STEPS:-500}"
SYNTHETIC_N="${SYNTHETIC_N:-8}"
ADAPTER_BOTTLENECK="${ADAPTER_BOTTLENECK:-32}"
REGION_MODES="${REGION_MODES:-utr5 cds utr3 all}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-${ROOT}/ckpts/stage_a_public_full_10k_bs8ga4_seed0/stage_a_best.pt}"
RECORDS_JSONL="${RECORDS_JSONL:-${ROOT}/benchmark/multiseed_t5_public_${SLICE}_hardneg_v2_top64/sources.jsonl}"
CKPT_ROOT="${CKPT_ROOT:-${ROOT}/ckpts}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs}"
WAIT_PID="${WAIT_PID:-}"
POLL_SECONDS="${POLL_SECONDS:-60}"
# Resource-safety gate: if MAX_LOADAVG > 0, defer until host 1-min loadavg is
# below the threshold. This keeps region-adapter training from preempting shared
# CPU/GPU jobs. 0 disables the gate.
MAX_LOADAVG="${MAX_LOADAVG:-0}"
LOAD_MAX_WAIT_SECONDS="${LOAD_MAX_WAIT_SECONDS:-86400}"
# Optional selected-GPU safety gate. If either threshold is >0, wait until the
# first CUDA_VISIBLE_DEVICES index is below both limits. 0 disables each limit.
MAX_GPU_UTIL="${MAX_GPU_UTIL:-0}"
MAX_GPU_MEM_USED_MB="${MAX_GPU_MEM_USED_MB:-0}"
GPU_MAX_WAIT_SECONDS="${GPU_MAX_WAIT_SECONDS:-86400}"

usage() {
  cat <<'EOF'
Usage:
  run_region_adapter_ablation.sh [--dry-run]

Trains region-specialized Stage B adapters with:
  python -m mrna_editflow.train_adapter --adapter-kind region

Environment overrides:
  ROOT, PYTHON_BIN, SLICE, CUDA_VISIBLE_DEVICES, DEVICE, TASK_ID, STEPS,
  ADAPTER_BOTTLENECK, REGION_MODES, BASE_CHECKPOINT, RECORDS_JSONL,
  CKPT_ROOT, LOG_ROOT, WAIT_PID, POLL_SECONDS, MAX_LOADAVG,
  LOAD_MAX_WAIT_SECONDS, MAX_GPU_UTIL, MAX_GPU_MEM_USED_MB,
  GPU_MAX_WAIT_SECONDS
EOF
}

regions_for_mode() {
  case "$1" in
    utr5|5utr) echo "5utr" ;;
    cds) echo "cds" ;;
    utr3|3utr) echo "3utr" ;;
    all) echo "5utr,cds,3utr" ;;
    *) echo "$1" ;;
  esac
}

save_dir_for_mode() {
  echo "${CKPT_ROOT}/region_adapter_${TASK_ID_LC}_$1_${SLICE}"
}

profile_for_mode() {
  echo "${LOG_ROOT}/region_adapter_${TASK_ID_LC}_$1_${SLICE}.profile.jsonl"
}

print_plan() {
  echo "ROOT=${ROOT}"
  echo "SLICE=${SLICE}  CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER}  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}  DEVICE=${DEVICE}"
  echo "TASK_ID=${TASK_ID}  STEPS=${STEPS}  ADAPTER_BOTTLENECK=${ADAPTER_BOTTLENECK}"
  echo "WAIT_PID=${WAIT_PID:-<none>}  POLL_SECONDS=${POLL_SECONDS}  MAX_LOADAVG=${MAX_LOADAVG}  LOAD_MAX_WAIT_SECONDS=${LOAD_MAX_WAIT_SECONDS}"
  echo "MAX_GPU_UTIL=${MAX_GPU_UTIL}  MAX_GPU_MEM_USED_MB=${MAX_GPU_MEM_USED_MB}  GPU_MAX_WAIT_SECONDS=${GPU_MAX_WAIT_SECONDS}"
  echo "BASE_CHECKPOINT=${BASE_CHECKPOINT}"
  echo "RECORDS_JSONL=${RECORDS_JSONL}"
  for mode in ${REGION_MODES}; do
    regions="$(regions_for_mode "${mode}")"
    echo "--- mode=${mode} regions=${regions}"
    echo "    save_dir -> $(save_dir_for_mode "${mode}")"
    echo "    profile  -> $(profile_for_mode "${mode}")"
  done
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then usage; exit 0; fi
if [[ "${1:-}" == "--dry-run" ]]; then print_plan; exit 0; fi

export CUDA_DEVICE_ORDER
export CUDA_VISIBLE_DEVICES
export PYTHONPATH="$(dirname "${ROOT}")${PYTHONPATH:+:${PYTHONPATH}}"

if [[ -n "${WAIT_PID}" ]]; then
  echo "[$(date -Iseconds)] gating region-adapter training on PID ${WAIT_PID} (poll ${POLL_SECONDS}s)"
  while kill -0 "${WAIT_PID}" 2>/dev/null; do
    sleep "${POLL_SECONDS}"
  done
  echo "[$(date -Iseconds)] PID ${WAIT_PID} exited; starting region-adapter training"
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
    if [[ -z "${load1}" ]]; then
      echo "[$(date -Iseconds)] unable to read loadavg; aborting load-gated run" >&2
      exit 2
    fi
    if awk "BEGIN{exit !(${load1} < ${MAX_LOADAVG})}"; then
      echo "[$(date -Iseconds)] loadavg ${load1} < ${MAX_LOADAVG}; proceeding"
      break
    fi
    if (( waited >= LOAD_MAX_WAIT_SECONDS )); then
      echo "[$(date -Iseconds)] loadavg gate timed out after ${waited}s (load ${load1} >= ${MAX_LOADAVG}); aborting to avoid preemption" >&2
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

if [[ "${DEVICE}" == cuda* ]] && awk "BEGIN{exit !(${MAX_GPU_UTIL} > 0 || ${MAX_GPU_MEM_USED_MB} > 0)}"; then
  waited=0
  while :; do
    if ! state="$(read_gpu_state)"; then
      echo "[$(date -Iseconds)] unable to read GPU ${CUDA_VISIBLE_DEVICES%%,*} state; aborting GPU-gated run" >&2
      exit 2
    fi
    read -r mem_used gpu_util <<< "${state}"
    if gpu_within_limits "${mem_used}" "${gpu_util}"; then
      echo "[$(date -Iseconds)] GPU ${CUDA_VISIBLE_DEVICES%%,*} mem=${mem_used}MiB util=${gpu_util}% within limits; proceeding"
      break
    fi
    if (( waited >= GPU_MAX_WAIT_SECONDS )); then
      echo "[$(date -Iseconds)] GPU gate timed out after ${waited}s (gpu ${CUDA_VISIBLE_DEVICES%%,*} mem=${mem_used}MiB util=${gpu_util}%); aborting to avoid preemption" >&2
      exit 2
    fi
    echo "[$(date -Iseconds)] GPU ${CUDA_VISIBLE_DEVICES%%,*} busy (mem=${mem_used}MiB util=${gpu_util}%); waiting ${POLL_SECONDS}s (waited ${waited}s)"
    sleep "${POLL_SECONDS}"
    waited=$(( waited + POLL_SECONDS ))
  done
fi

for mode in ${REGION_MODES}; do
  regions="$(regions_for_mode "${mode}")"
  save_dir="$(save_dir_for_mode "${mode}")"
  profile="$(profile_for_mode "${mode}")"
  echo "[$(date -Iseconds)] TRAIN region adapter mode=${mode} regions=${regions}"
  "${PYTHON_BIN}" -m mrna_editflow.train_adapter \
    --run-mode development \
    --adapter-kind region \
    --task-id "${TASK_ID}" \
    --steps "${STEPS}" \
    --synthetic-n "${SYNTHETIC_N}" \
    --records-jsonl "${RECORDS_JSONL}" \
    --adapter-bottleneck "${ADAPTER_BOTTLENECK}" \
    --regions "${regions}" \
    --base-checkpoint "${BASE_CHECKPOINT}" \
    --save-dir "${save_dir}" \
    --profile-path "${profile}" \
    --device "${DEVICE}"
done

echo "[$(date -Iseconds)] REGION ADAPTER ABLATION TRAINING COMPLETE"
