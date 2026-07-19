#!/usr/bin/env bash
# End-to-end region-specialized adapter ablation chain (roadmap upgrade #2).
#
# Runs the durable region-adapter path for one slice:
#   1. train  : train Stage B region adapters for utr5/cds/utr3/all
#               (run_region_adapter_ablation.sh)
#   2. eval   : verify checkpoints, apply resource gates, run multiseed eval,
#               compare against hardneg_v2, and refresh the SOTA gap report
#               (watch_region_adapter_eval.sh -> eval_region_adapter_ablation.sh)
#
# Existing train/watch scripts own the actual implementation and resource gates;
# this wrapper gives the whole train -> eval -> report path a single reproducible
# repo entry point for head256 and later head1024 reruns.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-/home/cunyuliu/mrna_editflow_goal/mrna_editflow}"
PYTHON_BIN="${PYTHON_BIN:-/home/cunyuliu/miniconda3/envs/editflow/bin/python3.10}"
SLICE="${SLICE:-head256}"
CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
DEVICE="${DEVICE:-cuda}"
TASK_ID="${TASK_ID:-T5}"
REGION_MODES="${REGION_MODES:-utr5 cds utr3 all}"
TOP_K="${TOP_K:-64}"
POLL_SECONDS="${POLL_SECONDS:-120}"
WAIT_PID="${WAIT_PID:-}"
WAIT_MAX_SECONDS="${WAIT_MAX_SECONDS:-86400}"

CKPT_ROOT="${CKPT_ROOT:-${ROOT}/ckpts}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs}"
BENCH_ROOT="${BENCH_ROOT:-${ROOT}/benchmark}"
BASELINE_LABEL="${BASELINE_LABEL:-hardneg_v2_top${TOP_K}}"
BASELINE_SUMMARY="${BASELINE_SUMMARY:-${BENCH_ROOT}/multiseed_t5_public_${SLICE}_hardneg_v2_top${TOP_K}/multiseed_summary.json}"
COMPARE_PREFIX="${COMPARE_PREFIX:-region_adapter_vs_${BASELINE_LABEL}_${SLICE}}"
REFRESH_SOTA="${REFRESH_SOTA:-1}"
SOTA_JSON="${SOTA_JSON:-${ROOT}/docs/sota_gap_report.json}"
SOTA_MD="${SOTA_MD:-${ROOT}/docs/sota_gap_report.md}"

MAX_LOADAVG="${MAX_LOADAVG:-0}"
LOAD_MAX_WAIT_SECONDS="${LOAD_MAX_WAIT_SECONDS:-86400}"
MAX_GPU_UTIL="${MAX_GPU_UTIL:-0}"
MAX_GPU_MEM_USED_MB="${MAX_GPU_MEM_USED_MB:-0}"
GPU_MAX_WAIT_SECONDS="${GPU_MAX_WAIT_SECONDS:-86400}"

TRAIN_SCRIPT="${TRAIN_SCRIPT:-${SCRIPT_DIR}/run_region_adapter_ablation.sh}"
WATCH_SCRIPT="${WATCH_SCRIPT:-${SCRIPT_DIR}/watch_region_adapter_eval.sh}"

usage() {
  cat <<'EOF'
Usage:
  run_region_adapter_ablation_chain.sh [--dry-run]

Runs the full region-specialized adapter chain for one data slice:
train adapters -> verify checkpoints -> multiseed eval -> paired compare ->
refresh SOTA report.

Environment overrides:
  ROOT, PYTHON_BIN, SLICE, CUDA_VISIBLE_DEVICES, DEVICE, TASK_ID,
  REGION_MODES, TOP_K, WAIT_PID, WAIT_MAX_SECONDS, POLL_SECONDS, CKPT_ROOT,
  LOG_ROOT, BENCH_ROOT, BASELINE_LABEL, BASELINE_SUMMARY, COMPARE_PREFIX,
  REFRESH_SOTA, SOTA_JSON, SOTA_MD, MAX_LOADAVG, LOAD_MAX_WAIT_SECONDS,
  MAX_GPU_UTIL, MAX_GPU_MEM_USED_MB, GPU_MAX_WAIT_SECONDS, TRAIN_SCRIPT,
  WATCH_SCRIPT, plus variables accepted by the train/eval sub-scripts.
EOF
}

print_plan() {
  echo "REGION ADAPTER CHAIN"
  echo "ROOT=${ROOT}"
  echo "SLICE=${SLICE}  TASK_ID=${TASK_ID}  REGION_MODES=${REGION_MODES}"
  echo "CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER}  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}  DEVICE=${DEVICE}"
  echo "WAIT_PID=${WAIT_PID:-<none>}  WAIT_MAX_SECONDS=${WAIT_MAX_SECONDS}  POLL_SECONDS=${POLL_SECONDS}"
  echo "MAX_LOADAVG=${MAX_LOADAVG}  LOAD_MAX_WAIT_SECONDS=${LOAD_MAX_WAIT_SECONDS}"
  echo "MAX_GPU_UTIL=${MAX_GPU_UTIL}  MAX_GPU_MEM_USED_MB=${MAX_GPU_MEM_USED_MB}  GPU_MAX_WAIT_SECONDS=${GPU_MAX_WAIT_SECONDS}"
  echo "step 1 train : ${TRAIN_SCRIPT}"
  echo "step 2 eval  : ${WATCH_SCRIPT}"
  echo "compare -> ${BENCH_ROOT}/${COMPARE_PREFIX}.{json,md}"
  echo "SOTA report -> ${SOTA_JSON} and ${SOTA_MD} (REFRESH_SOTA=${REFRESH_SOTA})"
  echo "--- train sub-script plan ---"
  ROOT="${ROOT}" PYTHON_BIN="${PYTHON_BIN}" WAIT_PID="" POLL_SECONDS="${POLL_SECONDS}" \
    SLICE="${SLICE}" TASK_ID="${TASK_ID}" REGION_MODES="${REGION_MODES}" \
    CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER}" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    DEVICE="${DEVICE}" CKPT_ROOT="${CKPT_ROOT}" LOG_ROOT="${LOG_ROOT}" \
    MAX_LOADAVG="${MAX_LOADAVG}" LOAD_MAX_WAIT_SECONDS="${LOAD_MAX_WAIT_SECONDS}" \
    MAX_GPU_UTIL="${MAX_GPU_UTIL}" MAX_GPU_MEM_USED_MB="${MAX_GPU_MEM_USED_MB}" \
    GPU_MAX_WAIT_SECONDS="${GPU_MAX_WAIT_SECONDS}" bash "${TRAIN_SCRIPT}" --dry-run
  echo "--- watch/eval sub-script plan ---"
  ROOT="${ROOT}" PYTHON_BIN="${PYTHON_BIN}" WAIT_PID="" POLL_SECONDS="${POLL_SECONDS}" \
    SLICE="${SLICE}" TASK_ID="${TASK_ID}" REGION_MODES="${REGION_MODES}" \
    CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER}" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    CKPT_ROOT="${CKPT_ROOT}" BENCH_ROOT="${BENCH_ROOT}" BASELINE_LABEL="${BASELINE_LABEL}" \
    BASELINE_SUMMARY="${BASELINE_SUMMARY}" COMPARE_PREFIX="${COMPARE_PREFIX}" \
    REFRESH_SOTA="${REFRESH_SOTA}" SOTA_JSON="${SOTA_JSON}" SOTA_MD="${SOTA_MD}" \
    MAX_LOADAVG="${MAX_LOADAVG}" LOAD_MAX_WAIT_SECONDS="${LOAD_MAX_WAIT_SECONDS}" \
    MAX_GPU_UTIL="${MAX_GPU_UTIL}" MAX_GPU_MEM_USED_MB="${MAX_GPU_MEM_USED_MB}" \
    GPU_MAX_WAIT_SECONDS="${GPU_MAX_WAIT_SECONDS}" bash "${WATCH_SCRIPT}" --dry-run
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then usage; exit 0; fi
if [[ "${1:-}" == "--dry-run" ]]; then print_plan; exit 0; fi

if [[ -n "${WAIT_PID}" ]]; then
  waited=0
  echo "[$(date -Iseconds)] gating region-adapter chain on PID ${WAIT_PID}"
  while kill -0 "${WAIT_PID}" 2>/dev/null; do
    if (( waited >= WAIT_MAX_SECONDS )); then
      echo "[$(date -Iseconds)] wait timed out after ${waited}s for PID ${WAIT_PID}" >&2
      exit 2
    fi
    sleep "${POLL_SECONDS}"
    waited=$(( waited + POLL_SECONDS ))
  done
  echo "[$(date -Iseconds)] PID ${WAIT_PID} exited; starting region-adapter chain"
fi

export ROOT PYTHON_BIN SLICE CUDA_DEVICE_ORDER CUDA_VISIBLE_DEVICES DEVICE TASK_ID
export REGION_MODES POLL_SECONDS CKPT_ROOT LOG_ROOT BENCH_ROOT
export BASELINE_LABEL BASELINE_SUMMARY COMPARE_PREFIX REFRESH_SOTA SOTA_JSON SOTA_MD
export MAX_LOADAVG LOAD_MAX_WAIT_SECONDS MAX_GPU_UTIL MAX_GPU_MEM_USED_MB GPU_MAX_WAIT_SECONDS

echo "[$(date -Iseconds)] STEP 1/2 train region adapters (SLICE=${SLICE})"
WAIT_PID="" bash "${TRAIN_SCRIPT}"

echo "[$(date -Iseconds)] STEP 2/2 eval + compare + SOTA refresh (SLICE=${SLICE})"
WAIT_PID="" bash "${WATCH_SCRIPT}"

echo "[$(date -Iseconds)] REGION ADAPTER CHAIN COMPLETE (SLICE=${SLICE})"
