#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/cunyuliu/mrna_editflow_goal/worktrees/route_a_v3_route2_method_repair_20260817}"
ROUTE2_ROOT="${ROUTE2_ROOT:-/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2}"
PYTHON="${PYTHON:-/home/cunyuliu/miniconda3/envs/editflow/bin/python}"
POLL_SECONDS="${POLL_SECONDS:-900}"
GPU=0

HUBER_SUMMARY="${ROUTE2_ROOT}/runs/mrnabert_scaleup_v2/max_mean_only_seed20260816_gpu0_bf16_v1/training_summary.json"
CONFIG="${REPO_ROOT}/configs/route_a_v3_route2_mrnabert_dataloader_benchmark_gpu0_v1.json"
OUTPUT="${ROUTE2_ROOT}/benchmarks/mrnabert_dataloader_benchmark_v1_gpu0/benchmark_summary.json"

while [[ ! -f "${HUBER_SUMMARY}" ]]; do
  printf '%s waiting_for_huber_before_dataloader_benchmark\n' "$(date -Is)"
  sleep "${POLL_SECONDS}"
done

while true; do
  free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${GPU}" | tr -d ' ')
  util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "${GPU}" | tr -d ' ')
  printf '%s waiting_for_dataloader_benchmark_gpu=%s free_mb=%s util=%s\n' \
    "$(date -Is)" "${GPU}" "${free_mb}" "${util}"
  if [[ "${free_mb}" -ge 24000 && "${util}" -le 70 ]]; then
    break
  fi
  sleep "${POLL_SECONDS}"
done

if [[ -e "${OUTPUT}" ]]; then
  printf 'dataloader benchmark output already exists: %s\n' "${OUTPUT}" >&2
  exit 1
fi

cd "${REPO_ROOT}"
printf '%s starting_mrnabert_dataloader_benchmark gpu=%s\n' "$(date -Is)" "${GPU}"
"${PYTHON}" -u scripts/route_a_v3/benchmark_route2_delta_gpu_training_v1.py \
  --config "${CONFIG}"
printf '%s mrnabert_dataloader_benchmark_finished\n' "$(date -Is)"
