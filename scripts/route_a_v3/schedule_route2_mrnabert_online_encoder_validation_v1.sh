#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/cunyuliu/mrna_editflow_goal/worktrees/route_a_v3_route2_method_repair_20260817}"
ROUTE2_ROOT="${ROUTE2_ROOT:-/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2}"
PYTHON="${PYTHON:-/home/cunyuliu/miniconda3/envs/editflow/bin/python}"
POLL_SECONDS="${POLL_SECONDS:-900}"
GPU=4
CONFIG="${REPO_ROOT}/configs/route_a_v3_route2_mrnabert_online_encoder_validation_gpu4_v1.json"
OUTPUT="${ROUTE2_ROOT}/runs/mrnabert_online_encoder_validation_v1/validation_summary.json"

if [[ -e "${OUTPUT}" ]]; then
  printf 'online encoder validation output already exists: %s\n' "${OUTPUT}" >&2
  exit 1
fi

while true; do
  free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${GPU}" | tr -d ' ')
  util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "${GPU}" | tr -d ' ')
  printf '%s waiting_for_gpu=%s free_mb=%s util=%s\n' "$(date -Is)" "${GPU}" "${free_mb}" "${util}"
  if [[ "${free_mb}" -ge 24000 && "${util}" -le 70 ]]; then
    break
  fi
  sleep "${POLL_SECONDS}"
done

cd "${REPO_ROOT}"
printf '%s starting_online_mrnabert_encoder_validation gpu=%s\n' "$(date -Is)" "${GPU}"
"${PYTHON}" -u scripts/route_a_v3/validate_route2_mrnabert_online_encoder_v1.py \
  --config "${CONFIG}"
printf '%s online_mrnabert_encoder_validation_finished\n' "$(date -Is)"
