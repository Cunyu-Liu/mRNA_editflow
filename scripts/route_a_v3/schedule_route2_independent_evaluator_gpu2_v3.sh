#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/cunyuliu/mrna_editflow_goal/worktrees/route_a_v3_route2_method_repair_20260817}"
ROUTE2_ROOT="${ROUTE2_ROOT:-/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2}"
PYTHON="${PYTHON:-/home/cunyuliu/miniconda3/envs/editflow/bin/python}"
POLL_SECONDS="${POLL_SECONDS:-900}"
GPU=2
FLOW_VALIDATION="${ROUTE2_ROOT}/runs/base_flow_g0/position_progress_validation_gpu_v2/validation_summary.json"
CONFIG="${REPO_ROOT}/configs/route_a_v3_route2_independent_evaluator_neural_medium_task_scaled_gpu2_v3.json"
PROTOCOL="${REPO_ROOT}/configs/route_a_v3_route2_mrnabert_independent_evaluator_qualification_v1.json"
RUN_ROOT="${ROUTE2_ROOT}/runs/independent_generation_evaluator/neural_medium_siamese_task_scaled_seed20260816_frozen_development_validation_gpu2_v3"
ADJUDICATION="${ROUTE2_ROOT}/comparisons/mrnabert_independent_evaluator_adjudication_v1.json"

while [[ ! -f "${FLOW_VALIDATION}" ]]; do
  printf '%s waiting_for_base_flow_validation_before_independent_evaluator\n' "$(date -Is)"
  sleep "${POLL_SECONDS}"
done

while true; do
  free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${GPU}" | tr -d ' ')
  util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "${GPU}" | tr -d ' ')
  printf '%s waiting_for_independent_evaluator_gpu=%s free_mb=%s util=%s\n' "$(date -Is)" "${GPU}" "${free_mb}" "${util}"
  if [[ "${free_mb}" -ge 24000 && "${util}" -le 70 ]]; then
    break
  fi
  sleep "${POLL_SECONDS}"
done

if [[ -e "${RUN_ROOT}" || -e "${ADJUDICATION}" ]]; then
  printf 'independent evaluator output already exists\n' >&2
  exit 1
fi

cd "${REPO_ROOT}"
printf '%s starting_development_only_independent_evaluator gpu=%s\n' "$(date -Is)" "${GPU}"
"${PYTHON}" -u scripts/route_a_v3/train_route2_delta_predictor_v1.py \
  --config "${CONFIG}" \
  >"${ROUTE2_ROOT}/independent_evaluator_task_scaled_gpu2_v3.log" 2>&1

"${PYTHON}" scripts/route_a_v3/adjudicate_route2_independent_generation_evaluator_v1.py \
  --training-summary "${RUN_ROOT}/training_summary.json" \
  --protocol "${PROTOCOL}" \
  --output "${ADJUDICATION}"
printf '%s independent_evaluator_adjudication_finished\n' "$(date -Is)"
