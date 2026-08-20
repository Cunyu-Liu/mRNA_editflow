#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/cunyuliu/mrna_editflow_goal/worktrees/route_a_v3_route2_method_repair_20260817}"
ROUTE2_ROOT="${ROUTE2_ROOT:-/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2}"
PYTHON="${PYTHON:-/home/cunyuliu/miniconda3/envs/editflow/bin/python}"
POLL_SECONDS="${POLL_SECONDS:-900}"
GPU_CANDIDATES="${GPU_CANDIDATES:-0 1 2 3 4 5}"
MINIMUM_FREE_MB="${MINIMUM_FREE_MB:-1024}"
FLOW_VALIDATION="${ROUTE2_ROOT}/runs/base_flow_g0/position_progress_validation_gpu_v2/validation_summary.json"
CONFIG_TEMPLATE="${REPO_ROOT}/configs/route_a_v3_route2_independent_evaluator_neural_medium_task_scaled_gpu2_v3.json"
RUNTIME_CONFIG_ROOT="${ROUTE2_ROOT}/runs/independent_generation_evaluator/runtime_configs"
PROTOCOL="${REPO_ROOT}/configs/route_a_v3_route2_mrnabert_independent_evaluator_qualification_v1.json"
RUN_ROOT="${ROUTE2_ROOT}/runs/independent_generation_evaluator/neural_medium_siamese_task_scaled_seed20260816_frozen_development_validation_gpu2_v3"
ADJUDICATION="${ROUTE2_ROOT}/comparisons/mrnabert_independent_evaluator_adjudication_v1.json"

while [[ ! -f "${FLOW_VALIDATION}" ]]; do
  printf '%s waiting_for_base_flow_validation_before_independent_evaluator\n' "$(date -Is)"
  sleep "${POLL_SECONDS}"
done

read -r -a candidate_gpus <<<"${GPU_CANDIDATES}"
selected_gpu=""
while [[ -z "${selected_gpu}" ]]; do
  best_free_mb=-1
  for gpu in "${candidate_gpus[@]}"; do
    free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${gpu}" | tr -d ' ')
    util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "${gpu}" | tr -d ' ')
    printf '%s independent_evaluator_gpu_candidate=%s free_mb=%s util=%s\n' \
      "$(date -Is)" "${gpu}" "${free_mb}" "${util}"
    if [[ "${free_mb}" -ge "${MINIMUM_FREE_MB}" \
      && "${free_mb}" -gt "${best_free_mb}" ]]; then
      selected_gpu="${gpu}"
      best_free_mb="${free_mb}"
    fi
  done
  if [[ -z "${selected_gpu}" ]]; then
    sleep "${POLL_SECONDS}"
  fi
done

if [[ -e "${RUN_ROOT}" || -e "${ADJUDICATION}" ]]; then
  printf 'independent evaluator output already exists\n' >&2
  exit 1
fi

cd "${REPO_ROOT}"
runtime_config="${RUNTIME_CONFIG_ROOT}/independent_evaluator_gpu${selected_gpu}_v3.json"
mkdir -p "${RUNTIME_CONFIG_ROOT}"
if [[ -e "${runtime_config}" ]]; then
  printf 'independent evaluator runtime config already exists: %s\n' "${runtime_config}" >&2
  exit 1
fi
"${PYTHON}" - "${CONFIG_TEMPLATE}" "${runtime_config}" "${selected_gpu}" "${MINIMUM_FREE_MB}" <<'PY'
import json
import sys
from pathlib import Path

template_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
gpu = int(sys.argv[3])
minimum_free_mb = int(sys.argv[4])
config = json.loads(template_path.read_text(encoding="utf-8"))
config["device"] = f"cuda:{gpu}"
config["physical_gpu_index"] = gpu
config["gpu_selection_policy"] = "MOST_FREE_MEMORY_GPU_0_TO_5_WITH_TASK_MINIMUM_NO_UTILIZATION_GATE"
config["gpu_selection_minimum_free_mb"] = minimum_free_mb
output_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

printf '%s starting_development_only_independent_evaluator gpu=%s\n' "$(date -Is)" "${selected_gpu}"
"${PYTHON}" -u scripts/route_a_v3/train_route2_delta_predictor_v1.py \
  --config "${runtime_config}" \
  >"${ROUTE2_ROOT}/independent_evaluator_task_scaled_gpu2_v3.log" 2>&1

"${PYTHON}" scripts/route_a_v3/adjudicate_route2_independent_generation_evaluator_v1.py \
  --training-summary "${RUN_ROOT}/training_summary.json" \
  --protocol "${PROTOCOL}" \
  --output "${ADJUDICATION}"
printf '%s independent_evaluator_adjudication_finished\n' "$(date -Is)"
