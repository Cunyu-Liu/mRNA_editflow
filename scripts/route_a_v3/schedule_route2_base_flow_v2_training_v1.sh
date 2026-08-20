#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/cunyuliu/mrna_editflow_goal/worktrees/route_a_v3_route2_method_repair_20260817}"
ROUTE2_ROOT="${ROUTE2_ROOT:-/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2}"
PYTHON="${PYTHON:-/home/cunyuliu/miniconda3/envs/editflow/bin/python}"
POLL_SECONDS="${POLL_SECONDS:-300}"
GPU_CANDIDATES="${GPU_CANDIDATES:-0 1 2 3 4 5}"
MINIMUM_FREE_MB="${MINIMUM_FREE_MB:-1024}"

TEMPLATE_CONFIG="${REPO_ROOT}/configs/route_a_v3_route2_base_flow_g0_position_progress_gpu_v2.json"
OUTPUT_DIRECTORY="${ROUTE2_ROOT}/runs/base_flow_g0/position_progress_gpu_v2"
RUNTIME_CONFIG_ROOT="${ROUTE2_ROOT}/runs/base_flow_g0/runtime_configs"

if [[ -e "${OUTPUT_DIRECTORY}" ]]; then
  printf 'base-flow V2 output already exists: %s\n' "${OUTPUT_DIRECTORY}" >&2
  exit 1
fi

read -r -a candidate_gpus <<<"${GPU_CANDIDATES}"
if [[ "${#candidate_gpus[@]}" -eq 0 ]]; then
  printf 'no GPU candidates were supplied\n' >&2
  exit 1
fi

selected_gpu=""
while [[ -z "${selected_gpu}" ]]; do
  best_free_mb=-1
  for gpu in "${candidate_gpus[@]}"; do
    free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${gpu}" | tr -d ' ')
    util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "${gpu}" | tr -d ' ')
    printf '%s base_flow_gpu_candidate=%s free_mb=%s util=%s\n' \
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

runtime_config="${RUNTIME_CONFIG_ROOT}/base_flow_g0_position_progress_gpu${selected_gpu}_v2.json"
mkdir -p "${RUNTIME_CONFIG_ROOT}"
if [[ -e "${runtime_config}" ]]; then
  printf 'base-flow runtime config already exists: %s\n' "${runtime_config}" >&2
  exit 1
fi

"${PYTHON}" - "${TEMPLATE_CONFIG}" "${runtime_config}" "${selected_gpu}" "${MINIMUM_FREE_MB}" <<'PY'
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

cd "${REPO_ROOT}"
printf '%s starting_base_flow_v2_training gpu=%s config=%s\n' \
  "$(date -Is)" "${selected_gpu}" "${runtime_config}"
"${PYTHON}" -u scripts/route_a_v3/train_route2_base_flow_g0_v1.py \
  --config "${runtime_config}"
printf '%s base_flow_v2_training_finished gpu=%s\n' "$(date -Is)" "${selected_gpu}"
