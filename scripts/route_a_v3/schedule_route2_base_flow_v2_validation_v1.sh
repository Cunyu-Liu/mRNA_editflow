#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/cunyuliu/mrna_editflow_goal/worktrees/route_a_v3_route2_method_repair_20260817}"
ROUTE2_ROOT="${ROUTE2_ROOT:-/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2}"
PYTHON="${PYTHON:-/home/cunyuliu/miniconda3/envs/editflow/bin/python}"
POLL_SECONDS="${POLL_SECONDS:-900}"
GPU=2

TRAINING_SUMMARY="${ROUTE2_ROOT}/runs/base_flow_g0/position_progress_gpu_v2/training_summary.json"
VALIDATION_CONFIG="${REPO_ROOT}/configs/route_a_v3_route2_base_flow_g0_position_progress_validation_gpu2_v2.json"
VALIDATION_OUTPUT="${ROUTE2_ROOT}/runs/base_flow_g0/position_progress_validation_gpu_v2"

while [[ ! -f "${TRAINING_SUMMARY}" ]]; do
  printf '%s waiting_for_base_flow_v2_training_summary\n' "$(date -Is)"
  sleep "${POLL_SECONDS}"
done

"${PYTHON}" - "${TRAINING_SUMMARY}" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1]))
if summary.get("status") != "LEARNED_BASE_FLOW_GPU_UPDATE_COMPLETE":
    raise SystemExit("base-flow V2 training did not complete")
if summary.get("position_progress_features") is not True:
    raise SystemExit("base-flow V2 position/progress features were not active")
if summary.get("guided_critic_used") is not False:
    raise SystemExit("guided critic entered unguided Flow G0")
if summary.get("evaluation_records_read") != 0:
    raise SystemExit("Evaluation records entered unguided Flow G0")
PY

while true; do
  free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${GPU}" | tr -d ' ')
  util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "${GPU}" | tr -d ' ')
  printf '%s waiting_for_gpu=%s free_mb=%s util=%s\n' "$(date -Is)" "${GPU}" "${free_mb}" "${util}"
  if [[ "${free_mb}" -ge 24000 && "${util}" -le 70 ]]; then
    break
  fi
  sleep 300
done

if [[ -e "${VALIDATION_OUTPUT}" ]]; then
  printf 'validation output already exists: %s\n' "${VALIDATION_OUTPUT}" >&2
  exit 1
fi

cd "${REPO_ROOT}"
printf '%s starting_base_flow_v2_validation gpu=%s\n' "$(date -Is)" "${GPU}"
"${PYTHON}" -u scripts/route_a_v3/run_route2_base_flow_g0_validation_v1.py \
  --config "${VALIDATION_CONFIG}"
printf '%s base_flow_v2_validation_finished\n' "$(date -Is)"
