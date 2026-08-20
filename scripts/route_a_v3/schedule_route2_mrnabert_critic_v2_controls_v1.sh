#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/cunyuliu/mrna_editflow_goal/worktrees/route_a_v3_route2_method_repair_20260817}"
ROUTE2_ROOT="${ROUTE2_ROOT:-/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2}"
PYTHON="${PYTHON:-/home/cunyuliu/miniconda3/envs/editflow/bin/python}"
POLL_SECONDS="${POLL_SECONDS:-900}"
MINIMUM_FREE_MB="${MINIMUM_FREE_MB:-4096}"
GPU_CANDIDATES="${GPU_CANDIDATES:-0 1 2 3 4 5}"

PROTOCOL="${REPO_ROOT}/configs/route_a_v3_route2_mrnabert_critic_v2_protocol_v1.json"
BASE_CONFIG="${REPO_ROOT}/configs/route_a_v3_route2_mrnabert_edit_max_mean_only_gpu6_v1.json"
RUNTIME_CONFIG_ROOT="${ROUTE2_ROOT}/runs/mrnabert_critic_v2/runtime_configs/task_study_macro_screen_seed20260825_v1"
RUN_ROOT="${ROUTE2_ROOT}/runs/mrnabert_critic_v2/task_study_macro_screen_seed20260825_v1"
LOG_ROOT="${ROUTE2_ROOT}/logs/mrnabert_critic_v2/task_study_macro_screen_seed20260825_v1"
ADJUDICATION="${ROUTE2_ROOT}/comparisons/mrnabert_critic_v2_task_study_macro_controls_adjudication_v1.json"
ARMS=(full candidate_permutation source_only source_edit_metadata)

if [[ -e "${ADJUDICATION}" || -e "${RUNTIME_CONFIG_ROOT}" || -e "${RUN_ROOT}" ]]; then
  printf 'Critic V2 runtime or terminal already exists; refusing duplicate launch\n' >&2
  exit 1
fi

read -r -a candidate_gpus <<<"${GPU_CANDIDATES}"
selected_gpus=()
while [[ "${#selected_gpus[@]}" -lt "${#ARMS[@]}" ]]; do
  selected_gpus=()
  remaining=()
  for gpu in "${candidate_gpus[@]}"; do
    free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${gpu}" | tr -d ' ')
    util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "${gpu}" | tr -d ' ')
    printf '%s critic_v2_gpu_candidate=%s free_mb=%s util=%s\n' \
      "$(date -Is)" "${gpu}" "${free_mb}" "${util}"
    if [[ "${free_mb}" -ge "${MINIMUM_FREE_MB}" ]]; then
      remaining+=("${free_mb}:${gpu}")
    fi
  done
  if [[ "${#remaining[@]}" -ge "${#ARMS[@]}" ]]; then
    mapfile -t selected_gpus < <(
      printf '%s\n' "${remaining[@]}" | sort -t: -k1,1nr | head -n "${#ARMS[@]}" | cut -d: -f2
    )
    break
  fi
  printf '%s waiting_for_four_critic_v2_gpus minimum_free_mb=%s available=%s\n' \
    "$(date -Is)" "${MINIMUM_FREE_MB}" "${#remaining[@]}"
  sleep "${POLL_SECONDS}"
done

mkdir -p "${LOG_ROOT}"
cd "${REPO_ROOT}"
pids=()
for index in "${!ARMS[@]}"; do
  arm="${ARMS[$index]}"
  gpu="${selected_gpus[$index]}"
  runtime_config="${RUNTIME_CONFIG_ROOT}/${arm}.json"
  "${PYTHON}" scripts/route_a_v3/prepare_route2_mrnabert_critic_v2_controls_v1.py \
    --base-config "${BASE_CONFIG}" \
    --protocol "${PROTOCOL}" \
    --arm "${arm}" \
    --gpu "${gpu}" \
    --output-config "${runtime_config}"
  printf '%s starting_critic_v2_arm=%s gpu=%s\n' "$(date -Is)" "${arm}" "${gpu}"
  "${PYTHON}" -u scripts/route_a_v3/train_route2_delta_predictor_v1.py \
    --config "${runtime_config}" \
    >"${LOG_ROOT}/${arm}.log" 2>&1 &
  pids+=("$!")
done

failed=0
set +e
for index in "${!pids[@]}"; do
  wait "${pids[$index]}"
  status=$?
  printf '%s critic_v2_arm_finished=%s status=%s\n' \
    "$(date -Is)" "${ARMS[$index]}" "${status}"
  if [[ "${status}" -ne 0 ]]; then
    failed=1
  fi
done
set -e
if [[ "${failed}" -ne 0 ]]; then
  printf '%s critic_v2_controls_failed_preserving_evidence\n' "$(date -Is)" >&2
  exit 1
fi

"${PYTHON}" scripts/route_a_v3/adjudicate_route2_mrnabert_critic_v2_controls_v1.py \
  --protocol "${PROTOCOL}" \
  --full-summary "${RUN_ROOT}/full/training_summary.json" \
  --candidate-permutation-summary "${RUN_ROOT}/candidate_permutation/training_summary.json" \
  --source-only-summary "${RUN_ROOT}/source_only/training_summary.json" \
  --source-edit-metadata-summary "${RUN_ROOT}/source_edit_metadata/training_summary.json" \
  --output "${ADJUDICATION}"

supports_seeds=$("${PYTHON}" -c \
  'import json,sys; print(str(json.load(open(sys.argv[1]))["supports_three_frozen_seeds"]).lower())' \
  "${ADJUDICATION}")
if [[ "${supports_seeds}" == "true" ]]; then
  printf '%s critic_v2_controls_passed_three_frozen_seeds_authorized_not_started\n' "$(date -Is)"
else
  printf '%s critic_v2_controls_terminal_no_go_test_remains_closed\n' "$(date -Is)"
fi
