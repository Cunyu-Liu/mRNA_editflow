#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/cunyuliu/mrna_editflow_goal/worktrees/route_a_v3_route2_method_repair_20260817}"
ROUTE2_ROOT="${ROUTE2_ROOT:-/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2}"
PYTHON="${PYTHON:-/home/cunyuliu/miniconda3/envs/editflow/bin/python}"
POLL_SECONDS="${POLL_SECONDS:-900}"
MINIMUM_FREE_MB="${MINIMUM_FREE_MB:-4096}"
GPU_CANDIDATES="${GPU_CANDIDATES:-0 1 2 3 4 5}"

BASE_CONFIG="${REPO_ROOT}/configs/route_a_v3_route2_mrnabert_edit_max_mean_only_gpu6_v1.json"
CONTROL_PROTOCOL="${REPO_ROOT}/configs/route_a_v3_route2_mrnabert_critic_v2_protocol_v1.json"
PROTOCOL="${REPO_ROOT}/configs/route_a_v3_route2_mrnabert_critic_v2_three_seed_protocol_v1.json"
CONTROL_ADJUDICATION="${ROUTE2_ROOT}/comparisons/mrnabert_critic_v2_task_study_macro_controls_adjudication_v1.json"
RUNTIME_CONFIG_ROOT="${ROUTE2_ROOT}/runs/mrnabert_critic_v2/runtime_configs/task_study_macro_confirmation_seeds_v1"
RUN_ROOT="${ROUTE2_ROOT}/runs/mrnabert_critic_v2/task_study_macro_confirmation_seeds_v1"
LOG_ROOT="${ROUTE2_ROOT}/logs/mrnabert_critic_v2/task_study_macro_confirmation_seeds_v1"
ADJUDICATION="${ROUTE2_ROOT}/comparisons/mrnabert_critic_v2_task_study_macro_three_seed_adjudication_v1.json"
SEEDS=(20260822 20260823 20260824)

if [[ -e "${RUNTIME_CONFIG_ROOT}" || -e "${RUN_ROOT}" || -e "${ADJUDICATION}" ]]; then
  printf 'Critic V2 confirmation runtime or terminal already exists; refusing duplicate launch\n' >&2
  exit 1
fi

while [[ ! -f "${CONTROL_ADJUDICATION}" ]]; do
  printf '%s waiting_for_critic_v2_control_adjudication\n' "$(date -Is)"
  sleep "${POLL_SECONDS}"
done

supports_seeds=$("${PYTHON}" -c \
  'import json,sys; p=json.load(open(sys.argv[1])); print(str(p.get("status") == "CRITIC_V2_CONTROLS_SUPPORT_THREE_FROZEN_SEEDS" and p.get("supports_three_frozen_seeds") is True).lower())' \
  "${CONTROL_ADJUDICATION}")
if [[ "${supports_seeds}" != "true" ]]; then
  printf '%s critic_v2_control_gate_terminal_no_go_confirmation_seeds_not_started\n' "$(date -Is)"
  exit 0
fi

read -r -a candidate_gpus <<<"${GPU_CANDIDATES}"
selected_gpus=()
while [[ "${#selected_gpus[@]}" -lt "${#SEEDS[@]}" ]]; do
  selected_gpus=()
  remaining=()
  for gpu in "${candidate_gpus[@]}"; do
    free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${gpu}" | tr -d ' ')
    util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "${gpu}" | tr -d ' ')
    printf '%s critic_v2_confirmation_gpu_candidate=%s free_mb=%s util=%s\n' \
      "$(date -Is)" "${gpu}" "${free_mb}" "${util}"
    if [[ "${free_mb}" -ge "${MINIMUM_FREE_MB}" ]]; then
      remaining+=("${free_mb}:${gpu}")
    fi
  done
  if [[ "${#remaining[@]}" -ge "${#SEEDS[@]}" ]]; then
    mapfile -t selected_gpus < <(
      printf '%s\n' "${remaining[@]}" | sort -t: -k1,1nr | head -n "${#SEEDS[@]}" | cut -d: -f2
    )
    break
  fi
  printf '%s waiting_for_three_critic_v2_confirmation_gpus minimum_free_mb=%s available=%s\n' \
    "$(date -Is)" "${MINIMUM_FREE_MB}" "${#remaining[@]}"
  sleep "${POLL_SECONDS}"
done

cd "${REPO_ROOT}"
gpu_args=()
for gpu in "${selected_gpus[@]}"; do
  gpu_args+=(--gpu "${gpu}")
done
"${PYTHON}" scripts/route_a_v3/prepare_route2_mrnabert_critic_v2_three_seed_configs_v1.py \
  --base-config "${BASE_CONFIG}" \
  --control-protocol "${CONTROL_PROTOCOL}" \
  --confirmation-protocol "${PROTOCOL}" \
  --control-adjudication "${CONTROL_ADJUDICATION}" \
  "${gpu_args[@]}" \
  --output-config-dir "${RUNTIME_CONFIG_ROOT}"

mkdir -p "${LOG_ROOT}"
pids=()
for index in "${!SEEDS[@]}"; do
  seed="${SEEDS[$index]}"
  gpu="${selected_gpus[$index]}"
  printf '%s starting_critic_v2_confirmation_seed=%s gpu=%s\n' "$(date -Is)" "${seed}" "${gpu}"
  "${PYTHON}" -u scripts/route_a_v3/train_route2_delta_predictor_v1.py \
    --config "${RUNTIME_CONFIG_ROOT}/seed${seed}.json" \
    >"${LOG_ROOT}/seed${seed}.log" 2>&1 &
  pids+=("$!")
done

failed=0
set +e
for index in "${!pids[@]}"; do
  wait "${pids[$index]}"
  status=$?
  printf '%s critic_v2_confirmation_seed_finished=%s status=%s\n' \
    "$(date -Is)" "${SEEDS[$index]}" "${status}"
  if [[ "${status}" -ne 0 ]]; then
    failed=1
  fi
done
set -e
if [[ "${failed}" -ne 0 ]]; then
  printf '%s critic_v2_confirmation_failed_preserving_evidence\n' "$(date -Is)" >&2
  exit 1
fi

summary_args=()
for seed in "${SEEDS[@]}"; do
  summary_args+=(--summary "${RUN_ROOT}/seed${seed}/training_summary.json")
done
"${PYTHON}" scripts/route_a_v3/adjudicate_route2_mrnabert_critic_v2_three_seeds_v1.py \
  --protocol "${PROTOCOL}" \
  --control-adjudication "${CONTROL_ADJUDICATION}" \
  "${summary_args[@]}" \
  --output "${ADJUDICATION}"

supports_test=$("${PYTHON}" -c \
  'import json,sys; print(str(json.load(open(sys.argv[1]))["supports_single_frozen_development_test"]).lower())' \
  "${ADJUDICATION}")
if [[ "${supports_test}" == "true" ]]; then
  printf '%s critic_v2_three_seeds_passed_frozen_test_authorized_not_started\n' "$(date -Is)"
else
  printf '%s critic_v2_three_seeds_terminal_no_go_test_remains_closed\n' "$(date -Is)"
fi
