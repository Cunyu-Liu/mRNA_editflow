#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/cunyuliu/mrna_editflow_goal/worktrees/route_a_v3_route2_method_repair_20260817}"
ROUTE2_ROOT="${ROUTE2_ROOT:-/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2}"
PYTHON="${PYTHON:-/home/cunyuliu/miniconda3/envs/editflow/bin/python}"
POLL_SECONDS="${POLL_SECONDS:-900}"

RUN_ROOT="${ROUTE2_ROOT}/runs/mrnabert_scaleup_v2"
HUBER_DIR="${RUN_ROOT}/max_mean_only_seed20260816_gpu0_bf16_v1"
FIXED_DIR="${RUN_ROOT}/max_fixed_variance_seed20260816_gpu5_bf16_v1"
LEARNED_DIR="${RUN_ROOT}/max_learned_variance_seed20260816_gpu3_bf16_v1"
COMPARISON="${ROUTE2_ROOT}/comparisons/mrnabert_loss_comparison_seed20260816_v1.json"
CONTROL_ADJUDICATION="${ROUTE2_ROOT}/comparisons/mrnabert_signal_control_adjudication_seed20260816_v1.json"
RUNTIME_CONFIG_ROOT="${RUN_ROOT}/runtime_configs"
PERMUTATION_CONFIG="${RUNTIME_CONFIG_ROOT}/selected_loss_candidate_permutation_seed20260816_gpu0_v1.json"
SOURCE_ONLY_CONFIG="${RUNTIME_CONFIG_ROOT}/selected_loss_source_only_seed20260816_gpu5_v1.json"
PERMUTATION_RUN="${RUN_ROOT}/selected_loss_candidate_permutation_seed20260816_gpu0_v1"
SOURCE_ONLY_RUN="${RUN_ROOT}/selected_loss_source_only_seed20260816_gpu5_v1"
FINAL_CONFIG_DIR="${RUNTIME_CONFIG_ROOT}/final_seed_validation_v1"
FINAL_RUN_ROOT="${RUN_ROOT}/final_seed_validation_v1"
THREE_SEED_ADJUDICATION="${ROUTE2_ROOT}/comparisons/mrnabert_three_seed_adjudication_v1.json"
FROZEN_TEST_CONFIG="${RUNTIME_CONFIG_ROOT}/selected_loss_frozen_development_test_seed20260823_gpu0_v1.json"
FROZEN_TEST_RUN="${RUN_ROOT}/selected_loss_frozen_development_test_seed20260823_gpu0_v1"
FINAL_REFIT_CONFIG="${RUNTIME_CONFIG_ROOT}/selected_loss_all126165_seed20260823_gpu0_v1.json"
FINAL_REFIT_RUN="${RUN_ROOT}/selected_loss_all126165_seed20260823_gpu0_v1"
LOSO_CONFIG_DIR="${RUNTIME_CONFIG_ROOT}/test_preserving_loso_v1"
LOSO_RUN_ROOT="${RUN_ROOT}/test_preserving_loso_v1"
BASELINE_LOSO_CONFIG_DIR="${RUNTIME_CONFIG_ROOT}/global_scaled_test_preserving_loso_v1"
BASELINE_LOSO_RUN_ROOT="${RUN_ROOT}/global_scaled_test_preserving_loso_v1"
LOSO_AGGREGATION_INPUT_DIR="${ROUTE2_ROOT}/comparisons/mrnabert_test_preserving_loso_inputs_v1"
LOSO_AGGREGATION_DIR="${ROUTE2_ROOT}/comparisons/mrnabert_test_preserving_loso_v1"
FLOW_V2_TRAINING_SUMMARY="${ROUTE2_ROOT}/runs/base_flow_g0/position_progress_gpu_v2/training_summary.json"
FLOW_V2_VALIDATION_SUMMARY="${ROUTE2_ROOT}/runs/base_flow_g0/position_progress_validation_gpu_v2/validation_summary.json"
ONLINE_ENCODER_VALIDATION="${ROUTE2_ROOT}/runs/mrnabert_online_encoder_validation_v1/validation_summary.json"
READINESS_INPUT="${ROUTE2_ROOT}/comparisons/mrnabert_guidance_readiness_input_v1.json"
READINESS_ADJUDICATION="${ROUTE2_ROOT}/comparisons/mrnabert_guidance_readiness_adjudication_v1.json"

summaries=(
  "${HUBER_DIR}/training_summary.json"
  "${FIXED_DIR}/training_summary.json"
  "${LEARNED_DIR}/training_summary.json"
)

while true; do
  missing=0
  for summary in "${summaries[@]}"; do
    if [[ ! -f "${summary}" ]]; then
      missing=$((missing + 1))
    fi
  done
  printf '%s completed_loss_summaries=%s/3\n' "$(date -Is)" "$((3 - missing))"
  if [[ "${missing}" -eq 0 ]]; then
    break
  fi
  sleep "${POLL_SECONDS}"
done

cd "${REPO_ROOT}"
"${PYTHON}" scripts/route_a_v3/summarize_route2_mrnabert_loss_comparison_v1.py \
  --summary "${summaries[0]}" \
  --summary "${summaries[1]}" \
  --summary "${summaries[2]}" \
  --output "${COMPARISON}"

selected_loss=$("${PYTHON}" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["selected_loss_for_controls"])' \
  "${COMPARISON}")
case "${selected_loss}" in
  huber)
    selected_config="${HUBER_DIR}/training_config.json"
    selected_summary="${HUBER_DIR}/training_summary.json"
    ;;
  fixed_variance_gaussian_nll)
    selected_config="${FIXED_DIR}/training_config.json"
    selected_summary="${FIXED_DIR}/training_summary.json"
    ;;
  learned_variance_gaussian_nll)
    selected_config="${LEARNED_DIR}/training_config.json"
    selected_summary="${LEARNED_DIR}/training_summary.json"
    ;;
  *)
    printf 'unexpected selected loss: %s\n' "${selected_loss}" >&2
    exit 1
    ;;
esac

"${PYTHON}" scripts/route_a_v3/prepare_route2_mrnabert_signal_controls_v1.py \
  --selected-config "${selected_config}" \
  --candidate-permutation-config "${PERMUTATION_CONFIG}" \
  --source-only-config "${SOURCE_ONLY_CONFIG}" \
  --candidate-permutation-run-dir "${PERMUTATION_RUN}" \
  --source-only-run-dir "${SOURCE_ONLY_RUN}" \
  --candidate-permutation-gpu 0 \
  --source-only-gpu 5

printf '%s starting_selected_loss_controls=%s\n' "$(date -Is)" "${selected_loss}"
"${PYTHON}" -u scripts/route_a_v3/train_route2_delta_predictor_v1.py \
  --config "${PERMUTATION_CONFIG}" \
  >"${ROUTE2_ROOT}/mrnabert_selected_loss_candidate_permutation_gpu0_v1.log" 2>&1 &
permutation_pid=$!
"${PYTHON}" -u scripts/route_a_v3/train_route2_delta_predictor_v1.py \
  --config "${SOURCE_ONLY_CONFIG}" \
  >"${ROUTE2_ROOT}/mrnabert_selected_loss_source_only_gpu5_v1.log" 2>&1 &
source_only_pid=$!

set +e
wait "${permutation_pid}"
permutation_status=$?
wait "${source_only_pid}"
source_only_status=$?
set -e
printf '%s controls_finished permutation_status=%s source_only_status=%s\n' \
  "$(date -Is)" "${permutation_status}" "${source_only_status}"
if [[ "${permutation_status}" -ne 0 || "${source_only_status}" -ne 0 ]]; then
  exit 1
fi

"${PYTHON}" scripts/route_a_v3/adjudicate_route2_mrnabert_signal_controls_v1.py \
  --protocol configs/route_a_v3_route2_mrnabert_signal_control_gate_v1.json \
  --comparison "${COMPARISON}" \
  --primary-summary "${selected_summary}" \
  --permutation-summary "${PERMUTATION_RUN}/training_summary.json" \
  --source-only-summary "${SOURCE_ONLY_RUN}/training_summary.json" \
  --output "${CONTROL_ADJUDICATION}"

supports_final_seeds=$("${PYTHON}" -c \
  'import json,sys; print(str(json.load(open(sys.argv[1]))["supports_final_seed_confirmation"]).lower())' \
  "${CONTROL_ADJUDICATION}")
if [[ "${supports_final_seeds}" != "true" ]]; then
  printf '%s signal_controls_stop_before_final_seeds\n' "$(date -Is)"
  exit 0
fi

"${PYTHON}" scripts/route_a_v3/prepare_route2_mrnabert_final_seed_configs_v1.py \
  --selected-config "${selected_config}" \
  --signal-adjudication "${CONTROL_ADJUDICATION}" \
  --output-config-dir "${FINAL_CONFIG_DIR}" \
  --run-root "${FINAL_RUN_ROOT}"

final_seed_pids=()
for seed_gpu in "20260822:0" "20260823:3" "20260824:5"; do
  seed=${seed_gpu%%:*}
  gpu=${seed_gpu##*:}
  config="${FINAL_CONFIG_DIR}/mrnabert_edit_centered_${selected_loss}_final_seed${seed}.json"
  log="${ROUTE2_ROOT}/mrnabert_final_seed${seed}_gpu${gpu}_v1.log"
  "${PYTHON}" -u scripts/route_a_v3/train_route2_delta_predictor_v1.py \
    --config "${config}" >"${log}" 2>&1 &
  final_seed_pids+=("$!")
done

final_seed_failure=0
set +e
for pid in "${final_seed_pids[@]}"; do
  wait "${pid}"
  status=$?
  if [[ "${status}" -ne 0 ]]; then
    final_seed_failure=1
  fi
done
set -e
printf '%s final_seed_runs_finished failure=%s\n' "$(date -Is)" "${final_seed_failure}"
if [[ "${final_seed_failure}" -ne 0 ]]; then
  exit 1
fi

final_seed_summaries=()
for seed_gpu in "20260822:0" "20260823:3" "20260824:5"; do
  seed=${seed_gpu%%:*}
  gpu=${seed_gpu##*:}
  final_seed_summaries+=(
    "${FINAL_RUN_ROOT}/seed${seed}_gpu${gpu}_${selected_loss}_v1/training_summary.json"
  )
done

"${PYTHON}" scripts/route_a_v3/adjudicate_route2_mrnabert_three_seeds_v1.py \
  --protocol configs/route_a_v3_route2_mrnabert_three_seed_gate_v1.json \
  --summary "${final_seed_summaries[0]}" \
  --summary "${final_seed_summaries[1]}" \
  --summary "${final_seed_summaries[2]}" \
  --output "${THREE_SEED_ADJUDICATION}"

supports_frozen_test=$("${PYTHON}" -c \
  'import json,sys; print(str(json.load(open(sys.argv[1]))["supports_single_frozen_development_test"]).lower())' \
  "${THREE_SEED_ADJUDICATION}")
if [[ "${supports_frozen_test}" != "true" ]]; then
  printf '%s three_seed_gate_stop_before_frozen_development_test\n' "$(date -Is)"
  exit 0
fi

selected_seed_config="${FINAL_CONFIG_DIR}/mrnabert_edit_centered_${selected_loss}_final_seed20260823.json"
"${PYTHON}" scripts/route_a_v3/prepare_route2_mrnabert_frozen_test_config_v1.py \
  --selected-config "${selected_seed_config}" \
  --three-seed-adjudication "${THREE_SEED_ADJUDICATION}" \
  --gpu 0 \
  --output-directory "${FROZEN_TEST_RUN}" \
  --output-config "${FROZEN_TEST_CONFIG}"

printf '%s starting_single_frozen_development_test loss=%s seed=20260823 gpu=0\n' \
  "$(date -Is)" "${selected_loss}"
"${PYTHON}" -u scripts/route_a_v3/train_route2_delta_predictor_v1.py \
  --config "${FROZEN_TEST_CONFIG}" \
  >"${ROUTE2_ROOT}/mrnabert_frozen_development_test_seed20260823_gpu0_v1.log" 2>&1

"${PYTHON}" scripts/route_a_v3/prepare_route2_mrnabert_all_development_refit_config_v1.py \
  --frozen-test-config "${FROZEN_TEST_CONFIG}" \
  --frozen-test-summary "${FROZEN_TEST_RUN}/training_summary.json" \
  --gpu 0 \
  --output-directory "${FINAL_REFIT_RUN}" \
  --output-config "${FINAL_REFIT_CONFIG}"

printf '%s starting_final_all126165_refit loss=%s seed=20260823 gpu=0\n' \
  "$(date -Is)" "${selected_loss}"
"${PYTHON}" -u scripts/route_a_v3/train_route2_delta_predictor_v1.py \
  --config "${FINAL_REFIT_CONFIG}" \
  >"${ROUTE2_ROOT}/mrnabert_final_all126165_seed20260823_gpu0_v1.log" 2>&1
printf '%s final_all126165_refit_finished\n' "$(date -Is)"

"${PYTHON}" scripts/route_a_v3/prepare_route2_mrnabert_test_preserving_loso_configs_v1.py \
  --selected-config "${selected_config}" \
  --three-seed-adjudication "${THREE_SEED_ADJUDICATION}" \
  --output-config-dir "${LOSO_CONFIG_DIR}" \
  --run-root "${LOSO_RUN_ROOT}"

loso_studies=(
  GSE200304 GSE114002 GSE149487 GSE217518 ENCSR854RUF GSE186455 GSE269595
)
for study in "${loso_studies[@]}"; do
  study_label=$(printf '%s' "${study}" | tr '[:upper:]-' '[:lower:]_')
  loso_pids=()
  for seed_gpu in "20260822:0" "20260823:3" "20260824:5"; do
    seed=${seed_gpu%%:*}
    gpu=${seed_gpu##*:}
    config="${LOSO_CONFIG_DIR}/mrnabert_${selected_loss}_loso_${study_label}_seed${seed}.json"
    log="${ROUTE2_ROOT}/mrnabert_loso_${study_label}_seed${seed}_gpu${gpu}_v1.log"
    "${PYTHON}" -u scripts/route_a_v3/train_route2_delta_predictor_v1.py \
      --config "${config}" >"${log}" 2>&1 &
    loso_pids+=("$!")
  done
  loso_failure=0
  set +e
  for pid in "${loso_pids[@]}"; do
    wait "${pid}"
    status=$?
    if [[ "${status}" -ne 0 ]]; then
      loso_failure=1
    fi
  done
  set -e
  printf '%s test_preserving_loso_finished study=%s failure=%s\n' \
    "$(date -Is)" "${study}" "${loso_failure}"
  if [[ "${loso_failure}" -ne 0 ]]; then
    exit 1
  fi
done
printf '%s all_test_preserving_loso_runs_finished test_remained_excluded=true\n' "$(date -Is)"

"${PYTHON}" scripts/route_a_v3/prepare_route2_global_scaled_test_preserving_loso_configs_v1.py \
  --base-config configs/route_a_v3_route2_method_repair_global_scaled_seed20260821_gpu0_v1.json \
  --three-seed-adjudication "${THREE_SEED_ADJUDICATION}" \
  --output-config-dir "${BASELINE_LOSO_CONFIG_DIR}" \
  --run-root "${BASELINE_LOSO_RUN_ROOT}"

for study in "${loso_studies[@]}"; do
  study_label=$(printf '%s' "${study}" | tr '[:upper:]-' '[:lower:]_')
  baseline_loso_pids=()
  for seed_gpu in "20260822:0" "20260823:3" "20260824:5"; do
    seed=${seed_gpu%%:*}
    gpu=${seed_gpu##*:}
    config="${BASELINE_LOSO_CONFIG_DIR}/global_scaled_loso_${study_label}_seed${seed}.json"
    log="${ROUTE2_ROOT}/global_scaled_loso_${study_label}_seed${seed}_gpu${gpu}_v1.log"
    "${PYTHON}" -u scripts/route_a_v3/train_route2_delta_predictor_v1.py \
      --config "${config}" >"${log}" 2>&1 &
    baseline_loso_pids+=("$!")
  done
  baseline_loso_failure=0
  set +e
  for pid in "${baseline_loso_pids[@]}"; do
    wait "${pid}"
    status=$?
    if [[ "${status}" -ne 0 ]]; then
      baseline_loso_failure=1
    fi
  done
  set -e
  printf '%s matched_baseline_test_preserving_loso_finished study=%s failure=%s\n' \
    "$(date -Is)" "${study}" "${baseline_loso_failure}"
  if [[ "${baseline_loso_failure}" -ne 0 ]]; then
    exit 1
  fi
done
printf '%s all_matched_baseline_test_preserving_loso_runs_finished\n' "$(date -Is)"

"${PYTHON}" scripts/route_a_v3/build_route2_test_preserving_loso_aggregation_inputs_v1.py \
  --model-run-root "${LOSO_RUN_ROOT}" \
  --baseline-run-root "${BASELINE_LOSO_RUN_ROOT}" \
  --loss-kind "${selected_loss}" \
  --output-dir "${LOSO_AGGREGATION_INPUT_DIR}"

if [[ -e "${LOSO_AGGREGATION_DIR}" ]]; then
  printf 'LOSO aggregation output directory already exists: %s\n' "${LOSO_AGGREGATION_DIR}" >&2
  exit 1
fi
mkdir -p "${LOSO_AGGREGATION_DIR}"
for seed in 20260822 20260823 20260824; do
  "${PYTHON}" scripts/route_a_v3/aggregate_route2_loso_v1.py \
    --input "${LOSO_AGGREGATION_INPUT_DIR}/test_preserving_loso_aggregation_input_seed${seed}.json" \
    --output "${LOSO_AGGREGATION_DIR}/test_preserving_loso_seed${seed}.json"
done
printf '%s three_test_preserving_loso_aggregations_finished\n' "$(date -Is)"

while [[ ! -f "${FLOW_V2_TRAINING_SUMMARY}" \
  || ! -f "${FLOW_V2_VALIDATION_SUMMARY}" \
  || ! -f "${ONLINE_ENCODER_VALIDATION}" ]]; do
  printf '%s waiting_for_flow_v2_and_online_encoder_validation\n' "$(date -Is)"
  sleep "${POLL_SECONDS}"
done

"${PYTHON}" scripts/route_a_v3/build_route2_mrnabert_guidance_readiness_input_v1.py \
  --validation-training-summary "${FINAL_RUN_ROOT}/seed20260823_gpu3_${selected_loss}_v1/training_summary.json" \
  --final-refit-summary "${FINAL_REFIT_RUN}/training_summary.json" \
  --final-refit-checkpoint "${FINAL_REFIT_RUN}/delta_predictor_checkpoint.pt" \
  --signal-adjudication "${CONTROL_ADJUDICATION}" \
  --loso-result "${LOSO_AGGREGATION_DIR}/test_preserving_loso_seed20260822.json" \
  --loso-result "${LOSO_AGGREGATION_DIR}/test_preserving_loso_seed20260823.json" \
  --loso-result "${LOSO_AGGREGATION_DIR}/test_preserving_loso_seed20260824.json" \
  --flow-training-summary "${FLOW_V2_TRAINING_SUMMARY}" \
  --flow-validation-summary "${FLOW_V2_VALIDATION_SUMMARY}" \
  --reward-policy configs/route_a_v3_route2_mrnabert_guidance_reward_policy_v1.json \
  --online-encoder-validation "${ONLINE_ENCODER_VALIDATION}" \
  --output "${READINESS_INPUT}"

"${PYTHON}" scripts/route_a_v3/adjudicate_route2_readiness_v1.py \
  --input "${READINESS_INPUT}" \
  --output "${READINESS_ADJUDICATION}"

guided_unlocked=$("${PYTHON}" -c \
  'import json,sys; print(str(json.load(open(sys.argv[1]))["guided_unlocked"]).lower())' \
  "${READINESS_ADJUDICATION}")
if [[ "${guided_unlocked}" != "true" ]]; then
  printf '%s readiness_stop_before_guided_xeditflow\n' "$(date -Is)"
  exit 0
fi
printf '%s critic_and_flow_ready_guided_runner_is_separate_next_step\n' "$(date -Is)"
