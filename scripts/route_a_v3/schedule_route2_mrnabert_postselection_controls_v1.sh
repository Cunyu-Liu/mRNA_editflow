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
