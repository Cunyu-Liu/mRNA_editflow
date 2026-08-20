#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/cunyuliu/mrna_editflow_goal/worktrees/route_a_v3_route2_method_repair_20260817}"
ROUTE2_ROOT="${ROUTE2_ROOT:-/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2}"
PYTHON="${PYTHON:-/home/cunyuliu/miniconda3/envs/editflow/bin/python}"
POLL_SECONDS="${POLL_SECONDS:-900}"
GPU_CANDIDATES="${GPU_CANDIDATES:-0 1 2 3 4 5}"
MINIMUM_FREE_MB="${MINIMUM_FREE_MB:-4096}"

FLOW_SUMMARY="${ROUTE2_ROOT}/runs/base_flow_g0/position_progress_validation_gpu_v2/validation_summary.json"
FLOW_CHECKPOINT="${ROUTE2_ROOT}/runs/base_flow_g0/position_progress_gpu_v2/best.pt"
EVALUATOR_ADJUDICATION="${ROUTE2_ROOT}/comparisons/mrnabert_independent_evaluator_adjudication_v1.json"
EVALUATOR_RUN="${ROUTE2_ROOT}/runs/independent_generation_evaluator/neural_medium_siamese_task_scaled_seed20260816_frozen_development_validation_gpu2_v3"
EVALUATOR_SUMMARY="${EVALUATOR_RUN}/training_summary.json"
EVALUATOR_CHECKPOINT="${EVALUATOR_RUN}/delta_predictor_checkpoint.pt"
GUIDING_CHECKPOINT="${ROUTE2_ROOT}/runs/development_frozen_validation/delta_main_2m_lr3e4_seed20260816_frozen_development_validation_gpu4_v1/delta_predictor_checkpoint.pt"
PROTOCOL_TEMPLATE="${REPO_ROOT}/configs/route_a_v3_route2_generation_matched_compute_repair_protocol_v1.json"
JOBS_TEMPLATE="${REPO_ROOT}/configs/route_a_v3_route2_generation_independent_evaluator_jobs_gpu6_v1.json"
FLOW_TEMPLATE="${REPO_ROOT}/configs/route_a_v3_route2_base_flow_g0_matched_compute_candidates_seed20260816_gpu6_v1.json"
RUNTIME_ROOT="${ROUTE2_ROOT}/runs/generation_search_baselines/runtime_configs_v2"
OUTPUT_ROOT="${ROUTE2_ROOT}/runs/generation_search_baselines/matched_compute_position_progress_seed20260816_v2"
AUDIT_ROOT="${ROUTE2_ROOT}/audits/generation_search_baselines/matched_compute_position_progress_seed20260816_v2"
FLOW_OUTPUT="${ROUTE2_ROOT}/runs/base_flow_g0/matched_compute_position_progress_seed20260816_v2"
SUITE_SUMMARY="${AUDIT_ROOT}/matched_generation_suite_summary_v2.json"

while [[ ! -f "${FLOW_SUMMARY}" || ! -f "${EVALUATOR_ADJUDICATION}" || ! -f "${EVALUATOR_SUMMARY}" ]]; do
  printf '%s waiting_for_flow_validation_and_independent_evaluator\n' "$(date -Is)"
  sleep "${POLL_SECONDS}"
done

"${PYTHON}" - "${FLOW_SUMMARY}" "${EVALUATOR_ADJUDICATION}" "${EVALUATOR_SUMMARY}" <<'PY'
import json
import sys

flow = json.load(open(sys.argv[1]))
adjudication = json.load(open(sys.argv[2]))
evaluator = json.load(open(sys.argv[3]))
if flow.get("status") != "FLOW_G0_READY":
    raise SystemExit("Base Flow validation is not ready")
if flow.get("evaluation_outcomes_read") != 0:
    raise SystemExit("Base Flow validation accessed Evaluation")
if adjudication.get("status") not in {
    "INDEPENDENT_GENERATION_EVALUATOR_QUALIFIED",
    "INDEPENDENT_GENERATION_EVALUATOR_NO_GO",
}:
    raise SystemExit("independent evaluator adjudication is not terminal")
if adjudication.get("development_test_outcomes_accessed") is not False:
    raise SystemExit("independent evaluator accessed Development TEST")
if adjudication.get("evaluation_outcomes_accessed") is not False:
    raise SystemExit("independent evaluator accessed Evaluation")
if evaluator.get("status") != "DELTA_PREDICTOR_DEVELOPMENT_GPU_RUN_COMPLETE":
    raise SystemExit("independent evaluator training is incomplete")
if evaluator.get("evaluation_outcomes_read") != 0:
    raise SystemExit("independent evaluator training accessed Evaluation")
PY

evaluator_training_gpu=$("${PYTHON}" -c \
  'import json,sys; print(int(json.load(open(sys.argv[1]))["physical_gpu_index"]))' \
  "${EVALUATOR_SUMMARY}")
EVALUATOR_RUNTIME_CONFIG="${ROUTE2_ROOT}/runs/independent_generation_evaluator/runtime_configs/independent_evaluator_gpu${evaluator_training_gpu}_v3.json"

for required_path in "${FLOW_CHECKPOINT}" "${EVALUATOR_CHECKPOINT}" "${GUIDING_CHECKPOINT}" "${EVALUATOR_RUNTIME_CONFIG}"; do
  if [[ ! -f "${required_path}" ]]; then
    printf 'required checkpoint is absent: %s\n' "${required_path}" >&2
    exit 1
  fi
done

read -r -a candidate_gpus <<<"${GPU_CANDIDATES}"
selected_gpu=""
while [[ -z "${selected_gpu}" ]]; do
  best_free_mb=-1
  for gpu in "${candidate_gpus[@]}"; do
    free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${gpu}" | tr -d ' ')
    util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "${gpu}" | tr -d ' ')
    printf '%s matched_generation_gpu_candidate=%s free_mb=%s util=%s\n' \
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

mkdir -p "${RUNTIME_ROOT}"
protocol_runtime="${RUNTIME_ROOT}/matched_compute_protocol_gpu${selected_gpu}_v2.json"
jobs_runtime="${RUNTIME_ROOT}/independent_evaluator_jobs_gpu${selected_gpu}_v2.json"
flow_runtime="${RUNTIME_ROOT}/base_flow_matched_candidates_gpu${selected_gpu}_v2.json"
for runtime_path in "${protocol_runtime}" "${jobs_runtime}" "${flow_runtime}" "${SUITE_SUMMARY}"; do
  if [[ -e "${runtime_path}" ]]; then
    printf 'matched-generation runtime/output already exists: %s\n' "${runtime_path}" >&2
    exit 1
  fi
done

"${PYTHON}" - \
  "${PROTOCOL_TEMPLATE}" "${JOBS_TEMPLATE}" "${FLOW_TEMPLATE}" \
  "${protocol_runtime}" "${jobs_runtime}" "${flow_runtime}" \
  "${selected_gpu}" "${MINIMUM_FREE_MB}" \
  "${OUTPUT_ROOT}" "${AUDIT_ROOT}" "${FLOW_OUTPUT}" \
  "${FLOW_CHECKPOINT}" "${EVALUATOR_CHECKPOINT}" "${GUIDING_CHECKPOINT}" \
  "${EVALUATOR_RUNTIME_CONFIG}" <<'PY'
import json
import sys
from pathlib import Path

(
    protocol_template,
    jobs_template,
    flow_template,
    protocol_output,
    jobs_output,
    flow_output_config,
    gpu_text,
    minimum_free_mb_text,
    candidate_root_text,
    audit_root_text,
    flow_output_text,
    flow_checkpoint_text,
    evaluator_checkpoint_text,
    guiding_checkpoint_text,
    evaluator_runtime_config_text,
) = sys.argv[1:]
gpu = int(gpu_text)
device = f"cuda:{gpu}"
candidate_root = Path(candidate_root_text)
flow_output = Path(flow_output_text)

protocol = json.load(open(protocol_template))
protocol.update({
    "candidate_output_root": str(candidate_root),
    "independent_evaluation_output_root": audit_root_text,
    "guiding_checkpoint_path": guiding_checkpoint_text,
    "independent_evaluator_config": evaluator_runtime_config_text,
    "independent_evaluator_checkpoint_path": evaluator_checkpoint_text,
    "execution_device": device,
    "physical_gpu_index": gpu,
    "candidate_generation_may_continue_when_no_go": True,
    "runtime_gpu_selection_policy": "MOST_FREE_MEMORY_GPU_0_TO_5_WITH_TASK_MINIMUM_NO_UTILIZATION_GATE",
    "runtime_gpu_selection_minimum_free_mb": int(minimum_free_mb_text),
})

jobs = json.load(open(jobs_template))
jobs.update({
    "evaluator_checkpoint_path": evaluator_checkpoint_text,
    "guiding_checkpoint_path": guiding_checkpoint_text,
    "source_manifest_path": protocol["source_manifest_path"],
    "device": device,
    "physical_gpu_index": gpu,
})
for job in jobs["jobs"]:
    method = str(job["method_id"])
    if method == "unguided_learned_base_flow_g0":
        job["candidate_path"] = str(flow_output / "trajectories.private.jsonl")
    else:
        job["candidate_path"] = str(candidate_root / f"{method}_candidates.jsonl")
    job["output_path"] = str(candidate_root / f"{method}_independent_scored_candidates.jsonl")

flow = json.load(open(flow_template))
flow.update({
    "checkpoint_path": flow_checkpoint_text,
    "device": device,
    "physical_gpu_index": gpu,
    "output_directory": str(flow_output),
    "gpu_selection_policy": "MOST_FREE_MEMORY_GPU_0_TO_5_WITH_TASK_MINIMUM_NO_UTILIZATION_GATE",
    "gpu_selection_minimum_free_mb": int(minimum_free_mb_text),
})

for path_text, payload in (
    (protocol_output, protocol),
    (jobs_output, jobs),
    (flow_output_config, flow),
):
    Path(path_text).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
PY

cd "${REPO_ROOT}"
printf '%s starting_matched_generation_suite_v2 gpu=%s\n' "$(date -Is)" "${selected_gpu}"
"${PYTHON}" -u scripts/route_a_v3/run_route2_matched_generation_suite_v1.py \
  --protocol "${protocol_runtime}" \
  --evaluator-jobs "${jobs_runtime}" \
  --evaluator-adjudication "${EVALUATOR_ADJUDICATION}" \
  --flow-config "${flow_runtime}" \
  --output-summary "${SUITE_SUMMARY}"
printf '%s matched_generation_suite_v2_finished\n' "$(date -Is)"
