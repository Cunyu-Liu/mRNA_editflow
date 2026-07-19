#!/usr/bin/env bash
# Downstream eval for region-specialized Stage B adapters (roadmap upgrade #2).
#
# Expects checkpoints from run_region_adapter_ablation.sh:
#   ckpts/region_adapter_t5_{utr5,cds,utr3,all}_${SLICE}/stage_b_region_t5_best.pt
#
# Benchmarks each checkpoint under the same T5 top-k/edit-budget protocol as the
# ranker baselines, then writes a paired comparison against a configurable
# baseline summary (hardneg_v2 by default).
set -euo pipefail

ROOT="${ROOT:-/home/cunyuliu/mrna_editflow_goal/mrna_editflow}"
PYTHON_BIN="${PYTHON_BIN:-/home/cunyuliu/miniconda3/envs/editflow/bin/python3.10}"
CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
SLICE="${SLICE:-head256}"
case "${SLICE}" in
  head[0-9]*) DEFAULT_LIMIT="${SLICE#head}" ;;
  *) DEFAULT_LIMIT="256" ;;
esac
LIMIT="${LIMIT:-${DEFAULT_LIMIT}}"
TOP_K="${TOP_K:-64}"
EDIT_BUDGET="${EDIT_BUDGET:-3}"
N_BOOTSTRAP="${N_BOOTSTRAP:-1000}"
N_PERMUTATIONS="${N_PERMUTATIONS:-2000}"
MAX_NOVELTY_SOURCES="${MAX_NOVELTY_SOURCES:-0}"

TASK_ID="${TASK_ID:-T5}"
TASK_ID_LC="$(printf '%s' "${TASK_ID}" | tr '[:upper:]' '[:lower:]')"
REGION_MODES="${REGION_MODES:-utr5 cds utr3 all}"
SOURCES="${SOURCES:-${ROOT}/benchmark/multiseed_t5_public_${SLICE}_hardneg_v2_top64/sources.jsonl}"
CKPT_ROOT="${CKPT_ROOT:-${ROOT}/ckpts}"
BENCH_ROOT="${BENCH_ROOT:-${ROOT}/benchmark}"
BASELINE_LABEL="${BASELINE_LABEL:-hardneg_v2_top${TOP_K}}"
BASELINE_SUMMARY="${BASELINE_SUMMARY:-${BENCH_ROOT}/multiseed_t5_public_${SLICE}_hardneg_v2_top${TOP_K}/multiseed_summary.json}"
COMPARE_PREFIX="${COMPARE_PREFIX:-region_adapter_vs_${BASELINE_LABEL}_${SLICE}}"
# Optional extra baselines to compare against after the primary baseline. Tokens
# without "=" are resolved as multiseed_t5_public_${SLICE}_${token}_top${TOP_K}.
# Example: "mo_grpo mo_scalar mo_pareto mo_te_only" (default).
EXTRA_BASELINES="${EXTRA_BASELINES:-mo_grpo mo_scalar mo_pareto mo_te_only}"
REFRESH_SOTA="${REFRESH_SOTA:-1}"
SOTA_JSON="${SOTA_JSON:-${ROOT}/docs/sota_gap_report.json}"
SOTA_MD="${SOTA_MD:-${ROOT}/docs/sota_gap_report.md}"
DECISION_JSON="${DECISION_JSON:-${BENCH_ROOT}/region_adapter_decision_report_${SLICE}.json}"
DECISION_MD="${DECISION_MD:-${BENCH_ROOT}/region_adapter_decision_report_${SLICE}.md}"
RESULT_AUDIT_JSON="${RESULT_AUDIT_JSON:-${BENCH_ROOT}/region_adapter_result_audit_${SLICE}.json}"
RESULT_AUDIT_MD="${RESULT_AUDIT_MD:-${BENCH_ROOT}/region_adapter_result_audit_${SLICE}.md}"

usage() {
  cat <<'EOF'
Usage:
  eval_region_adapter_ablation.sh [--dry-run]

Benchmarks region-specialized Stage B adapter checkpoints (utr5/cds/utr3/all)
and compares them against BASELINE_SUMMARY. Defaults to hardneg_v2 top64.

Environment overrides:
  ROOT, PYTHON_BIN, CUDA_VISIBLE_DEVICES, SEEDS, SLICE, LIMIT, TOP_K,
  EDIT_BUDGET, N_BOOTSTRAP, N_PERMUTATIONS, MAX_NOVELTY_SOURCES, TASK_ID,
  REGION_MODES, SOURCES, CKPT_ROOT, BENCH_ROOT, BASELINE_LABEL,
  BASELINE_SUMMARY, COMPARE_PREFIX, EXTRA_BASELINES, REFRESH_SOTA,
  SOTA_JSON, SOTA_MD, DECISION_JSON, DECISION_MD, RESULT_AUDIT_JSON,
  RESULT_AUDIT_MD
EOF
}

ckpt_for_mode() {
  echo "${CKPT_ROOT}/region_adapter_${TASK_ID_LC}_$1_${SLICE}/stage_b_region_${TASK_ID_LC}_best.pt"
}

bench_dir() {
  echo "${BENCH_ROOT}/multiseed_t5_public_${SLICE}_region_adapter_$1_top${TOP_K}"
}

extra_baseline_label() {
  spec="$1"
  if [[ "${spec}" == *"="* ]]; then
    echo "${spec%%=*}"
  else
    echo "${spec}_top${TOP_K}"
  fi
}

extra_baseline_summary() {
  spec="$1"
  if [[ "${spec}" == *"="* ]]; then
    echo "${spec#*=}"
  else
    echo "${BENCH_ROOT}/multiseed_t5_public_${SLICE}_${spec}_top${TOP_K}/multiseed_summary.json"
  fi
}

print_plan() {
  echo "ROOT=${ROOT}  CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER}  GPU=${CUDA_VISIBLE_DEVICES}  SEEDS=${SEEDS}  LIMIT=${LIMIT}  TOP_K=${TOP_K}  EDIT_BUDGET=${EDIT_BUDGET}  SLICE=${SLICE}"
  echo "TASK_ID=${TASK_ID}  REGION_MODES=${REGION_MODES}"
  echo "SOURCES=${SOURCES}"
  echo "BASELINE=${BASELINE_LABEL}=${BASELINE_SUMMARY}"
  echo "EXTRA_BASELINES=${EXTRA_BASELINES}"
  echo "REFRESH_SOTA=${REFRESH_SOTA}  SOTA_JSON=${SOTA_JSON}  SOTA_MD=${SOTA_MD}"
  echo "DECISION_JSON=${DECISION_JSON}  DECISION_MD=${DECISION_MD}"
  echo "RESULT_AUDIT_JSON=${RESULT_AUDIT_JSON}  RESULT_AUDIT_MD=${RESULT_AUDIT_MD}"
  for mode in ${REGION_MODES}; do
    echo "--- mode=${mode}"
    echo "    ckpt  -> $(ckpt_for_mode "${mode}")"
    echo "    bench -> $(bench_dir "${mode}")"
  done
  echo "compare -> ${BENCH_ROOT}/${COMPARE_PREFIX}.{json,md}"
  for spec in ${EXTRA_BASELINES}; do
    label="$(extra_baseline_label "${spec}")"
    summary="$(extra_baseline_summary "${spec}")"
    echo "extra compare -> ${BENCH_ROOT}/region_adapter_vs_${label}_${SLICE}.{json,md} baseline=${label}=${summary}"
  done
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then usage; exit 0; fi
if [[ "${1:-}" == "--dry-run" ]]; then print_plan; exit 0; fi

export CUDA_DEVICE_ORDER
export CUDA_VISIBLE_DEVICES
export PYTHONPATH="$(dirname "${ROOT}")${PYTHONPATH:+:${PYTHONPATH}}"

if [[ ! -f "${BASELINE_SUMMARY}" ]]; then
  echo "Missing baseline summary: ${BASELINE_SUMMARY}" >&2
  exit 2
fi

RUN_ARGS=()
for mode in ${REGION_MODES}; do
  ckpt="$(ckpt_for_mode "${mode}")"
  out_dir="$(bench_dir "${mode}")"
  if [[ ! -f "${ckpt}" ]]; then
    echo "Missing region-adapter checkpoint for mode=${mode}: ${ckpt}" >&2
    exit 2
  fi
  echo "[$(date -Iseconds)] BENCHMARK region_adapter mode=${mode}"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "${PYTHON_BIN}" -m mrna_editflow.eval.run_multiseed_benchmark \
    --run-mode development \
    --records-jsonl "${SOURCES}" \
    --checkpoint "${ckpt}" \
    --task-id "${TASK_ID}" \
    --edit-budget "${EDIT_BUDGET}" \
    --proposal-top-k "${TOP_K}" \
    --proposal-temperature 1.0 \
    --limit "${LIMIT}" \
    --device cuda \
    --seeds ${SEEDS} \
    --n-bootstrap "${N_BOOTSTRAP}" \
    --max-novelty-sources "${MAX_NOVELTY_SOURCES}" \
    --resume \
    --out-dir "${out_dir}"
  RUN_ARGS+=(--run "region_adapter_${mode}_top${TOP_K}=${out_dir}/multiseed_summary.json")
done

echo "[$(date -Iseconds)] COMPARE region adapters vs ${BASELINE_LABEL}"
"${PYTHON_BIN}" -m mrna_editflow.eval.compare_benchmarks \
  --baseline "${BASELINE_LABEL}=${BASELINE_SUMMARY}" \
  "${RUN_ARGS[@]}" \
  --metrics delta_oracle_te_vs_source mean_oracle_te legal_fraction mean_protein_identity within_budget_fraction reading_frame_intact_fraction mean_edit_distance \
  --out-json "${BENCH_ROOT}/${COMPARE_PREFIX}.json" \
  --out-md "${BENCH_ROOT}/${COMPARE_PREFIX}.md" \
  --n-bootstrap "${N_BOOTSTRAP}" \
  --n-permutations "${N_PERMUTATIONS}" \
  --require-default-matching-config

cat "${BENCH_ROOT}/${COMPARE_PREFIX}.md"

for spec in ${EXTRA_BASELINES}; do
  extra_label="$(extra_baseline_label "${spec}")"
  extra_summary="$(extra_baseline_summary "${spec}")"
  extra_prefix="region_adapter_vs_${extra_label}_${SLICE}"
  if [[ ! -f "${extra_summary}" ]]; then
    echo "[$(date -Iseconds)] SKIP extra compare: missing ${extra_label} summary ${extra_summary}"
    continue
  fi
  echo "[$(date -Iseconds)] EXTRA COMPARE region adapters vs ${extra_label}"
  "${PYTHON_BIN}" -m mrna_editflow.eval.compare_benchmarks \
    --baseline "${extra_label}=${extra_summary}" \
    "${RUN_ARGS[@]}" \
    --metrics delta_oracle_te_vs_source mean_oracle_te legal_fraction mean_protein_identity within_budget_fraction reading_frame_intact_fraction mean_edit_distance \
    --out-json "${BENCH_ROOT}/${extra_prefix}.json" \
    --out-md "${BENCH_ROOT}/${extra_prefix}.md" \
    --n-bootstrap "${N_BOOTSTRAP}" \
    --n-permutations "${N_PERMUTATIONS}" \
    --require-default-matching-config
  cat "${BENCH_ROOT}/${extra_prefix}.md"
done

echo "[$(date -Iseconds)] SUMMARIZE region-adapter decision report"
"${PYTHON_BIN}" -m mrna_editflow.eval.summarize_region_adapter_comparisons \
  --project-root "${ROOT}" \
  --slice "${SLICE}" \
  --top-k "${TOP_K}" \
  --out-json "${DECISION_JSON}" \
  --out-md "${DECISION_MD}"
cat "${DECISION_MD}"

echo "[$(date -Iseconds)] AUDIT region-adapter result artifacts"
"${PYTHON_BIN}" -m mrna_editflow.eval.audit_region_adapter_results \
  --project-root "${ROOT}" \
  --slice "${SLICE}" \
  --top-k "${TOP_K}" \
  --out-json "${RESULT_AUDIT_JSON}" \
  --out-md "${RESULT_AUDIT_MD}"
cat "${RESULT_AUDIT_MD}"

if [[ "${REFRESH_SOTA}" == "1" ]]; then
  echo "[$(date -Iseconds)] REFRESH SOTA gap report"
  "${PYTHON_BIN}" -m mrna_editflow.eval.sota_gap_report \
    --project-root "${ROOT}" \
    --out-json "${SOTA_JSON}" \
    --out-md "${SOTA_MD}"
fi
