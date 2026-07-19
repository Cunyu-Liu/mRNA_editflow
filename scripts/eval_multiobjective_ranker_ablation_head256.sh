#!/usr/bin/env bash
# Downstream eval for the multi-objective reward-head ablation (roadmap upgrade #3.1).
#
# Benchmarks the 4 mode rankers (te_only control + scalar/pareto/grpo) on the head256
# T5 slice with the SAME eval config as hardneg_v2 (10 seeds, edit_budget=3, top64),
# then runs guarded paired permutation tests with te_only as the baseline so any TE
# gain is attributable ONLY to the multi-objective reward.
set -euo pipefail

ROOT="${ROOT:-/home/cunyuliu/mrna_editflow_goal/mrna_editflow}"
PYTHON_BIN="${PYTHON_BIN:-/home/cunyuliu/miniconda3/envs/editflow/bin/python3.10}"
CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
# SLICE selects the data-slice scale by SOURCE ROW COUNT (head256 / head1024).
# Must match the SLICE used by run_multiobjective_ranker_ablation_*.sh so the
# benchmark loads the matching per-mode rankers. Defaults to head256 for back-compat.
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

SOURCES="${SOURCES:-${ROOT}/benchmark/multiseed_t5_public_${SLICE}_hardneg_v2_top64/sources.jsonl}"
CKPT_ROOT="${CKPT_ROOT:-${ROOT}/ckpts}"
BENCH_ROOT="${BENCH_ROOT:-${ROOT}/benchmark}"
MODES="${MODES:-te_only scalar pareto grpo}"
HARDNEG_SUMMARY="${HARDNEG_SUMMARY:-${BENCH_ROOT}/multiseed_t5_public_${SLICE}_hardneg_v2_top${TOP_K}/multiseed_summary.json}"

usage() {
  cat <<'EOF'
Usage:
  eval_multiobjective_ranker_ablation_head256.sh [--dry-run]

Benchmarks each mode ranker (te_only/scalar/pareto/grpo) head256 x 10-seed top64,
then runs paired permutation comparisons vs the te_only single-TE control.
EOF
}

bench_dir() { echo "${BENCH_ROOT}/multiseed_t5_public_${SLICE}_mo_$1_top${TOP_K}"; }

print_plan() {
  echo "ROOT=${ROOT}  CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER}  GPU=${CUDA_VISIBLE_DEVICES}  SEEDS=${SEEDS}  LIMIT=${LIMIT}  TOP_K=${TOP_K}  EDIT_BUDGET=${EDIT_BUDGET}  SLICE=${SLICE}"
  echo "SOURCES=${SOURCES}"
  for mode in ${MODES}; do
    echo "--- mode=${mode}"
    echo "    ckpt  -> ${CKPT_ROOT}/proposal_ranker_t5_mo_${mode}_${SLICE}/proposal_ranker_best.pt"
    echo "    bench -> $(bench_dir "${mode}")"
  done
  echo "compare baseline=te_only vs scalar/pareto/grpo -> ${BENCH_ROOT}/compare_mo_fusion_vs_te_only_${SLICE}.{json,md}"
  echo "compare baseline=hardneg_v2 vs scalar/pareto/grpo -> ${BENCH_ROOT}/compare_mo_fusion_vs_hardneg_v2_${SLICE}.{json,md}"
  echo "HARDNEG_SUMMARY=${HARDNEG_SUMMARY}"
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then usage; exit 0; fi
if [[ "${1:-}" == "--dry-run" ]]; then print_plan; exit 0; fi

export CUDA_DEVICE_ORDER
export CUDA_VISIBLE_DEVICES
export PYTHONPATH="$(dirname "${ROOT}")${PYTHONPATH:+:${PYTHONPATH}}"

for mode in ${MODES}; do
  ckpt="${CKPT_ROOT}/proposal_ranker_t5_mo_${mode}_${SLICE}/proposal_ranker_best.pt"
  out_dir="$(bench_dir "${mode}")"
  echo "[$(date -Iseconds)] BENCHMARK mode=${mode}"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "${PYTHON_BIN}" -m mrna_editflow.eval.run_multiseed_benchmark \
    --run-mode development \
    --records-jsonl "${SOURCES}" \
    --checkpoint "${ckpt}" \
    --task-id T5 \
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
done

RUN_ARGS=()
for mode in scalar pareto grpo; do
  RUN_ARGS+=(--run "mo_${mode}_top${TOP_K}=$(bench_dir "${mode}")/multiseed_summary.json")
done

echo "[$(date -Iseconds)] COMPARE vs te_only control"
"${PYTHON_BIN}" -m mrna_editflow.eval.compare_benchmarks \
  --baseline "mo_te_only_top${TOP_K}=$(bench_dir te_only)/multiseed_summary.json" \
  "${RUN_ARGS[@]}" \
  --metrics delta_oracle_te_vs_source mean_oracle_te mean_protein_identity within_budget_fraction reading_frame_intact_fraction \
  --out-json "${BENCH_ROOT}/compare_mo_fusion_vs_te_only_${SLICE}.json" \
  --out-md "${BENCH_ROOT}/compare_mo_fusion_vs_te_only_${SLICE}.md" \
  --n-bootstrap "${N_BOOTSTRAP}" \
  --n-permutations "${N_PERMUTATIONS}" \
  --require-default-matching-config

echo "[$(date -Iseconds)] ABLATION EVAL COMPLETE"
cat "${BENCH_ROOT}/compare_mo_fusion_vs_te_only_${SLICE}.md"

if [[ -f "${HARDNEG_SUMMARY}" ]]; then
  echo "[$(date -Iseconds)] COMPARE vs hardneg_v2 prior champion"
  "${PYTHON_BIN}" -m mrna_editflow.eval.compare_benchmarks \
    --baseline "hardneg_v2_top${TOP_K}=${HARDNEG_SUMMARY}" \
    "${RUN_ARGS[@]}" \
    --metrics delta_oracle_te_vs_source mean_oracle_te mean_protein_identity within_budget_fraction reading_frame_intact_fraction \
    --out-json "${BENCH_ROOT}/compare_mo_fusion_vs_hardneg_v2_${SLICE}.json" \
    --out-md "${BENCH_ROOT}/compare_mo_fusion_vs_hardneg_v2_${SLICE}.md" \
    --n-bootstrap "${N_BOOTSTRAP}" \
    --n-permutations "${N_PERMUTATIONS}" \
    --require-default-matching-config
  cat "${BENCH_ROOT}/compare_mo_fusion_vs_hardneg_v2_${SLICE}.md"
else
  echo "[$(date -Iseconds)] SKIP hardneg_v2 compare; missing ${HARDNEG_SUMMARY}"
fi
