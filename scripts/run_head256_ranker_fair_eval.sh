#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/cunyuliu/mrna_editflow_goal/mrna_editflow}"
PYTHON_BIN="${PYTHON_BIN:-/home/cunyuliu/miniconda3/envs/editflow/bin/python3.10}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"
LIMIT="${LIMIT:-256}"
TOP_K="${TOP_K:-32}"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
N_BOOTSTRAP="${N_BOOTSTRAP:-1000}"
N_PERMUTATIONS="${N_PERMUTATIONS:-2000}"
MAX_NOVELTY_SOURCES="${MAX_NOVELTY_SOURCES:-0}"

RECORDS_JSONL="${RECORDS_JSONL:-${ROOT}/data/processed/gencode_human_transcripts.records.jsonl}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-${ROOT}/ckpts/stage_a_public_full_1k_bs8ga4/stage_a_best_for_proposal_ranker.pt}"
RANKER_CHECKPOINT="${RANKER_CHECKPOINT:-${ROOT}/ckpts/proposal_ranker_t5_full1k/proposal_ranker_best.pt}"
BASE_OUT="${BASE_OUT:-${ROOT}/benchmark/multiseed_t5_public_head256_base_full1k_top32}"
RANKER_OUT="${RANKER_OUT:-${ROOT}/benchmark/multiseed_t5_public_head256_ranker_full1k_top32}"
COMPARISON_JSON="${COMPARISON_JSON:-${ROOT}/benchmark/t5_ranker_full1k_head256_comparison.json}"
COMPARISON_MD="${COMPARISON_MD:-${ROOT}/benchmark/t5_ranker_full1k_head256_comparison.md}"
SOTA_JSON="${SOTA_JSON:-${ROOT}/docs/sota_gap_report.json}"
SOTA_MD="${SOTA_MD:-${ROOT}/docs/sota_gap_report.md}"

usage() {
  cat <<'EOF'
Usage:
  run_head256_ranker_fair_eval.sh [--dry-run]

Purpose:
  Run the fair head256 baseline-vs-TE-ranker benchmark with seed-level resume,
  then generate a guarded paired comparison and refresh the SOTA gap report.

Environment overrides:
  ROOT, PYTHON_BIN, CUDA_VISIBLE_DEVICES, LIMIT, TOP_K, SEEDS, N_BOOTSTRAP,
  N_PERMUTATIONS, MAX_NOVELTY_SOURCES, RECORDS_JSONL, BASE_CHECKPOINT,
  RANKER_CHECKPOINT, BASE_OUT, RANKER_OUT, COMPARISON_JSON, COMPARISON_MD,
  SOTA_JSON, SOTA_MD
EOF
}

print_plan() {
  cat <<EOF
ROOT=${ROOT}
PYTHON_BIN=${PYTHON_BIN}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
LIMIT=${LIMIT}
TOP_K=${TOP_K}
SEEDS=${SEEDS}
N_BOOTSTRAP=${N_BOOTSTRAP}
MAX_NOVELTY_SOURCES=${MAX_NOVELTY_SOURCES}
BASE_OUT=${BASE_OUT}
RANKER_OUT=${RANKER_OUT}
COMPARISON_JSON=${COMPARISON_JSON}
COMPARISON_MD=${COMPARISON_MD}

Commands:
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" PYTHONPATH="\$(dirname "${ROOT}")" "${PYTHON_BIN}" -m mrna_editflow.eval.run_multiseed_benchmark ... base --resume
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" PYTHONPATH="\$(dirname "${ROOT}")" "${PYTHON_BIN}" -m mrna_editflow.eval.run_multiseed_benchmark ... ranker --resume
  PYTHONPATH="\$(dirname "${ROOT}")" "${PYTHON_BIN}" -m mrna_editflow.eval.compare_benchmarks ... --require-matching-config ... max_novelty_sources
  PYTHONPATH="\$(dirname "${ROOT}")" "${PYTHON_BIN}" -m mrna_editflow.eval.sota_gap_report ...
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

if [[ "${1:-}" == "--dry-run" ]]; then
  print_plan
  exit 0
fi

export CUDA_VISIBLE_DEVICES
export PYTHONPATH="$(dirname "${ROOT}")${PYTHONPATH:+:${PYTHONPATH}}"

"${PYTHON_BIN}" -m mrna_editflow.eval.run_multiseed_benchmark \
  --run-mode development \
  --records-jsonl "${RECORDS_JSONL}" \
  --checkpoint "${BASE_CHECKPOINT}" \
  --task-id T5 \
  --edit-budget 3 \
  --proposal-top-k "${TOP_K}" \
  --proposal-temperature 1.0 \
  --limit "${LIMIT}" \
  --device cuda \
  --seeds ${SEEDS} \
  --n-bootstrap "${N_BOOTSTRAP}" \
  --max-novelty-sources "${MAX_NOVELTY_SOURCES}" \
  --resume \
  --out-dir "${BASE_OUT}"

"${PYTHON_BIN}" -m mrna_editflow.eval.run_multiseed_benchmark \
  --run-mode development \
  --records-jsonl "${RECORDS_JSONL}" \
  --checkpoint "${RANKER_CHECKPOINT}" \
  --task-id T5 \
  --edit-budget 3 \
  --proposal-top-k "${TOP_K}" \
  --proposal-temperature 1.0 \
  --limit "${LIMIT}" \
  --device cuda \
  --seeds ${SEEDS} \
  --n-bootstrap "${N_BOOTSTRAP}" \
  --max-novelty-sources "${MAX_NOVELTY_SOURCES}" \
  --resume \
  --out-dir "${RANKER_OUT}"

"${PYTHON_BIN}" -m mrna_editflow.eval.compare_benchmarks \
  --baseline "base_full1k_top32=${BASE_OUT}/multiseed_summary.json" \
  --run "ranker_full1k_top32=${RANKER_OUT}/multiseed_summary.json" \
  --out-json "${COMPARISON_JSON}" \
  --out-md "${COMPARISON_MD}" \
  --metrics mean_oracle_te delta_oracle_te_vs_source mean_protein_identity within_budget_fraction mean_edit_distance \
  --n-bootstrap "${N_BOOTSTRAP}" \
  --n-permutations "${N_PERMUTATIONS}" \
  --require-matching-config task_id edit_budget proposal_top_k proposal_temperature n_records seeds target_length_delta max_novelty_sources

"${PYTHON_BIN}" -m mrna_editflow.eval.sota_gap_report \
  --project-root "${ROOT}" \
  --out-json "${SOTA_JSON}" \
  --out-md "${SOTA_MD}"
