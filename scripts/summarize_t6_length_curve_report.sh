#!/usr/bin/env bash
# Rebuild the T6 length-control curve report from completed multiseed summaries.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OUT_JSON="${OUT_JSON:-${ROOT}/benchmark/t6_length_curve_report_head256_head1024.json}"
OUT_MD="${OUT_MD:-${ROOT}/benchmark/t6_length_curve_report_head256_head1024.md}"
TOP_K="${TOP_K:-64}"
EDIT_BUDGET="${EDIT_BUDGET:-30}"
CHECKPOINT="${CHECKPOINT:-ckpts/stage_a_public_full_10k_bs8ga4_seed0/stage_a_best.pt}"
RUNNING_DELTAS="${RUNNING_DELTAS:-}"
RUNNING_LOGS="${RUNNING_LOGS:-}"

usage() {
  cat <<'EOF'
Usage:
  summarize_t6_length_curve_report.sh [--dry-run]

Rebuilds benchmark/t6_length_curve_report_head256_head1024.{json,md} from
completed multiseed_summary.json files. RUNNING_DELTAS may annotate missing
rows as running, e.g. RUNNING_DELTAS="head1024:-30 head1024:0".

Environment overrides:
  ROOT, PYTHON_BIN, OUT_JSON, OUT_MD, TOP_K, EDIT_BUDGET, CHECKPOINT,
  RUNNING_DELTAS, RUNNING_LOGS
EOF
}

build_command() {
  local cmd=(
    "${PYTHON_BIN}" -m mrna_editflow.eval.summarize_t6_length_curve
    --project-root "${ROOT}"
    --out-json "${OUT_JSON}"
    --out-md "${OUT_MD}"
    --top-k "${TOP_K}"
    --edit-budget "${EDIT_BUDGET}"
    --checkpoint "${CHECKPOINT}"
  )
  for item in ${RUNNING_DELTAS}; do
    cmd+=(--running-delta "${item}")
  done
  for item in ${RUNNING_LOGS}; do
    cmd+=(--running-log "${item}")
  done
  printf '%q ' "${cmd[@]}"
  printf '\n'
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then usage; exit 0; fi

export PYTHONPATH="$(dirname "${ROOT}")${PYTHONPATH:+:${PYTHONPATH}}"

if [[ "${1:-}" == "--dry-run" ]]; then
  echo "T6 LENGTH CURVE REPORT"
  echo "ROOT=${ROOT}"
  echo "OUT_JSON=${OUT_JSON}"
  echo "OUT_MD=${OUT_MD}"
  echo "TOP_K=${TOP_K}"
  echo "RUNNING_DELTAS=${RUNNING_DELTAS}"
  echo "RUNNING_LOGS=${RUNNING_LOGS}"
  echo "command -> $(build_command)"
  exit 0
fi

eval "$(build_command)"
