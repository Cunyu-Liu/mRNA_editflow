#!/usr/bin/env bash
# Wait for the active LinearDesign runner, then refresh the T4 comparison.
set -euo pipefail

ROOT="${ROOT:-/home/cunyuliu/mrna_editflow_goal/mrna_editflow}"
PYTHON_BIN="${PYTHON_BIN:-/home/cunyuliu/miniconda3/envs/editflow/bin/python3.10}"
RUN_PID="${RUN_PID:-3923723}"
POLL_SECONDS="${POLL_SECONDS:-60}"
LOG_PATH="${LOG_PATH:-${ROOT}/logs/t4_external_cds_comparison.refresh.log}"

if [[ "${1:-}" == "--dry-run" ]]; then
  echo "WATCH EXTERNAL LINEARDESIGN T4 COMPARISON"
  echo "RUN_PID=${RUN_PID} POLL_SECONDS=${POLL_SECONDS}"
  echo "LOG_PATH=${LOG_PATH}"
  exit 0
fi

while kill -0 "${RUN_PID}" 2>/dev/null; do
  sleep "${POLL_SECONDS}"
done

cd "${ROOT}"
export PYTHONPATH="$(dirname "${ROOT}")${PYTHONPATH:+:${PYTHONPATH}}"
"${PYTHON_BIN}" -m mrna_editflow.eval.build_t4_external_cds_comparison \
  --project-root "${ROOT}" \
  --out-json docs/t4_external_cds_baseline_comparison.json \
  --out-md docs/t4_external_cds_baseline_comparison.md \
  > "${LOG_PATH}" 2>&1
