#!/usr/bin/env bash
# Retry missing EnsembleDesign rows after the currently running first pass.
set -euo pipefail

ROOT="${ROOT:-/home/cunyuliu/mrna_editflow_goal/mrna_editflow}"
PYTHON_BIN="${PYTHON_BIN:-/home/cunyuliu/miniconda3/envs/editflow/bin/python3.10}"
WAIT_PID="${WAIT_PID:-2529673}"
POLL_S="${POLL_S:-60}"
WORKERS="${WORKERS:-8}"
TIMEOUT_S="${TIMEOUT_S:-1800}"
BEAM_SIZE="${BEAM_SIZE:-100}"
NUM_ITERS="${NUM_ITERS:-3}"
NUM_RUNS="${NUM_RUNS:-1}"
MAX_PASSES="${MAX_PASSES:-3}"

if [[ "${1:-}" == "--dry-run" ]]; then
  echo "WATCH EXTERNAL ENSEMBLEDESIGN RETRY"
  echo "WAIT_PID=${WAIT_PID} POLL_S=${POLL_S}"
  echo "WORKERS=${WORKERS} TIMEOUT_S=${TIMEOUT_S}"
  echo "BEAM_SIZE=${BEAM_SIZE} NUM_ITERS=${NUM_ITERS} NUM_RUNS=${NUM_RUNS}"
  echo "MAX_PASSES=${MAX_PASSES}"
  exit 0
fi

while kill -0 "${WAIT_PID}" 2>/dev/null; do
  sleep "${POLL_S}"
done

ROOT="${ROOT}" \
PYTHON_BIN="${PYTHON_BIN}" \
WORKERS="${WORKERS}" \
TIMEOUT_S="${TIMEOUT_S}" \
BEAM_SIZE="${BEAM_SIZE}" \
NUM_ITERS="${NUM_ITERS}" \
NUM_RUNS="${NUM_RUNS}" \
RESUME=1 \
MAX_PASSES="${MAX_PASSES}" \
bash "${ROOT}/scripts/run_external_ensembledesign_head1024.sh"
