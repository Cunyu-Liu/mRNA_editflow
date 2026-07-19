#!/usr/bin/env bash
# Run the official EnsembleDesign budgeted head1024 adapter with resume.
set -euo pipefail

ROOT="${ROOT:-/home/cunyuliu/mrna_editflow_goal/mrna_editflow}"
PYTHON_BIN="${PYTHON_BIN:-/home/cunyuliu/miniconda3/envs/editflow/bin/python3.10}"
INPUT_PACK_SUMMARY="${INPUT_PACK_SUMMARY:-${ROOT}/benchmark/external_sota/input_pack_t5_head1024/summary.json}"
EXECUTABLE="${ENSEMBLEDESIGN_BIN:-${ROOT}/scripts/external_ensembledesign.sh}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/benchmark/external_sota/real_runs_t5_head1024}"
OUT_DIR="${OUT_DIR:-${OUT_ROOT}/EnsembleDesign}"
PROGRESS_JSONL="${PROGRESS_JSONL:-${OUT_DIR}/progress.jsonl}"
STATUS_JSON="${STATUS_JSON:-${OUT_ROOT}/EnsembleDesign.status.json}"
WORKERS="${WORKERS:-8}"
TIMEOUT_S="${TIMEOUT_S:-1800}"
BEAM_SIZE="${BEAM_SIZE:-100}"
RESCUE_BEAM_SIZE="${RESCUE_BEAM_SIZE:-200}"
NUM_ITERS="${NUM_ITERS:-3}"
NUM_RUNS="${NUM_RUNS:-1}"
LEARNING_RATE="${LEARNING_RATE:-0.03}"
EPSILON="${EPSILON:-0.5}"
LIMIT="${LIMIT:-}"
RESUME="${RESUME:-1}"
MAX_PASSES="${MAX_PASSES:-3}"
LOG_PATH="${LOG_PATH:-${ROOT}/logs/external_ensembledesign_head1024.log}"

usage() {
  cat <<'EOF'
Usage:
  run_external_ensembledesign_head1024.sh [--dry-run]

Defaults to the predeclared budgeted protocol:
  beam=100, iterations=3, runs=1, workers=8.

The official paper/default driver uses beam=200, iterations=30, runs=20.
Only those settings pass the protocol-fidelity gate. This runner's default
head1024 artifact is descriptive and keeps the SOTA reproduction claim closed.
Each completed row is persisted immediately and RESUME=1 skips completed rows.

Environment overrides:
  ROOT, PYTHON_BIN, INPUT_PACK_SUMMARY, ENSEMBLEDESIGN_BIN, OUT_ROOT,
  OUT_DIR, PROGRESS_JSONL, STATUS_JSON, WORKERS, TIMEOUT_S, BEAM_SIZE,
  RESCUE_BEAM_SIZE, NUM_ITERS, NUM_RUNS, LEARNING_RATE, EPSILON, LIMIT,
  RESUME, MAX_PASSES, LOG_PATH
EOF
}

write_status() {
  local status="$1"
  local message="$2"
  STATUS="${status}" MESSAGE="${message}" STATUS_JSON="${STATUS_JSON}" \
    WORKERS="${WORKERS}" TIMEOUT_S="${TIMEOUT_S}" BEAM_SIZE="${BEAM_SIZE}" \
    RESCUE_BEAM_SIZE="${RESCUE_BEAM_SIZE}" NUM_ITERS="${NUM_ITERS}" \
    NUM_RUNS="${NUM_RUNS}" LIMIT="${LIMIT}" \
    "${PYTHON_BIN}" - <<'PY'
import json
import os
import time

payload = {
    "artifact_kind": "external_ensembledesign_head1024_status",
    "time": time.time(),
    "status": os.environ["STATUS"],
    "message": os.environ["MESSAGE"],
    "workers": int(os.environ["WORKERS"]),
    "timeout_s": float(os.environ["TIMEOUT_S"]),
    "beam_size": int(os.environ["BEAM_SIZE"]),
    "rescue_beam_size": int(os.environ["RESCUE_BEAM_SIZE"]),
    "num_iters": int(os.environ["NUM_ITERS"]),
    "num_runs": int(os.environ["NUM_RUNS"]),
    "limit": int(os.environ["LIMIT"]) if os.environ["LIMIT"] else None,
}
path = os.environ["STATUS_JSON"]
os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
with open(path, "w", encoding="utf-8") as fh:
    json.dump(payload, fh, indent=2, sort_keys=True)
PY
}

print_plan() {
  echo "EXTERNAL ENSEMBLEDESIGN HEAD1024"
  echo "ROOT=${ROOT}"
  echo "INPUT_PACK_SUMMARY=${INPUT_PACK_SUMMARY}"
  echo "EXECUTABLE=${EXECUTABLE}"
  echo "OUT_DIR=${OUT_DIR}"
  echo "WORKERS=${WORKERS} TIMEOUT_S=${TIMEOUT_S} LIMIT=${LIMIT:-all}"
  echo "BEAM_SIZE=${BEAM_SIZE} RESCUE_BEAM_SIZE=${RESCUE_BEAM_SIZE}"
  echo "NUM_ITERS=${NUM_ITERS} NUM_RUNS=${NUM_RUNS}"
  echo "LEARNING_RATE=${LEARNING_RATE} EPSILON=${EPSILON}"
  echo "RESUME=${RESUME}"
  echo "MAX_PASSES=${MAX_PASSES}"
  echo "PROGRESS_JSONL=${PROGRESS_JSONL}"
  echo "STATUS_JSON=${STATUS_JSON}"
  echo "LOG_PATH=${LOG_PATH}"
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then usage; exit 0; fi
if [[ "${1:-}" == "--dry-run" ]]; then print_plan; exit 0; fi

mkdir -p "${OUT_DIR}" "$(dirname "${LOG_PATH}")"
export PYTHONPATH="$(dirname "${ROOT}")${PYTHONPATH:+:${PYTHONPATH}}"

on_error() {
  local code=$?
  write_status "failed" "EnsembleDesign adapter or report refresh failed with exit ${code}"
  exit "${code}"
}
trap on_error ERR
write_status "running" "official EnsembleDesign budgeted head1024 adapter is running"

limit_args=()
if [[ -n "${LIMIT}" ]]; then limit_args+=(--limit "${LIMIT}"); fi

pass=1
while (( pass <= MAX_PASSES )); do
  resume_args=()
  if [[ "${RESUME}" == "1" || "${pass}" -gt 1 ]]; then
    resume_args+=(--resume)
  fi
  echo "[$(date -Iseconds)] EnsembleDesign adapter pass ${pass}/${MAX_PASSES}" \
    >> "${LOG_PATH}"
  "${PYTHON_BIN}" -m mrna_editflow.baselines.external_ensembledesign_adapter \
    --input-pack-summary "${INPUT_PACK_SUMMARY}" \
    --executable "${EXECUTABLE}" \
    --out-dir "${OUT_DIR}" \
    --workers "${WORKERS}" \
    --timeout-s "${TIMEOUT_S}" \
    --beam-size "${BEAM_SIZE}" \
    --rescue-beam-size "${RESCUE_BEAM_SIZE}" \
    --num-iters "${NUM_ITERS}" \
    --num-runs "${NUM_RUNS}" \
    --learning-rate "${LEARNING_RATE}" \
    --epsilon "${EPSILON}" \
    --progress-jsonl "${PROGRESS_JSONL}" \
    "${limit_args[@]}" \
    "${resume_args[@]}" \
    >> "${LOG_PATH}"
  if OUT_DIR="${OUT_DIR}" "${PYTHON_BIN}" - <<'PY'
import json
import os

with open(os.path.join(os.environ["OUT_DIR"], "summary.json"), "r", encoding="utf-8") as fh:
    summary = json.load(fh)
raise SystemExit(
    0
    if summary.get("n_outputs") == summary.get("n_inputs")
    and summary.get("n_failures") == 0
    else 1
)
PY
  then
    break
  fi
  pass=$((pass + 1))
done
if (( pass > MAX_PASSES )); then
  echo "EnsembleDesign remained incomplete after ${MAX_PASSES} passes" >&2
  write_status "failed" \
    "EnsembleDesign remained incomplete after ${MAX_PASSES} passes"
  trap - ERR
  exit 3
fi

"${PYTHON_BIN}" -m mrna_editflow.eval.audit_external_sota_real_runs \
  --project-root "${ROOT}" \
  --out-json docs/external_sota_real_run_audit.json \
  --out-md docs/external_sota_real_run_audit.md

"${PYTHON_BIN}" -m mrna_editflow.eval.build_t4_external_cds_comparison \
  --project-root "${ROOT}" \
  --out-json docs/t4_external_cds_baseline_comparison.json \
  --out-md docs/t4_external_cds_baseline_comparison.md

"${PYTHON_BIN}" -m mrna_editflow.eval.build_paper_table3_external_baselines \
  --project-root "${ROOT}" \
  --out-json docs/paper_table3_external_baseline_readiness.json \
  --out-md docs/paper_table3_external_baseline_readiness.md

"${PYTHON_BIN}" -m mrna_editflow.eval.audit_sota_readiness \
  --project-root "${ROOT}" \
  --slice head256 \
  --top-k 64 \
  --out-json docs/sota_readiness_audit_head256.json \
  --out-md docs/sota_readiness_audit_head256.md

"${PYTHON_BIN}" -m mrna_editflow.eval.sota_gap_report \
  --project-root "${ROOT}" \
  --out-json docs/sota_gap_report.json \
  --out-md docs/sota_gap_report.md

write_status "complete" "official EnsembleDesign budgeted head1024 adapter completed"
trap - ERR
