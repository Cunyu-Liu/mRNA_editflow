#!/usr/bin/env bash
# Run official UTRGAN and the constrained UTR-only comparator on head1024.
set -euo pipefail

ROOT="${ROOT:-/home/cunyuliu/mrna_editflow_goal/mrna_editflow}"
PYTHON_BIN="${PYTHON_BIN:-/home/cunyuliu/miniconda3/envs/editflow/bin/python3.10}"
INPUT_PACK_SUMMARY="${INPUT_PACK_SUMMARY:-${ROOT}/benchmark/external_sota/input_pack_t5_head1024/summary.json}"
SOURCE_RECORDS="${SOURCE_RECORDS:-${ROOT}/benchmark/multiseed_t5_public_head1024_sources.jsonl}"
EXECUTABLE="${UTRGAN_BIN:-${ROOT}/scripts/external_utrgan.sh}"
TOOL_ROOT="${UTRGAN_ROOT:-${ROOT}/external_tools/UTRGAN}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/benchmark/external_sota/real_runs_t5_head1024}"
UTRGAN_OUT_DIR="${UTRGAN_OUT_DIR:-${OUT_ROOT}/UTRGAN}"
LOCAL_JSON="${LOCAL_JSON:-${ROOT}/benchmark/utr_local_search_head1024.json}"
LOCAL_JSONL="${LOCAL_JSONL:-${ROOT}/benchmark/utr_local_search_head1024.records.jsonl}"
UTRGAN_STEPS="${UTRGAN_STEPS:-10}"
UTRGAN_LR_EXPONENT="${UTRGAN_LR_EXPONENT:-5}"
UTRGAN_TIMEOUT_S="${UTRGAN_TIMEOUT_S:-1800}"
UTRGAN_GPU="${UTRGAN_GPU:-2}"
LOCAL_WORKERS="${LOCAL_WORKERS:-16}"
RUN_UTRGAN="${RUN_UTRGAN:-1}"
RUN_LOCAL_SEARCH="${RUN_LOCAL_SEARCH:-1}"
LOG_PATH="${LOG_PATH:-${ROOT}/logs/external_utrgan_head1024.log}"
STATUS_JSON="${STATUS_JSON:-${OUT_ROOT}/UTRGAN.status.json}"

usage() {
  cat <<'EOF'
Usage:
  run_external_utrgan_head1024.sh [--dry-run]

Runs:
  1. official UTRGAN on the standardized head1024 5'UTR input pack;
  2. the shared-oracle constrained 5'UTR local-search ceiling;
  3. external real-run audit, T5 comparison, Table 3, readiness, and gap reports.

Defaults intentionally use the budgeted 10-step UTRGAN protocol. This is not
paper-faithful UTRGAN reproduction; the generated reports keep the SOTA claim
gate closed. Set UTRGAN_STEPS=10000 only for a predeclared paper-faithful run.

Environment overrides:
  ROOT, PYTHON_BIN, INPUT_PACK_SUMMARY, SOURCE_RECORDS, UTRGAN_BIN,
  UTRGAN_ROOT, OUT_ROOT, UTRGAN_OUT_DIR, LOCAL_JSON, LOCAL_JSONL,
  UTRGAN_STEPS, UTRGAN_LR_EXPONENT, UTRGAN_TIMEOUT_S, UTRGAN_GPU,
  LOCAL_WORKERS, RUN_UTRGAN, RUN_LOCAL_SEARCH, LOG_PATH, STATUS_JSON
EOF
}

write_status() {
  local status="$1"
  local message="$2"
  STATUS="${status}" MESSAGE="${message}" STATUS_JSON="${STATUS_JSON}" \
    UTRGAN_STEPS="${UTRGAN_STEPS}" UTRGAN_GPU="${UTRGAN_GPU}" \
    LOCAL_WORKERS="${LOCAL_WORKERS}" \
    "${PYTHON_BIN}" - <<'PY'
import json
import os
import time

payload = {
    "artifact_kind": "external_utrgan_head1024_status",
    "time": time.time(),
    "status": os.environ["STATUS"],
    "message": os.environ["MESSAGE"],
    "utrgan_steps": int(os.environ["UTRGAN_STEPS"]),
    "utrgan_gpu": os.environ["UTRGAN_GPU"],
    "local_workers": int(os.environ["LOCAL_WORKERS"]),
}
path = os.environ["STATUS_JSON"]
os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
with open(path, "w", encoding="utf-8") as fh:
    json.dump(payload, fh, indent=2, sort_keys=True)
PY
}

print_plan() {
  echo "EXTERNAL UTRGAN HEAD1024"
  echo "ROOT=${ROOT}"
  echo "INPUT_PACK_SUMMARY=${INPUT_PACK_SUMMARY}"
  echo "SOURCE_RECORDS=${SOURCE_RECORDS}"
  echo "EXECUTABLE=${EXECUTABLE}"
  echo "TOOL_ROOT=${TOOL_ROOT}"
  echo "UTRGAN_OUT_DIR=${UTRGAN_OUT_DIR}"
  echo "UTRGAN_STEPS=${UTRGAN_STEPS} UTRGAN_GPU=${UTRGAN_GPU}"
  echo "LOCAL_JSON=${LOCAL_JSON}"
  echo "LOCAL_JSONL=${LOCAL_JSONL}"
  echo "LOCAL_WORKERS=${LOCAL_WORKERS}"
  echo "RUN_UTRGAN=${RUN_UTRGAN} RUN_LOCAL_SEARCH=${RUN_LOCAL_SEARCH}"
  echo "LOG_PATH=${LOG_PATH}"
  echo "STATUS_JSON=${STATUS_JSON}"
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then usage; exit 0; fi
if [[ "${1:-}" == "--dry-run" ]]; then print_plan; exit 0; fi

mkdir -p "${OUT_ROOT}" "$(dirname "${LOG_PATH}")"
export PYTHONPATH="$(dirname "${ROOT}")${PYTHONPATH:+:${PYTHONPATH}}"

on_error() {
  local code=$?
  write_status "failed" "UTRGAN adapter, local comparator, or report refresh failed with exit ${code}"
  exit "${code}"
}

on_local_error() {
  local code=$?
  rm -f "${tmp_json:-}" "${tmp_jsonl:-}"
  write_status "failed" "UTR local comparator failed with exit ${code}"
  exit "${code}"
}

trap on_error ERR
write_status "running" "official UTRGAN head1024 protocol is running"

if [[ "${RUN_UTRGAN}" == "1" ]]; then
  UTRGAN_CUDA_VISIBLE_DEVICES="${UTRGAN_GPU}" \
  "${PYTHON_BIN}" -m mrna_editflow.baselines.external_utrgan_adapter \
    --input-pack-summary "${INPUT_PACK_SUMMARY}" \
    --executable "${EXECUTABLE}" \
    --tool-root "${TOOL_ROOT}" \
    --out-dir "${UTRGAN_OUT_DIR}" \
    --steps "${UTRGAN_STEPS}" \
    --learning-rate-exponent "${UTRGAN_LR_EXPONENT}" \
    --timeout-s "${UTRGAN_TIMEOUT_S}" \
    > "${LOG_PATH}"
fi

if [[ "${RUN_LOCAL_SEARCH}" == "1" ]]; then
  tmp_json="${LOCAL_JSON}.tmp.$$"
  tmp_jsonl="${LOCAL_JSONL}.tmp.$$"
  trap on_local_error ERR
  "${PYTHON_BIN}" -m mrna_editflow.baselines.utr_local_search \
    --records-jsonl "${SOURCE_RECORDS}" \
    --out-jsonl "${tmp_jsonl}" \
    --out-json "${tmp_json}" \
    --limit 1024 \
    --workers "${LOCAL_WORKERS}" \
    --edit-budget 3 \
    --beam-width 16 \
    --max-length-delta 6 \
    --start-window-nt 90 \
    --max-edit-positions 90
  mv "${tmp_json}" "${LOCAL_JSON}"
  mv "${tmp_jsonl}" "${LOCAL_JSONL}"
  LOCAL_JSON="${LOCAL_JSON}" LOCAL_JSONL="${LOCAL_JSONL}" \
    "${PYTHON_BIN}" - <<'PY'
import json
import os

path = os.environ["LOCAL_JSON"]
with open(path, "r", encoding="utf-8") as fh:
    payload = json.load(fh)
payload["out_jsonl"] = os.environ["LOCAL_JSONL"]
with open(path, "w", encoding="utf-8") as fh:
    json.dump(payload, fh, indent=2, sort_keys=True)
PY
  trap on_error ERR
fi

"${PYTHON_BIN}" -m mrna_editflow.eval.audit_external_sota_real_runs \
  --project-root "${ROOT}" \
  --out-json docs/external_sota_real_run_audit.json \
  --out-md docs/external_sota_real_run_audit.md

"${PYTHON_BIN}" -m mrna_editflow.eval.build_t5_external_utr_comparison \
  --project-root "${ROOT}" \
  --out-json docs/t5_external_utr_baseline_comparison.json \
  --out-md docs/t5_external_utr_baseline_comparison.md

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

write_status "complete" "official UTRGAN head1024 and T5 report refresh completed"
trap - ERR
