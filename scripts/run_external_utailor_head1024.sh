#!/usr/bin/env bash
# Run official UTailoR on the strict 25-100 nt head1024 protocol subset.
set -euo pipefail

ROOT="${ROOT:-/home/cunyuliu/mrna_editflow_goal/mrna_editflow}"
PYTHON_BIN="${PYTHON_BIN:-/home/cunyuliu/miniconda3/envs/editflow/bin/python3.10}"
INPUT_PACK_SUMMARY="${INPUT_PACK_SUMMARY:-${ROOT}/benchmark/external_sota/input_pack_t5_head1024/summary.json}"
SOURCE_RECORDS="${SOURCE_RECORDS:-${ROOT}/benchmark/multiseed_t5_public_head1024_sources.jsonl}"
EXECUTABLE="${UTAILOR_BIN:-${ROOT}/scripts/external_utailor.sh}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/benchmark/external_sota/real_runs_t5_head1024}"
OUT_DIR="${OUT_DIR:-${OUT_ROOT}/UTailoR}"
STATUS_JSON="${STATUS_JSON:-${OUT_ROOT}/UTailoR.status.json}"
GPU="${UTAILOR_GPU:-2}"
TIMEOUT_S="${TIMEOUT_S:-1800}"
LOG_PATH="${LOG_PATH:-${ROOT}/logs/external_utailor_head1024.log}"

usage() {
  cat <<'EOF'
Usage:
  run_external_utailor_head1024.sh [--dry-run]

Runs the official public UTailoR workflow and SavedModels only on source 5'UTRs
with lengths 25-100 nt, matching the web tool's declared input domain. Inputs
outside that range are persisted as ineligible, never silently truncated or
filled. The public RAR archives do not contain an explicit license file, so the
artifact is marked internal-research execution only and no redistribution
rights are assumed.

Environment overrides:
  ROOT, PYTHON_BIN, INPUT_PACK_SUMMARY, SOURCE_RECORDS, UTAILOR_BIN,
  OUT_ROOT, OUT_DIR, STATUS_JSON, UTAILOR_GPU, TIMEOUT_S, LOG_PATH
EOF
}

write_status() {
  local status="$1"
  local message="$2"
  STATUS="${status}" MESSAGE="${message}" STATUS_JSON="${STATUS_JSON}" \
    GPU="${GPU}" TIMEOUT_S="${TIMEOUT_S}" \
    "${PYTHON_BIN}" - <<'PY'
import json
import os
import time

payload = {
    "artifact_kind": "external_utailor_head1024_status",
    "time": time.time(),
    "status": os.environ["STATUS"],
    "message": os.environ["MESSAGE"],
    "gpu": os.environ["GPU"],
    "timeout_s": float(os.environ["TIMEOUT_S"]),
    "eligibility_policy": "official_input_length_25_100_strict",
}
path = os.environ["STATUS_JSON"]
os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
with open(path, "w", encoding="utf-8") as fh:
    json.dump(payload, fh, indent=2, sort_keys=True)
PY
}

print_plan() {
  echo "EXTERNAL UTAILOR HEAD1024"
  echo "ROOT=${ROOT}"
  echo "INPUT_PACK_SUMMARY=${INPUT_PACK_SUMMARY}"
  echo "SOURCE_RECORDS=${SOURCE_RECORDS}"
  echo "EXECUTABLE=${EXECUTABLE}"
  echo "OUT_DIR=${OUT_DIR}"
  echo "GPU=${GPU} TIMEOUT_S=${TIMEOUT_S}"
  echo "ELIGIBILITY_POLICY=official_input_length_25_100_strict"
  echo "STATUS_JSON=${STATUS_JSON}"
  echo "LOG_PATH=${LOG_PATH}"
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then usage; exit 0; fi
if [[ "${1:-}" == "--dry-run" ]]; then print_plan; exit 0; fi

mkdir -p "${OUT_DIR}" "$(dirname "${LOG_PATH}")"
export PYTHONPATH="$(dirname "${ROOT}")${PYTHONPATH:+:${PYTHONPATH}}"

on_error() {
  local code=$?
  write_status "failed" "UTailoR adapter or report refresh failed with exit ${code}"
  exit "${code}"
}
trap on_error ERR
write_status "running" "official UTailoR strict-domain head1024 subset is running"

UTAILOR_CUDA_VISIBLE_DEVICES="${GPU}" \
"${PYTHON_BIN}" -m mrna_editflow.baselines.external_utailor_adapter \
  --input-pack-summary "${INPUT_PACK_SUMMARY}" \
  --executable "${EXECUTABLE}" \
  --out-dir "${OUT_DIR}" \
  --timeout-s "${TIMEOUT_S}" \
  > "${LOG_PATH}"

LINEARDESIGN_BIN="${ROOT}/scripts/external_lineardesign.sh" \
ENSEMBLEDESIGN_BIN="${ROOT}/scripts/external_ensembledesign.sh" \
UTRGAN_BIN="${ROOT}/scripts/external_utrgan.sh" \
UTAILOR_BIN="${EXECUTABLE}" \
"${PYTHON_BIN}" -m mrna_editflow.baselines.external_sota_dry_run \
  --out-dir "${ROOT}/benchmark/external_sota/dry_run_t5_head1024" \
  --task-id T5 \
  --records-jsonl "${SOURCE_RECORDS}" \
  --limit 1024 \
  --split-name public_head1024 \
  --seed 0 \
  --hardware-label a100-server

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

write_status "complete" "official UTailoR strict-domain head1024 subset completed"
trap - ERR
