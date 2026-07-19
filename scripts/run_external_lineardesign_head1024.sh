#!/usr/bin/env bash
# Run the official LinearDesign baseline on the full public head1024 input pack.
set -euo pipefail

ROOT="${ROOT:-/home/cunyuliu/mrna_editflow_goal/mrna_editflow}"
PYTHON_BIN="${PYTHON_BIN:-/home/cunyuliu/miniconda3/envs/editflow/bin/python3.10}"
INPUT_PACK_SUMMARY="${INPUT_PACK_SUMMARY:-${ROOT}/benchmark/external_sota/input_pack_t5_head1024/summary.json}"
EXECUTABLE="${LINEARDESIGN_BIN:-${ROOT}/scripts/external_lineardesign.sh}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/benchmark/external_sota/real_runs_t5_head1024}"
OUT_DIR="${OUT_DIR:-${OUT_ROOT}/LinearDesign}"
PROGRESS_JSONL="${PROGRESS_JSONL:-${OUT_ROOT}/LinearDesign.progress.jsonl}"
STATUS_JSON="${STATUS_JSON:-${OUT_ROOT}/LinearDesign.status.json}"
STATUS_MD="${STATUS_MD:-${OUT_ROOT}/LinearDesign.status.md}"
WORKERS="${WORKERS:-8}"
TIMEOUT_S="${TIMEOUT_S:-900}"
LIMIT="${LIMIT:-}"
LOG_PATH="${LOG_PATH:-${ROOT}/logs/external_lineardesign_head1024.log}"

usage() {
  cat <<'EOF'
Usage:
  run_external_lineardesign_head1024.sh [--dry-run]

Runs the official LinearDesign executable against the standardized
protein-conditioned head1024 input pack. On completion it refreshes:
  - docs/external_sota_real_run_audit.{json,md}
  - docs/t4_external_cds_baseline_comparison.{json,md}
  - docs/t5_external_utr_baseline_comparison.{json,md}
  - docs/paper_table3_external_baseline_readiness.{json,md}
  - docs/sota_readiness_audit_head256.{json,md}
  - docs/sota_gap_report.{json,md}

Environment overrides:
  ROOT, PYTHON_BIN, INPUT_PACK_SUMMARY, LINEARDESIGN_BIN, OUT_ROOT, OUT_DIR,
  PROGRESS_JSONL, STATUS_JSON, STATUS_MD, WORKERS, TIMEOUT_S, LIMIT, LOG_PATH
EOF
}

write_status() {
  local status="$1"
  local message="$2"
  STATUS="${status}" MESSAGE="${message}" STATUS_JSON="${STATUS_JSON}" STATUS_MD="${STATUS_MD}" \
    WORKERS="${WORKERS}" TIMEOUT_S="${TIMEOUT_S}" LIMIT="${LIMIT}" \
    "${PYTHON_BIN}" - <<'PY'
import json
import os
import time

payload = {
    "artifact_kind": "external_lineardesign_head1024_status",
    "time": time.time(),
    "status": os.environ["STATUS"],
    "message": os.environ["MESSAGE"],
    "workers": int(os.environ["WORKERS"]),
    "timeout_s": float(os.environ["TIMEOUT_S"]),
    "limit": int(os.environ["LIMIT"]) if os.environ["LIMIT"] else None,
}
for path in (os.environ["STATUS_JSON"], os.environ["STATUS_MD"]):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
with open(os.environ["STATUS_JSON"], "w", encoding="utf-8") as fh:
    json.dump(payload, fh, indent=2, sort_keys=True)
with open(os.environ["STATUS_MD"], "w", encoding="utf-8") as fh:
    fh.write("# External LinearDesign Head1024 Status\n\n")
    for key, value in payload.items():
        fh.write(f"- {key}: `{value}`\n")
PY
}

print_plan() {
  echo "EXTERNAL LINEARDESIGN HEAD1024"
  echo "ROOT=${ROOT}"
  echo "INPUT_PACK_SUMMARY=${INPUT_PACK_SUMMARY}"
  echo "EXECUTABLE=${EXECUTABLE}"
  echo "OUT_DIR=${OUT_DIR}"
  echo "WORKERS=${WORKERS} TIMEOUT_S=${TIMEOUT_S} LIMIT=${LIMIT:-all}"
  echo "PROGRESS_JSONL=${PROGRESS_JSONL}"
  echo "STATUS_JSON=${STATUS_JSON}"
  echo "LOG_PATH=${LOG_PATH}"
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then usage; exit 0; fi
if [[ "${1:-}" == "--dry-run" ]]; then print_plan; exit 0; fi

mkdir -p "${OUT_ROOT}" "$(dirname "${LOG_PATH}")"
export PYTHONPATH="$(dirname "${ROOT}")${PYTHONPATH:+:${PYTHONPATH}}"

on_error() {
  local code=$?
  write_status "failed" "LinearDesign adapter or report refresh failed with exit ${code}"
  exit "${code}"
}
trap on_error ERR

write_status "running" "official LinearDesign head1024 adapter is running"

limit_args=()
if [[ -n "${LIMIT}" ]]; then
  limit_args+=(--limit "${LIMIT}")
fi

"${PYTHON_BIN}" -m mrna_editflow.baselines.external_lineardesign_adapter \
  --input-pack-summary "${INPUT_PACK_SUMMARY}" \
  --executable "${EXECUTABLE}" \
  --out-dir "${OUT_DIR}" \
  --workers "${WORKERS}" \
  --timeout-s "${TIMEOUT_S}" \
  --progress-jsonl "${PROGRESS_JSONL}" \
  "${limit_args[@]}"

"${PYTHON_BIN}" -m mrna_editflow.eval.audit_external_sota_real_runs \
  --project-root "${ROOT}" \
  --out-json docs/external_sota_real_run_audit.json \
  --out-md docs/external_sota_real_run_audit.md

"${PYTHON_BIN}" -m mrna_editflow.eval.build_t4_external_cds_comparison \
  --project-root "${ROOT}" \
  --out-json docs/t4_external_cds_baseline_comparison.json \
  --out-md docs/t4_external_cds_baseline_comparison.md

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

write_status "complete" "official LinearDesign head1024 adapter and report refresh completed"
trap - ERR
