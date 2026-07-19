#!/usr/bin/env bash
# Wait for GENCODE family split report, then refresh data readiness docs.
set -euo pipefail

ROOT="${ROOT:-/home/cunyuliu/mrna_editflow_goal/mrna_editflow}"
PYTHON_BIN="${PYTHON_BIN:-/home/cunyuliu/miniconda3/envs/editflow/bin/python3.10}"
REPORT_JSON="${REPORT_JSON:-${ROOT}/benchmark/gencode_family_leakage_protocol/report.json}"
POLL_SECONDS="${POLL_SECONDS:-120}"
OUT_MANIFEST_JSON="${OUT_MANIFEST_JSON:-${ROOT}/docs/dataset_manifest_audit.json}"
OUT_MANIFEST_MD="${OUT_MANIFEST_MD:-${ROOT}/docs/dataset_manifest_audit.md}"
OUT_READINESS_JSON="${OUT_READINESS_JSON:-${ROOT}/docs/data_scaleup_readiness.json}"
OUT_READINESS_MD="${OUT_READINESS_MD:-${ROOT}/docs/data_scaleup_readiness.md}"
MARKER="${MARKER:-${ROOT}/logs/gencode_family_readiness.done.json}"

usage() {
  cat <<'EOF'
Usage:
  watch_gencode_family_readiness.sh [--dry-run]

Purpose:
  Wait for benchmark/gencode_family_leakage_protocol/report.json, then refresh
  docs/dataset_manifest_audit.{json,md} and docs/data_scaleup_readiness.{json,md}.
  This is a lightweight watcher; it does not run the split/leakage audit itself.

Environment overrides:
  ROOT, PYTHON_BIN, REPORT_JSON, POLL_SECONDS, OUT_MANIFEST_JSON, OUT_MANIFEST_MD,
  OUT_READINESS_JSON, OUT_READINESS_MD, MARKER
EOF
}

print_plan() {
  cat <<EOF
GENCODE_FAMILY_READINESS_WATCHER
artifact_kind=gencode_family_readiness_watcher
ROOT=${ROOT}
PYTHON_BIN=${PYTHON_BIN}
REPORT_JSON=${REPORT_JSON}
POLL_SECONDS=${POLL_SECONDS}
OUT_MANIFEST_JSON=${OUT_MANIFEST_JSON}
OUT_MANIFEST_MD=${OUT_MANIFEST_MD}
OUT_READINESS_JSON=${OUT_READINESS_JSON}
OUT_READINESS_MD=${OUT_READINESS_MD}
MARKER=${MARKER}
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

mkdir -p "$(dirname "${MARKER}")"
while [[ ! -s "${REPORT_JSON}" ]]; do
  sleep "${POLL_SECONDS}"
done

PYTHONPATH="$(dirname "${ROOT}")${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PYTHON_BIN}" -m mrna_editflow.eval.dataset_manifest_audit \
    --project-root "${ROOT}" \
    --out-json "${OUT_MANIFEST_JSON}" \
    --out-md "${OUT_MANIFEST_MD}"

PYTHONPATH="$(dirname "${ROOT}")${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PYTHON_BIN}" -m mrna_editflow.eval.build_data_scaleup_readiness \
    --project-root "${ROOT}" \
    --out-json "${OUT_READINESS_JSON}" \
    --out-md "${OUT_READINESS_MD}"

"${PYTHON_BIN}" - <<PY
import json
import time

payload = {
    "artifact_kind": "gencode_family_readiness_watcher",
    "report_json": "${REPORT_JSON}",
    "manifest_json": "${OUT_MANIFEST_JSON}",
    "readiness_json": "${OUT_READINESS_JSON}",
    "status": "refreshed",
    "time": time.time(),
}
with open("${MARKER}", "w", encoding="utf-8") as fh:
    json.dump(payload, fh, indent=2, sort_keys=True)
PY
