#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/cunyuliu/mrna_editflow_goal/mrna_editflow}"
PYTHON_BIN="${PYTHON_BIN:-/home/cunyuliu/miniconda3/envs/editflow/bin/python3.10}"
POLL_SECONDS="${POLL_SECONDS:-60}"
COMPARISON_JSON="${COMPARISON_JSON:-${ROOT}/benchmark/t5_ranker_full1k_head256_comparison.json}"
OUT_JSON="${OUT_JSON:-${ROOT}/docs/sota_gap_report.json}"
OUT_MD="${OUT_MD:-${ROOT}/docs/sota_gap_report.md}"
MARKER="${MARKER:-${ROOT}/logs/head256_sota_refresh.done.json}"

usage() {
  cat <<'EOF'
Usage:
  watch_head256_refresh_sota.sh [--dry-run]

Purpose:
  Wait for the strict head256 ranker comparison artifact, then regenerate the
  measured SOTA gap report. This script is intentionally read-only with respect
  to benchmark results; it only writes docs/sota_gap_report.{json,md} and a
  small completion marker.

Environment overrides:
  ROOT, PYTHON_BIN, POLL_SECONDS, COMPARISON_JSON, OUT_JSON, OUT_MD, MARKER
EOF
}

print_plan() {
  cat <<EOF
ROOT=${ROOT}
PYTHON_BIN=${PYTHON_BIN}
POLL_SECONDS=${POLL_SECONDS}
COMPARISON_JSON=${COMPARISON_JSON}
OUT_JSON=${OUT_JSON}
OUT_MD=${OUT_MD}
MARKER=${MARKER}

Commands:
  while [[ ! -s "${COMPARISON_JSON}" ]]; do sleep "${POLL_SECONDS}"; done
  PYTHONPATH="\$(dirname "${ROOT}")" "${PYTHON_BIN}" -m mrna_editflow.eval.sota_gap_report \\
    --project-root "${ROOT}" \\
    --out-json "${OUT_JSON}" \\
    --out-md "${OUT_MD}"
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
while [[ ! -s "${COMPARISON_JSON}" ]]; do
  sleep "${POLL_SECONDS}"
done

PYTHONPATH="$(dirname "${ROOT}")${PYTHONPATH:+:${PYTHONPATH}}" \
"${PYTHON_BIN}" -m mrna_editflow.eval.sota_gap_report \
  --project-root "${ROOT}" \
  --out-json "${OUT_JSON}" \
  --out-md "${OUT_MD}"

"${PYTHON_BIN}" - <<PY
import json
import time
payload = {
    "comparison_json": "${COMPARISON_JSON}",
    "out_json": "${OUT_JSON}",
    "out_md": "${OUT_MD}",
    "status": "refreshed",
    "time": time.time(),
}
with open("${MARKER}", "w", encoding="utf-8") as fh:
    json.dump(payload, fh, indent=2, sort_keys=True)
PY
