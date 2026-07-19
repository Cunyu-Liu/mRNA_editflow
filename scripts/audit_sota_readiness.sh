#!/usr/bin/env bash
# Read-only SOTA readiness audit for current pending evidence streams.
#
# This does not run training or evaluation. It joins the result-level audits for
# region-adapter comparisons and protein-conditioned CDS GC sweep artifacts.
set -euo pipefail

ROOT="${ROOT:-/home/cunyuliu/mrna_editflow_goal/mrna_editflow}"
PYTHON_BIN="${PYTHON_BIN:-/home/cunyuliu/miniconda3/envs/editflow/bin/python3.10}"
SLICE="${SLICE:-head256}"
TOP_K="${TOP_K:-64}"
OUT_JSON="${OUT_JSON:-${ROOT}/docs/sota_readiness_audit_${SLICE}.json}"
OUT_MD="${OUT_MD:-${ROOT}/docs/sota_readiness_audit_${SLICE}.md}"
MO_CLAIM_JSON="${MO_CLAIM_JSON:-${ROOT}/docs/multiobjective_scaleup_claim_audit_head256_head1024.json}"
MO_CLAIM_MD="${MO_CLAIM_MD:-${ROOT}/docs/multiobjective_scaleup_claim_audit_head256_head1024.md}"

usage() {
  cat <<'EOF'
Usage:
  audit_sota_readiness.sh [--dry-run]

Runs the read-only SOTA readiness audit. It reports whether the currently
expected region-adapter and protein-conditioned GC sweep artifacts are complete
enough to support a claim audit. It does not execute any benchmark.

Environment overrides:
  ROOT, PYTHON_BIN, SLICE, TOP_K, OUT_JSON, OUT_MD,
  MO_CLAIM_JSON, MO_CLAIM_MD
EOF
}

print_plan() {
  echo "SOTA READINESS AUDIT"
  echo "ROOT=${ROOT}"
  echo "SLICE=${SLICE}  TOP_K=${TOP_K}"
  echo "OUT_JSON=${OUT_JSON}"
  echo "OUT_MD=${OUT_MD}"
  echo "MO_CLAIM_JSON=${MO_CLAIM_JSON}"
  echo "MO_CLAIM_MD=${MO_CLAIM_MD}"
  echo "command -> ${PYTHON_BIN} -m mrna_editflow.eval.audit_multiobjective_scaleup_claims"
  echo "command -> ${PYTHON_BIN} -m mrna_editflow.eval.audit_sota_readiness"
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then usage; exit 0; fi
if [[ "${1:-}" == "--dry-run" ]]; then print_plan; exit 0; fi

export PYTHONPATH="$(dirname "${ROOT}")${PYTHONPATH:+:${PYTHONPATH}}"

"${PYTHON_BIN}" -m mrna_editflow.eval.audit_multiobjective_scaleup_claims \
  --project-root "${ROOT}" \
  --out-json "${MO_CLAIM_JSON}" \
  --out-md "${MO_CLAIM_MD}"

"${PYTHON_BIN}" -m mrna_editflow.eval.audit_sota_readiness \
  --project-root "${ROOT}" \
  --slice "${SLICE}" \
  --top-k "${TOP_K}" \
  --out-json "${OUT_JSON}" \
  --out-md "${OUT_MD}"
cat "${OUT_MD}"
