#!/usr/bin/env bash
# Frozen-foundation protocol readiness check (CPU, leakage-gated).
#
# This is a small protocol check, not a real external foundation-model metric.
# It verifies that the matched-budget frozen-backbone harness can run with a
# leakage-free split and that placeholder external arms are clearly marked as
# non-quotable quality signals.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SLICE="${SLICE:-head256}"
OUT_DIR="${OUT_DIR:-${ROOT}/benchmark/frozen_backbone_protocol_${SLICE}}"
QUERY_JSONL="${QUERY_JSONL:-${OUT_DIR}/query.jsonl}"
REFERENCE_JSONL="${REFERENCE_JSONL:-${OUT_DIR}/reference.jsonl}"
OUT_JSON="${OUT_JSON:-${OUT_DIR}/summary.json}"
OUT_MD="${OUT_MD:-${OUT_DIR}/table.md}"
LEAKAGE_JSON="${LEAKAGE_JSON:-${OUT_DIR}/leakage.json}"
BACKBONES="${BACKBONES:-none helix_mrna mrnabert}"
STEPS="${STEPS:-1}"
SYNTHETIC_N="${SYNTHETIC_N:-4}"
SEED="${SEED:-0}"
QUERY_SEED="${QUERY_SEED:-101}"
REFERENCE_SEED="${REFERENCE_SEED:-202}"

usage() {
  cat <<'EOF'
Usage:
  run_frozen_backbone_protocol_check.sh [--dry-run]

Runs a tiny CPU frozen-backbone protocol check with a leakage-free synthetic
query/reference split. The output proves protocol readiness only; placeholder
external arms are not valid SOTA metrics.

Environment overrides:
  ROOT, PYTHON_BIN, SLICE, OUT_DIR, QUERY_JSONL, REFERENCE_JSONL, OUT_JSON,
  OUT_MD, LEAKAGE_JSON, BACKBONES, STEPS, SYNTHETIC_N, SEED, QUERY_SEED,
  REFERENCE_SEED
EOF
}

print_plan() {
  echo "FROZEN BACKBONE PROTOCOL CHECK"
  echo "ROOT=${ROOT}"
  echo "SLICE=${SLICE}"
  echo "OUT_DIR=${OUT_DIR}"
  echo "QUERY_JSONL=${QUERY_JSONL}"
  echo "REFERENCE_JSONL=${REFERENCE_JSONL}"
  echo "OUT_JSON=${OUT_JSON}"
  echo "OUT_MD=${OUT_MD}"
  echo "LEAKAGE_JSON=${LEAKAGE_JSON}"
  echo "BACKBONES=${BACKBONES}"
  echo "STEPS=${STEPS}  SYNTHETIC_N=${SYNTHETIC_N}  SEED=${SEED}"
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then usage; exit 0; fi
if [[ "${1:-}" == "--dry-run" ]]; then print_plan; exit 0; fi

mkdir -p "${OUT_DIR}"
export PYTHONPATH="$(dirname "${ROOT}")${PYTHONPATH:+:${PYTHONPATH}}"

"${PYTHON_BIN}" - "${QUERY_JSONL}" "${REFERENCE_JSONL}" "${QUERY_SEED}" "${REFERENCE_SEED}" <<'PY'
import sys
from mrna_editflow.data.download_mrna import synthesize_corpus, write_records_jsonl

query_path, reference_path, query_seed, reference_seed = sys.argv[1:5]
write_records_jsonl(synthesize_corpus(3, seed=int(query_seed)), query_path)
write_records_jsonl(synthesize_corpus(3, seed=int(reference_seed)), reference_path)
PY

# shellcheck disable=SC2086
"${PYTHON_BIN}" -m mrna_editflow.eval.frozen_backbone_comparison \
  --query-jsonl "${QUERY_JSONL}" \
  --reference-jsonl "${REFERENCE_JSONL}" \
  --backbones ${BACKBONES} \
  --steps "${STEPS}" \
  --synthetic-n "${SYNTHETIC_N}" \
  --seed "${SEED}" \
  --device cpu \
  --out-json "${OUT_JSON}" \
  --out-md "${OUT_MD}" \
  --leakage-json "${LEAKAGE_JSON}"

echo "Frozen-backbone protocol summary -> ${OUT_JSON}"
