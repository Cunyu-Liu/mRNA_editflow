#!/usr/bin/env bash
# Watch P3 artifacts and refresh readiness reports when inputs change.
set -euo pipefail

ROOT="${ROOT:-/home/cunyuliu/mrna_editflow_goal/mrna_editflow}"
PYTHON_BIN="${PYTHON_BIN:-/home/cunyuliu/miniconda3/envs/editflow/bin/python3.10}"
POLL_SECONDS="${POLL_SECONDS:-120}"
MARKER="${MARKER:-${ROOT}/logs/p3_readiness_watcher.state.json}"
LOG_PATH="${LOG_PATH:-${ROOT}/logs/p3_readiness_watcher.log}"
GENCODE_REPORT_JSON="${GENCODE_REPORT_JSON:-${ROOT}/benchmark/gencode_family_leakage_protocol/report.json}"
REFSEQ_MANIFEST_JSON="${REFSEQ_MANIFEST_JSON:-${ROOT}/data/processed/refseq_human_rna.data_manifest.json}"
STAGE_A_SUMMARY_JSON="${STAGE_A_SUMMARY_JSON:-${ROOT}/benchmark/stage_a_scalelaw_p3p4_20260715_0714/summary.json}"
OUT_MANIFEST_JSON="${OUT_MANIFEST_JSON:-${ROOT}/docs/dataset_manifest_audit.json}"
OUT_MANIFEST_MD="${OUT_MANIFEST_MD:-${ROOT}/docs/dataset_manifest_audit.md}"
OUT_ACQUISITION_JSON="${OUT_ACQUISITION_JSON:-${ROOT}/docs/downstream_data_acquisition_audit.json}"
OUT_ACQUISITION_MD="${OUT_ACQUISITION_MD:-${ROOT}/docs/downstream_data_acquisition_audit.md}"
OUT_READINESS_JSON="${OUT_READINESS_JSON:-${ROOT}/docs/data_scaleup_readiness.json}"
OUT_READINESS_MD="${OUT_READINESS_MD:-${ROOT}/docs/data_scaleup_readiness.md}"
OUT_STAGE_A_DOWNSTREAM_JSON="${OUT_STAGE_A_DOWNSTREAM_JSON:-${ROOT}/docs/stage_a_downstream_eval_readiness.json}"
OUT_STAGE_A_DOWNSTREAM_MD="${OUT_STAGE_A_DOWNSTREAM_MD:-${ROOT}/docs/stage_a_downstream_eval_readiness.md}"
OUT_TABLE5_JSON="${OUT_TABLE5_JSON:-${ROOT}/docs/paper_table5_scale_law_readiness.json}"
OUT_TABLE5_MD="${OUT_TABLE5_MD:-${ROOT}/docs/paper_table5_scale_law_readiness.md}"
RUN_ONCE="${RUN_ONCE:-0}"

usage() {
  cat <<'EOF'
Usage:
  watch_p3_readiness.sh [--dry-run]

Purpose:
  Watch P3 readiness inputs and refresh paper/data governance reports when
  GENCODE family split, RefSeq corpus, Stage A scale-law, or downstream
  MPRA/stability table artifacts appear. This watcher is lightweight: it does
  not run training, download RefSeq, or execute leakage audits.

Environment overrides:
  ROOT, PYTHON_BIN, POLL_SECONDS, MARKER, LOG_PATH, GENCODE_REPORT_JSON,
  REFSEQ_MANIFEST_JSON, STAGE_A_SUMMARY_JSON, OUT_* paths, RUN_ONCE
EOF
}

print_plan() {
  cat <<EOF
P3_READINESS_WATCHER
artifact_kind=p3_readiness_watcher
ROOT=${ROOT}
PYTHON_BIN=${PYTHON_BIN}
POLL_SECONDS=${POLL_SECONDS}
GENCODE_REPORT_JSON=${GENCODE_REPORT_JSON}
REFSEQ_MANIFEST_JSON=${REFSEQ_MANIFEST_JSON}
STAGE_A_SUMMARY_JSON=${STAGE_A_SUMMARY_JSON}
OUT_MANIFEST_JSON=${OUT_MANIFEST_JSON}
OUT_ACQUISITION_JSON=${OUT_ACQUISITION_JSON}
OUT_READINESS_JSON=${OUT_READINESS_JSON}
OUT_STAGE_A_DOWNSTREAM_JSON=${OUT_STAGE_A_DOWNSTREAM_JSON}
OUT_TABLE5_JSON=${OUT_TABLE5_JSON}
MARKER=${MARKER}
LOG_PATH=${LOG_PATH}
RUN_ONCE=${RUN_ONCE}
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
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing executable PYTHON_BIN: ${PYTHON_BIN}" >&2
  exit 1
fi

mkdir -p "$(dirname "${MARKER}")" "$(dirname "${LOG_PATH}")"

state_digest() {
  ROOT="${ROOT}" \
  GENCODE_REPORT_JSON="${GENCODE_REPORT_JSON}" \
  REFSEQ_MANIFEST_JSON="${REFSEQ_MANIFEST_JSON}" \
  STAGE_A_SUMMARY_JSON="${STAGE_A_SUMMARY_JSON}" \
  "${PYTHON_BIN}" - <<'PY'
import glob
import hashlib
import json
import os

root = os.environ["ROOT"]
paths = [
    os.environ["GENCODE_REPORT_JSON"],
    os.environ["REFSEQ_MANIFEST_JSON"],
    os.environ["STAGE_A_SUMMARY_JSON"],
]
patterns = [
    "data/raw/*mpra*.csv",
    "data/raw/*mpra*.tsv",
    "data/raw/*mrl*.csv",
    "data/raw/*mrl*.tsv",
    "data/raw/*stability*.csv",
    "data/raw/*stability*.tsv",
    "data/raw/*half_life*.csv",
    "data/raw/*half_life*.tsv",
    "data/raw/*degradation*.csv",
    "data/raw/*degradation*.tsv",
    "data/processed/*mpra*.csv",
    "data/processed/*mpra*.tsv",
    "data/processed/*stability*.csv",
    "data/processed/*stability*.tsv",
    "data/processed/*half_life*.csv",
    "data/processed/*half_life*.tsv",
    "data/processed/*degradation*.csv",
    "data/processed/*degradation*.tsv",
    "data/processed/*mpra*.data_manifest.json",
    "data/processed/*stability*.data_manifest.json",
    "data/processed/*half_life*.data_manifest.json",
    "data/processed/*degradation*.data_manifest.json",
    "benchmark/refseq_family_leakage_protocol/report.json",
    "benchmark/refseq_family_leakage_protocol/status.json",
    "benchmark/refseq_family_leakage_protocol/splits/*.idx",
    "benchmark/mpra_te_predictor_protocol_real/report.json",
    "benchmark/mpra_te_predictor_protocol_real/predictions.jsonl",
    "benchmark/stability_predictor_protocol_real/report.json",
    "benchmark/stability_predictor_protocol_real/predictions.jsonl",
    "benchmark/downstream_predictor_protocol/status.json",
    "benchmark/stage_a_scalelaw_downstream/*/*.json",
    "benchmark/stage_a_scalelaw_downstream/status.json",
    "docs/stage_a_scalelaw_downstream_eval_summary.json",
    "docs/stage_a_scalelaw_downstream_trend_audit.json",
]
for pattern in patterns:
    paths.extend(glob.glob(os.path.join(root, pattern)))

rows = []
for path in sorted(set(paths)):
    exists = os.path.exists(path)
    sha = None
    size = None
    if exists and os.path.isfile(path):
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(chunk)
        sha = h.hexdigest()
        size = os.path.getsize(path)
    rows.append({
        "path": os.path.relpath(path, root) if path.startswith(root) else path,
        "exists": exists,
        "size_bytes": size,
        "sha256": sha,
    })
payload = {"watched": rows}
encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
print(hashlib.sha256(encoded).hexdigest())
PY
}

refresh_reports() {
  PYTHONPATH="$(dirname "${ROOT}")${PYTHONPATH:+:${PYTHONPATH}}" \
    "${PYTHON_BIN}" -m mrna_editflow.eval.dataset_manifest_audit \
      --project-root "${ROOT}" \
      --out-json "${OUT_MANIFEST_JSON}" \
      --out-md "${OUT_MANIFEST_MD}"

  PYTHONPATH="$(dirname "${ROOT}")${PYTHONPATH:+:${PYTHONPATH}}" \
    "${PYTHON_BIN}" -m mrna_editflow.eval.downstream_data_acquisition_audit \
      --project-root "${ROOT}" \
      --out-json "${OUT_ACQUISITION_JSON}" \
      --out-md "${OUT_ACQUISITION_MD}"

  PYTHONPATH="$(dirname "${ROOT}")${PYTHONPATH:+:${PYTHONPATH}}" \
    "${PYTHON_BIN}" -m mrna_editflow.eval.build_data_scaleup_readiness \
      --project-root "${ROOT}" \
      --out-json "${OUT_READINESS_JSON}" \
      --out-md "${OUT_READINESS_MD}"

  PYTHONPATH="$(dirname "${ROOT}")${PYTHONPATH:+:${PYTHONPATH}}" \
    "${PYTHON_BIN}" -m mrna_editflow.eval.stage_a_downstream_eval_readiness \
      --project-root "${ROOT}" \
      --out-json "${OUT_STAGE_A_DOWNSTREAM_JSON}" \
      --out-md "${OUT_STAGE_A_DOWNSTREAM_MD}"

  PYTHONPATH="$(dirname "${ROOT}")${PYTHONPATH:+:${PYTHONPATH}}" \
    "${PYTHON_BIN}" -m mrna_editflow.eval.build_paper_table5_scale_law_readiness \
      --project-root "${ROOT}" \
      --out-json "${OUT_TABLE5_JSON}" \
      --out-md "${OUT_TABLE5_MD}"
}

previous_digest=""
if [[ -s "${MARKER}" ]]; then
  previous_digest="$("${PYTHON_BIN}" - "${MARKER}" <<'PY'
import json
import sys
try:
    with open(sys.argv[1], "r", encoding="utf-8") as fh:
        print(json.load(fh).get("state_digest", ""))
except Exception:
    print("")
PY
)"
fi

while true; do
  current_digest="$(state_digest)"
  if [[ "${current_digest}" != "${previous_digest}" ]]; then
    {
      echo "[$(date -Is)] refresh state_digest=${current_digest}"
      refresh_reports
    } >> "${LOG_PATH}" 2>&1
    "${PYTHON_BIN}" - "${MARKER}" "${current_digest}" <<PY
import json
import time
import sys

payload = {
    "artifact_kind": "p3_readiness_watcher",
    "status": "refreshed",
    "state_digest": sys.argv[2],
    "time": time.time(),
    "manifest_json": "${OUT_MANIFEST_JSON}",
    "acquisition_json": "${OUT_ACQUISITION_JSON}",
    "readiness_json": "${OUT_READINESS_JSON}",
    "stage_a_downstream_json": "${OUT_STAGE_A_DOWNSTREAM_JSON}",
    "table5_json": "${OUT_TABLE5_JSON}",
}
with open(sys.argv[1], "w", encoding="utf-8") as fh:
    json.dump(payload, fh, indent=2, sort_keys=True)
PY
    previous_digest="${current_digest}"
  fi
  if [[ "${RUN_ONCE}" == "1" ]]; then
    exit 0
  fi
  sleep "${POLL_SECONDS}"
done
