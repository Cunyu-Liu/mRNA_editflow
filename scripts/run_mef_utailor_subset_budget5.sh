#!/usr/bin/env bash
# Run the best MEF UTR teacher on the UTailoR-eligible subset at budget five.
set -euo pipefail

ROOT="${ROOT:-/home/cunyuliu/mrna_editflow_goal/mrna_editflow}"
PYTHON_BIN="${PYTHON_BIN:-/home/cunyuliu/miniconda3/envs/editflow/bin/python3.10}"
SOURCE_ALL="${SOURCE_ALL:-${ROOT}/benchmark/multiseed_t5_public_head1024_sources.jsonl}"
SUBSET_SOURCES="${SUBSET_SOURCES:-${ROOT}/benchmark/utailor_strict_25_100_sources.jsonl}"
SUBSET_SUMMARY="${SUBSET_SUMMARY:-${ROOT}/benchmark/utailor_strict_25_100_sources.summary.json}"
CHECKPOINT="${CHECKPOINT:-${ROOT}/ckpts/proposal_ranker_t5_utr_teacher_head256/proposal_ranker_best.pt}"
OUT_DIR="${OUT_DIR:-${ROOT}/benchmark/multiseed_t5_utailor_strict315_pure_utr_teacher_budget5_top64}"
STATUS_JSON="${STATUS_JSON:-${OUT_DIR}/status.json}"
SHARD_GPUS="${SHARD_GPUS:-0 2}"

if [[ "${1:-}" == "--dry-run" ]]; then
  echo "MEF UTAILOR SUBSET BUDGET5"
  echo "SOURCE_ALL=${SOURCE_ALL}"
  echo "SUBSET_SOURCES=${SUBSET_SOURCES}"
  echo "SUBSET_SUMMARY=${SUBSET_SUMMARY}"
  echo "CHECKPOINT=${CHECKPOINT}"
  echo "OUT_DIR=${OUT_DIR}"
  echo "EDIT_BUDGET=5 LIMIT=315 SHARD_GPUS=${SHARD_GPUS}"
  exit 0
fi

SOURCE_ALL="${SOURCE_ALL}" SUBSET_SOURCES="${SUBSET_SOURCES}" \
SUBSET_SUMMARY="${SUBSET_SUMMARY}" "${PYTHON_BIN}" - <<'PY'
import hashlib
import json
import os

source_path = os.environ["SOURCE_ALL"]
out_path = os.environ["SUBSET_SOURCES"]
summary_path = os.environ["SUBSET_SUMMARY"]
rows = [json.loads(line) for line in open(source_path) if line.strip()]
eligible = [row for row in rows if 25 <= len(row.get("five_utr", "")) <= 100]
os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
with open(out_path, "w", encoding="utf-8") as fh:
    for row in eligible:
        fh.write(json.dumps(row, sort_keys=True) + "\n")

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

payload = {
    "artifact_kind": "utailor_strict_input_domain_subset",
    "eligibility_policy": "official_input_length_25_100_strict",
    "source_path": source_path,
    "source_sha256": sha(source_path),
    "source_n": len(rows),
    "subset_path": out_path,
    "subset_sha256": sha(out_path),
    "subset_n": len(eligible),
}
with open(summary_path, "w", encoding="utf-8") as fh:
    json.dump(payload, fh, indent=2, sort_keys=True)
if len(eligible) != 315:
    raise SystemExit(f"expected 315 eligible rows, observed {len(eligible)}")
print(json.dumps(payload, sort_keys=True))
PY

ROOT="${ROOT}" \
PYTHON_BIN="${PYTHON_BIN}" \
SOURCES="${SUBSET_SOURCES}" \
CHECKPOINT="${CHECKPOINT}" \
OUT_DIR="${OUT_DIR}" \
STATUS_JSON="${STATUS_JSON}" \
LIMIT=315 \
EDIT_BUDGET=5 \
SHARDED=1 \
SHARD_GPUS="${SHARD_GPUS}" \
REFRESH_REPORTS=0 \
bash "${ROOT}/scripts/run_mef_utr5only_head1024.sh"
