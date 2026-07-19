#!/usr/bin/env bash
# Wait for RefSeq records, then run a load-gated family/leakage audit.
set -euo pipefail

ROOT="${ROOT:-/home/cunyuliu/mrna_editflow_goal/mrna_editflow}"
PYTHON_BIN="${PYTHON_BIN:-/home/cunyuliu/miniconda3/envs/editflow/bin/python3.10}"
RECORDS_JSONL="${RECORDS_JSONL:-${ROOT}/data/processed/refseq_human_rna.records.jsonl}"
MANIFEST_JSON="${MANIFEST_JSON:-${ROOT}/data/processed/refseq_human_rna.data_manifest.json}"
REFERENCE_JSONL="${REFERENCE_JSONL:-${ROOT}/data/processed/gencode_human_transcripts.records.jsonl}"
OUT_DIR="${OUT_DIR:-${ROOT}/benchmark/refseq_family_leakage_protocol}"
SPLIT_DIR="${SPLIT_DIR:-${OUT_DIR}/splits}"
REPORT_JSON="${REPORT_JSON:-${OUT_DIR}/report.json}"
REPORT_MD="${REPORT_MD:-${OUT_DIR}/report.md}"
STATUS_JSON="${STATUS_JSON:-${OUT_DIR}/status.json}"
STATUS_MD="${STATUS_MD:-${OUT_DIR}/status.md}"
PROGRESS_JSONL="${PROGRESS_JSONL:-${OUT_DIR}/progress.jsonl}"
LOG_PATH="${LOG_PATH:-${ROOT}/logs/refseq_family_leakage_protocol.log}"
MAX_LOADAVG="${MAX_LOADAVG:-80}"
WAIT_SECONDS="${WAIT_SECONDS:-120}"
SEED="${SEED:-20260714}"
USE_MMSEQS="${USE_MMSEQS:-never}"
KMER="${KMER:-15}"
TOP_K="${TOP_K:-3}"
FORCE="${FORCE:-0}"

usage() {
  cat <<'EOF'
Usage:
  run_refseq_family_leakage_audit.sh [--dry-run]

Purpose:
  Queue the RefSeq family-disjoint split and GENCODE-vs-RefSeq k-mer leakage
  audit. The script waits for RefSeq records/manifest, then waits for the load
  gate before running the audit. It does not build RefSeq and does not bypass
  MAX_LOADAVG.

Environment overrides:
  ROOT, PYTHON_BIN, RECORDS_JSONL, MANIFEST_JSON, REFERENCE_JSONL, OUT_DIR,
  SPLIT_DIR, REPORT_JSON, REPORT_MD, STATUS_JSON, STATUS_MD, PROGRESS_JSONL,
  LOG_PATH, MAX_LOADAVG, WAIT_SECONDS, SEED, USE_MMSEQS, KMER, TOP_K, FORCE
EOF
}

print_plan() {
  cat <<EOF
REFSEQ_FAMILY_LEAKAGE_PROTOCOL
artifact_kind=refseq_family_leakage_protocol
ROOT=${ROOT}
PYTHON_BIN=${PYTHON_BIN}
RECORDS_JSONL=${RECORDS_JSONL}
MANIFEST_JSON=${MANIFEST_JSON}
REFERENCE_JSONL=${REFERENCE_JSONL}
OUT_DIR=${OUT_DIR}
SPLIT_DIR=${SPLIT_DIR}
REPORT_JSON=${REPORT_JSON}
STATUS_JSON=${STATUS_JSON}
PROGRESS_JSONL=${PROGRESS_JSONL}
LOG_PATH=${LOG_PATH}
MAX_LOADAVG=${MAX_LOADAVG}
WAIT_SECONDS=${WAIT_SECONDS}
SEED=${SEED}
USE_MMSEQS=${USE_MMSEQS}
KMER=${KMER}
TOP_K=${TOP_K}
FORCE=${FORCE}
EOF
}

json_log() {
  "${PYTHON_BIN}" - "$PROGRESS_JSONL" "$@" <<'PY'
import json
import sys
import time

path = sys.argv[1]
event = sys.argv[2]
fields = {}
for item in sys.argv[3:]:
    if "=" not in item:
        continue
    key, value = item.split("=", 1)
    fields[key] = value
fields.update({"event": event, "time": time.time()})
with open(path, "a", encoding="utf-8") as fh:
    fh.write(json.dumps(fields, sort_keys=True) + "\n")
PY
}

write_status() {
  mkdir -p "${OUT_DIR}"
  PYTHONPATH="$(dirname "${ROOT}")${PYTHONPATH:+:${PYTHONPATH}}" \
    "${PYTHON_BIN}" - \
    "${ROOT}" "${RECORDS_JSONL}" "${MANIFEST_JSON}" "${REFERENCE_JSONL}" \
    "${OUT_DIR}" "${SPLIT_DIR}" "${REPORT_JSON}" "${REPORT_MD}" \
    "${STATUS_JSON}" "${STATUS_MD}" "${PROGRESS_JSONL}" "${LOG_PATH}" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

(
    root,
    records_jsonl,
    manifest_json,
    reference_jsonl,
    out_dir,
    split_dir,
    report_json,
    report_md,
    status_json,
    status_md,
    progress_jsonl,
    log_path,
) = sys.argv[1:13]

def sha(path):
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def read_jsonl(path):
    rows = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh if line.strip()]
    return rows

def load_json(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return payload if isinstance(payload, dict) else {}

progress = read_jsonl(progress_jsonl)
last = progress[-1] if progress else {}
report = load_json(report_json)
summary = report.get("summary", {}) if isinstance(report, dict) else {}
split = report.get("split", {}) if isinstance(report, dict) else {}
split_paths = {
    name: os.path.join(split_dir, f"{name}.idx")
    for name in ("train", "val", "test")
}
complete = bool(
    os.path.exists(report_json)
    and os.path.exists(split_paths["train"])
    and os.path.exists(split_paths["val"])
    and os.path.exists(split_paths["test"])
)
if complete:
    status = "complete"
elif not os.path.exists(records_jsonl) or not os.path.exists(manifest_json):
    status = "waiting_for_refseq_records"
elif not os.path.exists(reference_jsonl):
    status = "waiting_for_reference_records"
else:
    status = "queued_or_running"

payload = {
    "artifact_kind": "refseq_family_leakage_protocol_status",
    "claim_policy": (
        "RefSeq family leakage status is corpus-governance evidence. Do not "
        "claim leakage-free RefSeq scale-up until report/splits are complete "
        "and dataset manifests are audited."
    ),
    "root": root,
    "status": status,
    "records": {
        "path": records_jsonl,
        "exists": os.path.exists(records_jsonl),
        "sha256": sha(records_jsonl),
    },
    "manifest": {
        "path": manifest_json,
        "exists": os.path.exists(manifest_json),
        "sha256": sha(manifest_json),
    },
    "reference": {
        "path": reference_jsonl,
        "exists": os.path.exists(reference_jsonl),
        "sha256": sha(reference_jsonl),
    },
    "report": {
        "path": report_json,
        "exists": os.path.exists(report_json),
        "sha256": sha(report_json),
        "summary": summary,
    },
    "split": {
        "dir": split_dir,
        "paths": {
            name: {
                "path": path,
                "exists": os.path.exists(path),
                "sha256": sha(path),
            }
            for name, path in split_paths.items()
        },
        "n_train": split.get("n_train"),
        "n_val": split.get("n_val"),
        "n_test": split.get("n_test"),
        "n_clusters": split.get("n_clusters"),
    },
    "progress": {
        "path": progress_jsonl,
        "n_events": len(progress),
        "last_event": last.get("event"),
        "last_loadavg": last.get("loadavg"),
    },
    "log": {"path": log_path, "exists": os.path.exists(log_path)},
}
Path(status_json).parent.mkdir(parents=True, exist_ok=True)
with open(status_json, "w", encoding="utf-8") as fh:
    json.dump(payload, fh, indent=2, sort_keys=True)
lines = [
    "# RefSeq Family Leakage Protocol Status",
    "",
    f"- Status: `{payload['status']}`",
    f"- Last event: `{payload['progress']['last_event']}`; loadavg: `{payload['progress']['last_loadavg']}`",
    f"- Records exists: `{payload['records']['exists']}`; manifest exists: `{payload['manifest']['exists']}`",
    f"- Reference exists: `{payload['reference']['exists']}`",
    f"- Report exists: `{payload['report']['exists']}`; report SHA: `{payload['report']['sha256']}`",
    f"- Split train/val/test: `{payload['split']['n_train']}/{payload['split']['n_val']}/{payload['split']['n_test']}`",
    f"- Claim policy: {payload['claim_policy']}",
]
with open(status_md, "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines) + "\n")
print(json.dumps({"status": payload["status"], "json_path": status_json, "markdown_path": status_md}, sort_keys=True))
PY
}

wait_refseq_records() {
  while [[ ! -s "${RECORDS_JSONL}" || ! -s "${MANIFEST_JSON}" ]]; do
    echo "[$(date -Is)] waiting for RefSeq records/manifest; sleeping ${WAIT_SECONDS}s"
    json_log wait_refseq_records records_jsonl="${RECORDS_JSONL}" manifest_json="${MANIFEST_JSON}" wait_seconds="${WAIT_SECONDS}"
    write_status || true
    sleep "${WAIT_SECONDS}"
  done
  json_log refseq_records_ready records_jsonl="${RECORDS_JSONL}" manifest_json="${MANIFEST_JSON}"
  write_status || true
}

wait_reference_records() {
  while [[ ! -s "${REFERENCE_JSONL}" ]]; do
    echo "[$(date -Is)] waiting for reference records; sleeping ${WAIT_SECONDS}s"
    json_log wait_reference_records reference_jsonl="${REFERENCE_JSONL}" wait_seconds="${WAIT_SECONDS}"
    write_status || true
    sleep "${WAIT_SECONDS}"
  done
  json_log reference_records_ready reference_jsonl="${REFERENCE_JSONL}"
  write_status || true
}

wait_load_gate() {
  while true; do
    local load
    load="$(awk '{print $1}' /proc/loadavg 2>/dev/null || echo 0)"
    if "${PYTHON_BIN}" - "$load" "$MAX_LOADAVG" <<'PY'
import sys
raise SystemExit(0 if float(sys.argv[1]) < float(sys.argv[2]) else 1)
PY
    then
      json_log load_gate_pass loadavg="${load}" max_loadavg="${MAX_LOADAVG}"
      write_status || true
      return 0
    fi
    echo "[$(date -Is)] loadavg ${load} >= ${MAX_LOADAVG}; waiting ${WAIT_SECONDS}s"
    json_log load_gate_wait loadavg="${load}" max_loadavg="${MAX_LOADAVG}" wait_seconds="${WAIT_SECONDS}"
    write_status || true
    sleep "${WAIT_SECONDS}"
  done
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

mkdir -p "${OUT_DIR}" "${SPLIT_DIR}" "$(dirname "${LOG_PATH}")"
touch "${PROGRESS_JSONL}"
json_log start records_jsonl="${RECORDS_JSONL}" manifest_json="${MANIFEST_JSON}" reference_jsonl="${REFERENCE_JSONL}" out_dir="${OUT_DIR}"
write_status || true

if [[ -s "${REPORT_JSON}" && "${FORCE}" != "1" ]]; then
  json_log skip_existing_report report_json="${REPORT_JSON}"
  write_status || true
  exit 0
fi

wait_refseq_records
wait_reference_records
wait_load_gate
json_log launch
set +e
PYTHONPATH="$(dirname "${ROOT}")${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PYTHON_BIN}" -m mrna_editflow.eval.family_leakage_protocol \
    --records-jsonl "${RECORDS_JSONL}" \
    --reference-jsonl "${REFERENCE_JSONL}" \
    --out-json "${REPORT_JSON}" \
    --out-md "${REPORT_MD}" \
    --out-split-dir "${SPLIT_DIR}" \
    --seed "${SEED}" \
    --use-mmseqs "${USE_MMSEQS}" \
    --kmer "${KMER}" \
    --top-k "${TOP_K}" \
    >"${LOG_PATH}" 2>&1
rc=$?
set -e
json_log command_exit exit_code="${rc}"
write_status || true
exit "${rc}"
