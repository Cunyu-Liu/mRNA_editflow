#!/usr/bin/env bash
# Load-gated RefSeq public-corpus build.
#
# This queues the official RefSeq GenBank download/build path without bypassing
# the project load gate. It writes progress and status artifacts so parser
# readiness, download state, and manifest completion remain auditable.
set -euo pipefail

ROOT="${ROOT:-/home/cunyuliu/mrna_editflow_goal/mrna_editflow}"
PYTHON_BIN="${PYTHON_BIN:-/home/cunyuliu/miniconda3/envs/editflow/bin/python3.10}"
DATASET="${DATASET:-refseq_human_rna}"
DATA_DIR="${DATA_DIR:-${ROOT}/data/raw}"
OUT_DIR="${OUT_DIR:-${ROOT}/data/processed}"
SEED="${SEED:-20260714}"
LIMIT="${LIMIT:-}"
DOWNLOAD="${DOWNLOAD:-1}"
FORCE="${FORCE:-0}"
MAX_LOADAVG="${MAX_LOADAVG:-80}"
WAIT_SECONDS="${WAIT_SECONDS:-120}"
RUN_ID="${RUN_ID:-refseq_public_build_$(date +%Y%m%d_%H%M%S)}"
STATUS_DIR="${STATUS_DIR:-${ROOT}/benchmark/${RUN_ID}}"
PROGRESS_JSONL="${PROGRESS_JSONL:-${STATUS_DIR}/progress.jsonl}"
STATUS_JSON="${STATUS_JSON:-${STATUS_DIR}/status.json}"
STATUS_MD="${STATUS_MD:-${STATUS_DIR}/status.md}"
LOG_PATH="${LOG_PATH:-${ROOT}/logs/${RUN_ID}.log}"

usage() {
  cat <<'EOF'
Usage:
  run_refseq_public_build.sh [--dry-run]

Purpose:
  Queue a load-gated RefSeq public-corpus build through
  mrna_editflow.data.public_pipeline --dataset refseq_human_rna. The script
  records progress/status artifacts and does not claim RefSeq scale-up until the
  official file is downloaded, parsed, cleaned, and the manifest is written.

Environment overrides:
  ROOT, PYTHON_BIN, DATASET, DATA_DIR, OUT_DIR, SEED, LIMIT, DOWNLOAD, FORCE,
  MAX_LOADAVG, WAIT_SECONDS, RUN_ID, STATUS_DIR, PROGRESS_JSONL, STATUS_JSON,
  STATUS_MD, LOG_PATH
EOF
}

print_plan() {
  cat <<EOF
REFSEQ_PUBLIC_BUILD
artifact_kind=refseq_public_build
ROOT=${ROOT}
PYTHON_BIN=${PYTHON_BIN}
DATASET=${DATASET}
DATA_DIR=${DATA_DIR}
OUT_DIR=${OUT_DIR}
SEED=${SEED}
LIMIT=${LIMIT}
DOWNLOAD=${DOWNLOAD}
FORCE=${FORCE}
MAX_LOADAVG=${MAX_LOADAVG}
WAIT_SECONDS=${WAIT_SECONDS}
RUN_ID=${RUN_ID}
STATUS_JSON=${STATUS_JSON}
LOG_PATH=${LOG_PATH}
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
  mkdir -p "${STATUS_DIR}"
  PYTHONPATH="$(dirname "${ROOT}")${PYTHONPATH:+:${PYTHONPATH}}" \
    "${PYTHON_BIN}" - \
    "${ROOT}" "${DATASET}" "${DATA_DIR}" "${OUT_DIR}" "${STATUS_JSON}" "${STATUS_MD}" \
    "${PROGRESS_JSONL}" "${LOG_PATH}" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

root, dataset, data_dir, out_dir, out_json, out_md, progress_path, log_path = sys.argv[1:9]

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
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows

raw_path = os.path.join(data_dir, "human.1.rna.gbff.gz")
records_path = os.path.join(out_dir, f"{dataset}.records.jsonl")
manifest_path = os.path.join(out_dir, f"{dataset}.data_manifest.json")
progress = read_jsonl(progress_path)
last_event = progress[-1] if progress else {}
manifest = {}
if os.path.exists(manifest_path):
    with open(manifest_path, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
status = (
    "complete"
    if os.path.exists(records_path) and os.path.exists(manifest_path)
    else "queued_or_running"
)
payload = {
    "artifact_kind": "refseq_public_build_status",
    "claim_policy": (
        "RefSeq build status is corpus-readiness evidence only. Do not claim "
        "data scale-up until official raw file, cleaned records, manifest, and "
        "downstream leakage/family split audits are complete."
    ),
    "root": root,
    "dataset": dataset,
    "status": status,
    "raw": {
        "path": raw_path,
        "exists": os.path.exists(raw_path),
        "size_bytes": os.path.getsize(raw_path) if os.path.exists(raw_path) else 0,
        "sha256": sha(raw_path),
    },
    "records": {
        "path": records_path,
        "exists": os.path.exists(records_path),
        "size_bytes": os.path.getsize(records_path) if os.path.exists(records_path) else 0,
        "sha256": sha(records_path),
    },
    "manifest": {
        "path": manifest_path,
        "exists": os.path.exists(manifest_path),
        "sha256": sha(manifest_path),
        "clean_n_records": manifest.get("clean_summary", {}).get("n_records"),
        "raw_n_records": manifest.get("raw_summary", {}).get("n_records"),
    },
    "progress": {
        "path": progress_path,
        "n_events": len(progress),
        "last_event": last_event.get("event"),
        "last_loadavg": last_event.get("loadavg"),
    },
    "log": {"path": log_path, "exists": os.path.exists(log_path)},
}
Path(out_json).parent.mkdir(parents=True, exist_ok=True)
with open(out_json, "w", encoding="utf-8") as fh:
    json.dump(payload, fh, indent=2, sort_keys=True)
lines = [
    "# RefSeq Public Build Status",
    "",
    f"- Dataset: `{dataset}`",
    f"- Status: `{status}`",
    f"- Last event: `{payload['progress']['last_event']}`",
    f"- Raw exists: `{payload['raw']['exists']}`; raw SHA: `{payload['raw']['sha256']}`",
    f"- Records exists: `{payload['records']['exists']}`; records SHA: `{payload['records']['sha256']}`",
    f"- Manifest exists: `{payload['manifest']['exists']}`; clean/raw records: `{payload['manifest']['clean_n_records']}/{payload['manifest']['raw_n_records']}`",
    f"- Claim policy: {payload['claim_policy']}",
]
with open(out_md, "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines) + "\n")
print(json.dumps({"json_path": out_json, "markdown_path": out_md, "status": status}, sort_keys=True))
PY
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

mkdir -p "${STATUS_DIR}" "$(dirname "${LOG_PATH}")" "${DATA_DIR}" "${OUT_DIR}"
: > "${PROGRESS_JSONL}"
json_log queued dataset="${DATASET}" data_dir="${DATA_DIR}" out_dir="${OUT_DIR}"
write_status || true
wait_load_gate

cmd=("${PYTHON_BIN}" -m mrna_editflow.data.public_pipeline --dataset "${DATASET}" --data-dir "${DATA_DIR}" --out-dir "${OUT_DIR}" --seed "${SEED}")
if [[ "${DOWNLOAD}" == "1" ]]; then
  cmd+=(--download)
fi
if [[ "${FORCE}" == "1" ]]; then
  cmd+=(--force)
fi
if [[ -n "${LIMIT}" ]]; then
  cmd+=(--limit "${LIMIT}")
fi
json_log build_start command="${cmd[*]}"
write_status || true
(
  export PYTHONPATH="$(dirname "${ROOT}")${PYTHONPATH:+:${PYTHONPATH}}"
  "${cmd[@]}"
) > "${LOG_PATH}" 2>&1
json_log build_complete log_path="${LOG_PATH}"
write_status || true
