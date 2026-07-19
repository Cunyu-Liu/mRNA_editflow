#!/usr/bin/env bash
# Wait for real MPRA/stability tables, then build manifests and predictor reports.
set -euo pipefail

ROOT="${ROOT:-/home/cunyuliu/mrna_editflow_goal/mrna_editflow}"
PYTHON_BIN="${PYTHON_BIN:-/home/cunyuliu/miniconda3/envs/editflow/bin/python3.10}"
PROCESSED_DIR="${PROCESSED_DIR:-${ROOT}/data/processed}"
OUT_DIR="${OUT_DIR:-${ROOT}/benchmark/downstream_predictor_protocol}"
STATUS_JSON="${STATUS_JSON:-${OUT_DIR}/status.json}"
STATUS_MD="${STATUS_MD:-${OUT_DIR}/status.md}"
PROGRESS_JSONL="${PROGRESS_JSONL:-${OUT_DIR}/progress.jsonl}"
LOG_PATH="${LOG_PATH:-${ROOT}/logs/downstream_predictor_protocol.log}"
MPRA_TABLE="${MPRA_TABLE:-${MRNA_EDITFLOW_MPRA_TE_TABLE:-${MPRA_TE_TABLE:-${MPRA_TABLE:-}}}}"
MPRA_SOURCE_URL="${MPRA_SOURCE_URL:-${MRNA_EDITFLOW_MPRA_TE_SOURCE_URL:-}}"
MPRA_LICENSE="${MPRA_LICENSE:-unknown; verify before publication}"
STABILITY_TABLE="${STABILITY_TABLE:-${MRNA_EDITFLOW_STABILITY_TABLE:-${HALF_LIFE_TABLE:-}}}"
STABILITY_SOURCE_URL="${STABILITY_SOURCE_URL:-${MRNA_EDITFLOW_STABILITY_SOURCE_URL:-}}"
STABILITY_TARGET_COL="${STABILITY_TARGET_COL:-${MRNA_EDITFLOW_STABILITY_TARGET_COL:-}}"
STABILITY_LICENSE="${STABILITY_LICENSE:-unknown; verify before publication}"
MPRA_REPORT_DIR="${MPRA_REPORT_DIR:-${ROOT}/benchmark/mpra_te_predictor_protocol_real}"
STABILITY_REPORT_DIR="${STABILITY_REPORT_DIR:-${ROOT}/benchmark/stability_predictor_protocol_real}"
MAX_LOADAVG="${MAX_LOADAVG:-80}"
WAIT_SECONDS="${WAIT_SECONDS:-120}"
SEED="${SEED:-20260714}"
RIDGE_ALPHA="${RIDGE_ALPHA:-1.0}"
MIN_TEST_N="${MIN_TEST_N:-2}"
FORCE="${FORCE:-0}"

usage() {
  cat <<'EOF'
Usage:
  run_downstream_predictor_protocol.sh [--dry-run]

Purpose:
  Wait for real MPRA/TE and stability/half-life tables, require verified source
  URLs, then build dataset manifests and held-out predictor reports. This script
  never falls back to synthetic data for real predictor evidence.

Environment overrides:
  ROOT, PYTHON_BIN, PROCESSED_DIR, OUT_DIR, STATUS_JSON, STATUS_MD,
  PROGRESS_JSONL, LOG_PATH, MPRA_TABLE, MPRA_SOURCE_URL, MPRA_LICENSE,
  STABILITY_TABLE, STABILITY_SOURCE_URL, STABILITY_TARGET_COL, STABILITY_LICENSE,
  MPRA_REPORT_DIR, STABILITY_REPORT_DIR, MAX_LOADAVG, WAIT_SECONDS, SEED,
  RIDGE_ALPHA, MIN_TEST_N, FORCE
EOF
}

print_plan() {
  cat <<EOF
DOWNSTREAM_PREDICTOR_PROTOCOL
artifact_kind=downstream_predictor_protocol
ROOT=${ROOT}
PYTHON_BIN=${PYTHON_BIN}
PROCESSED_DIR=${PROCESSED_DIR}
OUT_DIR=${OUT_DIR}
STATUS_JSON=${STATUS_JSON}
PROGRESS_JSONL=${PROGRESS_JSONL}
LOG_PATH=${LOG_PATH}
MPRA_TABLE=${MPRA_TABLE}
MPRA_SOURCE_URL=${MPRA_SOURCE_URL}
STABILITY_TABLE=${STABILITY_TABLE}
STABILITY_SOURCE_URL=${STABILITY_SOURCE_URL}
STABILITY_TARGET_COL=${STABILITY_TARGET_COL}
MPRA_REPORT_DIR=${MPRA_REPORT_DIR}
STABILITY_REPORT_DIR=${STABILITY_REPORT_DIR}
MAX_LOADAVG=${MAX_LOADAVG}
WAIT_SECONDS=${WAIT_SECONDS}
SEED=${SEED}
RIDGE_ALPHA=${RIDGE_ALPHA}
MIN_TEST_N=${MIN_TEST_N}
FORCE=${FORCE}
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

mkdir -p "${OUT_DIR}" "$(dirname "${LOG_PATH}")" "${MPRA_REPORT_DIR}" "${STABILITY_REPORT_DIR}"
touch "${PROGRESS_JSONL}"

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

discover_table() {
  local explicit="$1"
  shift
  "${PYTHON_BIN}" - "$ROOT" "$explicit" "$@" <<'PY'
import glob
import os
import sys

root = sys.argv[1]
explicit = sys.argv[2]
patterns = sys.argv[3:]
if explicit:
    path = explicit if os.path.isabs(explicit) else os.path.join(root, explicit)
    if os.path.exists(path):
        print(path)
        raise SystemExit(0)
for pattern in patterns:
    for path in sorted(glob.glob(os.path.join(root, pattern))):
        if os.path.isfile(path):
            print(path)
            raise SystemExit(0)
print("")
PY
}

mpra_table_path() {
  discover_table "${MPRA_TABLE}" \
    "data/raw/*mpra*.csv" "data/raw/*mpra*.tsv" \
    "data/raw/*mrl*.csv" "data/raw/*mrl*.tsv" \
    "data/processed/*mpra*.csv" "data/processed/*mpra*.tsv" \
    "data/processed/*mrl*.csv" "data/processed/*mrl*.tsv"
}

stability_table_path() {
  discover_table "${STABILITY_TABLE}" \
    "data/raw/*stability*.csv" "data/raw/*stability*.tsv" \
    "data/raw/*half_life*.csv" "data/raw/*half_life*.tsv" \
    "data/raw/*degradation*.csv" "data/raw/*degradation*.tsv" \
    "data/processed/*stability*.csv" "data/processed/*stability*.tsv" \
    "data/processed/*half_life*.csv" "data/processed/*half_life*.tsv" \
    "data/processed/*degradation*.csv" "data/processed/*degradation*.tsv"
}

write_status() {
  local mpra_path stability_path
  mpra_path="$(mpra_table_path)"
  stability_path="$(stability_table_path)"
  PYTHONPATH="$(dirname "${ROOT}")${PYTHONPATH:+:${PYTHONPATH}}" \
    "${PYTHON_BIN}" - \
    "${STATUS_JSON}" "${STATUS_MD}" "${PROGRESS_JSONL}" \
    "${mpra_path}" "${MPRA_SOURCE_URL}" "${MPRA_REPORT_DIR}" \
    "${stability_path}" "${STABILITY_SOURCE_URL}" "${STABILITY_REPORT_DIR}" \
    "${PROCESSED_DIR}" "${LOG_PATH}" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

(
    status_json,
    status_md,
    progress_jsonl,
    mpra_path,
    mpra_source_url,
    mpra_report_dir,
    stability_path,
    stability_source_url,
    stability_report_dir,
    processed_dir,
    log_path,
) = sys.argv[1:12]

def sha(path):
    if not path or not os.path.isfile(path):
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

def report_state(report_dir, name):
    report_json = os.path.join(report_dir, "report.json")
    predictions = os.path.join(report_dir, "predictions.jsonl")
    manifest = os.path.join(processed_dir, f"{name}.data_manifest.json")
    return {
        "report_json": report_json,
        "report_exists": os.path.exists(report_json),
        "report_sha256": sha(report_json),
        "predictions_jsonl": predictions,
        "predictions_exists": os.path.exists(predictions),
        "predictions_sha256": sha(predictions),
        "manifest_json": manifest,
        "manifest_exists": os.path.exists(manifest),
        "manifest_sha256": sha(manifest),
    }

progress = read_jsonl(progress_jsonl)
last = progress[-1] if progress else {}
mpra = {
    "table_path": mpra_path or None,
    "table_exists": bool(mpra_path and os.path.exists(mpra_path)),
    "table_sha256": sha(mpra_path),
    "source_url_present": bool(mpra_source_url),
    **report_state(mpra_report_dir, "mpra_te"),
}
stability = {
    "table_path": stability_path or None,
    "table_exists": bool(stability_path and os.path.exists(stability_path)),
    "table_sha256": sha(stability_path),
    "source_url_present": bool(stability_source_url),
    **report_state(stability_report_dir, "stability_half_life"),
}
complete = bool(mpra["report_exists"] and stability["report_exists"])
if complete:
    status = "complete"
elif not mpra["table_exists"] and not stability["table_exists"]:
    status = "waiting_for_downstream_tables"
elif (mpra["table_exists"] and not mpra["source_url_present"]) or (
    stability["table_exists"] and not stability["source_url_present"]
):
    status = "waiting_for_source_urls"
else:
    status = "queued_or_running"

payload = {
    "artifact_kind": "downstream_predictor_protocol_status",
    "claim_policy": (
        "Real downstream predictor status is protocol evidence only. Do not "
        "claim real TE/stability performance until both external tables, "
        "manifests, held-out predictor reports, and leakage docs are complete."
    ),
    "status": status,
    "mpra_te": mpra,
    "stability_half_life": stability,
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
    "# Downstream Predictor Protocol Status",
    "",
    f"- Status: `{payload['status']}`",
    f"- MPRA table/report: `{mpra['table_exists']}/{mpra['report_exists']}`; source URL: `{mpra['source_url_present']}`",
    f"- Stability table/report: `{stability['table_exists']}/{stability['report_exists']}`; source URL: `{stability['source_url_present']}`",
    f"- Last event: `{payload['progress']['last_event']}`; loadavg: `{payload['progress']['last_loadavg']}`",
    f"- Claim policy: {payload['claim_policy']}",
]
with open(status_md, "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines) + "\n")
print(json.dumps({"status": status, "json_path": status_json, "markdown_path": status_md}, sort_keys=True))
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

refresh_governance_reports() {
  PYTHONPATH="$(dirname "${ROOT}")${PYTHONPATH:+:${PYTHONPATH}}" \
    "${PYTHON_BIN}" -m mrna_editflow.eval.dataset_manifest_audit \
      --project-root "${ROOT}" \
      --out-json "${ROOT}/docs/dataset_manifest_audit.json" \
      --out-md "${ROOT}/docs/dataset_manifest_audit.md"
  PYTHONPATH="$(dirname "${ROOT}")${PYTHONPATH:+:${PYTHONPATH}}" \
    "${PYTHON_BIN}" -m mrna_editflow.eval.downstream_data_acquisition_audit \
      --project-root "${ROOT}" \
      --out-json "${ROOT}/docs/downstream_data_acquisition_audit.json" \
      --out-md "${ROOT}/docs/downstream_data_acquisition_audit.md"
  PYTHONPATH="$(dirname "${ROOT}")${PYTHONPATH:+:${PYTHONPATH}}" \
    "${PYTHON_BIN}" -m mrna_editflow.eval.build_data_scaleup_readiness \
      --project-root "${ROOT}" \
      --out-json "${ROOT}/docs/data_scaleup_readiness.json" \
      --out-md "${ROOT}/docs/data_scaleup_readiness.md"
}

run_mpra() {
  local table_path
  table_path="$(mpra_table_path)"
  if [[ -z "${table_path}" || -z "${MPRA_SOURCE_URL}" ]]; then
    return 0
  fi
  if [[ -s "${MPRA_REPORT_DIR}/report.json" && "${FORCE}" != "1" ]]; then
    return 0
  fi
  mkdir -p "${MPRA_REPORT_DIR}"
  json_log mpra_manifest_start table="${table_path}"
  PYTHONPATH="$(dirname "${ROOT}")${PYTHONPATH:+:${PYTHONPATH}}" \
    "${PYTHON_BIN}" -m mrna_editflow.eval.build_downstream_table_manifest \
      --dataset mpra_te \
      --input "${table_path}" \
      --out-dir "${PROCESSED_DIR}" \
      --source-url "${MPRA_SOURCE_URL}" \
      --license "${MPRA_LICENSE}" >>"${LOG_PATH}" 2>&1
  json_log mpra_predictor_start table="${table_path}"
  PYTHONPATH="$(dirname "${ROOT}")${PYTHONPATH:+:${PYTHONPATH}}" \
    "${PYTHON_BIN}" -m mrna_editflow.eval.mpra_te_predictor \
      --input "${table_path}" \
      --out-json "${MPRA_REPORT_DIR}/report.json" \
      --out-md "${MPRA_REPORT_DIR}/report.md" \
      --predictions-jsonl "${MPRA_REPORT_DIR}/predictions.jsonl" \
      --seed "${SEED}" \
      --ridge-alpha "${RIDGE_ALPHA}" \
      --min-test-n "${MIN_TEST_N}" >>"${LOG_PATH}" 2>&1
  json_log mpra_complete report="${MPRA_REPORT_DIR}/report.json"
}

run_stability() {
  local table_path
  table_path="$(stability_table_path)"
  if [[ -z "${table_path}" || -z "${STABILITY_SOURCE_URL}" ]]; then
    return 0
  fi
  if [[ -s "${STABILITY_REPORT_DIR}/report.json" && "${FORCE}" != "1" ]]; then
    return 0
  fi
  mkdir -p "${STABILITY_REPORT_DIR}"
  json_log stability_manifest_start table="${table_path}"
  cmd=(
    "${PYTHON_BIN}" -m mrna_editflow.eval.build_downstream_table_manifest
    --dataset stability_half_life
    --input "${table_path}"
    --out-dir "${PROCESSED_DIR}"
    --source-url "${STABILITY_SOURCE_URL}"
    --license "${STABILITY_LICENSE}"
  )
  if [[ -n "${STABILITY_TARGET_COL}" ]]; then
    cmd+=(--target-col "${STABILITY_TARGET_COL}")
  fi
  PYTHONPATH="$(dirname "${ROOT}")${PYTHONPATH:+:${PYTHONPATH}}" "${cmd[@]}" >>"${LOG_PATH}" 2>&1
  json_log stability_predictor_start table="${table_path}"
  cmd=(
    "${PYTHON_BIN}" -m mrna_editflow.eval.stability_predictor
    --input "${table_path}"
    --out-json "${STABILITY_REPORT_DIR}/report.json"
    --out-md "${STABILITY_REPORT_DIR}/report.md"
    --predictions-jsonl "${STABILITY_REPORT_DIR}/predictions.jsonl"
    --seed "${SEED}"
    --ridge-alpha "${RIDGE_ALPHA}"
    --min-test-n "${MIN_TEST_N}"
  )
  if [[ -n "${STABILITY_TARGET_COL}" ]]; then
    cmd+=(--target-col "${STABILITY_TARGET_COL}")
  fi
  PYTHONPATH="$(dirname "${ROOT}")${PYTHONPATH:+:${PYTHONPATH}}" "${cmd[@]}" >>"${LOG_PATH}" 2>&1
  json_log stability_complete report="${STABILITY_REPORT_DIR}/report.json"
}

json_log start out_dir="${OUT_DIR}"
write_status || true

while true; do
  mpra_path="$(mpra_table_path)"
  stability_path="$(stability_table_path)"
  if [[ -z "${mpra_path}" && -z "${stability_path}" ]]; then
    json_log wait_downstream_tables wait_seconds="${WAIT_SECONDS}"
    write_status || true
    sleep "${WAIT_SECONDS}"
    continue
  fi
  if [[ (-n "${mpra_path}" && -z "${MPRA_SOURCE_URL}") || (-n "${stability_path}" && -z "${STABILITY_SOURCE_URL}") ]]; then
    json_log wait_source_urls mpra_table="${mpra_path}" stability_table="${stability_path}" wait_seconds="${WAIT_SECONDS}"
    write_status || true
    sleep "${WAIT_SECONDS}"
    continue
  fi
  wait_load_gate
  set +e
  run_mpra
  mpra_rc=$?
  run_stability
  stability_rc=$?
  set -e
  json_log command_exit mpra_exit_code="${mpra_rc}" stability_exit_code="${stability_rc}"
  write_status || true
  refresh_governance_reports || true
  exit $(( mpra_rc != 0 ? mpra_rc : stability_rc ))
done
