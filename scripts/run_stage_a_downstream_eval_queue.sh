#!/usr/bin/env bash
# Wait for Stage A scale-law checkpoints, then run downstream evals per run.
set -euo pipefail

ROOT="${ROOT:-/home/cunyuliu/mrna_editflow_goal/mrna_editflow}"
PYTHON_BIN="${PYTHON_BIN:-/home/cunyuliu/miniconda3/envs/editflow/bin/python3.10}"
SWEEP_DIR="${SWEEP_DIR:-${ROOT}/benchmark/stage_a_scalelaw_p3p4_20260715_0714}"
OUT_DIR="${OUT_DIR:-${ROOT}/benchmark/stage_a_scalelaw_downstream}"
STATUS_JSON="${STATUS_JSON:-${OUT_DIR}/status.json}"
STATUS_MD="${STATUS_MD:-${OUT_DIR}/status.md}"
PROGRESS_JSONL="${PROGRESS_JSONL:-${OUT_DIR}/progress.jsonl}"
LOG_PATH="${LOG_PATH:-${ROOT}/logs/stage_a_scalelaw_downstream_eval.log}"
MAX_LOADAVG="${MAX_LOADAVG:-80}"
WAIT_SECONDS="${WAIT_SECONDS:-120}"
DEVICE="${DEVICE:-cuda}"
TASK_ID="${TASK_ID:-T5}"
EVAL_LIMIT="${EVAL_LIMIT:-128}"
PROPOSAL_TOP_K="${PROPOSAL_TOP_K:-32}"
CANDIDATE_CAP="${CANDIDATE_CAP:-0}"
MULTISEED_SEEDS="${MULTISEED_SEEDS:-0}"
FORCE="${FORCE:-0}"

usage() {
  cat <<'EOF'
Usage:
  run_stage_a_downstream_eval_queue.sh [--dry-run]

Purpose:
  Wait for all Stage A scale-law sweep checkpoints, then run per-run downstream
  proposal-ranking and T5 multiseed evaluation artifacts. This script does not
  train Stage A and does not claim a scale law by itself.

Environment overrides:
  ROOT, PYTHON_BIN, SWEEP_DIR, OUT_DIR, STATUS_JSON, STATUS_MD, PROGRESS_JSONL,
  LOG_PATH, MAX_LOADAVG, WAIT_SECONDS, DEVICE, TASK_ID, EVAL_LIMIT,
  PROPOSAL_TOP_K, CANDIDATE_CAP, MULTISEED_SEEDS, FORCE
EOF
}

print_plan() {
  cat <<EOF
STAGE_A_DOWNSTREAM_EVAL_QUEUE
artifact_kind=stage_a_downstream_eval_queue
ROOT=${ROOT}
PYTHON_BIN=${PYTHON_BIN}
SWEEP_DIR=${SWEEP_DIR}
OUT_DIR=${OUT_DIR}
STATUS_JSON=${STATUS_JSON}
PROGRESS_JSONL=${PROGRESS_JSONL}
LOG_PATH=${LOG_PATH}
MAX_LOADAVG=${MAX_LOADAVG}
WAIT_SECONDS=${WAIT_SECONDS}
DEVICE=${DEVICE}
TASK_ID=${TASK_ID}
EVAL_LIMIT=${EVAL_LIMIT}
PROPOSAL_TOP_K=${PROPOSAL_TOP_K}
CANDIDATE_CAP=${CANDIDATE_CAP}
MULTISEED_SEEDS=${MULTISEED_SEEDS}
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

mkdir -p "${OUT_DIR}" "$(dirname "${LOG_PATH}")"
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

summarize_sweep_status() {
  PYTHONPATH="$(dirname "${ROOT}")${PYTHONPATH:+:${PYTHONPATH}}" \
    "${PYTHON_BIN}" -m mrna_editflow.eval.summarize_stage_a_scalelaw_sweep \
      --sweep-dir "${SWEEP_DIR}" \
      --out-json "${SWEEP_DIR}/status.json" \
      --out-md "${SWEEP_DIR}/status.md" >/dev/null
}

write_status() {
  summarize_sweep_status || true
  PYTHONPATH="$(dirname "${ROOT}")${PYTHONPATH:+:${PYTHONPATH}}" \
    "${PYTHON_BIN}" - \
    "${ROOT}" "${SWEEP_DIR}" "${OUT_DIR}" "${STATUS_JSON}" "${STATUS_MD}" \
    "${PROGRESS_JSONL}" "${LOG_PATH}" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

root, sweep_dir, out_dir, status_json, status_md, progress_jsonl, log_path = sys.argv[1:8]

def sha(path):
    if not path or not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def load_json(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return payload if isinstance(payload, dict) else {}

def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]

sweep_status_path = os.path.join(sweep_dir, "status.json")
sweep = load_json(sweep_status_path)
summary = sweep.get("summary", {}) if isinstance(sweep.get("summary"), dict) else {}
runs = sweep.get("runs", []) if isinstance(sweep.get("runs"), list) else []
progress = read_jsonl(progress_jsonl)
last = progress[-1] if progress else {}

rows = []
for run in runs:
    if not isinstance(run, dict):
        continue
    run_id = str(run.get("run_id", ""))
    run_dir = os.path.join(out_dir, run_id)
    proposal_json = os.path.join(run_dir, "proposal_ranking_t5.json")
    proposal_jsonl = os.path.join(run_dir, "proposal_ranking_t5.candidates.jsonl")
    eval_dir = os.path.join(run_dir, "t1_t7_multiseed")
    eval_summary = os.path.join(run_dir, "t1_t7_eval_summary.json")
    runtime_audit = os.path.join(run_dir, "runtime_audit.json")
    training_complete = str(run.get("status")) == "complete" and bool(run.get("checkpoint_exists"))
    downstream_ready = all(os.path.exists(path) for path in (proposal_json, eval_summary, runtime_audit))
    rows.append({
        "run_id": run_id,
        "training_status": run.get("status"),
        "training_complete": training_complete,
        "checkpoint_exists": bool(run.get("checkpoint_exists")),
        "proposal_json": proposal_json,
        "proposal_exists": os.path.exists(proposal_json),
        "proposal_sha256": sha(proposal_json),
        "proposal_jsonl": proposal_jsonl,
        "proposal_jsonl_exists": os.path.exists(proposal_jsonl),
        "eval_dir": eval_dir,
        "eval_summary": eval_summary,
        "eval_summary_exists": os.path.exists(eval_summary),
        "eval_summary_sha256": sha(eval_summary),
        "runtime_audit": runtime_audit,
        "runtime_audit_exists": os.path.exists(runtime_audit),
        "downstream_ready": downstream_ready,
    })

n_runs = len(rows)
n_training_complete = sum(1 for row in rows if row["training_complete"])
n_downstream_ready = sum(1 for row in rows if row["downstream_ready"])
if n_runs == 0:
    status = "missing_stage_a_sweep"
elif n_training_complete < n_runs:
    status = "waiting_for_stage_a_completion"
elif n_downstream_ready < n_runs:
    status = "running_or_missing_downstream_eval"
else:
    status = "complete"

payload = {
    "artifact_kind": "stage_a_downstream_eval_queue_status",
    "claim_policy": (
        "Stage A downstream eval queue status is automation evidence only. "
        "Do not claim a true scale law until per-run downstream reports, "
        "aggregate report, and trend audit are complete."
    ),
    "status": status,
    "sweep_status": {
        "path": sweep_status_path,
        "sha256": sha(sweep_status_path),
        "summary": summary,
    },
    "summary": {
        "n_runs": n_runs,
        "n_training_complete": n_training_complete,
        "n_downstream_ready": n_downstream_ready,
        "ready_for_stage_a_downstream_eval_claim": False,
        "ready_for_true_scale_law_claim": False,
    },
    "progress": {
        "path": progress_jsonl,
        "n_events": len(progress),
        "last_event": last.get("event"),
        "last_loadavg": last.get("loadavg"),
    },
    "log": {"path": log_path, "exists": os.path.exists(log_path), "sha256": sha(log_path)},
    "runs": rows,
}
Path(status_json).parent.mkdir(parents=True, exist_ok=True)
with open(status_json, "w", encoding="utf-8") as fh:
    json.dump(payload, fh, indent=2, sort_keys=True)
lines = [
    "# Stage A Downstream Eval Queue Status",
    "",
    f"- Status: `{payload['status']}`",
    f"- Training complete: `{n_training_complete}/{n_runs}`; downstream ready: `{n_downstream_ready}/{n_runs}`",
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
    json_log load_gate_wait loadavg="${load}" max_loadavg="${MAX_LOADAVG}" wait_seconds="${WAIT_SECONDS}"
    write_status || true
    sleep "${WAIT_SECONDS}"
  done
}

stage_a_complete() {
  summarize_sweep_status || return 1
  "${PYTHON_BIN}" - "${SWEEP_DIR}/status.json" <<'PY'
import json
import sys
with open(sys.argv[1], "r", encoding="utf-8") as fh:
    payload = json.load(fh)
summary = payload.get("summary", {})
raise SystemExit(0 if summary.get("n_runs", 0) > 0 and summary.get("n_complete") == summary.get("n_runs") else 1)
PY
}

run_downstream() {
  PYTHONPATH="$(dirname "${ROOT}")${PYTHONPATH:+:${PYTHONPATH}}" \
    "${PYTHON_BIN}" - "${ROOT}" "${SWEEP_DIR}/status.json" "${OUT_DIR}" \
    "${TASK_ID}" "${EVAL_LIMIT}" "${PROPOSAL_TOP_K}" "${CANDIDATE_CAP}" \
    "${MULTISEED_SEEDS}" "${DEVICE}" "${FORCE}" "${LOG_PATH}" <<'PY'
import json
import os
import subprocess
import sys
import time

(
    root,
    status_path,
    out_dir,
    task_id,
    eval_limit,
    proposal_top_k,
    candidate_cap,
    multiseed_seeds,
    device,
    force,
    log_path,
) = sys.argv[1:12]
with open(status_path, "r", encoding="utf-8") as fh:
    status = json.load(fh)
runs = [row for row in status.get("runs", []) if isinstance(row, dict)]
plan = {}
plan_path = status.get("plan_path")
if isinstance(plan_path, str) and os.path.exists(plan_path):
    with open(plan_path, "r", encoding="utf-8") as fh:
        plan = json.load(fh)
plan_runs = {
    str(row.get("run_id")): row
    for row in plan.get("runs", [])
    if isinstance(row, dict) and row.get("run_id")
}
env = os.environ.copy()
env["PYTHONPATH"] = os.path.dirname(root) + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
os.makedirs(out_dir, exist_ok=True)
os.makedirs(os.path.dirname(log_path), exist_ok=True)

def run_cmd(cmd):
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"[{time.time()}] {' '.join(cmd)}\n")
        log.flush()
        subprocess.run(cmd, cwd=root, env=env, stdout=log, stderr=log, check=True)

for run in runs:
    run_id = str(run.get("run_id", ""))
    plan_run = plan_runs.get(run_id, {})
    if str(run.get("status")) != "complete" or not run.get("checkpoint_exists"):
        continue
    paths = run.get("paths", {}) if isinstance(run.get("paths"), dict) else {}
    checkpoint = str(paths.get("checkpoint") or plan_run.get("checkpoint_path") or "")
    records = str(plan_run.get("records_jsonl") or "")
    if not records:
        records = str(run.get("records_jsonl", ""))
    if not records and isinstance(status.get("plan"), dict):
        records = str(status["plan"].get("records_jsonl", ""))
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(checkpoint)
    if not records or not os.path.exists(records):
        records = str(run.get("records_jsonl") or "")
    if not records or not os.path.exists(records):
        raise FileNotFoundError(f"records_jsonl missing for {run_id}")
    run_dir = os.path.join(out_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    proposal_json = os.path.join(run_dir, "proposal_ranking_t5.json")
    proposal_jsonl = os.path.join(run_dir, "proposal_ranking_t5.candidates.jsonl")
    if force == "1" or not os.path.exists(proposal_json):
        run_cmd([
            sys.executable, "-m", "mrna_editflow.eval.proposal_ranking",
            "--records-jsonl", records,
            "--checkpoint", checkpoint,
            "--task-id", task_id,
            "--limit", str(eval_limit),
            "--candidate-cap", str(candidate_cap),
            "--top-k", str(proposal_top_k),
            "--device", device,
            "--out-json", proposal_json,
            "--out-jsonl", proposal_jsonl,
        ])
    eval_dir = os.path.join(run_dir, "t1_t7_multiseed")
    eval_summary = os.path.join(run_dir, "t1_t7_eval_summary.json")
    if force == "1" or not os.path.exists(eval_summary):
        seeds = [s for s in str(multiseed_seeds).split() if s.strip()]
        run_cmd([
            sys.executable, "-m", "mrna_editflow.eval.run_multiseed_benchmark",
            "--run-mode", "development",
            "--records-jsonl", records,
            "--checkpoint", checkpoint,
            "--out-dir", eval_dir,
            "--task-id", task_id,
            "--seeds", *seeds,
            "--limit", str(eval_limit),
            "--device", device,
            "--proposal-top-k", "8",
            "--resume",
        ])
        source_summary = os.path.join(eval_dir, "multiseed_summary.json")
        if os.path.exists(source_summary):
            with open(source_summary, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            payload["artifact_kind"] = "stage_a_scalelaw_downstream_t1_t7_eval"
            payload["source_summary_path"] = source_summary
            with open(eval_summary, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, sort_keys=True)
    runtime_audit = os.path.join(run_dir, "runtime_audit.json")
    if force == "1" or not os.path.exists(runtime_audit):
        payload = {
            "artifact_kind": "stage_a_scalelaw_downstream_runtime_audit",
            "run_id": run_id,
            "checkpoint": checkpoint,
            "records_jsonl": records,
            "proposal_json": proposal_json,
            "eval_summary": eval_summary,
            "time": time.time(),
            "claim_policy": "Runtime audit is downstream-eval execution metadata only, not a hardware benchmark.",
        }
        with open(runtime_audit, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
PY
}

json_log start sweep_dir="${SWEEP_DIR}" out_dir="${OUT_DIR}"
write_status || true

while ! stage_a_complete; do
  json_log wait_stage_a_completion wait_seconds="${WAIT_SECONDS}"
  write_status || true
  sleep "${WAIT_SECONDS}"
done

wait_load_gate
json_log downstream_start
set +e
run_downstream
rc=$?
set -e
json_log downstream_exit exit_code="${rc}"
write_status || true
exit "${rc}"
