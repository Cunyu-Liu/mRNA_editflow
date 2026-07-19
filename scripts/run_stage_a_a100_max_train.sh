#!/usr/bin/env bash
# Load-gated A100 max-parameter Stage A training launcher.
#
# This is the current "best measured architecture switches, scaled up" run:
# region FiLM + codon constraint + RoPE + aux structure, using the largest
# currently available public corpus (GENCODE cleaned records) and an A100-sized
# head config.
set -euo pipefail

ROOT="${ROOT:-/home/cunyuliu/mrna_editflow_goal/mrna_editflow}"
PYTHON_BIN="${PYTHON_BIN:-/home/cunyuliu/miniconda3/envs/editflow/bin/python3.10}"
CONFIG="${CONFIG:-${ROOT}/configs/stage_a_full_a100_max.json}"
RECORDS_JSONL="${RECORDS_JSONL:-${ROOT}/data/processed/gencode_human_transcripts.records.jsonl}"
STEPS="${STEPS:-100000}"
SEED="${SEED:-0}"
RUN_NAME="${RUN_NAME:-stage_a_full_a100_max_gencode_100k_seed${SEED}}"
SAVE_DIR="${SAVE_DIR:-${ROOT}/ckpts/${RUN_NAME}}"
PROFILE_PATH="${PROFILE_PATH:-${ROOT}/logs/${RUN_NAME}.profile.jsonl}"
LOG_PATH="${LOG_PATH:-${ROOT}/logs/${RUN_NAME}.train.log}"
METADATA_JSON="${METADATA_JSON:-${ROOT}/logs/${RUN_NAME}.metadata.json}"
STATUS_JSON="${STATUS_JSON:-${ROOT}/benchmark/${RUN_NAME}/status.json}"
STATUS_MD="${STATUS_MD:-${ROOT}/benchmark/${RUN_NAME}/status.md}"
PROGRESS_JSONL="${PROGRESS_JSONL:-${ROOT}/benchmark/${RUN_NAME}/progress.jsonl}"
MAX_LOADAVG="${MAX_LOADAVG:-80}"
WAIT_SECONDS="${WAIT_SECONDS:-120}"
MIN_FREE_MEM_MB="${MIN_FREE_MEM_MB:-35000}"
ALLOW_NON_A100="${ALLOW_NON_A100:-0}"
REQUESTED_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-auto}"
ALLOW_EXISTING="${ALLOW_EXISTING:-0}"

usage() {
  cat <<'EOF'
Usage:
  run_stage_a_a100_max_train.sh [--dry-run]

Purpose:
  Queue the largest current Stage A training run on the freest A100 while
  preserving MAX_LOADAVG and GPU-memory gates. The script writes metadata,
  progress and status artifacts before training starts, during waiting, and
  after completion/failure.

Environment overrides:
  ROOT, PYTHON_BIN, CONFIG, RECORDS_JSONL, STEPS, SEED, RUN_NAME, SAVE_DIR,
  PROFILE_PATH, LOG_PATH, METADATA_JSON, STATUS_JSON, STATUS_MD,
  PROGRESS_JSONL, MAX_LOADAVG, WAIT_SECONDS, MIN_FREE_MEM_MB,
  CUDA_VISIBLE_DEVICES, ALLOW_NON_A100, ALLOW_EXISTING
EOF
}

print_plan() {
  cat <<EOF
STAGE_A_A100_MAX_TRAIN
artifact_kind=stage_a_a100_max_train
ROOT=${ROOT}
PYTHON_BIN=${PYTHON_BIN}
CONFIG=${CONFIG}
RECORDS_JSONL=${RECORDS_JSONL}
STEPS=${STEPS}
SEED=${SEED}
RUN_NAME=${RUN_NAME}
SAVE_DIR=${SAVE_DIR}
PROFILE_PATH=${PROFILE_PATH}
LOG_PATH=${LOG_PATH}
METADATA_JSON=${METADATA_JSON}
STATUS_JSON=${STATUS_JSON}
STATUS_MD=${STATUS_MD}
PROGRESS_JSONL=${PROGRESS_JSONL}
MAX_LOADAVG=${MAX_LOADAVG}
WAIT_SECONDS=${WAIT_SECONDS}
MIN_FREE_MEM_MB=${MIN_FREE_MEM_MB}
CUDA_VISIBLE_DEVICES=${REQUESTED_CUDA_VISIBLE_DEVICES}
ALLOW_NON_A100=${ALLOW_NON_A100}
ALLOW_EXISTING=${ALLOW_EXISTING}

Command:
  CUDA_VISIBLE_DEVICES=<selected A100> PYTHONPATH="\$(dirname "${ROOT}")" "${PYTHON_BIN}" -m mrna_editflow.train_backbone \\
    --config "${CONFIG}" \\
    --records-jsonl "${RECORDS_JSONL}" \\
    --steps "${STEPS}" \\
    --save-dir "${SAVE_DIR}" \\
    --profile-path "${PROFILE_PATH}" \\
    --device cuda \\
    --seed "${SEED}"
EOF
}

json_log() {
  local event="$1"
  shift || true
  mkdir -p "$(dirname "${PROGRESS_JSONL}")"
  "${PYTHON_BIN}" - "$event" "$@" >> "${PROGRESS_JSONL}" <<'PY'
import json
import sys
import time

event = sys.argv[1]
payload = {"event": event, "time": time.time()}
for item in sys.argv[2:]:
    if "=" in item:
        key, value = item.split("=", 1)
        payload[key] = value
print(json.dumps(payload, sort_keys=True))
PY
}

write_status() {
  local status="$1"
  local selected_gpu="${2:-}"
  local note="${3:-}"
  mkdir -p "$(dirname "${STATUS_JSON}")"
  PYTHONPATH="$(dirname "${ROOT}")${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON_BIN}" - \
    "${STATUS_JSON}" "${STATUS_MD}" "${status}" "${selected_gpu}" "${note}" \
    "${ROOT}" "${RUN_NAME}" "${MAX_LOADAVG}" "${MIN_FREE_MEM_MB}" "${STEPS}" "${SEED}" \
    "${CONFIG}" "${RECORDS_JSONL}" "${SAVE_DIR}" "${PROFILE_PATH}" "${LOG_PATH}" \
    "${METADATA_JSON}" "${PROGRESS_JSONL}" <<'PY'
import hashlib
import json
import os
import sys
import time

(
    status_json,
    status_md,
    status,
    selected_gpu,
    note,
    root,
    run_name,
    max_loadavg,
    min_free_mem_mb,
    steps,
    seed,
    config,
    records,
    save_dir,
    profile,
    log_path,
    metadata_json,
    progress,
) = sys.argv[1:19]
ckpt = os.path.join(save_dir, "stage_a_best.pt")

def sha(path):
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def count_lines(path):
    if not os.path.exists(path):
        return 0
    with open(path, "r", encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())

payload = {
    "artifact_kind": "stage_a_a100_max_train_status",
    "claim_policy": (
        "This is a max-scale Stage A training run using current best validated "
        "architecture switches. It is training infrastructure evidence only "
        "until downstream proposal-ranking and T1-T7 audits are complete."
    ),
    "status": status,
    "note": note,
    "time": time.time(),
    "root": root,
    "run_name": run_name,
    "selected_gpu": selected_gpu or None,
    "max_loadavg": float(max_loadavg),
    "min_free_mem_mb": int(min_free_mem_mb),
    "steps_target": int(steps),
    "seed": int(seed),
    "config": {"path": config, "sha256": sha(config)},
    "records": {
        "path": records,
        "exists": os.path.exists(records),
        "sha256": sha(records),
        "n_records": count_lines(records),
    },
    "artifacts": {
        "checkpoint": {"path": ckpt, "exists": os.path.exists(ckpt), "sha256": sha(ckpt)},
        "profile": {"path": profile, "exists": os.path.exists(profile), "n_rows": count_lines(profile), "sha256": sha(profile)},
        "log": {"path": log_path, "exists": os.path.exists(log_path), "sha256": sha(log_path)},
        "metadata": {"path": metadata_json, "exists": os.path.exists(metadata_json), "sha256": sha(metadata_json)},
        "progress": {"path": progress, "exists": os.path.exists(progress), "n_rows": count_lines(progress), "sha256": sha(progress)},
    },
    "ready_for_training_claim": False,
    "ready_for_downstream_claim": False,
}
with open(status_json, "w", encoding="utf-8") as fh:
    json.dump(payload, fh, indent=2, sort_keys=True)
with open(status_md, "w", encoding="utf-8") as fh:
    fh.write("# Stage A A100 Max Train Status\\n\\n")
    fh.write(f"- Status: `{status}`\\n")
    fh.write(f"- Run: `{payload['run_name']}`\\n")
    fh.write(f"- Selected GPU: `{payload['selected_gpu']}`\\n")
    fh.write(f"- Records: `{payload['records']['n_records']}`; SHA `{payload['records']['sha256']}`\\n")
    fh.write(f"- Profile rows: `{payload['artifacts']['profile']['n_rows']}`\\n")
    fh.write(f"- Checkpoint exists: `{payload['artifacts']['checkpoint']['exists']}`\\n")
    fh.write(f"- Claim policy: {payload['claim_policy']}\\n")
PY
}

load_ok() {
  if [[ ! -r /proc/loadavg ]]; then
    return 0
  fi
  local load_value
  load_value="$(awk '{print $1}' /proc/loadavg)"
  awk -v lval="${load_value}" -v max="${MAX_LOADAVG}" 'BEGIN { exit !(lval < max) }'
}

current_load() {
  if [[ -r /proc/loadavg ]]; then
    awk '{print $1}' /proc/loadavg
  else
    echo "NA"
  fi
}

select_gpu() {
  if [[ "${REQUESTED_CUDA_VISIBLE_DEVICES}" != "auto" ]]; then
    echo "${REQUESTED_CUDA_VISIBLE_DEVICES}"
    return 0
  fi
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return 1
  fi
  local gpu_csv
  gpu_csv="$(nvidia-smi --query-gpu=index,name,memory.free,utilization.gpu --format=csv,noheader,nounits)"
  GPU_CSV="${gpu_csv}" "${PYTHON_BIN}" - "${MIN_FREE_MEM_MB}" "${ALLOW_NON_A100}" <<'PY'
import os
import sys

min_free = int(sys.argv[1])
allow_non_a100 = sys.argv[2] == "1"
best = None
for line in os.environ.get("GPU_CSV", "").splitlines():
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 4:
        continue
    idx, name, free, util = parts[:4]
    try:
        free_i = int(float(free))
        util_i = int(float(util))
    except ValueError:
        continue
    is_a100 = "A100" in name.upper()
    if not is_a100 and not allow_non_a100:
        continue
    if free_i < min_free:
        continue
    row = (free_i, -util_i, idx, name)
    if best is None or row > best:
        best = row
if best is None:
    raise SystemExit(1)
print(best[2])
PY
}

write_metadata() {
  mkdir -p "$(dirname "${METADATA_JSON}")"
  PYTHONPATH="$(dirname "${ROOT}")${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON_BIN}" - \
    "${METADATA_JSON}" "${SELECTED_GPU}" "${RECORDS_JSONL}" "${CONFIG}" "${RUN_NAME}" \
    "${STEPS}" "${SEED}" "${SAVE_DIR}" "${PROFILE_PATH}" "${LOG_PATH}" <<'PY'
import hashlib
import json
import os
import platform
import socket
import sys
import time

from mrna_editflow.core.config import MEFConfig
from mrna_editflow.train_backbone import build_stage_a_model

(
    out,
    selected_gpu,
    records,
    config_path,
    run_name,
    steps,
    seed,
    save_dir,
    profile_path,
    log_path,
) = sys.argv[1:11]

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

with open(records, "r", encoding="utf-8") as fh:
    n_records = sum(1 for line in fh if line.strip())
cfg = MEFConfig.from_json(config_path)
backbone, model = build_stage_a_model(cfg, device="cpu")
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in backbone.parameters())
payload = {
    "artifact_kind": "stage_a_a100_max_train_metadata",
    "time": time.time(),
    "run_name": run_name,
    "stage": "stage_a_a100_max",
    "selected_gpu": selected_gpu,
    "steps": int(steps),
    "seed": int(seed),
    "records_jsonl": records,
    "dataset_sha256": sha(records),
    "record_count": n_records,
    "config": config_path,
    "config_sha256": sha(config_path),
    "save_dir": save_dir,
    "profile_path": profile_path,
    "log_path": log_path,
    "trainable_model_params": trainable,
    "total_backbone_plus_model_params": total,
    "hostname": socket.gethostname(),
    "machine": platform.machine(),
    "platform": platform.platform(),
    "claim_policy": (
        "Max-parameter Stage A training metadata. Do not claim SOTA until "
        "proposal-ranking, T1-T7, external baselines and real-data audits pass."
    ),
}
with open(out, "w", encoding="utf-8") as fh:
    json.dump(payload, fh, indent=2, sort_keys=True)
print(json.dumps(payload, sort_keys=True))
PY
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then usage; exit 0; fi
if [[ "${1:-}" == "--dry-run" ]]; then print_plan; exit 0; fi

if [[ ! -x "${PYTHON_BIN}" ]]; then echo "Missing PYTHON_BIN=${PYTHON_BIN}" >&2; exit 1; fi
if [[ ! -s "${CONFIG}" ]]; then echo "Missing CONFIG=${CONFIG}" >&2; exit 1; fi
if [[ ! -s "${RECORDS_JSONL}" ]]; then echo "Missing RECORDS_JSONL=${RECORDS_JSONL}" >&2; exit 1; fi
if [[ "${ALLOW_EXISTING}" != "1" ]]; then
  if [[ -e "${SAVE_DIR}/stage_a_best.pt" || -e "${PROFILE_PATH}" || -e "${LOG_PATH}" ]]; then
    echo "Refusing to overwrite existing artifacts for RUN_NAME=${RUN_NAME}." >&2
    echo "Set ALLOW_EXISTING=1 or choose a new RUN_NAME." >&2
    exit 1
  fi
fi

mkdir -p "${SAVE_DIR}" "$(dirname "${PROFILE_PATH}")" "$(dirname "${LOG_PATH}")" "$(dirname "${PROGRESS_JSONL}")"
write_status "queued" "" "waiting for load and A100 memory gates"
json_log "queued" run_name="${RUN_NAME}" steps="${STEPS}"

SELECTED_GPU=""
while true; do
  if ! load_ok; then
    load="$(current_load)"
    write_status "waiting_load_gate" "" "loadavg=${load} >= MAX_LOADAVG=${MAX_LOADAVG}"
    json_log "load_gate_wait" loadavg="${load}" max_loadavg="${MAX_LOADAVG}" wait_seconds="${WAIT_SECONDS}"
    sleep "${WAIT_SECONDS}"
    continue
  fi
  if SELECTED_GPU="$(select_gpu)"; then
    break
  fi
  write_status "waiting_a100_memory" "" "no A100 with free memory >= ${MIN_FREE_MEM_MB} MiB"
  json_log "a100_memory_wait" min_free_mem_mb="${MIN_FREE_MEM_MB}" wait_seconds="${WAIT_SECONDS}"
  sleep "${WAIT_SECONDS}"
done

write_metadata
write_status "training_running" "${SELECTED_GPU}" "training command launched"
json_log "training_start" selected_gpu="${SELECTED_GPU}" config="${CONFIG}" records="${RECORDS_JSONL}"

export CUDA_VISIBLE_DEVICES="${SELECTED_GPU}"
export PYTHONPATH="$(dirname "${ROOT}")${PYTHONPATH:+:${PYTHONPATH}}"
set +e
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"${PYTHON_BIN}" -m mrna_editflow.train_backbone \
  --run-mode development \
  --config "${CONFIG}" \
  --records-jsonl "${RECORDS_JSONL}" \
  --steps "${STEPS}" \
  --save-dir "${SAVE_DIR}" \
  --profile-path "${PROFILE_PATH}" \
  --device cuda \
  --seed "${SEED}" \
  > "${LOG_PATH}" 2>&1
rc=$?
set -e

if [[ "${rc}" -eq 0 ]]; then
  json_log "training_complete" selected_gpu="${SELECTED_GPU}" exit_code="${rc}"
  write_status "training_complete" "${SELECTED_GPU}" "train_backbone exited 0"
else
  json_log "training_failed" selected_gpu="${SELECTED_GPU}" exit_code="${rc}"
  write_status "training_failed" "${SELECTED_GPU}" "train_backbone exit_code=${rc}"
fi
exit "${rc}"
