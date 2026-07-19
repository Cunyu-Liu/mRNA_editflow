#!/usr/bin/env bash
# Launch a load-gated Stage A scale-law sweep.
#
# The sweep is deliberately modest by default. It creates controlled
# data-size x model-size x training-step runs, writes a manifest before launch,
# waits for the load gate, assigns free GPUs, and summarizes the resulting
# profiles/checkpoints. It does not lower the project load gate.
set -euo pipefail

ROOT="${ROOT:-/home/cunyuliu/mrna_editflow_goal/mrna_editflow}"
PYTHON_BIN="${PYTHON_BIN:-/home/cunyuliu/miniconda3/envs/editflow/bin/python3.10}"
BASE_CONFIG="${BASE_CONFIG:-${ROOT}/configs/stage_a_full_bs8_gradaccum4.json}"
RECORDS_JSONL="${RECORDS_JSONL:-${ROOT}/data/processed/gencode_human_transcripts.records.jsonl}"
SWEEP_ID="${SWEEP_ID:-stage_a_scalelaw_$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/benchmark/${SWEEP_ID}}"
CKPT_ROOT="${CKPT_ROOT:-${ROOT}/ckpts/${SWEEP_ID}}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/${SWEEP_ID}}"
DATA_SIZES="${DATA_SIZES:-256 1024}"
MODEL_SIZES="${MODEL_SIZES:-tiny small}"
STEP_COUNTS="${STEP_COUNTS:-200 500}"
SEEDS="${SEEDS:-0}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
MAX_LOADAVG="${MAX_LOADAVG:-80}"
MIN_FREE_MEM_MB="${MIN_FREE_MEM_MB:-18000}"
WAIT_SECONDS="${WAIT_SECONDS:-120}"
DEVICE="${DEVICE:-cuda}"
ALLOW_EXISTING="${ALLOW_EXISTING:-0}"
RESUME_EXISTING_PLAN="${RESUME_EXISTING_PLAN:-0}"
PLAN_TSV="${PLAN_TSV:-${OUT_ROOT}/plan.tsv}"
PLAN_JSON="${PLAN_JSON:-${OUT_ROOT}/plan.json}"
PROGRESS_JSONL="${PROGRESS_JSONL:-${OUT_ROOT}/progress.jsonl}"
SUMMARY_JSON="${SUMMARY_JSON:-${OUT_ROOT}/summary.json}"
SUMMARY_MD="${SUMMARY_MD:-${OUT_ROOT}/summary.md}"
STATUS_JSON="${STATUS_JSON:-${OUT_ROOT}/status.json}"
STATUS_MD="${STATUS_MD:-${OUT_ROOT}/status.md}"
GPU_LOCK_DIR="${GPU_LOCK_DIR:-${OUT_ROOT}/gpu_locks}"

usage() {
  cat <<'EOF'
Usage:
  run_stage_a_scalelaw_sweep.sh [--dry-run]

Purpose:
  Launch a controlled Stage A scale-law sweep over training data size, model
  size and training steps. This is the first real P3/P4 scale-law axis runner;
  it records manifests and waits for MAX_LOADAVG before using free GPUs.

Environment overrides:
  ROOT, PYTHON_BIN, BASE_CONFIG, RECORDS_JSONL, SWEEP_ID, OUT_ROOT, CKPT_ROOT,
  LOG_ROOT, DATA_SIZES, MODEL_SIZES, STEP_COUNTS, SEEDS, MAX_PARALLEL,
  MAX_LOADAVG, MIN_FREE_MEM_MB, WAIT_SECONDS, DEVICE, ALLOW_EXISTING,
  RESUME_EXISTING_PLAN, PLAN_TSV, PLAN_JSON, PROGRESS_JSONL, SUMMARY_JSON, SUMMARY_MD,
  STATUS_JSON, STATUS_MD

Default matrix:
  DATA_SIZES="256 1024"
  MODEL_SIZES="tiny small"
  STEP_COUNTS="200 500"
  SEEDS="0"
EOF
}

print_plan() {
  cat <<EOF
STAGE_A_SCALELAW_SWEEP
artifact_kind=stage_a_scalelaw_sweep
ROOT=${ROOT}
PYTHON_BIN=${PYTHON_BIN}
BASE_CONFIG=${BASE_CONFIG}
RECORDS_JSONL=${RECORDS_JSONL}
SWEEP_ID=${SWEEP_ID}
OUT_ROOT=${OUT_ROOT}
CKPT_ROOT=${CKPT_ROOT}
LOG_ROOT=${LOG_ROOT}
DATA_SIZES=${DATA_SIZES}
MODEL_SIZES=${MODEL_SIZES}
STEP_COUNTS=${STEP_COUNTS}
SEEDS=${SEEDS}
MAX_PARALLEL=${MAX_PARALLEL}
MAX_LOADAVG=${MAX_LOADAVG}
MIN_FREE_MEM_MB=${MIN_FREE_MEM_MB}
WAIT_SECONDS=${WAIT_SECONDS}
DEVICE=${DEVICE}
RESUME_EXISTING_PLAN=${RESUME_EXISTING_PLAN}
PLAN_JSON=${PLAN_JSON}
SUMMARY_JSON=${SUMMARY_JSON}
STATUS_JSON=${STATUS_JSON}
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

generate_plan() {
  mkdir -p "${OUT_ROOT}" "${CKPT_ROOT}" "${LOG_ROOT}"
  if [[ "${RESUME_EXISTING_PLAN}" == "1" && -s "${PLAN_JSON}" && -s "${PLAN_TSV}" ]]; then
    json_log resume_existing_plan plan_json="${PLAN_JSON}" plan_tsv="${PLAN_TSV}"
    write_status || true
    return 0
  fi
  "${PYTHON_BIN}" - "${BASE_CONFIG}" "${RECORDS_JSONL}" "${OUT_ROOT}" "${CKPT_ROOT}" "${LOG_ROOT}" \
    "${DATA_SIZES}" "${MODEL_SIZES}" "${STEP_COUNTS}" "${SEEDS}" "${PLAN_JSON}" "${PLAN_TSV}" <<'PY'
import hashlib
import json
import os
import socket
import sys
import time

(
    base_config,
    records_jsonl,
    out_root,
    ckpt_root,
    log_root,
    data_sizes_s,
    model_sizes_s,
    step_counts_s,
    seeds_s,
    plan_json,
    plan_tsv,
) = sys.argv[1:12]

with open(base_config, "r", encoding="utf-8") as fh:
    base = json.load(fh)
with open(records_jsonl, "rb") as fh:
    data = fh.read()
all_lines = [line for line in data.decode("utf-8").splitlines() if line.strip()]
dataset_sha = hashlib.sha256(data).hexdigest()
record_count = len(all_lines)

model_presets = {
    "tiny": {"model_dim": 192, "num_layers": 3, "num_heads": 4, "batch_size": 4, "grad_accum": 4},
    "small": {"model_dim": 256, "num_layers": 4, "num_heads": 4, "batch_size": 6, "grad_accum": 4},
    "base": {"model_dim": 384, "num_layers": 6, "num_heads": 8, "batch_size": 8, "grad_accum": 4},
}
data_sizes = [int(x) for x in data_sizes_s.split()]
model_sizes = model_sizes_s.split()
step_counts = [int(x) for x in step_counts_s.split()]
seeds = [int(x) for x in seeds_s.split()]
for model_size in model_sizes:
    if model_size not in model_presets:
        raise ValueError(f"unknown MODEL_SIZES entry: {model_size}")
for size in data_sizes:
    if size <= 0 or size > record_count:
        raise ValueError(f"invalid data size {size}; record_count={record_count}")

runs = []
for data_size in data_sizes:
    subset_rel = f"records/stage_a_scalelaw_data{data_size}.jsonl"
    subset_path = os.path.join(out_root, subset_rel)
    os.makedirs(os.path.dirname(subset_path), exist_ok=True)
    subset_text = "\n".join(all_lines[:data_size]) + "\n"
    with open(subset_path, "w", encoding="utf-8") as fh:
        fh.write(subset_text)
    subset_sha = hashlib.sha256(subset_text.encode("utf-8")).hexdigest()
    for model_size in model_sizes:
        preset = model_presets[model_size]
        for steps in step_counts:
            for seed in seeds:
                run_id = f"data{data_size}_{model_size}_steps{steps}_seed{seed}"
                cfg = json.loads(json.dumps(base))
                cfg["model"]["model_dim"] = preset["model_dim"]
                cfg["model"]["num_layers"] = preset["num_layers"]
                cfg["model"]["num_heads"] = preset["num_heads"]
                cfg["train"]["batch_size"] = preset["batch_size"]
                cfg["train"]["grad_accum"] = preset["grad_accum"]
                cfg["train"]["log_every"] = min(50, max(10, steps // 5))
                cfg["data"]["seed"] = seed
                save_dir = os.path.join(ckpt_root, run_id)
                profile_path = os.path.join(log_root, f"{run_id}.profile.jsonl")
                log_path = os.path.join(log_root, f"{run_id}.train.log")
                config_path = os.path.join(out_root, "configs", f"{run_id}.json")
                metadata_path = os.path.join(out_root, "metadata", f"{run_id}.json")
                cfg["train"]["save_dir"] = save_dir
                cfg["train"]["profile_path"] = profile_path
                os.makedirs(os.path.dirname(config_path), exist_ok=True)
                os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
                with open(config_path, "w", encoding="utf-8") as fh:
                    json.dump(cfg, fh, indent=2, sort_keys=True)
                metadata = {
                    "status": "planned",
                    "run_id": run_id,
                    "created_at": time.time(),
                    "hostname": socket.gethostname(),
                    "data_size": data_size,
                    "model_size": model_size,
                    "steps": steps,
                    "seed": seed,
                    "records_jsonl": subset_path,
                    "records_sha256": subset_sha,
                    "source_records_jsonl": records_jsonl,
                    "source_records_sha256": dataset_sha,
                    "source_record_count": record_count,
                    "config_path": config_path,
                    "save_dir": save_dir,
                    "checkpoint_path": os.path.join(save_dir, "stage_a_best.pt"),
                    "profile_path": profile_path,
                    "log_path": log_path,
                    "model_dim": preset["model_dim"],
                    "num_layers": preset["num_layers"],
                    "num_heads": preset["num_heads"],
                    "batch_size": preset["batch_size"],
                    "grad_accum": preset["grad_accum"],
                    "effective_batch_records": preset["batch_size"] * preset["grad_accum"],
                }
                with open(metadata_path, "w", encoding="utf-8") as fh:
                    json.dump(metadata, fh, indent=2, sort_keys=True)
                metadata["metadata_path"] = metadata_path
                runs.append(metadata)

payload = {
    "artifact_kind": "stage_a_scalelaw_sweep_plan",
    "created_at": time.time(),
    "base_config": base_config,
    "source_records_jsonl": records_jsonl,
    "source_records_sha256": dataset_sha,
    "source_record_count": record_count,
    "axes": {
        "data_sizes": data_sizes,
        "model_sizes": model_sizes,
        "step_counts": step_counts,
        "seeds": seeds,
    },
    "n_runs": len(runs),
    "runs": runs,
    "claim_policy": (
        "This sweep is controlled Stage A infrastructure evidence. Do not claim "
        "a monotonic scale law until all planned runs finish and downstream "
        "T1-T7/proposal-ranking audits are generated."
    ),
}
with open(plan_json, "w", encoding="utf-8") as fh:
    json.dump(payload, fh, indent=2, sort_keys=True)
with open(plan_tsv, "w", encoding="utf-8") as fh:
    fields = [
        "run_id",
        "data_size",
        "model_size",
        "steps",
        "seed",
        "config_path",
        "records_jsonl",
        "save_dir",
        "profile_path",
        "log_path",
        "metadata_path",
    ]
    fh.write("\t".join(fields) + "\n")
    for run in runs:
        fh.write("\t".join(str(run[field]) for field in fields) + "\n")
print(json.dumps({"plan_json": plan_json, "plan_tsv": plan_tsv, "n_runs": len(runs)}, sort_keys=True))
PY
  : > "${PROGRESS_JSONL}"
  json_log plan_ready plan_json="${PLAN_JSON}" n_runs="$(($(wc -l < "${PLAN_TSV}") - 1))"
  write_status || true
}

write_status() {
  if [[ ! -s "${PLAN_JSON}" ]]; then
    return 0
  fi
  PYTHONPATH="$(dirname "${ROOT}")${PYTHONPATH:+:${PYTHONPATH}}" \
    "${PYTHON_BIN}" -m mrna_editflow.eval.summarize_stage_a_scalelaw_sweep \
      --sweep-dir "${OUT_ROOT}" \
      --out-json "${STATUS_JSON}" \
      --out-md "${STATUS_MD}" \
    >/dev/null
}

wait_load_gate() {
  while true; do
    local load
    load="$(awk '{print $1}' /proc/loadavg 2>/dev/null || echo 0)"
    if "${PYTHON_BIN}" - "$load" "$MAX_LOADAVG" <<'PY'
import sys
load = float(sys.argv[1])
limit = float(sys.argv[2])
raise SystemExit(0 if load < limit else 1)
PY
    then
      json_log load_gate_pass loadavg="${load}" max_loadavg="${MAX_LOADAVG}"
      return 0
    fi
    echo "[$(date -Is)] loadavg ${load} >= ${MAX_LOADAVG}; waiting ${WAIT_SECONDS}s"
    json_log load_gate_wait loadavg="${load}" max_loadavg="${MAX_LOADAVG}" wait_seconds="${WAIT_SECONDS}"
    write_status || true
    sleep "${WAIT_SECONDS}"
  done
}

select_gpu() {
  if [[ "${DEVICE}" != "cuda" ]]; then
    echo ""
    return 0
  fi
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi not found for DEVICE=cuda" >&2
    return 1
  fi
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | awk -F',' -v min="${MIN_FREE_MEM_MB}" '
        {
          gsub(/ /, "", $1);
          gsub(/ /, "", $2);
          if (($2 + 0) >= min) print $1 "\t" ($2 + 0);
        }
      ' \
    | sort -k2,2nr \
    | while IFS=$'\t' read -r idx _free; do
        if [[ ! -e "${GPU_LOCK_DIR}/gpu_${idx}.lock" ]]; then
          echo "${idx}"
          break
        fi
      done
}

launch_one() {
  local run_id="$1" steps="$2" config_path="$3" records_jsonl="$4" save_dir="$5" profile_path="$6" log_path="$7" metadata_path="$8"
  if [[ "${ALLOW_EXISTING}" != "1" && ( -e "${save_dir}/stage_a_best.pt" || -e "${profile_path}" || -e "${log_path}" ) ]]; then
    echo "Skipping existing artifacts for ${run_id}; set ALLOW_EXISTING=1 to overwrite." >&2
    json_log skip_existing_artifacts run_id="${run_id}" save_dir="${save_dir}" profile_path="${profile_path}" log_path="${log_path}"
    write_status || true
    LAST_LAUNCHED_PID=""
    return 0
  fi
  wait_load_gate
  local gpu
  gpu="$(select_gpu)"
  if [[ "${DEVICE}" == "cuda" && -z "${gpu}" ]]; then
    echo "No GPU has >= ${MIN_FREE_MEM_MB} MiB free; waiting ${WAIT_SECONDS}s"
    json_log gpu_wait run_id="${run_id}" min_free_mem_mb="${MIN_FREE_MEM_MB}"
    sleep "${WAIT_SECONDS}"
    wait_load_gate
    gpu="$(select_gpu)"
    if [[ -z "${gpu}" ]]; then
      echo "No suitable GPU after retry for ${run_id}" >&2
      return 1
    fi
  fi
  json_log launch run_id="${run_id}" gpu="${gpu:-cpu}" steps="${steps}" config_path="${config_path}" records_jsonl="${records_jsonl}"
  write_status || true
  mkdir -p "${GPU_LOCK_DIR}"
  if [[ "${DEVICE}" == "cuda" ]]; then
    echo "${run_id}" > "${GPU_LOCK_DIR}/gpu_${gpu}.lock"
  fi
  (
    if [[ "${DEVICE}" == "cuda" ]]; then
      trap 'rm -f "'"${GPU_LOCK_DIR}/gpu_${gpu}.lock"'"' EXIT
    fi
    export PYTHONPATH="$(dirname "${ROOT}")${PYTHONPATH:+:${PYTHONPATH}}"
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
    if [[ "${DEVICE}" == "cuda" ]]; then
      export CUDA_VISIBLE_DEVICES="${gpu}"
    fi
    "${PYTHON_BIN}" -m mrna_editflow.train_backbone \
      --run-mode development \
      --config "${config_path}" \
      --records-jsonl "${records_jsonl}" \
      --steps "${steps}" \
      --save-dir "${save_dir}" \
      --profile-path "${profile_path}" \
      --device "${DEVICE}" \
      > "${log_path}" 2>&1
    "${PYTHON_BIN}" - "${metadata_path}" "${gpu:-cpu}" <<'PY'
import json
import sys
import time

path, gpu = sys.argv[1:3]
with open(path, "r", encoding="utf-8") as fh:
    payload = json.load(fh)
payload.update({"status": "train_command_exited", "finished_at": time.time(), "gpu": gpu})
with open(path, "w", encoding="utf-8") as fh:
    json.dump(payload, fh, indent=2, sort_keys=True)
PY
  ) &
  local pid=$!
  echo "[$(date -Is)] launched ${run_id} pid=${pid} gpu=${gpu:-cpu}"
  json_log launched run_id="${run_id}" pid="${pid}" gpu="${gpu:-cpu}"
  LAST_LAUNCHED_PID="${pid}"
}

prune_pids() {
  local alive=()
  local pid
  for pid in "${RUNNING_PIDS[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      alive+=("${pid}")
    else
      wait "${pid}" || true
    fi
  done
  RUNNING_PIDS=("${alive[@]}")
}

wait_for_slot() {
  while true; do
    prune_pids
    if (( ${#RUNNING_PIDS[@]} < MAX_PARALLEL )); then
      return 0
    fi
    wait "${RUNNING_PIDS[0]}" || true
    prune_pids
    summarize || true
  done
}

summarize() {
  "${PYTHON_BIN}" - "${PLAN_JSON}" "${SUMMARY_JSON}" "${SUMMARY_MD}" <<'PY'
import json
import math
import os
import sys
import time

plan_json, summary_json, summary_md = sys.argv[1:4]
with open(plan_json, "r", encoding="utf-8") as fh:
    plan = json.load(fh)
rows = []
for run in plan["runs"]:
    profile = run["profile_path"]
    ckpt = run["checkpoint_path"]
    log_path = run["log_path"]
    last = None
    best_loss = None
    n_profile_rows = 0
    if os.path.exists(profile):
        with open(profile, "r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                n_profile_rows += 1
                row = json.loads(line)
                last = row
                loss = row.get("loss")
                if isinstance(loss, (int, float)) and math.isfinite(float(loss)):
                    best_loss = float(loss) if best_loss is None else min(best_loss, float(loss))
    status = "complete" if os.path.exists(ckpt) and n_profile_rows >= int(run["steps"]) else "running_or_incomplete"
    rows.append({
        "run_id": run["run_id"],
        "data_size": run["data_size"],
        "model_size": run["model_size"],
        "steps": run["steps"],
        "seed": run["seed"],
        "status": status,
        "checkpoint_exists": os.path.exists(ckpt),
        "profile_exists": os.path.exists(profile),
        "n_profile_rows": n_profile_rows,
        "best_loss": best_loss,
        "last_loss": last.get("loss") if isinstance(last, dict) else None,
        "last_samples_per_s": last.get("samples_per_s") if isinstance(last, dict) else None,
        "log_path": log_path,
        "profile_path": profile,
        "checkpoint_path": ckpt,
    })
payload = {
    "artifact_kind": "stage_a_scalelaw_sweep_summary",
    "created_at": time.time(),
    "plan_json": plan_json,
    "n_runs": len(rows),
    "n_complete": sum(1 for row in rows if row["status"] == "complete"),
    "n_incomplete": sum(1 for row in rows if row["status"] != "complete"),
    "claim_policy": plan["claim_policy"],
    "rows": rows,
}
os.makedirs(os.path.dirname(summary_json), exist_ok=True)
with open(summary_json, "w", encoding="utf-8") as fh:
    json.dump(payload, fh, indent=2, sort_keys=True)
lines = [
    "# Stage A Scale-Law Sweep Summary",
    "",
    f"- Plan: `{plan_json}`",
    f"- Complete/incomplete: `{payload['n_complete']}/{payload['n_incomplete']}`",
    f"- Claim policy: {payload['claim_policy']}",
    "",
    "| run | data | model | steps | status | best loss | profile rows | samples/s |",
    "|---|---:|---|---:|---|---:|---:|---:|",
]
for row in rows:
    lines.append(
        f"| `{row['run_id']}` | {row['data_size']} | {row['model_size']} | {row['steps']} | "
        f"{row['status']} | {row['best_loss']} | {row['n_profile_rows']} | {row['last_samples_per_s']} |"
    )
with open(summary_md, "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines) + "\n")
print(json.dumps({"summary_json": summary_json, "summary_md": summary_md, "n_complete": payload["n_complete"], "n_incomplete": payload["n_incomplete"]}, sort_keys=True))
PY
  write_status || true
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
if [[ ! -s "${BASE_CONFIG}" ]]; then
  echo "Missing base config: ${BASE_CONFIG}" >&2
  exit 1
fi
if [[ ! -s "${RECORDS_JSONL}" ]]; then
  echo "Missing records JSONL: ${RECORDS_JSONL}" >&2
  exit 1
fi

generate_plan
mkdir -p "${GPU_LOCK_DIR}"
declare -a RUNNING_PIDS=()
LAST_LAUNCHED_PID=""
while IFS=$'\t' read -r run_id _data_size _model_size steps _seed config_path records_jsonl save_dir profile_path log_path metadata_path; do
  wait_for_slot
  LAST_LAUNCHED_PID=""
  launch_one "${run_id}" "${steps}" "${config_path}" "${records_jsonl}" "${save_dir}" "${profile_path}" "${log_path}" "${metadata_path}"
  if [[ -n "${LAST_LAUNCHED_PID}" ]]; then
    RUNNING_PIDS+=("${LAST_LAUNCHED_PID}")
  fi
done < <(tail -n +2 "${PLAN_TSV}")
if (( ${#RUNNING_PIDS[@]} > 0 )); then
  for pid in "${RUNNING_PIDS[@]}"; do
    wait "${pid}" || true
  done
fi
summarize
