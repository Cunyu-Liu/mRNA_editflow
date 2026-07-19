#!/usr/bin/env bash
# Run the MEF UTR-specific adapter with a hard 5'UTR-only edit canvas.
set -euo pipefail

ROOT="${ROOT:-/home/cunyuliu/mrna_editflow_goal/mrna_editflow}"
PYTHON_BIN="${PYTHON_BIN:-/home/cunyuliu/miniconda3/envs/editflow/bin/python3.10}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
SOURCES="${SOURCES:-${ROOT}/benchmark/multiseed_t5_public_head1024_sources.jsonl}"
CHECKPOINT="${CHECKPOINT:-${ROOT}/ckpts/region_adapter_t5_utr5_head1024/stage_b_region_t5_best.pt}"
OUT_DIR="${OUT_DIR:-${ROOT}/benchmark/multiseed_t5_public_head1024_region_adapter_utr5only_top64}"
STATUS_JSON="${STATUS_JSON:-${OUT_DIR}/status.json}"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
LIMIT="${LIMIT:-1024}"
TOP_K="${TOP_K:-64}"
EDIT_BUDGET="${EDIT_BUDGET:-3}"
N_BOOTSTRAP="${N_BOOTSTRAP:-1000}"
MAX_NOVELTY_SOURCES="${MAX_NOVELTY_SOURCES:-0}"
SHARDED="${SHARDED:-1}"
SHARD_GPUS="${SHARD_GPUS:-0 2}"
SHARD_A_SEEDS="${SHARD_A_SEEDS:-0 1 2 3 4}"
SHARD_B_SEEDS="${SHARD_B_SEEDS:-5 6 7 8 9}"
REFRESH_REPORTS="${REFRESH_REPORTS:-1}"

usage() {
  cat <<'EOF'
Usage:
  run_mef_utr5only_head1024.sh [--dry-run]

Runs the UTR-specific Stage B checkpoint under a hard candidate-generation
constraint: only 5'UTR substitutions are enumerated. CDS and 3'UTR positions
are absent from the legal candidate set. The default run uses head1024,
10 seeds, edit budget 3, and top64 decoding.

Environment overrides:
  ROOT, PYTHON_BIN, CUDA_VISIBLE_DEVICES, SOURCES, CHECKPOINT, OUT_DIR,
  STATUS_JSON, SEEDS, LIMIT, TOP_K, EDIT_BUDGET, N_BOOTSTRAP,
  MAX_NOVELTY_SOURCES, SHARDED, SHARD_GPUS, SHARD_A_SEEDS, SHARD_B_SEEDS,
  REFRESH_REPORTS
EOF
}

write_status() {
  local status="$1"
  local message="$2"
  STATUS="${status}" MESSAGE="${message}" STATUS_JSON="${STATUS_JSON}" \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" LIMIT="${LIMIT}" \
    TOP_K="${TOP_K}" EDIT_BUDGET="${EDIT_BUDGET}" \
    "${PYTHON_BIN}" - <<'PY'
import json
import os
import time

payload = {
    "artifact_kind": "mef_utr5only_head1024_status",
    "time": time.time(),
    "status": os.environ["STATUS"],
    "message": os.environ["MESSAGE"],
    "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
    "limit": int(os.environ["LIMIT"]),
    "top_k": int(os.environ["TOP_K"]),
    "edit_budget": int(os.environ["EDIT_BUDGET"]),
    "editable_regions": ["utr5"],
}
path = os.environ["STATUS_JSON"]
os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
with open(path, "w", encoding="utf-8") as fh:
    json.dump(payload, fh, indent=2, sort_keys=True)
PY
}

print_plan() {
  echo "MEF UTR5-ONLY HEAD1024"
  echo "ROOT=${ROOT}"
  echo "SOURCES=${SOURCES}"
  echo "CHECKPOINT=${CHECKPOINT}"
  echo "OUT_DIR=${OUT_DIR}"
  echo "GPU=${CUDA_VISIBLE_DEVICES}"
  echo "SEEDS=${SEEDS}"
  echo "LIMIT=${LIMIT} TOP_K=${TOP_K} EDIT_BUDGET=${EDIT_BUDGET}"
  echo "EDITABLE_REGIONS=utr5"
  echo "SHARDED=${SHARDED} SHARD_GPUS=${SHARD_GPUS}"
  echo "SHARD_A_SEEDS=${SHARD_A_SEEDS}"
  echo "SHARD_B_SEEDS=${SHARD_B_SEEDS}"
  echo "REFRESH_REPORTS=${REFRESH_REPORTS}"
  echo "STATUS_JSON=${STATUS_JSON}"
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then usage; exit 0; fi
if [[ "${1:-}" == "--dry-run" ]]; then print_plan; exit 0; fi

export CUDA_VISIBLE_DEVICES
export PYTHONPATH="$(dirname "${ROOT}")${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${OUT_DIR}"

on_error() {
  local code=$?
  write_status "failed" "MEF UTR5-only benchmark or report refresh failed with exit ${code}"
  exit "${code}"
}
trap on_error ERR
write_status "running" "MEF UTR5-only 10-seed head1024 benchmark is running"

run_benchmark() {
  local out_dir="$1"
  local gpu="$2"
  shift 2
  CUDA_VISIBLE_DEVICES="${gpu}" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "${PYTHON_BIN}" -m mrna_editflow.eval.run_multiseed_benchmark \
    --run-mode development \
    --records-jsonl "${SOURCES}" \
    --checkpoint "${CHECKPOINT}" \
    --task-id T5 \
    --editable-regions utr5 \
    --edit-budget "${EDIT_BUDGET}" \
    --proposal-top-k "${TOP_K}" \
    --proposal-temperature 1.0 \
    --limit "${LIMIT}" \
    --device cuda \
    --seeds "$@" \
    --n-bootstrap "${N_BOOTSTRAP}" \
    --max-novelty-sources "${MAX_NOVELTY_SOURCES}" \
    --resume \
    --out-dir "${out_dir}"
}

if [[ "${SHARDED}" == "1" ]]; then
  read -r -a shard_gpus <<< "${SHARD_GPUS}"
  if [[ "${#shard_gpus[@]}" -lt 2 ]]; then
    echo "SHARD_GPUS must contain at least two GPU indices" >&2
    exit 2
  fi
  read -r -a shard_a_seeds <<< "${SHARD_A_SEEDS}"
  read -r -a shard_b_seeds <<< "${SHARD_B_SEEDS}"
  shard_a="${OUT_DIR}/shard_0_4"
  shard_b="${OUT_DIR}/shard_5_9"
  run_benchmark "${shard_a}" "${shard_gpus[0]}" "${shard_a_seeds[@]}" &
  pid_a=$!
  run_benchmark "${shard_b}" "${shard_gpus[1]}" "${shard_b_seeds[@]}" &
  pid_b=$!
  wait "${pid_a}"
  wait "${pid_b}"
  "${PYTHON_BIN}" -m mrna_editflow.eval.merge_multiseed_shards \
    --source-dir "${shard_a}" \
    --source-dir "${shard_b}" \
    --out-dir "${OUT_DIR}" \
    --expected-seeds ${SEEDS} \
    --source-path "${SOURCES}" \
    --task-id T5 \
    --checkpoint "${CHECKPOINT}" \
    --edit-budget "${EDIT_BUDGET}" \
    --proposal-top-k "${TOP_K}" \
    --proposal-temperature 1.0 \
    --editable-regions utr5 \
    --n-bootstrap "${N_BOOTSTRAP}" \
    --max-novelty-sources "${MAX_NOVELTY_SOURCES}"
else
  read -r -a seeds <<< "${SEEDS}"
  run_benchmark "${OUT_DIR}" "${CUDA_VISIBLE_DEVICES}" "${seeds[@]}"
fi

if [[ "${REFRESH_REPORTS}" == "1" ]]; then
  "${PYTHON_BIN}" -m mrna_editflow.eval.build_t5_external_utr_comparison \
    --project-root "${ROOT}" \
    --out-json docs/t5_external_utr_baseline_comparison.json \
    --out-md docs/t5_external_utr_baseline_comparison.md

  "${PYTHON_BIN}" -m mrna_editflow.eval.audit_sota_readiness \
    --project-root "${ROOT}" \
    --slice head256 \
    --top-k 64 \
    --out-json docs/sota_readiness_audit_head256.json \
    --out-md docs/sota_readiness_audit_head256.md

  "${PYTHON_BIN}" -m mrna_editflow.eval.sota_gap_report \
    --project-root "${ROOT}" \
    --out-json docs/sota_gap_report.json \
    --out-md docs/sota_gap_report.md
fi

write_status "complete" "MEF UTR5-only 10-seed head1024 benchmark completed"
trap - ERR
