#!/usr/bin/env bash
# Post-training evaluation queue for A100 max Stage A runs.
#
# Waits for one Stage A max training process, snapshots its best checkpoint,
# runs proposal-ranking, distills a TE ranker, runs T5 multiseed evaluation,
# compares against current controls, and generates the sequence-spectrum audit.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-/home/cunyuliu/mrna_editflow_goal/mrna_editflow}"
PYTHON_BIN="${PYTHON_BIN:-/home/cunyuliu/miniconda3/envs/editflow/bin/python3.10}"
SEED="${SEED:-0}"
RUN_NAME="${RUN_NAME:-stage_a_full_a100_max_gencode_100k_seed${SEED}}"
STAGE_A_PID="${STAGE_A_PID:-}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
POLL_SECONDS="${POLL_SECONDS:-300}"
LIMIT="${LIMIT:-1024}"
TOP_K="${TOP_K:-64}"
PROPOSAL_TOP_K="${PROPOSAL_TOP_K:-64}"
RANKER_STEPS="${RANKER_STEPS:-500}"
RANKER_AUDIT_LIMIT="${RANKER_AUDIT_LIMIT:-64}"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
N_BOOTSTRAP="${N_BOOTSTRAP:-1000}"
N_PERMUTATIONS="${N_PERMUTATIONS:-2000}"
EDIT_BUDGET="${EDIT_BUDGET:-3}"
RECORDS_JSONL="${RECORDS_JSONL:-${ROOT}/data/processed/gencode_human_transcripts.records.jsonl}"
STAGE_A_DIR="${STAGE_A_DIR:-${ROOT}/ckpts/${RUN_NAME}}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-${STAGE_A_DIR}/stage_a_best.pt}"
SNAPSHOT="${SNAPSHOT:-${STAGE_A_DIR}/stage_a_best_a100max_final_for_proposal_ranker.pt}"
POST_ROOT="${POST_ROOT:-${ROOT}/benchmark/${RUN_NAME}_posteval}"
STATUS_JSON="${STATUS_JSON:-${POST_ROOT}/status.json}"
STATUS_MD="${STATUS_MD:-${POST_ROOT}/status.md}"
PROGRESS_JSONL="${PROGRESS_JSONL:-${POST_ROOT}/progress.jsonl}"

AUDIT_JSON="${AUDIT_JSON:-${POST_ROOT}/proposal_ranking_t5_head${LIMIT}.json}"
AUDIT_JSONL="${AUDIT_JSONL:-${POST_ROOT}/proposal_ranking_t5_head${LIMIT}.candidates.jsonl}"
BASE_AUDIT_JSON="${BASE_AUDIT_JSON:-${POST_ROOT}/proposal_ranking_t5_base_head${RANKER_AUDIT_LIMIT}.json}"
BASE_AUDIT_JSONL="${BASE_AUDIT_JSONL:-${POST_ROOT}/proposal_ranking_t5_base_head${RANKER_AUDIT_LIMIT}.candidates.jsonl}"
RANKER_DIR="${RANKER_DIR:-${ROOT}/ckpts/proposal_ranker_t5_${RUN_NAME}_head${LIMIT}_teacher}"
RANKER_PROFILE="${RANKER_PROFILE:-${ROOT}/logs/proposal_ranker_t5_${RUN_NAME}_head${LIMIT}_teacher.profile.jsonl}"
RANKER_AUDIT_JSON="${RANKER_AUDIT_JSON:-${POST_ROOT}/proposal_ranking_t5_ranker_head${RANKER_AUDIT_LIMIT}.json}"
RANKER_AUDIT_JSONL="${RANKER_AUDIT_JSONL:-${POST_ROOT}/proposal_ranking_t5_ranker_head${RANKER_AUDIT_LIMIT}.candidates.jsonl}"
BENCH_DIR="${BENCH_DIR:-${ROOT}/benchmark/multiseed_t5_public_head${LIMIT}_${RUN_NAME}_ranker_top${TOP_K}}"
COMPARE_TE_JSON="${COMPARE_TE_JSON:-${ROOT}/benchmark/compare_${RUN_NAME}_ranker_vs_te_only_head${LIMIT}.json}"
COMPARE_TE_MD="${COMPARE_TE_MD:-${ROOT}/benchmark/compare_${RUN_NAME}_ranker_vs_te_only_head${LIMIT}.md}"
COMPARE_HARDNEG_JSON="${COMPARE_HARDNEG_JSON:-${ROOT}/benchmark/compare_${RUN_NAME}_ranker_vs_hardneg_v2_head${LIMIT}.json}"
COMPARE_HARDNEG_MD="${COMPARE_HARDNEG_MD:-${ROOT}/benchmark/compare_${RUN_NAME}_ranker_vs_hardneg_v2_head${LIMIT}.md}"
SPECTRUM_JSON="${SPECTRUM_JSON:-${ROOT}/benchmark/multi_scale_sequence_spectrum_${RUN_NAME}.json}"
SPECTRUM_MD="${SPECTRUM_MD:-${ROOT}/benchmark/multi_scale_sequence_spectrum_${RUN_NAME}.md}"
SPECTRUM_FIG_DIR="${SPECTRUM_FIG_DIR:-${ROOT}/benchmark/multi_scale_sequence_spectrum_${RUN_NAME}_figures}"

usage() {
  cat <<'EOF'
Usage:
  run_after_stage_a_a100_max.sh [--dry-run]

Purpose:
  Queue downstream evaluation after one A100-max Stage A run completes.
  It waits for STAGE_A_PID when provided; otherwise it waits until no
  train_backbone process for RUN_NAME remains.

Environment overrides:
  ROOT, PYTHON_BIN, SEED, RUN_NAME, STAGE_A_PID, CUDA_VISIBLE_DEVICES,
  POLL_SECONDS, LIMIT, TOP_K, PROPOSAL_TOP_K, RANKER_STEPS,
  RANKER_AUDIT_LIMIT, SEEDS, N_BOOTSTRAP, N_PERMUTATIONS, EDIT_BUDGET,
  RECORDS_JSONL, STAGE_A_DIR, BASE_CHECKPOINT, SNAPSHOT, POST_ROOT,
  STATUS_JSON, STATUS_MD, PROGRESS_JSONL
EOF
}

print_plan() {
  cat <<EOF
A100_MAX_POSTEVAL
artifact_kind=a100_max_posteval
ROOT=${ROOT}
PYTHON_BIN=${PYTHON_BIN}
RUN_NAME=${RUN_NAME}
STAGE_A_PID=${STAGE_A_PID}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
LIMIT=${LIMIT}
TOP_K=${TOP_K}
PROPOSAL_TOP_K=${PROPOSAL_TOP_K}
RANKER_STEPS=${RANKER_STEPS}
RANKER_AUDIT_LIMIT=${RANKER_AUDIT_LIMIT}
POST_ROOT=${POST_ROOT}
AUDIT_JSON=${AUDIT_JSON}
RANKER_DIR=${RANKER_DIR}
BENCH_DIR=${BENCH_DIR}
COMPARE_TE_JSON=${COMPARE_TE_JSON}
SPECTRUM_JSON=${SPECTRUM_JSON}
EOF
}

json_log() {
  mkdir -p "$(dirname "${PROGRESS_JSONL}")"
  "${PYTHON_BIN}" - "$PROGRESS_JSONL" "$@" <<'PY'
import json
import sys
import time

path = sys.argv[1]
event = sys.argv[2]
fields = {}
for item in sys.argv[3:]:
    if "=" in item:
        key, value = item.split("=", 1)
        fields[key] = value
fields.update({"event": event, "time": time.time()})
with open(path, "a", encoding="utf-8") as fh:
    fh.write(json.dumps(fields, sort_keys=True) + "\n")
PY
}

write_status() {
  local status="$1"
  local note="${2:-}"
  mkdir -p "$(dirname "${STATUS_JSON}")"
  "${PYTHON_BIN}" - "${STATUS_JSON}" "${STATUS_MD}" "${status}" "${note}" <<PY
import json
import os
import time
import sys

out_json, out_md, status, note = sys.argv[1:5]
paths = {
    "base_checkpoint": "${BASE_CHECKPOINT}",
    "snapshot": "${SNAPSHOT}",
    "audit_json": "${AUDIT_JSON}",
    "audit_jsonl": "${AUDIT_JSONL}",
    "ranker_checkpoint": "${RANKER_DIR}/proposal_ranker_best.pt",
    "ranker_audit_json": "${RANKER_AUDIT_JSON}",
    "bench_summary": "${BENCH_DIR}/multiseed_summary.json",
    "compare_te_json": "${COMPARE_TE_JSON}",
    "compare_hardneg_json": "${COMPARE_HARDNEG_JSON}",
    "spectrum_json": "${SPECTRUM_JSON}",
}
payload = {
    "artifact_kind": "a100_max_posteval_status",
    "claim_policy": (
        "Post-evaluation status is downstream proxy evidence only. Do not claim "
        "SOTA until paired comparisons and T1-T7 audits are inspected."
    ),
    "run_name": "${RUN_NAME}",
    "status": status,
    "note": note,
    "time": time.time(),
    "paths": {key: {"path": path, "exists": os.path.exists(path)} for key, path in paths.items()},
}
with open(out_json, "w", encoding="utf-8") as fh:
    json.dump(payload, fh, indent=2, sort_keys=True)
with open(out_md, "w", encoding="utf-8") as fh:
    fh.write("# A100 Max Post-Eval Status\n\n")
    fh.write(f"- Run: `{payload['run_name']}`\n")
    fh.write(f"- Status: `{status}`\n")
    fh.write(f"- Note: {note}\n")
    fh.write(f"- Claim policy: {payload['claim_policy']}\n\n")
    fh.write("| artifact | exists | path |\n|---|---:|---|\n")
    for key, row in payload["paths"].items():
        fh.write(f"| {key} | `{row['exists']}` | `{row['path']}` |\n")
PY
}

stage_a_running() {
  if [[ -n "${STAGE_A_PID}" ]]; then
    kill -0 "${STAGE_A_PID}" 2>/dev/null
    return $?
  fi
  pgrep -f "train_backbone .*${RUN_NAME}" >/dev/null 2>&1
}

wait_stage_a() {
  write_status "waiting_for_stage_a" "waiting for ${RUN_NAME}"
  json_log waiting_for_stage_a run_name="${RUN_NAME}" stage_a_pid="${STAGE_A_PID}"
  while stage_a_running; do
    sleep "${POLL_SECONDS}"
    json_log wait_tick run_name="${RUN_NAME}"
    write_status "waiting_for_stage_a" "still waiting for ${RUN_NAME}"
  done
  json_log stage_a_done run_name="${RUN_NAME}"
}

run_cmd() {
  json_log command_start label="$1"
  shift
  "$@"
  json_log command_done label="$1"
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then usage; exit 0; fi
if [[ "${1:-}" == "--dry-run" ]]; then print_plan; exit 0; fi

mkdir -p "${POST_ROOT}" "$(dirname "${RANKER_PROFILE}")" "${RANKER_DIR}"
export PYTHONPATH="$(dirname "${ROOT}")${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

wait_stage_a
if [[ ! -s "${BASE_CHECKPOINT}" ]]; then
  write_status "failed_missing_checkpoint" "missing ${BASE_CHECKPOINT}"
  echo "Missing checkpoint: ${BASE_CHECKPOINT}" >&2
  exit 2
fi

cp "${BASE_CHECKPOINT}" "${SNAPSHOT}"
write_status "proposal_ranking" "snapshot ready"

"${PYTHON_BIN}" -m mrna_editflow.eval.proposal_ranking \
  --records-jsonl "${RECORDS_JSONL}" \
  --checkpoint "${SNAPSHOT}" \
  --task-id T5 \
  --limit "${LIMIT}" \
  --candidate-cap 0 \
  --top-k "${PROPOSAL_TOP_K}" \
  --device cuda \
  --out-json "${AUDIT_JSON}" \
  --out-jsonl "${AUDIT_JSONL}"

"${PYTHON_BIN}" -m mrna_editflow.eval.proposal_ranking \
  --records-jsonl "${RECORDS_JSONL}" \
  --checkpoint "${SNAPSHOT}" \
  --task-id T5 \
  --limit "${RANKER_AUDIT_LIMIT}" \
  --candidate-cap 0 \
  --top-k "${PROPOSAL_TOP_K}" \
  --device cuda \
  --out-json "${BASE_AUDIT_JSON}" \
  --out-jsonl "${BASE_AUDIT_JSONL}"

write_status "ranker_distill" "proposal-ranking complete"
"${PYTHON_BIN}" -m mrna_editflow.train_proposal_ranker \
  --run-mode development \
  --records-jsonl "${RECORDS_JSONL}" \
  --teacher-jsonl "${AUDIT_JSONL}" \
  --base-checkpoint "${SNAPSHOT}" \
  --save-dir "${RANKER_DIR}" \
  --profile-path "${RANKER_PROFILE}" \
  --steps "${RANKER_STEPS}" \
  --batch-records 4 \
  --max-pairs-per-record 32 \
  --lr 2e-5 \
  --pair-source-mode global \
  --device cuda \
  --seed "${SEED}"

write_status "ranker_audit" "ranker distilled"
"${PYTHON_BIN}" -m mrna_editflow.eval.proposal_ranking \
  --records-jsonl "${RECORDS_JSONL}" \
  --checkpoint "${RANKER_DIR}/proposal_ranker_best.pt" \
  --task-id T5 \
  --limit "${RANKER_AUDIT_LIMIT}" \
  --candidate-cap 0 \
  --top-k "${PROPOSAL_TOP_K}" \
  --device cuda \
  --out-json "${RANKER_AUDIT_JSON}" \
  --out-jsonl "${RANKER_AUDIT_JSONL}"

write_status "multiseed_t5" "ranker audit complete"
"${PYTHON_BIN}" -m mrna_editflow.eval.run_multiseed_benchmark \
  --run-mode development \
  --records-jsonl "${RECORDS_JSONL}" \
  --checkpoint "${RANKER_DIR}/proposal_ranker_best.pt" \
  --task-id T5 \
  --edit-budget "${EDIT_BUDGET}" \
  --proposal-top-k "${TOP_K}" \
  --proposal-temperature 1.0 \
  --limit "${LIMIT}" \
  --device cuda \
  --seeds ${SEEDS} \
  --n-bootstrap "${N_BOOTSTRAP}" \
  --max-novelty-sources 0 \
  --resume \
  --out-dir "${BENCH_DIR}"

write_status "paired_compare" "multiseed complete"
"${PYTHON_BIN}" -m mrna_editflow.eval.compare_benchmarks \
  --baseline "te_only_top${TOP_K}=${ROOT}/benchmark/multiseed_t5_public_head${LIMIT}_mo_te_only_top${TOP_K}/multiseed_summary.json" \
  --run "a100_max_ranker_top${TOP_K}=${BENCH_DIR}/multiseed_summary.json" \
  --metrics delta_oracle_te_vs_source mean_oracle_te mean_protein_identity within_budget_fraction reading_frame_intact_fraction \
  --out-json "${COMPARE_TE_JSON}" \
  --out-md "${COMPARE_TE_MD}" \
  --n-bootstrap "${N_BOOTSTRAP}" \
  --n-permutations "${N_PERMUTATIONS}" \
  --require-default-matching-config

"${PYTHON_BIN}" -m mrna_editflow.eval.compare_benchmarks \
  --baseline "hardneg_v2_top${TOP_K}=${ROOT}/benchmark/multiseed_t5_public_head${LIMIT}_hardneg_v2_top${TOP_K}/multiseed_summary.json" \
  --run "a100_max_ranker_top${TOP_K}=${BENCH_DIR}/multiseed_summary.json" \
  --metrics delta_oracle_te_vs_source mean_oracle_te mean_protein_identity within_budget_fraction reading_frame_intact_fraction \
  --out-json "${COMPARE_HARDNEG_JSON}" \
  --out-md "${COMPARE_HARDNEG_MD}" \
  --n-bootstrap "${N_BOOTSTRAP}" \
  --n-permutations "${N_PERMUTATIONS}" \
  --require-default-matching-config

write_status "spectrum_audit" "paired compare complete"
"${PYTHON_BIN}" -m mrna_editflow.eval.multi_scale_sequence_spectrum_audit \
  --candidate-glob "${BENCH_DIR}/seed_*/candidates.jsonl" \
  --sources "${BENCH_DIR}/sources.jsonl" \
  --out-json "${SPECTRUM_JSON}" \
  --out-md "${SPECTRUM_MD}" \
  --out-fig-dir "${SPECTRUM_FIG_DIR}"

write_status "complete" "post-eval complete"
json_log complete run_name="${RUN_NAME}"
