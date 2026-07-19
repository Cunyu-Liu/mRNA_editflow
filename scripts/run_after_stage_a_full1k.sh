#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/cunyuliu/mrna_editflow_goal/mrna_editflow}"
PYTHON_BIN="${PYTHON_BIN:-/home/cunyuliu/miniconda3/envs/editflow/bin/python3.10}"
STAGE_A_PID="${STAGE_A_PID:-255505}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
POLL_SECONDS="${POLL_SECONDS:-120}"
LIMIT="${LIMIT:-256}"
TOP_K="${TOP_K:-32}"
RANKER_STEPS="${RANKER_STEPS:-500}"
RANKER_AUDIT_LIMIT="${RANKER_AUDIT_LIMIT:-64}"

STAGE_A_DIR="${STAGE_A_DIR:-${ROOT}/ckpts/stage_a_public_full_1k_bs8ga4}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-${STAGE_A_DIR}/stage_a_best.pt}"
SNAPSHOT="${SNAPSHOT:-${STAGE_A_DIR}/stage_a_best_full1k_final_for_proposal_ranker.pt}"
RECORDS_JSONL="${RECORDS_JSONL:-${ROOT}/data/processed/gencode_human_transcripts.records.jsonl}"
AUDIT_JSON="${AUDIT_JSON:-${ROOT}/benchmark/proposal_ranking_t5_full1k_final.json}"
AUDIT_JSONL="${AUDIT_JSONL:-${ROOT}/benchmark/proposal_ranking_t5_full1k_final.candidates.jsonl}"
BASE_AUDIT_JSON="${BASE_AUDIT_JSON:-}"
BASE_AUDIT_JSONL="${BASE_AUDIT_JSONL:-}"
RANKER_DIR="${RANKER_DIR:-${ROOT}/ckpts/proposal_ranker_t5_full1k_final}"
RANKER_PROFILE="${RANKER_PROFILE:-${ROOT}/logs/proposal_ranker_t5_full1k_final.profile.jsonl}"
RANKER_AUDIT_JSON="${RANKER_AUDIT_JSON:-${ROOT}/benchmark/proposal_ranking_t5_ranker_full1k_final_head64.json}"
RANKER_AUDIT_JSONL="${RANKER_AUDIT_JSONL:-${ROOT}/benchmark/proposal_ranking_t5_ranker_full1k_final_head64.candidates.jsonl}"

usage() {
  cat <<'EOF'
Usage:
  run_after_stage_a_full1k.sh [--dry-run]

Purpose:
  Wait for the Stage A full-corpus 1000-step process to finish, freeze the best
  checkpoint into a race-free snapshot, run a full-pool proposal-ranking audit,
  train a TE-aware proposal ranker from that teacher file, and audit the final
  ranker on a head64 slice.

Environment overrides:
  ROOT, PYTHON_BIN, STAGE_A_PID, CUDA_VISIBLE_DEVICES, POLL_SECONDS, LIMIT,
  TOP_K, RANKER_STEPS, RANKER_AUDIT_LIMIT, STAGE_A_DIR, BASE_CHECKPOINT,
  SNAPSHOT, RECORDS_JSONL, AUDIT_JSON, AUDIT_JSONL, BASE_AUDIT_JSON,
  BASE_AUDIT_JSONL, RANKER_DIR, RANKER_PROFILE, RANKER_AUDIT_JSON,
  RANKER_AUDIT_JSONL
EOF
}

print_plan() {
  cat <<EOF
ROOT=${ROOT}
PYTHON_BIN=${PYTHON_BIN}
STAGE_A_PID=${STAGE_A_PID}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
LIMIT=${LIMIT}
TOP_K=${TOP_K}
RANKER_STEPS=${RANKER_STEPS}
RANKER_AUDIT_LIMIT=${RANKER_AUDIT_LIMIT}
BASE_CHECKPOINT=${BASE_CHECKPOINT}
SNAPSHOT=${SNAPSHOT}
AUDIT_JSON=${AUDIT_JSON}
AUDIT_JSONL=${AUDIT_JSONL}
BASE_AUDIT_JSON=${BASE_AUDIT_JSON}
BASE_AUDIT_JSONL=${BASE_AUDIT_JSONL}
RANKER_DIR=${RANKER_DIR}
RANKER_PROFILE=${RANKER_PROFILE}
RANKER_AUDIT_JSON=${RANKER_AUDIT_JSON}

Commands:
  while kill -0 "${STAGE_A_PID}" 2>/dev/null; do sleep "${POLL_SECONDS}"; done
  cp "${BASE_CHECKPOINT}" "${SNAPSHOT}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" PYTHONPATH="\$(dirname "${ROOT}")" "${PYTHON_BIN}" -m mrna_editflow.eval.proposal_ranking ...
  if [[ -n "${BASE_AUDIT_JSON}" ]]; then CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" PYTHONPATH="\$(dirname "${ROOT}")" "${PYTHON_BIN}" -m mrna_editflow.eval.proposal_ranking ... base head${RANKER_AUDIT_LIMIT}; fi
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" PYTHONPATH="\$(dirname "${ROOT}")" "${PYTHON_BIN}" -m mrna_editflow.train_proposal_ranker ...
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" PYTHONPATH="\$(dirname "${ROOT}")" "${PYTHON_BIN}" -m mrna_editflow.eval.proposal_ranking ... final ranker head${RANKER_AUDIT_LIMIT}
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

while kill -0 "${STAGE_A_PID}" 2>/dev/null; do
  sleep "${POLL_SECONDS}"
done

if [[ ! -s "${BASE_CHECKPOINT}" ]]; then
  echo "Missing Stage A checkpoint: ${BASE_CHECKPOINT}" >&2
  exit 1
fi

mkdir -p "$(dirname "${SNAPSHOT}")" "$(dirname "${AUDIT_JSON}")" "${RANKER_DIR}" "$(dirname "${RANKER_PROFILE}")"
cp "${BASE_CHECKPOINT}" "${SNAPSHOT}"

export CUDA_VISIBLE_DEVICES
export PYTHONPATH="$(dirname "${ROOT}")${PYTHONPATH:+:${PYTHONPATH}}"

"${PYTHON_BIN}" -m mrna_editflow.eval.proposal_ranking \
  --records-jsonl "${RECORDS_JSONL}" \
  --checkpoint "${SNAPSHOT}" \
  --task-id T5 \
  --limit "${LIMIT}" \
  --candidate-cap 0 \
  --top-k "${TOP_K}" \
  --device cuda \
  --out-json "${AUDIT_JSON}" \
  --out-jsonl "${AUDIT_JSONL}"

if [[ -n "${BASE_AUDIT_JSON}" ]]; then
  if [[ -z "${BASE_AUDIT_JSONL}" ]]; then
    echo "BASE_AUDIT_JSONL must be set when BASE_AUDIT_JSON is set." >&2
    exit 1
  fi
  mkdir -p "$(dirname "${BASE_AUDIT_JSON}")"
  "${PYTHON_BIN}" -m mrna_editflow.eval.proposal_ranking \
    --records-jsonl "${RECORDS_JSONL}" \
    --checkpoint "${SNAPSHOT}" \
    --task-id T5 \
    --limit "${RANKER_AUDIT_LIMIT}" \
    --candidate-cap 0 \
    --top-k "${TOP_K}" \
    --device cuda \
    --out-json "${BASE_AUDIT_JSON}" \
    --out-jsonl "${BASE_AUDIT_JSONL}"
fi

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
  --device cuda \
  --seed 0

"${PYTHON_BIN}" -m mrna_editflow.eval.proposal_ranking \
  --records-jsonl "${RECORDS_JSONL}" \
  --checkpoint "${RANKER_DIR}/proposal_ranker_best.pt" \
  --task-id T5 \
  --limit "${RANKER_AUDIT_LIMIT}" \
  --candidate-cap 0 \
  --top-k "${TOP_K}" \
  --device cuda \
  --out-json "${RANKER_AUDIT_JSON}" \
  --out-jsonl "${RANKER_AUDIT_JSONL}"
