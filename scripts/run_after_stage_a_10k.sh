#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-/home/cunyuliu/mrna_editflow_goal/mrna_editflow}"
STAGE_A_PID="${STAGE_A_PID:-4037283}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"
LIMIT="${LIMIT:-1024}"
TOP_K="${TOP_K:-32}"
RANKER_STEPS="${RANKER_STEPS:-500}"
RUN_NAME="${RUN_NAME:-stage_a_public_full_10k_bs8ga4_seed0}"

export ROOT
export STAGE_A_PID
export CUDA_VISIBLE_DEVICES
export LIMIT
export TOP_K
export RANKER_STEPS
export RANKER_AUDIT_LIMIT="${RANKER_AUDIT_LIMIT:-64}"
export STAGE_A_DIR="${STAGE_A_DIR:-${ROOT}/ckpts/${RUN_NAME}}"
export BASE_CHECKPOINT="${BASE_CHECKPOINT:-${STAGE_A_DIR}/stage_a_best.pt}"
export SNAPSHOT="${SNAPSHOT:-${STAGE_A_DIR}/stage_a_best_10k_final_for_proposal_ranker.pt}"
export AUDIT_JSON="${AUDIT_JSON:-${ROOT}/benchmark/proposal_ranking_t5_stage_a10k_head1024.json}"
export AUDIT_JSONL="${AUDIT_JSONL:-${ROOT}/benchmark/proposal_ranking_t5_stage_a10k_head1024.candidates.jsonl}"
export BASE_AUDIT_JSON="${BASE_AUDIT_JSON:-${ROOT}/benchmark/proposal_ranking_t5_base_stage_a10k_head64.json}"
export BASE_AUDIT_JSONL="${BASE_AUDIT_JSONL:-${ROOT}/benchmark/proposal_ranking_t5_base_stage_a10k_head64.candidates.jsonl}"
export RANKER_DIR="${RANKER_DIR:-${ROOT}/ckpts/proposal_ranker_t5_stage_a10k_head1024_teacher}"
export RANKER_PROFILE="${RANKER_PROFILE:-${ROOT}/logs/proposal_ranker_t5_stage_a10k_head1024_teacher.profile.jsonl}"
export RANKER_AUDIT_JSON="${RANKER_AUDIT_JSON:-${ROOT}/benchmark/proposal_ranking_t5_ranker_stage_a10k_head64.json}"
export RANKER_AUDIT_JSONL="${RANKER_AUDIT_JSONL:-${ROOT}/benchmark/proposal_ranking_t5_ranker_stage_a10k_head64.candidates.jsonl}"
DELEGATE="${DELEGATE:-${SCRIPT_DIR}/run_after_stage_a_full1k.sh}"

exec "${DELEGATE}" "$@"
