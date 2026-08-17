#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/cunyuliu/mrna_editflow_goal/worktrees/route_a_v3_route2_method_repair_20260817}"
ROUTE2_ROOT="${ROUTE2_ROOT:-/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2}"
PYTHON="${PYTHON:-/home/cunyuliu/miniconda3/envs/editflow/bin/python}"
POLL_SECONDS="${POLL_SECONDS:-900}"
OFFICIAL="${ROUTE2_ROOT}/runs/mrnabert_online_encoder_validation_v1/validation_summary.json"
SCREEN="${ROUTE2_ROOT}/benchmarks/mrnabert_alibi_attention_backend_gpu4_v1/report.json"
CANDIDATE="${ROUTE2_ROOT}/runs/mrnabert_sdpa_full_encoder_alignment_gpu4_v1/validation_summary.json"
ADJUDICATION="${ROUTE2_ROOT}/comparisons/mrnabert_sdpa_backend_adjudication_v1.json"

while [[ ! -f "${OFFICIAL}" || ! -f "${SCREEN}" ]]; do
  printf '%s waiting_for_online_and_attention_screen\n' "$(date -Is)"
  sleep "${POLL_SECONDS}"
done

screen_decision=$("${PYTHON}" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["summary"]["decision"])' \
  "${SCREEN}")
candidate_args=()
if [[ "${screen_decision}" == "ELIGIBLE_FOR_FULL_ENCODER_CACHE_ALIGNMENT_BENCHMARK" ]]; then
  while true; do
    free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 4 | tr -d ' ')
    util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 4 | tr -d ' ')
    if [[ "${free_mb}" -ge 30000 && "${util}" -le 20 ]]; then
      break
    fi
    sleep "${POLL_SECONDS}"
  done
  cd "${REPO_ROOT}"
  "${PYTHON}" \
    scripts/route_a_v3/validate_route2_mrnabert_online_encoder_v1.py \
    --config configs/route_a_v3_route2_mrnabert_sdpa_full_encoder_alignment_gpu4_v1.json
  candidate_args=(--candidate-validation "${CANDIDATE}")
fi

cd "${REPO_ROOT}"
"${PYTHON}" scripts/route_a_v3/adjudicate_route2_mrnabert_sdpa_backend_v1.py \
  --attention-screen "${SCREEN}" \
  --official-validation "${OFFICIAL}" \
  "${candidate_args[@]}" \
  --minimum-speedup 1.1 \
  --output "${ADJUDICATION}"
