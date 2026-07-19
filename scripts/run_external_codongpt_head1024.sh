#!/usr/bin/env bash
# Run the official public codonGPT checkpoint on the T4 head1024 input pack.
set -euo pipefail

ROOT="${ROOT:-/home/cunyuliu/mrna_editflow_goal/mrna_editflow}"
REPORT_PYTHON="${PYTHON_BIN:-/home/cunyuliu/miniconda3/envs/editflow/bin/python3.10}"
INPUT_PACK_SUMMARY="${INPUT_PACK_SUMMARY:-${ROOT}/benchmark/external_sota/input_pack_t5_head1024/summary.json}"
SOURCE_RECORDS="${SOURCE_RECORDS:-${ROOT}/benchmark/multiseed_t5_public_head1024_sources.jsonl}"
EXECUTABLE="${CODONGPT_BIN:-${ROOT}/scripts/external_codongpt.sh}"
MODEL_DIR="${CODONGPT_MODEL_DIR:-${ROOT}/external_tools/codonGPT_hf_ee7017c4}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/benchmark/external_sota/real_runs_t5_head1024}"
OUT_DIR="${OUT_DIR:-${OUT_ROOT}/codonGPT}"
STATUS_JSON="${STATUS_JSON:-${OUT_ROOT}/codonGPT.status.json}"
LOG_PATH="${LOG_PATH:-${ROOT}/logs/external_codongpt_head1024.log}"
GPU="${CODONGPT_GPU:-6}"
LIMIT="${LIMIT:-1024}"
BATCH_SIZE="${BATCH_SIZE:-64}"
SEED="${SEED:-0}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_K="${TOP_K:-50}"
TOP_P="${TOP_P:-0.9}"
RESUME="${RESUME:-1}"
AUTO_SETUP="${AUTO_SETUP:-1}"
REFRESH_REPORTS="${REFRESH_REPORTS:-1}"

usage() {
  cat <<'EOF'
Usage:
  run_external_codongpt_head1024.sh [--dry-run]

Runs the pinned public naniltx/codonGPT Hugging Face checkpoint with the model
card's synonymous-mask sampling recipe. The public artifact is a pretrained
checkpoint, not a released task-specific RL policy, so protocol fidelity for
the paper's RL results remains false.

Environment overrides:
  ROOT, PYTHON_BIN, INPUT_PACK_SUMMARY, SOURCE_RECORDS, CODONGPT_BIN,
  CODONGPT_MODEL_DIR, OUT_ROOT, OUT_DIR, STATUS_JSON, LOG_PATH,
  CODONGPT_GPU, LIMIT, BATCH_SIZE, SEED, TEMPERATURE, TOP_K, TOP_P,
  RESUME, AUTO_SETUP, REFRESH_REPORTS
EOF
}

write_status() {
  local status="$1"
  local message="$2"
  STATUS="${status}" MESSAGE="${message}" STATUS_JSON="${STATUS_JSON}" \
    GPU="${GPU}" LIMIT="${LIMIT}" BATCH_SIZE="${BATCH_SIZE}" \
    "${REPORT_PYTHON}" - <<'PY'
import json
import os
import time

payload = {
    "artifact_kind": "external_codongpt_head1024_status",
    "time": time.time(),
    "status": os.environ["STATUS"],
    "message": os.environ["MESSAGE"],
    "gpu": os.environ["GPU"],
    "limit": int(os.environ["LIMIT"]),
    "batch_size": int(os.environ["BATCH_SIZE"]),
    "protocol": "official_hf_pretrained_synonymous_masked_sampling",
    "paper_rl_protocol_reproduced": False,
}
path = os.environ["STATUS_JSON"]
os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
with open(path, "w", encoding="utf-8") as fh:
    json.dump(payload, fh, indent=2, sort_keys=True)
PY
}

print_plan() {
  echo "EXTERNAL CODONGPT HEAD1024"
  echo "ROOT=${ROOT}"
  echo "INPUT_PACK_SUMMARY=${INPUT_PACK_SUMMARY}"
  echo "EXECUTABLE=${EXECUTABLE}"
  echo "MODEL_DIR=${MODEL_DIR}"
  echo "OUT_DIR=${OUT_DIR}"
  echo "GPU=${GPU} LIMIT=${LIMIT} BATCH_SIZE=${BATCH_SIZE}"
  echo "SEED=${SEED} TEMPERATURE=${TEMPERATURE} TOP_K=${TOP_K} TOP_P=${TOP_P}"
  echo "RESUME=${RESUME} AUTO_SETUP=${AUTO_SETUP}"
  echo "REFRESH_REPORTS=${REFRESH_REPORTS}"
  echo "STATUS_JSON=${STATUS_JSON}"
  echo "LOG_PATH=${LOG_PATH}"
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then usage; exit 0; fi
if [[ "${1:-}" == "--dry-run" ]]; then print_plan; exit 0; fi

mkdir -p "${OUT_DIR}" "$(dirname "${LOG_PATH}")"
export PYTHONPATH="$(dirname "${ROOT}")${PYTHONPATH:+:${PYTHONPATH}}"

on_error() {
  local code=$?
  write_status "failed" "codonGPT adapter or report refresh failed with exit ${code}"
  exit "${code}"
}
trap on_error ERR

if [[ "${AUTO_SETUP}" == "1" ]] && {
  [[ ! -x "${ROOT}/external_tools/envs/codongpt/bin/python" ]] ||
  [[ ! -f "${MODEL_DIR}/model_manifest.json" ]];
}; then
  ROOT="${ROOT}" CODONGPT_MODEL_DIR="${MODEL_DIR}" \
    bash "${ROOT}/scripts/setup_external_codongpt.sh"
fi

write_status "running" "official codonGPT HF checkpoint head1024 is running"
resume_args=()
if [[ "${RESUME}" == "1" ]]; then resume_args+=(--resume); fi

CUDA_VISIBLE_DEVICES="${GPU}" "${EXECUTABLE}" \
  --input-pack-summary "${INPUT_PACK_SUMMARY}" \
  --executable "${EXECUTABLE}" \
  --out-dir "${OUT_DIR}" \
  --limit "${LIMIT}" \
  --batch-size "${BATCH_SIZE}" \
  --seed "${SEED}" \
  --temperature "${TEMPERATURE}" \
  --top-k "${TOP_K}" \
  --top-p "${TOP_P}" \
  --device cuda \
  --progress-jsonl "${OUT_DIR}/progress.jsonl" \
  "${resume_args[@]}" > "${LOG_PATH}"

if [[ "${REFRESH_REPORTS}" == "1" ]]; then
  LINEARDESIGN_BIN="${ROOT}/scripts/external_lineardesign.sh" \
  ENSEMBLEDESIGN_BIN="${ROOT}/scripts/external_ensembledesign.sh" \
  CODONGPT_BIN="${EXECUTABLE}" \
  UTRGAN_BIN="${ROOT}/scripts/external_utrgan.sh" \
  UTAILOR_BIN="${ROOT}/scripts/external_utailor.sh" \
  "${REPORT_PYTHON}" -m mrna_editflow.baselines.external_sota_dry_run \
    --out-dir "${ROOT}/benchmark/external_sota/dry_run_t5_head1024" \
    --task-id T5 \
    --records-jsonl "${SOURCE_RECORDS}" \
    --limit 1024 \
    --split-name public_head1024 \
    --seed 0 \
    --hardware-label a100-server

  "${REPORT_PYTHON}" -m mrna_editflow.eval.audit_external_sota_real_runs \
    --project-root "${ROOT}" \
    --out-json docs/external_sota_real_run_audit.json \
    --out-md docs/external_sota_real_run_audit.md

  "${REPORT_PYTHON}" -m mrna_editflow.eval.build_t4_external_cds_comparison \
    --project-root "${ROOT}" \
    --out-json docs/t4_external_cds_baseline_comparison.json \
    --out-md docs/t4_external_cds_baseline_comparison.md

  "${REPORT_PYTHON}" -m mrna_editflow.eval.build_paper_table3_external_baselines \
    --project-root "${ROOT}" \
    --out-json docs/paper_table3_external_baseline_readiness.json \
    --out-md docs/paper_table3_external_baseline_readiness.md

  "${REPORT_PYTHON}" -m mrna_editflow.eval.audit_sota_readiness \
    --project-root "${ROOT}" \
    --slice head256 \
    --top-k 64 \
    --out-json docs/sota_readiness_audit_head256.json \
    --out-md docs/sota_readiness_audit_head256.md

  "${REPORT_PYTHON}" -m mrna_editflow.eval.sota_gap_report \
    --project-root "${ROOT}" \
    --out-json docs/sota_gap_report.json \
    --out-md docs/sota_gap_report.md
fi

write_status "complete" "official codonGPT HF checkpoint head1024 completed"
trap - ERR
