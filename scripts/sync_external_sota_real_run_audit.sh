#!/usr/bin/env bash
# Retry sync and remote refresh for external SOTA real-run audit artifacts.
set -euo pipefail

LOCAL_ROOT="${LOCAL_ROOT:-/Users/bytedance/Documents/research/mrna_editflow}"
REMOTE_HOST="${REMOTE_HOST:-cunyuliu@36.137.135.49}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/cunyuliu/mrna_editflow_goal/mrna_editflow}"
REMOTE_PYTHON="${REMOTE_PYTHON:-/home/cunyuliu/miniconda3/envs/editflow/bin/python3.10}"
TRIES="${TRIES:-20}"
SLEEP_SECONDS="${SLEEP_SECONDS:-60}"
CONNECT_TIMEOUT="${CONNECT_TIMEOUT:-45}"
SLICE="${SLICE:-head256}"
TOP_K="${TOP_K:-64}"
RECORDS_JSONL="${RECORDS_JSONL:-${REMOTE_ROOT}/benchmark/multiseed_t5_public_head1024_sources.jsonl}"

usage() {
  cat <<'EOF'
Usage:
  sync_external_sota_real_run_audit.sh [--dry-run]

Retries the SSH/rsync upload for the external SOTA real-run audit code and then
refreshes the remote read-only reports:
  - docs/external_sota_real_run_audit.{json,md}
  - docs/paper_table3_external_baseline_readiness.{json,md}
  - docs/sota_readiness_audit_head256.{json,md}
  - docs/sota_gap_report.{json,md}

This script does not run external SOTA executables and does not fabricate
metrics. It only validates measured-output contracts if adapter outputs exist.

Environment overrides:
  LOCAL_ROOT, REMOTE_HOST, REMOTE_ROOT, REMOTE_PYTHON, TRIES, SLEEP_SECONDS,
  CONNECT_TIMEOUT, SLICE, TOP_K, RECORDS_JSONL
EOF
}

print_plan() {
  echo "EXTERNAL SOTA REAL-RUN AUDIT SYNC"
  echo "LOCAL_ROOT=${LOCAL_ROOT}"
  echo "REMOTE=${REMOTE_HOST}:${REMOTE_ROOT}"
  echo "REMOTE_PYTHON=${REMOTE_PYTHON}"
  echo "TRIES=${TRIES} SLEEP_SECONDS=${SLEEP_SECONDS} CONNECT_TIMEOUT=${CONNECT_TIMEOUT}"
  echo "SLICE=${SLICE} TOP_K=${TOP_K}"
  echo "RECORDS_JSONL=${RECORDS_JSONL}"
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then usage; exit 0; fi
if [[ "${1:-}" == "--dry-run" ]]; then print_plan; exit 0; fi

cd "${LOCAL_ROOT}"

FILES=(
  baselines/external_sota_input_pack.py
  baselines/external_sota_dry_run.py
  eval/audit_external_sota_real_runs.py
  eval/build_paper_table3_external_baselines.py
  eval/audit_sota_readiness.py
  eval/sota_gap_report.py
  tests/test_eval.py
  tests/test_baselines_ablation.py
  scripts/check_remote_sota_status.sh
  scripts/external_utrgan.sh
  scripts/sync_external_sota_real_run_audit.sh
  docs/next_steps_sota_roadmap.md
)

for attempt in $(seq 1 "${TRIES}"); do
  echo "[$(date '+%Y-%m-%dT%H:%M:%S%z')] sync attempt ${attempt}/${TRIES}"
  if rsync -avR -e "ssh -o BatchMode=yes -o ConnectTimeout=${CONNECT_TIMEOUT} -o ServerAliveInterval=10 -o ServerAliveCountMax=3" \
    "${FILES[@]}" "${REMOTE_HOST}:${REMOTE_ROOT}/"; then
    echo "[$(date '+%Y-%m-%dT%H:%M:%S%z')] sync succeeded"
    ssh -o BatchMode=yes -o ConnectTimeout="${CONNECT_TIMEOUT}" \
      -o ServerAliveInterval=10 -o ServerAliveCountMax=3 \
      "${REMOTE_HOST}" \
      "REMOTE_ROOT='${REMOTE_ROOT}' REMOTE_PYTHON='${REMOTE_PYTHON}' SLICE='${SLICE}' TOP_K='${TOP_K}' RECORDS_JSONL='${RECORDS_JSONL}' bash -s" <<'REMOTE'
set -euo pipefail
cd "${REMOTE_ROOT}"
export PYTHONPATH="$(dirname "${REMOTE_ROOT}")${PYTHONPATH:+:${PYTHONPATH}}"

"${REMOTE_PYTHON}" -m mrna_editflow.baselines.external_sota_input_pack \
  --records-jsonl "${RECORDS_JSONL}" \
  --out-dir benchmark/external_sota/input_pack_t5_head1024 \
  --limit 1024 \
  --split-name public_head1024 \
  --seed 0

LINEARDESIGN_BIN="${REMOTE_ROOT}/scripts/external_lineardesign.sh" \
ENSEMBLEDESIGN_BIN="${REMOTE_ROOT}/scripts/external_ensembledesign.sh" \
UTRGAN_BIN="${REMOTE_ROOT}/scripts/external_utrgan.sh" \
"${REMOTE_PYTHON}" -m mrna_editflow.baselines.external_sota_dry_run \
  --records-jsonl "${RECORDS_JSONL}" \
  --out-dir benchmark/external_sota/dry_run_t5_head1024 \
  --limit 1024 \
  --split-name public_head1024 \
  --seed 0 \
  --hardware-label 36.137.135.49

"${REMOTE_PYTHON}" -m mrna_editflow.eval.audit_external_sota_real_runs \
  --project-root "${REMOTE_ROOT}" \
  --out-json docs/external_sota_real_run_audit.json \
  --out-md docs/external_sota_real_run_audit.md

"${REMOTE_PYTHON}" -m mrna_editflow.eval.build_paper_table3_external_baselines \
  --project-root "${REMOTE_ROOT}" \
  --out-json docs/paper_table3_external_baseline_readiness.json \
  --out-md docs/paper_table3_external_baseline_readiness.md

"${REMOTE_PYTHON}" -m mrna_editflow.eval.audit_sota_readiness \
  --project-root "${REMOTE_ROOT}" \
  --slice "${SLICE}" \
  --top-k "${TOP_K}" \
  --out-json "docs/sota_readiness_audit_${SLICE}.json" \
  --out-md "docs/sota_readiness_audit_${SLICE}.md"

"${REMOTE_PYTHON}" -m mrna_editflow.eval.sota_gap_report \
  --project-root "${REMOTE_ROOT}" \
  --out-json docs/sota_gap_report.json \
  --out-md docs/sota_gap_report.md

sha256sum \
  benchmark/external_sota/input_pack_t5_head1024/summary.json \
  benchmark/external_sota/dry_run_t5_head1024/summary.json \
  docs/external_sota_real_run_audit.json \
  docs/paper_table3_external_baseline_readiness.json \
  "docs/sota_readiness_audit_${SLICE}.json" \
  docs/sota_gap_report.json
REMOTE
    exit 0
  fi
  echo "[$(date '+%Y-%m-%dT%H:%M:%S%z')] sync failed; retrying after ${SLEEP_SECONDS}s"
  sleep "${SLEEP_SECONDS}"
done

echo "[$(date '+%Y-%m-%dT%H:%M:%S%z')] sync failed after ${TRIES} attempts" >&2
exit 1
