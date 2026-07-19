#!/usr/bin/env bash
# Protein-conditioned CDS CAI-GC Pareto sweep (roadmap upgrade #3).
#
# Runs the audited protein->CDS DP over a gc_weight grid and writes a Pareto
# frontier artifact. This is CPU-only by design; optional load gating prevents
# the sweep from adding pressure to a busy shared server.
set -euo pipefail

ROOT="${ROOT:-/home/cunyuliu/mrna_editflow_goal/mrna_editflow}"
PYTHON_BIN="${PYTHON_BIN:-/home/cunyuliu/miniconda3/envs/editflow/bin/python3.10}"
SLICE="${SLICE:-head256}"
case "${SLICE}" in
  head[0-9]*) DEFAULT_LIMIT="${SLICE#head}" ;;
  *) DEFAULT_LIMIT="" ;;
esac
LIMIT="${LIMIT:-${DEFAULT_LIMIT}}"
TOP_K="${TOP_K:-64}"
GC_WEIGHTS="${GC_WEIGHTS:-0,0.1,0.5,1,2,4,8,16}"
TARGET_GC="${TARGET_GC:-0.55}"
CAI_WEIGHT="${CAI_WEIGHT:-1.0}"
BOUNDARY_WEIGHT="${BOUNDARY_WEIGHT:-0.05}"
MAX_CODON_CHANGES="${MAX_CODON_CHANGES:-}"
USE_NATIVE_BASELINE="${USE_NATIVE_BASELINE:-1}"

BENCH_ROOT="${BENCH_ROOT:-${ROOT}/benchmark}"
RECORDS_JSONL="${RECORDS_JSONL:-${BENCH_ROOT}/multiseed_t5_public_${SLICE}_hardneg_v2_top${TOP_K}/sources.jsonl}"
REFERENCE_RECORDS_JSONL="${REFERENCE_RECORDS_JSONL:-${BENCH_ROOT}/multiseed_t5_public_head1024_sources.jsonl}"
OUT_PREFIX="${OUT_PREFIX:-${BENCH_ROOT}/protein_conditioned_cds_gc_sweep_${SLICE}}"
OUT_JSONL="${OUT_JSONL:-${OUT_PREFIX}.jsonl}"
OUT_JSON="${OUT_JSON:-${OUT_PREFIX}.summary.json}"
OUT_MD="${OUT_MD:-${OUT_PREFIX}.md}"
AUDIT_JSON="${AUDIT_JSON:-${OUT_PREFIX}.audit.json}"
AUDIT_MD="${AUDIT_MD:-${OUT_PREFIX}.audit.md}"
CODON_METRICS_JSONL="${CODON_METRICS_JSONL:-${BENCH_ROOT}/protein_conditioned_cds_${SLICE}.jsonl}"
CODON_METRICS_JSON="${CODON_METRICS_JSON:-${BENCH_ROOT}/protein_conditioned_codon_metrics_${SLICE}.json}"
CODON_METRICS_MD="${CODON_METRICS_MD:-${BENCH_ROOT}/protein_conditioned_codon_metrics_${SLICE}.md}"
CODON_METRICS_TOP_N="${CODON_METRICS_TOP_N:-20}"
RUN_CODON_METRICS="${RUN_CODON_METRICS:-1}"

POLL_SECONDS="${POLL_SECONDS:-120}"
WAIT_PID="${WAIT_PID:-}"
WAIT_MAX_SECONDS="${WAIT_MAX_SECONDS:-86400}"
MAX_LOADAVG="${MAX_LOADAVG:-0}"
LOAD_MAX_WAIT_SECONDS="${LOAD_MAX_WAIT_SECONDS:-86400}"

usage() {
  cat <<'EOF'
Usage:
  run_protein_conditioned_cds_gc_sweep.sh [--dry-run]

Runs protein-conditioned CDS design over a gc_weight grid and writes JSONL,
JSON summary, and Markdown Pareto-front artifacts.

Environment overrides:
  ROOT, PYTHON_BIN, SLICE, LIMIT, TOP_K, GC_WEIGHTS, TARGET_GC, CAI_WEIGHT,
  BOUNDARY_WEIGHT, MAX_CODON_CHANGES, USE_NATIVE_BASELINE, BENCH_ROOT,
  RECORDS_JSONL, REFERENCE_RECORDS_JSONL, OUT_PREFIX, OUT_JSONL, OUT_JSON,
  OUT_MD, AUDIT_JSON, AUDIT_MD, CODON_METRICS_JSONL, CODON_METRICS_JSON,
  CODON_METRICS_MD, CODON_METRICS_TOP_N, RUN_CODON_METRICS, WAIT_PID,
  WAIT_MAX_SECONDS, POLL_SECONDS, MAX_LOADAVG, LOAD_MAX_WAIT_SECONDS
EOF
}

print_plan() {
  echo "PROTEIN CONDITIONED CDS GC SWEEP"
  echo "ROOT=${ROOT}"
  echo "SLICE=${SLICE}  LIMIT=${LIMIT:-<none>}  TOP_K=${TOP_K}"
  echo "GC_WEIGHTS=${GC_WEIGHTS}  TARGET_GC=${TARGET_GC}  CAI_WEIGHT=${CAI_WEIGHT}  BOUNDARY_WEIGHT=${BOUNDARY_WEIGHT}"
  echo "MAX_CODON_CHANGES=${MAX_CODON_CHANGES:-<none>}  USE_NATIVE_BASELINE=${USE_NATIVE_BASELINE}"
  echo "WAIT_PID=${WAIT_PID:-<none>}  WAIT_MAX_SECONDS=${WAIT_MAX_SECONDS}  POLL_SECONDS=${POLL_SECONDS}"
  echo "MAX_LOADAVG=${MAX_LOADAVG}  LOAD_MAX_WAIT_SECONDS=${LOAD_MAX_WAIT_SECONDS}"
  echo "RECORDS_JSONL=${RECORDS_JSONL}"
  echo "REFERENCE_RECORDS_JSONL=${REFERENCE_RECORDS_JSONL}"
  echo "OUT_JSONL=${OUT_JSONL}"
  echo "OUT_JSON=${OUT_JSON}"
  echo "OUT_MD=${OUT_MD}"
  echo "AUDIT_JSON=${AUDIT_JSON}"
  echo "AUDIT_MD=${AUDIT_MD}"
  echo "RUN_CODON_METRICS=${RUN_CODON_METRICS}"
  echo "CODON_METRICS_JSONL=${CODON_METRICS_JSONL}"
  echo "CODON_METRICS_JSON=${CODON_METRICS_JSON}"
  echo "CODON_METRICS_MD=${CODON_METRICS_MD}"
  echo "command -> ${PYTHON_BIN} -m mrna_editflow.baselines.protein_conditioned_cds --gc-weight-sweep ${GC_WEIGHTS}"
  echo "command -> ${PYTHON_BIN} -m mrna_editflow.eval.audit_protein_conditioned_codon_metrics"
}

read_loadavg1() {
  if [[ -r /proc/loadavg ]]; then
    cut -d' ' -f1 /proc/loadavg
  else
    uptime | awk -F'load average[s]?: ' '{print $2}' | awk -F',' '{gsub(/ /, "", $1); print $1}'
  fi
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then usage; exit 0; fi
if [[ "${1:-}" == "--dry-run" ]]; then print_plan; exit 0; fi

if [[ -n "${WAIT_PID}" ]]; then
  waited=0
  echo "[$(date -Iseconds)] gating GC sweep on PID ${WAIT_PID}"
  while kill -0 "${WAIT_PID}" 2>/dev/null; do
    if (( waited >= WAIT_MAX_SECONDS )); then
      echo "[$(date -Iseconds)] wait timed out after ${waited}s for PID ${WAIT_PID}" >&2
      exit 2
    fi
    sleep "${POLL_SECONDS}"
    waited=$(( waited + POLL_SECONDS ))
  done
  echo "[$(date -Iseconds)] PID ${WAIT_PID} exited; starting GC sweep"
fi

if awk "BEGIN{exit !(${MAX_LOADAVG} > 0)}"; then
  waited=0
  while :; do
    load1="$(read_loadavg1)"
    if [[ -z "${load1}" ]]; then
      echo "[$(date -Iseconds)] unable to read loadavg; aborting load-gated run" >&2
      exit 2
    fi
    if awk "BEGIN{exit !(${load1} < ${MAX_LOADAVG})}"; then
      echo "[$(date -Iseconds)] loadavg ${load1} < ${MAX_LOADAVG}; proceeding"
      break
    fi
    if (( waited >= LOAD_MAX_WAIT_SECONDS )); then
      echo "[$(date -Iseconds)] loadavg gate timed out after ${waited}s (load ${load1} >= ${MAX_LOADAVG})" >&2
      exit 2
    fi
    echo "[$(date -Iseconds)] loadavg ${load1} >= ${MAX_LOADAVG}; waiting ${POLL_SECONDS}s (waited ${waited}s)"
    sleep "${POLL_SECONDS}"
    waited=$(( waited + POLL_SECONDS ))
  done
fi

if [[ ! -f "${RECORDS_JSONL}" ]]; then
  echo "Missing records JSONL: ${RECORDS_JSONL}" >&2
  exit 2
fi
if [[ -n "${REFERENCE_RECORDS_JSONL}" && ! -f "${REFERENCE_RECORDS_JSONL}" ]]; then
  echo "Missing reference records JSONL: ${REFERENCE_RECORDS_JSONL}" >&2
  exit 2
fi

export PYTHONPATH="$(dirname "${ROOT}")${PYTHONPATH:+:${PYTHONPATH}}"
args=(
  -m mrna_editflow.baselines.protein_conditioned_cds
  --records-jsonl "${RECORDS_JSONL}"
  --out-jsonl "${OUT_JSONL}"
  --out-json "${OUT_JSON}"
  --out-md "${OUT_MD}"
  --gc-weight-sweep "${GC_WEIGHTS}"
  --target-gc "${TARGET_GC}"
  --cai-weight "${CAI_WEIGHT}"
  --boundary-weight "${BOUNDARY_WEIGHT}"
)
if [[ -n "${LIMIT}" ]]; then
  args+=(--limit "${LIMIT}")
fi
if [[ "${USE_NATIVE_BASELINE}" == "1" ]]; then
  args+=(--use-native-baseline)
fi
if [[ -n "${REFERENCE_RECORDS_JSONL}" ]]; then
  args+=(--reference-records-jsonl "${REFERENCE_RECORDS_JSONL}")
fi
if [[ -n "${MAX_CODON_CHANGES}" ]]; then
  args+=(--max-codon-changes "${MAX_CODON_CHANGES}")
fi

echo "[$(date -Iseconds)] RUN protein-conditioned CDS GC sweep (SLICE=${SLICE})"
"${PYTHON_BIN}" "${args[@]}"
echo "[$(date -Iseconds)] GC SWEEP COMPLETE -> ${OUT_JSON}"

echo "[$(date -Iseconds)] AUDIT protein-conditioned CDS GC sweep"
"${PYTHON_BIN}" -m mrna_editflow.eval.audit_protein_conditioned_gc_sweep \
  --project-root "${ROOT}" \
  --summary-json "${OUT_JSON}" \
  --jsonl "${OUT_JSONL}" \
  --md "${OUT_MD}" \
  --out-json "${AUDIT_JSON}" \
  --out-md "${AUDIT_MD}"
cat "${AUDIT_MD}"

if [[ "${RUN_CODON_METRICS}" == "1" ]]; then
  if [[ ! -f "${CODON_METRICS_JSONL}" ]]; then
    echo "Missing protein-conditioned CDS JSONL for codon metrics: ${CODON_METRICS_JSONL}" >&2
    echo "Run mrna_editflow.baselines.protein_conditioned_cds without --gc-weight-sweep first, or set CODON_METRICS_JSONL." >&2
    exit 2
  fi
  echo "[$(date -Iseconds)] AUDIT protein-conditioned codon-level metrics"
  "${PYTHON_BIN}" -m mrna_editflow.eval.audit_protein_conditioned_codon_metrics \
    --project-root "${ROOT}" \
    --jsonl "${CODON_METRICS_JSONL}" \
    --top-n "${CODON_METRICS_TOP_N}" \
    --out-json "${CODON_METRICS_JSON}" \
    --out-md "${CODON_METRICS_MD}"
  cat "${CODON_METRICS_MD}"
fi
