#!/usr/bin/env bash
# Protein-conditioned T4/CDS audit queue for one slice.
#
# Runs the CDS-only codon-lattice DP, protein->CDS constructive design,
# codon-level metrics, CAI-GC sweep, sweep audit, and slice-specific T4 report.
set -euo pipefail

ROOT="${ROOT:-/home/cunyuliu/mrna_editflow_goal/mrna_editflow}"
PYTHON_BIN="${PYTHON_BIN:-/home/cunyuliu/miniconda3/envs/editflow/bin/python3.10}"
SLICE="${SLICE:-head1024}"
case "${SLICE}" in
  head[0-9]*) DEFAULT_LIMIT="${SLICE#head}" ;;
  *) DEFAULT_LIMIT="" ;;
esac
LIMIT="${LIMIT:-${DEFAULT_LIMIT}}"
TOP_K="${TOP_K:-64}"
BENCH_ROOT="${BENCH_ROOT:-${ROOT}/benchmark}"
RECORDS_JSONL="${RECORDS_JSONL:-${BENCH_ROOT}/multiseed_t5_public_${SLICE}_hardneg_v2_top${TOP_K}/sources.jsonl}"
REFERENCE_RECORDS_JSONL="${REFERENCE_RECORDS_JSONL:-${BENCH_ROOT}/multiseed_t5_public_head1024_sources.jsonl}"
GC_WEIGHTS="${GC_WEIGHTS:-0,0.1,0.5,1,2,4,8,16}"
TARGET_GC="${TARGET_GC:-0.55}"
CAI_WEIGHT="${CAI_WEIGHT:-1.0}"
GC_WEIGHT="${GC_WEIGHT:-0.10}"
BOUNDARY_WEIGHT="${BOUNDARY_WEIGHT:-0.05}"
CODON_DP_MAX_CODON_CHANGES="${CODON_DP_MAX_CODON_CHANGES:-3}"
PROTEIN_MAX_CODON_CHANGES="${PROTEIN_MAX_CODON_CHANGES:-${MAX_CODON_CHANGES:-}}"

OUT_DIR="${OUT_DIR:-${BENCH_ROOT}/protein_conditioned_t4_${SLICE}}"
STATUS_JSON="${STATUS_JSON:-${OUT_DIR}/status.json}"
STATUS_MD="${STATUS_MD:-${OUT_DIR}/status.md}"
PROGRESS_JSONL="${PROGRESS_JSONL:-${OUT_DIR}/progress.jsonl}"

CODON_DP_JSONL="${CODON_DP_JSONL:-${BENCH_ROOT}/codon_lattice_dp_${SLICE}.jsonl}"
CODON_DP_JSON="${CODON_DP_JSON:-${BENCH_ROOT}/codon_lattice_dp_${SLICE}.json}"
PROTEIN_JSONL="${PROTEIN_JSONL:-${BENCH_ROOT}/protein_conditioned_cds_${SLICE}.jsonl}"
PROTEIN_JSON="${PROTEIN_JSON:-${BENCH_ROOT}/protein_conditioned_cds_${SLICE}.summary.json}"
PROTEIN_PROGRESS_JSONL="${PROTEIN_PROGRESS_JSONL:-${OUT_DIR}/protein_design.progress.jsonl}"
PROTEIN_PROGRESS_EVERY="${PROTEIN_PROGRESS_EVERY:-25}"
CODON_METRICS_JSON="${CODON_METRICS_JSON:-${BENCH_ROOT}/protein_conditioned_codon_metrics_${SLICE}.json}"
CODON_METRICS_MD="${CODON_METRICS_MD:-${BENCH_ROOT}/protein_conditioned_codon_metrics_${SLICE}.md}"
GC_SWEEP_PREFIX="${GC_SWEEP_PREFIX:-${BENCH_ROOT}/protein_conditioned_cds_gc_sweep_${SLICE}}"
T4_JSON="${T4_JSON:-${BENCH_ROOT}/t4_protein_identity_cai_gc_report_${SLICE}.json}"
T4_MD="${T4_MD:-${BENCH_ROOT}/t4_protein_identity_cai_gc_report_${SLICE}.md}"

RUN_CODON_DP="${RUN_CODON_DP:-1}"
RUN_PROTEIN_DESIGN="${RUN_PROTEIN_DESIGN:-1}"
RUN_CODON_METRICS="${RUN_CODON_METRICS:-1}"
RUN_GC_SWEEP="${RUN_GC_SWEEP:-1}"
RUN_T4_SUMMARY="${RUN_T4_SUMMARY:-1}"

usage() {
  cat <<'EOF'
Usage:
  run_protein_conditioned_t4_slice.sh [--dry-run]

Purpose:
  Queue a slice-specific protein-conditioned T4/CDS audit chain:
  codon-lattice DP -> protein-conditioned CDS design -> codon metrics ->
  CAI-GC sweep/audit -> T4 summary.

Environment overrides:
  ROOT, PYTHON_BIN, SLICE, LIMIT, TOP_K, BENCH_ROOT, RECORDS_JSONL,
  REFERENCE_RECORDS_JSONL, GC_WEIGHTS, TARGET_GC, CAI_WEIGHT, GC_WEIGHT,
  BOUNDARY_WEIGHT, CODON_DP_MAX_CODON_CHANGES, PROTEIN_MAX_CODON_CHANGES,
  OUT_DIR, CODON_DP_JSON[L], PROTEIN_JSON[L], PROTEIN_PROGRESS_JSONL,
  PROTEIN_PROGRESS_EVERY, CODON_METRICS_JSON/MD, GC_SWEEP_PREFIX, T4_JSON/MD,
  RUN_CODON_DP, RUN_PROTEIN_DESIGN, RUN_CODON_METRICS, RUN_GC_SWEEP,
  RUN_T4_SUMMARY
EOF
}

write_status() {
  local status="$1"
  local note="$2"
  mkdir -p "${OUT_DIR}"
  "${PYTHON_BIN}" - "${STATUS_JSON}" "${STATUS_MD}" "${status}" "${note}" "${SLICE}" <<'PY'
import json
import sys
import time
from pathlib import Path

status_json, status_md, status, note, slice_name = sys.argv[1:6]
payload = {
    "artifact_kind": "protein_conditioned_t4_slice_status",
    "claim_policy": "Protein-conditioned T4 slice status is proxy/offline CDS evidence only. Do not claim external codonGPT/Prot2RNA or wet-lab SOTA from this queue.",
    "slice": slice_name,
    "status": status,
    "note": note,
    "time": time.time(),
}
Path(status_json).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
Path(status_md).write_text(
    "# Protein-Conditioned T4 Slice Status\n\n"
    f"- slice: `{slice_name}`\n"
    f"- status: `{status}`\n"
    f"- note: {note}\n",
    encoding="utf-8",
)
PY
}

json_log() {
  local event="$1"
  shift || true
  mkdir -p "${OUT_DIR}"
  "${PYTHON_BIN}" - "${PROGRESS_JSONL}" "${event}" "$@" <<'PY'
import json
import sys
import time

path = sys.argv[1]
event = sys.argv[2]
payload = {"time": time.time(), "event": event}
for item in sys.argv[3:]:
    if "=" in item:
        k, v = item.split("=", 1)
        payload[k] = v
with open(path, "a", encoding="utf-8") as fh:
    fh.write(json.dumps(payload, sort_keys=True) + "\n")
PY
}

print_plan() {
  cat <<EOF
PROTEIN_CONDITIONED_T4_SLICE
artifact_kind=protein_conditioned_t4_slice
ROOT=${ROOT}
SLICE=${SLICE}  LIMIT=${LIMIT:-<none>}  TOP_K=${TOP_K}
RECORDS_JSONL=${RECORDS_JSONL}
REFERENCE_RECORDS_JSONL=${REFERENCE_RECORDS_JSONL}
CODON_DP_JSON=${CODON_DP_JSON}
PROTEIN_JSON=${PROTEIN_JSON}
PROTEIN_PROGRESS_JSONL=${PROTEIN_PROGRESS_JSONL}
PROTEIN_PROGRESS_EVERY=${PROTEIN_PROGRESS_EVERY}
CODON_METRICS_JSON=${CODON_METRICS_JSON}
GC_SWEEP_PREFIX=${GC_SWEEP_PREFIX}
T4_JSON=${T4_JSON}
OUT_DIR=${OUT_DIR}
CODON_DP_MAX_CODON_CHANGES=${CODON_DP_MAX_CODON_CHANGES:-<none>}
PROTEIN_MAX_CODON_CHANGES=${PROTEIN_MAX_CODON_CHANGES:-<none>}
RUNS=codon_dp:${RUN_CODON_DP},protein_design:${RUN_PROTEIN_DESIGN},codon_metrics:${RUN_CODON_METRICS},gc_sweep:${RUN_GC_SWEEP},t4_summary:${RUN_T4_SUMMARY}
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then usage; exit 0; fi
if [[ "${1:-}" == "--dry-run" ]]; then print_plan; exit 0; fi

if [[ ! -f "${RECORDS_JSONL}" ]]; then
  echo "Missing records JSONL: ${RECORDS_JSONL}" >&2
  exit 2
fi
if [[ -n "${REFERENCE_RECORDS_JSONL}" && ! -f "${REFERENCE_RECORDS_JSONL}" ]]; then
  echo "Missing reference records JSONL: ${REFERENCE_RECORDS_JSONL}" >&2
  exit 2
fi

export PYTHONPATH="$(dirname "${ROOT}")${PYTHONPATH:+:${PYTHONPATH}}"
write_status "running" "starting protein-conditioned T4 chain"
json_log "start" slice="${SLICE}" limit="${LIMIT:-}"

limit_args=()
if [[ -n "${LIMIT}" ]]; then
  limit_args+=(--limit "${LIMIT}")
fi
codon_dp_max_change_args=()
if [[ -n "${CODON_DP_MAX_CODON_CHANGES}" ]]; then
  codon_dp_max_change_args+=(--max-codon-changes "${CODON_DP_MAX_CODON_CHANGES}")
fi
protein_max_change_args=()
if [[ -n "${PROTEIN_MAX_CODON_CHANGES}" ]]; then
  protein_max_change_args+=(--max-codon-changes "${PROTEIN_MAX_CODON_CHANGES}")
fi

if [[ "${RUN_CODON_DP}" == "1" ]]; then
  write_status "running_codon_lattice_dp" "running CDS synonymous codon-lattice DP"
  json_log "codon_lattice_dp_start"
  "${PYTHON_BIN}" -m mrna_editflow.baselines.codon_lattice_dp \
    --records-jsonl "${RECORDS_JSONL}" \
    --out-jsonl "${CODON_DP_JSONL}" \
    --out-json "${CODON_DP_JSON}" \
    "${limit_args[@]}" \
    --target-gc "${TARGET_GC}" \
    --cai-weight "${CAI_WEIGHT}" \
    --gc-weight "${GC_WEIGHT}" \
    --boundary-weight "${BOUNDARY_WEIGHT}" \
    "${codon_dp_max_change_args[@]}"
  json_log "codon_lattice_dp_done" out_json="${CODON_DP_JSON}"
fi

if [[ "${RUN_PROTEIN_DESIGN}" == "1" ]]; then
  write_status "running_protein_design" "running protein-conditioned CDS design"
  json_log "protein_design_start"
  "${PYTHON_BIN}" -m mrna_editflow.baselines.protein_conditioned_cds \
    --records-jsonl "${RECORDS_JSONL}" \
    --reference-records-jsonl "${REFERENCE_RECORDS_JSONL}" \
    --use-native-baseline \
    --out-jsonl "${PROTEIN_JSONL}" \
    --out-json "${PROTEIN_JSON}" \
    "${limit_args[@]}" \
    --target-gc "${TARGET_GC}" \
    --cai-weight "${CAI_WEIGHT}" \
    --gc-weight "${GC_WEIGHT}" \
    --boundary-weight "${BOUNDARY_WEIGHT}" \
    --progress-jsonl "${PROTEIN_PROGRESS_JSONL}" \
    --progress-every "${PROTEIN_PROGRESS_EVERY}" \
    "${protein_max_change_args[@]}"
  json_log "protein_design_done" out_json="${PROTEIN_JSON}"
fi

if [[ "${RUN_CODON_METRICS}" == "1" ]]; then
  write_status "running_codon_metrics" "auditing protein-conditioned codon-level metrics"
  json_log "codon_metrics_start"
  "${PYTHON_BIN}" -m mrna_editflow.eval.audit_protein_conditioned_codon_metrics \
    --project-root "${ROOT}" \
    --jsonl "${PROTEIN_JSONL}" \
    --top-n 20 \
    --out-json "${CODON_METRICS_JSON}" \
    --out-md "${CODON_METRICS_MD}"
  json_log "codon_metrics_done" out_json="${CODON_METRICS_JSON}"
fi

if [[ "${RUN_GC_SWEEP}" == "1" ]]; then
  write_status "running_gc_sweep" "running CAI-GC Pareto sweep"
  json_log "gc_sweep_start"
  env \
    ROOT="${ROOT}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    SLICE="${SLICE}" \
    LIMIT="${LIMIT}" \
    TOP_K="${TOP_K}" \
    RECORDS_JSONL="${RECORDS_JSONL}" \
    REFERENCE_RECORDS_JSONL="${REFERENCE_RECORDS_JSONL}" \
    GC_WEIGHTS="${GC_WEIGHTS}" \
    TARGET_GC="${TARGET_GC}" \
    CAI_WEIGHT="${CAI_WEIGHT}" \
    BOUNDARY_WEIGHT="${BOUNDARY_WEIGHT}" \
    MAX_CODON_CHANGES="${PROTEIN_MAX_CODON_CHANGES}" \
    OUT_PREFIX="${GC_SWEEP_PREFIX}" \
    CODON_METRICS_JSONL="${PROTEIN_JSONL}" \
    CODON_METRICS_JSON="${CODON_METRICS_JSON}" \
    CODON_METRICS_MD="${CODON_METRICS_MD}" \
    RUN_CODON_METRICS="1" \
    MAX_LOADAVG="0" \
    bash "${ROOT}/scripts/run_protein_conditioned_cds_gc_sweep.sh"
  json_log "gc_sweep_done" out_prefix="${GC_SWEEP_PREFIX}"
fi

if [[ "${RUN_T4_SUMMARY}" == "1" ]]; then
  write_status "running_t4_summary" "building slice-specific T4 summary"
  json_log "t4_summary_start"
  "${PYTHON_BIN}" -m mrna_editflow.eval.summarize_t4_protein_identity_cai_gc \
    --project-root "${ROOT}" \
    --slice "${SLICE}" \
    --out-json "${T4_JSON}" \
    --out-md "${T4_MD}"
  json_log "t4_summary_done" out_json="${T4_JSON}"
fi

write_status "complete" "protein-conditioned T4 chain complete"
json_log "complete" slice="${SLICE}"
