#!/usr/bin/env bash
# End-to-end multi-objective reward-head ablation chain (roadmap upgrade #3.1).
#
# Runs, for a single data slice, the full controlled ablation:
#   1. train : export 4 teacher rewards (te_only/scalar/pareto/grpo) + distill 4 rankers
#              (run_multiobjective_ranker_ablation_head256.sh)
#   2. eval  : head256/head1024 x 10-seed top64 benchmark of all 4 rankers + guarded
#              paired permutation vs the te_only single-TE control
#              (eval_multiobjective_ranker_ablation_head256.sh)
#   3. decoded-property analysis: mean the non-TE decoded properties (uAUG, start
#              accessibility, GC, CAI) across seeds so the multi-objective tradeoff is
#              visible, not just oracle TE (eval/analyze_mo_fusion_decoded_properties.py)
#
# This replaces the ad-hoc /tmp chain wrapper used for the head256 run so the exact
# reproduction path lives in the repo. Everything is slice-parameterized via SLICE, so
# the head1024 scale-up (upgrade #3.5) is a single command:
#     SLICE=head1024 CUDA_VISIBLE_DEVICES=<free_gpu> scripts/run_multiobjective_ablation_chain.sh
#
# Optionally gate the start on a currently-running job (e.g. wait for a prior benchmark
# to release the GPU) with WAIT_PID; the chain sleeps until that PID exits before step 1.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-/home/cunyuliu/mrna_editflow_goal/mrna_editflow}"
PYTHON_BIN="${PYTHON_BIN:-/home/cunyuliu/miniconda3/envs/editflow/bin/python3.10}"
SLICE="${SLICE:-head256}"
CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
WAIT_PID="${WAIT_PID:-}"
POLL_SECONDS="${POLL_SECONDS:-60}"
# Resource-safety gate: if MAX_LOADAVG > 0, defer the chain until the host 1-min
# loadavg drops below it (avoids preempting shared CPU). 0 disables the gate.
MAX_LOADAVG="${MAX_LOADAVG:-0}"
LOAD_MAX_WAIT_SECONDS="${LOAD_MAX_WAIT_SECONDS:-86400}"
BENCH_ROOT="${BENCH_ROOT:-${ROOT}/benchmark}"

TRAIN_SCRIPT="${TRAIN_SCRIPT:-${SCRIPT_DIR}/run_multiobjective_ranker_ablation_head256.sh}"
EVAL_SCRIPT="${EVAL_SCRIPT:-${SCRIPT_DIR}/eval_multiobjective_ranker_ablation_head256.sh}"

usage() {
  cat <<'EOF'
Usage:
  run_multiobjective_ablation_chain.sh [--dry-run]

Runs the full multi-objective ablation chain (train -> eval -> decoded-property
analysis) for one data slice. Slice-parameterized via SLICE (head256 default,
head1024 for scale-up). Optionally gate on WAIT_PID to defer until a running job
(e.g. a prior GPU benchmark) exits, and/or MAX_LOADAVG to defer until the host
1-min loadavg drops below a threshold (resource-safety; 0 disables).

Environment overrides:
  ROOT, PYTHON_BIN, SLICE, CUDA_VISIBLE_DEVICES, WAIT_PID, POLL_SECONDS,
  MAX_LOADAVG, LOAD_MAX_WAIT_SECONDS, BENCH_ROOT, TRAIN_SCRIPT, EVAL_SCRIPT
  (plus all overrides accepted by the train/eval sub-scripts)
EOF
}

print_plan() {
  echo "ROOT=${ROOT}"
  echo "SLICE=${SLICE}  CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER}  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  echo "WAIT_PID=${WAIT_PID:-<none>}  POLL_SECONDS=${POLL_SECONDS}  MAX_LOADAVG=${MAX_LOADAVG}  LOAD_MAX_WAIT_SECONDS=${LOAD_MAX_WAIT_SECONDS}"
  echo "step 1 train : SLICE=${SLICE} ${TRAIN_SCRIPT}"
  echo "step 2 eval  : SLICE=${SLICE} ${EVAL_SCRIPT}"
  echo "step 3 decoded analysis : ${PYTHON_BIN} ${ROOT}/eval/analyze_mo_fusion_decoded_properties.py ${BENCH_ROOT} ${SLICE}"
  echo "final compare -> ${BENCH_ROOT}/compare_mo_fusion_vs_te_only_${SLICE}.{json,md}"
  echo "--- sub-script plans ---"
  SLICE="${SLICE}" bash "${TRAIN_SCRIPT}" --dry-run
  SLICE="${SLICE}" bash "${EVAL_SCRIPT}" --dry-run
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then usage; exit 0; fi
if [[ "${1:-}" == "--dry-run" ]]; then print_plan; exit 0; fi

export ROOT PYTHON_BIN SLICE CUDA_DEVICE_ORDER CUDA_VISIBLE_DEVICES BENCH_ROOT

if [[ -n "${WAIT_PID}" ]]; then
  echo "[$(date -Iseconds)] gating chain on PID ${WAIT_PID} (poll ${POLL_SECONDS}s)"
  while kill -0 "${WAIT_PID}" 2>/dev/null; do
    sleep "${POLL_SECONDS}"
  done
  echo "[$(date -Iseconds)] PID ${WAIT_PID} exited; starting chain"
fi

# Resource-safety loadavg gate (opt-in via MAX_LOADAVG>0). Defers until the host
# 1-min loadavg is below the threshold, so head1024 scale-up does not preempt a
# busy shared machine. Bounded by LOAD_MAX_WAIT_SECONDS so it cannot hang forever.
if awk "BEGIN{exit !(${MAX_LOADAVG} > 0)}"; then
  waited=0
  while :; do
    load1="$(cut -d' ' -f1 /proc/loadavg)"
    if awk "BEGIN{exit !(${load1} < ${MAX_LOADAVG})}"; then
      echo "[$(date -Iseconds)] loadavg ${load1} < ${MAX_LOADAVG}; proceeding"
      break
    fi
    if (( waited >= LOAD_MAX_WAIT_SECONDS )); then
      echo "[$(date -Iseconds)] loadavg gate timed out after ${waited}s (load ${load1} >= ${MAX_LOADAVG}); aborting to avoid preemption" >&2
      exit 2
    fi
    echo "[$(date -Iseconds)] loadavg ${load1} >= ${MAX_LOADAVG}; waiting ${POLL_SECONDS}s (waited ${waited}s)"
    sleep "${POLL_SECONDS}"
    waited=$(( waited + POLL_SECONDS ))
  done
fi

echo "[$(date -Iseconds)] STEP 1/3 train (SLICE=${SLICE})"
SLICE="${SLICE}" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" bash "${TRAIN_SCRIPT}"

echo "[$(date -Iseconds)] STEP 2/3 eval + compare (SLICE=${SLICE})"
SLICE="${SLICE}" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" bash "${EVAL_SCRIPT}"

echo "[$(date -Iseconds)] STEP 3/3 decoded-property analysis (SLICE=${SLICE})"
export PYTHONPATH="$(dirname "${ROOT}")${PYTHONPATH:+:${PYTHONPATH}}"
"${PYTHON_BIN}" "${ROOT}/eval/analyze_mo_fusion_decoded_properties.py" "${BENCH_ROOT}" "${SLICE}"

echo "[$(date -Iseconds)] CHAIN COMPLETE (SLICE=${SLICE})"
