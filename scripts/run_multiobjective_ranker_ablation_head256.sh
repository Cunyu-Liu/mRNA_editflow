#!/usr/bin/env bash
# Multi-objective reward-head ablation (roadmap upgrade #3.1) on the head256 T5 slice.
#
# Controlled comparison: hold candidate generation + ranker architecture + distill
# recipe fixed, vary ONLY the teacher reward:
#   - te_only        : single-TE control (w-te=1, others 0) via the SAME exporter
#   - scalar         : weighted scalarization over 6 objectives (default weights)
#   - pareto         : NSGA-II front-major fusion
#   - grpo           : ProMORNA-style per-metric z-score standardized fusion
#
# Motivation: the 10k-recall -> hardneg cascade did NOT beat the single hardneg_v2
# precision ranker (paired p=0.36632), so the bottleneck is precision top-1
# quality, not candidate recall. This ablation asks whether a richer multi-objective
# teacher reward improves the precision ranker itself.
#
# IMPORTANT (pairing mode): this ablation MUST use --pair-source-mode global.
# The exporter emits per-objective source_scores that are IDENTICAL across fusion
# modes; only teacher_score encodes the fusion. In global mode the Bradley-Terry
# pairs are formed from teacher_score, so te_only/scalar/pareto/grpo train genuinely
# different rankers. In source_balanced mode pairs are formed from the per-objective
# source_scores, which ignores teacher_score entirely -> all 4 modes would collapse
# to the SAME ranker (null by construction).
set -euo pipefail

ROOT="${ROOT:-/home/cunyuliu/mrna_editflow_goal/mrna_editflow}"
PYTHON_BIN="${PYTHON_BIN:-/home/cunyuliu/miniconda3/envs/editflow/bin/python3.10}"
CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
STEPS="${STEPS:-500}"
BATCH_RECORDS="${BATCH_RECORDS:-4}"
MAX_PAIRS="${MAX_PAIRS:-32}"
LR="${LR:-2e-5}"
CANDIDATE_CAP="${CANDIDATE_CAP:-256}"
TRAIN_SEED="${TRAIN_SEED:-0}"
PAIR_SOURCE_MODE="${PAIR_SOURCE_MODE:-global}"

# SLICE selects the data-slice scale by SOURCE ROW COUNT (head256 = 256 rows,
# head1024 = 1024 rows). The controlled-ablation methodology is identical across
# slices; only the source pool size grows. Defaults to head256 for back-compat.
SLICE="${SLICE:-head256}"

SOURCES="${SOURCES:-${ROOT}/benchmark/multiseed_t5_public_${SLICE}_hardneg_v2_top64/sources.jsonl}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-${ROOT}/ckpts/proposal_ranker_t5_stage_a10k_head1024_teacher/proposal_ranker_best.pt}"
TEACHER_DIR="${TEACHER_DIR:-${ROOT}/benchmark/multiobjective_teacher_${SLICE}}"
CKPT_ROOT="${CKPT_ROOT:-${ROOT}/ckpts}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs}"

MODES="${MODES:-te_only scalar pareto grpo}"

usage() {
  cat <<'EOF'
Usage:
  run_multiobjective_ranker_ablation_head256.sh [--dry-run]

Exports 4 head256 multi-objective teacher sets (te_only control + scalar/pareto/grpo
treatments) from the SAME candidate pool, then distills one ranker per mode with an
identical recipe. Downstream benchmark/compare is run separately after this completes.

Environment overrides:
  ROOT, PYTHON_BIN, CUDA_VISIBLE_DEVICES, SEEDS, STEPS, BATCH_RECORDS, MAX_PAIRS, LR,
  CANDIDATE_CAP, TRAIN_SEED, PAIR_SOURCE_MODE, SOURCES, BASE_CHECKPOINT, TEACHER_DIR,
  CKPT_ROOT, LOG_ROOT, MODES
EOF
}

teacher_flags() {
  # Only the reward differs across modes; te_only zeroes non-TE weights.
  local mode="$1"
  case "${mode}" in
    te_only) echo "--fusion-mode scalar --w-te 1.0 --w-mrl 0.0 --w-cai 0.0 --w-gc 0.0 --w-access 0.0 --w-uaug 0.0" ;;
    scalar)  echo "--fusion-mode scalar" ;;
    pareto)  echo "--fusion-mode pareto" ;;
    grpo)    echo "--fusion-mode grpo_standardized" ;;
    *) echo "unknown mode: ${mode}" >&2; return 1 ;;
  esac
}

print_plan() {
  echo "ROOT=${ROOT}"
  echo "CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  echo "SOURCES=${SOURCES}"
  echo "BASE_CHECKPOINT=${BASE_CHECKPOINT}"
  echo "STEPS=${STEPS} BATCH_RECORDS=${BATCH_RECORDS} MAX_PAIRS=${MAX_PAIRS} LR=${LR}"
  echo "PAIR_SOURCE_MODE=${PAIR_SOURCE_MODE} CANDIDATE_CAP=${CANDIDATE_CAP}"
  echo "MODES=${MODES}"
  for mode in ${MODES}; do
    echo "--- mode=${mode} teacher_flags: $(teacher_flags "${mode}")"
    echo "    teacher -> ${TEACHER_DIR}/mo_teacher_${mode}.jsonl"
    echo "    ranker  -> ${CKPT_ROOT}/proposal_ranker_t5_mo_${mode}_${SLICE}/proposal_ranker_best.pt"
  done
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then usage; exit 0; fi
if [[ "${1:-}" == "--dry-run" ]]; then print_plan; exit 0; fi

export CUDA_DEVICE_ORDER
export CUDA_VISIBLE_DEVICES
export PYTHONPATH="$(dirname "${ROOT}")${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${TEACHER_DIR}" "${LOG_ROOT}"

for mode in ${MODES}; do
  teacher_jsonl="${TEACHER_DIR}/mo_teacher_${mode}.jsonl"
  teacher_json="${TEACHER_DIR}/mo_teacher_${mode}.summary.json"
  save_dir="${CKPT_ROOT}/proposal_ranker_t5_mo_${mode}_${SLICE}"
  profile="${LOG_ROOT}/proposal_ranker_t5_mo_${mode}_${SLICE}.profile.jsonl"

  echo "[$(date -Iseconds)] EXPORT teacher mode=${mode}"
  if [[ -f "${teacher_jsonl}" ]]; then
    echo "  teacher exists, skipping export: ${teacher_jsonl}"
  else
    # shellcheck disable=SC2046
    "${PYTHON_BIN}" -m mrna_editflow.baselines.multiobjective_teacher_export \
      --run-mode development \
      --records-jsonl "${SOURCES}" \
      --out-jsonl "${teacher_jsonl}" \
      --out-json "${teacher_json}" \
      --candidate-cap "${CANDIDATE_CAP}" \
      $(teacher_flags "${mode}")
  fi

  echo "[$(date -Iseconds)] DISTILL ranker mode=${mode} on GPU ${CUDA_VISIBLE_DEVICES}"
  # Skip only if a COMPLETE ranker trained under the expected pairing mode + step
  # count already exists. This guards against silently reusing a stale partial
  # checkpoint from an aborted run with a different --pair-source-mode.
  reuse_ok=0
  if [[ -f "${save_dir}/proposal_ranker_best.pt" && -f "${profile}" ]]; then
    read -r prof_mode prof_step < <(
      tail -n 1 "${profile}" | "${PYTHON_BIN}" -c \
        'import sys,json;d=json.loads(sys.stdin.read());print(d.get("pair_source_mode"), d.get("step"))' \
        2>/dev/null || echo "unknown 0"
    )
    if [[ "${prof_mode}" == "${PAIR_SOURCE_MODE}" && "${prof_step}" == "${STEPS}" ]]; then
      reuse_ok=1
    else
      echo "  stale ranker (mode=${prof_mode} step=${prof_step}, expected mode=${PAIR_SOURCE_MODE} step=${STEPS}); retraining"
      rm -rf "${save_dir}" "${profile}"
    fi
  fi
  if [[ "${reuse_ok}" -eq 1 ]]; then
    echo "  ranker complete under ${PAIR_SOURCE_MODE}/${STEPS}, skipping distill: ${save_dir}/proposal_ranker_best.pt"
  else
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${PYTHON_BIN}" -m mrna_editflow.train_proposal_ranker \
      --run-mode development \
      --records-jsonl "${SOURCES}" \
      --teacher-jsonl "${teacher_jsonl}" \
      --base-checkpoint "${BASE_CHECKPOINT}" \
      --save-dir "${save_dir}" \
      --profile-path "${profile}" \
      --steps "${STEPS}" \
      --batch-records "${BATCH_RECORDS}" \
      --max-pairs-per-record "${MAX_PAIRS}" \
      --lr "${LR}" \
      --seed "${TRAIN_SEED}" \
      --pair-source-mode "${PAIR_SOURCE_MODE}" \
      --device cuda
  fi
  echo "[$(date -Iseconds)] DONE mode=${mode}"
done

echo "[$(date -Iseconds)] ALL MODES COMPLETE"
