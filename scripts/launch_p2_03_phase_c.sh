#!/usr/bin/env bash
# P2-03 Phase C: Train additional ranker seeds (1, 2) for the 4 multiobjective
# modes (te_only, scalar, pareto, grpo). hardneg_v2 is handled separately.
#
# This wrapper invokes the existing run_multiobjective_ranker_ablation_head256.sh
# with CKPT_ROOT and TRAIN_SEED overrides, so the new seed-specific checkpoints
# don't collide with the Phase B (seed 0) checkpoints.
#
# Usage:
#   bash scripts/launch_p2_03_phase_c.sh [--dry-run]
#
# Environment overrides:
#   PHASE_C_SEEDS   (default: "1 2")         — training seeds to run
#   PHASE_C_GPUS    (default: "6")           — GPU to use
#   PHASE_C_STEPS   (default: 500)           — training steps per ranker
#   PHASE_C_SLICE   (default: head256)       — data slice
#
# *** GPU MEMORY REQUIREMENT ***
# The T5 proposal ranker training requires ~4 GB for PyTorch activations alone.
# On 2026-07-21, GPU 6 and GPU 7 both have MIG enabled (1g.5gb slices, ~5 GB each),
# AND each MIG slice is shared with 5-7 other processes. This makes OOM inevitable:
#   torch.OutOfMemoryError: GPU 0 has a total capacity of 4.75 GiB of which
#   12.81 MiB is free. Process 2540550 has 356.00 MiB ...
#
# DO NOT launch Phase C on GPU 6 or GPU 7 until either:
#   (a) MIG is disabled (requires root: `nvidia-smi -i 6 -mig 0`), OR
#   (b) A full A100 GPU becomes free (e.g., GPU 2 after P2-02 finishes, or
#       GPU 0/1/3/5 after a backbone training process finishes).
#
# The script checks GPU memory before launching and aborts if < 8 GB is available.
#
# Constraints respected:
#   - Does NOT touch GPU 4 (calibrate PID 2544995).
#   - Does NOT touch GPU 2 (P2-02 PID 265498) unless explicitly set.
#   - Does NOT delete/modify existing Phase B checkpoints.
#   - Writes to new seed-specific CKPT_ROOT: ckpts/phase_c_seed{N}/
#   - Teacher export is reused from Phase B (TEACHER_DIR unchanged).
#
# After all training completes, run the Phase B evaluation script for each
# new (mode, seed) combination to populate benchmark/dev/leakage_free_headline_phase_c/.
set -euo pipefail

ROOT="${ROOT:-/home/cunyuliu/mrna_editflow_goal/mrna_editflow}"
PHASE_C_SEEDS="${PHASE_C_SEEDS:-1 2}"
PHASE_C_GPUS="${PHASE_C_GPUS:-6}"
PHASE_C_STEPS="${PHASE_C_STEPS:-500}"
PHASE_C_SLICE="${PHASE_C_SLICE:-head256}"
MIN_GPU_MEMORY_GB="${MIN_GPU_MEMORY_GB:-8}"

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then DRY_RUN=1; fi

echo "=== P2-03 Phase C Launcher ==="
echo "ROOT=${ROOT}"
echo "SEEDS=${PHASE_C_SEEDS}"
echo "GPUS=${PHASE_C_GPUS}"
echo "STEPS=${PHASE_C_STEPS}"
echo "SLICE=${PHASE_C_SLICE}"
echo "MIN_GPU_MEMORY_GB=${MIN_GPU_MEMORY_GB}"
echo "DRY_RUN=${DRY_RUN}"
echo ""

# GPU memory check (skip in dry-run mode).
# IMPORTANT: nvidia-smi reports the FULL GPU memory, but when MIG is enabled,
# PyTorch only sees the MIG slice (~5 GB for 1g.5gb). We must check the
# PyTorch-visible memory, not the nvidia-smi memory.
if [[ "${DRY_RUN}" -eq 0 ]]; then
  echo "=== GPU memory check (PyTorch-visible) ==="
  PYTHON_BIN="${PYTHON_BIN:-/home/cunyuliu/miniconda3/envs/editflow/bin/python3.10}"
  MEM_CHECK=$(CUDA_VISIBLE_DEVICES="${PHASE_C_GPUS}" "${PYTHON_BIN}" -c "
import torch
if not torch.cuda.is_available():
    print('NO_CUDA')
else:
    props = torch.cuda.get_device_properties(0)
    print(f'{props.total_memory // (1024*1024)}')
" 2>/dev/null || echo "ERROR")
  if [[ "${MEM_CHECK}" == "NO_CUDA" || "${MEM_CHECK}" == "ERROR" || -z "${MEM_CHECK}" ]]; then
    echo "  ERROR: Cannot determine PyTorch-visible memory on GPU ${PHASE_C_GPUS}."
    echo "  CUDA might not be available. ABORTING."
    exit 1
  fi
  FREE_MB="${MEM_CHECK}"
  FREE_GB=$(awk "BEGIN{printf \"%.2f\", ${FREE_MB:-0}/1024}")
  echo "  GPU ${PHASE_C_GPUS}: PyTorch sees ${FREE_GB} GB total"
  MIN_FREE_MB=$((MIN_GPU_MEMORY_GB * 1024))
  if [[ "${FREE_MB}" -lt "${MIN_FREE_MB}" ]]; then
    echo "  ERROR: GPU ${PHASE_C_GPUS} PyTorch-visible memory is ${FREE_GB} GB (need >= ${MIN_GPU_MEMORY_GB} GB)."
    echo "  This is likely because MIG is enabled (1g.5gb slices = ~5 GB each)."
    echo "  nvidia-smi may show 40 GB, but PyTorch only sees the MIG slice."
    echo ""
    echo "  To fix:"
    echo "    1. Disable MIG on GPU ${PHASE_C_GPUS} (requires root): nvidia-smi -i ${PHASE_C_GPUS} -mig 0"
    echo "    2. Or wait for a full A100 GPU to become free."
    echo "    3. Or override: MIN_GPU_MEMORY_GB=4 PHASE_C_GPUS=<full_gpu> bash $0"
    echo ""
    echo "  ABORTING."
    exit 1
  fi
  echo "  OK: ${FREE_GB} GB PyTorch-visible >= ${MIN_GPU_MEMORY_GB} GB required."
  echo ""
fi

for seed in ${PHASE_C_SEEDS}; do
  CKPT_ROOT="${ROOT}/ckpts/phase_c_seed${seed}"
  LOG_ROOT="${ROOT}/logs/phase_c_seed${seed}"
  echo "=== Phase C seed=${seed} ==="
  echo "  CKPT_ROOT=${CKPT_ROOT}"
  echo "  LOG_ROOT=${LOG_ROOT}"
  echo "  GPU=${PHASE_C_GPUS}"

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "  [dry-run] would invoke:"
    echo "    CUDA_VISIBLE_DEVICES=${PHASE_C_GPUS} \\"
    echo "    TRAIN_SEED=${seed} \\"
    echo "    STEPS=${PHASE_C_STEPS} \\"
    echo "    SLICE=${PHASE_C_SLICE} \\"
    echo "    CKPT_ROOT=${CKPT_ROOT} \\"
    echo "    LOG_ROOT=${LOG_ROOT} \\"
    echo "    bash scripts/run_multiobjective_ranker_ablation_head256.sh"
  else
    mkdir -p "${CKPT_ROOT}" "${LOG_ROOT}"
    CUDA_VISIBLE_DEVICES="${PHASE_C_GPUS}" \
    TRAIN_SEED="${seed}" \
    STEPS="${PHASE_C_STEPS}" \
    SLICE="${PHASE_C_SLICE}" \
    CKPT_ROOT="${CKPT_ROOT}" \
    LOG_ROOT="${LOG_ROOT}" \
    bash "${ROOT}/scripts/run_multiobjective_ranker_ablation_head256.sh"
  fi
  echo "=== seed=${seed} done ==="
  echo ""
done

echo "=== Phase C training complete ==="
echo ""
echo "Next steps:"
echo "  1. For each (mode, seed) combination, run the Phase B evaluation:"
echo "     bash scripts/run_head256_ranker_fair_eval.sh \\"
echo "       --checkpoint ckpts/phase_c_seed{N}/proposal_ranker_t5_mo_{mode}_${PHASE_C_SLICE}/proposal_ranker_best.pt \\"
echo "       --out-dir benchmark/dev/leakage_free_headline_phase_c/{mode}_seed{N}"
echo "  2. Re-run the P2-03 aggregator with Phase C results."
echo "  3. Re-run P2-09 OOD stress test with Phase C results."
echo ""
echo "NOTE: hardneg_v2 Phase C training is NOT covered by this script."
echo "      It requires a separate cascade hardneg teacher training command"
echo "      (pair_source_mode=source_balanced). See"
echo "      logs/proposal_ranker_t5_cascade_hardneg_teacher_head256_20260712_221014.log"
echo "      for the original training configuration."
