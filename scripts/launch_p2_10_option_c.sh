#!/bin/bash
# P2-10 Option C: Stage A recovery with stricter fixes.
#
# Restarts the Stage A training with:
#   - AMP disabled (pure fp32 for numerical stability)
#   - batch_size=8, grad_accum=8 (effective batch 64, 4x larger than P2-02)
#   - LR warmup 500 steps (linear 0 -> 1e-6, then constant)
#   - grad_clip=1.0 (enforced, same as P2-02)
#   - eval_every=250, save_every=500 (more frequent monitoring)
#
# Triggered by P2-02 failure:
#   - Crashed at step 7000 (I/O error during torch.save)
#   - val_loss never improved past step 2000 (10174.49)
#   - 5000 steps without improvement
#   - grad_norm P99 ~9000+ (target <1000 FAILING)
#
# Constraints respected:
#   - Does NOT touch GPU 4 (calibrate — now ended, but still forbidden by convention).
#   - Does NOT terminate or modify any running process.
#   - Uses paper mode with --train-idx/--val-idx/--test-idx (split contract enforced).
#   - All new code has unit tests (tests/test_stage_a_recovery_p2_10_option_c.py).
#   - Does NOT modify train_backbone.py or scripts/run_stage_a_recovery_p2_02.py.
#
# Usage:
#   bash scripts/launch_p2_10_option_c.sh
#
# Environment overrides:
#   P2_10_GPU       (default: 0)    — GPU to use
#   P2_10_SEED      (default: 42)   — training seed
#   P2_10_STEPS     (default: 10000) — total training steps
#   P2_10_WARMUP    (default: 500)  — LR warmup steps
set -euo pipefail

ROOT="${ROOT:-/home/cunyuliu/mrna_editflow_goal/mrna_editflow}"
PY="${PY:-/home/cunyuliu/miniconda3/envs/editflow/bin/python}"
P2_10_GPU="${P2_10_GPU:-0}"
P2_10_SEED="${P2_10_SEED:-42}"
P2_10_STEPS="${P2_10_STEPS:-10000}"
P2_10_WARMUP="${P2_10_WARMUP:-500}"

RECORDS="${ROOT}/data/reconstructed/p0_data_reconstruction_v1/combined/combined_model_view.records.jsonl"
MANIFEST="${ROOT}/benchmark/dev/p0_data_reconstruction_v1/combined_family/split_manifest.json"
TRAIN_IDX="${ROOT}/benchmark/dev/p0_data_reconstruction_v1/combined_family/train.idx"
VAL_IDX="${ROOT}/benchmark/dev/p0_data_reconstruction_v1/combined_family/val.idx"
TEST_IDX="${ROOT}/benchmark/dev/p0_data_reconstruction_v1/combined_family/test.idx"
SAVE_DIR="${ROOT}/benchmark/paper/stage_a_recovery_p2_10_option_c_seed${P2_10_SEED}"
PROFILE="${ROOT}/benchmark/paper/stage_a_recovery_p2_10_option_c_seed${P2_10_SEED}.profile.jsonl"
CONFIG="${ROOT}/configs/stage_a_recovery_p2_10_option_c.json"

mkdir -p "${SAVE_DIR}"

echo "=== $(date) Launching P2-10 Option C Stage A recovery ==="
echo "=== config=${CONFIG} seed=${P2_10_SEED} steps=${P2_10_STEPS} device=cuda:${P2_10_GPU} warmup=${P2_10_WARMUP} ==="
echo "=== split_manifest=${MANIFEST} ==="
echo "=== save_dir=${SAVE_DIR} ==="
echo "=== amp=false batch_size=8 grad_accum=8 lr=1e-6 grad_clip=1.0 ==="

cd "${ROOT}"
CUDA_VISIBLE_DEVICES="${P2_10_GPU}" \
"${PY}" -m scripts.run_stage_a_recovery_p2_10_option_c \
    --config "${CONFIG}" \
    --records-jsonl "${RECORDS}" \
    --split-manifest "${MANIFEST}" \
    --split-role train \
    --train-idx "${TRAIN_IDX}" \
    --val-idx "${VAL_IDX}" \
    --test-idx "${TEST_IDX}" \
    --save-dir "${SAVE_DIR}" \
    --profile-path "${PROFILE}" \
    --steps "${P2_10_STEPS}" \
    --seed "${P2_10_SEED}" \
    --device "cuda" \
    --warmup-steps "${P2_10_WARMUP}" \
    --run-mode paper

echo "=== $(date) P2-10 Option C recovery run COMPLETE ==="
