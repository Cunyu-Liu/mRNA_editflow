#!/bin/bash
# P2-05 seed 2 re-launch on GPU 0 (mitigation for GPU 7 MIG limit)
#
# GPU 7 has MIG enabled (4.75 GB visible to PyTorch), too small for GRPO pilot.
# This script re-launches seed 2 on GPU 0 after P2-10 completes and frees it.
#
# Usage:
#   bash scripts/relaunch_p2_05_seed2_gpu0.sh
#
# Prerequisites:
#   - P2-10 training has completed (GPU 0 is free)
#   - stage_a_step10000.pt exists (or stage_a_best.pt as fallback)
#   - Seeds 0 and 1 are running or completed on GPUs 2 and 5

set -uo pipefail

ROOT=/home/cunyuliu/mrna_editflow_goal/mrna_editflow
PY=/home/cunyuliu/miniconda3/envs/editflow/bin/python
RECORDS=data/reconstructed/p0_data_reconstruction_v1/combined/combined_model_view.records.jsonl
MANIFEST=benchmark/dev/p0_data_reconstruction_v1/combined_family/split_manifest.json
TRAIN_IDX=benchmark/dev/p0_data_reconstruction_v1/combined_family/train.idx
VAL_IDX=benchmark/dev/p0_data_reconstruction_v1/combined_family/val.idx
TEST_IDX=benchmark/dev/p0_data_reconstruction_v1/combined_family/test.idx
ORACLE_MANIFEST=benchmark/paper/leakage_free_headline/oracle_manifest.json

CKPT_DIR="${ROOT}/benchmark/paper/stage_a_recovery_p2_10_option_c_seed42"
CKPT_10K="${CKPT_DIR}/stage_a_step10000.pt"
CKPT_BEST="${CKPT_DIR}/stage_a_best.pt"

# Select checkpoint: prefer 10k, fall back to best
if [ -f "${CKPT_10K}" ]; then
    CKPT="${CKPT_10K}"
elif [ -f "${CKPT_BEST}" ]; then
    CKPT="${CKPT_BEST}"
    echo "WARN: 10k checkpoint not found, using stage_a_best.pt"
else
    echo "ERROR: No checkpoint found in ${CKPT_DIR}"
    exit 1
fi

echo "[$(date)] Using checkpoint: ${CKPT}"
CKPT_SHA=$(${PY} -c "import hashlib; print(hashlib.sha256(open('${CKPT}','rb').read()).hexdigest())")
echo "[$(date)] Checkpoint SHA-256: ${CKPT_SHA}"

# Check if seed 2 is already running
if pgrep -f "run_p2_05_grpo_pilot.*--policy-seed 2" > /dev/null 2>&1; then
    echo "[$(date)] Seed 2 is already running. Aborting re-launch."
    exit 0
fi

# Check if seed 2 already completed
OUT_DIR=benchmark/dev/grpo_pilot_preliminary/cds_seed2
if [ -f "${ROOT}/${OUT_DIR}/run_metadata.json" ]; then
    echo "[$(date)] Seed 2 already completed (run_metadata.json exists). Aborting re-launch."
    exit 0
fi

# Check GPU 0 is free (P2-10 completed)
GPU0_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 2>/dev/null | tr -d ' ')
if [ "${GPU0_MEM:-0}" -gt 5000 ]; then
    echo "[$(date)] WARN: GPU 0 has ${GPU0_MEM} MiB used. P2-10 may still be running."
    echo "[$(date)] Proceeding anyway (seed 2 may fail with OOM if GPU 0 is not actually free)."
fi

cd "${ROOT}"
export PYTHONPATH=/home/cunyuliu/mrna_editflow_goal

LOG_STDOUT=logs/p2_05_grpo_cds_seed2_gpu0.stdout
LOG_STDERR=logs/p2_05_grpo_cds_seed2_gpu0.stderr

echo "[$(date)] Re-launching policy seed 2 on GPU 0 → ${OUT_DIR}"

nohup ${PY} -m scripts.run_p2_05_grpo_pilot \
    --checkpoint "${CKPT}" \
    --checkpoint-sha256 "${CKPT_SHA}" \
    --oracle-manifest "${ORACLE_MANIFEST}" \
    --records-jsonl "${RECORDS}" \
    --split-manifest "${MANIFEST}" \
    --split-role train \
    --train-idx "${TRAIN_IDX}" \
    --val-idx "${VAL_IDX}" \
    --test-idx "${TEST_IDX}" \
    --task cds \
    --group-size 8 \
    --kl-coef 0.05 \
    --entropy-coef 0.01 \
    --lr 0.01 \
    --n-iter 500 \
    --n-groups 4 \
    --policy-seed 2 \
    --rollout-seeds 0 1 2 3 4 5 6 7 8 9 \
    --out-dir "${OUT_DIR}" \
    --device "cuda:0" \
    --run-mode development \
    --limit 1024 \
    > "${LOG_STDOUT}" 2> "${LOG_STDERR}" &

PID=$!
echo "[$(date)]   PID=${PID} GPU=0 LOG=${LOG_STDOUT}"
echo "[$(date)] Monitor: tail -f ${LOG_STDOUT} ${LOG_STDERR}"
echo "[$(date)] Output: ${OUT_DIR}/"
