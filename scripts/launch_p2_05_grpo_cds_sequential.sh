#!/bin/bash
# P2-05: Sequential GRPO pilot launcher — 3 policy seeds on GPU 0 (one at a time)
#
# Previous parallel launch on GPU 0 caused OOM (each seed needs ~40 GB).
# This script runs seeds 0, 1, 2 sequentially on GPU 0.
#
# Usage:
#   nohup bash scripts/launch_p2_05_grpo_cds_sequential.sh > logs/p2_05_grpo_cds_sequential.nohup.log 2>&1 &
set -uo pipefail

ROOT=/home/cunyuliu/mrna_editflow_goal/mrna_editflow
PY=/home/cunyuliu/miniconda3/envs/editflow/bin/python
RECORDS=data/reconstructed/p0_data_reconstruction_v1/combined/combined_model_view.records.jsonl
MANIFEST=benchmark/dev/p0_data_reconstruction_v1/combined_family/split_manifest.json
TRAIN_IDX=benchmark/dev/p0_data_reconstruction_v1/combined_family/train.idx
VAL_IDX=benchmark/dev/p0_data_reconstruction_v1/combined_family/val.idx
TEST_IDX=benchmark/dev/p0_data_reconstruction_v1/combined_family/test.idx
ORACLE_MANIFEST=benchmark/paper/leakage_free_headline/oracle_manifest.json

CKPT="${ROOT}/benchmark/paper/stage_a_recovery_p2_10_option_c_seed42/stage_a_step10000.pt"
CKPT_SHA="4e5e7b500882af65989b65f460d1b659315ca7dae9bb083447877e5f1aea48dd"
GPU=0

cd "${ROOT}"
export PYTHONPATH=/home/cunyuliu/mrna_editflow_goal

SEEDS="0 1 2"

for seed in ${SEEDS}; do
    OUT_DIR=benchmark/dev/grpo_pilot_preliminary/cds_seed${seed}
    LOG_STDOUT=logs/p2_05_grpo_cds_seed${seed}.stdout
    LOG_STDERR=logs/p2_05_grpo_cds_seed${seed}.stderr

    echo "[$(date)] === Launching seed ${seed} on GPU ${GPU} ==="

    ${PY} -m scripts.run_p2_05_grpo_pilot \
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
        --policy-seed ${seed} \
        --rollout-seeds 0 1 2 3 4 5 6 7 8 9 \
        --out-dir "${OUT_DIR}" \
        --device "cuda:${GPU}" \
        --run-mode development \
        --limit 1024 \
        > "${LOG_STDOUT}" 2> "${LOG_STDERR}"

    EXIT_CODE=$?
    echo "[$(date)] Seed ${seed} exited with code ${EXIT_CODE}"

    if [ ${EXIT_CODE} -ne 0 ]; then
        echo "[$(date)] ERROR: Seed ${seed} failed. Check ${LOG_STDERR}"
        echo "[$(date)] Continuing to next seed..."
    fi

    # Brief pause between seeds to release GPU memory
    sleep 10
done

echo "[$(date)] All seeds completed."
echo "[$(date)] Next: aggregate results + family-cluster bootstrap CI."
