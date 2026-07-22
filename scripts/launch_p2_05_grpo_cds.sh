#!/bin/bash
# P2-05: RL-2 GRPO pilot launcher — 3 policy seeds × 10 rollout seeds (CDS task)
#
# Waits for the P2-10 Option C 10k checkpoint to exist, then launches 3 policy
# seeds in parallel on GPUs 2, 5, 7 (avoiding GPU 0/1/3/4/6).
#
# P2-02 crashed at step 7000 (I/O error) with val_loss never improving past
# step 2000. P2-10 Option C (AMP disabled, batch=8/grad_accum=8, LR warmup
# 500 steps) was selected as the pivot. This launcher now waits for the
# P2-10 checkpoint instead of P2-02.
#
# Usage:
#   nohup bash scripts/launch_p2_05_grpo_cds.sh > logs/p2_05_grpo_cds.nohup.log 2>&1 &
#
# Environment overrides:
#   P2_05_CKPT_STAGE   (default: p2_10_option_c)  — checkpoint stage name
#   P2_05_CKPT_SEED    (default: 42)               — checkpoint training seed
#   P2_05_SEEDS        (default: "0 1 2")          — policy seeds
#   P2_05_GPUS         (default: "2 5 7")          — GPUs for policy seeds
#
# Constraints:
#   - Does NOT terminate any running process.
#   - Uses --train-idx/--val-idx/--test-idx (split contract enforced).
#   - Reward field: predicted_te_internal_proxy (Oracle #3 delta).
#   - run-mode=development (preliminary; paper mode requires paper-eligible
#     checkpoint + oracle manifest, which P2-10 10k will provide after locking).
set -uo pipefail

ROOT=/home/cunyuliu/mrna_editflow_goal/mrna_editflow
PY=/home/cunyuliu/miniconda3/envs/editflow/bin/python
RECORDS=data/reconstructed/p0_data_reconstruction_v1/combined/combined_model_view.records.jsonl
MANIFEST=benchmark/dev/p0_data_reconstruction_v1/combined_family/split_manifest.json
TRAIN_IDX=benchmark/dev/p0_data_reconstruction_v1/combined_family/train.idx
VAL_IDX=benchmark/dev/p0_data_reconstruction_v1/combined_family/val.idx
TEST_IDX=benchmark/dev/p0_data_reconstruction_v1/combined_family/test.idx
ORACLE_MANIFEST=benchmark/paper/leakage_free_headline/oracle_manifest.json

# P2-10 Option C checkpoint (pivot from P2-02 which crashed at step 7000).
P2_05_CKPT_STAGE="${P2_05_CKPT_STAGE:-p2_10_option_c}"
P2_05_CKPT_SEED="${P2_05_CKPT_SEED:-42}"
CKPT_DIR="${ROOT}/benchmark/paper/stage_a_recovery_${P2_05_CKPT_STAGE}_seed${P2_05_CKPT_SEED}"
CKPT_10K="${CKPT_DIR}/stage_a_step10000.pt"
CKPT_BEST="${CKPT_DIR}/stage_a_best.pt"

# Wait for 10k checkpoint (poll every 5 min, max 48 hours).
echo "[$(date)] Waiting for ${P2_05_CKPT_STAGE} 10k checkpoint: ${CKPT_10K}"
WAIT_SEC=0
MAX_WAIT=172800  # 48 hours
while [ ! -f "${CKPT_10K}" ]; do
    if [ ${WAIT_SEC} -ge ${MAX_WAIT} ]; then
        echo "[$(date)] ERROR: 10k checkpoint not found after ${MAX_WAIT}s. Aborting."
        exit 1
    fi
    # Check if P2-10 process is still alive.
    if ! pgrep -f "run_stage_a_recovery_p2_10" > /dev/null 2>&1; then
        echo "[$(date)] WARN: P2-10 process not found. Checking for 10k checkpoint one more time..."
        if [ ! -f "${CKPT_10K}" ]; then
            echo "[$(date)] 10k checkpoint missing and P2-10 not running. Falling back to stage_a_best.pt."
            CKPT="${CKPT_BEST}"
            break
        fi
    fi
    sleep 300
    WAIT_SEC=$((WAIT_SEC + 300))
    echo "[$(date)] Still waiting... (${WAIT_SEC}s elapsed)"
done

if [ -z "${CKPT:-}" ]; then
    CKPT="${CKPT_10K}"
fi

echo "[$(date)] Using checkpoint: ${CKPT}"
CKPT_SHA=$(${PY} -c "import hashlib; print(hashlib.sha256(open('${CKPT}','rb').read()).hexdigest())")
echo "[$(date)] Checkpoint SHA-256: ${CKPT_SHA}"

cd "${ROOT}"
export PYTHONPATH=/home/cunyuliu/mrna_editflow_goal

# Launch 3 policy seeds in parallel on GPUs 2, 5, 7.
# Each seed: 10 rollout seeds, 500 GRPO iterations, 4 groups per step.
SEEDS="${P2_05_SEEDS:-0 1 2}"
GPUS="${P2_05_GPUS:-2 5 7}"
GPU_ARR=(${GPUS})
SEED_ARR=(${SEEDS})

for i in "${!SEED_ARR[@]}"; do
    seed=${SEED_ARR[$i]}
    GPU=${GPU_ARR[$i]:-${GPU_ARR[0]}}
    OUT_DIR=benchmark/dev/grpo_pilot_preliminary/cds_seed${seed}
    LOG_STDOUT=logs/p2_05_grpo_cds_seed${seed}.stdout
    LOG_STDERR=logs/p2_05_grpo_cds_seed${seed}.stderr

    echo "[$(date)] Launching policy seed ${seed} on GPU ${GPU} → ${OUT_DIR}"

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
        --policy-seed ${seed} \
        --rollout-seeds 0 1 2 3 4 5 6 7 8 9 \
        --out-dir "${OUT_DIR}" \
        --device "cuda:${GPU}" \
        --run-mode development \
        --limit 1024 \
        > "${LOG_STDOUT}" 2> "${LOG_STDERR}" &

    echo "[$(date)]   PID=$! GPU=${GPU} LOG=${LOG_STDOUT}"
done

echo "[$(date)] All ${#SEED_ARR[@]} policy seeds launched. Monitoring..."
echo "[$(date)] Logs: logs/p2_05_grpo_cds_seed{${SEEDS// /,}}.{stdout,stderr}"
echo "[$(date)] Output: benchmark/dev/grpo_pilot_preliminary/cds_seed{${SEEDS// /,}}/"

# Wait for all to finish.
wait
echo "[$(date)] All ${#SEED_ARR[@]} policy seeds completed."
echo "[$(date)] Next: aggregate results + family-cluster bootstrap CI."
