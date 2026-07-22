#!/bin/bash
# P2-05: GRPO pilot with per-trajectory backward (gradient accumulation) fix.
#
# After the OOM fix in rl/grpo.py (_compute_loss_grad_accum), peak memory is
# bounded to ONE trajectory's forward-pass activations instead of B*N*T.
# This allows running 3 seeds in PARALLEL on separate GPUs.
#
# Config: K=8 (spec), n_groups=2 (degraded from 4), max_steps=5 (degraded from 256)
# Documented as DEGRADED in docs/p2_05_grpo_pilot.md
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

cd "${ROOT}"
export PYTHONPATH=/home/cunyuliu/mrna_editflow_goal
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p logs

SEEDS="0 1 2"
# GPUs: 0 and 1 are fully free; 2 has 3.2 GB used (37 GB avail)
GPUS="0 1 2"

PIDS=()
i=0
for seed in ${SEEDS}; do
    GPU=$(echo ${GPUS} | cut -d' ' -f$((i+1)))
    OUT_DIR=benchmark/dev/grpo_pilot_preliminary/cds_seed${seed}
    LOG_STDOUT=logs/p2_05_grpo_cds_seed${seed}.stdout
    LOG_STDERR=logs/p2_05_grpo_cds_seed${seed}.stderr

    echo "[$(date)] === Launching seed ${seed} on GPU ${GPU} (K=8, n_groups=2, max_steps=5, grad_accum fix) ==="

    CUDA_VISIBLE_DEVICES=${GPU} ${PY} -m scripts.run_p2_05_grpo_pilot \
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
        --n-groups 2 \
        --policy-seed ${seed} \
        --rollout-seeds 0 1 2 3 4 5 6 7 8 9 \
        --out-dir "${OUT_DIR}" \
        --device "cuda:0" \
        --run-mode development \
        --limit 1024 \
        --max-steps 5 \
        > "${LOG_STDOUT}" 2> "${LOG_STDERR}" &

    PIDS+=($!)
    i=$((i+1))
    sleep 5  # stagger launches to avoid concurrent CUDA init spike
done

echo "[$(date)] All ${#PIDS[@]} seeds launched. PIDs: ${PIDS[@]}"
echo "[$(date)] Waiting for completion..."

FAIL=0
for pid in "${PIDS[@]}"; do
    wait ${pid}
    EC=$?
    if [ ${EC} -ne 0 ]; then
        echo "[$(date)] ERROR: PID ${pid} exited with code ${EC}"
        FAIL=1
    else
        echo "[$(date)] PID ${pid} completed successfully"
    fi
done

if [ ${FAIL} -eq 0 ]; then
    echo "[$(date)] All seeds completed successfully."
else
    echo "[$(date)] Some seeds failed. Check logs/p2_05_grpo_cds_seed*.stderr"
fi
