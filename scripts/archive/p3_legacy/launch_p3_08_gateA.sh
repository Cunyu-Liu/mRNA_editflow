#!/bin/bash
# P3-08 Gate A: 3-seed production pilot (1000 updates, edit_budget=1)
# Uses batched oracle scoring + PrecomputedSingleEditOracle for ~30x speedup.
set -euo pipefail

cd /home/cunyuliu/mrna_editflow_goal/mrna_editflow
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pc_cng_gpu

export CUDA_VISIBLE_DEVICES=7
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

LOG_DIR="logs/p3_08"
mkdir -p "$LOG_DIR"
mkdir -p "checkpoints/p3_08_gateA"
mkdir -p "docs"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/gateA_${TIMESTAMP}.log"

echo "=== P3-08 Gate A Launch ===" | tee "$LOG_FILE"
echo "Timestamp: $(date)" | tee -a "$LOG_FILE"
echo "GPU: $CUDA_VISIBLE_DEVICES" | tee -a "$LOG_FILE"
echo "Config: 3 seeds [42,123,456], 1000 updates, edit_budget=1" | tee -a "$LOG_FILE"
echo "Batch: sources_per_batch=8, group_size=4 (32 trajectories/update)" | tee -a "$LOG_FILE"
echo "Validation: interval=100, n_trajectories=8" | tee -a "$LOG_FILE"
echo "Checkpoint: interval=200" | tee -a "$LOG_FILE"
echo "Hyperparams: lr=1e-4, beta_kl=0.3, beta_entropy=0.05, max_kl=0.15, warmup=100, stop_penalty=0.1" | tee -a "$LOG_FILE"
echo "Fixes: NaN in compute_kl_entropy_fast (index legal positions), proactive KL tier at 0.5*max_kl, MIN_COEFF=0.3, reference reset after 30 KL_SKIPs" | tee -a "$LOG_FILE"
echo "Log: $LOG_FILE" | tee -a "$LOG_FILE"
echo "==========================" | tee -a "$LOG_FILE"

python scripts/run_p3_08.py \
    --gate A \
    --benchmark-dir data/p3/benchmark \
    --output-json docs/p3_08_grpo_results_gateA.json \
    --n-updates 1000 \
    --sources-per-batch 8 \
    --group-size 4 \
    --lr 1e-4 \
    --validation-interval 100 \
    --checkpoint-interval 200 \
    --n-validation-trajectories 8 \
    --n-test-sources 24 \
    --n-train-sources 24 \
    --max-proxy 10000 \
    --device cuda \
    --save-dir checkpoints/p3_08_gateA \
    2>&1 | tee -a "$LOG_FILE"

EXIT_CODE=$?
echo "=== Gate A finished with exit code $EXIT_CODE ===" | tee -a "$LOG_FILE"
exit $EXIT_CODE
