#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/cunyuliu/mrna_editflow_goal
MEF=$ROOT/mrna_editflow
PY=/home/cunyuliu/miniconda3/envs/editflow/bin/python3.10
export PYTHONPATH=$ROOT
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SOURCES=$MEF/benchmark/multiseed_t5_public_head256_hardneg_v2_top64/sources.jsonl
RANKER=$MEF/ckpts/proposal_ranker_t5_stage_a10k_head1024_teacher/proposal_ranker_best.pt
OUT=$MEF/benchmark/multiseed_t5_public_head256_stage_a10k_ranker_top64

echo "[$(date -Is)] START head256 10k-ranker direct decode on GPU $CUDA_VISIBLE_DEVICES"
$PY -m mrna_editflow.eval.run_multiseed_benchmark   --run-mode development   --records-jsonl $SOURCES   --checkpoint $RANKER   --out-dir $OUT   --task-id T5   --seeds 0 1 2 3 4 5 6 7 8 9   --edit-budget 3   --proposal-top-k 64   --device cuda   --resume
echo "[$(date -Is)] DONE head256 10k-ranker benchmark"
