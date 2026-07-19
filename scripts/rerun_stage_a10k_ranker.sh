#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/cunyuliu/mrna_editflow_goal
MEF=$ROOT/mrna_editflow
PY=/home/cunyuliu/miniconda3/envs/editflow/bin/python3.10
export PYTHONPATH=$ROOT
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SNAP=$MEF/ckpts/stage_a_public_full_10k_bs8ga4_seed0/stage_a_best_10k_final_for_proposal_ranker.pt
RECORDS=$MEF/data/processed/gencode_human_transcripts.records.jsonl
TEACHER=$MEF/benchmark/proposal_ranking_t5_stage_a10k_head1024.candidates.jsonl
RDIR=$MEF/ckpts/proposal_ranker_t5_stage_a10k_head1024_teacher
RPROF=$MEF/logs/proposal_ranker_t5_stage_a10k_head1024_teacher.profile.jsonl
RAUDIT=$MEF/benchmark/proposal_ranking_t5_ranker_stage_a10k_head64.json
RAUDITJSONL=$MEF/benchmark/proposal_ranking_t5_ranker_stage_a10k_head64.candidates.jsonl

echo "[$(date -Is)] START ranker distill on GPU $CUDA_VISIBLE_DEVICES"
$PY -m mrna_editflow.train_proposal_ranker   --run-mode development   --records-jsonl $RECORDS   --teacher-jsonl $TEACHER   --base-checkpoint $SNAP   --save-dir $RDIR   --profile-path $RPROF   --steps 500   --batch-records 4   --max-pairs-per-record 32   --lr 2e-5   --device cuda   --seed 0
echo "[$(date -Is)] ranker distill DONE"

$PY -m mrna_editflow.eval.proposal_ranking   --records-jsonl $RECORDS   --checkpoint $RDIR/proposal_ranker_best.pt   --task-id T5   --limit 64   --candidate-cap 0   --top-k 32   --device cuda   --out-json $RAUDIT   --out-jsonl $RAUDITJSONL
echo "[$(date -Is)] ranker head64 audit DONE"
