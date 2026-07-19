#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/cunyuliu/mrna_editflow_goal
MEF=$ROOT/mrna_editflow
PY=/home/cunyuliu/miniconda3/envs/editflow/bin/python3.10
export PYTHONPATH=$ROOT
B=$MEF/benchmark
NEW=$B/multiseed_t5_public_head256_stage_a10k_ranker_top64/multiseed_summary.json
HN=$B/multiseed_t5_public_head256_hardneg_v2_top64/multiseed_summary.json
F1=$B/multiseed_t5_public_head256_ranker_full1k_top32/multiseed_summary.json
$PY -m mrna_editflow.eval.compare_benchmarks   --baseline hardneg_v2_top64=$HN   --run stage_a10k_ranker_top64=$NEW   --metrics delta_oracle_te_vs_source mean_oracle_te   --out-json $B/compare_stage_a10k_ranker_vs_hardneg_v2_head256.json   --out-md $B/compare_stage_a10k_ranker_vs_hardneg_v2_head256.md
echo COMPARE_HN_DONE
