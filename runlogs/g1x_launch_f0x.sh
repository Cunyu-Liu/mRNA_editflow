#!/usr/bin/env bash
# G1-X launcher: produce frozen F0-X base flow checkpoint (GPU 3, forbidden GPU4 avoided)
set -e
WORKROOT=/home/cunyuliu/mrna_editflow_goal/worktrees/xeditflow_migration_20260806T024650Z
cd "$WORKROOT"
export PYTHONPATH="$WORKROOT"
mkdir -p runlogs
exec /home/cunyuliu/miniconda3/envs/editflow/bin/python -u -m scripts.f0x.run_f0x \
  --dataset artifacts/b0x/effect_dataset.jsonl \
  --out-dir artifacts/f0x \
  --benchmark 5U-A1 \
  --gpu cuda:3 \
  --ckpt artifacts/f0x/f0x_base_flow_5U-A1.pt