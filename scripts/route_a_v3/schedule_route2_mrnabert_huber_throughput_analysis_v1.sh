#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/cunyuliu/mrna_editflow_goal/worktrees/route_a_v3_route2_method_repair_20260817}"
ROUTE2_ROOT="${ROUTE2_ROOT:-/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2}"
PYTHON="${PYTHON:-/home/cunyuliu/miniconda3/envs/editflow/bin/python}"
POLL_SECONDS="${POLL_SECONDS:-900}"

HUBER_SUMMARY="${ROUTE2_ROOT}/runs/mrnabert_scaleup_v2/max_mean_only_seed20260816_gpu0_bf16_v1/training_summary.json"
GPU_BENCHMARK="${ROUTE2_ROOT}/benchmarks/mrnabert_gpu_training_benchmark_v4_full_a100_gpu0/benchmark_summary.json"
DATALOADER_BENCHMARK="${ROUTE2_ROOT}/benchmarks/mrnabert_dataloader_benchmark_v1_gpu0/benchmark_summary.json"
OUTPUT="${ROUTE2_ROOT}/comparisons/mrnabert_huber_throughput_bottleneck_v1.json"

while [[ ! -f "${HUBER_SUMMARY}" || ! -f "${DATALOADER_BENCHMARK}" ]]; do
  printf '%s waiting_for_huber_and_dataloader_benchmark\n' "$(date -Is)"
  sleep "${POLL_SECONDS}"
done

if [[ -e "${OUTPUT}" ]]; then
  printf 'throughput analysis output already exists: %s\n' "${OUTPUT}" >&2
  exit 1
fi

cd "${REPO_ROOT}"
"${PYTHON}" scripts/route_a_v3/analyze_route2_mrnabert_huber_throughput_v1.py \
  --huber-summary "${HUBER_SUMMARY}" \
  --gpu-benchmark "${GPU_BENCHMARK}" \
  --dataloader-benchmark "${DATALOADER_BENCHMARK}" \
  --output "${OUTPUT}"
