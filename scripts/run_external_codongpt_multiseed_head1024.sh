#!/usr/bin/env bash
# Run codonGPT seeds 1-9 in two GPU workers and aggregate with canonical seed0.
set -euo pipefail

ROOT="${ROOT:-/home/cunyuliu/mrna_editflow_goal/mrna_editflow}"
PYTHON_BIN="${PYTHON_BIN:-/home/cunyuliu/miniconda3/envs/editflow/bin/python3.10}"
INPUT_PACK_SUMMARY="${INPUT_PACK_SUMMARY:-${ROOT}/benchmark/external_sota/input_pack_t5_head1024/summary.json}"
EXECUTABLE="${CODONGPT_BIN:-${ROOT}/scripts/external_codongpt.sh}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/benchmark/external_sota/codongpt_multiseed_head1024}"
STATUS_JSON="${STATUS_JSON:-${OUT_ROOT}/status.json}"
GPUS="${CODONGPT_GPUS:-6 7}"
GPU_A_SEEDS="${GPU_A_SEEDS:-1 3 5 7 9}"
GPU_B_SEEDS="${GPU_B_SEEDS:-2 4 6 8}"
BATCH_SIZE="${BATCH_SIZE:-64}"

if [[ "${1:-}" == "--dry-run" ]]; then
  echo "EXTERNAL CODONGPT MULTISEED HEAD1024"
  echo "INPUT_PACK_SUMMARY=${INPUT_PACK_SUMMARY}"
  echo "EXECUTABLE=${EXECUTABLE}"
  echo "OUT_ROOT=${OUT_ROOT}"
  echo "GPUS=${GPUS}"
  echo "GPU_A_SEEDS=${GPU_A_SEEDS}"
  echo "GPU_B_SEEDS=${GPU_B_SEEDS}"
  echo "BATCH_SIZE=${BATCH_SIZE}"
  echo "CANONICAL_SEED0=benchmark/external_sota/real_runs_t5_head1024/codonGPT"
  exit 0
fi

write_status() {
  local status="$1"
  local message="$2"
  STATUS="${status}" MESSAGE="${message}" STATUS_JSON="${STATUS_JSON}" \
    "${PYTHON_BIN}" - <<'PY'
import json
import os
import time

payload = {
    "artifact_kind": "external_codongpt_multiseed_head1024_status",
    "time": time.time(),
    "status": os.environ["STATUS"],
    "message": os.environ["MESSAGE"],
    "expected_seeds": list(range(10)),
}
path = os.environ["STATUS_JSON"]
os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
with open(path, "w", encoding="utf-8") as fh:
    json.dump(payload, fh, indent=2, sort_keys=True)
PY
}

mkdir -p "${OUT_ROOT}" "${ROOT}/logs"
export PYTHONPATH="$(dirname "${ROOT}")${PYTHONPATH:+:${PYTHONPATH}}"
on_error() {
  local code=$?
  write_status "failed" "codonGPT multiseed run failed with exit ${code}"
  exit "${code}"
}
trap on_error ERR

read -r -a gpu_list <<< "${GPUS}"
if [[ "${#gpu_list[@]}" -lt 2 ]]; then
  echo "CODONGPT_GPUS must contain two GPU indices" >&2
  exit 2
fi

run_seed_group() {
  local gpu="$1"
  shift
  local seed out_dir log_path
  for seed in "$@"; do
    out_dir="${OUT_ROOT}/seed_$(printf '%03d' "${seed}")"
    log_path="${ROOT}/logs/external_codongpt_seed$(printf '%03d' "${seed}").log"
    CUDA_VISIBLE_DEVICES="${gpu}" "${EXECUTABLE}" \
      --input-pack-summary "${INPUT_PACK_SUMMARY}" \
      --executable "${EXECUTABLE}" \
      --out-dir "${out_dir}" \
      --limit 1024 \
      --batch-size "${BATCH_SIZE}" \
      --seed "${seed}" \
      --temperature 1.0 \
      --top-k 50 \
      --top-p 0.9 \
      --device cuda \
      --resume \
      --progress-jsonl "${out_dir}/progress.jsonl" > "${log_path}"
  done
}

write_status "running" "codonGPT seeds 1-9 are running on two GPU workers"
read -r -a seeds_a <<< "${GPU_A_SEEDS}"
read -r -a seeds_b <<< "${GPU_B_SEEDS}"
run_seed_group "${gpu_list[0]}" "${seeds_a[@]}" &
pid_a=$!
run_seed_group "${gpu_list[1]}" "${seeds_b[@]}" &
pid_b=$!
wait "${pid_a}"
wait "${pid_b}"

"${PYTHON_BIN}" -m mrna_editflow.eval.build_codongpt_multiseed_summary \
  --project-root "${ROOT}" \
  --multiseed-root "benchmark/external_sota/codongpt_multiseed_head1024" \
  --out-json "${OUT_ROOT}/summary.json" \
  --out-md "${OUT_ROOT}/summary.md"

write_status "complete" "codonGPT 10-seed head1024 summary completed"
trap - ERR
