#!/bin/bash
# P2-03 Phase C evaluation: Run multiseed benchmark for 4 modes × 2 training seeds.
# Each eval: 10 decoder seeds × 1024 sources on the frozen combined_family test split.
#
# Usage:
#   bash scripts/launch_p2_03_phase_c_eval.sh
#
# Environment overrides:
#   PHASE_C_EVAL_GPU  (default: 2)  — GPU to use
#   PHASE_C_EVAL_SEEDS (default: "1 2") — training seeds to evaluate
#
# Output:
#   benchmark/dev/leakage_free_headline_preliminary/{mode}_seed{N}/
#
# Timing: ~1 hour per eval, 8 evals total = ~8 hours sequential.
# This script is idempotent: it skips evals that already have a complete
# multiseed_summary.json with the correct training_seed.
set -uo pipefail

ROOT=/home/cunyuliu/mrna_editflow_goal/mrna_editflow
PY=/home/cunyuliu/miniconda3/envs/editflow/bin/python
RECORDS=data/reconstructed/p0_data_reconstruction_v1/combined/combined_model_view.records.jsonl
MANIFEST=benchmark/dev/p0_data_reconstruction_v1/combined_family/split_manifest.json
TRAIN_IDX=benchmark/dev/p0_data_reconstruction_v1/combined_family/train.idx
VAL_IDX=benchmark/dev/p0_data_reconstruction_v1/combined_family/val.idx
TEST_IDX=benchmark/dev/p0_data_reconstruction_v1/combined_family/test.idx
OUT_ROOT=benchmark/dev/leakage_free_headline_preliminary
LIMIT=1024
DECODER_SEEDS="0 1 2 3 4 5 6 7 8 9"
EDIT_BUDGET=3
TOP_K=64
N_BOOTSTRAP=1000

GPU="${PHASE_C_EVAL_GPU:-2}"
TRAIN_SEEDS="${PHASE_C_EVAL_SEEDS:-1 2}"

cd "${ROOT}"
export PYTHONPATH=/home/cunyuliu/mrna_editflow_goal
mkdir -p "${OUT_ROOT}"

MODES="te_only scalar pareto grpo"

run_eval() {
    local mode=$1
    local train_seed=$2
    local ckpt="ckpts/phase_c_seed${train_seed}/proposal_ranker_t5_mo_${mode}_head256/proposal_ranker_best.pt"
    local out_dir="${OUT_ROOT}/${mode}_seed${train_seed}"

    # Skip if already complete
    local summary="${out_dir}/multiseed_summary.json"
    if [ -f "${summary}" ]; then
        local existing_seed=$("${PY}" -c "
import json
with open('${summary}') as f:
    d = json.load(f)
sv = d.get('scientific_validity', d.get('provenance', {})).get('scientific_validity', d.get('scientific_validity', {}))
ts = sv.get('training_seed', sv.get('config', {}).get('training_seed', 'MISSING'))
print(ts)
" 2>/dev/null || echo "PARSE_FAIL")
        if [ "${existing_seed}" == "${train_seed}" ]; then
            echo "=== [SKIP] ${mode} seed${train_seed} already complete (training_seed=${existing_seed}) ==="
            return 0
        fi
    fi

    if [ ! -f "${ckpt}" ]; then
        echo "=== [ERROR] checkpoint missing: ${ckpt} ==="
        return 1
    fi

    echo "=== $(date -Iseconds) START ${mode} seed${train_seed} on GPU ${GPU} ==="
    CUDA_VISIBLE_DEVICES="${GPU}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        "${PY}" -m mrna_editflow.eval.run_multiseed_benchmark \
        --run-mode development \
        --records-jsonl "${RECORDS}" \
        --checkpoint "${ckpt}" \
        --task-id T5 \
        --edit-budget "${EDIT_BUDGET}" \
        --proposal-top-k "${TOP_K}" \
        --cascade-recall-top-k "${TOP_K}" \
        --proposal-temperature 1.0 \
        --seeds ${DECODER_SEEDS} \
        --training-seed "${train_seed}" \
        --n-bootstrap "${N_BOOTSTRAP}" \
        --split-manifest "${MANIFEST}" \
        --split-role test \
        --train-idx "${TRAIN_IDX}" \
        --val-idx "${VAL_IDX}" \
        --test-idx "${TEST_IDX}" \
        --out-dir "${out_dir}" \
        --device cuda \
        --limit "${LIMIT}" \
        2>"${ROOT}/logs/p2_03_${mode}_seed${train_seed}.stderr" \
        >"${ROOT}/logs/p2_03_${mode}_seed${train_seed}.stdout"
    echo "=== $(date -Iseconds) DONE ${mode} seed${train_seed} (exit=$?) ==="
}

echo "=== P2-03 Phase C Evaluation Launcher ==="
echo "GPU=${GPU}  TRAIN_SEEDS=${TRAIN_SEEDS}  MODES=${MODES}"
echo "Output: ${OUT_ROOT}/{mode}_seed{N}/"
echo

for train_seed in ${TRAIN_SEEDS}; do
    for mode in ${MODES}; do
        run_eval "${mode}" "${train_seed}"
    done
done

echo "=== $(date -Iseconds) ALL PHASE C EVALUATIONS COMPLETE ==="
