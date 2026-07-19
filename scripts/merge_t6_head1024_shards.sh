#!/usr/bin/env bash
# Merge completed head1024 T6 seed shards into full 10-seed summaries.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TOP_K="${TOP_K:-64}"
EDIT_BUDGET="${EDIT_BUDGET:-30}"
CHECKPOINT="${CHECKPOINT:-${ROOT}/ckpts/stage_a_public_full_10k_bs8ga4_seed0/stage_a_best.pt}"
SOURCE_PATH="${SOURCE_PATH:-${ROOT}/benchmark/multiseed_t5_public_head1024_hardneg_v2_top64/sources.jsonl}"
MERGE_TAG="${MERGE_TAG:-merged_$(date +%Y%m%d_%H%M%S)}"
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage:
  merge_t6_head1024_shards.sh [--dry-run]

Scans standard, non-standard parallel, and benchmark/t6_shards head1024 T6
directories. When a target delta has completed seed_000..seed_009 artifacts,
it writes benchmark/multiseed_t6_public_head1024_stagea10k_len_<delta>_<tag>_top64/multiseed_summary.json
with the same aggregation math as run_multiseed_benchmark.
If a complete 10-seed summary already exists for a delta, it skips that delta
to keep repeated polling idempotent.

Environment overrides:
  ROOT, PYTHON_BIN, TOP_K, EDIT_BUDGET, CHECKPOINT, SOURCE_PATH, MERGE_TAG
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then usage; exit 0; fi
if [[ "${1:-}" == "--dry-run" ]]; then DRY_RUN=1; fi

export PYTHONPATH="$(dirname "${ROOT}")${PYTHONPATH:+:${PYTHONPATH}}"
shopt -s nullglob

delta_name() {
  local delta="$1"
  if [[ "${delta}" == -* ]]; then
    echo "neg${delta#-}"
  else
    echo "pos${delta}"
  fi
}

collect_source_dirs() {
  local name="$1"
  local standard="${ROOT}/benchmark/multiseed_t6_public_head1024_stagea10k_len_${name}_top${TOP_K}"
  if [[ -d "${standard}" ]]; then
    printf '%s\n' "${standard}"
  fi
  local d
  for d in "${ROOT}/benchmark"/multiseed_t6_public_head1024_stagea10k_len_"${name}"_*_top"${TOP_K}"; do
    if [[ -d "${d}" && "${d}" != "${standard}" ]]; then
      printf '%s\n' "${d}"
    fi
  done
  for d in "${ROOT}/benchmark/t6_shards"/head1024_stagea10k_len_"${name}"_*_top"${TOP_K}"; do
    if [[ -d "${d}" ]]; then
      printf '%s\n' "${d}"
    fi
  done
}

find_complete_summary() {
  local name="$1"
  local candidates_text=""
  local standard="${ROOT}/benchmark/multiseed_t6_public_head1024_stagea10k_len_${name}_top${TOP_K}/multiseed_summary.json"
  if [[ -f "${standard}" ]]; then
    candidates_text="${candidates_text}${standard}"$'\n'
  fi
  local d
  for d in "${ROOT}/benchmark"/multiseed_t6_public_head1024_stagea10k_len_"${name}"_*_top"${TOP_K}"; do
    if [[ -f "${d}/multiseed_summary.json" ]]; then
      candidates_text="${candidates_text}${d}/multiseed_summary.json"$'\n'
    fi
  done
  if [[ -z "${candidates_text}" ]]; then
    return 1
  fi
  CANDIDATES="${candidates_text}" "${PYTHON_BIN}" - <<'PY'
import json
import os

EXPECTED = list(range(10))
SUMMARY_METRICS = (
    "mean_abs_length_error",
    "legal_fraction",
    "mean_protein_identity",
    "within_budget_fraction",
    "reading_frame_intact_fraction",
    "delta_oracle_te_vs_source",
    "mean_oracle_te",
    "mean_edit_distance",
)


def has_complete_summary(path: str) -> bool:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            summary = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(summary, dict):
        return False
    config = summary.get("config", {})
    if not isinstance(config, dict):
        return False
    try:
        if sorted(int(seed) for seed in config.get("seeds", [])) != EXPECTED:
            return False
    except (TypeError, ValueError):
        return False
    per_seed = summary.get("per_seed", [])
    if not isinstance(per_seed, list):
        return False
    found = []
    for row in per_seed:
        if not isinstance(row, dict):
            return False
        try:
            found.append(int(row.get("seed")))
        except (TypeError, ValueError):
            return False
    if sorted(found) != EXPECTED:
        return False
    aggregate = summary.get("aggregate", {})
    if not isinstance(aggregate, dict):
        return False
    for metric in SUMMARY_METRICS:
        entry = aggregate.get(metric, {})
        if not isinstance(entry, dict):
            return False
        try:
            if int(entry.get("n", 0)) < len(EXPECTED):
                return False
        except (TypeError, ValueError):
            return False
    return True


for candidate in os.environ.get("CANDIDATES", "").splitlines():
    if candidate and has_complete_summary(candidate):
        print(candidate)
        break
PY
}

run_one_delta() {
  local delta="$1"
  local name
  name="$(delta_name "${delta}")"
  local source_dirs_text
  source_dirs_text="$(collect_source_dirs "${name}" || true)"
  local source_count=0
  local out_dir="${ROOT}/benchmark/multiseed_t6_public_head1024_stagea10k_len_${name}_${MERGE_TAG}_top${TOP_K}"
  local cmd=(
    "${PYTHON_BIN}" -m mrna_editflow.eval.merge_multiseed_shards
    --out-dir "${out_dir}"
    --source-path "${SOURCE_PATH}"
    --task-id T6
    --checkpoint "${CHECKPOINT}"
    --edit-budget "${EDIT_BUDGET}"
    --target-length-delta "${delta}"
    --proposal-top-k "${TOP_K}"
    --cascade-recall-top-k "${TOP_K}"
    --proposal-temperature 1.0
    --n-bootstrap 1000
    --max-novelty-sources 0
  )
  local source_dir
  while IFS= read -r source_dir; do
    if [[ -z "${source_dir}" ]]; then
      continue
    fi
    source_count=$((source_count + 1))
    cmd+=(--source-dir "${source_dir}")
  done <<< "${source_dirs_text}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "merge_multiseed_shards delta=${delta} sources=${source_count} out=${out_dir}"
    printf 'command ->'
    printf ' %q' "${cmd[@]}"
    printf '\n'
    return 0
  fi
  if [[ "${source_count}" == "0" ]]; then
    echo "SKIP delta=${delta}: no source directories"
    return 0
  fi
  local existing_summary
  existing_summary="$(find_complete_summary "${name}" || true)"
  if [[ -n "${existing_summary}" ]]; then
    echo "SKIP delta=${delta}: already complete summary ${existing_summary}"
    return 0
  fi
  local output
  if output="$("${cmd[@]}" 2>&1)"; then
    echo "MERGED delta=${delta}: ${output}"
  else
    reason="$(printf '%s\n' "${output}" | tail -n 1)"
    echo "SKIP delta=${delta}: ${reason}"
  fi
}

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "T6 HEAD1024 SHARD MERGE"
  echo "ROOT=${ROOT}"
  echo "MERGE_TAG=${MERGE_TAG}"
  echo "SOURCE_PATH=${SOURCE_PATH}"
  echo "CHECKPOINT=${CHECKPOINT}"
fi

for delta in -30 -15 15 30; do
  run_one_delta "${delta}"
done
