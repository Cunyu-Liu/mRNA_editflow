#!/usr/bin/env bash
# Read-only remote status snapshot for the active SOTA queue.
#
# This script does not start, stop, or modify remote jobs. It records the load
# gate, watcher PIDs, recent logs, and target artifact presence into local
# docs/remote_execution_status.{json,md}.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_ROOT="${LOCAL_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
REMOTE_HOST="${REMOTE_HOST:-cunyuliu@36.137.135.49}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/cunyuliu/mrna_editflow_goal/mrna_editflow}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SLICE="${SLICE:-head256}"
TOP_K="${TOP_K:-64}"
WATCH_PID="${WATCH_PID:-3486627}"
GC_SWEEP_PID="${GC_SWEEP_PID:-1842634}"
MAX_LOADAVG="${MAX_LOADAVG:-80}"
OUT_JSON="${OUT_JSON:-${LOCAL_ROOT}/docs/remote_execution_status.json}"
OUT_MD="${OUT_MD:-${LOCAL_ROOT}/docs/remote_execution_status.md}"

usage() {
  cat <<'EOF'
Usage:
  check_remote_sota_status.sh [--dry-run]

Collects a read-only remote execution snapshot for the SOTA queue and writes
docs/remote_execution_status.{json,md}. It does not run any benchmark.

Environment overrides:
  LOCAL_ROOT, REMOTE_HOST, REMOTE_ROOT, SLICE, TOP_K, WATCH_PID, GC_SWEEP_PID,
  MAX_LOADAVG, PYTHON_BIN, OUT_JSON, OUT_MD
EOF
}

target_list() {
  cat <<EOF
benchmark/region_adapter_vs_hardneg_v2_top${TOP_K}_${SLICE}.json
benchmark/region_adapter_vs_mo_grpo_top${TOP_K}_${SLICE}.json
benchmark/region_adapter_vs_mo_scalar_top${TOP_K}_${SLICE}.json
benchmark/region_adapter_vs_mo_pareto_top${TOP_K}_${SLICE}.json
benchmark/region_adapter_vs_mo_te_only_top${TOP_K}_${SLICE}.json
benchmark/region_adapter_decision_report_${SLICE}.json
benchmark/region_adapter_result_audit_${SLICE}.json
benchmark/protein_conditioned_cds_gc_sweep_${SLICE}.summary.json
benchmark/protein_conditioned_cds_gc_sweep_${SLICE}.audit.json
benchmark/protein_conditioned_codon_metrics_${SLICE}.json
benchmark/protein_conditioned_codon_metrics_${SLICE}.md
benchmark/protein_conditioned_t4_head1024/status.json
benchmark/protein_conditioned_t4_head1024/status.md
benchmark/protein_conditioned_t4_head1024/progress.jsonl
benchmark/codon_lattice_dp_head1024.json
benchmark/protein_conditioned_cds_head1024.summary.json
benchmark/protein_conditioned_codon_metrics_head1024.json
benchmark/protein_conditioned_codon_metrics_head1024.md
benchmark/protein_conditioned_cds_gc_sweep_head1024.summary.json
benchmark/protein_conditioned_cds_gc_sweep_head1024.audit.json
benchmark/t4_protein_identity_cai_gc_report_head1024.json
benchmark/t4_protein_identity_cai_gc_report_head1024.md
benchmark/multiseed_t5_public_head256_mo_te_only_top64/multiseed_summary.json
benchmark/multiseed_t5_public_head256_mo_scalar_top64/multiseed_summary.json
benchmark/multiseed_t5_public_head256_mo_pareto_top64/multiseed_summary.json
benchmark/multiseed_t5_public_head256_mo_grpo_top64/multiseed_summary.json
benchmark/multiseed_t5_public_head256_hardneg_v2_top64/multiseed_summary.json
benchmark/multiseed_t5_public_head1024_mo_te_only_top64/multiseed_summary.json
benchmark/multiseed_t5_public_head1024_mo_scalar_top64/multiseed_summary.json
benchmark/multiseed_t5_public_head1024_mo_pareto_top64/multiseed_summary.json
benchmark/multiseed_t5_public_head1024_mo_grpo_top64/multiseed_summary.json
benchmark/multiseed_t5_public_head1024_hardneg_v2_top64/multiseed_summary.json
benchmark/t1_t7_evidence_status_head256.json
benchmark/multi_scale_sequence_spectrum_head32_ranker_full1k.json
benchmark/multi_scale_sequence_spectrum_head32_ranker_full1k.md
benchmark/multi_scale_sequence_spectrum_head32_ranker_full1k_figures/base_composition_full.svg
benchmark/multi_scale_sequence_spectrum_head32_ranker_full1k_figures/base_composition_five_utr.svg
benchmark/multi_scale_sequence_spectrum_head32_ranker_full1k_figures/base_composition_cds.svg
benchmark/multi_scale_sequence_spectrum_head32_ranker_full1k_figures/base_composition_three_utr.svg
benchmark/multi_scale_sequence_spectrum_head32_ranker_full1k_figures/length_histogram.svg
benchmark/multi_scale_sequence_spectrum_head32_ranker_full1k_figures/gc_histogram.svg
benchmark/multi_scale_sequence_spectrum_head32_ranker_full1k_figures/kmer_top_delta.svg
benchmark/multi_scale_sequence_spectrum_head32_ranker_full1k_figures/codon_pair_top_delta.svg
benchmark/external_sota/input_pack_t5_head1024/summary.json
benchmark/external_sota/input_pack_t5_head1024/table.md
benchmark/external_sota/input_pack_t5_head1024/cds_protein_inputs.jsonl
benchmark/external_sota/input_pack_t5_head1024/utr5_inputs.jsonl
benchmark/external_sota/input_pack_t5_head1024/metric_schema.json
docs/external_sota_real_run_audit.json
docs/external_sota_real_run_audit.md
benchmark/external_sota/real_runs_t5_head1024/LinearDesign.status.json
benchmark/external_sota/real_runs_t5_head1024/LinearDesign.status.md
benchmark/external_sota/real_runs_t5_head1024/LinearDesign.progress.jsonl
benchmark/external_sota/real_runs_t5_head1024/LinearDesign/summary.json
benchmark/external_sota/real_runs_t5_head1024/LinearDesign/cds_outputs.jsonl
scripts/external_lineardesign.sh
scripts/external_ensembledesign.sh
scripts/external_utrgan.sh
scripts/run_external_lineardesign_head1024.sh
docs/t4_external_cds_baseline_comparison.json
docs/t4_external_cds_baseline_comparison.md
external_tools/UTRGAN/environment.yml
external_tools/UTRGAN/models/checkpoint_3000.h5
logs/utrgan_env_create.log
logs/utrgan_env_create.pid
logs/utrgan_cf_env_create.log
logs/utrgan_cf_env_create.pid
logs/utrgan_cf_deps_install.log
logs/utrgan_cf_deps_install.pid
benchmark/edit_budget_curve_report_head256_head1024.json
benchmark/t6_length_curve_report_head256_head1024.json
benchmark/region_adapter_decision_report_head1024.json
benchmark/region_adapter_result_audit_head1024.json
docs/multiobjective_scaleup_claim_audit_head256_head1024.json
docs/sota_readiness_audit_${SLICE}.json
docs/sota_gap_report.json
configs/stage_a_full_a100_max.json
scripts/run_stage_a_a100_max_train.sh
scripts/run_after_stage_a_a100_max.sh
scripts/run_protein_conditioned_t4_slice.sh
docs/reproduce_full_training_eval_commands.md
benchmark/stage_a_full_a100_max_gencode_100k_seed0/status.json
benchmark/stage_a_full_a100_max_gencode_100k_seed0/status.md
benchmark/stage_a_full_a100_max_gencode_100k_seed0/progress.jsonl
benchmark/stage_a_full_a100_max_gencode_100k_seed0_posteval/status.json
benchmark/stage_a_full_a100_max_gencode_100k_seed0_posteval/status.md
benchmark/stage_a_full_a100_max_gencode_100k_seed0_posteval/progress.jsonl
logs/stage_a_full_a100_max_gencode_100k_seed0.metadata.json
logs/stage_a_full_a100_max_gencode_100k_seed0.profile.jsonl
logs/stage_a_full_a100_max_gencode_100k_seed0.train.log
docs/data_scaleup_readiness.json
docs/data_scaleup_readiness.md
docs/stage_a_downstream_eval_readiness.json
docs/stage_a_downstream_eval_readiness.md
docs/stage_a_scalelaw_downstream_eval_summary.json
docs/stage_a_scalelaw_downstream_trend_audit.json
benchmark/stage_a_scalelaw_downstream/status.json
benchmark/stage_a_scalelaw_downstream/status.md
benchmark/stage_a_scalelaw_downstream/progress.jsonl
docs/downstream_data_acquisition_audit.json
docs/downstream_data_acquisition_audit.md
docs/dataset_manifest_audit.json
docs/dataset_manifest_audit.md
logs/p3_readiness_watcher.state.json
logs/p3_readiness_watcher.log
benchmark/downstream_predictor_protocol/status.json
benchmark/downstream_predictor_protocol/status.md
benchmark/downstream_predictor_protocol/progress.jsonl
benchmark/mpra_te_predictor_protocol_real/report.json
benchmark/mpra_te_predictor_protocol_real/report.md
benchmark/mpra_te_predictor_protocol_real/predictions.jsonl
benchmark/stability_predictor_protocol_real/report.json
benchmark/stability_predictor_protocol_real/report.md
benchmark/stability_predictor_protocol_real/predictions.jsonl
benchmark/mpra_te_predictor_protocol_smoke/report.json
benchmark/mpra_te_predictor_protocol_smoke/report.md
benchmark/mpra_te_predictor_protocol_smoke/predictions.jsonl
benchmark/stability_predictor_protocol_smoke/report.json
benchmark/stability_predictor_protocol_smoke/report.md
benchmark/stability_predictor_protocol_smoke/predictions.jsonl
benchmark/family_leakage_protocol_smoke/report.json
benchmark/family_leakage_protocol_smoke/report.md
benchmark/family_leakage_protocol_smoke/splits/train.idx
benchmark/family_leakage_protocol_smoke/splits/val.idx
benchmark/family_leakage_protocol_smoke/splits/test.idx
benchmark/gencode_family_leakage_protocol/report.json
benchmark/gencode_family_leakage_protocol/report.md
benchmark/gencode_family_leakage_protocol/status.json
benchmark/gencode_family_leakage_protocol/status.md
benchmark/gencode_family_leakage_protocol/progress.jsonl
benchmark/gencode_family_leakage_protocol/splits/train.idx
benchmark/gencode_family_leakage_protocol/splits/val.idx
benchmark/gencode_family_leakage_protocol/splits/test.idx
benchmark/refseq_family_leakage_protocol/report.json
benchmark/refseq_family_leakage_protocol/report.md
benchmark/refseq_family_leakage_protocol/status.json
benchmark/refseq_family_leakage_protocol/status.md
benchmark/refseq_family_leakage_protocol/progress.jsonl
benchmark/refseq_family_leakage_protocol/splits/train.idx
benchmark/refseq_family_leakage_protocol/splits/val.idx
benchmark/refseq_family_leakage_protocol/splits/test.idx
docs/paper_table1_sota_landscape.json
docs/paper_table1_sota_landscape.md
docs/paper_table2_t1_t7_main_results.json
docs/paper_table2_t1_t7_main_results.md
docs/paper_table3_external_baseline_readiness.json
docs/paper_table3_external_baseline_readiness.md
docs/paper_table4_architecture_ablation.json
docs/paper_table4_architecture_ablation.md
docs/paper_table5_scale_law_readiness.json
docs/paper_table5_scale_law_readiness.md
docs/paper_figure1_full_length_edit_flow.json
docs/paper_figure1_full_length_edit_flow.md
docs/paper_figure2_cascade_recall_precision.json
docs/paper_figure2_cascade_recall_precision.md
docs/paper_figure3_oracle_gap_closure.json
docs/paper_figure3_oracle_gap_closure.md
EOF
}

print_plan() {
  echo "REMOTE SOTA STATUS SNAPSHOT"
  echo "LOCAL_ROOT=${LOCAL_ROOT}"
  echo "REMOTE=${REMOTE_HOST}:${REMOTE_ROOT}"
  echo "SLICE=${SLICE}  TOP_K=${TOP_K}"
  echo "WATCH_PID=${WATCH_PID}  GC_SWEEP_PID=${GC_SWEEP_PID}  MAX_LOADAVG=${MAX_LOADAVG}"
  echo "PYTHON_BIN=${PYTHON_BIN}"
  echo "OUT_JSON=${OUT_JSON}"
  echo "OUT_MD=${OUT_MD}"
  echo "targets:"
  target_list | sed 's/^/  - /'
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then usage; exit 0; fi
if [[ "${1:-}" == "--dry-run" ]]; then print_plan; exit 0; fi

remote_report="$(mktemp)"
trap 'rm -f "${remote_report}"' EXIT

ssh "${REMOTE_HOST}" "REMOTE_ROOT='${REMOTE_ROOT}' WATCH_PID='${WATCH_PID}' GC_SWEEP_PID='${GC_SWEEP_PID}' SLICE='${SLICE}' TOP_K='${TOP_K}' bash -s" > "${remote_report}" <<'REMOTE'
set -u
cd "${REMOTE_ROOT}"
printf 'REMOTE_DATE\t'; date -Iseconds
printf 'UPTIME\t'; uptime
if [[ -r /proc/loadavg ]]; then
  awk '{printf "LOADAVG\t%s\t%s\t%s\n", $1, $2, $3}' /proc/loadavg
else
  printf 'LOADAVG\t\t\t\n'
fi
printf 'PIDS_BEGIN\n'
ps -p "${WATCH_PID},${GC_SWEEP_PID}" -o pid,stat,etime,cmd || true
printf 'PIDS_END\n'
printf 'DYNAMIC_PROCS_BEGIN\n'
ps -eo pid,stat,etime,cmd \
  | grep -E 'run_stage_a_a100_max_train|stage_a_full_a100_max|run_stage_a_downstream_eval_queue|stage_a_downstream_eval_queue|run_downstream_predictor_protocol|downstream_predictor_protocol|watch_p3_readiness|p3_readiness_watcher|watch_gencode_family_readiness|gencode_family_readiness|run_refseq_family_leakage_audit|refseq_family_leakage_protocol|run_gencode_family_leakage_audit|gencode_family_leakage_protocol|family_leakage_protocol|run_refseq_public_build|refseq_public_build|public_pipeline|run_stage_a_scalelaw_sweep|stage_a_scalelaw|train_backbone|run_region_adapter_ablation_chain|train_adapter|eval_region_adapter|run_multiseed_benchmark|t6_length_curve|edit_budget_curve|run_protein_conditioned_t4_slice|protein_conditioned_cds|external_lineardesign|LinearDesign_2D|UTRGAN|conda env create|compare_benchmarks|summarize_region|audit_region|sota_gap' \
  | grep -v -E '/python[0-9.]*[[:space:]]+-[[:space:]]' \
  | grep -v grep || true
printf 'DYNAMIC_PROCS_END\n'
printf 'WATCH_TAIL_BEGIN\n'
ls -t logs/watch_region_adapter_eval_${SLICE:-head256}.*.log 2>/dev/null | head -n 1 | xargs -r tail -n 12
printf 'WATCH_TAIL_END\n'
printf 'GC_TAIL_BEGIN\n'
ls -t logs/protein_conditioned_cds_gc_sweep_${SLICE:-head256}.*.log 2>/dev/null | head -n 1 | xargs -r tail -n 8
printf 'GC_TAIL_END\n'
printf 'REGION_DIRS_BEGIN\n'
find benchmark -maxdepth 1 -type d -name "multiseed_t5_public_${SLICE:-head256}_region_adapter_*_top64" -print | sort
printf 'REGION_DIRS_END\n'
printf 'T6_HEAD1024_PROGRESS_BEGIN\n'
for spec in neg30:-30 neg15:-15 pos0:0 pos15:15 pos30:30; do
  name="${spec%%:*}"
  delta="${spec##*:}"
  dir="benchmark/multiseed_t6_public_head1024_stagea10k_len_${name}_top64"
  summary="${dir}/multiseed_summary.json"
  alt_summary="$(find benchmark -maxdepth 2 -path "benchmark/multiseed_t6_public_head1024_stagea10k_len_${name}_*_top64/multiseed_summary.json" -print 2>/dev/null | sort | head -n 1)"
  alt_dir="$(find benchmark -maxdepth 1 -type d -name "multiseed_t6_public_head1024_stagea10k_len_${name}_*_top64" -print 2>/dev/null | sort | head -n 1)"
  if [[ -f "${summary}" ]]; then
    sha="$(sha256sum "${summary}" | awk '{print $1}')"
    seed_count="$(find "${dir}" -maxdepth 1 -type d -name 'seed_*' 2>/dev/null | wc -l | tr -d ' ')"
    eval_count="$(find "${dir}" -maxdepth 2 -type f -path '*/seed_*/eval_summary.json' 2>/dev/null | wc -l | tr -d ' ')"
    printf 'T6_PROGRESS\t%s\tsummary\t%s\t%s\t%s\t%s\n' "${delta}" "${seed_count}" "${eval_count}" "${sha}" "${summary}"
  elif [[ -n "${alt_summary}" && -f "${alt_summary}" ]]; then
    sha="$(sha256sum "${alt_summary}" | awk '{print $1}')"
    alt_summary_dir="$(dirname "${alt_summary}")"
    seed_count="$(find "${alt_summary_dir}" -maxdepth 1 -type d -name 'seed_*' 2>/dev/null | wc -l | tr -d ' ')"
    eval_count="$(find "${alt_summary_dir}" -maxdepth 2 -type f -path '*/seed_*/eval_summary.json' 2>/dev/null | wc -l | tr -d ' ')"
    printf 'T6_PROGRESS\t%s\tsummary\t%s\t%s\t%s\t%s\n' "${delta}" "${seed_count}" "${eval_count}" "${sha}" "${alt_summary}"
  elif [[ -d "${dir}" ]]; then
    seed_count="$(find "${dir}" -maxdepth 1 -type d -name 'seed_*' 2>/dev/null | wc -l | tr -d ' ')"
    eval_count="$(find "${dir}" -maxdepth 2 -type f -path '*/seed_*/eval_summary.json' 2>/dev/null | wc -l | tr -d ' ')"
    status="running"
    if [[ "${seed_count}" == "0" ]]; then
      status="pending"
    fi
    printf 'T6_PROGRESS\t%s\t%s\t%s\t%s\t-\t%s\n' "${delta}" "${status}" "${seed_count}" "${eval_count}" "${dir}"
  elif [[ -n "${alt_dir}" && -d "${alt_dir}" ]]; then
    seed_count="$(find "${alt_dir}" -maxdepth 1 -type d -name 'seed_*' 2>/dev/null | wc -l | tr -d ' ')"
    eval_count="$(find "${alt_dir}" -maxdepth 2 -type f -path '*/seed_*/eval_summary.json' 2>/dev/null | wc -l | tr -d ' ')"
    status="running"
    if [[ "${seed_count}" == "0" ]]; then
      status="pending"
    fi
    printf 'T6_PROGRESS\t%s\t%s\t%s\t%s\t-\t%s\n' "${delta}" "${status}" "${seed_count}" "${eval_count}" "${alt_dir}"
  else
    printf 'T6_PROGRESS\t%s\tpending\t0\t0\t-\t%s\n' "${delta}" "${dir}"
  fi
done
printf 'T6_HEAD1024_PROGRESS_END\n'
printf 'T6_HEAD1024_SHARDS_BEGIN\n'
if [[ -d benchmark/t6_shards ]]; then
  find benchmark/t6_shards -maxdepth 1 -type d -name 'head1024_stagea10k_len_*_top64' -print 2>/dev/null | sort | while read -r shard_dir; do
    base="$(basename "${shard_dir}")"
    label="${base#head1024_stagea10k_len_}"
    delta_name="${label%%_*}"
    case "${delta_name}" in
      neg*) delta="-${delta_name#neg}" ;;
      pos*) delta="${delta_name#pos}" ;;
      *) delta="0" ;;
    esac
    summary="${shard_dir}/multiseed_summary.json"
    seed_count="$(find "${shard_dir}" -maxdepth 1 -type d -name 'seed_*' 2>/dev/null | wc -l | tr -d ' ')"
    eval_count="$(find "${shard_dir}" -maxdepth 2 -type f -path '*/seed_*/eval_summary.json' 2>/dev/null | wc -l | tr -d ' ')"
    if [[ -f "${summary}" ]]; then
      sha="$(sha256sum "${summary}" | awk '{print $1}')"
      printf 'T6_SHARD\t%s\tsummary\t%s\t%s\t%s\t%s\n' "${delta}" "${seed_count}" "${eval_count}" "${sha}" "${summary}"
    else
      complete_summary="$(
        {
          find benchmark -maxdepth 2 \
            -path "benchmark/multiseed_t6_public_head1024_stagea10k_len_${delta_name}_top64/multiseed_summary.json" -print 2>/dev/null
          find benchmark -maxdepth 2 \
            -path "benchmark/multiseed_t6_public_head1024_stagea10k_len_${delta_name}_*_top64/multiseed_summary.json" -print 2>/dev/null
        } | sort | head -n 1
      )"
      active_count="$(
        ps -eo cmd \
          | grep -F -- "${REMOTE_ROOT%/}/${shard_dir}" \
          | grep -v grep \
          | wc -l \
          | tr -d ' '
      )"
      status="running"
      if [[ -n "${complete_summary}" ]]; then
        status="superseded"
      elif [[ "${active_count}" == "0" && "${seed_count}" != "0" ]]; then
        status="stopped"
      elif [[ "${active_count}" == "0" && "${seed_count}" == "0" ]]; then
        status="pending"
      fi
      printf 'T6_SHARD\t%s\t%s\t%s\t%s\t-\t%s\n' "${delta}" "${status}" "${seed_count}" "${eval_count}" "${shard_dir}"
    fi
  done
fi
printf 'T6_HEAD1024_SHARDS_END\n'
printf 'T6_HEAD1024_COVERAGE_BEGIN\n'
for spec in neg30:-30 neg15:-15 pos0:0 pos15:15 pos30:30; do
  name="${spec%%:*}"
  delta="${spec##*:}"
  seeds="$(
    {
      find benchmark -maxdepth 1 -type d \
        \( -name "multiseed_t6_public_head1024_stagea10k_len_${name}_top64" \
        -o -name "multiseed_t6_public_head1024_stagea10k_len_${name}_*_top64" \) -print 2>/dev/null
      if [[ -d benchmark/t6_shards ]]; then
        find benchmark/t6_shards -maxdepth 1 -type d -name "head1024_stagea10k_len_${name}_*_top64" -print 2>/dev/null
      fi
    } | while read -r coverage_dir; do
      find "${coverage_dir}" -maxdepth 2 -type f -path '*/seed_*/eval_summary.json' -print 2>/dev/null
    done | sed -E 's#.*seed_([0-9]+)/eval_summary.json#\1#' | sort -n -u
  )"
  completed="$(printf '%s\n' "${seeds}" | sed '/^$/d' | paste -sd, -)"
  if [[ -z "${completed}" ]]; then
    completed="-"
  fi
  missing=""
  for seed in 000 001 002 003 004 005 006 007 008 009; do
    if ! printf '%s\n' "${seeds}" | grep -qx "${seed}"; then
      missing="${missing}${missing:+,}${seed}"
    fi
  done
  if [[ -z "${missing}" ]]; then
    missing="-"
  fi
  eval_seed_count="$(printf '%s\n' "${seeds}" | sed '/^$/d' | wc -l | tr -d ' ')"
  printf 'T6_COVERAGE\t%s\t%s\t%s\t%s\n' "${delta}" "${eval_seed_count}" "${completed}" "${missing}"
done
printf 'T6_HEAD1024_COVERAGE_END\n'
printf 'ARTIFACTS_BEGIN\n'
for path in \
  "benchmark/region_adapter_vs_hardneg_v2_top${TOP_K}_${SLICE}.json" \
  "benchmark/region_adapter_vs_mo_grpo_top${TOP_K}_${SLICE}.json" \
  "benchmark/region_adapter_vs_mo_scalar_top${TOP_K}_${SLICE}.json" \
  "benchmark/region_adapter_vs_mo_pareto_top${TOP_K}_${SLICE}.json" \
  "benchmark/region_adapter_vs_mo_te_only_top${TOP_K}_${SLICE}.json" \
  "benchmark/region_adapter_decision_report_${SLICE}.json" \
  "benchmark/region_adapter_result_audit_${SLICE}.json" \
  "benchmark/protein_conditioned_cds_gc_sweep_${SLICE}.summary.json" \
  "benchmark/protein_conditioned_cds_gc_sweep_${SLICE}.audit.json" \
  "benchmark/protein_conditioned_codon_metrics_${SLICE}.json" \
  "benchmark/protein_conditioned_codon_metrics_${SLICE}.md" \
  "benchmark/protein_conditioned_t4_head1024/status.json" \
  "benchmark/protein_conditioned_t4_head1024/status.md" \
  "benchmark/protein_conditioned_t4_head1024/progress.jsonl" \
  "benchmark/codon_lattice_dp_head1024.json" \
  "benchmark/protein_conditioned_cds_head1024.summary.json" \
  "benchmark/protein_conditioned_codon_metrics_head1024.json" \
  "benchmark/protein_conditioned_codon_metrics_head1024.md" \
  "benchmark/protein_conditioned_cds_gc_sweep_head1024.summary.json" \
  "benchmark/protein_conditioned_cds_gc_sweep_head1024.audit.json" \
  "benchmark/t4_protein_identity_cai_gc_report_head1024.json" \
  "benchmark/t4_protein_identity_cai_gc_report_head1024.md" \
  "benchmark/multiseed_t5_public_head256_mo_te_only_top64/multiseed_summary.json" \
  "benchmark/multiseed_t5_public_head256_mo_scalar_top64/multiseed_summary.json" \
  "benchmark/multiseed_t5_public_head256_mo_pareto_top64/multiseed_summary.json" \
  "benchmark/multiseed_t5_public_head256_mo_grpo_top64/multiseed_summary.json" \
  "benchmark/multiseed_t5_public_head256_hardneg_v2_top64/multiseed_summary.json" \
  "benchmark/multiseed_t5_public_head1024_mo_te_only_top64/multiseed_summary.json" \
  "benchmark/multiseed_t5_public_head1024_mo_scalar_top64/multiseed_summary.json" \
  "benchmark/multiseed_t5_public_head1024_mo_pareto_top64/multiseed_summary.json" \
  "benchmark/multiseed_t5_public_head1024_mo_grpo_top64/multiseed_summary.json" \
  "benchmark/multiseed_t5_public_head1024_hardneg_v2_top64/multiseed_summary.json" \
  "benchmark/frozen_backbone_protocol_${SLICE}/summary.json" \
  "benchmark/frozen_backbone_protocol_${SLICE}/leakage.json" \
  "benchmark/t1_t7_evidence_status_head256.json" \
  "benchmark/multi_scale_sequence_spectrum_head32_ranker_full1k.json" \
  "benchmark/multi_scale_sequence_spectrum_head32_ranker_full1k.md" \
  "benchmark/multi_scale_sequence_spectrum_head32_ranker_full1k_figures/base_composition_full.svg" \
  "benchmark/multi_scale_sequence_spectrum_head32_ranker_full1k_figures/base_composition_five_utr.svg" \
  "benchmark/multi_scale_sequence_spectrum_head32_ranker_full1k_figures/base_composition_cds.svg" \
  "benchmark/multi_scale_sequence_spectrum_head32_ranker_full1k_figures/base_composition_three_utr.svg" \
  "benchmark/multi_scale_sequence_spectrum_head32_ranker_full1k_figures/length_histogram.svg" \
  "benchmark/multi_scale_sequence_spectrum_head32_ranker_full1k_figures/gc_histogram.svg" \
  "benchmark/multi_scale_sequence_spectrum_head32_ranker_full1k_figures/kmer_top_delta.svg" \
  "benchmark/multi_scale_sequence_spectrum_head32_ranker_full1k_figures/codon_pair_top_delta.svg" \
  "benchmark/external_sota/input_pack_t5_head1024/summary.json" \
  "benchmark/external_sota/input_pack_t5_head1024/table.md" \
  "benchmark/external_sota/input_pack_t5_head1024/cds_protein_inputs.jsonl" \
  "benchmark/external_sota/input_pack_t5_head1024/utr5_inputs.jsonl" \
  "benchmark/external_sota/input_pack_t5_head1024/metric_schema.json" \
  "docs/external_sota_real_run_audit.json" \
  "docs/external_sota_real_run_audit.md" \
  "benchmark/external_sota/real_runs_t5_head1024/LinearDesign.status.json" \
  "benchmark/external_sota/real_runs_t5_head1024/LinearDesign.status.md" \
  "benchmark/external_sota/real_runs_t5_head1024/LinearDesign.progress.jsonl" \
  "benchmark/external_sota/real_runs_t5_head1024/LinearDesign/summary.json" \
  "benchmark/external_sota/real_runs_t5_head1024/LinearDesign/cds_outputs.jsonl" \
  "scripts/external_lineardesign.sh" \
  "scripts/external_ensembledesign.sh" \
  "scripts/external_utrgan.sh" \
  "scripts/run_external_lineardesign_head1024.sh" \
  "docs/t4_external_cds_baseline_comparison.json" \
  "docs/t4_external_cds_baseline_comparison.md" \
  "external_tools/UTRGAN/environment.yml" \
  "external_tools/UTRGAN/models/checkpoint_3000.h5" \
  "logs/utrgan_env_create.log" \
  "logs/utrgan_env_create.pid" \
  "logs/utrgan_cf_env_create.log" \
  "logs/utrgan_cf_env_create.pid" \
  "logs/utrgan_cf_deps_install.log" \
  "logs/utrgan_cf_deps_install.pid" \
  "benchmark/edit_budget_curve_report_head256_head1024.json" \
  "benchmark/t6_length_curve_report_head256_head1024.json" \
  "benchmark/region_adapter_decision_report_head1024.json" \
  "benchmark/region_adapter_result_audit_head1024.json" \
  "docs/multiobjective_scaleup_claim_audit_head256_head1024.json" \
  "docs/sota_readiness_audit_${SLICE}.json" \
  "docs/sota_gap_report.json" \
  "configs/stage_a_full_a100_max.json" \
  "configs/stage_a_mig_tiny_gencode.json" \
  "scripts/run_stage_a_a100_max_train.sh" \
  "scripts/run_after_stage_a_a100_max.sh" \
  "scripts/run_protein_conditioned_t4_slice.sh" \
  "docs/reproduce_full_training_eval_commands.md" \
  "benchmark/stage_a_full_a100_max_gencode_100k_seed0/status.json" \
  "benchmark/stage_a_full_a100_max_gencode_100k_seed0/status.md" \
  "benchmark/stage_a_full_a100_max_gencode_100k_seed0/progress.jsonl" \
  "benchmark/stage_a_full_a100_max_gencode_100k_seed0_posteval/status.json" \
  "benchmark/stage_a_full_a100_max_gencode_100k_seed0_posteval/status.md" \
  "benchmark/stage_a_full_a100_max_gencode_100k_seed0_posteval/progress.jsonl" \
  "logs/stage_a_full_a100_max_gencode_100k_seed0.metadata.json" \
  "logs/stage_a_full_a100_max_gencode_100k_seed0.profile.jsonl" \
  "logs/stage_a_full_a100_max_gencode_100k_seed0.train.log" \
  "docs/data_scaleup_readiness.json" \
  "docs/data_scaleup_readiness.md" \
  "docs/dataset_manifest_audit.json" \
  "docs/dataset_manifest_audit.md" \
  "benchmark/mpra_te_predictor_protocol_smoke/report.json" \
  "benchmark/mpra_te_predictor_protocol_smoke/report.md" \
  "benchmark/mpra_te_predictor_protocol_smoke/predictions.jsonl" \
  "benchmark/stability_predictor_protocol_smoke/report.json" \
  "benchmark/stability_predictor_protocol_smoke/report.md" \
  "benchmark/stability_predictor_protocol_smoke/predictions.jsonl" \
  "benchmark/family_leakage_protocol_smoke/report.json" \
  "benchmark/family_leakage_protocol_smoke/report.md" \
  "benchmark/family_leakage_protocol_smoke/splits/train.idx" \
  "benchmark/family_leakage_protocol_smoke/splits/val.idx" \
  "benchmark/family_leakage_protocol_smoke/splits/test.idx" \
  "benchmark/gencode_family_leakage_protocol/status.json" \
  "benchmark/gencode_family_leakage_protocol/status.md" \
  "benchmark/gencode_family_leakage_protocol/progress.jsonl" \
  "docs/paper_table1_sota_landscape.json" \
  "docs/paper_table1_sota_landscape.md" \
  "docs/paper_table2_t1_t7_main_results.json" \
  "docs/paper_table2_t1_t7_main_results.md" \
  "docs/paper_table3_external_baseline_readiness.json" \
  "docs/paper_table3_external_baseline_readiness.md" \
  "docs/paper_table4_architecture_ablation.json" \
  "docs/paper_table4_architecture_ablation.md" \
  "docs/paper_table5_scale_law_readiness.json" \
  "docs/paper_table5_scale_law_readiness.md" \
  "docs/paper_figure1_full_length_edit_flow.json" \
  "docs/paper_figure1_full_length_edit_flow.md" \
  "docs/paper_figure2_cascade_recall_precision.json" \
  "docs/paper_figure2_cascade_recall_precision.md" \
  "docs/paper_figure3_oracle_gap_closure.json" \
  "docs/paper_figure3_oracle_gap_closure.md"; do
  if [[ -e "${path}" ]]; then
    sha="$(sha256sum "${path}" | awk '{print $1}')"
    printf 'ARTIFACT\tpresent\t%s\t%s\n' "${sha}" "${path}"
  else
    printf 'ARTIFACT\tmissing\t-\t%s\n' "${path}"
  fi
done
find benchmark -maxdepth 3 -type f \( \
  -path "benchmark/gencode_family_leakage_protocol/report.json" -o \
  -path "benchmark/gencode_family_leakage_protocol/report.md" -o \
  -path "benchmark/gencode_family_leakage_protocol/splits/*.idx" \
\) -print 2>/dev/null | sort | while read -r path; do
  sha="$(sha256sum "${path}" | awk '{print $1}')"
  printf 'ARTIFACT\tpresent\t%s\t%s\n' "${sha}" "${path}"
done
if [[ -f "logs/gencode_family_readiness.done.json" ]]; then
  sha="$(sha256sum "logs/gencode_family_readiness.done.json" | awk '{print $1}')"
  printf 'ARTIFACT\tpresent\t%s\t%s\n' "${sha}" "logs/gencode_family_readiness.done.json"
fi
find benchmark -maxdepth 2 -type f \( \
  -path "benchmark/stage_a_scalelaw*/plan.json" -o \
  -path "benchmark/stage_a_scalelaw*/plan.tsv" -o \
  -path "benchmark/stage_a_scalelaw*/progress.jsonl" -o \
  -path "benchmark/stage_a_scalelaw*/status.json" -o \
  -path "benchmark/stage_a_scalelaw*/status.md" -o \
  -path "benchmark/stage_a_scalelaw*/summary.json" -o \
  -path "benchmark/stage_a_scalelaw*/summary.md" \
\) -print 2>/dev/null | sort | while read -r path; do
  sha="$(sha256sum "${path}" | awk '{print $1}')"
  printf 'ARTIFACT\tpresent\t%s\t%s\n' "${sha}" "${path}"
done
find benchmark -maxdepth 2 -type f \( \
  -path "benchmark/refseq_public_build*/progress.jsonl" -o \
  -path "benchmark/refseq_public_build*/status.json" -o \
  -path "benchmark/refseq_public_build*/status.md" \
\) -print 2>/dev/null | sort | while read -r path; do
  sha="$(sha256sum "${path}" | awk '{print $1}')"
  printf 'ARTIFACT\tpresent\t%s\t%s\n' "${sha}" "${path}"
done
for path in \
  "data/raw/human.1.rna.gbff.gz" \
  "data/processed/refseq_human_rna.records.jsonl" \
  "data/processed/refseq_human_rna.data_manifest.json"; do
  if [[ -e "${path}" ]]; then
    sha="$(sha256sum "${path}" | awk '{print $1}')"
    printf 'ARTIFACT\tpresent\t%s\t%s\n' "${sha}" "${path}"
  fi
done
printf 'ARTIFACTS_END\n'
REMOTE

mkdir -p "$(dirname "${OUT_JSON}")" "$(dirname "${OUT_MD}")"
"${PYTHON_BIN}" - "${remote_report}" "${OUT_JSON}" "${OUT_MD}" "${REMOTE_HOST}" "${REMOTE_ROOT}" "${SLICE}" "${TOP_K}" "${WATCH_PID}" "${GC_SWEEP_PID}" "${MAX_LOADAVG}" <<'PY'
import json
import sys
from datetime import datetime, timezone

report_path, out_json, out_md, remote_host, remote_root, slice_name, top_k, watch_pid, gc_pid, max_load = sys.argv[1:11]

with open(report_path, "r", encoding="utf-8") as fh:
    lines = [line.rstrip("\n") for line in fh]

payload = {
    "artifact_kind": "remote_execution_status",
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "remote_host": remote_host,
    "remote_root": remote_root,
    "slice": slice_name,
    "top_k": int(top_k),
    "max_loadavg": float(max_load),
    "watch_pid": int(watch_pid),
    "gc_sweep_pid": int(gc_pid),
    "remote_date": None,
    "uptime": None,
    "loadavg": None,
    "ps_block": [],
    "dynamic_processes": [],
    "watch_tail": [],
    "gc_tail": [],
    "region_dirs": [],
    "t6_head1024_progress": [],
    "t6_head1024_shards": [],
    "t6_head1024_coverage": [],
    "artifacts": [],
}

section = None
for line in lines:
    if line.startswith("REMOTE_DATE\t"):
        payload["remote_date"] = line.split("\t", 1)[1]
    elif line.startswith("UPTIME\t"):
        payload["uptime"] = line.split("\t", 1)[1]
    elif line.startswith("LOADAVG\t"):
        parts = line.split("\t")
        payload["loadavg"] = [float(x) for x in parts[1:4] if x]
    elif line == "PIDS_BEGIN":
        section = "ps_block"
    elif line == "PIDS_END":
        section = None
    elif line == "DYNAMIC_PROCS_BEGIN":
        section = "dynamic_processes"
    elif line == "DYNAMIC_PROCS_END":
        section = None
    elif line == "WATCH_TAIL_BEGIN":
        section = "watch_tail"
    elif line == "WATCH_TAIL_END":
        section = None
    elif line == "GC_TAIL_BEGIN":
        section = "gc_tail"
    elif line == "GC_TAIL_END":
        section = None
    elif line == "REGION_DIRS_BEGIN":
        section = "region_dirs"
    elif line == "REGION_DIRS_END":
        section = None
    elif line == "T6_HEAD1024_PROGRESS_BEGIN":
        section = "t6_head1024_progress"
    elif line == "T6_HEAD1024_PROGRESS_END":
        section = None
    elif line == "T6_HEAD1024_SHARDS_BEGIN":
        section = "t6_head1024_shards"
    elif line == "T6_HEAD1024_SHARDS_END":
        section = None
    elif line == "T6_HEAD1024_COVERAGE_BEGIN":
        section = "t6_head1024_coverage"
    elif line == "T6_HEAD1024_COVERAGE_END":
        section = None
    elif line == "ARTIFACTS_BEGIN":
        section = "artifacts"
    elif line == "ARTIFACTS_END":
        section = None
    elif section == "t6_head1024_progress" and line.startswith("T6_PROGRESS\t"):
        _, delta, status, seed_count, eval_count, sha, path = line.split("\t", 6)
        payload["t6_head1024_progress"].append(
            {
                "target_length_delta": int(delta),
                "status": status,
                "seed_count": int(seed_count),
                "eval_summary_count": int(eval_count),
                "sha256": None if sha == "-" else sha,
                "path": path,
            }
        )
    elif section == "t6_head1024_shards" and line.startswith("T6_SHARD\t"):
        _, delta, status, seed_count, eval_count, sha, path = line.split("\t", 6)
        payload["t6_head1024_shards"].append(
            {
                "target_length_delta": int(delta),
                "status": status,
                "seed_count": int(seed_count),
                "eval_summary_count": int(eval_count),
                "sha256": None if sha == "-" else sha,
                "path": path,
            }
        )
    elif section == "t6_head1024_coverage" and line.startswith("T6_COVERAGE\t"):
        _, delta, eval_seed_count, completed, missing = line.split("\t", 4)
        completed_list = [] if completed == "-" else [int(seed) for seed in completed.split(",") if seed]
        missing_list = [] if missing == "-" else [int(seed) for seed in missing.split(",") if seed]
        payload["t6_head1024_coverage"].append(
            {
                "target_length_delta": int(delta),
                "eval_seed_count": int(eval_seed_count),
                "completed_seeds": completed_list,
                "missing_seeds": missing_list,
                "complete": len(missing_list) == 0,
            }
        )
    elif section == "artifacts" and line.startswith("ARTIFACT\t"):
        _, status, sha, path = line.split("\t", 3)
        payload["artifacts"].append({"path": path, "status": status, "sha256": None if sha == "-" else sha})
    elif section in {"ps_block", "dynamic_processes", "watch_tail", "gc_tail", "region_dirs"}:
        payload[section].append(line)

present = [row for row in payload["artifacts"] if row["status"] == "present"]
missing = [row for row in payload["artifacts"] if row["status"] == "missing"]
payload["summary"] = {
    "n_artifacts_present": len(present),
    "n_artifacts_missing": len(missing),
    "n_dynamic_processes": len(payload["dynamic_processes"]),
    "n_t6_head1024_summaries": sum(1 for row in payload["t6_head1024_progress"] if row["status"] == "summary"),
    "n_t6_head1024_shards": len(payload["t6_head1024_shards"]),
    "n_t6_head1024_complete_coverages": sum(1 for row in payload["t6_head1024_coverage"] if row["complete"]),
    "region_eval_started": bool(payload["region_dirs"]),
    "readiness_present": any(row["path"] == f"docs/sota_readiness_audit_{slice_name}.json" and row["status"] == "present" for row in payload["artifacts"]),
}

with open(out_json, "w", encoding="utf-8") as fh:
    json.dump(payload, fh, indent=2, sort_keys=True)

load = payload["loadavg"] or []
load_text = " / ".join(f"{x:.2f}" for x in load) if load else "unknown"
lines_md = [
    "# Remote Execution Status",
    "",
    f"Last refreshed: {payload['remote_date']}",
    "",
    "## Server",
    "",
    f"- Host: `{remote_host}`",
    f"- Remote root: `{remote_root}`",
    f"- Resource gate: `MAX_LOADAVG={max_load}`",
    f"- Current load average: `{load_text}`",
    "",
    "## Active Processes",
    "",
    "```text",
    *payload["ps_block"],
    "```",
    "",
    "## Dynamic MEF Processes",
    "",
    "```text",
    *payload["dynamic_processes"],
    "```",
    "",
    "## T6 Head1024 Progress",
    "",
    "| target length delta | status | seed dirs | eval summaries | sha256/path |",
    "|---:|---|---:|---:|---|",
]
for row in payload["t6_head1024_progress"]:
    sha_or_path = row["sha256"] or row["path"]
    lines_md.append(
        f"| {row['target_length_delta']:+d} | {row['status']} | {row['seed_count']} | {row['eval_summary_count']} | `{sha_or_path}` |"
    )
if payload["t6_head1024_shards"]:
    lines_md.extend(
        [
            "",
            "## T6 Head1024 Shards",
            "",
            "| target length delta | status | seed dirs | eval summaries | sha256/path |",
            "|---:|---|---:|---:|---|",
        ]
    )
    for row in payload["t6_head1024_shards"]:
        sha_or_path = row["sha256"] or row["path"]
        lines_md.append(
            f"| {row['target_length_delta']:+d} | {row['status']} | {row['seed_count']} | {row['eval_summary_count']} | `{sha_or_path}` |"
        )
if payload["t6_head1024_coverage"]:
    lines_md.extend(
        [
            "",
            "## T6 Head1024 Merge Coverage",
            "",
            "| target length delta | eval seeds | completed seeds | missing seeds | complete |",
            "|---:|---:|---|---|---|",
        ]
    )
    for row in payload["t6_head1024_coverage"]:
        completed = ",".join(f"{int(seed):03d}" for seed in row["completed_seeds"]) or "-"
        missing = ",".join(f"{int(seed):03d}" for seed in row["missing_seeds"]) or "-"
        lines_md.append(
            f"| {row['target_length_delta']:+d} | {row['eval_seed_count']} | `{completed}` | `{missing}` | `{row['complete']}` |"
        )
lines_md.extend([
    "",
    "## Gate Evidence",
    "",
    "Latest watcher tail:",
    "",
    "```text",
    *payload["watch_tail"],
    "```",
    "",
    "GC sweep queue tail:",
    "",
    "```text",
    *payload["gc_tail"],
    "```",
    "",
    "## Artifact Status",
    "",
    "| artifact | status | sha256 |",
    "|---|---|---|",
])
for row in payload["artifacts"]:
    lines_md.append(f"| `{row['path']}` | {row['status']} | `{row['sha256'] or ''}` |")
lines_md.extend([
    "",
    "## Current Readiness",
    "",
    f"- Region eval started: `{payload['summary']['region_eval_started']}`",
    f"- Target artifacts present/missing: `{payload['summary']['n_artifacts_present']}/{payload['summary']['n_artifacts_missing']}`",
    f"- Readiness artifact present: `{payload['summary']['readiness_present']}`",
    "",
    "## Policy",
    "",
    f"Do not lower `MAX_LOADAVG={max_load}`. Wait for the watcher to pass the load gate, then run `scripts/harvest_sota_artifacts.sh` and inspect the unified readiness report before making any positive SOTA claim.",
])
with open(out_md, "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines_md) + "\n")
PY

echo "Remote status JSON -> ${OUT_JSON}"
echo "Remote status MD   -> ${OUT_MD}"
