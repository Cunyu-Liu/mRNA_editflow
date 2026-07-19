#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON:-${REPO_ROOT}/editflow/.venv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python3"
fi
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

usage() {
  cat <<'EOF'
Usage:
  mrna_editflow/scripts/run_ablation.sh --dry-run
  mrna_editflow/scripts/run_ablation.sh --list
  mrna_editflow/scripts/run_ablation.sh --smoke

Purpose:
  Enumerate and smoke-run Task 6 ablations without external data, GPU, pytest,
  or network access.
EOF
}

list_ablations() {
  cat <<'EOF'
Ablation switches:
  rope_on_vs_off
    question: does rotary position encoding improve long mRNA sequence modelling?
    switch: ModelConfig(use_rope=True/False)

  region_film_on_vs_off
    question: does region-aware conditioning matter for 5UTR/CDS/3UTR heterogeneity?
    switch: ModelConfig(use_region_film=True/False)

  codon_constraint_on_vs_off
    question: does synonymous codon-lattice restriction preserve CDS legality?
    switch: ModelConfig(use_codon_constraint=True/False)

  whole_codon_indel_on_vs_off
    question: are frame-safe whole-codon indels useful inside CDS?
    switch: ModelConfig(codon_indel=True/False)

  aux_structure_on_vs_off
    question: do MFE/start-accessibility auxiliary heads stabilize useful designs?
    switch: ModelConfig(use_aux_struct=True/False, aux_loss_weight)

  guidance_scale
    question: does oracle/property guidance improve design objectives without breaking legality?
    switch: guidance_scale in sampling/eval protocol (0.0 vs >0.0)

  editflow_vs_masked_diffusion
    question: does variable-length edit flow beat fixed-canvas masked denoising?
    switch: train_stage_a/sample_mrna vs baselines.masked_diffusion

  coupling_mixture_empty_corruption_ortholog
    question: which three-way coupling ratio drives robustness and evolution-aware edits?
    switch: CouplingConfig(empty_prob, corruption_prob, ortholog_prob)

  backbone_family
    question: how much comes from the foundation encoder versus MEF head?
    switch: BackboneConfig(name=none vs mrnabert/helix_mrna/orthrus_mlm/lamar vs rna_fm/rinalmo)

  ar_lm_baseline
    question: how does a traditional left-to-right generator compare to edit flow?
    switch: baselines.ar_lm.AutoregressiveLM

  codon_lattice_dp_baseline
    question: how does a transparent synonymous codon-lattice optimizer compare on CDS-only objectives?
    switch: baselines.codon_lattice_dp.CodonLatticeDPConfig

  utr_local_search_baseline
    question: how much TE gain is reachable by UTR-only predictor-guided local search?
    switch: baselines.utr_local_search.UTRLocalSearchConfig

  utr_teacher_export
    question: can UTR oracle headroom be converted into ranker-compatible one-step supervision?
    switch: baselines.utr_teacher_export.export_utr_teacher_jsonl

  hybrid_teacher_export
    question: can full-pool TE ranking and UTR-search teacher extremes improve both top-1 regret and top-k recall?
    switch: baselines.hybrid_teacher_export.export_hybrid_teacher_jsonl
EOF
}

dry_run() {
  list_ablations
  cat <<'EOF'

Dry-run command matrix:
  # RoPE
  python - <<'PY'  # ModelConfig(use_rope=True/False) with same seed/data split
  # Region FiLM
  python - <<'PY'  # ModelConfig(use_region_film=True/False)
  # Codon constraint
  python - <<'PY'  # ModelConfig(use_codon_constraint=True/False)
  # Whole-codon indel
  python - <<'PY'  # ModelConfig(codon_indel=True/False)
  # Auxiliary structural head
  python - <<'PY'  # ModelConfig(use_aux_struct=True/False)
  # Guidance
  python - <<'PY'  # sample/eval with guidance_scale=0.0 and target scale
  # EditFlow vs diffusion
  python - <<'PY'  # train_stage_a/sample_mrna vs train_masked_diffusion/sample_masked_diffusion
  # Coupling ratios, including ortholog
  python - <<'PY'  # CouplingConfig(1,0,0), (0,1,0), (0,0,1), mixed default
  # Backbone controls
  python - <<'PY'  # BackboneConfig(name='none'), mRNA-native, ncRNA-control
  # AR LM
  python - <<'PY'  # train_ar_lm/sample_ar_lm
  # Codon lattice DP
  python - <<'PY'  # baselines.codon_lattice_dp.run_codon_lattice_dp
  # UTR local search
  python - <<'PY'  # baselines.utr_local_search.run_utr_local_search
  # UTR teacher export
  python - <<'PY'  # baselines.utr_teacher_export.export_utr_teacher_jsonl
  # Hybrid teacher export
  python - <<'PY'  # baselines.hybrid_teacher_export.export_hybrid_teacher_jsonl

Local executable smoke:
  mrna_editflow/scripts/run_ablation.sh --smoke
EOF
}

smoke() {
  "${PYTHON_BIN}" - <<'PY'
import json
import math

from mrna_editflow.baselines.ar_lm import ARLMConfig, sample_ar_lm, train_ar_lm
from mrna_editflow.baselines.external_models import list_external_results
from mrna_editflow.baselines.masked_diffusion import (
    MaskedDiffusionConfig,
    sample_masked_diffusion,
    train_masked_diffusion,
)
from mrna_editflow.baselines.mrnabench_probe import (
    MRNABenchProbeConfig,
    run_mrnabench_probe,
)
from mrna_editflow.baselines.utr_local_search import (
    UTRLocalSearchConfig,
    optimize_records_five_utr,
)
from mrna_editflow.baselines.hybrid_teacher_export import export_hybrid_teacher_jsonl
from mrna_editflow.baselines.utr_teacher_export import score_record_utr_teacher_rows
from mrna_editflow.data.download_mrna import synthesize_corpus

records = synthesize_corpus(4, seed=20260711)

diff = train_masked_diffusion(
    records,
    MaskedDiffusionConfig(
        max_len=64,
        hidden_dim=16,
        num_layers=1,
        num_heads=2,
        batch_size=2,
        steps=1,
        mask_prob=0.4,
        seed=11,
    ),
)
diff_seq = sample_masked_diffusion(diff.model, length=24, denoise_steps=2, seed=12)

ar = train_ar_lm(
    records,
    ARLMConfig(max_len=64, hidden_dim=16, num_layers=1, batch_size=2, steps=1, seed=13),
)
ar_seq = sample_ar_lm(ar.model, length=24, seed=14)

probe = run_mrnabench_probe(
    records=records,
    config=MRNABenchProbeConfig(steps=2, synthetic_size=4, seed=15),
)
utr_optimized, utr_results = optimize_records_five_utr(
    records,
    config=UTRLocalSearchConfig(
        edit_budget=1,
        beam_width=4,
        start_window_nt=24,
        max_edit_positions=24,
    ),
)
teacher_rows = score_record_utr_teacher_rows(
    records[0],
    config=UTRLocalSearchConfig(
        edit_budget=1,
        beam_width=4,
        start_window_nt=24,
        max_edit_positions=24,
    ),
    candidate_cap=8,
)
import tempfile
with tempfile.TemporaryDirectory(prefix="mef_hybrid_smoke_") as tmp:
    full_path = f"{tmp}/full.jsonl"
    utr_path = f"{tmp}/utr.jsonl"
    out_path = f"{tmp}/hybrid.jsonl"
    summary_path = f"{tmp}/hybrid.json"
    with open(full_path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "transcript_id": records[0].transcript_id,
            "task_id": "T5",
            "op": "sub",
            "pos": 0,
            "nt": "A",
            "teacher_score": 0.10,
        }) + "\n")
    with open(utr_path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "transcript_id": records[0].transcript_id,
            "task_id": "T5",
            "op": "sub",
            "pos": 0,
            "nt": "A",
            "teacher_score": 0.30,
        }) + "\n")
    hybrid = export_hybrid_teacher_jsonl(
        full_jsonl=full_path,
        utr_jsonl=utr_path,
        out_jsonl=out_path,
        out_json=summary_path,
        max_rows_per_record=8,
        run_mode="development",
    )
external = list_external_results(task_id="T1")

assert math.isfinite(diff.final_loss)
assert math.isfinite(ar.final_loss)
assert math.isfinite(probe.final_loss)
assert len(diff_seq) == 24 and set(diff_seq) <= set("ACGU")
assert len(ar_seq) == 24 and set(ar_seq) <= set("ACGU")
assert all(a.cds == b.cds for a, b in zip(records, utr_optimized))
assert all(a.three_utr == b.three_utr for a, b in zip(records, utr_optimized))
assert all(item.utr_edit_distance <= 1 for item in utr_results)
assert len(teacher_rows) == 8
assert all(row.transcript_id == records[0].transcript_id for row in teacher_rows)
assert hybrid["summary"]["overlap_rows"] == 1
assert external and all(item.status == "offline_placeholder" for item in external)

print(json.dumps({
    "status": "ok",
    "masked_diffusion_loss": diff.final_loss,
    "masked_diffusion_sample": diff_seq,
    "ar_lm_loss": ar.final_loss,
    "ar_lm_sample": ar_seq,
    "mrnabench_trainable_params": probe.trainable_params,
    "utr_local_search_mean_delta_te": sum(item.delta_te for item in utr_results) / len(utr_results),
    "utr_teacher_rows": len(teacher_rows),
    "hybrid_teacher_rows": hybrid["summary"]["n_rows"],
    "external_records": len(external),
}, indent=2, sort_keys=True))
PY
}

case "${1:---dry-run}" in
  --dry-run)
    dry_run
    ;;
  --list)
    list_ablations
    ;;
  --smoke)
    smoke
    ;;
  -h|--help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
