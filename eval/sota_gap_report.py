"""Generate a measured SOTA gap report for mRNA-EditFlow.

The report joins two evidence streams:

* measured local artifacts, such as paired multi-seed benchmark comparisons and
  proposal-ranking audits;
* external SOTA protocol records, which document what must still be integrated
  before paper-grade claims can be made.

No network access is performed. Literature metadata is intentionally static so
the report can be regenerated inside the training server and in offline tests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from typing import Mapping, Optional, Sequence

from mrna_editflow.baselines.external_models import available_external_models
from mrna_editflow.eval.artifact_contract import normalize_run_mode, paper_builder_gate, validate_report_output_namespaces, write_paper_report_sidecars


@dataclass(frozen=True)
class MeasuredEvidence:
    """One quantitative result extracted from local artifacts.

    ``baseline`` and ``run`` are metric means. ``delta = run - baseline`` follows
    the convention used by :mod:`mrna_editflow.eval.compare_benchmarks`; for
    lower-is-better metrics the narrative should interpret negative deltas as
    improvements.

    Complexity: construction and conversion are ``O(1)``.
    """

    name: str
    metric: str
    baseline: float
    run: float
    delta: float
    ci_low: Optional[float]
    ci_high: Optional[float]
    paired_p: Optional[float]
    n: Optional[int]
    source: str
    interpretation: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready mapping. Complexity: ``O(fields)``."""
        return dict(asdict(self))


SOTA_REFERENCE_ROWS: tuple[dict[str, str], ...] = (
    {
        "method": "LinearDesign",
        "venue_year": "Nature 2023",
        "scope": "protein-conditioned CDS structure/codon dynamic programming",
        "reported_signal": "Strong experimental validation; SARS-CoV-2 spike design reported about 11 min runtime and high antibody titer gains.",
        "accuracy_f1": "Not an accuracy/F1 task; optimize folding/codon objective.",
        "speed_scale": "Dynamic-programming CDS optimizer; strong deterministic baseline.",
        "mef_gap": "Run CDS-only T4 benchmark with CAI/MFE/proxy TE and wall-clock comparison.",
        "citation_url": "https://doi.org/10.1038/s41586-023-06127-z",
    },
    {
        "method": "EnsembleDesign",
        "venue_year": "Bioinformatics 2025",
        "scope": "CDS ensemble free-energy optimization",
        "reported_signal": "Probabilistic lattice parsing over synonymous codons.",
        "accuracy_f1": "Objective-optimization task, not classification.",
        "speed_scale": "External lattice parser; runtime benchmark pending.",
        "mef_gap": "Add ensemble-free-energy objective to T4/CDS benchmark.",
        "citation_url": "https://doi.org/10.1093/bioinformatics/btaf245",
    },
    {
        "method": "mRNA-LM",
        "venue_year": "Nucleic Acids Research 2025",
        "scope": "full-length mRNA analysis and representation",
        "reported_signal": "Segment-integrated small language model trained on millions of mRNAs.",
        "accuracy_f1": "Task dependent; compare frozen-probe accuracy/F1 on mRNABench-style tasks.",
        "speed_scale": "Foundation encoder; adapter/probe runtime pending.",
        "mef_gap": "Install checkpoint or embedding cache and run frozen-probe plus MEF-head ablation.",
        "citation_url": "https://doi.org/10.1093/nar/gkaf044",
    },
    {
        "method": "codonGPT",
        "venue_year": "Nucleic Acids Research 2025",
        "scope": "codon-level generative LM plus RL for CDS design",
        "reported_signal": "Trained on 338,417 mRNAs and optimized expression, stability and GC rewards.",
        "accuracy_f1": "Generative/RL task; use reward, validity and protein-preservation metrics.",
        "speed_scale": "Scalable codon LM; exact local throughput pending.",
        "mef_gap": "Compare CDS-only T4 generation, then show MEF advantage from joint UTR editing.",
        "citation_url": "https://doi.org/10.1093/nar/gkaf1345",
    },
    {
        "method": "GEMORNA",
        "venue_year": "Science 2025",
        "scope": "de novo full-length mRNA design (encoder-decoder CDS + decoder-only UTR, then combined)",
        "reported_signal": "Wet-lab validated: up to ~41x protein and ~128x antibody response vs codon optimization; circRNA ~121x in vivo. CDS beats CAI/GC baselines; 5'UTRs rival BNT162b2.",
        "accuracy_f1": "Generation task; report wet-lab/oracle protein output, CAI, MFE, rare-codon, and TE proxy.",
        "speed_scale": "Zero-shot autoregressive decoders; modular per-region generation.",
        "mef_gap": "Align T4 CDS-only + T5 UTR-only design under one protocol; MEF must show full-length constrained-edit advantage vs GEMORNA's modular combine, and report that MEF preserves 100% protein identity (hard constraint) rather than free CDS regeneration.",
        "citation_url": "https://doi.org/10.1126/science.adr8470",
    },
    {
        "method": "mRNA-GPT",
        "venue_year": "ICLR 2026 submission",
        "scope": "end-to-end full-length mRNA design/optimization (decoder-only, joint 5'UTR+CDS+3'UTR)",
        "reported_signal": "Pretrained on 10M full-length natural mRNAs across species; iterative oracle-reward optimization; reports higher predicted translation rate than LinearDesign and GEMORNA plus higher full-length diversity.",
        "accuracy_f1": "Generation task; report predicted TE/half-life, diversity, cross-region interaction gain, protein preservation.",
        "speed_scale": "Decoder-only LM; flexible single-region / full-length / conditional generation modes.",
        "mef_gap": "This is the closest full-length rival: run head-to-head full-length T1-T7 with matched proxy oracle; MEF's differentiator is constrained edit-flow (protein/frame/budget hard constraints + interpretable edit distance) vs mRNA-GPT's free autoregressive generation. Quantify TE delta AND constraint satisfaction jointly.",
        "citation_url": "https://openreview.net/pdf?id=juUrI9kCBw",
    },
    {
        "method": "ProMORNA",
        "venue_year": "arXiv 2026",
        "scope": "protein-conditioned de novo full-length mRNA via multi-objective RL (BART encoder-decoder + MO-GRPO)",
        "reported_signal": "Pretrained on 6M protein-mRNA pairs; MO-GRPO standardizes per-metric advantages before aggregation; improves in-silico Pareto frontier of predicted half-life vs TE on held-out luciferase.",
        "accuracy_f1": "Multi-objective generation; report per-objective Pareto frontier (TE, half-life, immune-safety) and protein-conditioned validity.",
        "speed_scale": "GRPO RL over protein prompts; no wild-type template needed at inference.",
        "mef_gap": "Directly informs MEF upgrade #1 (multi-objective) and #3 (protein-conditioned): adopt per-metric advantage standardization as an alternative to scalar/Pareto fusion, and align MEF protein-conditioned CDS + multi-objective ranker against ProMORNA's Pareto-frontier metric on a shared held-out protein.",
        "citation_url": "https://arxiv.org/abs/2605.01513",
    },
    {
        "method": "RNAGenScape",
        "venue_year": "ICML 2025 GenBio workshop / arXiv 2025",
        "scope": "property-guided optimization + interpolation of existing mRNAs via manifold Langevin dynamics (organized autoencoder + manifold projector)",
        "reported_signal": "Across 3 real mRNA datasets (2 orders of magnitude in size, incl. zebrafish 5'UTR); up to +148% median property gain and +30% success rate while staying on the viable manifold; interpretable latent trajectories; robust with as few as ~2000 points.",
        "accuracy_f1": "Optimization task; report property gain, success rate, on-manifold biological plausibility, and interpolation smoothness.",
        "speed_scale": "Latent Langevin dynamics; efficient inference, no explicit score/density learning.",
        "mef_gap": "Closest philosophy to MEF ('optimize, not invent' from real sequences): benchmark MEF's discrete constrained edit-flow trajectories against RNAGenScape's continuous latent trajectories on shared 5'UTR TE optimization; report interpretability (edit distance/budget) and constraint guarantees MEF adds over latent projection.",
        "citation_url": "https://arxiv.org/abs/2510.24736",
    },
    {
        "method": "CodonFM",
        "venue_year": "NVIDIA open model 2025",
        "scope": "codon-level RNA foundation encoder",
        "reported_signal": "Open codon-aware models announced at 80M, 600M and 1B parameter scales.",
        "accuracy_f1": "Task dependent; compare property-prediction accuracy/F1 after frozen probing.",
        "speed_scale": "Large RefSeq-scale pretraining; embedding-cache runtime pending.",
        "mef_gap": "Add leakage-audited frozen embeddings and measure TE/stability probe transfer.",
        "citation_url": "https://developer.nvidia.com/blog/introducing-the-codonfm-open-model-for-rna-design-and-analysis/",
    },
    {
        "method": "UTailoR",
        "venue_year": "iScience 2025",
        "scope": "5'UTR discriminative and generative optimization",
        "reported_signal": "Optimized 5'UTRs reported around 200% translation-efficiency improvement.",
        "accuracy_f1": "Predictor/generator task; use TE proxy and held-out predictor agreement.",
        "speed_scale": "5'UTR-specific service/model; local throughput pending.",
        "mef_gap": "Run T5/T7 UTR-only benchmark and verify MEF preserves full transcript constraints.",
        "citation_url": "https://doi.org/10.1016/j.isci.2025.113544",
    },
    {
        "method": "Helix-mRNA",
        "venue_year": "arXiv 2025",
        "scope": "long-context full mRNA foundation encoder",
        "reported_signal": "Hybrid long-context model for full-sequence mRNA therapeutics.",
        "accuracy_f1": "Task dependent; compare frozen-probe accuracy/F1 and generation-head transfer.",
        "speed_scale": "Long-context embedding cache needed for fair runtime.",
        "mef_gap": "Use as frozen backbone and compare against backbone=none under same trainable budget.",
        "citation_url": "https://arxiv.org/abs/2502.13785",
    },
    {
        "method": "Prot2RNA",
        "venue_year": "OpenReview 2026 submission",
        "scope": "protein-conditioned CDS diffusion",
        "reported_signal": "Codon-level generation claims for protein-conditioned coding sequence design.",
        "accuracy_f1": "Use codon accuracy, codon-usage profile and protein-preservation metrics.",
        "speed_scale": "External diffusion model; runtime pending.",
        "mef_gap": "Isolate CDS-conditioned T4 benchmark; report that Prot2RNA has no UTR edit task.",
        "citation_url": "https://openreview.net/forum?id=BPNK5HDEMh",
    },
)

DATASET_SURVEY_METHODS: tuple[str, ...] = (
    "GEMORNA",
    "mRNA-GPT",
    "ProMORNA",
    "RNAGenScape",
    "codonGPT",
)


def _load_json(path: str) -> Optional[Mapping[str, object]]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _finite_or_none(value: object) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _comparison_row(
    payload: Mapping[str, object],
    *,
    run_label: str,
    metric: str,
) -> Optional[Mapping[str, object]]:
    rows = payload.get("rows", [])
    if not isinstance(rows, Sequence):
        return None
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        if row.get("run") == run_label and row.get("metric") == metric:
            return row
    return None


def _records_label(payload: Mapping[str, object]) -> str:
    baseline = payload.get("baseline", {})
    if not isinstance(baseline, Mapping):
        return "unknown"
    config = baseline.get("config", {})
    if not isinstance(config, Mapping):
        return "unknown"
    value = config.get("n_records", "unknown")
    return str(value)


def _evidence_from_comparison(
    project_root: str,
    relative_path: str,
    *,
    run_label: str,
    metric: str,
    name_prefix: str,
    interpretation: str,
) -> Optional[MeasuredEvidence]:
    payload = _load_json(os.path.join(project_root, relative_path))
    if payload is None:
        return None
    row = _comparison_row(payload, run_label=run_label, metric=metric)
    if row is None:
        return None
    baseline = _finite_or_none(row.get("baseline_mean"))
    run = _finite_or_none(row.get("run_mean"))
    delta = _finite_or_none(row.get("delta"))
    if baseline is None or run is None or delta is None:
        return None
    label = _records_label(payload)
    return MeasuredEvidence(
        name=f"{name_prefix} (n={label})",
        metric=metric,
        baseline=baseline,
        run=run,
        delta=delta,
        ci_low=_finite_or_none(row.get("ci_low")),
        ci_high=_finite_or_none(row.get("ci_high")),
        paired_p=_finite_or_none(row.get("paired_p")),
        n=int(row["n_paired_seeds"]) if isinstance(row.get("n_paired_seeds"), int) else None,
        source=relative_path,
        interpretation=interpretation,
    )


def _proposal_metric(
    project_root: str,
    base_path: str,
    ranker_path: str,
    *,
    metric: str,
    name: str,
    interpretation: str,
) -> Optional[MeasuredEvidence]:
    base_payload = _load_json(os.path.join(project_root, base_path))
    ranker_payload = _load_json(os.path.join(project_root, ranker_path))
    if base_payload is None or ranker_payload is None:
        return None
    base_agg = base_payload.get("aggregate", {})
    ranker_agg = ranker_payload.get("aggregate", {})
    if not isinstance(base_agg, Mapping) or not isinstance(ranker_agg, Mapping):
        return None
    baseline = _finite_or_none(base_agg.get(metric))
    run = _finite_or_none(ranker_agg.get(metric))
    if baseline is None or run is None:
        return None
    return MeasuredEvidence(
        name=name,
        metric=metric,
        baseline=baseline,
        run=run,
        delta=run - baseline,
        ci_low=None,
        ci_high=None,
        paired_p=None,
        n=int(ranker_agg["n_records"]) if isinstance(ranker_agg.get("n_records"), int) else None,
        source=f"{base_path} vs {ranker_path}",
        interpretation=interpretation,
    )


def _aggregate_metric(
    project_root: str,
    relative_path: str,
    *,
    key: str,
    metric: str,
    name: str,
    interpretation: str,
) -> Optional[MeasuredEvidence]:
    """Collect an absolute metric from a ``{"aggregate": ...}`` artifact.

    Proposal-ranking audits sometimes represent a single checkpoint without a
    matched baseline. For those rows the baseline is recorded as zero and the
    run value is interpreted as an absolute diagnostic, not a paired gain.
    Complexity is ``O(file_size)`` for JSON loading and ``O(1)`` extraction.
    """
    payload = _load_json(os.path.join(project_root, relative_path))
    if payload is None:
        return None
    aggregate = payload.get("aggregate", {})
    if not isinstance(aggregate, Mapping):
        return None
    value = _finite_or_none(aggregate.get(key))
    if value is None:
        return None
    n = aggregate.get("n_records")
    return MeasuredEvidence(
        name=name,
        metric=metric,
        baseline=0.0,
        run=value,
        delta=value,
        ci_low=None,
        ci_high=None,
        paired_p=None,
        n=int(n) if isinstance(n, int) else None,
        source=relative_path,
        interpretation=interpretation,
    )


def _codon_lattice_metric(
    project_root: str,
    relative_path: str,
    *,
    source_key: str,
    optimized_key: str,
    metric: str,
    name: str,
    interpretation: str,
) -> Optional[MeasuredEvidence]:
    payload = _load_json(os.path.join(project_root, relative_path))
    if payload is None:
        return None
    summary = payload.get("summary", {})
    if not isinstance(summary, Mapping):
        return None
    baseline = _finite_or_none(summary.get(source_key))
    run = _finite_or_none(summary.get(optimized_key))
    if baseline is None or run is None:
        return None
    n = summary.get("n", summary.get("n_records"))
    return MeasuredEvidence(
        name=name,
        metric=metric,
        baseline=baseline,
        run=run,
        delta=run - baseline,
        ci_low=None,
        ci_high=None,
        paired_p=None,
        n=int(n) if isinstance(n, int) else None,
        source=relative_path,
        interpretation=interpretation,
    )


def _summary_delta_metric(
    project_root: str,
    relative_path: str,
    *,
    source_key: str,
    optimized_key: str,
    metric: str,
    name: str,
    interpretation: str,
) -> Optional[MeasuredEvidence]:
    """Collect a source-vs-optimized metric from a benchmark summary.

    This generic artifact reader is used by executable baselines that write a
    ``{"summary": ...}`` payload. For baseline value ``b`` and optimized value
    ``r``, the reported delta is ``r-b``. Complexity is ``O(file_size)`` for
    JSON loading and ``O(1)`` for extraction.
    """
    payload = _load_json(os.path.join(project_root, relative_path))
    if payload is None:
        return None
    summary = payload.get("summary", {})
    if not isinstance(summary, Mapping):
        return None
    baseline = _finite_or_none(summary.get(source_key))
    run = _finite_or_none(summary.get(optimized_key))
    if baseline is None or run is None:
        return None
    n = summary.get("n", summary.get("n_records"))
    return MeasuredEvidence(
        name=name,
        metric=metric,
        baseline=baseline,
        run=run,
        delta=run - baseline,
        ci_low=None,
        ci_high=None,
        paired_p=None,
        n=int(n) if isinstance(n, int) else None,
        source=relative_path,
        interpretation=interpretation,
    )


def _cascade_metric(
    project_root: str,
    relative_path: str,
    *,
    metric_key: str,
    baseline_key: str,
    metric: str,
    name: str,
    interpretation: str,
) -> Optional[MeasuredEvidence]:
    """Collect a cascade audit metric from ``{"aggregate": ...}`` artifacts.

    ``baseline_key`` is usually a source or full-pool reference metric from the
    same audit, so the delta remains self-contained. Complexity is ``O(file)``.
    """
    payload = _load_json(os.path.join(project_root, relative_path))
    if payload is None:
        return None
    aggregate = payload.get("aggregate", {})
    if not isinstance(aggregate, Mapping):
        return None
    baseline = 0.0 if baseline_key == "__zero__" else _finite_or_none(aggregate.get(baseline_key))
    run = _finite_or_none(aggregate.get(metric_key))
    if baseline is None or run is None:
        return None
    n = aggregate.get("n_records")
    return MeasuredEvidence(
        name=name,
        metric=metric,
        baseline=baseline,
        run=run,
        delta=run - baseline,
        ci_low=None,
        ci_high=None,
        paired_p=None,
        n=int(n) if isinstance(n, int) else None,
        source=relative_path,
        interpretation=interpretation,
    )


def _leakage_metric(
    project_root: str,
    relative_path: str,
    *,
    key: str,
    metric: str,
    name: str,
    interpretation: str,
) -> Optional[MeasuredEvidence]:
    payload = _load_json(os.path.join(project_root, relative_path))
    if payload is None:
        return None
    summary = payload.get("summary", {})
    if not isinstance(summary, Mapping):
        return None
    value = _finite_or_none(summary.get(key))
    if value is None:
        return None
    n = summary.get("n_query")
    return MeasuredEvidence(
        name=name,
        metric=metric,
        baseline=0.0,
        run=value,
        delta=value,
        ci_low=None,
        ci_high=None,
        paired_p=None,
        n=int(n) if isinstance(n, int) else None,
        source=relative_path,
        interpretation=interpretation,
    )


def collect_measured_evidence(project_root: str) -> list[MeasuredEvidence]:
    """Collect measured local evidence from benchmark artifacts.

    The oracle-gap health check uses measured deltas:

    ``remaining_gap = delta_TE(all_legal_oracle) - delta_TE(model_ranker)``.

    This function only extracts raw evidence rows; gap arithmetic is performed
    in :func:`summarize_oracle_gap`. Complexity is ``O(F + R)`` for ``F`` files
    and comparison rows ``R``.
    """
    evidence: list[MeasuredEvidence] = []
    final_base = "benchmark/proposal_ranking_t5_base_full1k_head64.json"
    final_ranker = "benchmark/proposal_ranking_t5_ranker_full1k_head64.json"
    if os.path.exists(os.path.join(project_root, "benchmark/proposal_ranking_t5_full1k_final.json")):
        final_base = "benchmark/proposal_ranking_t5_base_full1k_head64.json"
    if os.path.exists(os.path.join(project_root, "benchmark/proposal_ranking_t5_ranker_full1k_final_head64.json")):
        final_ranker = "benchmark/proposal_ranking_t5_ranker_full1k_final_head64.json"
    utr_teacher_ranker = "benchmark/proposal_ranking_t5_utr_teacher_head64.json"
    hybrid_teacher_ranker = "benchmark/proposal_ranking_t5_hybrid_teacher_head64.json"
    sequential_teacher_ranker = "benchmark/proposal_ranking_t5_full1k_then_utr_teacher_head64.json"
    sourceaware_hybrid_ranker = "benchmark/proposal_ranking_t5_sourceaware_hybrid_teacher_head64.json"
    cascade_hardneg_ranker = "benchmark/proposal_ranking_t5_cascade_hardneg_teacher_head64.json"
    stage_a10k_teacher = "benchmark/proposal_ranking_t5_stage_a10k_head1024.json"
    stage_a10k_base_head64 = "benchmark/proposal_ranking_t5_base_stage_a10k_head64.json"
    stage_a10k_ranker_head64 = "benchmark/proposal_ranking_t5_ranker_stage_a10k_head64.json"

    candidates = [
        _evidence_from_comparison(
            project_root,
            "benchmark/t5_ranker_comparison.json",
            run_label="ranker_full1k_top32",
            metric="delta_oracle_te_vs_source",
            name_prefix="TE-ranker fair head decoding",
            interpretation="Matched-config model-only ranker gain over the same base snapshot.",
        ),
        _evidence_from_comparison(
            project_root,
            "benchmark/t5_ranker_full1k_head256_comparison.json",
            run_label="ranker_full1k_top32",
            metric="delta_oracle_te_vs_source",
            name_prefix="TE-ranker fair head decoding",
            interpretation="Larger held-out slice; generated automatically when the head256 job finishes.",
        ),
        _evidence_from_comparison(
            project_root,
            "benchmark/t5_guidance_comparison.json",
            run_label="all_proposal_te_guided",
            metric="delta_oracle_te_vs_source",
            name_prefix="All-legal-proposal TE oracle upper bound",
            interpretation="Oracle-guided legal proposal upper bound; use as gap target, not model-only claim.",
        ),
        _evidence_from_comparison(
            project_root,
            "benchmark/compare_t5_head256_cascade_vs_seq_top64.json",
            run_label="cascade_top64",
            metric="delta_oracle_te_vs_source",
            name_prefix="Cascade decoding source-aware->sequential top64",
            interpretation=(
                "Claim-grade decoding comparison on head256: source-aware "
                "hybrid recall keeps top-64 candidates and sequential precision "
                "reranks them under the same effective candidate width."
            ),
        ),
        _evidence_from_comparison(
            project_root,
            "benchmark/compare_t5_head256_cascade_vs_seq_top64.json",
            run_label="cascade_top64",
            metric="mean_oracle_te",
            name_prefix="Cascade decoding source-aware->sequential top64",
            interpretation=(
                "Absolute TE proxy in the same paired head256 cascade-vs-"
                "sequential top64 benchmark."
            ),
        ),
        _evidence_from_comparison(
            project_root,
            "benchmark/compare_t5_head256_hardneg_v2_top64.json",
            run_label="hardneg_v2_top64",
            metric="delta_oracle_te_vs_source",
            name_prefix="Hard-negative v2 direct top64 decoding",
            interpretation=(
                "Claim-grade head256 direct decoding with the cascade hard-"
                "negative v2 ranker. This is the first current head256 model-"
                "only decoding run with paired TE p < 0.05 versus sequential "
                "precision top64."
            ),
        ),
        _evidence_from_comparison(
            project_root,
            "benchmark/compare_t5_head256_hardneg_v2_top64.json",
            run_label="hardneg_v2_top64",
            metric="mean_oracle_te",
            name_prefix="Hard-negative v2 direct top64 decoding",
            interpretation=(
                "Absolute TE proxy for the same hard-negative v2 direct top64 "
                "head256 benchmark."
            ),
        ),
        _evidence_from_comparison(
            project_root,
            "benchmark/compare_t5_head1024_hardneg_v2_top64.json",
            run_label="hardneg_v2_top64",
            metric="delta_oracle_te_vs_source",
            name_prefix="Hard-negative v2 direct top64 decoding",
            interpretation=(
                "Scale-up validation on head1024. The same hard-negative v2 "
                "top64 decoder remains significantly better than sequential "
                "precision top64 under matched seeds and evaluation caps."
            ),
        ),
        _evidence_from_comparison(
            project_root,
            "benchmark/compare_t5_head1024_hardneg_v2_top64.json",
            run_label="hardneg_v2_top64",
            metric="mean_oracle_te",
            name_prefix="Hard-negative v2 direct top64 decoding",
            interpretation=(
                "Absolute TE proxy for the head1024 scale-up validation."
            ),
        ),
        _evidence_from_comparison(
            project_root,
            "benchmark/compare_mo_fusion_vs_te_only_head256.json",
            run_label="mo_grpo_top64",
            metric="delta_oracle_te_vs_source",
            name_prefix="Multi-objective grpo-fusion ranker top64 (vs single-TE control)",
            interpretation=(
                "Roadmap upgrade #1 controlled ablation on head256: holding "
                "candidate generation, ranker architecture and distill recipe "
                "fixed and varying ONLY the teacher reward, the grpo per-metric "
                "standardized fusion is the numerically top fusion "
                "(delta +0.01114) and significantly beats the te_only single-TE "
                "control; all hard constraints stay 1.0. grpo/scalar/pareto are "
                "statistically tied with each other (see head-to-head compares)."
            ),
        ),
        _evidence_from_comparison(
            project_root,
            "benchmark/compare_mo_fusion_vs_te_only_head256.json",
            run_label="mo_scalar_top64",
            metric="delta_oracle_te_vs_source",
            name_prefix="Multi-objective scalar-fusion ranker top64 (vs single-TE control)",
            interpretation=(
                "Weighted scalarization over 6 objectives (TE/MRL/CAI/GC/access/"
                "uAUG). Significantly beats the te_only single-TE control on "
                "head256 with all hard constraints 1.0."
            ),
        ),
        _evidence_from_comparison(
            project_root,
            "benchmark/compare_mo_fusion_vs_te_only_head256.json",
            run_label="mo_pareto_top64",
            metric="delta_oracle_te_vs_source",
            name_prefix="Multi-objective pareto-fusion ranker top64 (vs single-TE control)",
            interpretation=(
                "NSGA-II front-major Pareto fusion. Significantly beats the "
                "te_only single-TE control on head256 with all hard constraints "
                "1.0; ties scalar/grpo on TE."
            ),
        ),
        _evidence_from_comparison(
            project_root,
            "benchmark/compare_grpo_vs_hardneg_v2_head256.json",
            run_label="mo_grpo_top64",
            metric="delta_oracle_te_vs_source",
            name_prefix="Multi-objective grpo-fusion ranker top64 (vs prior champion hardneg_v2)",
            interpretation=(
                "The multi-objective fusion ranker also significantly beats the "
                "prior head256 champion (hard-negative v2 direct top64) under "
                "matched seeds and eval caps, so richer reward signal — not wider "
                "candidate recall — is what finally converts into top-1 TE."
            ),
        ),
        _evidence_from_comparison(
            project_root,
            "benchmark/compare_mo_fusion_vs_te_only_head1024.json",
            run_label="mo_grpo_top64",
            metric="delta_oracle_te_vs_source",
            name_prefix="Multi-objective grpo-fusion ranker top64 scale-up (vs single-TE control)",
            interpretation=(
                "Roadmap upgrade #1 scale-up check on head1024: same controlled "
                "ablation as head256, but with the true 1024-record slice and "
                "limit=1024. This row verifies whether the multi-objective "
                "fusion gain survives the larger evaluation slice."
            ),
        ),
        _evidence_from_comparison(
            project_root,
            "benchmark/compare_mo_fusion_vs_te_only_head1024.json",
            run_label="mo_scalar_top64",
            metric="delta_oracle_te_vs_source",
            name_prefix="Multi-objective scalar-fusion ranker top64 scale-up (vs single-TE control)",
            interpretation=(
                "Head1024 scale-up for weighted scalarization over TE/MRL/CAI/"
                "GC/access/uAUG under the same candidate generation, ranker "
                "architecture and distillation recipe as the single-TE control."
            ),
        ),
        _evidence_from_comparison(
            project_root,
            "benchmark/compare_mo_fusion_vs_te_only_head1024.json",
            run_label="mo_pareto_top64",
            metric="delta_oracle_te_vs_source",
            name_prefix="Multi-objective pareto-fusion ranker top64 scale-up (vs single-TE control)",
            interpretation=(
                "Head1024 scale-up for NSGA-II front-major Pareto fusion. "
                "Interpret together with paired p-values and hard-constraint "
                "rows from the comparison artifact."
            ),
        ),
        _evidence_from_comparison(
            project_root,
            "benchmark/compare_mo_fusion_vs_hardneg_v2_head1024.json",
            run_label="mo_scalar_top64",
            metric="delta_oracle_te_vs_source",
            name_prefix="Multi-objective scalar-fusion ranker top64 scale-up (vs prior champion hardneg_v2)",
            interpretation=(
                "Head1024 SOTA-trajectory check against the prior hard-negative "
                "v2 champion under matched seeds and evaluation cap."
            ),
        ),
        _evidence_from_comparison(
            project_root,
            "benchmark/compare_mo_fusion_vs_hardneg_v2_head1024.json",
            run_label="mo_pareto_top64",
            metric="delta_oracle_te_vs_source",
            name_prefix="Multi-objective pareto-fusion ranker top64 scale-up (vs prior champion hardneg_v2)",
            interpretation=(
                "Head1024 SOTA-trajectory check against the prior hard-negative "
                "v2 champion. Pareto is the numerically strongest head1024 "
                "fusion mode in the completed true-scale run."
            ),
        ),
        _evidence_from_comparison(
            project_root,
            "benchmark/compare_mo_fusion_vs_hardneg_v2_head1024.json",
            run_label="mo_grpo_top64",
            metric="delta_oracle_te_vs_source",
            name_prefix="Multi-objective grpo-fusion ranker top64 scale-up (vs prior champion hardneg_v2)",
            interpretation=(
                "Optional head1024 SOTA-trajectory check against the prior "
                "hard-negative v2 champion, using the same matched seeds and "
                "evaluation cap as the multi-objective ablation."
            ),
        ),
        _evidence_from_comparison(
            project_root,
            "benchmark/region_adapter_vs_hardneg_v2_top64_head256.json",
            run_label="region_adapter_utr5_top64",
            metric="delta_oracle_te_vs_source",
            name_prefix="Region-specialized 5UTR adapter top64 (vs hardneg_v2)",
            interpretation=(
                "Roadmap upgrade #2 head256 ablation: only the 5UTR region "
                "adapter is trainable over the frozen Stage A head; compare "
                "against the hard-negative v2 top64 prior champion."
            ),
        ),
        _evidence_from_comparison(
            project_root,
            "benchmark/region_adapter_vs_hardneg_v2_top64_head256.json",
            run_label="region_adapter_cds_top64",
            metric="delta_oracle_te_vs_source",
            name_prefix="Region-specialized CDS adapter top64 (vs hardneg_v2)",
            interpretation=(
                "Roadmap upgrade #2 head256 ablation: only the CDS region "
                "adapter is trainable, testing whether coding-lattice capacity "
                "helps T5 decoding without violating hard constraints."
            ),
        ),
        _evidence_from_comparison(
            project_root,
            "benchmark/region_adapter_vs_hardneg_v2_top64_head256.json",
            run_label="region_adapter_utr3_top64",
            metric="delta_oracle_te_vs_source",
            name_prefix="Region-specialized 3UTR adapter top64 (vs hardneg_v2)",
            interpretation=(
                "Roadmap upgrade #2 head256 ablation: only the 3UTR region "
                "adapter is trainable, isolating stability-canvas capacity."
            ),
        ),
        _evidence_from_comparison(
            project_root,
            "benchmark/region_adapter_vs_hardneg_v2_top64_head256.json",
            run_label="region_adapter_all_top64",
            metric="delta_oracle_te_vs_source",
            name_prefix="Region-specialized all-region adapters top64 (vs hardneg_v2)",
            interpretation=(
                "Roadmap upgrade #2 head256 ablation with all three region "
                "adapters trainable. This is the main region-specialized "
                "adapter comparison against the prior hard-negative v2 champion."
            ),
        ),
        _evidence_from_comparison(
            project_root,
            "benchmark/region_adapter_vs_mo_grpo_top64_head256.json",
            run_label="region_adapter_all_top64",
            metric="delta_oracle_te_vs_source",
            name_prefix="Region-specialized all-region adapters top64 (vs MO-GRPO)",
            interpretation=(
                "Roadmap upgrade #2 head256 ablation compared against the "
                "current strongest multi-objective fusion ranker. This guards "
                "against over-claiming wins that only hold versus the older "
                "hard-negative v2 baseline."
            ),
        ),
        _evidence_from_comparison(
            project_root,
            "benchmark/region_adapter_vs_mo_scalar_top64_head256.json",
            run_label="region_adapter_all_top64",
            metric="delta_oracle_te_vs_source",
            name_prefix="Region-specialized all-region adapters top64 (vs MO-scalar)",
            interpretation=(
                "Roadmap upgrade #2 head256 ablation compared against the "
                "scalar multi-objective fusion baseline under matched seeds."
            ),
        ),
        _evidence_from_comparison(
            project_root,
            "benchmark/region_adapter_vs_mo_pareto_top64_head256.json",
            run_label="region_adapter_all_top64",
            metric="delta_oracle_te_vs_source",
            name_prefix="Region-specialized all-region adapters top64 (vs MO-pareto)",
            interpretation=(
                "Roadmap upgrade #2 head256 ablation compared against the "
                "Pareto multi-objective fusion baseline under matched seeds."
            ),
        ),
        _evidence_from_comparison(
            project_root,
            "benchmark/region_adapter_vs_mo_te_only_top64_head256.json",
            run_label="region_adapter_all_top64",
            metric="delta_oracle_te_vs_source",
            name_prefix="Region-specialized all-region adapters top64 (vs MO te_only)",
            interpretation=(
                "Roadmap upgrade #2 head256 ablation compared against the "
                "single-TE control trained with the same multi-objective "
                "distillation recipe."
            ),
        ),
        _proposal_metric(
            project_root,
            final_base,
            final_ranker,
            metric="mean_model_regret",
            name="Head64 proposal-ranking regret",
            interpretation="Lower regret means the model ranks oracle-preferred legal edits nearer the top.",
        ),
        _proposal_metric(
            project_root,
            final_base,
            final_ranker,
            metric="oracle_best_in_model_top_k_fraction",
            name="Head64 oracle-best-in-model-top32 fraction",
            interpretation="Higher fraction means the oracle-best legal edit is inside the model top-32 pool.",
        ),
        _proposal_metric(
            project_root,
            final_base,
            utr_teacher_ranker,
            metric="mean_model_regret",
            name="UTR-teacher head64 proposal-ranking regret",
            interpretation=(
                "UTR one-step teacher fine-tune evaluated under the same full-pool "
                "head64 audit; lower regret is better."
            ),
        ),
        _proposal_metric(
            project_root,
            final_base,
            utr_teacher_ranker,
            metric="oracle_best_in_model_top_k_fraction",
            name="UTR-teacher head64 oracle-best-in-model-top32 fraction",
            interpretation=(
                "UTR one-step teacher fine-tune substantially improves whether "
                "oracle-best proposals are reachable by top-k decoding."
            ),
        ),
        _proposal_metric(
            project_root,
            final_base,
            hybrid_teacher_ranker,
            metric="mean_model_regret",
            name="Hybrid-teacher head64 proposal-ranking regret",
            interpretation=(
                "Direct full-pool plus UTR-teacher fusion trained from Stage A; "
                "lower regret is better."
            ),
        ),
        _proposal_metric(
            project_root,
            final_base,
            hybrid_teacher_ranker,
            metric="oracle_best_in_model_top_k_fraction",
            name="Hybrid-teacher head64 oracle-best-in-model-top32 fraction",
            interpretation=(
                "Direct full-pool plus UTR-teacher fusion evaluated for top-k "
                "oracle-best recall."
            ),
        ),
        _proposal_metric(
            project_root,
            final_base,
            sequential_teacher_ranker,
            metric="mean_model_regret",
            name="Full-then-UTR head64 proposal-ranking regret",
            interpretation=(
                "Sequential fine-tune from the previous full-pool TE ranker onto "
                "the UTR teacher; lower regret is better."
            ),
        ),
        _proposal_metric(
            project_root,
            final_base,
            sequential_teacher_ranker,
            metric="oracle_best_in_model_top_k_fraction",
            name="Full-then-UTR head64 oracle-best-in-model-top32 fraction",
            interpretation=(
                "Sequential fine-tune tradeoff: improves top-1 regret while "
                "tracking whether top-k oracle recall is preserved."
            ),
        ),
        _proposal_metric(
            project_root,
            final_base,
            sourceaware_hybrid_ranker,
            metric="mean_model_regret",
            name="Source-aware hybrid head64 proposal-ranking regret",
            interpretation=(
                "Hybrid teacher trained with source-balanced full/UTR pair "
                "sampling; lower regret is better."
            ),
        ),
        _proposal_metric(
            project_root,
            final_base,
            sourceaware_hybrid_ranker,
            metric="oracle_best_in_model_top_k_fraction",
            name="Source-aware hybrid head64 oracle-best-in-model-top32 fraction",
            interpretation=(
                "Source-balanced full/UTR pair sampling targets top-k recall "
                "without collapsing source-specific teacher objectives."
            ),
        ),
        _proposal_metric(
            project_root,
            final_base,
            cascade_hardneg_ranker,
            metric="mean_model_regret",
            name="Cascade hard-negative v2 head64 proposal-ranking regret",
            interpretation=(
                "Hard-negative teacher mined from cascade win/loss transcripts; "
                "lower regret indicates the precision ranker learned from "
                "cascade failures without losing full-pool TE ordering."
            ),
        ),
        _proposal_metric(
            project_root,
            final_base,
            cascade_hardneg_ranker,
            metric="oracle_best_in_model_top_k_fraction",
            name="Cascade hard-negative v2 head64 oracle-best-in-model-top32 fraction",
            interpretation=(
                "Tracks whether hard-negative v2 preserves source-aware top-k "
                "recall while improving top-1 proposal quality."
            ),
        ),
        _aggregate_metric(
            project_root,
            stage_a10k_teacher,
            key="mean_model_regret",
            metric="mean_model_regret",
            name="Stage A 10k head1024 proposal-ranking regret",
            interpretation=(
                "Absolute proposal-ranking regret for the 10k Stage A snapshot "
                "on the head1024 teacher-export slice; lower is better."
            ),
        ),
        _aggregate_metric(
            project_root,
            stage_a10k_teacher,
            key="oracle_best_in_model_top_k_fraction",
            metric="oracle_best_in_model_top_k_fraction",
            name="Stage A 10k head1024 oracle-best-in-model-top32 fraction",
            interpretation=(
                "Absolute top-k oracle recall for the 10k Stage A snapshot on "
                "the head1024 teacher-export slice."
            ),
        ),
        _proposal_metric(
            project_root,
            stage_a10k_base_head64,
            stage_a10k_ranker_head64,
            metric="mean_model_regret",
            name="Stage A 10k teacher-ranker head64 proposal-ranking regret",
            interpretation=(
                "Matched head64 proposal-ranking audit after distilling the "
                "10k Stage A head1024 teacher; lower regret versus the same "
                "10k base snapshot indicates ranker gain."
            ),
        ),
        _proposal_metric(
            project_root,
            stage_a10k_base_head64,
            stage_a10k_ranker_head64,
            metric="oracle_best_in_model_top_k_fraction",
            name="Stage A 10k teacher-ranker head64 oracle-best-in-model-top32 fraction",
            interpretation=(
                "Matched head64 top-k oracle recall after 10k teacher "
                "distillation."
            ),
        ),
        _cascade_metric(
            project_root,
            "benchmark/cascade_sourceaware_to_sequential_head64.json",
            metric_key="mean_cascade_regret",
            baseline_key="mean_recall_model_regret",
            metric="mean_cascade_regret",
            name="Cascade source-aware->sequential head64 regret (k=32)",
            interpretation=(
                "Two-stage pipeline: source-aware hybrid recall top-32 followed "
                "by sequential precision reranking; lower regret is better."
            ),
        ),
        _cascade_metric(
            project_root,
            "benchmark/cascade_sourceaware_to_sequential_head64.json",
            metric_key="oracle_best_in_recall_top_k_fraction",
            baseline_key="__zero__",
            metric="oracle_best_in_recall_top_k_fraction",
            name="Cascade source-aware->sequential oracle-best recall (k=32)",
            interpretation="Fraction of oracle-best proposals retained by the recall stage.",
        ),
        _cascade_metric(
            project_root,
            "benchmark/cascade_sourceaware_to_sequential_head64_k64.json",
            metric_key="mean_cascade_regret",
            baseline_key="mean_precision_full_regret",
            metric="mean_cascade_regret",
            name="Cascade source-aware->sequential head64 regret (k=64)",
            interpretation=(
                "Top-64 recall stage followed by sequential precision reranking; "
                "compared against precision full-pool regret from the same audit."
            ),
        ),
        _cascade_metric(
            project_root,
            "benchmark/cascade_sourceaware_to_sequential_head64_k64.json",
            metric_key="oracle_best_in_recall_top_k_fraction",
            baseline_key="__zero__",
            metric="oracle_best_in_recall_top_k_fraction",
            name="Cascade source-aware->sequential oracle-best recall (k=64)",
            interpretation="Fraction of oracle-best proposals retained by the top-64 recall stage.",
        ),
        _cascade_metric(
            project_root,
            "benchmark/cascade_sourceaware_to_hardneg_head64_k64.json",
            metric_key="mean_cascade_regret",
            baseline_key="mean_precision_full_regret",
            metric="mean_cascade_regret",
            name="Cascade source-aware->hard-negative v2 head64 regret (k=64)",
            interpretation=(
                "Source-aware recall followed by hard-negative v2 precision. "
                "Compare with the v2 full-pool precision regret to quantify "
                "whether recall truncation is now limiting."
            ),
        ),
        _cascade_metric(
            project_root,
            "benchmark/cascade_sourceaware_to_hardneg_head64_k64.json",
            metric_key="oracle_best_in_recall_top_k_fraction",
            baseline_key="__zero__",
            metric="oracle_best_in_recall_top_k_fraction",
            name="Cascade source-aware->hard-negative v2 oracle-best recall (k=64)",
            interpretation="Fraction of oracle-best proposals retained by the source-aware top-64 recall stage.",
        ),
        _codon_lattice_metric(
            project_root,
            "benchmark/codon_lattice_dp_head256.json",
            source_key="mean_source_cai",
            optimized_key="mean_optimized_cai",
            metric="mean_cai",
            name="CDS codon-lattice DP CAI",
            interpretation="Executable CDS-only synonymous codon-lattice baseline; higher CAI is better.",
        ),
        _codon_lattice_metric(
            project_root,
            "benchmark/codon_lattice_dp_head256.json",
            source_key="mean_source_gc",
            optimized_key="mean_optimized_gc",
            metric="mean_gc",
            name="CDS codon-lattice DP GC shift",
            interpretation="GC shift induced by the executable codon-lattice DP baseline.",
        ),
        _leakage_metric(
            project_root,
            "benchmark/leakage_ranker_head256_vs_gencode.json",
            key="flagged_fraction",
            metric="flagged_fraction",
            name="Foundation leakage audit head256 vs GENCODE",
            interpretation="Fraction of query transcripts flagged as exact or near-overlap with the reference corpus.",
        ),
        _leakage_metric(
            project_root,
            "benchmark/leakage_ranker_head256_vs_gencode.json",
            key="exact_match_count",
            metric="exact_match_count",
            name="Foundation leakage audit exact matches",
            interpretation="Exact sequence matches between head256 query sources and the GENCODE reference corpus.",
        ),
        _summary_delta_metric(
            project_root,
            "benchmark/utr_local_search_head256.json",
            source_key="mean_source_te",
            optimized_key="mean_optimized_te",
            metric="mean_oracle_te",
            name="UTR local-search TE baseline",
            interpretation=(
                "Executable Optimus/UTailoR-style 5'UTR-only predictor-guided "
                "beam search; CDS and 3'UTR are unchanged."
            ),
        ),
        _summary_delta_metric(
            project_root,
            "benchmark/utr_teacher_head256.json",
            source_key="mean_source_te",
            optimized_key="mean_best_candidate_te",
            metric="best_one_step_oracle_te",
            name="UTR one-step teacher headroom",
            interpretation=(
                "Ranker-compatible one-step 5'UTR teacher export; delta is the "
                "mean best oracle-labelled single edit available for distillation."
            ),
        ),
    ]
    for item in candidates:
        if item is not None:
            evidence.append(item)
    return evidence


def _next_gates(evidence: Sequence[MeasuredEvidence]) -> list[str]:
    """Return context-aware next gates from currently available artifacts.

    Complexity is ``O(E)`` over measured evidence rows.
    """
    has_head256 = any("(n=256)" in row.name for row in evidence)
    has_codon_dp = any("codon-lattice DP" in row.name for row in evidence)
    has_leakage = any("leakage audit" in row.name.lower() for row in evidence)
    has_utr_search = any("UTR local-search" in row.name for row in evidence)
    has_utr_teacher = any("UTR one-step teacher" in row.name for row in evidence)
    has_mo_head1024 = any(
        row.source == "benchmark/compare_mo_fusion_vs_te_only_head1024.json"
        for row in evidence
    )
    has_region_adapter_head256 = any(
        row.source == "benchmark/region_adapter_vs_hardneg_v2_top64_head256.json"
        for row in evidence
    )
    gates = []
    if not has_head256:
        gates.append("Finish fair head256 ranker comparison and refresh this report.")
    if not has_mo_head1024:
        gates.append("Finish true head1024 multi-objective fusion scale-up and refresh this report.")
    if not has_region_adapter_head256:
        gates.append("Finish head256 region-specialized adapter ablation and refresh this report.")
    if not has_codon_dp:
        gates.append("Run the executable CDS codon-lattice DP baseline on the public head256 slice.")
    if not has_leakage:
        gates.append("Run k-mer leakage audit before frozen mRNA foundation-model comparisons.")
    if not has_utr_search:
        gates.append("Run the executable UTR local-search baseline on the public head256 slice.")
    if not has_utr_teacher:
        gates.append("Export UTR one-step oracle teacher JSONL for proposal-ranker distillation.")
    gates.extend(
        [
            "Finish EnsembleDesign, UTRGAN paper-default, and codonGPT 10-seed runs; obtain Prot2RNA artifacts if released.",
            "Add frozen mRNA foundation embeddings with leakage audits.",
            "Scale beyond the 1000-step Stage A snapshot and repeat proposal-ranking distillation.",
        ]
    )
    return gates


def _load_external_sota_dry_run(project_root: str) -> Optional[Mapping[str, object]]:
    """Load the latest external SOTA dry-run readiness artifact if present."""
    path = os.path.join(project_root, "benchmark/external_sota/dry_run_t5_head1024/summary.json")
    return _load_json(path)


def _load_external_sota_input_pack(project_root: str) -> Optional[Mapping[str, object]]:
    """Load the latest external SOTA standardized input pack if present."""
    path = os.path.join(project_root, "benchmark/external_sota/input_pack_t5_head1024/summary.json")
    return _load_json(path)


def _load_external_sota_real_run_audit(project_root: str) -> Optional[Mapping[str, object]]:
    """Load the latest external SOTA real-run output audit if present."""
    path = os.path.join(project_root, "docs/external_sota_real_run_audit.json")
    return _load_json(path)


def _load_t5_external_utr_comparison(
    project_root: str,
) -> Optional[Mapping[str, object]]:
    """Load the shared-oracle MEF/UTRGAN T5 comparison if present."""
    path = os.path.join(
        project_root,
        "docs",
        "t5_external_utr_baseline_comparison.json",
    )
    return _load_json(path)


def _load_sota_readiness(project_root: str) -> Optional[Mapping[str, object]]:
    """Load the unified SOTA-readiness audit if present."""
    path = os.path.join(project_root, "docs/sota_readiness_audit_head256.json")
    return _load_json(path)


def _load_dataset_survey(project_root: str) -> dict[str, object]:
    """Return an audit summary for the dataset/split survey document.

    The survey is protocol evidence only. It records benchmark alignment and
    leakage questions for external SOTA methods, but it is not a real external
    metric reproduction. Complexity is ``O(file_size + M)``.
    """
    rel_path = "docs/mrna_dataset_survey.md"
    path = os.path.join(project_root, rel_path)
    if not os.path.exists(path):
        return {
            "status": "missing",
            "path": rel_path,
            "covered_methods": [],
            "missing_methods": list(DATASET_SURVEY_METHODS),
            "protocol_ready": False,
            "claim_policy": "Missing survey; do not claim external SOTA dataset alignment.",
        }
    with open(path, "rb") as fh:
        data = fh.read()
    text = data.decode("utf-8")
    covered = [method for method in DATASET_SURVEY_METHODS if method in text]
    missing = [method for method in DATASET_SURVEY_METHODS if method not in covered]
    leakage_terms = ("leakage", "split", "license", "许可")
    return {
        "status": "ready" if not missing else "incomplete",
        "path": rel_path,
        "sha256": hashlib.sha256(data).hexdigest(),
        "n_bytes": len(data),
        "covered_methods": covered,
        "missing_methods": missing,
        "mentions_leakage_or_split": any(term in text for term in leakage_terms),
        "protocol_ready": not missing and any(term in text for term in leakage_terms),
        "claim_policy": (
            "Dataset survey is protocol/landscape evidence only. Do not claim "
            "external SOTA reproduction until official weights/executables and "
            "leakage-free splits are configured."
        ),
    }


def summarize_oracle_gap(evidence: Sequence[MeasuredEvidence]) -> dict[str, object]:
    """Summarize remaining model-only gap to the oracle legal-proposal ceiling.

    Let ``u`` be the measured all-legal oracle upper-bound improvement and
    ``r`` be the best measured ranker improvement. The remaining gap is

    ``G = u - r``.

    ``G > 0`` means the legal edit space still contains higher-TE candidates
    than the current model ranks into its stochastic top-k decoding path.
    Complexity is ``O(E)`` for ``E`` evidence rows.
    """
    oracle = None
    rankers: list[MeasuredEvidence] = []
    for row in evidence:
        if row.metric != "delta_oracle_te_vs_source":
            continue
        if "upper bound" in row.name.lower():
            oracle = row
        elif "ranker" in row.name.lower():
            rankers.append(row)
    best_ranker = max(rankers, key=lambda item: item.run, default=None)
    if oracle is None or best_ranker is None:
        return {
            "status": "insufficient_evidence",
            "message": "Need both all-legal oracle upper bound and ranker comparison artifacts.",
        }
    return {
        "status": "measured",
        "oracle_upper_bound_delta_te": oracle.run,
        "best_ranker_delta_te": best_ranker.run,
        "remaining_gap": oracle.run - best_ranker.run,
        "oracle_source": oracle.source,
        "ranker_source": best_ranker.source,
        "caution": (
            "The upper bound is an oracle-guided target. Treat the gap as a health check "
            "unless generated under the same guarded config as the model-only comparison."
        ),
    }


def _build_sota_gap_report_development(project_root: str) -> dict[str, object]:
    """Build the full SOTA gap report object.

    Complexity is ``O(E + S + M)`` where ``E`` is measured evidence rows,
    ``S`` is static SOTA rows and ``M`` is the external-model registry size.
    """
    evidence = collect_measured_evidence(project_root)
    registered = set(available_external_models())
    sota_rows = []
    for row in SOTA_REFERENCE_ROWS:
        enriched = dict(row)
        enriched["registered_external_model"] = row["method"] in registered
        sota_rows.append(enriched)
    return {
        "project_root": os.path.abspath(project_root),
        "measured_evidence": [item.to_dict() for item in evidence],
        "oracle_gap_health_check": summarize_oracle_gap(evidence),
        "sota_references": sota_rows,
        "external_sota_dry_run": _load_external_sota_dry_run(project_root),
        "external_sota_input_pack": _load_external_sota_input_pack(project_root),
        "external_sota_real_run_audit": _load_external_sota_real_run_audit(project_root),
        "t5_external_utr_comparison": _load_t5_external_utr_comparison(project_root),
        "dataset_survey_audit": _load_dataset_survey(project_root),
        "sota_readiness_audit": _load_sota_readiness(project_root),
        "next_gates": _next_gates(evidence),
    }


def build_sota_gap_report(
    project_root: str,
    run_mode: str = "development",
    artifact_paths: Optional[Sequence[str]] = None,
) -> dict[str, object]:
    if normalize_run_mode(run_mode) == "paper":
        return paper_builder_gate("sota_gap_report", project_root, artifact_paths, __file__)
    report = _build_sota_gap_report_development(project_root)
    report.update({"claim_tier": "development_only", "paper_eligible": False})
    return report


def _fmt(value: object) -> str:
    number = _finite_or_none(value)
    if number is None:
        return "NA"
    if abs(number) >= 1:
        return f"{number:.4f}"
    return f"{number:.5f}"


def write_report_markdown(report: Mapping[str, object], path: str) -> str:
    """Write a Markdown SOTA gap report. Complexity: ``O(E + S)``."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    evidence = report.get("measured_evidence", [])
    sota = report.get("sota_references", [])
    gap = report.get("oracle_gap_health_check", {})
    external_dry_run = report.get("external_sota_dry_run")
    external_input_pack = report.get("external_sota_input_pack")
    external_real_run = report.get("external_sota_real_run_audit")
    t5_external_utr = report.get("t5_external_utr_comparison")
    dataset_survey = report.get("dataset_survey_audit")
    readiness = report.get("sota_readiness_audit")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# mRNA-EditFlow SOTA Gap Report\n\n")
        fh.write("## Measured MEF Evidence\n\n")
        fh.write("| Evidence | Metric | Baseline | Run | Delta | 95% CI | paired p | n | Source |\n")
        fh.write("|---|---|---:|---:|---:|---:|---:|---:|---|\n")
        if isinstance(evidence, Sequence):
            for row in evidence:
                if not isinstance(row, Mapping):
                    continue
                ci = "NA"
                if row.get("ci_low") is not None and row.get("ci_high") is not None:
                    ci = f"[{_fmt(row.get('ci_low'))}, {_fmt(row.get('ci_high'))}]"
                fh.write(
                    f"| {row.get('name', '')} | `{row.get('metric', '')}` | "
                    f"{_fmt(row.get('baseline'))} | {_fmt(row.get('run'))} | "
                    f"{_fmt(row.get('delta'))} | {ci} | {_fmt(row.get('paired_p'))} | "
                    f"{row.get('n', 'NA')} | `{row.get('source', '')}` |\n"
                )
        fh.write("\n## Oracle Gap Health Check\n\n")
        if isinstance(gap, Mapping) and gap.get("status") == "measured":
            fh.write(
                f"- All-legal oracle delta_TE: `{_fmt(gap.get('oracle_upper_bound_delta_te'))}`\n"
            )
            fh.write(f"- Best ranker delta_TE: `{_fmt(gap.get('best_ranker_delta_te'))}`\n")
            fh.write(f"- Remaining oracle gap: `{_fmt(gap.get('remaining_gap'))}`\n")
            fh.write(f"- Caution: {gap.get('caution', '')}\n")
        else:
            fh.write(f"- Status: `{gap.get('status', 'missing') if isinstance(gap, Mapping) else 'missing'}`\n")
        fh.write("\n## External SOTA Integration Table\n\n")
        fh.write("| Method | Venue/year | Scope | Accuracy/F1 signal | Speed/scale | MEF gap/action | Registered | Citation |\n")
        fh.write("|---|---|---|---|---|---|---:|---|\n")
        if isinstance(sota, Sequence):
            for row in sota:
                if not isinstance(row, Mapping):
                    continue
                registered = "yes" if row.get("registered_external_model") else "no"
                fh.write(
                    f"| {row.get('method', '')} | {row.get('venue_year', '')} | "
                    f"{row.get('scope', '')} | {row.get('accuracy_f1', '')} | "
                    f"{row.get('speed_scale', '')} | {row.get('mef_gap', '')} | "
                    f"{registered} | {row.get('citation_url', '')} |\n"
                )
        fh.write("\n## External SOTA Dry-Run Readiness\n\n")
        if isinstance(external_dry_run, Mapping):
            fh.write(
                f"- Status: `{external_dry_run.get('status', '')}`; "
                f"task: `{external_dry_run.get('task_id', '')}`; "
                f"ready: `{external_dry_run.get('n_executable_ready', 0)}` / "
                f"{external_dry_run.get('n_models', 0)}\n\n"
            )
            dataset = external_dry_run.get("dataset", {})
            if isinstance(dataset, Mapping):
                fh.write(
                    "- Dataset audit: "
                    f"split=`{dataset.get('split_name', '')}`, "
                    f"seed=`{dataset.get('seed')}`, "
                    f"records=`{dataset.get('record_count_effective')}` / "
                    f"{dataset.get('record_count_total')}, "
                    f"sha256=`{dataset.get('sha256') or 'NA'}`\n\n"
                )
            hardware = external_dry_run.get("hardware", {})
            if isinstance(hardware, Mapping):
                fh.write(
                    "- Hardware audit: "
                    f"label=`{hardware.get('label')}`, "
                    f"hostname=`{hardware.get('hostname')}`, "
                    f"machine=`{hardware.get('machine')}`\n\n"
                )
            contract = external_dry_run.get("artifact_contract", {})
            if isinstance(contract, Mapping) and contract.get("real_metric_policy"):
                fh.write(f"- Real metric policy: {contract.get('real_metric_policy')}\n\n")
            fh.write("| Model | Status | Command candidates | Protocol scope | Next setup |\n")
            fh.write("|---|---|---|---|---|\n")
            rows = external_dry_run.get("rows", [])
            if isinstance(rows, Sequence):
                for row in rows:
                    if not isinstance(row, Mapping):
                        continue
                    candidates = row.get("command_candidates", [])
                    if isinstance(candidates, Sequence) and not isinstance(candidates, (str, bytes)):
                        candidate_text = ", ".join(str(x) for x in candidates)
                    else:
                        candidate_text = str(candidates)
                    protocol_scope = row.get("protocol_difference") or row.get("notes", "")
                    if row.get("status") == "executable_ready":
                        next_setup = "Run the real adapter and write measured outputs under `benchmark/external_sota/`."
                    else:
                        next_setup = "Set one candidate as an environment variable or install it on PATH, then rerun."
                    candidate_audit = row.get("candidate_audit", [])
                    if isinstance(candidate_audit, Sequence) and not isinstance(candidate_audit, (str, bytes)):
                        audit_text = "; ".join(
                            f"{item.get('candidate')}={item.get('status')}"
                            for item in candidate_audit
                            if isinstance(item, Mapping)
                        )
                        if audit_text:
                            next_setup = f"{next_setup} Candidate audit: {audit_text}."
                    fh.write(
                        f"| {row.get('model_name', '')} | `{row.get('status', '')}` | "
                        f"`{candidate_text}` | {protocol_scope} | {next_setup} |\n"
                    )
        else:
            fh.write("- Missing `benchmark/external_sota/dry_run_t5_head1024/summary.json`.\n")
        fh.write("\n## External SOTA Input Pack\n\n")
        if isinstance(external_input_pack, Mapping):
            dataset = external_input_pack.get("dataset", {})
            if not isinstance(dataset, Mapping):
                dataset = {}
            outputs = external_input_pack.get("outputs", {})
            if not isinstance(outputs, Mapping):
                outputs = {}
            fh.write(
                f"- Ready for external real run: "
                f"`{external_input_pack.get('ready_for_external_real_run')}`; "
                f"ready for external SOTA claim: "
                f"`{external_input_pack.get('ready_for_external_sota_claim')}`\n"
            )
            fh.write(
                f"- Dataset: split=`{dataset.get('split_name')}`, seed=`{dataset.get('seed')}`, "
                f"records=`{dataset.get('record_count_effective')}` / "
                f"{dataset.get('record_count_total')}, "
                f"sha256=`{dataset.get('records_jsonl_sha256')}`\n"
            )
            fh.write(
                f"- Rows: CDS/protein-conditioned=`{external_input_pack.get('n_cds_protein_rows')}`, "
                f"5'UTR-only=`{external_input_pack.get('n_utr5_rows')}`, "
                f"skipped invalid CDS=`{external_input_pack.get('n_skipped_invalid_cds')}`\n"
            )
            fh.write(
                f"- Pack SHA: CDS=`{outputs.get('cds_protein_jsonl_sha256')}`, "
                f"5'UTR=`{outputs.get('utr5_jsonl_sha256')}`, "
                f"metric schema=`{outputs.get('metric_schema_json_sha256')}`\n"
            )
            claim_policy = external_input_pack.get("claim_policy")
            if claim_policy:
                fh.write(f"- Claim policy: {claim_policy}\n")
        else:
            fh.write("- Missing `benchmark/external_sota/input_pack_t5_head1024/summary.json`.\n")
        fh.write("\n## External SOTA Real-Run Audit\n\n")
        if isinstance(external_real_run, Mapping):
            summary = external_real_run.get("summary", {})
            if not isinstance(summary, Mapping):
                summary = {}
            fh.write(
                f"- Audit complete: `{summary.get('audit_complete')}`; "
                f"real metric table ready: "
                f"`{summary.get('ready_for_external_real_metric_table')}`; "
                f"external metric claim ready: "
                f"`{summary.get('ready_for_external_sota_metric_claim')}`\n"
            )
            fh.write(
                f"- Measured models: `{summary.get('n_models_measured')}` / "
                f"`{summary.get('n_models_expected')}`; invalid: "
                f"`{summary.get('n_models_invalid')}`; missing: "
                f"`{summary.get('n_models_missing')}`\n"
            )
            rows = external_real_run.get("rows", [])
            if isinstance(rows, Sequence):
                fh.write("\n| Model | Task family | Status | Outputs | Constraints | Reasons |\n")
                fh.write("|---|---|---|---:|---|---|\n")
                for row in rows:
                    if not isinstance(row, Mapping):
                        continue
                    reasons = row.get("failure_reasons", [])
                    if isinstance(reasons, Sequence) and not isinstance(reasons, (str, bytes)):
                        reason_text = ", ".join(str(item) for item in reasons)
                    else:
                        reason_text = str(reasons)
                    fh.write(
                        f"| {row.get('model_name')} | {row.get('task_family')} | "
                        f"`{row.get('status')}` | {row.get('n_outputs')} / "
                        f"{row.get('expected_input_rows')} | "
                        f"`{row.get('hard_constraints_exact_1')}` | {reason_text} |\n"
                    )
        else:
            fh.write("- Missing `docs/external_sota_real_run_audit.json`.\n")
        fh.write("\n## T5 External 5'UTR Comparison\n\n")
        if isinstance(t5_external_utr, Mapping):
            summary = t5_external_utr.get("summary", {})
            if not isinstance(summary, Mapping):
                summary = {}
            fh.write(
                f"- Descriptive table ready: "
                f"`{summary.get('ready_for_t5_utr_descriptive_table')}`; "
                f"model-only head-to-head ready: "
                f"`{summary.get('ready_for_model_only_head_to_head')}`; "
                f"MEF superiority claim ready: "
                f"`{summary.get('ready_for_mef_superiority_claim')}`\n"
            )
            fh.write(
                f"- Hard constraints exact-1: "
                f"`{summary.get('hard_constraints_exact_1')}`; "
                f"paired inference ready: "
                f"`{summary.get('ready_for_paired_per_record_inference')}`\n\n"
            )
            fh.write(
                "| Method | Status | n | TE | delta TE | uAUG | Kozak | "
                "access | UTR edit | CDS/3UTR/protein |\n"
            )
            fh.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---|\n")
            rows = t5_external_utr.get("rows", [])
            if isinstance(rows, Sequence):
                for row in rows:
                    if not isinstance(row, Mapping):
                        continue
                    fh.write(
                        f"| {row.get('method')} | `{row.get('status')}` | "
                        f"{row.get('n', 'NA')} | "
                        f"{_fmt(row.get('mean_te_proxy'))} | "
                        f"{_fmt(row.get('mean_te_proxy_delta_vs_native'))} | "
                        f"{_fmt(row.get('mean_uaug_count'))} | "
                        f"{_fmt(row.get('mean_kozak_score'))} | "
                        f"{_fmt(row.get('mean_start_accessibility_proxy'))} | "
                        f"{_fmt(row.get('mean_utr_edit_distance_vs_native'))} | "
                        f"{_fmt(row.get('cds_unchanged_fraction'))}/"
                        f"{_fmt(row.get('three_utr_unchanged_fraction'))}/"
                        f"{_fmt(row.get('protein_identity_exact_1_fraction'))} |\n"
                    )
            fh.write(
                f"\n- Remaining fairness action: "
                f"{summary.get('remaining_model_fairness_action')}\n"
            )
        else:
            fh.write(
                "- Missing `docs/t5_external_utr_baseline_comparison.json`.\n"
            )
        fh.write("\n## Dataset Survey / Leakage Alignment\n\n")
        if isinstance(dataset_survey, Mapping):
            fh.write(
                f"- Status: `{dataset_survey.get('status')}`; "
                f"protocol ready: `{dataset_survey.get('protocol_ready')}`; "
                f"path: `{dataset_survey.get('path')}`\n"
            )
            if dataset_survey.get("sha256"):
                fh.write(f"- SHA-256: `{dataset_survey.get('sha256')}`\n")
            fh.write(
                f"- Covered methods: `{dataset_survey.get('covered_methods')}`; "
                f"missing methods: `{dataset_survey.get('missing_methods')}`\n"
            )
            fh.write(
                f"- Mentions leakage/split/license controls: "
                f"`{dataset_survey.get('mentions_leakage_or_split')}`\n"
            )
            fh.write(f"- Claim policy: {dataset_survey.get('claim_policy')}\n")
        else:
            fh.write("- Dataset survey audit unavailable.\n")
        fh.write("\n## Unified SOTA Readiness\n\n")
        if isinstance(readiness, Mapping):
            summary = readiness.get("summary", {})
            if not isinstance(summary, Mapping):
                summary = {}
            fh.write(
                f"- All ready for SOTA claim audit: `{summary.get('all_ready_for_sota_claim_audit')}`\n"
            )
            if "positive_sota_claim_ready" in summary:
                fh.write(
                    f"- Positive SOTA claim ready: `{summary.get('positive_sota_claim_ready')}`\n"
                )
                fh.write(
                    "- Internal proxy constrained-optimization claim ready: "
                    f"`{summary.get('ready_for_internal_proxy_constrained_optimization_claim')}`\n"
                )
                fh.write(
                    f"- External SOTA metric claim ready: "
                    f"`{summary.get('ready_for_external_sota_metric_claim')}`\n"
                )
                fh.write(
                    f"- Full de novo claim ready: `{summary.get('ready_for_full_de_novo_claim')}`\n"
                )
                fh.write(
                    f"- Real TE/stability claim ready: "
                    f"`{summary.get('ready_for_real_te_or_stability_claim')}`\n"
                )
                fh.write(
                    f"- True scale-law claim ready: `{summary.get('ready_for_true_scale_law_claim')}`\n"
                )
                fh.write(f"- Wet-lab claim ready: `{summary.get('ready_for_wet_lab_claim')}`\n")
                fh.write(
                    f"- Positive SOTA blockers: `{summary.get('positive_sota_block_reasons')}`\n"
                )
                fh.write(f"- Allowed claim scope: {summary.get('allowed_claim_scope')}\n")
            fh.write(
                f"- Sections ready: `{summary.get('n_sections_ready')}` / `{summary.get('n_sections_expected')}`\n"
            )
            fh.write(f"- Pending sections: `{summary.get('pending_sections')}`\n")
            claim_policy = summary.get("claim_policy")
            if claim_policy:
                fh.write(f"- Claim policy: {claim_policy}\n")
            fh.write("\n| Section | Ready | Key status |\n")
            fh.write("|---|---:|---|\n")
            sections = readiness.get("sections", {})
            if isinstance(sections, Mapping):
                for name, row in sections.items():
                    if not isinstance(row, Mapping):
                        continue
                    audit = row.get("audit", {})
                    audit_summary = audit.get("summary", {}) if isinstance(audit, Mapping) else {}
                    if not isinstance(audit_summary, Mapping):
                        audit_summary = {}
                    if name == "external_sota_protocol":
                        status = (
                            f"protocol ready={audit_summary.get('protocol_ready')}; "
                            f"input pack ready={audit_summary.get('input_pack_ready')}; "
                            f"real-run audit={audit_summary.get('real_run_audit_complete')}; "
                            f"measured={audit_summary.get('n_real_run_models_measured')}/"
                            f"{audit_summary.get('n_real_run_models_expected')}; "
                            f"models={audit_summary.get('n_models')}; "
                            f"executable ready={audit_summary.get('n_executable_ready')}; "
                            "real metrics not claimed"
                        )
                    elif name == "region_adapter":
                        status = (
                            f"compare files {audit_summary.get('n_compare_files_found')}/"
                            f"{audit_summary.get('n_compare_files_expected')}; "
                            f"constraints exact 1={audit_summary.get('all_constraints_exact_1')}"
                        )
                    elif name == "protein_conditioned_gc_sweep":
                        status = (
                            f"points={audit_summary.get('n_points')}; "
                            f"identity exact 1={audit_summary.get('all_points_identity_exact_1')}; "
                            f"pareto metadata={audit_summary.get('pareto_metadata_ok')}"
                        )
                    elif name == "multiobjective_scaleup_claims":
                        status = (
                            f"comparison rows={audit_summary.get('comparison_rows_ready')}; "
                            f"hard constraints={audit_summary.get('summary_constraints_complete')}; "
                            f"head256 strict={audit_summary.get('head256_fusion_vs_te_only_all_strict')}; "
                            "head1024 vs te_only strict="
                            f"{audit_summary.get('head1024_vs_te_only_strict_claim_allowed')}; "
                            f"best signal={audit_summary.get('head1024_vs_te_only_best_signal')}"
                        )
                    elif name == "frozen_foundation_protocol":
                        status = (
                            f"protocol ready={audit_summary.get('protocol_ready')}; "
                            f"leakage gate={audit_summary.get('leakage_gate_passed')}; "
                            f"matched budget={audit_summary.get('matched_budget')}; "
                            f"real/stub arms={audit_summary.get('n_real_arms')}/"
                            f"{audit_summary.get('n_stub_arms')}; real metrics not claimed"
                        )
                    elif name == "t1_t7_evidence_bundle":
                        status = (
                            f"reports ready={audit_summary.get('n_reports_ready')}/"
                            f"{audit_summary.get('n_reports_expected')}; "
                            f"failed checks={audit_summary.get('failed_checks')}; "
                            "proxy/constraint reports only"
                        )
                    else:
                        status = ""
                    fh.write(f"| {name} | `{row.get('ready')}` | {status} |\n")
        else:
            fh.write("- Missing `docs/sota_readiness_audit_head256.json`.\n")
        fh.write("\n## Next Gates\n\n")
        for item in report.get("next_gates", []):
            fh.write(f"- {item}\n")
    return path


def write_report_json(report: Mapping[str, object], path: str) -> str:
    """Write report JSON. Complexity: ``O(report_size)``."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
    return path


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=os.getcwd())
    parser.add_argument("--out-json", default="docs/sota_gap_report.json")
    parser.add_argument("--out-md", default="docs/sota_gap_report.md")
    parser.add_argument("--run-mode", choices=("development", "paper"), default="development")
    parser.add_argument("--paper-artifact", action="append", default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    report = build_sota_gap_report(args.project_root, args.run_mode, args.paper_artifact)
    out_json = args.out_json
    out_md = args.out_md
    if not os.path.isabs(out_json):
        out_json = os.path.join(args.project_root, out_json)
    if not os.path.isabs(out_md):
        out_md = os.path.join(args.project_root, out_md)
    validate_report_output_namespaces(report, (out_json, out_md))
    write_report_json(report, out_json)
    write_report_markdown(report, out_md)
    write_paper_report_sidecars(report, (out_json, out_md))
    print(json.dumps({"json_path": out_json, "markdown_path": out_md}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "MeasuredEvidence",
    "SOTA_REFERENCE_ROWS",
    "collect_measured_evidence",
    "_next_gates",
    "summarize_oracle_gap",
    "build_sota_gap_report",
    "write_report_json",
    "write_report_markdown",
    "main",
]
