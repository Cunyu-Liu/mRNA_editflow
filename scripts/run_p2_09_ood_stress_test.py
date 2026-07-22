"""P2-09: OOD Robustness Stress Test.

Stress-tests the ranker baselines from P2-03 on out-of-distribution (OOD)
subsets of the frozen combined_family test split. Compares OOD vs
in-distribution (ID) performance to quantify robustness.

OOD subsets (8 total, defined on the evaluated sources):
  - length_total_p10 / p90: bottom/top 10% by total mRNA length
  - length_5utr_p90: top 10% by 5'UTR length
  - length_3utr_p90: top 10% by 3'UTR length
  - gc_total_p10 / p90: bottom/top 10% by GC content (full sequence)
  - family_rare: sources whose gene family has <=2 members in the eval set
  - cds_long_p90: top 10% by CDS length

Reward qualifier: all delta_oracle_te values are *predicted TE (internal
proxy)* from Oracle #3. No unqualified "TE" claims.

Usage:
    python -m scripts.run_p2_09_ood_stress_test \\
        --p2-03-root benchmark/dev/leakage_free_headline_preliminary \\
        --baselines te_only scalar pareto grpo hardneg_v2 \\
        --out-dir benchmark/dev/ood_robustness_p2_09 \\
        --n-bootstrap 1000

Outputs:
    benchmark/dev/ood_robustness_p2_09/
      ├── ood_subsets.json              # subset definitions + membership
      ├── <baseline>_<subset>.json      # per-baseline per-subset stats
      ├── robustness_summary.json       # aggregate robustness scores
      └── robustness_summary.md         # human-readable table
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


# Pre-registered endpoints / qualifiers.
PRIMARY_ENDPOINT = "delta_oracle_te_vs_source"
REWARD_QUALIFIER = "predicted_te_internal_proxy"
# Bonferroni correction for 8 OOD subsets.
N_SUBSETS = 8
ALPHA_CORRECTED = 0.05 / N_SUBSETS  # = 0.00625


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_sources_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load sources.jsonl from a P2-03 baseline directory."""
    out: List[Dict[str, Any]] = []
    with path.open("r") as fh:
        for line in fh:
            if line.strip():
                out.append(json.loads(line))
    return out


def load_per_record_metrics(seed_dir: Path) -> Optional[Dict[str, List[float]]]:
    """Load per_record_metrics from a seed's eval_summary.json.

    Returns a dict mapping metric_name -> list of values (one per source),
    or None if the file is missing or malformed.
    """
    eval_path = seed_dir / "eval_summary.json"
    if not eval_path.exists():
        return None
    try:
        with eval_path.open("r") as fh:
            d = json.load(fh)
    except json.JSONDecodeError:
        return None
    prm = d.get("per_record_metrics")
    if not isinstance(prm, dict):
        return None
    return prm


def compute_delta_per_record(
    prm: Dict[str, List[float]],
) -> Optional[np.ndarray]:
    """Compute delta_oracle_te_vs_source per record.

    delta = oracle_ensemble_te[i] - source_oracle_ensemble_te[i]
    """
    te = prm.get("oracle_ensemble_te")
    src_te = prm.get("source_oracle_ensemble_te")
    if te is None or src_te is None:
        return None
    if len(te) != len(src_te):
        return None
    return np.array(te, dtype=float) - np.array(src_te, dtype=float)


# ---------------------------------------------------------------------------
# OOD subset membership
# ---------------------------------------------------------------------------


def _gc_content(seq: str) -> float:
    """GC content as a fraction in [0, 1]. Empty -> 0.0."""
    if not seq:
        return 0.0
    gc = sum(1 for c in seq.upper() if c in "GC")
    return gc / len(seq)


def _gene_id_from_transcript(transcript_id: str) -> str:
    """Extract a gene-level identifier from a transcript_id.

    Examples:
      'gencode_v45:ENST00000328596.10' -> 'ENSG00000328596' (PARSED)
      'refseq:NM_001005484.1'          -> 'NM_001005484' (strip version)

    For ENSEMBL transcripts (ENST/S), we attempt to map to the parent gene
    (ENSG/S) by replacing the leading 'T' with 'G'. This is an
    approximation; for non-ENSEMBL ids we fall back to stripping the
    version suffix. The version suffix (.N) is always stripped.
    """
    if not transcript_id:
        return ""
    tid = transcript_id.split(":")[-1].strip()
    # Strip version suffix (.N) first.
    tid = tid.split(".")[0]
    # ENSEMBL transcript -> gene approximation.
    if tid.startswith("ENST"):
        return tid.replace("ENST", "ENSG", 1)
    if tid.startswith("ENSMUST"):
        return tid.replace("ENSMUST", "ENSMUSG", 1)
    if tid.startswith("ENSDART"):
        return tid.replace("ENSDART", "ENSDARG", 1)
    return tid


@dataclass
class SourceMetadata:
    """Per-source metadata used for OOD partitioning."""

    index: int  # position in sources.jsonl
    total_length: int
    five_utr_length: int
    cds_length: int
    three_utr_length: int
    gc_total: float
    gene_id: str
    species: str
    cluster_id: int = -1  # family-cluster ID from split contract (-1 = unknown)


def extract_source_metadata(
    sources: Sequence[Mapping[str, Any]],
    cluster_assignments: Optional[Sequence[int]] = None,
    test_idx: Optional[Sequence[int]] = None,
) -> List[SourceMetadata]:
    """Extract metadata for OOD partitioning from source records.

    Args:
      sources: list of source record dicts (from sources.jsonl).
      cluster_assignments: optional list of cluster IDs for every record in
        the full dataset (e.g. from cluster_assignments.json). When provided
        together with ``test_idx``, each source's cluster_id is looked up as
        ``cluster_assignments[test_idx[i]]``.
      test_idx: optional list of test-split indices into the full dataset.
        Must be the same length as ``sources`` (i.e. the first N test
        indices correspond to the N evaluated sources when --limit N was
        used in P2-03).
    """
    use_clusters = (
        cluster_assignments is not None
        and test_idx is not None
        and len(test_idx) >= len(sources)
    )
    out: List[SourceMetadata] = []
    for i, s in enumerate(sources):
        five_utr = s.get("five_utr", "") or ""
        cds = s.get("cds", "") or ""
        three_utr = s.get("three_utr", "") or ""
        full = five_utr + cds + three_utr
        cid = -1
        if use_clusters:
            tid = int(test_idx[i])
            if 0 <= tid < len(cluster_assignments):
                cid = int(cluster_assignments[tid])
        out.append(
            SourceMetadata(
                index=i,
                total_length=len(full),
                five_utr_length=len(five_utr),
                cds_length=len(cds),
                three_utr_length=len(three_utr),
                gc_total=_gc_content(full),
                gene_id=_gene_id_from_transcript(s.get("transcript_id", "")),
                species=s.get("species", ""),
                cluster_id=cid,
            )
        )
    return out


def _top_n_indices(values: Sequence[float], n: int) -> set:
    """Return the indices of the top-n values (descending)."""
    if n <= 0 or len(values) == 0:
        return set()
    n = min(n, len(values))
    arr = np.array(values, dtype=float)
    # Get indices that would sort the array descending.
    order = np.argsort(-arr, kind="stable")
    return {int(i) for i in order[:n]}


def _bottom_n_indices(values: Sequence[float], n: int) -> set:
    """Return the indices of the bottom-n values (ascending)."""
    if n <= 0 or len(values) == 0:
        return set()
    n = min(n, len(values))
    arr = np.array(values, dtype=float)
    order = np.argsort(arr, kind="stable")
    return {int(i) for i in order[:n]}


def compute_ood_subsets(
    meta: Sequence[SourceMetadata],
) -> Dict[str, Dict[str, Any]]:
    """Compute OOD subset membership.

    Returns a mapping {subset_name: {"indices": [int], "definition": str, "size": int}}.

    For length/GC subsets, we use top-N / bottom-N (N = max(1, len(meta)//10))
    instead of percentile thresholds, to avoid tie-induced inflation when many
    sources share the same value (e.g., empty 5'UTR).
    """
    total_lens = [m.total_length for m in meta]
    five_utr_lens = [m.five_utr_length for m in meta]
    three_utr_lens = [m.three_utr_length for m in meta]
    cds_lens = [m.cds_length for m in meta]
    gcs = [m.gc_total for m in meta]
    n = len(meta)
    target = max(1, n // 10)  # ~10% of the eval set.

    # Family rarity: use cluster_id from split contract when available
    # (cluster_id >= 0); otherwise fall back to gene_id from transcript_id.
    has_clusters = any(m.cluster_id >= 0 for m in meta)
    if has_clusters:
        cluster_counts: Dict[int, int] = {}
        for m in meta:
            if m.cluster_id >= 0:
                cluster_counts[m.cluster_id] = cluster_counts.get(m.cluster_id, 0) + 1
        family_rare_idx = {
            m.index for m in meta
            if m.cluster_id >= 0 and cluster_counts.get(m.cluster_id, 0) <= 2
        }
        family_criterion = "cluster_count <= 2 (from split contract cluster_assignments)"
        family_definition = (
            "sources whose family cluster has <=2 members in the eval set "
            "(cluster_id from split contract)"
        )
    else:
        gene_counts: Dict[str, int] = {}
        for m in meta:
            gene_counts[m.gene_id] = gene_counts.get(m.gene_id, 0) + 1
        family_rare_idx = {
            m.index for m in meta if gene_counts.get(m.gene_id, 0) <= 2
        }
        family_criterion = "gene_count <= 2 (gene_id from transcript_id; approximation)"
        family_definition = (
            "sources whose gene family has <=2 members in the eval set "
            "(gene_id approximated from transcript_id; may capture most sources)"
        )

    subsets: Dict[str, Dict[str, Any]] = {
        "length_total_p10": {
            "indices": sorted(_bottom_n_indices(total_lens, target)),
            "definition": "bottom ~10% by len(5'UTR)+len(CDS)+len(3'UTR)",
            "criterion": f"bottom-{target} by total_length",
        },
        "length_total_p90": {
            "indices": sorted(_top_n_indices(total_lens, target)),
            "definition": "top ~10% by total length",
            "criterion": f"top-{target} by total_length",
        },
        "length_5utr_p90": {
            "indices": sorted(_top_n_indices(five_utr_lens, target)),
            "definition": "top ~10% by len(5'UTR)",
            "criterion": f"top-{target} by five_utr_length",
        },
        "length_3utr_p90": {
            "indices": sorted(_top_n_indices(three_utr_lens, target)),
            "definition": "top ~10% by len(3'UTR)",
            "criterion": f"top-{target} by three_utr_length",
        },
        "gc_total_p10": {
            "indices": sorted(_bottom_n_indices(gcs, target)),
            "definition": "bottom ~10% by GC content (full sequence)",
            "criterion": f"bottom-{target} by gc_total",
        },
        "gc_total_p90": {
            "indices": sorted(_top_n_indices(gcs, target)),
            "definition": "top ~10% by GC content",
            "criterion": f"top-{target} by gc_total",
        },
        "family_rare": {
            "indices": sorted(family_rare_idx),
            "definition": family_definition,
            "criterion": family_criterion,
        },
        "cds_long_p90": {
            "indices": sorted(_top_n_indices(cds_lens, target)),
            "definition": "top ~10% by len(CDS)",
            "criterion": f"top-{target} by cds_length",
        },
    }
    for name, s in subsets.items():
        s["size"] = len(s["indices"])
    return subsets


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def cohens_d(a: Sequence[float], b: Sequence[float]) -> float:
    """Cohen's d (pooled-SD) for two independent samples."""
    a_arr = np.array(a, dtype=float)
    b_arr = np.array(b, dtype=float)
    if len(a_arr) < 2 or len(b_arr) < 2:
        return 0.0
    pooled_var = (
        ((len(a_arr) - 1) * a_arr.var(ddof=1))
        + ((len(b_arr) - 1) * b_arr.var(ddof=1))
    ) / (len(a_arr) + len(b_arr) - 2)
    if pooled_var <= 0:
        return 0.0
    return float((a_arr.mean() - b_arr.mean()) / math.sqrt(pooled_var))


def paired_permutation_test(
    ood_values: Sequence[float],
    id_values: Sequence[float],
    n_perm: int = 1000,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, float]:
    """Two-sided paired permutation test on the mean difference.

    Tests H0: mean(ood - id) == 0 by randomly swapping labels within pairs.
    Returns {"diff": float, "p_value": float, "n_perm": int}.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    ood = np.array(ood_values, dtype=float)
    idd = np.array(id_values, dtype=float)
    n = min(len(ood), len(idd))
    if n < 2:
        return {"diff": 0.0, "p_value": 1.0, "n_perm": n_perm}
    ood = ood[:n]
    idd = idd[:n]
    diffs = ood - idd
    obs = float(diffs.mean())
    # Permutation: randomly flip sign of each diff.
    count = 0
    for _ in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=n)
        perm_mean = float((diffs * signs).mean())
        if abs(perm_mean) >= abs(obs):
            count += 1
    p_value = (count + 1) / (n_perm + 1)
    return {"diff": obs, "p_value": p_value, "n_perm": n_perm}


def bootstrap_ci(
    values: Sequence[float],
    n_boot: int = 1000,
    rng: Optional[np.random.Generator] = None,
    level: float = 0.95,
) -> Tuple[float, float, float]:
    """Bootstrap CI for the mean. Returns (mean, ci_low, ci_high)."""
    if rng is None:
        rng = np.random.default_rng(42)
    arr = np.array(values, dtype=float)
    if len(arr) == 0:
        return 0.0, 0.0, 0.0
    mean = float(arr.mean())
    if len(arr) < 2:
        return mean, mean, mean
    boot_means = np.array(
        [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_boot)]
    )
    alpha = (1 - level) / 2
    ci_low = float(np.percentile(boot_means, 100 * alpha))
    ci_high = float(np.percentile(boot_means, 100 * (1 - alpha)))
    return mean, ci_low, ci_high


# ---------------------------------------------------------------------------
# Per-baseline per-subset analysis
# ---------------------------------------------------------------------------


@dataclass
class SubsetResult:
    """OOD analysis result for one baseline x subset pair."""

    baseline: str
    subset: str
    n_ood: int
    n_id: int
    ood_mean: float
    ood_ci_low: float
    ood_ci_high: float
    id_mean: float
    id_ci_low: float
    id_ci_high: float
    diff_ood_minus_id: float
    cohens_d: float
    p_value: float
    n_perm: int
    significant_after_bonferroni: bool
    qualifier: str = REWARD_QUALIFIER

    def to_dict(self) -> Dict[str, Any]:
        return {
            "baseline": self.baseline,
            "subset": self.subset,
            "n_ood": self.n_ood,
            "n_id": self.n_id,
            "ood_mean": self.ood_mean,
            "ood_ci_low": self.ood_ci_low,
            "ood_ci_high": self.ood_ci_high,
            "id_mean": self.id_mean,
            "id_ci_low": self.id_ci_low,
            "id_ci_high": self.id_ci_high,
            "diff_ood_minus_id": self.diff_ood_minus_id,
            "cohens_d": self.cohens_d,
            "p_value": self.p_value,
            "n_perm": self.n_perm,
            "significant_after_bonferroni": self.significant_after_bonferroni,
            "qualifier": self.qualifier,
            "alpha_corrected": ALPHA_CORRECTED,
        }


def analyze_baseline_subset(
    baseline: str,
    subset_name: str,
    subset_indices: set,
    seed_deltas: List[np.ndarray],
    n_bootstrap: int = 1000,
    n_perm: int = 1000,
    rng: Optional[np.random.Generator] = None,
) -> Optional[SubsetResult]:
    """Analyze one baseline x subset pair.

    Args:
      baseline: name of the baseline.
      subset_name: name of the OOD subset.
      subset_indices: set of source indices belonging to the OOD subset.
      seed_deltas: list of per-seed delta arrays (each shape [n_sources]).
      n_bootstrap: bootstrap iterations for CI.
      n_perm: permutation test iterations.
      rng: RNG for reproducibility.

    Returns:
      SubsetResult, or None if no valid data.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    if not seed_deltas:
        return None
    n_sources = seed_deltas[0].shape[0]
    # Sanity: all seeds must have the same length.
    for sd in seed_deltas:
        if sd.shape[0] != n_sources:
            return None

    ood_idx = sorted(i for i in subset_indices if 0 <= i < n_sources)
    id_idx = sorted(i for i in range(n_sources) if i not in subset_indices)
    if not ood_idx or not id_idx:
        return None

    # Pool delta values across seeds, per partition.
    ood_vals: List[float] = []
    id_vals: List[float] = []
    for sd in seed_deltas:
        ood_vals.extend(float(sd[i]) for i in ood_idx)
        id_vals.extend(float(sd[i]) for i in id_idx)

    ood_mean, ood_lo, ood_hi = bootstrap_ci(ood_vals, n_boot=n_bootstrap, rng=rng)
    id_mean, id_lo, id_hi = bootstrap_ci(id_vals, n_boot=n_bootstrap, rng=rng)
    d = cohens_d(ood_vals, id_vals)
    perm = paired_permutation_test(ood_vals, id_vals, n_perm=n_perm, rng=rng)

    return SubsetResult(
        baseline=baseline,
        subset=subset_name,
        n_ood=len(ood_idx),
        n_id=len(id_idx),
        ood_mean=ood_mean,
        ood_ci_low=ood_lo,
        ood_ci_high=ood_hi,
        id_mean=id_mean,
        id_ci_low=id_lo,
        id_ci_high=id_hi,
        diff_ood_minus_id=perm["diff"],
        cohens_d=d,
        p_value=perm["p_value"],
        n_perm=perm["n_perm"],
        significant_after_bonferroni=perm["p_value"] < ALPHA_CORRECTED,
    )


# ---------------------------------------------------------------------------
# Baseline-level aggregation
# ---------------------------------------------------------------------------


def load_baseline_seed_deltas(
    baseline_dir: Path,
) -> Tuple[List[np.ndarray], List[int]]:
    """Load per-seed delta arrays for a baseline.

    Returns (seed_deltas, seed_ids). seed_deltas[i] is an array of
    delta_oracle_te_vs_source values, one per source.
    """
    seed_dirs = sorted(
        [d for d in baseline_dir.iterdir() if d.is_dir() and d.name.startswith("seed_")]
    )
    seed_deltas: List[np.ndarray] = []
    seed_ids: List[int] = []
    for sd in seed_dirs:
        try:
            sid = int(sd.name.replace("seed_", ""))
        except ValueError:
            continue
        prm = load_per_record_metrics(sd)
        if prm is None:
            continue
        delta = compute_delta_per_record(prm)
        if delta is None:
            continue
        seed_deltas.append(delta)
        seed_ids.append(sid)
    return seed_deltas, seed_ids


def robustness_score(
    results: Sequence[SubsetResult],
) -> float:
    """Aggregate robustness score: mean(diff_ood_minus_id) across subsets.

    A score near 0 means robust; a large negative score means OOD
    degradation.
    """
    if not results:
        return 0.0
    return float(np.mean([r.diff_ood_minus_id for r in results]))


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def write_markdown_summary(
    subsets: Dict[str, Dict[str, Any]],
    baseline_results: Dict[str, List[SubsetResult]],
    out_path: Path,
) -> None:
    """Write a human-readable markdown summary."""
    lines: List[str] = []
    lines.append("# P2-09: OOD Robustness Stress Test")
    lines.append("")
    lines.append(
        f"**Reward qualifier**: `{REWARD_QUALIFIER}` — all "
        "`delta_oracle_te` values are predicted TE (internal proxy) from "
        "Oracle #3."
    )
    lines.append(
        f"**Significance**: paired permutation test, Bonferroni-corrected "
        f"α = {ALPHA_CORRECTED:.5f} ({N_SUBSETS} subsets)."
    )
    lines.append("")
    lines.append("## OOD subset sizes")
    lines.append("")
    lines.append("| Subset | Definition | Size |")
    lines.append("|--------|-----------|-----:|")
    for name, s in subsets.items():
        lines.append(f"| `{name}` | {s['definition']} | {s['size']} |")
    lines.append("")
    lines.append("## Per-baseline robustness scores")
    lines.append("")
    lines.append(
        "Robustness score = mean(diff_ood_minus_id) across 8 subsets. "
        "Near 0 = robust; large negative = OOD degradation."
    )
    lines.append("")
    lines.append("| Baseline | Robustness score | # significant (Bonferroni) |")
    lines.append("|----------|-----------------:|---------------------------:|")
    for baseline, results in baseline_results.items():
        score = robustness_score(results)
        n_sig = sum(1 for r in results if r.significant_after_bonferroni)
        lines.append(f"| `{baseline}` | {score:.6f} | {n_sig}/{len(results)} |")
    lines.append("")
    lines.append("## Per-baseline per-subset details")
    lines.append("")
    for baseline, results in baseline_results.items():
        lines.append(f"### `{baseline}`")
        lines.append("")
        lines.append(
            "| Subset | n_ood | n_id | OOD mean | OOD 95% CI | "
            "ID mean | ID 95% CI | Δ (OOD−ID) | Cohen's d | p-value | "
            "Sig. (Bonf.) |"
        )
        lines.append(
            "|--------|------:|-----:|----------:|------------|--------:|"
            "-----------|----------:|----------:|---------:|:------------:|"
        )
        for r in results:
            sig = "✓" if r.significant_after_bonferroni else ""
            lines.append(
                f"| `{r.subset}` | {r.n_ood} | {r.n_id} | "
                f"{r.ood_mean:.6f} | [{r.ood_ci_low:.6f}, {r.ood_ci_high:.6f}] | "
                f"{r.id_mean:.6f} | [{r.id_ci_low:.6f}, {r.id_ci_high:.6f}] | "
                f"{r.diff_ood_minus_id:.6f} | {r.cohens_d:.4f} | "
                f"{r.p_value:.4f} | {sig} |"
            )
        lines.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_p2_09(
    p2_03_root: Path,
    baselines: Sequence[str],
    out_dir: Path,
    n_bootstrap: int = 1000,
    n_perm: int = 1000,
    seed: int = 42,
    cluster_assignments_path: Optional[Path] = None,
    test_idx_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run the full P2-09 OOD stress test.

    Args:
      cluster_assignments_path: optional path to cluster_assignments.json
        (list of cluster IDs for every record in the full dataset). When
        provided together with ``test_idx_path``, the ``family_rare`` subset
        uses proper family-cluster IDs from the split contract.
      test_idx_path: optional path to test.idx (list of test-split indices
        into the full dataset).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    # Load sources from the first available baseline (they should be identical).
    sources: Optional[List[Dict[str, Any]]] = None
    for b in baselines:
        p = p2_03_root / f"{b}_seed0" / "sources.jsonl"
        if p.exists():
            sources = load_sources_jsonl(p)
            break
    if not sources:
        raise FileNotFoundError(
            f"sources.jsonl not found under {p2_03_root} for any of {baselines}"
        )

    # Optionally load cluster assignments + test idx for proper family_rare.
    cluster_assignments: Optional[List[int]] = None
    test_idx: Optional[List[int]] = None
    if cluster_assignments_path is not None and cluster_assignments_path.exists():
        with cluster_assignments_path.open("r") as fh:
            cluster_assignments = json.load(fh)
        print(
            f"[info] loaded {len(cluster_assignments)} cluster assignments from "
            f"{cluster_assignments_path}",
            file=sys.stderr,
        )
    if test_idx_path is not None and test_idx_path.exists():
        with test_idx_path.open("r") as fh:
            test_idx = [int(line.strip()) for line in fh if line.strip()]
        print(
            f"[info] loaded {len(test_idx)} test indices from {test_idx_path}",
            file=sys.stderr,
        )

    meta = extract_source_metadata(
        sources,
        cluster_assignments=cluster_assignments,
        test_idx=test_idx,
    )
    subsets = compute_ood_subsets(meta)

    # Save subset definitions.
    subsets_out = {
        name: {
            "definition": s["definition"],
            "criterion": s["criterion"],
            "size": s["size"],
            "indices": s["indices"],
        }
        for name, s in subsets.items()
    }
    (out_dir / "ood_subsets.json").write_text(json.dumps(subsets_out, indent=2))

    baseline_results: Dict[str, List[SubsetResult]] = {}
    for b in baselines:
        bdir = p2_03_root / f"{b}_seed0"
        if not bdir.exists():
            print(f"[warn] baseline dir not found: {bdir}", file=sys.stderr)
            continue
        seed_deltas, seed_ids = load_baseline_seed_deltas(bdir)
        if not seed_deltas:
            print(f"[warn] no seed deltas for baseline {b}", file=sys.stderr)
            continue
        results: List[SubsetResult] = []
        for subset_name, sinfo in subsets.items():
            r = analyze_baseline_subset(
                baseline=b,
                subset_name=subset_name,
                subset_indices=set(sinfo["indices"]),
                seed_deltas=seed_deltas,
                n_bootstrap=n_bootstrap,
                n_perm=n_perm,
                rng=rng,
            )
            if r is not None:
                results.append(r)
                # Save per-baseline per-subset JSON.
                out_path = out_dir / f"{b}_{subset_name}.json"
                out_path.write_text(json.dumps(r.to_dict(), indent=2))
        baseline_results[b] = results

    # Aggregate.
    robustness: Dict[str, float] = {
        b: robustness_score(rs) for b, rs in baseline_results.items()
    }
    summary = {
        "reward_qualifier": REWARD_QUALIFIER,
        "primary_endpoint": PRIMARY_ENDPOINT,
        "alpha_corrected": ALPHA_CORRECTED,
        "n_subsets": N_SUBSETS,
        "n_baselines": len(baseline_results),
        "n_sources": len(sources),
        "subset_sizes": {name: s["size"] for name, s in subsets.items()},
        "robustness_scores": robustness,
        "baselines": {
            b: [r.to_dict() for r in rs] for b, rs in baseline_results.items()
        },
    }
    (out_dir / "robustness_summary.json").write_text(json.dumps(summary, indent=2))
    write_markdown_summary(subsets, baseline_results, out_dir / "robustness_summary.md")
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="P2-09: OOD Robustness Stress Test")
    parser.add_argument(
        "--p2-03-root",
        type=Path,
        required=True,
        help="Root dir of P2-03 results (e.g. benchmark/dev/leakage_free_headline_preliminary)",
    )
    parser.add_argument(
        "--baselines",
        nargs="+",
        default=["te_only", "scalar", "pareto", "grpo", "hardneg_v2"],
        help="Baselines to analyze",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory",
    )
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--n-perm", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--cluster-assignments",
        type=Path,
        default=None,
        help=(
            "Path to cluster_assignments.json (list of cluster IDs for every "
            "record in the full dataset). When provided together with --test-idx, "
            "the family_rare subset uses proper family-cluster IDs from the split "
            "contract instead of the gene_id approximation."
        ),
    )
    parser.add_argument(
        "--test-idx",
        type=Path,
        default=None,
        help=(
            "Path to test.idx (list of test-split indices into the full dataset). "
            "Required when --cluster-assignments is provided."
        ),
    )
    args = parser.parse_args(argv)
    if args.cluster_assignments is not None and args.test_idx is None:
        parser.error("--test-idx is required when --cluster-assignments is provided")
    summary = run_p2_09(
        p2_03_root=args.p2_03_root,
        baselines=args.baselines,
        out_dir=args.out_dir,
        n_bootstrap=args.n_bootstrap,
        n_perm=args.n_perm,
        seed=args.seed,
        cluster_assignments_path=args.cluster_assignments,
        test_idx_path=args.test_idx,
    )
    print(
        f"[ok] P2-09 done: {summary['n_baselines']} baselines, "
        f"{summary['n_sources']} sources, {summary['n_subsets']} subsets"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
