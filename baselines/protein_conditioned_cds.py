"""Protein-conditioned CDS design for mRNA-EditFlow.

Roadmap architecture upgrade #3. The existing T4 pipeline *edits an existing CDS*
under a synonymous-codon lattice. This module upgrades T4 to *design a CDS from a
protein sequence*: given a target amino-acid string, it emits a valid in-frame
coding sequence whose translation exactly equals the target protein, then
co-optimizes codon usage (CAI) and GC toward a target using the verified
synonymous-codon dynamic program in :mod:`mrna_editflow.baselines.codon_lattice_dp`.

Pipeline
--------
1. **Seed CDS** ``x0`` from protein ``p = a_1 ... a_n``:

   ``x0 = AUG + c(a_1) + c(a_2) + ... + c(a_n) + stop``,

   where ``c(a)`` is a deterministic synonymous codon for amino acid ``a`` (the
   lexicographically-first codon, for reproducibility) and ``stop`` is the
   first stop codon. If ``p`` already starts with Met (``M``) the leading AUG is
   that residue; otherwise an initiator AUG (Met) is prepended and reported.

2. **Optimize** ``x*`` over the synonymous lattice with
   :func:`optimize_cds_synonymous`, which provably preserves
   ``translate(x*) == translate(x0)`` (protein identity is a hard constraint,
   never traded off) while maximizing

   ``sum_i [ alpha log CAI(c_i) - beta (GC(c_i)-g*)^2 - gamma 1[c_i != x0_i] ]``
   ``+ boundary term``.

Because step 2 reuses the audited DP, protein identity, in-frame validity and a
single terminal stop are guaranteed by construction; this module only adds the
protein->seed construction and reporting.

Complexity is ``O(n)`` for seeding plus the DP cost ``O(n * K * C^2)``.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass, replace
from typing import Mapping, Optional, Sequence

from mrna_editflow.baselines.codon_lattice_dp import (
    CodonLatticeDPConfig,
    optimize_cds_synonymous,
)
from mrna_editflow.core.constants import (
    CODON_TABLE,
    START_CODON,
    STOP_CODONS,
    SYNONYMOUS_CODONS,
    translate,
)
from mrna_editflow.core.schema import MRNARecord
from mrna_editflow.data.download_mrna import load_records_jsonl
from mrna_editflow.eval.metrics import cai, codon_weights_from_reference, gc_fraction


# Amino-acid letter -> deterministic (lexicographically first) synonymous codon.
# Stop ("*") is excluded here; the terminal stop is added explicitly.
_AA_DEFAULT_CODON: dict[str, str] = {
    aa: sorted(codons)[0]
    for aa, codons in SYNONYMOUS_CODONS.items()
    if aa != "*"
}
_VALID_AA: frozenset[str] = frozenset(_AA_DEFAULT_CODON)
_DEFAULT_STOP: str = sorted(STOP_CODONS)[0]


@dataclass(frozen=True)
class ProteinConditionedCDSResult:
    """Result of designing a CDS for one target protein."""

    protein: str
    prepended_start_met: bool
    seed_cds: str
    designed_cds: str
    designed_protein: str
    protein_identity: float
    codon_changes: int
    seed_cai: float
    designed_cai: float
    seed_gc: float
    designed_gc: float
    n_codons: int
    # Optional native-CDS baseline (populated when designing from real records):
    # the natural human CDS that encodes this protein, so the design can be
    # compared against the biological reference (the CDS-optimizer SOTA question)
    # rather than only the naive lexicographic seed. ``native_protein_identity``
    # verifies the designed CDS still encodes the same protein as the native one.
    native_cds: Optional[str] = None
    native_cai: Optional[float] = None
    native_gc: Optional[float] = None
    native_protein_identity: Optional[float] = None

    def to_dict(self) -> dict[str, object]:
        return dict(asdict(self))


def _normalise_protein(protein: str) -> str:
    """Upper-case, strip whitespace, drop a trailing stop ``*``.

    Complexity is ``O(len(protein))``.
    """
    seq = "".join(str(protein or "").upper().split())
    if seq.endswith("*"):
        seq = seq[:-1]
    return seq


def seed_cds_from_protein(protein: str) -> tuple[str, bool]:
    """Build a deterministic valid seed CDS for a target protein.

    Returns ``(seed_cds, prepended_start_met)``. The seed always begins with an
    AUG start codon and ends with a stop codon, so it is a valid in-frame CDS.
    If the protein does not already start with Met, an initiator Met is
    prepended (and flagged) so the design remains translatable from AUG.

    Raises ``ValueError`` on empty or non-standard amino acids. Complexity is
    ``O(len(protein))``.
    """
    seq = _normalise_protein(protein)
    if not seq:
        raise ValueError("protein sequence is empty")
    bad = sorted(set(seq) - _VALID_AA)
    if bad:
        raise ValueError(f"protein contains non-standard amino acids: {bad}")

    prepended = seq[0] != "M"
    residues = ("M" + seq) if prepended else seq
    codons = [START_CODON]  # AUG encodes the leading Met.
    for aa in residues[1:]:
        codons.append(_AA_DEFAULT_CODON[aa])
    codons.append(_DEFAULT_STOP)
    return "".join(codons), prepended


def design_cds_for_protein(
    protein: str,
    *,
    config: Optional[CodonLatticeDPConfig] = None,
    codon_weights: Optional[Mapping[str, float]] = None,
    native_cds: Optional[str] = None,
) -> ProteinConditionedCDSResult:
    """Design a CAI/GC-optimized CDS whose translation equals ``protein``.

    Protein identity is enforced by the synonymous-lattice DP and independently
    verified here (``designed_protein`` vs the target). When ``native_cds`` is
    given (the real transcript CDS this protein came from), its CAI/GC are
    recorded so the design can be compared against the biological reference, not
    only the naive seed. Complexity is the DP cost ``O(n * K * C^2)``.
    """
    cfg = config or CodonLatticeDPConfig()
    seed, prepended = seed_cds_from_protein(protein)
    result = optimize_cds_synonymous(seed, config=cfg, codon_weights=codon_weights)

    designed = result.optimized_cds
    designed_protein = translate(designed)
    # Target protein as encoded by the seed (drops the terminal stop marker).
    target_protein = translate(seed)
    identity = _protein_identity(target_protein, designed_protein)
    weights = dict(codon_weights) if codon_weights else None

    native_cai = native_gc = native_identity = None
    if native_cds:
        native_cai = float(cai(native_cds, weights))
        native_gc = float(gc_fraction(native_cds))
        # Confirm the designed CDS encodes the same protein as the native CDS.
        native_identity = float(
            _protein_identity(translate(native_cds), designed_protein)
        )
    return ProteinConditionedCDSResult(
        protein=_normalise_protein(protein),
        prepended_start_met=bool(prepended),
        seed_cds=seed,
        designed_cds=designed,
        designed_protein=designed_protein.rstrip("*"),
        protein_identity=float(identity),
        codon_changes=int(result.codon_changes),
        seed_cai=float(cai(seed, weights)),
        designed_cai=float(cai(designed, weights)),
        seed_gc=float(gc_fraction(seed)),
        designed_gc=float(gc_fraction(designed)),
        n_codons=int(len(seed) // 3),
        native_cds=native_cds,
        native_cai=native_cai,
        native_gc=native_gc,
        native_protein_identity=native_identity,
    )


def _protein_identity(a: str, b: str) -> float:
    """Fraction of matching residues over the longer length (stop-insensitive).

    Complexity is ``O(len)``.
    """
    a = a.rstrip("*")
    b = b.rstrip("*")
    if not a and not b:
        return 1.0
    n = max(len(a), len(b))
    if n == 0:
        return 1.0
    matches = sum(1 for x, y in zip(a, b) if x == y)
    # Length mismatch counts as non-identity for the trailing residues.
    return float(matches / n)


def design_records_for_proteins(
    proteins: Sequence[str],
    *,
    config: Optional[CodonLatticeDPConfig] = None,
    codon_weights: Optional[Mapping[str, float]] = None,
    native_cds: Optional[Sequence[Optional[str]]] = None,
) -> list[ProteinConditionedCDSResult]:
    """Design CDS for a batch of proteins. Complexity is ``O(sum DP costs)``.

    ``native_cds`` optionally supplies, per protein, the real transcript CDS so
    each design is scored against its biological reference.
    """
    cfg = config or CodonLatticeDPConfig()
    natives: Sequence[Optional[str]]
    if native_cds is None:
        natives = [None] * len(proteins)
    else:
        natives = list(native_cds)
    return [
        design_cds_for_protein(p, config=cfg, codon_weights=codon_weights, native_cds=nat)
        for p, nat in zip(proteins, natives)
    ]


def _targets_from_inputs(
    *,
    proteins: Optional[Sequence[str]] = None,
    records_jsonl: Optional[str] = None,
    limit: Optional[int] = None,
    use_native_baseline: bool = False,
) -> tuple[list[str], Optional[list[Optional[str]]]]:
    """Load protein targets and optional native CDS references.

    Complexity is ``O(N + total CDS length)`` in records mode.
    """
    targets: list[str]
    natives: Optional[list[Optional[str]]] = None
    if proteins is not None:
        targets = list(proteins)
    elif records_jsonl is not None:
        records = load_records_jsonl(records_jsonl)
        targets = [translate(r.cds).rstrip("*") for r in records]
        if use_native_baseline:
            natives = [r.cds for r in records]
    else:
        raise ValueError("provide either proteins or records_jsonl")
    if limit is not None:
        targets = targets[: int(limit)]
        if natives is not None:
            natives = natives[: int(limit)]
    return targets, natives


def _append_progress_jsonl(path: Optional[str], event: str, **payload: object) -> None:
    """Append one JSONL progress event when ``path`` is configured."""
    if not path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    row = {"time": time.time(), "event": event}
    row.update(payload)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")


def summarize_designs(results: Sequence[ProteinConditionedCDSResult]) -> dict[str, object]:
    """Summarize a batch of protein-conditioned designs.

    Complexity is ``O(len(results))``.
    """
    if not results:
        return {"n": 0, "mean_protein_identity": 1.0}
    n = len(results)
    summary = {
        "n": n,
        "mean_protein_identity": float(sum(r.protein_identity for r in results) / n),
        "min_protein_identity": float(min(r.protein_identity for r in results)),
        "protein_identity_eq_1_fraction": float(
            sum(1 for r in results if r.protein_identity >= 0.999) / n
        ),
        "mean_seed_cai": float(sum(r.seed_cai for r in results) / n),
        "mean_designed_cai": float(sum(r.designed_cai for r in results) / n),
        "mean_cai_delta": float(sum(r.designed_cai - r.seed_cai for r in results) / n),
        "mean_seed_gc": float(sum(r.seed_gc for r in results) / n),
        "mean_designed_gc": float(sum(r.designed_gc for r in results) / n),
        "mean_codon_changes": float(sum(r.codon_changes for r in results) / n),
        "n_prepended_start_met": int(sum(1 for r in results if r.prepended_start_met)),
    }
    # Native-CDS baseline comparison (only over designs that carry a native CDS):
    # the biologically meaningful CDS-optimizer question is whether the design
    # matches or beats the real human transcript CDS at protein identity 1.0.
    native = [r for r in results if r.native_cai is not None]
    if native:
        m = len(native)
        summary.update(
            {
                "n_with_native": m,
                "mean_native_cai": float(sum(r.native_cai for r in native) / m),
                "mean_designed_vs_native_cai_delta": float(
                    sum(r.designed_cai - r.native_cai for r in native) / m
                ),
                "designed_cai_ge_native_fraction": float(
                    sum(1 for r in native if r.designed_cai + 1e-9 >= r.native_cai) / m
                ),
                "mean_native_gc": float(sum(r.native_gc for r in native) / m),
                "mean_designed_vs_native_gc_delta": float(
                    sum(r.designed_gc - r.native_gc for r in native) / m
                ),
                "native_protein_identity_eq_1_fraction": float(
                    sum(
                        1
                        for r in native
                        if r.native_protein_identity is not None
                        and r.native_protein_identity >= 0.999
                    )
                    / m
                ),
            }
        )
    return summary


def _dominates_cai_gc(a: Mapping[str, float], b: Mapping[str, float]) -> bool:
    """Return whether sweep point ``a`` dominates ``b``.

    The CAI-GC frontier maximizes mean CAI and minimizes mean absolute GC error
    from the target GC. Complexity is ``O(1)``.
    """
    cai_a = float(a["mean_designed_cai"])
    cai_b = float(b["mean_designed_cai"])
    err_a = float(a["mean_abs_gc_error"])
    err_b = float(b["mean_abs_gc_error"])
    return (cai_a >= cai_b and err_a <= err_b) and (cai_a > cai_b or err_a < err_b)


def _cai_gc_pareto_ranks(points: Sequence[Mapping[str, float]]) -> list[int]:
    """Assign non-dominated-sort ranks for CAI-GC sweep points.

    Rank 0 is the Pareto frontier. Complexity is ``O(M^3)`` in the number of
    sweep weights, which is intentionally tiny (usually <20).
    """
    remaining = set(range(len(points)))
    ranks = [0 for _ in points]
    rank = 0
    while remaining:
        front = []
        for i in sorted(remaining):
            if not any(_dominates_cai_gc(points[j], points[i]) for j in remaining if j != i):
                front.append(i)
        for i in front:
            ranks[i] = rank
        remaining.difference_update(front)
        rank += 1
    return ranks


def _write_gc_sweep_markdown(payload: Mapping[str, object], out_md: str) -> None:
    """Write a compact human-readable CAI-GC frontier table."""
    os.makedirs(os.path.dirname(os.path.abspath(out_md)), exist_ok=True)
    points = list(payload.get("points", []))
    lines = [
        "# Protein-conditioned CDS CAI-GC Pareto Sweep",
        "",
        f"- Targets: {payload.get('n_targets')}",
        f"- Target GC: {payload.get('target_gc')}",
        f"- Pareto front weights: {payload.get('pareto_front_gc_weights')}",
        "",
        "| gc_weight | mean CAI | mean GC | mean abs GC error | codon changes | identity=1 frac | rank |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for point in points:
        if not isinstance(point, Mapping):
            continue
        summary = point.get("summary", {})
        if not isinstance(summary, Mapping):
            summary = {}
        lines.append(
            "| {gc_weight:.6g} | {cai:.5f} | {gc:.5f} | {err:.5f} | {changes:.2f} | {identity:.3f} | {rank} |".format(
                gc_weight=float(point.get("gc_weight", 0.0)),
                cai=float(summary.get("mean_designed_cai", 0.0)),
                gc=float(summary.get("mean_designed_gc", 0.0)),
                err=float(summary.get("mean_abs_gc_error", 0.0)),
                changes=float(summary.get("mean_codon_changes", 0.0)),
                identity=float(summary.get("protein_identity_eq_1_fraction", 0.0)),
                rank=int(point.get("pareto_rank", 0)),
            )
        )
    with open(out_md, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def run_protein_conditioned_gc_sweep(
    *,
    gc_weights: Sequence[float],
    proteins: Optional[Sequence[str]] = None,
    records_jsonl: Optional[str] = None,
    out_jsonl: str,
    out_json: str,
    out_md: Optional[str] = None,
    limit: Optional[int] = None,
    config: Optional[CodonLatticeDPConfig] = None,
    reference_records_jsonl: Optional[str] = None,
    use_native_baseline: bool = False,
) -> dict[str, object]:
    """Run a CAI-GC Pareto sweep over ``gc_weight``.

    Each weight designs the same protein set with the same reference codon
    weights. The sweep frontier uses two objectives: maximize mean designed CAI
    and minimize mean absolute GC error to ``target_gc``. Protein identity must
    remain 1.0 for every point. Complexity is ``O(M * N * DP)``.
    """
    weights = [float(w) for w in gc_weights]
    if not weights:
        raise ValueError("gc_weights must contain at least one value")
    cfg = config or CodonLatticeDPConfig()
    codon_weights: Optional[Mapping[str, float]] = None
    if reference_records_jsonl:
        codon_weights = codon_weights_from_reference(load_records_jsonl(reference_records_jsonl))
    targets, natives = _targets_from_inputs(
        proteins=proteins,
        records_jsonl=records_jsonl,
        limit=limit,
        use_native_baseline=use_native_baseline,
    )

    points: list[dict[str, object]] = []
    point_objectives: list[dict[str, float]] = []
    rows: list[dict[str, object]] = []
    for gc_weight in weights:
        sweep_cfg = replace(cfg, gc_weight=float(gc_weight))
        results = design_records_for_proteins(
            targets,
            config=sweep_cfg,
            codon_weights=codon_weights,
            native_cds=natives,
        )
        summary = summarize_designs(results)
        if results:
            mean_abs_gc_error = float(
                sum(abs(r.designed_gc - sweep_cfg.target_gc) for r in results) / len(results)
            )
        else:
            mean_abs_gc_error = 0.0
        summary.update(
            {
                "gc_weight": float(gc_weight),
                "target_gc": float(sweep_cfg.target_gc),
                "mean_abs_gc_error": mean_abs_gc_error,
                "mean_gc_target_error": float(
                    summary.get("mean_designed_gc", 0.0)
                )
                - float(sweep_cfg.target_gc),
            }
        )
        objective = {
            "mean_designed_cai": float(summary.get("mean_designed_cai", 0.0)),
            "mean_abs_gc_error": float(summary.get("mean_abs_gc_error", 0.0)),
        }
        point_objectives.append(objective)
        point = {
            "gc_weight": float(gc_weight),
            "target_gc": float(sweep_cfg.target_gc),
            "config": asdict(sweep_cfg),
            "objectives": {
                "maximize_mean_designed_cai": objective["mean_designed_cai"],
                "minimize_mean_abs_gc_error": objective["mean_abs_gc_error"],
            },
            "summary": summary,
        }
        points.append(point)
        for result in results:
            row = result.to_dict()
            row["gc_weight"] = float(gc_weight)
            row["target_gc"] = float(sweep_cfg.target_gc)
            rows.append(row)

    ranks = _cai_gc_pareto_ranks(point_objectives)
    for point, rank in zip(points, ranks):
        point["pareto_rank"] = int(rank)
        point["is_pareto_front"] = bool(rank == 0)
    pareto_front = [point for point in points if point["is_pareto_front"]]
    pareto_front = sorted(
        pareto_front,
        key=lambda p: (
            float(p["summary"]["mean_abs_gc_error"]),  # type: ignore[index]
            -float(p["summary"]["mean_designed_cai"]),  # type: ignore[index]
        ),
    )

    os.makedirs(os.path.dirname(os.path.abspath(out_jsonl)), exist_ok=True)
    with open(out_jsonl, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    payload = {
        "sweep_kind": "protein_conditioned_cai_gc_pareto",
        "config_template": asdict(cfg),
        "gc_weights": weights,
        "target_gc": float(cfg.target_gc),
        "n_targets": len(targets),
        "uses_reference_codon_weights": bool(reference_records_jsonl),
        "uses_native_baseline": bool(natives is not None),
        "out_jsonl": out_jsonl,
        "out_md": out_md,
        "points": points,
        "pareto_front": pareto_front,
        "pareto_front_gc_weights": [float(p["gc_weight"]) for p in pareto_front],
        "artifact_contract": {
            "frontier_objectives": [
                "maximize mean_designed_cai",
                "minimize mean_abs_gc_error",
            ],
            "hard_constraint": "protein_identity_eq_1_fraction must remain 1.0",
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    if out_md:
        _write_gc_sweep_markdown(payload, out_md)
    return payload


def run_protein_conditioned_design(
    *,
    proteins: Optional[Sequence[str]] = None,
    records_jsonl: Optional[str] = None,
    out_jsonl: str,
    out_json: str,
    limit: Optional[int] = None,
    config: Optional[CodonLatticeDPConfig] = None,
    reference_records_jsonl: Optional[str] = None,
    use_native_baseline: bool = False,
    progress_jsonl: Optional[str] = None,
    progress_every: int = 25,
) -> dict[str, object]:
    """Design CDS for proteins (given directly or derived from record CDS).

    When ``proteins`` is omitted, target proteins are taken from
    ``records_jsonl`` by translating each record's CDS -- this makes the module a
    drop-in T4 "design from protein" benchmark on the same public split used for
    the editing pipeline. ``reference_records_jsonl`` optionally defines the CAI
    codon weights. When ``use_native_baseline`` and targets come from records,
    each design is also scored against its native transcript CDS (the CDS-only
    SOTA reference). Complexity is ``O(N * DP)``.
    """
    cfg = config or CodonLatticeDPConfig()
    codon_weights: Optional[Mapping[str, float]] = None
    if reference_records_jsonl:
        codon_weights = codon_weights_from_reference(load_records_jsonl(reference_records_jsonl))

    targets, natives = _targets_from_inputs(
        proteins=proteins,
        records_jsonl=records_jsonl,
        limit=limit,
        use_native_baseline=use_native_baseline,
    )

    native_rows: Sequence[Optional[str]]
    if natives is None:
        native_rows = [None] * len(targets)
    else:
        native_rows = natives
    progress_every = max(1, int(progress_every))
    _append_progress_jsonl(
        progress_jsonl,
        "protein_design_start",
        n_targets=len(targets),
        use_native_baseline=bool(natives is not None),
    )
    results: list[ProteinConditionedCDSResult] = []
    for index, (target, native) in enumerate(zip(targets, native_rows), start=1):
        result = design_cds_for_protein(
            target,
            config=cfg,
            codon_weights=codon_weights,
            native_cds=native,
        )
        results.append(result)
        if index == 1 or index == len(targets) or index % progress_every == 0:
            _append_progress_jsonl(
                progress_jsonl,
                "protein_design_progress",
                completed=index,
                total=len(targets),
                current_n_codons=result.n_codons,
                current_codon_changes=result.codon_changes,
                current_protein_identity=result.protein_identity,
            )
    _append_progress_jsonl(
        progress_jsonl,
        "protein_design_complete",
        completed=len(results),
        total=len(targets),
    )

    os.makedirs(os.path.dirname(os.path.abspath(out_jsonl)), exist_ok=True)
    with open(out_jsonl, "w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r.to_dict(), sort_keys=True) + "\n")
    payload = {
        "config": asdict(cfg),
        "n_targets": len(targets),
        "uses_reference_codon_weights": bool(reference_records_jsonl),
        "uses_native_baseline": bool(natives is not None),
        "out_jsonl": out_jsonl,
        "progress_jsonl": progress_jsonl,
        "summary": summarize_designs(results),
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    return payload


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-jsonl", default=None, help="derive target proteins from record CDS")
    parser.add_argument("--proteins", nargs="*", default=None, help="explicit target protein strings")
    parser.add_argument("--out-jsonl", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", default=None, help="optional markdown summary for gc-weight sweep")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--reference-records-jsonl", default=None)
    parser.add_argument(
        "--use-native-baseline",
        action="store_true",
        help="score designs against the native transcript CDS (records mode only)",
    )
    parser.add_argument(
        "--gc-weight-sweep",
        default=None,
        help="comma/space separated gc_weight values; when set, run CAI-GC Pareto sweep",
    )
    parser.add_argument("--target-gc", type=float, default=0.55)
    parser.add_argument("--cai-weight", type=float, default=1.0)
    parser.add_argument("--gc-weight", type=float, default=0.10)
    parser.add_argument("--boundary-weight", type=float, default=0.05)
    parser.add_argument("--max-codon-changes", type=int, default=None)
    parser.add_argument("--progress-jsonl", default=None)
    parser.add_argument("--progress-every", type=int, default=25)
    return parser.parse_args(argv)


def _parse_float_list(text: str) -> list[float]:
    values = [
        float(part)
        for part in text.replace(",", " ").split()
        if part.strip()
    ]
    if not values:
        raise ValueError("expected at least one float")
    return values


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    cfg = CodonLatticeDPConfig(
        cai_weight=args.cai_weight,
        gc_weight=args.gc_weight,
        boundary_weight=args.boundary_weight,
        target_gc=args.target_gc,
        max_codon_changes=args.max_codon_changes,
    )
    if args.gc_weight_sweep:
        payload = run_protein_conditioned_gc_sweep(
            gc_weights=_parse_float_list(args.gc_weight_sweep),
            proteins=args.proteins,
            records_jsonl=args.records_jsonl,
            out_jsonl=args.out_jsonl,
            out_json=args.out_json,
            out_md=args.out_md,
            limit=args.limit,
            config=cfg,
            reference_records_jsonl=args.reference_records_jsonl,
            use_native_baseline=args.use_native_baseline,
        )
        print(
            json.dumps(
                {
                    "out_json": args.out_json,
                    "out_jsonl": args.out_jsonl,
                    "out_md": args.out_md,
                    "pareto_front_gc_weights": payload["pareto_front_gc_weights"],
                },
                sort_keys=True,
            )
        )
    else:
        payload = run_protein_conditioned_design(
            proteins=args.proteins,
            records_jsonl=args.records_jsonl,
            out_jsonl=args.out_jsonl,
            out_json=args.out_json,
            limit=args.limit,
            config=cfg,
            reference_records_jsonl=args.reference_records_jsonl,
            use_native_baseline=args.use_native_baseline,
        )
        print(json.dumps({"out_json": args.out_json, "summary": payload["summary"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ProteinConditionedCDSResult",
    "seed_cds_from_protein",
    "design_cds_for_protein",
    "design_records_for_proteins",
    "summarize_designs",
    "run_protein_conditioned_design",
    "run_protein_conditioned_gc_sweep",
    "main",
]
