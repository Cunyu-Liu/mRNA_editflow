"""Leakage-gated, matched-budget frozen-backbone adapter comparison.

Architecture upgrade #4 of the SOTA roadmap asks us to plug frozen mRNA
foundation-model encoders (mRNA-LM / Helix-mRNA / CodonFM ...) under the
Edit-Flow head and compare them *fairly*. Two ingredients make such a
comparison publishable rather than misleading:

1. **Leakage gate.** A frozen encoder can silently inflate downstream numbers
   if the evaluation transcripts (or close homologues) were in its pretraining
   corpus. Before any comparison we audit the evaluation split against a
   reference corpus with :func:`mrna_editflow.data.leakage_audit.audit_leakage`.
   If the split is flagged we *refuse* the fair-comparison claim and mark the
   run ``leaked``.
2. **Matched training budget.** Every backbone is placed under the *same*
   :class:`MEFConfig` head (identical ``ModelConfig``, identical optimizer
   steps, identical records and seed). Only the frozen encoder changes, so the
   trainable-parameter budget is provably identical across arms. We record the
   trainable/frozen parameter counts to prove the budget really matched.

Honesty contract
----------------
In this offline environment the external foundation models resolve to
deterministic *placeholder* encoders (:class:`FrozenBackbone.is_real == False`).
Their training loss is a real number, but it reflects **pipeline / adapter
plumbing under a matched budget only** -- it is emphatically *not* a proxy for
the real foundation model's downstream quality. Every stub arm is therefore
tagged ``is_real=False`` and ``valid_quality_signal=False``. The only arm whose
loss is a genuine trainable-quality signal is the real from-scratch ``none``
encoder. We never fabricate SOTA metrics for a stub.

The module is importable (for tests / notebooks) and has an offline CLI.
Complexity is dominated by the small-step Stage A training per backbone.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass, field
from typing import Mapping, Optional, Sequence

import torch

from mrna_editflow.core.config import MEFConfig
from mrna_editflow.core.schema import MRNARecord
from mrna_editflow.data.leakage_audit import audit_leakage, write_leakage_report
from mrna_editflow.data.download_mrna import load_records_jsonl
from mrna_editflow.models.backbones import SUPPORTED_BACKBONES
from mrna_editflow.train_backbone import train_stage_a


DEFAULT_BACKBONES: tuple[str, ...] = ("none", "helix_mrna", "mrnabert")


@dataclass(frozen=True)
class LeakageGate:
    """Outcome of the pre-comparison leakage audit.

    ``passed`` is ``True`` when the comparison may be reported as *fair*: either
    leakage was audited and nothing was flagged, or auditing was skipped because
    no reference corpus was supplied (in which case ``audited`` is ``False`` and
    the caller is told leakage was *not* checked).
    """

    enabled: bool
    audited: bool
    passed: bool
    flagged_fraction: float
    exact_match_count: int
    max_jaccard: float
    max_containment: float
    n_query: int
    n_reference: int
    note: str
    summary: Mapping[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        payload = dict(asdict(self))
        payload["summary"] = dict(self.summary)
        return payload


@dataclass(frozen=True)
class BackboneRun:
    """Matched-budget Stage A result for one frozen backbone."""

    backbone: str
    is_real: bool
    kind: str
    granularity: str
    hidden_dim: int
    steps: int
    trainable_params: int
    backbone_params: int
    backbone_trainable_params: int
    best_loss: float
    last_loss: float
    finite_loss: bool
    valid_quality_signal: bool
    note: str

    def to_dict(self) -> dict[str, object]:
        return dict(asdict(self))


def _coerce_records(records: Optional[Sequence[object]]) -> list[MRNARecord]:
    out: list[MRNARecord] = []
    for rec in records or ():
        if isinstance(rec, MRNARecord):
            out.append(rec)
        elif isinstance(rec, Mapping):
            out.append(MRNARecord.from_dict(dict(rec)))
        else:
            raise TypeError(f"unsupported record type: {type(rec)!r}")
    return out


def _run_leakage_gate(
    query_records: Sequence[object],
    reference_records: Optional[Sequence[object]],
    *,
    kmer: int,
    top_k: int,
    jaccard_threshold: float,
    containment_threshold: float,
) -> LeakageGate:
    if not reference_records:
        return LeakageGate(
            enabled=False,
            audited=False,
            passed=True,
            flagged_fraction=0.0,
            exact_match_count=0,
            max_jaccard=0.0,
            max_containment=0.0,
            n_query=len(query_records),
            n_reference=0,
            note=(
                "no reference corpus provided; leakage was NOT audited. "
                "Cross-backbone quality claims remain unsupported until a "
                "pretraining-corpus proxy is supplied."
            ),
        )

    audit = audit_leakage(
        query_records,
        reference_records,
        k=kmer,
        top_k=top_k,
        jaccard_threshold=jaccard_threshold,
        containment_threshold=containment_threshold,
    )
    summary = audit["summary"]
    flagged_fraction = float(summary["flagged_fraction"])
    exact = int(summary["exact_match_count"])
    passed = flagged_fraction <= 0.0 and exact == 0
    note = (
        "clean: no evaluation transcript overlapped the reference corpus above "
        "threshold; matched-budget comparison may be reported as fair."
        if passed
        else (
            "LEAKED: evaluation transcripts overlap the reference corpus above "
            "threshold. Fair-comparison claim refused; frozen-encoder numbers "
            "would be inflated by memorisation."
        )
    )
    return LeakageGate(
        enabled=True,
        audited=True,
        passed=passed,
        flagged_fraction=flagged_fraction,
        exact_match_count=exact,
        max_jaccard=float(summary["max_jaccard"]),
        max_containment=float(summary["max_containment"]),
        n_query=int(summary["n_query"]),
        n_reference=int(summary["n_reference"]),
        note=note,
        summary=summary,
    )


def _make_backbone_config(
    base: MEFConfig,
    backbone_name: str,
    hidden_dim: Optional[int],
    workdir: str,
) -> MEFConfig:
    cfg = copy.deepcopy(base)
    cfg.backbone.name = backbone_name
    cfg.backbone.freeze = True
    if hidden_dim is not None:
        cfg.backbone.hidden_dim = int(hidden_dim)
    run_dir = os.path.join(workdir, backbone_name)
    cfg.train.save_dir = os.path.join(run_dir, "ckpts")
    cfg.train.profile_path = os.path.join(run_dir, "profile.jsonl")
    return cfg


def _count_params(module: torch.nn.Module) -> tuple[int, int]:
    total = 0
    trainable = 0
    for p in module.parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    return total, trainable


def _train_one_backbone(
    backbone_name: str,
    base_config: MEFConfig,
    hidden_dim: Optional[int],
    records: Optional[Sequence[MRNARecord]],
    steps: int,
    synthetic_n: int,
    device: Optional[object],
    seed: int,
    workdir: str,
) -> BackboneRun:
    cfg = _make_backbone_config(base_config, backbone_name, hidden_dim, workdir)
    result = train_stage_a(
        cfg,
        records=list(records) if records is not None else None,
        steps=steps,
        synthetic_n=synthetic_n,
        device=device,
        seed=seed,
    )
    backbone = result["backbone"]
    model = result["model"]
    _model_total, model_trainable = _count_params(model)
    backbone_total, backbone_trainable = _count_params(backbone)
    best_loss = float(result["best_loss"])
    last_stats = result.get("last_stats") or {}
    last_loss = float(last_stats.get("loss", best_loss))
    finite = bool(math.isfinite(best_loss) and math.isfinite(last_loss))
    is_real = bool(getattr(backbone, "is_real", False))
    kind = "real" if is_real else "adapter-stub"
    note = (
        "genuine trainable-quality signal (from-scratch light encoder)."
        if is_real
        else (
            "placeholder encoder: loss reflects matched-budget pipeline plumbing "
            "ONLY, not the real foundation model's downstream quality. Do not "
            "cite as a SOTA metric."
        )
    )
    return BackboneRun(
        backbone=backbone_name,
        is_real=is_real,
        kind=kind,
        granularity=str(cfg.backbone.granularity),
        hidden_dim=int(cfg.backbone.hidden_dim),
        steps=int(steps),
        trainable_params=int(model_trainable),
        backbone_params=int(backbone_total),
        backbone_trainable_params=int(backbone_trainable),
        best_loss=best_loss,
        last_loss=last_loss,
        finite_loss=finite,
        valid_quality_signal=is_real,
        note=note,
    )


def run_frozen_backbone_comparison(
    query_records: Sequence[object],
    reference_records: Optional[Sequence[object]] = None,
    *,
    backbones: Sequence[str] = DEFAULT_BACKBONES,
    train_records: Optional[Sequence[object]] = None,
    base_config: Optional[MEFConfig] = None,
    hidden_dim: Optional[int] = None,
    steps: int = 3,
    synthetic_n: int = 8,
    device: Optional[object] = None,
    seed: int = 0,
    require_gate: bool = True,
    kmer: int = 15,
    top_k: int = 3,
    jaccard_threshold: float = 0.80,
    containment_threshold: float = 0.95,
    workdir: Optional[str] = None,
) -> dict[str, object]:
    """Run a leakage-gated, matched-budget frozen-backbone comparison.

    Parameters
    ----------
    query_records : sequence of MRNARecord / dict
        The evaluation split whose leakage against ``reference_records`` is
        audited (also the default training records when ``train_records`` is
        ``None`` and the records are non-empty).
    reference_records : sequence, optional
        Reference/pretraining-corpus proxy for the leakage gate. When ``None``
        the gate is *skipped* and the result documents that leakage was not
        audited.
    backbones : sequence of str
        Backbone names to compare (must be in ``SUPPORTED_BACKBONES``). Always
        include ``none`` as the real reference arm.
    require_gate : bool
        When ``True`` (default) and the gate is enabled but fails, training arms
        are **not** run; the result records the refusal. Set ``False`` only for
        diagnostics where you explicitly want the (inflated) numbers.

    Returns
    -------
    dict
        JSON-ready comparison with ``leakage_gate``, ``matched_budget`` and
        ``runs`` sections plus a plain-language ``interpretation``.
    """
    unknown = [b for b in backbones if b not in SUPPORTED_BACKBONES]
    if unknown:
        raise ValueError(
            f"unknown backbone(s) {unknown}; supported: {sorted(SUPPORTED_BACKBONES)}"
        )
    if not backbones:
        raise ValueError("backbones must be non-empty")

    q_records = _coerce_records(query_records)
    r_records = _coerce_records(reference_records) if reference_records else []
    if train_records is not None:
        t_records: Optional[list[MRNARecord]] = _coerce_records(train_records)
    elif q_records:
        t_records = q_records
    else:
        t_records = None  # -> synthesize inside train_stage_a

    base_cfg = base_config if base_config is not None else MEFConfig()

    gate = _run_leakage_gate(
        q_records,
        r_records,
        kmer=kmer,
        top_k=top_k,
        jaccard_threshold=jaccard_threshold,
        containment_threshold=containment_threshold,
    )

    runs: list[BackboneRun] = []
    skipped_reason: Optional[str] = None
    if gate.enabled and not gate.passed and require_gate:
        skipped_reason = (
            "leakage gate failed and require_gate=True; training arms skipped to "
            "avoid publishing memorisation-inflated frozen-encoder numbers."
        )
    else:
        owns_workdir = workdir is None
        run_workdir = workdir or tempfile.mkdtemp(prefix="mef_backbone_cmp_")
        try:
            for name in backbones:
                runs.append(
                    _train_one_backbone(
                        backbone_name=name,
                        base_config=base_cfg,
                        hidden_dim=hidden_dim,
                        records=t_records,
                        steps=steps,
                        synthetic_n=synthetic_n,
                        device=device,
                        seed=seed,
                        workdir=run_workdir,
                    )
                )
        finally:
            if owns_workdir:
                _safe_rmtree(run_workdir)

    trainable_counts = {r.trainable_params for r in runs}
    matched = len(trainable_counts) == 1 if runs else False
    matched_budget = {
        "steps": int(steps),
        "seed": int(seed),
        "model_config": asdict(base_cfg.model),
        "trainable_params_consistent": bool(matched),
        "trainable_params": (next(iter(trainable_counts)) if matched else None),
        "note": (
            "identical MEFConfig head + steps + seed across arms; only the frozen "
            "encoder differs, so the trainable budget is matched by construction."
        ),
    }

    real_runs = [r for r in runs if r.is_real]
    stub_runs = [r for r in runs if not r.is_real]
    interpretation = _interpret(gate, runs, real_runs, stub_runs, skipped_reason)

    return {
        "config": {
            "backbones": list(backbones),
            "steps": int(steps),
            "seed": int(seed),
            "hidden_dim": (int(hidden_dim) if hidden_dim is not None else int(base_cfg.backbone.hidden_dim)),
            "require_gate": bool(require_gate),
            "leakage": {
                "kmer": int(kmer),
                "top_k": int(top_k),
                "jaccard_threshold": float(jaccard_threshold),
                "containment_threshold": float(containment_threshold),
            },
        },
        "leakage_gate": gate.to_dict(),
        "matched_budget": matched_budget,
        "runs": [r.to_dict() for r in runs],
        "skipped_reason": skipped_reason,
        "n_real_arms": len(real_runs),
        "n_stub_arms": len(stub_runs),
        "interpretation": interpretation,
    }


def _safe_rmtree(path: str) -> None:
    import shutil

    try:
        shutil.rmtree(path, ignore_errors=True)
    except Exception:  # pragma: no cover - best-effort cleanup
        pass


def _interpret(
    gate: LeakageGate,
    runs: Sequence[BackboneRun],
    real_runs: Sequence[BackboneRun],
    stub_runs: Sequence[BackboneRun],
    skipped_reason: Optional[str],
) -> str:
    parts: list[str] = []
    if not gate.enabled:
        parts.append(
            "Leakage was NOT audited (no reference corpus); no fair cross-backbone "
            "quality claim can be made."
        )
    elif gate.passed:
        parts.append(
            f"Leakage gate PASSED (flagged_fraction={gate.flagged_fraction:.3f}, "
            f"exact_match_count={gate.exact_match_count}); the matched-budget "
            "comparison is fair with respect to memorisation."
        )
    else:
        parts.append(
            f"Leakage gate FAILED (flagged_fraction={gate.flagged_fraction:.3f}, "
            f"exact_match_count={gate.exact_match_count}); fair-comparison claim "
            "refused."
        )
    if skipped_reason:
        parts.append(skipped_reason)
        return " ".join(parts)
    if stub_runs:
        stub_names = ", ".join(sorted({r.backbone for r in stub_runs}))
        parts.append(
            f"Stub (placeholder) arms [{stub_names}] ran end-to-end under the "
            "matched budget; their losses index pipeline plumbing only and are "
            "NOT SOTA metrics."
        )
    if real_runs:
        real_names = ", ".join(sorted({r.backbone for r in real_runs}))
        parts.append(
            f"Real arm(s) [{real_names}] carry a genuine trainable-quality "
            "signal; real external checkpoints must be dropped into "
            "FrozenBackbone before their numbers become quotable."
        )
    return " ".join(parts)


def write_comparison_report(
    result: Mapping[str, object],
    out_json: str,
    out_md: Optional[str] = None,
) -> tuple[str, Optional[str]]:
    """Write JSON and optional Markdown report. Complexity: O(report size)."""
    os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)
    if out_md is None:
        return out_json, None
    os.makedirs(os.path.dirname(os.path.abspath(out_md)), exist_ok=True)
    gate = result.get("leakage_gate", {})
    runs = result.get("runs", [])
    with open(out_md, "w", encoding="utf-8") as fh:
        fh.write("# Frozen-Backbone Adapter Comparison (leakage-gated)\n\n")
        fh.write("## Leakage gate\n\n")
        fh.write("| Field | Value |\n|---|---:|\n")
        if isinstance(gate, Mapping):
            for key in (
                "enabled",
                "audited",
                "passed",
                "flagged_fraction",
                "exact_match_count",
                "max_jaccard",
                "max_containment",
                "n_query",
                "n_reference",
            ):
                fh.write(f"| `{key}` | {gate.get(key, '')} |\n")
            fh.write(f"\n> {gate.get('note', '')}\n\n")
        fh.write("## Matched-budget runs\n\n")
        fh.write(
            "| Backbone | Real? | Kind | Trainable params | Best loss | Last loss | "
            "Finite | Valid quality signal |\n"
        )
        fh.write("|---|---:|---|---:|---:|---:|---:|---:|\n")
        if isinstance(runs, Sequence):
            for row in runs:
                if not isinstance(row, Mapping):
                    continue
                fh.write(
                    f"| {row.get('backbone', '')} | {row.get('is_real', '')} | "
                    f"{row.get('kind', '')} | {row.get('trainable_params', '')} | "
                    f"{float(row.get('best_loss', float('nan'))):.4f} | "
                    f"{float(row.get('last_loss', float('nan'))):.4f} | "
                    f"{row.get('finite_loss', '')} | {row.get('valid_quality_signal', '')} |\n"
                )
        fh.write(f"\n## Interpretation\n\n{result.get('interpretation', '')}\n")
    return out_json, out_md


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-jsonl", default=None, help="evaluation split JSONL")
    parser.add_argument("--reference-jsonl", default=None, help="reference/pretraining corpus JSONL")
    parser.add_argument("--train-jsonl", default=None, help="training records JSONL (defaults to query split)")
    parser.add_argument("--backbones", nargs="+", default=list(DEFAULT_BACKBONES))
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--synthetic-n", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-require-gate", action="store_true")
    parser.add_argument("--kmer", type=int, default=15)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--jaccard-threshold", type=float, default=0.80)
    parser.add_argument("--containment-threshold", type=float, default=0.95)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", default=None)
    parser.add_argument("--leakage-json", default=None, help="also write the raw leakage audit")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    query = load_records_jsonl(args.query_jsonl) if args.query_jsonl else []
    reference = load_records_jsonl(args.reference_jsonl) if args.reference_jsonl else None
    train = load_records_jsonl(args.train_jsonl) if args.train_jsonl else None
    result = run_frozen_backbone_comparison(
        query,
        reference,
        backbones=args.backbones,
        train_records=train,
        hidden_dim=args.hidden_dim,
        steps=args.steps,
        synthetic_n=args.synthetic_n,
        device=args.device,
        seed=args.seed,
        require_gate=not args.no_require_gate,
        kmer=args.kmer,
        top_k=args.top_k,
        jaccard_threshold=args.jaccard_threshold,
        containment_threshold=args.containment_threshold,
    )
    out_json, out_md = write_comparison_report(result, args.out_json, args.out_md)
    if args.leakage_json and reference:
        audit = audit_leakage(
            query,
            reference,
            k=args.kmer,
            top_k=args.top_k,
            jaccard_threshold=args.jaccard_threshold,
            containment_threshold=args.containment_threshold,
        )
        write_leakage_report(audit, args.leakage_json)
    print(
        json.dumps(
            {
                "out_json": out_json,
                "out_md": out_md,
                "gate_passed": result["leakage_gate"]["passed"],
                "n_runs": len(result["runs"]),
                "n_real_arms": result["n_real_arms"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "LeakageGate",
    "BackboneRun",
    "run_frozen_backbone_comparison",
    "write_comparison_report",
    "main",
    "DEFAULT_BACKBONES",
]
