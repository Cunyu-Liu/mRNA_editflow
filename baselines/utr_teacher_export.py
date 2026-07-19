"""Export one-step 5'UTR oracle teachers for proposal-ranker distillation.

The UTR local-search baseline shows that a predictor-guided optimizer can find
large translation-efficiency gains while preserving CDS and 3'UTR constraints.
This module turns that upper-bound evidence into a training signal for
``train_proposal_ranker.py``.

For a source transcript ``x`` with 5'UTR ``u`` and fixed CDS context ``c``, we
enumerate one-step 5'UTR proposals

``C_1(x) = {sub_i^a(u), ins_i^a(u), del_i(u)}``

over biologically salient positions selected by the same motif/start-window
rule as :mod:`mrna_editflow.baselines.utr_local_search`. Each candidate ``y`` is
labelled by an independent oracle

``teacher_score(y) = TE_oracle(y, c) - TE_oracle(u, c)``.

The emitted JSONL rows are intentionally compatible with
``train_proposal_ranker.ProposalTeacherRow``:

``{"transcript_id": ..., "op": "sub|ins|del", "pos": i, "nt": a,
"teacher_score": ...}``.

The ranker then optimizes a Bradley-Terry pairwise objective over rows from the
same transcript. Export complexity is
``O(N * P * |alphabet| * Q)`` time and ``O(P * |alphabet|)`` memory per record,
where ``N`` is record count, ``P`` selected 5'UTR positions and ``Q`` oracle
feature cost. Candidate capping is deterministic and keeps both high and low
teacher extremes, preserving pairwise ranking contrast.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Mapping, Optional, Sequence

from mrna_editflow.baselines.utr_local_search import (
    RNA_ALPHABET,
    UTRLocalSearchConfig,
    _editable_positions,
    _normalise_utr,
    _objective_from_score,
)
from mrna_editflow.core.schema import MRNARecord
from mrna_editflow.data.download_mrna import load_records_jsonl
from mrna_editflow.data.split_contract import VerifiedSplitContract, load_and_verify_split_manifest
from mrna_editflow.eval.artifact_contract import (
    OracleContractError,
    build_run_metadata,
    load_and_verify_oracle_manifest,
    normalize_run_mode,
    prepare_scientific_records,
    require_paper_cli_inputs,
    validate_output_namespace,
    write_provenance_sidecar,
)
from mrna_editflow.eval.metrics import edit_distance, gc_fraction
from mrna_editflow.eval.oracle import LocalTranslationOracle


@dataclass(frozen=True)
class UTRTeacherRow:
    """One oracle-labelled 5'UTR edit proposal.

    ``pos`` follows MEF sampler/ranker semantics: substitutions and deletions
    act at the full-sequence position, and insertions add ``nt`` after ``pos``.
    Since this teacher is 5'UTR-only, full-sequence and 5'UTR positions are the
    same. ``teacher_score`` is the TE delta, while
    ``teacher_objective_delta`` records the optional GC/uAUG-penalized objective
    delta for audit use. Conversion complexity is ``O(sequence_length)`` because
    the candidate record is embedded for reproducibility.
    """

    transcript_id: str
    task_id: str
    op: str
    pos: int
    nt: str
    teacher_score: float
    source_te: float
    candidate_te: float
    source_mrl: float
    candidate_mrl: float
    source_objective: float
    candidate_objective: float
    teacher_objective_delta: float
    utr_edit_distance: int
    length_delta: int
    source_gc: float
    candidate_gc: float
    candidate: Mapping[str, object]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable mapping. Complexity is ``O(fields)``."""
        return dict(asdict(self))


def _candidate_record(record: MRNARecord, five_utr: str, suffix: str) -> MRNARecord:
    """Materialize a candidate transcript with CDS and 3'UTR fixed.

    Complexity is ``O(len(five_utr))`` for string storage.
    """
    return MRNARecord(
        transcript_id=f"{record.transcript_id}_{suffix}",
        five_utr=five_utr,
        cds=record.cds,
        three_utr=record.three_utr,
        species=record.species,
    )


def _one_step_candidates(
    record: MRNARecord,
    config: UTRLocalSearchConfig,
) -> list[tuple[str, int, str, MRNARecord]]:
    """Enumerate one-step 5'UTR proposals compatible with MEF ranker ops.

    Returned tuples are ``(op, pos, nt, candidate_record)``. The position ``pos``
    is the source-record full-sequence index. Insertion means "insert after
    ``pos``", matching :func:`mrna_editflow.sample._insert_nt_after` and
    ``train_proposal_ranker``. Complexity is ``O(P * |alphabet|)``.
    """
    source = _normalise_utr(record.five_utr)
    source_record = _candidate_record(record, source, "utr_teacher_source")
    positions = _editable_positions(source, config)
    rows: list[tuple[str, int, str, MRNARecord]] = []
    max_delta = max(0, int(config.max_length_delta))
    if config.allow_substitution:
        for pos in positions:
            old = source[pos]
            for nt in RNA_ALPHABET:
                if nt == old:
                    continue
                cand = source[:pos] + nt + source[pos + 1:]
                rows.append(("sub", pos, nt, _candidate_record(source_record, cand, f"sub_{pos}_{nt}")))
    if config.allow_insertion and source and max_delta >= 1:
        for pos in positions:
            for nt in RNA_ALPHABET:
                cand = source[:pos + 1] + nt + source[pos + 1:]
                rows.append(("ins", pos, nt, _candidate_record(source_record, cand, f"ins_{pos}_{nt}")))
    if config.allow_deletion and source and max_delta >= 1:
        for pos in positions:
            cand = source[:pos] + source[pos + 1:]
            rows.append(("del", pos, "", _candidate_record(source_record, cand, f"del_{pos}")))
    return rows


def _score_record_utr(record: MRNARecord, oracle: LocalTranslationOracle, config: UTRLocalSearchConfig) -> tuple[float, float, float]:
    """Return ``(objective, TE, MRL)`` for a record's 5'UTR.

    Complexity is the oracle feature cost, ``O(len(5'UTR))`` for the local
    deterministic fallback.
    """
    score = oracle.score_utr(record.five_utr, record.cds[:12])
    return _objective_from_score(score, config=config)


def _cap_rows(rows: Sequence[UTRTeacherRow], candidate_cap: int) -> list[UTRTeacherRow]:
    """Deterministically cap rows while preserving ranking extremes.

    If ``candidate_cap`` is positive and the candidate pool is larger, we keep
    alternating high-score and low-score rows. This gives pairwise ranker
    training a broad teacher margin instead of retaining only near-tied positive
    proposals. Complexity is ``O(R log R)`` for ``R`` rows.
    """
    cap = int(candidate_cap)
    ordered = sorted(rows, key=lambda row: (-row.teacher_score, row.op, row.pos, row.nt))
    if cap <= 0 or len(ordered) <= cap:
        return ordered
    selected: list[UTRTeacherRow] = []
    lo = 0
    hi = len(ordered) - 1
    while len(selected) < cap and lo <= hi:
        selected.append(ordered[lo])
        lo += 1
        if len(selected) >= cap or lo > hi:
            break
        selected.append(ordered[hi])
        hi -= 1
    return selected


def score_record_utr_teacher_rows(
    record: MRNARecord,
    *,
    config: Optional[UTRLocalSearchConfig] = None,
    oracle: Optional[LocalTranslationOracle] = None,
    candidate_cap: int = 0,
) -> list[UTRTeacherRow]:
    """Return one-step oracle teacher rows for one transcript.

    The source CDS and 3'UTR are never changed. The teacher label is a TE delta,
    so rows can be fed directly into ``train_proposal_ranker``. Complexity is
    ``O(P * |alphabet| * Q)``.
    """
    cfg = config or UTRLocalSearchConfig()
    pred = oracle or LocalTranslationOracle()
    source_utr = _normalise_utr(record.five_utr)
    source = _candidate_record(record, source_utr, "utr_teacher_source")
    source_objective, source_te, source_mrl = _score_record_utr(source, pred, cfg)
    rows: list[UTRTeacherRow] = []
    for op, pos, nt, cand in _one_step_candidates(source, cfg):
        cand_objective, cand_te, cand_mrl = _score_record_utr(cand, pred, cfg)
        rows.append(
            UTRTeacherRow(
                transcript_id=record.transcript_id,
                task_id="T5",
                op=op,
                pos=int(pos),
                nt=str(nt),
                teacher_score=float(cand_te - source_te),
                source_te=float(source_te),
                candidate_te=float(cand_te),
                source_mrl=float(source_mrl),
                candidate_mrl=float(cand_mrl),
                source_objective=float(source_objective),
                candidate_objective=float(cand_objective),
                teacher_objective_delta=float(cand_objective - source_objective),
                utr_edit_distance=int(edit_distance(source_utr, cand.five_utr)),
                length_delta=int(len(cand.five_utr) - len(source_utr)),
                source_gc=float(gc_fraction(source_utr)),
                candidate_gc=float(gc_fraction(cand.five_utr)),
                candidate=cand.to_dict(),
            )
        )
    return _cap_rows(rows, candidate_cap)


def summarize_teacher_rows(rows_by_record: Sequence[Sequence[UTRTeacherRow]]) -> dict[str, object]:
    """Summarize a grouped UTR teacher export.

    ``mean_best_candidate_te`` reports the mean best one-step candidate per
    source, which estimates how much teacher signal is immediately available to
    the ranker before multi-step UTR search. Complexity is ``O(total_rows)``.
    """
    groups = [list(rows) for rows in rows_by_record]
    non_empty = [rows for rows in groups if rows]
    all_rows = [row for rows in non_empty for row in rows]
    if not groups:
        return {
            "n_records": 0,
            "n_records_with_rows": 0,
            "n_rows": 0,
            "mean_rows_per_record": 0.0,
        }
    if not all_rows:
        return {
            "n_records": len(groups),
            "n_records_with_rows": 0,
            "n_rows": 0,
            "mean_rows_per_record": 0.0,
        }
    best_rows = [max(rows, key=lambda row: row.teacher_score) for rows in non_empty]
    worst_rows = [min(rows, key=lambda row: row.teacher_score) for rows in non_empty]
    return {
        "n_records": len(groups),
        "n_records_with_rows": len(non_empty),
        "n_rows": len(all_rows),
        "mean_rows_per_record": float(len(all_rows) / max(1, len(non_empty))),
        "mean_source_te": float(sum(row.source_te for row in best_rows) / len(best_rows)),
        "mean_best_candidate_te": float(sum(row.candidate_te for row in best_rows) / len(best_rows)),
        "mean_best_teacher_score": float(sum(row.teacher_score for row in best_rows) / len(best_rows)),
        "mean_worst_teacher_score": float(sum(row.teacher_score for row in worst_rows) / len(worst_rows)),
        "max_teacher_score": float(max(row.teacher_score for row in all_rows)),
        "min_teacher_score": float(min(row.teacher_score for row in all_rows)),
        "mean_abs_teacher_score": float(sum(abs(row.teacher_score) for row in all_rows) / len(all_rows)),
        "sub_rows": int(sum(row.op == "sub" for row in all_rows)),
        "ins_rows": int(sum(row.op == "ins" for row in all_rows)),
        "del_rows": int(sum(row.op == "del" for row in all_rows)),
    }


def export_utr_teacher_jsonl(
    records: Sequence[MRNARecord],
    *,
    out_jsonl: str,
    out_json: str,
    limit: Optional[int] = None,
    config: Optional[UTRLocalSearchConfig] = None,
    candidate_cap: int = 0,
    run_mode: str = "development",
    split_contract: Optional[VerifiedSplitContract] = None,
    split_role: Optional[str] = None,
    oracle_manifest: Optional[str] = None,
) -> dict[str, object]:
    """Write UTR one-step teacher rows and a summary JSON artifact.

    ``out_jsonl`` is the training input for ``train_proposal_ranker``. The
    summary JSON records config, counts and one-step TE headroom. Complexity is
    ``O(N * P * |alphabet| * Q + rows_written)``.
    """
    cfg = config or UTRLocalSearchConfig()
    run_mode = normalize_run_mode(run_mode)
    role_records, data_provenance = prepare_scientific_records(
        records,
        run_mode=run_mode,
        split_contract=split_contract,
        split_role=split_role,
        allowed_roles=("train",),
    )
    oracle_metadata = load_and_verify_oracle_manifest(oracle_manifest, run_mode=run_mode)
    if run_mode == "paper":
        raise OracleContractError(
            "paper teacher export has no independent-oracle execution adapter in this goal"
        )
    validate_output_namespace(out_jsonl, run_mode)
    validate_output_namespace(out_json, run_mode)
    selected = list(role_records[: int(limit)]) if limit is not None else list(role_records)
    scientific_validity = build_run_metadata(
        run_mode=run_mode,
        data_provenance=data_provenance,
        config=asdict(cfg),
        code_paths=(__file__,),
        oracle=oracle_metadata,
        extra_block_reasons=("heuristic_functional_oracle",),
        functional_claim=True,
    )
    oracle = LocalTranslationOracle()
    grouped_rows = [
        score_record_utr_teacher_rows(
            record,
            config=cfg,
            oracle=oracle,
            candidate_cap=candidate_cap,
        )
        for record in selected
    ]
    os.makedirs(os.path.dirname(os.path.abspath(out_jsonl)), exist_ok=True)
    with open(out_jsonl, "w", encoding="utf-8") as fh:
        for rows in grouped_rows:
            for row in rows:
                fh.write(json.dumps(row.to_dict(), sort_keys=True) + "\n")
    payload = {
        "config": asdict(cfg),
        "candidate_cap": int(candidate_cap),
        "n_records": len(selected),
        "out_jsonl": out_jsonl,
        "summary": summarize_teacher_rows(grouped_rows),
        "scientific_validity": scientific_validity,
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    payload["provenance_sidecar"] = write_provenance_sidecar(out_jsonl, scientific_validity)
    payload["summary_provenance_sidecar"] = write_provenance_sidecar(
        out_json, scientific_validity
    )
    return payload


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments. Complexity is ``O(number_of_args)``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-jsonl", required=True)
    parser.add_argument("--out-jsonl", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--candidate-cap", type=int, default=256)
    parser.add_argument("--start-window-nt", type=int, default=90)
    parser.add_argument("--max-edit-positions", type=int, default=90)
    parser.add_argument("--max-length-delta", type=int, default=1)
    parser.add_argument("--gc-target", type=float, default=0.52)
    parser.add_argument("--gc-penalty-weight", type=float, default=0.0)
    parser.add_argument("--uaug-penalty-weight", type=float, default=0.0)
    parser.add_argument("--disable-insertion", action="store_true")
    parser.add_argument("--disable-deletion", action="store_true")
    parser.add_argument("--disable-substitution", action="store_true")
    parser.add_argument("--run-mode", choices=("development", "paper"), default="development")
    parser.add_argument("--split-manifest", default=None)
    parser.add_argument("--split-role", choices=("train", "val", "test"), default=None)
    parser.add_argument("--oracle-manifest", default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point. Complexity is dominated by teacher export."""
    args = _parse_args(argv)
    require_paper_cli_inputs(
        run_mode=args.run_mode,
        split_manifest=args.split_manifest,
        split_role=args.split_role,
        allowed_roles=("train",),
        oracle_manifest=args.oracle_manifest,
        require_oracle=True,
    )
    config = UTRLocalSearchConfig(
        edit_budget=1,
        beam_width=1,
        max_length_delta=args.max_length_delta,
        start_window_nt=args.start_window_nt,
        max_edit_positions=args.max_edit_positions,
        gc_target=args.gc_target,
        gc_penalty_weight=args.gc_penalty_weight,
        uaug_penalty_weight=args.uaug_penalty_weight,
        allow_substitution=not args.disable_substitution,
        allow_insertion=not args.disable_insertion,
        allow_deletion=not args.disable_deletion,
    )
    payload = export_utr_teacher_jsonl(
        load_records_jsonl(args.records_jsonl),
        out_jsonl=args.out_jsonl,
        out_json=args.out_json,
        limit=args.limit,
        config=config,
        candidate_cap=args.candidate_cap,
        run_mode=args.run_mode,
        split_contract=(
            load_and_verify_split_manifest(args.split_manifest, records_path=args.records_jsonl)
            if args.split_manifest else None
        ),
        split_role=args.split_role,
        oracle_manifest=args.oracle_manifest,
    )
    print(json.dumps({"out_json": args.out_json, "out_jsonl": args.out_jsonl, "summary": payload["summary"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "UTRTeacherRow",
    "export_utr_teacher_jsonl",
    "score_record_utr_teacher_rows",
    "summarize_teacher_rows",
    "main",
]
