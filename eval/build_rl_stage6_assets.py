"""Build immutable development-only Stage 6 evaluation split/oracle assets.

The builder derives small, deterministic test-role slices from an already
verified family-disjoint contract.  It never changes train/validation members,
never relaxes split validation, and preserves the source contract's scientific
blockers.  The output is explicitly development-only: it is useful for
reproducible engineering comparisons, not for experimental claims.

It also writes manifests for the three Oracle roles used by Stage 6:
``training`` (P1-04 teacher provenance), ``heldout`` (frozen P1-05 Oracle #3),
and an alternative heuristic.  The latter is intentionally marked
non-independent and can only support disagreement/reward-hacking diagnostics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PACKAGE_PARENT = _REPO_ROOT.parent
for _path in (str(_PACKAGE_PARENT), str(_REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from mrna_editflow.data.download_mrna import load_records_jsonl
from mrna_editflow.data.split_contract import (
    VerifiedSplitContract,
    build_split_manifest,
    load_and_verify_split_manifest,
    sha256_file,
)


SCHEMA_VERSION = 1
DEFAULT_MAX_TEST_RECORDS = 512


def _write_indices(path: Path, indices: Sequence[int]) -> None:
    path.write_text("".join(f"{int(index)}\n" for index in sorted(indices)), encoding="utf-8")


def _gc_fraction(sequence: str) -> float:
    sequence = str(sequence).upper().replace("T", "U")
    return (sequence.count("G") + sequence.count("C")) / max(1, len(sequence))


def _stable_rank(seed: int, value: object) -> str:
    return hashlib.sha256(f"stage6:{seed}:{value}".encode("utf-8")).hexdigest()


def select_ood_family_indices(
    test_indices: Sequence[int], clusters: Sequence[int] | None, *, max_records: int, seed: int
) -> list[int]:
    """Select source-test OOD families where mappings exist; otherwise a hash slice.

    A source contract can certify family-disjointness without publishing its
    cluster vector.  In that case every source-test record remains OOD relative
    to source train/validation, but we cannot claim a whole-family slice.
    """
    if clusters is None or any(index >= len(clusters) for index in test_indices):
        return sorted(test_indices, key=lambda index: _stable_rank(seed, index))[:max_records]
    by_cluster: dict[int, list[int]] = defaultdict(list)
    for index in test_indices:
        by_cluster[int(clusters[index])].append(int(index))
    selected: list[int] = []
    for cluster in sorted(by_cluster, key=lambda item: _stable_rank(seed, item)):
        members = sorted(by_cluster[cluster])
        if selected and len(selected) + len(members) > max_records:
            continue
        selected.extend(members)
    if not selected and by_cluster:
        selected.extend(sorted(next(iter(by_cluster.values()))))
    return sorted(selected)


def select_extreme_indices(
    values: Mapping[int, float], *, largest: bool, max_records: int
) -> list[int]:
    """Deterministic extreme-value selector; ties break by immutable index."""
    return [
        index for index, _ in sorted(values.items(), key=lambda item: (item[1], item[0]), reverse=largest)[:max_records]
    ]


def _write_derived_split(
    *,
    name: str,
    output_root: Path,
    source: VerifiedSplitContract,
    test_indices: Sequence[int],
    source_manifest: Path,
    selector: Mapping[str, object],
) -> dict[str, object]:
    if not test_indices:
        raise ValueError(f"derived split {name!r} selected zero test records")
    split_root = output_root / "splits" / name
    split_root.mkdir(parents=True, exist_ok=True)
    train = list(source.roles["train"].indices)
    val = list(source.roles["val"].indices)
    test = sorted(set(int(index) for index in test_indices))
    occupied = set(train) | set(val) | set(test)
    if len(occupied) != len(train) + len(val) + len(test):
        raise ValueError("derived roles overlap")
    excluded = sorted(set(range(source.records_count)) - occupied)
    paths = {role: split_root / f"{role}.idx" for role in ("train", "val", "test")}
    for role, indices in (("train", train), ("val", val), ("test", test)):
        _write_indices(paths[role], indices)
    excluded_path = split_root / "excluded.idx"
    _write_indices(excluded_path, excluded)
    leakage_path = split_root / "leakage_report.json"
    leakage_path.write_text(json.dumps({
        "schema_version": 1,
        "split": {"cluster_disjoint": True, "derived_from": str(source_manifest.resolve())},
        "summary": {
            "exact_match_count": 0,
            "leakage_exact_match_count": 0,
            "near_neighbor_threshold_passed": bool(source.near_neighbor_threshold_passed),
        },
        "selector": dict(selector),
        "source_manifest_sha256": source.manifest_sha256,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    blockers = list(source.block_reasons) + ["derived_stage6_development_evaluation", f"selector:{name}"]
    manifest = build_split_manifest(
        dataset_id=f"{source.dataset_id}__stage6_{name}",
        records_path=source.records_path,
        role_idx_paths={key: str(value) for key, value in paths.items()},
        leakage_report_path=str(leakage_path),
        algorithm="stage6_deterministic_source_test_slice",
        seed=int(selector.get("seed", 0)),
        family_threshold=0.8,
        family_disjoint=True,
        exact_cross_role_matches=0,
        near_neighbor_threshold_passed=bool(source.near_neighbor_threshold_passed),
        cluster_assignment_path=source.cluster_assignment_path,
        excluded_idx_path=str(excluded_path),
        excluded_reason="not_selected_for_this_development_evaluation_slice",
        paper_eligible=False,
        block_reasons=blockers,
    )
    manifest_path = split_root / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    verified = load_and_verify_split_manifest(str(manifest_path))
    return {
        "name": name,
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": verified.manifest_sha256,
        "test_count": verified.roles["test"].count,
        "excluded_count": verified.excluded.count if verified.excluded else 0,
        "selector": dict(selector),
        "paper_eligible": verified.paper_eligible,
        "block_reasons": list(verified.block_reasons),
    }


def _write_oracle_manifest(path: Path, payload: Mapping[str, object]) -> dict[str, object]:
    artifact = Path(str(payload["artifact_path"])).resolve()
    if not artifact.is_file():
        raise FileNotFoundError(f"Oracle artifact is missing: {artifact}")
    row = dict(payload)
    row.update({"schema_version": SCHEMA_VERSION, "artifact_path": str(artifact), "artifact_sha256": sha256_file(artifact)})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"manifest_path": str(path.resolve()), "manifest_sha256": sha256_file(path), **row}


def build_assets(project_root: str | Path, output_root: str | Path, *, max_test_records: int = DEFAULT_MAX_TEST_RECORDS, seed: int = 20260722) -> dict[str, object]:
    root = Path(project_root).resolve()
    out = Path(output_root).resolve()
    source_manifest = root / "benchmark/dev/gencode_family_leakage_protocol/split_manifest.json"
    source = load_and_verify_split_manifest(str(source_manifest))
    records = load_records_jsonl(source.records_path)
    test_indices = list(source.roles["test"].indices)
    lengths = {index: float(len(records[index].seq)) for index in test_indices}
    gcs = {index: _gc_fraction(records[index].seq) for index in test_indices}
    has_cluster_assignments = len(source.cluster_assignments) == source.records_count
    ood = select_ood_family_indices(test_indices, source.cluster_assignments if has_cluster_assignments else None, max_records=max_test_records, seed=seed)
    length = select_extreme_indices(lengths, largest=True, max_records=max_test_records)
    low_n = max_test_records // 2
    high_n = max_test_records - low_n
    low_gc = select_extreme_indices(gcs, largest=False, max_records=low_n)
    high_gc = select_extreme_indices(gcs, largest=True, max_records=high_n)
    gc = sorted(set(low_gc) | set(high_gc))
    split_rows = [
        _write_derived_split(name="ood_family_test", output_root=out, source=source, test_indices=ood, source_manifest=source_manifest, selector={"kind": "whole_source_test_families" if has_cluster_assignments else "deterministic_source_family_disjoint_test_hash_slice", "seed": seed, "max_records": max_test_records, "source_cluster_assignments_available": has_cluster_assignments}),
        _write_derived_split(name="length_shift_test", output_root=out, source=source, test_indices=length, source_manifest=source_manifest, selector={"kind": "longest_source_test_records", "max_records": max_test_records, "min_length": min(lengths[index] for index in length)}),
        _write_derived_split(name="gc_shift_test", output_root=out, source=source, test_indices=gc, source_manifest=source_manifest, selector={"kind": "low_and_high_gc_source_test_tails", "max_records": max_test_records, "low_max_gc": max(gcs[index] for index in low_gc), "high_min_gc": min(gcs[index] for index in high_gc)}),
    ]
    oracle_root = out / "oracles"
    training = _write_oracle_manifest(oracle_root / "training_oracle_manifest.json", {
        "oracle_type": "p1_04_crossfit_training_teacher_predictions",
        "independent": False,
        "source": "P1-04 training teacher prediction artifact; never admissible as final evaluation Oracle",
        "independence_statement": "This is the training teacher provenance record and is intentionally non-independent.",
        "artifact_path": root / "data/processed/p1_04_predictions/cnn_50mer__sample2019_mpra__crossfit.json",
    })
    heldout_source = root / "benchmark/paper/leakage_free_headline/oracle_manifest.json"
    heldout = json.loads(heldout_source.read_text(encoding="utf-8"))
    heldout_row = {"manifest_path": str(heldout_source.resolve()), "manifest_sha256": sha256_file(heldout_source), **heldout}
    alternative = _write_oracle_manifest(oracle_root / "alternative_heuristic_oracle_manifest.json", {
        "oracle_type": "alternative_heuristic_local_translation_oracle",
        "independent": False,
        "source": "Deterministic LocalTranslationOracle code path used only for disagreement and reward-hacking diagnostics",
        "independence_statement": "Alternative heuristic; it is deliberately not treated as an independent final evaluation Oracle.",
        "artifact_path": root / "eval/oracle.py",
    })
    report = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "rl_stage6_development_assets",
        "claim_tier": "development_only",
        "paper_eligible": False,
        "source_split_manifest": str(source_manifest.resolve()),
        "source_split_manifest_sha256": source.manifest_sha256,
        "source_dataset_hash": source.records_sha256,
        "splits": split_rows,
        "oracles": {"training": training, "heldout": heldout_row, "alternative": alternative, "public_experimental": {"status": "unavailable", "reason": "no frozen public experimental predictor manifest"}},
        "species_shift": {"status": "unavailable", "reason": "source GENCODE evaluation corpus contains one species (human)"},
        "claim_boundary": "Derived shifts are development-only source-test slices. They preserve the original family contract and blockers; they do not establish experimental performance or paper eligibility.",
    }
    report_path = out / "rl_stage6_asset_manifest.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output-root", default="benchmark/dev/rl_stage6_assets")
    parser.add_argument("--max-test-records", type=int, default=DEFAULT_MAX_TEST_RECORDS)
    parser.add_argument("--seed", type=int, default=20260722)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = build_assets(args.project_root, args.output_root, max_test_records=args.max_test_records, seed=args.seed)
    print(json.dumps({"splits": report["splits"], "asset_manifest": str(Path(args.output_root).resolve() / "rl_stage6_asset_manifest.json")}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
