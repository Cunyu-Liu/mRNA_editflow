from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts" / "route_a_v3" / "audit_a1_public_data.py"
CONFIG_PATH = ROOT / "configs" / "route_a_v3_a1_qualification.json"
SPEC = importlib.util.spec_from_file_location("audit_a1_public_data", MODULE_PATH)
assert SPEC and SPEC.loader
A1 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(A1)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _manifest(p0_root: Path, dataset_id: str, name: str, content: bytes) -> None:
    data_path = p0_root / dataset_id / name
    data_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.write_bytes(content)
    _write_json(
        data_path.parent / "manifest.json",
        {
            "provider": "PUBLIC_FIXTURE",
            "accession": dataset_id,
            "files": [
                {
                    "name": name,
                    "sha256": hashlib.sha256(content).hexdigest(),
                    "downloaded": True,
                }
            ],
        },
    )


def test_gap_inventory_is_aggregate_only_and_cannot_auto_qualify(tmp_path: Path) -> None:
    source = "ACGUACGUACGU"
    candidate = "ACGUACGUACGA"
    canonical = tmp_path / "ordinary" / "canonical.jsonl"
    _write_jsonl(
        canonical,
        [
            {
                "record_id": "PLUMAGE_FIXTURE",
                "accession": "GSE149487",
                "source_sequence": source,
                "candidate_sequence": candidate,
                "edit_script": [{"op": "SUB", "pos": 11, "token": "A"}],
                "edit_script_verified": True,
                "region": "5UTR",
                "labels": {
                    "te_log_fold_change": 0.5,
                    "mutant_293T_DNA_rep1": 10,
                    "mutant_293T_DNA_rep2": 11,
                },
                "metadata": {
                    "gene": "GENE_FIXTURE",
                    "wt_id": "WT_FIXTURE",
                    "source_file": "counts.tsv",
                },
            },
            {
                "record_id": "DENSE_FIXTURE",
                "accession": "GSE145046",
                "source_sequence": "ACGUACGUAC",
                "candidate_sequence": "ACGUACGUAC",
                "edit_script": [],
                "edit_script_verified": True,
                "region": "5UTR",
                "labels": {"count": 100, "norm_count": 1.0},
                "metadata": {"record_type": "input_support", "source_file": "input.tsv"},
            },
        ],
    )
    p0_root = tmp_path / "p0"
    _manifest(p0_root, "GSE149487", "counts.tsv", b"fixture-counts\n")
    _manifest(p0_root, "GSE145046", "input.tsv", b"fixture-input\n")

    report = A1.build_report(
        protocol_path=CONFIG_PATH,
        canonical_records=canonical,
        p0_root=p0_root,
        expected_canonical_sha256=_sha256(canonical),
        verify_dataset_ids={"GSE145046", "GSE149487"},
    )

    studies = {study["dataset_id"]: study for study in report["studies"]}
    plumage = studies["GSE149487"]
    assert plumage["nominal_rows"] == 1
    assert plumage["distinct_candidates"] == 1
    assert plumage["biological_source_groups"]["distinct_source_sequence_proxy"] == 1
    assert plumage["replicate_and_se_coverage"]["records_with_two_or_more_replicate_labels"] == 1
    assert plumage["beneficial_and_noise_zone_balance"]["legacy_explicit_delta_sign_counts"]["positive"] == 1
    assert plumage["qualified"] is False
    assert plumage["qualification_status"] == "BLOCKED_PENDING_PUBLIC_EVIDENCE"

    dense = studies["GSE145046"]
    assert dense["eligible_multi_candidate_pools"]["status"] == "NOT_REPRESENTED_BY_LEGACY_SOURCE_FIELDS"
    assert "LEGACY_CANONICAL_CONSUMES_INPUT_SUPPORT_NOT_30_SAMPLE_LABEL_COMPLETE_JOIN" in dense["blockers"]
    assert report["gate"]["qualified_independent_ordinary_studies"] == 0
    assert report["gate"]["next_phase_authorized"] is False
    serialized = json.dumps(report, sort_keys=True)
    assert source not in serialized
    assert candidate not in serialized


def test_forbidden_path_is_rejected_before_read(tmp_path: Path) -> None:
    forbidden = tmp_path / "restricted" / "ordinary.jsonl"
    with pytest.raises(A1.OrdinaryScopeError, match="rejected path before read"):
        A1.ensure_ordinary_path(forbidden, ["restricted", "gse246381"])


def test_canonical_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    canonical = tmp_path / "ordinary.jsonl"
    _write_jsonl(canonical, [])
    p0_root = tmp_path / "p0"
    p0_root.mkdir()
    with pytest.raises(ValueError, match="canonical SHA-256 mismatch"):
        A1.build_report(
            protocol_path=CONFIG_PATH,
            canonical_records=canonical,
            p0_root=p0_root,
            expected_canonical_sha256="0" * 64,
        )


def test_input_manifest_hash_drift_is_reported(tmp_path: Path) -> None:
    p0_root = tmp_path / "p0"
    data_path = p0_root / "GSE149487" / "counts.tsv"
    data_path.parent.mkdir(parents=True)
    data_path.write_bytes(b"current\n")
    _write_json(
        data_path.parent / "manifest.json",
        {
            "files": [
                {
                    "name": "counts.tsv",
                    "sha256": hashlib.sha256(b"stale\n").hexdigest(),
                    "downloaded": True,
                }
            ]
        },
    )
    summary = A1.summarize_p0_manifest("GSE149487", p0_root, verify_file_hashes=True)
    assert summary["status"] == "INCOMPLETE_BLOCKED"
    assert summary["verified_file_hashes"] == 0
    assert summary["hash_mismatches"] == ["counts.tsv"]


def test_unlisted_payload_is_not_silently_accepted(tmp_path: Path) -> None:
    p0_root = tmp_path / "p0"
    _manifest(p0_root, "GSE149487", "declared.tsv", b"declared\n")
    (p0_root / "GSE149487" / "supplement.xlsx").write_bytes(b"unlisted\n")
    summary = A1.summarize_p0_manifest("GSE149487", p0_root, verify_file_hashes=True)
    assert summary["status"] == "INCOMPLETE_BLOCKED"
    assert summary["unlisted_payload_files"] == ["supplement.xlsx"]


def test_missing_manifest_keeps_closed_summary_shape(tmp_path: Path) -> None:
    p0_root = tmp_path / "p0"
    p0_root.mkdir()
    summary = A1.summarize_p0_manifest("GSE232572", p0_root, verify_file_hashes=False)
    assert summary["status"] == "MISSING_BLOCKED"
    assert summary["unlisted_payload_files"] == []
    assert summary["quarantined_extra_files"] == []
    assert summary["hash_verification_requested"] is False


def test_protocol_keeps_training_and_auto_qualification_disabled() -> None:
    protocol = A1.load_protocol(CONFIG_PATH)
    assert protocol["scope"]["training_allowed"] is False
    assert protocol["scope"]["model_selection_allowed"] is False
    assert protocol["legacy_canonical_policy"]["may_auto_qualify_study"] is False
    assert protocol["gate"]["minimum_independent_ordinary_studies"] == 3
    assert protocol["gate"]["minimum_qualified_a1_studies"] == 2
    assert protocol["gate"]["minimum_qualified_a2_dense_studies"] == 1
