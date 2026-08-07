"""R1 ordinary-only grouping audit tests using synthetic records."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.m4_sparse.audit_ordinary_grouping import (  # noqa: E402
    assert_ordinary_input_path,
    audit_rows,
    load_ordinary_rows,
    write_json_exclusive,
)
from scripts.m4_sparse.run_o0x import load_rows as load_o0_rows  # noqa: E402


def _row(
    study,
    source,
    candidate,
    endpoint="ep",
    benchmark="5U-A1",
    edit_count=1,
    source_id=None,
    source_value=0.0,
    **extra,
):
    row = {
        "study": study,
        "benchmark": benchmark,
        "endpoint": endpoint,
        "context_id": "ctx",
        "source_id": source_id or f"{study}::{candidate}::src",
        "source_sequence": source,
        "candidate_sequence": candidate,
        "candidate_id": f"{study}::{candidate}",
        "edit_list": [{"op": "SUB", "pos": 0, "token": candidate[0]}],
        "edit_count": edit_count,
        "source_value": source_value,
        "candidate_value": 1.0,
        "delta": 1.0 - source_value,
    }
    row.update(extra)
    return row


def test_audit_reports_provisional_source_sequence_candidate_groups():
    rows = [
        _row("S1", "AAAA", "CAAA", source_id="mechanical_1"),
        _row("S1", "AAAA", "GAAA", source_id="mechanical_2"),
        _row("S1", "CCCC", "GCCC", source_id="mechanical_3"),
    ]
    report = audit_rows(rows)
    groups = report["source_sequence_candidate_groups"]
    assert report["status"] == "ORDINARY_ONLY_GROUPING_DIAGNOSTIC"
    assert report["publication_grouping_qualified"] is False
    assert groups["qualification_status"] == "PROVISIONAL_REQUIRES_BIOLOGICAL_PROVENANCE"
    assert groups["n_groups"] == 2
    assert groups["n_groups_with_at_least_2_distinct_candidates"] == 1
    assert groups["n_rows_in_groups_with_at_least_2_distinct_candidates"] == 2
    assert groups["max_distinct_candidates"] == 2
    assert groups["n_groups_with_multiple_mechanical_source_ids"] == 1


def test_audit_never_mixes_source_sequence_groups_across_endpoint_or_study():
    rows = [
        _row("S1", "AAAA", "CAAA", endpoint="A"),
        _row("S1", "AAAA", "GAAA", endpoint="B"),
        _row("S2", "AAAA", "UAAA", endpoint="A"),
    ]
    groups = audit_rows(rows)["source_sequence_candidate_groups"]
    assert groups["n_groups"] == 3
    assert groups["n_groups_with_at_least_2_distinct_candidates"] == 0


def test_audit_excludes_missing_or_conflicting_context_from_candidate_groups():
    missing = _row("S1", "AAAA", "CAAA")
    missing["context_id"] = None
    conflicting = _row("S1", "CCCC", "GCCC")
    conflicting["context_id"] = "ctx_a"
    conflicting["assay_context"] = "ctx_b"
    groups = audit_rows([missing, conflicting])["source_sequence_candidate_groups"]
    assert groups["n_groups"] == 0
    assert groups["n_rows_missing_safe_group_key"] == 2
    assert groups["n_rows_missing_source_sequence"] == 0
    assert groups["missing_group_key_atom_counts"] == {
        "conflicting_context": 1,
        "context": 1,
    }


def test_r2_input_coverage_reports_fail_closed_reasons_and_zero_edit_controls():
    valid = _row("S1", "A" * 120, "A" * 110 + "C" + "A" * 9)
    valid["edit_list"] = [{"op": "SUB", "pos": 110, "token": "C"}]
    zero_edit = _row("S1", "CCCC", "CCCC")
    zero_edit["edit_list"] = []
    zero_edit["edit_count"] = 0
    missing = _row("S1", "GGGG", "UGGG")
    missing["source_sequence"] = ""
    wide_source = "A" * 150
    wide_candidate = "C" + "A" * 119 + "C" + "A" * 29
    wide = _row("S1", wide_source, wide_candidate)
    wide["edit_list"] = [
        {"op": "SUB", "pos": 0, "token": "C"},
        {"op": "SUB", "pos": 120, "token": "C"},
    ]
    coverage = audit_rows([valid, zero_edit, missing, wide])["r2_input_coverage"]
    assert coverage["n_records"] == 4
    assert coverage["n_compatible"] == 2
    assert coverage["n_incompatible"] == 2
    assert coverage["n_compatible_zero_edit_controls"] == 1
    assert coverage["incompatibility_reason_counts"] == {
        "EDIT_SPAN_EXCEEDS_SINGLE_WINDOW": 1,
        "EMPTY_SEQUENCE": 1,
    }
    assert coverage["by_study"]["S1"]["n_compatible"] == 2
    assert coverage["by_study"]["S1"]["n_incompatible"] == 2
    assert coverage["by_benchmark"]["5U-A1"][
        "incompatibility_reason_counts"] == {
            "EDIT_SPAN_EXCEEDS_SINGLE_WINDOW": 1,
            "EMPTY_SEQUENCE": 1,
        }


def test_audit_reports_missing_and_conflicting_source_values_separately():
    missing = _row("S1", "AAAA", "CAAA", source_id="m1")
    missing["source_value"] = None
    first = _row("S1", "CCCC", "GCCC", source_id="c1", source_value=0.1)
    second = _row("S1", "CCCC", "UCCC", source_id="c2", source_value=0.2)
    groups = audit_rows([missing, first, second])["source_sequence_candidate_groups"]
    assert groups["n_groups_with_missing_or_nonfinite_source_value"] == 1
    assert groups["n_groups_with_inconsistent_source_value"] == 1


def test_audit_reports_study_dominance_edit_budget_and_missing_atoms():
    rows = [
        _row("S1", "AAAA", "CAAA", edit_count=1),
        _row("S1", "CCCC", "GCCC", edit_count=2),
        _row("S1", "GGGG", "UGGG", edit_count=10),
        _row("S2", "UUUU", "AUUU", edit_count=6),
    ]
    report = audit_rows(rows)
    dominance = report["study_dominance"]["5U-A1"]
    assert dominance["dominant_study"] == "S1"
    assert dominance["dominant_study_share"] == pytest.approx(0.75)
    assert dominance["exceeds_reference_max_share"] is True
    edit = report["edit_budget"]["classification_counts"]
    assert edit["primary_exact"] == 1
    assert edit["exploratory_exact"] == 1
    assert edit["outside_declared_exact_budgets"] == 2
    assert edit["above_primary_max"] == 2
    assert report["missing_grouping_atoms"]["biological_parent_group"][
        "missing_fraction"
    ] == pytest.approx(1.0)


def test_audit_exposes_legacy_endpoint_duplicate_candidate_risk():
    common = dict(source_id="legacy_src")
    rows = [
        _row("S1", "AAAA", "CAAA", endpoint="HEK", **common),
        _row("S1", "AAAA", "CAAA", endpoint="SH", **common),
    ]
    risk = audit_rows(rows)["legacy_source_id_pool_risk"]
    assert risk["n_groups_spanning_multiple_endpoint_contexts"] == 1
    assert risk["n_endpoint_mixed_groups_with_at_most_1_distinct_candidate"] == 1


def test_audit_refuses_sealed_study_row():
    rows = [_row("GSE246381", "AAAA", "CAAA")]
    with pytest.raises(ValueError, match="protected_identity"):
        audit_rows(rows)


@pytest.mark.parametrize(
    "name",
    [
        "sealed/final.jsonl",
        "sealed.final/data.jsonl",
        "restricted/data.jsonl",
        "GSE246381/x.jsonl",
    ],
)
def test_audit_refuses_sealed_or_restricted_path(name):
    with pytest.raises(ValueError, match="refuses protected path"):
        assert_ordinary_input_path(Path(name))


def test_loader_rejects_sealed_role_inside_ordinary_named_file(tmp_path):
    path = tmp_path / "ordinary.jsonl"
    path.write_text(
        json.dumps(_row("S1", "AAAA", "CAAA", data_role="SEALED_FINAL")) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="protected_role"):
        load_ordinary_rows(path)


def test_every_field_guard_rejects_deeply_nested_protected_reference(tmp_path):
    row = _row("S1", "AAAA", "CAAA")
    row["metadata"] = {
        "provenance": [{"alternate_accession": "GSE246381"}]
    }
    path = tmp_path / "ordinary.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"protected_field_value:.*alternate_accession"):
        load_ordinary_rows(path)


def test_every_field_guard_rejects_truthy_protected_field_name(tmp_path):
    row = _row("S1", "AAAA", "CAAA")
    row["metadata"] = {"restricted_payload": True}
    path = tmp_path / "ordinary.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"protected_field_name:.*restricted_payload"):
        load_ordinary_rows(path)


def test_negative_protection_flags_and_unsealed_word_do_not_false_positive(tmp_path):
    row = _row("S1", "AAAA", "CAAA")
    row.update({"is_sealed": False, "is_restricted": False, "state": "UNSEALED"})
    path = tmp_path / "ordinary.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    assert load_ordinary_rows(path) == [row]


def test_o0_runner_loader_enforces_same_row_boundary(tmp_path):
    row = _row("S1", "AAAA", "CAAA")
    row["alternate_dataset"] = "restricted_final"
    path = tmp_path / "ordinary.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="protected_field_value"):
        load_o0_rows(path)


def test_loader_rejects_ordinary_symlink_resolving_into_protected_path(tmp_path):
    protected_dir = tmp_path / "restricted_data"
    protected_dir.mkdir()
    target = protected_dir / "data.jsonl"
    target.write_text(json.dumps(_row("S1", "AAAA", "CAAA")) + "\n", encoding="utf-8")
    link = tmp_path / "ordinary.jsonl"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="protected path"):
        load_ordinary_rows(link)


def test_write_json_is_exclusive_and_preserves_previous_bytes(tmp_path):
    path = tmp_path / "audit.json"
    write_json_exclusive(path, {"version": 1})
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        write_json_exclusive(path, {"version": 2})
    assert path.read_bytes() == original


def test_write_json_refuses_protected_output_namespace(tmp_path):
    path = tmp_path / "restricted" / "audit.json"
    with pytest.raises(ValueError, match="protected path"):
        write_json_exclusive(path, {"version": 1})
    assert not path.exists()


@pytest.mark.parametrize(
    "script",
    [
        "scripts/m4_sparse/audit_ordinary_grouping.py",
        "scripts/m4_sparse/run_o0x.py",
    ],
)
def test_repo_root_direct_cli_help_smoke(script):
    completed = subprocess.run(
        [sys.executable, script, "--help"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--dataset" in completed.stdout
