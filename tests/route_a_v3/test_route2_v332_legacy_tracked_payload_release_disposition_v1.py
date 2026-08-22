import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
AUDIT = (
    ROOT
    / "audits/route_a_v3_route2_v332_legacy_tracked_payload_release_disposition_v1.json"
)
MEMO = (
    ROOT
    / "docs/paper/route2_v332_legacy_tracked_payload_release_disposition_memo_v1.md"
)
DRAFT = ROOT / "docs/paper/route2_v332_methods_results_draft_v1.md"
CODE_AVAILABILITY = (
    ROOT / "audits/route_a_v3_route2_v332_code_availability_completion_v1.json"
)
RELEASE_CANDIDATE = (
    ROOT / "audits/route_a_v3_route2_v332_github_release_candidate_v1.json"
)

EXPECTED_SIZES = {
    "data_registry/excel_inventory.parquet": 46498,
    "data/b0_splits/split_study_disjoint.jsonl": 11462850,
    "data/b0_splits/split_cross_region_transfer.jsonl": 11299013,
    "data/b0_splits/split_5utr_source_disjoint.jsonl": 7638905,
    "data/b0_splits/split_3utr_source_disjoint.jsonl": 4338809,
}

EXPECTED_DESTINATIONS = {
    source: (
        "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/"
        f"legacy_repository_payloads/{Path(source).name}"
    )
    for source in EXPECTED_SIZES
}

DIRECT_READER_MARKERS = {
    "d1_staging/scripts/b0/audit_split_manifests.py": (
        'default="data/b0_splits"',
        "reject_legacy_b0_splits(splits_dir)",
    ),
    "d1_staging/scripts/b0/eval_tracks.py": (
        'default="data/b0_splits"',
        "reject_legacy_b0_splits(splits_dir)",
    ),
    "d1_staging/scripts/b0/leakage_audit.py": (
        'default="data/b0_splits"',
        "reject_legacy_b0_splits(splits_dir)",
    ),
    "d1_staging/scripts/fm0/fm0_exposure_audit.py": (
        'default="data/b0_splits"',
        "reject_legacy_b0_splits(b0_splits_dir)",
    ),
}


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_five_payloads_are_absent_from_current_head_and_exactly_ignored() -> None:
    tracked = subprocess.run(
        ["git", "ls-files", "--", *EXPECTED_SIZES],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    ignored = subprocess.run(
        ["git", "check-ignore", "--", *EXPECTED_SIZES],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    ignore_lines = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()

    assert tracked == []
    assert set(ignored) == set(EXPECTED_SIZES)
    assert all(ignore_lines.count(path) == 1 for path in EXPECTED_SIZES)
    assert "data_registry/" not in ignore_lines
    assert "data/b0_splits/" not in ignore_lines


def test_migration_audit_records_five_size_verified_mnt_copies() -> None:
    audit = _load(AUDIT)
    rows = {row["source_path"]: row for row in audit["migrated_payloads"]}

    assert set(rows) == set(EXPECTED_SIZES)
    assert {path: row["size_bytes"] for path, row in rows.items()} == EXPECTED_SIZES
    assert {
        path: row["destination_path"] for path, row in rows.items()
    } == EXPECTED_DESTINATIONS
    assert all(row["tracked_in_current_head"] is False for row in rows.values())
    assert all(row["destination_file_present"] is True for row in rows.values())
    assert all(row["destination_size_verified"] is True for row in rows.values())
    assert audit["current_head_tracked_payload_file_count"] == 0
    assert audit["current_head_tracked_payload_total_size_bytes"] == 0
    assert audit["migrated_payload_file_count"] == 5
    assert audit["migrated_payload_total_size_bytes"] == 34786075
    assert audit["legacy_b0_jsonl_total_size_bytes"] == 34739577
    assert audit["destination"]["payload_file_count"] == 5
    assert audit["destination"]["payload_total_size_bytes"] == 34786075
    assert audit["destination"]["provenance_note"].endswith(
        "/legacy_repository_payloads/PROVENANCE.md"
    )
    assert audit["destination"]["provenance_note_present"] is True
    assert audit["destination"]["project_generated_checksum_files_created"] is False


def test_four_direct_legacy_readers_remain_fail_closed() -> None:
    audit = _load(AUDIT)
    controls = audit["repository_controls"]

    assert controls["legacy_b0_direct_reader_entrypoints"] == list(
        DIRECT_READER_MARKERS
    )
    assert controls["legacy_b0_direct_reader_entrypoint_count"] == 4
    assert controls["legacy_b0_guarded_direct_reader_count"] == 4
    assert controls["legacy_b0_unguarded_direct_reader_count"] == 0
    assert controls["legacy_b0_active_loader_negative_test_evidence_present"] is True
    for path, markers in DIRECT_READER_MARKERS.items():
        text = (ROOT / path).read_text(encoding="utf-8")
        assert all(marker in text for marker in markers)


def test_release_documents_resolve_only_the_payload_boundary() -> None:
    audit = _load(AUDIT)
    code = _load(CODE_AVAILABILITY)
    release = _load(RELEASE_CANDIDATE)
    draft = " ".join(DRAFT.read_text(encoding="utf-8").split())

    adjudication = audit["release_adjudication"]
    assert adjudication["current_head_formal_release_payload_boundary_compliant"] is True
    assert adjudication["legacy_tracked_payload_policy_resolved"] is True
    assert adjudication["formal_release_or_tag_authorized"] is False
    assert adjudication["submission_ready"] is False
    assert code["repository_facts"]["formal_release_payload_boundary_compliant"] is True
    assert code["repository_facts"]["legacy_payload_migration_authorized"] is True
    assert code["repository_facts"]["legacy_payload_migration_complete"] is True
    policy = release["tracked_data_policy"]
    assert policy["contract_compliant_for_formal_release"] is True
    assert policy["legacy_payload_migration_authorized"] is True
    assert policy["legacy_payload_migration_complete"] is True
    assert policy["current_head_tracked_payload_file_count"] == 0
    assert "removed from current-HEAD tracking" in draft
    assert "This resolves the formal-release payload-boundary component only" in draft
    assert all(value is False for value in audit["protected_outcomes"].values())


def test_migration_actions_and_authorization_boundaries_are_exact() -> None:
    audit = _load(AUDIT)
    actions = audit["actions_performed"]
    authority = audit["authority"]
    memo = MEMO.read_text(encoding="utf-8")

    assert authority["user_authorized_current_head_migration"] is True
    assert authority["shared_git_history_rewrite_authorized"] is False
    assert authority["formal_tag_or_release_authorized"] is False
    assert authority["public_payload_redistribution_authorized"] is False
    assert actions == {
        "payload_content_opened": False,
        "payload_copied_to_authorized_mnt_root": True,
        "payload_removed_from_current_head_tracking": True,
        "source_worktree_payload_deleted_during_local_migration": False,
        "payload_moved": False,
        "destination_payload_overwritten": False,
        "exact_ignore_rules_added": True,
        "provenance_note_written": True,
        "git_history_rewritten": False,
        "legacy_reader_behavior_changed": True,
        "excel_inventory_producer_default_changed": True,
        "formal_release_or_tag_created": False,
    }
    assert audit["release_adjudication"]["ordered_migration_actions_not_yet_executed"] == []
    assert "Current HEAD tracks none of the five payloads" in memo
    assert "does not authorize a formal tag or GitHub Release" in memo
    assert "Shared Git history was not rewritten" in memo
