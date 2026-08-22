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

DIRECT_READER_MARKERS = {
    "d1_staging/scripts/b0/audit_split_manifests.py": (
        'default="data/b0_splits"',
        "entries = load_manifest(manifest_path)",
    ),
    "d1_staging/scripts/b0/eval_tracks.py": (
        'default="data/b0_splits"',
        "entries = load_jsonl(path)",
    ),
    "d1_staging/scripts/b0/leakage_audit.py": (
        'default="data/b0_splits"',
        "load_manifest(path)",
    ),
    "d1_staging/scripts/fm0/fm0_exposure_audit.py": (
        'default="data/b0_splits"',
        'b0_splits_dir.glob("split_*.jsonl")',
    ),
}


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_five_payloads_are_tracked_with_declared_sizes_without_content_reads() -> None:
    audit = _load(AUDIT)
    tracked = subprocess.run(
        ["git", "ls-files", "--", *EXPECTED_SIZES],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()

    assert set(tracked) == set(EXPECTED_SIZES)
    assert {row["path"]: row["size_bytes"] for row in audit["tracked_payloads"]} == (
        EXPECTED_SIZES
    )
    assert all((ROOT / path).stat().st_size == size for path, size in EXPECTED_SIZES.items())
    assert audit["tracked_payload_file_count"] == 5
    assert audit["tracked_payload_total_size_bytes"] == sum(EXPECTED_SIZES.values()) == 34786075
    assert audit["legacy_b0_jsonl_total_size_bytes"] == 34739577
    assert audit["scope"] == (
        "PATH_TRACKING_SIZE_AND_TEXT_REFERENCE_AUDIT_ONLY_NO_PAYLOAD_CONTENT_OPENED"
    )
    assert audit["actions_performed"] == {
        "payload_content_opened": False,
        "payload_copied": False,
        "payload_deleted": False,
        "payload_moved": False,
        "git_history_rewritten": False,
        "legacy_reader_behavior_changed": False,
        "formal_release_or_tag_created": False,
    }


def test_four_direct_legacy_readers_exist_and_negative_test_evidence_is_absent() -> None:
    audit = _load(AUDIT)
    readers = audit["text_reference_inventory"]["legacy_b0_direct_reader_entrypoints"]

    assert readers == list(DIRECT_READER_MARKERS)
    assert audit["legacy_b0_direct_reader_entrypoint_count"] == 4
    for path, markers in DIRECT_READER_MARKERS.items():
        text = (ROOT / path).read_text(encoding="utf-8")
        assert all(marker in text for marker in markers)

    test_files = list((ROOT / "tests").rglob("*.py"))
    staging_tests = ROOT / "d1_staging/tests"
    if staging_tests.is_dir():
        test_files.extend(staging_tests.rglob("*.py"))
    this_file = Path(__file__).resolve()
    other_test_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in test_files
        if path.resolve() != this_file
    )
    assert "SUPERSEDED_NOT_LOADABLE" not in other_test_text
    assert "LEGACY_B0_INVALIDATION_MANIFEST" not in other_test_text
    assert audit["legacy_b0_active_loader_negative_test_evidence_present"] is False
    assert audit["text_reference_inventory"]["legacy_b0_active_loader_negative_test_files"] == []


def test_release_documents_report_the_conflict_without_claiming_migration() -> None:
    audit = _load(AUDIT)
    code = _load(CODE_AVAILABILITY)
    release = _load(RELEASE_CANDIDATE)
    draft = " ".join(DRAFT.read_text(encoding="utf-8").split())

    assert audit["current_head_formal_release_payload_boundary_compliant"] is False
    assert code["repository_facts"]["formal_release_payload_boundary_compliant"] is False
    assert code["repository_facts"]["legacy_payload_migration_authorized"] is False
    policy = release["tracked_data_policy"]
    assert policy["legacy_b0_direct_reader_entrypoint_count"] == 4
    assert policy["legacy_b0_active_loader_negative_test_evidence_present"] is False
    assert policy["automatic_removal_performed"] is False
    assert "five files total 34,786,075 bytes" in draft
    assert "no corresponding negative-loader test was found" in draft
    assert "not eligible for a formal release" in draft
    assert all(value is False for value in audit["protected_outcomes"].values())


def test_recommendation_is_fail_close_then_migrate_and_requires_authorization() -> None:
    audit = _load(AUDIT)
    recommendation = audit["recommended_disposition"]
    memo = MEMO.read_text(encoding="utf-8")

    assert recommendation["decision"] == (
        "AFTER_EXPLICIT_USER_AUTHORIZATION_FAIL_CLOSE_LEGACY_READERS_THEN_"
        "MIGRATE_FIVE_PAYLOADS_OUT_OF_CURRENT_HEAD"
    )
    assert len(recommendation["ordered_actions_not_yet_executed"]) == 5
    assert recommendation["shared_git_history_rewrite_recommended_in_this_task"] is False
    assert recommendation["shared_git_history_rewrite_requires_separate_explicit_authorization"] is True
    assert recommendation["formal_release_or_tag_before_resolution_authorized"] is False
    assert "Do not create a formal tag or GitHub Release from the current HEAD" in memo
    assert "This memo authorizes no deletion, move, copy, history rewrite, release or tag" in memo
