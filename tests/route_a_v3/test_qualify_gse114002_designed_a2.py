from __future__ import annotations

import csv
import errno
import gzip
import hashlib
import importlib.util
import io
import json
import os
import stat
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts" / "route_a_v3" / "qualify_gse114002_designed_a2.py"
PROTOCOL_PATH = ROOT / "configs" / "route_a_v3_gse114002_a2_qualification.json"
SPEC = importlib.util.spec_from_file_location("qualify_gse114002_designed_a2", MODULE_PATH)
assert SPEC and SPEC.loader
QUALIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(QUALIFY)


def _counts_with_mrl(weight: int) -> tuple[list[str], str]:
    counts = [0] * 14
    counts[weight] = 10
    return [str(value) for value in counts], str(float(weight))


def _row(
    *,
    utr: str,
    mother: str,
    designed: bool,
    row_id: str,
    library: str = "human_utrs",
    mrl_weight: int = 4,
    info4: str = "NOT_A_CONTEXT",
) -> list[str]:
    values = [""] * len(QUALIFY.EXPECTED_HEADER)
    index = {name: position for position, name in enumerate(QUALIFY.EXPECTED_HEADER)}
    counts, rl = _counts_with_mrl(mrl_weight)
    values[0] = row_id
    values[index["utr"]] = utr
    for column, count in zip(QUALIFY.FRACTION_COLUMNS, counts):
        values[index[column]] = count
        values[index["r" + column]] = str(float(count) / 10.0)
    values[index["total"]] = "10"
    values[index["r_total"]] = "1"
    values[index["rl"]] = rl
    values[index["id"]] = row_id
    values[index["info1"]] = "opaque-info-one"
    values[index["info2"]] = "opaque-info-two"
    values[index["info3"]] = "opaque-info-three"
    values[index["info4"]] = info4
    values[index["library"]] = library
    values[index["mother"]] = mother
    values[index["designed"]] = "True" if designed else "False"
    values[index["match_score"]] = "1.0"
    return values


def _fixture_rows(*, duplicate_id: bool = False) -> list[list[str]]:
    mother = "AACCGGTTAACCGGTTAACCGGTTAA"
    edited = [
        "CACCGGTTAACCGGTTAACCGGTTAA",
        "AGCCGGTTAACCGGTTAACCGGTTAA",
        "AATCGGTTAACCGGTTAACCGGTTAA",
    ]
    rows = [
        _row(
            utr=mother,
            mother=mother,
            designed=True,
            row_id="raw-row-secret-anchor-7f3a",
        ),
    ]
    for position, candidate in enumerate(edited):
        rows.append(
            _row(
                utr=candidate,
                mother=mother,
                designed=False,
                row_id=(
                    "raw-row-secret-anchor-7f3a"
                    if duplicate_id and position == 0
                    else f"raw-row-secret-edited-{position}-7f3a"
                ),
            )
        )
    return rows


def _gzip_csv(rows: list[list[str]], *, header: tuple[str, ...] | None = None) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header or QUALIFY.EXPECTED_HEADER)
    writer.writerows(rows)
    return gzip.compress(buffer.getvalue().encode("utf-8"), mtime=0)


def _production_protocol() -> dict[str, object]:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def _fixture_protocol(payload: bytes) -> dict[str, object]:
    protocol = _production_protocol()
    # Synthetic integration fixtures live outside a Git worktree.  Keep their
    # implementation state explicitly unresolved; the dedicated temporary-Git
    # tests below exercise the fully bound production branch.
    protocol["authority"]["implementation_commit"] = "UNKNOWN_NOT_ASSERTED"  # type: ignore[index]
    blockers = protocol["known_blockers"]  # type: ignore[assignment]
    if QUALIFY.IMPLEMENTATION_BLOCKER not in blockers:
        blockers.append(QUALIFY.IMPLEMENTATION_BLOCKER)
    source = protocol["ordinary_public_asset_allowlist"][0]  # type: ignore[index]
    source["compressed_sha256"] = hashlib.sha256(payload).hexdigest()
    source["compressed_bytes"] = len(payload)
    pool = protocol["provisional_pool_rule"]  # type: ignore[assignment]
    pool["expected_pool_count"] = 1
    pool["expected_distinct_candidate_count"] = 3
    return protocol


def _write_integration_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, duplicate_id: bool = False
) -> tuple[Path, Path, bytes, dict[str, object]]:
    payload = _gzip_csv(_fixture_rows(duplicate_id=duplicate_id))
    protocol = _fixture_protocol(payload)
    protocol_path = tmp_path / "route_a_v3_gse114002_a2_qualification.json"
    protocol_path.write_text(json.dumps(protocol, sort_keys=True) + "\n", encoding="utf-8")
    source = tmp_path / QUALIFY.SOURCE_BASENAME
    source.write_bytes(payload)
    monkeypatch.setattr(QUALIFY, "EXPECTED_SOURCE_SHA256", hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(QUALIFY, "EXPECTED_SOURCE_BYTES", len(payload))
    monkeypatch.setattr(QUALIFY, "EXPECTED_POOL_COUNT", 1)
    monkeypatch.setattr(QUALIFY, "EXPECTED_CANDIDATE_COUNT", 3)
    return protocol_path, source, payload, protocol


def _read_bundle(output: Path) -> dict[str, object]:
    return {
        path.name: json.loads(path.read_text(encoding="utf-8"))
        for path in output.iterdir()
        if path.suffix == ".json"
    }


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _git_binding_fixture(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "Route A Test")
    _git(repo, "config", "user.email", "route-a@example.invalid")
    (repo / "README.md").write_text("accepted\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-q", "-m", "accepted")
    accepted = _git(repo, "rev-parse", "HEAD")

    contract = repo / "docs" / "goals" / "contract.md"
    registry = repo / "docs" / "execution" / "registry.yaml"
    contract.parent.mkdir(parents=True)
    registry.parent.mkdir(parents=True)
    contract.write_text("contract authority\n", encoding="utf-8")
    registry.write_text("registry authority\n", encoding="utf-8")
    _git(repo, "add", "docs")
    _git(repo, "commit", "-q", "-m", "authority")
    active = _git(repo, "rev-parse", "HEAD")

    qualifier = repo / "scripts" / "qualifier.py"
    focused_test = repo / "tests" / "test_qualifier.py"
    qualifier.parent.mkdir()
    focused_test.parent.mkdir()
    qualifier.write_text("print('qualifier')\n", encoding="utf-8")
    focused_test.write_text("def test_bound(): pass\n", encoding="utf-8")
    _git(repo, "add", "scripts", "tests")
    _git(repo, "commit", "-q", "-m", "implementation")
    implementation = _git(repo, "rev-parse", "HEAD")
    protocol = {
        "authority": {
            "accepted_a0_base_commit": accepted,
            "active_authority_commit": active,
            "implementation_commit": implementation,
            "contract_path": "docs/goals/contract.md",
            "contract_sha256": _file_sha256(contract),
            "data_role_registry_path": "docs/execution/registry.yaml",
            "data_role_registry_sha256": _file_sha256(registry),
            "qualifier_path": "scripts/qualifier.py",
            "qualifier_sha256": _file_sha256(qualifier),
            "focused_test_path": "tests/test_qualifier.py",
            "focused_test_sha256": _file_sha256(focused_test),
        }
    }
    return repo, protocol


def test_production_protocol_is_immutable_blocked_true_a2_candidate() -> None:
    protocol = _production_protocol()
    QUALIFY._validate_protocol(protocol)
    source = protocol["ordinary_public_asset_allowlist"][0]
    assert source["compressed_sha256"] == (
        "b72ac298cb0f4d21f911d330c0def06f8d94f15d9f8cc22f3a50ae87a7ef7ee5"
    )
    assert source["compressed_bytes"] == 17_332_142
    assert tuple(source["exact_header"]) == QUALIFY.EXPECTED_HEADER
    assert protocol["data_role"] == "A2_RECOVERY_CANDIDATE_NOT_QUALIFIED"
    assert protocol["scope"]["training_allowed"] is False
    assert protocol["claim_boundary"]["canonical_record_count"] == 0
    implementation_commit = protocol["authority"]["implementation_commit"]
    if implementation_commit == "UNKNOWN_NOT_ASSERTED":
        assert QUALIFY.IMPLEMENTATION_BLOCKER in protocol["known_blockers"]
    else:
        assert QUALIFY.COMMIT_RE.fullmatch(implementation_commit)
        assert QUALIFY.IMPLEMENTATION_BLOCKER not in protocol["known_blockers"]
    assert protocol["authority"]["qualifier_sha256"] == _file_sha256(MODULE_PATH)
    assert protocol["authority"]["focused_test_sha256"] == _file_sha256(Path(__file__))
    output_contract = protocol["output_contract"]
    assert output_contract["output_id"] == QUALIFY.OUTPUT_ID
    assert output_contract["primary_publication_mode"] == (
        QUALIFY.PRIMARY_PUBLICATION_MODE
    )
    assert output_contract["fallback_publication_mode"] == (
        QUALIFY.FALLBACK_PUBLICATION_MODE
    )
    assert output_contract["atomic_no_replace_unsupported_errno_fallback"] == list(
        QUALIFY.ATOMIC_NOREPLACE_UNSUPPORTED_ERRNO_NAMES
    )
    assert output_contract["required_terminal_metadata_files_all_publication_modes"] == [
        QUALIFY.PUBLICATION_COMMIT_FILENAME
    ]
    assert output_contract["commit_marker_written_last"] is True
    assert output_contract[
        "commit_marker_validation_required_before_published_return"
    ] is True
    assert output_contract["fallback_pre_marker_failure_status"] == (
        "PARTIAL_NOT_COMMITTED"
    )
    assert output_contract["fallback_post_marker_validation_failure_status"] == (
        "COMMITTED_NOT_ACCEPTED"
    )


def test_git_binding_enforces_full_ancestry_blobs_head_and_clean_worktree(
    tmp_path: Path,
) -> None:
    repo, protocol = _git_binding_fixture(tmp_path)
    result = QUALIFY._verify_git_binding(protocol, repo)
    assert result["status"] == "PASS"
    assert result["accepted_a0_is_ancestor_of_active_authority"] is True
    assert result["active_authority_is_ancestor_of_implementation"] is True
    assert result["implementation_is_ancestor_of_head"] is True
    assert result["active_authority_file_hashes_match"] is True
    assert result["implementation_file_hashes_match"] is True
    assert result["worktree_clean"] is True


def test_git_binding_rejects_wrong_authority_blob(tmp_path: Path) -> None:
    repo, protocol = _git_binding_fixture(tmp_path)
    protocol["authority"]["contract_sha256"] = "0" * 64
    with pytest.raises(QUALIFY.QualificationError, match="active authority file hash"):
        QUALIFY._verify_git_binding(protocol, repo)


def test_git_binding_rejects_dirty_worktree(tmp_path: Path) -> None:
    repo, protocol = _git_binding_fixture(tmp_path)
    (repo / "README.md").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(QUALIFY.QualificationError, match="worktree is not clean"):
        QUALIFY._verify_git_binding(protocol, repo)


def test_git_binding_rejects_active_to_implementation_ancestry_break(
    tmp_path: Path,
) -> None:
    repo, protocol = _git_binding_fixture(tmp_path)
    accepted = protocol["authority"]["accepted_a0_base_commit"]
    _git(repo, "checkout", "-q", "-b", "sibling", accepted)
    (repo / "sibling.txt").write_text("sibling\n", encoding="utf-8")
    _git(repo, "add", "sibling.txt")
    _git(repo, "commit", "-q", "-m", "sibling")
    protocol["authority"]["implementation_commit"] = _git(repo, "rev-parse", "HEAD")
    with pytest.raises(QUALIFY.QualificationError, match="active authority to implementation"):
        QUALIFY._verify_git_binding(protocol, repo)


def test_identity_anchor_and_minimum_three_edited_pool_rule() -> None:
    protocol = _production_protocol()
    pool, _, blockers = QUALIFY._analyze_verified_source(
        _gzip_csv(_fixture_rows()), protocol
    )
    assert pool["provisional_source_pool_count"] == 1
    assert pool["eligible_provisional_pool_count"] == 1
    assert pool["eligible_provisional_distinct_candidate_count"] == 3
    assert pool["pool_with_nonunique_or_missing_identity_count"] == 0
    assert pool["pool_with_fewer_than_three_distinct_edited_count"] == 0
    assert "MOTHER_INFO1_INFO2_INFO3_MATCH_SCORE_ID_AUTHORITY_UNKNOWN_NOT_ASSERTED" in blockers


@pytest.mark.parametrize(
    ("rows", "expected_identity_bad", "expected_min_bad"),
    [
        (_fixture_rows()[1:], 1, 0),
        (_fixture_rows()[:3], 0, 1),
        (_fixture_rows() + [_fixture_rows()[0]], 1, 0),
    ],
)
def test_pool_rule_fails_closed_without_unique_identity_or_three_edited(
    rows: list[list[str]], expected_identity_bad: int, expected_min_bad: int
) -> None:
    pool, _, _ = QUALIFY._analyze_verified_source(
        _gzip_csv(rows), _production_protocol()
    )
    assert pool["eligible_provisional_pool_count"] == 0
    assert pool["pool_with_nonunique_or_missing_identity_count"] == expected_identity_bad
    assert pool["pool_with_fewer_than_three_distinct_edited_count"] == expected_min_bad


def test_duplicate_id_is_a_blocker_even_when_pool_geometry_is_mechanical() -> None:
    pool, _, blockers = QUALIFY._analyze_verified_source(
        _gzip_csv(_fixture_rows(duplicate_id=True)), _production_protocol()
    )
    assert pool["duplicate_id_value_count"] == 1
    assert "SOURCE_ID_NOT_GLOBALLY_UNIQUE" in blockers


def test_id_uniqueness_is_global_across_included_and_excluded_libraries() -> None:
    rows = _fixture_rows()
    mother = "TTGGCCAATTGGCCAATTGGCCAATT"
    rows.append(
        _row(
            utr=mother,
            mother=mother,
            designed=True,
            row_id="raw-row-secret-anchor-7f3a",
            library="excluded_control_library",
        )
    )
    pool, _, blockers = QUALIFY._analyze_verified_source(
        _gzip_csv(rows), _production_protocol()
    )
    assert pool["included_library_row_count"] == 4
    assert pool["duplicate_id_value_count"] == 1
    assert "SOURCE_ID_NOT_GLOBALLY_UNIQUE" in blockers


def test_info_fields_and_info4_never_close_context_authority() -> None:
    rows = _fixture_rows()
    index = {name: position for position, name in enumerate(QUALIFY.EXPECTED_HEADER)}
    for row in rows:
        row[index["info1"]] = "HEK293"
        row[index["info2"]] = "replicate-1"
        row[index["info3"]] = "paper-looking-token"
        row[index["info4"]] = "tempting-context-token"
    pool, _, blockers = QUALIFY._analyze_verified_source(
        _gzip_csv(rows), _production_protocol()
    )
    assert pool["info4_used_as_context"] is False
    assert pool["field_authority_status"] == "UNKNOWN_NOT_ASSERTED"
    assert "MOTHER_INFO1_INFO2_INFO3_MATCH_SCORE_ID_AUTHORITY_UNKNOWN_NOT_ASSERTED" in blockers


def test_fraction_count_mrl_weights_small_fixture() -> None:
    mean, technical_se = QUALIFY.derive_fraction_count_mrl_and_technical_se(
        [1, 1, 2], [0, 2, 4]
    )
    assert mean == pytest.approx(2.5)
    assert technical_se == pytest.approx((2.75 / 4) ** 0.5)
    with pytest.raises(QUALIFY.QualificationError, match="positive total"):
        QUALIFY.derive_fraction_count_mrl_and_technical_se([0, 0], [0, 1])


def test_technical_fraction_uncertainty_is_never_biological_se() -> None:
    _, uncertainty, blockers = QUALIFY._analyze_verified_source(
        _gzip_csv(_fixture_rows()), _production_protocol()
    )
    assert uncertainty["technical_fraction_uncertainty_derived_row_count"] == 4
    assert uncertainty["biological_replicate_status"] == "ABSENT_BY_DESIGN"
    assert uncertainty["paper_standard_error_status"] == "ABSENT"
    assert uncertainty["biological_standard_error_derived"] is False
    assert "OWNER_TECHNICAL_UNCERTAINTY_POLICY_UNKNOWN_NOT_ASSERTED" in blockers


def test_split_assignment_is_outcome_independent() -> None:
    key = QUALIFY.provisional_source_key(
        "AACCGG", "human_utrs", assay_id="assay", context_id="context",
        endpoint_id="mrl",
    )
    first = QUALIFY.outcome_independent_partition(key, "human_utrs", "assay", "context")
    # The API deliberately accepts no MRL, delta, significance, or candidate outcome.
    second = QUALIFY.outcome_independent_partition(key, "human_utrs", "assay", "context")
    assert first == second
    assert first in {"DEVELOPMENT_A", "DEVELOPMENT_B"}


@pytest.mark.parametrize(
    ("pool_count", "candidate_count", "expected_status"),
    [
        (959, 3899, "PROVISIONAL_COUNTS_MATCH_NOT_AUTHORITY"),
        (958, 3899, "PROVISIONAL_COUNTS_MISMATCH_BLOCKED"),
        (960, 3899, "PROVISIONAL_COUNTS_MISMATCH_BLOCKED"),
        (959, 3898, "PROVISIONAL_COUNTS_MISMATCH_BLOCKED"),
        (959, 3900, "PROVISIONAL_COUNTS_MISMATCH_BLOCKED"),
    ],
)
def test_production_959_3899_reconciliation_expectations_fail_closed_at_plus_minus_one(
    pool_count: int, candidate_count: int, expected_status: str
) -> None:
    audit = QUALIFY.reconcile_provisional_geometry(pool_count, candidate_count)
    assert audit["status"] == expected_status
    assert audit["expected_counts_are_authority"] is False


def test_unknown_gates_publish_aggregate_blocked_bundle_with_zero_canonical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol_path, source, _, _ = _write_integration_fixture(tmp_path, monkeypatch)
    output = tmp_path / "blocked-bundle"
    staging_write_order: list[str] = []
    original_write = QUALIFY._write_exclusive

    def tracked_write(
        path: Path, payload: bytes, *, directory_descriptor: int | None = None
    ) -> None:
        if path.parent.name.startswith(f".{output.name}.staging-"):
            staging_write_order.append(path.name)
        original_write(
            path, payload, directory_descriptor=directory_descriptor
        )

    monkeypatch.setattr(QUALIFY, "_write_exclusive", tracked_write)
    result = QUALIFY.qualify_gse114002_designed_a2(
        protocol_path=protocol_path,
        protocol_sha256=_file_sha256(protocol_path),
        source_path=source,
        output_directory=output,
    )
    bundle = _read_bundle(output)
    report = bundle["QUALIFICATION_REPORT.json"]
    assert result["published"] is True
    assert result["canonical_record_count"] == 0
    assert report["status"] == "BLOCKED_NOT_QUALIFIED"
    assert report["true_a2_status"] == "NOT_QUALIFIED"
    assert report["training_allowed"] is False
    assert report["canonical_materialization_allowed"] is False
    assert report["canonical_record_count"] == 0
    assert set(path.name for path in output.iterdir()) == (
        set(QUALIFY.ALWAYS_OUTPUT_FILES) | {QUALIFY.PUBLICATION_COMMIT_FILENAME}
    )
    marker = bundle[QUALIFY.PUBLICATION_COMMIT_FILENAME]
    assert result["publication_mode"] == QUALIFY.PRIMARY_PUBLICATION_MODE
    assert result["terminal_commit_marker_validated"] is True
    assert marker["protocol_id"] == QUALIFY.PROTOCOL_ID
    assert marker["output_id"] == QUALIFY.OUTPUT_ID
    assert marker["publication_mode"] == QUALIFY.PRIMARY_PUBLICATION_MODE
    assert marker["bundle_member_names"] == sorted(QUALIFY.ALWAYS_OUTPUT_FILES)
    assert marker["bundle_file_count_excluding_commit_marker"] == len(
        QUALIFY.ALWAYS_OUTPUT_FILES
    )
    assert marker["final_output_directory_name_sha256"] == (
        QUALIFY._final_output_directory_name_sha256(output.name)
    )
    assert marker["final_output_target_sha256"] == (
        QUALIFY._final_output_target_sha256(output)
    )
    assert marker["committed"] is True
    assert staging_write_order[-1] == QUALIFY.PUBLICATION_COMMIT_FILENAME
    assert staging_write_order.count(QUALIFY.PUBLICATION_COMMIT_FILENAME) == 1
    QUALIFY._validate_publication_commit(
        output, expected_publication_mode=QUALIFY.PRIMARY_PUBLICATION_MODE
    )
    assert not (output / "canonical_intervention_records.jsonl").exists()


def test_launch_protocol_hash_mismatch_fails_before_source_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol_path, source, _, _ = _write_integration_fixture(tmp_path, monkeypatch)
    observed_reads: list[Path] = []
    original = QUALIFY._read_regular_verified_snapshot

    def track(path: Path, **kwargs: object) -> object:
        observed_reads.append(Path(path))
        return original(path, **kwargs)

    monkeypatch.setattr(QUALIFY, "_read_regular_verified_snapshot", track)
    with pytest.raises(QUALIFY.QualificationError, match="protocol.*SHA-256 mismatch"):
        QUALIFY.qualify_gse114002_designed_a2(
            protocol_path=protocol_path,
            protocol_sha256="0" * 64,
            source_path=source,
            output_directory=tmp_path / "must-not-exist",
        )
    assert observed_reads == [protocol_path.resolve()]
    assert not (tmp_path / "must-not-exist").exists()


@pytest.mark.parametrize(
    "mutation", ["extra_root_key", "freeform_blocker", "critical_string_injection"]
)
def test_protocol_shape_and_blocker_enum_reject_injection_before_source_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    protocol_path, source, _, protocol = _write_integration_fixture(tmp_path, monkeypatch)
    if mutation == "extra_root_key":
        protocol["injected"] = "AACCGGTTAACCGGTTAACCGGTTAA"
    elif mutation == "critical_string_injection":
        protocol["scope"]["context_id"] = "AACCGGTTAACCGGTTAACCGGTTAA"
    else:
        protocol["known_blockers"].append("AACCGGTTAACCGGTTAACCGGTTAA")
    protocol_path.write_text(json.dumps(protocol, sort_keys=True) + "\n", encoding="utf-8")
    source_reads = 0
    original = QUALIFY._read_regular_verified_snapshot

    def track(path: Path, **kwargs: object) -> object:
        nonlocal source_reads
        if Path(path) == source.resolve():
            source_reads += 1
        return original(path, **kwargs)

    monkeypatch.setattr(QUALIFY, "_read_regular_verified_snapshot", track)
    with pytest.raises(QUALIFY.QualificationError):
        QUALIFY.qualify_gse114002_designed_a2(
            protocol_path=protocol_path,
            protocol_sha256=_file_sha256(protocol_path),
            source_path=source,
            output_directory=tmp_path / "must-not-exist",
        )
    assert source_reads == 0
    assert not (tmp_path / "must-not-exist").exists()


def test_blocked_bundle_never_leaks_sequence_or_raw_source_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol_path, source, _, _ = _write_integration_fixture(tmp_path, monkeypatch)
    output = tmp_path / "aggregate-only"
    QUALIFY.qualify_gse114002_designed_a2(
        protocol_path=protocol_path,
        protocol_sha256=_file_sha256(protocol_path),
        source_path=source,
        output_directory=output,
    )
    combined = b"\n".join(path.read_bytes() for path in sorted(output.iterdir()))
    assert b"AACCGGTTAACCGGTTAACCGGTTAA" not in combined
    for secret in (
        b"opaque-info-one", b"opaque-info-two", b"opaque-info-three",
        b"NOT_A_CONTEXT", b"raw-row-secret-anchor-7f3a",
        b"raw-row-secret-edited-0-7f3a",
    ):
        assert secret not in combined
    for document in _read_bundle(output).values():
        QUALIFY._assert_aggregate_safe(document)


def test_forbidden_scope_is_rejected_before_any_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden_read(*args: object, **kwargs: object) -> object:
        raise AssertionError("payload read happened before scope rejection")

    monkeypatch.setattr(QUALIFY, "_read_regular_verified_snapshot", forbidden_read)
    with pytest.raises(QUALIFY.ScopeViolation, match="rejected before read"):
        QUALIFY.qualify_gse114002_designed_a2(
            protocol_path=tmp_path / "restricted" / PROTOCOL_PATH.name,
            protocol_sha256="0" * 64,
            source_path=tmp_path / QUALIFY.SOURCE_BASENAME,
            output_directory=tmp_path / "out",
        )


def test_verified_snapshot_accepts_descriptor_bytes_despite_stale_preopen_lstat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "ordinary-public.bin"
    payload = b"descriptor-bytes-are-the-trust-root\n"
    source.write_bytes(payload)
    observed = source.stat()
    stale = SimpleNamespace(
        st_mode=stat.S_IFREG | 0o640,
        st_dev=observed.st_dev,
        st_ino=observed.st_ino + 101,
        st_size=observed.st_size + 17,
        st_mtime_ns=observed.st_mtime_ns - 1,
        st_ctime_ns=observed.st_ctime_ns - 1,
    )
    original_lstat = Path.lstat

    def stale_lstat(path: Path) -> object:
        if path == source:
            return stale
        return original_lstat(path)

    monkeypatch.setattr(Path, "lstat", stale_lstat)
    captured, provenance = QUALIFY._read_regular_verified_snapshot(
        source,
        label="ordinary public stale-lstat fixture",
        expected_sha256=hashlib.sha256(payload).hexdigest(),
        expected_bytes=len(payload),
    )
    assert captured == payload
    assert provenance["sha256"] == hashlib.sha256(payload).hexdigest()
    assert provenance["bytes"] == len(payload)


@pytest.mark.parametrize("kind", ["symlink", "directory"])
def test_verified_snapshot_rejects_symlink_and_nonregular_targets(
    tmp_path: Path, kind: str
) -> None:
    target = tmp_path / "target.bin"
    target.write_bytes(b"ordinary-public\n")
    candidate = tmp_path / "candidate.bin"
    if kind == "symlink":
        candidate.symlink_to(target)
    else:
        candidate.mkdir()
    with pytest.raises(QUALIFY.QualificationError):
        QUALIFY._read_regular_verified_snapshot(
            candidate, label=f"{kind} fixture"
        )


def test_verified_snapshot_rejects_intermediate_symlink_before_payload_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    target = real_parent / "ordinary-public.bin"
    target.write_bytes(b"must-not-be-read-through-an-intermediate-symlink\n")
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)

    def forbidden_payload_read(*_: object) -> bytes:
        raise AssertionError("payload read occurred through an intermediate symlink")

    monkeypatch.setattr(QUALIFY.os, "read", forbidden_payload_read)
    with pytest.raises(QUALIFY.QualificationError, match="parent path contains a symlink"):
        QUALIFY._read_regular_verified_snapshot(
            linked_parent / target.name,
            label="intermediate-symlink ordinary-public fixture",
        )


def test_verified_snapshot_rejects_same_descriptor_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "mutating.bin"
    source.write_bytes(b"A" * (2 << 20))
    original_read = QUALIFY.os.read
    mutated = False

    def read_then_mutate(descriptor: int, amount: int) -> bytes:
        nonlocal mutated
        block = original_read(descriptor, amount)
        if block and not mutated:
            mutated = True
            with source.open("ab") as handle:
                handle.write(b"MUTATION")
                handle.flush()
                os.fsync(handle.fileno())
        return block

    monkeypatch.setattr(QUALIFY.os, "read", read_then_mutate)
    with pytest.raises(QUALIFY.QualificationError, match="changed during descriptor"):
        QUALIFY._read_regular_verified_snapshot(
            source, label="mutating ordinary-public fixture"
        )


def test_verified_compressed_snapshot_closes_path_replacement_toctou(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol_path, source, original, _ = _write_integration_fixture(tmp_path, monkeypatch)
    replacement = _gzip_csv(_fixture_rows()[:1])

    def replace_original_path() -> None:
        source.rename(tmp_path / "captured-original.csv.gz")
        source.write_bytes(replacement)

    monkeypatch.setattr(QUALIFY, "_POST_VERIFIED_SNAPSHOT_HOOK", replace_original_path)
    output = tmp_path / "toctou-closed"
    QUALIFY.qualify_gse114002_designed_a2(
        protocol_path=protocol_path,
        protocol_sha256=_file_sha256(protocol_path),
        source_path=source,
        output_directory=output,
    )
    pool = json.loads((output / "POOL_GEOMETRY_AUDIT.json").read_text(encoding="utf-8"))
    assert pool["source_row_count"] == 4
    assert pool["eligible_provisional_pool_count"] == 1
    assert (tmp_path / "captured-original.csv.gz").read_bytes() == original
    assert source.read_bytes() == replacement


def test_atomic_publish_never_overwrites_existing_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol_path, source, _, _ = _write_integration_fixture(tmp_path, monkeypatch)
    output = tmp_path / "already-present"
    output.mkdir()
    sentinel = output / "sentinel.txt"
    sentinel.write_bytes(b"preserve-me")
    result = QUALIFY.qualify_gse114002_designed_a2(
        protocol_path=protocol_path,
        protocol_sha256=_file_sha256(protocol_path),
        source_path=source,
        output_directory=output,
    )
    assert result["kind"] == "CONTENDED"
    assert result["published"] is False
    assert result["contention_status"] == "ATOMIC_NO_REPLACE_CONTENDED"
    assert sentinel.read_bytes() == b"preserve-me"
    assert list(output.iterdir()) == [sentinel]


def test_atomic_publish_race_is_explicitly_contended(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol_path, source, _, _ = _write_integration_fixture(tmp_path, monkeypatch)
    output = tmp_path / "race-target"
    original = QUALIFY._rename_directory_noreplace

    def competing_writer(source_dir: Path, target: Path, parent_fd: int) -> None:
        target.mkdir()
        (target / "winner.txt").write_text("winner\n", encoding="utf-8")
        original(source_dir, target, parent_fd)

    monkeypatch.setattr(QUALIFY, "_rename_directory_noreplace", competing_writer)
    result = QUALIFY.qualify_gse114002_designed_a2(
        protocol_path=protocol_path,
        protocol_sha256=_file_sha256(protocol_path),
        source_path=source,
        output_directory=output,
    )
    assert result["kind"] == "CONTENDED"
    assert result["published"] is False
    assert result["status"] == "CONTENDED"
    assert (output / "winner.txt").read_text(encoding="utf-8") == "winner\n"


def test_primary_namespace_replacement_after_rename_is_not_accepted_as_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol_path, source, _, _ = _write_integration_fixture(tmp_path, monkeypatch)
    output = tmp_path / "primary-namespace-target"
    displaced = tmp_path / "primary-namespace-displaced"
    original_validate = QUALIFY._validate_publication_commit
    replaced = False

    def replace_final_namespace_before_descriptor_validation(
        output_directory: Path, **kwargs: object
    ) -> dict[str, object]:
        nonlocal replaced
        if output_directory == output and not replaced:
            replaced = True
            output.rename(displaced)
            output.mkdir()
        return original_validate(output_directory, **kwargs)

    monkeypatch.setattr(
        QUALIFY,
        "_validate_publication_commit",
        replace_final_namespace_before_descriptor_validation,
    )
    with pytest.raises(
        QUALIFY.CommittedPublicationValidationError,
        match="namespace entry",
    ):
        QUALIFY.qualify_gse114002_designed_a2(
            protocol_path=protocol_path,
            protocol_sha256=_file_sha256(protocol_path),
            source_path=source,
            output_directory=output,
        )
    assert replaced is True
    assert output.is_dir()
    assert list(output.iterdir()) == []
    assert (displaced / QUALIFY.PUBLICATION_COMMIT_FILENAME).is_file()


def test_primary_parent_path_replacement_is_not_accepted_as_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol_path, source, _, _ = _write_integration_fixture(tmp_path, monkeypatch)
    output = tmp_path / "primary-parent-target"
    displaced_parent = tmp_path.parent / f"{tmp_path.name}-displaced-parent"
    original_identity_check = QUALIFY._require_directory_entry_identity
    replaced = False

    def replace_parent_before_final_identity_check(**kwargs: object) -> None:
        nonlocal replaced
        if kwargs.get("label") == "published final output" and not replaced:
            replaced = True
            tmp_path.rename(displaced_parent)
            tmp_path.mkdir()
            (tmp_path / output.name).mkdir()
        original_identity_check(**kwargs)

    monkeypatch.setattr(
        QUALIFY,
        "_require_directory_entry_identity",
        replace_parent_before_final_identity_check,
    )
    with pytest.raises(
        QUALIFY.CommittedPublicationValidationError,
        match="absolute final output namespace entry",
    ):
        QUALIFY.qualify_gse114002_designed_a2(
            protocol_path=protocol_path,
            protocol_sha256=_file_sha256(protocol_path),
            source_path=source,
            output_directory=output,
        )
    assert replaced is True
    assert output.is_dir()
    assert list(output.iterdir()) == []
    displaced_output = displaced_parent / output.name
    assert (displaced_output / QUALIFY.PUBLICATION_COMMIT_FILENAME).is_file()


def test_explicitly_unsupported_rename_uses_exact_terminal_marker_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol_path, source, _, _ = _write_integration_fixture(tmp_path, monkeypatch)
    output = tmp_path / "fallback-success"
    write_order: list[str] = []
    original_write = QUALIFY._write_exclusive

    def unsupported_rename(*_: object) -> None:
        raise QUALIFY.AtomicNoReplaceUnsupported(errno.EOPNOTSUPP)

    def tracked_write(
        path: Path, payload: bytes, *, directory_descriptor: int | None = None
    ) -> None:
        if path.parent == output:
            write_order.append(path.name)
        original_write(
            path, payload, directory_descriptor=directory_descriptor
        )

    monkeypatch.setattr(QUALIFY, "_rename_directory_noreplace", unsupported_rename)
    monkeypatch.setattr(QUALIFY, "_write_exclusive", tracked_write)
    result = QUALIFY.qualify_gse114002_designed_a2(
        protocol_path=protocol_path,
        protocol_sha256=_file_sha256(protocol_path),
        source_path=source,
        output_directory=output,
    )
    assert result["kind"] == "PUBLISHED"
    assert result["published"] is True
    assert result["publication_mode"] == QUALIFY.FALLBACK_PUBLICATION_MODE
    assert result["atomic_no_replace"] is False
    assert result["atomic_exclusive_final_mkdir"] is True
    assert result["terminal_commit_marker_validated"] is True
    assert write_order[-1] == QUALIFY.PUBLICATION_COMMIT_FILENAME
    assert write_order.count(QUALIFY.PUBLICATION_COMMIT_FILENAME) == 1
    assert set(path.name for path in output.iterdir()) == (
        set(QUALIFY.ALWAYS_OUTPUT_FILES) | {QUALIFY.PUBLICATION_COMMIT_FILENAME}
    )
    marker = json.loads(
        (output / QUALIFY.PUBLICATION_COMMIT_FILENAME).read_text(encoding="utf-8")
    )
    assert marker["protocol_id"] == QUALIFY.PROTOCOL_ID
    assert marker["output_id"] == QUALIFY.OUTPUT_ID
    assert marker["publication_mode"] == QUALIFY.FALLBACK_PUBLICATION_MODE
    assert marker["bundle_file_count_excluding_commit_marker"] == len(
        QUALIFY.ALWAYS_OUTPUT_FILES
    )
    assert marker["bundle_member_names"] == sorted(QUALIFY.ALWAYS_OUTPUT_FILES)
    assert marker["final_output_directory_name_sha256"] == (
        QUALIFY._final_output_directory_name_sha256(output.name)
    )
    assert marker["final_output_target_sha256"] == (
        QUALIFY._final_output_target_sha256(output)
    )
    assert marker["sha256sums_sha256"] == hashlib.sha256(
        (output / "SHA256SUMS").read_bytes()
    ).hexdigest()
    assert marker["committed"] is True
    QUALIFY._validate_publication_commit(
        output, expected_publication_mode=QUALIFY.FALLBACK_PUBLICATION_MODE
    )
    retained_staging = list(tmp_path.glob(f".{output.name}.staging-*"))
    assert len(retained_staging) == 1
    with pytest.raises(
        QUALIFY.QualificationError, match="final output directory-name hash"
    ):
        QUALIFY._validate_publication_commit(
            retained_staging[0],
            expected_publication_mode=QUALIFY.PRIMARY_PUBLICATION_MODE,
        )


def test_fallback_midwrite_failure_preserves_uncommitted_directory_without_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol_path, source, _, _ = _write_integration_fixture(tmp_path, monkeypatch)
    output = tmp_path / "fallback-midwrite"
    original_write = QUALIFY._write_exclusive

    def unsupported_rename(*_: object) -> None:
        raise QUALIFY.AtomicNoReplaceUnsupported(errno.EINVAL)

    def fail_second_member(
        path: Path, payload: bytes, *, directory_descriptor: int | None = None
    ) -> None:
        if path.parent == output and path.name == "POOL_GEOMETRY_AUDIT.json":
            raise OSError(errno.EIO, "injected fallback member write failure")
        original_write(
            path, payload, directory_descriptor=directory_descriptor
        )

    monkeypatch.setattr(QUALIFY, "_rename_directory_noreplace", unsupported_rename)
    monkeypatch.setattr(QUALIFY, "_write_exclusive", fail_second_member)
    with pytest.raises(QUALIFY.PartialPublicationError, match="PARTIAL_NOT_COMMITTED"):
        QUALIFY.qualify_gse114002_designed_a2(
            protocol_path=protocol_path,
            protocol_sha256=_file_sha256(protocol_path),
            source_path=source,
            output_directory=output,
        )
    assert output.is_dir()
    assert (output / "MEASUREMENT_UNCERTAINTY_AUDIT.json").is_file()
    assert not (output / QUALIFY.PUBLICATION_COMMIT_FILENAME).exists()
    with pytest.raises(QUALIFY.QualificationError):
        QUALIFY._validate_publication_commit(
            output, expected_publication_mode=QUALIFY.FALLBACK_PUBLICATION_MODE
        )


def test_fallback_fsynced_marker_with_persistent_validation_failure_is_committed_not_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol_path, source, _, _ = _write_integration_fixture(tmp_path, monkeypatch)
    output = tmp_path / "fallback-committed-not-accepted"
    original_validate = QUALIFY._validate_publication_commit

    def unsupported_rename(*_: object) -> None:
        raise QUALIFY.AtomicNoReplaceUnsupported(errno.EINVAL)

    def persistent_final_validation_failure(
        output_directory: Path, **kwargs: object
    ) -> dict[str, object]:
        if output_directory == output:
            raise QUALIFY.QualificationError("injected persistent marker validation failure")
        return original_validate(output_directory, **kwargs)

    monkeypatch.setattr(QUALIFY, "_rename_directory_noreplace", unsupported_rename)
    monkeypatch.setattr(
        QUALIFY, "_validate_publication_commit", persistent_final_validation_failure
    )
    with pytest.raises(
        QUALIFY.CommittedPublicationValidationError,
        match="COMMITTED_NOT_ACCEPTED",
    ):
        QUALIFY.qualify_gse114002_designed_a2(
            protocol_path=protocol_path,
            protocol_sha256=_file_sha256(protocol_path),
            source_path=source,
            output_directory=output,
        )
    assert output.is_dir()
    assert (output / QUALIFY.PUBLICATION_COMMIT_FILENAME).is_file()


def test_fallback_namespace_replacement_cannot_redirect_descriptor_anchored_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol_path, source, _, _ = _write_integration_fixture(tmp_path, monkeypatch)
    output = tmp_path / "fallback-namespace-target"
    displaced = tmp_path / "fallback-namespace-displaced"
    original_write = QUALIFY._write_exclusive
    replaced = False

    def unsupported_rename(*_: object) -> None:
        raise QUALIFY.AtomicNoReplaceUnsupported(errno.EOPNOTSUPP)

    def replace_namespace_then_write(
        path: Path, payload: bytes, *, directory_descriptor: int | None = None
    ) -> None:
        nonlocal replaced
        if path.parent == output and not replaced:
            replaced = True
            output.rename(displaced)
            output.mkdir()
        original_write(
            path, payload, directory_descriptor=directory_descriptor
        )

    monkeypatch.setattr(QUALIFY, "_rename_directory_noreplace", unsupported_rename)
    monkeypatch.setattr(QUALIFY, "_write_exclusive", replace_namespace_then_write)
    with pytest.raises(
        QUALIFY.CommittedPublicationValidationError,
        match="namespace entry",
    ):
        QUALIFY.qualify_gse114002_designed_a2(
            protocol_path=protocol_path,
            protocol_sha256=_file_sha256(protocol_path),
            source_path=source,
            output_directory=output,
        )
    assert replaced is True
    assert output.is_dir()
    assert list(output.iterdir()) == []
    assert (displaced / QUALIFY.PUBLICATION_COMMIT_FILENAME).is_file()
    assert set(path.name for path in displaced.iterdir()) == (
        set(QUALIFY.ALWAYS_OUTPUT_FILES) | {QUALIFY.PUBLICATION_COMMIT_FILENAME}
    )


def test_fallback_atomic_mkdir_contention_is_explicit_and_preserves_winner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol_path, source, _, _ = _write_integration_fixture(tmp_path, monkeypatch)
    output = tmp_path / "fallback-contention"
    output.mkdir()
    sentinel = output / "winner.txt"
    sentinel.write_text("winner\n", encoding="utf-8")

    def unsupported_rename(*_: object) -> None:
        raise QUALIFY.AtomicNoReplaceUnsupported(errno.ENOSYS)

    monkeypatch.setattr(QUALIFY, "_rename_directory_noreplace", unsupported_rename)
    result = QUALIFY.qualify_gse114002_designed_a2(
        protocol_path=protocol_path,
        protocol_sha256=_file_sha256(protocol_path),
        source_path=source,
        output_directory=output,
    )
    assert result["kind"] == "CONTENDED"
    assert result["published"] is False
    assert result["publication_mode"] == QUALIFY.FALLBACK_PUBLICATION_MODE
    assert result["contention_status"] == "ATOMIC_FALLBACK_MKDIR_CONTENDED"
    assert list(output.iterdir()) == [sentinel]
    assert sentinel.read_text(encoding="utf-8") == "winner\n"


def test_nonunsupported_rename_error_never_enters_atomic_mkdir_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol_path, source, _, _ = _write_integration_fixture(tmp_path, monkeypatch)
    output = tmp_path / "no-fallback-for-eio"

    def hard_rename_failure(*_: object) -> None:
        raise QUALIFY.QualificationError("atomic no-replace failed with errno EIO")

    def forbidden_fallback(**_: object) -> dict[str, object]:
        raise AssertionError("non-unsupported rename error entered fallback")

    monkeypatch.setattr(QUALIFY, "_rename_directory_noreplace", hard_rename_failure)
    monkeypatch.setattr(
        QUALIFY, "_publish_bundle_with_terminal_marker", forbidden_fallback
    )
    with pytest.raises(QUALIFY.QualificationError, match="errno EIO"):
        QUALIFY.qualify_gse114002_designed_a2(
            protocol_path=protocol_path,
            protocol_sha256=_file_sha256(protocol_path),
            source_path=source,
            output_directory=output,
        )
    assert not output.exists()


def test_publication_marker_validation_rejects_member_hash_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol_path, source, _, _ = _write_integration_fixture(tmp_path, monkeypatch)
    output = tmp_path / "fallback-tamper"

    def unsupported_rename(*_: object) -> None:
        raise QUALIFY.AtomicNoReplaceUnsupported(errno.EOPNOTSUPP)

    monkeypatch.setattr(QUALIFY, "_rename_directory_noreplace", unsupported_rename)
    result = QUALIFY.qualify_gse114002_designed_a2(
        protocol_path=protocol_path,
        protocol_sha256=_file_sha256(protocol_path),
        source_path=source,
        output_directory=output,
    )
    assert result["published"] is True
    report_path = output / "QUALIFICATION_REPORT.json"
    report_path.write_bytes(report_path.read_bytes() + b" ")
    with pytest.raises(QUALIFY.QualificationError, match="member SHA-256 mismatch"):
        QUALIFY._validate_publication_commit(
            output, expected_publication_mode=QUALIFY.FALLBACK_PUBLICATION_MODE
        )


def test_fallback_capability_and_post_marker_fsync_failures_are_published_warnings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol_path, source, _, _ = _write_integration_fixture(tmp_path, monkeypatch)
    output = tmp_path / "fallback-fsync-warnings"
    original_fsync_directory = QUALIFY._fsync_directory

    def unsupported_rename(*_: object) -> None:
        raise QUALIFY.AtomicNoReplaceUnsupported(errno.EINVAL)

    def injected_directory_fsync(
        path: Path, *, directory_descriptor: int | None = None
    ) -> None:
        if path.name.startswith(f".{output.name}.staging-"):
            raise OSError(errno.EINVAL, "directory fsync unsupported")
        if path == output:
            assert (output / QUALIFY.PUBLICATION_COMMIT_FILENAME).is_file()
            raise OSError(errno.EIO, "post-marker output fsync failure")
        original_fsync_directory(
            path, directory_descriptor=directory_descriptor
        )

    monkeypatch.setattr(QUALIFY, "_rename_directory_noreplace", unsupported_rename)
    monkeypatch.setattr(QUALIFY, "_fsync_directory", injected_directory_fsync)
    result = QUALIFY.qualify_gse114002_designed_a2(
        protocol_path=protocol_path,
        protocol_sha256=_file_sha256(protocol_path),
        source_path=source,
        output_directory=output,
    )
    assert result["kind"] == "PUBLISHED"
    assert result["published"] is True
    assert result["publication_mode"] == QUALIFY.FALLBACK_PUBLICATION_MODE
    assert result["terminal_commit_marker_validated"] is True
    assert result["durability_warning_codes"] == [
        "POST_COMMIT_OUTPUT_DIRECTORY_FSYNC_FAILED",
        "PRECOMMIT_STAGING_DIRECTORY_FSYNC_UNSUPPORTED",
    ]


def test_publication_parent_fd_is_opened_before_commit_and_not_reopened_after(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol_path, source, _, _ = _write_integration_fixture(tmp_path, monkeypatch)
    output = tmp_path / "parent-fd-order"
    original_open = QUALIFY.os.open
    original_rename = QUALIFY._rename_directory_noreplace
    parent_fds: list[int] = []

    def tracked_open(
        path: object, flags: int, mode: int = 0o777, *, dir_fd: int | None = None
    ) -> int:
        fd = original_open(path, flags, mode, dir_fd=dir_fd)
        if Path(path) == tmp_path:
            parent_fds.append(fd)
        return fd

    def assert_open_before_commit(source_dir: Path, target: Path, parent_fd: int) -> None:
        assert parent_fds == [parent_fd]
        original_rename(source_dir, target, parent_fd)

    monkeypatch.setattr(QUALIFY.os, "open", tracked_open)
    monkeypatch.setattr(QUALIFY, "_rename_directory_noreplace", assert_open_before_commit)
    result = QUALIFY.qualify_gse114002_designed_a2(
        protocol_path=protocol_path,
        protocol_sha256=_file_sha256(protocol_path),
        source_path=source,
        output_directory=output,
    )
    assert result["published"] is True
    assert len(parent_fds) == 1


def test_post_commit_parent_fsync_failure_returns_published_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol_path, source, _, _ = _write_integration_fixture(tmp_path, monkeypatch)
    output = tmp_path / "post-commit-fsync"
    original_open = QUALIFY.os.open
    original_fsync = QUALIFY.os.fsync
    parent_fd: list[int] = []

    def tracked_open(
        path: object, flags: int, mode: int = 0o777, *, dir_fd: int | None = None
    ) -> int:
        fd = original_open(path, flags, mode, dir_fd=dir_fd)
        if Path(path) == tmp_path:
            parent_fd[:] = [fd]
        return fd

    def injected_fsync(fd: int) -> None:
        if parent_fd and fd == parent_fd[0]:
            raise OSError("injected post-commit parent fsync failure")
        original_fsync(fd)

    monkeypatch.setattr(QUALIFY.os, "open", tracked_open)
    monkeypatch.setattr(QUALIFY.os, "fsync", injected_fsync)
    result = QUALIFY.qualify_gse114002_designed_a2(
        protocol_path=protocol_path,
        protocol_sha256=_file_sha256(protocol_path),
        source_path=source,
        output_directory=output,
    )
    assert result["kind"] == "PUBLISHED"
    assert result["published"] is True
    assert result["durability_warning_codes"] == [
        "POST_COMMIT_PARENT_DIRECTORY_FSYNC_FAILED"
    ]
    assert (output / "QUALIFICATION_REPORT.json").is_file()


def test_post_commit_parent_close_failure_returns_published_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol_path, source, _, _ = _write_integration_fixture(tmp_path, monkeypatch)
    output = tmp_path / "post-commit-close"
    original_open = QUALIFY.os.open
    original_close = QUALIFY.os.close
    parent_fd: list[int] = []

    def tracked_open(
        path: object, flags: int, mode: int = 0o777, *, dir_fd: int | None = None
    ) -> int:
        fd = original_open(path, flags, mode, dir_fd=dir_fd)
        if Path(path) == tmp_path:
            parent_fd[:] = [fd]
        return fd

    def injected_close(fd: int) -> None:
        if parent_fd and fd == parent_fd[0]:
            original_close(fd)
            raise OSError("injected post-commit parent close failure")
        original_close(fd)

    monkeypatch.setattr(QUALIFY.os, "open", tracked_open)
    monkeypatch.setattr(QUALIFY.os, "close", injected_close)
    result = QUALIFY.qualify_gse114002_designed_a2(
        protocol_path=protocol_path,
        protocol_sha256=_file_sha256(protocol_path),
        source_path=source,
        output_directory=output,
    )
    assert result["kind"] == "PUBLISHED"
    assert result["published"] is True
    assert result["durability_warning_codes"] == [
        "POST_COMMIT_PARENT_DIRECTORY_CLOSE_FAILED"
    ]
    assert (output / "QUALIFICATION_REPORT.json").is_file()


def test_exact_header_is_required() -> None:
    wrong = list(QUALIFY.EXPECTED_HEADER)
    wrong[-1] = "untrusted_field"
    with pytest.raises(QUALIFY.QualificationError, match="exact header mismatch"):
        QUALIFY._analyze_verified_source(
            _gzip_csv(_fixture_rows(), header=tuple(wrong)), _production_protocol()
        )
