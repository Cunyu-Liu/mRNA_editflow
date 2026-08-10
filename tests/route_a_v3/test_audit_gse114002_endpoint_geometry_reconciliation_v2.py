from __future__ import annotations

import copy
import csv
import gzip
import hashlib
import importlib.util
import io
import json
import subprocess
from collections import Counter
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = (
    ROOT
    / "scripts"
    / "route_a_v3"
    / "audit_gse114002_endpoint_geometry_reconciliation_v2.py"
)
PROTOCOL_PATH = (
    ROOT
    / "configs"
    / "route_a_v3_gse114002_endpoint_geometry_reconciliation_v2.json"
)
SPEC = importlib.util.spec_from_file_location("gse114002_reconciliation_v2", MODULE_PATH)
assert SPEC and SPEC.loader
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


MOTHER = "AACCGGTTAACCGGTTAACCGGTTAA"
CANDIDATES = (
    "CACCGGTTAACCGGTTAACCGGTTAA",
    "CGCCGGTTAACCGGTTAACCGGTTAA",
    "CGTCGGTTAACCGGTTAACCGGTTAA",
)
SECRET_ID = "secret-raw-id-collision-7f3a"


def _production_protocol() -> dict[str, object]:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def _explicit_unknown_protocol() -> dict[str, object]:
    protocol = _production_protocol()
    protocol["implementation_binding"] = copy.deepcopy(
        AUDIT.IMPLEMENTATION_BINDING_UNKNOWN
    )
    protocol["unresolved_blockers"] = [
        *AUDIT.BASE_BLOCKERS,
        AUDIT.IMPLEMENTATION_BINDING_BLOCKER,
    ]
    return protocol


def _row(
    *,
    utr: str,
    designed: bool,
    raw_id: str,
    first_index: str,
    mother: str = MOTHER,
    library: str = "human_utrs",
    published_rl: float | None = None,
    total: float = 105.0,
) -> list[str]:
    values = [""] * len(AUDIT.EXPECTED_HEADER)
    index = {name: position for position, name in enumerate(AUDIT.EXPECTED_HEADER)}
    values[0] = first_index
    values[index["utr"]] = utr
    for position, column in enumerate(AUDIT.RAW_FRACTION_COLUMNS):
        values[index[column]] = str(position + 1)
    values[index["total"]] = str(total)
    for column in AUDIT.NORMALIZED_FRACTION_COLUMNS:
        values[index[column]] = str(1.0 / 14.0)
    values[index["r_total"]] = "3.5"
    endpoint = sum(AUDIT.FRACTION_WEIGHTS) / 14.0
    values[index["rl"]] = str(endpoint if published_rl is None else published_rl)
    values[index["id"]] = raw_id
    values[index["info1"]] = "opaque-one"
    values[index["info2"]] = "opaque-two"
    values[index["info3"]] = "opaque-three"
    values[index["info4"]] = "opaque-four"
    values[index["library"]] = library
    values[index["mother"]] = mother
    values[index["designed"]] = "True" if designed else "False"
    values[index["match_score"]] = "1.0"
    return values


def _retune_two_stage_cohort(rows: list[list[str]]) -> None:
    r_total_index = AUDIT.EXPECTED_HEADER.index("r_total")
    cohort_r_total = 14.0 / len(rows)
    for row in rows:
        row[r_total_index] = str(cohort_r_total)


def _base_rows() -> list[list[str]]:
    return [
        _row(
            utr=MOTHER,
            designed=True,
            raw_id=SECRET_ID,
            first_index="duplicate-index",
        ),
        _row(
            utr=CANDIDATES[0],
            designed=False,
            raw_id=SECRET_ID,
            first_index="duplicate-index",
        ),
        _row(
            utr=CANDIDATES[1],
            designed=False,
            raw_id="",
            first_index="index-two",
        ),
        _row(
            utr=CANDIDATES[2],
            designed=False,
            raw_id="unique-id",
            first_index="index-three",
        ),
    ]


def _gzip_csv(rows: list[list[str]], *, header: tuple[str, ...] | None = None) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header or AUDIT.EXPECTED_HEADER)
    writer.writerows(rows)
    return gzip.compress(buffer.getvalue().encode("utf-8"), mtime=0)


def _raw_id_expected(rows: list[list[str]], *, included_only: bool) -> dict[str, int]:
    index = {name: position for position, name in enumerate(AUDIT.EXPECTED_HEADER)}
    selected = [
        row
        for row in rows
        if not included_only or row[index["library"]] in AUDIT.INCLUDED_LIBRARIES
    ]
    values = Counter(row[index["id"]] for row in selected if row[index["id"]])
    return {
        "blank_record_count": sum(not row[index["id"]] for row in selected),
        "nonblank_distinct_token_count": len(values),
        "duplicated_nonblank_token_count": sum(count > 1 for count in values.values()),
        "record_count_in_duplicated_nonblank_tokens": sum(
            count for count in values.values() if count > 1
        ),
        "maximum_nonblank_token_multiplicity": max(values.values(), default=0),
    }


def _fixture_protocol(
    rows: list[list[str]],
    *,
    expected_malformed: int = 0,
    expected_out_of_scope: int = 0,
    expected_noneligible_rule_records: int = 0,
) -> dict[str, object]:
    protocol = _production_protocol()
    row_count = len(rows)
    input_contract = protocol["input_contract"]
    endpoint = protocol["endpoint_reconciliation_contract"]
    pool = protocol["pool_geometry_contract"]
    input_contract["expected_source_row_count"] = row_count
    for key in (
        "expected_normalized_fraction_sum_match_count",
        "expected_paper_endpoint_match_count",
        "expected_sum_normalized_equivalent_match_count",
        "expected_raw_total_match_count",
    ):
        endpoint[key] = row_count
    endpoint["expected_paper_endpoint_max_abs_residual"] = 0
    endpoint["expected_normalized_fraction_sum_max_abs_residual"] = 0
    endpoint["expected_sum_normalized_equivalent_max_abs_residual"] = 0
    endpoint["expected_raw_total_max_abs_residual"] = 0
    endpoint["expected_two_stage_r_total_match_count"] = row_count
    endpoint["expected_two_stage_stored_fraction_vector_match_count"] = row_count
    included = sum(
        row[{name: pos for pos, name in enumerate(AUDIT.EXPECTED_HEADER)}["library"]]
        in AUDIT.INCLUDED_LIBRARIES
        for row in rows
        if len(row) == len(AUDIT.EXPECTED_HEADER)
    )
    pool["expected_included_library_record_count"] = included
    pool["expected_malformed_included_record_count"] = expected_malformed
    pool["expected_valid_out_of_rule_included_record_count"] = expected_out_of_scope
    pool["expected_provisional_source_pool_count"] = 1
    pool["expected_valid_rule_record_count_in_noneligible_pools"] = (
        expected_noneligible_rule_records
    )
    pool["expected_eligible_rule_record_count"] = 4
    pool["expected_eligible_identity_record_count"] = 1
    pool["expected_eligible_provisional_pool_count"] = 1
    pool["expected_eligible_distinct_candidate_count"] = 3
    pool["expected_hamming_distance_candidate_counts"] = {"1": 1, "2": 1, "3": 1}
    pool["expected_pool_with_hamming_distance_candidate_counts"] = {
        "1": 1,
        "2": 1,
        "3": 1,
    }
    pool["expected_pool_with_hamming_distance_3_candidate_count"] = 1
    pool["expected_hamming_distance_5_candidate_count"] = 0
    pool["expected_global_raw_id_audit"] = _raw_id_expected(rows, included_only=False)
    pool["expected_included_scope_raw_id_audit"] = _raw_id_expected(
        rows, included_only=True
    )
    first = Counter(row[0] for row in rows if len(row) == len(AUDIT.EXPECTED_HEADER))
    pool["expected_first_unnamed_index_audit"] = {
        "distinct_token_count": len(first),
        "duplicate_excess_record_count": sum(first.values()) - len(first),
    }
    return protocol


def _analyze(
    rows: list[list[str]],
    *,
    protocol: dict[str, object] | None = None,
) -> tuple[dict[str, object], dict[str, object], dict[str, object], list[str]]:
    frozen = _fixture_protocol(rows) if protocol is None else protocol
    return AUDIT._analyze_verified_source(_gzip_csv(rows), frozen)


def _payloads(rows: list[list[str]]) -> dict[str, object]:
    protocol = _fixture_protocol(rows)
    input_audit, endpoint, geometry, blockers = _analyze(rows, protocol=protocol)
    conditional = bool(set(blockers) & AUDIT.CONDITIONAL_BLOCKERS)
    report = {
        "contract_id": AUDIT.CONTRACT_ID,
        "protocol_id": AUDIT.PROTOCOL_ID,
        "dataset_id": AUDIT.DATASET_ID,
        "status": AUDIT.FAILED_MECHANICAL_STATUS if conditional else AUDIT.MECHANICAL_STATUS,
        "qualified": False,
        "data_role": AUDIT.DATA_ROLE,
        "scientific_claim_status": "NOT_ESTABLISHED",
        "ordinary_study_contribution": 0,
        "a1_intervention_study_contribution": 0,
        "true_a2_dense_study_contribution": 0,
        "canonical_record_count": 0,
        "canonical_materialization_allowed": False,
        "training_allowed": False,
        "model_selection_allowed": False,
        "next_phase_authorized": False,
        "true_a2_claim_established": False,
        "aggregate_only": True,
        "blockers": blockers,
        "protocol_provenance": {
            "basename": AUDIT.PROTOCOL_BASENAME,
            "sha256": "0" * 64,
            "bytes": 1,
            "parser_input_mode": "SAME_DESCRIPTOR_VERIFIED_SNAPSHOT",
            "launch_expected_sha256": "0" * 64,
        },
        "source_provenance": {
            "basename": AUDIT.SOURCE_BASENAME,
            "sha256": "1" * 64,
            "bytes": 1,
            "parser_input_mode": "SAME_DESCRIPTOR_VERIFIED_SNAPSHOT",
        },
        "implementation_binding": {
            "status": "PASS_BOUND_IMPLEMENTATION",
            "verified": True,
            "implementation_commit": "2" * 40,
            "binding_commit": "3" * 40,
            "clean_worktree": True,
            "implementation_direct_child_of_staging_parent": True,
            "implementation_changed_paths_exact": True,
            "config_only_direct_child": True,
            "active_authority_blobs_match": True,
            "head_authority_blobs_match": True,
            "implementation_blobs_match": True,
            "running_script_matches_bound_blob": True,
        },
    }
    return {
        "INPUT_INTEGRITY_AUDIT.json": input_audit,
        "ENDPOINT_RECONCILIATION_AUDIT.json": endpoint,
        "POOL_GEOMETRY_RECONCILIATION_AUDIT.json": geometry,
        "QUALIFICATION_REPORT.json": report,
    }


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_production_protocol_freezes_correct_endpoint_and_exact_aggregates() -> None:
    protocol = _production_protocol()
    AUDIT._validate_protocol(protocol)
    source = protocol["input_contract"]
    endpoint = protocol["endpoint_reconciliation_contract"]
    pool = protocol["pool_geometry_contract"]
    boundary = protocol["decision_neutral_boundary"]
    assert source["compressed_bytes"] == 17_332_142
    assert source["compressed_sha256"] == (
        "b72ac298cb0f4d21f911d330c0def06f8d94f15d9f8cc22f3a50ae87a7ef7ee5"
    )
    assert tuple(source["exact_header"]) == AUDIT.EXPECTED_HEADER
    assert endpoint["fraction_weights"] == list(AUDIT.FRACTION_WEIGHTS)
    assert endpoint["paper_endpoint_formula"] == (
        "SUM_RI_TIMES_FROZEN_WEIGHT_NO_DENOMINATOR"
    )
    assert endpoint["expected_paper_endpoint_max_abs_residual"] == (
        2.2500223906263273e-11
    )
    assert endpoint["expected_normalized_fraction_sum_max_abs_residual"] == (
        2.099875828776021e-12
    )
    assert endpoint["expected_sum_normalized_equivalent_max_abs_residual"] == (
        1.2160050744114415e-11
    )
    assert endpoint["raw_fraction_count_endpoint_role"] == (
        "RAW_RECONSTRUCTS_ONLY_VIA_PAPER_TWO_STAGE_GLOBAL_NORMALIZATION"
    )
    assert endpoint["expected_two_stage_r_total_match_count"] == 100_017
    assert endpoint["two_stage_r_total_max_abs_residual_cap"] == 1e-12
    assert endpoint["expected_two_stage_stored_fraction_vector_match_count"] == 100_017
    assert endpoint["two_stage_stored_fraction_vector_max_abs_residual_cap"] == 1e-12
    assert endpoint["two_stage_global_fraction_totals_output_allowed"] is False
    assert pool["expected_included_library_record_count"] == 50_600
    assert pool["expected_provisional_source_pool_count"] == 23_177
    assert pool["expected_valid_out_of_rule_included_record_count"] == 8_909
    assert pool["expected_valid_rule_record_count_in_noneligible_pools"] == 36_833
    assert pool["expected_eligible_rule_record_count"] == 4_858
    assert pool["expected_eligible_provisional_pool_count"] == 959
    assert pool["expected_eligible_distinct_candidate_count"] == 3_899
    assert pool["expected_hamming_distance_candidate_counts"] == {
        "1": 2_925,
        "2": 870,
        "3": 104,
    }
    assert pool["expected_pool_with_hamming_distance_candidate_counts"] == {
        "1": 952,
        "2": 574,
        "3": 91,
    }
    assert boundary["status"] == AUDIT.MECHANICAL_STATUS
    assert boundary["qualified"] is False
    assert boundary["ordinary_study_contribution"] == 0
    assert boundary["a1_intervention_study_contribution"] == 0
    assert boundary["true_a2_dense_study_contribution"] == 0
    assert boundary["training_allowed"] is False
    binding = protocol["implementation_binding"]
    AUDIT._validate_binding_document(binding)
    if binding["status"] == "UNKNOWN_NOT_ASSERTED":
        assert binding == AUDIT.IMPLEMENTATION_BINDING_UNKNOWN
        assert AUDIT.IMPLEMENTATION_BINDING_BLOCKER in protocol["unresolved_blockers"]
    else:
        assert binding["status"] == "BOUND"
        assert AUDIT.COMMIT_RE.fullmatch(binding["implementation_commit"])
        assert AUDIT.SHA256_RE.fullmatch(binding["qualifier_blob_sha256"])
        assert AUDIT.SHA256_RE.fullmatch(binding["test_blob_sha256"])
        assert protocol["unresolved_blockers"] == list(AUDIT.BASE_BLOCKERS)


def test_correct_processed_formula_and_provisional_geometry_reconcile() -> None:
    input_audit, endpoint, geometry, blockers = _analyze(_base_rows())
    assert input_audit["status"] == "PASS_EXACT_PUBLIC_ASSET_AND_HEADER"
    assert endpoint["status"] == "PASS_EXACT_PAPER_ENDPOINT_RECONCILIATION"
    assert endpoint["paper_endpoint_match_count"] == 4
    assert endpoint["normalized_fraction_sum_match_count"] == 4
    assert endpoint["sum_normalized_equivalent_match_count"] == 4
    assert endpoint["raw_total_match_count"] == 4
    assert endpoint["two_stage_r_total_match_count"] == 4
    assert endpoint["two_stage_r_total_max_abs_residual"] <= 1e-12
    assert endpoint["two_stage_stored_fraction_vector_match_count"] == 4
    assert endpoint["two_stage_stored_fraction_vector_max_abs_residual"] <= 1e-12
    assert endpoint["global_fraction_totals_emitted"] is False
    assert geometry["status"] == "PASS_PROVISIONAL_GEOMETRY_MATCH_NOT_AUTHORITY"
    assert geometry["eligible_provisional_pool_count"] == 1
    assert geometry["eligible_provisional_distinct_candidate_count"] == 3
    assert geometry["hamming_distance_candidate_counts"] == {"1": 1, "2": 1, "3": 1}
    assert geometry["pool_with_hamming_distance_candidate_counts"] == {
        "1": 1,
        "2": 1,
        "3": 1,
    }
    assert not set(blockers) & AUDIT.CONDITIONAL_BLOCKERS


def test_legacy_integer_weights_and_second_normalization_are_rejected() -> None:
    normalized = [0.0] * 14
    normalized[6] = 1.0
    assert AUDIT.reconstruct_paper_rl(normalized) == 4.8
    with pytest.raises(AUDIT.BoundaryViolation, match="frozen paper weights"):
        AUDIT.reconstruct_paper_rl(
            normalized, weights=AUDIT.LEGACY_INCORRECT_WEIGHTS
        )
    with pytest.raises(AUDIT.BoundaryViolation, match="divided a second time"):
        AUDIT.reconstruct_paper_rl(normalized, denominator=37.0)


def test_raw_counts_and_p_or_fdr_cannot_create_endpoint_or_standard_error() -> None:
    with pytest.raises(AUDIT.BoundaryViolation, match="naive raw-row endpoint"):
        AUDIT.reconstruct_paper_rl_from_raw_counts([1.0] * 14)
    with pytest.raises(AUDIT.BoundaryViolation, match="technical or biological SE"):
        AUDIT.derive_standard_error_from_fraction_counts([1.0] * 14)
    with pytest.raises(AUDIT.BoundaryViolation, match="p/FDR"):
        AUDIT.infer_standard_error_from_p_or_fdr(0.05, 0.1)


def test_analysis_never_calls_prohibited_raw_endpoint_or_se_helpers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> float:
        raise AssertionError("prohibited derivation was called")

    monkeypatch.setattr(AUDIT, "reconstruct_paper_rl_from_raw_counts", forbidden)
    monkeypatch.setattr(AUDIT, "derive_standard_error_from_fraction_counts", forbidden)
    _input, endpoint, _geometry, _blockers = _analyze(_base_rows())
    assert endpoint["raw_fraction_count_endpoint_role"] == (
        "RAW_RECONSTRUCTS_ONLY_VIA_PAPER_TWO_STAGE_GLOBAL_NORMALIZATION"
    )
    assert endpoint["global_fraction_totals_emitted"] is False
    assert endpoint["naive_raw_row_endpoint_reconstruction_allowed"] is False
    assert endpoint["technical_standard_error_derived"] is False
    assert endpoint["biological_standard_error_derived"] is False
    assert endpoint["p_or_fdr_used_to_back_calculate_standard_error"] is False


def test_wrong_paper_endpoint_is_a_mechanical_blocker() -> None:
    rows = _base_rows()
    endpoint_index = AUDIT.EXPECTED_HEADER.index("rl")
    rows[0][endpoint_index] = "6.0"
    protocol = _fixture_protocol(rows)
    _input, endpoint, _geometry, blockers = _analyze(rows, protocol=protocol)
    assert endpoint["status"] == "FAIL_ENDPOINT_OR_TOTAL_RECONCILIATION"
    assert "ENDPOINT_RECONCILIATION_MISMATCH" in blockers
    assert "SUM_NORMALIZED_EQUIVALENT_RECONCILIATION_MISMATCH" in blockers


def test_raw_total_mismatch_blocks_without_using_raw_counts_as_endpoint() -> None:
    rows = _base_rows()
    total_index = AUDIT.EXPECTED_HEADER.index("total")
    rows[0][total_index] = "106"
    protocol = _fixture_protocol(rows)
    _input, endpoint, _geometry, blockers = _analyze(rows, protocol=protocol)
    assert endpoint["raw_total_match_count"] == 3
    assert endpoint["naive_raw_row_endpoint_reconstruction_allowed"] is False
    assert "RAW_TOTAL_RECONCILIATION_MISMATCH" in blockers


def test_two_stage_global_normalization_mismatch_is_a_mechanical_blocker() -> None:
    rows = _base_rows()
    r_total_index = AUDIT.EXPECTED_HEADER.index("r_total")
    rows[0][r_total_index] = "4.0"
    protocol = _fixture_protocol(rows)
    _input, endpoint, _geometry, blockers = _analyze(rows, protocol=protocol)
    assert endpoint["two_stage_r_total_match_count"] == 3
    assert endpoint["two_stage_stored_fraction_vector_match_count"] == 3
    assert "TWO_STAGE_GLOBAL_NORMALIZATION_RECONCILIATION_MISMATCH" in blockers


def test_global_raw_id_collisions_are_aggregate_boundary_not_automatic_blocker() -> None:
    _input, _endpoint, geometry, blockers = _analyze(_base_rows())
    assert geometry["raw_identifier_collision_distinct_token_count"] == 1
    assert geometry["raw_identifier_collision_record_count"] == 2
    assert geometry["raw_identifier_missing_record_count"] == 1
    assert geometry["raw_identifier_collision_role"] == (
        "AGGREGATE_COLLISION_AUDIT_ONLY_NOT_AUTOMATIC_BLOCKER"
    )
    assert not any("ID" in blocker or "IDENTIFIER" in blocker for blocker in blockers)


def test_valid_out_of_rule_record_is_pending_not_malformed_or_blocking() -> None:
    rows = _base_rows()
    rows.append(
        _row(
            utr=MOTHER,
            designed=False,
            raw_id="out-of-scope-id",
            first_index="index-four",
        )
    )
    _retune_two_stage_cohort(rows)
    protocol = _fixture_protocol(rows, expected_out_of_scope=1)
    _input, _endpoint, geometry, blockers = _analyze(rows, protocol=protocol)
    assert geometry["valid_out_of_rule_included_record_count"] == 1
    assert geometry["valid_out_of_rule_included_record_status"] == (
        "OUT_OF_SCOPE_DISPOSITION_PENDING"
    )
    assert geometry["malformed_included_record_count"] == 0
    assert "MALFORMED_INCLUDED_ROWS_PRESENT" not in blockers
    assert "PROVISIONAL_POOL_GEOMETRY_RECONCILIATION_MISMATCH" not in blockers


def test_truly_malformed_included_record_is_a_blocker() -> None:
    rows = _base_rows()
    rows.append(
        _row(
            utr="N" + MOTHER[1:],
            designed=False,
            raw_id="malformed-id",
            first_index="index-four",
        )
    )
    _retune_two_stage_cohort(rows)
    protocol = _fixture_protocol(rows, expected_malformed=0)
    _input, _endpoint, geometry, blockers = _analyze(rows, protocol=protocol)
    assert geometry["malformed_included_record_count"] == 1
    assert geometry["valid_out_of_rule_included_record_count"] == 0
    assert "MALFORMED_INCLUDED_ROWS_PRESENT" in blockers


def test_payload_privacy_and_closed_schema_reject_sensitive_arrays() -> None:
    payloads = _payloads(_base_rows())
    AUDIT._validate_closed_output_payloads(payloads)
    rendered = json.dumps(payloads, sort_keys=True)
    for secret in (*CANDIDATES, MOTHER, SECRET_ID):
        assert secret not in rendered
    with pytest.raises(AUDIT.PublicationError, match="sensitive array"):
        AUDIT._assert_aggregate_safe({"sequence_values": [MOTHER]})
    with pytest.raises(AUDIT.PublicationError, match="26-nt-or-longer"):
        AUDIT._assert_aggregate_safe({"allowed_status": MOTHER})
    with pytest.raises(AUDIT.PublicationError, match="26-nt-or-longer"):
        AUDIT._assert_aggregate_safe({"allowed_status": "AUGC" * 7})
    bad = copy.deepcopy(payloads)
    bad["POOL_GEOMETRY_RECONCILIATION_AUDIT.json"]["sequence_values"] = [MOTHER]
    with pytest.raises(AUDIT.PublicationError, match="closed schema"):
        AUDIT._validate_closed_output_payloads(bad)


def test_unknown_binding_stops_before_any_source_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol_payload = (
        json.dumps(_explicit_unknown_protocol(), sort_keys=True) + "\n"
    ).encode("utf-8")
    protocol_path = tmp_path / AUDIT.PROTOCOL_BASENAME
    protocol_path.write_bytes(protocol_payload)
    source_path = tmp_path / AUDIT.SOURCE_BASENAME
    source_path.write_bytes(b"must-not-be-read")
    touched = False

    def forbidden_read(*_args: object, **_kwargs: object) -> object:
        nonlocal touched
        touched = True
        raise AssertionError("source snapshot was attempted")

    monkeypatch.setattr(AUDIT, "_read_source_snapshot", forbidden_read)
    with pytest.raises(AUDIT.BindingNotFrozen, match="source access stopped"):
        AUDIT.audit_gse114002_endpoint_geometry_reconciliation_v2(
            protocol_path=protocol_path,
            protocol_sha256=hashlib.sha256(protocol_payload).hexdigest(),
            source_path=source_path,
            output_directory=tmp_path / "output",
        )
    assert touched is False


def test_forbidden_scope_is_rejected_before_protocol_or_asset_read(tmp_path: Path) -> None:
    with pytest.raises(AUDIT.ScopeViolation, match="before read"):
        AUDIT._preflight_paths_before_read(
            tmp_path / AUDIT.PROTOCOL_BASENAME,
            tmp_path / "restricted" / AUDIT.SOURCE_BASENAME,
            tmp_path / "output",
        )


def test_two_commit_binding_requires_exact_config_only_direct_child(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "Route A Test")
    _git(repo, "config", "user.email", "route-a@example.invalid")

    contract = repo / "docs" / "goals" / "contract.md"
    registry = repo / "docs" / "execution" / "registry.yaml"
    decision = repo / "docs" / "execution" / "decision.yaml"
    contract.parent.mkdir(parents=True)
    registry.parent.mkdir(parents=True)
    contract.write_text("contract\n", encoding="utf-8")
    registry.write_text("registry\n", encoding="utf-8")
    decision.write_text("decision\n", encoding="utf-8")
    _git(repo, "add", "docs")
    _git(repo, "commit", "-q", "-m", "active authority")
    active = _git(repo, "rev-parse", "HEAD")

    (repo / "STAGING_PARENT").write_text("parent\n", encoding="utf-8")
    _git(repo, "add", "STAGING_PARENT")
    _git(repo, "commit", "-q", "-m", "staging parent")
    staging_parent = _git(repo, "rev-parse", "HEAD")

    config_relative = AUDIT.IMPLEMENTATION_BINDING_UNKNOWN[
        "post_implementation_allowed_changed_paths"
    ][0]
    qualifier_relative = AUDIT.IMPLEMENTATION_BINDING_UNKNOWN["qualifier_path"]
    test_relative = AUDIT.IMPLEMENTATION_BINDING_UNKNOWN["test_path"]
    config = repo / config_relative
    qualifier = repo / qualifier_relative
    focused_test = repo / test_relative
    config.parent.mkdir(parents=True)
    qualifier.parent.mkdir(parents=True)
    focused_test.parent.mkdir(parents=True)
    qualifier.write_text("print('bound auditor')\n", encoding="utf-8")
    focused_test.write_text("def test_bound(): pass\n", encoding="utf-8")
    implementation_protocol = {
        "implementation_binding": AUDIT.IMPLEMENTATION_BINDING_UNKNOWN,
        "unresolved_blockers": [
            *AUDIT.BASE_BLOCKERS,
            AUDIT.IMPLEMENTATION_BINDING_BLOCKER,
        ],
        "core": {"frozen": True},
    }
    config.write_text(
        json.dumps(implementation_protocol, sort_keys=True) + "\n", encoding="utf-8"
    )
    _git(repo, "add", "configs", "scripts", "tests")
    _git(repo, "commit", "-q", "-m", "implementation I")
    implementation = _git(repo, "rev-parse", "HEAD")

    binding = copy.deepcopy(AUDIT.IMPLEMENTATION_BINDING_UNKNOWN)
    binding.update(
        {
            "status": "BOUND",
            "implementation_commit": implementation,
            "qualifier_blob_sha256": _file_sha256(qualifier),
            "test_blob_sha256": _file_sha256(focused_test),
        }
    )
    binding_protocol = {
        "implementation_binding": binding,
        "unresolved_blockers": list(AUDIT.BASE_BLOCKERS),
        "core": {"frozen": True},
    }
    config.write_text(
        json.dumps(binding_protocol, sort_keys=True) + "\n", encoding="utf-8"
    )
    _git(repo, "add", config_relative)
    _git(repo, "commit", "-q", "-m", "config-only binding B")

    authority = {
        "contract_path": "docs/goals/contract.md",
        "contract_sha256": _file_sha256(contract),
        "data_role_registry_path": "docs/execution/registry.yaml",
        "data_role_registry_sha256": _file_sha256(registry),
        "decision_log_path": "docs/execution/decision.yaml",
        "decision_log_sha256": _file_sha256(decision),
        "active_authority_commit": active,
        "staging_parent_head": staging_parent,
    }
    result = AUDIT._verify_implementation_binding(
        binding, authority, repo, running_script_path=qualifier
    )
    assert result == {
        "status": "PASS_BOUND_IMPLEMENTATION",
        "verified": True,
        "implementation_commit": implementation,
        "binding_commit": _git(repo, "rev-parse", "HEAD"),
        "clean_worktree": True,
        "implementation_direct_child_of_staging_parent": True,
        "implementation_changed_paths_exact": True,
        "config_only_direct_child": True,
        "active_authority_blobs_match": True,
        "head_authority_blobs_match": True,
        "implementation_blobs_match": True,
        "running_script_matches_bound_blob": True,
    }


def _invalid_i_binding_fixture(
    tmp_path: Path,
    *,
    extra_i_file: bool = False,
    non_direct_i: bool = False,
    authority_drift_in_staging_parent: bool = False,
) -> tuple[Path, dict[str, object], dict[str, object], Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "Route A Test")
    _git(repo, "config", "user.email", "route-a@example.invalid")
    contract = repo / "docs" / "goals" / "contract.md"
    registry = repo / "docs" / "execution" / "registry.yaml"
    decision = repo / "docs" / "execution" / "decision.yaml"
    contract.parent.mkdir(parents=True)
    registry.parent.mkdir(parents=True)
    contract.write_text("contract\n", encoding="utf-8")
    registry.write_text("registry\n", encoding="utf-8")
    decision.write_text("decision\n", encoding="utf-8")
    _git(repo, "add", "docs")
    _git(repo, "commit", "-q", "-m", "active authority")
    active = _git(repo, "rev-parse", "HEAD")
    active_contract_sha256 = _file_sha256(contract)
    active_registry_sha256 = _file_sha256(registry)
    active_decision_sha256 = _file_sha256(decision)
    (repo / "STAGING_PARENT").write_text("parent\n", encoding="utf-8")
    staging_paths = ["STAGING_PARENT"]
    if authority_drift_in_staging_parent:
        registry.write_text("drifted registry\n", encoding="utf-8")
        staging_paths.append("docs/execution/registry.yaml")
    _git(repo, "add", *staging_paths)
    _git(repo, "commit", "-q", "-m", "staging parent")
    staging_parent = _git(repo, "rev-parse", "HEAD")
    if non_direct_i:
        (repo / "INTERMEDIATE").write_text("not allowed\n", encoding="utf-8")
        _git(repo, "add", "INTERMEDIATE")
        _git(repo, "commit", "-q", "-m", "unexpected intermediate")

    config_relative = AUDIT.IMPLEMENTATION_BINDING_UNKNOWN[
        "post_implementation_allowed_changed_paths"
    ][0]
    qualifier_relative = AUDIT.IMPLEMENTATION_BINDING_UNKNOWN["qualifier_path"]
    test_relative = AUDIT.IMPLEMENTATION_BINDING_UNKNOWN["test_path"]
    config = repo / config_relative
    qualifier = repo / qualifier_relative
    focused_test = repo / test_relative
    config.parent.mkdir(parents=True)
    qualifier.parent.mkdir(parents=True)
    focused_test.parent.mkdir(parents=True)
    qualifier.write_text("print('bound auditor')\n", encoding="utf-8")
    focused_test.write_text("def test_bound(): pass\n", encoding="utf-8")
    implementation_protocol = {
        "implementation_binding": AUDIT.IMPLEMENTATION_BINDING_UNKNOWN,
        "unresolved_blockers": [
            *AUDIT.BASE_BLOCKERS,
            AUDIT.IMPLEMENTATION_BINDING_BLOCKER,
        ],
        "core": {"frozen": True},
    }
    config.write_text(
        json.dumps(implementation_protocol, sort_keys=True) + "\n", encoding="utf-8"
    )
    add_paths = [config_relative, qualifier_relative, test_relative]
    if extra_i_file:
        (repo / "EXTRA_I_FILE").write_text("not allowed\n", encoding="utf-8")
        add_paths.append("EXTRA_I_FILE")
    _git(repo, "add", *add_paths)
    _git(repo, "commit", "-q", "-m", "implementation I")
    implementation = _git(repo, "rev-parse", "HEAD")
    binding = copy.deepcopy(AUDIT.IMPLEMENTATION_BINDING_UNKNOWN)
    binding.update(
        {
            "status": "BOUND",
            "implementation_commit": implementation,
            "qualifier_blob_sha256": _file_sha256(qualifier),
            "test_blob_sha256": _file_sha256(focused_test),
        }
    )
    config.write_text(
        json.dumps(
            {
                "implementation_binding": binding,
                "unresolved_blockers": list(AUDIT.BASE_BLOCKERS),
                "core": {"frozen": True},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _git(repo, "add", config_relative)
    _git(repo, "commit", "-q", "-m", "config-only binding B")
    authority = {
        "contract_path": "docs/goals/contract.md",
        "contract_sha256": active_contract_sha256,
        "data_role_registry_path": "docs/execution/registry.yaml",
        "data_role_registry_sha256": active_registry_sha256,
        "decision_log_path": "docs/execution/decision.yaml",
        "decision_log_sha256": active_decision_sha256,
        "active_authority_commit": active,
        "staging_parent_head": staging_parent,
    }
    return repo, binding, authority, qualifier


def test_binding_rejects_extra_file_in_implementation_commit(tmp_path: Path) -> None:
    repo, binding, authority, qualifier = _invalid_i_binding_fixture(
        tmp_path, extra_i_file=True
    )
    with pytest.raises(AUDIT.ProtocolError, match="changed paths are not exactly"):
        AUDIT._verify_implementation_binding(
            binding, authority, repo, running_script_path=qualifier
        )


def test_binding_rejects_non_direct_implementation_commit(tmp_path: Path) -> None:
    repo, binding, authority, qualifier = _invalid_i_binding_fixture(
        tmp_path, non_direct_i=True
    )
    with pytest.raises(AUDIT.ProtocolError, match="not the direct staging-parent child"):
        AUDIT._verify_implementation_binding(
            binding, authority, repo, running_script_path=qualifier
        )


def test_binding_rejects_current_head_authority_drift(tmp_path: Path) -> None:
    repo, binding, authority, qualifier = _invalid_i_binding_fixture(
        tmp_path, authority_drift_in_staging_parent=True
    )
    with pytest.raises(AUDIT.ProtocolError, match="current-HEAD authority blob"):
        AUDIT._verify_implementation_binding(
            binding, authority, repo, running_script_path=qualifier
        )


def test_atomic_no_overwrite_exact_six_members_and_marker_last(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "bundle"
    writes: list[str] = []
    original_write = AUDIT._write_exclusive

    def recording_write(path: Path, payload: bytes) -> None:
        writes.append(path.name)
        original_write(path, payload)

    monkeypatch.setattr(AUDIT, "_write_exclusive", recording_write)
    first = AUDIT._publish_bundle(output, _payloads(_base_rows()))
    assert first["kind"] == "PUBLISHED"
    assert first["file_count"] == 6
    assert set(path.name for path in output.iterdir()) == set(AUDIT.EXACT_BUNDLE_MEMBERS)
    assert writes[-1] == AUDIT.PUBLICATION_COMMIT_FILENAME
    marker = AUDIT._validate_publication_commit(
        output, expected_mode=first["publication_mode"]
    )
    assert marker["committed"] is True
    before = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in output.iterdir()}
    second = AUDIT._publish_bundle(output, _payloads(_base_rows()))
    after = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in output.iterdir()}
    assert second["kind"] == "ALREADY_COMMITTED_EXACT"
    assert second["published"] is True
    assert second["committed"] is True
    assert second["accepted"] is True
    assert before == after


def test_existing_self_consistent_but_different_bundle_is_not_already_accepted(
    tmp_path: Path,
) -> None:
    output = tmp_path / "bundle"
    first = AUDIT._publish_bundle(output, _payloads(_base_rows()))
    assert first["accepted"] is True
    input_path = output / "INPUT_INTEGRITY_AUDIT.json"
    input_document = json.loads(input_path.read_text(encoding="utf-8"))
    input_document["source_asset_bytes"] += 1
    input_path.write_bytes(AUDIT._pretty_json_bytes(input_document))
    rendered = {
        name: (output / name).read_bytes() for name in AUDIT.JSON_PAYLOAD_FILENAMES
    }
    sums = "".join(
        f"{hashlib.sha256(rendered[name]).hexdigest()}  {name}\n"
        for name in sorted(rendered)
    ).encode("ascii")
    (output / AUDIT.SHA256SUMS_FILENAME).write_bytes(sums)
    marker_path = output / AUDIT.PUBLICATION_COMMIT_FILENAME
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["sha256sums_sha256"] = hashlib.sha256(sums).hexdigest()
    marker_path.write_bytes(AUDIT._pretty_json_bytes(marker))
    AUDIT._validate_publication_commit(
        output, expected_mode=first["publication_mode"]
    )

    second = AUDIT._publish_bundle(output, _payloads(_base_rows()))
    assert second["kind"] == "COMMITTED_NOT_ACCEPTED"
    assert second["accepted"] is False
    assert "EXISTING_BUNDLE_DIFFERS_FROM_CURRENT_EXPECTED_BYTES" in second[
        "post_commit_warning_codes"
    ]


def test_rename_then_raise_is_classified_as_committed_exact_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "bundle"

    def rename_then_raise(source: Path, target: Path) -> None:
        source.rename(target)
        raise AUDIT.PublicationError("injected wrapper error after rename")

    monkeypatch.setattr(AUDIT, "_rename_directory_noreplace", rename_then_raise)
    result = AUDIT._publish_bundle(output, _payloads(_base_rows()))
    assert output.is_dir()
    assert not list(tmp_path.glob(".bundle.staging-*"))
    assert result["kind"] == "COMMITTED_WITH_POST_COMMIT_WARNING"
    assert result["committed"] is True
    assert result["accepted"] is True
    assert "RENAME_REPORTED_ERROR_AFTER_EXACT_FINAL_VISIBILITY" in result[
        "post_commit_warning_codes"
    ]


def test_post_rename_validation_failure_is_committed_not_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "bundle"
    original_validate = AUDIT._validate_publication_commit

    def fail_only_after_visibility(
        directory: Path, *, expected_mode: str, final_output: Path | None = None
    ) -> dict[str, object]:
        if directory == output:
            raise AUDIT.PublicationError("injected post-rename validation failure")
        return original_validate(
            directory, expected_mode=expected_mode, final_output=final_output
        )

    monkeypatch.setattr(AUDIT, "_validate_publication_commit", fail_only_after_visibility)
    result = AUDIT._publish_bundle(output, _payloads(_base_rows()))
    assert output.is_dir()
    assert result["kind"] == "COMMITTED_NOT_ACCEPTED"
    assert result["committed"] is True
    assert result["accepted"] is False
    assert result["published"] is False
    assert result["requires_manual_adjudication"] is True
    assert "POST_COMMIT_EXACT_VALIDATION_FAILED" in result["post_commit_warning_codes"]


def test_post_commit_parent_fsync_failure_returns_committed_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "bundle"
    original_fsync = AUDIT._fsync_directory

    def fail_parent_only(path: Path) -> None:
        if path == tmp_path:
            raise OSError("injected parent fsync failure")
        original_fsync(path)

    monkeypatch.setattr(AUDIT, "_fsync_directory", fail_parent_only)
    result = AUDIT._publish_bundle(output, _payloads(_base_rows()))
    assert result["kind"] == "COMMITTED_WITH_POST_COMMIT_WARNING"
    assert result["committed"] is True
    assert result["accepted"] is True
    assert result["published"] is True
    assert result["requires_manual_adjudication"] is False
    assert "POST_COMMIT_PARENT_DIRECTORY_FSYNC_FAILED" in result[
        "post_commit_warning_codes"
    ]


def test_fallback_mid_write_preserves_explicit_unmarked_partial_and_retry_stops(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "bundle"
    original_write = AUDIT._write_exclusive

    def unsupported_rename(_source: Path, _target: Path) -> None:
        raise AUDIT.AtomicNoReplaceUnsupported(38)

    def fail_second_visible_member(path: Path, payload: bytes) -> None:
        if path.parent == output and path.name == "INPUT_INTEGRITY_AUDIT.json":
            raise AUDIT.PublicationError("injected fallback mid-write failure")
        original_write(path, payload)

    monkeypatch.setattr(AUDIT, "_rename_directory_noreplace", unsupported_rename)
    monkeypatch.setattr(AUDIT, "_write_exclusive", fail_second_visible_member)
    first = AUDIT._publish_bundle(output, _payloads(_base_rows()))
    assert output.is_dir()
    assert not (output / AUDIT.PUBLICATION_COMMIT_FILENAME).exists()
    assert first["kind"] == "PARTIAL_REQUIRES_MANUAL_ADJUDICATION"
    assert first["committed"] is False
    assert first["accepted"] is False
    assert first["requires_manual_adjudication"] is True
    assert not list(tmp_path.glob(".bundle.staging-*"))

    monkeypatch.setattr(AUDIT, "_write_exclusive", original_write)
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    second = AUDIT._publish_bundle(output, _payloads(_base_rows()))
    after = {path.name: path.read_bytes() for path in output.iterdir()}
    assert second["kind"] == "PARTIAL_REQUIRES_MANUAL_ADJUDICATION"
    assert second["published"] is False
    assert before == after


def test_primary_precommit_validation_failure_never_creates_final_truth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "bundle"
    original_validate = AUDIT._validate_publication_commit

    def fail_staging(
        directory: Path, *, expected_mode: str, final_output: Path | None = None
    ) -> dict[str, object]:
        if directory.name.startswith(".bundle.staging-"):
            raise AUDIT.PublicationError("injected precommit validation failure")
        return original_validate(
            directory, expected_mode=expected_mode, final_output=final_output
        )

    monkeypatch.setattr(AUDIT, "_validate_publication_commit", fail_staging)
    with pytest.raises(AUDIT.PublicationError, match="injected precommit"):
        AUDIT._publish_bundle(output, _payloads(_base_rows()))
    assert not output.exists()
    assert not list(tmp_path.glob(".bundle.staging-*"))


def test_cli_returns_nonzero_for_manual_publication_state(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    result = {
        "kind": "PARTIAL_REQUIRES_MANUAL_ADJUDICATION",
        "accepted": False,
        "published": False,
    }

    def partial_result(**_kwargs: object) -> dict[str, object]:
        return result

    monkeypatch.setattr(
        AUDIT,
        "audit_gse114002_endpoint_geometry_reconciliation_v2",
        partial_result,
    )
    exit_code = AUDIT.main(
        [
            "--protocol",
            AUDIT.PROTOCOL_BASENAME,
            "--protocol-sha256",
            "0" * 64,
            "--source",
            AUDIT.SOURCE_BASENAME,
            "--output-directory",
            "output",
        ]
    )
    assert exit_code == 4
    assert json.loads(capsys.readouterr().out) == result


def test_output_report_cannot_upgrade_any_scientific_boundary() -> None:
    report = _payloads(_base_rows())["QUALIFICATION_REPORT.json"]
    assert report["status"] == AUDIT.MECHANICAL_STATUS
    assert report["qualified"] is False
    assert report["ordinary_study_contribution"] == 0
    assert report["a1_intervention_study_contribution"] == 0
    assert report["true_a2_dense_study_contribution"] == 0
    assert report["canonical_record_count"] == 0
    assert report["training_allowed"] is False
    assert report["model_selection_allowed"] is False
    assert report["next_phase_authorized"] is False
    assert report["true_a2_claim_established"] is False
