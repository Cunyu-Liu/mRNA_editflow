from __future__ import annotations

import copy
import csv
import gzip
import hashlib
import importlib.util
import inspect
import io
import json
import sys
import tarfile
from pathlib import Path
from unittest import mock

import pytest


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / (
    "configs/route_a_v3_gse295080_independence_overlap_adjudication_v1.json"
)
MODULE_PATH = ROOT / (
    "scripts/route_a_v3/preflight_gse295080_independence_overlap_adjudication.py"
)
REPORT_PATH = ROOT / "reports/GSE295080_INDEPENDENCE_OVERLAP_AGGREGATE_PREFLIGHT_V1.json"
SPEC = importlib.util.spec_from_file_location("gse295080_overlap_preflight", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
PREFLIGHT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PREFLIGHT
SPEC.loader.exec_module(PREFLIGHT)


def _protocol() -> dict:
    protocol = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    PREFLIGHT.validate_protocol(protocol)
    return protocol


def _bound_protocol() -> dict:
    protocol = copy.deepcopy(_protocol())
    binding = protocol["implementation_binding"]
    own = binding["own_preflight_group"]
    own["status"] = PREFLIGHT.BOUND
    own["implementation_commit"] = "9" * 40
    own["implementation_script_sha256"] = "a" * 64
    own["implementation_test_sha256"] = "b" * 64
    PREFLIGHT.validate_protocol(protocol)
    PREFLIGHT._require_all_bindings(protocol)
    return protocol


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_stability(path: Path, headers: list[str]) -> None:
    rows = [
        {"Element": "PRIVATE_L1_A", "Library": "1", "familyID": "PRIVATE_F1", "sampleID": "PRIVATE_S1", "Type": "SNV"},
        {"Element": "PRIVATE_L1_A", "Library": "1", "familyID": "PRIVATE_F1", "sampleID": "PRIVATE_S2", "Type": "SNV"},
        {"Element": "PRIVATE_L1_B", "Library": "1", "familyID": "PRIVATE_F2", "sampleID": "PRIVATE_S3", "Type": "Indel"},
        {"Element": "PRIVATE_L2_A", "Library": "2", "familyID": "PRIVATE_F3", "sampleID": "PRIVATE_S4", "Type": "SNV"},
        {"Element": "PRIVATE_L2_B", "Library": "2", "familyID": "PRIVATE_F3", "sampleID": "PRIVATE_S5", "Type": "SNV"},
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers, delimiter="\t")
        writer.writeheader()
        for partial in rows:
            row = {header: "synthetic" for header in headers}
            row.update(partial)
            writer.writerow(row)


def _write_fasta(path: Path) -> None:
    path.write_text(
        ">PRIVATE_REF_A\nAAAA\n>PRIVATE_REF_B\nCCCCC\n",
        encoding="utf-8",
    )


def _write_soft(path: Path) -> None:
    text = "\n".join(
        (
            "^SAMPLE = PRIVATE_ACCESSION_1",
            "!Sample_title = HEK cells, Pre-Splice BiolRep1",
            "^SAMPLE = PRIVATE_ACCESSION_2",
            "!Sample_title = HEK cells, Post-Splice TechRep1",
            "^SAMPLE = PRIVATE_ACCESSION_3",
            "!Sample_title = HEK cells, Pre-Splice Rep1",
            "^SAMPLE = PRIVATE_ACCESSION_4",
            "!Sample_title = DNA control",
        )
    )
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(text + "\n")


def _write_inventory(path: Path) -> None:
    path.write_text(
        "#Archive/File\tName\tTime\tSize\tType\n"
        "Archive\tPRIVATE_RAW.tar\tdate\t100\tTAR\n"
        "File\tPRIVATE_MEMBER.txt.gz\tdate\t75\tTXT\n",
        encoding="utf-8",
    )


def _write_reference_archive(path: Path) -> None:
    rows = (
        "\tbc\tsequence\tseqName\tbcCount\n"
        "0\tPRIVATE_BC_1\tAAAA\tPRIVATE_L1_A_ref\t1\n"
        "1\tPRIVATE_BC_2\tCCCC\tPRIVATE_L1_A_alt\t1\n"
        "2\tPRIVATE_BC_3\tGGGG\tPRIVATE_L1_B_ref\t1\n"
        "3\tPRIVATE_BC_4\tTTTT\tPRIVATE_REFERENCE_ONLY_ref\t1\n"
    ).encode("utf-8")
    payload = gzip.compress(rows, mtime=0)
    with tarfile.open(path, "w") as archive:
        for index in (1, 2):
            info = tarfile.TarInfo(f"PRIVATE_PROCESSED_MEMBER_{index}.txt.gz")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))


def _synthetic_assets(tmp_path: Path) -> tuple[dict[str, Path], dict]:
    protocol = _protocol()
    inputs = protocol["ordinary_public_input_contract"]
    paths = {
        key: tmp_path / value["required_basename"] for key, value in inputs.items()
    }
    _write_stability(
        paths["stability_table"],
        inputs["stability_table"]["required_header_names_exactly"],
    )
    _write_fasta(paths["author_reference_fasta"])
    _write_soft(paths["geo_family_soft"])
    _write_inventory(paths["geo_file_inventory"])
    _write_reference_archive(paths["gse186455_processed_archive"])
    for key, path in paths.items():
        inputs[key]["byte_count"] = path.stat().st_size
        inputs[key]["sha256"] = _hash(path)
    geometry = PREFLIGHT._build_actual_geometry(protocol, paths)
    protocol["expected_aggregate_replay"] = geometry
    return paths, protocol


def test_protocol_freezes_dec027_exact7_zero_credit_and_no_row_level_authority() -> None:
    protocol = _protocol()
    assert tuple(protocol["gate_contract"]["gate_ids_exactly"]) == PREFLIGHT.GATE_IDS
    assert protocol["fresh_baseline"]["dec027_authority_commit"] == PREFLIGHT.AUTHORITY_COMMIT
    assert protocol["fresh_baseline"]["bound_runtime_event"] == "A1-EVT-059"
    assert protocol["implementation_binding"]["runtime_group"]["frozen_i1_commit"] == PREFLIGHT.RUNTIME_I1_COMMIT
    assert protocol["implementation_binding"]["runtime_group"]["implementation_commit"] == PREFLIGHT.RUNTIME_I2_COMMIT
    assert protocol["implementation_binding"]["runtime_group"]["binding_commit"] == PREFLIGHT.RUNTIME_B_COMMIT
    assert protocol["implementation_binding"]["gse217518_predecessor"] == {
        "status": PREFLIGHT.BOUND,
        "append_only_history": list(PREFLIGHT.GSE217_HISTORY),
        "terminal_binding_commit": PREFLIGHT.GSE217_FINAL_B,
    }
    assert protocol["implementation_binding"]["encsr854ruf_predecessor"] == {
        "status": PREFLIGHT.BOUND,
        "append_only_history": list(PREFLIGHT.ENCSR_HISTORY),
        "terminal_binding_commit": PREFLIGHT.ENCSR_FINAL_B,
    }
    expected_future_histories = {
        "gse232572_predecessor": (
            ["I1", "B1"],
            "0f2c00868b6581edd9a429c7a8a67bb43f6b7776",
        ),
        "gse113849_predecessor": (
            ["I1", "B1"],
            "6372ddcb4b006d587a40ce628f9e193324c28b17",
        ),
        "gse269595_predecessor": (
            ["I1", "I2", "B2"],
            "19ca49229c9ff2814bad2c58b8b84be14624b7ea",
        ),
    }
    for name, (steps, terminal) in expected_future_histories.items():
        group = protocol["implementation_binding"][name]
        assert group["status"] == PREFLIGHT.BOUND
        assert [entry["step"] for entry in group["append_only_history"]] == steps
        assert group["terminal_binding_commit"] == terminal
        assert group["append_only_history"][-1]["exact_changed_paths"] == (
            group["binding_exact_changed_paths"]
        )
    own = protocol["implementation_binding"]["own_preflight_group"]
    assert own["status"] == PREFLIGHT.UNKNOWN
    assert {own[field] for field in own["unknown_to_bound_fields"]} == {
        PREFLIGHT.UNKNOWN
    }
    assert protocol["decision_boundary"]["row_level_qualification_execution_allowed"] is False
    assert protocol["scientific_state"]["gse295080_contribution"] == {
        "ordinary": 0,
        "a1": 0,
        "true_a2": 0,
        "canonical_records": 0,
    }
    assert protocol["frozen_gate_snapshot"]["normalized_gate_counts"] == {
        "PASS": 3,
        "PARTIAL_OR_CONDITIONAL": 1,
        "FAIL": 1,
        "UNKNOWN_NOT_ASSERTED": 1,
        "BLOCKED_OR_STOP": 1,
        "TOTAL": 7,
    }


def test_production_stops_before_asset_loader_or_output_path_inspection(tmp_path: Path) -> None:
    calls = {"git": 0, "asset": 0, "output": 0}

    def poison(name):
        def callback(*args, **kwargs):
            calls[name] += 1
            raise AssertionError(f"{name} crossed grouped UNKNOWN barrier")

        return callback

    asset_dir = tmp_path / "must-not-be-inspected-assets"
    output_dir = tmp_path / "must-not-be-inspected-output"
    with mock.patch.object(
        PREFLIGHT, "_audit_bound_repository", poison("git")
    ), mock.patch.object(
        PREFLIGHT, "_build_actual_geometry", poison("asset")
    ), mock.patch.object(PREFLIGHT, "_write_report", poison("output")):
        with pytest.raises(PREFLIGHT.ActivationBlocked, match="GROUPED_BINDINGS_UNKNOWN"):
            PREFLIGHT.execute_production(
                protocol_path=CONFIG_PATH,
                asset_dir=asset_dir,
                output_dir=output_dir,
                recorded_at="2026-08-15T02:25:00+08:00",
            )
    assert calls == {"git": 0, "asset": 0, "output": 0}
    assert not asset_dir.exists()
    assert not output_dir.exists()


def test_later_grouped_unknown_also_stops_before_git_asset_or_output(
    tmp_path: Path,
) -> None:
    protocol = copy.deepcopy(_protocol())
    for name in ("gse113849_predecessor", "gse269595_predecessor"):
        group = protocol["implementation_binding"][name]
        for field in PREFLIGHT.FUTURE_PREDECESSOR_FIELDS:
            group[field] = PREFLIGHT.UNKNOWN
    PREFLIGHT.validate_protocol(protocol)
    calls = {"git": 0, "asset": 0, "output": 0}

    def poison(name):
        def callback(*args, **kwargs):
            calls[name] += 1
            raise AssertionError(f"{name} crossed later grouped UNKNOWN barrier")

        return callback

    asset_dir = tmp_path / "must-not-be-inspected-assets"
    output_dir = tmp_path / "must-not-be-inspected-output"
    with mock.patch.object(
        PREFLIGHT, "load_protocol", return_value=protocol
    ), mock.patch.object(
        PREFLIGHT, "_audit_bound_repository", poison("git")
    ), mock.patch.object(
        PREFLIGHT, "_build_actual_geometry", poison("asset")
    ), mock.patch.object(PREFLIGHT, "_write_report", poison("output")):
        with pytest.raises(PREFLIGHT.ActivationBlocked, match="GROUPED_BINDINGS_UNKNOWN"):
            PREFLIGHT.execute_production(
                protocol_path=CONFIG_PATH,
                asset_dir=asset_dir,
                output_dir=output_dir,
                recorded_at="2026-08-15T02:25:00+08:00",
            )
    assert calls == {"git": 0, "asset": 0, "output": 0}
    assert not asset_dir.exists()
    assert not output_dir.exists()


def test_partial_unknown_binding_is_rejected() -> None:
    protocol = copy.deepcopy(_protocol())
    protocol["implementation_binding"]["gse217518_predecessor"][
        "terminal_binding_commit"
    ] = "1" * 40
    with pytest.raises(PREFLIGHT.ProtocolError, match="frozen append-only history"):
        PREFLIGHT.validate_protocol(protocol)

    protocol = copy.deepcopy(_protocol())
    gse232 = protocol["implementation_binding"]["gse232572_predecessor"]
    gse232["status"] = PREFLIGHT.UNKNOWN
    gse232["append_only_history"] = []
    gse232["terminal_binding_commit"] = PREFLIGHT.UNKNOWN
    with pytest.raises(PREFLIGHT.ProtocolError, match="partially bound"):
        PREFLIGHT.validate_protocol(protocol)


def test_config_only_successor_can_bind_all_groups_and_production_permissions() -> None:
    protocol = _bound_protocol()
    assert protocol["implementation_binding"]["authority_group"]["status"] == PREFLIGHT.BOUND
    assert protocol["implementation_binding"]["runtime_group"]["runtime_event_id"] == "A1-EVT-059"
    assert protocol["implementation_binding"]["gse269595_predecessor"][
        "terminal_binding_commit"
    ] == "19ca49229c9ff2814bad2c58b8b84be14624b7ea"
    assert protocol["implementation_binding"]["own_preflight_group"][
        "implementation_commit"
    ] == "9" * 40
    assert protocol["decision_boundary"]["production_preflight_execution_allowed"] is True


def test_future_history_allows_repairs_but_every_binding_is_config_only() -> None:
    protocol = _bound_protocol()
    gse269 = protocol["implementation_binding"]["gse269595_predecessor"]
    assert [step["step"] for step in gse269["append_only_history"]] == [
        "I1",
        "I2",
        "B2",
    ]
    assert gse269["append_only_history"][-1]["exact_changed_paths"] == [
        PREFLIGHT.GSE269_CONFIG_PATH
    ]
    invalid = copy.deepcopy(protocol)
    invalid["implementation_binding"]["gse269595_predecessor"][
        "append_only_history"
    ][-1]["exact_changed_paths"] = list(PREFLIGHT.GSE269_EXACT3)
    with pytest.raises(PREFLIGHT.ProtocolError, match="history changed paths differ"):
        PREFLIGHT.validate_protocol(invalid)


def test_clean_normalised_disk_i_and_legal_disk_b_are_accepted() -> None:
    bound = _bound_protocol()
    implementation_i = PREFLIGHT._normalise_own_binding(bound)
    PREFLIGHT.validate_protocol(implementation_i)
    assert {
        implementation_i["implementation_binding"]["own_preflight_group"][field]
        for field in PREFLIGHT.OWN_BINDING_FIELDS
    } == {PREFLIGHT.UNKNOWN}
    with pytest.raises(PREFLIGHT.ActivationBlocked, match="GROUPED_BINDINGS_UNKNOWN"):
        PREFLIGHT._require_all_bindings(implementation_i)
    PREFLIGHT._require_all_bindings(bound)


def test_single_entry_has_no_loader_or_local_candidate_bypass() -> None:
    assert set(inspect.signature(PREFLIGHT.execute_production).parameters) == {
        "protocol_path",
        "asset_dir",
        "output_dir",
        "recorded_at",
    }
    assert not hasattr(PREFLIGHT, "execute_local_candidate")
    parser_actions = {action.dest for action in PREFLIGHT._parser()._actions}
    assert "asset_dir" in parser_actions
    assert "mode" not in parser_actions
    assert "asset_loader" not in parser_actions
    assert "repository_root" not in parser_actions


def _repository_audit_fixture(tmp_path: Path, *, stale_copy: bool):
    protocol = _bound_protocol()
    repo_root = tmp_path / "repo"
    script_path = repo_root / PREFLIGHT.SCRIPT_REPO_PATH
    test_path = repo_root / PREFLIGHT.TEST_REPO_PATH
    protocol_path = repo_root / PREFLIGHT.CONFIG_REPO_PATH
    for path in (script_path, test_path, protocol_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_bytes(MODULE_PATH.read_bytes())
    test_path.write_bytes(Path(__file__).read_bytes())
    disk_protocol = copy.deepcopy(protocol)
    disk_protocol["repository_authority"]["production_repo_root"] = str(repo_root)
    protocol_path.write_text(
        json.dumps(disk_protocol, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    implementation_bytes = (
        json.dumps(
            PREFLIGHT._normalise_own_binding(disk_protocol),
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")
    head = "e" * 40
    own_i = "9" * 40
    verified = []

    def fake_run_git(_repo_root, *arguments):
        return {
            ("rev-parse", "HEAD"): head,
            ("rev-parse", "@{upstream}"): head,
            ("rev-parse", "--abbrev-ref", "HEAD"): PREFLIGHT.PRODUCTION_BRANCH,
            ("rev-parse", "--abbrev-ref", "@{upstream}"): PREFLIGHT.PRODUCTION_UPSTREAM,
            ("status", "--porcelain=v1", "--untracked-files=all"): "",
        }[arguments]

    def fake_verify(_repo_root, **kwargs):
        verified.append(
            (
                kwargs["label"],
                kwargs["commit"],
                kwargs["expected_parent"],
                tuple(kwargs["expected_paths"]),
            )
        )

    def fake_blob(_repo_root, commit, path):
        if commit == own_i and path == PREFLIGHT.CONFIG_REPO_PATH:
            return implementation_bytes
        if commit == head and path == PREFLIGHT.CONFIG_REPO_PATH:
            return protocol_path.read_bytes()
        if commit == own_i and path == PREFLIGHT.SCRIPT_REPO_PATH:
            return script_path.read_bytes()
        if commit == own_i and path == PREFLIGHT.TEST_REPO_PATH:
            return test_path.read_bytes()
        raise AssertionError(f"unexpected blob request: {commit}:{path}")

    executing = MODULE_PATH if stale_copy else script_path
    with mock.patch.object(PREFLIGHT, "_run_git", fake_run_git), mock.patch.object(
        PREFLIGHT, "_live_origin_head", return_value=head
    ), mock.patch.object(
        PREFLIGHT, "_verify_frozen_commit", fake_verify
    ), mock.patch.object(
        PREFLIGHT, "_git_blob", fake_blob
    ), mock.patch.object(
        PREFLIGHT, "__file__", str(executing)
    ):
        if stale_copy:
            with pytest.raises(PREFLIGHT.ActivationBlocked, match="STALE_COPY"):
                PREFLIGHT._audit_bound_repository(
                    disk_protocol, protocol_path, repo_root
                )
            return verified
        result = PREFLIGHT._audit_bound_repository(
            disk_protocol, protocol_path, repo_root
        )
    assert result["binding_commit"] == head
    return verified


def test_full_legal_disk_i_b_chain_audits_in_exact_order(tmp_path: Path) -> None:
    verified = _repository_audit_fixture(tmp_path, stale_copy=False)
    assert [item[0] for item in verified] == [
        "DEC027_AUTHORITY_A",
        "DEC027_RUNTIME_I1",
        "DEC027_RUNTIME_I2",
        "DEC027_RUNTIME_B2",
        "GSE217518_I1",
        "GSE217518_I2",
        "GSE217518_B2",
        "GSE217518_I3",
        "GSE217518_B3",
        "ENCSR854RUF_I1",
        "ENCSR854RUF_I2",
        "ENCSR854RUF_B2",
        "ENCSR854RUF_I3",
        "ENCSR854RUF_B3",
        "ENCSR854RUF_I4",
        "ENCSR854RUF_B4",
        "GSE232572_I1",
        "GSE232572_B1",
        "GSE113849_I1",
        "GSE113849_B1",
        "GSE269595_I1",
        "GSE269595_I2",
        "GSE269595_B2",
        "GSE295080_I",
        "GSE295080_B",
    ]
    assert verified[4][2] == PREFLIGHT.RUNTIME_B_COMMIT
    assert verified[9][2] == PREFLIGHT.GSE217_FINAL_B
    assert verified[16][2] == PREFLIGHT.ENCSR_FINAL_B
    assert verified[18][2] == "0f2c00868b6581edd9a429c7a8a67bb43f6b7776"
    assert verified[20][2] == "6372ddcb4b006d587a40ce628f9e193324c28b17"
    assert verified[23][2] == "19ca49229c9ff2814bad2c58b8b84be14624b7ea"
    assert verified[24][2] == "9" * 40
    assert verified[24][3] == (PREFLIGHT.CONFIG_REPO_PATH,)


def test_stale_executing_copy_is_rejected_before_asset_read(tmp_path: Path) -> None:
    assert len(_repository_audit_fixture(tmp_path, stale_copy=True)) == 25


def test_all_five_asset_identities_are_checked_before_any_parse(tmp_path: Path) -> None:
    protocol = _protocol()
    contracts = protocol["ordinary_public_input_contract"]
    paths = {
        key: tmp_path / contracts[key]["required_basename"] for key in PREFLIGHT.INPUT_KEYS
    }
    verified = []
    parse_count = 0

    def fake_verify(path, contract, *, label):
        verified.append(label)
        if len(verified) == len(PREFLIGHT.INPUT_KEYS):
            raise PREFLIGHT.AssetError("fifth frozen identity differs")

    def forbidden_parse(*args, **kwargs):
        nonlocal parse_count
        parse_count += 1
        raise AssertionError("parse crossed all-five identity barrier")

    with mock.patch.object(PREFLIGHT, "_verify_asset", fake_verify), mock.patch.object(
        PREFLIGHT, "_audit_stability", forbidden_parse
    ):
        with pytest.raises(PREFLIGHT.AssetError, match="fifth frozen identity"):
            PREFLIGHT._build_actual_geometry(protocol, paths)
    assert verified == list(PREFLIGHT.INPUT_KEYS)
    assert parse_count == 0


def test_synthetic_replay_closes_library1_reuse_and_library2_nonoverlap(tmp_path: Path) -> None:
    paths, protocol = _synthetic_assets(tmp_path)
    geometry = PREFLIGHT._build_actual_geometry(protocol, paths)
    assert geometry["library1"]["unique_design_count"] == 2
    assert geometry["library1_exact_name_overlap_count"] == 2
    assert geometry["library1_gse295080_only_count"] == 0
    assert geometry["library1_gse186455_only_count"] == 1
    assert geometry["library2_exact_name_overlap_count"] == 0
    assert geometry["library2_gse295080_only_count"] == 2
    assert geometry["first_two_gse186455_unique_element_base_count_each"] == [3, 3]
    assert geometry["first_two_gse186455_element_base_intersection_count"] == 3
    assert geometry["first_gse186455_member_only_element_base_count"] == 0
    assert geometry["second_gse186455_member_only_element_base_count"] == 0


def test_synthetic_fixture_builds_one_aggregate_terminal_record_without_payload(tmp_path: Path) -> None:
    asset_root = tmp_path / "assets"
    asset_root.mkdir()
    paths, protocol = _synthetic_assets(asset_root)
    report = PREFLIGHT._build_report(
        protocol,
        PREFLIGHT._build_actual_geometry(protocol, paths),
        "2026-08-15T02:15:00+08:00",
    )
    output_dir = tmp_path / "terminal"
    output = PREFLIGHT._write_report(report, output_dir)
    assert output.is_file()
    assert list(output_dir.iterdir()) == [output]
    assert report["p0"]["status"] == "FAIL_CLOSED_STOP"
    assert report["p1"]["row_level_status"] == "NOT_AUTHORIZED_NOT_RUN"
    assert report["terminal_disposition"]["row_level_successor_authority_request"] == "DO_NOT_REQUEST_ON_CURRENT_EVIDENCE"
    serialized = output.read_text(encoding="utf-8")
    for secret in (
        "PRIVATE_L1_A",
        "PRIVATE_L2_A",
        "PRIVATE_BC",
        "PRIVATE_ACCESSION",
        "PRIVATE_PROCESSED_MEMBER",
        "AAAA",
    ):
        assert secret not in serialized


def test_replicate_labels_never_become_independence_or_standard_error(tmp_path: Path) -> None:
    paths, protocol = _synthetic_assets(tmp_path)
    report = PREFLIGHT._build_report(
        protocol,
        PREFLIGHT._build_actual_geometry(protocol, paths),
        "2026-08-15T02:15:00+08:00",
    )
    boundary = report["replicate_label_boundary"]
    assert boundary["official_metadata_label_geometry_closed"] is True
    assert boundary["labels_establish_biological_independence"] is False
    assert boundary["labels_establish_valid_standard_error"] is False
    statuses = {gate["gate_id"]: gate["normalized_status"] for gate in report["scientific_gates"]}
    assert statuses["BIOLOGICAL_REPLICATE_LABEL_GEOMETRY_CLOSED"] == "PASS"
    assert statuses["INDEPENDENT_STUDY_OR_REUSED_LIBRARY_BOUNDARY_CLOSED"] == "FAIL"


def test_asset_identity_tamper_stops_before_replay(tmp_path: Path) -> None:
    paths, protocol = _synthetic_assets(tmp_path)
    with paths["stability_table"].open("a", encoding="utf-8") as handle:
        handle.write("tamper\n")
    with pytest.raises(PREFLIGHT.AssetError, match="byte count differs"):
        PREFLIGHT._build_actual_geometry(protocol, paths)


def test_aggregate_output_guard_rejects_member_level_keys() -> None:
    with pytest.raises(PREFLIGHT.OutputError, match="forbidden member-level key"):
        PREFLIGHT._assert_finite({"aggregate": {"member_id": "PRIVATE"}})


def test_all_seven_are_not_pass_and_credit_remains_zero() -> None:
    protocol = _protocol()
    snapshot = protocol["frozen_gate_snapshot"]
    assert snapshot["normalized_gate_counts"]["PASS"] == 3
    assert snapshot["normalized_gate_counts"]["TOTAL"] == 7
    assert snapshot["p0_status"] == "FAIL_CLOSED_STOP"
    assert snapshot["p1_row_level_status"] == "NOT_AUTHORIZED_NOT_RUN"
    assert protocol["aggregate_output_contract"]["row_level_successor_authority_request_eligible"] is False
    assert protocol["scientific_state"]["qualified"] is False


def test_existing_exact7_aggregate_record_preserves_stop_and_zero_credit() -> None:
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    PREFLIGHT._assert_finite(report)
    assert report["normalized_gate_counts"] == PREFLIGHT.EXPECTED_STATUS_COUNTS
    assert report["p0"]["status"] == "FAIL_CLOSED_STOP"
    assert report["p1"]["row_level_status"] == "NOT_AUTHORIZED_NOT_RUN"
    assert report["terminal_disposition"]["verdict"] == (
        "STOP_NO_INDEPENDENT_CREDIT_AND_NO_ROW_LEVEL_AUTHORITY_REQUEST"
    )
    assert report["scientific_state"]["gse295080_contribution"] == {
        "ordinary": 0,
        "a1": 0,
        "true_a2": 0,
        "canonical_records": 0,
    }


def test_fixed_public_asset_identities_are_exact() -> None:
    protocol = _protocol()
    contracts = protocol["ordinary_public_input_contract"]
    assert {
        value["required_basename"]: (value["byte_count"], value["sha256"])
        for value in contracts.values()
    } == PREFLIGHT.PUBLIC_ASSET_IDENTITIES


def test_single_aggregate_publication_is_atomic_no_replace_and_idempotent(
    tmp_path: Path,
) -> None:
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    output_dir = tmp_path / "terminal"
    output = PREFLIGHT._write_report(report, output_dir)
    assert output.name == PREFLIGHT.REPORT_FILENAME
    assert list(output_dir.iterdir()) == [output]
    assert PREFLIGHT._write_report(report, output_dir) == output
    different = copy.deepcopy(report)
    different["recorded_at"] = "2026-08-15T02:25:01+08:00"
    with pytest.raises(PREFLIGHT.OutputError, match="different aggregate report"):
        PREFLIGHT._write_report(different, output_dir)
