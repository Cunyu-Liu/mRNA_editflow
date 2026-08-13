from __future__ import annotations

import copy
import gzip
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = (
    ROOT
    / "configs"
    / "route_a_v3_gse256185_aggregate_row_level_qualification_preflight_v1.json"
)
MODULE_PATH = (
    ROOT
    / "scripts"
    / "route_a_v3"
    / "preflight_gse256185_aggregate_row_level_qualification.py"
)
SPEC = importlib.util.spec_from_file_location("gse256185_dec022_preflight", MODULE_PATH)
assert SPEC and SPEC.loader
PREFLIGHT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PREFLIGHT
SPEC.loader.exec_module(PREFLIGHT)

OFFICIAL_TSV = Path(
    "/mnt/cunyuliu/mrna_xeditflow_routea_v3/raw/GSE256185/"
    "GSE256185_CPMandRRS_VCE_Var.tsv.gz"
)
OFFICIAL_FASTA = Path(
    "/mnt/cunyuliu/mrna_xeditflow_routea_v3/raw/GSE256185/"
    "GSE256185_DNAPool_ref.fa.gz"
)


def _protocol() -> dict[str, object]:
    value = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    PREFLIGHT._validate_protocol(value)
    return value


def _i_protocol() -> dict[str, object]:
    value = copy.deepcopy(_protocol())
    binding = value["implementation_binding"]
    for key in PREFLIGHT.UNKNOWN_BINDING_SCALARS:
        binding[key] = PREFLIGHT.UNKNOWN
    PREFLIGHT._validate_protocol(value)
    return value


def _bound_protocol() -> dict[str, object]:
    value = _i_protocol()
    binding = value["implementation_binding"]
    binding["status"] = PREFLIGHT.BOUND
    binding["implementation_commit"] = "2" * 40
    binding["implementation_script_sha256"] = "3" * 64
    binding["implementation_test_sha256"] = "4" * 64
    PREFLIGHT._validate_protocol(value)
    return value


def _write_protocol(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return path


def _disk_lifecycle_state() -> str:
    protocol = _protocol()
    authority = protocol["implementation_binding"]["authority_group"]
    assert authority["status"] == PREFLIGHT.BOUND
    assert authority["authority_commit"] == PREFLIGHT.AUTHORITY_COMMIT
    assert authority["authority_runtime_binding_commit"] == PREFLIGHT.RUNTIME_B_COMMIT
    binding = protocol["implementation_binding"]
    values = [binding[key] for key in PREFLIGHT.UNKNOWN_BINDING_SCALARS]
    if values == [PREFLIGHT.UNKNOWN] * 4:
        return "I"
    assert binding["status"] == PREFLIGHT.BOUND
    assert PREFLIGHT.HEX40_RE.fullmatch(binding["implementation_commit"])
    assert hashlib.sha256(MODULE_PATH.read_bytes()).hexdigest() == binding[
        "implementation_script_sha256"
    ]
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == binding[
        "implementation_test_sha256"
    ]
    return "B"


def _official_observation() -> dict[str, object]:
    protocol = _protocol()
    universe = {
        key: copy.deepcopy(value)
        for key, value in protocol["candidate_universe_contract"].items()
        if key not in {"selection_rule", "count_or_rule_drift_action"}
    }
    return {
        "candidate_universe": universe,
        **copy.deepcopy(protocol["expected_aggregate_observation"]),
        "internal_access_attestation": {
            "ordinary_public_assets_read_count": 2,
            "candidate_universe_closed_before_row_level_field_access": True,
            "row_level_values_persisted_count": 0,
            "row_level_values_serialized_count": 0,
            "private_or_restricted_input_read_count": 0,
            "sealed_contact_count": 0,
            "gse246381_contact_count": 0,
        },
    }


def _fixture_binding(*args: object) -> dict[str, str]:
    return {
        "status": "TEST_FIXTURE_BOUND_WITHOUT_GIT",
        "authority_commit": PREFLIGHT.AUTHORITY_COMMIT,
        "authority_runtime_binding_commit": PREFLIGHT.RUNTIME_B_COMMIT,
        "implementation_commit": "2" * 40,
        "binding_commit": "3" * 40,
    }


def _fixture_identity(*args: object) -> dict[str, object]:
    result: dict[str, object] = {}
    for key in ("processed_tsv", "reference_fasta"):
        asset = _protocol()["official_public_assets"][key]
        result[key] = {
            "filename": asset["filename"],
            "compressed_bytes": asset["compressed_bytes"],
            "compressed_sha256": asset["compressed_sha256"],
            "identity_status": "PASS_FROZEN_ORDINARY_PUBLIC_ASSET",
        }
    return result


def test_protocol_supports_real_i_or_b_and_freezes_dec022_chain() -> None:
    protocol = _protocol()
    assert _disk_lifecycle_state() in {"I", "B"}
    binding = protocol["implementation_binding"]
    authority = binding["authority_group"]
    assert authority["authority_expected_parent"] == PREFLIGHT.AUTHORITY_PARENT
    assert tuple(authority["authority_exact_changed_paths"]) == PREFLIGHT.AUTHORITY_EXACT10
    runtime = authority["authority_runtime_lifecycle"]
    assert runtime["implementation_commit"] == PREFLIGHT.RUNTIME_I_COMMIT
    assert runtime["implementation_expected_parent"] == PREFLIGHT.AUTHORITY_COMMIT
    assert runtime["binding_commit"] == PREFLIGHT.RUNTIME_B_COMMIT
    assert runtime["binding_expected_parent"] == PREFLIGHT.RUNTIME_I_COMMIT
    assert tuple(binding["implementation_commit_exact_changed_paths"]) == PREFLIGHT.EXPECTED_EXACT3
    assert binding["binding_commit_exact_changed_paths"] == [PREFLIGHT.CONFIG_PATH]

    synthetic_b = _bound_protocol()
    assert synthetic_b["implementation_binding"]["status"] == PREFLIGHT.BOUND


def test_partial_authority_or_implementation_groups_are_rejected() -> None:
    protocol = _protocol()
    protocol["implementation_binding"]["authority_group"]["authority_commit"] = PREFLIGHT.UNKNOWN
    with pytest.raises(PREFLIGHT.ProtocolError, match="authority group"):
        PREFLIGHT._validate_protocol(protocol)

    protocol = _i_protocol()
    protocol["implementation_binding"]["implementation_commit"] = "2" * 40
    with pytest.raises(PREFLIGHT.ProtocolError, match="partial implementation"):
        PREFLIGHT._validate_protocol(protocol)


def test_default_binding_auditor_checks_a_runtime_i_b_and_preflight_i_b(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    protocol = _bound_protocol()
    monkeypatch.setattr(PREFLIGHT, "PRODUCTION_REPO_ROOT", str(repo))
    protocol["repository_authority"]["production_repo_root"] = str(repo)
    binding = protocol["implementation_binding"]
    script_blob = b"bound producer\n"
    test_blob = b"bound focused test\n"
    binding["implementation_script_sha256"] = hashlib.sha256(script_blob).hexdigest()
    binding["implementation_test_sha256"] = hashlib.sha256(test_blob).hexdigest()
    PREFLIGHT._validate_protocol(protocol)

    i_protocol = PREFLIGHT._normalise_binding(protocol)
    i_payload = (json.dumps(i_protocol, indent=2) + "\n").encode()
    b_payload = (json.dumps(protocol, indent=2) + "\n").encode()
    protocol_path = repo / PREFLIGHT.CONFIG_PATH
    protocol_path.parent.mkdir(parents=True)
    protocol_path.write_bytes(b_payload)
    script_path = repo / PREFLIGHT.SCRIPT_PATH
    script_path.parent.mkdir(parents=True)
    script_path.write_bytes(script_blob)
    test_path = repo / PREFLIGHT.TEST_PATH
    test_path.parent.mkdir(parents=True)
    test_path.write_bytes(test_blob)

    implementation = binding["implementation_commit"]
    bound = "3" * 40
    verified: list[tuple[str, str, str, tuple[str, ...]]] = []

    def fake_verify(
        root: Path,
        *,
        label: str,
        commit: str,
        expected_parent: str,
        expected_paths: tuple[str, ...],
        expected_blobs: object = None,
    ) -> None:
        assert root == repo
        verified.append((label, commit, expected_parent, tuple(expected_paths)))

    def fake_run(root: Path, *args: str) -> str:
        assert root == repo
        mapping = {
            ("rev-parse", "HEAD"): bound,
            ("rev-parse", "@{upstream}"): bound,
            (
                "rev-parse",
                "--verify",
                f"refs/remotes/origin/{PREFLIGHT.PRODUCTION_BRANCH}",
            ): bound,
            ("rev-parse", "--abbrev-ref", "HEAD"): PREFLIGHT.PRODUCTION_BRANCH,
            ("rev-parse", "--abbrev-ref", "@{upstream}"): (
                f"origin/{PREFLIGHT.PRODUCTION_BRANCH}"
            ),
            ("status", "--porcelain=v1", "--untracked-files=all"): "",
            ("rev-parse", f"{implementation}^"): PREFLIGHT.RUNTIME_B_COMMIT,
            ("rev-parse", f"{bound}^"): implementation,
        }
        return mapping[args]

    def fake_changed(root: Path, commit: str) -> tuple[str, ...]:
        assert root == repo
        return (
            tuple(sorted(PREFLIGHT.EXPECTED_EXACT3))
            if commit == implementation
            else (PREFLIGHT.CONFIG_PATH,)
        )

    def fake_blob(root: Path, commit: str, path: str) -> bytes:
        assert root == repo
        return {
            (implementation, PREFLIGHT.CONFIG_PATH): i_payload,
            (implementation, PREFLIGHT.SCRIPT_PATH): script_blob,
            (implementation, PREFLIGHT.TEST_PATH): test_blob,
            (bound, PREFLIGHT.CONFIG_PATH): b_payload,
        }[(commit, path)]

    monkeypatch.setattr(PREFLIGHT, "_verify_frozen_commit", fake_verify)
    monkeypatch.setattr(PREFLIGHT, "_run_git", fake_run)
    monkeypatch.setattr(PREFLIGHT, "_changed_paths", fake_changed)
    monkeypatch.setattr(PREFLIGHT, "_git_blob", fake_blob)
    monkeypatch.setattr(PREFLIGHT, "__file__", str(script_path))
    result = PREFLIGHT._default_binding_auditor(
        protocol, protocol_path, b_payload, repo
    )
    assert verified == [
        ("DEC022 authority A", PREFLIGHT.AUTHORITY_COMMIT, PREFLIGHT.AUTHORITY_PARENT, PREFLIGHT.AUTHORITY_EXACT10),
        ("DEC022 runtime I", PREFLIGHT.RUNTIME_I_COMMIT, PREFLIGHT.AUTHORITY_COMMIT, PREFLIGHT.RUNTIME_EXACT3),
        ("DEC022 runtime B", PREFLIGHT.RUNTIME_B_COMMIT, PREFLIGHT.RUNTIME_I_COMMIT, (PREFLIGHT.RUNTIME_CONFIG_PATH,)),
    ]
    assert result["status"].startswith("BOUND_DEC022_AUTHORITY_RUNTIME")
    assert result["binding_commit"] == bound

    copied_producer = tmp_path / "stale-copy" / "preflight.py"
    copied_producer.parent.mkdir()
    copied_producer.write_bytes(b"older producer bytes\n")
    monkeypatch.setattr(PREFLIGHT, "__file__", str(copied_producer))
    calls = {"asset": 0, "aggregate": 0}

    def forbidden_asset(*args: object) -> dict[str, object]:
        calls["asset"] += 1
        raise AssertionError("asset auditor must not run for a copied producer")

    def forbidden_aggregate(*args: object) -> dict[str, object]:
        calls["aggregate"] += 1
        raise AssertionError("asset body must not be read for a copied producer")

    with pytest.raises(PREFLIGHT.ProtocolError, match="executing producer"):
        PREFLIGHT.execute(
            protocol_path,
            tmp_path / "missing.tsv.gz",
            tmp_path / "missing.fa.gz",
            tmp_path / "must-not-exist",
            repo_root=repo,
            asset_identity_auditor=forbidden_asset,
            aggregator=forbidden_aggregate,
        )
    assert calls == {"asset": 0, "aggregate": 0}
    assert not (tmp_path / "must-not-exist").exists()

    def dirty_run(root: Path, *args: str) -> str:
        if args == ("status", "--porcelain=v1", "--untracked-files=all"):
            return " M configs/unrelated.json"
        return fake_run(root, *args)

    monkeypatch.setattr(PREFLIGHT, "_run_git", dirty_run)
    monkeypatch.setattr(PREFLIGHT, "__file__", str(script_path))
    with pytest.raises(PREFLIGHT.ProtocolError, match="dirty"):
        PREFLIGHT._default_binding_auditor(protocol, protocol_path, b_payload, repo)


def test_unknown_implementation_stops_before_assets_or_output(tmp_path: Path) -> None:
    calls = {"asset": 0, "aggregate": 0}

    def forbidden_asset(*args: object) -> dict[str, object]:
        calls["asset"] += 1
        raise AssertionError("asset identity must not be inspected")

    def forbidden_aggregate(*args: object) -> dict[str, object]:
        calls["aggregate"] += 1
        raise AssertionError("asset body must not be inspected")

    path = _write_protocol(tmp_path / "repo" / PREFLIGHT.CONFIG_PATH, _i_protocol())
    output = tmp_path / "must-not-exist"
    with pytest.raises(PREFLIGHT.BindingNotFrozen, match="not BOUND"):
        PREFLIGHT.execute(
            path,
            tmp_path / "missing.tsv.gz",
            tmp_path / "missing.fa.gz",
            output,
            repo_root=tmp_path / "repo",
            asset_identity_auditor=forbidden_asset,
            aggregator=forbidden_aggregate,
        )
    assert calls == {"asset": 0, "aggregate": 0}
    assert not output.exists()


def test_sequence_edit_replay_and_endpoint_math() -> None:
    assert PREFLIGHT._publisher_transform("GCTAATACGACTCACTATAACCC") == "GCCC"
    assert PREFLIGHT.replay_edit("AAAAAAATG", "ATG", "win0") == ("DIRECT", True)
    assert PREFLIGHT.replay_edit("AAAAAAATG", "AGT", "win0") == (
        "PUBLISHER_ASSISTED",
        True,
    )
    assert PREFLIGHT.replay_edit("ATG", "CCCATG", "+1CCC") == ("DIRECT", True)
    assert PREFLIGHT.replay_edit("CCCATG", "ATG", "-1CCC") == ("DIRECT", True)
    assert PREFLIGHT.replay_edit("CCCATG", "AGT", "-1CCC") == (
        "PUBLISHER_ASSISTED",
        True,
    )
    assert PREFLIGHT.replay_edit("CCCATG", "AAT", "-1CCC")[0] == "UNEXPLAINED"
    assert PREFLIGHT.replay_edit("AAAA", "", "win0") == ("UNEXPLAINED", False)

    values = [22.51486017, 29.34298267, 21.13522429, 26.45138195, 63.74621344, 65.28337223, 70.4163538]
    result = PREFLIGHT.recompute_endpoint(values)
    assert result == pytest.approx(-1.4312419549108963, abs=1e-15)
    with pytest.raises(PREFLIGHT.ObservationError, match="undefined"):
        PREFLIGHT.recompute_endpoint([0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])


def test_tsv_parser_accepts_authorized_fields_but_report_whitelist_blocks_poison(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    poison_identifier = "ENSG1-ENST1-9.parent.ROW_POISON"
    poison_sequence = "SEQUENCE_POISON"
    asset = tmp_path / "fixture.tsv.gz"
    with gzip.open(asset, "wt", encoding="ascii", newline="") as handle:
        handle.write("\t".join(PREFLIGHT.EXPECTED_HEADER) + "\n")
        handle.write(
            "\t".join(
                [poison_identifier, "0", "1", "1", "1", "1", "1", "1", "1", poison_sequence]
            )
            + "\n"
        )
    rows, parsed = PREFLIGHT._load_tsv(asset)
    assert rows[0]["identifier"] == poison_identifier
    assert rows[0]["sequence"] == poison_sequence
    assert parsed["geometry"]["total_body_row_count"] == 1

    calls = {"fasta": 0}

    def forbidden_fasta(*args: object) -> dict[str, str]:
        calls["fasta"] += 1
        raise AssertionError("FASTA sequence must not be read before universe closure")

    monkeypatch.setattr(PREFLIGHT, "_read_fasta", forbidden_fasta)
    with pytest.raises(PREFLIGHT.ObservationError, match="before row-level"):
        PREFLIGHT.aggregate_assets(_protocol(), asset, tmp_path / "missing.fa.gz")
    assert calls["fasta"] == 0

    observation = _official_observation()
    observation["row_material_poison"] = {
        "identifier": poison_identifier,
        "sequence": poison_sequence,
        "effect": 123.0,
    }
    report = PREFLIGHT.build_report(
        _protocol(),
        observation,
        binding=_fixture_binding(),
        asset_identity=_fixture_identity(),
        recorded_at="2026-08-13T15:00:00Z",
    )
    serialized = json.dumps(report, sort_keys=True)
    assert poison_identifier not in serialized
    assert poison_sequence not in serialized
    assert "row_material_poison" not in serialized


def test_complete_report_has_exact17_fail_closed_gates_and_no_promotion(
    tmp_path: Path,
) -> None:
    path = _write_protocol(tmp_path / "repo" / PREFLIGHT.CONFIG_PATH, _bound_protocol())
    output = tmp_path / "output"
    report = PREFLIGHT.execute(
        path,
        tmp_path / "tsv",
        tmp_path / "fasta",
        output,
        repo_root=tmp_path / "repo",
        binding_auditor=_fixture_binding,
        asset_identity_auditor=_fixture_identity,
        aggregator=lambda *args: _official_observation(),
        recorded_at="2026-08-13T15:00:00Z",
    )
    assert report["status"] == PREFLIGHT.STATUS_STOP
    assert report["all_required_gates_pass"] is False
    assert report["qualified"] is False
    gates = report["required_gate_results"]
    assert tuple(gate["gate_id"] for gate in gates) == PREFLIGHT.GATE_IDS
    statuses = {gate["gate_id"]: gate["status"] for gate in gates}
    assert statuses[PREFLIGHT.GATE_IDS[3]] == "PARTIAL_FAIL_CURRENT_PROTOCOL"
    assert statuses[PREFLIGHT.GATE_IDS[8]] == "FAIL"
    assert statuses[PREFLIGHT.GATE_IDS[10]] == "NOT_RUN_FORMAL"
    assert statuses[PREFLIGHT.GATE_IDS[11]] == "INELIGIBLE_NOT_RUN"
    assert statuses[PREFLIGHT.GATE_IDS[13]] == (
        "CONDITIONAL_PENDING_ZERO_EXTERNAL_LEARNED_INPUT_RUNTIME_ATTESTATION"
    )
    assert report["terminal_truth"]["gse256185_contribution"] == {
        "ordinary": 0,
        "a1": 0,
        "true_a2": 0,
        "canonical_records": 0,
    }
    assert report["terminal_truth"]["training_allowed"] is False
    assert report["terminal_truth"]["model_selection_allowed"] is False
    assert report["terminal_truth"]["next_phase_authorized"] is False
    assert [entry.name for entry in output.iterdir()] == [PREFLIGHT.REPORT_FILENAME]


def test_asset_drift_stops_before_decompression_aggregation_or_output(
    tmp_path: Path,
) -> None:
    path = _write_protocol(tmp_path / "repo" / PREFLIGHT.CONFIG_PATH, _bound_protocol())
    tsv = tmp_path / "GSE256185_CPMandRRS_VCE_Var.tsv.gz"
    tsv.write_bytes(b"drift")
    calls = {"aggregate": 0}

    def forbidden(*args: object) -> dict[str, object]:
        calls["aggregate"] += 1
        raise AssertionError("aggregation must not run")

    output = tmp_path / "must-not-exist"
    with pytest.raises(PREFLIGHT.AssetIdentityError, match="byte count"):
        PREFLIGHT.execute(
            path,
            tsv,
            tmp_path / "GSE256185_DNAPool_ref.fa.gz",
            output,
            repo_root=tmp_path / "repo",
            binding_auditor=_fixture_binding,
            aggregator=forbidden,
        )
    assert calls["aggregate"] == 0
    assert not output.exists()


def test_atomic_failure_leaves_no_final_or_temporary_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "output"

    def fail_after_partial(path: Path, payload: bytes) -> None:
        path.write_bytes(payload[:9])
        raise OSError("injected")

    monkeypatch.setattr(PREFLIGHT, "_write_temp_payload", fail_after_partial)
    with pytest.raises(PREFLIGHT.OutputError, match="cannot publish"):
        PREFLIGHT._write_exclusive(output, {"aggregate": True})
    assert list(output.iterdir()) == []


@pytest.mark.skipif(
    not OFFICIAL_TSV.is_file() or not OFFICIAL_FASTA.is_file(),
    reason="frozen production public assets are not mounted",
)
def test_frozen_public_assets_recompute_exact_aggregate_observation() -> None:
    protocol = _protocol()
    identity = PREFLIGHT._default_asset_identity_auditor(
        protocol, OFFICIAL_TSV, OFFICIAL_FASTA
    )
    assert identity["processed_tsv"]["identity_status"].startswith("PASS")
    observation = PREFLIGHT.aggregate_assets(protocol, OFFICIAL_TSV, OFFICIAL_FASTA)
    PREFLIGHT._validate_observation(protocol, observation)
    assert observation["endpoint_transform"][
        "maximum_absolute_formula_difference"
    ] == pytest.approx(2.7253181933417636e-9, abs=1e-20)
    assert observation["edit_replay"]["unexplained_count"] == 3
    assert observation["reject_closure"]["mutually_exclusive_row_reason_total"] == 11404
