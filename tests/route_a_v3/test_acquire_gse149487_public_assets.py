from __future__ import annotations

import copy
import hashlib
import importlib.util
import io
import json
import os
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts" / "route_a_v3" / "acquire_gse149487_public_assets.py"
DEFAULT_PROTOCOL_PATH = (
    ROOT
    / "configs"
    / "route_a_v3_gse149487_public_asset_acquisition_v1.json"
)
PROTOCOL_STATE_OVERRIDE = os.environ.get("GSE149487_TEST_PROTOCOL_STATE")
SPEC = importlib.util.spec_from_file_location("gse149487_public_asset_acquisition", MODULE_PATH)
assert SPEC and SPEC.loader
ACQUIRE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ACQUIRE
SPEC.loader.exec_module(ACQUIRE)


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


class MemoryTransport:
    def __init__(self, payload_by_url: dict[str, bytes]) -> None:
        self.payload_by_url = payload_by_url
        self.opened: list[str] = []

    def open(self, url: str, *, timeout_seconds: int) -> io.BytesIO:
        assert timeout_seconds == 120
        self.opened.append(url)
        return io.BytesIO(self.payload_by_url[url])


class ForbiddenTransport:
    def __init__(self) -> None:
        self.opened: list[str] = []

    def open(self, url: str, *, timeout_seconds: int) -> io.BytesIO:
        self.opened.append(url)
        raise AssertionError("network must not be reached")


def _disk_protocol() -> dict[str, object]:
    protocol = json.loads(DEFAULT_PROTOCOL_PATH.read_text(encoding="utf-8"))
    ACQUIRE._validate_protocol(protocol)
    return protocol


def _normalized_i_protocol(
    protocol: dict[str, object] | None = None,
) -> dict[str, object]:
    normalized = copy.deepcopy(protocol if protocol is not None else _disk_protocol())
    binding = normalized["implementation_binding"]
    assert binding["status"] in {"UNKNOWN_NOT_ASSERTED", "BOUND"}
    for key in ACQUIRE.UNKNOWN_BINDING_SCALARS:
        binding[key] = "UNKNOWN_NOT_ASSERTED"
    ACQUIRE._validate_protocol(normalized)
    return normalized


def _synthetic_b_protocol(
    protocol_i: dict[str, object] | None = None,
) -> dict[str, object]:
    bound = copy.deepcopy(
        protocol_i if protocol_i is not None else _normalized_i_protocol()
    )
    binding = bound["implementation_binding"]
    binding["status"] = "BOUND"
    binding["implementation_commit"] = "1" * 40
    binding["implementation_script_sha256"] = "2" * 64
    binding["implementation_test_sha256"] = "3" * 64
    ACQUIRE._validate_protocol(bound)
    return bound


def _production_protocol() -> dict[str, object]:
    protocol = _disk_protocol()
    if PROTOCOL_STATE_OVERRIDE is None:
        return protocol
    if PROTOCOL_STATE_OVERRIDE == "BOUND":
        return _synthetic_b_protocol(_normalized_i_protocol(protocol))
    raise AssertionError("GSE149487_TEST_PROTOCOL_STATE must be BOUND when set")


def _write_authority(path: Path, value: object) -> tuple[int, str]:
    payload = _json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return len(payload), _sha256(payload)


def _fixture_tree(tmp_path: Path) -> tuple[Path, Path, MemoryTransport, dict[str, bytes]]:
    repo = tmp_path / "repo"
    configs = repo / "configs"
    configs.mkdir(parents=True)
    protocol = _normalized_i_protocol(_production_protocol())

    download_payloads: dict[str, bytes] = {}
    r4_assets: list[dict[str, object]] = []
    manifest_assets: list[dict[str, object]] = []
    for context in ("PC3", "293T"):
        for assay in ("DNA", "POLYSOME", "TOTALRNA"):
            for replicate in (1, 2, 3):
                asset_id = f"GSE149487_GEO_{context}_{assay}_REP{replicate}"
                filename = f"{asset_id}.txt.gz"
                url = f"https://example.test/{filename}"
                payload = f"fixture:{asset_id}".encode("ascii")
                download_payloads[url] = payload
                r4_assets.append(
                    {
                        "asset_id": asset_id,
                        "asset_kind": "GEO_RAW_COUNT",
                        "context": context,
                        "assay": assay,
                        "biological_replicate": replicate,
                        "filename": filename,
                        "bytes": len(payload),
                        "sha256": _sha256(payload),
                        "source_uri": url,
                    }
                )
                manifest_assets.append(
                    {
                        "asset_id": asset_id,
                        "asset_kind": "GEO_RAW_COUNT",
                        "context": context,
                        "assay": assay,
                        "biological_replicate": replicate,
                    }
                )

    supplement_rows = (
        ("GSE149487_MOESM3", "41467_2021_24445_MOESM3_ESM.xlsx"),
        ("GSE149487_MOESM8", "41467_2021_24445_MOESM8_ESM.xlsx"),
        ("GSE149487_LIM6C_293T", "Lim_et_al_Supp_Tbl_6c_293T.xlsx"),
    )
    replacement_by_id = {
        record["asset_id"]: record
        for record in ACQUIRE.EXPECTED_LOCATOR_REPLACEMENTS
    }
    for asset_id, filename in supplement_rows:
        replacement = replacement_by_id.get(asset_id)
        authority_url = (
            replacement["frozen_authority_source_uri"]
            if replacement is not None
            else f"https://example.test/{filename}"
        )
        resolved_url = (
            replacement["resolved_current_official_source_uri"]
            if replacement is not None
            else authority_url
        )
        payload = f"fixture:{asset_id}".encode("ascii")
        download_payloads[resolved_url] = payload
        record = {
            "asset_id": asset_id,
            "asset_kind": "SUPPLEMENT_WORKBOOK",
            "filename": filename,
            "bytes": len(payload),
            "sha256": _sha256(payload),
            "source_uri": authority_url,
        }
        r4_assets.append(record)
        manifest_assets.append(dict(record))

    asset_manifest = {
        "dataset_id": "GSE149487",
        "expected_asset_count": 21,
        "assets": manifest_assets,
    }
    asset_path = configs / "route_a_v3_gse149487_asset_manifest_v2.json"
    asset_bytes, asset_sha = _write_authority(asset_path, asset_manifest)

    r4 = {
        "dataset_id": "GSE149487",
        "all_input_hashes_verified": True,
        "asset_count": 21,
        "assets": r4_assets,
    }
    r4_path = tmp_path / "historical-r4" / "ASSET_MANIFEST_EFFECTIVE.json"
    r4_bytes, r4_sha = _write_authority(r4_path, r4)

    external = {
        "dataset_id": "GSE149487",
        "authority_bindings": {"asset_manifest": {"sha256": asset_sha}},
        "historical_r4_closure": {
            "bundle_path": str(r4_path.parent),
            "qualification_report": {"qualified": False},
            "reuse_policy": "REFERENCE_AGGREGATE_ONLY_DO_NOT_REOPEN_OR_REHASH",
        },
        "scientific_evidence_boundaries": {
            "license_boundary": {
                "moesm3_and_moesm8": "CC_BY_4_0_ONLY",
                "geo_raw_18": "NONREDISTRIBUTABLE_LOCATOR_HASH_ONLY",
                "lim6c": "NO_EXPLICIT_LICENSE",
                "all_21_assets_license_status": "BLOCKED",
                "qualification_effect": "BLOCK",
            }
        },
    }
    external_path = configs / "route_a_v3_gse149487_external_evidence_roots_v1.json"
    external_bytes, external_sha = _write_authority(external_path, external)

    gate = {
        "qualification_status": "BLOCKED_PENDING_PUBLIC_EVIDENCE",
        "qualified": False,
        "training_allowed": False,
        "model_selection_allowed": False,
        "ordinary_study_contribution": 0,
        "a1_study_contribution": 0,
        "true_a2_study_contribution": 0,
        "canonical_record_count": 0,
        "next_phase_authorized": False,
    }
    qualification = {
        "authority": {"asset_manifest_sha256": asset_sha},
        "scope": {"full_raw_geo_table_count": 18, "supplement_count": 3},
        "current_gate_contract": gate,
    }
    qualification_path = configs / "route_a_v3_gse149487_a1_qualification.json"
    qualification_bytes, qualification_sha = _write_authority(
        qualification_path, qualification
    )

    metadata = protocol["metadata_authorities"]
    metadata["asset_manifest"].update(bytes=asset_bytes, sha256=asset_sha)
    metadata["external_evidence_roots"].update(
        bytes=external_bytes, sha256=external_sha
    )
    metadata["a1_qualification"].update(
        bytes=qualification_bytes, sha256=qualification_sha
    )
    metadata["historical_r4_effective_asset_manifest"].update(
        path=str(r4_path), bytes=r4_bytes, sha256=r4_sha
    )
    protocol["asset_contract"]["expected_total_payload_bytes"] = sum(
        len(payload) for payload in download_payloads.values()
    )
    output_base = tmp_path / "data" / "A1" / "GSE149487"
    protocol["output_contract"]["base_directory"] = str(output_base)
    protocol_path = configs / ACQUIRE.PROTOCOL_BASENAME
    protocol_path.write_bytes(_json_bytes(protocol))
    output = output_base / "GSE149487_PUBLIC_ASSETS_20260812T120000Z"
    return (
        protocol_path,
        output,
        MemoryTransport(download_payloads),
        download_payloads,
    )


def _fixture_binding(
    protocol: dict[str, object], protocol_path: Path, payload: bytes
) -> dict[str, str]:
    assert protocol == _normalized_i_protocol(protocol)
    assert payload
    return {
        "status": "TEST_FIXTURE_BOUND_WITHOUT_GIT",
        "implementation_commit": "0" * 40,
        "binding_commit": "1" * 40,
    }


def test_production_protocol_freezes_exact_authorities_and_honest_stop() -> None:
    protocol = _production_protocol()
    protocol_i = _normalized_i_protocol(protocol)
    protocol_b = _synthetic_b_protocol(protocol_i)
    assert protocol["implementation_binding"]["status"] in {
        "UNKNOWN_NOT_ASSERTED",
        "BOUND",
    }
    assert protocol_i["implementation_binding"]["status"] == "UNKNOWN_NOT_ASSERTED"
    assert protocol_b["implementation_binding"]["status"] == "BOUND"
    changed_binding_scalars = {
        key
        for key in ACQUIRE.UNKNOWN_BINDING_SCALARS
        if protocol_i["implementation_binding"][key]
        != protocol_b["implementation_binding"][key]
    }
    assert changed_binding_scalars == set(ACQUIRE.UNKNOWN_BINDING_SCALARS)
    r4 = protocol["metadata_authorities"]["historical_r4_effective_asset_manifest"]
    assert r4 == {
        "path": (
            "/mnt/cunyuliu/mrna_xeditflow_routea_v3/runs/A1/"
            "A1_GSE149487_PLUMAGE_FULL_QUAL_20260810T131156P0800_a859166_R4/"
            "ASSET_MANIFEST_EFFECTIVE.json"
        ),
        "bytes": 18355,
        "sha256": "f035494964b48a29440306164ca467674aaf76adac8f8c6b526da14ffa71c2de",
        "reuse_policy": "FROZEN_AGGREGATE_EXACT21_ACQUISITION_AUTHORITY_ONLY",
        "payload_reopen_allowed": False,
    }
    assert protocol["asset_contract"]["expected_asset_count"] == 21
    assert protocol["asset_contract"]["expected_geo_raw_count"] == 18
    assert protocol["asset_contract"]["expected_supplement_count"] == 3
    assert protocol["asset_contract"]["expected_total_payload_bytes"] == 70_032_274
    assert protocol["asset_contract"]["payload_parse_allowed"] is False
    assert protocol["asset_contract"]["qualifier_execution_allowed"] is False
    assert protocol["output_contract"]["base_directory"] == (
        "/mnt/cunyuliu/mrna_xeditflow_routea_v3/data/A1/GSE149487"
    )
    assert protocol["output_contract"]["single_aggregate_report_only"] is True
    assert protocol["output_contract"]["sha256sums_file_written"] is False
    assert protocol["output_contract"]["terminal_marker_written"] is False
    assert protocol["terminal_truth"] == ACQUIRE.EXPECTED_TERMINAL_TRUTH
    assert protocol["implementation_binding"]["base_commit"] == (
        "b39da87060e7794351a373aaf3bb66892a6b36b3"
    )
    assert protocol["download_policy"] == ACQUIRE.EXPECTED_DOWNLOAD_POLICY
    assert set(protocol["unknown_not_asserted"].values()) == {
        "UNKNOWN_NOT_ASSERTED"
    }
    assert protocol["confirmed_public_evidence"]["paper_jats_method_surface"] == {
        "status": "CONFIRMED_METHOD_SURFACE_ONLY",
        "original_cpm_minimum_inclusive": 0.5,
        "test_sidedness": "TWO_SIDED",
        "test_family": "MANN_WHITNEY_U",
        "reported_r_call": "wilcox.test",
        "multiple_testing_adjustment": "FDR",
        "significance_rule": "FDR_LT_0.1",
    }
    assert protocol["confirmed_public_evidence"]["license_and_redistribution"] == {
        "paper_and_moesm3_moesm8_license": (
            "CC_BY_4_0_CONFIRMED_FOR_PAPER_AND_PAPER_SUPPLEMENTS_ONLY"
        ),
        "geo_raw_18": (
            "PRIVATE_CANONICAL_LOCATOR_HASH_USE_ONLY_NO_DATA_SPECIFIC_"
            "REDISTRIBUTION_GRANT"
        ),
        "lim6c_github_blob": "NO_EXPLICIT_LICENSE_CONFIRMED",
    }


def test_unknown_binding_stops_before_metadata_output_or_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol_path, output, _, _ = _fixture_tree(tmp_path)
    transport = ForbiddenTransport()
    git_called = False

    def forbidden_git(*args: object, **kwargs: object) -> bytes:
        nonlocal git_called
        git_called = True
        raise AssertionError("UNKNOWN binding must stop before git")

    monkeypatch.setattr(ACQUIRE, "_run_git", forbidden_git)
    with pytest.raises(ACQUIRE.BindingNotFrozen, match="config-only-B"):
        ACQUIRE.execute(protocol_path, output, transport=transport)
    assert git_called is False
    assert transport.opened == []
    assert not output.exists()


def test_exact21_acquisition_writes_assets_and_one_stopped_aggregate(
    tmp_path: Path,
) -> None:
    protocol_path, output, transport, payloads = _fixture_tree(tmp_path)
    report = ACQUIRE.execute(
        protocol_path,
        output,
        transport=transport,
        binding_auditor=_fixture_binding,
        recorded_at="2026-08-12T12:00:00Z",
    )

    assert len(transport.opened) == 21
    assert report["status"] == "STOPPED_WITH_PUBLIC_EVIDENCE_BLOCKER"
    assert report["acquisition_status"] == (
        "EXACT_21_ASSETS_ACQUIRED_AND_INTEGRITY_VERIFIED"
    )
    assert report["asset_counts"] == {
        "asset_count": 21,
        "geo_raw_count": 18,
        "supplement_count": 3,
        "total_verified_bytes": sum(len(payload) for payload in payloads.values()),
    }
    assert report["terminal_truth"] == ACQUIRE.EXPECTED_TERMINAL_TRUTH
    assert report["execution_counters"] == {
        "payload_files_opened_for_scientific_parsing": 0,
        "row_sequence_effect_records_read": 0,
        "qualifier_execution_count": 0,
        "model_download_count": 0,
        "training_run_count": 0,
        "model_selection_run_count": 0,
        "canonical_record_count": 0,
    }
    assert len(report["retained_blockers"]) == len(ACQUIRE.EXPECTED_RETAINED_BLOCKERS)
    report_by_id = {record["asset_id"]: record for record in report["assets"]}
    for replacement in ACQUIRE.EXPECTED_LOCATOR_REPLACEMENTS:
        record = report_by_id[replacement["asset_id"]]
        assert record["authority_source_uri"] == replacement[
            "frozen_authority_source_uri"
        ]
        assert record["resolved_current_official_source_uri"] == replacement[
            "resolved_current_official_source_uri"
        ]
        assert record["official_locator_replacement_applied"] is True
        assert replacement["resolved_current_official_source_uri"] in transport.opened
        assert replacement["frozen_authority_source_uri"] not in transport.opened
    unchanged = [
        record
        for record in report["assets"]
        if not record["official_locator_replacement_applied"]
    ]
    assert len(unchanged) == 19
    assert all(
        record["authority_source_uri"]
        == record["resolved_current_official_source_uri"]
        for record in unchanged
    )

    expected_names = {
        Path(url).name for url in payloads
    } | {ACQUIRE.REPORT_FILENAME}
    assert {path.name for path in output.iterdir()} == expected_names
    assert not list(output.glob("*.part"))
    for url, payload in payloads.items():
        assert (output / Path(url).name).read_bytes() == payload
    written_report = json.loads((output / ACQUIRE.REPORT_FILENAME).read_text())
    assert written_report == report
    assert not (output / "SHA256SUMS").exists()
    assert not (output / "PUBLICATION_COMMIT.json").exists()


@pytest.mark.parametrize(
    "mutation",
    ("WRONG_OLD_LOCATOR", "WRONG_CURRENT_LOCATOR", "THIRD_ASSET_REPLACEMENT"),
)
def test_closed_locator_policy_rejects_wrong_or_expanded_mapping(
    mutation: str,
) -> None:
    protocol = _normalized_i_protocol(_production_protocol())
    replacements = protocol["download_policy"][
        "closed_official_locator_replacements"
    ]
    if mutation == "WRONG_OLD_LOCATOR":
        replacements[0]["frozen_authority_source_uri"] += ".wrong"
    elif mutation == "WRONG_CURRENT_LOCATOR":
        replacements[1]["resolved_current_official_source_uri"] += ".wrong"
    else:
        replacements.append(
            {
                "asset_id": "GSE149487_LIM6C_293T",
                "frozen_authority_source_uri": (
                    "https://raw.githubusercontent.com/sonali-bioc/Lim-5utr-Paper/"
                    "d613b541d192d6c502a1ef8849c27e801a7fbfb9/data/"
                    "Lim_et_al_Supp_Tbl_6c_293T.xlsx"
                ),
                "resolved_current_official_source_uri": (
                    "https://example.test/Lim_et_al_Supp_Tbl_6c_293T.xlsx"
                ),
            }
        )
    with pytest.raises(ACQUIRE.ProtocolError, match="closed two-locator repair"):
        ACQUIRE._validate_protocol(protocol)


def test_download_hash_mismatch_retains_part_and_never_writes_report(
    tmp_path: Path,
) -> None:
    protocol_path, output, transport, _ = _fixture_tree(tmp_path)
    first_url = next(iter(transport.payload_by_url))
    transport.payload_by_url[first_url] += b"tampered"

    with pytest.raises(ACQUIRE.IntegrityError, match="byte count mismatch"):
        ACQUIRE.execute(
            protocol_path,
            output,
            transport=transport,
            binding_auditor=_fixture_binding,
        )
    assert transport.opened == [first_url]
    assert (output / f"{Path(first_url).name}.part").exists()
    assert not (output / Path(first_url).name).exists()
    assert not (output / ACQUIRE.REPORT_FILENAME).exists()


def test_r4_authority_mismatch_stops_before_output_and_network(tmp_path: Path) -> None:
    protocol_path, output, _, _ = _fixture_tree(tmp_path)
    protocol = json.loads(protocol_path.read_text())
    r4_path = Path(
        protocol["metadata_authorities"]["historical_r4_effective_asset_manifest"][
            "path"
        ]
    )
    r4_path.write_bytes(r4_path.read_bytes() + b" ")
    transport = ForbiddenTransport()

    with pytest.raises(ACQUIRE.AuthorityError, match="byte count mismatch"):
        ACQUIRE.execute(
            protocol_path,
            output,
            transport=transport,
            binding_auditor=_fixture_binding,
        )
    assert transport.opened == []
    assert not output.exists()


def test_output_must_be_exclusive_child_of_public_asset_root(tmp_path: Path) -> None:
    protocol_path, output, transport, _ = _fixture_tree(tmp_path)
    outside = output.parent.parent / output.name
    binding_called = False

    def forbidden_binding(*args: object) -> dict[str, object]:
        nonlocal binding_called
        binding_called = True
        raise AssertionError("wrong output must stop before binding")

    with pytest.raises(ACQUIRE.OutputScopeError, match="one direct child"):
        ACQUIRE.execute(
            protocol_path,
            outside,
            transport=transport,
            binding_auditor=forbidden_binding,
        )
    assert binding_called is False
    assert transport.opened == []
    assert not outside.exists()
