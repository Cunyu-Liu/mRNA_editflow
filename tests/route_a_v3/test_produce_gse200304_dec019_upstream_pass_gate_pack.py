from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


STAGING_ROOT = Path(__file__).resolve().parents[2]
WORK_ROOT = STAGING_ROOT.parent
SCRIPT_PATH = STAGING_ROOT / (
    "scripts/route_a_v3/produce_gse200304_dec019_upstream_pass_gate_pack.py"
)
CONFIG_PATH = STAGING_ROOT / (
    "configs/route_a_v3_gse200304_dec019_upstream_pass_gate_pack_v1.json"
)


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PRODUCER = _load_module(SCRIPT_PATH, "gse200304_dec019_upstream_pass_gate_pack")


def _config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _unknown_config() -> dict[str, Any]:
    config = _config()
    binding = config["implementation_binding"]
    binding["status"] = PRODUCER.UNKNOWN
    binding["implementation_commit"] = PRODUCER.UNKNOWN
    binding["implementation_script_sha256"] = PRODUCER.UNKNOWN
    binding["implementation_test_sha256"] = PRODUCER.UNKNOWN
    PRODUCER.validate_static_config(config)
    return config


def _bound_config() -> dict[str, Any]:
    config = _unknown_config()
    binding = config["implementation_binding"]
    binding["status"] = PRODUCER.BOUND
    binding["implementation_commit"] = "1" * 40
    binding["implementation_script_sha256"] = PRODUCER.sha256(SCRIPT_PATH.read_bytes())
    binding["implementation_test_sha256"] = PRODUCER.sha256(Path(__file__).read_bytes())
    PRODUCER.validate_static_config(config)
    PRODUCER.validate_implementation_binding(config)
    return config


def _consumer_sources() -> dict[str, Path]:
    integrated = {
        "config": STAGING_ROOT / PRODUCER.CONSUMER_CONFIG_REPO_PATH,
        "script": STAGING_ROOT / PRODUCER.CONSUMER_SCRIPT_REPO_PATH,
        "test": STAGING_ROOT / PRODUCER.CONSUMER_TEST_REPO_PATH,
    }
    expected = _config()["consumer_authority"]
    if all(path.is_file() for path in integrated.values()):
        if PRODUCER.sha256(integrated["config"].read_bytes()) == expected[
            "frozen_config_sha256"
        ]:
            return integrated
    staged = {
        "config": WORK_ROOT
        / "g200_d2_descriptor_bind_staging/configs/"
        "route_a_v3_gse200304_dec019_reported_endpoint_a1_activation_v3.json",
        "script": WORK_ROOT
        / "g200_consumer_descriptor_lifecycle_final_staging/scripts/route_a_v3/"
        "adjudicate_gse200304_dec019_reported_endpoint_a1_v3.py",
        "test": WORK_ROOT
        / "g200_consumer_descriptor_lifecycle_final_staging/tests/route_a_v3/"
        "test_adjudicate_gse200304_dec019_reported_endpoint_a1_v3.py",
    }
    assert all(path.is_file() for path in staged.values())
    assert PRODUCER.sha256(staged["config"].read_bytes()) == expected[
        "frozen_config_sha256"
    ]
    return staged


def _materialize_consumer_repo(
    root: Path,
    *,
    current_config_payload: bytes | None = None,
) -> bytes:
    sources = _consumer_sources()
    frozen_payload = sources["config"].read_bytes()
    payloads = {
        PRODUCER.CONSUMER_CONFIG_REPO_PATH: current_config_payload or frozen_payload,
        PRODUCER.CONSUMER_SCRIPT_REPO_PATH: sources["script"].read_bytes(),
        PRODUCER.CONSUMER_TEST_REPO_PATH: sources["test"].read_bytes(),
    }
    for relative, payload in payloads.items():
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
    return frozen_payload


def _consumer_context(
    tmp_path: Path,
    config: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], ModuleType, Path]:
    bound = config or _bound_config()
    repo = tmp_path / "consumer_repo"
    frozen_payload = _materialize_consumer_repo(repo)
    frozen, _current, module, result = PRODUCER.load_verified_consumer(
        bound,
        repo=repo,
        production=False,
        frozen_config_payload=frozen_payload,
    )
    assert result is None
    return bound, frozen, module, repo


def _valid_audit(config: dict[str, Any]) -> dict[str, Any]:
    upstream = config["upstream_authority"]
    semantics = config["expected_audit_semantics"]
    raw = upstream["exact_members"][:3]
    return {
        "schema_version": PRODUCER.UPSTREAM_AUDIT_SCHEMA_VERSION,
        "record_type": PRODUCER.UPSTREAM_AUDIT_RECORD_TYPE,
        "protocol_id": PRODUCER.UPSTREAM_PROTOCOL_ID,
        "contract_id": PRODUCER.CONTRACT_ID,
        "phase_id": PRODUCER.PHASE_ID,
        "dataset_id": PRODUCER.DATASET_ID,
        "decision_id": PRODUCER.DECISION_ID,
        "status": PRODUCER.UPSTREAM_AUDIT_STATUS,
        "mode": PRODUCER.UPSTREAM_AUDIT_MODE,
        "producer_binding": {
            "status": "PASS_BOUND_IMPLEMENTATION",
            "implementation_commit": upstream["producer_implementation_commit"],
            "binding_commit": upstream["producer_binding_commit"],
            "implementation_script_sha256": upstream["producer_script_sha256"],
            "implementation_test_sha256": upstream["producer_test_sha256"],
            "config_core_sha256": upstream["producer_config_core_sha256"],
        },
        "predecessor_authority": {
            "published_endpoint_config_sha256": (
                "92fc3a3859f7a8949ace67fa4b03a14e8ad102eb257d4f95cace01ea535b41af"
            ),
            "published_endpoint_trio_manifest_sha256": "a" * 64,
            "source_exact7_manifest_sha256": "b" * 64,
            "published_endpoint_bundle_manifest_sha256": "c" * 64,
            "source_exact7_member_count": 7,
            "published_endpoint_bundle_member_count": 5,
            "table_s3_selective_pair_count": 6772,
            "table_s3_finite_totalpoly_pair_count": 6547,
            "table_s3_gene_column_selected_or_persisted": False,
            "table_s3_translation_significance_selected_or_persisted": False,
        },
        "official_source_authority": {
            "status": "PASS_EXACT_THREE_OFFICIAL_SOURCE_SNAPSHOTS",
            "network_download_count": 3,
            "verbatim_source_member_count": 3,
            "sources": [
                {
                    "source_kind": f"FIXTURE_{index}",
                    "url": f"https://example.invalid/{index}",
                    "output_name": item["name"],
                    "bytes": item["bytes"],
                    "sha256": item["sha256"],
                    "same_fd_size_and_hash_verified": True,
                }
                for index, item in enumerate(raw)
            ],
        },
        "jats_authority": {
            "status": "PASS_EXACT_JATS_IDENTITY_LICENSE_LINKAGE_AND_PARAGRAPHS",
            "identity": {
                "doi": "10.1016/j.celrep.2023.112840",
                "pmcid": "PMC10540565",
                "pmid": "37516102",
            },
            "license_ref": "https://creativecommons.org/licenses/by/4.0/",
            "license_text_verified": True,
            "supplement_table_cross_reference_counts": {
                "Table S2": 1,
                "Table S3": 1,
            },
            "normalized_paragraphs": {
                "endpoint_and_six_biological_replicates": {
                    "utf8_bytes": 798,
                    "sha256": (
                        "45dd0d8b9c7976748615f2c7b620bcc403fe7bf5c832b2dbb8516d758b27ac3d"
                    ),
                },
                "ratio_of_ratios_methods": {
                    "utf8_bytes": 534,
                    "sha256": (
                        "0fb681090cf10597751369a11ab72fa19552e14ae9ce579d8b350f135b274fd2"
                    ),
                },
            },
        },
        "geo_soft_authority": {
            "status": "PASS_EXACT_GSE200302_SUBSERIES_AND_24_SAMPLE_ROLE_GRID",
            "series_accession": "GSE200302",
            "subseries_of_gse200304": True,
            "sample_count": 24,
            "role_counts": {
                "High_Poly": 6,
                "Low_Poly": 6,
                "Total_RNA": 6,
                "pDNA": 6,
            },
            "replicates_per_role": [1, 2, 3, 4, 5, 6],
            "sample_supplementary_none_count": 24,
            "series_supplementary_file_count": 2,
            "series_processed_matrix_reference_count": 1,
            "processed_matrix_payload_embedded_in_soft": False,
            "geo_dataset_restriction_field_count": 0,
        },
        "processed_matrix_authority": {
            "status": "PASS_EXACT_6772_BY_61_MATRIX_AND_S3_MEMBERSHIP_CROSSCHECK",
            "header_field_count": 61,
            "value_field_count": 60,
            "row_count": 6772,
            "row_width_error_count": 0,
            "duplicate_key_count": 0,
            "missing_value_count": 0,
            "invalid_numeric_count": 0,
            "closed_role_geometry_count": 60,
            "required_endpoint_families": ["High_Poly", "Low_Poly", "Total_RNA"],
            "required_arms": ["WT", "Mutant"],
            "required_replicates": [1, 2, 3, 4, 5, 6],
            "endpoint_excluded_families": ["80S_RNA", "pDNA"],
            "matrix_key_set_equals_s3_key_set": True,
            "matrix_key_count": 6772,
            "s3_key_count": 6772,
            "finite_totalpoly_key_count": 6547,
            "matrix_covers_every_finite_totalpoly_key": True,
            "standard_error_status": "ABSENT_NOT_REPORTED_NOT_DERIVED_NOT_USED",
            "p_or_fdr_back_calculation_used": False,
        },
        "endpoint_crosswalk": copy.deepcopy(semantics["endpoint_crosswalk"]),
        "replicate_branch": copy.deepcopy(semantics["replicate_branch"]),
        "private_only_rights": copy.deepcopy(semantics["private_only_rights"]),
        "biological_group_authority": copy.deepcopy(
            semantics["biological_group_authority"]
        ),
        "unchanged_gates": copy.deepcopy(semantics["unchanged_gates"]),
        "decision_boundary": copy.deepcopy(semantics["decision_boundary"]),
        "execution_boundary": copy.deepcopy(semantics["execution_boundary"]),
        "privacy": {
            "derived_row_payload": False,
            "derived_sequence_payload": False,
            "derived_row_identifier_payload": False,
            "derived_effect_value_payload": False,
            "derived_gene_payload": False,
            "verbatim_raw_source_members_are_not_derived_payload": True,
        },
    }


def _upstream_marker(config: dict[str, Any]) -> dict[str, Any]:
    upstream = config["upstream_authority"]
    specs = {item["name"]: item for item in upstream["exact_members"]}
    return {
        "schema_version": "1.0.0",
        "record_type": PRODUCER.UPSTREAM_MARKER_RECORD_TYPE,
        "protocol_id": PRODUCER.UPSTREAM_PROTOCOL_ID,
        "contract_id": PRODUCER.CONTRACT_ID,
        "dataset_id": PRODUCER.DATASET_ID,
        "bundle_id": upstream["bundle_id"],
        "preterminal_member_names": sorted(
            name for name in specs if name != PRODUCER.MARKER_MEMBER_NAME
        ),
        "preterminal_member_count": 5,
        "exact_final_member_count": 6,
        "sha256sums_sha256": specs[PRODUCER.CHECKSUMS_MEMBER_NAME]["sha256"],
        "final_output_target_sha256": upstream[
            "terminal_marker_final_output_target_sha256"
        ],
        "publication_mode": PRODUCER.UPSTREAM_PUBLICATION_MODE,
        "committed": True,
        "terminal_marker_written_last": True,
        "no_overwrite": True,
        "partial_default": PRODUCER.UPSTREAM_PARTIAL_DEFAULT,
    }


def _materialize_upstream(
    tmp_path: Path,
    config: dict[str, Any],
    *,
    audit: dict[str, Any] | None = None,
    marker_mutator: Any = None,
) -> Path:
    root = tmp_path / "upstream_exact6"
    root.mkdir(parents=True)
    upstream = config["upstream_authority"]
    specs = {item["name"]: item for item in upstream["exact_members"]}
    raw_payloads = {
        upstream["raw_source_member_names_hash_only"][0]: b"jats-fixture\n",
        upstream["raw_source_member_names_hash_only"][1]: b"soft-fixture\n",
        upstream["raw_source_member_names_hash_only"][2]: b"matrix-fixture\n",
    }
    for name, payload in raw_payloads.items():
        specs[name]["bytes"] = len(payload)
        specs[name]["sha256"] = PRODUCER.sha256(payload)
    closed_audit = audit or _valid_audit(config)
    # Official-source bytes/hashes are part of the semantic audit.
    for source, name in zip(
        closed_audit["official_source_authority"]["sources"], raw_payloads
    ):
        source["bytes"] = specs[name]["bytes"]
        source["sha256"] = specs[name]["sha256"]
    audit_payload = PRODUCER.json_bytes(closed_audit)
    audit_name = upstream["audit_member_name"]
    specs[audit_name]["bytes"] = len(audit_payload)
    specs[audit_name]["sha256"] = PRODUCER.sha256(audit_payload)
    sums = "".join(
        f"{specs[name]['sha256']}  {name}\n"
        for name in sorted((*raw_payloads, audit_name))
    ).encode("ascii")
    specs[PRODUCER.CHECKSUMS_MEMBER_NAME]["bytes"] = len(sums)
    specs[PRODUCER.CHECKSUMS_MEMBER_NAME]["sha256"] = PRODUCER.sha256(sums)
    marker = _upstream_marker(config)
    if marker_mutator is not None:
        marker_mutator(marker)
    marker_payload = PRODUCER.json_bytes(marker)
    specs[PRODUCER.MARKER_MEMBER_NAME]["bytes"] = len(marker_payload)
    specs[PRODUCER.MARKER_MEMBER_NAME]["sha256"] = PRODUCER.sha256(marker_payload)
    payloads = {
        **raw_payloads,
        audit_name: audit_payload,
        PRODUCER.CHECKSUMS_MEMBER_NAME: sums,
        PRODUCER.MARKER_MEMBER_NAME: marker_payload,
    }
    for name, payload in payloads.items():
        (root / name).write_bytes(payload)
    return root


def _sample_preterminal() -> dict[str, bytes]:
    content = {
        name: f"fixture:{name}\n".encode("utf-8")
        for name in PRODUCER.CONTENT_MEMBER_NAMES
    }
    sums = "".join(
        f"{PRODUCER.sha256(content[name])}  {name}\n" for name in sorted(content)
    ).encode("ascii")
    return {**content, PRODUCER.CHECKSUMS_MEMBER_NAME: sums}


def test_unknown_stops_before_repo_upstream_or_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _unknown_config()
    calls = {"consumer": 0, "upstream": 0, "output": 0}

    def forbidden(name: str):
        def inner(*_args: Any, **_kwargs: Any) -> None:
            calls[name] += 1
            raise AssertionError(f"{name} touched before binding")

        return inner

    monkeypatch.setattr(PRODUCER, "load_verified_consumer", forbidden("consumer"))
    monkeypatch.setattr(PRODUCER, "inspect_upstream_bundle", forbidden("upstream"))
    monkeypatch.setattr(PRODUCER, "publish_pack", forbidden("output"))
    with pytest.raises(PRODUCER.BindingError):
        PRODUCER.produce(
            config,
            tmp_path / "output",
            repo=tmp_path / "repo",
            production=False,
        )
    assert calls == {"consumer": 0, "upstream": 0, "output": 0}


def test_production_inspect_rejects_alternate_output_before_any_authority_or_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _bound_config()
    calls = {"git": 0, "consumer": 0, "upstream": 0, "output": 0}

    def forbidden(name: str):
        def inner(*_args: Any, **_kwargs: Any) -> None:
            calls[name] += 1
            raise AssertionError(f"{name} touched before production scope rejection")

        return inner

    monkeypatch.setattr(PRODUCER, "validate_production_authority", forbidden("git"))
    monkeypatch.setattr(PRODUCER, "load_verified_consumer", forbidden("consumer"))
    monkeypatch.setattr(PRODUCER, "inspect_upstream_bundle", forbidden("upstream"))
    monkeypatch.setattr(PRODUCER, "inspect_published_pack", forbidden("output"))
    with pytest.raises(PRODUCER.ScopeViolation):
        PRODUCER.produce(
            config,
            tmp_path / "alternate-pack",
            repo=tmp_path / "repo",
            production=True,
            config_payload=b"not-reached",
            config_path=PRODUCER.PRODUCTION_CONFIG_PATH,
            inspect_only=True,
        )
    assert calls == {"git": 0, "consumer": 0, "upstream": 0, "output": 0}


def test_direct_production_api_rejects_alternate_upstream_before_any_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _bound_config()
    calls = {"git": 0, "consumer": 0, "upstream": 0, "output": 0}

    def forbidden(name: str):
        def inner(*_args: Any, **_kwargs: Any) -> None:
            calls[name] += 1
            raise AssertionError(f"{name} touched before production scope rejection")

        return inner

    monkeypatch.setattr(PRODUCER, "validate_production_authority", forbidden("git"))
    monkeypatch.setattr(PRODUCER, "load_verified_consumer", forbidden("consumer"))
    monkeypatch.setattr(PRODUCER, "inspect_upstream_bundle", forbidden("upstream"))
    monkeypatch.setattr(PRODUCER, "publish_pack", forbidden("output"))
    with pytest.raises(PRODUCER.ScopeViolation):
        PRODUCER.produce(
            config,
            Path(config["output_contract"]["trusted_final_directory"]),
            repo=tmp_path / "repo",
            production=True,
            config_payload=b"not-reached",
            config_path=PRODUCER.PRODUCTION_CONFIG_PATH,
            upstream_root=tmp_path / "alternate-upstream",
        )
    assert calls == {"git": 0, "consumer": 0, "upstream": 0, "output": 0}


def test_i_to_b_is_exact_four_scalars() -> None:
    i_config = _unknown_config()
    b_config = copy.deepcopy(i_config)
    binding = b_config["implementation_binding"]
    binding["status"] = PRODUCER.BOUND
    binding["implementation_commit"] = "1" * 40
    binding["implementation_script_sha256"] = "2" * 64
    binding["implementation_test_sha256"] = "3" * 64
    PRODUCER.validate_i_to_b_transition(
        i_config,
        b_config,
        implementation_commit="1" * 40,
        implementation_script_sha256="2" * 64,
        implementation_test_sha256="3" * 64,
    )
    changed = copy.deepcopy(b_config)
    changed["record_policy"]["training_allowed"] = True
    with pytest.raises(PRODUCER.ProducerError):
        PRODUCER.validate_i_to_b_transition(
            i_config,
            changed,
            implementation_commit="1" * 40,
            implementation_script_sha256="2" * 64,
            implementation_test_sha256="3" * 64,
        )


def test_exact6_inspection_decodes_only_closed_audit(tmp_path: Path) -> None:
    config = _bound_config()
    root = _materialize_upstream(tmp_path, config)
    result = PRODUCER.inspect_upstream_bundle(config, root=root)
    assert result["exact_member_count"] == 6
    assert result["raw_source_same_fd_hash_only_count"] == 3
    assert result["decoded_raw_source_count"] == 0
    assert result["decoded_semantic_input_count"] == 1


def test_upstream_hash_marker_and_semantic_drift_fail_closed(tmp_path: Path) -> None:
    config = _bound_config()
    hash_root = _materialize_upstream(tmp_path / "hash", config)
    raw_name = config["upstream_authority"]["raw_source_member_names_hash_only"][0]
    (hash_root / raw_name).write_bytes(b"changed-byte\n")
    with pytest.raises(PRODUCER.InputIntegrityError):
        PRODUCER.inspect_upstream_bundle(config, root=hash_root)

    marker_config = _bound_config()
    marker_root = _materialize_upstream(
        tmp_path / "marker",
        marker_config,
        marker_mutator=lambda marker: marker.__setitem__("committed", False),
    )
    with pytest.raises(PRODUCER.ProducerError):
        PRODUCER.inspect_upstream_bundle(marker_config, root=marker_root)

    semantic_config = _bound_config()
    audit = _valid_audit(semantic_config)
    audit["endpoint_crosswalk"]["consumer_gate_pass"] = True
    semantic_root = _materialize_upstream(
        tmp_path / "semantic", semantic_config, audit=audit
    )
    with pytest.raises(PRODUCER.ProducerError):
        PRODUCER.inspect_upstream_bundle(semantic_config, root=semantic_root)


def test_records_use_actual_consumer_both_apis_and_frozen_provenance(
    tmp_path: Path,
) -> None:
    config, frozen, module, _repo = _consumer_context(tmp_path)
    payloads = PRODUCER.build_gate_records(config, frozen, module)
    assert tuple(sorted(payloads)) == PRODUCER.GATE_MEMBER_NAMES
    slots = {slot["slot_id"]: slot for slot in frozen["evidence_contract"]["slots"]}
    predecessor = frozen["evidence_contract"]["required_predecessor_authority"]
    acceptance = frozen["evidence_contract"]["gate_record_provenance_contract"][
        "acceptance_authority"
    ]
    for payload in payloads.values():
        record = json.loads(payload)
        accepted = module._validate_gate_record(
            payload, slots[record["gate_id"]], frozen
        )
        assert module._slot_gate_pass(record["gate_id"], accepted["facts"]) is True
        assert record["status"] == "PASS"
        assert record["accepted"] is True
        assert record["aggregate_only"] is True
        assert record["unknown_fields"] == []
        assert record["reason_codes"] == []
        assert set(record["privacy"].values()) == {False}
        assert record["provenance"]["predecessor_authority"] == predecessor
        assert record["provenance"]["acceptance_authority"] == acceptance
        assert record["provenance"]["source_bundle_id"] == predecessor["bundle_id"]


def test_current_descriptor_descendant_may_change_only_descriptor_truth(
    tmp_path: Path,
) -> None:
    config = _bound_config()
    sources = _consumer_sources()
    frozen_payload = sources["config"].read_bytes()
    current = json.loads(frozen_payload)
    consumer_module = _load_module(sources["script"], "consumer_descriptor_fixture")
    selected = {
        "CANONICAL_REPORTED_ENDPOINT_SEMANTICS",
        "ROW_REPLICATE_OR_VALID_SE",
        "LICENSE_RIGHTS",
    }
    for slot in current["evidence_descriptor_bindings"]["slots"]:
        if slot["slot_id"] in selected:
            slot["absolute_path"] = f"/mnt/private/pass/{slot['slot_id']}.json"
            slot["sha256"] = "a" * 64
            slot["bytes"] = 123
    current["evidence_descriptor_bindings"]["descriptor_set_sha256"] = (
        consumer_module.descriptor_set_sha256(current)
    )
    current_payload = PRODUCER.json_bytes(current)
    repo = tmp_path / "descriptor_repo"
    _materialize_consumer_repo(repo, current_config_payload=current_payload)
    frozen, observed, module, result = PRODUCER.load_verified_consumer(
        config,
        repo=repo,
        production=False,
        frozen_config_payload=frozen_payload,
    )
    assert result is None
    assert module.config_core_sha256(observed) == config["consumer_authority"][
        "science_core_sha256"
    ]
    assert observed["evidence_descriptor_bindings"]["descriptor_set_sha256"] != (
        frozen["evidence_descriptor_bindings"]["descriptor_set_sha256"]
    )
    assert observed["evidence_contract"]["required_predecessor_authority"] == frozen[
        "evidence_contract"
    ]["required_predecessor_authority"]


def test_private_only_rights_and_other_pass_facts_are_frozen() -> None:
    config = _bound_config()
    assert config["pass_gate_records"] == PRODUCER.GATE_SPECS
    license_spec = next(
        spec for spec in PRODUCER.GATE_SPECS if spec["gate_id"] == "LICENSE_RIGHTS"
    )
    assert license_spec["facts"]["redistribution_scope"] == "PRIVATE_CANONICAL_ONLY"
    assert config["expected_audit_semantics"]["private_only_rights"][
        "public_redistribution_pass_claimed"
    ] is False
    drifted = copy.deepcopy(config)
    drifted["pass_gate_records"][2]["facts"][
        "redistribution_scope"
    ] = "PUBLIC_REDISTRIBUTION_ALLOWED"
    with pytest.raises(PRODUCER.ProducerError):
        PRODUCER.validate_static_config(drifted)


def test_pack_audit_keeps_counts_zero_and_training_locked(tmp_path: Path) -> None:
    config, frozen, module, _repo = _consumer_context(tmp_path)
    gates = PRODUCER.build_gate_records(config, frozen, module)
    summary = {
        "bundle_id": config["upstream_authority"]["bundle_id"],
        "final_output_target_sha256": config["upstream_authority"][
            "terminal_marker_final_output_target_sha256"
        ],
        "raw_source_same_fd_hash_only_count": 3,
    }
    payloads = PRODUCER.build_preterminal_payloads(config, gates, summary)
    audit = json.loads(payloads[PRODUCER.AUDIT_MEMBER_NAME])
    for key in (
        "ordinary_study_contribution_delta",
        "a1_study_contribution_delta",
        "true_a2_study_contribution_delta",
        "canonical_record_count_delta",
    ):
        assert audit[key] == 0
    for key in ("training_allowed", "model_selection_allowed", "next_phase_authorized"):
        assert audit[key] is False
    assert audit["consumer_run"] is False
    assert audit["adjudicator_run"] is False
    assert audit["decoded_raw_source_count"] == 0


def test_primary_publish_is_marker_last_exact6_and_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _bound_config()
    payloads = _sample_preterminal()
    output = tmp_path / "parent" / "pack"
    output.parent.mkdir()
    writes: list[str] = []
    original = PRODUCER._write_exclusive_regular_at

    def observing_write(directory_fd: int, name: str, payload: bytes) -> None:
        writes.append(name)
        original(directory_fd, name, payload)

    monkeypatch.setattr(PRODUCER, "_write_exclusive_regular_at", observing_write)
    status = PRODUCER.publish_pack(
        output, payloads, production=False, config=config
    )
    assert status == "PUBLISHED_PRIMARY"
    assert writes[:5] == list(PRODUCER.PRETERMINAL_MEMBER_NAMES)
    assert writes[5] == PRODUCER.MARKER_MEMBER_NAME
    inspected = PRODUCER.inspect_published_pack(output, payloads)
    assert inspected["exact_member_names"] == list(PRODUCER.PUBLISHED_MEMBER_NAMES)
    assert PRODUCER.publish_pack(
        output, payloads, production=False, config=config
    ) == "EXISTING_EXACT"


def test_fallback_partial_is_preserved_and_requires_explicit_recovery(
    tmp_path: Path,
) -> None:
    config = _bound_config()
    payloads = _sample_preterminal()
    output = tmp_path / "parent" / "pack"
    output.parent.mkdir()

    def unsupported(*_args: Any) -> None:
        raise PRODUCER.AtomicNoReplaceUnsupported(getattr(os, "ENOSYS", 38))

    fired = False

    def fault(event: str) -> None:
        nonlocal fired
        if event.startswith("after_fallback_write:") and not fired:
            fired = True
            raise RuntimeError("fixture fallback interruption")

    with pytest.raises(PRODUCER.PartialPublicationError):
        PRODUCER.publish_pack(
            output,
            payloads,
            production=False,
            config=config,
            rename_noreplace=unsupported,
            fault_injector=fault,
        )
    assert output.is_dir()
    assert PRODUCER.MARKER_MEMBER_NAME not in {path.name for path in output.iterdir()}
    with pytest.raises(PRODUCER.PartialPublicationError):
        PRODUCER.publish_pack(
            output, payloads, production=False, config=config
        )
    status = PRODUCER.publish_pack(
        output,
        payloads,
        production=False,
        config=config,
        recover_partial=True,
    )
    assert status == "RECOVERED_PARTIAL_FALLBACK"
    assert PRODUCER.inspect_published_pack(output, payloads)[
        "terminal_commit_marker_validated"
    ] is True


def test_symlink_and_hardlink_inputs_and_outputs_are_rejected(tmp_path: Path) -> None:
    config = _bound_config()
    root = _materialize_upstream(tmp_path / "input", config)
    outside = tmp_path / "raw-hardlink-outside"
    raw = root / config["upstream_authority"]["raw_source_member_names_hash_only"][0]
    os.link(raw, outside)
    with pytest.raises(PRODUCER.InputIntegrityError):
        PRODUCER.inspect_upstream_bundle(config, root=root)

    clean_config = _bound_config()
    clean_root = _materialize_upstream(tmp_path / "symlink", clean_config)
    alias = tmp_path / "upstream-alias"
    alias.symlink_to(clean_root, target_is_directory=True)
    with pytest.raises(PRODUCER.ScopeViolation):
        PRODUCER.inspect_upstream_bundle(clean_config, root=alias)

    payloads = _sample_preterminal()
    output = tmp_path / "published" / "pack"
    output.parent.mkdir()
    PRODUCER.publish_pack(output, payloads, production=False, config=config)
    os.link(output / PRODUCER.GATE_MEMBER_NAMES[0], tmp_path / "output-hardlink")
    with pytest.raises(PRODUCER.ScopeViolation):
        PRODUCER.inspect_published_pack(output, payloads)


def test_canonical_parent_rename_preserves_unpublished_temp(tmp_path: Path) -> None:
    config = _bound_config()
    payloads = _sample_preterminal()
    parent = tmp_path / "canonical"
    parent.mkdir()
    moved = tmp_path / "moved"
    output = parent / "pack"

    def rename_parent(event: str) -> None:
        if event == "before_atomic_rename":
            parent.rename(moved)
            parent.mkdir()

    with pytest.raises(PRODUCER.PartialPublicationError):
        PRODUCER.publish_pack(
            output,
            payloads,
            production=False,
            config=config,
            fault_injector=rename_parent,
        )
    assert not output.exists()
    assert any(path.name.startswith(".pack.tmp.") for path in moved.iterdir())


def test_post_rename_and_post_marker_failures_report_committed_truth(
    tmp_path: Path,
) -> None:
    config = _bound_config()
    payloads = _sample_preterminal()
    primary = tmp_path / "primary" / "pack"
    primary.parent.mkdir()

    def after_rename(event: str) -> None:
        if event == "after_atomic_rename":
            raise RuntimeError("fixture after rename")

    with pytest.raises(PRODUCER.PublicationStateError) as primary_error:
        PRODUCER.publish_pack(
            primary,
            payloads,
            production=False,
            config=config,
            fault_injector=after_rename,
        )
    assert primary_error.value.publication_state == "COMMITTED_EXACT"
    PRODUCER.inspect_published_pack(primary, payloads)

    fallback = tmp_path / "fallback" / "pack"
    fallback.parent.mkdir()

    def unsupported(*_args: Any) -> None:
        raise PRODUCER.AtomicNoReplaceUnsupported(38)

    def after_marker(event: str) -> None:
        if event == "after_fallback_marker":
            raise RuntimeError("fixture after marker")

    with pytest.raises(PRODUCER.PublicationStateError) as fallback_error:
        PRODUCER.publish_pack(
            fallback,
            payloads,
            production=False,
            config=config,
            rename_noreplace=unsupported,
            fault_injector=after_marker,
        )
    assert fallback_error.value.publication_state in {
        "COMMIT_MARKER_PRESENT_NOT_ACCEPTED",
        "COMMIT_MARKER_EXACT_DURABILITY_UNVERIFIED",
    }
    PRODUCER.inspect_published_pack(fallback, payloads)
