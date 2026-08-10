from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

import pytest


STAGING_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = STAGING_ROOT / "configs/route_a_v3_gse200304_published_endpoint_runtime_sync_v1.json"
SCRIPT_PATH = STAGING_ROOT / "scripts/route_a_v3/gse200304_published_endpoint_runtime_sync.py"
SPEC = importlib.util.spec_from_file_location("gse200304_evt037_runtime_sync", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
runtime_sync = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runtime_sync)


def read_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def refresh_core(config: dict[str, Any]) -> None:
    config["implementation_binding"]["compiled_core_sha256"] = runtime_sync.compiled_core_sha256(config)


def bind_config(config: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(config)
    binding = value["implementation_binding"]
    binding["status"] = "BOUND"
    binding["implementation_commit"] = "1" * 40
    binding["implementation_script_sha256"] = "2" * 64
    binding["implementation_test_sha256"] = "3" * 64
    refresh_core(value)
    return value


def artifact_documents(config: dict[str, Any]) -> dict[str, bytes]:
    blockers = config["unresolved_blockers"]
    integrity = {
        "contract_id": config["contract_id"], "protocol_id": "ROUTE_A_V3_GSE200304_PUBLISHED_ENDPOINT_A1_V1",
        "dataset_id": "GSE200304", "network_accessed": False,
        "raw_fastq_or_alignment_input_count": 0, "external_code_executed": False,
        "aggregate_only": True,
    }
    endpoint = {
        "contract_id": config["contract_id"], "protocol_id": "ROUTE_A_V3_GSE200304_PUBLISHED_ENDPOINT_A1_V1",
        "dataset_id": "GSE200304", "aggregate_only": True,
        "published_endpoint_is_not_raw_replay": True,
        "published_endpoint_is_not_canonical_materialization": True,
        "table_s2": {
            "raw_row_count": 13850, "unique_content_row_count": 13836,
            "duplicate_extra_row_count": 14, "duplicated_pair_count": 7,
            "deduplicated_pair_count": 6885, "deduplicated_control_count": 66,
            "all_pair_orientation_counts": {"FORWARD": 3497, "REVERSE_COMPLEMENT": 3388, "UNRESOLVED": 0},
        },
        "table_s3": {
            "primary_data_row_count": 13544, "primary_pair_count": 6772,
            "finite_statistic_rows": {"HighPoly:RNA": 6538, "TotalPoly:RNA": 6547},
            "na_statistic_rows": {"HighPoly:RNA": 234, "TotalPoly:RNA": 225},
            "both_comparisons_finite_pair_count": 6538, "primary_only_finite_pair_count": 9,
            "secondary_only_finite_pair_count": 0, "neither_comparison_finite_pair_count": 225,
            "translation_formula_count": 13544, "cached_translation_string_count": 13544,
            "opaque_control_data_cell_read_count": 0,
            "cached_translation_counts_role": "DESCRIPTIVE_ONLY_NOT_MEMBERSHIP_OR_GATE",
        },
        "endpoint_boundary": {
            "joined_pair_count": 6772, "table_s2_absent_from_table_s3_count": 113,
            "joined_pair_orientation_counts": {"FORWARD": 3451, "REVERSE_COMPLEMENT": 3321, "UNRESOLVED": 0},
            "primary_finite_effect_pair_count": 6547, "primary_na_pair_count": 225,
            "primary_total_attrition_count": 338,
            "primary_complete_distinct_wt_201nt_proxy_group_count": 6544,
            "primary_complete_wt_201nt_proxy_pool_size_counts": {"1": 6541, "2": 3},
            "biological_source_group_authority_closed": False,
            "study_level_reported_biological_replicate_count": 6,
            "row_level_effective_replicate_count": None, "standard_error": None,
            "power_effective_n": None, "true_a2_dense_candidate_count": 0,
        },
    }
    report = {
        "contract_id": config["contract_id"], "protocol_id": "ROUTE_A_V3_GSE200304_PUBLISHED_ENDPOINT_A1_V1",
        "dataset_id": "GSE200304", "execution_outcome": "ENGINEERING_SUCCESS_IMMUTABLY_BLOCKED",
        "qualification_status": "BLOCKED_NOT_QUALIFIED", "scientific_claim_status": "NOT_ESTABLISHED",
        "qualified": False, "ordinary_study_contribution": 0,
        "a1_intervention_study_contribution": 0, "true_a2_dense_study_contribution": 0,
        "canonical_record_count": 0, "training_allowed": False,
        "model_selection_allowed": False, "next_phase_authorized": False,
        "unresolved_blockers": blockers,
        "implementation_binding": {
            "implementation_commit": "84fc6932de32fe0de8e5ddf540e14dee62a2b723",
            "binding_commit": "d06bb991ca9c9052671ee5c5ad7d92dfb69b0189",
        },
    }
    payloads = {
        "INPUT_INTEGRITY_AUDIT.json": runtime_sync.json_bytes(integrity),
        "PUBLISHED_ENDPOINT_AUDIT.json": runtime_sync.json_bytes(endpoint),
        "QUALIFICATION_REPORT.json": runtime_sync.json_bytes(report),
    }
    payloads["SHA256SUMS"] = "".join(
        f"{runtime_sync.sha256(payloads[name])}  {name}\n" for name in sorted(payloads)
    ).encode("ascii")
    marker = {
        "record_type": "GSE200304_PUBLISHED_ENDPOINT_A1_PUBLICATION_COMMIT",
        "execution_outcome": "ENGINEERING_SUCCESS_IMMUTABLY_BLOCKED",
        "bundle_member_names": sorted(payloads), "bundle_member_count": 4,
        "sha256sums_sha256": runtime_sync.sha256(payloads["SHA256SUMS"]),
        "committed": True, "terminal_marker_written_last": True,
        "terminal_publication_operation": "FSYNCED_STAGED_HARDLINK_NO_REPLACE",
    }
    payloads["PUBLICATION_COMMIT.json"] = runtime_sync.json_bytes(marker)
    return payloads


def predecessor_payloads() -> dict[str, bytes]:
    status = {
        "run_status": "IN_PROGRESS", "evidence_status": "BLOCKED_PENDING_PUBLIC_EVIDENCE",
        "gate_status": "A1_BLOCKED_PENDING_PUBLIC_EVIDENCE", "claim_status": "NOT_ESTABLISHED",
        "qualified_a1_studies": 0, "qualified_a2_dense_studies": 0,
        "metadata_only_qualification_count": 0, "training_started": False,
        "next_phase_authorized": False, "updated_at": "2026-08-11T03:09:21+08:00",
        "code_commit": "28cd2f132d022fea6ac43e1f89d6673d02a9c97d", "legacy": "PRESERVED",
    }
    manifest = {
        "run_status": "IN_PROGRESS", "evidence_status": "BLOCKED_PENDING_PUBLIC_EVIDENCE",
        "claim_status": "NOT_ESTABLISHED", "code_commit": "28cd2f132d022fea6ac43e1f89d6673d02a9c97d",
        "active_authority_commit": "d078060c81114687db5068902a5aad5d9bedbee6",
        "outputs": [
            {"artifact_type": f"OLD_{index:03d}", "absolute_path": f"/old/{index:03d}", "sha256": f"{index:064x}", "status": "COMPLETE"}
            for index in range(93)
        ],
    }
    events = [
        {"event_id": f"A1-EVT-{index:03d}", "at": "2026-08-10T00:00:00+08:00", "event": "OLD"}
        for index in range(1, 36)
    ]
    events.append({"event_id": "A1-EVT-036", "at": "2026-08-11T03:09:21+08:00", "event": "EXACT_PREDECESSOR"})
    return {
        "STATUS.json": runtime_sync.json_bytes(status),
        "RUN_MANIFEST.json": runtime_sync.json_bytes(manifest),
        "EVENT_LOG.jsonl": b"".join(runtime_sync.compact_json_line(item) for item in events),
    }


def make_context(tmp_path: Path) -> tuple[dict[str, Any], dict[str, bytes], dict[str, bytes], dict[str, Any], Path, Path]:
    config = bind_config(read_config())
    run_root = tmp_path / "run"
    artifact_root = run_root / "GSE200304_PUBLISHED_ENDPOINT_A1_SYNTHETIC"
    allowed = tmp_path / "allowed"
    prepared = allowed / "job"
    config["runtime"]["run_root"] = str(run_root)
    config["runtime"]["artifact_root"] = str(artifact_root)
    config["runtime"]["allowed_prepared_root"] = str(allowed)
    predecessor = predecessor_payloads()
    for name, payload in predecessor.items():
        config["runtime"]["predecessor_mutables"][name]["bytes"] = len(payload)
        config["runtime"]["predecessor_mutables"][name]["sha256"] = runtime_sync.sha256(payload)
    artifact = artifact_documents(config)
    for item in config["runtime"]["artifact_members"]:
        payload = artifact[item["name"]]
        item["bytes"], item["sha256"] = len(payload), runtime_sync.sha256(payload)
    refresh_core(config)
    runtime_sync.validate_bound_config(config)
    run_root.mkdir()
    artifact_root.mkdir()
    allowed.mkdir()
    for name, payload in predecessor.items():
        (run_root / name).write_bytes(payload)
    for name, payload in artifact.items():
        (artifact_root / name).write_bytes(payload)
    authority = {
        "status": "PASS_EXACT_LEDGER_L_TO_I_TO_CONFIG_ONLY_B",
        "binding_commit": "4" * 40, "head_commit": "4" * 40,
        "origin_branch_head_commit": "4" * 40,
        "config_sha256": runtime_sync.sha256(runtime_sync.json_bytes(config)),
    }
    return config, predecessor, artifact, authority, run_root, prepared


def prepare_context(tmp_path: Path) -> tuple[dict[str, Any], dict[str, bytes], dict[str, bytes], dict[str, Any], Path, Path]:
    context = make_context(tmp_path)
    config, _predecessor, _artifact, authority, run_root, prepared = context
    result = runtime_sync.prepare_runtime_sync(
        prepared_directory=prepared, recorded_at="2026-08-11T05:00:00+08:00",
        production=False, config_override=config, authority_override=authority,
        run_root_override=run_root,
    )
    assert result["status"] == "PREPARED_NOT_PUBLISHED"
    return context


def tree_bytes(root: Path) -> dict[str, bytes]:
    return {item.name: item.read_bytes() for item in root.iterdir() if item.is_file() and not item.is_symlink()}


def test_production_config_is_exact_unknown_schema_and_frozen_metadata() -> None:
    config = read_config()
    assert config["implementation_binding"]["status"] == "UNKNOWN_NOT_ASSERTED"
    assert config["implementation_binding"]["implementation_commit"] == "UNKNOWN_NOT_ASSERTED"
    assert config["repository_authority"]["ledger_commit"] == "bdd30d50c04a565b68a1b33b5cb164e4eda3fa9f"
    assert config["repository_authority"]["ledger_semantics"]["runtime_sync_status"] == "PENDING_NO_EVT_037"
    assert [(item["name"], item["bytes"], item["sha256"]) for item in config["runtime"]["artifact_members"]] == [
        ("INPUT_INTEGRITY_AUDIT.json", 3610, "e87723673dfea6dca654b670d1c05f331f240a53d52d81d1207fbfc50d9a4fe8"),
        ("PUBLISHED_ENDPOINT_AUDIT.json", 4981, "d849da8cc29a2a4419c85d69e5084736b6b41b03cac90263aa2620be3fe3acc7"),
        ("QUALIFICATION_REPORT.json", 2095, "006db8da47dc2bbc0c313a156ae16ab79a3f6aebe324d37806820ac9240b100d"),
        ("SHA256SUMS", 281, "e1720881f8bcfaaea1fef613dd4ee059c08da1bbd11bafc32a8fccdea0a43515"),
        ("PUBLICATION_COMMIT.json", 973, "f1e5d0752bcc12db0b0eaabe0e75efdb6f2c48dfba4c3bae6bff99a302194cfc"),
    ]
    bound = bind_config(config)
    runtime_sync.validate_bound_config(bound)
    assert runtime_sync.expected_unknown_i_config(bound) == config


def test_unknown_stops_before_runtime_artifact_or_output_access(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(runtime_sync, "open_directory", lambda _path: calls.append("runtime") or 1)
    monkeypatch.setattr(runtime_sync, "open_absolute_directory_nofollow", lambda _path: calls.append("artifact") or 1)
    with pytest.raises(runtime_sync.BindingError, match="not BOUND"):
        runtime_sync.prepare_runtime_sync(
            prepared_directory=tmp_path / "prepared", recorded_at="2026-08-11T05:00:00+08:00",
            production=False, config_override=read_config(), authority_override={}, run_root_override=tmp_path / "run",
        )
    assert calls == [] and not (tmp_path / "prepared").exists()


def test_binding_pending_hash_and_compiled_core_drift_fail_closed() -> None:
    config = bind_config(read_config())
    config["implementation_binding"]["implementation_script_sha256"] = "PENDING_SCRIPT"
    with pytest.raises(runtime_sync.BindingError, match="lowercase hex"):
        runtime_sync.validate_bound_config(config)
    config = bind_config(read_config())
    config["implementation_binding"]["compiled_core_sha256"] = "f" * 64
    with pytest.raises(runtime_sync.RuntimeSyncError, match="compiled core"):
        runtime_sync.validate_bound_config(config)


@pytest.mark.parametrize("failure", ["nonancestor", "non_config_binding", "dirty", "hash_drift"])
def test_repo_audit_positive_and_fail_closed_variants(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str) -> None:
    config = bind_config(read_config())
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr(runtime_sync, "PRODUCTION_REPO_ROOT", repo)
    authority = config["repository_authority"]
    implementation = config["implementation_binding"]["implementation_commit"]
    ledger, qualifier_b, head = authority["ledger_commit"], authority["ledger_commit_expected_parent"], "4" * 40
    config_payload = runtime_sync.json_bytes(config)
    i_payload = runtime_sync.json_bytes(runtime_sync.expected_unknown_i_config(config))
    blobs: dict[str, bytes] = {}
    for item in authority["ledger_files"]:
        if item["path"].endswith("route_a_v3_a1_interim.yaml"):
            payload = ("  gse200304_published_endpoint_a1_v1:\n"
                       "    publication_state: COMMITTED_ACCEPTED\n"
                       "    execution_outcome: ENGINEERING_SUCCESS_IMMUTABLY_BLOCKED\n"
                       "    qualification_status: BLOCKED_NOT_QUALIFIED\n"
                       "    runtime_sync_status: PENDING_NO_EVT_037\n").encode()
        else:
            payload = f"ledger {item['path']}\n".encode()
        blobs[item["path"]] = payload
        item["sha256"] = runtime_sync.sha256(payload)
    for path, field in ((runtime_sync.SCRIPT_REPO_PATH, "implementation_script_sha256"), (runtime_sync.TEST_REPO_PATH, "implementation_test_sha256")):
        blobs[path] = f"implementation {path}\n".encode()
        config["implementation_binding"][field] = runtime_sync.sha256(blobs[path])
    qualifier = authority["external_qualifier_binding"]
    for path_field, hash_field in (("config_path", "config_sha256"), ("script_path", "script_sha256"), ("test_path", "test_sha256")):
        blobs[qualifier[path_field]] = f"qualifier {path_field}\n".encode()
        qualifier[hash_field] = runtime_sync.sha256(blobs[qualifier[path_field]])
    refresh_core(config)
    config_payload = runtime_sync.json_bytes(config)
    i_payload = runtime_sync.json_bytes(runtime_sync.expected_unknown_i_config(config))

    def fake_git(_repo: Path, *args: str, allowed_returncodes: tuple[int, ...] = (0,)) -> bytes:
        if args == ("rev-parse", "HEAD"): return f"{head}\n".encode()
        if args == ("rev-parse", "--verify", f"refs/remotes/origin/{authority['branch']}"): return f"{head}\n".encode()
        if args == ("rev-parse", "--abbrev-ref", "@{upstream}"): return f"origin/{authority['branch']}\n".encode()
        if args == ("rev-parse", "@{upstream}"): return f"{head}\n".encode()
        if args == ("rev-parse", "--abbrev-ref", "HEAD"): return f"{authority['branch']}\n".encode()
        if args == ("rev-parse", f"{head}^"): return f"{implementation}\n".encode()
        if args == ("rev-parse", f"{implementation}^"): return f"{ledger}\n".encode()
        if args == ("rev-parse", f"{ledger}^"): return f"{qualifier_b}\n".encode()
        if args == ("status", "--porcelain=v1", "--untracked-files=all"): return b"M dirty\n" if failure == "dirty" else b""
        if args[:2] == ("merge-base", "--is-ancestor"):
            if failure == "nonancestor": raise runtime_sync.AuthorityError("not ancestor")
            return b""
        if args[:4] == ("diff-tree", "--no-commit-id", "--name-only", "-r"):
            commit = args[4]
            paths = authority["implementation_commit_exact_changed_paths"] if commit == implementation else authority["binding_commit_exact_changed_paths"]
            if failure == "non_config_binding" and commit == head: paths = paths + [runtime_sync.SCRIPT_REPO_PATH]
            return ("\n".join(paths) + "\n").encode()
        if args[0] == "show":
            commit, relative = args[1].split(":", 1)
            if relative == runtime_sync.CONFIG_REPO_PATH: return config_payload if commit == head else i_payload
            payload = blobs[relative]
            if failure == "hash_drift" and relative == runtime_sync.SCRIPT_REPO_PATH and commit == implementation: return payload + b"drift"
            return payload
        raise AssertionError(args)

    monkeypatch.setattr(runtime_sync, "_run_git", fake_git)
    monkeypatch.setattr(runtime_sync, "read_regular_path", lambda path: blobs[str(path.relative_to(repo))])
    if failure == "positive":
        assert runtime_sync.audit_repo_authority(repo, config, config_payload)["ledger_is_ancestor"] is True
    else:
        with pytest.raises(runtime_sync.RuntimeSyncError):
            runtime_sync.audit_repo_authority(repo, config, config_payload)


def test_repo_audit_positive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Reuse the parameterized harness with a sentinel that activates no failure.
    test_repo_audit_positive_and_fail_closed_variants(tmp_path, monkeypatch, "positive")


@pytest.mark.parametrize("failure", ["missing", "tamper", "symlink", "extra", "sha256sums", "terminal"])
def test_artifact_bundle_closure_failures(tmp_path: Path, failure: str) -> None:
    config, _predecessor, artifact, _authority, _run_root, _prepared = make_context(tmp_path)
    root = Path(config["runtime"]["artifact_root"])
    if failure == "missing":
        (root / "INPUT_INTEGRITY_AUDIT.json").unlink()
    elif failure == "tamper":
        (root / "PUBLISHED_ENDPOINT_AUDIT.json").write_bytes(artifact["PUBLISHED_ENDPOINT_AUDIT.json"] + b" ")
    elif failure == "symlink":
        leaf = root / "QUALIFICATION_REPORT.json"
        target = tmp_path / "target.json"
        target.write_bytes(artifact["QUALIFICATION_REPORT.json"])
        leaf.unlink(); leaf.symlink_to(target)
    elif failure == "extra":
        (root / "EXTRA").write_bytes(b"extra")
    elif failure == "sha256sums":
        payload = artifact["SHA256SUMS"] + b"extra\n"
        (root / "SHA256SUMS").write_bytes(payload)
        item = next(item for item in config["runtime"]["artifact_members"] if item["name"] == "SHA256SUMS")
        item["bytes"], item["sha256"] = len(payload), runtime_sync.sha256(payload)
    else:
        marker = json.loads(artifact["PUBLICATION_COMMIT.json"])
        marker["terminal_marker_written_last"] = False
        payload = runtime_sync.json_bytes(marker)
        (root / "PUBLICATION_COMMIT.json").write_bytes(payload)
        item = next(item for item in config["runtime"]["artifact_members"] if item["name"] == "PUBLICATION_COMMIT.json")
        item["bytes"], item["sha256"] = len(payload), runtime_sync.sha256(payload)
    refresh_core(config)
    with pytest.raises(runtime_sync.PublicationError):
        runtime_sync.validate_artifact_bundle(config)


def test_ledger_semantics_pending_hash_blocker_gate_and_type_drift() -> None:
    mutations = []
    config = bind_config(read_config()); config["repository_authority"]["ledger_semantics"]["runtime_sync_status"] = "SYNCED_EVT_037"; mutations.append(config)
    config = bind_config(read_config()); config["repository_authority"]["ledger_files"][0]["sha256"] = "PENDING_LEDGER"; mutations.append(config)
    config = bind_config(read_config()); config["unresolved_blockers"] = config["unresolved_blockers"][::-1]; mutations.append(config)
    config = bind_config(read_config()); config["successor_invariants"]["qualified"] = True; mutations.append(config)
    config = bind_config(read_config()); config["runtime"]["predecessor_event_count"] = True; mutations.append(config)
    for config in mutations:
        refresh_core(config)
        with pytest.raises(runtime_sync.RuntimeSyncError):
            runtime_sync.validate_bound_config(config)


def test_predecessor_tail_and_output_drift() -> None:
    config = bind_config(read_config())
    payloads = predecessor_payloads()
    status = json.loads(payloads["STATUS.json"])
    manifest = json.loads(payloads["RUN_MANIFEST.json"])
    events = runtime_sync.load_json_lines(payloads["EVENT_LOG.jsonl"], label="events")
    bad_events = copy.deepcopy(events); bad_events[-1]["event_id"] = "A1-EVT-035"
    with pytest.raises(runtime_sync.PublicationError, match="tail"):
        runtime_sync._validate_predecessor_objects(status, manifest, bad_events, config)
    bad_manifest = copy.deepcopy(manifest); bad_manifest["outputs"].pop()
    with pytest.raises(runtime_sync.PublicationError, match="93"):
        runtime_sync._validate_predecessor_objects(status, bad_manifest, events, config)


def test_prepare_exact_93_to_102_sync_and_privacy(tmp_path: Path) -> None:
    config, predecessor, _artifact, _authority, _run_root, prepared = prepare_context(tmp_path)
    payloads = {item.name: item.read_bytes() for item in prepared.iterdir()}
    assert set(payloads) == set(runtime_sync.MUTABLE_NAMES) | set(runtime_sync.immutable_names(config))
    manifest = runtime_sync.load_json(payloads["RUN_MANIFEST.json"], label="manifest")
    old_manifest = runtime_sync.load_json(predecessor["RUN_MANIFEST.json"], label="old manifest")
    assert len(manifest["outputs"]) == 102 and manifest["outputs"][:93] == old_manifest["outputs"]
    assert [Path(item["absolute_path"]).name for item in manifest["outputs"][93:]] == [
        "INPUT_INTEGRITY_AUDIT.json", "PUBLISHED_ENDPOINT_AUDIT.json", "QUALIFICATION_REPORT.json",
        "SHA256SUMS", "PUBLICATION_COMMIT.json",
        "STATUS_PRE_GSE200304_PUBLISHED_ENDPOINT_RUNTIME_SYNC_V1.json",
        "RUN_MANIFEST_PRE_GSE200304_PUBLISHED_ENDPOINT_RUNTIME_SYNC_V1.json",
        "EVENT_LOG_PRE_GSE200304_PUBLISHED_ENDPOINT_RUNTIME_SYNC_V1.jsonl",
        "A1_GSE200304_PUBLISHED_ENDPOINT_RUNTIME_SYNC_V1.json",
    ]
    sync_name = config["runtime"]["sync_name"]
    sync = runtime_sync.load_json(payloads[sync_name], label="sync")
    assert sync["scientific_blockers"] == {"count": 8, "exact": config["unresolved_blockers"]}
    assert sync["a1_gate_snapshot"] == config["successor_invariants"]
    assert sync["self_hash"] == "NOT_APPLICABLE_BOUND_BY_OUTER_RUN_MANIFEST"
    for name in runtime_sync.MUTABLE_NAMES:
        assert runtime_sync.sha256(payloads[name]).encode() not in payloads[sync_name]
    generated = payloads[sync_name] + payloads["EVENT_LOG.jsonl"][len(predecessor["EVENT_LOG.jsonl"]):]
    assert b'"rows":' not in generated and b'"row_payload":' not in generated
    assert sync["access_and_materialization_boundary"]["row_level_payload_included"] is False
    assert sync["access_and_materialization_boundary"]["row_identifier_payload_included"] is False
    assert b"ACGTACGTACGTACGTACGTACGT" not in generated
    events = runtime_sync.load_json_lines(payloads["EVENT_LOG.jsonl"], label="events")
    assert len(events) == 37 and events[-1]["event_id"] == "A1-EVT-037"


def test_first_publish_and_idempotent_retry(tmp_path: Path) -> None:
    config, _predecessor, _artifact, authority, run_root, prepared = prepare_context(tmp_path)
    first = runtime_sync.publish_prepared(prepared_directory=prepared, production=False, config_override=config, authority_override=authority, run_root_override=run_root)
    assert first["status"] == "PUBLISHED_VERIFIED"
    after = tree_bytes(run_root)
    second = runtime_sync.publish_prepared(prepared_directory=prepared, production=False, config_override=config, authority_override=authority, run_root_override=run_root)
    assert second["status"] == "PUBLISHED_VERIFIED" and tree_bytes(run_root) == after
    assert all(second["results"][name]["state"] == "EXISTING_NEW_EXACT_REUSED" for name in runtime_sync.MUTABLE_NAMES)


@pytest.mark.parametrize("new_prefix_count", [0, 1, 2, 3])
def test_all_four_allowed_recovery_states(tmp_path: Path, new_prefix_count: int) -> None:
    config, _predecessor, _artifact, authority, run_root, prepared = prepare_context(tmp_path)
    for name in runtime_sync.MUTABLE_NAMES[:new_prefix_count]:
        (run_root / name).write_bytes((prepared / name).read_bytes())
    result = runtime_sync.publish_prepared(prepared_directory=prepared, production=False, config_override=config, authority_override=authority, run_root_override=run_root)
    assert result["status"] == "PUBLISHED_VERIFIED"
    assert [result["mutable_preflight"][name] for name in runtime_sync.MUTABLE_NAMES] == ["NEW_EXACT"] * new_prefix_count + ["OLD_EXACT"] * (3 - new_prefix_count)


def test_postcommit_warning_is_truthful_and_retryable(tmp_path: Path) -> None:
    config, _predecessor, _artifact, authority, run_root, prepared = prepare_context(tmp_path)
    injected = False
    def inject(point: str) -> None:
        nonlocal injected
        if point == "immutable_post_link_directory_fsync" and not injected:
            injected = True
            raise OSError("synthetic postcommit warning")
    warning = runtime_sync.publish_prepared(prepared_directory=prepared, production=False, config_override=config, authority_override=authority, run_root_override=run_root, fault_injector=inject)
    assert warning["status"] == "COMMITTED_WITH_WARNINGS_REQUIRES_IDEMPOTENT_RETRY"
    assert (run_root / warning["warning_member"]).exists()
    recovered = runtime_sync.publish_prepared(prepared_directory=prepared, production=False, config_override=config, authority_override=authority, run_root_override=run_root)
    assert recovered["status"] == "PUBLISHED_VERIFIED"


def test_event_post_replace_directory_fsync_fault_retry_reconfirms_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _predecessor, _artifact, authority, run_root, prepared = prepare_context(tmp_path)
    mutable_fsync_attempts = 0

    def inject(point: str) -> None:
        nonlocal mutable_fsync_attempts
        if point == "mutable_post_replace_directory_fsync":
            mutable_fsync_attempts += 1
            if mutable_fsync_attempts == 3:
                raise OSError("synthetic EVENT_LOG directory fsync fault")

    warning = runtime_sync.publish_prepared(
        prepared_directory=prepared,
        production=False,
        config_override=config,
        authority_override=authority,
        run_root_override=run_root,
        fault_injector=inject,
    )
    assert warning["status"] == "COMMITTED_WITH_WARNINGS_REQUIRES_IDEMPOTENT_RETRY"
    assert warning["warning_member"] == "EVENT_LOG.jsonl"
    assert (run_root / "EVENT_LOG.jsonl").read_bytes() == (prepared / "EVENT_LOG.jsonl").read_bytes()

    real_fsync = runtime_sync.os.fsync
    retry_fsync_calls: list[int] = []

    def tracking_fsync(descriptor: int) -> None:
        retry_fsync_calls.append(descriptor)
        real_fsync(descriptor)

    monkeypatch.setattr(runtime_sync.os, "fsync", tracking_fsync)
    recovered = runtime_sync.publish_prepared(
        prepared_directory=prepared,
        production=False,
        config_override=config,
        authority_override=authority,
        run_root_override=run_root,
    )
    assert recovered["status"] == "PUBLISHED_VERIFIED"
    assert len(retry_fsync_calls) == len(runtime_sync.immutable_names(config)) + len(
        runtime_sync.MUTABLE_NAMES
    )
    assert all(
        recovered["results"][name]["state"] == "EXISTING_EXACT_REUSED"
        for name in runtime_sync.immutable_names(config)
    )
    assert all(
        recovered["results"][name]["state"] == "EXISTING_NEW_EXACT_REUSED"
        for name in runtime_sync.MUTABLE_NAMES
    )


def test_immutable_post_link_unlink_fault_leaves_closed_namespace_failure(
    tmp_path: Path,
) -> None:
    config, _predecessor, _artifact, authority, run_root, prepared = prepare_context(tmp_path)
    injected = False

    def inject(point: str) -> None:
        nonlocal injected
        if point == "immutable_post_link_unlink" and not injected:
            injected = True
            raise OSError("synthetic immutable temporary unlink fault")

    warning = runtime_sync.publish_prepared(
        prepared_directory=prepared,
        production=False,
        config_override=config,
        authority_override=authority,
        run_root_override=run_root,
        fault_injector=inject,
    )
    assert warning["status"] == "COMMITTED_WITH_WARNINGS_REQUIRES_IDEMPOTENT_RETRY"
    stale = [item for item in run_root.iterdir() if item.name.startswith(".evt037.")]
    assert len(stale) == 1
    assert stale[0].stat().st_ino == (run_root / warning["warning_member"]).stat().st_ino
    with pytest.raises(runtime_sync.PublicationError, match="namespace unclosed"):
        runtime_sync.publish_prepared(
            prepared_directory=prepared,
            production=False,
            config_override=config,
            authority_override=authority,
            run_root_override=run_root,
        )


def test_invalid_event_first_state_is_rejected_before_write(tmp_path: Path) -> None:
    config, _predecessor, _artifact, authority, run_root, prepared = prepare_context(tmp_path)
    (run_root / "EVENT_LOG.jsonl").write_bytes((prepared / "EVENT_LOG.jsonl").read_bytes())
    before = tree_bytes(run_root)
    with pytest.raises(runtime_sync.PublicationError, match="publication-order state"):
        runtime_sync.publish_prepared(prepared_directory=prepared, production=False, config_override=config, authority_override=authority, run_root_override=run_root)
    assert tree_bytes(run_root) == before
