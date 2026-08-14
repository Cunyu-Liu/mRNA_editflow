from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


STAGING_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    STAGING_ROOT
    / "configs/route_a_v3_dec027_six_rescue_terminal_aggregate_evidence_runtime_sync_v1.json"
)
SCRIPT_PATH = (
    STAGING_ROOT
    / "scripts/route_a_v3/dec027_six_rescue_terminal_aggregate_evidence_runtime_sync.py"
)


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("dec027_six_sync", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sync = _load_module()


def shipped_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def bound_config() -> dict[str, Any]:
    config = shipped_config()
    ledger = config["repository_authority"]["predecessor_ledger"]
    ledger.update(
        {
            "status": "BOUND",
            "commit": "a" * 40,
            "integration_id": sync.LEDGER_INTEGRATION_ID,
            "manifest_status": sync.LEDGER_MANIFEST_STATUS,
            "registered_lineage_ids": list(sync.LINEAGE_IDS),
        }
    )
    for index, item in enumerate(ledger["frozen_blobs"], start=1):
        item["sha256"] = f"{index:x}" * 64
    config["implementation_binding"].update(
        {
            "status": "BOUND",
            "implementation_commit": "b" * 40,
            "implementation_script_sha256": "c" * 64,
            "implementation_test_sha256": "d" * 64,
        }
    )
    sync.validate_bound_config(config)
    return config


def wholly_unknown_config() -> dict[str, Any]:
    """Normalize either legal disk I/B state to the original zero-I/O draft."""

    config = shipped_config()
    binding = config["implementation_binding"]
    for key in (
        "status",
        "implementation_commit",
        "implementation_script_sha256",
        "implementation_test_sha256",
    ):
        binding[key] = sync.UNKNOWN
    ledger = config["repository_authority"]["predecessor_ledger"]
    for key in ("status", "commit", "integration_id", "manifest_status"):
        ledger[key] = sync.UNKNOWN
    ledger["registered_lineage_ids"] = [sync.UNKNOWN] * 6
    for item in ledger["frozen_blobs"]:
        item["sha256"] = sync.UNKNOWN
    sync.validate_static_config(config)
    return config


def predecessor_fixture(config: dict[str, Any]) -> dict[str, bytes]:
    science = sync._science_fields(config)
    status = {
        **science,
        "updated_at": "2026-08-15T00:00:00+08:00",
        "dec027_authority_runtime_sync_status": "SYNCED_EVT_059",
        "unrelated_status_field": {"preserve": [1, 2, 3]},
    }
    outputs = [
        {
            "absolute_path": f"/runtime/predecessor_{index:03d}.json",
            "artifact_type": "PREDECESSOR",
            "bytes": index + 1,
            "sha256": f"{index % 16:x}" * 64,
            "status": "COMPLETE",
        }
        for index in range(256)
    ]
    manifest = {
        **science,
        "updated_at": "2026-08-15T00:00:00+08:00",
        "dec027_authority_runtime_sync_status": "SYNCED_EVT_059",
        "registered_artifact_count": 8,
        "outputs": outputs,
        "unrelated_manifest_field": "preserve-exactly",
    }
    events = []
    for index in range(1, 60):
        event = {
            "event_id": f"A1-EVT-{index:03d}",
            "at": f"2026-08-14T{index // 60:02d}:{index % 60:02d}:00+08:00",
            "predecessor_event_id": None if index == 1 else f"A1-EVT-{index - 1:03d}",
            "decision_id": "V3-DEC-027" if index == 59 else "HISTORICAL",
        }
        events.append(event)
    payloads = {
        "STATUS.json": sync.json_bytes(status),
        "RUN_MANIFEST.json": sync.json_bytes(manifest),
        "EVENT_LOG.jsonl": b"".join(sync.json_line(event) for event in events),
    }
    runtime = config["runtime"]
    for name, payload in payloads.items():
        runtime["predecessor_mutables"][name]["bytes"] = len(payload)
        runtime["predecessor_mutables"][name]["sha256"] = sync.sha256(payload)
    tail = payloads["EVENT_LOG.jsonl"].splitlines(keepends=True)[-1]
    runtime["predecessor_tail"]["bytes"] = len(tail)
    runtime["predecessor_tail"]["sha256"] = sync.sha256(tail)
    return payloads


def test_disk_candidate_is_legal_draft_i_or_bound_b_and_freezes_exact_facts() -> None:
    config = shipped_config()
    sync.validate_static_config(config)
    ledger_values = sync._ledger_values(
        config["repository_authority"]["predecessor_ledger"]
    )
    binding_values = sync._binding_values(config["implementation_binding"])
    assert ledger_values == [sync.UNKNOWN] * 14 or sync.UNKNOWN not in ledger_values
    assert binding_values == [sync.UNKNOWN] * 4 or sync.UNKNOWN not in binding_values
    if sync.UNKNOWN not in binding_values:
        clean_i = sync.expected_unknown_i_config(config)
        assert sync._binding_values(clean_i["implementation_binding"]) == [
            sync.UNKNOWN
        ] * 4
        assert sync.truth_projection(clean_i) == sync.truth_projection(config)
    assert config["repository_authority"]["base_lifecycle"]["binding_b2"]["commit"] == (
        "679a1c2ae89db7d6a9894f9299de7ce38b30ecdb"
    )
    expected_reports = [
        (
            "GSE217518",
            7833,
            "03de0d423604518653a5188696d8186c82fa66e7858e7f498052fc67256e8884",
        ),
        (
            "ENCSR854RUF",
            11423,
            "3753d6fc5fb4132e43e11f29f9c79a04078a592aaba760c1f8a6e6ed2c5fc6c2",
        ),
        (
            "GSE232572",
            9823,
            "20451d763b8b2bc2658a4bf6163bbef8a2449759fa7fbe1ff5a29f9146cdae2c",
        ),
        (
            "GSE113849",
            16280,
            "7ac51be90de8bbed2562e081a4063b6ed479f8a700d417623bfbefb269384839",
        ),
        (
            "GSE269595",
            13272,
            "7952a74690817f24c3dc1df1ccdb104b9997464ceb97f0320fddd227bd84ac4b",
        ),
        (
            "GSE295080",
            8989,
            "f3f258cd89f58d42270e05b40a67c55fcd18511e6b17d9dc6711f62d0db5aa63",
        ),
    ]
    assert [
        (item["dataset_id"], item["bytes"], item["sha256"])
        for item in config["registered_artifacts"]
    ] == expected_reports
    assert config["runtime"]["predecessor_mutables"] == {
        "STATUS.json": {
            "bytes": 32620,
            "sha256": "5fb567a2f081658206cd528b15c12ea36f59aea7d41df9c5d2e161ee5456058c",
            "snapshot_name": "STATUS_PRE_DEC027_SIX_RESCUE_TERMINAL_AGGREGATE_EVIDENCE_RUNTIME_SYNC_V1.json",
        },
        "RUN_MANIFEST.json": {
            "bytes": 112607,
            "sha256": "df4abe62508ca25c817d2ec9a17fa25a32671feba56b1e51e443833957a5ad32",
            "snapshot_name": "RUN_MANIFEST_PRE_DEC027_SIX_RESCUE_TERMINAL_AGGREGATE_EVIDENCE_RUNTIME_SYNC_V1.json",
        },
        "EVENT_LOG.jsonl": {
            "bytes": 150681,
            "sha256": "03836a81da41e0f81c5af78b1e0d9568e87abce245f42f582b605a7cde5ed177",
            "snapshot_name": "EVENT_LOG_PRE_DEC027_SIX_RESCUE_TERMINAL_AGGREGATE_EVIDENCE_RUNTIME_SYNC_V1.jsonl",
        },
    }
    assert config["access_boundary"]["registered_static_repository_leaf_count"] == 0
    assert all(
        not item["absolute_path"].startswith(config["repository_authority"]["production_repo_root"])
        for item in config["registered_artifacts"]
    )


@pytest.mark.parametrize("entrypoint", ["prepare", "publish", "validate"])
def test_unknown_groups_stop_before_git_report_prepared_or_runtime_io(
    monkeypatch: pytest.MonkeyPatch, entrypoint: str
) -> None:
    config = wholly_unknown_config()

    def poison(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("I/O occurred after an unbound contract")

    monkeypatch.setattr(sync, "audit_production_repository_authority", poison)
    monkeypatch.setattr(sync, "validate_registered_artifacts", poison)
    monkeypatch.setattr(sync, "_prepared_path", poison)
    monkeypatch.setattr(sync, "_read_runtime", poison)
    with pytest.raises(sync.BindingError, match="UNKNOWN_NOT_ASSERTED"):
        if entrypoint == "prepare":
            sync.prepare_runtime_sync(
                config,
                recorded_at="2026-08-15T01:01:00+08:00",
                prepared_directory=Path("/poison/prepared"),
            )
        elif entrypoint == "publish":
            sync.publish_prepared(config, prepared_directory=Path("/poison/prepared"))
        else:
            sync.validate_published(config, prepared_directory=Path("/poison/prepared"))


def test_partial_ledger_and_implementation_groups_fail_closed() -> None:
    for group in ("ledger", "implementation"):
        config = wholly_unknown_config()
        if group == "ledger":
            config["repository_authority"]["predecessor_ledger"]["status"] = "BOUND"
        else:
            config["implementation_binding"]["status"] = "BOUND"
        with pytest.raises(sync.BindingError, match="partially known"):
            sync.validate_static_config(config)


def test_reports_are_exact_byte_checked_without_json_body_parse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = bound_config()
    for index, item in enumerate(config["registered_artifacts"]):
        report = tmp_path / item["name"]
        payload = b"not-json aggregate body " + bytes([index])
        report.write_bytes(payload)
        item["absolute_path"] = str(report)
        item["bytes"] = len(payload)
        item["sha256"] = sync.sha256(payload)

    def poison(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("report body was parsed")

    monkeypatch.setattr(sync, "load_json", poison)
    result = sync.validate_registered_artifacts(config, verify_exact_bytes=True)
    assert result == {
        "artifact_count": 6,
        "exact_byte_validation_count": 6,
        "body_parse_count": 0,
        "payload_field_read_count": 0,
        "registered_artifacts_copied": False,
    }
    (tmp_path / config["registered_artifacts"][0]["name"]).write_bytes(b"drift")
    with pytest.raises(sync.PublicationError, match="identity drift"):
        sync.validate_registered_artifacts(config, verify_exact_bytes=True)


def test_successor_is_exact_59_to_60_256_to_266_and_preserves_science() -> None:
    config = bound_config()
    predecessor = predecessor_fixture(config)
    successors = sync.build_successors(
        config, predecessor, "2026-08-15T01:01:00+08:00"
    )
    sync.validate_successors(config, predecessor, successors)
    status = sync.load_json(successors["STATUS.json"], label="status")
    manifest = sync.load_json(successors["RUN_MANIFEST.json"], label="manifest")
    events = sync.load_events(successors["EVENT_LOG.jsonl"], label="events")
    assert len(events) == 60
    assert successors["EVENT_LOG.jsonl"].startswith(predecessor["EVENT_LOG.jsonl"])
    assert len(manifest["outputs"]) == 266
    assert manifest["registered_artifact_count"] == 14
    assert manifest["outputs"][:256] == sync.load_json(
        predecessor["RUN_MANIFEST.json"], label="old manifest"
    )["outputs"]
    assert manifest["outputs"][256:262] == config["registered_artifacts"]
    assert status["all_six_terminal_reports_registered"] is True
    assert status["stop_rule_evaluation_ready_after_commit"] is True
    assert status["stop_rule_evaluated_by_this_event"] is False
    assert status["conditional_successor_activated"] is False
    for document in (status, manifest):
        for key, value in sync._science_fields(config).items():
            assert document[key] == value
    assert events[-1]["evidence_gate_statuses_changed"] is False
    assert events[-1]["qualification_changed"] is False
    assert events[-1]["frozen_scientific_state"]["current_qualified_counts"] == {
        "ordinary": 1,
        "a1": 1,
        "true_a2": 0,
        "canonical_records": 6547,
    }


def test_predecessor_cas_and_structure_drift_fail_closed() -> None:
    config = bound_config()
    predecessor = predecessor_fixture(config)
    corrupt = dict(predecessor)
    corrupt["STATUS.json"] += b" "
    with pytest.raises(sync.PredecessorError, match="identity drift"):
        sync.validate_predecessor(config, corrupt)
    events = sync.load_events(predecessor["EVENT_LOG.jsonl"], label="events")
    events[-1]["predecessor_event_id"] = "A1-EVT-001"
    structural = dict(predecessor)
    structural["EVENT_LOG.jsonl"] = b"".join(sync.json_line(event) for event in events)
    payload = structural["EVENT_LOG.jsonl"]
    config["runtime"]["predecessor_mutables"]["EVENT_LOG.jsonl"].update(
        {"bytes": len(payload), "sha256": sync.sha256(payload)}
    )
    tail = payload.splitlines(keepends=True)[-1]
    config["runtime"]["predecessor_tail"].update(
        {"bytes": len(tail), "sha256": sync.sha256(tail)}
    )
    with pytest.raises(sync.RuntimeSyncError, match="tail parent"):
        sync.validate_predecessor(config, structural)


def _repository_fixture(
    tmp_path: Path,
) -> tuple[dict[str, Any], dict[tuple[str, ...], bytes], Path]:
    config = bound_config()
    repo = tmp_path / "repo"
    repo.mkdir()
    authority = config["repository_authority"]
    authority["production_repo_root"] = str(repo)
    ledger = authority["predecessor_ledger"]
    ledger_commit = "a" * 40
    implementation = "b" * 40
    head = "e" * 40
    ledger["commit"] = ledger_commit
    config["implementation_binding"]["implementation_commit"] = implementation
    ledger_payloads = {
        item["path"]: f"ledger-{index}".encode()
        for index, item in enumerate(ledger["frozen_blobs"])
    }
    for item in ledger["frozen_blobs"]:
        item["sha256"] = sync.sha256(ledger_payloads[item["path"]])
    base_i = authority["base_lifecycle"]["implementation_i2"]
    base_b = authority["base_lifecycle"]["binding_b2"]
    base_i_payloads = {
        path: f"base-i-{index}".encode()
        for index, path in enumerate(base_i["exact_changed_paths"])
    }
    base_b_payloads = dict(base_i_payloads)
    base_b_payloads[base_b["exact_changed_paths"][0]] = b"base-bound-config"
    base_i["blob_sha256_by_path"] = {
        path: sync.sha256(payload) for path, payload in base_i_payloads.items()
    }
    base_b["blob_sha256_by_path"] = {
        path: sync.sha256(payload) for path, payload in base_b_payloads.items()
    }
    script_payload = b"#!/usr/bin/env python3\n# synthetic exact I script\n"
    test_payload = b"# synthetic exact I test\n"
    config["implementation_binding"]["implementation_script_sha256"] = sync.sha256(
        script_payload
    )
    config["implementation_binding"]["implementation_test_sha256"] = sync.sha256(
        test_payload
    )
    i_config = sync.expected_unknown_i_config(config)
    i_config_payload = sync.json_bytes(i_config)
    b_config_payload = sync.json_bytes(config)
    for path, payload in (
        (sync.CONFIG_REPO_PATH, b_config_payload),
        (sync.SCRIPT_REPO_PATH, script_payload),
        (sync.TEST_REPO_PATH, test_payload),
    ):
        destination = repo / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
    responses: dict[tuple[str, ...], bytes] = {
        ("rev-parse", f"{base_i['commit']}^"): (base_i["expected_parent"] + "\n").encode(),
        ("rev-parse", f"{base_b['commit']}^"): (base_i["commit"] + "\n").encode(),
        ("rev-parse", f"{ledger_commit}^"): (base_b["commit"] + "\n").encode(),
        ("rev-parse", f"{implementation}^"): (ledger_commit + "\n").encode(),
        ("rev-parse", f"{head}^"): (implementation + "\n").encode(),
        ("rev-parse", "HEAD"): (head + "\n").encode(),
        ("status", "--porcelain"): b"",
        ("symbolic-ref", "--short", "HEAD"): (authority["branch"] + "\n").encode(),
        ("rev-parse", "@{u}"): (head + "\n").encode(),
        ("ls-remote", "--heads", "origin", authority["branch"]): (
            f"{head}\trefs/heads/{authority['branch']}\n".encode()
        ),
    }
    paths_by_commit = {
        base_i["commit"]: base_i["exact_changed_paths"],
        base_b["commit"]: base_b["exact_changed_paths"],
        ledger_commit: list(sync.LEDGER_PATHS),
        implementation: authority["implementation_exact_changed_paths"],
        head: authority["binding_exact_changed_paths"],
    }
    for commit, paths in paths_by_commit.items():
        responses[
            ("diff-tree", "--no-commit-id", "--name-only", "-r", commit)
        ] = ("\n".join(paths) + "\n").encode()
    for path, payload in base_i_payloads.items():
        responses[("show", f"{base_i['commit']}:{path}")] = payload
    for path, payload in base_b_payloads.items():
        responses[("show", f"{base_b['commit']}:{path}")] = payload
    for path, payload in ledger_payloads.items():
        responses[("show", f"{ledger_commit}:{path}")] = payload
    for commit in (implementation, head):
        responses[("show", f"{commit}:{sync.SCRIPT_REPO_PATH}")] = script_payload
        responses[("show", f"{commit}:{sync.TEST_REPO_PATH}")] = test_payload
    responses[("show", f"{implementation}:{sync.CONFIG_REPO_PATH}")] = i_config_payload
    responses[("show", f"{head}:{sync.CONFIG_REPO_PATH}")] = b_config_payload
    return config, responses, repo


def test_repository_audit_accepts_exact_i_or_b_config_and_rejects_stale_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, responses, repo = _repository_fixture(tmp_path)

    def fake_git(_repo: Path, *args: str) -> bytes:
        try:
            return responses[tuple(args)]
        except KeyError as exc:
            raise AssertionError(f"unexpected Git call: {args}") from exc

    monkeypatch.setattr(sync, "_run_git", fake_git)
    result = sync.audit_production_repository_authority(config)
    assert result["base_b2"] == "679a1c2ae89db7d6a9894f9299de7ce38b30ecdb"
    assert result["ledger"] == "a" * 40
    assert result["i1"] == "b" * 40
    assert result["b1"] == "e" * 40
    assert sync.expected_unknown_i_config(config)["implementation_binding"]["status"] == (
        sync.UNKNOWN
    )
    (repo / sync.SCRIPT_REPO_PATH).write_bytes(b"stale copy")
    with pytest.raises(sync.AuthorityError, match="executing script/test"):
        sync.audit_production_repository_authority(config)


def _publication_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, Any], dict[str, bytes], dict[str, bytes], Path, Path]:
    config = bound_config()
    predecessor = predecessor_fixture(config)
    run_root = tmp_path / "run"
    prepared_root = tmp_path / "prepared-root"
    prepared = prepared_root / "evt060"
    run_root.mkdir()
    prepared_root.mkdir()
    config["runtime"]["run_root"] = str(run_root)
    config["runtime"]["allowed_prepared_root"] = str(prepared_root)
    for name, payload in predecessor.items():
        (run_root / name).write_bytes(payload)
    successors = sync.build_successors(
        config, predecessor, "2026-08-15T01:01:00+08:00"
    )
    sync._write_prepared(prepared, successors)
    monkeypatch.setattr(sync, "_production_preflight", lambda _config: None)
    return config, predecessor, successors, run_root, prepared


def test_atomic_publication_recovers_prefix_and_is_exactly_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _predecessor, successors, run_root, prepared = _publication_fixture(
        tmp_path, monkeypatch
    )

    def fail_after_status(stage: str) -> None:
        if stage == "mutable:STATUS.json":
            raise RuntimeError("injected stop")

    with pytest.raises(RuntimeError, match="injected stop"):
        sync.publish_prepared(
            config, prepared_directory=prepared, fault_injector=fail_after_status
        )
    assert (run_root / "STATUS.json").read_bytes() == successors["STATUS.json"]
    assert (run_root / "RUN_MANIFEST.json").read_bytes() != successors["RUN_MANIFEST.json"]
    sync.publish_prepared(config, prepared_directory=prepared)
    sync.publish_prepared(config, prepared_directory=prepared)
    sync.validate_published(config, prepared_directory=prepared)
    for name in sync.MUTABLE_NAMES:
        assert (run_root / name).read_bytes() == successors[name]
    for name in [*sync._snapshot_names(config).values(), config["runtime"]["sync_name"]]:
        assert (run_root / name).read_bytes() == successors[name]


def test_different_immutable_and_nonprefix_runtime_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, predecessor, successors, run_root, prepared = _publication_fixture(
        tmp_path, monkeypatch
    )
    first_snapshot = next(iter(sync._snapshot_names(config).values()))
    (run_root / first_snapshot).write_bytes(b"different")
    with pytest.raises(sync.PublicationError, match="different bytes"):
        sync.publish_prepared(config, prepared_directory=prepared)
    (run_root / first_snapshot).unlink()
    (run_root / "RUN_MANIFEST.json").write_bytes(successors["RUN_MANIFEST.json"])
    assert (run_root / "STATUS.json").read_bytes() == predecessor["STATUS.json"]
    with pytest.raises(sync.PredecessorError, match="publication prefix"):
        sync.publish_prepared(config, prepared_directory=prepared)


def test_atomic_failure_cleans_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "STATUS.json"

    def fail_replace(_source: Path, _destination: Path) -> None:
        raise OSError("injected replace failure")

    monkeypatch.setattr(sync.os, "replace", fail_replace)
    with pytest.raises(sync.PublicationError, match="atomic write failed"):
        sync._write_atomic(destination, b"new status")
    assert not destination.exists()
    assert list(tmp_path.iterdir()) == []
