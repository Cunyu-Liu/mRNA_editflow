from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
from typing import Any

import pytest


STAGING_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = STAGING_ROOT / "configs/route_a_v3_gse200304_dec020_v4_qualification_runtime_sync_v1.json"
SCRIPT_PATH = STAGING_ROOT / "scripts/route_a_v3/gse200304_dec020_v4_qualification_runtime_sync.py"
SPEC = importlib.util.spec_from_file_location("evt051_runtime_sync", SCRIPT_PATH)
assert SPEC and SPEC.loader
runtime_sync = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runtime_sync)


def disk_config() -> dict[str, Any]:
    return runtime_sync.load_config(CONFIG_PATH, require_bound=False)


def bound_config() -> dict[str, Any]:
    config = copy.deepcopy(disk_config())
    binding = config["implementation_binding"]
    binding.update(
        {
            "status": "BOUND",
            "implementation_commit": "b" * 40,
            "implementation_script_sha256": "c" * 64,
            "implementation_test_sha256": "d" * 64,
        }
    )
    binding["compiled_core_sha256"] = runtime_sync.compiled_core_sha256(config)
    return config


def unknown_l_and_i_config() -> dict[str, Any]:
    config = copy.deepcopy(disk_config())
    ledger = config["repository_authority"]["predecessor_ledger"]
    ledger.update(
        {
            "status": runtime_sync.UNKNOWN,
            "commit": runtime_sync.UNKNOWN,
            "integration_id": runtime_sync.UNKNOWN,
            "manifest_status": runtime_sync.UNKNOWN,
        }
    )
    ledger["registered_lineage_ids"] = [runtime_sync.UNKNOWN]
    for item in ledger["frozen_blobs"]:
        item["sha256"] = runtime_sync.UNKNOWN
    binding = config["implementation_binding"]
    for key in (
        "status",
        "implementation_commit",
        "implementation_script_sha256",
        "implementation_test_sha256",
    ):
        binding[key] = runtime_sync.UNKNOWN
    binding["compiled_core_sha256"] = runtime_sync.compiled_core_sha256(config)
    return config


def predecessor_payloads(config: dict[str, Any]) -> dict[str, bytes]:
    scientific = copy.deepcopy(runtime_sync.PREDECESSOR_SCIENTIFIC_STATE)
    status = {
        **scientific,
        "claim_status": "NOT_ESTABLISHED",
        "canonical_record_count": 0,
        "updated_at": "2026-08-12T22:43:38+08:00",
    }
    manifest = {
        **scientific,
        "run_status": "IN_PROGRESS",
        "claim_status": "NOT_ESTABLISHED",
        "canonical_record_count": 0,
        "registered_artifact_count": 0,
        "outputs": [
            {
                "absolute_path": f"/existing/{index:03d}",
                "artifact_type": f"EXISTING_{index:03d}",
                "bytes": index,
                "sha256": f"{index:064x}",
            }
            for index in range(212)
        ],
    }
    events = [
        {
            "event_id": f"A1-EVT-{index:03d}",
            "at": "2026-08-12T00:00:00+08:00",
            "event": "HISTORICAL",
        }
        for index in range(1, 50)
    ]
    events.append(
        {
            "event_id": "A1-EVT-050",
            "at": "2026-08-12T22:43:38+08:00",
            "event": "GSE200304_DEC020_PREDECESSOR_SETTLED",
            "decision_id": "V3-DEC-020",
        }
    )
    payloads = {
        "STATUS.json": runtime_sync.json_bytes(status),
        "RUN_MANIFEST.json": runtime_sync.json_bytes(manifest),
        "EVENT_LOG.jsonl": b"".join(runtime_sync.compact_json_line(event) for event in events),
    }
    for name, payload in payloads.items():
        config["runtime"]["predecessor_mutables"][name].update(
            {"bytes": len(payload), "sha256": runtime_sync.sha256(payload)}
        )
    tail = runtime_sync.compact_json_line(events[-1])
    config["runtime"]["predecessor_tail"].update(
        {"bytes": len(tail), "sha256": runtime_sync.sha256(tail)}
    )
    return payloads


def make_context(tmp_path: Path) -> tuple[dict[str, Any], dict[str, bytes], Path, Path]:
    config = bound_config()
    run_root = tmp_path / "run"
    allowed_root = tmp_path / "prepared-root"
    prepared = allowed_root / "evt051-job"
    run_root.mkdir()
    allowed_root.mkdir()
    config["runtime"]["run_root"] = str(run_root)
    config["runtime"]["allowed_prepared_root"] = str(allowed_root)
    predecessor = predecessor_payloads(config)
    config["implementation_binding"]["compiled_core_sha256"] = runtime_sync.compiled_core_sha256(config)
    for name, payload in predecessor.items():
        (run_root / name).write_bytes(payload)
    return config, predecessor, run_root, prepared


def read_runtime(run_root: Path) -> dict[str, bytes]:
    return {name: (run_root / name).read_bytes() for name in runtime_sync.MUTABLE_NAMES}


def test_disk_static_config_and_truth_axes() -> None:
    config = disk_config()
    runtime_sync.validate_static_config(config)
    ledger = config["repository_authority"]["predecessor_ledger"]
    assert ledger["status"] == "BOUND"
    runtime_sync._validate_ledger_binding(ledger)
    binding = config["implementation_binding"]
    values = [binding[key] for key in (
        "status", "implementation_commit", "implementation_script_sha256",
        "implementation_test_sha256",
    )]
    assert all(value == runtime_sync.UNKNOWN for value in values) or all(
        value != runtime_sync.UNKNOWN for value in values
    )
    normalized = runtime_sync.expected_unknown_i2_config(config)
    assert runtime_sync._binding_values_are_unknown(normalized["implementation_binding"])
    assert runtime_sync.compiled_core_projection(normalized) == runtime_sync.compiled_core_projection(config)
    assert config["runtime"]["predecessor_event_count"] == 50
    assert config["runtime"]["successor_event_count"] == 51
    assert (
        config["runtime"]["predecessor_manifest_output_count"],
        config["runtime"]["successor_manifest_output_count"],
        config["runtime"]["output_delta_count"],
    ) == (212, 220, 8)
    successor = config["successor_scientific_state"]
    assert successor["input_status_counts"] == {
        "PASS": 7,
        "BLOCKED": 0,
        "UNKNOWN_NOT_ASSERTED": 0,
        "NOT_RUN": 0,
    }
    assert successor["unresolved_blockers"] == []
    assert config["registered_evidence_truth"]["historical_dec019"]["input_status_counts"]["UNKNOWN_NOT_ASSERTED"] == 1
    assert config["registered_evidence_truth"]["global_phase"]["unresolved_blockers"] == [
        "INSUFFICIENT_QUALIFIED_ORDINARY_STUDIES",
        "INSUFFICIENT_QUALIFIED_A1_STUDIES",
        "NO_QUALIFIED_TRUE_A2_DENSE_STUDY",
    ]
    assert len(config["registered_artifacts"]) == 4
    assert config["access_boundary"]["registered_artifact_body_parse_count"] == 0
    assert config["access_boundary"]["private_payload_read_count"] == 0
    assert all(not item["absolute_path"].endswith(".jsonl") for item in config["registered_artifacts"])


def test_partial_unknown_ledger_group_rejected() -> None:
    config = unknown_l_and_i_config()
    config["repository_authority"]["predecessor_ledger"]["status"] = "BOUND"
    config["implementation_binding"]["compiled_core_sha256"] = runtime_sync.compiled_core_sha256(config)
    with pytest.raises(runtime_sync.BindingError, match="partially known"):
        runtime_sync.validate_static_config(config)


def test_bound_l_unknown_i_stops_before_external_io(tmp_path: Path) -> None:
    config = bound_config()
    binding = config["implementation_binding"]
    for key in ("status", "implementation_commit", "implementation_script_sha256", "implementation_test_sha256"):
        binding[key] = runtime_sync.UNKNOWN
    binding["compiled_core_sha256"] = runtime_sync.compiled_core_sha256(config)
    with pytest.raises(runtime_sync.BindingError, match="implementation is not BOUND"):
        runtime_sync.prepare_runtime_sync(
            prepared_directory=tmp_path / "prepared",
            recorded_at="2026-08-12T23:00:00+08:00",
            production=False,
            config_override=config,
            run_root_override=tmp_path / "run",
        )


def test_exact_byte_validation_reads_four_members_without_body_parse(monkeypatch: pytest.MonkeyPatch) -> None:
    config = bound_config()
    payloads = {item["absolute_path"]: f"opaque bundle member {index}\n".encode() for index, item in enumerate(config["registered_artifacts"])}
    config["registered_artifacts"] = copy.deepcopy(config["registered_artifacts"])
    for item in config["registered_artifacts"]:
        payload = payloads[item["absolute_path"]]
        item["bytes"] = len(payload)
        item["sha256"] = runtime_sync.sha256(payload)
    config["implementation_binding"]["compiled_core_sha256"] = runtime_sync.compiled_core_sha256(config)
    original_read = Path.read_bytes
    def read_member(path: Path) -> bytes:
        if str(path) in payloads:
            return payloads[str(path)]
        return original_read(path)
    monkeypatch.setattr(Path, "read_bytes", read_member)
    monkeypatch.setattr(runtime_sync, "validate_bound_config", lambda _config: None)
    monkeypatch.setattr(runtime_sync, "load_json", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("body parse")))
    result = runtime_sync.validate_registered_artifacts(config, verify_exact_bytes=True)
    assert result["status"] == "EXACT4_BYTES_AND_SHA256_VALIDATED"
    assert result["artifact_count"] == 4
    assert result["exact_byte_validation_count"] == 4
    assert result["body_parse_count"] == 0


def test_normal_transaction_is_212_to_220_with_exact4_and_sync(tmp_path: Path) -> None:
    config, predecessor, run_root, prepared = make_context(tmp_path)
    result = runtime_sync.prepare_runtime_sync(
        prepared_directory=prepared,
        recorded_at="2026-08-12T23:00:00+08:00",
        production=False,
        config_override=config,
        run_root_override=run_root,
    )
    assert result["manifest_output_transition"] == "212_TO_220"
    assert result["new_runtime_output_count"] == 8
    runtime_sync.publish_prepared(
        prepared_directory=prepared,
        production=False,
        config_override=config,
        run_root_override=run_root,
    )
    status, manifest, events = runtime_sync._parse_runtime(read_runtime(run_root))
    old_status, old_manifest, old_events = runtime_sync._parse_runtime(predecessor)
    assert len(events) == 51 and events[:-1] == old_events
    event = events[-1]
    assert event["registered_artifact_count"] == 4
    assert event["registered_lineage_ids"] == runtime_sync.LEDGER_LINEAGE_IDS
    assert event["scientific_state_changed"] is True
    assert event["dataset_qualification_changed"] is True
    assert event["overall_A1_phase_completion_gate_changed"] is False
    assert event["private_payload_read_count"] == 0
    assert len(manifest["outputs"]) == 220
    assert manifest["outputs"][:212] == old_manifest["outputs"]
    assert manifest["outputs"][212:216] == config["registered_artifacts"]
    assert manifest["registered_artifact_count"] == 4
    assert status["input_status_counts"]["UNKNOWN_NOT_ASSERTED"] == 0
    assert status["unresolved_blockers"] == []
    sync = runtime_sync.load_json((run_root / config["runtime"]["sync_name"]).read_bytes(), label="sync")
    assert sync["registered_artifact_count"] == 4
    assert sync["output_delta_count"] == 8
    assert sync["dataset_qualification_changed"] is True


def test_predecessor_drift_stops_before_prepared_write(tmp_path: Path) -> None:
    config, _predecessor, run_root, prepared = make_context(tmp_path)
    status = runtime_sync.load_json((run_root / "STATUS.json").read_bytes(), label="status")
    status["drift"] = True
    (run_root / "STATUS.json").write_bytes(runtime_sync.json_bytes(status))
    with pytest.raises(runtime_sync.PredecessorError, match="identity drift"):
        runtime_sync.prepare_runtime_sync(
            prepared_directory=prepared,
            recorded_at="2026-08-12T23:00:00+08:00",
            production=False,
            config_override=config,
            run_root_override=run_root,
        )
    assert not prepared.exists()


def test_immutable_first_prefix_recovery(tmp_path: Path) -> None:
    config, predecessor, run_root, prepared = make_context(tmp_path)
    runtime_sync.prepare_runtime_sync(
        prepared_directory=prepared,
        recorded_at="2026-08-12T23:00:00+08:00",
        production=False,
        config_override=config,
        run_root_override=run_root,
    )

    def interrupt(point: str) -> None:
        if point == "before_replace:RUN_MANIFEST.json":
            raise OSError("supported prefix interruption")

    with pytest.raises(runtime_sync.PublicationError, match="not committed"):
        runtime_sync.publish_prepared(
            prepared_directory=prepared,
            production=False,
            config_override=config,
            run_root_override=run_root,
            fault_injector=interrupt,
        )
    partial = read_runtime(run_root)
    assert partial["STATUS.json"] != predecessor["STATUS.json"]
    assert partial["RUN_MANIFEST.json"] == predecessor["RUN_MANIFEST.json"]
    recovered = runtime_sync.publish_prepared(
        prepared_directory=prepared,
        production=False,
        config_override=config,
        run_root_override=run_root,
    )
    assert recovered["status"] == "PUBLISHED_VERIFIED"
    assert len(runtime_sync.load_events(read_runtime(run_root)["EVENT_LOG.jsonl"], label="events")) == 51


@pytest.mark.parametrize("drift", [None, "parent", "path"])
def test_production_authority_exact_l_i1_b1_i2_b2(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, drift: str | None
) -> None:
    config = bound_config()
    binding = config["implementation_binding"]
    authority = config["repository_authority"]
    ledger = authority["predecessor_ledger"]
    i2 = binding["implementation_commit"]
    head = "e" * 40
    branch = authority["branch"]
    config_payload = runtime_sync.json_bytes(config)
    unknown_i2_payload = runtime_sync.json_bytes(runtime_sync.expected_unknown_i2_config(config))

    parents = {
        head: i2,
        i2: runtime_sync.FROZEN_B1_COMMIT,
        runtime_sync.FROZEN_B1_COMMIT: runtime_sync.FROZEN_I1_COMMIT,
        runtime_sync.FROZEN_I1_COMMIT: ledger["commit"],
    }
    if drift == "parent":
        parents[i2] = "f" * 40
    changed = {
        head: authority["binding_exact_changed_paths"],
        i2: authority["implementation_exact_changed_paths"],
        runtime_sync.FROZEN_B1_COMMIT: authority["binding_exact_changed_paths"],
        runtime_sync.FROZEN_I1_COMMIT: authority["implementation_exact_changed_paths"],
        ledger["commit"]: ledger["exact_changed_paths"],
    }
    if drift == "path":
        changed[i2] = [*changed[i2], "unexpected"]

    digest_by_payload: dict[bytes, str] = {}
    blobs: dict[tuple[str, str], bytes] = {}
    for label, commit, digests in (
        ("i1", runtime_sync.FROZEN_I1_COMMIT, runtime_sync.FROZEN_I1_BLOB_SHA256),
        ("b1", runtime_sync.FROZEN_B1_COMMIT, runtime_sync.FROZEN_B1_BLOB_SHA256),
    ):
        for path, digest in digests.items():
            payload = f"{label}:{path}\n".encode()
            blobs[(commit, path)] = payload
            digest_by_payload[payload] = digest

    dynamic_script = b"dynamic I2 script\n"
    dynamic_test = b"dynamic I2 test\n"
    digest_by_payload[dynamic_script] = binding["implementation_script_sha256"]
    digest_by_payload[dynamic_test] = binding["implementation_test_sha256"]
    for commit in (i2, head):
        blobs[(commit, runtime_sync.SCRIPT_REPO_PATH)] = dynamic_script
        blobs[(commit, runtime_sync.TEST_REPO_PATH)] = dynamic_test
    blobs[(i2, runtime_sync.CONFIG_REPO_PATH)] = unknown_i2_payload
    blobs[(head, runtime_sync.CONFIG_REPO_PATH)] = config_payload

    worktree: dict[str, bytes] = {
        runtime_sync.CONFIG_REPO_PATH: config_payload,
        runtime_sync.SCRIPT_REPO_PATH: dynamic_script,
        runtime_sync.TEST_REPO_PATH: dynamic_test,
    }
    for item in ledger["frozen_blobs"]:
        payload = f"ledger:{item['path']}\n".encode()
        digest_by_payload[payload] = item["sha256"]
        worktree[item["path"]] = payload
        for commit in (ledger["commit"], runtime_sync.FROZEN_I1_COMMIT,
                       runtime_sync.FROZEN_B1_COMMIT, i2, head):
            blobs[(commit, item["path"])] = payload

    def fake_git(_repo: Path, *args: str) -> bytes:
        if args in (("rev-parse", "HEAD"), ("rev-parse", "@{upstream}"),
                    ("rev-parse", "--verify", f"refs/remotes/origin/{branch}")):
            return f"{head}\n".encode()
        if args == ("rev-parse", "--abbrev-ref", "HEAD"):
            return f"{branch}\n".encode()
        if args == ("rev-parse", "--abbrev-ref", "@{upstream}"):
            return f"origin/{branch}\n".encode()
        if args == ("status", "--porcelain=v1", "--untracked-files=all"):
            return b""
        if len(args) == 2 and args[0] == "rev-parse" and args[1].endswith("^"):
            return f"{parents[args[1][:-1]]}\n".encode()
        raise AssertionError(args)

    real_sha256 = runtime_sync.sha256
    monkeypatch.setattr(runtime_sync, "sha256", lambda payload: digest_by_payload.get(payload, real_sha256(payload)))
    monkeypatch.setattr(runtime_sync, "_run_git", fake_git)
    monkeypatch.setattr(runtime_sync, "_changed_paths", lambda _repo, commit: sorted(changed[commit]))
    monkeypatch.setattr(runtime_sync, "_git_blob", lambda _repo, commit, path: blobs[(commit, path)])
    monkeypatch.setattr(runtime_sync, "_read_repo_file", lambda _repo, path: worktree[path])

    if drift is None:
        result = runtime_sync.audit_production_repository_authority(config, config_payload)
        assert result["frozen_i1_commit"] == runtime_sync.FROZEN_I1_COMMIT
        assert result["frozen_b1_commit"] == runtime_sync.FROZEN_B1_COMMIT
        assert result["implementation_i2_commit"] == i2
        assert result["binding_b2_commit"] == head
    else:
        with pytest.raises(runtime_sync.RuntimeSyncError):
            runtime_sync.audit_production_repository_authority(config, config_payload)
