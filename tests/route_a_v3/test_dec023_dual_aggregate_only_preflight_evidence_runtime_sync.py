from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from typing import Callable

import pytest


STAGING_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = (
    STAGING_ROOT
    / "scripts/route_a_v3/dec023_dual_aggregate_only_preflight_evidence_runtime_sync.py"
)
CONFIG_PATH = (
    STAGING_ROOT
    / "configs/route_a_v3_dec023_dual_aggregate_only_preflight_evidence_runtime_sync_v1.json"
)
SPEC = importlib.util.spec_from_file_location("dec023_dual_evt057_runtime_sync", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
sync = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sync)

RECORDED_AT = "2026-08-14T11:00:00+08:00"


def candidate_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def clean_grouped_unknown_config(config: dict) -> dict:
    clean = copy.deepcopy(config)
    ledger = clean["repository_authority"]["predecessor_ledger"]
    ledger.update(
        {
            "status": sync.UNKNOWN,
            "commit": sync.UNKNOWN,
            "integration_id": sync.UNKNOWN,
            "manifest_status": sync.UNKNOWN,
            "registered_lineage_ids": [sync.UNKNOWN, sync.UNKNOWN],
        }
    )
    for item in ledger["frozen_blobs"]:
        item["sha256"] = sync.UNKNOWN
    binding = clean["implementation_binding"]
    binding.update(
        {
            "status": sync.UNKNOWN,
            "implementation_commit": sync.UNKNOWN,
            "implementation_script_sha256": sync.UNKNOWN,
            "implementation_test_sha256": sync.UNKNOWN,
        }
    )
    sync.validate_static_config(clean)
    return clean


def bind_ledger(config: dict) -> dict:
    bound = copy.deepcopy(config)
    ledger = bound["repository_authority"]["predecessor_ledger"]
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
    return bound


def clean_i_config(config: dict) -> dict:
    clean_i = sync.expected_unknown_i_config(bind_ledger(config))
    sync.validate_static_config(clean_i)
    assert not sync._ledger_is_unknown(
        clean_i["repository_authority"]["predecessor_ledger"]
    )
    assert sync._binding_is_unknown(clean_i["implementation_binding"])
    return clean_i


def bind_config(config: dict) -> dict:
    bound = clean_i_config(config)
    binding = bound["implementation_binding"]
    binding.update(
        {
            "status": "BOUND",
            "implementation_commit": "b" * 40,
            "implementation_script_sha256": "c" * 64,
            "implementation_test_sha256": "d" * 64,
        }
    )
    sync.validate_bound_config(bound)
    return bound


def predecessor_fixture(config: dict) -> dict[str, bytes]:
    fields = sync._runtime_truth_fields(config)
    authority = sync._predecessor_authority_fields()
    status = {
        "schema_version": "test.status.v1",
        "updated_at": "2026-08-14T10:30:00+08:00",
        "opaque_status_field": "PRESERVE_EXACTLY",
        **copy.deepcopy(fields),
        **copy.deepcopy(authority),
    }
    manifest = {
        "schema_version": "test.manifest.v1",
        "opaque_manifest_field": {"preserve": True},
        "outputs": [
            {
                "absolute_path": f"/fixture/predecessor/{index:03d}.json",
                "artifact_type": "FIXTURE",
            }
            for index in range(242)
        ],
        "registered_artifact_count": 6,
        **copy.deepcopy(fields),
        **copy.deepcopy(authority),
    }
    events = [
        {
            "event_id": f"A1-EVT-{index:03d}",
            "at": "2026-08-14T10:00:00+08:00",
            "decision_id": "V3-DEC-022",
        }
        for index in range(1, 56)
    ]
    events.append(
        {
            "event_id": "A1-EVT-056",
            "predecessor_event_id": "A1-EVT-055",
            "at": "2026-08-14T10:30:00+08:00",
            "decision_id": "V3-DEC-023",
        }
    )
    payloads = {
        "STATUS.json": sync.json_bytes(status),
        "RUN_MANIFEST.json": sync.json_bytes(manifest),
        "EVENT_LOG.jsonl": b"".join(sync.compact_json_line(item) for item in events),
    }
    refresh_predecessor_identities(config, payloads)
    sync.validate_bound_config(config)
    return payloads


def refresh_predecessor_identities(
    config: dict, payloads: dict[str, bytes]
) -> None:
    for name, payload in payloads.items():
        spec = config["runtime"]["predecessor_mutables"][name]
        spec["bytes"] = len(payload)
        spec["sha256"] = sync.sha256(payload)
    tail = payloads["EVENT_LOG.jsonl"].splitlines(keepends=True)[-1]
    config["runtime"]["predecessor_tail"]["bytes"] = len(tail)
    config["runtime"]["predecessor_tail"]["sha256"] = sync.sha256(tail)


def write_runtime(run_root: Path, payloads: dict[str, bytes]) -> None:
    run_root.mkdir(parents=True)
    for name in sync.MUTABLE_NAMES:
        (run_root / name).write_bytes(payloads[name])


def test_disk_candidate_is_staging_unknown_or_exact_real_i_or_b() -> None:
    config = candidate_config()
    sync.validate_static_config(config)
    ledger_unknown = sync._ledger_is_unknown(
        config["repository_authority"]["predecessor_ledger"]
    )
    binding_unknown = sync._binding_is_unknown(config["implementation_binding"])

    if ledger_unknown:
        assert binding_unknown
        state = "STAGING_GROUPED_UNKNOWN"
    elif binding_unknown:
        state = "REAL_DISK_I"
    else:
        sync.validate_bound_config(config)
        binding = config["implementation_binding"]
        assert binding["implementation_script_sha256"] == sync.sha256(
            SCRIPT_PATH.read_bytes()
        )
        assert binding["implementation_test_sha256"] == sync.sha256(
            Path(__file__).read_bytes()
        )
        state = "REAL_DISK_B"
    assert state in {"STAGING_GROUPED_UNKNOWN", "REAL_DISK_I", "REAL_DISK_B"}
    if binding_unknown:
        with pytest.raises(sync.BindingError):
            sync.validate_bound_config(config)


def test_partial_ledger_or_implementation_groups_are_rejected() -> None:
    candidate = candidate_config()
    partial_l = clean_grouped_unknown_config(candidate)
    partial_l["repository_authority"]["predecessor_ledger"]["status"] = "BOUND"
    with pytest.raises(sync.BindingError, match="ledger is partially known"):
        sync.validate_static_config(partial_l)

    partial_i = clean_i_config(candidate)
    partial_i["implementation_binding"]["status"] = "BOUND"
    with pytest.raises(sync.BindingError, match="implementation binding is partially known"):
        sync.validate_static_config(partial_i)

    i_before_l = clean_grouped_unknown_config(candidate)
    i_before_l["implementation_binding"].update(
        {
            "status": "BOUND",
            "implementation_commit": "b" * 40,
            "implementation_script_sha256": "c" * 64,
            "implementation_test_sha256": "d" * 64,
        }
    )
    with pytest.raises(sync.BindingError, match="before the ledger"):
        sync.validate_static_config(i_before_l)


def test_bound_b_normalizes_to_exact_clean_i_only() -> None:
    bound = bind_config(candidate_config())
    normalized = sync.expected_unknown_i_config(bound)
    sync.validate_static_config(normalized)
    assert sync._binding_is_unknown(normalized["implementation_binding"])
    assert not sync._ledger_is_unknown(
        normalized["repository_authority"]["predecessor_ledger"]
    )
    changed = []
    for key in bound["implementation_binding"]:
        if bound["implementation_binding"][key] != normalized["implementation_binding"][key]:
            changed.append(f"implementation_binding.{key}")
    assert changed == [
        "implementation_binding.status",
        "implementation_binding.implementation_commit",
        "implementation_binding.implementation_script_sha256",
        "implementation_binding.implementation_test_sha256",
    ]
    assert sync.runtime_science_report_truth_projection(normalized) == (
        sync.runtime_science_report_truth_projection(bound)
    )


@pytest.mark.parametrize("entrypoint", ["prepare", "publish", "validate"])
def test_unknown_i_stops_before_git_report_prepared_or_runtime_io(
    entrypoint: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    clean_i = clean_i_config(candidate_config())
    touched: list[str] = []

    def forbidden(*args: object, **kwargs: object) -> object:
        touched.append("io")
        raise AssertionError("external I/O occurred before grouped binding rejection")

    for name in (
        "_run_git",
        "validate_registered_artifacts",
        "_read_prepared",
        "_read_runtime",
        "_write_prepared",
    ):
        monkeypatch.setattr(sync, name, forbidden)

    kwargs = {
        "prepared_directory": tmp_path / "never-created",
        "production": False,
        "config_override": clean_i,
        "run_root_override": tmp_path / "never-opened",
    }
    with pytest.raises(sync.BindingError, match="implementation binding"):
        if entrypoint == "prepare":
            sync.prepare_runtime_sync(recorded_at=RECORDED_AT, **kwargs)
        elif entrypoint == "publish":
            sync.publish_prepared(**kwargs)
        else:
            sync.validate_published(**kwargs)
    assert touched == []
    assert not (tmp_path / "never-created").exists()
    assert not (tmp_path / "never-opened").exists()


def test_exact2_reports_are_hashed_without_body_parse_and_drift_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = bind_config(candidate_config())
    artifacts = config["registered_artifacts"]
    payload_by_path = {
        Path(item["absolute_path"]): b"not-json\x00" + bytes([index]) * (item["bytes"] - 9)
        for index, item in enumerate(artifacts, start=1)
    }
    digest_by_payload = {
        payload_by_path[Path(item["absolute_path"])]: item["sha256"]
        for item in artifacts
    }
    real_read_bytes = Path.read_bytes
    real_sha256 = sync.sha256

    def fake_read_bytes(path: Path) -> bytes:
        if path in payload_by_path:
            return payload_by_path[path]
        return real_read_bytes(path)

    def fake_sha256(payload: bytes) -> str:
        return digest_by_payload.get(payload, real_sha256(payload))

    monkeypatch.setattr(Path, "read_bytes", fake_read_bytes)
    monkeypatch.setattr(sync, "sha256", fake_sha256)
    result = sync.validate_registered_artifacts(config, verify_exact_bytes=True)
    assert result["status"] == "EXACT2_BYTES_AND_SHA256_VALIDATED"
    assert result["artifact_count"] == 2
    assert result["exact_byte_validation_count"] == 2
    assert result["body_parse_count"] == 0
    assert result["payload_field_read_count"] == 0
    assert result["public_asset_read_count"] == 0

    first_path = Path(artifacts[0]["absolute_path"])
    payload_by_path[first_path] += b"x"
    with pytest.raises(sync.PublicationError, match="identity drift"):
        sync.validate_registered_artifacts(config, verify_exact_bytes=True)


def test_successor_is_exact_56_to_57_242_to_248_and_preserves_science() -> None:
    config = bind_config(candidate_config())
    predecessor = predecessor_fixture(config)
    old_status, old_manifest, old_events = sync._parse_runtime(predecessor)
    successors = sync.build_successors(config, predecessor, RECORDED_AT)
    new_status, new_manifest, events = sync._parse_runtime(
        {name: successors[name] for name in sync.MUTABLE_NAMES}
    )

    assert len(old_events) == 56
    assert len(events) == 57
    assert events[:-1] == old_events
    event = events[-1]
    assert event["event_id"] == "A1-EVT-057"
    assert event["predecessor_event_id"] == "A1-EVT-056"
    assert event["decision_id"] == "V3-DEC-023"
    assert event["registered_lineage_ids"] == list(sync.LINEAGE_IDS)
    assert event["scientific_state_changed"] is False
    assert event["evidence_surface_changed"] is True
    assert event["evidence_gate_statuses_changed"] is True
    assert event["overall_qualification_gate_changed"] is False
    assert event["qualification_changed"] is False
    assert event["registered_artifact_body_parse_count"] == 0
    assert len(new_manifest["outputs"]) == 248
    assert new_manifest["outputs"][:242] == old_manifest["outputs"]
    assert new_manifest["outputs"][242:244] == config["registered_artifacts"]
    assert new_manifest["registered_artifact_count"] == 8
    expected_status = copy.deepcopy(old_status)
    expected_status["updated_at"] = RECORDED_AT
    assert new_status == expected_status

    frozen = event["frozen_scientific_state"]
    assert frozen["current_qualified_counts"] == {
        "ordinary": 1,
        "a1": 1,
        "true_a2": 0,
        "canonical_records": 6547,
    }
    assert frozen["gse261709_contribution"] == {
        "ordinary": 0,
        "a1": 0,
        "true_a2": 0,
        "canonical_records": 0,
    }
    assert frozen["gse207584_contribution"] == frozen["gse261709_contribution"]
    assert frozen["a1_complete"] is False
    for key in (
        "training_started",
        "training_allowed",
        "training_authorized",
        "gpu_work_started",
        "gpu_work_allowed",
        "model_selection_allowed",
        "a7_allowed",
        "next_phase_authorized",
    ):
        assert frozen[key] is False

    truth = event["registered_evidence_truth"]
    assert truth[
        "gse261709_public_identifier_asset_schema_aggregate_geometry_preflight"
    ]["gate_status_counts"] == {
        "PASS": 1,
        "BLOCKED": 2,
        "UNKNOWN_NOT_ASSERTED": 0,
    }
    assert truth[
        "gse207584_aggregate_dense_family_qualification_preflight"
    ]["gate_status_counts"] == {
        "PASS_PREFLIGHT_ONLY": 1,
        "FAIL_CLOSED": 2,
        "UNKNOWN_NOT_ASSERTED": 8,
    }


@pytest.mark.parametrize("drift", ["status_identity", "tail_parent", "manifest_count"])
def test_predecessor_identity_and_structure_drift_fail_closed(drift: str) -> None:
    config = bind_config(candidate_config())
    predecessor = predecessor_fixture(config)
    if drift == "status_identity":
        predecessor["STATUS.json"] += b" "
        match = "identity drift"
    elif drift == "tail_parent":
        events = sync.load_events(predecessor["EVENT_LOG.jsonl"], label="fixture")
        events[-1]["predecessor_event_id"] = "A1-EVT-054"
        predecessor["EVENT_LOG.jsonl"] = b"".join(
            sync.compact_json_line(item) for item in events
        )
        refresh_predecessor_identities(config, predecessor)
        match = "tail parent"
    else:
        manifest = sync.load_json(predecessor["RUN_MANIFEST.json"], label="fixture")
        manifest["outputs"].pop()
        predecessor["RUN_MANIFEST.json"] = sync.json_bytes(manifest)
        refresh_predecessor_identities(config, predecessor)
        match = "output count"
    with pytest.raises(sync.RuntimeSyncError, match=match):
        sync.build_successors(config, predecessor, RECORDED_AT)


def test_immutable_partial_temp_failure_leaves_no_final_or_temp(tmp_path: Path) -> None:
    target = tmp_path / "IMMUTABLE.json"
    payload = b"complete immutable payload" * 32

    def stop_after_partial(point: str) -> None:
        if point == "after_partial_temp_write":
            raise RuntimeError("injected partial temp write")

    with pytest.raises(RuntimeError, match="partial temp write"):
        sync._write_immutable_once(target, payload, fault_injector=stop_after_partial)
    assert not target.exists()
    assert list(tmp_path.iterdir()) == []
    assert sync._write_immutable_once(target, payload) == "CREATED"
    assert sync._write_immutable_once(target, payload) == "EXISTING_EXACT"
    with pytest.raises(sync.PublicationError, match="immutable output differs"):
        sync._write_immutable_once(target, b"different")


def _prepared_fixture(tmp_path: Path) -> tuple[dict, Path, Path, dict[str, bytes]]:
    config = bind_config(candidate_config())
    run_root = tmp_path / "runtime"
    allowed = tmp_path / "prepared-root"
    prepared = allowed / "evt057"
    config["runtime"]["run_root"] = str(run_root)
    config["runtime"]["allowed_prepared_root"] = str(allowed)
    predecessor = predecessor_fixture(config)
    write_runtime(run_root, predecessor)
    result = sync.prepare_runtime_sync(
        prepared_directory=prepared,
        recorded_at=RECORDED_AT,
        production=False,
        config_override=config,
        run_root_override=run_root,
    )
    assert result["status"] == "PREPARED_NOT_PUBLISHED"
    assert result["manifest_output_transition"] == "242_TO_248"
    assert result["manifest_registered_artifact_transition"] == "6_TO_8"
    assert len(list(prepared.iterdir())) == 7
    return config, run_root, prepared, predecessor


def test_publish_recovers_only_in_status_manifest_event_order(tmp_path: Path) -> None:
    config, run_root, prepared, predecessor = _prepared_fixture(tmp_path)
    triggered = False

    def stop_before_manifest(point: str) -> None:
        nonlocal triggered
        if point == "before_replace:RUN_MANIFEST.json" and not triggered:
            triggered = True
            raise RuntimeError("fixture interruption")

    with pytest.raises(sync.PublicationError, match="retry"):
        sync.publish_prepared(
            prepared_directory=prepared,
            production=False,
            config_override=config,
            run_root_override=run_root,
            fault_injector=stop_before_manifest,
        )
    assert (run_root / "STATUS.json").read_bytes() == (prepared / "STATUS.json").read_bytes()
    assert (run_root / "RUN_MANIFEST.json").read_bytes() == predecessor["RUN_MANIFEST.json"]
    assert (run_root / "EVENT_LOG.jsonl").read_bytes() == predecessor["EVENT_LOG.jsonl"]

    assert sync.publish_prepared(
        prepared_directory=prepared,
        production=False,
        config_override=config,
        run_root_override=run_root,
    )["status"] == "PUBLISHED_VERIFIED"
    assert sync.validate_published(
        prepared_directory=prepared,
        production=False,
        config_override=config,
        run_root_override=run_root,
    ) == {
        "status": "PUBLISHED_VERIFIED",
        "event_id": "A1-EVT-057",
        "registered_artifact_body_parse_count": 0,
    }


def test_publish_rejects_nonprefix_and_prepared_extras(tmp_path: Path) -> None:
    config, run_root, prepared, predecessor = _prepared_fixture(tmp_path)
    (run_root / "RUN_MANIFEST.json").write_bytes(
        (prepared / "RUN_MANIFEST.json").read_bytes()
    )
    with pytest.raises(sync.PredecessorError, match="prefix is not recoverable"):
        sync.publish_prepared(
            prepared_directory=prepared,
            production=False,
            config_override=config,
            run_root_override=run_root,
        )

    (run_root / "RUN_MANIFEST.json").write_bytes(predecessor["RUN_MANIFEST.json"])
    (prepared / "UNEXPECTED").write_text("unexpected", encoding="utf-8")
    with pytest.raises(sync.PublicationError, match="extras"):
        sync.publish_prepared(
            prepared_directory=prepared,
            production=False,
            config_override=config,
            run_root_override=run_root,
        )


def _repository_audit_fixture(
    monkeypatch: pytest.MonkeyPatch,
    drift: str | None,
) -> tuple[dict, bytes, str, str, str]:
    config = bind_config(candidate_config())
    monkeypatch.setattr(sync, "PRODUCTION_REPO_ROOT", STAGING_ROOT)
    config["repository_authority"]["production_repo_root"] = str(STAGING_ROOT)
    ledger = config["repository_authority"]["predecessor_ledger"]
    binding = config["implementation_binding"]
    l_commit = ledger["commit"]
    i_commit = "b" * 40
    head = "e" * 40
    binding["implementation_commit"] = i_commit
    ledger_payloads = {path: f"ledger:{path}".encode() for path in sync.LEDGER_PATHS}
    for item in ledger["frozen_blobs"]:
        item["sha256"] = sync.sha256(ledger_payloads[item["path"]])
    script_payload = SCRIPT_PATH.read_bytes()
    test_payload = Path(__file__).read_bytes()
    binding["implementation_script_sha256"] = sync.sha256(script_payload)
    binding["implementation_test_sha256"] = sync.sha256(test_payload)
    sync.validate_bound_config(config)
    i_config_payload = sync.json_bytes(sync.expected_unknown_i_config(config))
    config_payload = sync.json_bytes(config)

    def fake_run_git(repo_root: Path, *args: str) -> bytes:
        assert repo_root == sync.PRODUCTION_REPO_ROOT
        mapping = {
            ("rev-parse", "HEAD"): f"{head}\n".encode(),
            ("rev-parse", "--abbrev-ref", "HEAD"): f"{sync.BRANCH}\n".encode(),
            ("rev-parse", "--abbrev-ref", "@{upstream}"): f"origin/{sync.BRANCH}\n".encode(),
            ("rev-parse", "@{upstream}"): f"{head}\n".encode(),
            ("rev-parse", "--verify", f"refs/remotes/origin/{sync.BRANCH}"): f"{head}\n".encode(),
            ("status", "--porcelain=v1", "--untracked-files=all"): b"",
            ("rev-parse", f"{head}^"): f"{i_commit}\n".encode(),
            ("rev-parse", f"{i_commit}^"): (
                f"{'f' * 40}\n".encode() if drift == "parent" else f"{l_commit}\n".encode()
            ),
        }
        return mapping[args]

    def fake_changed_paths(repo_root: Path, commit: str) -> list[str]:
        assert repo_root == sync.PRODUCTION_REPO_ROOT
        if commit == l_commit:
            return sorted(sync.LEDGER_PATHS)
        if commit == i_commit:
            paths = sorted(
                config["repository_authority"]["implementation_exact_changed_paths"]
            )
            return paths + ["unexpected.txt"] if drift == "path" else paths
        if commit == head:
            return [sync.CONFIG_REPO_PATH]
        raise AssertionError(commit)

    def fake_git_blob(repo_root: Path, commit: str, path: str) -> bytes:
        assert repo_root == sync.PRODUCTION_REPO_ROOT
        if path in ledger_payloads and commit in {l_commit, i_commit, head}:
            if drift == "blob" and commit == i_commit and path == sync.LEDGER_PATHS[0]:
                return b"drift"
            return ledger_payloads[path]
        if path == sync.CONFIG_REPO_PATH and commit == i_commit:
            return i_config_payload
        if path == sync.CONFIG_REPO_PATH and commit == head:
            return config_payload
        if path == sync.SCRIPT_REPO_PATH and commit in {i_commit, head}:
            return script_payload
        if path == sync.TEST_REPO_PATH and commit in {i_commit, head}:
            return test_payload
        raise AssertionError((commit, path))

    def fake_repo_file(repo_root: Path, path: str) -> bytes:
        assert repo_root == sync.PRODUCTION_REPO_ROOT
        if path in ledger_payloads:
            return ledger_payloads[path]
        if path == sync.CONFIG_REPO_PATH:
            return config_payload
        if path == sync.SCRIPT_REPO_PATH:
            return script_payload
        if path == sync.TEST_REPO_PATH:
            return test_payload
        raise AssertionError(path)

    monkeypatch.setattr(sync, "_run_git", fake_run_git)
    monkeypatch.setattr(sync, "_changed_paths", fake_changed_paths)
    monkeypatch.setattr(sync, "_git_blob", fake_git_blob)
    monkeypatch.setattr(sync, "_read_repo_file", fake_repo_file)
    return config, config_payload, l_commit, i_commit, head


def test_repository_audit_proves_direct_l_i_b_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, payload, l_commit, i_commit, head = _repository_audit_fixture(
        monkeypatch, None
    )
    assert sync.audit_production_repository_authority(config, payload) == {
        "status": "PASS_DIRECT_EXACT4_L_TO_EXACT3_I_TO_CONFIG_ONLY_B",
        "ledger_l_commit": l_commit,
        "implementation_i_commit": i_commit,
        "binding_b_commit": head,
        "upstream_head_commit": head,
        "origin_branch_head_commit": head,
        "worktree_and_index_clean": True,
    }


def test_production_stale_script_copy_stops_before_report_prepared_or_runtime_io(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config, payload, _, _, _ = _repository_audit_fixture(monkeypatch, None)
    stale_root = tmp_path / "stale-copy"
    stale_script = stale_root / sync.SCRIPT_REPO_PATH
    stale_config = stale_root / sync.CONFIG_REPO_PATH
    stale_script.parent.mkdir(parents=True)
    stale_config.parent.mkdir(parents=True)
    stale_script.write_bytes(SCRIPT_PATH.read_bytes())
    stale_config.write_bytes(payload)

    stale_spec = importlib.util.spec_from_file_location(
        "dec023_dual_evt057_runtime_sync_stale_copy", stale_script
    )
    assert stale_spec is not None and stale_spec.loader is not None
    stale_sync = importlib.util.module_from_spec(stale_spec)
    stale_spec.loader.exec_module(stale_sync)
    monkeypatch.setattr(stale_sync, "PRODUCTION_REPO_ROOT", STAGING_ROOT)
    for name in ("_run_git", "_changed_paths", "_git_blob", "_read_repo_file"):
        monkeypatch.setattr(stale_sync, name, getattr(sync, name))

    touched = {"report": 0, "prepared": 0, "runtime": 0}

    def forbidden_report(*args: object, **kwargs: object) -> object:
        touched["report"] += 1
        raise AssertionError("report I/O occurred before stale-copy rejection")

    def forbidden_prepared(*args: object, **kwargs: object) -> object:
        touched["prepared"] += 1
        raise AssertionError("prepared I/O occurred before stale-copy rejection")

    def forbidden_runtime(*args: object, **kwargs: object) -> object:
        touched["runtime"] += 1
        raise AssertionError("runtime I/O occurred before stale-copy rejection")

    monkeypatch.setattr(stale_sync, "validate_registered_artifacts", forbidden_report)
    monkeypatch.setattr(stale_sync, "_prepared_path", forbidden_prepared)
    monkeypatch.setattr(stale_sync, "_write_prepared", forbidden_prepared)
    monkeypatch.setattr(stale_sync, "_read_runtime", forbidden_runtime)
    monkeypatch.setattr(stale_sync, "_locked_run", forbidden_runtime)

    prepared = tmp_path / "never-created"
    with pytest.raises(
        stale_sync.AuthorityError,
        match="executing script path differs from bound repository path",
    ):
        stale_sync.prepare_runtime_sync(
            prepared_directory=prepared,
            recorded_at=RECORDED_AT,
            config_path=stale_config,
            production=True,
        )
    assert touched == {"report": 0, "prepared": 0, "runtime": 0}
    assert not prepared.exists()


@pytest.mark.parametrize(
    ("drift", "match"),
    [
        ("parent", "I parent/L"),
        ("path", "I changed paths"),
        ("blob", "I frozen ledger blob drift"),
    ],
)
def test_repository_audit_rejects_parent_path_or_blob_drift(
    drift: str,
    match: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, payload, _, _, _ = _repository_audit_fixture(monkeypatch, drift)
    with pytest.raises(sync.RuntimeSyncError, match=match):
        sync.audit_production_repository_authority(config, payload)
