from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest


STAGING_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = (
    STAGING_ROOT
    / "scripts/route_a_v3/gse256185_aggregate_row_level_qualification_preflight_runtime_sync.py"
)
CONFIG_PATH = (
    STAGING_ROOT
    / "configs/route_a_v3_gse256185_aggregate_row_level_qualification_preflight_runtime_sync_v1.json"
)
SPEC = importlib.util.spec_from_file_location("gse256185_evt055_runtime_sync", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
sync = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sync)


def candidate_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def bind_config(config: dict) -> dict:
    bound = copy.deepcopy(config)
    binding = bound["implementation_binding"]
    binding["status"] = "BOUND"
    binding["implementation_commit"] = "b" * 40
    binding["implementation_script_sha256"] = "c" * 64
    binding["implementation_test_sha256"] = "d" * 64
    ledger = bound["repository_authority"]["predecessor_ledger"]
    ledger["status"] = "BOUND"
    ledger["commit"] = "a" * 40
    for index, item in enumerate(ledger["frozen_blobs"], start=1):
        item["sha256"] = f"{index:x}" * 64
    sync.validate_bound_config(bound)
    return bound


def predecessor_fixture(config: dict) -> dict[str, bytes]:
    fields = sync._runtime_truth_fields(config)
    status = {
        "schema_version": "test.status.v1",
        "updated_at": "2026-08-13T22:55:31+08:00",
        "opaque_status_field": "PRESERVE_EXACTLY",
        **copy.deepcopy(fields),
    }
    manifest = {
        "schema_version": "test.manifest.v1",
        "opaque_manifest_field": {"preserve": True},
        "outputs": [
            {
                "absolute_path": f"/fixture/predecessor/{index:03d}.json",
                "artifact_type": "FIXTURE",
            }
            for index in range(233)
        ],
        "registered_artifact_count": 5,
        **copy.deepcopy(fields),
    }
    events = [
        {
            "event_id": f"A1-EVT-{index:03d}",
            "at": "2026-08-13T22:00:00+08:00",
            "decision_id": "V3-DEC-021",
        }
        for index in range(1, 54)
    ]
    events.append(
        {
            "event_id": "A1-EVT-054",
            "at": "2026-08-13T22:55:31+08:00",
            "decision_id": "V3-DEC-022",
        }
    )
    payloads = {
        "STATUS.json": sync.json_bytes(status),
        "RUN_MANIFEST.json": sync.json_bytes(manifest),
        "EVENT_LOG.jsonl": b"".join(sync.compact_json_line(item) for item in events),
    }
    for name, payload in payloads.items():
        spec = config["runtime"]["predecessor_mutables"][name]
        spec["bytes"] = len(payload)
        spec["sha256"] = sync.sha256(payload)
    tail = payloads["EVENT_LOG.jsonl"].splitlines(keepends=True)[-1]
    config["runtime"]["predecessor_tail"]["bytes"] = len(tail)
    config["runtime"]["predecessor_tail"]["sha256"] = sync.sha256(tail)
    sync.validate_bound_config(config)
    return payloads


def write_runtime(run_root: Path, payloads: dict[str, bytes]) -> None:
    run_root.mkdir(parents=True)
    for name in sync.MUTABLE_NAMES:
        (run_root / name).write_bytes(payloads[name])


def test_candidate_binds_l_keeps_i_grouped_unknown_and_fails_before_io(
    tmp_path: Path,
) -> None:
    config = candidate_config()
    sync.validate_static_config(config)
    assert sync._binding_is_unknown(config["implementation_binding"])
    ledger = config["repository_authority"]["predecessor_ledger"]
    assert not sync._ledger_is_unknown(ledger)
    assert ledger["status"] == "BOUND"
    assert ledger["commit"] == "7f1546c1012df413781796aeb9d614a6601e7322"
    assert [item["sha256"] for item in ledger["frozen_blobs"]] == [
        "d62d7485b848571d618c797ad04284e8725b3fc6ba2e0dc42becffbfd0fc5810",
        "7fdafb7db153c2560e8275ebdd83ee75620cbd2a55405a096128d6e3fe01632e",
        "3ea7ab270e33d7eebad9288265aa3d5a07ba2884698c11d3b277f5163393b4cf",
        "5a690fe50cb7db99ff1ee0c702fe0dbc2a1996b282dab53a980b1b8a27e18821",
    ]
    with pytest.raises(sync.BindingError, match="implementation binding"):
        sync.validate_bound_config(config)

    partial_i = copy.deepcopy(config)
    partial_i["implementation_binding"]["status"] = "BOUND"
    with pytest.raises(sync.BindingError, match="partially known"):
        sync.validate_static_config(partial_i)

    unknown_l = copy.deepcopy(config)
    unknown_ledger = unknown_l["repository_authority"]["predecessor_ledger"]
    unknown_ledger["status"] = sync.UNKNOWN
    unknown_ledger["commit"] = sync.UNKNOWN
    for item in unknown_ledger["frozen_blobs"]:
        item["sha256"] = sync.UNKNOWN
    sync.validate_static_config(unknown_l)
    partial_l = copy.deepcopy(unknown_l)
    partial_l["repository_authority"]["predecessor_ledger"]["status"] = "BOUND"
    with pytest.raises(sync.BindingError, match="partially known"):
        sync.validate_static_config(partial_l)

    prepared = tmp_path / "never-created"
    with pytest.raises(sync.BindingError):
        sync.prepare_runtime_sync(
            prepared_directory=prepared,
            recorded_at="2026-08-13T23:45:00+08:00",
            production=False,
            config_override=config,
            run_root_override=tmp_path / "also-never-opened",
        )
    assert not prepared.exists()


def test_fully_bound_candidate_normalizes_only_four_i_scalars() -> None:
    bound = bind_config(candidate_config())
    normalized = sync.expected_unknown_i_config(bound)
    sync.validate_static_config(normalized)
    assert sync._binding_is_unknown(normalized["implementation_binding"])
    assert not sync._ledger_is_unknown(
        normalized["repository_authority"]["predecessor_ledger"]
    )
    assert sync.runtime_science_report_truth_projection(normalized) == (
        sync.runtime_science_report_truth_projection(bound)
    )


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


def test_exact_report_validation_checks_bytes_without_parsing_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = bind_config(candidate_config())
    report_path = Path(config["registered_artifacts"][0]["absolute_path"])
    invalid_json = b"not-json\x00" + b"x" * (15214 - len(b"not-json\x00"))
    real_read_bytes = Path.read_bytes
    real_sha256 = sync.sha256

    def fake_read_bytes(path: Path) -> bytes:
        return invalid_json if path == report_path else real_read_bytes(path)

    def fake_sha256(payload: bytes) -> str:
        if payload == invalid_json:
            return config["registered_artifacts"][0]["sha256"]
        return real_sha256(payload)

    monkeypatch.setattr(Path, "read_bytes", fake_read_bytes)
    monkeypatch.setattr(sync, "sha256", fake_sha256)
    result = sync.validate_registered_artifacts(config, verify_exact_bytes=True)
    assert result["status"] == "EXACT1_BYTES_AND_SHA256_VALIDATED"
    assert result["body_parse_count"] == 0
    assert result["payload_field_read_count"] == 0
    assert result["raw_asset_read_count"] == 0


def test_build_successor_is_exact_233_to_238_and_preserves_science() -> None:
    config = bind_config(candidate_config())
    predecessor = predecessor_fixture(config)
    old_status, old_manifest, _ = sync._parse_runtime(predecessor)
    successors = sync.build_successors(
        config, predecessor, "2026-08-13T23:45:00+08:00"
    )
    new_status, new_manifest, events = sync._parse_runtime(
        {name: successors[name] for name in sync.MUTABLE_NAMES}
    )

    assert len(events) == 55
    event = events[-1]
    assert event["event_id"] == "A1-EVT-055"
    assert event["decision_id"] == "V3-DEC-022"
    assert event["scientific_state_changed"] is False
    assert event["evidence_surface_changed"] is True
    assert event["evidence_gate_statuses_changed"] is True
    assert event["overall_qualification_gate_changed"] is False
    assert event["registered_artifact_body_parse_count"] == 0
    assert len(new_manifest["outputs"]) == 238
    assert new_manifest["outputs"][:233] == old_manifest["outputs"]
    assert new_manifest["outputs"][233] == config["registered_artifacts"][0]
    assert new_manifest["registered_artifact_count"] == 6
    expected_status = copy.deepcopy(old_status)
    expected_status["updated_at"] = "2026-08-13T23:45:00+08:00"
    assert new_status == expected_status
    truth = event["registered_evidence_truth"]
    assert truth["status"] == "STOP_CURRENT_PROTOCOL_NOT_QUALIFIED"
    assert truth["required_gate_count"] == 17
    assert truth["pass_like_gate_count"] == 7
    assert truth["nonpass_gate_count"] == 10
    assert truth["qualified"] is False
    assert event["frozen_scientific_state"]["current_qualified_counts"] == {
        "ordinary": 1,
        "a1": 1,
        "true_a2": 0,
        "canonical_records": 6547,
    }


def test_predecessor_identity_drift_stops_before_successor_build() -> None:
    config = bind_config(candidate_config())
    predecessor = predecessor_fixture(config)
    predecessor["STATUS.json"] += b" "
    with pytest.raises(sync.PredecessorError, match="identity drift"):
        sync.build_successors(config, predecessor, "2026-08-13T23:45:00+08:00")


def test_publish_recovers_only_from_ordered_mutable_prefix(tmp_path: Path) -> None:
    config = bind_config(candidate_config())
    run_root = tmp_path / "runtime"
    allowed = tmp_path / "prepared-root"
    prepared = allowed / "evt055"
    config["runtime"]["run_root"] = str(run_root)
    config["runtime"]["allowed_prepared_root"] = str(allowed)
    predecessor = predecessor_fixture(config)
    write_runtime(run_root, predecessor)

    result = sync.prepare_runtime_sync(
        prepared_directory=prepared,
        recorded_at="2026-08-13T23:45:00+08:00",
        production=False,
        config_override=config,
        run_root_override=run_root,
    )
    assert result["status"] == "PREPARED_NOT_PUBLISHED"
    assert result["manifest_output_transition"] == "233_TO_238"
    assert result["manifest_registered_artifact_transition"] == "5_TO_6"
    assert len(list(prepared.iterdir())) == 7

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
        "event_id": "A1-EVT-055",
        "registered_artifact_body_parse_count": 0,
    }


def test_repository_audit_proves_exact_l_i_b_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = bind_config(candidate_config())
    ledger = config["repository_authority"]["predecessor_ledger"]
    binding = config["implementation_binding"]
    l_commit, i_commit, head = ledger["commit"], "b" * 40, "e" * 40
    binding["implementation_commit"] = i_commit
    ledger_payloads = {path: f"ledger:{path}".encode() for path in sync.LEDGER_PATHS}
    for item in ledger["frozen_blobs"]:
        item["sha256"] = sync.sha256(ledger_payloads[item["path"]])
    script_payload = b"implementation script"
    test_payload = b"implementation test"
    binding["implementation_script_sha256"] = sync.sha256(script_payload)
    binding["implementation_test_sha256"] = sync.sha256(test_payload)
    sync.validate_bound_config(config)
    i_config_payload = sync.json_bytes(sync.expected_unknown_i_config(config))
    config_payload = sync.json_bytes(config)
    git_overrides: dict[tuple[str, ...], bytes] = {}

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
            ("rev-parse", f"{i_commit}^"): f"{l_commit}\n".encode(),
        }
        mapping.update(git_overrides)
        return mapping[args]

    def fake_changed_paths(repo_root: Path, commit: str) -> list[str]:
        assert repo_root == sync.PRODUCTION_REPO_ROOT
        if commit == l_commit:
            return sorted(sync.LEDGER_PATHS)
        if commit == i_commit:
            return sorted(config["repository_authority"]["implementation_exact_changed_paths"])
        if commit == head:
            return [sync.CONFIG_REPO_PATH]
        raise AssertionError(commit)

    def fake_git_blob(repo_root: Path, commit: str, path: str) -> bytes:
        assert repo_root == sync.PRODUCTION_REPO_ROOT
        if path in ledger_payloads and commit in {l_commit, i_commit, head}:
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
    result = sync.audit_production_repository_authority(config, config_payload)
    assert result == {
        "status": "PASS_EXACT4_L_TO_EXACT3_I_TO_CONFIG_ONLY_B",
        "ledger_l_commit": l_commit,
        "implementation_i_commit": i_commit,
        "binding_b_commit": head,
        "upstream_head_commit": head,
        "origin_branch_head_commit": head,
        "worktree_and_index_clean": True,
    }

    git_overrides[("rev-parse", f"{i_commit}^")] = f"{'f' * 40}\n".encode()
    with pytest.raises(sync.RuntimeSyncError, match="I parent/L"):
        sync.audit_production_repository_authority(config, config_payload)
    git_overrides.clear()
    git_overrides[("status", "--porcelain=v1", "--untracked-files=all")] = b"dirty\n"
    with pytest.raises(sync.AuthorityError, match="dirty"):
        sync.audit_production_repository_authority(config, config_payload)
