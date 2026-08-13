from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest


STAGING_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = (
    STAGING_ROOT
    / "scripts/route_a_v3/gse256185_public_identifier_pool_geometry_preflight_runtime_sync.py"
)
CONFIG_PATH = (
    STAGING_ROOT
    / "configs/route_a_v3_gse256185_public_identifier_pool_geometry_preflight_runtime_sync_v1.json"
)
SPEC = importlib.util.spec_from_file_location("gse256185_evt053_runtime_sync", SCRIPT_PATH)
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
    sync.validate_bound_config(bound)
    return bound


def predecessor_fixture(config: dict) -> dict[str, bytes]:
    fields = sync._runtime_truth_fields(config)
    status = {
        "schema_version": "test.status.v1",
        "updated_at": "2026-08-13T20:41:15+08:00",
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
            for index in range(224)
        ],
        "registered_artifact_count": 4,
        **copy.deepcopy(fields),
    }
    events = [
        {
            "event_id": f"A1-EVT-{index:03d}",
            "at": "2026-08-13T20:00:00+08:00",
            "decision_id": "V3-DEC-020",
        }
        for index in range(1, 52)
    ]
    events.append(
        {
            "event_id": "A1-EVT-052",
            "at": "2026-08-13T20:41:15+08:00",
            "decision_id": "V3-DEC-021",
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


def test_disk_i2_or_b2_normalizes_only_four_binding_scalars(
    tmp_path: Path,
) -> None:
    config = candidate_config()
    sync.validate_static_config(config)
    ledger = config["repository_authority"]["predecessor_ledger"]
    assert ledger["status"] == "BOUND"
    assert ledger["commit"] == "5b4b763b3a0b9a8886b5a526b3dbfef9c7349fbd"
    assert [item["sha256"] for item in ledger["frozen_blobs"]] == [
        "f178750dfed15fde42699840caef314e39b6b4fa4acf72dd8606695ae7c930c4",
        "d0d3eeb99cc79bc80ee33a6d98cd78f3d602a7bcc25fce0e529e64089e0026d4",
        "7c7f6d47adc8914d1a0b3d92e59167c1743e6dc4125eec0d73d74e8f0f2a0cdf",
        "89702695c37b759a1c78e5b9d73653b0f158861e41a0232c7b6b6b061259cfc2",
    ]
    i1 = config["repository_authority"]["predecessor_implementation_i1"]
    assert i1["commit"] == sync.I1_COMMIT
    assert i1["blob_sha256_by_path"] == sync.I1_BLOBS

    if sync._binding_is_unknown(config["implementation_binding"]):
        with pytest.raises(sync.BindingError, match="implementation binding"):
            sync.validate_bound_config(config)
    else:
        sync.validate_bound_config(config)
    normalized = copy.deepcopy(config)
    for key in (
        "status",
        "implementation_commit",
        "implementation_script_sha256",
        "implementation_test_sha256",
    ):
        normalized["implementation_binding"][key] = sync.UNKNOWN
    sync.validate_static_config(normalized)
    assert sync._binding_is_unknown(normalized["implementation_binding"])
    restored = bind_config(normalized)
    assert sync.runtime_science_report_truth_projection(restored) == (
        sync.runtime_science_report_truth_projection(config)
    )

    partial = copy.deepcopy(config)
    partial["implementation_binding"]["status"] = "BOUND"
    with pytest.raises(sync.BindingError, match="partially known"):
        sync.validate_static_config(partial)

    partial = copy.deepcopy(config)
    partial["repository_authority"]["predecessor_ledger"]["frozen_blobs"][0][
        "sha256"
    ] = sync.UNKNOWN
    with pytest.raises(sync.BindingError, match="partially known"):
        sync.validate_static_config(partial)

    prepared = tmp_path / "never-created"
    with pytest.raises(sync.BindingError):
        sync.prepare_runtime_sync(
            prepared_directory=prepared,
            recorded_at="2026-08-13T21:30:00+08:00",
            production=False,
            config_override=normalized,
            run_root_override=tmp_path / "also-never-opened",
        )
    assert not prepared.exists()


def test_synthetic_disk_b_accepts_fully_bound_implementation() -> None:
    config = bind_config(candidate_config())
    sync.validate_bound_config(config)
    assert config["implementation_binding"]["status"] == "BOUND"
    assert not sync._binding_is_unknown(config["implementation_binding"])


def test_exact_report_validation_does_not_parse_invalid_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = bind_config(candidate_config())
    report_path = Path(config["registered_artifacts"][0]["absolute_path"])
    invalid_json = b"not-json\x00" + b"x" * (7441 - len(b"not-json\x00"))
    real_read_bytes = Path.read_bytes
    real_sha256 = sync.sha256

    def fake_read_bytes(path: Path) -> bytes:
        if path == report_path:
            return invalid_json
        return real_read_bytes(path)

    def fake_sha256(payload: bytes) -> str:
        if payload == invalid_json:
            return config["registered_artifacts"][0]["sha256"]
        return real_sha256(payload)

    monkeypatch.setattr(Path, "read_bytes", fake_read_bytes)
    monkeypatch.setattr(sync, "sha256", fake_sha256)
    result = sync.validate_registered_artifacts(config, verify_exact_bytes=True)
    assert result == {
        "status": "EXACT1_BYTES_AND_SHA256_VALIDATED",
        "artifact_count": 1,
        "exact_byte_validation_count": 1,
        "body_parse_count": 0,
        "payload_field_read_count": 0,
        "raw_asset_read_count": 0,
        "registered_artifacts": config["registered_artifacts"],
    }


def test_build_successor_is_exact_224_to_229_and_preserves_scientific_state() -> None:
    config = bind_config(candidate_config())
    predecessor = predecessor_fixture(config)
    old_status, old_manifest, _ = sync._parse_runtime(predecessor)
    successors = sync.build_successors(
        config, predecessor, "2026-08-13T21:30:00+08:00"
    )
    new_status, new_manifest, events = sync._parse_runtime(
        {name: successors[name] for name in sync.MUTABLE_NAMES}
    )

    assert len(events) == 53
    assert events[-1]["event_id"] == "A1-EVT-053"
    assert events[-1]["scientific_state_changed"] is False
    assert events[-1]["evidence_surface_changed"] is True
    assert events[-1]["registered_artifact_body_parse_count"] == 0
    assert len(new_manifest["outputs"]) == 229
    assert new_manifest["outputs"][:224] == old_manifest["outputs"]
    assert new_manifest["outputs"][224] == config["registered_artifacts"][0]
    assert new_manifest["registered_artifact_count"] == 5
    expected_status = copy.deepcopy(old_status)
    expected_status["updated_at"] = "2026-08-13T21:30:00+08:00"
    assert new_status == expected_status
    assert events[-1]["registered_evidence_truth"]["aggregate_geometry"] == {
        "identifier_row_count": 11404,
        "identifier_pool_count": 652,
        "single_parent_pool_count": 637,
        "two_parent_pool_count": 15,
        "strict_eligible_pool_count": 634,
        "strict_controlled_candidate_identifier_count": 7292,
        "family_closure_inference_controlled_candidate_identifier_count": 7294,
        "missing_dot_identifier_anomaly_count": 1,
        "unsigned_ccc_identifier_anomaly_count": 1,
    }
    assert events[-1]["frozen_scientific_state"]["current_qualified_counts"] == {
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
        sync.build_successors(
            config, predecessor, "2026-08-13T21:30:00+08:00"
        )


def test_publish_recovers_only_from_ordered_mutable_prefix(tmp_path: Path) -> None:
    config = bind_config(candidate_config())
    run_root = tmp_path / "runtime"
    allowed = tmp_path / "prepared-root"
    prepared = allowed / "evt053"
    config["runtime"]["run_root"] = str(run_root)
    config["runtime"]["allowed_prepared_root"] = str(allowed)
    predecessor = predecessor_fixture(config)
    write_runtime(run_root, predecessor)

    result = sync.prepare_runtime_sync(
        prepared_directory=prepared,
        recorded_at="2026-08-13T21:30:00+08:00",
        production=False,
        config_override=config,
        run_root_override=run_root,
    )
    assert result["status"] == "PREPARED_NOT_PUBLISHED"
    assert result["manifest_output_transition"] == "224_TO_229"
    assert result["registered_artifact_body_parse_count"] == 0
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

    published = sync.publish_prepared(
        prepared_directory=prepared,
        production=False,
        config_override=config,
        run_root_override=run_root,
    )
    assert published["status"] == "PUBLISHED_VERIFIED"
    verified = sync.validate_published(
        prepared_directory=prepared,
        production=False,
        config_override=config,
        run_root_override=run_root,
    )
    assert verified == {
        "status": "PUBLISHED_VERIFIED",
        "event_id": "A1-EVT-053",
        "registered_artifact_body_parse_count": 0,
    }
    for name in config["runtime"]["immutable_publish_order"]:
        assert (run_root / name).read_bytes() == (prepared / name).read_bytes()


def test_repository_audit_proves_exact_l_i1_i2_b2_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = bind_config(candidate_config())
    ledger = config["repository_authority"]["predecessor_ledger"]
    binding = config["implementation_binding"]
    l_commit = ledger["commit"]
    frozen_i1 = config["repository_authority"]["predecessor_implementation_i1"]
    i1_commit = frozen_i1["commit"]
    i2_commit, head = "b" * 40, "e" * 40
    binding["implementation_commit"] = i2_commit
    ledger_payloads = {
        path: f"ledger:{path}".encode() for path in sync.LEDGER_PATHS
    }
    script_payload = b"implementation script"
    test_payload = b"implementation test"
    binding["implementation_script_sha256"] = sync.sha256(script_payload)
    binding["implementation_test_sha256"] = sync.sha256(test_payload)
    i2_config_payload = sync.json_bytes(sync.expected_unknown_i_config(config))
    config_payload = sync.json_bytes(config)
    i1_config = candidate_config()
    i1_config_payload = sync.json_bytes(i1_config)
    i1_script_payload = b"frozen I1 script"
    i1_test_payload = b"frozen I1 test"
    sync.validate_bound_config(config)

    exact_identity_by_payload = {
        **{
            ledger_payloads[item["path"]]: item["sha256"]
            for item in ledger["frozen_blobs"]
        },
        i1_config_payload: frozen_i1["blob_sha256_by_path"][sync.CONFIG_REPO_PATH],
        i1_script_payload: frozen_i1["blob_sha256_by_path"][sync.SCRIPT_REPO_PATH],
        i1_test_payload: frozen_i1["blob_sha256_by_path"][sync.TEST_REPO_PATH],
    }
    real_sha256 = sync.sha256

    def fake_sha256(payload: bytes) -> str:
        return exact_identity_by_payload.get(payload, real_sha256(payload))

    git_overrides: dict[tuple[str, ...], bytes] = {}

    def fake_run_git(repo_root: Path, *args: str) -> bytes:
        assert repo_root == sync.PRODUCTION_REPO_ROOT
        mapping = {
            ("rev-parse", "HEAD"): f"{head}\n".encode(),
            ("rev-parse", "--abbrev-ref", "HEAD"): f"{sync.BRANCH}\n".encode(),
            ("rev-parse", "--abbrev-ref", "@{upstream}"): (
                f"origin/{sync.BRANCH}\n".encode()
            ),
            ("rev-parse", "@{upstream}"): f"{head}\n".encode(),
            (
                "rev-parse",
                "--verify",
                f"refs/remotes/origin/{sync.BRANCH}",
            ): f"{head}\n".encode(),
            ("status", "--porcelain=v1", "--untracked-files=all"): b"",
            ("rev-parse", f"{head}^"): f"{i2_commit}\n".encode(),
            ("rev-parse", f"{i2_commit}^"): f"{i1_commit}\n".encode(),
            ("rev-parse", f"{i1_commit}^"): f"{l_commit}\n".encode(),
        }
        mapping.update(git_overrides)
        return mapping[args]

    def fake_changed_paths(repo_root: Path, commit: str) -> list[str]:
        assert repo_root == sync.PRODUCTION_REPO_ROOT
        if commit == l_commit:
            return sorted(sync.LEDGER_PATHS)
        if commit == i1_commit:
            return sorted(frozen_i1["exact_changed_paths"])
        if commit == i2_commit:
            return sorted(config["repository_authority"]["implementation_exact_changed_paths"])
        if commit == head:
            return [sync.CONFIG_REPO_PATH]
        raise AssertionError(commit)

    def fake_git_blob(repo_root: Path, commit: str, path: str) -> bytes:
        assert repo_root == sync.PRODUCTION_REPO_ROOT
        if path in ledger_payloads and commit in {l_commit, head}:
            return ledger_payloads[path]
        if path == sync.CONFIG_REPO_PATH and commit == i1_commit:
            return i1_config_payload
        if path == sync.SCRIPT_REPO_PATH and commit == i1_commit:
            return i1_script_payload
        if path == sync.TEST_REPO_PATH and commit == i1_commit:
            return i1_test_payload
        if path == sync.CONFIG_REPO_PATH and commit == i2_commit:
            return i2_config_payload
        if path == sync.CONFIG_REPO_PATH and commit == head:
            return config_payload
        if path == sync.SCRIPT_REPO_PATH and commit in {i2_commit, head}:
            return script_payload
        if path == sync.TEST_REPO_PATH and commit in {i2_commit, head}:
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
    monkeypatch.setattr(sync, "sha256", fake_sha256)
    result = sync.audit_production_repository_authority(config, config_payload)
    assert result == {
        "status": "PASS_EXACT_L_TO_I1_TO_I2_TO_CONFIG_ONLY_B2",
        "ledger_l_commit": l_commit,
        "frozen_i1_commit": i1_commit,
        "implementation_i2_commit": i2_commit,
        "binding_b2_commit": head,
        "upstream_head_commit": head,
        "origin_branch_head_commit": head,
        "worktree_and_index_clean": True,
    }

    git_overrides[("rev-parse", f"{i2_commit}^")] = f"{'f' * 40}\n".encode()
    with pytest.raises(sync.RuntimeSyncError, match="I2 parent/I1"):
        sync.audit_production_repository_authority(config, config_payload)
    git_overrides.clear()

    def changed_path_drift(repo_root: Path, commit: str) -> list[str]:
        paths = fake_changed_paths(repo_root, commit)
        return paths + ["unexpected.txt"] if commit == i2_commit else paths

    monkeypatch.setattr(sync, "_changed_paths", changed_path_drift)
    with pytest.raises(sync.RuntimeSyncError, match="I2 changed paths"):
        sync.audit_production_repository_authority(config, config_payload)
    monkeypatch.setattr(sync, "_changed_paths", fake_changed_paths)

    def frozen_i1_blob_drift(repo_root: Path, commit: str, path: str) -> bytes:
        if commit == i1_commit and path == sync.SCRIPT_REPO_PATH:
            return b"drift"
        return fake_git_blob(repo_root, commit, path)

    monkeypatch.setattr(sync, "_git_blob", frozen_i1_blob_drift)
    with pytest.raises(sync.AuthorityError, match="frozen I1 blob drift"):
        sync.audit_production_repository_authority(config, config_payload)
