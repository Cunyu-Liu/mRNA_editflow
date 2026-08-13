from __future__ import annotations

import copy
import hashlib
import importlib.util
from pathlib import Path
from typing import Any

import pytest


STAGING_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = STAGING_ROOT / "configs/route_a_v3_dec021_authority_runtime_sync_v1.json"
SCRIPT_PATH = STAGING_ROOT / "scripts/route_a_v3/dec021_authority_runtime_sync.py"
SPEC = importlib.util.spec_from_file_location("dec021_authority_runtime_sync", SCRIPT_PATH)
assert SPEC and SPEC.loader
RUNTIME_SYNC = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNTIME_SYNC)


def _disk_config() -> dict[str, Any]:
    return RUNTIME_SYNC.load_config(CONFIG_PATH, require_bound=False)


def _assert_disk_lifecycle(
    config: dict[str, Any],
    *,
    script_path: Path = SCRIPT_PATH,
    test_path: Path = Path(__file__),
) -> str:
    RUNTIME_SYNC.validate_static_config(config)
    assert RUNTIME_SYNC._authority_binding_state(config["repository_authority"]) == (
        "BOUND"
    )
    i1 = config["repository_authority"]["predecessor_implementation_i1"]
    assert i1 == {
        "commit": RUNTIME_SYNC.I1_COMMIT,
        "expected_parent": config["repository_authority"]["authority_commit"],
        "exact_changed_paths": RUNTIME_SYNC.IMPLEMENTATION_PATHS,
        "blob_sha256_by_path": RUNTIME_SYNC.I1_BLOB_SHA256_BY_PATH,
    }
    binding = config["implementation_binding"]
    assert binding["implementation_commit_exact_changed_paths"] == (
        RUNTIME_SYNC.IMPLEMENTATION_PATHS
    )
    assert binding["binding_commit_exact_changed_paths"] == [
        RUNTIME_SYNC.CONFIG_REPO_PATH
    ]
    state = RUNTIME_SYNC._implementation_binding_state(binding)
    if state == "UNKNOWN":
        assert [binding[field] for field in RUNTIME_SYNC.UNKNOWN_BINDING_FIELDS] == [
            RUNTIME_SYNC.UNKNOWN
        ] * 4
        return state

    assert state == "BOUND"
    assert RUNTIME_SYNC.HEX40.fullmatch(binding["implementation_commit"])
    assert hashlib.sha256(script_path.read_bytes()).hexdigest() == (
        binding["implementation_script_sha256"]
    )
    assert hashlib.sha256(test_path.read_bytes()).hexdigest() == (
        binding["implementation_test_sha256"]
    )
    i_config = RUNTIME_SYNC.expected_unknown_i2_config(config)
    RUNTIME_SYNC.validate_static_config(i_config)
    assert [
        i_config["implementation_binding"][field]
        for field in RUNTIME_SYNC.UNKNOWN_BINDING_FIELDS
    ] == [RUNTIME_SYNC.UNKNOWN] * 4
    assert {
        key: value
        for key, value in config["implementation_binding"].items()
        if key not in RUNTIME_SYNC.UNKNOWN_BINDING_FIELDS
    } == {
        key: value
        for key, value in i_config["implementation_binding"].items()
        if key not in RUNTIME_SYNC.UNKNOWN_BINDING_FIELDS
    }
    return state


def _bound_config() -> dict[str, Any]:
    config = _unknown_config()
    config["implementation_binding"].update(
        {
            "status": RUNTIME_SYNC.BOUND,
            "implementation_commit": "2" * 40,
            "implementation_script_sha256": "3" * 64,
            "implementation_test_sha256": "4" * 64,
        }
    )
    RUNTIME_SYNC.validate_static_config(config)
    return config


def _unknown_config() -> dict[str, Any]:
    config = RUNTIME_SYNC.expected_unknown_i2_config(copy.deepcopy(_disk_config()))
    RUNTIME_SYNC.validate_static_config(config)
    return config


def _outer_runtime_fields() -> dict[str, Any]:
    return {
        "qualified_ordinary_studies": 1,
        "qualified_a1_studies": 1,
        "qualified_a2_dense_studies": 0,
        "canonical_intervention_record_count": 6547,
        "canonical_record_count": 6547,
        "run_status": "IN_PROGRESS",
        "evidence_status": "SCRATCH_ROUTE_QUALIFIED_GLOBAL_PHASE_INCOMPLETE",
        "gate_status": "A1_PHASE_INCOMPLETE_GLOBAL_REQUIREMENTS",
        "qualified": False,
        "training_started": False,
        "training_allowed": False,
        "training_authorized": False,
        "gpu_work_started": False,
        "gpu_work_allowed": False,
        "model_selection_allowed": False,
        "next_phase_authorized": False,
        "scientific_claim_status": "NOT_ESTABLISHED",
    }


def _synthetic_predecessor(
    config: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> dict[str, bytes]:
    prior_decisions = ["V3-DEC-017", "V3-DEC-018", "V3-DEC-019", "V3-DEC-020"]
    prior_authority = {
        "decision_id": "V3-DEC-020",
        "authority_commit": "8" * 40,
        "scope": "HISTORICAL_DEC020_AUTHORITY",
    }
    status = {
        **_outer_runtime_fields(),
        "updated_at": "2026-08-13T18:53:08+08:00",
        "active_amendment_decision_ids": prior_decisions,
        "current_contract_authority": prior_authority,
        "existing_status_field": "PRESERVED",
    }
    manifest = {
        **_outer_runtime_fields(),
        "active_authority_commit": "9" * 40,
        "active_amendment_decision_ids": prior_decisions,
        "current_contract_authority": prior_authority,
        "registered_artifact_count": 4,
        "outputs": [
            {
                "absolute_path": f"/existing/{index:03d}",
                "artifact_type": f"EXISTING_{index:03d}",
                "bytes": index + 1,
                "sha256": f"{index + 1:064x}",
            }
            for index in range(220)
        ],
        "existing_manifest_field": "PRESERVED",
    }
    events = [
        {
            "event_id": f"A1-EVT-{index:03d}",
            "at": "2026-08-12T00:00:00+08:00",
            "event": "HISTORICAL",
        }
        for index in range(1, 51)
    ]
    events.append(
        {
            "event_id": "A1-EVT-051",
            "at": "2026-08-13T18:53:08+08:00",
            "event": (
                "GSE200304_DEC020_SCRATCH_ROUTE_A1_QUALIFICATION_"
                "REGISTERED_RUNTIME_SYNCED"
            ),
            "decision_id": "V3-DEC-020",
            "predecessor_event_id": "A1-EVT-050",
            "registered_artifact_count": 4,
            "manifest_output_count_before": 212,
            "manifest_output_count_after": 220,
        }
    )
    payloads = {
        "STATUS.json": RUNTIME_SYNC.json_bytes(status),
        "RUN_MANIFEST.json": RUNTIME_SYNC.json_bytes(manifest),
        "EVENT_LOG.jsonl": b"".join(
            RUNTIME_SYNC.compact_json_line(event) for event in events
        ),
    }
    identities: dict[str, dict[str, Any]] = {}
    for name, payload in payloads.items():
        identities[name] = {
            "bytes": len(payload),
            "sha256": RUNTIME_SYNC.sha256(payload),
            "snapshot_name": config["runtime"]["predecessor_mutables"][name][
                "snapshot_name"
            ],
        }
    tail_payload = RUNTIME_SYNC.compact_json_line(events[-1])
    tail = {
        "event_id": "A1-EVT-051",
        "decision_id": "V3-DEC-020",
        "bytes": len(tail_payload),
        "sha256": RUNTIME_SYNC.sha256(tail_payload),
    }
    config["runtime"]["predecessor_mutables"] = copy.deepcopy(identities)
    config["runtime"]["predecessor_tail"] = copy.deepcopy(tail)
    monkeypatch.setattr(RUNTIME_SYNC, "PREDECESSOR_IDENTITIES", identities)
    monkeypatch.setattr(RUNTIME_SYNC, "PREDECESSOR_TAIL", tail)
    RUNTIME_SYNC.validate_static_config(config)
    return payloads


def _make_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, Any], dict[str, bytes], Path, Path]:
    config = _bound_config()
    run_root = tmp_path / "run"
    allowed_root = tmp_path / "prepared-root"
    prepared = allowed_root / "evt052-job"
    run_root.mkdir()
    allowed_root.mkdir()
    config["runtime"]["run_root"] = str(run_root)
    config["runtime"]["allowed_prepared_root"] = str(allowed_root)
    predecessor = _synthetic_predecessor(config, monkeypatch)
    for name, payload in predecessor.items():
        (run_root / name).write_bytes(payload)
    return config, predecessor, run_root, prepared


def _read_runtime(run_root: Path) -> dict[str, bytes]:
    return {
        name: (run_root / name).read_bytes() for name in RUNTIME_SYNC.MUTABLE_NAMES
    }


def _install_fake_repository(
    monkeypatch: pytest.MonkeyPatch,
    config: dict[str, Any],
    *,
    drift_authority_paths: bool = False,
    drift_i1_blob: bool = False,
) -> bytes:
    authority_commit = config["repository_authority"]["authority_commit"]
    i1_commit = config["repository_authority"]["predecessor_implementation_i1"][
        "commit"
    ]
    implementation_commit = "6" * 40
    binding_commit = "b" * 40
    script_payload = b"DEC021 authority runtime sync producer I\n"
    test_payload = b"DEC021 authority runtime sync focused test I\n"
    binding = config["implementation_binding"]
    binding.update(
        {
            "status": RUNTIME_SYNC.BOUND,
            "implementation_commit": implementation_commit,
            "implementation_script_sha256": hashlib.sha256(script_payload).hexdigest(),
            "implementation_test_sha256": hashlib.sha256(test_payload).hexdigest(),
        }
    )
    RUNTIME_SYNC.validate_static_config(config)
    config_payload = RUNTIME_SYNC.json_bytes(config)
    i_payload = RUNTIME_SYNC.json_bytes(
        RUNTIME_SYNC.expected_unknown_i2_config(config)
    )

    authority_payloads: dict[str, bytes] = {}
    digest_overrides: dict[bytes, str] = {}
    for index, item in enumerate(config["repository_authority"]["authority_files"]):
        seed = f"DEC021 authority blob {index}\n".encode()
        payload = (seed * (item["bytes"] // len(seed) + 1))[: item["bytes"]]
        authority_payloads[item["path"]] = payload
        digest_overrides[payload] = item["sha256"]
    i1_payloads = {
        relative: f"frozen I1 blob {index} for {relative}\n".encode()
        for index, relative in enumerate(RUNTIME_SYNC.IMPLEMENTATION_PATHS)
    }
    for relative, payload in i1_payloads.items():
        digest_overrides[payload] = RUNTIME_SYNC.I1_BLOB_SHA256_BY_PATH[relative]
    if drift_i1_blob:
        i1_payloads[RUNTIME_SYNC.CONFIG_REPO_PATH] += b"drift"
    real_sha256 = RUNTIME_SYNC.sha256

    def fake_sha256(payload: bytes) -> str:
        return digest_overrides.get(payload, real_sha256(payload))

    changed = {
        authority_commit: list(RUNTIME_SYNC.AUTHORITY_PATHS),
        i1_commit: list(RUNTIME_SYNC.IMPLEMENTATION_PATHS),
        implementation_commit: list(RUNTIME_SYNC.IMPLEMENTATION_PATHS),
        binding_commit: [RUNTIME_SYNC.CONFIG_REPO_PATH],
    }
    if drift_authority_paths:
        changed[authority_commit].append("unexpected/path")
    blobs: dict[tuple[str, str], bytes] = {
        **{
            (authority_commit, relative): payload
            for relative, payload in authority_payloads.items()
        },
        **{
            (binding_commit, relative): payload
            for relative, payload in authority_payloads.items()
        },
        **{
            (i1_commit, relative): payload
            for relative, payload in i1_payloads.items()
        },
        (implementation_commit, RUNTIME_SYNC.CONFIG_REPO_PATH): i_payload,
        (implementation_commit, RUNTIME_SYNC.SCRIPT_REPO_PATH): script_payload,
        (implementation_commit, RUNTIME_SYNC.TEST_REPO_PATH): test_payload,
        (binding_commit, RUNTIME_SYNC.CONFIG_REPO_PATH): config_payload,
    }

    def fake_git(_repo: Path, *arguments: str) -> bytes:
        if arguments == ("rev-parse", "HEAD"):
            return f"{binding_commit}\n".encode()
        if arguments == ("rev-parse", "@{upstream}"):
            return f"{binding_commit}\n".encode()
        if arguments == (
            "rev-parse",
            "--verify",
            f"refs/remotes/origin/{RUNTIME_SYNC.BRANCH}",
        ):
            return f"{binding_commit}\n".encode()
        if arguments == ("rev-parse", "--abbrev-ref", "HEAD"):
            return f"{RUNTIME_SYNC.BRANCH}\n".encode()
        if arguments == ("rev-parse", "--abbrev-ref", "@{upstream}"):
            return f"origin/{RUNTIME_SYNC.BRANCH}\n".encode()
        if arguments == ("status", "--porcelain=v1", "--untracked-files=all"):
            return b""
        if arguments == ("rev-parse", f"{binding_commit}^"):
            return f"{implementation_commit}\n".encode()
        if arguments == ("rev-parse", f"{implementation_commit}^"):
            return f"{i1_commit}\n".encode()
        if arguments == ("rev-parse", f"{i1_commit}^"):
            return f"{authority_commit}\n".encode()
        if arguments == ("rev-parse", f"{authority_commit}^"):
            return f"{RUNTIME_SYNC.AUTHORITY_PARENT}\n".encode()
        if arguments[:4] == (
            "diff-tree",
            "--no-commit-id",
            "--name-only",
            "-r",
        ):
            return ("\n".join(changed[arguments[4]]) + "\n").encode()
        if arguments[0] == "show":
            commit, relative = arguments[1].split(":", 1)
            return blobs[(commit, relative)]
        raise AssertionError(arguments)

    worktree = {
        **authority_payloads,
        RUNTIME_SYNC.CONFIG_REPO_PATH: config_payload,
        RUNTIME_SYNC.SCRIPT_REPO_PATH: script_payload,
        RUNTIME_SYNC.TEST_REPO_PATH: test_payload,
    }
    monkeypatch.setattr(RUNTIME_SYNC, "sha256", fake_sha256)
    monkeypatch.setattr(RUNTIME_SYNC, "_run_git", fake_git)
    monkeypatch.setattr(
        RUNTIME_SYNC,
        "_read_repo_file",
        lambda _repo, relative: worktree[relative],
    )
    return config_payload


def test_disk_config_is_strict_real_i_or_b_with_exact10_authority_bound() -> None:
    config = _disk_config()
    assert _assert_disk_lifecycle(config) in {"UNKNOWN", "BOUND"}
    authority = config["repository_authority"]
    assert authority["authority_commit"] == (
        "1ee575799a4b3289f9b7d684b4b31885dde0bd50"
    )
    assert authority["authority_expected_parent"] == RUNTIME_SYNC.AUTHORITY_PARENT
    assert len(authority["authority_files"]) == 10
    assert authority["authority_exact_changed_paths"] == RUNTIME_SYNC.AUTHORITY_PATHS
    assert config["registered_artifacts"] == []
    assert config["runtime"]["predecessor_event_id"] == "A1-EVT-051"
    assert config["runtime"]["successor_event_id"] == "A1-EVT-052"
    assert config["runtime"]["predecessor_manifest_output_count"] == 220
    assert config["runtime"]["successor_manifest_output_count"] == 224
    assert config["runtime"]["predecessor_mutables"] == (
        RUNTIME_SYNC.PREDECESSOR_IDENTITIES
    )


def test_synthetic_bound_disk_config_hashes_and_normalises_to_i(
    tmp_path: Path,
) -> None:
    script_payload = b"DEC021 runtime implementation exact3 I2\n"
    test_payload = b"DEC021 runtime focused test exact3 I2\n"
    script_path = tmp_path / "runtime_sync.py"
    test_path = tmp_path / "test_runtime_sync.py"
    config_path = tmp_path / CONFIG_PATH.name
    script_path.write_bytes(script_payload)
    test_path.write_bytes(test_payload)
    config = _disk_config()
    config["implementation_binding"].update(
        {
            "status": RUNTIME_SYNC.BOUND,
            "implementation_commit": "2" * 40,
            "implementation_script_sha256": hashlib.sha256(
                script_payload
            ).hexdigest(),
            "implementation_test_sha256": hashlib.sha256(test_payload).hexdigest(),
        }
    )
    config_path.write_bytes(RUNTIME_SYNC.json_bytes(config))
    config = RUNTIME_SYNC.load_config(config_path, require_bound=True)
    assert _assert_disk_lifecycle(
        config, script_path=script_path, test_path=test_path
    ) == "BOUND"


def test_partial_implementation_or_authority_binding_is_rejected() -> None:
    config = _unknown_config()
    config["implementation_binding"]["status"] = RUNTIME_SYNC.BOUND
    with pytest.raises(RUNTIME_SYNC.BindingError, match="BOUND implementation"):
        RUNTIME_SYNC.validate_static_config(config)

    config = _unknown_config()
    config["implementation_binding"]["implementation_commit"] = "2" * 40
    with pytest.raises(RUNTIME_SYNC.BindingError, match="partially known"):
        RUNTIME_SYNC.validate_static_config(config)

    config = _unknown_config()
    config["repository_authority"]["authority_files"][0]["sha256"] = (
        RUNTIME_SYNC.UNKNOWN
    )
    with pytest.raises(RUNTIME_SYNC.RuntimeSyncError):
        RUNTIME_SYNC.validate_static_config(config)


def test_unknown_exact3_i_stops_before_git_runtime_or_prepared_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        calls.append("external")
        raise AssertionError("external I/O occurred")

    monkeypatch.setattr(RUNTIME_SYNC, "audit_production_repository_authority", forbidden)
    monkeypatch.setattr(RUNTIME_SYNC, "_read_runtime", forbidden)
    monkeypatch.setattr(RUNTIME_SYNC, "_read_prepared", forbidden)
    with pytest.raises(RUNTIME_SYNC.BindingError, match="not BOUND"):
        RUNTIME_SYNC._context(
            CONFIG_PATH,
            production=False,
            config_override=_unknown_config(),
        )
    assert calls == []


def test_repository_a_i1_i2_b2_lifecycle_is_audited(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _bound_config()
    payload = _install_fake_repository(monkeypatch, config)
    audit = RUNTIME_SYNC.audit_production_repository_authority(config, payload)
    assert audit["status"] == (
        "PASS_EXACT10_A_I1_EXACT3_I2_EXACT3_CONFIG_ONLY_B2"
    )
    assert audit["predecessor_implementation_i1_commit"] == RUNTIME_SYNC.I1_COMMIT
    assert audit["authority_blob_count"] == 10
    assert audit["worktree_and_index_clean"] is True


def test_repository_authority_path_drift_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _bound_config()
    payload = _install_fake_repository(
        monkeypatch, config, drift_authority_paths=True
    )
    with pytest.raises(RUNTIME_SYNC.RuntimeSyncError, match="A exact10 drift"):
        RUNTIME_SYNC.audit_production_repository_authority(config, payload)


def test_repository_i1_blob_drift_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _bound_config()
    payload = _install_fake_repository(monkeypatch, config, drift_i1_blob=True)
    with pytest.raises(RUNTIME_SYNC.RuntimeSyncError, match="I1 blob identity"):
        RUNTIME_SYNC.audit_production_repository_authority(config, payload)


def test_successor_is_exact_evt052_and_preserves_current_scientific_truth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _bound_config()
    predecessor = _synthetic_predecessor(config, monkeypatch)
    successors = RUNTIME_SYNC.build_successors(
        config,
        predecessor,
        "2026-08-13T20:00:00+08:00",
    )
    RUNTIME_SYNC.validate_successors(config, predecessor, successors)
    status, manifest, events = RUNTIME_SYNC._parse_runtime(
        {name: successors[name] for name in RUNTIME_SYNC.MUTABLE_NAMES}
    )
    assert len(events) == 52
    event = events[-1]
    assert event["event_id"] == "A1-EVT-052"
    assert event["decision_id"] == "V3-DEC-021"
    assert event["registered_artifacts"] == []
    assert event["registered_artifact_count"] == 0
    assert event["preflight_executed"] is False
    assert event["scientific_state_changed"] is False
    assert event["qualification_changed"] is False
    assert status["qualified_ordinary_studies"] == 1
    assert status["qualified_a1_studies"] == 1
    assert status["qualified_a2_dense_studies"] == 0
    assert status["canonical_intervention_record_count"] == 6547
    assert status["canonical_record_count"] == 6547
    assert status["qualified"] is False
    assert status["training_allowed"] is False
    assert status["gpu_work_allowed"] is False
    assert status["model_selection_allowed"] is False
    assert status["next_phase_authorized"] is False
    assert status["scientific_claim_status"] == "NOT_ESTABLISHED"
    assert status["gse256185_contribution"] == {
        "ordinary": 0,
        "a1": 0,
        "true_a2": 0,
        "canonical_records": 0,
    }
    assert status["gse256185_public_preflight_status"] == "AUTHORIZED_NOT_RUN"
    assert manifest["registered_artifact_count"] == 4
    assert len(manifest["outputs"]) == 224
    assert manifest["outputs"][:220] == RUNTIME_SYNC.load_json(
        predecessor["RUN_MANIFEST.json"], label="old manifest"
    )["outputs"]
    assert status["existing_status_field"] == "PRESERVED"
    assert manifest["existing_manifest_field"] == "PRESERVED"
    sync = RUNTIME_SYNC.load_json(
        successors[config["runtime"]["sync_name"]], label="sync"
    )
    assert sync["registered_artifacts"] == []
    assert sync["registered_artifact_count"] == 0
    assert sync["manifest_registered_artifact_count_before"] == 4
    assert sync["manifest_registered_artifact_count_after"] == 4
    assert sync["frozen_outer_truth"] == RUNTIME_SYNC.FROZEN_OUTER_TRUTH


def test_fresh_predecessor_drift_stops_before_prepared_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, predecessor, run_root, prepared = _make_context(tmp_path, monkeypatch)
    (run_root / "STATUS.json").write_bytes(predecessor["STATUS.json"] + b"drift")
    with pytest.raises(RUNTIME_SYNC.PredecessorError, match="identity drift"):
        RUNTIME_SYNC.prepare_runtime_sync(
            prepared_directory=prepared,
            recorded_at="2026-08-13T20:00:00+08:00",
            production=False,
            config_override=config,
            run_root_override=run_root,
        )
    assert not prepared.exists()


def test_partial_publication_retries_as_exact_prefix_and_validates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, predecessor, run_root, prepared = _make_context(tmp_path, monkeypatch)
    result = RUNTIME_SYNC.prepare_runtime_sync(
        prepared_directory=prepared,
        recorded_at="2026-08-13T20:00:00+08:00",
        production=False,
        config_override=config,
        run_root_override=run_root,
    )
    assert result["status"] == "PREPARED_NOT_PUBLISHED"
    assert result["event_id"] == "A1-EVT-052"
    assert result["manifest_output_transition"] == "220_TO_224"
    assert result["new_registered_artifact_count"] == 0
    assert len(list(prepared.iterdir())) == 7

    def fail_before_manifest(stage: str) -> None:
        if stage == "before_replace:RUN_MANIFEST.json":
            raise RuntimeError("simulated interruption")

    with pytest.raises(RUNTIME_SYNC.PublicationError, match="retry"):
        RUNTIME_SYNC.publish_prepared(
            prepared_directory=prepared,
            production=False,
            config_override=config,
            run_root_override=run_root,
            fault_injector=fail_before_manifest,
        )
    partial = _read_runtime(run_root)
    assert partial["STATUS.json"] == (prepared / "STATUS.json").read_bytes()
    assert partial["RUN_MANIFEST.json"] == predecessor["RUN_MANIFEST.json"]
    assert partial["EVENT_LOG.jsonl"] == predecessor["EVENT_LOG.jsonl"]

    published = RUNTIME_SYNC.publish_prepared(
        prepared_directory=prepared,
        production=False,
        config_override=config,
        run_root_override=run_root,
    )
    assert published["status"] == "PUBLISHED_VERIFIED"
    assert published["event_id"] == "A1-EVT-052"
    assert published["reused"] is False
    validated = RUNTIME_SYNC.validate_published(
        prepared_directory=prepared,
        production=False,
        config_override=config,
        run_root_override=run_root,
    )
    assert validated == {"status": "PUBLISHED_VERIFIED", "event_id": "A1-EVT-052"}
    final = _read_runtime(run_root)
    events = RUNTIME_SYNC.load_events(final["EVENT_LOG.jsonl"], label="final events")
    manifest = RUNTIME_SYNC.load_json(final["RUN_MANIFEST.json"], label="final manifest")
    assert events[-1]["event_id"] == "A1-EVT-052"
    assert len(events) == 52
    assert len(manifest["outputs"]) == 224
    assert manifest["registered_artifact_count"] == 4
    for name in config["runtime"]["immutable_publish_order"]:
        assert (run_root / name).read_bytes() == (prepared / name).read_bytes()


def test_immutable_partial_temp_write_failure_leaves_final_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "run" / "immutable.json"

    def injected_partial_write(temp_path: Path, payload: bytes) -> None:
        temp_path.write_bytes(payload[:5])
        raise OSError("injected partial write")

    monkeypatch.setattr(RUNTIME_SYNC, "_write_temp_payload", injected_partial_write)
    with pytest.raises(RUNTIME_SYNC.PublicationError, match="cannot create"):
        RUNTIME_SYNC._write_immutable_once(target, b"complete immutable payload\n")
    assert not target.exists()
    assert list(target.parent.iterdir()) == []


def test_immutable_exclusive_publish_accepts_exact_existing_and_rejects_drift(
    tmp_path: Path,
) -> None:
    target = tmp_path / "run" / "immutable.json"
    payload = b"complete immutable payload\n"
    assert RUNTIME_SYNC._write_immutable_once(target, payload) == "CREATED"
    assert target.read_bytes() == payload
    assert RUNTIME_SYNC._write_immutable_once(target, payload) == "EXISTING_EXACT"
    with pytest.raises(RUNTIME_SYNC.PublicationError, match="differs"):
        RUNTIME_SYNC._write_immutable_once(target, b"different payload\n")
    assert target.read_bytes() == payload
    assert [item.name for item in target.parent.iterdir()] == [target.name]
