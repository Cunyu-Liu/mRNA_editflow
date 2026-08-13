from __future__ import annotations

import copy
import hashlib
import importlib.util
from pathlib import Path
from typing import Any

import pytest


STAGING_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = STAGING_ROOT / "configs/route_a_v3_dec020_authority_runtime_sync_v1.json"
SCRIPT_PATH = STAGING_ROOT / "scripts/route_a_v3/dec020_authority_runtime_sync.py"
SPEC = importlib.util.spec_from_file_location("dec020_authority_runtime_sync", SCRIPT_PATH)
assert SPEC and SPEC.loader
runtime_sync = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runtime_sync)


def read_disk_config() -> dict[str, Any]:
    return runtime_sync.load_config(CONFIG_PATH, require_bound=False)


def unknown_i_config() -> dict[str, Any]:
    config = copy.deepcopy(read_disk_config())
    for key in (
        "status",
        "implementation_commit",
        "implementation_script_sha256",
        "implementation_test_sha256",
    ):
        config["implementation_binding"][key] = runtime_sync.UNKNOWN
    return config


def bound_config() -> dict[str, Any]:
    config = unknown_i_config()
    config["implementation_binding"].update(
        {
            "status": "BOUND",
            "implementation_commit": "1" * 40,
            "implementation_script_sha256": "2" * 64,
            "implementation_test_sha256": "3" * 64,
        }
    )
    return config


def predecessor_payloads(config: dict[str, Any]) -> dict[str, bytes]:
    scientific = config["successor_scientific_state"]
    runtime_science = {
        key: copy.deepcopy(scientific[key]) for key in runtime_sync.RUNTIME_SCIENTIFIC_KEYS
    }
    previous_decisions = ["V3-DEC-017", "V3-DEC-018", "V3-DEC-019"]
    previous_contract_authority = {
        "decision_id": "V3-DEC-019",
        "authority_commit": "8" * 40,
        "scope": "HISTORICAL_PRE_DEC020_CONTRACT_AUTHORITY",
    }
    status = {
        **runtime_science,
        **copy.deepcopy(config["outer_a1_state"]),
        "claim_status": "NOT_ESTABLISHED",
        "canonical_record_count": 0,
        "updated_at": "2026-08-13T01:01:16+08:00",
        "active_amendment_decision_ids": previous_decisions,
        "current_contract_authority": previous_contract_authority,
        "historical_status_field": "PRESERVED",
    }
    manifest = {
        **runtime_science,
        "run_status": "IN_PROGRESS",
        "evidence_status": "BLOCKED_PENDING_PUBLIC_EVIDENCE",
        "claim_status": "NOT_ESTABLISHED",
        "canonical_record_count": 0,
        "active_authority_commit": "9" * 40,
        "active_amendment_decision_ids": previous_decisions,
        "current_contract_authority": previous_contract_authority,
        "registered_artifact_count": 1,
        "outputs": [
            {
                "absolute_path": f"/existing/{index:03d}",
                "artifact_type": f"EXISTING_{index:03d}",
                "bytes": index,
                "sha256": f"{index:064x}",
            }
            for index in range(208)
        ],
        "historical_manifest_field": "PRESERVED",
    }
    events = [
        {
            "event_id": f"A1-EVT-{index:03d}",
            "at": "2026-08-12T00:00:00+08:00",
            "event": "HISTORICAL",
        }
        for index in range(1, 49)
    ]
    events.append(
        {
            "event_id": "A1-EVT-049",
            "at": "2026-08-13T01:01:16+08:00",
            "event": "HISTORICAL_EVT049",
            "decision_id": "V3-DEC-019",
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
    tail_payload = runtime_sync.compact_json_line(events[-1])
    config["runtime"]["predecessor_tail"].update(
        {"bytes": len(tail_payload), "sha256": runtime_sync.sha256(tail_payload)}
    )
    return payloads


def make_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, Any], dict[str, bytes], Path, Path]:
    config = bound_config()
    run_root = tmp_path / "run"
    allowed_root = tmp_path / "prepared-root"
    prepared = allowed_root / "evt050-job"
    run_root.mkdir()
    allowed_root.mkdir()
    config["runtime"]["run_root"] = str(run_root)
    config["runtime"]["allowed_prepared_root"] = str(allowed_root)
    predecessor = predecessor_payloads(config)
    # The production config freezes real predecessor sizes/digests. Synthetic
    # transaction fixtures replace only those identities and paths, so bypass
    # static config revalidation while exercising the real predecessor checks.
    monkeypatch.setattr(runtime_sync, "validate_bound_config", lambda _config: None)
    for name, payload in predecessor.items():
        (run_root / name).write_bytes(payload)
    return config, predecessor, run_root, prepared


def read_runtime(run_root: Path) -> dict[str, bytes]:
    return {name: (run_root / name).read_bytes() for name in runtime_sync.MUTABLE_NAMES}


def install_fake_repository_authority(
    monkeypatch: pytest.MonkeyPatch,
    config: dict[str, Any],
    *,
    mode: str = "valid",
) -> bytes:
    authority_commit = runtime_sync.AUTHORITY_COMMIT
    implementation_commit = "1" * 40
    binding_commit = "b" * 40
    script_payload = b"DEC020 authority runtime publisher\n"
    test_payload = b"DEC020 authority runtime focused test\n"
    binding = config["implementation_binding"]
    binding.update(
        {
            "status": "BOUND",
            "implementation_commit": implementation_commit,
            "implementation_script_sha256": hashlib.sha256(script_payload).hexdigest(),
            "implementation_test_sha256": hashlib.sha256(test_payload).hexdigest(),
        }
    )
    config_payload = runtime_sync.json_bytes(config)
    i_payload = runtime_sync.json_bytes(runtime_sync.expected_unknown_i_config(config))

    authority_payloads: dict[str, bytes] = {}
    digest_overrides: dict[bytes, str] = {}
    for index, item in enumerate(runtime_sync.AUTHORITY_FILES):
        seed = f"DEC020 authority blob {index}\n".encode()
        payload = (seed * (item["bytes"] // len(seed) + 1))[: item["bytes"]]
        authority_payloads[item["path"]] = payload
        digest_overrides[payload] = item["sha256"]
    real_sha256 = runtime_sync.sha256

    def fake_sha256(payload: bytes) -> str:
        return digest_overrides.get(payload, real_sha256(payload))

    changed_paths = {
        authority_commit: list(runtime_sync.AUTHORITY_PATHS),
        implementation_commit: list(runtime_sync.IMPLEMENTATION_PATHS),
        binding_commit: [runtime_sync.CONFIG_REPO_PATH],
    }
    if mode in {"A_paths", "I_paths", "B_paths"}:
        commit = {
            "A_paths": authority_commit,
            "I_paths": implementation_commit,
            "B_paths": binding_commit,
        }[mode]
        changed_paths[commit].append("unexpected/path")

    blobs: dict[tuple[str, str], bytes] = {
        **{
            (authority_commit, path): payload
            for path, payload in authority_payloads.items()
        },
        **{
            (binding_commit, path): payload for path, payload in authority_payloads.items()
        },
        (implementation_commit, runtime_sync.CONFIG_REPO_PATH): i_payload,
        (implementation_commit, runtime_sync.SCRIPT_REPO_PATH): script_payload,
        (implementation_commit, runtime_sync.TEST_REPO_PATH): test_payload,
        (binding_commit, runtime_sync.CONFIG_REPO_PATH): config_payload,
        (binding_commit, runtime_sync.SCRIPT_REPO_PATH): script_payload,
        (binding_commit, runtime_sync.TEST_REPO_PATH): test_payload,
    }
    if mode == "A_blob":
        first = runtime_sync.AUTHORITY_PATHS[0]
        blobs[(authority_commit, first)] = blobs[(authority_commit, first)][:-1] + b"x"
    if mode == "current_A_blob":
        first = runtime_sync.AUTHORITY_PATHS[0]
        blobs[(binding_commit, first)] = blobs[(binding_commit, first)][:-1] + b"x"
    if mode == "I_config":
        blobs[(implementation_commit, runtime_sync.CONFIG_REPO_PATH)] += b"drift"
    if mode == "script_hash":
        blobs[(implementation_commit, runtime_sync.SCRIPT_REPO_PATH)] += b"drift"

    def fake_git(_repo: Path, *arguments: str) -> bytes:
        branch = config["repository_authority"]["branch"]
        if arguments == ("rev-parse", "HEAD"):
            return f"{binding_commit}\n".encode()
        if arguments == ("rev-parse", "@{upstream}"):
            observed = "c" * 40 if mode == "upstream" else binding_commit
            return f"{observed}\n".encode()
        if arguments == (
            "rev-parse",
            "--verify",
            f"refs/remotes/origin/{branch}",
        ):
            observed = "d" * 40 if mode == "origin" else binding_commit
            return f"{observed}\n".encode()
        if arguments == ("rev-parse", "--abbrev-ref", "HEAD"):
            return f"{branch}\n".encode()
        if arguments == ("rev-parse", "--abbrev-ref", "@{upstream}"):
            return f"origin/{branch}\n".encode()
        if arguments == ("status", "--porcelain=v1", "--untracked-files=all"):
            return b" M dirty\n" if mode == "dirty" else b""
        if arguments == ("rev-parse", f"{binding_commit}^"):
            parent = "e" * 40 if mode == "B_parent" else implementation_commit
            return f"{parent}\n".encode()
        if arguments == ("rev-parse", f"{implementation_commit}^"):
            parent = "e" * 40 if mode == "I_parent" else authority_commit
            return f"{parent}\n".encode()
        if arguments == ("rev-parse", f"{authority_commit}^"):
            parent = "e" * 40 if mode == "A_parent" else runtime_sync.AUTHORITY_PARENT
            return f"{parent}\n".encode()
        if arguments[:4] == (
            "diff-tree",
            "--no-commit-id",
            "--name-only",
            "-r",
        ):
            return ("\n".join(changed_paths[arguments[4]]) + "\n").encode()
        if arguments[0] == "show":
            commit, path = arguments[1].split(":", 1)
            return blobs[(commit, path)]
        raise AssertionError(arguments)

    worktree_payloads = {
        **authority_payloads,
        runtime_sync.CONFIG_REPO_PATH: config_payload,
        runtime_sync.SCRIPT_REPO_PATH: script_payload,
        runtime_sync.TEST_REPO_PATH: test_payload,
    }
    if mode == "current_A_blob":
        first = runtime_sync.AUTHORITY_PATHS[0]
        worktree_payloads[first] = worktree_payloads[first][:-1] + b"x"
    monkeypatch.setattr(runtime_sync, "sha256", fake_sha256)
    monkeypatch.setattr(runtime_sync, "_run_git", fake_git)
    monkeypatch.setattr(
        runtime_sync,
        "_read_repo_file",
        lambda _repo, path: worktree_payloads[path],
    )
    return config_payload


def test_static_config_core_and_frozen_candidates_are_exact() -> None:
    config = read_disk_config()
    assert CONFIG_PATH.read_bytes() == runtime_sync.json_bytes(config)
    assert runtime_sync._binding_values_are_unknown(config["implementation_binding"])
    assert config["implementation_binding"]["compiled_core_sha256"] == (
        runtime_sync.compiled_core_sha256(config)
    )
    assert config["repository_authority"]["authority_commit"] == runtime_sync.AUTHORITY_COMMIT
    assert config["repository_authority"]["authority_expected_parent"] == (
        runtime_sync.AUTHORITY_PARENT
    )
    assert config["repository_authority"]["authority_files"] == runtime_sync.AUTHORITY_FILES
    assert config["registered_artifacts"] == []
    assert config["runtime"]["predecessor_event_count"] == 49
    assert config["runtime"]["successor_event_count"] == 50
    assert config["runtime"]["predecessor_manifest_output_count"] == 208
    assert config["runtime"]["successor_manifest_output_count"] == 212
    assert config["runtime"]["output_delta_count"] == 4
    assert config["runtime"]["predecessor_candidate_status"].startswith("FROZEN_")
    assert config["runtime"]["fresh_production_validation_required"] is True


def test_config_only_bound_form_loads_and_partial_binding_is_rejected(tmp_path: Path) -> None:
    bound = bound_config()
    bound_path = tmp_path / CONFIG_PATH.name
    bound_path.write_bytes(runtime_sync.json_bytes(bound))
    assert runtime_sync.load_bound_config(bound_path) == bound
    partial = unknown_i_config()
    partial["implementation_binding"]["status"] = "BOUND"
    with pytest.raises(runtime_sync.BindingError, match="partially known"):
        runtime_sync.validate_static_config(partial)


def test_unknown_i_stops_before_git_runtime_or_prepared_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        calls.append("external")
        raise AssertionError("external I/O occurred")

    monkeypatch.setattr(runtime_sync, "audit_production_repository_authority", forbidden)
    monkeypatch.setattr(runtime_sync, "_read_runtime", forbidden)
    monkeypatch.setattr(runtime_sync, "_read_prepared", forbidden)
    with pytest.raises(runtime_sync.BindingError, match="not BOUND"):
        runtime_sync._context(
            CONFIG_PATH,
            production=False,
            config_override=unknown_i_config(),
        )
    assert calls == []


def test_production_authority_is_first_before_runtime_or_prepared_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = bound_config()
    payload = runtime_sync.json_bytes(config)
    calls: list[str] = []
    monkeypatch.setattr(
        runtime_sync,
        "_load_config_payload",
        lambda *_args, **_kwargs: (copy.deepcopy(config), payload),
    )

    def stop_at_authority(*_args: Any, **_kwargs: Any) -> Any:
        calls.append("authority")
        raise runtime_sync.AuthorityError("ordering proof")

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        calls.append("runtime-or-prepared")
        raise AssertionError("runtime/prepared I/O occurred before Git authority")

    monkeypatch.setattr(runtime_sync, "audit_production_repository_authority", stop_at_authority)
    monkeypatch.setattr(runtime_sync, "_read_runtime", forbidden)
    monkeypatch.setattr(runtime_sync, "_read_prepared", forbidden)
    monkeypatch.setattr(runtime_sync, "_write_prepared", forbidden)
    with pytest.raises(runtime_sync.AuthorityError, match="ordering proof"):
        runtime_sync.prepare_runtime_sync(
            prepared_directory=Path(config["runtime"]["allowed_prepared_root"]) / "proof",
            recorded_at="2026-08-13T02:00:00+08:00",
            production=True,
        )
    assert calls == ["authority"]


def test_production_authority_accepts_exact_a_i_b(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = bound_config()
    payload = install_fake_repository_authority(monkeypatch, config)
    result = runtime_sync.audit_production_repository_authority(config, payload)
    assert result == {
        "status": "PASS_EXACT_A_TO_I_TO_CONFIG_ONLY_B",
        "authority_commit": runtime_sync.AUTHORITY_COMMIT,
        "implementation_commit": "1" * 40,
        "binding_commit": "b" * 40,
        "head_commit": "b" * 40,
        "upstream_head_commit": "b" * 40,
        "origin_branch_head_commit": "b" * 40,
        "authority_blob_count": 14,
        "worktree_and_index_clean": True,
    }


@pytest.mark.parametrize(
    "mode",
    [
        "dirty",
        "upstream",
        "origin",
        "A_parent",
        "I_parent",
        "B_parent",
        "A_paths",
        "I_paths",
        "B_paths",
        "A_blob",
        "current_A_blob",
        "I_config",
        "script_hash",
    ],
)
def test_production_authority_drift_is_rejected(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    config = bound_config()
    payload = install_fake_repository_authority(monkeypatch, config, mode=mode)
    with pytest.raises(runtime_sync.RuntimeSyncError):
        runtime_sync.audit_production_repository_authority(config, payload)


def test_predecessor_drift_stops_with_zero_prepared_or_runtime_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _predecessor, run_root, prepared = make_context(tmp_path, monkeypatch)
    status = runtime_sync.load_json((run_root / "STATUS.json").read_bytes(), label="status")
    status["drift"] = True
    (run_root / "STATUS.json").write_bytes(runtime_sync.json_bytes(status))
    before = read_runtime(run_root)
    with pytest.raises(runtime_sync.PredecessorError, match="identity drift"):
        runtime_sync.prepare_runtime_sync(
            prepared_directory=prepared,
            recorded_at="2026-08-13T02:00:00+08:00",
            production=False,
            config_override=config,
            run_root_override=run_root,
        )
    assert read_runtime(run_root) == before
    assert not prepared.exists()


def test_normal_authority_only_transaction_adds_exact4_and_no_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, predecessor, run_root, prepared = make_context(tmp_path, monkeypatch)
    result = runtime_sync.prepare_runtime_sync(
        prepared_directory=prepared,
        recorded_at="2026-08-13T02:00:00+08:00",
        production=False,
        config_override=config,
        run_root_override=run_root,
    )
    assert result == {
        "status": "PREPARED_NOT_PUBLISHED",
        "event_id": "A1-EVT-050",
        "prepared_directory": str(prepared),
        "prepared_member_count": 7,
        "manifest_output_transition": "208_TO_212",
        "new_runtime_output_count": 4,
        "registered_artifact_count": 0,
    }
    assert {item.name for item in prepared.iterdir()} == {
        *runtime_sync.MUTABLE_NAMES,
        *config["runtime"]["immutable_publish_order"],
    }
    published = runtime_sync.publish_prepared(
        prepared_directory=prepared,
        production=False,
        config_override=config,
        run_root_override=run_root,
    )
    assert published["status"] == "PUBLISHED_VERIFIED"
    old_status, old_manifest, old_events = runtime_sync._parse_runtime(predecessor)
    status, manifest, events = runtime_sync._parse_runtime(read_runtime(run_root))
    assert len(events) == 50
    assert events[:-1] == old_events
    event = events[-1]
    assert event["event_id"] == "A1-EVT-050"
    assert event["decision_id"] == "V3-DEC-020"
    assert event["scientific_state_changed"] is False
    assert event["registered_artifacts"] == []
    assert event["registered_artifact_count"] == 0
    assert event["access_boundary"] == runtime_sync.ACCESS_BOUNDARY
    assert len(manifest["outputs"]) == 212
    assert manifest["outputs"][:208] == old_manifest["outputs"]
    assert [Path(item["absolute_path"]).name for item in manifest["outputs"][-4:]] == (
        config["runtime"]["immutable_publish_order"]
    )
    assert status["active_amendment_decision_ids"] == runtime_sync.ACTIVE_DECISION_IDS
    assert manifest["active_amendment_decision_ids"] == runtime_sync.ACTIVE_DECISION_IDS
    assert status["current_contract_authority"] == runtime_sync.CURRENT_CONTRACT_AUTHORITY
    assert manifest["current_contract_authority"] == runtime_sync.CURRENT_CONTRACT_AUTHORITY
    assert manifest["active_authority_commit"] == old_manifest["active_authority_commit"]
    assert status["historical_status_field"] == old_status["historical_status_field"]
    sync = runtime_sync.load_json(
        (run_root / config["runtime"]["sync_name"]).read_bytes(), label="sync"
    )
    assert sync["registered_artifacts"] == []
    assert sync["registered_artifact_count"] == 0
    assert sync["scientific_state_changed"] is False
    assert sync["access_boundary"] == runtime_sync.ACCESS_BOUNDARY
    assert sync["historical_outer_runtime_authority"]["active_authority_commit"] == (
        old_manifest["active_authority_commit"]
    )
    assert runtime_sync.validate_published(
        prepared_directory=prepared,
        production=False,
        config_override=config,
        run_root_override=run_root,
    ) == {"status": "PUBLISHED_VERIFIED", "event_id": "A1-EVT-050"}


def test_immutables_first_mutable_prefix_recovery_and_idempotence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, predecessor, run_root, prepared = make_context(tmp_path, monkeypatch)
    runtime_sync.prepare_runtime_sync(
        prepared_directory=prepared,
        recorded_at="2026-08-13T02:00:00+08:00",
        production=False,
        config_override=config,
        run_root_override=run_root,
    )

    def fail_before_manifest(point: str) -> None:
        if point == "before_replace:RUN_MANIFEST.json":
            raise OSError("injected supported-prefix interruption")

    with pytest.raises(runtime_sync.PublicationError, match="not committed"):
        runtime_sync.publish_prepared(
            prepared_directory=prepared,
            production=False,
            config_override=config,
            run_root_override=run_root,
            fault_injector=fail_before_manifest,
        )
    partial = read_runtime(run_root)
    assert partial["STATUS.json"] != predecessor["STATUS.json"]
    assert partial["RUN_MANIFEST.json"] == predecessor["RUN_MANIFEST.json"]
    assert partial["EVENT_LOG.jsonl"] == predecessor["EVENT_LOG.jsonl"]
    assert all(
        (run_root / name).read_bytes() == (prepared / name).read_bytes()
        for name in config["runtime"]["immutable_publish_order"]
    )
    recovered = runtime_sync.publish_prepared(
        prepared_directory=prepared,
        production=False,
        config_override=config,
        run_root_override=run_root,
    )
    assert recovered["status"] == "PUBLISHED_VERIFIED"
    reused = runtime_sync.publish_prepared(
        prepared_directory=prepared,
        production=False,
        config_override=config,
        run_root_override=run_root,
    )
    assert reused["status"] == "PUBLISHED_VERIFIED"
    assert reused["reused"] is True
    assert len(runtime_sync.load_events(read_runtime(run_root)["EVENT_LOG.jsonl"], label="events")) == 50


def test_no_registered_private_sealed_row_sequence_or_effect_payload_surface() -> None:
    config = read_disk_config()
    assert config["registered_artifacts"] == []
    assert config["access_boundary"] == runtime_sync.ACCESS_BOUNDARY
    assert all(value == 0 for key, value in config["access_boundary"].items() if key.endswith("_count"))
    assert config["access_boundary"]["restricted_or_sealed_path_accessed"] is False
    assert config["access_boundary"]["gse246381_contact"] is False
    source = SCRIPT_PATH.read_text()
    assert "validate_registered_artifacts" not in source
    assert "registered artifact bytes" not in source
    assert "/restricted/" not in source
    assert "/sealed_external/" not in source
    forbidden_true_science_flag = "scientific_state_" + 'changed\": True'
    assert forbidden_true_science_flag not in source
