from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest


STAGING_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = STAGING_ROOT / "configs/route_a_v3_dec023_authority_runtime_sync_v1.json"
SCRIPT_PATH = STAGING_ROOT / "scripts/route_a_v3/dec023_authority_runtime_sync.py"
SPEC = importlib.util.spec_from_file_location("dec023_authority_runtime_sync", SCRIPT_PATH)
assert SPEC and SPEC.loader
SYNC = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SYNC)


def disk_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def clean_i_config() -> dict[str, Any]:
    config = SYNC.expected_unknown_i_config(disk_config())
    SYNC.validate_static_config(config)
    assert SYNC._implementation_binding_state(config["implementation_binding"]) == (
        "UNKNOWN"
    )
    return config


def bind_authority(config: dict[str, Any]) -> None:
    authority = config["repository_authority"]
    authority.update(
        {
            "authority_binding_status": "FROZEN_BOUND_EXACT10",
            "authority_commit": "a" * 40,
            "authority_expected_parent": "b" * 40,
        }
    )
    for index, item in enumerate(authority["authority_files"], start=1):
        item["bytes"] = 1000 + index
        item["sha256"] = f"{index:064x}"


def predecessor_fixture(config: dict[str, Any]) -> dict[str, bytes]:
    fields = SYNC._outer_runtime_fields(config)
    prior_decisions = [
        "V3-DEC-017",
        "V3-DEC-018",
        "V3-DEC-019",
        "V3-DEC-020",
        "V3-DEC-021",
        "V3-DEC-022",
    ]
    status = {
        **copy.deepcopy(fields),
        "updated_at": "2026-08-14T00:01:00+08:00",
        "active_amendment_decision_ids": prior_decisions,
        "existing_status_field": "PRESERVE_EXACTLY",
    }
    manifest = {
        **copy.deepcopy(fields),
        "active_authority_commit": "c" * 40,
        "active_amendment_decision_ids": prior_decisions,
        "registered_artifact_count": 6,
        "outputs": [
            {
                "absolute_path": f"/existing/{index:03d}.json",
                "artifact_type": "EXISTING_FIXTURE",
                "bytes": index + 1,
                "sha256": f"{index + 1:064x}",
            }
            for index in range(238)
        ],
        "existing_manifest_field": {"preserve": True},
    }
    events = [
        {
            "event_id": f"A1-EVT-{index:03d}",
            "at": "2026-08-13T00:00:00+08:00",
            "event": "HISTORICAL_FIXTURE",
        }
        for index in range(1, 55)
    ]
    events.append(
        {
            "event_id": "A1-EVT-055",
            "at": "2026-08-14T00:01:00+08:00",
            "event": "GSE256185_AGGREGATE_PREFLIGHT_EVIDENCE_REGISTERED",
            "decision_id": "V3-DEC-022",
            "predecessor_event_id": "A1-EVT-054",
        }
    )
    payloads = {
        "STATUS.json": SYNC.json_bytes(status),
        "RUN_MANIFEST.json": SYNC.json_bytes(manifest),
        "EVENT_LOG.jsonl": b"".join(
            SYNC.compact_json_line(event) for event in events
        ),
    }
    runtime = config["runtime"]
    runtime["predecessor_binding_status"] = "FROZEN_BOUND_EVT055"
    for name, payload in payloads.items():
        runtime["predecessor_mutables"][name]["bytes"] = len(payload)
        runtime["predecessor_mutables"][name]["sha256"] = SYNC.sha256(payload)
    tail_payload = SYNC.compact_json_line(events[-1])
    runtime["predecessor_tail"]["bytes"] = len(tail_payload)
    runtime["predecessor_tail"]["sha256"] = SYNC.sha256(tail_payload)
    return payloads


def bound_context() -> tuple[dict[str, Any], dict[str, bytes]]:
    config = disk_config()
    bind_authority(config)
    predecessor = predecessor_fixture(config)
    config["implementation_binding"].update(
        {
            "status": SYNC.BOUND,
            "implementation_commit": "d" * 40,
            "implementation_script_sha256": "e" * 64,
            "implementation_test_sha256": "f" * 64,
        }
    )
    SYNC.validate_bound_config(config)
    return config, predecessor


def write_runtime(run_root: Path, payloads: dict[str, bytes]) -> None:
    run_root.mkdir(parents=True)
    for name in SYNC.MUTABLE_NAMES:
        (run_root / name).write_bytes(payloads[name])


def test_disk_candidate_is_strict_valid_i_or_b_and_normalizes_to_i() -> None:
    config = disk_config()
    SYNC.validate_static_config(config)
    authority = config["repository_authority"]
    assert SYNC._authority_binding_state(authority) == SYNC.BOUND
    assert authority["authority_commit"] == (
        "f7cfff896a1a30d25a3b73ea7f89957d70d95d39"
    )
    assert authority["authority_expected_parent"] == (
        "ae8e730d726754466e5c914d7ff962377607ac50"
    )
    assert [(item["bytes"], item["sha256"]) for item in authority["authority_files"]] == [
        (33739, "df38455904d67f22a2fea1fb08a3314cd4fb120e91ea711427ad1689653ba8ce"),
        (24375, "98de408ec423836efac75bcd75b4fd940e9fbd52a0bf1b3c397ea0c67e548740"),
        (9773, "44622c7f589d841105cb21d0b35219aa9163fe4d54350671106408d4c8439e4a"),
        (7701, "0edfddd90ebea11db1cebb7084bdf28dd8a99426c3956a81cdfd7a4a9ccb12e2"),
        (196200, "fbeef398ff59764375edf9a35c2c35fd3a93db89eb3b06326ec1c73968646eff"),
        (30320, "bb577d4ce7d7dc673f41bb182b7868f66816c15a3ed4235c98e0839292e75d6b"),
        (37941, "8e514512ccd63d87a596231b11183c06765ba50ba3736adc165f141da8fa13d0"),
        (24934, "abebbe62f7b6dbac8e0a7673fc5580a56fd96198a37362ec206519028b457c83"),
        (735770, "a6a71a82e90352c6c9bc02fad95c54436e268d1ee58233093dc8412c5e5739bb"),
        (188940, "a0c9b3cc457011e57046506a02e57a4314a26a8885b3d110f88797a000daca0a"),
    ]
    assert authority["predecessor_implementation_i1"] == {
        "status": "FROZEN_BOUND_EXACT3",
        "commit": "b0afa92eea9718c15a5989cfa67bac57036617d9",
        "expected_parent": "f7cfff896a1a30d25a3b73ea7f89957d70d95d39",
        "exact_changed_paths": SYNC.IMPLEMENTATION_PATHS,
        "blob_sha256_by_path": SYNC.FROZEN_I1_BLOBS,
    }
    assert SYNC._predecessor_binding_state(config["runtime"]) == SYNC.BOUND
    binding = config["implementation_binding"]
    implementation_state = SYNC._implementation_binding_state(binding)
    assert config["runtime"]["predecessor_event_id"] == "A1-EVT-055"
    assert config["runtime"]["predecessor_event_count"] == 55
    assert config["runtime"]["predecessor_manifest_output_count"] == 238
    assert config["runtime"]["predecessor_manifest_registered_artifact_count"] == 6
    assert config["runtime"]["successor_event_id"] == "A1-EVT-056"
    assert config["runtime"]["successor_manifest_output_count"] == 242
    assert {
        name: (
            config["runtime"]["predecessor_mutables"][name]["bytes"],
            config["runtime"]["predecessor_mutables"][name]["sha256"],
        )
        for name in SYNC.MUTABLE_NAMES
    } == {
        "STATUS.json": (
            29057,
            "00876189c12627a1121d7a264927255131023b69fe365a12f4ebf91c2936c578",
        ),
        "RUN_MANIFEST.json": (
            101455,
            "32cef1adeb28832dbaaac29e8a170223e59ec076c6fb5f8a4a265bcd58aea090",
        ),
        "EVENT_LOG.jsonl": (
            121688,
            "b5ecd3e99425e60e1b5fff9e31201582aa0a19470009578b06d2db5d3cfbab45",
        ),
    }
    assert config["runtime"]["predecessor_tail"] == {
        "event_id": "A1-EVT-055",
        "decision_id": "V3-DEC-022",
        "bytes": 4769,
        "sha256": "e83ca46853724c9f9b0e28daa33f99b141affee464d1725b2b23863b49d462e3",
    }
    assert config["registered_artifacts"] == []
    normalized_i = SYNC.expected_unknown_i_config(config)
    SYNC.validate_static_config(normalized_i)
    if implementation_state == "UNKNOWN":
        assert normalized_i == config
        with pytest.raises(SYNC.BindingError, match="implementation remains"):
            SYNC.validate_bound_config(config)
    else:
        assert implementation_state == SYNC.BOUND
        assert binding["status"] == SYNC.BOUND
        assert SYNC.HEX40.fullmatch(binding["implementation_commit"])
        assert binding["implementation_script_sha256"] == SYNC.sha256(
            SCRIPT_PATH.read_bytes()
        )
        assert binding["implementation_test_sha256"] == SYNC.sha256(
            Path(__file__).read_bytes()
        )
        SYNC.validate_bound_config(config)
        restored_b = copy.deepcopy(normalized_i)
        for field in SYNC.UNKNOWN_BINDING_FIELDS:
            restored_b["implementation_binding"][field] = binding[field]
        assert restored_b == config

    synthetic_b = copy.deepcopy(normalized_i)
    synthetic_b["implementation_binding"].update(
        {
            "status": SYNC.BOUND,
            "implementation_commit": "9" * 40,
            "implementation_script_sha256": SYNC.sha256(SCRIPT_PATH.read_bytes()),
            "implementation_test_sha256": SYNC.sha256(Path(__file__).read_bytes()),
        }
    )
    SYNC.validate_bound_config(synthetic_b)
    assert SYNC.expected_unknown_i_config(synthetic_b) == normalized_i


@pytest.mark.parametrize("group", ["authority", "predecessor", "implementation"])
def test_partial_group_is_rejected(group: str) -> None:
    config = clean_i_config()
    if group == "authority":
        authority = config["repository_authority"]
        authority["authority_binding_status"] = SYNC.UNKNOWN
        authority["authority_commit"] = SYNC.UNKNOWN
        authority["authority_expected_parent"] = SYNC.UNKNOWN
        for item in authority["authority_files"]:
            item["bytes"] = SYNC.UNKNOWN
            item["sha256"] = SYNC.UNKNOWN
        config["repository_authority"]["authority_binding_status"] = (
            "FROZEN_BOUND_EXACT10"
        )
    elif group == "predecessor":
        runtime = config["runtime"]
        runtime["predecessor_binding_status"] = SYNC.UNKNOWN
        for item in runtime["predecessor_mutables"].values():
            item["bytes"] = SYNC.UNKNOWN
            item["sha256"] = SYNC.UNKNOWN
        runtime["predecessor_tail"]["bytes"] = SYNC.UNKNOWN
        runtime["predecessor_tail"]["sha256"] = SYNC.UNKNOWN
        config["runtime"]["predecessor_binding_status"] = "FROZEN_BOUND_EVT055"
    else:
        config["implementation_binding"]["status"] = SYNC.BOUND
    with pytest.raises(SYNC.BindingError, match="partially known"):
        SYNC.validate_static_config(config)


def test_any_unknown_stops_before_prepared_or_runtime_io(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = clean_i_config()
    touched: list[str] = []

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        touched.append("I/O")
        raise AssertionError("prepared/runtime I/O reached")

    monkeypatch.setattr(SYNC, "_prepared_path", forbidden)
    monkeypatch.setattr(SYNC, "_read_runtime", forbidden)
    monkeypatch.setattr(SYNC, "_write_prepared", forbidden)
    with pytest.raises(SYNC.BindingError):
        SYNC.prepare_runtime_sync(
            prepared_directory=tmp_path / "must-not-exist",
            recorded_at="2026-08-14T00:05:00+08:00",
            production=False,
            config_override=config,
            run_root_override=tmp_path / "must-not-open",
        )
    assert touched == []
    assert not (tmp_path / "must-not-exist").exists()
    assert not (tmp_path / "must-not-open").exists()


def test_stale_copied_producer_is_rejected_before_git_commands(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _predecessor = bound_context()
    stale_script = tmp_path / "stale-copy" / SCRIPT_PATH.name
    stale_script.parent.mkdir()
    stale_script.write_bytes(SCRIPT_PATH.read_bytes())
    stale_spec = importlib.util.spec_from_file_location(
        "dec023_authority_runtime_sync_stale_copy", stale_script
    )
    assert stale_spec and stale_spec.loader
    stale_sync = importlib.util.module_from_spec(stale_spec)
    stale_spec.loader.exec_module(stale_sync)
    touched: list[str] = []

    def forbidden_git(*_args: Any, **_kwargs: Any) -> bytes:
        touched.append("git")
        raise AssertionError("git I/O should not run for a copied producer")

    def forbidden_prepared(*_args: Any, **_kwargs: Any) -> Any:
        touched.append("prepared")
        raise AssertionError("prepared I/O should not run for a copied producer")

    def forbidden_runtime(*_args: Any, **_kwargs: Any) -> Any:
        touched.append("runtime")
        raise AssertionError("runtime I/O should not run for a copied producer")

    monkeypatch.setattr(stale_sync, "_run_git", forbidden_git)
    monkeypatch.setattr(stale_sync, "_prepared_path", forbidden_prepared)
    monkeypatch.setattr(stale_sync, "_read_runtime", forbidden_runtime)
    with pytest.raises(stale_sync.AuthorityError, match="executing producer"):
        stale_sync.audit_production_repository_authority(
            config, stale_sync.json_bytes(config)
        )
    assert touched == []


def test_repository_audit_proves_a_i1_i2_b2_and_frozen_i1_blobs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = clean_i_config()
    authority = config["repository_authority"]
    frozen_i1 = authority["predecessor_implementation_i1"]
    authority_commit = authority["authority_commit"]
    i1_commit = frozen_i1["commit"]
    i2_commit = "2" * 40
    b2_commit = "3" * 40
    script_payload = b"dynamic I2 implementation script\n"
    test_payload = b"dynamic I2 focused test\n"
    config["implementation_binding"].update(
        {
            "status": SYNC.BOUND,
            "implementation_commit": i2_commit,
            "implementation_script_sha256": SYNC.sha256(script_payload),
            "implementation_test_sha256": SYNC.sha256(test_payload),
        }
    )
    SYNC.validate_bound_config(config)
    b2_config_payload = SYNC.json_bytes(config)
    i2_config_payload = SYNC.json_bytes(SYNC.expected_unknown_i_config(config))
    i1_config_payload = SYNC.json_bytes(
        {
            "implementation_binding": {
                field: SYNC.UNKNOWN for field in SYNC.UNKNOWN_BINDING_FIELDS
            }
        }
    )
    i1_payloads = {
        SYNC.CONFIG_REPO_PATH: i1_config_payload,
        SYNC.SCRIPT_REPO_PATH: b"frozen I1 implementation script\n",
        SYNC.TEST_REPO_PATH: b"frozen I1 focused test\n",
    }
    authority_payloads = {
        item["path"]: bytes([index]) * item["bytes"]
        for index, item in enumerate(authority["authority_files"], start=1)
    }
    frozen_digest_by_payload = {
        payload: frozen_i1["blob_sha256_by_path"][path]
        for path, payload in i1_payloads.items()
    }
    frozen_digest_by_payload.update(
        {
            authority_payloads[item["path"]]: item["sha256"]
            for item in authority["authority_files"]
        }
    )
    real_sha256 = SYNC.sha256

    def fake_sha256(payload: bytes) -> str:
        return frozen_digest_by_payload.get(payload, real_sha256(payload))

    def fake_run_git(_repo_root: Path, *args: str) -> bytes:
        mapping = {
            ("rev-parse", "HEAD"): f"{b2_commit}\n".encode(),
            ("rev-parse", "--abbrev-ref", "HEAD"): f"{SYNC.BRANCH}\n".encode(),
            ("rev-parse", "--abbrev-ref", "@{upstream}"): f"origin/{SYNC.BRANCH}\n".encode(),
            ("rev-parse", "@{upstream}"): f"{b2_commit}\n".encode(),
            (
                "rev-parse",
                "--verify",
                f"refs/remotes/origin/{SYNC.BRANCH}",
            ): f"{b2_commit}\n".encode(),
            ("status", "--porcelain=v1", "--untracked-files=all"): b"",
            ("rev-parse", f"{b2_commit}^"): f"{i2_commit}\n".encode(),
            ("rev-parse", f"{i2_commit}^"): f"{i1_commit}\n".encode(),
            ("rev-parse", f"{i1_commit}^"): f"{authority_commit}\n".encode(),
            (
                "rev-parse",
                f"{authority_commit}^",
            ): f"{authority['authority_expected_parent']}\n".encode(),
        }
        return mapping[args]

    def fake_changed_paths(_repo_root: Path, commit: str) -> list[str]:
        if commit == authority_commit:
            return sorted(SYNC.AUTHORITY_PATHS)
        if commit in {i1_commit, i2_commit}:
            return sorted(SYNC.IMPLEMENTATION_PATHS)
        if commit == b2_commit:
            return [SYNC.CONFIG_REPO_PATH]
        raise AssertionError(commit)

    i1_drift_path: str | None = None

    def fake_git_blob(_repo_root: Path, commit: str, path: str) -> bytes:
        if path in authority_payloads and commit in {
            authority_commit,
            i1_commit,
            i2_commit,
            b2_commit,
        }:
            return authority_payloads[path]
        if commit == i1_commit and path in i1_payloads:
            if path == i1_drift_path:
                return b"drifted frozen I1 blob"
            return i1_payloads[path]
        if commit == i2_commit:
            return {
                SYNC.CONFIG_REPO_PATH: i2_config_payload,
                SYNC.SCRIPT_REPO_PATH: script_payload,
                SYNC.TEST_REPO_PATH: test_payload,
            }[path]
        if commit == b2_commit:
            return {
                SYNC.CONFIG_REPO_PATH: b2_config_payload,
                SYNC.SCRIPT_REPO_PATH: script_payload,
                SYNC.TEST_REPO_PATH: test_payload,
            }[path]
        raise AssertionError((commit, path))

    def fake_repo_file(_repo_root: Path, path: str) -> bytes:
        if path in authority_payloads:
            return authority_payloads[path]
        return {
            SYNC.CONFIG_REPO_PATH: b2_config_payload,
            SYNC.SCRIPT_REPO_PATH: script_payload,
            SYNC.TEST_REPO_PATH: test_payload,
        }[path]

    monkeypatch.setattr(
        SYNC, "__file__", str(SYNC.PRODUCTION_REPO_ROOT / SYNC.SCRIPT_REPO_PATH)
    )
    monkeypatch.setattr(SYNC, "_run_git", fake_run_git)
    monkeypatch.setattr(SYNC, "_changed_paths", fake_changed_paths)
    monkeypatch.setattr(SYNC, "_git_blob", fake_git_blob)
    monkeypatch.setattr(SYNC, "_read_repo_file", fake_repo_file)
    monkeypatch.setattr(SYNC, "sha256", fake_sha256)
    assert SYNC.audit_production_repository_authority(
        config, b2_config_payload
    ) == {
        "status": "PASS_EXACT10_A_TO_FROZEN_EXACT3_I1_TO_EXACT3_I2_TO_CONFIG_ONLY_B2",
        "authority_commit": authority_commit,
        "frozen_i1_commit": i1_commit,
        "implementation_i2_commit": i2_commit,
        "binding_b2_commit": b2_commit,
        "authority_blob_count": 10,
        "worktree_and_index_clean": True,
    }

    i1_drift_path = SYNC.TEST_REPO_PATH
    with pytest.raises(SYNC.AuthorityError, match="frozen I1 blob identity"):
        SYNC.audit_production_repository_authority(config, b2_config_payload)


def test_build_successor_is_exact_238_to_242_no_science_change() -> None:
    config, predecessor = bound_context()
    old_status, old_manifest, _ = SYNC._parse_runtime(predecessor)
    successors = SYNC.build_successors(
        config, predecessor, "2026-08-14T00:05:00+08:00"
    )
    new_status, new_manifest, events = SYNC._parse_runtime(
        {name: successors[name] for name in SYNC.MUTABLE_NAMES}
    )
    assert len(successors) == 7
    assert len(events) == 56
    event = events[-1]
    assert event["event_id"] == "A1-EVT-056"
    assert event["decision_id"] == "V3-DEC-023"
    assert event["registered_artifacts"] == []
    assert event["new_registered_artifact_count"] == 0
    assert event["preflight_executed"] is False
    assert event["scientific_state_changed"] is False
    assert event["qualification_changed"] is False
    assert event["a7_allowed"] is False
    assert len(new_manifest["outputs"]) == 242
    assert new_manifest["outputs"][:238] == old_manifest["outputs"]
    assert new_manifest["registered_artifact_count"] == 6
    assert new_manifest["active_authority_commit"] == old_manifest[
        "active_authority_commit"
    ]
    expected_status = copy.deepcopy(old_status)
    expected_status["updated_at"] = "2026-08-14T00:05:00+08:00"
    sync_payload = successors[config["runtime"]["sync_name"]]
    expected_status.update(
        SYNC._successor_updates(
            config,
            "2026-08-14T00:05:00+08:00",
            SYNC.sha256(sync_payload),
        )
    )
    assert new_status == expected_status
    assert event["frozen_outer_truth"]["current_qualified_counts"] == {
        "ordinary": 1,
        "a1": 1,
        "true_a2": 0,
        "canonical_records": 6547,
    }
    assert all(value == 0 for value in event["access_boundary"].values() if type(value) is int)
    assert event["access_boundary"]["restricted_or_sealed_path_accessed"] is False


def test_predecessor_identity_and_semantic_drift_fail_closed() -> None:
    config, predecessor = bound_context()
    byte_drift = copy.deepcopy(predecessor)
    byte_drift["STATUS.json"] += b" "
    with pytest.raises(SYNC.PredecessorError, match="identity drift"):
        SYNC.build_successors(
            config, byte_drift, "2026-08-14T00:05:00+08:00"
        )

    semantic_drift = copy.deepcopy(predecessor)
    manifest = SYNC.load_json(semantic_drift["RUN_MANIFEST.json"], label="fixture")
    manifest["registered_artifact_count"] = 7
    semantic_drift["RUN_MANIFEST.json"] = SYNC.json_bytes(manifest)
    spec = config["runtime"]["predecessor_mutables"]["RUN_MANIFEST.json"]
    spec["bytes"] = len(semantic_drift["RUN_MANIFEST.json"])
    spec["sha256"] = SYNC.sha256(semantic_drift["RUN_MANIFEST.json"])
    with pytest.raises(SYNC.RuntimeSyncError, match="registered artifact count"):
        SYNC.build_successors(
            config, semantic_drift, "2026-08-14T00:05:00+08:00"
        )


def test_prepare_publish_and_ordered_prefix_recovery(
    tmp_path: Path,
) -> None:
    config, predecessor = bound_context()
    run_root = tmp_path / "runtime"
    allowed_root = tmp_path / "prepared-root"
    prepared = allowed_root / "evt056"
    write_runtime(run_root, predecessor)
    allowed_root.mkdir()
    config["runtime"]["run_root"] = str(run_root)
    config["runtime"]["allowed_prepared_root"] = str(allowed_root)

    result = SYNC.prepare_runtime_sync(
        prepared_directory=prepared,
        recorded_at="2026-08-14T00:05:00+08:00",
        production=False,
        config_override=config,
        run_root_override=run_root,
    )
    assert result["status"] == "PREPARED_NOT_PUBLISHED"
    assert result["manifest_output_transition"] == "238_TO_242"
    assert result["manifest_registered_artifact_transition"] == "6_TO_6"
    assert result["scientific_state_changed"] is False
    assert {item.name for item in prepared.iterdir()} == {
        *SYNC.MUTABLE_NAMES,
        *config["runtime"]["immutable_publish_order"],
    }

    triggered = False

    def interrupt_before_manifest(point: str) -> None:
        nonlocal triggered
        if point == "before_replace:RUN_MANIFEST.json" and not triggered:
            triggered = True
            raise RuntimeError("synthetic interruption")

    with pytest.raises(SYNC.PublicationError, match="retry"):
        SYNC.publish_prepared(
            prepared_directory=prepared,
            production=False,
            config_override=config,
            run_root_override=run_root,
            fault_injector=interrupt_before_manifest,
        )
    assert (run_root / "STATUS.json").read_bytes() == (
        prepared / "STATUS.json"
    ).read_bytes()
    assert (run_root / "RUN_MANIFEST.json").read_bytes() == predecessor[
        "RUN_MANIFEST.json"
    ]
    assert (run_root / "EVENT_LOG.jsonl").read_bytes() == predecessor[
        "EVENT_LOG.jsonl"
    ]

    published = SYNC.publish_prepared(
        prepared_directory=prepared,
        production=False,
        config_override=config,
        run_root_override=run_root,
    )
    assert published["status"] == "PUBLISHED_VERIFIED"
    assert SYNC.validate_published(
        prepared_directory=prepared,
        production=False,
        config_override=config,
        run_root_override=run_root,
    ) == {
        "status": "PUBLISHED_VERIFIED",
        "event_id": "A1-EVT-056",
        "scientific_state_changed": False,
    }


def test_authority_boundaries_cannot_be_relaxed_by_config_only_change() -> None:
    config = disk_config()
    for path, value in (
        (("dec023_authority", "gse261709", "member_payload_read_allowed"), True),
        (("dec023_authority", "gse207584", "sequence_output_allowed"), True),
        (("dec023_authority", "qualification_allowed"), True),
        (("frozen_outer_truth", "training_allowed"), True),
    ):
        drift = copy.deepcopy(config)
        cursor: dict[str, Any] = drift
        for key in path[:-1]:
            cursor = cursor[key]
        cursor[path[-1]] = value
        with pytest.raises(SYNC.RuntimeSyncError):
            SYNC.validate_static_config(drift)
