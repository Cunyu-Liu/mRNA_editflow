from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest


STAGING_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    STAGING_ROOT
    / "configs/route_a_v3_gse232572_qualification_authority_preflight_runtime_sync_v1.json"
)
SCRIPT_PATH = (
    STAGING_ROOT
    / "scripts/route_a_v3/gse232572_qualification_authority_preflight_runtime_sync.py"
)
SPEC = importlib.util.spec_from_file_location("evt049_runtime_sync", SCRIPT_PATH)
assert SPEC and SPEC.loader
runtime_sync = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runtime_sync)


def read_disk_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text())


def normalized_unknown_implementation_config() -> dict[str, Any]:
    config = copy.deepcopy(read_disk_config())
    for key in (
        "status",
        "implementation_commit",
        "implementation_script_sha256",
        "implementation_test_sha256",
    ):
        config["implementation_binding"][key] = runtime_sync.UNKNOWN
    return config


def unknown_l_and_i_config() -> dict[str, Any]:
    config = normalized_unknown_implementation_config()
    ledger = config["repository_authority"]["predecessor_ledger"]
    ledger.update(
        {
            "status": runtime_sync.UNKNOWN,
            "commit": runtime_sync.UNKNOWN,
            "integration_id": runtime_sync.UNKNOWN,
            "manifest_status": runtime_sync.UNKNOWN,
            "registered_lineage_ids": [runtime_sync.UNKNOWN],
        }
    )
    for item in ledger["frozen_blobs"]:
        item["sha256"] = runtime_sync.UNKNOWN
    config["implementation_binding"]["compiled_core_sha256"] = (
        runtime_sync.compiled_core_sha256(config)
    )
    return config


def bound_l_unknown_i_config() -> dict[str, Any]:
    config = normalized_unknown_implementation_config()
    ledger = config["repository_authority"]["predecessor_ledger"]
    ledger.update(
        {
            "status": "BOUND",
            "commit": "a" * 40,
            "integration_id": "GSE232572_A1_QUALIFICATION_AUTHORITY_PREFLIGHT_V1",
            "manifest_status": (
                "A1_GSE232572_QUALIFICATION_AUTHORITY_PREFLIGHT_"
                "LEDGER_REGISTERED_PENDING_EVT049"
            ),
            "registered_lineage_ids": copy.deepcopy(runtime_sync.LEDGER_LINEAGE_IDS),
        }
    )
    for index, item in enumerate(ledger["frozen_blobs"], start=4):
        item["sha256"] = str(index) * 64
    config["implementation_binding"]["compiled_core_sha256"] = (
        runtime_sync.compiled_core_sha256(config)
    )
    return config


def bound_config() -> dict[str, Any]:
    config = bound_l_unknown_i_config()
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
    scientific = copy.deepcopy(config["successor_scientific_state"])
    outer = copy.deepcopy(config["outer_a1_state"])
    status = {
        **scientific,
        **outer,
        "claim_status": "NOT_ESTABLISHED",
        "canonical_record_count": 0,
        "updated_at": "2026-08-12T22:43:38+08:00",
        "historical_status_field": "PRESERVED",
    }
    manifest = {
        **scientific,
        "run_status": "IN_PROGRESS",
        "evidence_status": "BLOCKED_PENDING_PUBLIC_EVIDENCE",
        "claim_status": "NOT_ESTABLISHED",
        "canonical_record_count": 0,
        "active_authority_commit": "9" * 40,
        "registered_artifact_count": 1,
        "outputs": [
            {
                "absolute_path": f"/existing/{index:03d}",
                "artifact_type": f"EXISTING_{index:03d}",
                "bytes": index,
                "sha256": f"{index:064x}",
            }
            for index in range(203)
        ],
        "historical_manifest_field": "PRESERVED",
    }
    events = [
        {
            "event_id": f"A1-EVT-{index:03d}",
            "at": "2026-08-12T00:00:00+08:00",
            "event": "HISTORICAL",
        }
        for index in range(1, 48)
    ]
    events.append(
        {
            "event_id": "A1-EVT-048",
            "at": "2026-08-12T22:43:38+08:00",
            "event": (
                "GSE232572_PUBLIC_RECOVERY_AUDIT_REGISTERED_"
                "QUALIFICATION_GATE_UNCHANGED"
            ),
            "decision_id": "V3-DEC-019",
        }
    )
    payloads = {
        "STATUS.json": runtime_sync.json_bytes(status),
        "RUN_MANIFEST.json": runtime_sync.json_bytes(manifest),
        "EVENT_LOG.jsonl": b"".join(
            runtime_sync.compact_json_line(event) for event in events
        ),
    }
    for name, payload in payloads.items():
        config["runtime"]["predecessor_mutables"][name].update(
            {"bytes": len(payload), "sha256": runtime_sync.sha256(payload)}
        )
    tail_payload = runtime_sync.compact_json_line(events[-1])
    config["runtime"]["predecessor_tail"].update(
        {
            "bytes": len(tail_payload),
            "sha256": runtime_sync.sha256(tail_payload),
        }
    )
    return payloads


def make_context(
    tmp_path: Path,
) -> tuple[dict[str, Any], dict[str, bytes], Path, Path]:
    config = bound_config()
    run_root = tmp_path / "run"
    allowed_root = tmp_path / "prepared-root"
    prepared = allowed_root / "evt049-job"
    run_root.mkdir()
    allowed_root.mkdir()
    config["runtime"]["run_root"] = str(run_root)
    config["runtime"]["allowed_prepared_root"] = str(allowed_root)
    predecessor = predecessor_payloads(config)
    config["implementation_binding"]["compiled_core_sha256"] = (
        runtime_sync.compiled_core_sha256(config)
    )
    for name, payload in predecessor.items():
        (run_root / name).write_bytes(payload)
    return config, predecessor, run_root, prepared


def read_runtime(run_root: Path) -> dict[str, bytes]:
    return {
        name: (run_root / name).read_bytes() for name in runtime_sync.MUTABLE_NAMES
    }


def install_fake_repository_authority(
    monkeypatch: pytest.MonkeyPatch,
    config: dict[str, Any],
    *,
    mode: str = "valid",
) -> bytes:
    """Install deterministic exact4 L -> exact3 I -> config-only B."""

    ledger_commit = config["repository_authority"]["predecessor_ledger"]["commit"]
    implementation_commit = "1" * 40
    binding_commit = "b" * 40
    script_payload = b"bound I EVT049 runtime publisher\n"
    test_payload = b"bound I EVT049 focused test\n"
    binding = config["implementation_binding"]
    binding.update(
        {
            "status": "BOUND",
            "implementation_commit": implementation_commit,
            "implementation_script_sha256": hashlib.sha256(script_payload).hexdigest(),
            "implementation_test_sha256": hashlib.sha256(test_payload).hexdigest(),
        }
    )
    binding["compiled_core_sha256"] = runtime_sync.compiled_core_sha256(config)
    config_payload = runtime_sync.json_bytes(config)
    i_config_payload = runtime_sync.json_bytes(
        runtime_sync.expected_unknown_i_config(config)
    )

    ledger_payloads = {
        item["path"]: f"frozen ledger payload {index}\n".encode()
        for index, item in enumerate(
            config["repository_authority"]["predecessor_ledger"]["frozen_blobs"]
        )
    }
    digest_overrides = {
        payload: item["sha256"]
        for payload, item in zip(
            ledger_payloads.values(),
            config["repository_authority"]["predecessor_ledger"]["frozen_blobs"],
        )
    }
    real_sha256 = runtime_sync.sha256

    def fake_sha256(payload: bytes) -> str:
        return digest_overrides.get(payload, real_sha256(payload))

    changed_paths = {
        ledger_commit: list(
            config["repository_authority"]["predecessor_ledger"][
                "exact_changed_paths"
            ]
        ),
        implementation_commit: list(
            config["repository_authority"]["implementation_exact_changed_paths"]
        ),
        binding_commit: list(
            config["repository_authority"]["binding_exact_changed_paths"]
        ),
    }
    if mode in {"L_paths", "I_paths", "B_paths"}:
        commit = {
            "L_paths": ledger_commit,
            "I_paths": implementation_commit,
            "B_paths": binding_commit,
        }[mode]
        changed_paths[commit].append("unexpected/path")

    blobs: dict[tuple[str, str], bytes] = {
        **{
            (ledger_commit, path): payload
            for path, payload in ledger_payloads.items()
        },
        **{
            (binding_commit, path): payload
            for path, payload in ledger_payloads.items()
        },
        (implementation_commit, runtime_sync.CONFIG_REPO_PATH): i_config_payload,
        (implementation_commit, runtime_sync.SCRIPT_REPO_PATH): script_payload,
        (implementation_commit, runtime_sync.TEST_REPO_PATH): test_payload,
        (binding_commit, runtime_sync.CONFIG_REPO_PATH): config_payload,
        (binding_commit, runtime_sync.SCRIPT_REPO_PATH): script_payload,
        (binding_commit, runtime_sync.TEST_REPO_PATH): test_payload,
    }
    if mode == "I_config":
        blobs[(implementation_commit, runtime_sync.CONFIG_REPO_PATH)] += b"drift"
    if mode == "script_hash":
        blobs[(implementation_commit, runtime_sync.SCRIPT_REPO_PATH)] += b"drift"
    if mode == "current_ledger_blob":
        first_path = config["repository_authority"]["predecessor_ledger"][
            "frozen_blobs"
        ][0]["path"]
        blobs[(binding_commit, first_path)] += b"drift"

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
            return f"{implementation_commit}\n".encode()
        if arguments == ("rev-parse", f"{implementation_commit}^"):
            parent = "e" * 40 if mode == "I_parent" else ledger_commit
            return f"{parent}\n".encode()
        if arguments[:4] == (
            "diff-tree",
            "--no-commit-id",
            "--name-only",
            "-r",
        ):
            return ("\n".join(changed_paths[arguments[4]]) + "\n").encode()
        if arguments[0] == "show":
            commit, blob_path = arguments[1].split(":", 1)
            return blobs[(commit, blob_path)]
        raise AssertionError(arguments)

    worktree_payloads = {
        runtime_sync.CONFIG_REPO_PATH: config_payload,
        runtime_sync.SCRIPT_REPO_PATH: script_payload,
        runtime_sync.TEST_REPO_PATH: test_payload,
        **ledger_payloads,
    }
    if mode == "current_ledger_blob":
        first_path = config["repository_authority"]["predecessor_ledger"][
            "frozen_blobs"
        ][0]["path"]
        worktree_payloads[first_path] += b"drift"
    monkeypatch.setattr(runtime_sync, "sha256", fake_sha256)
    monkeypatch.setattr(runtime_sync, "_run_git", fake_git)
    monkeypatch.setattr(
        runtime_sync,
        "_read_repo_file",
        lambda _repo, blob_path: worktree_payloads[blob_path],
    )
    return config_payload

def test_disk_bound_l_unknown_i_is_exact_and_delta5() -> None:
    config = read_disk_config()
    runtime_sync.validate_static_config(config)
    ledger = config["repository_authority"]["predecessor_ledger"]
    assert not runtime_sync._ledger_values_are_unknown(ledger)
    assert ledger["status"] == "BOUND"
    assert ledger["commit"] == "b0a68155500359300e0246af912e71b3d43dfbe5"
    assert ledger["integration_id"] == (
        "GSE232572_A1_QUALIFICATION_AUTHORITY_PREFLIGHT_V1"
    )
    assert ledger["manifest_status"] == (
        "A1_GSE232572_QUALIFICATION_AUTHORITY_PREFLIGHT_"
        "LEDGER_REGISTERED_PENDING_EVT049"
    )
    assert ledger["registered_lineage_ids"] == runtime_sync.LEDGER_LINEAGE_IDS
    binding = config["implementation_binding"]
    assert runtime_sync._binding_values_are_unknown(binding)
    assert binding["compiled_core_sha256"] == runtime_sync.compiled_core_sha256(config)
    assert config["registered_artifacts"] == runtime_sync.REGISTERED_ARTIFACTS
    assert [
        item["lineage_id"] for item in config["registered_artifacts"]
    ] == runtime_sync.LEDGER_LINEAGE_IDS
    runtime = config["runtime"]
    assert (
        runtime["predecessor_event_count"],
        runtime["successor_event_count"],
    ) == (48, 49)
    assert (
        runtime["predecessor_manifest_output_count"],
        runtime["successor_manifest_output_count"],
        runtime["output_delta_count"],
    ) == (203, 208, 5)
    assert runtime["predecessor_tail"] == {
        "event_id": "A1-EVT-048",
        "decision_id": "V3-DEC-019",
        "bytes": 7395,
        "sha256": (
            "42b5440c7a287cf8c15e5bd3a6717e462af3bf096f423766932390f51f483c84"
        ),
    }
    preflight = config["registered_evidence_truth"][
        "gse232572_a1_qualification_authority_preflight"
    ]
    assert preflight["overall_decision"] == "BLOCKED_MISSING_EXTERNAL_AUTHORITY"
    assert preflight["terminal_status"] == (
        "STOP_BEFORE_PRIVATE_ROW_ACCESS_AND_CANONICAL_MATERIALIZATION"
    )
    assert preflight["registered_aggregate_pass_count"] == 3
    assert preflight["open_qualification_blocker_count"] == 12
    assert preflight["canonical_record_count"] == 0
    assert preflight["qualified"] is False
    assert preflight["training_allowed"] is False
    assert preflight["model_selection_allowed"] is False
    assert preflight["next_phase_authorized"] is False
    boundary = config["access_boundary"]
    assert boundary["private_jsonl_read_count"] == 0
    assert boundary["private_jsonl_registered_artifact_count"] == 0
    assert boundary["private_jsonl_copied"] is False
    assert boundary["private_jsonl_listed"] is False
    assert all(
        not item["absolute_path"].endswith(".jsonl")
        for item in config["registered_artifacts"]
    )


def test_ledger_unknown_group_cannot_be_partially_backfilled() -> None:
    config = unknown_l_and_i_config()
    config["repository_authority"]["predecessor_ledger"][
        "integration_id"
    ] = "GSE232572_A1_QUALIFICATION_AUTHORITY_PREFLIGHT_V1"
    config["implementation_binding"]["compiled_core_sha256"] = (
        runtime_sync.compiled_core_sha256(config)
    )
    with pytest.raises(runtime_sync.BindingError, match="partially known"):
        runtime_sync.validate_static_config(config)


def test_bound_l_unknown_i_stops_before_report_or_runtime_io(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    accessed: list[str] = []

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        accessed.append("external-io")
        raise AssertionError("external I/O occurred before binding")

    monkeypatch.setattr(runtime_sync, "audit_production_repository_authority", forbidden)
    monkeypatch.setattr(runtime_sync, "validate_registered_artifacts", forbidden)
    monkeypatch.setattr(runtime_sync, "_read_runtime", forbidden)
    with pytest.raises(runtime_sync.BindingError, match="implementation is not BOUND"):
        runtime_sync.prepare_runtime_sync(
            prepared_directory=tmp_path / "prepared",
            recorded_at="2026-08-12T23:00:00+08:00",
            production=False,
            config_override=bound_l_unknown_i_config(),
            run_root_override=tmp_path / "run",
        )
    assert accessed == []


def test_exact_byte_validation_hashes_exact1_without_parsing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = bound_config()
    payloads = {
        item["absolute_path"]: f"opaque public report {index}\n".encode()
        for index, item in enumerate(config["registered_artifacts"])
    }
    digest_overrides = {
        payloads[item["absolute_path"]]: item["sha256"]
        for item in config["registered_artifacts"]
    }
    real_sha256 = runtime_sync.sha256
    real_read_bytes = Path.read_bytes

    def fake_read_bytes(path: Path) -> bytes:
        if str(path) in payloads:
            return payloads[str(path)]
        return real_read_bytes(path)

    def fake_sha256(payload: bytes) -> str:
        return digest_overrides.get(payload, real_sha256(payload))

    for item in config["registered_artifacts"]:
        item["bytes"] = len(payloads[item["absolute_path"]])
    monkeypatch.setattr(Path, "read_bytes", fake_read_bytes)
    monkeypatch.setattr(runtime_sync, "sha256", fake_sha256)
    monkeypatch.setattr(runtime_sync, "validate_static_config", lambda _config: None)
    monkeypatch.setattr(
        runtime_sync,
        "load_json",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("registered report JSON was parsed")
        ),
    )
    result = runtime_sync.validate_registered_artifacts(
        config, verify_exact_bytes=True
    )
    assert result["status"] == "EXACT1_BYTES_AND_SHA256_VALIDATED"
    assert result["exact_byte_validation_count"] == 1
    assert result["body_parse_count"] == 0
    assert result["payload_field_read_count"] == 0


def test_normal_transaction_adds_exact1_plus_snapshots_and_sync(
    tmp_path: Path,
) -> None:
    config, predecessor, run_root, prepared = make_context(tmp_path)
    result = runtime_sync.prepare_runtime_sync(
        prepared_directory=prepared,
        recorded_at="2026-08-12T23:00:00+08:00",
        production=False,
        config_override=config,
        run_root_override=run_root,
    )
    assert result == {
        "status": "PREPARED_NOT_PUBLISHED",
        "event_id": "A1-EVT-049",
        "prepared_directory": str(prepared),
        "prepared_member_count": 7,
        "manifest_output_transition": "203_TO_208",
        "new_runtime_output_count": 5,
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
    status, manifest, events = runtime_sync._parse_runtime(read_runtime(run_root))
    old_status, old_manifest, old_events = runtime_sync._parse_runtime(predecessor)
    assert len(events) == 49
    assert events[:-1] == old_events
    event = events[-1]
    assert event["event_id"] == "A1-EVT-049"
    assert event["scientific_state_changed"] is True
    assert event["evidence_surface_changed_since_evt048"] is True
    assert event["evidence_gate_statuses_changed_since_evt048"] is False
    assert event["overall_qualification_gate_changed"] is False
    assert event["registered_lineage_ids"] == runtime_sync.LEDGER_LINEAGE_IDS
    assert event["private_jsonl_read_count"] == 0
    assert event["private_jsonl_registered_artifact_count"] == 0
    assert event["private_jsonl_copied"] is False
    assert event["private_jsonl_listed"] is False
    assert {key: value for key, value in status.items() if key != "updated_at"} == {
        key: value for key, value in old_status.items() if key != "updated_at"
    }
    assert status["updated_at"] == "2026-08-12T23:00:00+08:00"
    assert len(manifest["outputs"]) == 208
    assert manifest["outputs"][:203] == old_manifest["outputs"]
    assert manifest["outputs"][203:204] == config["registered_artifacts"]
    assert [Path(item["absolute_path"]).name for item in manifest["outputs"][-4:]] == (
        config["runtime"]["immutable_publish_order"]
    )
    assert manifest["registered_artifact_count"] == 1
    sync = runtime_sync.load_json(
        (run_root / config["runtime"]["sync_name"]).read_bytes(), label="sync"
    )
    assert sync["registered_artifacts"] == config["registered_artifacts"]
    assert sync["registered_artifact_body_parse_count"] == 0
    assert sync["registered_artifact_payload_field_read_count"] == 0
    assert sync["private_jsonl_read_count"] == 0
    assert sync["private_jsonl_registered_artifact_count"] == 0
    assert sync["private_jsonl_copied"] is False
    assert sync["private_jsonl_listed"] is False
    assert runtime_sync.validate_published(
        prepared_directory=prepared,
        production=False,
        config_override=config,
        run_root_override=run_root,
    )["status"] == "PUBLISHED_VERIFIED"


def test_predecessor_drift_stops_before_prepared_or_runtime_write(
    tmp_path: Path,
) -> None:
    config, _predecessor, run_root, prepared = make_context(tmp_path)
    status = runtime_sync.load_json((run_root / "STATUS.json").read_bytes(), label="status")
    status["drift"] = True
    (run_root / "STATUS.json").write_bytes(runtime_sync.json_bytes(status))
    before = read_runtime(run_root)
    with pytest.raises(runtime_sync.PredecessorError, match="identity drift"):
        runtime_sync.prepare_runtime_sync(
            prepared_directory=prepared,
            recorded_at="2026-08-12T23:00:00+08:00",
            production=False,
            config_override=config,
            run_root_override=run_root,
        )
    assert read_runtime(run_root) == before
    assert not prepared.exists()


def test_immutables_first_prefix_recovery_and_idempotent_retry(
    tmp_path: Path,
) -> None:
    config, predecessor, run_root, prepared = make_context(tmp_path)
    runtime_sync.prepare_runtime_sync(
        prepared_directory=prepared,
        recorded_at="2026-08-12T23:00:00+08:00",
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
    assert len(
        runtime_sync.load_events(
            read_runtime(run_root)["EVENT_LOG.jsonl"], label="events"
        )
    ) == 49


def test_production_authority_accepts_exact_l_i_b(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = bound_config()
    config_payload = install_fake_repository_authority(monkeypatch, config)
    result = runtime_sync.audit_production_repository_authority(config, config_payload)
    assert result == {
        "status": "PASS_EXACT_L_TO_I_TO_CONFIG_ONLY_B",
        "ledger_commit": config["repository_authority"]["predecessor_ledger"][
            "commit"
        ],
        "implementation_i_commit": "1" * 40,
        "binding_b_commit": "b" * 40,
        "head_commit": "b" * 40,
        "upstream_head_commit": "b" * 40,
        "origin_branch_head_commit": "b" * 40,
        "worktree_and_index_clean": True,
    }


@pytest.mark.parametrize(
    "mode",
    [
        "dirty",
        "upstream",
        "origin",
        "L_paths",
        "I_paths",
        "B_paths",
        "I_parent",
        "I_config",
        "script_hash",
        "current_ledger_blob",
    ],
)
def test_production_authority_drift_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    config = bound_config()
    config_payload = install_fake_repository_authority(monkeypatch, config, mode=mode)
    with pytest.raises(runtime_sync.RuntimeSyncError):
        runtime_sync.audit_production_repository_authority(config, config_payload)


def test_context_orders_authority_before_report_and_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = bound_config()
    config_payload = runtime_sync.json_bytes(config)
    calls: list[str] = []
    monkeypatch.setattr(
        runtime_sync,
        "_load_config_payload",
        lambda *_args, **_kwargs: (copy.deepcopy(config), config_payload),
    )

    def authority(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        calls.append("authority")
        return {"status": "PASS"}

    def evidence(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        calls.append("reports")
        raise runtime_sync.PublicationError("stop after ordering proof")

    def runtime(*_args: Any, **_kwargs: Any) -> Any:
        calls.append("runtime")
        raise AssertionError("runtime should not be reached")

    monkeypatch.setattr(runtime_sync, "audit_production_repository_authority", authority)
    monkeypatch.setattr(runtime_sync, "validate_registered_artifacts", evidence)
    monkeypatch.setattr(runtime_sync, "_read_runtime", runtime)
    with pytest.raises(runtime_sync.PublicationError, match="ordering proof"):
        runtime_sync.prepare_runtime_sync(
            prepared_directory=(
                Path(config["runtime"]["allowed_prepared_root"]) / "evt049-test"
            ),
            recorded_at="2026-08-12T23:00:00+08:00",
            production=True,
        )
    assert calls == ["authority", "reports"]
