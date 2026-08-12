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
    / "configs/route_a_v3_gse232572_public_recovery_audit_runtime_sync_v1.json"
)
SCRIPT_PATH = (
    STAGING_ROOT
    / "scripts/route_a_v3/gse232572_public_recovery_audit_runtime_sync.py"
)
SPEC = importlib.util.spec_from_file_location("evt047_runtime_sync", SCRIPT_PATH)
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
    config["implementation_binding"]["compiled_core_sha256"] = (
        runtime_sync.compiled_core_sha256(config)
    )
    return config


def bound_config() -> dict[str, Any]:
    config = normalized_unknown_implementation_config()
    config["implementation_binding"].update(
        {
            "status": "BOUND",
            "implementation_commit": "1" * 40,
            "implementation_script_sha256": "2" * 64,
            "implementation_test_sha256": "3" * 64,
        }
    )
    config["implementation_binding"]["compiled_core_sha256"] = (
        runtime_sync.compiled_core_sha256(config)
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
        "updated_at": "2026-08-12T17:15:17+08:00",
        "historical_status_field": "PRESERVED",
    }
    manifest = {
        **scientific,
        "run_status": "IN_PROGRESS",
        "evidence_status": "BLOCKED_PENDING_PUBLIC_EVIDENCE",
        "claim_status": "NOT_ESTABLISHED",
        "canonical_record_count": 0,
        "active_authority_commit": "a" * 40,
        "registered_artifact_count": 14,
        "outputs": [
            {
                "absolute_path": f"/existing/{index:03d}",
                "artifact_type": f"EXISTING_{index:03d}",
                "bytes": index,
                "sha256": f"{index:064x}",
            }
            for index in range(192)
        ],
        "historical_manifest_field": "PRESERVED",
    }
    events = [
        {
            "event_id": f"A1-EVT-{index:03d}",
            "at": "2026-08-12T00:00:00+08:00",
            "event": "HISTORICAL",
        }
        for index in range(1, 46)
    ]
    events.append(
        {
            "event_id": "A1-EVT-046",
            "at": "2026-08-12T17:15:17+08:00",
            "event": "EVT046",
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
    prepared = allowed_root / "evt047-job"
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
    """Install a deterministic fake of L -> frozen I1 -> exact2 I2 -> B2."""

    ledger_commit = config["repository_authority"]["predecessor_ledger"]["commit"]
    frozen_i1_commit = runtime_sync.FROZEN_I1_COMMIT
    implementation_i2_commit = "1" * 40
    binding_b2_commit = "b" * 40
    frozen_i1_script_payload = b"frozen I1 EVT047 runtime publisher\n"
    frozen_i1_test_payload = b"frozen I1 EVT047 focused test\n"
    i2_script_payload = b"bound I2 EVT047 runtime publisher\n"
    i2_test_payload = b"bound I2 EVT047 focused test\n"
    binding = config["implementation_binding"]
    binding.update(
        {
            "status": "BOUND",
            "implementation_commit": implementation_i2_commit,
            "implementation_script_sha256": hashlib.sha256(
                i2_script_payload
            ).hexdigest(),
            "implementation_test_sha256": hashlib.sha256(
                i2_test_payload
            ).hexdigest(),
        }
    )
    binding["compiled_core_sha256"] = runtime_sync.compiled_core_sha256(config)
    config_payload = runtime_sync.json_bytes(config)
    frozen_i1_config_payload = runtime_sync.json_bytes(
        runtime_sync.expected_unknown_i2_config(config)
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
    digest_overrides.update(
        {
            frozen_i1_config_payload: runtime_sync.FROZEN_I1_BLOB_SHA256[
                runtime_sync.CONFIG_REPO_PATH
            ],
            frozen_i1_script_payload: runtime_sync.FROZEN_I1_BLOB_SHA256[
                runtime_sync.SCRIPT_REPO_PATH
            ],
            frozen_i1_test_payload: runtime_sync.FROZEN_I1_BLOB_SHA256[
                runtime_sync.TEST_REPO_PATH
            ],
        }
    )
    real_sha256 = runtime_sync.sha256

    def fake_sha256(payload: bytes) -> str:
        return digest_overrides.get(payload, real_sha256(payload))

    changed_paths = {
        ledger_commit: list(
            config["repository_authority"]["predecessor_ledger"][
                "exact_changed_paths"
            ]
        ),
        frozen_i1_commit: list(
            config["repository_authority"]["implementation_exact_changed_paths"]
        ),
        implementation_i2_commit: list(runtime_sync.I2_EXACT_CHANGED_PATHS),
        binding_b2_commit: list(
            config["repository_authority"]["binding_exact_changed_paths"]
        ),
    }
    if mode in {"L_paths", "I1_paths", "I2_paths", "B2_paths"}:
        commit = {
            "L_paths": ledger_commit,
            "I1_paths": frozen_i1_commit,
            "I2_paths": implementation_i2_commit,
            "B2_paths": binding_b2_commit,
        }[mode]
        changed_paths[commit].append("unexpected/path")

    blobs: dict[tuple[str, str], bytes] = {
        **{
            (ledger_commit, path): payload
            for path, payload in ledger_payloads.items()
        },
        **{
            (binding_b2_commit, path): payload
            for path, payload in ledger_payloads.items()
        },
        (frozen_i1_commit, runtime_sync.CONFIG_REPO_PATH): frozen_i1_config_payload,
        (frozen_i1_commit, runtime_sync.SCRIPT_REPO_PATH): frozen_i1_script_payload,
        (frozen_i1_commit, runtime_sync.TEST_REPO_PATH): frozen_i1_test_payload,
        (implementation_i2_commit, runtime_sync.CONFIG_REPO_PATH): (
            frozen_i1_config_payload
        ),
        (implementation_i2_commit, runtime_sync.SCRIPT_REPO_PATH): i2_script_payload,
        (implementation_i2_commit, runtime_sync.TEST_REPO_PATH): i2_test_payload,
        (binding_b2_commit, runtime_sync.CONFIG_REPO_PATH): config_payload,
        (binding_b2_commit, runtime_sync.SCRIPT_REPO_PATH): i2_script_payload,
        (binding_b2_commit, runtime_sync.TEST_REPO_PATH): i2_test_payload,
    }
    if mode == "I1_config_hash":
        blobs[(frozen_i1_commit, runtime_sync.CONFIG_REPO_PATH)] += b"drift"
    if mode == "I1_script_hash":
        blobs[(frozen_i1_commit, runtime_sync.SCRIPT_REPO_PATH)] += b"drift"
    if mode == "I2_config":
        blobs[(implementation_i2_commit, runtime_sync.CONFIG_REPO_PATH)] += b"drift"
    if mode == "script_hash":
        blobs[(implementation_i2_commit, runtime_sync.SCRIPT_REPO_PATH)] += b"drift"
    if mode == "current_ledger_blob":
        first_path = config["repository_authority"]["predecessor_ledger"][
            "frozen_blobs"
        ][0]["path"]
        blobs[(binding_b2_commit, first_path)] += b"drift"

    def fake_git(_repo: Path, *arguments: str) -> bytes:
        branch = config["repository_authority"]["branch"]
        if arguments == ("rev-parse", "HEAD"):
            return f"{binding_b2_commit}\n".encode()
        if arguments == ("rev-parse", "@{upstream}"):
            observed = "c" * 40 if mode == "upstream" else binding_b2_commit
            return f"{observed}\n".encode()
        if arguments == (
            "rev-parse",
            "--verify",
            f"refs/remotes/origin/{branch}",
        ):
            observed = "d" * 40 if mode == "origin" else binding_b2_commit
            return f"{observed}\n".encode()
        if arguments == ("rev-parse", "--abbrev-ref", "HEAD"):
            return f"{branch}\n".encode()
        if arguments == ("rev-parse", "--abbrev-ref", "@{upstream}"):
            return f"origin/{branch}\n".encode()
        if arguments == ("status", "--porcelain=v1", "--untracked-files=all"):
            return b" M dirty\n" if mode == "dirty" else b""
        if arguments == ("rev-parse", f"{binding_b2_commit}^"):
            return f"{implementation_i2_commit}\n".encode()
        if arguments == ("rev-parse", f"{implementation_i2_commit}^"):
            parent = "e" * 40 if mode == "I2_parent" else frozen_i1_commit
            return f"{parent}\n".encode()
        if arguments == ("rev-parse", f"{frozen_i1_commit}^"):
            parent = "e" * 40 if mode == "I1_parent" else ledger_commit
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
        runtime_sync.CONFIG_REPO_PATH: config_payload,
        runtime_sync.SCRIPT_REPO_PATH: i2_script_payload,
        runtime_sync.TEST_REPO_PATH: i2_test_payload,
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
        lambda _repo, path: worktree_payloads[path],
    )
    return config_payload


def assert_disk_candidate_freeze_and_runtime_delta(config: dict[str, Any]) -> None:
    runtime_sync.validate_static_config(config)
    assert config["implementation_binding"]["compiled_core_sha256"] == (
        runtime_sync.compiled_core_sha256(config)
    )
    ledger = config["repository_authority"]["predecessor_ledger"]
    assert ledger == {
        "status": "BOUND",
        "commit": "cd6c6ef90b8905e5e2fe067402a37c0661e89edf",
        "integration_id": "GSE232572_PUBLIC_RECOVERY_AUDIT_V1",
        "manifest_status": (
            "A1_GSE232572_PUBLIC_RECOVERY_AUDIT_"
            "LEDGER_REGISTERED_PENDING_EVT047"
        ),
        "registered_lineage_ids": ["gse232572_public_recovery_audit_v1"],
        "exact_changed_paths": runtime_sync.LEDGER_PATHS,
        "frozen_blobs": [
            {
                "path": runtime_sync.LEDGER_PATHS[0],
                "sha256": (
                    "7498c4fe6f1e63d2df9c9e2da08feff62e45c43647962d653875eb40ba2f868d"
                ),
            },
            {
                "path": runtime_sync.LEDGER_PATHS[1],
                "sha256": (
                    "89fe70ae69c14cfdf76bcd2c86fc28bbf6a8957cf430de22314a61c48281e297"
                ),
            },
            {
                "path": runtime_sync.LEDGER_PATHS[2],
                "sha256": (
                    "4313db7f5d7651be5fda93dd70eed318354115985a0b19ba010dd5cb262061cd"
                ),
            },
            {
                "path": runtime_sync.LEDGER_PATHS[3],
                "sha256": (
                    "10379c502afa010dc8dd2b550fe7085f9240110de02c170f2f5963f766dc5ee8"
                ),
            },
        ],
    }
    binding = config["implementation_binding"]
    scalar_keys = (
        "status",
        "implementation_commit",
        "implementation_script_sha256",
        "implementation_test_sha256",
    )
    binding_is_unknown = all(binding[key] == runtime_sync.UNKNOWN for key in scalar_keys)
    binding_is_bound = binding["status"] == "BOUND" and all(
        binding[key] != runtime_sync.UNKNOWN for key in scalar_keys[1:]
    )
    assert binding_is_unknown or binding_is_bound
    frozen_i1_config = runtime_sync.json_bytes(
        runtime_sync.expected_unknown_i2_config(config)
    )
    assert runtime_sync.sha256(frozen_i1_config) == (
        runtime_sync.FROZEN_I1_BLOB_SHA256[runtime_sync.CONFIG_REPO_PATH]
    )
    assert config["registered_artifacts"] == runtime_sync.REGISTERED_ARTIFACTS
    assert config["runtime"]["predecessor_event_count"] == 46
    assert config["runtime"]["successor_event_count"] == 47
    assert config["runtime"]["predecessor_manifest_output_count"] == 192
    assert config["runtime"]["successor_manifest_output_count"] == 197
    assert config["runtime"]["output_delta_count"] == 5
    assert config["successor_scientific_state"] == runtime_sync.SUCCESSOR_SCIENTIFIC_STATE
    assert config["registered_evidence_truth"]["gse232572_public_recovery_audit"] == {
        "schema_version": "1.0.0",
        "record_type": "GSE232572_PUBLIC_RECOVERY_AUDIT_AGGREGATE_ONLY",
        "status": "DEVELOPMENT_PRIVATE_RECONSTRUCTION_COMPLETE_NOT_QUALIFIED",
        "scientific_disposition": (
            "DEVELOPMENT_RECONSTRUCTION_ONLY_AUDIT_PENDING_NOT_QUALIFIED"
        ),
        "registry_role": "AUDIT_ONLY",
        "qualification_status": "AUDIT_PENDING",
        "published_universe_row_count": 11929,
        "development_reconstruction_record_count": 8068,
        "accepted_pair_count": 8068,
        "rejected_published_row_count": 3861,
        "qualified": False,
        "ordinary_study_contribution": 0,
        "a1_study_contribution": 0,
        "true_a2_study_contribution": 0,
        "canonical_record_count": 0,
        "canonical_materialization_allowed": False,
        "training_allowed": False,
        "model_selection_allowed": False,
        "next_phase_authorized": False,
    }
    assert all(
        not item["name"].endswith(".private.jsonl")
        for item in config["registered_artifacts"]
    )


def test_disk_bound_l_unknown_i2_freeze_evt046_report_and_runtime_delta() -> None:
    assert_disk_candidate_freeze_and_runtime_delta(read_disk_config())


def test_temporary_disk_bound_b2_preserves_frozen_i1_config_and_runtime_delta(
    tmp_path: Path,
) -> None:
    config = normalized_unknown_implementation_config()
    config["implementation_binding"].update(
        {
            "status": "BOUND",
            "implementation_commit": "1" * 40,
            "implementation_script_sha256": "2" * 64,
            "implementation_test_sha256": "3" * 64,
        }
    )
    temporary_config = tmp_path / CONFIG_PATH.name
    temporary_config.write_bytes(runtime_sync.json_bytes(config))
    assert_disk_candidate_freeze_and_runtime_delta(
        json.loads(temporary_config.read_text())
    )


def test_ledger_authority_unknowns_cannot_be_partially_backfilled() -> None:
    config = unknown_l_and_i_config()
    config["repository_authority"]["predecessor_ledger"][
        "integration_id"
    ] = "GSE232572_PUBLIC_RECOVERY_AUDIT_V1"
    config["implementation_binding"]["compiled_core_sha256"] = (
        runtime_sync.compiled_core_sha256(config)
    )
    with pytest.raises(runtime_sync.BindingError, match="partially known"):
        runtime_sync.validate_static_config(config)


def test_bound_l_unknown_i_stops_before_authority_report_or_runtime_io(
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
            recorded_at="2026-08-12T20:00:00+08:00",
            production=False,
            config_override=bound_l_unknown_i_config(),
            run_root_override=tmp_path / "run",
        )
    assert accessed == []


def test_unknown_l_stops_direct_exact_report_validation_before_report_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    accessed: list[str] = []

    def forbidden(_path: Path) -> bytes:
        accessed.append("report")
        raise AssertionError("report I/O occurred before L binding")

    monkeypatch.setattr(Path, "read_bytes", forbidden)
    with pytest.raises(runtime_sync.BindingError, match="ledger authority is not BOUND"):
        runtime_sync.validate_registered_artifacts(
            unknown_l_and_i_config(), verify_exact_bytes=True
        )
    assert accessed == []


def test_normal_transaction_adds_exact1_plus_exact4_and_preserves_truth(
    tmp_path: Path,
) -> None:
    config, predecessor, run_root, prepared = make_context(tmp_path)
    result = runtime_sync.prepare_runtime_sync(
        prepared_directory=prepared,
        recorded_at="2026-08-12T20:00:00+08:00",
        production=False,
        config_override=config,
        run_root_override=run_root,
    )
    assert result == {
        "status": "PREPARED_NOT_PUBLISHED",
        "event_id": "A1-EVT-047",
        "prepared_directory": str(prepared),
        "prepared_member_count": 7,
        "manifest_output_transition": "192_TO_197",
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
    assert len(events) == 47
    assert events[:-1] == old_events
    assert events[-1]["event_id"] == "A1-EVT-047"
    assert events[-1]["scientific_state_changed"] is True
    assert events[-1]["evidence_surface_changed_since_evt046"] is True
    assert events[-1]["evidence_gate_statuses_changed_since_evt046"] is False
    assert events[-1]["overall_qualification_gate_changed"] is False
    assert events[-1]["successor_scientific_state"] == (
        config["successor_scientific_state"]
    )
    assert events[-1]["registered_evidence_truth"] == (
        config["registered_evidence_truth"]
    )
    assert {key: value for key, value in status.items() if key != "updated_at"} == {
        key: value for key, value in old_status.items() if key != "updated_at"
    }
    assert status["updated_at"] == "2026-08-12T20:00:00+08:00"
    assert len(manifest["outputs"]) == 197
    assert manifest["outputs"][:192] == old_manifest["outputs"]
    assert manifest["outputs"][192:193] == config["registered_artifacts"]
    assert [Path(item["absolute_path"]).name for item in manifest["outputs"][-4:]] == (
        config["runtime"]["immutable_publish_order"]
    )
    assert manifest["registered_artifact_count"] == 1
    sync = runtime_sync.load_json(
        (run_root / config["runtime"]["sync_name"]).read_bytes(), label="sync"
    )
    assert sync["registered_artifacts"] == config["registered_artifacts"]
    assert sync["registered_artifact_body_parse_count"] == 0
    assert all(
        not item["name"].endswith(".private.jsonl")
        for item in sync["registered_artifacts"]
    )
    assert sync["outer_a1_state"] == config["outer_a1_state"]
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
            recorded_at="2026-08-12T20:00:00+08:00",
            production=False,
            config_override=config,
            run_root_override=run_root,
        )
    assert read_runtime(run_root) == before
    assert not prepared.exists()


def test_immutables_first_prefix_recovery_and_idempotent_retry(tmp_path: Path) -> None:
    config, predecessor, run_root, prepared = make_context(tmp_path)
    runtime_sync.prepare_runtime_sync(
        prepared_directory=prepared,
        recorded_at="2026-08-12T20:00:00+08:00",
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
    ) == 47


def test_exact_byte_validation_hashes_one_artifact_without_parsing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = bound_config()
    payloads = {
        item["absolute_path"]: f"opaque artifact {index}\n".encode()
        for index, item in enumerate(config["registered_artifacts"])
    }
    expected_digest = {
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
        return expected_digest.get(payload, real_sha256(payload))

    for item in config["registered_artifacts"]:
        item["bytes"] = len(payloads[item["absolute_path"]])
    monkeypatch.setattr(Path, "read_bytes", fake_read_bytes)
    monkeypatch.setattr(runtime_sync, "sha256", fake_sha256)
    # This test exercises the byte/hash operation itself; static frozen-metadata
    # validation is covered separately and is not repeated after fixture sizing.
    monkeypatch.setattr(runtime_sync, "validate_static_config", lambda _config: None)
    monkeypatch.setattr(
        runtime_sync,
        "load_json",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("artifact JSON was parsed")
        ),
    )
    result = runtime_sync.validate_registered_artifacts(
        config, verify_exact_bytes=True
    )
    assert result["status"] == "EXACT1_BYTES_AND_SHA256_VALIDATED"
    assert result["exact_byte_validation_count"] == 1
    assert result["body_parse_count"] == 0
    assert result["payload_field_read_count"] == 0


def test_production_authority_accepts_exact_l_i1_i2_b2_before_external_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = bound_config()
    config_payload = install_fake_repository_authority(monkeypatch, config)
    result = runtime_sync.audit_production_repository_authority(config, config_payload)
    assert result == {
        "status": "PASS_EXACT_L_TO_FROZEN_I1_TO_EXACT2_I2_TO_CONFIG_ONLY_B2",
        "ledger_commit": config["repository_authority"]["predecessor_ledger"][
            "commit"
        ],
        "frozen_i1_commit": runtime_sync.FROZEN_I1_COMMIT,
        "implementation_i2_commit": "1" * 40,
        "binding_b2_commit": "b" * 40,
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
        "I1_paths",
        "I2_paths",
        "B2_paths",
        "I1_parent",
        "I2_parent",
        "I1_config_hash",
        "I1_script_hash",
        "I2_config",
        "script_hash",
        "current_ledger_blob",
    ],
)
def test_production_authority_drift_stops_before_evidence_or_runtime_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    config = bound_config()
    config_payload = install_fake_repository_authority(monkeypatch, config, mode=mode)
    accessed: list[str] = []

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        accessed.append("evidence-or-runtime")
        raise AssertionError("evidence/runtime I/O occurred before authority")

    monkeypatch.setattr(runtime_sync, "validate_registered_artifacts", forbidden)
    monkeypatch.setattr(runtime_sync, "_read_runtime", forbidden)
    with pytest.raises(runtime_sync.RuntimeSyncError):
        runtime_sync.audit_production_repository_authority(config, config_payload)
    assert accessed == []
    assert not (tmp_path / "prepared").exists()


def test_context_orders_authority_before_evidence_and_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
        calls.append("evidence")
        raise runtime_sync.PublicationError("stop after evidence ordering proof")

    def runtime(*_args: Any, **_kwargs: Any) -> Any:
        calls.append("runtime")
        raise AssertionError("runtime should not be reached")

    monkeypatch.setattr(runtime_sync, "audit_production_repository_authority", authority)
    monkeypatch.setattr(runtime_sync, "validate_registered_artifacts", evidence)
    monkeypatch.setattr(runtime_sync, "_read_runtime", runtime)
    with pytest.raises(runtime_sync.PublicationError, match="ordering proof"):
        runtime_sync.prepare_runtime_sync(
            prepared_directory=(
                Path(config["runtime"]["allowed_prepared_root"]) / "evt047-test"
            ),
            recorded_at="2026-08-12T20:00:00+08:00",
            production=True,
        )
    assert calls == ["authority", "evidence"]
    assert not (tmp_path / "prepared").exists()
