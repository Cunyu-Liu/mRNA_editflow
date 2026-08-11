from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest


STAGING_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = STAGING_ROOT / "configs/route_a_v3_gse114002_endpoint_geometry_reconciliation_v2_runtime_sync_v1.json"
SCRIPT_PATH = STAGING_ROOT / "scripts/route_a_v3/gse114002_endpoint_geometry_reconciliation_v2_runtime_sync.py"
SPEC = importlib.util.spec_from_file_location("gse114002_evt039_runtime_sync", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
runtime_sync = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runtime_sync)


def read_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def refresh_core(config: dict[str, Any]) -> None:
    config["implementation_binding"]["compiled_core_sha256"] = (
        runtime_sync.compiled_core_sha256(config)
    )


def unknown_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    value = copy.deepcopy(config if config is not None else read_config())
    for key in (
        "status",
        "implementation_commit",
        "implementation_script_sha256",
        "implementation_test_sha256",
    ):
        value["implementation_binding"][key] = "UNKNOWN_NOT_ASSERTED"
    return value


def bind_config(config: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(config)
    binding = value["implementation_binding"]
    binding["status"] = "BOUND"
    binding["implementation_commit"] = "1" * 40
    binding["implementation_script_sha256"] = "2" * 64
    binding["implementation_test_sha256"] = "3" * 64
    refresh_core(value)
    return value


def render_attempt_bundle(
    attempt: dict[str, Any],
    *,
    binding_commit: str | None = None,
) -> dict[str, bytes]:
    blockers = list(runtime_sync.EXPECTED_BLOCKERS)
    if attempt["attempt_id"] == "ATTEMPT_1_FAILED_PRESERVED":
        blockers.append("ZZ_SYNTHETIC_CONDITIONAL_MECHANICAL_BLOCKER")
        blockers.sort()
    report = {
        "status": attempt["status"],
        "qualified": False,
        "scientific_claim_status": "NOT_ESTABLISHED",
        "ordinary_study_contribution": 0,
        "a1_intervention_study_contribution": 0,
        "true_a2_dense_study_contribution": 0,
        "canonical_record_count": 0,
        "canonical_materialization_allowed": False,
        "training_allowed": False,
        "model_selection_allowed": False,
        "next_phase_authorized": False,
        "true_a2_claim_established": False,
        "aggregate_only": True,
        "blockers": blockers,
        "implementation_binding": {
            "status": "PASS_BOUND_IMPLEMENTATION",
            "verified": True,
            "implementation_commit": attempt["implementation_commit"],
            "binding_commit": binding_commit or attempt["binding_commit"],
        },
    }
    payloads = {
        "INPUT_INTEGRITY_AUDIT.json": runtime_sync.json_bytes(
            {"dataset_id": "GSE114002", "status": "PASS_SYNTHETIC_INPUT"}
        ),
        "ENDPOINT_RECONCILIATION_AUDIT.json": runtime_sync.json_bytes(
            {"dataset_id": "GSE114002", "status": attempt["status"]}
        ),
        "POOL_GEOMETRY_RECONCILIATION_AUDIT.json": runtime_sync.json_bytes(
            {"dataset_id": "GSE114002", "status": "SYNTHETIC_AGGREGATE_ONLY"}
        ),
        "QUALIFICATION_REPORT.json": runtime_sync.json_bytes(report),
    }
    payloads["SHA256SUMS"] = "".join(
        f"{runtime_sync.sha256(payloads[name])}  {name}\n"
        for name in sorted(runtime_sync.BUNDLE_JSON_NAMES)
    ).encode("ascii")
    root = Path(attempt["run_root"])
    marker = {
        "schema_version": "1.0.0",
        "record_type": "GSE114002_ENDPOINT_GEOMETRY_RECONCILIATION_V2_COMMIT",
        "contract_id": "mrna_xeditflow_route_a_v3",
        "protocol_id": "ROUTE_A_V3_GSE114002_ENDPOINT_GEOMETRY_RECONCILIATION_V2",
        "dataset_id": "GSE114002",
        "output_id": "ROUTE_A_V3_GSE114002_ENDPOINT_GEOMETRY_RECONCILIATION_V2_BUNDLE",
        "scientific_status": attempt["status"],
        "publication_mode": runtime_sync.PRIMARY_PUBLICATION_MODE,
        "sha256sums_sha256": runtime_sync.sha256(payloads["SHA256SUMS"]),
        "bundle_file_count_excluding_commit_marker": 5,
        "bundle_member_names_excluding_commit_marker": sorted(
            set(runtime_sync.BUNDLE_JSON_NAMES) | {"SHA256SUMS"}
        ),
        "final_output_directory_name_sha256": runtime_sync.sha256(
            root.name.encode("utf-8")
        ),
        "final_output_target_sha256": runtime_sync._final_target_sha256(root),
        "committed": True,
        "commit_marker_written_last": True,
        "aggregate_acceptance_requires_exact_marker": True,
    }
    payloads["PUBLICATION_COMMIT.json"] = runtime_sync.json_bytes(marker)
    return payloads


def materialize_attempt(
    attempt: dict[str, Any], *, binding_commit: str | None = None
) -> dict[str, bytes]:
    root = Path(attempt["run_root"])
    root.mkdir(exist_ok=True)
    payloads = render_attempt_bundle(attempt, binding_commit=binding_commit)
    for name, payload in payloads.items():
        (root / name).write_bytes(payload)
        attempt["members"][name] = {
            "bytes": len(payload),
            "sha256": runtime_sync.sha256(payload),
        }
    return payloads


def predecessor_payloads(config: dict[str, Any]) -> dict[str, bytes]:
    status = {
        "run_status": "IN_PROGRESS",
        "evidence_status": "BLOCKED_PENDING_PUBLIC_EVIDENCE",
        "gate_status": "A1_BLOCKED_PENDING_PUBLIC_EVIDENCE",
        "claim_status": "NOT_ESTABLISHED",
        "training_started": False,
        "next_phase_authorized": False,
        "code_commit": "28cd2f132d022fea6ac43e1f89d6673d02a9c97d",
        "updated_at": "2026-08-11T06:00:00+08:00",
        "legacy": "PRESERVED",
    }
    outputs = [
        {
            "artifact_type": f"OLD_{index:03d}",
            "absolute_path": f"/old/{index:03d}",
            "sha256": f"{index:064x}",
            "status": "COMPLETE",
        }
        for index in range(106)
    ]
    manifest = {
        "run_status": "IN_PROGRESS",
        "evidence_status": "BLOCKED_PENDING_PUBLIC_EVIDENCE",
        "claim_status": "NOT_ESTABLISHED",
        "code_commit": "28cd2f132d022fea6ac43e1f89d6673d02a9c97d",
        "active_authority_commit": "d078060c81114687db5068902a5aad5d9bedbee6",
        "outputs": outputs,
    }
    events = [
        {
            "event_id": f"A1-EVT-{index:03d}",
            "at": "2026-08-10T00:00:00+08:00",
            "event": "OLD",
        }
        for index in range(1, 38)
    ]
    events.append(
        {
            "event_id": "A1-EVT-038",
            "at": "2026-08-11T06:30:00+08:00",
            "event": "GSE200304_EVT037_TRAINING_STARTED_FALSE_APPEND_ONLY_CORRECTION_GATE_UNCHANGED",
            "training_started": False,
            "training_allowed": False,
            "next_phase_authorized": False,
        }
    )
    return {
        "STATUS.json": runtime_sync.json_bytes(status),
        "RUN_MANIFEST.json": runtime_sync.json_bytes(manifest),
        "EVENT_LOG.jsonl": b"".join(
            runtime_sync.compact_json_line(item) for item in events
        ),
    }


def make_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, Any], dict[str, bytes], dict[str, Any], Path, Path]:
    config = bind_config(unknown_config())
    run_root, allowed = tmp_path / "run", tmp_path / "allowed"
    prepared = allowed / "job"
    config["runtime"]["run_root"] = str(run_root)
    config["runtime"]["allowed_prepared_root"] = str(allowed)
    for index, attempt in enumerate(config["runtime"]["attempts"], start=1):
        attempt["run_root"] = str(tmp_path / f"attempt_{index}")
        materialize_attempt(attempt)
    predecessor = predecessor_payloads(config)
    for name, payload in predecessor.items():
        config["runtime"]["predecessor_mutables"][name]["bytes"] = len(payload)
        config["runtime"]["predecessor_mutables"][name]["sha256"] = (
            runtime_sync.sha256(payload)
        )
    tail_line = predecessor["EVENT_LOG.jsonl"].splitlines(keepends=True)[-1]
    config["runtime"]["predecessor_tail_event"]["sha256"] = runtime_sync.sha256(
        tail_line
    )
    refresh_core(config)
    run_root.mkdir()
    allowed.mkdir()
    for name, payload in predecessor.items():
        (run_root / name).write_bytes(payload)
    authority = {
        "status": "PASS_EXACT_SYNTHETIC_I_TO_CONFIG_ONLY_B",
        "binding_commit": "4" * 40,
        "head_commit": "4" * 40,
        "origin_branch_head_commit": "4" * 40,
        "config_sha256": runtime_sync.sha256(runtime_sync.json_bytes(config)),
    }
    monkeypatch.setattr(runtime_sync, "validate_bound_config", lambda _config: None)
    return config, predecessor, authority, run_root, prepared


def prepare_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, Any], dict[str, bytes], dict[str, Any], Path, Path]:
    context = make_context(tmp_path, monkeypatch)
    config, _predecessor, authority, run_root, prepared = context
    result = runtime_sync.prepare_runtime_sync(
        prepared_directory=prepared,
        recorded_at="2026-08-11T08:30:00+08:00",
        production=False,
        config_override=config,
        authority_override=authority,
        run_root_override=run_root,
    )
    assert result["status"] == "PREPARED_NOT_PUBLISHED"
    assert result["manifest_output_transition"] == "106_TO_122"
    return context


def tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        item.name: item.read_bytes()
        for item in root.iterdir()
        if item.is_file() and not item.is_symlink()
    }


def test_static_config_supports_i_or_b_and_freezes_evt038_and_two_attempts() -> None:
    config = read_config()
    assert config["implementation_binding"]["status"] in {
        "UNKNOWN_NOT_ASSERTED",
        "BOUND",
    }
    assert config["implementation_binding"]["compiled_core_sha256"] == (
        runtime_sync.compiled_core_sha256(config)
    )
    runtime = config["runtime"]
    assert (runtime["predecessor_event_count"], runtime["successor_event_count"]) == (
        38,
        39,
    )
    assert (
        runtime["predecessor_manifest_output_count"],
        runtime["successor_manifest_output_count"],
        runtime["output_delta_count"],
    ) == (106, 122, 16)
    assert runtime["predecessor_mutables"] == {
        "STATUS.json": {
            "bytes": 20114,
            "sha256": "80f0b7a987afa27d6be500c7177e03405dd5602c350cf5ff99ed7ecbe0f9def4",
            "snapshot_name": "STATUS_PRE_GSE114002_ENDPOINT_GEOMETRY_RECONCILIATION_V2_RUNTIME_SYNC_V1.json",
        },
        "RUN_MANIFEST.json": {
            "bytes": 43342,
            "sha256": "389f6efeb10af0a6d9d9907736d668c016bc913682b879391f853761a514adfd",
            "snapshot_name": "RUN_MANIFEST_PRE_GSE114002_ENDPOINT_GEOMETRY_RECONCILIATION_V2_RUNTIME_SYNC_V1.json",
        },
        "EVENT_LOG.jsonl": {
            "bytes": 45298,
            "sha256": "fcfb4155191548f228fe823552641906a2abedc9321401da1822e9a1bdf9b496",
            "snapshot_name": "EVENT_LOG_PRE_GSE114002_ENDPOINT_GEOMETRY_RECONCILIATION_V2_RUNTIME_SYNC_V1.jsonl",
        },
    }
    assert runtime["predecessor_tail_event"]["sha256"] == (
        "f532c7ade9f60898bd993739c779326d2c6990def17760c92395029c5684b423"
    )
    assert runtime["attempts"] == runtime_sync.EXPECTED_ATTEMPTS
    assert config["unresolved_blockers"] == runtime_sync.EXPECTED_BLOCKERS
    unknown = unknown_config(config)
    bound = config if config["implementation_binding"]["status"] == "BOUND" else bind_config(unknown)
    runtime_sync.validate_bound_config(bound)
    assert runtime_sync.expected_unknown_i_config(bound) == unknown
    assert runtime_sync.compiled_core_projection(bound) == (
        runtime_sync.compiled_core_projection(unknown)
    )


def test_frozen_focused_test_remains_valid_after_config_only_b(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bound_path = tmp_path / CONFIG_PATH.name
    bound_path.write_bytes(runtime_sync.json_bytes(bind_config(read_config())))
    monkeypatch.setattr(sys.modules[__name__], "CONFIG_PATH", bound_path)
    test_static_config_supports_i_or_b_and_freezes_evt038_and_two_attempts()


@pytest.mark.parametrize(
    "drift",
    [
        "root_extra",
        "privacy_true",
        "attempt_hash",
        "attempt_order",
        "training_integer",
        "ledger_parent",
        "publication_extra",
    ],
)
def test_bound_config_is_closed_typed_and_frozen(drift: str) -> None:
    config = bind_config(unknown_config())
    if drift == "root_extra":
        config["extra"] = False
    elif drift == "privacy_true":
        config["access_and_materialization_boundary"]["real_row_level_data_opened"] = True
    elif drift == "attempt_hash":
        config["runtime"]["attempts"][0]["members"]["PUBLICATION_COMMIT.json"][
            "sha256"
        ] = "0" * 64
    elif drift == "attempt_order":
        config["runtime"]["attempts"].reverse()
    elif drift == "training_integer":
        config["successor_invariants"]["training_started"] = 0
    elif drift == "ledger_parent":
        config["repository_authority"]["base_commit_expected_parent"] = "0" * 40
    else:
        config["publication_policy"]["extra"] = False
    refresh_core(config)
    with pytest.raises(runtime_sync.RuntimeSyncError):
        runtime_sync.validate_bound_config(config)


def test_unknown_binding_stops_before_runtime_or_attempt_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened = False

    def forbidden(_path: Path) -> int:
        nonlocal opened
        opened = True
        raise AssertionError("runtime or attempt accessed")

    monkeypatch.setattr(runtime_sync, "open_directory", forbidden)
    monkeypatch.setattr(runtime_sync, "open_absolute_directory_nofollow", forbidden)
    with pytest.raises(runtime_sync.BindingError, match="not BOUND"):
        runtime_sync.prepare_runtime_sync(
            prepared_directory=tmp_path / "prepared",
            recorded_at="2026-08-11T08:30:00+08:00",
            production=False,
            config_override=unknown_config(),
            authority_override={},
            run_root_override=tmp_path / "run",
        )
    assert opened is False


@pytest.mark.parametrize(
    "mode", ["positive", "dirty", "non_config_binding", "script_drift", "lineage_drift"]
)
def test_repo_binding_audit_requires_exact_lineage_and_direct_config_only_b(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    config = bind_config(unknown_config())
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr(runtime_sync, "PRODUCTION_REPO_ROOT", repo)
    authority, binding = config["repository_authority"], config["implementation_binding"]
    implementation, base, head = binding["implementation_commit"], authority["base_commit"], "4" * 40
    blobs = {
        runtime_sync.SCRIPT_REPO_PATH: b"synthetic runtime sync script\n",
        runtime_sync.TEST_REPO_PATH: b"synthetic runtime sync tests\n",
    }
    binding["implementation_script_sha256"] = runtime_sync.sha256(
        blobs[runtime_sync.SCRIPT_REPO_PATH]
    )
    binding["implementation_test_sha256"] = runtime_sync.sha256(
        blobs[runtime_sync.TEST_REPO_PATH]
    )
    refresh_core(config)
    config_payload = runtime_sync.json_bytes(config)
    i_payload = runtime_sync.json_bytes(runtime_sync.expected_unknown_i_config(config))
    lineage = authority["gse114002_v2_attempt_commit_lineage"]
    parent_map = {
        head: implementation,
        implementation: base,
        base: lineage[1]["binding_commit"],
        lineage[1]["binding_commit"]: lineage[1]["implementation_commit"],
        lineage[1]["implementation_commit"]: lineage[0]["binding_commit"],
        lineage[0]["binding_commit"]: lineage[0]["implementation_commit"],
    }

    def fake_git(
        _repo: Path, *args: str, allowed_returncodes: tuple[int, ...] = (0,)
    ) -> bytes:
        del allowed_returncodes
        if args == ("rev-parse", "HEAD"):
            return f"{head}\n".encode()
        if args == ("rev-parse", "--verify", f"refs/remotes/origin/{authority['branch']}"):
            return f"{head}\n".encode()
        if args == ("rev-parse", "--abbrev-ref", "HEAD"):
            return f"{authority['branch']}\n".encode()
        if args == ("rev-parse", "--abbrev-ref", "@{upstream}"):
            return f"origin/{authority['branch']}\n".encode()
        if args == ("rev-parse", "@{upstream}"):
            return f"{head}\n".encode()
        if len(args) == 2 and args[0] == "rev-parse" and args[1].endswith("^"):
            commit = args[1][:-1]
            if mode == "lineage_drift" and commit == base:
                return ("9" * 40 + "\n").encode()
            return f"{parent_map[commit]}\n".encode()
        if args == ("status", "--porcelain=v1", "--untracked-files=all"):
            return b"M dirty\n" if mode == "dirty" else b""
        if args[:2] == ("merge-base", "--is-ancestor"):
            return b""
        if args[:4] == ("diff-tree", "--no-commit-id", "--name-only", "-r"):
            paths = (
                authority["implementation_commit_exact_changed_paths"]
                if args[4] == implementation
                else authority["binding_commit_exact_changed_paths"]
            )
            if mode == "non_config_binding" and args[4] == head:
                paths = paths + [runtime_sync.SCRIPT_REPO_PATH]
            return ("\n".join(paths) + "\n").encode()
        if args[0] == "show":
            commit, path = args[1].split(":", 1)
            if path == runtime_sync.CONFIG_REPO_PATH:
                return config_payload if commit == head else i_payload
            payload = blobs[path]
            if mode == "script_drift" and path == runtime_sync.SCRIPT_REPO_PATH and commit == implementation:
                return payload + b"drift"
            return payload
        raise AssertionError(args)

    monkeypatch.setattr(runtime_sync, "_run_git", fake_git)
    monkeypatch.setattr(
        runtime_sync,
        "read_regular_path",
        lambda path: blobs[str(path.relative_to(repo))],
    )
    if mode == "positive":
        result = runtime_sync.audit_repo_authority(repo, config, config_payload)
        assert result["binding_commit"] == head
        assert result["base_commit_parent"] == lineage[1]["binding_commit"]
    else:
        with pytest.raises(runtime_sync.RuntimeSyncError):
            runtime_sync.audit_repo_authority(repo, config, config_payload)


def test_prepare_registers_twelve_attempt_members_then_snapshots_and_sync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, predecessor, _authority, _run_root, prepared = prepare_context(
        tmp_path, monkeypatch
    )
    artifacts = {item.name: item.read_bytes() for item in prepared.iterdir()}
    manifest = runtime_sync.load_json(artifacts["RUN_MANIFEST.json"], label="manifest")
    old_manifest = runtime_sync.load_json(
        predecessor["RUN_MANIFEST.json"], label="old manifest"
    )
    assert len(manifest["outputs"]) == 122
    assert manifest["outputs"][:106] == old_manifest["outputs"]
    delta = manifest["outputs"][106:]
    assert len(delta) == 16
    assert [Path(item["absolute_path"]).name for item in delta[:12]] == list(
        runtime_sync.BUNDLE_NAMES
    ) * 2
    assert [Path(item["absolute_path"]).name for item in delta[12:]] == [
        "STATUS_PRE_GSE114002_ENDPOINT_GEOMETRY_RECONCILIATION_V2_RUNTIME_SYNC_V1.json",
        "RUN_MANIFEST_PRE_GSE114002_ENDPOINT_GEOMETRY_RECONCILIATION_V2_RUNTIME_SYNC_V1.json",
        "EVENT_LOG_PRE_GSE114002_ENDPOINT_GEOMETRY_RECONCILIATION_V2_RUNTIME_SYNC_V1.jsonl",
        "A1_GSE114002_ENDPOINT_GEOMETRY_RECONCILIATION_V2_RUNTIME_SYNC_V1.json",
    ]
    sync = runtime_sync.load_json(
        artifacts[config["runtime"]["sync_name"]], label="sync"
    )
    assert [item["status"] for item in sync["attempt_lineage"]] == [
        runtime_sync.FAILED_MECHANICAL_STATUS,
        runtime_sync.CURRENT_MECHANICAL_STATUS,
    ]
    assert all(item["terminal_commit"]["committed"] for item in sync["attempt_lineage"])
    assert sync["scientific_blockers"] == {
        "count": 7,
        "exact": runtime_sync.EXPECTED_BLOCKERS,
    }
    events = runtime_sync.load_json_lines(artifacts["EVENT_LOG.jsonl"], label="events")
    assert len(events) == 39
    assert events[:-1] == runtime_sync.load_json_lines(
        predecessor["EVENT_LOG.jsonl"], label="old events"
    )
    assert artifacts["EVENT_LOG.jsonl"].startswith(predecessor["EVENT_LOG.jsonl"])
    event = events[-1]
    assert event["event_id"] == "A1-EVT-039"
    assert event["manifest_output_count_before"] == 106
    assert event["manifest_output_count_after"] == 122
    assert event["training_started"] is False
    assert event["training_allowed"] is False
    assert event["model_selection_allowed"] is False
    assert event["next_phase_authorized"] is False
    assert event["gpu_work_started"] is False
    assert event["qualifier_execution_count"] == 0
    assert event["canonical_intervention_record_count"] == 0
    assert event["aggregate_reconciliation_artifact_registration_count"] == 12


@pytest.mark.parametrize("drift", ["marker_truth", "report_commit", "sha256sums"])
def test_attempt_marker_commit_and_checksum_truth_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, drift: str
) -> None:
    config, _predecessor, _authority, _run_root, _prepared = make_context(
        tmp_path, monkeypatch
    )
    attempt = config["runtime"]["attempts"][1]
    root = Path(attempt["run_root"])
    if drift == "report_commit":
        for item in root.iterdir():
            item.unlink()
        payloads = materialize_attempt(attempt, binding_commit="9" * 40)
        del payloads
    else:
        marker_path = root / "PUBLICATION_COMMIT.json"
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if drift == "marker_truth":
            marker["committed"] = False
        else:
            sums_path = root / "SHA256SUMS"
            lines = sums_path.read_text(encoding="ascii").splitlines()
            lines[0] = "0" * 64 + lines[0][64:]
            sums_payload = ("\n".join(lines) + "\n").encode("ascii")
            sums_path.write_bytes(sums_payload)
            attempt["members"]["SHA256SUMS"] = {
                "bytes": len(sums_payload),
                "sha256": runtime_sync.sha256(sums_payload),
            }
            marker["sha256sums_sha256"] = runtime_sync.sha256(sums_payload)
        marker_payload = runtime_sync.json_bytes(marker)
        marker_path.write_bytes(marker_payload)
        attempt["members"]["PUBLICATION_COMMIT.json"] = {
            "bytes": len(marker_payload),
            "sha256": runtime_sync.sha256(marker_payload),
        }
    refresh_core(config)
    with pytest.raises(runtime_sync.RuntimeSyncError):
        runtime_sync.read_exact_attempt_lineage(config)


def test_evt038_tail_has_independent_exact_line_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, predecessor, _authority, run_root, _prepared = make_context(
        tmp_path, monkeypatch
    )
    events = runtime_sync.load_json_lines(predecessor["EVENT_LOG.jsonl"], label="events")
    events[-1]["extra"] = False
    changed = b"".join(runtime_sync.compact_json_line(item) for item in events)
    (run_root / "EVENT_LOG.jsonl").write_bytes(changed)
    config["runtime"]["predecessor_mutables"]["EVENT_LOG.jsonl"]["bytes"] = len(changed)
    config["runtime"]["predecessor_mutables"]["EVENT_LOG.jsonl"]["sha256"] = (
        runtime_sync.sha256(changed)
    )
    descriptor = runtime_sync.open_directory(run_root)
    try:
        with pytest.raises(runtime_sync.PublicationError, match="tail line identity"):
            runtime_sync.read_exact_predecessor(descriptor, config)
    finally:
        runtime_sync.os.close(descriptor)


@pytest.mark.parametrize(
    "drift",
    [
        "missing_gate",
        "missing_access",
        "true_gate",
        "true_gpu",
        "nonzero_qualifier",
        "bool_as_integer",
        "integer_as_bool",
        "extra_key",
    ],
)
def test_evt039_closed_event_rejects_missing_true_nonzero_and_type_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, drift: str
) -> None:
    config, predecessor, _authority, _run_root, prepared = prepare_context(
        tmp_path, monkeypatch
    )
    artifacts = {item.name: item.read_bytes() for item in prepared.iterdir()}
    events = runtime_sync.load_json_lines(artifacts["EVENT_LOG.jsonl"], label="events")
    if drift == "missing_gate":
        events[-1].pop("run_status")
    elif drift == "missing_access":
        events[-1].pop("real_row_level_data_opened")
    elif drift == "true_gate":
        events[-1]["training_allowed"] = True
    elif drift == "true_gpu":
        events[-1]["gpu_work_started"] = True
    elif drift == "nonzero_qualifier":
        events[-1]["qualifier_execution_count"] = 1
    elif drift == "bool_as_integer":
        events[-1]["qualifier_execution_count"] = False
    elif drift == "integer_as_bool":
        events[-1]["training_started"] = 0
    else:
        events[-1]["unexpected"] = False
    artifacts["EVENT_LOG.jsonl"] = predecessor["EVENT_LOG.jsonl"] + (
        runtime_sync.compact_json_line(events[-1])
    )
    with pytest.raises(runtime_sync.RuntimeSyncError, match="EVT-039 closed event"):
        runtime_sync.validate_successors(
            config,
            artifacts,
            predecessor,
            runtime_sync.load_json(predecessor["STATUS.json"], label="status"),
            runtime_sync.load_json(
                predecessor["RUN_MANIFEST.json"], label="manifest"
            ),
            runtime_sync.sha256(artifacts[config["runtime"]["sync_name"]]),
        )


def test_first_publish_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _predecessor, authority, run_root, prepared = prepare_context(
        tmp_path, monkeypatch
    )
    first = runtime_sync.publish_prepared(
        prepared_directory=prepared,
        production=False,
        config_override=config,
        authority_override=authority,
        run_root_override=run_root,
    )
    assert first["status"] == "PUBLISHED_VERIFIED"
    after = tree_bytes(run_root)
    second = runtime_sync.publish_prepared(
        prepared_directory=prepared,
        production=False,
        config_override=config,
        authority_override=authority,
        run_root_override=run_root,
    )
    assert second["status"] == "PUBLISHED_VERIFIED"
    assert tree_bytes(run_root) == after


@pytest.mark.parametrize("new_count", [0, 1, 2, 3])
def test_four_allowed_mutable_recovery_prefixes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, new_count: int
) -> None:
    config, _predecessor, authority, run_root, prepared = prepare_context(
        tmp_path, monkeypatch
    )
    for name in runtime_sync.MUTABLE_NAMES[:new_count]:
        (run_root / name).write_bytes((prepared / name).read_bytes())
    result = runtime_sync.publish_prepared(
        prepared_directory=prepared,
        production=False,
        config_override=config,
        authority_override=authority,
        run_root_override=run_root,
    )
    assert result["status"] == "PUBLISHED_VERIFIED"
    assert [result["mutable_preflight"][name] for name in runtime_sync.MUTABLE_NAMES] == (
        ["NEW_EXACT"] * new_count + ["OLD_EXACT"] * (3 - new_count)
    )


def test_event_is_last_and_post_commit_warning_recovers_idempotently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _predecessor, authority, run_root, prepared = prepare_context(
        tmp_path, monkeypatch
    )
    attempts = 0

    def inject(point: str) -> None:
        nonlocal attempts
        if point == "mutable_post_replace_directory_fsync":
            attempts += 1
            if attempts == 3:
                raise OSError("EVENT fsync fault")

    warning = runtime_sync.publish_prepared(
        prepared_directory=prepared,
        production=False,
        config_override=config,
        authority_override=authority,
        run_root_override=run_root,
        fault_injector=inject,
    )
    assert warning["status"] == "COMMITTED_WITH_WARNINGS_REQUIRES_IDEMPOTENT_RETRY"
    assert warning["warning_member"] == "EVENT_LOG.jsonl"
    recovered = runtime_sync.publish_prepared(
        prepared_directory=prepared,
        production=False,
        config_override=config,
        authority_override=authority,
        run_root_override=run_root,
    )
    assert recovered["status"] == "PUBLISHED_VERIFIED"
    events = runtime_sync.load_json_lines(
        (run_root / "EVENT_LOG.jsonl").read_bytes(), label="published events"
    )
    assert events[-1]["event_id"] == "A1-EVT-039"


def test_immutable_temp_unlink_failure_is_committed_manual_truth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _predecessor, authority, run_root, prepared = prepare_context(
        tmp_path, monkeypatch
    )
    injected = False

    def inject(point: str) -> None:
        nonlocal injected
        if point == "immutable_post_link_unlink" and not injected:
            injected = True
            raise OSError("unlink fault")

    result = runtime_sync.publish_prepared(
        prepared_directory=prepared,
        production=False,
        config_override=config,
        authority_override=authority,
        run_root_override=run_root,
        fault_injector=inject,
    )
    assert result["status"] == "COMMITTED_REQUIRES_MANUAL_TEMP_ADJUDICATION"
    stale = [item.name for item in run_root.iterdir() if item.name.startswith(".evt039.")]
    assert len(stale) == 1
    with pytest.raises(runtime_sync.PublicationError, match="stale EVT-039 publisher"):
        runtime_sync.publish_prepared(
            prepared_directory=prepared,
            production=False,
            config_override=config,
            authority_override=authority,
            run_root_override=run_root,
        )
