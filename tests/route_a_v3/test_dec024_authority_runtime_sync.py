from __future__ import annotations

import copy
import hashlib
import importlib.util
from pathlib import Path
from typing import Any

import pytest


STAGING_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = STAGING_ROOT / "configs/route_a_v3_dec024_authority_runtime_sync_v1.json"
SCRIPT_PATH = STAGING_ROOT / "scripts/route_a_v3/dec024_authority_runtime_sync.py"
SPEC = importlib.util.spec_from_file_location("dec024_authority_runtime_sync", SCRIPT_PATH)
assert SPEC and SPEC.loader
SYNC = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SYNC)


def disk_config() -> dict[str, Any]:
    return SYNC.load_config(CONFIG_PATH, require_bound=False)


def unknown_config() -> dict[str, Any]:
    config = SYNC.expected_unknown_i_config(copy.deepcopy(disk_config()))
    SYNC.validate_static_config(config)
    return config


def bound_config() -> dict[str, Any]:
    config = unknown_config()
    config["implementation_binding"].update(
        {
            "status": SYNC.BOUND,
            "implementation_commit": "6" * 40,
            "implementation_script_sha256": "7" * 64,
            "implementation_test_sha256": "8" * 64,
        }
    )
    SYNC.validate_static_config(config)
    return config


def outer_runtime_fields() -> dict[str, Any]:
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


def synthetic_predecessor(
    config: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> dict[str, bytes]:
    prior_decisions = [
        "V3-DEC-017",
        "V3-DEC-018",
        "V3-DEC-019",
        "V3-DEC-020",
        "V3-DEC-021",
        "V3-DEC-022",
        "V3-DEC-023",
    ]
    status = {
        **outer_runtime_fields(),
        "updated_at": "2026-08-14T13:45:38+08:00",
        "active_amendment_decision_ids": prior_decisions,
        "existing_status_field": "PRESERVED",
    }
    manifest = {
        **outer_runtime_fields(),
        "active_authority_commit": "9" * 40,
        "active_amendment_decision_ids": prior_decisions,
        "registered_artifact_count": 8,
        "outputs": [
            {
                "absolute_path": f"/existing/{index:03d}.json",
                "artifact_type": "EXISTING_FIXTURE",
                "bytes": index + 1,
                "sha256": f"{index + 1:064x}",
            }
            for index in range(248)
        ],
        "existing_manifest_field": {"preserve": True},
    }
    events = [
        {
            "event_id": f"A1-EVT-{index:03d}",
            "at": "2026-08-13T00:00:00+08:00",
            "event": "HISTORICAL_FIXTURE",
        }
        for index in range(1, 57)
    ]
    events.append(
        {
            "event_id": "A1-EVT-057",
            "at": "2026-08-14T13:45:38+08:00",
            "event": "DEC023_DUAL_AGGREGATE_ONLY_PREFLIGHT_EVIDENCE_REGISTERED",
            "decision_id": "V3-DEC-023",
            "predecessor_event_id": "A1-EVT-056",
        }
    )
    payloads = {
        "STATUS.json": SYNC.json_bytes(status),
        "RUN_MANIFEST.json": SYNC.json_bytes(manifest),
        "EVENT_LOG.jsonl": b"".join(SYNC.compact_json_line(event) for event in events),
    }
    identities = {
        name: {
            "bytes": len(payload),
            "sha256": SYNC.sha256(payload),
            "snapshot_name": config["runtime"]["predecessor_mutables"][name][
                "snapshot_name"
            ],
        }
        for name, payload in payloads.items()
    }
    tail_payload = SYNC.compact_json_line(events[-1])
    tail = {
        "event_id": "A1-EVT-057",
        "decision_id": "V3-DEC-023",
        "bytes": len(tail_payload),
        "sha256": SYNC.sha256(tail_payload),
    }
    config["runtime"]["predecessor_mutables"] = copy.deepcopy(identities)
    config["runtime"]["predecessor_tail"] = copy.deepcopy(tail)
    monkeypatch.setattr(SYNC, "PREDECESSOR_IDENTITIES", identities)
    monkeypatch.setattr(SYNC, "PREDECESSOR_TAIL", tail)
    SYNC.validate_static_config(config)
    return payloads


def make_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, Any], dict[str, bytes], Path, Path]:
    config = bound_config()
    run_root = tmp_path / "run"
    allowed_root = tmp_path / "prepared-root"
    prepared = allowed_root / "evt058-job"
    run_root.mkdir()
    allowed_root.mkdir()
    config["runtime"]["run_root"] = str(run_root)
    config["runtime"]["allowed_prepared_root"] = str(allowed_root)
    predecessor = synthetic_predecessor(config, monkeypatch)
    for name, payload in predecessor.items():
        (run_root / name).write_bytes(payload)
    return config, predecessor, run_root, prepared


def read_runtime(run_root: Path) -> dict[str, bytes]:
    return {name: (run_root / name).read_bytes() for name in SYNC.MUTABLE_NAMES}


def install_fake_repository(
    monkeypatch: pytest.MonkeyPatch,
    config: dict[str, Any],
    *,
    drift_authority_paths: bool = False,
    drift_g_blob: bool = False,
    drift_i_config: bool = False,
) -> bytes:
    authority = config["repository_authority"]
    authority_commit = authority["authority_commit"]
    g_commit = authority["predecessor_nonauthoritative_a6_g0"]["commit"]
    implementation_commit = "6" * 40
    binding_commit = "b" * 40
    script_payload = b"DEC024 authority runtime sync producer I\n"
    test_payload = b"DEC024 authority runtime sync focused test I\n"
    config["implementation_binding"].update(
        {
            "status": SYNC.BOUND,
            "implementation_commit": implementation_commit,
            "implementation_script_sha256": hashlib.sha256(script_payload).hexdigest(),
            "implementation_test_sha256": hashlib.sha256(test_payload).hexdigest(),
        }
    )
    SYNC.validate_static_config(config)
    config_payload = SYNC.json_bytes(config)
    i_payload = SYNC.json_bytes(SYNC.expected_unknown_i_config(config))
    if drift_i_config:
        i_payload += b"drift"

    authority_payloads: dict[str, bytes] = {}
    g_payloads: dict[str, bytes] = {}
    digest_overrides: dict[bytes, str] = {}
    for index, item in enumerate(authority["authority_files"], start=1):
        seed = f"DEC024 authority blob {index}\n".encode()
        payload = (seed * (item["bytes"] // len(seed) + 1))[: item["bytes"]]
        authority_payloads[item["path"]] = payload
        digest_overrides[payload] = item["sha256"]
    for index, item in enumerate(SYNC.FROZEN_G_FILES, start=1):
        seed = f"A6 G0 frozen blob {index}\n".encode()
        payload = (seed * (item["bytes"] // len(seed) + 1))[: item["bytes"]]
        g_payloads[item["path"]] = payload
        digest_overrides[payload] = item["sha256"]
    real_sha256 = SYNC.sha256

    def fake_sha256(payload: bytes) -> str:
        return digest_overrides.get(payload, real_sha256(payload))

    changed = {
        authority_commit: list(SYNC.AUTHORITY_PATHS),
        g_commit: list(SYNC.FROZEN_G_PATHS),
        implementation_commit: list(SYNC.IMPLEMENTATION_PATHS),
        binding_commit: [SYNC.CONFIG_REPO_PATH],
    }
    if drift_authority_paths:
        changed[authority_commit].append("unexpected/path")

    blobs: dict[tuple[str, str], bytes] = {}
    for commit in (authority_commit, g_commit, implementation_commit, binding_commit):
        for relative, payload in authority_payloads.items():
            blobs[(commit, relative)] = payload
    for commit in (g_commit, implementation_commit, binding_commit):
        for relative, payload in g_payloads.items():
            blobs[(commit, relative)] = payload
    blobs.update(
        {
            (implementation_commit, SYNC.CONFIG_REPO_PATH): i_payload,
            (implementation_commit, SYNC.SCRIPT_REPO_PATH): script_payload,
            (implementation_commit, SYNC.TEST_REPO_PATH): test_payload,
            (binding_commit, SYNC.CONFIG_REPO_PATH): config_payload,
            (binding_commit, SYNC.SCRIPT_REPO_PATH): script_payload,
            (binding_commit, SYNC.TEST_REPO_PATH): test_payload,
        }
    )
    if drift_g_blob:
        blobs[(g_commit, SYNC.FROZEN_G_PATHS[0])] = b"drifted G blob"

    def fake_git(_repo: Path, *arguments: str) -> bytes:
        mapping = {
            ("rev-parse", "HEAD"): f"{binding_commit}\n".encode(),
            ("rev-parse", "@{upstream}"): f"{binding_commit}\n".encode(),
            (
                "rev-parse",
                "--verify",
                f"refs/remotes/origin/{SYNC.BRANCH}",
            ): f"{binding_commit}\n".encode(),
            ("rev-parse", "--abbrev-ref", "HEAD"): f"{SYNC.BRANCH}\n".encode(),
            (
                "rev-parse",
                "--abbrev-ref",
                "@{upstream}",
            ): f"origin/{SYNC.BRANCH}\n".encode(),
            ("status", "--porcelain=v1", "--untracked-files=all"): b"",
            ("rev-parse", f"{binding_commit}^"): f"{implementation_commit}\n".encode(),
            ("rev-parse", f"{implementation_commit}^"): f"{g_commit}\n".encode(),
            ("rev-parse", f"{g_commit}^"): f"{authority_commit}\n".encode(),
            (
                "rev-parse",
                f"{authority_commit}^",
            ): f"{SYNC.AUTHORITY_PARENT}\n".encode(),
        }
        if arguments in mapping:
            return mapping[arguments]
        if arguments[:4] == ("diff-tree", "--no-commit-id", "--name-only", "-r"):
            return ("\n".join(changed[arguments[4]]) + "\n").encode()
        if arguments[0] == "show":
            commit, relative = arguments[1].split(":", 1)
            return blobs[(commit, relative)]
        raise AssertionError(arguments)

    worktree = {
        **authority_payloads,
        **g_payloads,
        SYNC.CONFIG_REPO_PATH: config_payload,
        SYNC.SCRIPT_REPO_PATH: script_payload,
        SYNC.TEST_REPO_PATH: test_payload,
    }
    monkeypatch.setattr(
        SYNC, "__file__", str(SYNC.PRODUCTION_REPO_ROOT / SYNC.SCRIPT_REPO_PATH)
    )
    monkeypatch.setattr(SYNC, "sha256", fake_sha256)
    monkeypatch.setattr(SYNC, "_run_git", fake_git)
    monkeypatch.setattr(
        SYNC,
        "_read_repo_file",
        lambda _repo, relative: worktree[relative],
    )
    return config_payload


def test_disk_candidate_is_strict_i_or_b_with_bound_a_g_and_evt057() -> None:
    config = disk_config()
    SYNC.validate_static_config(config)
    authority = config["repository_authority"]
    assert SYNC._authority_binding_state(authority) == "BOUND"
    assert authority["authority_commit"] == "0bb84dffb1389b9eced7e92e36ef80b8a97ed0be"
    assert authority["authority_expected_parent"] == SYNC.AUTHORITY_PARENT
    assert authority["predecessor_nonauthoritative_a6_g0"] == SYNC.FROZEN_G
    assert len(authority["authority_files"]) == 12
    runtime = config["runtime"]
    assert runtime["predecessor_event_id"] == "A1-EVT-057"
    assert runtime["predecessor_event_count"] == 57
    assert runtime["predecessor_manifest_output_count"] == 248
    assert runtime["predecessor_manifest_registered_artifact_count"] == 8
    assert runtime["successor_event_id"] == "A1-EVT-058"
    assert runtime["successor_manifest_output_count"] == 252
    assert runtime["predecessor_mutables"] == SYNC.PREDECESSOR_IDENTITIES
    assert runtime["predecessor_tail"] == SYNC.PREDECESSOR_TAIL
    assert config["registered_artifacts"] == []
    binding = config["implementation_binding"]
    state = SYNC._implementation_binding_state(binding)
    assert state in {"UNKNOWN", "BOUND"}
    normalized = SYNC.expected_unknown_i_config(config)
    SYNC.validate_static_config(normalized)
    if state == "UNKNOWN":
        assert normalized == config
    else:
        assert binding["implementation_script_sha256"] == SYNC.sha256(
            SCRIPT_PATH.read_bytes()
        )
        assert binding["implementation_test_sha256"] == SYNC.sha256(
            Path(__file__).read_bytes()
        )


def test_unknown_i_stops_before_git_prepared_or_runtime_io(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = unknown_config()
    touched: list[str] = []

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        touched.append("I/O")
        raise AssertionError("external I/O reached")

    monkeypatch.setattr(SYNC, "audit_production_repository_authority", forbidden)
    monkeypatch.setattr(SYNC, "_prepared_path", forbidden)
    monkeypatch.setattr(SYNC, "_read_runtime", forbidden)
    monkeypatch.setattr(SYNC, "_write_prepared", forbidden)
    with pytest.raises(SYNC.BindingError, match="not BOUND"):
        SYNC.prepare_runtime_sync(
            prepared_directory=tmp_path / "must-not-exist",
            recorded_at="2026-08-14T18:30:00+08:00",
            production=False,
            config_override=config,
            run_root_override=tmp_path / "must-not-open",
        )
    assert touched == []
    assert not (tmp_path / "must-not-exist").exists()
    assert not (tmp_path / "must-not-open").exists()


def test_stale_copied_producer_rejected_before_git_io(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = bound_config()
    stale_script = tmp_path / SCRIPT_PATH.name
    stale_script.write_bytes(SCRIPT_PATH.read_bytes())
    stale_spec = importlib.util.spec_from_file_location("stale_dec024", stale_script)
    assert stale_spec and stale_spec.loader
    stale = importlib.util.module_from_spec(stale_spec)
    stale_spec.loader.exec_module(stale)
    touched: list[str] = []

    def forbidden(*_args: Any, **_kwargs: Any) -> bytes:
        touched.append("git")
        raise AssertionError("git reached")

    monkeypatch.setattr(stale, "_run_git", forbidden)
    with pytest.raises(stale.AuthorityError, match="executing producer"):
        stale.audit_production_repository_authority(config, stale.json_bytes(config))
    assert touched == []


def test_production_config_copy_is_rejected_before_config_or_runtime_io(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    touched: list[str] = []

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        touched.append("I/O")
        raise AssertionError("I/O reached")

    monkeypatch.setattr(SYNC, "_load_config_payload", forbidden)
    monkeypatch.setattr(SYNC, "_read_runtime", forbidden)
    with pytest.raises(SYNC.BindingError, match="executing repository config"):
        SYNC._context(
            tmp_path / CONFIG_PATH.name,
            production=True,
            config_override=None,
        )
    assert touched == []


def test_repository_a_g_i_b_lifecycle_and_blobs_are_audited(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = bound_config()
    payload = install_fake_repository(monkeypatch, config)
    audit = SYNC.audit_production_repository_authority(config, payload)
    assert audit == {
        "status": "PASS_EXACT12_A_NONAUTHORITATIVE_EXACT4_G_EXACT3_I_CONFIG_ONLY_B",
        "authority_commit": config["repository_authority"]["authority_commit"],
        "nonauthoritative_a6_g0_commit": SYNC.FROZEN_G_COMMIT,
        "implementation_commit": "6" * 40,
        "binding_commit": "b" * 40,
        "head_commit": "b" * 40,
        "upstream_head_commit": "b" * 40,
        "origin_branch_head_commit": "b" * 40,
        "authority_blob_count": 12,
        "nonauthoritative_a6_g0_blob_count": 4,
        "worktree_and_index_clean": True,
    }


@pytest.mark.parametrize(
    ("drift_flag", "message"),
    [
        ("drift_authority_paths", "A exact12 drift"),
        ("drift_g_blob", "frozen non-authoritative G exact4 blob"),
        ("drift_i_config", "invalid JSON"),
    ],
)
def test_repository_chain_drift_is_rejected(
    monkeypatch: pytest.MonkeyPatch, drift_flag: str, message: str
) -> None:
    config = bound_config()
    payload = install_fake_repository(
        monkeypatch,
        config,
        **{drift_flag: True},
    )
    with pytest.raises(SYNC.RuntimeSyncError, match=message):
        SYNC.audit_production_repository_authority(config, payload)


def test_authority_and_science_boundaries_cannot_be_relaxed() -> None:
    config = disk_config()
    drifts = [
        (("dec024_authority", "gse261709", "raw_fastq_or_sra_member_payload_read_allowed"), True),
        (("dec024_authority", "gse269595", "maximum_roles_if_later_qualified"), 2),
        (("dec024_authority", "gse269595", "a1_and_true_a2_double_credit_allowed"), True),
        (("dec024_authority", "emtab10902", "row_count_may_substitute_for_independent_source_group_n"), True),
        (("dec024_authority", "emtab10902", "prefrozen_required_effective_n_for_power_and_full_ci_width"), 16),
        (("dec024_authority", "qualification_allowed"), True),
        (("frozen_outer_truth", "training_allowed"), True),
        (("access_boundary", "row_payload_read_count"), 1),
    ]
    for keys, value in drifts:
        drift = copy.deepcopy(config)
        cursor: dict[str, Any] = drift
        for key in keys[:-1]:
            cursor = cursor[key]
        cursor[keys[-1]] = value
        with pytest.raises(SYNC.RuntimeSyncError):
            SYNC.validate_static_config(drift)


def test_successor_is_exact_evt058_and_preserves_all_science_locks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = bound_config()
    predecessor = synthetic_predecessor(config, monkeypatch)
    old_status, old_manifest, _ = SYNC._parse_runtime(predecessor)
    successors = SYNC.build_successors(
        config, predecessor, "2026-08-14T18:30:00+08:00"
    )
    status, manifest, events = SYNC._parse_runtime(
        {name: successors[name] for name in SYNC.MUTABLE_NAMES}
    )
    assert len(successors) == 7
    assert len(events) == 58
    event = events[-1]
    assert event["event_id"] == "A1-EVT-058"
    assert event["decision_id"] == "V3-DEC-024"
    assert event["registered_artifacts"] == []
    assert event["new_registered_artifact_count"] == 0
    assert event["preflight_executed"] is False
    assert event["scientific_state_changed"] is False
    assert event["qualification_changed"] is False
    assert event["qualified"] is False
    assert event["qualification_allowed"] is False
    assert event["canonical_materialization_allowed"] is False
    assert event["training_authorized"] is False
    assert event["a7_allowed"] is False
    assert event["frozen_outer_truth"]["current_qualified_counts"] == {
        "ordinary": 1,
        "a1": 1,
        "true_a2": 0,
        "canonical_records": 6547,
    }
    assert event["dec024_authority"]["gse269595"][
        "a1_and_true_a2_double_credit_allowed"
    ] is False
    assert event["dec024_authority"]["emtab10902"][
        "reported_source_group_count_approximate"
    ] == 16
    assert event["dec024_authority"]["emtab10902"][
        "reported_qc_design_row_count_reference_only"
    ] == 5679
    assert event["dec024_authority"]["emtab10902"][
        "prefrozen_required_effective_n_for_power_and_full_ci_width"
    ] == 156
    assert len(manifest["outputs"]) == 252
    assert manifest["outputs"][:248] == old_manifest["outputs"]
    assert manifest["registered_artifact_count"] == 8
    assert manifest["active_authority_commit"] == old_manifest["active_authority_commit"]
    assert status["existing_status_field"] == old_status["existing_status_field"]
    for dataset in ("gse261709", "gse269595", "emtab10902"):
        assert status[f"{dataset}_contribution"] == {
            "ordinary": 0,
            "a1": 0,
            "true_a2": 0,
            "canonical_records": 0,
        }
    assert status["qualified"] is False
    assert status["training_allowed"] is False
    assert status["gpu_work_allowed"] is False
    assert status["model_selection_allowed"] is False
    assert status["next_phase_authorized"] is False
    assert status["scientific_claim_status"] == "NOT_ESTABLISHED"
    assert all(
        value == 0
        for value in event["access_boundary"].values()
        if type(value) is int
    )


def test_predecessor_identity_drift_stops_before_prepared_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, predecessor, run_root, prepared = make_context(tmp_path, monkeypatch)
    (run_root / "STATUS.json").write_bytes(predecessor["STATUS.json"] + b"drift")
    with pytest.raises(SYNC.PredecessorError, match="identity drift"):
        SYNC.prepare_runtime_sync(
            prepared_directory=prepared,
            recorded_at="2026-08-14T18:30:00+08:00",
            production=False,
            config_override=config,
            run_root_override=run_root,
        )
    assert not prepared.exists()


def test_prepare_publish_retries_only_ordered_prefix_and_validates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, predecessor, run_root, prepared = make_context(tmp_path, monkeypatch)
    result = SYNC.prepare_runtime_sync(
        prepared_directory=prepared,
        recorded_at="2026-08-14T18:30:00+08:00",
        production=False,
        config_override=config,
        run_root_override=run_root,
    )
    assert result["status"] == "PREPARED_NOT_PUBLISHED"
    assert result["event_id"] == "A1-EVT-058"
    assert result["manifest_output_transition"] == "248_TO_252"
    assert result["manifest_registered_artifact_transition"] == "8_TO_8"
    assert result["new_registered_artifact_count"] == 0
    assert len(list(prepared.iterdir())) == 7

    triggered = False

    def interrupt_before_manifest(stage: str) -> None:
        nonlocal triggered
        if stage == "before_replace:RUN_MANIFEST.json" and not triggered:
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
    partial = read_runtime(run_root)
    assert partial["STATUS.json"] == (prepared / "STATUS.json").read_bytes()
    assert partial["RUN_MANIFEST.json"] == predecessor["RUN_MANIFEST.json"]
    assert partial["EVENT_LOG.jsonl"] == predecessor["EVENT_LOG.jsonl"]

    published = SYNC.publish_prepared(
        prepared_directory=prepared,
        production=False,
        config_override=config,
        run_root_override=run_root,
    )
    assert published["status"] == "PUBLISHED_VERIFIED"
    assert published["event_id"] == "A1-EVT-058"
    assert SYNC.validate_published(
        prepared_directory=prepared,
        production=False,
        config_override=config,
        run_root_override=run_root,
    ) == {"status": "PUBLISHED_VERIFIED", "event_id": "A1-EVT-058"}
    final = read_runtime(run_root)
    assert len(SYNC.load_events(final["EVENT_LOG.jsonl"], label="final")) == 58
    assert len(SYNC.load_json(final["RUN_MANIFEST.json"], label="final")["outputs"]) == 252
