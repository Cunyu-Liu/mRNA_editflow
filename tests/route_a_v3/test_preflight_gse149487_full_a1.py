from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest


STAGING_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = STAGING_ROOT / "scripts/route_a_v3/preflight_gse149487_full_a1.py"
PROTOCOL_PATH = STAGING_ROOT / "configs/route_a_v3_gse149487_external_evidence_roots_v1.json"

SPEC = importlib.util.spec_from_file_location("preflight_gse149487_full_a1", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
SP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SP)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_payload(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=True) + "\n").encode("utf-8")


def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def _bound_object(config_sha: str, script_sha: str, test_sha: str) -> dict[str, Any]:
    return {
        "binding_scheme": "CONFIG_ONLY_POST_IMPLEMENTATION_BINDING_V1",
        "status": SP.UNBOUND_TOKEN,
        "implementation_commit": SP.UNBOUND_TOKEN,
        "external_evidence_config_path": SP.PREFLIGHT_CONFIG_PATH,
        "external_evidence_config_sha256": config_sha,
        "preflight_script_path": SP.PREFLIGHT_SCRIPT_PATH,
        "preflight_script_sha256": script_sha,
        "preflight_test_path": SP.PREFLIGHT_TEST_PATH,
        "preflight_test_sha256": test_sha,
    }


def _synthetic_repo(
    root: Path,
    *,
    bound: bool = True,
    binding_commit_extra: bool = False,
) -> dict[str, Any]:
    repo = root / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "preflight-test@example.invalid")
    _git(repo, "config", "user.name", "Preflight Test")
    _write(repo / "seed.txt", b"authority seed\n")
    _git(repo, "add", "seed.txt")
    _git(repo, "commit", "-qm", "seed authority")
    accepted_a0_commit = _git(repo, "rev-parse", "HEAD")

    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    root_blob_keys = ("contract", "data_role_registry", "decision_log")
    root_blobs: dict[str, dict[str, str]] = {}
    for key in root_blob_keys:
        item = protocol["authority_bindings"][key]
        payload = f"synthetic immutable authority blob: {key}\n".encode("utf-8")
        item["sha256"] = _sha256(payload)
        _write(repo / item["repo_path"], payload)
        root_blobs[key] = dict(item)
    _git(repo, "add", "--all")
    _git(repo, "commit", "-qm", "activate synthetic authority root")
    active_authority_commit = _git(repo, "rev-parse", "HEAD")
    test_authority_root = {
        "accepted_a0_base_commit": accepted_a0_commit,
        "active_authority_commit": active_authority_commit,
        "active_amendment_decision_ids": ["V3-DEC-017", "V3-DEC-018"],
        "expected_branch": "synthetic-routea-v3-a1",
        "authority_blobs": root_blobs,
    }
    protocol["production_authority_root"] = test_authority_root

    for key, item in protocol["authority_bindings"].items():
        if key in root_blob_keys:
            continue
        payload = f"synthetic authority blob: {key}\n".encode("utf-8")
        item["sha256"] = _sha256(payload)
        _write(repo / item["repo_path"], payload)

    qualifier_script_payload = b"synthetic qualifier code\n"
    qualifier_test_payload = b"synthetic qualifier focused test\n"
    _write(repo / SP.QUALIFIER_SCRIPT_PATH, qualifier_script_payload)
    _write(repo / SP.QUALIFIER_TEST_PATH, qualifier_test_payload)

    script_payload = SCRIPT_PATH.read_bytes()
    test_payload = Path(__file__).read_bytes()
    protocol_payload = _json_payload(protocol)
    _write(repo / SP.PREFLIGHT_CONFIG_PATH, protocol_payload)
    _write(repo / SP.PREFLIGHT_SCRIPT_PATH, script_payload)
    _write(repo / SP.PREFLIGHT_TEST_PATH, test_payload)

    binding = _bound_object(
        _sha256(protocol_payload),
        _sha256(script_payload),
        _sha256(test_payload),
    )
    qualifier = {
        "protocol_id": protocol["qualifier_config_contract"]["protocol_id"],
        "authority": {
            "implementation_commit": SP.UNBOUND_TOKEN,
            "accepted_a0_base_commit": accepted_a0_commit,
            "active_authority_commit": active_authority_commit,
            "active_amendment_decision_ids": ["V3-DEC-017", "V3-DEC-018"],
            "contract_path": root_blobs["contract"]["repo_path"],
            "contract_sha256": root_blobs["contract"]["sha256"],
            "data_role_registry_path": root_blobs["data_role_registry"]["repo_path"],
            "data_role_registry_sha256": root_blobs["data_role_registry"]["sha256"],
            "decision_log_path": root_blobs["decision_log"]["repo_path"],
            "decision_log_sha256": root_blobs["decision_log"]["sha256"],
            "asset_manifest_sha256": protocol["authority_bindings"]["asset_manifest"]["sha256"],
            "qualifier_sha256": _sha256(qualifier_script_payload),
            "focused_test_sha256": _sha256(qualifier_test_payload),
        },
        "foundation_exposure": {
            "audit_status": "UNKNOWN_NOT_ASSERTED",
            "checkpoint_id": "UNKNOWN_NOT_ASSERTED",
            "checkpoint_sha256": "UNKNOWN_NOT_ASSERTED",
            "sequence_exposed": True,
            "label_exposed": True,
            "unknown_checkpoint_blocks_qualification": True,
        },
        "scope": {
            "training_allowed": False,
            "model_selection_allowed": False,
        },
        SP.PREFLIGHT_BINDING_KEY: binding,
    }
    qualifier_path = repo / SP.QUALIFIER_CONFIG_PATH
    _write(qualifier_path, _json_payload(qualifier))
    _git(repo, "add", "--all")
    _git(repo, "commit", "-qm", "freeze preflight implementation with unknown binding")
    implementation_commit = _git(repo, "rev-parse", "HEAD")

    binding_commit = None
    if bound:
        qualifier["authority"]["implementation_commit"] = implementation_commit
        qualifier[SP.PREFLIGHT_BINDING_KEY]["status"] = "BOUND"
        qualifier[SP.PREFLIGHT_BINDING_KEY]["implementation_commit"] = implementation_commit
        _write(qualifier_path, _json_payload(qualifier))
        if binding_commit_extra:
            _write(repo / "seed.txt", b"impermissible B-side change\n")
        _git(repo, "add", "--all")
        _git(repo, "commit", "-qm", "bind preflight from qualifier config only")
        binding_commit = _git(repo, "rev-parse", "HEAD")

    _git(repo, "branch", "-M", test_authority_root["expected_branch"])
    _git(
        repo,
        "update-ref",
        f"refs/remotes/origin/{test_authority_root['expected_branch']}",
        _git(repo, "rev-parse", "HEAD"),
    )

    return {
        "repo": repo,
        "protocol": repo / SP.PREFLIGHT_CONFIG_PATH,
        "implementation_commit": implementation_commit,
        "binding_commit": binding_commit,
        "test_authority_root": test_authority_root,
    }


def _make_inventory(root: Path, protocol: dict[str, Any]) -> Path:
    data_root = root / "ordinary_public_gse149487"
    data_root.mkdir(parents=True)
    for item in protocol["data_directory_contract"]["expected_entries"]:
        path = data_root / item["name"]
        path.touch()
        os.truncate(path, item["bytes"])
    return data_root


def _run_success_inputs(bundle: dict[str, Any], tmp_path: Path, data_root: Path) -> dict[str, Any]:
    output_root = tmp_path / "outputs"
    output_root.mkdir()
    return {
        "repo_root": bundle["repo"],
        "protocol_path": bundle["protocol"],
        "data_root": data_root,
        "output_path": output_root / "PREFLIGHT.json",
        "failure_path": output_root / "FAILURE.json",
        "claim_path": output_root / "CLAIM.json",
        "recorded_at_utc": "2026-08-11T00:00:00Z",
        "module_finder": lambda _name: object(),
        "disk_usage": lambda _path: SimpleNamespace(free=2 * 1024**3),
        "python_version": (3, 11, 9),
        "test_only_authority_root": bundle["test_authority_root"],
    }


def test_unknown_binding_fails_before_data_or_output_stat(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bundle = _synthetic_repo(tmp_path, bound=False)
    calls = {"environment": 0, "inventory": 0}

    def forbidden_environment(*_args: Any, **_kwargs: Any) -> None:
        calls["environment"] += 1
        raise AssertionError("output path was inspected")

    def forbidden_inventory(*_args: Any, **_kwargs: Any) -> None:
        calls["inventory"] += 1
        raise AssertionError("data root was inspected")

    monkeypatch.setattr(SP, "audit_environment", forbidden_environment)
    monkeypatch.setattr(SP, "audit_data_directory", forbidden_inventory)
    with pytest.raises(SP.BindingError, match="UNKNOWN_NOT_ASSERTED"):
        SP.run_preflight(
            repo_root=bundle["repo"],
            protocol_path=bundle["protocol"],
            data_root=tmp_path / "must_not_be_statted_data",
            output_path=tmp_path / "must_not_be_statted_output" / "PREFLIGHT.json",
            failure_path=tmp_path / "must_not_be_statted_output" / "FAILURE.json",
            claim_path=tmp_path / "must_not_be_statted_output" / "CLAIM.json",
            test_only_authority_root=bundle["test_authority_root"],
        )
    assert calls == {"environment": 0, "inventory": 0}
    assert not (tmp_path / "must_not_be_statted_output").exists()


@pytest.mark.parametrize("mutation", ["extra", "missing", "name", "size", "symlink"])
def test_inventory_rejects_extra_missing_name_size_and_symlink(tmp_path: Path, mutation: str) -> None:
    protocol, _ = SP.load_protocol(PROTOCOL_PATH)
    data_root = _make_inventory(tmp_path, protocol)
    entries = protocol["data_directory_contract"]["expected_entries"]
    first = data_root / entries[0]["name"]
    second = data_root / entries[1]["name"]
    if mutation == "extra":
        (data_root / "unexpected.txt").write_bytes(b"x")
    elif mutation == "missing":
        first.unlink()
    elif mutation == "name":
        first.rename(data_root / "wrong_manifest_name.json")
    elif mutation == "size":
        os.truncate(first, entries[0]["bytes"] + 1)
    elif mutation == "symlink":
        first.unlink()
        first.symlink_to(second.name)
    else:  # pragma: no cover - the parameter set is closed above
        raise AssertionError(mutation)
    with pytest.raises(SP.InventoryError):
        SP.audit_data_directory(data_root, protocol)


def test_protocol_blocker_tamper_fails_closed(tmp_path: Path) -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    protocol["historical_r4_closure"]["exact_blockers"][0] = "TAMPERED_BLOCKER"
    tampered = tmp_path / "tampered.json"
    _write(tampered, _json_payload(protocol))
    with pytest.raises(SP.ProtocolError):
        SP.load_protocol(tampered)


def test_production_default_rejects_arbitrary_self_consistent_git_seed(tmp_path: Path) -> None:
    bundle = _synthetic_repo(tmp_path)
    with pytest.raises(SP.ProtocolError, match="production authority root"):
        SP.run_preflight(
            repo_root=bundle["repo"],
            protocol_path=bundle["protocol"],
            data_root=tmp_path / "data_must_not_be_needed",
            output_path=tmp_path / "output_must_not_be_needed" / "PREFLIGHT.json",
            failure_path=tmp_path / "output_must_not_be_needed" / "FAILURE.json",
            claim_path=tmp_path / "output_must_not_be_needed" / "CLAIM.json",
        )


def test_protocol_alias_to_protected_entry_is_rejected_without_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _synthetic_repo(tmp_path)
    protocol, _ = SP.load_protocol(PROTOCOL_PATH)
    data_root = _make_inventory(tmp_path / "inventory", protocol)
    protected = data_root / "manifest.json"
    real_os_open: Callable[..., int] = SP.os.open
    protected_open_count = 0

    def guarded_open(path: Any, *args: Any, **kwargs: Any) -> int:
        nonlocal protected_open_count
        if Path(os.fspath(path)).name == protected.name:
            protected_open_count += 1
            raise AssertionError("protected protocol alias was opened")
        return real_os_open(path, *args, **kwargs)

    monkeypatch.setattr(SP.os, "open", guarded_open)
    with pytest.raises(SP.ScopeError, match="protocol path"):
        SP.run_preflight(
            repo_root=bundle["repo"],
            protocol_path=protected,
            data_root=data_root,
            output_path=tmp_path / "outputs" / "PREFLIGHT.json",
            failure_path=tmp_path / "outputs" / "FAILURE.json",
            claim_path=tmp_path / "outputs" / "CLAIM.json",
            test_only_authority_root=bundle["test_authority_root"],
        )
    assert protected_open_count == 0


def test_committed_authority_drift_fails_before_data(tmp_path: Path) -> None:
    bundle = _synthetic_repo(tmp_path)
    contract = bundle["repo"] / SP.EXPECTED_AUTHORITY_PATHS["contract"]
    _write(contract, b"committed authority drift\n")
    _git(bundle["repo"], "add", "--all")
    _git(bundle["repo"], "commit", "-qm", "tamper authority")
    _git(
        bundle["repo"],
        "update-ref",
        f"refs/remotes/origin/{bundle['test_authority_root']['expected_branch']}",
        _git(bundle["repo"], "rev-parse", "HEAD"),
    )
    with pytest.raises(SP.AuthorityError, match="SHA-256 drift"):
        SP.run_preflight(
            repo_root=bundle["repo"],
            protocol_path=bundle["protocol"],
            data_root=tmp_path / "data_must_not_be_needed",
            output_path=tmp_path / "output_must_not_be_needed" / "PREFLIGHT.json",
            failure_path=tmp_path / "output_must_not_be_needed" / "FAILURE.json",
            claim_path=tmp_path / "output_must_not_be_needed" / "CLAIM.json",
            test_only_authority_root=bundle["test_authority_root"],
        )
    assert not (tmp_path / "output_must_not_be_needed").exists()


def test_binding_commit_must_change_only_qualifier_config(tmp_path: Path) -> None:
    bundle = _synthetic_repo(tmp_path, binding_commit_extra=True)
    with pytest.raises(SP.AuthorityError, match="modify only the qualifier config"):
        SP.run_preflight(
            repo_root=bundle["repo"],
            protocol_path=bundle["protocol"],
            data_root=tmp_path / "data_must_not_be_needed",
            output_path=tmp_path / "output_must_not_be_needed" / "PREFLIGHT.json",
            failure_path=tmp_path / "output_must_not_be_needed" / "FAILURE.json",
            claim_path=tmp_path / "output_must_not_be_needed" / "CLAIM.json",
            test_only_authority_root=bundle["test_authority_root"],
        )


def test_output_collision_never_overwrites_existing_file(tmp_path: Path) -> None:
    bundle = _synthetic_repo(tmp_path)
    protocol, _ = SP.load_protocol(
        bundle["protocol"],
        test_only_authority_root=bundle["test_authority_root"],
    )
    data_root = _make_inventory(tmp_path / "inventory", protocol)
    inputs = _run_success_inputs(bundle, tmp_path, data_root)
    output = inputs["output_path"]
    output.write_bytes(b"preserve-me\n")
    with pytest.raises(SP.PublicationError, match="output already exists"):
        SP.run_preflight(**inputs)
    assert output.read_bytes() == b"preserve-me\n"
    with pytest.raises(SP.PublicationError, match="output already exists"):
        SP._publish_exclusive(output, b"also-must-not-overwrite\n")
    assert output.read_bytes() == b"preserve-me\n"
    assert list(output.parent.glob(f".{output.name}.tmp.*")) == []


@pytest.mark.parametrize("operation", ["unlink", "fsync", "close"])
def test_post_commit_cleanup_errors_return_warning_without_false_failure(
    tmp_path: Path,
    operation: str,
) -> None:
    output_root = tmp_path / operation
    output_root.mkdir()
    output = output_root / "PREFLIGHT.json"
    payload = b'{"closed":true}\n'
    kwargs: dict[str, Any] = {}

    if operation == "unlink":
        def failing_unlink(*_args: Any, **_kwargs: Any) -> None:
            raise OSError("injected post-link unlink failure")

        kwargs["post_link_unlink_fn"] = failing_unlink
    elif operation == "fsync":
        def failing_fsync(_fd: int) -> None:
            raise OSError("injected post-link fsync failure")

        kwargs["post_link_fsync_fn"] = failing_fsync
    elif operation == "close":
        real_close = os.close

        def close_then_report(fd: int) -> None:
            real_close(fd)
            raise OSError("injected post-link close report")

        kwargs["post_link_close_fn"] = close_then_report
    else:  # pragma: no cover - the parameter set is closed above
        raise AssertionError(operation)

    receipt = SP._publish_exclusive(output, payload, **kwargs)
    assert receipt["status"] == "COMMITTED_WITH_POST_COMMIT_WARNING"
    assert receipt["final_identity_verified"] is True
    assert receipt["warnings"]
    assert output.read_bytes() == payload


def test_name_swap_at_link_is_explicitly_committed_not_accepted(tmp_path: Path) -> None:
    output_root = tmp_path / "swap"
    output_root.mkdir()
    output = output_root / "PREFLIGHT.json"
    intended = b'{"intended":true}\n'
    swapped = b'{"swapped":true}\n'

    def swapping_link(
        source: str,
        destination: str,
        *,
        src_dir_fd: int,
        dst_dir_fd: int,
        follow_symlinks: bool,
    ) -> None:
        os.unlink(source, dir_fd=src_dir_fd)
        replacement = os.open(
            source,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o640,
            dir_fd=src_dir_fd,
        )
        try:
            os.write(replacement, swapped)
            os.fsync(replacement)
        finally:
            os.close(replacement)
        os.link(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
            follow_symlinks=follow_symlinks,
        )

    receipt = SP._publish_exclusive(output, intended, link_fn=swapping_link)
    assert receipt["status"] == "COMMITTED_NOT_ACCEPTED"
    assert receipt["final_identity_verified"] is False
    assert "FINAL_IDENTITY_MISMATCH_AFTER_COMMIT" in receipt["warnings"]
    assert output.read_bytes() == swapped


def test_success_publishes_closed_blocked_record_without_payload_opens(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _synthetic_repo(tmp_path)
    protocol, _ = SP.load_protocol(
        bundle["protocol"],
        test_only_authority_root=bundle["test_authority_root"],
    )
    data_root = _make_inventory(tmp_path / "inventory", protocol)
    inputs = _run_success_inputs(bundle, tmp_path, data_root)
    forbidden_names = {
        item["name"] for item in protocol["data_directory_contract"]["expected_entries"]
    }
    real_os_open: Callable[..., int] = SP.os.open
    opened_payloads: list[str] = []

    def guarded_open(path: Any, *args: Any, **kwargs: Any) -> int:
        name = Path(os.fspath(path)).name
        if name in forbidden_names:
            opened_payloads.append(name)
            raise AssertionError(f"preflight opened a protected inventory member: {name}")
        return real_os_open(path, *args, **kwargs)

    monkeypatch.setattr(SP.os, "open", guarded_open)
    document = SP.run_preflight(**inputs)
    output = inputs["output_path"]
    persisted = json.loads(output.read_text(encoding="utf-8"))

    assert opened_payloads == []
    assert persisted == document
    assert document["outcome"] == "NOT_READY_FOR_STUDY_QUALIFICATION"
    assert document["ready_for_study_qualification"] is False
    assert document["inventory_audit"]["payload_open_count"] == 0
    assert document["inventory_audit"]["manifest_open_count"] == 0
    assert document["inventory_audit"]["hash_reverification"] == "NOT_RUN_STOP_BEFORE_DATA"
    assert document["gate_truth"] == SP.EXPECTED_GATE_TRUTH
    assert document["counters"]["canonical_record_count"] == 0
    assert list(output.parent.glob(f".{output.name}.tmp.*")) == []
