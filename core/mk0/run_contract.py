"""Shared fail-closed helpers for the MK0 section 19--21 run contract."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Any, Mapping


EVIDENCE_LEVEL = "E0_MATH_ENGINEERING_ONLY"
RUN_DIRECTORIES = (
    "provenance",
    "git",
    "logs",
    "checkpoints",
    "evaluation",
    "failure",
    "summary",
)
LOG_FILENAMES = (
    "stdout.log",
    "stderr.log",
    "metrics.jsonl",
    "system_metrics.jsonl",
    "events.jsonl",
)


def _regular_run_files(root: Path, *, excluded: set[Path]) -> list[Path]:
    """Return regular files, rejecting every alias or special tree node."""

    root = root.resolve(strict=True)
    excluded_resolved = {path.resolve() for path in excluded}
    files: list[Path] = []
    seen_identities: dict[tuple[int, int], Path] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise RuntimeError(f"run tree contains a symlink: {relative}")
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError(f"run tree contains a special node: {relative}")
        resolved = path.resolve(strict=True)
        if metadata.st_nlink != 1:
            raise RuntimeError(f"run tree contains a hardlink alias: {relative}")
        identity = (metadata.st_dev, metadata.st_ino)
        if identity in seen_identities:
            raise RuntimeError(
                "run tree contains duplicate filesystem identity: "
                f"{seen_identities[identity].relative_to(root)} and {relative}"
            )
        seen_identities[identity] = path
        if resolved not in excluded_resolved:
            files.append(path)
    return files


def _verify_ledger(root: Path, ledger: Path, *, excluded_from_ledger: set[Path]) -> int:
    """Verify ledger bytes and exact coverage of every non-excluded run file."""

    root = root.resolve(strict=True)
    ledger = ledger.resolve(strict=True)
    excluded = {path.resolve() for path in excluded_from_ledger} | {ledger}
    seen: set[str] = set()
    seen_identities: set[tuple[int, int]] = set()
    count = 0
    for line in ledger.read_text(encoding="utf-8").splitlines():
        digest, separator, relative = line.partition("  ")
        if (
            not separator
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or not relative
            or relative in seen
        ):
            raise RuntimeError(f"invalid checksum-ledger line: {line!r}")
        lexical_path = root / relative
        metadata = lexical_path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise RuntimeError(f"checksum-ledger target is a symlink: {relative}")
        path = lexical_path.resolve(strict=True)
        try:
            path.relative_to(root)
        except ValueError as error:
            raise RuntimeError("checksum-ledger path escaped root") from error
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError(f"checksum-ledger target is not regular: {relative}")
        if metadata.st_nlink != 1:
            raise RuntimeError(f"checksum-ledger target is a hardlink: {relative}")
        identity = (metadata.st_dev, metadata.st_ino)
        if identity in seen_identities:
            raise RuntimeError(f"duplicate checksum-ledger identity: {relative}")
        if path in excluded:
            raise RuntimeError(f"excluded file appeared in checksum ledger: {relative}")
        if not path.is_file() or sha256_file(path) != digest:
            raise RuntimeError(f"checksum-ledger mismatch: {relative}")
        seen.add(relative)
        seen_identities.add(identity)
        count += 1
    if count == 0:
        raise RuntimeError("checksum ledger is empty")
    actual = {
        path.relative_to(root).as_posix()
        for path in _regular_run_files(root, excluded=excluded)
    }
    if seen != actual:
        omitted = sorted(actual - seen)
        unexpected = sorted(seen - actual)
        raise RuntimeError(
            "checksum ledger does not exactly cover run files: "
            f"omitted={omitted}, unexpected={unexpected}"
        )
    return count


def validate_terminal_chain(run_root: Path, *, run_id: str) -> str | None:
    """Validate DONE/FAILED without mutating a sealed run."""

    done = run_root / "DONE"
    failed = run_root / "FAILED"
    for name, sentinel in (("DONE", done), ("FAILED", failed)):
        if sentinel.exists() and (sentinel.is_symlink() or not sentinel.is_file()):
            raise RuntimeError(f"{name} sentinel is not a regular file")
    if done.exists() and failed.exists():
        raise RuntimeError("run has both DONE and FAILED sentinels")
    if done.exists():
        lines = done.read_text(encoding="utf-8").splitlines()
        freeze = run_root / "artifacts" / "mk0" / "mk0_freeze_manifest.json"
        completion = run_root / "summary" / "run_completion_manifest.json"
        ledger = run_root / "artifact_checksums.sha256"
        status = run_root / "status.json"
        mk0_status = run_root / "mk0_status.json"
        if (
            len(lines) != 6
            or lines[0] != run_id
            or not all(
                path.is_file()
                for path in (freeze, completion, ledger, status, mk0_status)
            )
            or lines[1] != sha256_file(freeze)
            or lines[2] != sha256_file(completion)
            or lines[3] != sha256_file(ledger)
            or lines[4] != sha256_file(status)
            or lines[5] != sha256_file(mk0_status)
        ):
            raise RuntimeError("existing DONE sentinel/hash chain is invalid")
        _verify_ledger(
            run_root,
            ledger,
            excluded_from_ledger={ledger, done},
        )
        status_payload = json.loads(status.read_text(encoding="utf-8"))
        mk0_status_payload = json.loads(mk0_status.read_text(encoding="utf-8"))
        completion_payload = json.loads(completion.read_text(encoding="utf-8"))
        if (
            status_payload.get("state") != "CLOSED_ACCEPTED"
            or status_payload.get("terminal") is not True
            or status_payload.get("run_id") != run_id
            or mk0_status_payload.get("status") != "DONE"
            or mk0_status_payload.get("run_id") != run_id
            or completion_payload.get("status") != "DONE"
            or completion_payload.get("run_id") != run_id
        ):
            raise RuntimeError("DONE terminal status semantics are invalid")
        return "DONE"
    if failed.exists():
        lines = failed.read_text(encoding="utf-8").splitlines()
        completion = run_root / "failure" / "run_failure_completion_manifest.json"
        ledger = run_root / "failure" / "artifact_checksums_at_failure.sha256"
        mk0_status = run_root / "mk0_status.json"
        root_status = run_root / "status.json"
        intent = run_root / "failure" / "failure_closure_intent.json"
        if (
            len(lines) != 5
            or lines[0] != run_id
            or not all(
                path.is_file()
                for path in (completion, ledger, mk0_status, root_status, intent)
            )
            or lines[2] != sha256_file(completion)
            or lines[3] != sha256_file(ledger)
            or lines[4] != sha256_file(mk0_status)
        ):
            raise RuntimeError("existing FAILED sentinel/hash chain is invalid")
        _verify_ledger(
            run_root,
            ledger,
            excluded_from_ledger={ledger, failed},
        )
        task_status = json.loads(mk0_status.read_text(encoding="utf-8"))
        status_payload = json.loads(root_status.read_text(encoding="utf-8"))
        completion_payload = json.loads(completion.read_text(encoding="utf-8"))
        intent_payload = json.loads(intent.read_text(encoding="utf-8"))
        if (
            task_status.get("status") != "FAILED_WITH_EVIDENCE"
            or task_status.get("run_id") != run_id
            or task_status.get("stage") != lines[1]
            or status_payload.get("state") != "FAILED_WITH_EVIDENCE"
            or status_payload.get("terminal") is not True
            or status_payload.get("run_id") != run_id
            or completion_payload.get("status") != "FAILED_WITH_EVIDENCE"
            or completion_payload.get("run_id") != run_id
            or completion_payload.get("stage") != lines[1]
            or completion_payload.get("failure_intent_sha256") != sha256_file(intent)
            or intent_payload.get("run_id") != run_id
            or intent_payload.get("stage") != lines[1]
        ):
            raise RuntimeError("FAILED terminal status semantics are invalid")
        return "FAILED"
    return None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_contract_tree(run_root: Path) -> None:
    for relative in RUN_DIRECTORIES:
        (run_root / relative).mkdir(parents=True, exist_ok=False)
    for name in LOG_FILENAMES:
        path = run_root / "logs" / name
        with path.open("xb") as handle:
            handle.flush()
            os.fsync(handle.fileno())


def append_text(path: Path, text: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"append target is absent: {path}")
    with path.open("ab") as handle:
        handle.write(text.encode("utf-8"))
        handle.flush()
        os.fsync(handle.fileno())


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    append_text(path, canonical_json_bytes(dict(payload)).decode("utf-8"))


def append_event(run_root: Path, event: str, **fields: Any) -> None:
    append_jsonl(
        run_root / "logs" / "events.jsonl",
        {"created_at_utc": utc_now(), "event": event, **fields},
    )


def _atomic_json_replace(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = canonical_json_bytes(dict(payload))
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def write_bytes_exclusive_atomic(path: Path, data: bytes) -> str:
    """Publish immutable evidence with create-if-absent and directory fsync."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    return hashlib.sha256(data).hexdigest()


def write_json_exclusive_atomic(path: Path, payload: Mapping[str, Any]) -> str:
    return write_bytes_exclusive_atomic(path, canonical_json_bytes(dict(payload)))


def update_status(
    run_root: Path,
    *,
    run_id: str,
    state: str,
    terminal: bool,
    stop_reason: str,
    exit_code: int | None = None,
) -> dict[str, Any]:
    payload = {
        "schema_version": "mk0_status_v1",
        "run_id": run_id,
        "state": state,
        "terminal": terminal,
        "evidence_level": EVIDENCE_LEVEL,
        "stop_reason": stop_reason,
        "exit_code": exit_code,
        "updated_at_utc": utc_now(),
    }
    append_jsonl(run_root / "logs" / "events.jsonl", {"status": payload})
    _atomic_json_replace(run_root / "status.json", payload)
    return payload


def replace_status_without_event(
    run_root: Path,
    *,
    run_id: str,
    state: str,
    terminal: bool,
    stop_reason: str,
    exit_code: int | None,
) -> dict[str, Any]:
    """Publish post-ledger status; the final sentinel binds its resulting hash."""

    payload = {
        "schema_version": "mk0_status_v1",
        "run_id": run_id,
        "state": state,
        "terminal": terminal,
        "evidence_level": EVIDENCE_LEVEL,
        "stop_reason": stop_reason,
        "exit_code": exit_code,
        "updated_at_utc": utc_now(),
    }
    _atomic_json_replace(run_root / "status.json", payload)
    return payload


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{label} is missing or invalid") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} is not a JSON object")
    return payload


def _require_failure_contract_base(run_root: Path, *, run_id: str) -> None:
    """Refuse to seal a failure after canonical preterminal evidence was lost."""

    for directory in RUN_DIRECTORIES:
        if not (run_root / directory).is_dir():
            raise RuntimeError(f"failure closure lacks run directory: {directory}")
    manifest = _read_json_object(
        run_root / "run_manifest.json", label="failure run manifest"
    )
    if (
        manifest.get("schema_version") != "mk0_run_manifest_v3"
        or manifest.get("run_id") != run_id
        or Path(str(manifest.get("run_root", ""))).resolve() != run_root.resolve()
    ):
        raise RuntimeError("failure run manifest binding is invalid")
    status_payload = _read_json_object(
        run_root / "status.json", label="failure status record"
    )
    if (
        status_payload.get("schema_version") != "mk0_status_v1"
        or status_payload.get("run_id") != run_id
    ):
        raise RuntimeError("failure status record binding is invalid")
    for name in LOG_FILENAMES:
        path = run_root / "logs" / name
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"failure closure lacks canonical log: {name}")
    for name in ("metrics.jsonl", "system_metrics.jsonl", "events.jsonl"):
        path = run_root / "logs" / name
        lines = path.read_text(encoding="utf-8").splitlines()
        if name == "events.jsonl" and not lines:
            raise RuntimeError("failure events log is empty")
        for line in lines:
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"failure structured log is invalid: {name}"
                ) from error
            if not isinstance(record, dict):
                raise RuntimeError(
                    f"failure structured log is not object JSONL: {name}"
                )
    # This scan must precede the immutable intent. Otherwise replacing a
    # hardlinked status file could detach its alias and make the final tree
    # appear canonical after a noncanonical preclosure state.
    _regular_run_files(run_root, excluded=set())


def _same_failure_record(
    path: Path,
    *,
    run_id: str,
    stage: str,
    reason: str,
    exit_code: int,
) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        isinstance(payload, dict)
        and payload.get("run_id") == run_id
        and payload.get("status") == "FAILED_WITH_EVIDENCE"
        and payload.get("stage") == stage
        and payload.get("stop_reason") == reason
        and payload.get("exit_code") == exit_code
    )


def write_failed_sentinel(
    run_root: Path,
    *,
    run_id: str,
    stage: str,
    reason: str,
    exit_code: int,
) -> Path:
    terminal = validate_terminal_chain(run_root, run_id=run_id)
    if terminal == "DONE":
        return run_root / "DONE"
    if terminal == "FAILED":
        return run_root / "FAILED"
    _require_failure_contract_base(run_root, run_id=run_id)
    failure_dir = run_root / "failure"
    failure_dir.mkdir(parents=True, exist_ok=True)
    failed = run_root / "FAILED"
    completion_path = failure_dir / "run_failure_completion_manifest.json"
    failure_ledger = failure_dir / "artifact_checksums_at_failure.sha256"
    intent_path = failure_dir / "failure_closure_intent.json"
    if failure_ledger.exists() and not completion_path.exists():
        raise RuntimeError("orphan failure ledger exists; refusing mutation")
    if completion_path.exists():
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        if completion.get("run_id") != run_id:
            raise RuntimeError("prior failure completion belongs to another run")
        stage = str(completion.get("stage"))
        if not intent_path.is_file() or completion.get(
            "failure_intent_sha256"
        ) != sha256_file(intent_path):
            raise RuntimeError("failure completion lacks its immutable intent binding")
        root_status = run_root / "mk0_status.json"
        if not root_status.is_file():
            raise RuntimeError("failure completion lacks root mk0_status.json")
        if not failure_ledger.exists():
            records = immutable_file_inventory(
                run_root, exclude={failure_ledger, failed}
            )
            ledger_bytes = "".join(
                f"{record['sha256']}  {record['path']}\n" for record in records
            ).encode("utf-8")
            write_bytes_exclusive_atomic(failure_ledger, ledger_bytes)
        _verify_ledger(
            run_root,
            failure_ledger,
            excluded_from_ledger={failure_ledger, failed},
        )
        completion_sha256 = sha256_file(completion_path)
        ledger_sha256 = sha256_file(failure_ledger)
        status_sha256 = sha256_file(root_status)
        write_bytes_exclusive_atomic(
            failed,
            (
                f"{run_id}\n{stage}\n{completion_sha256}\n{ledger_sha256}\n"
                f"{status_sha256}\n"
            ).encode("utf-8"),
        )
        if validate_terminal_chain(run_root, run_id=run_id) != "FAILED":
            raise RuntimeError("failure closure did not produce a valid FAILED chain")
        return failed
    if (
        not isinstance(stage, str)
        or not stage
        or not isinstance(reason, str)
        or not reason
        or not isinstance(exit_code, int)
        or isinstance(exit_code, bool)
        or exit_code == 0
    ):
        raise RuntimeError("failure closure intent semantics are invalid")
    if intent_path.exists():
        intent = _read_json_object(intent_path, label="failure closure intent")
        if (
            intent.get("schema_version") != "mk0_failure_closure_intent_v1"
            or intent.get("run_id") != run_id
            or intent.get("stage") != stage
            or intent.get("stop_reason") != reason
            or intent.get("exit_code") != exit_code
        ):
            raise RuntimeError("existing failure closure intent conflicts with request")
    else:
        intent = {
            "schema_version": "mk0_failure_closure_intent_v1",
            "run_id": run_id,
            "stage": stage,
            "stop_reason": reason,
            "exit_code": exit_code,
            "created_at_utc": utc_now(),
            "evidence_level": EVIDENCE_LEVEL,
            "paper_eligibility": False,
        }
        write_json_exclusive_atomic(intent_path, intent)
    done = run_root / "DONE"
    if done.exists():
        revoked = failure_dir / "DONE_REVOKED_AFTER_FAILURE"
        if revoked.exists():
            raise RuntimeError("DONE revocation artifact already exists")
        os.replace(done, revoked)

    terminal_at = str(intent["created_at_utc"])
    status = update_status(
        run_root,
        run_id=run_id,
        state="FAILED_WITH_EVIDENCE",
        terminal=True,
        stop_reason=reason,
        exit_code=exit_code,
    )
    root_summary = run_root / "summary.json"
    failure_summary_payload = {
        "schema_version": "mk0_failure_summary_v1",
        "run_id": run_id,
        "status": "FAILED_WITH_EVIDENCE",
        "stage": stage,
        "stop_reason": reason,
        "exit_code": exit_code,
        "terminal_at_utc": terminal_at,
        "evidence_level": EVIDENCE_LEVEL,
        "paper_eligibility": False,
        "artifact_checksums": {
            "failure_ledger_path": str(failure_ledger),
            "self_reference_exception": "FAILED sentinel binds the verified failure ledger hash",
        },
    }
    if root_summary.exists() and not _same_failure_record(
        root_summary,
        run_id=run_id,
        stage=stage,
        reason=reason,
        exit_code=exit_code,
    ):
        archived_summary = failure_dir / "PRE_FAILURE_ROOT_SUMMARY.json"
        if not archived_summary.exists():
            with archived_summary.open("xb") as handle:
                handle.write(root_summary.read_bytes())
                handle.flush()
                os.fsync(handle.fileno())
    _atomic_json_replace(root_summary, failure_summary_payload)

    root_status = run_root / "mk0_status.json"
    if root_status.exists() and not _same_failure_record(
        root_status,
        run_id=run_id,
        stage=stage,
        reason=reason,
        exit_code=exit_code,
    ):
        archived_status = failure_dir / "PRE_FAILURE_MK0_STATUS.json"
        if not archived_status.exists():
            with archived_status.open("xb") as handle:
                handle.write(root_status.read_bytes())
                handle.flush()
                os.fsync(handle.fileno())
    failure_status_payload = {
        "schema_version": "mk0_task_status_v1",
        "run_id": run_id,
        "task_id": "MK0-01",
        "status": "FAILED_WITH_EVIDENCE",
        "stage": stage,
        "stop_reason": reason,
        "failure_intent": {
            "path": str(intent_path),
            "sha256": sha256_file(intent_path),
        },
        "exit_code": exit_code,
        "evidence_level": EVIDENCE_LEVEL,
        "failure_completion_manifest_path": str(completion_path),
        "failure_checksum_ledger_path": str(failure_ledger),
        "FAILED_sentinel_binds_terminal_hashes": True,
        "updated_at_utc": terminal_at,
    }
    _atomic_json_replace(root_status, failure_status_payload)

    registration_path = run_root / "run_manifest.json"
    registration = None
    if registration_path.is_file():
        try:
            registration = json.loads(registration_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            registration = None
    completion = {
        "schema_version": "mk0_run_failure_completion_manifest_v1",
        "run_id": run_id,
        "task_id": "MK0-01",
        "status": "FAILED_WITH_EVIDENCE",
        "stage": stage,
        "evidence_level": EVIDENCE_LEVEL,
        "timing": {
            "start_utc": (
                registration.get("timing", {}).get("start_utc")
                if isinstance(registration, dict)
                else None
            ),
            "end_utc": terminal_at,
        },
        "process_identity": {
            "failure_closure_pid": os.getpid(),
            "registered": (
                registration.get("process_identity")
                if isinstance(registration, dict)
                else None
            ),
        },
        "exit_code": exit_code,
        "stop_reason": reason,
        "artifact_checksums": {
            "ledger_path": str(failure_ledger),
            "ledger_excludes_itself_and_FAILED": True,
            "FAILED_sentinel_binds_ledger_sha256": True,
        },
        "paper_eligibility": False,
        "known_deviations": (
            registration.get("known_deviations", [])
            if isinstance(registration, dict)
            else ["REGISTRATION_MANIFEST_UNAVAILABLE_AT_FAILURE"]
        ),
        "final_labels_accessed": False,
        "downstream_stage_started": False,
        "mutable_status_snapshot": status,
        "failure_intent_sha256": sha256_file(intent_path),
    }
    write_json_exclusive_atomic(completion_path, completion)
    completion_sha256 = sha256_file(completion_path)
    records = immutable_file_inventory(run_root, exclude={failure_ledger, failed})
    ledger_bytes = "".join(
        f"{record['sha256']}  {record['path']}\n" for record in records
    ).encode("utf-8")
    write_bytes_exclusive_atomic(failure_ledger, ledger_bytes)
    ledger_sha256 = sha256_file(failure_ledger)
    failure_status_sha256 = sha256_file(root_status)
    write_bytes_exclusive_atomic(
        failed,
        (
            f"{run_id}\n{stage}\n{completion_sha256}\n{ledger_sha256}\n"
            f"{failure_status_sha256}\n"
        ).encode("utf-8"),
    )
    if validate_terminal_chain(run_root, run_id=run_id) != "FAILED":
        raise RuntimeError("failure closure did not produce a valid FAILED chain")
    return failed


def resume_failure_closure_if_present(run_root: Path, *, run_id: str) -> str | None:
    """Seal an interrupted failure tail before a stage can append any new bytes.

    The immutable failure intent is published before any failure-state write.
    Once it, the completion manifest, or its ledger exists, a later stage may
    only finish and validate the same FAILED chain before other diagnostics.
    """

    terminal = validate_terminal_chain(run_root, run_id=run_id)
    if terminal is not None:
        return terminal
    completion_path = run_root / "failure" / "run_failure_completion_manifest.json"
    failure_ledger = run_root / "failure" / "artifact_checksums_at_failure.sha256"
    intent_path = run_root / "failure" / "failure_closure_intent.json"
    if failure_ledger.exists() and not completion_path.exists():
        raise RuntimeError("orphan failure ledger exists; refusing stage mutation")
    if completion_path.exists():
        source = _read_json_object(completion_path, label="partial failure completion")
        if source.get("run_id") != run_id:
            raise RuntimeError("partial failure completion belongs to another run")
    elif intent_path.exists():
        source = _read_json_object(intent_path, label="partial failure intent")
        if (
            source.get("schema_version") != "mk0_failure_closure_intent_v1"
            or source.get("run_id") != run_id
        ):
            raise RuntimeError("partial failure intent belongs to another run")
    else:
        return None
    stage = source.get("stage")
    reason = source.get("stop_reason")
    exit_code = source.get("exit_code")
    if (
        not isinstance(stage, str)
        or not stage
        or not isinstance(reason, str)
        or not reason
        or not isinstance(exit_code, int)
        or isinstance(exit_code, bool)
        or exit_code == 0
    ):
        raise RuntimeError("partial failure completion semantics are invalid")
    write_failed_sentinel(
        run_root,
        run_id=run_id,
        stage=stage,
        reason=reason,
        exit_code=exit_code,
    )
    if validate_terminal_chain(run_root, run_id=run_id) != "FAILED":
        raise RuntimeError("partial failure closure could not be sealed")
    return "FAILED"


def immutable_file_inventory(
    run_root: Path, *, exclude: set[Path] | None = None
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in _regular_run_files(run_root, excluded=exclude or set()):
        relative = path.relative_to(run_root).as_posix()
        records.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def write_whole_run_checksum_ledger(
    run_root: Path, *, exclude: set[Path] | None = None
) -> dict[str, Any]:
    ledger = run_root / "artifact_checksums.sha256"
    if ledger.exists():
        raise FileExistsError(f"refusing to overwrite {ledger}")
    records = immutable_file_inventory(run_root, exclude={ledger, *(exclude or set())})
    ledger_bytes = "".join(
        f"{record['sha256']}  {record['path']}\n" for record in records
    ).encode("utf-8")
    write_bytes_exclusive_atomic(ledger, ledger_bytes)
    return {
        "path": str(ledger),
        "sha256": sha256_file(ledger),
        "entry_count": len(records),
        "covered_files_sha256": hashlib.sha256(
            canonical_json_bytes(records)
        ).hexdigest(),
    }
