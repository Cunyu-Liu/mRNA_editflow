#!/usr/bin/env python3
"""Manifest-bound, fail-closed GSE200304 ENA FASTQ acquisition.

The program downloads only the 48 HTTPS objects frozen in the committed ENA
manifest.  It never invokes a shell, never overwrites a completed FASTQ, and
publishes a terminal commit marker only after every object passes exact byte,
ENA repository-MD5, and local SHA256 verification.

Partial transfers are intentionally simple and auditable.  A ``.part`` file
may be resumed only when its immutable sidecar binds the exact source URL,
expected bytes, repository MD5, manifest hash, and source terminal marker.  A
resumed response must be an exact HTTP 206 range response; ambiguity is a hard
failure and the partial evidence is retained.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import io
import json
import os
import re
import secrets
import ssl
import stat
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Callable, Mapping, Protocol, Sequence


SCHEMA_VERSION = "route_a_v3_gse200304_fastq_acquisition.v1"
PROTOCOL_ID = "ROUTE_A_V3_GSE200304_ENA_FASTQ_ACQUISITION_V1"
DATASET_ACCESSION = "GSE200304"
BIOPROJECT_ACCESSION = "PRJNA824033"
PROTOCOL_BASENAME = "route_a_v3_gse200304_fastq_acquisition.json"

EXPECTED_MANIFEST_BUNDLE_ROOT = Path(
    "/mnt/cunyuliu/mrna_xeditflow_routea_v3/data/A1/GSE200304/"
    "GSE200304_ENA_FASTQ_MANIFEST_20260810T145631P0800"
)
EXPECTED_MANIFEST_BUNDLE_BASENAME = (
    "GSE200304_ENA_FASTQ_MANIFEST_20260810T145631P0800"
)
EXPECTED_OUTPUT_ROOT = Path(
    "/mnt/cunyuliu/mrna_xeditflow_routea_v3/data/A1/GSE200304"
)
EXPECTED_SOURCE_COMMIT_FILENAME = "PUBLICATION_COMMIT.json"
EXPECTED_SOURCE_COMMIT_BYTES = 1578
EXPECTED_SOURCE_COMMIT_SHA256 = (
    "d3eed4a9408543c77f47aa2a0d8cff59ebfe863c1e3c2d0bb2324d7910d6014b"
)
EXPECTED_MANIFEST_FILENAME = "ENA_PRJNA824033_FASTQ_FILES.canonical.tsv"
EXPECTED_MANIFEST_BYTES = 10388
EXPECTED_MANIFEST_SHA256 = (
    "22cd317d961d07036cb2dad19555b5c2423671c33a76badeb7b325847ee68d7b"
)
EXPECTED_SUMMARY_FILENAME = "MANIFEST_SUMMARY.json"
EXPECTED_SUMMARY_BYTES = 3135
EXPECTED_SUMMARY_SHA256 = (
    "f92f944c825a255f3f1fb50f48cbf0e701980b7895101c1a2a6699d4b190e1e4"
)
EXPECTED_SOURCE_REPORT_FILENAME = "ENA_PRJNA824033_FASTQ_FILE_REPORT.source.tsv"
EXPECTED_SOURCE_REPORT_BYTES = 5998
EXPECTED_SOURCE_REPORT_SHA256 = (
    "c4a0b6152ec2a3480f280d8498345196d5095ec54967525463fa81961f0f4ea1"
)
EXPECTED_SOURCE_SHA256SUMS_FILENAME = "SHA256SUMS"
EXPECTED_SOURCE_SHA256SUMS_BYTES = 307
EXPECTED_SOURCE_SHA256SUMS_SHA256 = (
    "5217d3bd5494908d1886c6a00719014f4726ab3b61efde43184c2e475c6fdc78"
)
EXPECTED_MANIFEST_HEADER = (
    "run_accession",
    "mate",
    "fastq_https",
    "fastq_ftp",
    "fastq_bytes",
    "repository_md5",
)
EXPECTED_RUN_ACCESSIONS = (
    "SRR18656742",
    "SRR18656743",
    "SRR18656744",
    "SRR18656745",
    "SRR18656746",
    "SRR18656747",
    "SRR18656748",
    "SRR18656749",
    "SRR18656750",
    "SRR18656751",
    "SRR18656752",
    "SRR18656753",
    "SRR18656754",
    "SRR18656755",
    "SRR18656756",
    "SRR18656757",
    "SRR18656758",
    "SRR18656759",
    "SRR18656760",
    "SRR18656766",
    "SRR18656767",
    "SRR18656768",
    "SRR18656769",
    "SRR18656770",
)
EXPECTED_FILE_COUNT = 48
EXPECTED_RUN_COUNT = 24
EXPECTED_TOTAL_BYTES = 12_738_938_976
OFFICIAL_HOST = "ftp.sra.ebi.ac.uk"
DEFAULT_WORKERS = 2
MAXIMUM_WORKERS = 2
DEFAULT_TIMEOUT_SECONDS = 120
DEFAULT_CHUNK_BYTES = 8 * 1024 * 1024
MAX_CHUNK_BYTES = 64 * 1024 * 1024

OUTPUT_BASENAME_RE = re.compile(
    r"GSE200304_FASTQ_ACQUISITION_[A-Za-z0-9][A-Za-z0-9._-]{7,95}"
)
RUN_RE = re.compile(r"SRR[0-9]{8}")
MD5_RE = re.compile(r"[0-9a-f]{32}")
SHA256_RE = re.compile(r"[0-9a-f]{64}")
COMMIT_RE = re.compile(r"[0-9a-f]{40}")
POSITIVE_INTEGER_RE = re.compile(r"[1-9][0-9]*")
FORBIDDEN_PATH_TOKENS = ("gse246381", "sealed", "restricted", "access_log")

ACQUISITION_BINDING_FILENAME = "ACQUISITION_BINDING.json"
PROGRESS_STATUS_FILENAME = "ACQUISITION_PROGRESS.json"
FINAL_STATUS_FILENAME = "ACQUISITION_STATUS.json"
INTEGRITY_MANIFEST_FILENAME = "FASTQ_INTEGRITY_MANIFEST.json"
SHA256SUMS_FILENAME = "SHA256SUMS"
LOCK_FILENAME = "ACQUISITION_LOCK"
TERMINAL_MARKER_FILENAME = "PUBLICATION_COMMIT.json"
MAXIMUM_PROGRESS_STATUS_BYTES = 8 * 1024 * 1024
MAXIMUM_ATTEMPT_HISTORY_ENTRIES = 128
MINIMUM_CAPACITY_SAFETY_MARGIN_BYTES = 2 * 1024 * 1024 * 1024
CAPACITY_SAFETY_MARGIN_BASIS_POINTS = 500
GIT_BINARY = "/usr/bin/git"
IMPLEMENTATION_SCRIPT_REPO_PATH = "scripts/route_a_v3/acquire_gse200304_fastq.py"
PROTOCOL_REPO_PATH = "configs/route_a_v3_gse200304_fastq_acquisition.json"


class AcquisitionError(RuntimeError):
    """Base class for explicit fail-closed acquisition errors."""


class ScopeViolation(AcquisitionError):
    """A path or URL leaves the frozen ordinary-public scope."""


class ProtocolError(AcquisitionError):
    """The protocol or committed manifest authority is invalid."""


class ManifestError(AcquisitionError):
    """The canonical ENA manifest is not the exact closed authority."""


class ResumeError(AcquisitionError):
    """A partial transfer cannot be resumed without ambiguity."""


class TransportError(AcquisitionError):
    """The HTTPS response violates the exact transport contract."""


class IntegrityError(AcquisitionError):
    """A downloaded object fails bytes or digest verification."""


class CapacityError(AcquisitionError):
    """Trusted remaining bytes plus the frozen safety margin do not fit."""

    def __init__(self, message: str, evidence: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.evidence = dict(evidence)


class PublicationError(AcquisitionError):
    """Output publication would overwrite or misrepresent evidence."""


@dataclass(frozen=True)
class ManifestEntry:
    run_accession: str
    mate: int
    url: str
    ftp_path: str
    expected_bytes: int
    repository_md5: str

    @property
    def filename(self) -> str:
        return f"{self.run_accession}_{self.mate}.fastq.gz"

    @property
    def part_filename(self) -> str:
        return f"{self.filename}.part"

    @property
    def transfer_binding_filename(self) -> str:
        return f"{self.filename}.transfer.json"


@dataclass(frozen=True)
class FileIdentity:
    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int


@dataclass(frozen=True)
class FileResult:
    entry: ManifestEntry
    success: bool
    resumed_from_bytes: int
    bytes_verified: int = 0
    repository_md5: str | None = None
    local_sha256: str | None = None
    identity: FileIdentity | None = None
    reused_completed_file: bool = False
    error_code: str | None = None
    error_message: str | None = None


class HTTPResponse(Protocol):
    status: int
    headers: Mapping[str, str]

    def read(self, amount: int = -1) -> bytes: ...

    def geturl(self) -> str: ...

    def close(self) -> None: ...


class DownloadTransport(Protocol):
    def open(self, url: str, *, offset: int, timeout_seconds: int) -> HTTPResponse: ...


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(  # type: ignore[override]
        self,
        req: urllib.request.Request,
        fp: BinaryIO,
        code: int,
        msg: str,
        headers: Mapping[str, str],
        newurl: str,
    ) -> None:
        return None


class UrllibHttpsTransport:
    """HTTPS-only stdlib transport with redirects disabled."""

    def __init__(self) -> None:
        # An explicit empty ProxyHandler prevents HTTP(S)_PROXY, ALL_PROXY, and
        # platform proxy discovery from changing the single-host truth.
        self._opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}),
            _NoRedirect(),
            urllib.request.HTTPSHandler(context=ssl.create_default_context()),
        )

    def open(self, url: str, *, offset: int, timeout_seconds: int) -> HTTPResponse:
        headers = {
            "Accept-Encoding": "identity",
            "User-Agent": "mRNA-XEditFlow-RouteA-v3-GSE200304-acquirer/1",
        }
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(url, headers=headers, method="GET")
        try:
            return self._opener.open(request, timeout=timeout_seconds)  # type: ignore[return-value]
        except urllib.error.HTTPError as exc:
            raise TransportError(f"HTTP request failed with status {exc.code}") from exc
        except urllib.error.URLError as exc:
            raise TransportError("HTTPS connection failed") from exc


def _normalized_error(exc: BaseException) -> tuple[str, str]:
    """Return bounded, non-environment-leaking status text."""

    if isinstance(exc, AcquisitionError):
        message = str(exc)
        if len(message) > 512:
            message = message[:509] + "..."
        return type(exc).__name__, message
    if isinstance(exc, OSError):
        return "IOError", "A local I/O or fixed transport operation failed"
    return "InternalError", "An unexpected acquisition operation failed"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_bytes(document: Mapping[str, Any]) -> bytes:
    return (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _assert_no_forbidden_path(path: Path, *, label: str) -> None:
    if not path.is_absolute():
        raise ScopeViolation(f"{label} must be absolute")
    if any(component in {"", ".", ".."} for component in path.parts[1:]):
        raise ScopeViolation(f"{label} contains a non-canonical component")
    lowered = tuple(component.casefold() for component in path.parts)
    for token in FORBIDDEN_PATH_TOKENS:
        if any(token in component for component in lowered):
            raise ScopeViolation(f"{label} contains forbidden path token {token}")


def _safe_basename(name: str, *, label: str) -> str:
    pure = PurePosixPath(name)
    if (
        not name
        or pure.is_absolute()
        or len(pure.parts) != 1
        or pure.name != name
        or name in {".", ".."}
        or "\x00" in name
    ):
        raise ScopeViolation(f"{label} is not a safe basename")
    return name


def _open_directory_chain(path: Path, *, label: str) -> int:
    """Open every absolute path component with openat and O_NOFOLLOW."""

    _assert_no_forbidden_path(path, label=label)
    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if nofollow is None or directory is None:
        raise ScopeViolation(f"{label} requires O_NOFOLLOW and O_DIRECTORY")
    flags = os.O_RDONLY | directory | nofollow | getattr(os, "O_CLOEXEC", 0)
    current = os.open("/", flags)
    try:
        for component in path.parts[1:]:
            next_fd = os.open(component, flags, dir_fd=current)
            os.close(current)
            current = next_fd
        return current
    except BaseException:
        os.close(current)
        raise


def _read_regular_at(
    directory_fd: int,
    name: str,
    *,
    label: str,
    maximum_bytes: int,
) -> tuple[bytes, str, os.stat_result]:
    _safe_basename(name, label=label)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(name, flags, dir_fd=directory_fd)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ProtocolError(f"{label} is not a regular file")
        if before.st_size < 0 or before.st_size > maximum_bytes:
            raise ProtocolError(f"{label} exceeds the bounded read limit")
        chunks: list[bytes] = []
        digest = hashlib.sha256()
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise ProtocolError(f"{label} ended before its stat size")
            chunks.append(chunk)
            digest.update(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ProtocolError(f"{label} grew during its verified read")
        after = os.fstat(descriptor)
        identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
        if identity_before != identity_after:
            raise ProtocolError(f"{label} changed during its verified read")
        return b"".join(chunks), digest.hexdigest(), after
    finally:
        os.close(descriptor)


def _read_regular_path(path: Path, *, label: str, maximum_bytes: int) -> tuple[bytes, str]:
    _assert_no_forbidden_path(path, label=label)
    parent_fd = _open_directory_chain(path.parent, label=f"{label} parent")
    try:
        payload, digest, _ = _read_regular_at(
            parent_fd, path.name, label=label, maximum_bytes=maximum_bytes
        )
        return payload, digest
    finally:
        os.close(parent_fd)


def _load_json_bytes(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        decoded = payload.decode("utf-8", errors="strict")
        document = json.loads(decoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(document, dict):
        raise ProtocolError(f"{label} must contain one JSON object")
    return document


def _require_equal(actual: Any, expected: Any, *, label: str) -> None:
    if type(actual) is not type(expected) or actual != expected:
        raise ProtocolError(f"{label} does not match the frozen acquisition contract")


def _validate_protocol(document: Mapping[str, Any]) -> None:
    _require_equal(document.get("schema_version"), SCHEMA_VERSION, label="schema_version")
    _require_equal(document.get("protocol_id"), PROTOCOL_ID, label="protocol_id")
    _require_equal(document.get("dataset_accession"), DATASET_ACCESSION, label="dataset_accession")
    _require_equal(document.get("bioproject_accession"), BIOPROJECT_ACCESSION, label="bioproject_accession")

    authority = document.get("input_authority")
    if not isinstance(authority, Mapping):
        raise ProtocolError("input_authority must be an object")
    _require_equal(
        authority.get("bundle_directory"),
        str(EXPECTED_MANIFEST_BUNDLE_ROOT),
        label="input_authority.bundle_directory",
    )
    _require_equal(
        authority.get("bundle_basename"),
        EXPECTED_MANIFEST_BUNDLE_BASENAME,
        label="input_authority.bundle_basename",
    )
    source_marker = authority.get("terminal_marker")
    manifest = authority.get("canonical_manifest")
    summary = authority.get("manifest_summary")
    source_report = authority.get("source_file_report")
    source_sums = authority.get("source_sha256sums")
    if not all(
        isinstance(item, Mapping)
        for item in (source_marker, manifest, summary, source_report, source_sums)
    ):
        raise ProtocolError("input authority member specifications must be objects")
    for actual, expected, label in (
        (source_marker.get("filename"), EXPECTED_SOURCE_COMMIT_FILENAME, "terminal marker filename"),
        (source_marker.get("bytes"), EXPECTED_SOURCE_COMMIT_BYTES, "terminal marker bytes"),
        (source_marker.get("sha256"), EXPECTED_SOURCE_COMMIT_SHA256, "terminal marker sha256"),
        (manifest.get("filename"), EXPECTED_MANIFEST_FILENAME, "manifest filename"),
        (manifest.get("bytes"), EXPECTED_MANIFEST_BYTES, "manifest bytes"),
        (manifest.get("sha256"), EXPECTED_MANIFEST_SHA256, "manifest sha256"),
        (manifest.get("exact_header"), list(EXPECTED_MANIFEST_HEADER), "manifest header"),
        (manifest.get("exact_row_count"), EXPECTED_FILE_COUNT, "manifest row count"),
        (manifest.get("exact_run_count"), EXPECTED_RUN_COUNT, "manifest run count"),
        (manifest.get("total_fastq_bytes"), EXPECTED_TOTAL_BYTES, "manifest total bytes"),
        (manifest.get("expected_run_accessions"), list(EXPECTED_RUN_ACCESSIONS), "manifest run set"),
        (summary.get("filename"), EXPECTED_SUMMARY_FILENAME, "summary filename"),
        (summary.get("bytes"), EXPECTED_SUMMARY_BYTES, "summary bytes"),
        (summary.get("sha256"), EXPECTED_SUMMARY_SHA256, "summary sha256"),
        (source_report.get("filename"), EXPECTED_SOURCE_REPORT_FILENAME, "source report filename"),
        (source_report.get("bytes"), EXPECTED_SOURCE_REPORT_BYTES, "source report bytes"),
        (source_report.get("sha256"), EXPECTED_SOURCE_REPORT_SHA256, "source report sha256"),
        (source_sums.get("filename"), EXPECTED_SOURCE_SHA256SUMS_FILENAME, "source sums filename"),
        (source_sums.get("bytes"), EXPECTED_SOURCE_SHA256SUMS_BYTES, "source sums bytes"),
        (source_sums.get("sha256"), EXPECTED_SOURCE_SHA256SUMS_SHA256, "source sums sha256"),
    ):
        _require_equal(actual, expected, label=label)

    implementation = document.get("implementation_binding")
    if not isinstance(implementation, Mapping):
        raise ProtocolError("implementation_binding must be an object")
    if implementation.get("status") not in {"UNKNOWN_NOT_ASSERTED", "BOUND"}:
        raise ProtocolError("implementation binding status is not closed")
    _require_equal(
        implementation.get("binding_mode"),
        "TWO_COMMIT_NON_SELF_REFERENTIAL",
        label="implementation binding mode",
    )
    _require_equal(
        implementation.get("implementation_script_repo_path"),
        IMPLEMENTATION_SCRIPT_REPO_PATH,
        label="implementation script path",
    )
    _require_equal(
        implementation.get("protocol_repo_path"),
        PROTOCOL_REPO_PATH,
        label="implementation protocol path",
    )

    policy = document.get("download_policy")
    resume = document.get("resume_contract")
    output = document.get("output_contract")
    integrity = document.get("integrity_gate")
    scope = document.get("scope_guard")
    capacity = document.get("capacity_gate")
    concurrency = document.get("concurrency_truth")
    if not all(
        isinstance(item, Mapping)
        for item in (policy, resume, output, integrity, scope, capacity, concurrency)
    ):
        raise ProtocolError(
            "download, resume, output, integrity, scope, capacity, and concurrency contracts are required"
        )
    critical_pairs = (
        (policy.get("network_protocol"), "HTTPS_ONLY", "network protocol"),
        (policy.get("official_host_allowlist"), [OFFICIAL_HOST], "official host allowlist"),
        (policy.get("redirects_allowed"), False, "redirect policy"),
        (policy.get("environment_proxy_allowed"), False, "proxy policy"),
        (policy.get("default_workers"), DEFAULT_WORKERS, "default workers"),
        (policy.get("maximum_workers"), MAXIMUM_WORKERS, "maximum workers"),
        (policy.get("shell_allowed"), False, "shell policy"),
        (policy.get("fixed_argv_read_only_git_subprocess_allowed"), True, "Git verifier policy"),
        (policy.get("git_binary"), GIT_BINARY, "Git binary"),
        (resume.get("resume_requires_http_206"), True, "resume HTTP status gate"),
        (resume.get("resume_requires_exact_content_range"), True, "resume content-range gate"),
        (output.get("base_directory"), str(EXPECTED_OUTPUT_ROOT), "output base"),
        (output.get("progress_status_filename"), PROGRESS_STATUS_FILENAME, "progress status filename"),
        (output.get("final_status_filename"), FINAL_STATUS_FILENAME, "final status filename"),
        (output.get("maximum_progress_status_bytes"), MAXIMUM_PROGRESS_STATUS_BYTES, "progress status byte limit"),
        (output.get("maximum_attempt_history_entries"), MAXIMUM_ATTEMPT_HISTORY_ENTRIES, "attempt history limit"),
        (output.get("never_overwrite_completed_fastq"), True, "no-overwrite gate"),
        (output.get("terminal_marker_written_last"), True, "terminal write-order gate"),
        (output.get("derived_publication_members_reused_only_when_exact"), True, "derived reuse gate"),
        (integrity.get("required_verified_file_count"), EXPECTED_FILE_COUNT, "verified file gate"),
        (integrity.get("required_verified_run_count"), EXPECTED_RUN_COUNT, "verified run gate"),
        (integrity.get("required_verified_total_bytes"), EXPECTED_TOTAL_BYTES, "verified byte gate"),
        (integrity.get("per_file_ena_repository_md5_required"), True, "repository MD5 gate"),
        (integrity.get("per_file_local_sha256_required"), True, "local SHA256 gate"),
        (integrity.get("qualified_study_contribution"), 0, "study contribution boundary"),
        (integrity.get("training_allowed"), False, "training boundary"),
        (integrity.get("next_phase_authorized"), False, "phase boundary"),
        (scope.get("gpu_allowed"), False, "GPU boundary"),
        (scope.get("training_allowed"), False, "scope training boundary"),
        (capacity.get("required_before_any_network_request"), True, "capacity timing gate"),
        (capacity.get("minimum_safety_margin_bytes"), MINIMUM_CAPACITY_SAFETY_MARGIN_BYTES, "capacity minimum margin"),
        (capacity.get("safety_margin_basis_points_of_remaining"), CAPACITY_SAFETY_MARGIN_BASIS_POINTS, "capacity fractional margin"),
        (concurrency.get("mechanism"), "SINGLE_HOST_ADVISORY_FLOCK_ON_O_NOFOLLOW_REGULAR_FILE", "concurrency mechanism"),
        (concurrency.get("cross_host_exclusion"), "NOT_ESTABLISHED", "cross-host truth"),
        (concurrency.get("allowed_execution_scope"), "ONE_HOST_ONLY", "concurrency scope"),
    )
    for actual, expected, label in critical_pairs:
        _require_equal(actual, expected, label=label)


def load_protocol(path: Path) -> tuple[dict[str, Any], str]:
    if path.name != PROTOCOL_BASENAME:
        raise ProtocolError("protocol basename is outside the frozen allowlist")
    payload, digest = _read_regular_path(path, label="protocol", maximum_bytes=1024 * 1024)
    document = _load_json_bytes(payload, label="protocol")
    _validate_protocol(document)
    return document, digest


def _run_read_only_git(repo_root: Path, arguments: Sequence[str]) -> subprocess.CompletedProcess[bytes]:
    environment = {
        "LC_ALL": "C",
        "LANG": "C",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_OPTIONAL_LOCKS": "0",
    }
    try:
        return subprocess.run(
            [GIT_BINARY, "-C", str(repo_root), *arguments],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=30,
            env=environment,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ProtocolError("fixed read-only Git verification failed") from exc


def verify_implementation_binding(
    protocol_path: Path,
    protocol_sha256: str,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify the two-commit binding without a shell or self-reference."""

    binding = protocol.get("implementation_binding")
    if not isinstance(binding, Mapping):
        raise ProtocolError("implementation binding is missing")
    if binding.get("status") != "BOUND":
        raise ProtocolError(
            "implementation binding is UNKNOWN_NOT_ASSERTED; a separate binding commit is required"
        )
    implementation_commit = binding.get("implementation_commit")
    script_sha256 = binding.get("implementation_script_sha256")
    if not isinstance(implementation_commit, str) or not COMMIT_RE.fullmatch(
        implementation_commit
    ):
        raise ProtocolError("implementation commit is not a full Git object ID")
    if not isinstance(script_sha256, str) or not SHA256_RE.fullmatch(script_sha256):
        raise ProtocolError("implementation script SHA256 is not bound")
    _require_equal(
        binding.get("implementation_script_repo_path"),
        IMPLEMENTATION_SCRIPT_REPO_PATH,
        label="implementation script repo path",
    )
    _require_equal(
        binding.get("protocol_repo_path"),
        PROTOCOL_REPO_PATH,
        label="protocol repo path",
    )

    repo_root = protocol_path.parent.parent
    expected_protocol_path = repo_root / PROTOCOL_REPO_PATH
    expected_script_path = repo_root / IMPLEMENTATION_SCRIPT_REPO_PATH
    if protocol_path != expected_protocol_path:
        raise ProtocolError("protocol is not at its frozen repository-relative path")
    running_script_path = Path(os.path.abspath(__file__))
    if running_script_path != expected_script_path:
        raise ProtocolError("the executing script is not the Git-bound repository script")
    script_payload, current_script_sha = _read_regular_path(
        expected_script_path,
        label="implementation script",
        maximum_bytes=4 * 1024 * 1024,
    )
    if current_script_sha != script_sha256:
        raise ProtocolError("working-tree implementation script SHA256 is not bound")

    root_result = _run_read_only_git(repo_root, ("rev-parse", "--show-toplevel"))
    if root_result.returncode != 0:
        raise ProtocolError("protocol directory is not a verified Git worktree")
    try:
        reported_root = root_result.stdout.decode("utf-8", errors="strict").strip()
    except UnicodeDecodeError as exc:
        raise ProtocolError("Git worktree root is not UTF-8") from exc
    if reported_root != str(repo_root):
        raise ProtocolError("Git worktree root does not match the protocol root")

    head_result = _run_read_only_git(repo_root, ("rev-parse", "HEAD"))
    try:
        head_commit = head_result.stdout.decode("ascii", errors="strict").strip()
    except UnicodeDecodeError as exc:
        raise ProtocolError("current Git HEAD is not ASCII") from exc
    if head_result.returncode != 0 or not COMMIT_RE.fullmatch(head_commit):
        raise ProtocolError("current Git HEAD is not a full commit ID")
    ancestor = _run_read_only_git(
        repo_root,
        ("merge-base", "--is-ancestor", implementation_commit, head_commit),
    )
    if ancestor.returncode != 0:
        raise ProtocolError("implementation commit is not an ancestor of current HEAD")

    for cached in (False, True):
        arguments = ["diff", "--quiet"]
        if cached:
            arguments.append("--cached")
        arguments.extend(("--", PROTOCOL_REPO_PATH, IMPLEMENTATION_SCRIPT_REPO_PATH))
        if _run_read_only_git(repo_root, tuple(arguments)).returncode != 0:
            raise ProtocolError("protocol or implementation script has uncommitted Git changes")

    implementation_blob = _run_read_only_git(
        repo_root,
        ("show", f"{implementation_commit}:{IMPLEMENTATION_SCRIPT_REPO_PATH}"),
    )
    head_script_blob = _run_read_only_git(
        repo_root, ("show", f"{head_commit}:{IMPLEMENTATION_SCRIPT_REPO_PATH}")
    )
    head_protocol_blob = _run_read_only_git(
        repo_root, ("show", f"{head_commit}:{PROTOCOL_REPO_PATH}")
    )
    if any(
        result.returncode != 0
        for result in (implementation_blob, head_script_blob, head_protocol_blob)
    ):
        raise ProtocolError("Git binding blobs are unavailable")
    if _sha256_bytes(implementation_blob.stdout) != script_sha256:
        raise ProtocolError("implementation-commit script blob hash is not bound")
    if _sha256_bytes(head_script_blob.stdout) != script_sha256:
        raise ProtocolError("current-HEAD script blob hash is not bound")
    if head_script_blob.stdout != script_payload:
        raise ProtocolError("working-tree script bytes differ from the current Git blob")
    if _sha256_bytes(head_protocol_blob.stdout) != protocol_sha256:
        raise ProtocolError("working-tree protocol bytes differ from the current Git blob")
    return {
        "status": "BOUND",
        "binding_mode": "TWO_COMMIT_NON_SELF_REFERENTIAL",
        "implementation_commit": implementation_commit,
        "implementation_script_sha256": script_sha256,
        "binding_commit": head_commit,
        "protocol_sha256": protocol_sha256,
        "worktree_and_index_clean": True,
    }


def _validate_implementation_evidence(
    evidence: Mapping[str, Any],
    *,
    protocol_sha256: str,
    protocol: Mapping[str, Any],
) -> None:
    expected_keys = {
        "status",
        "binding_mode",
        "implementation_commit",
        "implementation_script_sha256",
        "binding_commit",
        "protocol_sha256",
        "worktree_and_index_clean",
    }
    if set(evidence) != expected_keys:
        raise ProtocolError("implementation verifier evidence field set is not exact")
    if evidence.get("status") != "BOUND":
        raise ProtocolError("implementation verifier did not establish BOUND status")
    if evidence.get("binding_mode") != "TWO_COMMIT_NON_SELF_REFERENTIAL":
        raise ProtocolError("implementation verifier binding mode is not exact")
    implementation_commit = evidence.get("implementation_commit")
    binding_commit = evidence.get("binding_commit")
    script_sha256 = evidence.get("implementation_script_sha256")
    if not isinstance(implementation_commit, str) or not COMMIT_RE.fullmatch(
        implementation_commit
    ):
        raise ProtocolError("verified implementation commit is invalid")
    if not isinstance(binding_commit, str) or not COMMIT_RE.fullmatch(binding_commit):
        raise ProtocolError("verified binding commit is invalid")
    if not isinstance(script_sha256, str) or not SHA256_RE.fullmatch(script_sha256):
        raise ProtocolError("verified implementation script SHA256 is invalid")
    if evidence.get("protocol_sha256") != protocol_sha256:
        raise ProtocolError("verified protocol SHA256 is inconsistent")
    if evidence.get("worktree_and_index_clean") is not True:
        raise ProtocolError("verified implementation worktree truth is not clean")
    binding = protocol.get("implementation_binding")
    if not isinstance(binding, Mapping):
        raise ProtocolError("protocol implementation binding is absent")
    if (
        binding.get("status") != "BOUND"
        or binding.get("implementation_commit") != implementation_commit
        or binding.get("implementation_script_sha256") != script_sha256
    ):
        raise ProtocolError("implementation evidence does not match the protocol binding")


def _expected_url(run_accession: str, mate: int) -> str:
    numeric = run_accession[3:]
    prefix = f"SRR{numeric[:3]}"
    # The exact frozen ENA objects use the extended-accession shard present in
    # the committed authority (042..070 for this closed run set).  Because the
    # run set itself is closed, deriving that three-character shard from the
    # final two accession digits is both stricter and clearer than accepting an
    # arbitrary manifest path component.
    shard = f"{int(numeric[-2:]):03d}"
    return (
        f"https://{OFFICIAL_HOST}/vol1/fastq/{prefix}/{shard}/"
        f"{run_accession}/{run_accession}_{mate}.fastq.gz"
    )


def _validate_official_url(url: str, *, run_accession: str, mate: int) -> None:
    if url != _expected_url(run_accession, mate):
        raise ManifestError("FASTQ URL is not the exact official ENA object path")
    split = urllib.parse.urlsplit(url)
    if (
        split.scheme != "https"
        or split.hostname != OFFICIAL_HOST
        or split.netloc != OFFICIAL_HOST
        or split.port is not None
        or split.username is not None
        or split.password is not None
        or split.query
        or split.fragment
        or "%" in split.path
    ):
        raise ManifestError("FASTQ URL violates the HTTPS host/path allowlist")


def parse_manifest(payload: bytes) -> tuple[ManifestEntry, ...]:
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ManifestError("canonical manifest is not UTF-8") from exc
    if "\x00" in text:
        raise ManifestError("canonical manifest contains a NUL byte")
    reader = csv.reader(io.StringIO(text, newline=""), delimiter="\t", strict=True)
    try:
        rows = list(reader)
    except csv.Error as exc:
        raise ManifestError("canonical manifest is not strict TSV") from exc
    if not rows or tuple(rows[0]) != EXPECTED_MANIFEST_HEADER:
        raise ManifestError("canonical manifest header is not exact")
    if len(rows) - 1 != EXPECTED_FILE_COUNT:
        raise ManifestError("canonical manifest does not contain exactly 48 rows")

    entries: list[ManifestEntry] = []
    observed_keys: set[tuple[str, int]] = set()
    observed_urls: set[str] = set()
    for number, row in enumerate(rows[1:], start=2):
        if len(row) != len(EXPECTED_MANIFEST_HEADER):
            raise ManifestError(f"manifest row {number} does not have exactly six fields")
        run, mate_text, url, ftp_path, byte_text, repository_md5 = row
        if not RUN_RE.fullmatch(run) or run not in EXPECTED_RUN_ACCESSIONS:
            raise ManifestError(f"manifest row {number} has an unregistered run accession")
        if mate_text not in {"1", "2"}:
            raise ManifestError(f"manifest row {number} has an invalid mate")
        mate = int(mate_text)
        _validate_official_url(url, run_accession=run, mate=mate)
        if ftp_path != url.removeprefix("https://"):
            raise ManifestError(f"manifest row {number} has inconsistent FTP and HTTPS paths")
        if not POSITIVE_INTEGER_RE.fullmatch(byte_text):
            raise ManifestError(f"manifest row {number} has an invalid byte count")
        expected_bytes = int(byte_text)
        if not MD5_RE.fullmatch(repository_md5):
            raise ManifestError(f"manifest row {number} has an invalid repository MD5")
        key = (run, mate)
        if key in observed_keys or url in observed_urls:
            raise ManifestError("canonical manifest contains a duplicate key or URL")
        observed_keys.add(key)
        observed_urls.add(url)
        entries.append(
            ManifestEntry(
                run_accession=run,
                mate=mate,
                url=url,
                ftp_path=ftp_path,
                expected_bytes=expected_bytes,
                repository_md5=repository_md5,
            )
        )

    expected_keys = {(run, mate) for run in EXPECTED_RUN_ACCESSIONS for mate in (1, 2)}
    if observed_keys != expected_keys:
        raise ManifestError("canonical manifest run/mate set is not exact")
    expected_order = sorted(entries, key=lambda item: (int(item.run_accession[3:]), item.mate))
    if entries != expected_order:
        raise ManifestError("canonical manifest row order is not canonical")
    if sum(entry.expected_bytes for entry in entries) != EXPECTED_TOTAL_BYTES:
        raise ManifestError("canonical manifest aggregate byte count is not exact")
    return tuple(entries)


def _parse_sha256sums(payload: bytes, *, label: str) -> dict[str, str]:
    try:
        text = payload.decode("ascii", errors="strict")
    except UnicodeDecodeError as exc:
        raise ProtocolError(f"{label} is not ASCII") from exc
    if not text.endswith("\n"):
        raise ProtocolError(f"{label} lacks its terminal newline")
    result: dict[str, str] = {}
    for line in text.splitlines():
        if len(line) < 67 or line[64:66] != "  ":
            raise ProtocolError(f"{label} has a malformed checksum row")
        digest, name = line[:64], line[66:]
        if not SHA256_RE.fullmatch(digest):
            raise ProtocolError(f"{label} contains an invalid SHA256")
        _safe_basename(name, label=f"{label} member")
        if name in result:
            raise ProtocolError(f"{label} contains a duplicate member")
        result[name] = digest
    return result


def load_manifest_authority() -> tuple[tuple[ManifestEntry, ...], dict[str, Any]]:
    if EXPECTED_MANIFEST_BUNDLE_ROOT.name != EXPECTED_MANIFEST_BUNDLE_BASENAME:
        raise ProtocolError("manifest bundle basename constant is inconsistent")
    bundle_fd = _open_directory_chain(
        EXPECTED_MANIFEST_BUNDLE_ROOT, label="committed manifest bundle"
    )
    try:
        marker_payload, marker_digest, marker_stat = _read_regular_at(
            bundle_fd,
            EXPECTED_SOURCE_COMMIT_FILENAME,
            label="source terminal marker",
            maximum_bytes=64 * 1024,
        )
        if marker_stat.st_size != EXPECTED_SOURCE_COMMIT_BYTES:
            raise ProtocolError("source terminal marker byte count is not exact")
        if marker_digest != EXPECTED_SOURCE_COMMIT_SHA256:
            raise ProtocolError("source terminal marker SHA256 is not exact")
        marker = _load_json_bytes(marker_payload, label="source terminal marker")
        required_marker_values = {
            "record_type": "GSE200304_ENA_FASTQ_MANIFEST_PUBLICATION_COMMIT",
            "dataset_accession": DATASET_ACCESSION,
            "bioproject_accession": BIOPROJECT_ACCESSION,
            "publication_status": "MANIFEST_COMMITTED_FASTQ_NOT_DOWNLOADED",
            "run_count": EXPECTED_RUN_COUNT,
            "paired_fastq_file_count": EXPECTED_FILE_COUNT,
            "total_fastq_bytes": EXPECTED_TOTAL_BYTES,
            "fastq_bodies_downloaded": 0,
            "repository_md5_recomputed_count": 0,
            "qualified_study_contribution": 0,
            "training_allowed": False,
            "next_phase_authorized": False,
        }
        for key, expected in required_marker_values.items():
            _require_equal(marker.get(key), expected, label=f"source terminal marker {key}")
        expected_member_sha = {
            EXPECTED_MANIFEST_FILENAME: EXPECTED_MANIFEST_SHA256,
            EXPECTED_SOURCE_REPORT_FILENAME: EXPECTED_SOURCE_REPORT_SHA256,
            EXPECTED_SUMMARY_FILENAME: EXPECTED_SUMMARY_SHA256,
            EXPECTED_SOURCE_SHA256SUMS_FILENAME: EXPECTED_SOURCE_SHA256SUMS_SHA256,
        }
        _require_equal(
            marker.get("member_set"),
            sorted(expected_member_sha),
            label="source terminal marker member set",
        )
        _require_equal(
            marker.get("member_sha256"),
            {name: expected_member_sha[name] for name in sorted(expected_member_sha)},
            label="source terminal marker member hashes",
        )
        actual_bundle_names = set(os.listdir(bundle_fd))
        expected_bundle_names = set(expected_member_sha) | {
            EXPECTED_SOURCE_COMMIT_FILENAME
        }
        if actual_bundle_names != expected_bundle_names:
            raise ProtocolError("source manifest bundle member set is not exact")

        member_specifications = {
            EXPECTED_MANIFEST_FILENAME: (EXPECTED_MANIFEST_BYTES, 1024 * 1024),
            EXPECTED_SOURCE_REPORT_FILENAME: (EXPECTED_SOURCE_REPORT_BYTES, 1024 * 1024),
            EXPECTED_SUMMARY_FILENAME: (EXPECTED_SUMMARY_BYTES, 1024 * 1024),
            EXPECTED_SOURCE_SHA256SUMS_FILENAME: (
                EXPECTED_SOURCE_SHA256SUMS_BYTES,
                1024 * 1024,
            ),
        }
        member_payloads: dict[str, bytes] = {}
        for name, (expected_bytes, limit) in member_specifications.items():
            payload, digest, member_stat = _read_regular_at(
                bundle_fd, name, label=f"source bundle member {name}", maximum_bytes=limit
            )
            if member_stat.st_size != expected_bytes or digest != expected_member_sha[name]:
                if name == EXPECTED_MANIFEST_FILENAME:
                    raise ManifestError(
                        "canonical manifest byte count or SHA256 is not exact"
                    )
                raise ProtocolError(f"source bundle member is not exact: {name}")
            member_payloads[name] = payload

        source_sums = _parse_sha256sums(
            member_payloads[EXPECTED_SOURCE_SHA256SUMS_FILENAME],
            label="source SHA256SUMS",
        )
        expected_source_sums = {
            EXPECTED_MANIFEST_FILENAME: EXPECTED_MANIFEST_SHA256,
            EXPECTED_SOURCE_REPORT_FILENAME: EXPECTED_SOURCE_REPORT_SHA256,
            EXPECTED_SUMMARY_FILENAME: EXPECTED_SUMMARY_SHA256,
        }
        if source_sums != expected_source_sums:
            raise ProtocolError("source SHA256SUMS semantic member map is not exact")

        summary = _load_json_bytes(
            member_payloads[EXPECTED_SUMMARY_FILENAME], label="manifest summary"
        )
        expected_summary_values = {
            "schema_version": "route_a_v3_ena_fastq_manifest_summary.v1",
            "dataset_accession": DATASET_ACCESSION,
            "bioproject_accession": BIOPROJECT_ACCESSION,
        }
        for key, expected in expected_summary_values.items():
            _require_equal(summary.get(key), expected, label=f"manifest summary {key}")
        canonical_summary = summary.get("canonical_manifest")
        aggregate_summary = summary.get("aggregate")
        verification_summary = summary.get("verification")
        if not all(
            isinstance(item, Mapping)
            for item in (canonical_summary, aggregate_summary, verification_summary)
        ):
            raise ProtocolError("manifest summary lacks its semantic sections")
        for actual, expected, label in (
            (canonical_summary.get("path"), EXPECTED_MANIFEST_FILENAME, "summary manifest path"),
            (canonical_summary.get("bytes"), EXPECTED_MANIFEST_BYTES, "summary manifest bytes"),
            (canonical_summary.get("sha256"), EXPECTED_MANIFEST_SHA256, "summary manifest SHA256"),
            (aggregate_summary.get("run_count"), EXPECTED_RUN_COUNT, "summary run count"),
            (aggregate_summary.get("paired_fastq_file_count"), EXPECTED_FILE_COUNT, "summary file count"),
            (aggregate_summary.get("total_fastq_bytes"), EXPECTED_TOTAL_BYTES, "summary total bytes"),
            (verification_summary.get("ena_two_files_per_run"), True, "summary paired gate"),
            (verification_summary.get("filename_run_and_mate_binding"), True, "summary filename gate"),
            (verification_summary.get("fastq_file_bodies_downloaded"), 0, "summary no-body truth"),
            (verification_summary.get("repository_md5_recomputed_from_fastq_bodies"), False, "summary MD5 truth"),
        ):
            _require_equal(actual, expected, label=label)

        manifest_payload = member_payloads[EXPECTED_MANIFEST_FILENAME]
        entries = parse_manifest(manifest_payload)
        return entries, marker
    finally:
        os.close(bundle_fd)


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise PublicationError("short write while publishing acquisition metadata")
        view = view[written:]


def _create_exclusive_at(
    directory_fd: int, name: str, payload: bytes, *, mode: int = 0o640
) -> None:
    _safe_basename(name, label="output member")
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(name, flags, mode, dir_fd=directory_fd)
    try:
        _write_all(descriptor, payload)
        os.fsync(descriptor)
    except BaseException:
        os.close(descriptor)
        raise
    else:
        os.close(descriptor)
    os.fsync(directory_fd)


def _atomic_replace_at(directory_fd: int, name: str, payload: bytes) -> None:
    _safe_basename(name, label="replaceable status member")
    temporary = f".{name}.tmp-{secrets.token_hex(12)}"
    _create_exclusive_at(directory_fd, temporary, payload)
    try:
        os.replace(temporary, name, src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
        os.fsync(directory_fd)
    except BaseException:
        try:
            os.unlink(temporary, dir_fd=directory_fd)
        except FileNotFoundError:
            pass
        raise


def _member_exists(directory_fd: int, name: str) -> bool:
    try:
        os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        return True
    except FileNotFoundError:
        return False


def _read_json_at(
    directory_fd: int,
    name: str,
    *,
    label: str,
    maximum_bytes: int = 16 * 1024 * 1024,
) -> dict[str, Any]:
    payload, _, _ = _read_regular_at(
        directory_fd, name, label=label, maximum_bytes=maximum_bytes
    )
    return _load_json_bytes(payload, label=label)


def _hash_regular_at(
    directory_fd: int, name: str, *, label: str
) -> tuple[int, str, str, FileIdentity]:
    _safe_basename(name, label=label)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(name, flags, dir_fd=directory_fd)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise IntegrityError(f"{label} is not a regular file")
        md5 = hashlib.md5(usedforsecurity=False)
        sha256 = hashlib.sha256()
        total = 0
        while True:
            chunk = os.read(descriptor, 8 * 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            md5.update(chunk)
            sha256.update(chunk)
        after = os.fstat(descriptor)
        before_identity = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        after_identity = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if before_identity != after_identity or total != after.st_size:
            raise IntegrityError(f"{label} changed during digest verification")
        identity = FileIdentity(
            device=after.st_dev,
            inode=after.st_ino,
            size=after.st_size,
            mtime_ns=after.st_mtime_ns,
            ctime_ns=after.st_ctime_ns,
        )
        return total, md5.hexdigest(), sha256.hexdigest(), identity
    finally:
        os.close(descriptor)


def _identity_at(directory_fd: int, name: str, *, label: str) -> FileIdentity:
    _safe_basename(name, label=label)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(name, flags, dir_fd=directory_fd)
    try:
        value = os.fstat(descriptor)
        if not stat.S_ISREG(value.st_mode):
            raise IntegrityError(f"{label} is not regular")
        return FileIdentity(
            device=value.st_dev,
            inode=value.st_ino,
            size=value.st_size,
            mtime_ns=value.st_mtime_ns,
            ctime_ns=value.st_ctime_ns,
        )
    finally:
        os.close(descriptor)


def _transfer_binding(entry: ManifestEntry) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": "GSE200304_FASTQ_IMMUTABLE_TRANSFER_BINDING",
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "source_terminal_marker_sha256": EXPECTED_SOURCE_COMMIT_SHA256,
        "run_accession": entry.run_accession,
        "mate": entry.mate,
        "filename": entry.filename,
        "url": entry.url,
        "expected_bytes": entry.expected_bytes,
        "repository_md5": entry.repository_md5,
        "qualified_study_contribution": 0,
        "training_allowed": False,
        "next_phase_authorized": False,
    }


def _ensure_transfer_binding(
    directory_fd: int, entry: ManifestEntry, *, resume: bool
) -> None:
    expected = _transfer_binding(entry)
    name = entry.transfer_binding_filename
    if _member_exists(directory_fd, name):
        if not resume:
            raise ResumeError("pre-existing transfer binding requires explicit resume")
        actual = _read_json_at(directory_fd, name, label=f"transfer binding {entry.filename}")
        if actual != expected:
            raise ResumeError("partial transfer binding does not exactly match the manifest")
        return
    if _member_exists(directory_fd, entry.part_filename) or _member_exists(
        directory_fd, entry.filename
    ):
        raise ResumeError("FASTQ or partial exists without its immutable transfer binding")
    _create_exclusive_at(directory_fd, name, _json_bytes(expected))


def _validate_existing_transfer_binding(
    directory_fd: int, entry: ManifestEntry
) -> bool:
    name = entry.transfer_binding_filename
    if not _member_exists(directory_fd, name):
        return False
    actual = _read_json_at(
        directory_fd,
        name,
        label=f"transfer binding {entry.filename}",
        maximum_bytes=64 * 1024,
    )
    if actual != _transfer_binding(entry):
        raise ResumeError("existing transfer binding does not exactly match the manifest")
    return True


def _default_capacity_probe(directory_fd: int) -> int:
    values = os.fstatvfs(directory_fd)
    available = values.f_bavail * values.f_frsize
    if available < 0:
        raise CapacityError(
            "fstatvfs reported a negative available-byte value",
            {
                "available_bytes": available,
                "capacity_source": "FSTATVFS_F_BAVAIL_TIMES_F_FRSIZE",
                "passed": False,
            },
        )
    return available


def _capacity_preflight(
    directory_fd: int,
    entries: Sequence[ManifestEntry],
    *,
    resume: bool,
    capacity_probe: Callable[[int], int],
) -> dict[str, Any]:
    remaining_bytes = 0
    trusted_partial_bytes = 0
    reverified_completed_bytes = 0
    for entry in entries:
        binding_exists = _validate_existing_transfer_binding(directory_fd, entry)
        final_exists = _member_exists(directory_fd, entry.filename)
        part_exists = _member_exists(directory_fd, entry.part_filename)
        if not resume and (binding_exists or final_exists or part_exists):
            raise ResumeError("fresh output unexpectedly contains transfer state")
        if (final_exists or part_exists) and not binding_exists:
            raise ResumeError("transfer state exists without an exact immutable binding")
        if final_exists:
            if part_exists:
                final_stat = os.stat(
                    entry.filename, dir_fd=directory_fd, follow_symlinks=False
                )
                part_stat = os.stat(
                    entry.part_filename, dir_fd=directory_fd, follow_symlinks=False
                )
                if (
                    not stat.S_ISREG(final_stat.st_mode)
                    or not stat.S_ISREG(part_stat.st_mode)
                    or (final_stat.st_dev, final_stat.st_ino)
                    != (part_stat.st_dev, part_stat.st_ino)
                ):
                    raise ResumeError(
                        "final FASTQ and partial coexist with different identities"
                    )
            _verify_entry_file(directory_fd, entry, name=entry.filename)
            reverified_completed_bytes += entry.expected_bytes
            continue
        if part_exists:
            partial_stat = os.stat(
                entry.part_filename, dir_fd=directory_fd, follow_symlinks=False
            )
            if (
                not stat.S_ISREG(partial_stat.st_mode)
                or partial_stat.st_size < 0
                or partial_stat.st_size > entry.expected_bytes
            ):
                raise ResumeError("bound partial has an invalid byte count")
            trusted_partial_bytes += partial_stat.st_size
            remaining_bytes += entry.expected_bytes - partial_stat.st_size
        else:
            remaining_bytes += entry.expected_bytes

    fractional_margin = (
        remaining_bytes * CAPACITY_SAFETY_MARGIN_BASIS_POINTS + 9_999
    ) // 10_000
    safety_margin = (
        max(MINIMUM_CAPACITY_SAFETY_MARGIN_BYTES, fractional_margin)
        if remaining_bytes
        else 0
    )
    required_available = remaining_bytes + safety_margin
    available = capacity_probe(directory_fd)
    if not isinstance(available, int) or isinstance(available, bool) or available < 0:
        raise CapacityError(
            "capacity probe did not return a non-negative integer",
            {
                "remaining_bytes": remaining_bytes,
                "required_available_bytes": required_available,
                "available_bytes": available,
                "passed": False,
            },
        )
    evidence = {
        "capacity_source": "FSTATVFS_F_BAVAIL_TIMES_F_FRSIZE",
        "remaining_bytes": remaining_bytes,
        "trusted_partial_bytes": trusted_partial_bytes,
        "reverified_completed_bytes": reverified_completed_bytes,
        "safety_margin_bytes": safety_margin,
        "minimum_safety_margin_bytes": MINIMUM_CAPACITY_SAFETY_MARGIN_BYTES,
        "safety_margin_basis_points": CAPACITY_SAFETY_MARGIN_BASIS_POINTS,
        "required_available_bytes": required_available,
        "available_bytes": available,
        "passed": available >= required_available,
    }
    if not evidence["passed"]:
        raise CapacityError(
            "insufficient filesystem capacity before network acquisition", evidence
        )
    return evidence


def _header(response: HTTPResponse, name: str) -> str | None:
    headers = response.headers
    value: Any = None
    if hasattr(headers, "get"):
        value = headers.get(name)
        if value is None:
            value = headers.get(name.lower())
    return None if value is None else str(value).strip()


def _validate_response(response: HTTPResponse, entry: ManifestEntry, *, offset: int) -> None:
    status = int(getattr(response, "status", 0))
    final_url = response.geturl()
    if final_url != entry.url:
        raise TransportError("redirected or substituted response URL is forbidden")
    _validate_official_url(final_url, run_accession=entry.run_accession, mate=entry.mate)
    remaining = entry.expected_bytes - offset
    if offset == 0:
        if status != 200:
            raise TransportError("fresh transfer requires HTTP 200")
        content_range = _header(response, "Content-Range")
        if content_range:
            raise TransportError("fresh transfer returned an unexpected Content-Range")
    else:
        if status != 206:
            raise ResumeError("resumed transfer requires HTTP 206")
        expected_range = f"bytes {offset}-{entry.expected_bytes - 1}/{entry.expected_bytes}"
        if _header(response, "Content-Range") != expected_range:
            raise ResumeError("resumed transfer Content-Range is not exact")
    if _header(response, "Content-Length") != str(remaining):
        error = ResumeError if offset else TransportError
        raise error("response Content-Length is not the exact remaining byte count")
    encoding = _header(response, "Content-Encoding")
    if encoding not in {None, "", "identity"}:
        raise TransportError("encoded response bodies are forbidden")


def _append_response_to_part(
    directory_fd: int,
    entry: ManifestEntry,
    response: HTTPResponse,
    *,
    offset: int,
    chunk_bytes: int,
) -> None:
    flags = (
        os.O_WRONLY
        | os.O_APPEND
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(entry.part_filename, flags, dir_fd=directory_fd)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size != offset:
            raise ResumeError("partial size changed before append")
        remaining = entry.expected_bytes - offset
        while remaining:
            chunk = response.read(min(chunk_bytes, remaining))
            if not chunk:
                raise TransportError("response ended before the expected byte count")
            if len(chunk) > remaining:
                raise TransportError("response exceeded the expected byte count")
            _write_all(descriptor, chunk)
            remaining -= len(chunk)
        if response.read(1):
            raise TransportError("response contains bytes beyond the exact manifest length")
        os.fsync(descriptor)
        after = os.fstat(descriptor)
        if after.st_dev != before.st_dev or after.st_ino != before.st_ino:
            raise ResumeError("partial file identity changed while appending")
        if after.st_size != entry.expected_bytes:
            raise TransportError("completed partial size is not exact")
    except BaseException:
        try:
            os.fsync(descriptor)
        except OSError:
            pass
        raise
    finally:
        os.close(descriptor)


def _create_empty_part(directory_fd: int, entry: ManifestEntry) -> None:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(entry.part_filename, flags, 0o640, dir_fd=directory_fd)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.fsync(directory_fd)


def _promote_verified_part(directory_fd: int, entry: ManifestEntry) -> None:
    """No-overwrite promotion using linkat followed by unlink of the old name."""

    try:
        os.link(
            entry.part_filename,
            entry.filename,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
            follow_symlinks=False,
        )
    except FileExistsError as exc:
        raise PublicationError("completed FASTQ already exists; it will not be overwritten") from exc
    os.fsync(directory_fd)
    os.unlink(entry.part_filename, dir_fd=directory_fd)
    os.fsync(directory_fd)


def _verify_entry_file(
    directory_fd: int, entry: ManifestEntry, *, name: str
) -> tuple[str, FileIdentity]:
    size, md5, sha256, identity = _hash_regular_at(
        directory_fd, name, label=f"FASTQ {entry.run_accession} mate {entry.mate}"
    )
    if size != entry.expected_bytes:
        raise IntegrityError("FASTQ byte count does not match the ENA manifest")
    if md5 != entry.repository_md5:
        raise IntegrityError("FASTQ MD5 does not match the ENA repository MD5")
    if not SHA256_RE.fullmatch(sha256):
        raise IntegrityError("local FASTQ SHA256 is invalid")
    return sha256, identity


def _remove_duplicate_part_after_link_crash(
    directory_fd: int, entry: ManifestEntry
) -> None:
    if not _member_exists(directory_fd, entry.part_filename):
        return
    final_stat = os.stat(entry.filename, dir_fd=directory_fd, follow_symlinks=False)
    part_stat = os.stat(entry.part_filename, dir_fd=directory_fd, follow_symlinks=False)
    if (
        not stat.S_ISREG(final_stat.st_mode)
        or not stat.S_ISREG(part_stat.st_mode)
        or (final_stat.st_dev, final_stat.st_ino) != (part_stat.st_dev, part_stat.st_ino)
    ):
        raise ResumeError("final FASTQ and partial both exist with different identities")
    os.unlink(entry.part_filename, dir_fd=directory_fd)
    os.fsync(directory_fd)


def _download_one(
    directory_fd: int,
    entry: ManifestEntry,
    *,
    resume: bool,
    transport_factory: Callable[[], DownloadTransport],
    timeout_seconds: int,
    chunk_bytes: int,
) -> FileResult:
    offset = 0
    try:
        _ensure_transfer_binding(directory_fd, entry, resume=resume)
        if _member_exists(directory_fd, entry.filename):
            if not resume:
                raise PublicationError("completed FASTQ unexpectedly exists on a fresh acquisition")
            _remove_duplicate_part_after_link_crash(directory_fd, entry)
            # Reuse truth comes only from a digest over the final name after any
            # crash-recovery unlink has completed.  A pre-cleanup digest plus a
            # later stat would leave a mutation window.
            sha256, identity = _verify_entry_file(
                directory_fd, entry, name=entry.filename
            )
            return FileResult(
                entry=entry,
                success=True,
                resumed_from_bytes=entry.expected_bytes,
                bytes_verified=entry.expected_bytes,
                repository_md5=entry.repository_md5,
                local_sha256=sha256,
                identity=identity,
                reused_completed_file=True,
            )

        if _member_exists(directory_fd, entry.part_filename):
            if not resume:
                raise ResumeError("pre-existing partial requires explicit resume")
            partial_stat = os.stat(
                entry.part_filename, dir_fd=directory_fd, follow_symlinks=False
            )
            if not stat.S_ISREG(partial_stat.st_mode):
                raise ResumeError("partial is not a regular file")
            offset = partial_stat.st_size
            if offset < 0 or offset > entry.expected_bytes:
                raise ResumeError("partial byte count exceeds the manifest object")
        else:
            _create_empty_part(directory_fd, entry)
            offset = 0

        if offset < entry.expected_bytes:
            transport = transport_factory()
            response = transport.open(
                entry.url, offset=offset, timeout_seconds=timeout_seconds
            )
            with closing(response):
                _validate_response(response, entry, offset=offset)
                _append_response_to_part(
                    directory_fd,
                    entry,
                    response,
                    offset=offset,
                    chunk_bytes=chunk_bytes,
                )
        _verify_entry_file(
            directory_fd, entry, name=entry.part_filename
        )
        _promote_verified_part(directory_fd, entry)
        # The hard-link/no-replace transition is followed by a full digest over
        # the final name; neither the pre-link digest nor an identity-only stat
        # is accepted as publication evidence.
        sha256, identity = _verify_entry_file(
            directory_fd, entry, name=entry.filename
        )
        return FileResult(
            entry=entry,
            success=True,
            resumed_from_bytes=offset,
            bytes_verified=entry.expected_bytes,
            repository_md5=entry.repository_md5,
            local_sha256=sha256,
            identity=identity,
        )
    except Exception as exc:
        error_code, error_message = _normalized_error(exc)
        return FileResult(
            entry=entry,
            success=False,
            resumed_from_bytes=offset,
            error_code=error_code,
            error_message=error_message,
        )


def _run_downloads(
    directory_fd: int,
    entries: Sequence[ManifestEntry],
    *,
    workers: int,
    resume: bool,
    transport_factory: Callable[[], DownloadTransport],
    timeout_seconds: int,
    chunk_bytes: int,
) -> list[FileResult]:
    results: list[FileResult] = []
    iterator = iter(entries)
    futures: dict[Future[FileResult], ManifestEntry] = {}
    failure_seen = False
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="gse200304-fastq") as pool:
        for _ in range(workers):
            try:
                entry = next(iterator)
            except StopIteration:
                break
            futures[
                pool.submit(
                    _download_one,
                    directory_fd,
                    entry,
                    resume=resume,
                    transport_factory=transport_factory,
                    timeout_seconds=timeout_seconds,
                    chunk_bytes=chunk_bytes,
                )
            ] = entry
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                futures.pop(future)
                result = future.result()
                results.append(result)
                if not result.success:
                    failure_seen = True
            if failure_seen:
                continue
            for _ in range(len(done)):
                try:
                    entry = next(iterator)
                except StopIteration:
                    break
                futures[
                    pool.submit(
                        _download_one,
                        directory_fd,
                        entry,
                        resume=resume,
                        transport_factory=transport_factory,
                        timeout_seconds=timeout_seconds,
                        chunk_bytes=chunk_bytes,
                    )
                ] = entry
    return sorted(results, key=lambda result: (int(result.entry.run_accession[3:]), result.entry.mate))


def _output_binding(
    *,
    protocol_sha256: str,
    output_directory: Path,
    implementation_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": "GSE200304_FASTQ_ACQUISITION_BINDING",
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": protocol_sha256,
        "dataset_accession": DATASET_ACCESSION,
        "bioproject_accession": BIOPROJECT_ACCESSION,
        "manifest_bundle_directory": str(EXPECTED_MANIFEST_BUNDLE_ROOT),
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "source_terminal_marker_sha256": EXPECTED_SOURCE_COMMIT_SHA256,
        "expected_run_count": EXPECTED_RUN_COUNT,
        "expected_file_count": EXPECTED_FILE_COUNT,
        "expected_total_bytes": EXPECTED_TOTAL_BYTES,
        "output_directory": str(output_directory),
        "implementation_binding": dict(implementation_evidence),
        "concurrency_truth": "SINGLE_HOST_FLOCK_ONLY_CROSS_HOST_NOT_ESTABLISHED",
        "qualified_study_contribution": 0,
        "training_allowed": False,
        "next_phase_authorized": False,
    }


def _status_document(
    existing: Mapping[str, Any] | None,
    *,
    output_directory: Path,
    attempt: Mapping[str, Any],
) -> dict[str, Any]:
    prior: list[Any] = []
    if existing is not None:
        raw = existing.get("attempts")
        if isinstance(raw, list):
            prior = list(raw)
    if len(prior) >= MAXIMUM_ATTEMPT_HISTORY_ENTRIES:
        raise ResumeError("progress status attempt history reached its frozen limit")
    document = {
        "schema_version": SCHEMA_VERSION,
        "record_type": "GSE200304_FASTQ_ACQUISITION_PROGRESS",
        "dataset_accession": DATASET_ACCESSION,
        "bioproject_accession": BIOPROJECT_ACCESSION,
        "output_directory": str(output_directory),
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "source_terminal_marker_sha256": EXPECTED_SOURCE_COMMIT_SHA256,
        "attempts": [*prior, dict(attempt)],
        "current_status": attempt.get("status"),
        "terminal_commit_present": False,
        "qualified_study_contribution": 0,
        "training_allowed": False,
        "next_phase_authorized": False,
    }
    _validate_progress_status(document, output_directory=output_directory)
    if len(_json_bytes(document)) > MAXIMUM_PROGRESS_STATUS_BYTES:
        raise ResumeError("progress status exceeds its frozen byte limit")
    return document


def _validate_progress_status(
    document: Mapping[str, Any], *, output_directory: Path
) -> None:
    required = {
        "schema_version": SCHEMA_VERSION,
        "record_type": "GSE200304_FASTQ_ACQUISITION_PROGRESS",
        "dataset_accession": DATASET_ACCESSION,
        "bioproject_accession": BIOPROJECT_ACCESSION,
        "output_directory": str(output_directory),
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "source_terminal_marker_sha256": EXPECTED_SOURCE_COMMIT_SHA256,
        "terminal_commit_present": False,
        "qualified_study_contribution": 0,
        "training_allowed": False,
        "next_phase_authorized": False,
    }
    for key, expected in required.items():
        if document.get(key) != expected or type(document.get(key)) is not type(expected):
            raise ResumeError(f"progress status field is not exact: {key}")
    expected_document_keys = set(required) | {"attempts", "current_status"}
    if set(document) != expected_document_keys:
        raise ResumeError("progress status field set is not exact")
    attempts = document.get("attempts")
    if (
        not isinstance(attempts, list)
        or not attempts
        or len(attempts) > MAXIMUM_ATTEMPT_HISTORY_ENTRIES
        or not all(isinstance(attempt, Mapping) for attempt in attempts)
    ):
        raise ResumeError("progress status attempt history is invalid")
    if document.get("current_status") != attempts[-1].get("status"):
        raise ResumeError("progress status current state is inconsistent")
    allowed_statuses = {
        "PREFLIGHT_FAILED_NOT_COMMITTED",
        "RUNNING_NOT_COMMITTED",
        "INTERRUPTED_NOT_COMMITTED",
        "FAILED_NOT_COMMITTED",
        "VERIFIED_ALL_FILES_READY_FOR_TERMINAL_COMMIT",
        "PUBLICATION_FAILED_NOT_COMMITTED",
    }
    for index, attempt in enumerate(attempts, start=1):
        attempt_number = attempt.get("attempt_number")
        workers = attempt.get("workers")
        if (
            type(attempt_number) is not int
            or attempt_number != index
            or type(attempt.get("resume_requested")) is not bool
            or type(workers) is not int
            or not (1 <= workers <= MAXIMUM_WORKERS)
            or not isinstance(attempt.get("started_at"), str)
            or attempt.get("status") not in allowed_statuses
            or type(attempt.get("attempted_files")) is not int
            or type(attempt.get("verified_files")) is not int
            or not isinstance(attempt.get("results"), list)
            or len(attempt.get("results")) > EXPECTED_FILE_COUNT
        ):
            raise ResumeError("progress status attempt schema is invalid")
        finished_at = attempt.get("finished_at")
        if finished_at is not None and not isinstance(finished_at, str):
            raise ResumeError("progress status finish timestamp is invalid")


def _read_progress_status(
    directory_fd: int, *, output_directory: Path
) -> dict[str, Any]:
    document = _read_json_at(
        directory_fd,
        PROGRESS_STATUS_FILENAME,
        label="existing acquisition progress",
        maximum_bytes=MAXIMUM_PROGRESS_STATUS_BYTES,
    )
    _validate_progress_status(document, output_directory=output_directory)
    return document


def _write_progress_status(directory_fd: int, document: Mapping[str, Any]) -> None:
    payload = _json_bytes(document)
    if len(payload) > MAXIMUM_PROGRESS_STATUS_BYTES:
        raise ResumeError("progress status exceeds its frozen byte limit")
    _atomic_replace_at(directory_fd, PROGRESS_STATUS_FILENAME, payload)


def _result_record(result: FileResult) -> dict[str, Any]:
    record: dict[str, Any] = {
        "run_accession": result.entry.run_accession,
        "mate": result.entry.mate,
        "filename": result.entry.filename,
        "success": result.success,
        "resumed_from_bytes": result.resumed_from_bytes,
        "expected_bytes": result.entry.expected_bytes,
        "expected_repository_md5": result.entry.repository_md5,
        "reused_completed_file": result.reused_completed_file,
    }
    if result.success:
        record.update(
            {
                "bytes_verified": result.bytes_verified,
                "repository_md5_verified": result.repository_md5,
                "local_sha256": result.local_sha256,
            }
        )
    else:
        record.update(
            {
                "error_code": result.error_code,
                "error_message": result.error_message,
            }
        )
    return record


def _reverify_final_results(
    directory_fd: int, results: Sequence[FileResult]
) -> list[FileResult]:
    reverified: list[FileResult] = []
    for result in results:
        if not result.success or result.identity is None:
            raise PublicationError("cannot commit a result without a verified file identity")
        sha256, identity = _verify_entry_file(
            directory_fd, result.entry, name=result.entry.filename
        )
        reverified.append(
            FileResult(
                entry=result.entry,
                success=True,
                resumed_from_bytes=result.resumed_from_bytes,
                bytes_verified=result.entry.expected_bytes,
                repository_md5=result.entry.repository_md5,
                local_sha256=sha256,
                identity=identity,
                reused_completed_file=result.reused_completed_file,
            )
        )
    return reverified


def _assert_reverified_identities_unchanged(
    directory_fd: int, results: Sequence[FileResult]
) -> None:
    for result in results:
        if result.identity is None:
            raise PublicationError("precommit FASTQ identity evidence is absent")
        current = _identity_at(
            directory_fd, result.entry.filename, label="terminal-adjacent FASTQ identity"
        )
        if current != result.identity:
            raise PublicationError(
                f"FASTQ changed after terminal precommit digest: {result.entry.filename}"
            )


def _list_output_names(directory_fd: int) -> set[str]:
    try:
        names = set(os.listdir(directory_fd))
    except (OSError, TypeError) as exc:
        raise PublicationError("exact descriptor-bound output listing is unavailable") from exc
    for name in names:
        _safe_basename(name, label="output directory member")
    return names


def _metadata_sha_at(directory_fd: int, name: str) -> str:
    _, _, sha256, _ = _hash_regular_at(directory_fd, name, label=f"metadata {name}")
    return sha256


def _ensure_exact_derived_member(
    directory_fd: int, name: str, payload: bytes
) -> str:
    if _member_exists(directory_fd, name):
        existing, digest, _ = _read_regular_at(
            directory_fd,
            name,
            label=f"reusable derived member {name}",
            maximum_bytes=max(len(payload), 1) + 1,
        )
        if existing != payload:
            raise PublicationError(
                f"pre-existing derived publication member is not exact: {name}"
            )
        return digest
    _create_exclusive_at(directory_fd, name, payload)
    return _sha256_bytes(payload)


def _final_status_document(
    *,
    output_directory: Path,
    implementation_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": "GSE200304_FASTQ_ACQUISITION_FINAL_STATUS",
        "publication_state": "VERIFIED_ALL_FILES_READY_FOR_TERMINAL_COMMIT",
        "dataset_accession": DATASET_ACCESSION,
        "bioproject_accession": BIOPROJECT_ACCESSION,
        "output_directory": str(output_directory),
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "source_terminal_marker_sha256": EXPECTED_SOURCE_COMMIT_SHA256,
        "verified_file_count": EXPECTED_FILE_COUNT,
        "verified_run_count": EXPECTED_RUN_COUNT,
        "verified_total_bytes": EXPECTED_TOTAL_BYTES,
        "repository_md5_verified_count": EXPECTED_FILE_COUNT,
        "local_sha256_recorded_count": EXPECTED_FILE_COUNT,
        "implementation_binding": dict(implementation_evidence),
        "terminal_marker_must_be_present_for_commit": True,
        "qualified_study_contribution": 0,
        "training_allowed": False,
        "next_phase_authorized": False,
    }


def _integrity_manifest_document(
    entries: Sequence[ManifestEntry],
    verified_results: Sequence[FileResult],
    *,
    implementation_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    by_name = {result.entry.filename: result for result in verified_results}
    rows: list[dict[str, Any]] = []
    for entry in entries:
        result = by_name.get(entry.filename)
        if result is None or result.local_sha256 is None:
            raise PublicationError("integrity manifest lacks a frozen FASTQ result")
        rows.append(
            {
                "run_accession": entry.run_accession,
                "mate": entry.mate,
                "filename": entry.filename,
                "source_url": entry.url,
                "bytes": entry.expected_bytes,
                "ena_repository_md5": entry.repository_md5,
                "local_sha256": result.local_sha256,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": "GSE200304_FASTQ_INTEGRITY_MANIFEST",
        "dataset_accession": DATASET_ACCESSION,
        "bioproject_accession": BIOPROJECT_ACCESSION,
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "source_terminal_marker_sha256": EXPECTED_SOURCE_COMMIT_SHA256,
        "verified_file_count": len(rows),
        "verified_run_count": len({row["run_accession"] for row in rows}),
        "verified_total_bytes": sum(row["bytes"] for row in rows),
        "files": rows,
        "implementation_binding": dict(implementation_evidence),
        "qualified_study_contribution": 0,
        "training_allowed": False,
        "next_phase_authorized": False,
        "claim_boundary": (
            "This is a transport-integrity result only. It does not establish raw-count "
            "reconstruction, xTail replay, A1 qualification, training authorization, or "
            "a scientific result."
        ),
    }


def _expected_committed_member_names(
    entries: Sequence[ManifestEntry],
) -> set[str]:
    names = {
        ACQUISITION_BINDING_FILENAME,
        FINAL_STATUS_FILENAME,
        INTEGRITY_MANIFEST_FILENAME,
        SHA256SUMS_FILENAME,
    }
    for entry in entries:
        names.add(entry.filename)
        names.add(entry.transfer_binding_filename)
    return names


def _publish_terminal_bundle(
    directory_fd: int,
    output_directory: Path,
    entries: Sequence[ManifestEntry],
    results: Sequence[FileResult],
    implementation_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    if len(results) != EXPECTED_FILE_COUNT or not all(result.success for result in results):
        raise PublicationError("terminal commit requires all 48 verified FASTQ files")
    if sum(result.bytes_verified for result in results) != EXPECTED_TOTAL_BYTES:
        raise PublicationError("terminal commit aggregate byte count is not exact")
    if len({result.entry.run_accession for result in results}) != EXPECTED_RUN_COUNT:
        raise PublicationError("terminal commit run count is not exact")
    reverified_results = _reverify_final_results(directory_fd, results)
    integrity_manifest = _integrity_manifest_document(
        entries,
        reverified_results,
        implementation_evidence=implementation_evidence,
    )
    integrity_payload = _json_bytes(integrity_manifest)
    integrity_sha = _ensure_exact_derived_member(
        directory_fd, INTEGRITY_MANIFEST_FILENAME, integrity_payload
    )
    final_status = _final_status_document(
        output_directory=output_directory,
        implementation_evidence=implementation_evidence,
    )
    final_status_payload = _json_bytes(final_status)
    final_status_sha = _ensure_exact_derived_member(
        directory_fd, FINAL_STATUS_FILENAME, final_status_payload
    )

    member_sha: dict[str, str] = {
        ACQUISITION_BINDING_FILENAME: _metadata_sha_at(
            directory_fd, ACQUISITION_BINDING_FILENAME
        ),
        FINAL_STATUS_FILENAME: final_status_sha,
        INTEGRITY_MANIFEST_FILENAME: integrity_sha,
    }
    result_by_name = {
        result.entry.filename: result for result in reverified_results
    }
    for entry in entries:
        result = result_by_name[entry.filename]
        assert result.local_sha256 is not None
        member_sha[entry.filename] = result.local_sha256
        member_sha[entry.transfer_binding_filename] = _metadata_sha_at(
            directory_fd, entry.transfer_binding_filename
        )
    sums_payload = "".join(
        f"{member_sha[name]}  {name}\n" for name in sorted(member_sha)
    ).encode("utf-8")
    member_sha[SHA256SUMS_FILENAME] = _ensure_exact_derived_member(
        directory_fd, SHA256SUMS_FILENAME, sums_payload
    )

    if set(member_sha) != _expected_committed_member_names(entries):
        raise PublicationError("derived committed member set is not manifest-closed")
    allowed_before_commit = set(member_sha) | {
        LOCK_FILENAME,
        PROGRESS_STATUS_FILENAME,
    }
    names_before_commit = _list_output_names(directory_fd)
    if names_before_commit != allowed_before_commit:
        unexpected = sorted(names_before_commit ^ allowed_before_commit)
        raise PublicationError(
            "output directory contains unexpected or missing members before commit: "
            + ", ".join(unexpected[:8])
        )
    _assert_reverified_identities_unchanged(directory_fd, reverified_results)

    marker = {
        "schema_version": SCHEMA_VERSION,
        "record_type": "GSE200304_FASTQ_ACQUISITION_PUBLICATION_COMMIT",
        "dataset_accession": DATASET_ACCESSION,
        "bioproject_accession": BIOPROJECT_ACCESSION,
        "generated_at": _utc_now(),
        "publication_status": "FASTQ_ACQUISITION_COMMITTED",
        "output_directory": str(output_directory),
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "source_terminal_marker_sha256": EXPECTED_SOURCE_COMMIT_SHA256,
        "implementation_binding": dict(implementation_evidence),
        "member_set": sorted(member_sha),
        "member_sha256": {name: member_sha[name] for name in sorted(member_sha)},
        "verified_file_count": EXPECTED_FILE_COUNT,
        "verified_run_count": EXPECTED_RUN_COUNT,
        "verified_total_bytes": EXPECTED_TOTAL_BYTES,
        "repository_md5_verified_count": EXPECTED_FILE_COUNT,
        "local_sha256_recorded_count": EXPECTED_FILE_COUNT,
        "qualified_study_contribution": 0,
        "training_allowed": False,
        "next_phase_authorized": False,
        "claim_boundary": (
            "This terminal commit establishes acquisition and transport integrity for "
            "the exact 48-file ENA manifest only. It does not establish count "
            "reconstruction, xTail replay, A1 qualification, training authorization, "
            "model performance, or a scientific conclusion."
        ),
    }
    # This must remain the final file-creation operation in the success path.
    _create_exclusive_at(directory_fd, TERMINAL_MARKER_FILENAME, _json_bytes(marker))
    return marker


def _validate_committed_output(
    directory_fd: int,
    output_directory: Path,
    entries: Sequence[ManifestEntry],
    *,
    protocol_sha256: str,
    implementation_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    marker = _read_json_at(
        directory_fd, TERMINAL_MARKER_FILENAME, label="output terminal marker"
    )
    required = {
        "schema_version": SCHEMA_VERSION,
        "record_type": "GSE200304_FASTQ_ACQUISITION_PUBLICATION_COMMIT",
        "dataset_accession": DATASET_ACCESSION,
        "bioproject_accession": BIOPROJECT_ACCESSION,
        "publication_status": "FASTQ_ACQUISITION_COMMITTED",
        "output_directory": str(output_directory),
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "source_terminal_marker_sha256": EXPECTED_SOURCE_COMMIT_SHA256,
        "implementation_binding": dict(implementation_evidence),
        "verified_file_count": EXPECTED_FILE_COUNT,
        "verified_run_count": EXPECTED_RUN_COUNT,
        "verified_total_bytes": EXPECTED_TOTAL_BYTES,
        "repository_md5_verified_count": EXPECTED_FILE_COUNT,
        "local_sha256_recorded_count": EXPECTED_FILE_COUNT,
        "qualified_study_contribution": 0,
        "training_allowed": False,
        "next_phase_authorized": False,
    }
    for key, expected in required.items():
        _require_equal(marker.get(key), expected, label=f"output terminal marker {key}")
    member_set = marker.get("member_set")
    member_sha = marker.get("member_sha256")
    expected_claim_boundary = (
        "This terminal commit establishes acquisition and transport integrity for "
        "the exact 48-file ENA manifest only. It does not establish count "
        "reconstruction, xTail replay, A1 qualification, training authorization, "
        "model performance, or a scientific conclusion."
    )
    if marker.get("claim_boundary") != expected_claim_boundary:
        raise PublicationError("output terminal marker claim boundary is not exact")
    generated_at = marker.get("generated_at")
    if not isinstance(generated_at, str):
        raise PublicationError("output terminal marker timestamp is missing")
    try:
        parsed_generated_at = datetime.fromisoformat(generated_at)
    except ValueError as exc:
        raise PublicationError("output terminal marker timestamp is invalid") from exc
    if parsed_generated_at.tzinfo is None:
        raise PublicationError("output terminal marker timestamp lacks a timezone")
    expected_marker_keys = set(required) | {
        "generated_at",
        "member_set",
        "member_sha256",
        "claim_boundary",
    }
    if set(marker) != expected_marker_keys:
        raise PublicationError("output terminal marker field set is not exact")
    if not isinstance(member_set, list) or not isinstance(member_sha, Mapping):
        raise PublicationError("output terminal marker has no closed member map")
    expected_member_set = _expected_committed_member_names(entries)
    if (
        member_set != sorted(expected_member_set)
        or set(member_sha) != expected_member_set
    ):
        raise PublicationError("output terminal marker member set is not exact")
    names = _list_output_names(directory_fd)
    expected_directory_names = expected_member_set | {
        LOCK_FILENAME,
        PROGRESS_STATUS_FILENAME,
        TERMINAL_MARKER_FILENAME,
    }
    if names != expected_directory_names:
        raise PublicationError("committed output directory member set is not exact")

    expected_binding = _output_binding(
        protocol_sha256=protocol_sha256,
        output_directory=output_directory,
        implementation_evidence=implementation_evidence,
    )
    actual_binding = _read_json_at(
        directory_fd,
        ACQUISITION_BINDING_FILENAME,
        label="committed acquisition binding",
        maximum_bytes=256 * 1024,
    )
    if actual_binding != expected_binding:
        raise PublicationError("committed acquisition binding is not exact")
    progress = _read_progress_status(
        directory_fd, output_directory=output_directory
    )
    final_attempt = progress["attempts"][-1]
    if (
        progress.get("current_status")
        != "VERIFIED_ALL_FILES_READY_FOR_TERMINAL_COMMIT"
        or final_attempt.get("verified_files") != EXPECTED_FILE_COUNT
        or final_attempt.get("verified_bytes") != EXPECTED_TOTAL_BYTES
    ):
        raise PublicationError("committed operational progress is not precommit-complete")

    reverified_results: list[FileResult] = []
    actual_member_sha: dict[str, str] = {
        ACQUISITION_BINDING_FILENAME: _metadata_sha_at(
            directory_fd, ACQUISITION_BINDING_FILENAME
        )
    }
    for entry in entries:
        binding = _read_json_at(
            directory_fd,
            entry.transfer_binding_filename,
            label=f"committed transfer binding {entry.filename}",
            maximum_bytes=64 * 1024,
        )
        if binding != _transfer_binding(entry):
            raise PublicationError("committed transfer binding is not manifest-exact")
        transfer_sha = _metadata_sha_at(
            directory_fd, entry.transfer_binding_filename
        )
        sha256, identity = _verify_entry_file(
            directory_fd, entry, name=entry.filename
        )
        actual_member_sha[entry.transfer_binding_filename] = transfer_sha
        actual_member_sha[entry.filename] = sha256
        reverified_results.append(
            FileResult(
                entry=entry,
                success=True,
                resumed_from_bytes=entry.expected_bytes,
                bytes_verified=entry.expected_bytes,
                repository_md5=entry.repository_md5,
                local_sha256=sha256,
                identity=identity,
                reused_completed_file=True,
            )
        )

    expected_integrity = _integrity_manifest_document(
        entries,
        reverified_results,
        implementation_evidence=implementation_evidence,
    )
    integrity_payload, integrity_sha, _ = _read_regular_at(
        directory_fd,
        INTEGRITY_MANIFEST_FILENAME,
        label="committed integrity manifest",
        maximum_bytes=4 * 1024 * 1024,
    )
    if integrity_payload != _json_bytes(expected_integrity):
        raise PublicationError("committed integrity manifest is not semantically exact")
    actual_member_sha[INTEGRITY_MANIFEST_FILENAME] = integrity_sha

    expected_final_status = _final_status_document(
        output_directory=output_directory,
        implementation_evidence=implementation_evidence,
    )
    final_status_payload, final_status_sha, _ = _read_regular_at(
        directory_fd,
        FINAL_STATUS_FILENAME,
        label="committed final status",
        maximum_bytes=1024 * 1024,
    )
    if final_status_payload != _json_bytes(expected_final_status):
        raise PublicationError("committed final status is not semantically exact")
    actual_member_sha[FINAL_STATUS_FILENAME] = final_status_sha

    expected_sums_payload = "".join(
        f"{actual_member_sha[name]}  {name}\n"
        for name in sorted(actual_member_sha)
    ).encode("ascii")
    sums_payload, sums_sha, _ = _read_regular_at(
        directory_fd,
        SHA256SUMS_FILENAME,
        label="committed SHA256SUMS",
        maximum_bytes=1024 * 1024,
    )
    if sums_payload != expected_sums_payload:
        raise PublicationError("committed SHA256SUMS is not semantically exact")
    if _parse_sha256sums(sums_payload, label="committed SHA256SUMS") != actual_member_sha:
        raise PublicationError("committed SHA256SUMS member map is not exact")
    actual_member_sha[SHA256SUMS_FILENAME] = sums_sha
    if actual_member_sha != member_sha:
        raise PublicationError("terminal marker member hashes do not match derived truth")
    return marker


def _validate_runtime_parameters(
    *, workers: int, timeout_seconds: int, chunk_bytes: int
) -> None:
    if not isinstance(workers, int) or isinstance(workers, bool) or not (1 <= workers <= MAXIMUM_WORKERS):
        raise ProtocolError("workers must be an integer between 1 and 2")
    if not isinstance(timeout_seconds, int) or isinstance(timeout_seconds, bool) or not (1 <= timeout_seconds <= 3600):
        raise ProtocolError("timeout_seconds must be an integer from 1 to 3600")
    if not isinstance(chunk_bytes, int) or isinstance(chunk_bytes, bool) or not (1 <= chunk_bytes <= MAX_CHUNK_BYTES):
        raise ProtocolError("chunk_bytes is outside the bounded streaming range")


def _open_output_directory(output_directory: Path, *, resume: bool) -> tuple[int, bool]:
    _assert_no_forbidden_path(output_directory, label="output directory")
    if output_directory.parent != EXPECTED_OUTPUT_ROOT:
        raise ScopeViolation("output must be one direct child of the frozen GSE200304 root")
    if not OUTPUT_BASENAME_RE.fullmatch(output_directory.name):
        raise ScopeViolation("output subdirectory basename is outside the frozen pattern")
    root_fd = _open_directory_chain(EXPECTED_OUTPUT_ROOT, label="output base directory")
    created = False
    try:
        flags = (
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        if resume:
            try:
                output_fd = os.open(output_directory.name, flags, dir_fd=root_fd)
            except FileNotFoundError as exc:
                raise ResumeError("resume requires an existing acquisition directory") from exc
        else:
            try:
                os.mkdir(output_directory.name, 0o750, dir_fd=root_fd)
            except FileExistsError as exc:
                raise PublicationError(
                    "fresh acquisition requires a unique nonexistent output directory"
                ) from exc
            created = True
            os.fsync(root_fd)
            output_fd = os.open(output_directory.name, flags, dir_fd=root_fd)
        output_stat = os.fstat(output_fd)
        if output_stat.st_uid != os.geteuid() or output_stat.st_mode & 0o022:
            os.close(output_fd)
            raise ScopeViolation("output directory must be owned by the caller and not group/world writable")
        return output_fd, created
    finally:
        os.close(root_fd)


def _lock_output(directory_fd: int) -> int:
    flags = (
        os.O_RDWR
        | os.O_CREAT
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(LOCK_FILENAME, flags, 0o600, dir_fd=directory_fd)
    value = os.fstat(descriptor)
    if not stat.S_ISREG(value.st_mode) or value.st_nlink != 1:
        os.close(descriptor)
        raise PublicationError("acquisition lock is not a unique regular file")
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        os.close(descriptor)
        raise PublicationError("another acquisition process holds the output lock") from exc
    return descriptor


def execute(
    protocol_path: Path,
    output_directory: Path,
    *,
    resume: bool = False,
    workers: int = DEFAULT_WORKERS,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    transport_factory: Callable[[], DownloadTransport] = UrllibHttpsTransport,
    implementation_verifier: Callable[
        [Path, str, Mapping[str, Any]], Mapping[str, Any]
    ] = verify_implementation_binding,
    capacity_probe: Callable[[int], int] = _default_capacity_probe,
) -> dict[str, Any]:
    """Execute one acquisition attempt and return an aggregate status object."""

    _validate_runtime_parameters(
        workers=workers, timeout_seconds=timeout_seconds, chunk_bytes=chunk_bytes
    )
    protocol, protocol_sha256 = load_protocol(protocol_path)
    implementation_evidence = dict(
        implementation_verifier(protocol_path, protocol_sha256, protocol)
    )
    _validate_implementation_evidence(
        implementation_evidence,
        protocol_sha256=protocol_sha256,
        protocol=protocol,
    )
    entries, _ = load_manifest_authority()
    output_fd, _ = _open_output_directory(output_directory, resume=resume)
    lock_fd: int | None = None
    try:
        lock_fd = _lock_output(output_fd)
        if _member_exists(output_fd, TERMINAL_MARKER_FILENAME):
            if not resume:
                raise PublicationError("terminal output already exists and cannot be overwritten")
            marker = _validate_committed_output(
                output_fd,
                output_directory,
                entries,
                protocol_sha256=protocol_sha256,
                implementation_evidence=implementation_evidence,
            )
            return {
                "success": True,
                "publication_status": "ALREADY_COMMITTED_VERIFIED",
                "output_directory": str(output_directory),
                "terminal_marker": marker,
                "qualified_study_contribution": 0,
                "training_allowed": False,
                "next_phase_authorized": False,
            }

        expected_binding = _output_binding(
            protocol_sha256=protocol_sha256,
            output_directory=output_directory,
            implementation_evidence=implementation_evidence,
        )
        if resume:
            if not _member_exists(output_fd, ACQUISITION_BINDING_FILENAME):
                raise ResumeError("resume directory lacks the immutable acquisition binding")
            actual_binding = _read_json_at(
                output_fd,
                ACQUISITION_BINDING_FILENAME,
                label="acquisition binding",
            )
            if actual_binding != expected_binding:
                raise ResumeError("acquisition binding does not exactly match this execution")
        else:
            _create_exclusive_at(
                output_fd,
                ACQUISITION_BINDING_FILENAME,
                _json_bytes(expected_binding),
            )

        existing_status: Mapping[str, Any] | None = None
        if _member_exists(output_fd, PROGRESS_STATUS_FILENAME):
            existing_status = _read_progress_status(
                output_fd, output_directory=output_directory
            )
        attempt_number = (
            len(existing_status.get("attempts", [])) + 1
            if isinstance(existing_status, Mapping)
            and isinstance(existing_status.get("attempts"), list)
            else 1
        )
        started_at = _utc_now()
        preflight_attempt = {
            "attempt_number": attempt_number,
            "started_at": started_at,
            "finished_at": None,
            "resume_requested": resume,
            "workers": workers,
            "status": "PREFLIGHT_NOT_COMMITTED",
            "attempted_files": 0,
            "verified_files": 0,
            "results": [],
        }
        try:
            capacity_evidence = _capacity_preflight(
                output_fd,
                entries,
                resume=resume,
                capacity_probe=capacity_probe,
            )
        except Exception as exc:
            error_code, error_message = _normalized_error(exc)
            evidence = exc.evidence if isinstance(exc, CapacityError) else None
            failed_preflight_attempt = {
                **preflight_attempt,
                "finished_at": _utc_now(),
                "status": "PREFLIGHT_FAILED_NOT_COMMITTED",
                "capacity_gate": evidence,
                "error_code": error_code,
                "error_message": error_message,
            }
            failed_preflight_status = _status_document(
                existing_status,
                output_directory=output_directory,
                attempt=failed_preflight_attempt,
            )
            _write_progress_status(output_fd, failed_preflight_status)
            return {
                "success": False,
                "publication_status": "PREFLIGHT_FAILED_NOT_COMMITTED",
                "output_directory": str(output_directory),
                "capacity_gate": evidence,
                "error_code": error_code,
                "error_message": error_message,
                "terminal_commit_present": False,
                "qualified_study_contribution": 0,
                "training_allowed": False,
                "next_phase_authorized": False,
            }

        running_attempt = {
            **preflight_attempt,
            "status": "RUNNING_NOT_COMMITTED",
            "capacity_gate": capacity_evidence,
        }
        running_status = _status_document(
            existing_status,
            output_directory=output_directory,
            attempt=running_attempt,
        )
        _write_progress_status(output_fd, running_status)

        try:
            results = _run_downloads(
                output_fd,
                entries,
                workers=workers,
                resume=resume,
                transport_factory=transport_factory,
                timeout_seconds=timeout_seconds,
                chunk_bytes=chunk_bytes,
            )
        except BaseException as exc:
            error_code, error_message = _normalized_error(exc)
            interrupted_attempt = {
                **running_attempt,
                "finished_at": _utc_now(),
                "status": "INTERRUPTED_NOT_COMMITTED",
                "error_code": error_code,
                "error_message": error_message,
            }
            interrupted_status = _status_document(
                existing_status,
                output_directory=output_directory,
                attempt=interrupted_attempt,
            )
            _write_progress_status(output_fd, interrupted_status)
            raise

        successful = [result for result in results if result.success]
        failed = [result for result in results if not result.success]
        if failed or len(successful) != EXPECTED_FILE_COUNT:
            failed_attempt = {
                **running_attempt,
                "finished_at": _utc_now(),
                "status": "FAILED_NOT_COMMITTED",
                "attempted_files": len(results),
                "verified_files": len(successful),
                "verified_bytes": sum(result.bytes_verified for result in successful),
                "results": [_result_record(result) for result in results],
            }
            failure_status = _status_document(
                existing_status,
                output_directory=output_directory,
                attempt=failed_attempt,
            )
            _write_progress_status(output_fd, failure_status)
            return {
                "success": False,
                "publication_status": "FAILED_NOT_COMMITTED",
                "output_directory": str(output_directory),
                "attempted_files": len(results),
                "verified_files": len(successful),
                "failure_count": len(failed),
                "failures": [_result_record(result) for result in failed],
                "terminal_commit_present": False,
                "qualified_study_contribution": 0,
                "training_allowed": False,
                "next_phase_authorized": False,
            }

        successful_attempt = {
            **running_attempt,
            "finished_at": _utc_now(),
            "status": "VERIFIED_ALL_FILES_READY_FOR_TERMINAL_COMMIT",
            "attempted_files": len(results),
            "verified_files": len(successful),
            "verified_bytes": sum(result.bytes_verified for result in successful),
            "results": [_result_record(result) for result in results],
        }
        success_status = _status_document(
            existing_status,
            output_directory=output_directory,
            attempt=successful_attempt,
        )
        _write_progress_status(output_fd, success_status)
        try:
            marker = _publish_terminal_bundle(
                output_fd,
                output_directory,
                entries,
                results,
                implementation_evidence,
            )
        except Exception as exc:
            # A marker write can fail after the directory entry becomes visible
            # (for example, on a late fsync error).  If a complete marker is
            # present, validate it instead of mutating any member it commits.
            if _member_exists(output_fd, TERMINAL_MARKER_FILENAME):
                try:
                    recovered_marker = _validate_committed_output(
                        output_fd,
                        output_directory,
                        entries,
                        protocol_sha256=protocol_sha256,
                        implementation_evidence=implementation_evidence,
                    )
                except Exception as validation_exc:
                    error_code, error_message = _normalized_error(exc)
                    validation_code, validation_message = _normalized_error(
                        validation_exc
                    )
                    return {
                        "success": False,
                        "publication_status": (
                            "PUBLICATION_FAILED_TERMINAL_MARKER_PRESENT_UNVERIFIED"
                        ),
                        "output_directory": str(output_directory),
                        "error_code": error_code,
                        "error_message": error_message,
                        "terminal_validation_error_code": validation_code,
                        "terminal_validation_error_message": validation_message,
                        "terminal_commit_present": True,
                        "qualified_study_contribution": 0,
                        "training_allowed": False,
                        "next_phase_authorized": False,
                    }
                return {
                    "success": True,
                    "publication_status": "FASTQ_ACQUISITION_COMMITTED_RECOVERED",
                    "output_directory": str(output_directory),
                    "terminal_marker": recovered_marker,
                    "qualified_study_contribution": 0,
                    "training_allowed": False,
                    "next_phase_authorized": False,
                }

            error_code, error_message = _normalized_error(exc)
            publication_failed_attempt = {
                **running_attempt,
                "finished_at": _utc_now(),
                "status": "PUBLICATION_FAILED_NOT_COMMITTED",
                "attempted_files": len(results),
                "verified_files": len(successful),
                "verified_bytes": sum(
                    result.bytes_verified for result in successful
                ),
                "error_code": error_code,
                "error_message": error_message,
                "results": [_result_record(result) for result in results],
            }
            publication_failure_status = _status_document(
                existing_status,
                output_directory=output_directory,
                attempt=publication_failed_attempt,
            )
            _write_progress_status(output_fd, publication_failure_status)
            return {
                "success": False,
                "publication_status": "PUBLICATION_FAILED_NOT_COMMITTED",
                "output_directory": str(output_directory),
                "error_code": error_code,
                "error_message": error_message,
                "terminal_commit_present": False,
                "qualified_study_contribution": 0,
                "training_allowed": False,
                "next_phase_authorized": False,
            }
        return {
            "success": True,
            "publication_status": "FASTQ_ACQUISITION_COMMITTED",
            "output_directory": str(output_directory),
            "verified_files": EXPECTED_FILE_COUNT,
            "verified_runs": EXPECTED_RUN_COUNT,
            "verified_bytes": EXPECTED_TOTAL_BYTES,
            "terminal_marker_filename": TERMINAL_MARKER_FILENAME,
            "terminal_marker_sha256": _sha256_bytes(_json_bytes(marker)),
            "qualified_study_contribution": 0,
            "training_allowed": False,
            "next_phase_authorized": False,
        }
    finally:
        if lock_fd is not None:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            finally:
                os.close(lock_fd)
        os.close(output_fd)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--chunk-bytes", type=int, default=DEFAULT_CHUNK_BYTES)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    try:
        result = execute(
            arguments.protocol,
            arguments.output_directory,
            resume=arguments.resume,
            workers=arguments.workers,
            timeout_seconds=arguments.timeout_seconds,
            chunk_bytes=arguments.chunk_bytes,
        )
    except Exception as exc:
        error_code, error_message = _normalized_error(exc)
        result = {
            "success": False,
            "publication_status": "FAIL_CLOSED",
            "error_code": error_code,
            "error_message": error_message,
            "qualified_study_contribution": 0,
            "training_allowed": False,
            "next_phase_authorized": False,
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("success") is True else 2


if __name__ == "__main__":
    sys.exit(main())
