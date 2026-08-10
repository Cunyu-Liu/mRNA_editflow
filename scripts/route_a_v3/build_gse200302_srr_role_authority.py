#!/usr/bin/env python3
"""Build and validate the exact official-metadata SRR/SRX role authority.

The production path is deliberately inert while the protocol implementation
binding is UNKNOWN_NOT_ASSERTED.  A BOUND run permits only fixed-argv,
read-only Git verification followed by reads of the two hash-pinned official
metadata files.  It never reads FASTQ bodies and never performs replay,
qualification, canonicalization, or training.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import os
import re
import secrets
import stat
import subprocess
import zlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


SCHEMA_VERSION = "route_a_v3_gse200302_srr_role_authority.v1"
PROTOCOL_ID = "ROUTE_A_V3_GSE200302_SRR_ROLE_AUTHORITY_V1"
CANONICALIZATION = "CANONICAL_SORTED_UTF8_V1"
PROTOCOL_CORE_SHA256 = "d407504d42c390b32aaa0eff953c168b1e9cc4991afcd8530870144c78a1d526"
TARGET_SERIES = "GSE200302"
TARGET_BIOPROJECT = "PRJNA824033"

PROTOCOL_REPO_PATH = "configs/route_a_v3_gse200302_srr_role_authority.json"
SCRIPT_REPO_PATH = "scripts/route_a_v3/build_gse200302_srr_role_authority.py"
TEST_REPO_PATH = "tests/route_a_v3/test_gse200302_srr_role_authority.py"
GIT_BINARY = "/usr/bin/git"
BINDING_MODE = "TWO_COMMIT_CONFIG_ONLY_NON_SELF_REFERENTIAL_V1"
BINDING_ACTIVATION_RULE = (
    "Commit this UNKNOWN protocol with the implementation script and test, then create exactly "
    "one separate config-only binding commit that changes only implementation_binding.status, "
    "implementation_commit, implementation_script_sha256, and implementation_test_sha256. Runtime "
    "fails before official-source or output access until that binding exists."
)

RUNINFO_BYTES = 12_042
RUNINFO_SHA256 = "34bcedafebc41ee9ccd79483f331b62f2443df31d12691abc0a961a7201848f4"
SOFT_BYTES = 4_699
SOFT_SHA256 = "6df39a3406fe1bdf5a37345fee5605510ca1086fbce54d5aeeb934b562bb7d2e"
MAPPING_BYTES = 1_200
MAPPING_SHA256 = "f69fa9af134b421439a2a90c09c75cb300e2e833de143d829bafe4ef7a1d094d"
EXPERIMENT_JOIN_BYTES = 1_509
EXPERIMENT_JOIN_SHA256 = "6684f3d1fde3666ac4bf07ff0aa29bd9b47240b5d6708fd8483aaa1d88a64ae4"

MAPPING_COLUMNS = (
    "run_accession",
    "geo_sample_accession",
    "biosample_accession",
    "measurement_family",
    "replicate",
)
EXPERIMENT_JOIN_COLUMNS = (
    "run_accession",
    "geo_sample_accession",
    "biosample_accession",
    "experiment_accession",
    "measurement_family",
    "replicate",
)
MEASUREMENT_FAMILIES = ("High_Poly", "Low_Poly", "pDNA", "Total_RNA")
REPLICATES = (1, 2, 3, 4, 5, 6)
FORBIDDEN_ALIASES = ("80S_RNA",)
RUNINFO_REQUIRED_COLUMNS = (
    "Run",
    "Experiment",
    "LibraryName",
    "BioProject",
    "BioSample",
    "SampleName",
)

MAPPING_FILENAME = "GSE200302_SRR_ROLE_AUTHORITY.tsv"
EXPERIMENT_JOIN_FILENAME = "GSE200302_SRR_SRX_ROLE_JOIN_AUTHORITY.tsv"
PROVENANCE_FILENAME = "ROLE_AUTHORITY.json"
CHECKSUMS_FILENAME = "SHA256SUMS"
MARKER_FILENAME = "PUBLICATION_COMMIT.json"

ROLE_AUTHORITY_STATUS = "EXACT_OFFICIAL_SRR_ROLE_AUTHORITY_CLOSED"
RAW_REPLAY_STATUS = "CONFLICT_WITH_CURRENT_80S_EXPECTATION"
PARTIAL_NOT_COMMITTED = "PARTIAL_NOT_COMMITTED"
COMMITTED_NOT_ACCEPTED = "COMMITTED_NOT_ACCEPTED"
PUBLISHED_WITH_DURABILITY_WARNING = "PUBLISHED_WITH_DURABILITY_WARNING"
COMMITTED_AND_ACCEPTED = "COMMITTED_AND_ACCEPTED"
CLAIM_BOUNDARY = (
    "This artifact closes only the exact 24-run official SRR/GSM/BioSample role authority "
    "for GSE200302 and PRJNA824033. It does not establish raw replay, A1 qualification, "
    "canonical materialization, training authorization, model performance, or any scientific result."
)

FORBIDDEN_PATH_TOKENS = (
    "gse246381",
    "access_log",
    "sealed_external",
    "sealed",
    "restricted",
)
SAFE_BASENAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
SHA256_RE = re.compile(r"[0-9a-f]{64}")
COMMIT_RE = re.compile(r"[0-9a-f]{40}")
RUN_RE = re.compile(r"SRR[0-9]+")
EXPERIMENT_RE = re.compile(r"SRX[0-9]+")
GSM_RE = re.compile(r"GSM[0-9]+")
BIOSAMPLE_RE = re.compile(r"SAMN[0-9]+")
SERIES_RE = re.compile(r"GSE[0-9]+")
SOFT_RECORD_RE = re.compile(r"\^([A-Z]+)\s*=\s*(\S+)\s*")
SOFT_FIELD_RE = re.compile(r"!([A-Za-z0-9_]+)\s*=\s*(.*)")
ROLE_TITLE_PATTERN = (
    r"^(High_Poly|Low_Poly|pDNA|Total_RNA)_([1-6])_(S[0-9]+)$"
)
ROLE_TITLE_RE = re.compile(ROLE_TITLE_PATTERN)
MAX_CONTRACT_BYTES = 256 * 1024
MAX_SOFT_DECOMPRESSED_BYTES = 4 * 1024 * 1024
MAX_PUBLICATION_MEMBER_BYTES = 4 * 1024 * 1024


class AuthorityError(RuntimeError):
    code = "ROLE_AUTHORITY_FAILED"
    publication_state = "FAIL_CLOSED"


class ScopeViolation(AuthorityError):
    code = "SCOPE_VIOLATION"


class ContractError(AuthorityError):
    code = "CONTRACT_INVALID"


class ImplementationBindingUnknown(ContractError):
    code = "IMPLEMENTATION_BINDING_UNKNOWN"


class ImplementationBindingError(ContractError):
    code = "IMPLEMENTATION_BINDING_INVALID"


class SourceError(AuthorityError):
    code = "SOURCE_FINGERPRINT_INVALID"


class MetadataError(AuthorityError):
    code = "OFFICIAL_METADATA_INVALID"


class JoinError(AuthorityError):
    code = "OFFICIAL_METADATA_JOIN_INVALID"


class PublicationError(AuthorityError):
    code = "PUBLICATION_FAILED"


class OutputExistsError(PublicationError):
    code = "OUTPUT_ALREADY_EXISTS"


class PartialPublicationError(PublicationError):
    code = PARTIAL_NOT_COMMITTED
    publication_state = PARTIAL_NOT_COMMITTED


class CommittedNotAcceptedError(PublicationError):
    code = COMMITTED_NOT_ACCEPTED
    publication_state = COMMITTED_NOT_ACCEPTED


@dataclass(frozen=True)
class ExpectedBindingRoot:
    """Caller-owned trust root required by every authority consumer."""

    implementation_commit: str
    binding_commit: str
    implementation_script_sha256: str
    implementation_test_sha256: str
    protocol_full_sha256: str
    protocol_core_sha256: str

    def as_dict(self) -> dict[str, str]:
        return {
            "implementation_commit": self.implementation_commit,
            "binding_commit": self.binding_commit,
            "implementation_script_sha256": self.implementation_script_sha256,
            "implementation_test_sha256": self.implementation_test_sha256,
            "protocol_full_sha256": self.protocol_full_sha256,
            "protocol_core_sha256": self.protocol_core_sha256,
        }

    @classmethod
    def from_verified_evidence(cls, evidence: Mapping[str, Any]) -> "ExpectedBindingRoot":
        verified = _validate_implementation_evidence(dict(evidence))
        return cls(
            implementation_commit=verified["implementation_commit"],
            binding_commit=verified["binding_commit"],
            implementation_script_sha256=verified["implementation_script_sha256"],
            implementation_test_sha256=verified["implementation_test_sha256"],
            protocol_full_sha256=verified["protocol_full_sha256"],
            protocol_core_sha256=verified["protocol_core_sha256"],
        )


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("ascii")


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")


def _strict_equal(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected):
        return False
    if type(expected) is dict:
        return set(actual) == set(expected) and all(
            _strict_equal(actual[key], expected[key]) for key in expected
        )
    if type(expected) is list:
        return len(actual) == len(expected) and all(
            _strict_equal(left, right) for left, right in zip(actual, expected)
        )
    return actual == expected


def _role_order_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    role_runs: tuple[tuple[str, Sequence[int]], ...] = (
        ("High_Poly", tuple(range(60, 54, -1))),
        ("Low_Poly", tuple(range(54, 48, -1))),
        ("pDNA", tuple(range(48, 42, -1))),
        ("Total_RNA", (42, 70, 69, 68, 67, 66)),
    )
    source_index = 0
    for family, run_suffixes in role_runs:
        for replicate, suffix in enumerate(run_suffixes, 1):
            rows.append(
                {
                    "run_accession": f"SRR186567{suffix:02d}",
                    "geo_sample_accession": f"GSM{6_030_613 + source_index}",
                    "biosample_accession": f"SAMN{27_381_278 - source_index}",
                    "measurement_family": family,
                    "replicate": replicate,
                }
            )
            source_index += 1
    return rows


def _expected_experiment(run_accession: str) -> str:
    suffix = int(run_accession[-2:])
    if 42 <= suffix <= 60:
        return f"SRX{14_759_751 + (60 - suffix)}"
    if 66 <= suffix <= 70:
        return f"SRX{14_759_741 + (70 - suffix)}"
    raise AssertionError("run outside compiled SRX authority")


EXPECTED_ROWS = tuple(
    sorted(_role_order_rows(), key=lambda row: int(row["run_accession"][3:]))
)
EXPECTED_JOIN_ROWS = tuple(
    {
        "run_accession": row["run_accession"],
        "geo_sample_accession": row["geo_sample_accession"],
        "biosample_accession": row["biosample_accession"],
        "experiment_accession": _expected_experiment(row["run_accession"]),
        "measurement_family": row["measurement_family"],
        "replicate": row["replicate"],
    }
    for row in EXPECTED_ROWS
)

EXPECTED_PROTOCOL_TRUST = {
    "canonicalization": CANONICALIZATION,
    "core_projection_excluded_top_level_keys": ["implementation_binding"],
    "compiled_core_projection_required": True,
}
EXPECTED_SCOPE = {
    "target_series_accession": TARGET_SERIES,
    "bioproject_accession": TARGET_BIOPROJECT,
    "authority_level": "OFFICIAL_METADATA_ROLE_AUTHORITY_ONLY",
    "ordinary_public_metadata_only": True,
}
EXPECTED_SOURCES = {
    "runinfo": {
        "source_kind": "NCBI_SRA_RUNINFO",
        "format": "CSV",
        "expected_bytes": RUNINFO_BYTES,
        "expected_sha256": RUNINFO_SHA256,
        "required_columns": list(RUNINFO_REQUIRED_COLUMNS),
    },
    "geo_soft": {
        "source_kind": "NCBI_GEO_FAMILY_SOFT",
        "format": "GEO_SOFT_UTF8_OR_GZIP_BY_MAGIC",
        "expected_bytes": SOFT_BYTES,
        "expected_sha256": SOFT_SHA256,
        "series_accession": TARGET_SERIES,
        "sra_relation_label": "SRA",
        "biosample_relation_label": "BioSample",
        "sample_title_role_contract": {
            "fullmatch_regex": ROLE_TITLE_PATTERN,
            "technical_suffix_group": 3,
            "technical_suffix_authority_parse_only": True,
            "technical_suffix_creates_study": False,
            "technical_suffix_creates_context": False,
            "technical_suffix_creates_endpoint": False,
            "technical_suffix_creates_label": False,
            "technical_suffix_output_allowed": False,
        },
    },
}
EXPECTED_JOIN_CONTRACT = {
    "expected_row_count": 24,
    "require_exact_24_of_24_join": True,
    "require_runinfo_experiment_equals_soft_sra_relation": True,
    "unique_accession_classes": [
        "run_accession",
        "geo_sample_accession",
        "biosample_accession",
        "experiment_accession",
    ],
}
EXPECTED_PUBLICATION = {
    "mapping_filename": MAPPING_FILENAME,
    "experiment_join_filename": EXPERIMENT_JOIN_FILENAME,
    "provenance_filename": PROVENANCE_FILENAME,
    "checksums_filename": CHECKSUMS_FILENAME,
    "terminal_marker_filename": MARKER_FILENAME,
    "atomic_terminal_marker_written_last": True,
    "no_overwrite": True,
}
EXPECTED_GATE_CONTRACT = {
    "role_authority_status": ROLE_AUTHORITY_STATUS,
    "raw_replay_role_grid_status": RAW_REPLAY_STATUS,
    "qualified": False,
    "training_authorized": False,
    "ordinary_study_contribution": 0,
    "a1_study_contribution": 0,
    "true_a2_study_contribution": 0,
    "canonical_record_count": 0,
    "next_phase_authorized": False,
}
EXPECTED_EXECUTION_POLICY = {
    "network_access_allowed": False,
    "subprocess_allowed": False,
    "fixed_argv_read_only_git_subprocess_allowed": True,
    "git_binary": GIT_BINARY,
    "fastq_body_read_allowed": False,
    "sequence_output_allowed": False,
    "barcode_output_allowed": False,
    "training_label_output_allowed": False,
    "qualification_allowed": False,
    "canonical_materialization_allowed": False,
    "training_allowed": False,
    "next_phase_unlock_allowed": False,
}
EXPECTED_VALIDATION = {
    "runinfo_row_count": 24,
    "geo_soft_sample_count": 24,
    "exact_join_count": 24,
    "run_accession_unique": True,
    "geo_sample_accession_unique": True,
    "biosample_accession_unique": True,
    "experiment_accession_unique": True,
    "runinfo_experiment_matches_compiled_authority": True,
    "soft_sra_relation_matches_compiled_authority": True,
    "runinfo_experiment_equals_soft_sra_relation": True,
    "runinfo_biosample_equals_soft_biosample_relation": True,
    "role_grid_exact": True,
    "forbidden_80s_alias_count": 0,
}


def _expected_mapping_contract() -> dict[str, Any]:
    return {
        "columns": list(MAPPING_COLUMNS),
        "sort": "RUN_ACCESSION_NUMERIC_ASC",
        "line_ending": "LF",
        "allowed_measurement_families": list(MEASUREMENT_FAMILIES),
        "replicates": list(REPLICATES),
        "forbidden_family_aliases": list(FORBIDDEN_ALIASES),
        "expected_bytes": MAPPING_BYTES,
        "expected_sha256": MAPPING_SHA256,
        "expected_rows": [dict(row) for row in EXPECTED_ROWS],
    }


def _expected_experiment_join_contract() -> dict[str, Any]:
    return {
        "columns": list(EXPERIMENT_JOIN_COLUMNS),
        "sort": "RUN_ACCESSION_NUMERIC_ASC",
        "line_ending": "LF",
        "expected_bytes": EXPERIMENT_JOIN_BYTES,
        "expected_sha256": EXPERIMENT_JOIN_SHA256,
        "expected_rows": [dict(row) for row in EXPECTED_JOIN_ROWS],
    }


def _expected_core_contract() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "protocol_trust": EXPECTED_PROTOCOL_TRUST,
        "scope": EXPECTED_SCOPE,
        "sources": EXPECTED_SOURCES,
        "join_contract": EXPECTED_JOIN_CONTRACT,
        "mapping_contract": _expected_mapping_contract(),
        "experiment_join_contract": _expected_experiment_join_contract(),
        "publication": EXPECTED_PUBLICATION,
        "gate_contract": EXPECTED_GATE_CONTRACT,
        "execution_policy": EXPECTED_EXECUTION_POLICY,
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _expected_unknown_binding() -> dict[str, Any]:
    return {
        "status": "UNKNOWN_NOT_ASSERTED",
        "binding_mode": BINDING_MODE,
        "implementation_commit": "UNKNOWN_NOT_ASSERTED",
        "implementation_script_repo_path": SCRIPT_REPO_PATH,
        "implementation_script_sha256": "UNKNOWN_NOT_ASSERTED",
        "implementation_test_repo_path": TEST_REPO_PATH,
        "implementation_test_sha256": "UNKNOWN_NOT_ASSERTED",
        "protocol_repo_path": PROTOCOL_REPO_PATH,
        "activation_rule": BINDING_ACTIVATION_RULE,
    }


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _parse_json(
    payload: bytes,
    *,
    label: str,
    error_type: type[AuthorityError] = ContractError,
) -> dict[str, Any]:
    try:
        text = payload.decode("utf-8")
        value = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, ValueError, RecursionError) as exc:
        raise error_type(f"{label} is not strict duplicate-free finite UTF-8 JSON") from exc
    if type(value) is not dict:
        raise error_type(f"{label} root must be an object")
    return value


def _canonical_protocol_projection(protocol: Mapping[str, Any]) -> dict[str, Any]:
    if type(protocol) is not dict:
        raise ContractError("protocol root must be an exact object")
    projection = copy.deepcopy(protocol)
    if "implementation_binding" not in projection:
        raise ContractError("protocol implementation binding is absent")
    del projection["implementation_binding"]
    return projection


def _protocol_core_sha256(protocol: Mapping[str, Any]) -> str:
    try:
        return _sha256(_canonical_json_bytes(_canonical_protocol_projection(protocol)))
    except (TypeError, ValueError, RecursionError) as exc:
        raise ContractError("protocol core projection is not canonicalizable") from exc


def _normalized_two_commit_protocol(protocol: Mapping[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(protocol)
    binding = normalized.get("implementation_binding")
    if type(binding) is not dict:
        raise ContractError("protocol binding cannot be normalized")
    for key in (
        "status",
        "implementation_commit",
        "implementation_script_sha256",
        "implementation_test_sha256",
    ):
        binding[key] = "TWO_COMMIT_DYNAMIC_FIELD"
    return normalized


def _validate_binding(binding: Any) -> str:
    expected_keys = set(_expected_unknown_binding())
    if type(binding) is not dict or set(binding) != expected_keys:
        raise ContractError("implementation binding keys are not exact")
    fixed = _expected_unknown_binding()
    for key in (
        "binding_mode",
        "implementation_script_repo_path",
        "implementation_test_repo_path",
        "protocol_repo_path",
        "activation_rule",
    ):
        if not _strict_equal(binding[key], fixed[key]):
            raise ContractError(f"implementation binding {key} drifted")
    status = binding.get("status")
    if status == "UNKNOWN_NOT_ASSERTED":
        if not _strict_equal(binding, fixed):
            raise ContractError("UNKNOWN implementation binding is not the exact closed state")
        return status
    if status != "BOUND":
        raise ContractError("implementation binding status is outside the closed enum")
    if COMMIT_RE.fullmatch(binding.get("implementation_commit", "")) is None:
        raise ContractError("BOUND implementation commit is not a full object ID")
    for key in ("implementation_script_sha256", "implementation_test_sha256"):
        if SHA256_RE.fullmatch(binding.get(key, "")) is None:
            raise ContractError(f"BOUND {key} is not a SHA256")
    return status


def _validate_contract(value: Any) -> str:
    expected_core = _expected_core_contract()
    if type(value) is not dict or set(value) != set(expected_core) | {"implementation_binding"}:
        raise ContractError("protocol top-level keys are not exact")
    if not _strict_equal(_canonical_protocol_projection(value), expected_core):
        raise ContractError("protocol differs from the closed role-authority core")
    observed_core = _protocol_core_sha256(value)
    if observed_core != PROTOCOL_CORE_SHA256:
        raise ContractError("protocol core differs from the compiled projection")
    return _validate_binding(value["implementation_binding"])


def _reject_forbidden_path(path: Path | str, *, label: str) -> None:
    if not isinstance(path, (str, os.PathLike)):
        raise ScopeViolation(f"{label} is not path-like")
    text = os.fspath(path).casefold()
    if any(token in text for token in FORBIDDEN_PATH_TOKENS):
        raise ScopeViolation(f"{label} rejected before read by the frozen scope guard")


def _absolute_without_resolving(path: Path | str, *, label: str) -> Path:
    _reject_forbidden_path(path, label=label)
    value = Path(path).expanduser()
    if ".." in value.parts:
        raise ScopeViolation(f"{label} contains parent traversal")
    absolute = value if value.is_absolute() else Path.cwd() / value
    _reject_forbidden_path(absolute, label=label)
    return absolute


def _safe_basename(value: str, *, label: str) -> str:
    if type(value) is not str or SAFE_BASENAME_RE.fullmatch(value) is None or Path(value).name != value:
        raise ScopeViolation(f"{label} is not an allowed basename")
    return value


def _close_once(descriptor: int) -> BaseException | None:
    try:
        os.close(descriptor)
    except BaseException as exc:
        return exc
    return None


def _require_platform_capabilities() -> None:
    for name in ("O_NOFOLLOW", "O_DIRECTORY", "O_CLOEXEC", "O_NONBLOCK"):
        if not hasattr(os, name):
            raise ScopeViolation(f"platform lacks required {name} capability")
    for function in (os.open, os.mkdir, os.stat, os.link, os.unlink):
        if function not in os.supports_dir_fd:
            raise ScopeViolation("platform lacks required dir_fd capability")
    if os.stat not in os.supports_follow_symlinks or os.link not in os.supports_follow_symlinks:
        raise ScopeViolation("platform lacks required no-follow metadata capability")


def _open_directory_chain(path: Path, *, label: str) -> int:
    _require_platform_capabilities()
    if not path.is_absolute() or ".." in path.parts:
        raise ScopeViolation(f"{label} is not an absolute closed path")
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptor: int | None = None
    try:
        descriptor = os.open("/", flags)
        for part in path.parts[1:]:
            if part in {"", ".", ".."}:
                raise ScopeViolation(f"{label} contains an unsafe path component")
            next_descriptor = os.open(part, flags, dir_fd=descriptor)
            close_error = _close_once(descriptor)
            descriptor = next_descriptor
            if close_error is not None:
                raise ScopeViolation(f"{label} directory transition close failed") from close_error
        opened = os.fstat(descriptor)
        if not stat.S_ISDIR(opened.st_mode):
            raise ScopeViolation(f"{label} is not a directory")
        result = descriptor
        descriptor = None
        return result
    except AuthorityError:
        raise
    except BaseException as exc:
        raise ScopeViolation(f"{label} could not be opened without following links") from exc
    finally:
        if descriptor is not None:
            _close_once(descriptor)


def _read_regular_snapshot(
    path: Path | str,
    *,
    label: str,
    maximum_bytes: int,
    expected_bytes: int | None = None,
    expected_sha256: str | None = None,
    _after_capture: Callable[[Path], None] | None = None,
) -> tuple[bytes, dict[str, Any]]:
    absolute = _absolute_without_resolving(path, label=label)
    parent_fd: int | None = None
    descriptor: int | None = None
    primary_error: BaseException | None = None
    result: tuple[bytes, dict[str, Any]] | None = None
    close_errors: list[BaseException] = []
    try:
        parent_fd = _open_directory_chain(absolute.parent, label=f"{label} parent")
        descriptor = os.open(
            absolute.name,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK,
            dir_fd=parent_fd,
        )
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or opened.st_size > maximum_bytes:
            raise SourceError(f"{label} is not a bounded regular file")
        if expected_bytes is not None and opened.st_size != expected_bytes:
            raise SourceError(f"{label} byte count drifted from the frozen authority")
        def read_bounded_pass() -> tuple[bytes, int, str]:
            blocks: list[bytes] = []
            digest = hashlib.sha256()
            total = 0
            while True:
                try:
                    block = os.read(
                        descriptor,
                        min(1 << 20, maximum_bytes + 1 - total),
                    )
                except OSError as exc:
                    raise SourceError(
                        f"{label} same-descriptor read failed"
                    ) from exc
                if not block:
                    break
                blocks.append(block)
                digest.update(block)
                total += len(block)
                if total > maximum_bytes:
                    raise SourceError(f"{label} exceeds the closed byte bound")
            return b"".join(blocks), total, digest.hexdigest()

        first_payload, total, observed_sha256 = read_bounded_pass()
        try:
            first_end_offset = os.lseek(descriptor, 0, os.SEEK_CUR)
        except OSError as exc:
            raise SourceError(f"{label} same-descriptor offset check failed") from exc
        if first_end_offset != total:
            raise SourceError(f"{label} changed during same-descriptor capture")
        if _after_capture is not None:
            _after_capture(absolute)
        try:
            rewind_offset = os.lseek(descriptor, 0, os.SEEK_SET)
        except OSError as exc:
            raise SourceError(f"{label} same-descriptor rewind failed") from exc
        if rewind_offset != 0:
            raise SourceError(f"{label} changed during same-descriptor capture")
        second_payload, second_total, second_sha256 = read_bounded_pass()
        try:
            second_end_offset = os.lseek(descriptor, 0, os.SEEK_CUR)
        except OSError as exc:
            raise SourceError(f"{label} same-descriptor offset check failed") from exc
        if (
            second_end_offset != second_total
            or second_total != total
            or second_sha256 != observed_sha256
            or second_payload != first_payload
        ):
            raise SourceError(f"{label} changed during same-descriptor capture")
        final = os.fstat(descriptor)
        entry = os.stat(absolute.name, dir_fd=parent_fd, follow_symlinks=False)
        opened_identity = (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        )
        final_identity = (
            final.st_dev,
            final.st_ino,
            final.st_size,
            final.st_mtime_ns,
            final.st_ctime_ns,
        )
        if (
            opened_identity != final_identity
            or total != opened.st_size
            or (entry.st_dev, entry.st_ino) != (opened.st_dev, opened.st_ino)
        ):
            raise SourceError(f"{label} changed during same-descriptor capture")
        if expected_sha256 is not None and observed_sha256 != expected_sha256:
            raise SourceError(f"{label} hash drifted from the frozen authority")
        result = (
            first_payload,
            {"bytes": total, "sha256": observed_sha256},
        )
    except BaseException as exc:
        primary_error = exc
    finally:
        if descriptor is not None:
            close_error = _close_once(descriptor)
            if close_error is not None:
                close_errors.append(close_error)
        if parent_fd is not None:
            close_error = _close_once(parent_fd)
            if close_error is not None:
                close_errors.append(close_error)
    if primary_error is not None:
        if isinstance(primary_error, AuthorityError):
            raise primary_error
        raise SourceError(f"{label} could not be captured safely") from primary_error
    if close_errors:
        raise SourceError(f"{label} descriptor close failed") from close_errors[0]
    if result is None:
        raise SourceError(f"{label} capture produced no result")
    return result


def load_contract(path: Path | str) -> tuple[dict[str, Any], dict[str, Any], bytes]:
    payload, observed = _read_regular_snapshot(
        path,
        label="role-authority protocol",
        maximum_bytes=MAX_CONTRACT_BYTES,
    )
    protocol = _parse_json(payload, label="role-authority protocol")
    binding_status = _validate_contract(protocol)
    provenance = {
        "full_file_bytes": observed["bytes"],
        "full_file_sha256": observed["sha256"],
        "core_projection_sha256": _protocol_core_sha256(protocol),
        "canonicalization": CANONICALIZATION,
        "binding_status": binding_status,
    }
    return protocol, provenance, payload


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
            stderr=subprocess.PIPE,
            check=False,
            timeout=30,
            env=environment,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ImplementationBindingError("fixed read-only Git verification failed") from exc


def _git_capture(repo_root: Path, arguments: Sequence[str], *, label: str) -> bytes:
    completed = _run_read_only_git(repo_root, arguments)
    if completed.returncode != 0:
        raise ImplementationBindingError(f"Git {label} query failed closed")
    return completed.stdout


def _git_is_ancestor(repo_root: Path, ancestor: str, descendant: str) -> bool:
    completed = _run_read_only_git(
        repo_root,
        ("merge-base", "--is-ancestor", ancestor, descendant),
    )
    if completed.returncode not in {0, 1}:
        raise ImplementationBindingError("Git ancestry query failed closed")
    return completed.returncode == 0


def _validate_implementation_evidence(
    evidence: Any,
    *,
    protocol_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    expected_keys = {
        "status",
        "binding_mode",
        "implementation_commit",
        "binding_commit",
        "implementation_script_sha256",
        "implementation_test_sha256",
        "protocol_full_sha256",
        "protocol_core_sha256",
        "worktree_and_index_clean",
        "implementation_is_direct_parent",
        "post_implementation_changed_paths",
        "implementation_protocol_binding_status",
        "head_protocol_binding_status",
    }
    if type(evidence) is not dict or set(evidence) != expected_keys:
        raise ImplementationBindingError("implementation evidence keys are not exact")
    if evidence["status"] != "BOUND" or evidence["binding_mode"] != BINDING_MODE:
        raise ImplementationBindingError("implementation evidence status or mode is invalid")
    for key in ("implementation_commit", "binding_commit"):
        if COMMIT_RE.fullmatch(evidence.get(key, "")) is None:
            raise ImplementationBindingError(f"implementation evidence {key} is invalid")
    if evidence["implementation_commit"] == evidence["binding_commit"]:
        raise ImplementationBindingError("binding commit must strictly descend from implementation")
    for key in (
        "implementation_script_sha256",
        "implementation_test_sha256",
        "protocol_full_sha256",
        "protocol_core_sha256",
    ):
        if SHA256_RE.fullmatch(evidence.get(key, "")) is None:
            raise ImplementationBindingError(f"implementation evidence {key} is invalid")
    if evidence["protocol_core_sha256"] != PROTOCOL_CORE_SHA256:
        raise ImplementationBindingError("implementation evidence core hash is not compiled")
    if evidence["worktree_and_index_clean"] is not True:
        raise ImplementationBindingError("implementation evidence worktree is not clean")
    if evidence["implementation_is_direct_parent"] is not True:
        raise ImplementationBindingError("implementation evidence is not a direct two-commit binding")
    if evidence["post_implementation_changed_paths"] != [PROTOCOL_REPO_PATH]:
        raise ImplementationBindingError("implementation evidence change set is not config-only")
    if evidence["implementation_protocol_binding_status"] != "UNKNOWN_NOT_ASSERTED":
        raise ImplementationBindingError("implementation commit did not preserve UNKNOWN binding")
    if evidence["head_protocol_binding_status"] != "BOUND":
        raise ImplementationBindingError("binding commit did not establish BOUND status")
    if protocol_provenance is not None:
        if evidence["protocol_full_sha256"] != protocol_provenance.get("full_file_sha256"):
            raise ImplementationBindingError("evidence full protocol hash is inconsistent")
        if evidence["protocol_core_sha256"] != protocol_provenance.get("core_projection_sha256"):
            raise ImplementationBindingError("evidence protocol core hash is inconsistent")
    return evidence


def _validate_expected_binding_root(
    value: ExpectedBindingRoot | Mapping[str, Any] | None,
) -> ExpectedBindingRoot:
    expected_keys = {
        "implementation_commit",
        "binding_commit",
        "implementation_script_sha256",
        "implementation_test_sha256",
        "protocol_full_sha256",
        "protocol_core_sha256",
    }
    if isinstance(value, ExpectedBindingRoot):
        fields: Any = value.as_dict()
    else:
        fields = value
    if type(fields) is not dict or set(fields) != expected_keys:
        raise ImplementationBindingError(
            "external ExpectedBindingRoot is required with an exact field set"
        )
    if (
        type(fields.get("implementation_commit")) is not str
        or COMMIT_RE.fullmatch(fields["implementation_commit"]) is None
    ):
        raise ImplementationBindingError("external implementation commit is not exact")
    if (
        type(fields.get("binding_commit")) is not str
        or COMMIT_RE.fullmatch(fields["binding_commit"]) is None
    ):
        raise ImplementationBindingError("external binding commit is not exact")
    if fields["implementation_commit"] == fields["binding_commit"]:
        raise ImplementationBindingError("external binding root commits are not strictly ordered")
    for key in (
        "implementation_script_sha256",
        "implementation_test_sha256",
        "protocol_full_sha256",
        "protocol_core_sha256",
    ):
        if type(fields.get(key)) is not str or SHA256_RE.fullmatch(fields[key]) is None:
            raise ImplementationBindingError(f"external binding root {key} is not exact")
    if fields["protocol_core_sha256"] != PROTOCOL_CORE_SHA256:
        raise ImplementationBindingError("external binding root core hash is not compiled")
    return ExpectedBindingRoot(**fields)


def _require_provenance_matches_external_root(
    provenance: Mapping[str, Any],
    expected_binding_root: ExpectedBindingRoot,
) -> None:
    evidence = provenance.get("implementation_binding")
    trust = provenance.get("protocol_trust")
    if type(evidence) is not dict or type(trust) is not dict:
        raise ImplementationBindingError("bundle binding evidence is absent")
    observed = {
        "implementation_commit": evidence.get("implementation_commit"),
        "binding_commit": evidence.get("binding_commit"),
        "implementation_script_sha256": evidence.get("implementation_script_sha256"),
        "implementation_test_sha256": evidence.get("implementation_test_sha256"),
        "protocol_full_sha256": trust.get("protocol_full_sha256"),
        "protocol_core_sha256": trust.get("protocol_core_sha256"),
    }
    if not _strict_equal(observed, expected_binding_root.as_dict()):
        raise ImplementationBindingError(
            "bundle binding evidence differs from the caller-supplied ExpectedBindingRoot"
        )


def verify_implementation_binding(
    *,
    protocol_path: Path,
    protocol: Mapping[str, Any],
    protocol_payload: bytes,
    protocol_provenance: Mapping[str, Any],
    running_script_path: Path | None = None,
) -> dict[str, Any]:
    binding = protocol.get("implementation_binding")
    if _validate_binding(binding) != "BOUND":
        raise ImplementationBindingUnknown(
            "implementation binding is UNKNOWN_NOT_ASSERTED; a separate config-only binding commit is required"
        )
    assert type(binding) is dict
    repo_root = protocol_path.parent.parent
    expected_protocol_path = repo_root / PROTOCOL_REPO_PATH
    expected_script_path = repo_root / SCRIPT_REPO_PATH
    expected_test_path = repo_root / TEST_REPO_PATH
    if protocol_path != expected_protocol_path:
        raise ImplementationBindingError("protocol is not at its frozen repository-relative path")
    script_path = Path(os.path.abspath(__file__)) if running_script_path is None else running_script_path
    if script_path != expected_script_path:
        raise ImplementationBindingError("executing script is not the repository-bound implementation")

    reported_root = _git_capture(
        repo_root,
        ("rev-parse", "--show-toplevel"),
        label="repository root",
    ).decode("utf-8", errors="strict").strip()
    if reported_root != str(repo_root):
        raise ImplementationBindingError("Git repository root differs from protocol root")
    head_commit = _git_capture(repo_root, ("rev-parse", "HEAD"), label="HEAD").decode(
        "ascii", errors="strict"
    ).strip()
    if COMMIT_RE.fullmatch(head_commit) is None:
        raise ImplementationBindingError("current HEAD is not a full commit ID")
    dirty = _git_capture(
        repo_root,
        ("status", "--porcelain=v1", "--untracked-files=all"),
        label="worktree status",
    )
    if dirty:
        raise ImplementationBindingError("implementation binding requires a clean worktree and index")

    implementation_commit = binding["implementation_commit"]
    if not _git_is_ancestor(repo_root, implementation_commit, head_commit):
        raise ImplementationBindingError("implementation commit is not an ancestor of current HEAD")
    commit_count = _git_capture(
        repo_root,
        ("rev-list", "--count", f"{implementation_commit}..{head_commit}"),
        label="binding commit count",
    ).decode("ascii", errors="strict").strip()
    if commit_count != "1":
        raise ImplementationBindingError("binding must be exactly one commit after implementation")
    parent_record = _git_capture(
        repo_root,
        ("rev-list", "--parents", "-n", "1", head_commit),
        label="binding parent set",
    ).decode("ascii", errors="strict").strip().split()
    if parent_record != [head_commit, implementation_commit]:
        raise ImplementationBindingError("binding commit must have exactly the implementation parent")
    first_parent = _git_capture(
        repo_root,
        ("rev-parse", f"{head_commit}^"),
        label="binding parent",
    ).decode("ascii", errors="strict").strip()
    if first_parent != implementation_commit:
        raise ImplementationBindingError("implementation commit is not the binding commit parent")
    changed_paths = _git_capture(
        repo_root,
        ("diff", "--name-only", implementation_commit, head_commit, "--"),
        label="post-implementation change set",
    ).decode("utf-8", errors="strict").splitlines()
    if changed_paths != [PROTOCOL_REPO_PATH]:
        raise ImplementationBindingError("post-implementation change set is not config-only")

    script_payload, script_observed = _read_regular_snapshot(
        expected_script_path,
        label="bound implementation script",
        maximum_bytes=4 * 1024 * 1024,
    )
    test_payload, test_observed = _read_regular_snapshot(
        expected_test_path,
        label="bound implementation test",
        maximum_bytes=4 * 1024 * 1024,
    )
    if script_observed["sha256"] != binding["implementation_script_sha256"]:
        raise ImplementationBindingError("working-tree script hash differs from binding")
    if test_observed["sha256"] != binding["implementation_test_sha256"]:
        raise ImplementationBindingError("working-tree test hash differs from binding")

    def git_blob(commit: str, path: str, label: str) -> bytes:
        return _git_capture(repo_root, ("show", f"{commit}:{path}"), label=label)

    implementation_script = git_blob(
        implementation_commit, SCRIPT_REPO_PATH, "implementation script blob"
    )
    implementation_test = git_blob(
        implementation_commit, TEST_REPO_PATH, "implementation test blob"
    )
    implementation_protocol = git_blob(
        implementation_commit, PROTOCOL_REPO_PATH, "implementation protocol blob"
    )
    head_script = git_blob(head_commit, SCRIPT_REPO_PATH, "HEAD script blob")
    head_test = git_blob(head_commit, TEST_REPO_PATH, "HEAD test blob")
    head_protocol = git_blob(head_commit, PROTOCOL_REPO_PATH, "HEAD protocol blob")
    if any(
        _sha256(blob) != expected_hash
        for blob, expected_hash in (
            (implementation_script, binding["implementation_script_sha256"]),
            (head_script, binding["implementation_script_sha256"]),
            (implementation_test, binding["implementation_test_sha256"]),
            (head_test, binding["implementation_test_sha256"]),
        )
    ):
        raise ImplementationBindingError("implementation or HEAD blob hash differs from binding")
    if script_payload != implementation_script or script_payload != head_script:
        raise ImplementationBindingError("running script bytes differ from bound Git blobs")
    if test_payload != implementation_test or test_payload != head_test:
        raise ImplementationBindingError("working-tree test bytes differ from bound Git blobs")
    if head_protocol != protocol_payload or _sha256(head_protocol) != protocol_provenance["full_file_sha256"]:
        raise ImplementationBindingError("working-tree protocol bytes differ from HEAD blob")

    implementation_protocol_value = _parse_json(
        implementation_protocol,
        label="implementation-commit protocol",
        error_type=ImplementationBindingError,
    )
    implementation_status = _validate_contract(implementation_protocol_value)
    if implementation_status != "UNKNOWN_NOT_ASSERTED":
        raise ImplementationBindingError("implementation commit protocol was not UNKNOWN_NOT_ASSERTED")
    if _protocol_core_sha256(implementation_protocol_value) != protocol_provenance["core_projection_sha256"]:
        raise ImplementationBindingError("binding commit changed the protocol core projection")
    if not _strict_equal(
        _normalized_two_commit_protocol(implementation_protocol_value),
        _normalized_two_commit_protocol(protocol),
    ):
        raise ImplementationBindingError("binding commit changed fields outside the four-field binding delta")

    evidence = {
        "status": "BOUND",
        "binding_mode": BINDING_MODE,
        "implementation_commit": implementation_commit,
        "binding_commit": head_commit,
        "implementation_script_sha256": binding["implementation_script_sha256"],
        "implementation_test_sha256": binding["implementation_test_sha256"],
        "protocol_full_sha256": protocol_provenance["full_file_sha256"],
        "protocol_core_sha256": protocol_provenance["core_projection_sha256"],
        "worktree_and_index_clean": True,
        "implementation_is_direct_parent": True,
        "post_implementation_changed_paths": [PROTOCOL_REPO_PATH],
        "implementation_protocol_binding_status": implementation_status,
        "head_protocol_binding_status": "BOUND",
    }
    return _validate_implementation_evidence(
        evidence,
        protocol_provenance=protocol_provenance,
    )


def _decode_runinfo(payload: bytes) -> str:
    try:
        value = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise MetadataError("RunInfo is not strict UTF-8") from exc
    if "\x00" in value:
        raise MetadataError("RunInfo contains a NUL byte")
    return value


def _parse_runinfo(payload: bytes) -> list[dict[str, str]]:
    reader = csv.DictReader(io.StringIO(_decode_runinfo(payload), newline=""))
    if reader.fieldnames is None or len(reader.fieldnames) != len(set(reader.fieldnames)):
        raise MetadataError("RunInfo header is missing or duplicated")
    if any(column not in reader.fieldnames for column in RUNINFO_REQUIRED_COLUMNS):
        raise MetadataError("RunInfo is missing a required authority column")
    expected_by_run = {row["run_accession"]: row for row in EXPECTED_JOIN_ROWS}
    rows: list[dict[str, str]] = []
    for source_row in reader:
        if None in source_row:
            raise MetadataError("RunInfo row width differs from its header")
        selected = {column: source_row[column].strip() for column in RUNINFO_REQUIRED_COLUMNS}
        if RUN_RE.fullmatch(selected["Run"]) is None:
            raise MetadataError("RunInfo contains an invalid run accession")
        if EXPERIMENT_RE.fullmatch(selected["Experiment"]) is None:
            raise MetadataError("RunInfo contains an invalid experiment accession")
        if GSM_RE.fullmatch(selected["LibraryName"]) is None:
            raise MetadataError("RunInfo LibraryName is not a GEO sample accession")
        if selected["LibraryName"] != selected["SampleName"]:
            raise MetadataError("RunInfo GEO sample columns disagree")
        if BIOSAMPLE_RE.fullmatch(selected["BioSample"]) is None:
            raise MetadataError("RunInfo contains an invalid BioSample accession")
        if selected["BioProject"] != TARGET_BIOPROJECT:
            raise MetadataError("RunInfo contains a row outside PRJNA824033")
        expected = expected_by_run.get(selected["Run"])
        if expected is None or selected["Experiment"] != expected["experiment_accession"]:
            raise MetadataError("RunInfo Experiment differs from the compiled SRX authority")
        rows.append(selected)
    if len(rows) != 24:
        raise MetadataError("RunInfo does not contain exactly 24 data rows")
    for column in ("Run", "Experiment", "LibraryName", "BioSample"):
        if len({row[column] for row in rows}) != 24:
            raise MetadataError(f"RunInfo {column} accessions are not unique")
    return rows


def _decode_soft(payload: bytes) -> str:
    decoded = payload
    if payload.startswith(b"\x1f\x8b"):
        try:
            inflater = zlib.decompressobj(16 + zlib.MAX_WBITS)
            decoded = inflater.decompress(payload, MAX_SOFT_DECOMPRESSED_BYTES + 1)
        except zlib.error as exc:
            raise MetadataError("GEO SOFT gzip stream is invalid") from exc
        if len(decoded) > MAX_SOFT_DECOMPRESSED_BYTES or inflater.unconsumed_tail:
            raise MetadataError("GEO SOFT gzip stream exceeds the decompressed byte bound")
        try:
            decoded += inflater.flush(MAX_SOFT_DECOMPRESSED_BYTES + 1 - len(decoded))
        except zlib.error as exc:
            raise MetadataError("GEO SOFT gzip stream is invalid") from exc
        if (
            not inflater.eof
            or inflater.unused_data
            or inflater.unconsumed_tail
            or len(decoded) > MAX_SOFT_DECOMPRESSED_BYTES
        ):
            raise MetadataError("GEO SOFT gzip stream is not one bounded complete member")
    elif len(decoded) > MAX_SOFT_DECOMPRESSED_BYTES:
        raise MetadataError("GEO SOFT exceeds the decompressed byte bound")
    try:
        text = decoded.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise MetadataError("GEO SOFT is not strict UTF-8") from exc
    if "\x00" in text:
        raise MetadataError("GEO SOFT contains a NUL byte")
    return text


def _relation_accession(value: str, *, label: str, pattern: re.Pattern[str]) -> str:
    prefix = f"{label}:"
    if not value.startswith(prefix):
        raise MetadataError(f"GEO SOFT {label} relation prefix is not exact")
    matches = pattern.findall(value[len(prefix) :])
    if len(matches) != 1:
        raise MetadataError(f"GEO SOFT {label} relation does not contain one accession")
    return matches[0]


def _normalize_title_role(title: str) -> tuple[str, int]:
    if "80s" in title.casefold():
        raise MetadataError("GEO SOFT uses the forbidden 80S_RNA role alias")
    match = ROLE_TITLE_RE.fullmatch(title.strip())
    if match is None:
        raise MetadataError("GEO SOFT sample title is outside the exact four-family grid")
    technical_suffix = match.group(3)
    if re.fullmatch(r"S[0-9]+", technical_suffix) is None:
        raise MetadataError("GEO SOFT sample title technical suffix is not exact")
    family_token = re.sub(r"[ _-]+", "_", match.group(1)).casefold()
    canonical = {
        "high_poly": "High_Poly",
        "low_poly": "Low_Poly",
        "pdna": "pDNA",
        "total_rna": "Total_RNA",
    }.get(family_token)
    if canonical is None or canonical in FORBIDDEN_ALIASES:
        raise MetadataError("GEO SOFT sample title cannot be canonicalized exactly")
    # The frozen technical suffix establishes parse shape only and is intentionally not returned.
    return canonical, int(match.group(2))


def _parse_soft(payload: bytes) -> dict[str, dict[str, Any]]:
    series_accessions: list[str] = []
    samples: dict[str, dict[str, Any]] = {}
    current_sample: str | None = None
    for line in _decode_soft(payload).splitlines():
        record_match = SOFT_RECORD_RE.fullmatch(line)
        if record_match is not None:
            record_type, accession = record_match.groups()
            current_sample = None
            if record_type == "SERIES":
                if SERIES_RE.fullmatch(accession) is None:
                    raise MetadataError("GEO SOFT contains an invalid series accession")
                series_accessions.append(accession)
            elif record_type == "SAMPLE":
                if GSM_RE.fullmatch(accession) is None or accession in samples:
                    raise MetadataError("GEO SOFT sample accessions are invalid or duplicated")
                current_sample = accession
                samples[accession] = {
                    "titles": [],
                    "experiments": [],
                    "biosamples": [],
                    "declared_geo_accessions": [],
                }
            continue
        if current_sample is None:
            continue
        field_match = SOFT_FIELD_RE.fullmatch(line)
        if field_match is None:
            continue
        field, value = field_match.groups()
        if field == "Sample_title":
            samples[current_sample]["titles"].append(value.strip())
        elif field == "Sample_geo_accession":
            samples[current_sample]["declared_geo_accessions"].append(value.strip())
        elif field == "Sample_relation":
            relation = value.strip()
            if relation.startswith("SRA:"):
                samples[current_sample]["experiments"].append(
                    _relation_accession(relation, label="SRA", pattern=EXPERIMENT_RE)
                )
            elif relation.startswith("BioSample:"):
                samples[current_sample]["biosamples"].append(
                    _relation_accession(relation, label="BioSample", pattern=BIOSAMPLE_RE)
                )
    if series_accessions != [TARGET_SERIES]:
        raise MetadataError("GEO SOFT series authority is not exactly GSE200302")
    expected_by_gsm = {row["geo_sample_accession"]: row for row in EXPECTED_JOIN_ROWS}
    if set(samples) != set(expected_by_gsm) or len(samples) != 24:
        raise MetadataError("GEO SOFT sample set is not the exact 24-GSM authority")
    normalized: dict[str, dict[str, Any]] = {}
    for gsm, sample in samples.items():
        declared = sample["declared_geo_accessions"]
        if declared and declared != [gsm]:
            raise MetadataError("GEO SOFT declared sample accession disagrees with its record")
        if (
            len(sample["titles"]) != 1
            or len(sample["experiments"]) != 1
            or len(sample["biosamples"]) != 1
        ):
            raise MetadataError("GEO SOFT sample authority fields are not exactly singular")
        if sample["experiments"][0] != expected_by_gsm[gsm]["experiment_accession"]:
            raise MetadataError("GEO SOFT SRA relation differs from the compiled SRX authority")
        family, replicate = _normalize_title_role(sample["titles"][0])
        normalized[gsm] = {
            "experiment_accession": sample["experiments"][0],
            "biosample_accession": sample["biosamples"][0],
            "measurement_family": family,
            "replicate": replicate,
        }
    for key in ("experiment_accession", "biosample_accession"):
        if len({sample[key] for sample in normalized.values()}) != 24:
            raise MetadataError(f"GEO SOFT {key} values are not unique")
    return normalized


def _serialize_rows(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output, delimiter="\t", lineterminator="\n")
    writer.writerow(columns)
    for row in rows:
        writer.writerow([row[column] for column in columns])
    try:
        return output.getvalue().encode("ascii")
    except UnicodeEncodeError as exc:
        raise JoinError("normalized authority table is not ASCII metadata") from exc


def _expected_mapping_payload() -> bytes:
    return _serialize_rows(EXPECTED_ROWS, MAPPING_COLUMNS)


def _expected_experiment_join_payload() -> bytes:
    return _serialize_rows(EXPECTED_JOIN_ROWS, EXPERIMENT_JOIN_COLUMNS)


def derive_role_authority(
    runinfo_payload: bytes,
    soft_payload: bytes,
) -> tuple[list[dict[str, Any]], bytes, bytes, dict[str, Any]]:
    runinfo_rows = _parse_runinfo(runinfo_payload)
    soft_samples = _parse_soft(soft_payload)
    expected_by_run = {row["run_accession"]: row for row in EXPECTED_JOIN_ROWS}
    if {row["Run"] for row in runinfo_rows} != set(expected_by_run):
        raise JoinError("RunInfo SRR set differs from the exact role authority")
    derived_join: list[dict[str, Any]] = []
    for runinfo in runinfo_rows:
        run = runinfo["Run"]
        gsm = runinfo["LibraryName"]
        expected = expected_by_run[run]
        if (
            gsm != expected["geo_sample_accession"]
            or runinfo["BioSample"] != expected["biosample_accession"]
            or runinfo["Experiment"] != expected["experiment_accession"]
        ):
            raise JoinError("RunInfo mapping differs from the compiled SRR/SRX authority")
        soft = soft_samples.get(gsm)
        if soft is None:
            raise JoinError("RunInfo GEO sample is absent from GEO SOFT")
        if soft["experiment_accession"] != expected["experiment_accession"]:
            raise JoinError("GEO SOFT experiment differs from compiled SRX authority")
        if runinfo["Experiment"] != soft["experiment_accession"]:
            raise JoinError("RunInfo Experiment differs from the GEO SOFT SRA relation")
        if runinfo["BioSample"] != soft["biosample_accession"]:
            raise JoinError("RunInfo BioSample differs from the GEO SOFT relation")
        if (soft["measurement_family"], soft["replicate"]) != (
            expected["measurement_family"],
            expected["replicate"],
        ):
            raise JoinError("GEO SOFT role is permuted relative to the exact SRR authority")
        derived_join.append(dict(expected))
    derived_join.sort(key=lambda row: int(row["run_accession"][3:]))
    if not _strict_equal(derived_join, [dict(row) for row in EXPECTED_JOIN_ROWS]):
        raise JoinError("derived rows are not exactly equal to the compiled SRR/SRX authority")
    role_rows = [
        {column: row[column] for column in MAPPING_COLUMNS}
        for row in derived_join
    ]
    if not _strict_equal(role_rows, [dict(row) for row in EXPECTED_ROWS]):
        raise JoinError("derived role rows are not exactly equal to the closed mapping")
    grid = {(row["measurement_family"], row["replicate"]) for row in role_rows}
    expected_grid = {(family, replicate) for family in MEASUREMENT_FAMILIES for replicate in REPLICATES}
    if grid != expected_grid or len(grid) != 24:
        raise JoinError("derived role grid is not exactly four families by six replicates")
    mapping_payload = _serialize_rows(role_rows, MAPPING_COLUMNS)
    join_payload = _serialize_rows(derived_join, EXPERIMENT_JOIN_COLUMNS)
    if len(mapping_payload) != MAPPING_BYTES or _sha256(mapping_payload) != MAPPING_SHA256:
        raise JoinError("normalized role mapping bytes differ from the frozen artifact")
    if len(join_payload) != EXPERIMENT_JOIN_BYTES or _sha256(join_payload) != EXPERIMENT_JOIN_SHA256:
        raise JoinError("normalized SRX join bytes differ from the compiled artifact")
    return role_rows, mapping_payload, join_payload, dict(EXPECTED_VALIDATION)


def _provenance_factory(
    *,
    protocol_provenance: Mapping[str, Any],
    implementation_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": "GSE200302_SRR_SRX_ROLE_AUTHORITY",
        "status": ROLE_AUTHORITY_STATUS,
        "target_series_accession": TARGET_SERIES,
        "bioproject_accession": TARGET_BIOPROJECT,
        "protocol_trust": {
            "canonicalization": CANONICALIZATION,
            "protocol_full_bytes": protocol_provenance["full_file_bytes"],
            "protocol_full_sha256": protocol_provenance["full_file_sha256"],
            "protocol_core_sha256": protocol_provenance["core_projection_sha256"],
        },
        "implementation_binding": dict(implementation_evidence),
        "source_provenance": {
            "runinfo": {
                "source_kind": "NCBI_SRA_RUNINFO",
                "bytes": RUNINFO_BYTES,
                "sha256": RUNINFO_SHA256,
            },
            "geo_soft": {
                "source_kind": "NCBI_GEO_FAMILY_SOFT",
                "bytes": SOFT_BYTES,
                "sha256": SOFT_SHA256,
            },
        },
        "validation": dict(EXPECTED_VALIDATION),
        "mapping_artifact": {
            "filename": MAPPING_FILENAME,
            "columns": list(MAPPING_COLUMNS),
            "row_count": 24,
            "sort": "RUN_ACCESSION_NUMERIC_ASC",
            "line_ending": "LF",
            "bytes": MAPPING_BYTES,
            "sha256": MAPPING_SHA256,
        },
        "experiment_join_artifact": {
            "filename": EXPERIMENT_JOIN_FILENAME,
            "columns": list(EXPERIMENT_JOIN_COLUMNS),
            "row_count": 24,
            "sort": "RUN_ACCESSION_NUMERIC_ASC",
            "line_ending": "LF",
            "bytes": EXPERIMENT_JOIN_BYTES,
            "sha256": EXPERIMENT_JOIN_SHA256,
        },
        "raw_replay_role_grid_status": RAW_REPLAY_STATUS,
        "qualified": False,
        "training_authorized": False,
        "ordinary_study_contribution": 0,
        "a1_study_contribution": 0,
        "true_a2_study_contribution": 0,
        "canonical_record_count": 0,
        "next_phase_authorized": False,
        "metadata_only": True,
        "fastq_body_included": False,
        "sequence_included": False,
        "barcode_included": False,
        "training_label_included": False,
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _validate_provenance_document(value: Any) -> dict[str, Any]:
    if type(value) is not dict:
        raise PublicationError("provenance root is not an exact object")
    trust = value.get("protocol_trust")
    evidence = value.get("implementation_binding")
    if type(trust) is not dict or set(trust) != {
        "canonicalization",
        "protocol_full_bytes",
        "protocol_full_sha256",
        "protocol_core_sha256",
    }:
        raise PublicationError("provenance protocol trust is not exact")
    if trust.get("canonicalization") != CANONICALIZATION:
        raise PublicationError("provenance canonicalization is invalid")
    if type(trust.get("protocol_full_bytes")) is not int or trust["protocol_full_bytes"] <= 0:
        raise PublicationError("provenance protocol byte count is invalid")
    if SHA256_RE.fullmatch(trust.get("protocol_full_sha256", "")) is None:
        raise PublicationError("provenance full protocol hash is invalid")
    if trust.get("protocol_core_sha256") != PROTOCOL_CORE_SHA256:
        raise PublicationError("provenance core protocol hash is invalid")
    validated_evidence = _validate_implementation_evidence(
        evidence,
        protocol_provenance={
            "full_file_sha256": trust["protocol_full_sha256"],
            "core_projection_sha256": trust["protocol_core_sha256"],
        },
    )
    expected = _provenance_factory(
        protocol_provenance={
            "full_file_bytes": trust["protocol_full_bytes"],
            "full_file_sha256": trust["protocol_full_sha256"],
            "core_projection_sha256": trust["protocol_core_sha256"],
        },
        implementation_evidence=validated_evidence,
    )
    if not _strict_equal(value, expected):
        raise PublicationError("provenance differs from the closed metadata-only factory")
    return value


def build_provenance_document(
    *,
    runinfo_provenance: Mapping[str, Any],
    soft_provenance: Mapping[str, Any],
    mapping_payload: bytes,
    experiment_join_payload: bytes,
    validation: Mapping[str, Any],
    protocol_provenance: Mapping[str, Any],
    implementation_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    if not _strict_equal(dict(runinfo_provenance), {"bytes": RUNINFO_BYTES, "sha256": RUNINFO_SHA256}):
        raise SourceError("RunInfo provenance is not the frozen source authority")
    if not _strict_equal(dict(soft_provenance), {"bytes": SOFT_BYTES, "sha256": SOFT_SHA256}):
        raise SourceError("GEO SOFT provenance is not the frozen source authority")
    if mapping_payload != _expected_mapping_payload():
        raise JoinError("mapping provenance does not bind the frozen artifact")
    if experiment_join_payload != _expected_experiment_join_payload():
        raise JoinError("experiment join provenance does not bind the compiled artifact")
    if not _strict_equal(dict(validation), EXPECTED_VALIDATION):
        raise JoinError("validation provenance is not the closed 24-of-24 result")
    if (
        protocol_provenance.get("core_projection_sha256") != PROTOCOL_CORE_SHA256
        or SHA256_RE.fullmatch(protocol_provenance.get("full_file_sha256", "")) is None
        or type(protocol_provenance.get("full_file_bytes")) is not int
    ):
        raise ContractError("protocol provenance is not exact")
    evidence = _validate_implementation_evidence(
        dict(implementation_evidence),
        protocol_provenance=protocol_provenance,
    )
    return _validate_provenance_document(
        _provenance_factory(
            protocol_provenance=protocol_provenance,
            implementation_evidence=evidence,
        )
    )


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise PublicationError("exclusive publication write made no progress")
        view = view[written:]


def _write_exclusive_at(directory_fd: int, name: str, payload: bytes) -> None:
    _safe_basename(name, label="publication member")
    descriptor: int | None = None
    primary_error: BaseException | None = None
    close_error: BaseException | None = None
    try:
        descriptor = os.open(
            name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
            dir_fd=directory_fd,
        )
        _write_all(descriptor, payload)
        os.fsync(descriptor)
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or opened.st_size != len(payload):
            raise PublicationError("published member identity or length is invalid")
    except BaseException as exc:
        primary_error = exc
    finally:
        if descriptor is not None:
            close_error = _close_once(descriptor)
    if primary_error is not None:
        if isinstance(primary_error, FileExistsError):
            raise OutputExistsError("publication member already exists") from primary_error
        if isinstance(primary_error, AuthorityError):
            raise primary_error
        raise PublicationError("exclusive publication member write failed") from primary_error
    if close_error is not None:
        raise PublicationError("publication member close failed") from close_error


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _read_member_snapshot(
    directory_fd: int,
    name: str,
    *,
    maximum_bytes: int,
) -> tuple[bytes, tuple[int, int, int, int, int]]:
    descriptor: int | None = None
    primary_error: BaseException | None = None
    close_error: BaseException | None = None
    result: tuple[bytes, tuple[int, int, int, int, int]] | None = None
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK,
            dir_fd=directory_fd,
        )
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or opened.st_size > maximum_bytes:
            raise PublicationError("publication member is not a bounded regular file")
        payload = bytearray()
        while True:
            block = os.read(descriptor, min(1 << 20, maximum_bytes + 1 - len(payload)))
            if not block:
                break
            payload.extend(block)
            if len(payload) > maximum_bytes:
                raise PublicationError("publication member exceeds the closed byte bound")
        final = os.fstat(descriptor)
        entry = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        opened_identity = _stat_identity(opened)
        final_identity = _stat_identity(final)
        if (
            opened_identity != final_identity
            or len(payload) != opened.st_size
            or (entry.st_dev, entry.st_ino) != (opened.st_dev, opened.st_ino)
        ):
            raise PublicationError("publication member changed during same-descriptor capture")
        result = (bytes(payload), opened_identity)
    except BaseException as exc:
        primary_error = exc
    finally:
        if descriptor is not None:
            close_error = _close_once(descriptor)
    if primary_error is not None:
        if isinstance(primary_error, AuthorityError):
            raise primary_error
        raise PublicationError("publication member capture failed") from primary_error
    if close_error is not None:
        raise PublicationError("publication member close failed") from close_error
    if result is None:
        raise PublicationError("publication member capture produced no snapshot")
    return result


def _target_binding(output: Path) -> dict[str, Any]:
    return {
        "absolute_output_directory_sha256": _sha256(str(output).encode("utf-8")),
        "output_directory_basename": output.name,
    }


def _bundle_digest(member_payloads: Mapping[str, bytes]) -> str:
    return _sha256(
        b"\x00".join(
            name.encode("ascii") + b"\x00" + member_payloads[name]
            for name in sorted(member_payloads)
        )
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_timestamp(value: Any) -> str:
    if type(value) is not str:
        raise PublicationError("marker timestamp is not a string")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise PublicationError("marker timestamp is invalid") from exc
    if parsed.tzinfo is None:
        raise PublicationError("marker timestamp is timezone-naive")
    return value


def _marker_factory(
    *,
    member_payloads: Mapping[str, bytes],
    provenance: Mapping[str, Any],
    output: Path,
    generated_at: str,
) -> dict[str, Any]:
    evidence = provenance["implementation_binding"]
    trust = provenance["protocol_trust"]
    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": "GSE200302_SRR_SRX_ROLE_AUTHORITY_PUBLICATION_COMMIT",
        "status": ROLE_AUTHORITY_STATUS,
        "generated_at": generated_at,
        "target_binding": _target_binding(output),
        "bundle_digest": _bundle_digest(member_payloads),
        "member_set": sorted(member_payloads),
        "member_sha256": {
            name: _sha256(member_payloads[name]) for name in sorted(member_payloads)
        },
        "protocol_full_sha256": trust["protocol_full_sha256"],
        "protocol_core_sha256": trust["protocol_core_sha256"],
        "implementation_commit": evidence["implementation_commit"],
        "binding_commit": evidence["binding_commit"],
        "implementation_script_sha256": evidence["implementation_script_sha256"],
        "implementation_test_sha256": evidence["implementation_test_sha256"],
        "terminal_marker_written_last": True,
        "terminal_marker_atomic_visibility": True,
        "no_overwrite": True,
        "metadata_only": True,
        "raw_replay_role_grid_status": RAW_REPLAY_STATUS,
        "qualified": False,
        "training_authorized": False,
        "ordinary_study_contribution": 0,
        "a1_study_contribution": 0,
        "true_a2_study_contribution": 0,
        "canonical_record_count": 0,
        "next_phase_authorized": False,
    }


def _validate_marker(
    value: Any,
    *,
    member_payloads: Mapping[str, bytes],
    provenance: Mapping[str, Any],
    output: Path,
) -> dict[str, Any]:
    if type(value) is not dict:
        raise PublicationError("marker root is not an exact object")
    timestamp = _validate_timestamp(value.get("generated_at"))
    expected = _marker_factory(
        member_payloads=member_payloads,
        provenance=provenance,
        output=output,
        generated_at=timestamp,
    )
    if not _strict_equal(value, expected):
        raise PublicationError("marker differs from recomputed publication truth")
    return value


def _expected_member_payloads(
    provenance_document: Mapping[str, Any],
    *,
    mapping_payload: bytes,
    experiment_join_payload: bytes,
) -> dict[str, bytes]:
    if mapping_payload != _expected_mapping_payload():
        raise PublicationError("publication role mapping is not the frozen artifact")
    if experiment_join_payload != _expected_experiment_join_payload():
        raise PublicationError("publication SRX join is not the compiled artifact")
    provenance = _validate_provenance_document(provenance_document)
    payloads = {
        MAPPING_FILENAME: mapping_payload,
        EXPERIMENT_JOIN_FILENAME: experiment_join_payload,
        PROVENANCE_FILENAME: _json_bytes(provenance),
    }
    payloads[CHECKSUMS_FILENAME] = "".join(
        f"{_sha256(payloads[name])}  {name}\n"
        for name in sorted(payloads)
    ).encode("ascii")
    return payloads


def _validate_open_bundle(
    *,
    parent_fd: int,
    output_fd: int,
    output: Path,
    opened_directory: os.stat_result,
    expected_binding_root: ExpectedBindingRoot,
    _after_member_snapshot: Callable[[str, int], None] | None = None,
) -> dict[str, Any]:
    member_order = (
        MAPPING_FILENAME,
        EXPERIMENT_JOIN_FILENAME,
        PROVENANCE_FILENAME,
        CHECKSUMS_FILENAME,
        MARKER_FILENAME,
    )
    expected_names = set(member_order)
    if set(os.listdir(output_fd)) != expected_names:
        raise PublicationError("published directory member set is not exact")
    current = os.stat(output.name, dir_fd=parent_fd, follow_symlinks=False)
    if (current.st_dev, current.st_ino) != (opened_directory.st_dev, opened_directory.st_ino):
        raise PublicationError("published target directory was displaced")
    payloads: dict[str, bytes] = {}
    identities: dict[str, tuple[int, int, int, int, int]] = {}
    for index, name in enumerate(member_order):
        payload, identity = _read_member_snapshot(
            output_fd,
            name,
            maximum_bytes=MAX_PUBLICATION_MEMBER_BYTES,
        )
        payloads[name] = payload
        identities[name] = identity
        if _after_member_snapshot is not None:
            _after_member_snapshot(name, index)
    mapping_payload = payloads[MAPPING_FILENAME]
    join_payload = payloads[EXPERIMENT_JOIN_FILENAME]
    if mapping_payload != _expected_mapping_payload():
        raise PublicationError("consumer role mapping bytes are not exact")
    if join_payload != _expected_experiment_join_payload():
        raise PublicationError("consumer SRX join bytes are not exact")
    provenance = _parse_json(
        payloads[PROVENANCE_FILENAME],
        label="published provenance",
        error_type=PublicationError,
    )
    _validate_provenance_document(provenance)
    _require_provenance_matches_external_root(provenance, expected_binding_root)
    non_checksum = {
        MAPPING_FILENAME: mapping_payload,
        EXPERIMENT_JOIN_FILENAME: join_payload,
        PROVENANCE_FILENAME: payloads[PROVENANCE_FILENAME],
    }
    expected_sums = "".join(
        f"{_sha256(non_checksum[name])}  {name}\n"
        for name in sorted(non_checksum)
    ).encode("ascii")
    if payloads[CHECKSUMS_FILENAME] != expected_sums:
        raise PublicationError("published SHA256SUMS differs from recomputed members")
    member_payloads = dict(non_checksum)
    member_payloads[CHECKSUMS_FILENAME] = expected_sums
    marker = _parse_json(
        payloads[MARKER_FILENAME],
        label="published terminal marker",
        error_type=PublicationError,
    )
    _validate_marker(
        marker,
        member_payloads=member_payloads,
        provenance=provenance,
        output=output,
    )
    for name in member_order:
        final_member = os.stat(name, dir_fd=output_fd, follow_symlinks=False)
        if _stat_identity(final_member) != identities[name]:
            raise PublicationError(
                "published member identity changed after its snapshot was consumed"
            )
    if set(os.listdir(output_fd)) != expected_names:
        raise PublicationError("published directory changed during consumer validation")
    final = os.stat(output.name, dir_fd=parent_fd, follow_symlinks=False)
    if (final.st_dev, final.st_ino) != (opened_directory.st_dev, opened_directory.st_ino):
        raise PublicationError("published target directory identity changed during validation")
    return {
        "status": ROLE_AUTHORITY_STATUS,
        "publication_state": COMMITTED_AND_ACCEPTED,
        "committed": True,
        "accepted": True,
        "bundle_digest": marker["bundle_digest"],
        "mapping_sha256": MAPPING_SHA256,
        "experiment_join_sha256": EXPERIMENT_JOIN_SHA256,
        "raw_replay_role_grid_status": RAW_REPLAY_STATUS,
        "qualified": False,
        "training_authorized": False,
        "ordinary_study_contribution": 0,
        "a1_study_contribution": 0,
        "true_a2_study_contribution": 0,
        "canonical_record_count": 0,
        "next_phase_authorized": False,
    }


def validate_published_authority(
    output_directory: Path | str,
    *,
    expected_binding_root: ExpectedBindingRoot | Mapping[str, Any] | None = None,
    _after_member_snapshot: Callable[[str, int], None] | None = None,
) -> dict[str, Any]:
    external_root = _validate_expected_binding_root(expected_binding_root)
    output = _absolute_without_resolving(output_directory, label="published authority")
    _safe_basename(output.name, label="published authority basename")
    parent_fd: int | None = None
    output_fd: int | None = None
    marker_present = False
    primary_error: BaseException | None = None
    result: dict[str, Any] | None = None
    close_errors: list[BaseException] = []
    try:
        parent_fd = _open_directory_chain(output.parent, label="published authority parent")
        output_fd = os.open(
            output.name,
            os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=parent_fd,
        )
        opened = os.fstat(output_fd)
        names = set(os.listdir(output_fd))
        marker_present = MARKER_FILENAME in names
        if not marker_present:
            raise PartialPublicationError("terminal marker is absent")
        result = _validate_open_bundle(
            parent_fd=parent_fd,
            output_fd=output_fd,
            output=output,
            opened_directory=opened,
            expected_binding_root=external_root,
            _after_member_snapshot=_after_member_snapshot,
        )
    except BaseException as exc:
        primary_error = exc
    finally:
        if output_fd is not None:
            error = _close_once(output_fd)
            if error is not None:
                close_errors.append(error)
        if parent_fd is not None:
            error = _close_once(parent_fd)
            if error is not None:
                close_errors.append(error)
    if primary_error is not None:
        if isinstance(primary_error, PartialPublicationError):
            raise primary_error
        if marker_present:
            raise CommittedNotAcceptedError("committed authority failed consumer validation") from primary_error
        if isinstance(primary_error, AuthorityError):
            raise primary_error
        raise PublicationError("published authority could not be opened safely") from primary_error
    if close_errors:
        raise CommittedNotAcceptedError("consumer descriptor close failed") from close_errors[0]
    if result is None:
        raise CommittedNotAcceptedError("consumer produced no accepted result")
    return result


def _trip_fault(faults: Mapping[str, BaseException] | None, phase: str) -> None:
    if faults is not None and phase in faults:
        raise faults[phase]


def _marker_stage(
    parent_fd: int,
    payload: bytes,
    *,
    faults: Mapping[str, BaseException] | None,
) -> str:
    for _ in range(16):
        name = f".gse200302-role-authority-{secrets.token_hex(16)}.stage"
        descriptor: int | None = None
        try:
            descriptor = os.open(
                name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
                0o600,
                dir_fd=parent_fd,
            )
        except FileExistsError:
            continue
        except BaseException as exc:
            raise PublicationError("terminal marker staging inode could not be created") from exc
        primary_error: BaseException | None = None
        close_error: BaseException | None = None
        try:
            _write_all(descriptor, payload)
            os.fsync(descriptor)
            _trip_fault(faults, "marker_stage_fsync")
            opened = os.fstat(descriptor)
            if not stat.S_ISREG(opened.st_mode) or opened.st_size != len(payload):
                raise PublicationError("terminal marker staging inode is invalid")
        except BaseException as exc:
            primary_error = exc
        finally:
            close_error = _close_once(descriptor)
            try:
                _trip_fault(faults, "marker_stage_close")
            except BaseException as exc:
                if close_error is None:
                    close_error = exc
        if primary_error is None and close_error is None:
            return name
        try:
            os.unlink(name, dir_fd=parent_fd)
        except BaseException:
            pass
        if primary_error is not None:
            if isinstance(primary_error, AuthorityError):
                raise primary_error
            raise PublicationError("terminal marker staging write failed") from primary_error
        raise PublicationError("terminal marker staging close failed") from close_error
    raise PublicationError("terminal marker staging name contention")


def _link_terminal_marker(parent_fd: int, output_fd: int, stage_name: str) -> None:
    os.link(
        stage_name,
        MARKER_FILENAME,
        src_dir_fd=parent_fd,
        dst_dir_fd=output_fd,
        follow_symlinks=False,
    )


def _terminal_marker_matches_stage(parent_fd: int, output_fd: int, stage_name: str) -> bool:
    try:
        staged = os.stat(stage_name, dir_fd=parent_fd, follow_symlinks=False)
        committed = os.stat(MARKER_FILENAME, dir_fd=output_fd, follow_symlinks=False)
    except BaseException:
        return False
    return (staged.st_dev, staged.st_ino) == (committed.st_dev, committed.st_ino)


def publish_authority(
    output_directory: Path | str,
    *,
    mapping_payload: bytes,
    experiment_join_payload: bytes,
    provenance_document: Mapping[str, Any],
    expected_binding_root: ExpectedBindingRoot | Mapping[str, Any] | None = None,
    _faults: Mapping[str, BaseException] | None = None,
    _post_link_hook: Callable[[Path], None] | None = None,
) -> dict[str, Any]:
    external_root = _validate_expected_binding_root(expected_binding_root)
    member_payloads = _expected_member_payloads(
        provenance_document,
        mapping_payload=mapping_payload,
        experiment_join_payload=experiment_join_payload,
    )
    provenance = _validate_provenance_document(provenance_document)
    _require_provenance_matches_external_root(provenance, external_root)
    output = _absolute_without_resolving(output_directory, label="output directory")
    _safe_basename(output.name, label="output directory basename")
    marker = _marker_factory(
        member_payloads=member_payloads,
        provenance=provenance,
        output=output,
        generated_at=_utc_now(),
    )
    marker_payload = _json_bytes(marker)

    parent_fd: int | None = None
    output_fd: int | None = None
    stage_name: str | None = None
    marker_visible = False
    accepted = False
    output_created = False
    opened_directory: os.stat_result | None = None
    primary_error: BaseException | None = None
    warnings: list[str] = []
    write_trace: list[str] = []
    acceptance: dict[str, Any] | None = None
    try:
        parent_fd = _open_directory_chain(output.parent, label="output parent")
        try:
            os.mkdir(output.name, 0o700, dir_fd=parent_fd)
            output_created = True
        except FileExistsError as exc:
            raise OutputExistsError("exclusive output directory already exists") from exc
        output_fd = os.open(
            output.name,
            os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=parent_fd,
        )
        opened_directory = os.fstat(output_fd)
        if not stat.S_ISDIR(opened_directory.st_mode) or os.listdir(output_fd):
            raise PublicationError("exclusive output directory is not empty")
        for name in (
            MAPPING_FILENAME,
            EXPERIMENT_JOIN_FILENAME,
            PROVENANCE_FILENAME,
            CHECKSUMS_FILENAME,
        ):
            _write_exclusive_at(output_fd, name, member_payloads[name])
            write_trace.append(name)
        if set(os.listdir(output_fd)) != set(member_payloads):
            raise PublicationError("precommit publication member set differs")
        for name, payload in member_payloads.items():
            observed, _ = _read_member_snapshot(
                output_fd,
                name,
                maximum_bytes=MAX_PUBLICATION_MEMBER_BYTES,
            )
            if observed != payload:
                raise PublicationError("precommit publication member bytes differ")
        current = os.stat(output.name, dir_fd=parent_fd, follow_symlinks=False)
        if (current.st_dev, current.st_ino) != (
            opened_directory.st_dev,
            opened_directory.st_ino,
        ):
            raise PublicationError("output directory identity changed before commit")
        os.fsync(output_fd)
        _trip_fault(_faults, "precommit_output_fsync")
        os.fsync(parent_fd)
        _trip_fault(_faults, "precommit_parent_fsync")
        stage_name = _marker_stage(parent_fd, marker_payload, faults=_faults)
        staged_marker, _ = _read_member_snapshot(
            parent_fd,
            stage_name,
            maximum_bytes=MAX_PUBLICATION_MEMBER_BYTES,
        )
        if staged_marker != marker_payload:
            raise PublicationError("terminal marker staging bytes differ")
        try:
            _link_terminal_marker(parent_fd, output_fd, stage_name)
            marker_visible = True
        except BaseException:
            if _terminal_marker_matches_stage(parent_fd, output_fd, stage_name):
                marker_visible = True
            else:
                raise
        write_trace.append(MARKER_FILENAME)
        if _post_link_hook is not None:
            _post_link_hook(output)

        try:
            _trip_fault(_faults, "post_link_marker_stage_unlink")
            os.unlink(stage_name, dir_fd=parent_fd)
            stage_name = None
        except BaseException:
            warnings.append("MARKER_STAGE_UNLINK_WARNING")
        try:
            _trip_fault(_faults, "post_link_output_fsync")
            os.fsync(output_fd)
        except BaseException:
            warnings.append("POST_LINK_OUTPUT_FSYNC_WARNING")
        try:
            _trip_fault(_faults, "post_link_parent_fsync")
            os.fsync(parent_fd)
        except BaseException:
            warnings.append("POST_LINK_PARENT_FSYNC_WARNING")

        acceptance = _validate_open_bundle(
            parent_fd=parent_fd,
            output_fd=output_fd,
            output=output,
            opened_directory=opened_directory,
            expected_binding_root=external_root,
        )
        accepted = True
    except BaseException as exc:
        primary_error = exc
    finally:
        if stage_name is not None and parent_fd is not None:
            try:
                _trip_fault(_faults, "final_marker_stage_unlink")
                os.unlink(stage_name, dir_fd=parent_fd)
                stage_name = None
            except BaseException:
                if marker_visible:
                    warnings.append("FINAL_MARKER_STAGE_UNLINK_WARNING")
                elif primary_error is None:
                    primary_error = PublicationError("precommit marker staging cleanup failed")
        if output_fd is not None:
            close_error = _close_once(output_fd)
            try:
                _trip_fault(_faults, "postcommit_output_close")
            except BaseException as exc:
                if close_error is None:
                    close_error = exc
            if close_error is not None:
                if marker_visible and accepted:
                    warnings.append("POSTCOMMIT_OUTPUT_CLOSE_WARNING")
                elif primary_error is None:
                    primary_error = PublicationError("output descriptor close failed")
        if parent_fd is not None:
            close_error = _close_once(parent_fd)
            try:
                _trip_fault(_faults, "postcommit_parent_close")
            except BaseException as exc:
                if close_error is None:
                    close_error = exc
            if close_error is not None:
                if marker_visible and accepted:
                    warnings.append("POSTCOMMIT_PARENT_CLOSE_WARNING")
                elif primary_error is None:
                    primary_error = PublicationError("parent descriptor close failed")

    if marker_visible and not accepted:
        raise CommittedNotAcceptedError("terminal marker committed but producer acceptance failed") from primary_error
    if not marker_visible:
        if isinstance(primary_error, OutputExistsError):
            raise primary_error
        if isinstance(primary_error, PartialPublicationError):
            raise primary_error
        state = "after output creation" if output_created else "before output creation"
        raise PartialPublicationError(f"publication stopped {state} before terminal marker commit") from primary_error
    if acceptance is None:
        raise CommittedNotAcceptedError("producer accepted no publication result")
    warning_codes = sorted(set(warnings))
    result = dict(acceptance)
    result.update(
        {
            "status": (
                PUBLISHED_WITH_DURABILITY_WARNING
                if warning_codes
                else ROLE_AUTHORITY_STATUS
            ),
            "publication_state": (
                PUBLISHED_WITH_DURABILITY_WARNING
                if warning_codes
                else COMMITTED_AND_ACCEPTED
            ),
            "durability_warning": bool(warning_codes),
            "durability_warning_codes": warning_codes,
            "write_trace": write_trace,
        }
    )
    return result


def _paths_before_read(
    *,
    protocol_path: Path | str,
    runinfo_path: Path | str,
    geo_soft_path: Path | str,
    output_directory: Path | str,
) -> tuple[Path, Path, Path, Path]:
    values = (
        (protocol_path, "protocol path"),
        (runinfo_path, "RunInfo path"),
        (geo_soft_path, "GEO SOFT path"),
        (output_directory, "output directory"),
    )
    for value, label in values:
        _reject_forbidden_path(value, label=label)
    absolute = tuple(_absolute_without_resolving(value, label=label) for value, label in values)
    protocol, runinfo, geo_soft, output = absolute
    if len(set(absolute)) != 4:
        raise ScopeViolation("role-authority paths must be distinct")
    for source, label in ((protocol, "protocol"), (runinfo, "RunInfo"), (geo_soft, "GEO SOFT")):
        if output in source.parents or source in output.parents:
            raise ScopeViolation(f"output and {label} authority paths may not be nested")
    _safe_basename(output.name, label="output directory basename")
    return protocol, runinfo, geo_soft, output


def build_role_authority(
    *,
    protocol_path: Path | str,
    runinfo_path: Path | str,
    geo_soft_path: Path | str,
    output_directory: Path | str,
) -> dict[str, Any]:
    protocol_path, runinfo, geo_soft, output = _paths_before_read(
        protocol_path=protocol_path,
        runinfo_path=runinfo_path,
        geo_soft_path=geo_soft_path,
        output_directory=output_directory,
    )
    protocol, protocol_provenance, protocol_payload = load_contract(protocol_path)
    if protocol_provenance["binding_status"] != "BOUND":
        raise ImplementationBindingUnknown(
            "implementation binding is UNKNOWN_NOT_ASSERTED; official sources and output were not accessed"
        )
    implementation_evidence = verify_implementation_binding(
        protocol_path=protocol_path,
        protocol=protocol,
        protocol_payload=protocol_payload,
        protocol_provenance=protocol_provenance,
    )
    expected_binding_root = ExpectedBindingRoot.from_verified_evidence(
        implementation_evidence
    )
    runinfo_contract = protocol["sources"]["runinfo"]
    soft_contract = protocol["sources"]["geo_soft"]
    runinfo_payload, runinfo_provenance = _read_regular_snapshot(
        runinfo,
        label="frozen RunInfo source",
        maximum_bytes=RUNINFO_BYTES,
        expected_bytes=runinfo_contract["expected_bytes"],
        expected_sha256=runinfo_contract["expected_sha256"],
    )
    soft_payload, soft_provenance = _read_regular_snapshot(
        geo_soft,
        label="frozen GEO SOFT source",
        maximum_bytes=SOFT_BYTES,
        expected_bytes=soft_contract["expected_bytes"],
        expected_sha256=soft_contract["expected_sha256"],
    )
    _, mapping_payload, experiment_join_payload, validation = derive_role_authority(
        runinfo_payload,
        soft_payload,
    )
    provenance = build_provenance_document(
        runinfo_provenance=runinfo_provenance,
        soft_provenance=soft_provenance,
        mapping_payload=mapping_payload,
        experiment_join_payload=experiment_join_payload,
        validation=validation,
        protocol_provenance=protocol_provenance,
        implementation_evidence=implementation_evidence,
    )
    return publish_authority(
        output,
        mapping_payload=mapping_payload,
        experiment_join_payload=experiment_join_payload,
        provenance_document=provenance,
        expected_binding_root=expected_binding_root,
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--runinfo", type=Path)
    parser.add_argument("--geo-soft", type=Path)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--validate-published", type=Path)
    parser.add_argument("--expected-binding-root-json")
    args = parser.parse_args(argv)
    build_values = (args.protocol, args.runinfo, args.geo_soft, args.output_directory)
    if args.validate_published is None and any(value is None for value in build_values):
        parser.error("build mode requires --protocol, --runinfo, --geo-soft, and --output-directory")
    if args.validate_published is not None and any(value is not None for value in build_values):
        parser.error("--validate-published is an exclusive consumer mode")
    if args.validate_published is None and args.expected_binding_root_json is not None:
        parser.error("--expected-binding-root-json is only valid in consumer mode")
    if args.validate_published is not None and args.expected_binding_root_json is None:
        parser.error("consumer mode requires --expected-binding-root-json")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        if args.validate_published is not None:
            expected_root = _parse_json(
                args.expected_binding_root_json.encode("utf-8"),
                label="caller-supplied expected binding root",
                error_type=ImplementationBindingError,
            )
            result = validate_published_authority(
                args.validate_published,
                expected_binding_root=expected_root,
            )
        else:
            result = build_role_authority(
                protocol_path=args.protocol,
                runinfo_path=args.runinfo,
                geo_soft_path=args.geo_soft,
                output_directory=args.output_directory,
            )
    except AuthorityError as exc:
        print(
            json.dumps(
                {
                    "status": exc.publication_state,
                    "failure_code": exc.code,
                    "committed": exc.publication_state == COMMITTED_NOT_ACCEPTED,
                    "accepted": False,
                },
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0 if result.get("accepted") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
