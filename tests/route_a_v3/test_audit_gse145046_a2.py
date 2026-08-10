from __future__ import annotations

import copy
import gzip
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import pytest


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts" / "route_a_v3" / "audit_gse145046_a2.py"
CONFIG_PATH = ROOT / "configs" / "route_a_v3_gse145046_a2_audit.json"
SPEC = importlib.util.spec_from_file_location("audit_gse145046_a2", MODULE_PATH)
assert SPEC and SPEC.loader
A2 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(A2)

A = "AAAAAAAAAA"
C = "AAAAAAAAAC"
G = "AAAAAAAAAG"
T = "AAAAAAAAAT"
INVALID = "NNNNNNNNNN"
FIXED_RUN_ID = "A1_GSE145046_SYNTHETIC_001"
FIXED_EXECUTION_ID = "GSE145046_AUDIT_SYNTHETIC_001"
FIXED_RECORDED_AT = "2026-08-10T12:00:00+08:00"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _gzip_rows(rows: list[tuple[str, int]]) -> bytes:
    denominator = sum(count for _key, count in rows)
    lines = []
    for key, count in rows:
        rpm = count / denominator * 1_000_000 if denominator else 0.0
        lines.append(f"{key}\t{count}\t{rpm:.12f}\n")
    return gzip.compress("".join(lines).encode("utf-8"), mtime=0)


def _run_checked(arguments: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> str:
    completed = subprocess.run(
        arguments,
        cwd=cwd,
        env=env,
        check=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        shell=False,
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout.strip()


def _commit(repo: Path, message: str) -> str:
    environment = dict(os.environ)
    environment.update(
        {
            "GIT_AUTHOR_DATE": "2026-08-10T04:00:00+00:00",
            "GIT_COMMITTER_DATE": "2026-08-10T04:00:00+00:00",
        }
    )
    _run_checked(["git", "add", "--all"], cwd=repo, env=environment)
    _run_checked(
        [
            "git", "-c", "user.name=Synthetic Audit", "-c",
            "user.email=audit@example.invalid", "commit", "-q", "-m", message,
        ],
        cwd=repo,
        env=environment,
    )
    return _run_checked(["git", "rev-parse", "HEAD"], cwd=repo)


def _load_isolated_module(path: Path) -> Any:
    module_name = "synthetic_a2_" + hashlib.sha256(os.fspath(path).encode()).hexdigest()[:16]
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def _keys_from_payload(payload: bytes, fallback: list[tuple[str, int]]) -> list[str]:
    try:
        raw = gzip.decompress(payload)
        if len(raw) > 100_000:
            return [key for key, _count in fallback]
        keys: list[str] = []
        for line in raw.splitlines():
            fields = line.split(b"\t")
            if not fields:
                continue
            keys.append(fields[0].decode("utf-8"))
        return keys or [key for key, _count in fallback]
    except Exception:
        return [key for key, _count in fallback]


def _preflight_from_keys(keys_by_name: dict[str, list[str]]) -> dict[str, int]:
    valid_sets: list[set[str]] = []
    total = valid = invalid = 0
    for name in A2.EXPECTED_FILENAMES:
        keys = keys_by_name[name]
        total += len(keys)
        valid_set = {key for key in keys if A2.VALID_10MER_RE.fullmatch(key)}
        valid += sum(1 for key in keys if A2.VALID_10MER_RE.fullmatch(key))
        invalid += sum(1 for key in keys if A2.VALID_10MER_RE.fullmatch(key) is None)
        valid_sets.append(valid_set)
    input_union = valid_sets[0] | valid_sets[1]
    input_intersection = valid_sets[0] & valid_sets[1]
    all_union = set().union(*valid_sets)
    all_intersection = set.intersection(*valid_sets)
    return {
        "total_rows": total,
        "valid_key_rows": valid,
        "invalid_key_rows": invalid,
        "input_union": len(input_union),
        "input_intersection": len(input_intersection),
        "all_30_union": len(all_union),
        "all_30_intersection": len(all_intersection),
    }


ManifestMutator = Callable[[dict[str, Any]], None]
ProtocolMutator = Callable[[dict[str, Any]], None]


class Fixture:
    def __init__(
        self,
        *,
        module: Any,
        repo: Path,
        contract: Path,
        protocol: Path,
        protocol_sha256: str,
        manifest: Path,
        test_path: Path,
        output_directory: Path,
        data_directory: Path,
        code_commit: str,
        test_sha256: str,
    ) -> None:
        self.module = module
        self.repo = repo
        self.contract = contract
        self.protocol = protocol
        self.protocol_sha256 = protocol_sha256
        self.manifest = manifest
        self.test_path = test_path
        self.output_directory = output_directory
        self.data_directory = data_directory
        self.code_commit = code_commit
        self.test_sha256 = test_sha256

    def output_path(self, execution_id: str = FIXED_EXECUTION_ID) -> Path:
        return self.output_directory / f"{execution_id}.json"

    def run(self, **overrides: Any) -> dict[str, Any]:
        values: dict[str, Any] = {
            "contract_path": self.contract,
            "protocol_path": self.protocol,
            "expected_protocol_sha256": self.protocol_sha256,
            "manifest_path": self.manifest,
            "repo_root": self.repo,
            "test_path": self.test_path,
            "run_id": FIXED_RUN_ID,
            "audit_execution_id": FIXED_EXECUTION_ID,
            "code_commit": self.code_commit,
            "recorded_at": FIXED_RECORDED_AT,
            "test_sha256": self.test_sha256,
        }
        values.update(overrides)
        values.setdefault("output_path", self.output_path(values["audit_execution_id"]))
        return self.module.audit_gse145046_a2(**values)

    def recommit(self, message: str) -> str:
        self.code_commit = _commit(self.repo, message)
        return self.code_commit


def _fixture(
    tmp_path: Path,
    *,
    rows_by_name: dict[str, list[tuple[str, int]]] | None = None,
    raw_by_name: dict[str, bytes] | None = None,
    manifest_mutator: ManifestMutator | None = None,
    protocol_mutator: ProtocolMutator | None = None,
) -> Fixture:
    rows_by_name = rows_by_name or {}
    raw_by_name = raw_by_name or {}
    repo = tmp_path / "repo"
    authority = tmp_path / "authority"
    data = tmp_path / "ordinary_public" / A2.DATASET_ID
    outputs = tmp_path / "outputs"
    for directory in (repo, authority, data, outputs):
        directory.mkdir(parents=True)

    contract = authority / "contract.md"
    contract.write_bytes(b"synthetic Route-A V3 contract fixture\n")
    contract_sha = _sha256_bytes(contract.read_bytes())

    entries: list[dict[str, Any]] = []
    keys_by_name: dict[str, list[str]] = {}
    for index, name in enumerate(A2.EXPECTED_FILENAMES):
        default_rows = [(C, 1), (G, 1)]
        if index == 0:
            default_rows = [(A, 1), (C, 1)]
        elif index == 1:
            default_rows = [(C, 1), (G, 1)]
        rows = rows_by_name.get(name, default_rows)
        payload = raw_by_name.get(name, _gzip_rows(rows))
        (data / name).write_bytes(payload)
        keys_by_name[name] = _keys_from_payload(payload, rows)
        entries.append(
            {
                "bytes": len(payload),
                "downloaded": True,
                "expected_bytes": len(payload),
                "name": name,
                "sha256": _sha256_bytes(payload),
                "url": f"https://ftp.ncbi.nlm.nih.gov/geo/{name}",
            }
        )

    manifest_document: dict[str, Any] = {
        "accession": A2.DATASET_ID,
        "files": entries,
        "provider": "NCBI_GEO_SYNTHETIC_FIXTURE",
        "retrieved_at_utc": "2026-08-10T00:00:00Z",
        "skipped": [],
        "source_url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE145046",
    }
    if manifest_mutator:
        manifest_mutator(manifest_document)
    manifest = data / "manifest.json"
    manifest_bytes = (json.dumps(manifest_document, indent=2, sort_keys=True) + "\n").encode()
    manifest.write_bytes(manifest_bytes)

    protocol_document = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    protocol_document["authority"]["contract_sha256"] = contract_sha
    protocol_document["inputs"]["p0_manifest"]["sha256"] = _sha256_bytes(manifest_bytes)
    protocol_document["aggregate_preflight_evidence"]["observed_aggregates"] = (
        _preflight_from_keys(keys_by_name)
    )
    if protocol_mutator:
        protocol_mutator(protocol_document)
    protocol_bytes = (json.dumps(protocol_document, indent=2, sort_keys=True) + "\n").encode()

    protocol = repo / A2.PROTOCOL_REPOSITORY_PATH
    script = repo / A2.SCRIPT_REPOSITORY_PATH
    test_path = repo / A2.TEST_REPOSITORY_PATH
    for path in (protocol, script, test_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    protocol.write_bytes(protocol_bytes)
    script.write_bytes(MODULE_PATH.read_bytes())
    test_bytes = Path(__file__).read_bytes()
    test_path.write_bytes(test_bytes)

    _run_checked(["git", "init", "-q"], cwd=repo)
    code_commit = _commit(repo, "synthetic formal audit fixture")
    module = _load_isolated_module(script)
    protocol_sha = _sha256_bytes(protocol_bytes)
    module.CANONICAL_PROTOCOL_SHA256 = protocol_sha
    assert _run_checked(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"], cwd=repo
    ) == ""
    return Fixture(
        module=module,
        repo=repo,
        contract=contract,
        protocol=protocol,
        protocol_sha256=protocol_sha,
        manifest=manifest,
        test_path=test_path,
        output_directory=outputs,
        data_directory=data,
        code_commit=code_commit,
        test_sha256=_sha256_bytes(test_bytes),
    )


def test_canonical_protocol_hash_exact_roles_and_closed_science() -> None:
    raw = CONFIG_PATH.read_bytes()
    protocol = json.loads(raw)
    assert _sha256_bytes(raw) == A2.CANONICAL_PROTOCOL_SHA256
    assert protocol["protocol_status"] == (
        "FROZEN_AFTER_AGGREGATE_PREFLIGHT_BEFORE_FORMAL_VERSIONED_AUDIT_AND_MODEL_RESULTS"
    )
    assert protocol["science_authority"]["version_of_record"]["doi"] == "10.1038/s41594-020-0465-x"
    assert protocol["science_authority"]["preprint"]["doi"] == "10.1101/2020.03.13.990887"
    assert protocol["inputs"]["p0_manifest"]["sha256"] == "539b47a2962f4d875c8b16162d87d98f20082f659ce0e4b54ae6b5b61e681a22"
    tuples = tuple(
        (item["filename"], item["gsm_accession"], item["sample_group"], item["condition"], item["replicate"])
        for item in protocol["inputs"]["samples"]
    )
    assert tuples == A2.EXPECTED_SAMPLE_SPECS
    boundary = protocol["qualification_boundary"]
    assert boundary["a2_status"] == "NOT_TRUE_A2_FIXED_REPORTER_ABSOLUTE_AUXILIARY"
    assert boundary["data_semantics"] == "FIXED_SCAFFOLD_ABSOLUTE_OUTCOMES_NOT_DIRECT_SOURCE_TO_CANDIDATE_INTERVENTIONS"
    assert boundary["qualified"] is False
    assert protocol["aggregate_preflight_evidence"]["formal_reconciliation_required"] is True
    assert protocol["output_policy"]["self_hash"] == "NOT_APPLICABLE_BOUND_BY_OUTER_RUN_MANIFEST_OR_LEDGER"


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"run_id": "unsafe/run"}, "run_id"),
        ({"audit_execution_id": "unsafe/run"}, "audit_execution_id"),
        ({"audit_execution_id": FIXED_RUN_ID}, "must differ"),
        ({"code_commit": "not-a-commit"}, "code_commit"),
        ({"recorded_at": "2026-08-10T12:00:00"}, "explicit offset"),
        ({"test_sha256": "bad"}, "test SHA-256"),
    ],
)
def test_required_execution_metadata_is_strict(overrides: dict[str, str], message: str) -> None:
    values = {
        "run_id": FIXED_RUN_ID,
        "audit_execution_id": FIXED_EXECUTION_ID,
        "code_commit": "c" * 40,
        "recorded_at": FIXED_RECORDED_AT,
        "test_sha256": "d" * 64,
    }
    values.update(overrides)
    with pytest.raises(A2.A2AuditError, match=message):
        A2._validate_run_metadata(**values)


def test_forbidden_path_wins_before_any_input_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output_parent = tmp_path / "outputs"
    output_parent.mkdir()
    monkeypatch.setattr(A2, "_verify_repo_binding", lambda **_kwargs: pytest.fail("must not read inputs"))
    with pytest.raises(A2.ScopeViolation, match="forbidden path token"):
        A2.audit_gse145046_a2(
            contract_path=tmp_path / "restricted" / "contract.md",
            protocol_path=tmp_path / "missing-protocol.json",
            expected_protocol_sha256="0" * 64,
            manifest_path=tmp_path / "missing-data" / "missing-manifest.json",
            output_path=output_parent / f"{FIXED_EXECUTION_ID}.json",
            repo_root=tmp_path,
            test_path=tmp_path / "missing-test.py",
            run_id=FIXED_RUN_ID,
            audit_execution_id=FIXED_EXECUTION_ID,
            code_commit="c" * 40,
            recorded_at=FIXED_RECORDED_AT,
            test_sha256="d" * 64,
        )


def test_existing_output_rejected_before_any_input_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / f"{FIXED_EXECUTION_ID}.json"
    output.write_bytes(b"do-not-overwrite")
    monkeypatch.setattr(A2, "_verify_repo_binding", lambda **_kwargs: pytest.fail("must not read inputs"))
    with pytest.raises(A2.A2AuditError, match="refusing to overwrite"):
        A2.audit_gse145046_a2(
            contract_path=tmp_path / "missing-contract.md",
            protocol_path=tmp_path / "missing-protocol.json",
            expected_protocol_sha256="0" * 64,
            manifest_path=tmp_path / "missing-data" / "missing-manifest.json",
            output_path=output,
            repo_root=tmp_path,
            test_path=tmp_path / "missing-test.py",
            run_id=FIXED_RUN_ID,
            audit_execution_id=FIXED_EXECUTION_ID,
            code_commit="c" * 40,
            recorded_at=FIXED_RECORDED_AT,
            test_sha256="d" * 64,
        )
    assert output.read_bytes() == b"do-not-overwrite"


def test_symlink_manifest_is_rejected_before_payload_open(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _fixture(tmp_path)
    real = fixture.data_directory / "manifest-real.json"
    fixture.manifest.rename(real)
    fixture.manifest.symlink_to(real)
    monkeypatch.setattr(
        fixture.module,
        "_open_payload_at",
        lambda *_args, **_kwargs: pytest.fail("payload must not open"),
    )
    with pytest.raises(fixture.module.ScopeViolation, match="symlink"):
        fixture.run()


def test_manifest_path_escape_and_exact_schema_fail_closed(tmp_path: Path) -> None:
    def mutate(document: dict[str, Any]) -> None:
        document["files"][0]["name"] = "../escape.txt.gz"

    fixture = _fixture(tmp_path, manifest_mutator=mutate)
    with pytest.raises(fixture.module.ScopeViolation, match="safe basename"):
        fixture.run()
    assert not fixture.output_path().exists()


def test_manifest_missing_key_fails_closed(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, manifest_mutator=lambda document: document.pop("provider"))
    with pytest.raises(fixture.module.A2AuditError, match="exact key set mismatch"):
        fixture.run()
    assert not fixture.output_path().exists()


def test_forbidden_data_directory_entry_fails_before_payload_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    (fixture.data_directory / "restricted-marker.txt").write_text(
        "must not be read", encoding="utf-8"
    )
    monkeypatch.setattr(
        fixture.module,
        "_open_payload_at",
        lambda *_args, **_kwargs: pytest.fail("payload must not open"),
    )
    with pytest.raises(fixture.module.ScopeViolation, match="forbidden path token"):
        fixture.run()


def test_protocol_unknown_qualification_key_is_rejected_by_closed_schema(tmp_path: Path) -> None:
    def mutate(document: dict[str, Any]) -> None:
        document["qualification_boundary"]["unknown"] = False

    fixture = _fixture(tmp_path, protocol_mutator=mutate)
    with pytest.raises(fixture.module.A2AuditError, match="qualification_boundary"):
        fixture.run()


def test_protocol_exact_sample_role_tuple_is_enforced(tmp_path: Path) -> None:
    def mutate(document: dict[str, Any]) -> None:
        document["inputs"]["samples"][4]["condition"] = "POLYSOME"

    fixture = _fixture(tmp_path, protocol_mutator=mutate)
    with pytest.raises(fixture.module.A2AuditError, match="role tuple"):
        fixture.run()


@pytest.mark.parametrize("case", ["boolean_as_integer", "float_as_integer", "numeric_string"])
def test_protocol_exact_schema_rejects_json_type_aliases(tmp_path: Path, case: str) -> None:
    def mutate(document: dict[str, Any]) -> None:
        if case == "boolean_as_integer":
            document["inputs"]["samples"][0]["replicate"] = True
        elif case == "float_as_integer":
            document["inputs"]["samples"][0]["replicate"] = 1.0
        else:
            document["inputs"]["rpm_definition"]["absolute_tolerance"] = "0.000001"

    fixture = _fixture(tmp_path, protocol_mutator=mutate)
    with pytest.raises(fixture.module.A2AuditError):
        fixture.run()


def test_canonical_trust_root_rejects_committed_drift_even_with_new_caller_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    original_canonical = fixture.module.CANONICAL_PROTOCOL_SHA256
    fixture.protocol.write_bytes(fixture.protocol.read_bytes() + b" ")
    new_hash = _sha256_bytes(fixture.protocol.read_bytes())
    new_commit = fixture.recommit("protocol drift")
    assert new_hash != original_canonical
    monkeypatch.setattr(
        fixture.module,
        "_open_payload_at",
        lambda *_args, **_kwargs: pytest.fail("payload must not open"),
    )
    with pytest.raises(fixture.module.A2AuditError, match="hardcoded canonical trust root"):
        fixture.run(expected_protocol_sha256=new_hash, code_commit=new_commit)


@pytest.mark.parametrize(
    "override, message",
    [
        ({"code_commit": "0" * 40}, "Git HEAD"),
        ({"test_sha256": "0" * 64}, "focused test SHA-256"),
    ],
)
def test_false_code_or_test_binding_fails_before_payload_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    override: dict[str, str],
    message: str,
) -> None:
    fixture = _fixture(tmp_path)
    monkeypatch.setattr(
        fixture.module,
        "_open_payload_at",
        lambda *_args, **_kwargs: pytest.fail("payload must not open"),
    )
    with pytest.raises(fixture.module.A2AuditError, match=message):
        fixture.run(**override)


def test_dirty_repo_fails_before_payload_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _fixture(tmp_path)
    (fixture.repo / "untracked.txt").write_text("dirty", encoding="utf-8")
    monkeypatch.setattr(
        fixture.module,
        "_open_payload_at",
        lambda *_args, **_kwargs: pytest.fail("payload must not open"),
    )
    with pytest.raises(fixture.module.A2AuditError, match="worktree must be clean"):
        fixture.run()


def test_payload_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    first = fixture.data_directory / A2.EXPECTED_FILENAMES[0]
    first.write_bytes(first.read_bytes() + b"tamper")
    with pytest.raises(fixture.module.A2AuditError, match="SHA-256 mismatch"):
        fixture.run()
    assert not fixture.output_path().exists()


def test_bad_gzip_fails_closed_with_bound_manifest_hash(tmp_path: Path) -> None:
    first = A2.EXPECTED_FILENAMES[0]
    fixture = _fixture(tmp_path, raw_by_name={first: b"not-a-gzip-stream"})
    with pytest.raises(fixture.module.A2AuditError, match="gzip integrity failed"):
        fixture.run()


@pytest.mark.parametrize(
    "payload, message",
    [
        (gzip.compress(f"{A}\t1\n".encode(), mtime=0), "expected 3 columns"),
        (gzip.compress(f"{A}\tNaN\t0\n".encode(), mtime=0), "invalid count"),
        (gzip.compress(f"{A}\t1.0\t0\n".encode(), mtime=0), "invalid count"),
        (gzip.compress(f"{A}\t1e0\t0\n".encode(), mtime=0), "invalid count"),
        (gzip.compress(f"{A}\t+1\t0\n".encode(), mtime=0), "invalid count"),
        (gzip.compress(f"{A}\t-1\t0\n".encode(), mtime=0), "invalid count"),
        (gzip.compress(f"{A}\t01\t0\n".encode(), mtime=0), "invalid count"),
        (gzip.compress(f"{A}\t1\t-1\n".encode(), mtime=0), "invalid RPM"),
        (gzip.compress(f"{A}\t1\tnan\n".encode(), mtime=0), "invalid RPM"),
    ],
)
def test_strict_three_column_integer_count_and_finite_rpm(
    tmp_path: Path, payload: bytes, message: str
) -> None:
    first = A2.EXPECTED_FILENAMES[0]
    fixture = _fixture(tmp_path, raw_by_name={first: payload})
    with pytest.raises(fixture.module.A2AuditError, match=message):
        fixture.run()


def test_bounded_gzip_read_rejects_highly_compressed_unterminated_long_line(tmp_path: Path) -> None:
    first = A2.EXPECTED_FILENAMES[0]
    payload = gzip.compress(b"A" * (A2.MAX_DECOMPRESSED_LINE_BYTES + 100_000), mtime=0)
    fixture = _fixture(tmp_path, raw_by_name={first: payload})
    with pytest.raises(fixture.module.A2AuditError, match="row too long"):
        fixture.run()
    assert not fixture.output_path().exists()


def test_invalid_key_missing_semantics_union_intersection_and_no_raw_leakage(tmp_path: Path) -> None:
    rows = {A2.EXPECTED_FILENAMES[2]: [(C, 0), (T, 1), (INVALID, 1)]}
    fixture = _fixture(tmp_path, rows_by_name=rows)
    report = fixture.run()
    aggregate = report["presence_aggregates"]
    assert aggregate["total_rows"] == 61
    assert aggregate["valid_key_rows"] == 60
    assert aggregate["invalid_key_rows"] == 1
    assert aggregate["invalid_key_reason_counts"] == {"INVALID_ALPHABET": 1}
    assert aggregate["input_two_file_union_keys"] == 3
    assert aggregate["input_two_file_intersection_keys"] == 1
    assert aggregate["all_30_file_union_keys"] == 4
    assert aggregate["all_30_file_intersection_keys"] == 1
    endpoint = report["endpoint_presence_relative_to_input_union"][0]
    assert endpoint["shared_with_input_union_keys"] == 1
    assert endpoint["endpoint_only_vs_input_union_keys"] == 1
    assert endpoint["missing_from_endpoint_vs_input_union_keys"] == 2
    assert report["row_semantics"]["absent_key_is_missing"] is True
    assert report["row_semantics"]["absent_key_imputed_as_zero"] is False
    serialized = fixture.output_path().read_text(encoding="utf-8")
    for raw_key in (A, C, G, T, INVALID):
        assert raw_key not in serialized
    assert '"raw_rows"' not in serialized
    assert '"labels"' not in serialized


def test_duplicate_valid_key_fails_closed(tmp_path: Path) -> None:
    rows = {A2.EXPECTED_FILENAMES[4]: [(C, 1), (G, 1), (C, 1)]}
    fixture = _fixture(tmp_path, rows_by_name=rows)
    with pytest.raises(fixture.module.A2AuditError, match="duplicate valid 10-mer key"):
        fixture.run()


def test_rpm_mismatch_is_a_distinct_failed_validation_axis(tmp_path: Path) -> None:
    first = A2.EXPECTED_FILENAMES[0]
    wrong = gzip.compress(f"{A}\t1\t1\n{C}\t1\t999999\n".encode(), mtime=0)
    fixture = _fixture(tmp_path, raw_by_name={first: wrong})
    report = fixture.run()
    per_file = report["files"][0]["rpm_consistency"]
    assert per_file["status"] == "FAIL_MISMATCH"
    assert per_file["mismatch_rows"] == 2
    aggregate = report["rpm_validation_aggregate"]
    assert aggregate["status"] == "FAIL_MISMATCH"
    assert aggregate["total_mismatch_rows"] == 2
    assert aggregate["files_with_mismatches"] == [first]
    assert report["rpm_validation_status"] == "FAIL_MISMATCH"
    assert report["payload_integrity_status"] == "PASS"
    assert report["dataset_qualification_status"] == "NOT_QUALIFIED"


def test_zero_denominator_nonzero_deposited_rpm_is_counted_as_anomaly(tmp_path: Path) -> None:
    first = A2.EXPECTED_FILENAMES[0]
    raw = gzip.compress(f"{A}\t0\t3\n{C}\t0\t0\n".encode(), mtime=0)
    fixture = _fixture(tmp_path, raw_by_name={first: raw})
    report = fixture.run()
    per_file = report["files"][0]["rpm_consistency"]
    assert per_file["status"] == "UNDEFINED_ZERO_DENOMINATOR"
    assert per_file["rows_not_checkable_zero_denominator"] == 2
    assert per_file["nonzero_deposited_rpm_rows_with_zero_denominator"] == 1
    aggregate = report["rpm_validation_aggregate"]
    assert aggregate["status"] == "UNDEFINED_ZERO_DENOMINATOR"
    assert aggregate["total_mismatch_rows"] == 0
    assert aggregate["total_nonzero_deposited_rpm_rows_with_zero_denominator"] == 1
    assert aggregate["files_with_nonzero_deposited_rpm_zero_denominator"] == [first]


def test_mismatch_and_zero_denominator_statuses_combine(tmp_path: Path) -> None:
    first, second = A2.EXPECTED_FILENAMES[:2]
    raw = {
        first: gzip.compress(f"{A}\t1\t0\n{C}\t1\t0\n".encode(), mtime=0),
        second: gzip.compress(f"{C}\t0\t0\n{G}\t0\t0\n".encode(), mtime=0),
    }
    fixture = _fixture(tmp_path, raw_by_name=raw)
    report = fixture.run()
    assert report["rpm_validation_status"] == "FAIL_MISMATCH_AND_UNDEFINED_ZERO_DENOMINATOR"


def test_formal_aggregate_reconciliation_cannot_be_disabled(tmp_path: Path) -> None:
    def mutate(document: dict[str, Any]) -> None:
        observed = document["aggregate_preflight_evidence"]["observed_aggregates"]
        observed["total_rows"] += 1
        observed["valid_key_rows"] += 1

    fixture = _fixture(tmp_path, protocol_mutator=mutate)
    with pytest.raises(fixture.module.A2AuditError, match="formal aggregate reconciliation mismatch"):
        fixture.run()
    assert not fixture.output_path().exists()


def test_each_payload_is_opened_once_for_hash_presence_and_rpm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    module = fixture.module
    original_open = module._open_payload_at
    opens: dict[str, int] = {}

    def count_open(data: Any, filename: str) -> tuple[int, Any]:
        opens[filename] = opens.get(filename, 0) + 1
        return original_open(data, filename)

    monkeypatch.setattr(module, "_open_payload_at", count_open)
    fixture.run()
    assert opens == {name: 1 for name in module.EXPECTED_FILENAMES}


def test_payload_leaf_swap_after_hash_fails_without_reading_symlink_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    module = fixture.module
    first = module.EXPECTED_FILENAMES[0]
    original_hash = module._hash_open_fd
    scans = 0

    def swap_after_hash(descriptor: int, filename: str) -> tuple[str, int]:
        result = original_hash(descriptor, filename)
        if filename == first:
            leaf = fixture.data_directory / filename
            saved = fixture.data_directory / "saved-original.gz"
            target = fixture.data_directory / "replacement-target.gz"
            leaf.rename(saved)
            target.write_bytes(b"replacement target must never be opened")
            leaf.symlink_to(target)
        return result

    original_scan = module._scan_presence_fd

    def count_scan(descriptor: int, filename: str) -> dict[str, Any]:
        nonlocal scans
        scans += 1
        return original_scan(descriptor, filename)

    monkeypatch.setattr(module, "_hash_open_fd", swap_after_hash)
    monkeypatch.setattr(module, "_scan_presence_fd", count_scan)
    with pytest.raises(module.ScopeViolation, match="symlink|leaf identity|descriptor changed"):
        fixture.run()
    assert scans == 0
    assert not fixture.output_path().exists()


def test_same_fd_identity_change_between_presence_and_rpm_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    module = fixture.module
    first = module.EXPECTED_FILENAMES[0]
    original_scan = module._scan_presence_fd

    def mutate_after_presence(descriptor: int, filename: str) -> dict[str, Any]:
        result = original_scan(descriptor, filename)
        if filename == first:
            leaf = fixture.data_directory / filename
            leaf.write_bytes(leaf.read_bytes() + b"changed-between-passes")
        return result

    monkeypatch.setattr(module, "_scan_presence_fd", mutate_after_presence)
    with pytest.raises(module.ScopeViolation, match="descriptor changed"):
        fixture.run()
    assert not fixture.output_path().exists()


def test_output_parent_swap_never_publishes_into_symlink_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    module = fixture.module
    original_json = module._json_bytes
    target = tmp_path / "attacker-target"
    bound = tmp_path / "bound-output-parent"

    def swap_parent(report: dict[str, Any]) -> bytes:
        payload = original_json(report)
        fixture.output_directory.rename(bound)
        target.mkdir()
        fixture.output_directory.symlink_to(target, target_is_directory=True)
        return payload

    monkeypatch.setattr(module, "_json_bytes", swap_parent)
    with pytest.raises(module.ScopeViolation, match="output parent"):
        fixture.run()
    assert not (target / f"{FIXED_EXECUTION_ID}.json").exists()
    assert not (bound / f"{FIXED_EXECUTION_ID}.json").exists()


def test_fixed_explicit_metadata_is_deterministic_across_isolated_repositories(tmp_path: Path) -> None:
    first = _fixture(tmp_path / "one")
    second = _fixture(tmp_path / "two")
    assert first.code_commit == second.code_commit
    first_report = first.run()
    second_report = second.run()
    assert first_report == second_report
    assert first.output_path().read_bytes() == second.output_path().read_bytes()


def test_no_replace_and_successful_hidden_staging_cleanup(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture.run()
    original = fixture.output_path().read_bytes()
    assert not list(fixture.output_directory.glob(".*.partial-staging-*"))
    with pytest.raises(fixture.module.A2AuditError, match="refusing to overwrite"):
        fixture.run()
    assert fixture.output_path().read_bytes() == original


def test_atomic_link_failure_retains_staging_and_never_creates_final(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)

    def fail_link(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("synthetic link failure")

    monkeypatch.setattr(fixture.module.os, "link", fail_link)
    with pytest.raises(fixture.module.A2AuditError, match="atomic no-replace publication failed"):
        fixture.run()
    assert not fixture.output_path().exists()
    staging = list(fixture.output_directory.glob(f".{FIXED_EXECUTION_ID}.json.partial-staging-*"))
    assert len(staging) == 1
    staged_report = json.loads(staging[0].read_text(encoding="utf-8"))
    assert staged_report["audit_execution_status"] == "COMPLETED"
    assert staged_report["dataset_qualification_status"] == "NOT_QUALIFIED"


def test_staging_name_swap_before_link_cannot_publish_replacement_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    module = fixture.module
    original_link = module.os.link

    def swap_source_then_link(
        source: str, target: str, **kwargs: Any
    ) -> None:
        source_path = fixture.output_directory / source
        source_path.rename(fixture.output_directory / "saved-open-staging-inode")
        source_path.write_bytes(b"replacement bytes must not publish")
        original_link(source, target, **kwargs)

    monkeypatch.setattr(module.os, "link", swap_source_then_link)
    with pytest.raises(module.ScopeViolation, match="did not bind the open staging inode"):
        fixture.run()
    assert not fixture.output_path().exists()


@pytest.mark.parametrize(
    "case",
    [
        "lowercase_sequence",
        "invalid_alphabet",
        "embedded_sequence",
        "raw_rows",
        "unknown_nested_key",
        "label_list",
        "boolean_integer_alias",
        "numeric_string",
        "reconciliation_boolean_alias",
    ],
)
def test_closed_report_allowlist_is_primary_no_raw_gate(tmp_path: Path, case: str) -> None:
    fixture = _fixture(tmp_path)
    report = fixture.run()
    candidate = copy.deepcopy(report)
    if case == "lowercase_sequence":
        candidate["raw_sequence"] = "aaaaaaaaaa"
    elif case == "invalid_alphabet":
        candidate["source_anchor"]["canonical_source_hash"] = "NNNNNNNNNN"
    elif case == "embedded_sequence":
        candidate["evidence_status"] = "BLOCKED_AAAAAAAAAA_PENDING"
    elif case == "raw_rows":
        candidate["raw_rows"] = []
    elif case == "unknown_nested_key":
        candidate["qualification"]["unknown"] = False
    elif case == "label_list":
        candidate["labels"] = ["sample-label"]
    elif case == "boolean_integer_alias":
        candidate["qualification"]["qualified"] = 0
    elif case == "numeric_string":
        candidate["files"][0]["rpm_sum"] = "1000000"
    elif case == "reconciliation_boolean_alias":
        candidate["aggregate_reconciliation"]["expected"][
            "input_intersection"
        ] = True
    with pytest.raises(fixture.module.A2AuditError):
        fixture.module._validate_closed_report_schema(candidate)


def test_status_bindings_and_zero_materialization_invariants(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    report = fixture.run()
    assert report["record_id"] == f"GSE145046_A2_AUDIT_{FIXED_EXECUTION_ID}"
    assert report["phase_id"] == "A1"
    assert report["run_id"] == FIXED_RUN_ID
    assert report["audit_execution_id"] == FIXED_EXECUTION_ID
    assert report["audit_execution_status"] == "COMPLETED"
    assert report["payload_integrity_status"] == "PASS"
    assert report["rpm_validation_status"] == "PASS"
    assert report["aggregate_reconciliation_status"] == "MATCH"
    assert report["dataset_qualification_status"] == "NOT_QUALIFIED"
    assert report["training_authorization"] == "DENIED"
    assert report["evidence_status"] == "BLOCKED_PENDING_PUBLIC_EVIDENCE"
    assert report["scientific_claim_status"] == "NOT_ESTABLISHED"
    assert report["paper_method_reproduced"] is False
    code = report["bindings"]["code"]
    assert code["code_commit"] == fixture.code_commit == code["git_head"]
    assert code["git_worktree_clean"] is True
    assert code["verification"] == "VERIFIED_HEAD_CLEAN_AND_EXACT_BYTES"
    assert all(code[key]["head_blob_match"] is True for key in ("protocol", "script", "test"))
    assert code["test"]["sha256"] == fixture.test_sha256
    assert report["bindings"]["protocol"]["verification"] == "VERIFIED_CANONICAL_CALLER_ACTUAL_AND_HEAD"
    assert report["qualification"] == {
        "classification": "CONDITIONALLY_RECOVERABLE_AS_ABSOLUTE_AUXILIARY",
        "a2_status": "NOT_TRUE_A2_FIXED_REPORTER_ABSOLUTE_AUXILIARY",
        "qualified": False,
        "training_allowed": False,
        "model_selection_allowed": False,
        "canonical_intervention_records_materialized": False,
        "measured_candidate_pools_materialized": False,
        "endpoint_values_materialized": False,
        "data_semantics": "FIXED_SCAFFOLD_ABSOLUTE_OUTCOMES_NOT_DIRECT_SOURCE_TO_CANDIDATE_INTERVENTIONS",
        "license": {"status": "UNKNOWN_BLOCKED", "redistribution_allowed": False},
        "foundation_exposure": {
            "checkpoint_specific_status": "UNKNOWN_NOT_ASSERTED",
            "project_engineering_consumption": "CONFIRMED",
            "independent_holdout_eligibility": False,
        },
    }
    assert report["record_materialization"] == {
        "canonical_intervention_record_count": 0,
        "measured_candidate_pool_count": 0,
        "canonical_or_pool_artifacts_written": False,
    }
    assert report["output_policy"]["self_hash"] == "NOT_APPLICABLE_BOUND_BY_OUTER_RUN_MANIFEST_OR_LEDGER"


def test_cli_requires_repo_test_and_audit_execution_bindings() -> None:
    with pytest.raises(SystemExit):
        A2._parse_args([])
    parsed = A2._parse_args(
        [
            "--contract", "contract.md", "--protocol", "protocol.json",
            "--protocol-sha256", "0" * 64, "--p0-manifest", "manifest.json",
            "--output", f"{FIXED_EXECUTION_ID}.json", "--repo-root", ".",
            "--test-path", A2.TEST_REPOSITORY_PATH, "--run-id", FIXED_RUN_ID,
            "--audit-execution-id", FIXED_EXECUTION_ID, "--code-commit", "0" * 40,
            "--recorded-at", FIXED_RECORDED_AT, "--test-sha256", "0" * 64,
        ]
    )
    assert parsed.audit_execution_id == FIXED_EXECUTION_ID
    assert parsed.repo_root == Path(".")
