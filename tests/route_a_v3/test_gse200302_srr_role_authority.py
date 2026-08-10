from __future__ import annotations

import copy
import csv
import gzip
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest


STAGING = Path(__file__).resolve().parents[2]
SCRIPT = STAGING / "scripts" / "route_a_v3" / "build_gse200302_srr_role_authority.py"
CONFIG = STAGING / "configs" / "route_a_v3_gse200302_srr_role_authority.json"
TEST = Path(__file__).resolve()

# Compile/exec avoids target-tree bytecode and package import side effects.
AUTHORITY = types.ModuleType("build_gse200302_srr_role_authority_test_target")
AUTHORITY.__file__ = str(SCRIPT)
sys.modules[AUTHORITY.__name__] = AUTHORITY
exec(compile(SCRIPT.read_bytes(), str(SCRIPT), "exec"), AUTHORITY.__dict__)


def _runinfo_bytes(
    *,
    mutations: Mapping[str, Mapping[str, str]] | None = None,
    omit_run: str | None = None,
) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output,
        fieldnames=list(AUTHORITY.RUNINFO_REQUIRED_COLUMNS),
        lineterminator="\n",
    )
    writer.writeheader()
    for expected in AUTHORITY.EXPECTED_JOIN_ROWS:
        if expected["run_accession"] == omit_run:
            continue
        row = {
            "Run": expected["run_accession"],
            "Experiment": expected["experiment_accession"],
            "LibraryName": expected["geo_sample_accession"],
            "BioProject": AUTHORITY.TARGET_BIOPROJECT,
            "BioSample": expected["biosample_accession"],
            "SampleName": expected["geo_sample_accession"],
        }
        if mutations and expected["run_accession"] in mutations:
            row.update(mutations[expected["run_accession"]])
        writer.writerow(row)
    return output.getvalue().encode("ascii")


def _soft_bytes(
    *,
    title_overrides: Mapping[str, str] | None = None,
    experiment_overrides: Mapping[str, str] | None = None,
    biosample_overrides: Mapping[str, str] | None = None,
    duplicate_field_by_gsm: Mapping[str, str] | None = None,
    compressed: bool = True,
) -> bytes:
    lines = [f"^SERIES = {AUTHORITY.TARGET_SERIES}"]
    for expected in sorted(
        AUTHORITY.EXPECTED_JOIN_ROWS,
        key=lambda row: int(row["geo_sample_accession"][3:]),
    ):
        gsm = expected["geo_sample_accession"]
        title = f"{expected['measurement_family']}_rep{expected['replicate']}"
        experiment = expected["experiment_accession"]
        biosample = expected["biosample_accession"]
        if title_overrides and gsm in title_overrides:
            title = title_overrides[gsm]
        if experiment_overrides and gsm in experiment_overrides:
            experiment = experiment_overrides[gsm]
        if biosample_overrides and gsm in biosample_overrides:
            biosample = biosample_overrides[gsm]
        lines.extend(
            (
                f"^SAMPLE = {gsm}",
                f"!Sample_title = {title}",
                f"!Sample_geo_accession = {gsm}",
                f"!Sample_relation = BioSample: https://www.ncbi.nlm.nih.gov/biosample/{biosample}",
                f"!Sample_relation = SRA: https://www.ncbi.nlm.nih.gov/sra?term={experiment}",
            )
        )
        if duplicate_field_by_gsm and gsm in duplicate_field_by_gsm:
            lines.append(duplicate_field_by_gsm[gsm])
    payload = ("\n".join(lines) + "\n").encode("ascii")
    return gzip.compress(payload, mtime=0) if compressed else payload


def _derived() -> tuple[list[dict[str, Any]], bytes, bytes, dict[str, Any]]:
    return AUTHORITY.derive_role_authority(_runinfo_bytes(), _soft_bytes())


def _fake_protocol_provenance() -> dict[str, Any]:
    return {
        "full_file_bytes": 12_345,
        "full_file_sha256": "5" * 64,
        "core_projection_sha256": AUTHORITY.PROTOCOL_CORE_SHA256,
        "canonicalization": AUTHORITY.CANONICALIZATION,
        "binding_status": "BOUND",
    }


def _fake_implementation_evidence() -> dict[str, Any]:
    protocol = _fake_protocol_provenance()
    return {
        "status": "BOUND",
        "binding_mode": AUTHORITY.BINDING_MODE,
        "implementation_commit": "1" * 40,
        "binding_commit": "2" * 40,
        "implementation_script_sha256": "3" * 64,
        "implementation_test_sha256": "4" * 64,
        "protocol_full_sha256": protocol["full_file_sha256"],
        "protocol_core_sha256": protocol["core_projection_sha256"],
        "worktree_and_index_clean": True,
        "implementation_is_direct_parent": True,
        "post_implementation_changed_paths": [AUTHORITY.PROTOCOL_REPO_PATH],
        "implementation_protocol_binding_status": "UNKNOWN_NOT_ASSERTED",
        "head_protocol_binding_status": "BOUND",
    }


def _expected_binding_root() -> Any:
    return AUTHORITY.ExpectedBindingRoot.from_verified_evidence(
        _fake_implementation_evidence()
    )


def _publication_inputs() -> tuple[bytes, bytes, dict[str, Any]]:
    _, mapping_payload, experiment_join_payload, validation = _derived()
    provenance = AUTHORITY.build_provenance_document(
        runinfo_provenance={
            "bytes": AUTHORITY.RUNINFO_BYTES,
            "sha256": AUTHORITY.RUNINFO_SHA256,
        },
        soft_provenance={
            "bytes": AUTHORITY.SOFT_BYTES,
            "sha256": AUTHORITY.SOFT_SHA256,
        },
        mapping_payload=mapping_payload,
        experiment_join_payload=experiment_join_payload,
        validation=validation,
        protocol_provenance=_fake_protocol_provenance(),
        implementation_evidence=_fake_implementation_evidence(),
    )
    return mapping_payload, experiment_join_payload, provenance


def _git_bytes(
    repo: Path,
    arguments: Sequence[str],
    *,
    input_bytes: bytes | None = None,
) -> bytes:
    environment = {
        "LC_ALL": "C",
        "LANG": "C",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_AUTHOR_NAME": "Synthetic Test",
        "GIT_AUTHOR_EMAIL": "synthetic@example.invalid",
        "GIT_COMMITTER_NAME": "Synthetic Test",
        "GIT_COMMITTER_EMAIL": "synthetic@example.invalid",
    }
    completed = subprocess.run(
        [AUTHORITY.GIT_BINARY, "-C", str(repo), *arguments],
        input=input_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=environment,
    )
    assert completed.returncode == 0, completed.stderr.decode("utf-8", errors="replace")
    return completed.stdout


def _load_module(path: Path, name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)
    return module


def _make_two_commit_repo(
    root: Path,
    *,
    wrong_test_hash: bool = False,
    nonancestor: bool = False,
    extra_binding_path: bool = False,
    second_binding_commit: bool = False,
) -> dict[str, Any]:
    repo = root / "repo"
    repo.mkdir(parents=True)
    script_path = repo / AUTHORITY.SCRIPT_REPO_PATH
    test_path = repo / AUTHORITY.TEST_REPO_PATH
    protocol_path = repo / AUTHORITY.PROTOCOL_REPO_PATH
    for destination, source in (
        (script_path, SCRIPT),
        (test_path, TEST),
        (protocol_path, CONFIG),
    ):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    _git_bytes(repo, ("init", "-q"))
    _git_bytes(repo, ("add", "--", AUTHORITY.SCRIPT_REPO_PATH, AUTHORITY.TEST_REPO_PATH, AUTHORITY.PROTOCOL_REPO_PATH))
    _git_bytes(repo, ("commit", "-q", "-m", "implementation"))
    implementation_commit = _git_bytes(repo, ("rev-parse", "HEAD")).decode("ascii").strip()

    bound_implementation = implementation_commit
    if nonancestor:
        empty_tree = _git_bytes(
            repo,
            ("hash-object", "-t", "tree", "--stdin"),
            input_bytes=b"",
        ).decode("ascii").strip()
        bound_implementation = _git_bytes(
            repo,
            ("commit-tree", empty_tree),
            input_bytes=b"unrelated\n",
        ).decode("ascii").strip()

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    binding = protocol["implementation_binding"]
    binding["status"] = "BOUND"
    binding["implementation_commit"] = bound_implementation
    binding["implementation_script_sha256"] = hashlib.sha256(script_path.read_bytes()).hexdigest()
    binding["implementation_test_sha256"] = (
        "0" * 64
        if wrong_test_hash
        else hashlib.sha256(test_path.read_bytes()).hexdigest()
    )
    protocol_path.write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    changed = [AUTHORITY.PROTOCOL_REPO_PATH]
    if extra_binding_path:
        extra = repo / "unexpected.txt"
        extra.write_text("not config only\n", encoding="utf-8")
        changed.append("unexpected.txt")
    _git_bytes(repo, ("add", "--", *changed))
    _git_bytes(repo, ("commit", "-q", "-m", "binding"))
    if second_binding_commit:
        protocol_path.write_bytes(protocol_path.read_bytes() + b"\n")
        _git_bytes(repo, ("add", "--", AUTHORITY.PROTOCOL_REPO_PATH))
        _git_bytes(repo, ("commit", "-q", "-m", "second binding"))

    module = _load_module(
        script_path,
        f"synthetic_binding_{hashlib.sha256(str(root).encode()).hexdigest()[:12]}",
    )
    protocol_value, protocol_provenance, protocol_payload = module.load_contract(protocol_path)
    return {
        "repo": repo,
        "module": module,
        "protocol_path": protocol_path,
        "protocol": protocol_value,
        "protocol_provenance": protocol_provenance,
        "protocol_payload": protocol_payload,
        "implementation_commit": implementation_commit,
    }


def _verify_case(case: Mapping[str, Any]) -> dict[str, Any]:
    module = case["module"]
    return module.verify_implementation_binding(
        protocol_path=case["protocol_path"],
        protocol=case["protocol"],
        protocol_payload=case["protocol_payload"],
        protocol_provenance=case["protocol_provenance"],
    )


def test_unknown_protocol_core_and_two_exact_authority_tables() -> None:
    protocol, provenance, _ = AUTHORITY.load_contract(CONFIG)
    rows, mapping_payload, join_payload, validation = _derived()

    assert provenance["binding_status"] == "UNKNOWN_NOT_ASSERTED"
    assert protocol["implementation_binding"] == AUTHORITY._expected_unknown_binding()
    assert provenance["core_projection_sha256"] == AUTHORITY.PROTOCOL_CORE_SHA256
    assert rows == [dict(row) for row in AUTHORITY.EXPECTED_ROWS]
    assert len(mapping_payload) == 1_200
    assert hashlib.sha256(mapping_payload).hexdigest() == AUTHORITY.MAPPING_SHA256
    assert len(join_payload) == 1_509
    assert hashlib.sha256(join_payload).hexdigest() == AUTHORITY.EXPERIMENT_JOIN_SHA256
    assert join_payload.startswith(
        b"run_accession\tgeo_sample_accession\tbiosample_accession\texperiment_accession\tmeasurement_family\treplicate\n"
    )
    assert b"\r" not in mapping_payload + join_payload
    assert b"80S_RNA" not in mapping_payload + join_payload
    assert validation == AUTHORITY.EXPECTED_VALIDATION


def test_runinfo_and_soft_must_independently_match_compiled_srx_authority() -> None:
    first = AUTHORITY.EXPECTED_JOIN_ROWS[0]
    with pytest.raises(AUTHORITY.MetadataError, match="RunInfo Experiment differs"):
        AUTHORITY.derive_role_authority(
            _runinfo_bytes(
                mutations={first["run_accession"]: {"Experiment": "SRX99999999"}}
            ),
            _soft_bytes(),
        )
    with pytest.raises(AUTHORITY.MetadataError, match="SOFT SRA relation differs"):
        AUTHORITY.derive_role_authority(
            _runinfo_bytes(),
            _soft_bytes(
                experiment_overrides={first["geo_sample_accession"]: "SRX99999999"}
            ),
        )


def test_role_permutation_80s_alias_missing_and_nonunique_fail_closed() -> None:
    high_one = next(
        row for row in AUTHORITY.EXPECTED_JOIN_ROWS
        if row["measurement_family"] == "High_Poly" and row["replicate"] == 1
    )
    high_two = next(
        row for row in AUTHORITY.EXPECTED_JOIN_ROWS
        if row["measurement_family"] == "High_Poly" and row["replicate"] == 2
    )
    with pytest.raises(AUTHORITY.JoinError, match="permuted"):
        AUTHORITY.derive_role_authority(
            _runinfo_bytes(),
            _soft_bytes(
                title_overrides={
                    high_one["geo_sample_accession"]: "High_Poly_rep2",
                    high_two["geo_sample_accession"]: "High_Poly_rep1",
                }
            ),
        )
    pdna_one = next(
        row for row in AUTHORITY.EXPECTED_JOIN_ROWS
        if row["measurement_family"] == "pDNA" and row["replicate"] == 1
    )
    with pytest.raises(AUTHORITY.MetadataError, match="forbidden 80S_RNA"):
        AUTHORITY.derive_role_authority(
            _runinfo_bytes(),
            _soft_bytes(
                title_overrides={pdna_one["geo_sample_accession"]: "80S_RNA_rep1"}
            ),
        )
    first_run = AUTHORITY.EXPECTED_JOIN_ROWS[0]["run_accession"]
    with pytest.raises(AUTHORITY.MetadataError, match="exactly 24"):
        AUTHORITY.derive_role_authority(
            _runinfo_bytes(omit_run=first_run),
            _soft_bytes(),
        )
    first = next(row for row in AUTHORITY.EXPECTED_JOIN_ROWS if row["run_accession"] == first_run)
    second = AUTHORITY.EXPECTED_JOIN_ROWS[1]
    with pytest.raises(AUTHORITY.MetadataError, match="Experiment differs|not unique"):
        AUTHORITY.derive_role_authority(
            _runinfo_bytes(
                mutations={
                    second["run_accession"]: {
                        "Experiment": first["experiment_accession"]
                    }
                }
            ),
            _soft_bytes(),
        )


def test_unknown_binding_fails_before_git_official_source_or_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_read = AUTHORITY._read_regular_snapshot
    read_labels: list[str] = []

    def guarded_read(*args: Any, **kwargs: Any) -> Any:
        read_labels.append(kwargs["label"])
        if kwargs["label"] != "role-authority protocol":
            raise AssertionError("UNKNOWN binding must not read official sources")
        return original_read(*args, **kwargs)

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("UNKNOWN binding must not reach Git or publication")

    monkeypatch.setattr(AUTHORITY, "_read_regular_snapshot", guarded_read)
    monkeypatch.setattr(AUTHORITY, "verify_implementation_binding", forbidden)
    monkeypatch.setattr(AUTHORITY, "publish_authority", forbidden)
    output = tmp_path / "must-not-exist"
    with pytest.raises(AUTHORITY.ImplementationBindingUnknown):
        AUTHORITY.build_role_authority(
            protocol_path=CONFIG,
            runinfo_path=tmp_path / "not-read-runinfo.csv",
            geo_soft_path=tmp_path / "not-read.soft.gz",
            output_directory=output,
        )
    assert read_labels == ["role-authority protocol"]
    assert not output.exists()


@pytest.mark.parametrize("forbidden_argument", ["protocol", "runinfo", "geo_soft", "output"])
def test_all_four_paths_reject_forbidden_scope_before_any_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    forbidden_argument: str,
) -> None:
    values = {
        "protocol": tmp_path / "protocol.json",
        "runinfo": tmp_path / "runinfo.csv",
        "geo_soft": tmp_path / "soft.gz",
        "output": tmp_path / "output",
    }
    values[forbidden_argument] = tmp_path / "restricted" / forbidden_argument

    def forbidden_open(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("path scope must fail before os.open")

    monkeypatch.setattr(AUTHORITY.os, "open", forbidden_open)
    with pytest.raises(AUTHORITY.ScopeViolation, match="before read"):
        AUTHORITY._paths_before_read(
            protocol_path=values["protocol"],
            runinfo_path=values["runinfo"],
            geo_soft_path=values["geo_soft"],
            output_directory=values["output"],
        )


def test_same_fd_source_mutation_is_detected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "snapshot.csv"
    payload = b"A" * 4096
    mutated_payload = b"B" * len(payload)
    source.write_bytes(payload)
    original_read = AUTHORITY.os.read
    captured_nonempty_reads: list[bytes] = []

    def counted_read(descriptor: int, maximum_bytes: int) -> bytes:
        block = original_read(descriptor, maximum_bytes)
        if block:
            captured_nonempty_reads.append(block)
        return block

    def mutate(path: Path) -> None:
        path.write_bytes(mutated_payload)

    monkeypatch.setattr(AUTHORITY.os, "read", counted_read)
    with pytest.raises(AUTHORITY.SourceError, match="changed during same-descriptor"):
        AUTHORITY._read_regular_snapshot(
            source,
            label="synthetic mutable source",
            maximum_bytes=len(payload),
            expected_bytes=len(payload),
            expected_sha256=hashlib.sha256(payload).hexdigest(),
            _after_capture=mutate,
        )
    assert captured_nonempty_reads == [payload, mutated_payload]


@pytest.mark.parametrize(
    ("label", "expected_bytes", "fill"),
    [
        ("frozen RunInfo source", AUTHORITY.RUNINFO_BYTES, b"R"),
        ("frozen GEO SOFT source", AUTHORITY.SOFT_BYTES, b"S"),
    ],
)
def test_correct_size_wrong_sha_source_fingerprints_fail_closed(
    tmp_path: Path,
    label: str,
    expected_bytes: int,
    fill: bytes,
) -> None:
    source = tmp_path / f"{label.replace(' ', '-')}.bin"
    payload = fill * expected_bytes
    source.write_bytes(payload)
    assert hashlib.sha256(payload).hexdigest() != "0" * 64
    with pytest.raises(AUTHORITY.SourceError, match="hash drifted"):
        AUTHORITY._read_regular_snapshot(
            source,
            label=label,
            maximum_bytes=expected_bytes,
            expected_bytes=expected_bytes,
            expected_sha256="0" * 64,
        )


def test_build_wires_each_contract_hash_and_size_into_its_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = json.loads(CONFIG.read_text(encoding="utf-8"))
    binding = protocol["implementation_binding"]
    evidence = _fake_implementation_evidence()
    binding["status"] = "BOUND"
    binding["implementation_commit"] = evidence["implementation_commit"]
    binding["implementation_script_sha256"] = evidence["implementation_script_sha256"]
    binding["implementation_test_sha256"] = evidence["implementation_test_sha256"]
    protocol_provenance = _fake_protocol_provenance()
    protocol_provenance["binding_status"] = "BOUND"
    monkeypatch.setattr(
        AUTHORITY,
        "load_contract",
        lambda path: (protocol, protocol_provenance, b"synthetic-bound-protocol"),
    )
    monkeypatch.setattr(
        AUTHORITY,
        "verify_implementation_binding",
        lambda **kwargs: evidence,
    )
    snapshot_calls: list[dict[str, Any]] = []

    def snapshot(path: Path, **kwargs: Any) -> Any:
        snapshot_calls.append(dict(kwargs))
        if kwargs["label"] == "frozen RunInfo source":
            return b"synthetic-runinfo", {
                "bytes": AUTHORITY.RUNINFO_BYTES,
                "sha256": AUTHORITY.RUNINFO_SHA256,
            }
        if kwargs["label"] == "frozen GEO SOFT source":
            return b"synthetic-soft", {
                "bytes": AUTHORITY.SOFT_BYTES,
                "sha256": AUTHORITY.SOFT_SHA256,
            }
        raise AssertionError("unexpected snapshot label")

    monkeypatch.setattr(AUTHORITY, "_read_regular_snapshot", snapshot)
    _, mapping, experiment_join, validation = _derived()
    monkeypatch.setattr(
        AUTHORITY,
        "derive_role_authority",
        lambda runinfo, soft: (
            [dict(row) for row in AUTHORITY.EXPECTED_ROWS],
            mapping,
            experiment_join,
            validation,
        ),
    )
    published: dict[str, Any] = {}

    def publish(output: Path, **kwargs: Any) -> dict[str, Any]:
        published.update(kwargs)
        return {"committed": True, "accepted": True}

    monkeypatch.setattr(AUTHORITY, "publish_authority", publish)
    result = AUTHORITY.build_role_authority(
        protocol_path=tmp_path / "protocol.json",
        runinfo_path=tmp_path / "synthetic-runinfo.csv",
        geo_soft_path=tmp_path / "synthetic-soft.gz",
        output_directory=tmp_path / "not-materialized-output",
    )
    assert result == {"committed": True, "accepted": True}
    assert snapshot_calls == [
        {
            "label": "frozen RunInfo source",
            "maximum_bytes": AUTHORITY.RUNINFO_BYTES,
            "expected_bytes": protocol["sources"]["runinfo"]["expected_bytes"],
            "expected_sha256": protocol["sources"]["runinfo"]["expected_sha256"],
        },
        {
            "label": "frozen GEO SOFT source",
            "maximum_bytes": AUTHORITY.SOFT_BYTES,
            "expected_bytes": protocol["sources"]["geo_soft"]["expected_bytes"],
            "expected_sha256": protocol["sources"]["geo_soft"]["expected_sha256"],
        },
    ]
    assert published["expected_binding_root"].as_dict() == _expected_binding_root().as_dict()


def test_gzip_bomb_concatenation_and_duplicate_soft_fields_fail_closed() -> None:
    bomb = gzip.compress(b"A" * (AUTHORITY.MAX_SOFT_DECOMPRESSED_BYTES + 1), mtime=0)
    with pytest.raises(AUTHORITY.MetadataError, match="byte bound"):
        AUTHORITY._decode_soft(bomb)
    concatenated = gzip.compress(b"^SERIES = GSE200302\n", mtime=0) + gzip.compress(
        b"extra\n", mtime=0
    )
    with pytest.raises(AUTHORITY.MetadataError, match="one bounded complete member"):
        AUTHORITY._decode_soft(concatenated)
    first = AUTHORITY.EXPECTED_JOIN_ROWS[0]
    duplicate_title = (
        f"!Sample_title = {first['measurement_family']}_rep{first['replicate']}"
    )
    with pytest.raises(AUTHORITY.MetadataError, match="not exactly singular"):
        AUTHORITY.derive_role_authority(
            _runinfo_bytes(),
            _soft_bytes(
                duplicate_field_by_gsm={first["geo_sample_accession"]: duplicate_title}
            ),
        )


def test_two_commit_git_binding_positive_and_protocol_hash_evidence(tmp_path: Path) -> None:
    case = _make_two_commit_repo(tmp_path / "positive")
    evidence = _verify_case(case)
    assert evidence["implementation_commit"] == case["implementation_commit"]
    assert evidence["binding_commit"] != evidence["implementation_commit"]
    assert evidence["protocol_full_sha256"] == case["protocol_provenance"]["full_file_sha256"]
    assert evidence["protocol_core_sha256"] == AUTHORITY.PROTOCOL_CORE_SHA256
    assert evidence["worktree_and_index_clean"] is True
    assert evidence["implementation_is_direct_parent"] is True
    assert evidence["post_implementation_changed_paths"] == [AUTHORITY.PROTOCOL_REPO_PATH]


def test_two_commit_git_binding_rejects_nonancestor_blob_dirty_and_non_config_delta(
    tmp_path: Path,
) -> None:
    nonancestor = _make_two_commit_repo(tmp_path / "nonancestor", nonancestor=True)
    with pytest.raises(nonancestor["module"].ImplementationBindingError, match="not an ancestor"):
        _verify_case(nonancestor)

    wrong_blob = _make_two_commit_repo(tmp_path / "wrong-blob", wrong_test_hash=True)
    with pytest.raises(wrong_blob["module"].ImplementationBindingError, match="test hash"):
        _verify_case(wrong_blob)

    dirty = _make_two_commit_repo(tmp_path / "dirty")
    (dirty["repo"] / "untracked.txt").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(dirty["module"].ImplementationBindingError, match="clean worktree"):
        _verify_case(dirty)

    extra = _make_two_commit_repo(tmp_path / "extra", extra_binding_path=True)
    with pytest.raises(extra["module"].ImplementationBindingError, match="config-only"):
        _verify_case(extra)

    two_bindings = _make_two_commit_repo(
        tmp_path / "two-bindings",
        second_binding_commit=True,
    )
    with pytest.raises(two_bindings["module"].ImplementationBindingError, match="exactly one commit"):
        _verify_case(two_bindings)


def test_producer_marker_last_consumer_accepts_and_no_overwrite(tmp_path: Path) -> None:
    mapping, experiment_join, provenance = _publication_inputs()
    output = tmp_path / "authority"
    result = AUTHORITY.publish_authority(
        output,
        mapping_payload=mapping,
        experiment_join_payload=experiment_join,
        provenance_document=provenance,
        expected_binding_root=_expected_binding_root(),
    )
    assert result["publication_state"] == AUTHORITY.COMMITTED_AND_ACCEPTED
    assert result["committed"] is True
    assert result["accepted"] is True
    assert result["write_trace"][-1] == AUTHORITY.MARKER_FILENAME
    assert result["durability_warning"] is False
    assert set(path.name for path in output.iterdir()) == {
        AUTHORITY.MAPPING_FILENAME,
        AUTHORITY.EXPERIMENT_JOIN_FILENAME,
        AUTHORITY.PROVENANCE_FILENAME,
        AUTHORITY.CHECKSUMS_FILENAME,
        AUTHORITY.MARKER_FILENAME,
    }
    accepted = AUTHORITY.validate_published_authority(
        output,
        expected_binding_root=_expected_binding_root(),
    )
    assert accepted["publication_state"] == AUTHORITY.COMMITTED_AND_ACCEPTED
    assert accepted["accepted"] is True
    frozen = {path.name: path.read_bytes() for path in output.iterdir()}
    with pytest.raises(AUTHORITY.OutputExistsError):
        AUTHORITY.publish_authority(
            output,
            mapping_payload=mapping,
            experiment_join_payload=experiment_join,
            provenance_document=provenance,
            expected_binding_root=_expected_binding_root(),
        )
    assert {path.name: path.read_bytes() for path in output.iterdir()} == frozen


@pytest.mark.parametrize(
    "malformation",
    ["absent", "missing", "unknown", "extra", "type-drift"],
)
def test_consumer_requires_closed_external_binding_root(
    tmp_path: Path,
    malformation: str,
) -> None:
    mapping, experiment_join, provenance = _publication_inputs()
    output = tmp_path / "authority"
    root = _expected_binding_root()
    AUTHORITY.publish_authority(
        output,
        mapping_payload=mapping,
        experiment_join_payload=experiment_join,
        provenance_document=provenance,
        expected_binding_root=root,
    )
    malformed: Any = root.as_dict()
    if malformation == "absent":
        malformed = None
    elif malformation == "missing":
        malformed.pop("binding_commit")
    elif malformation == "unknown":
        malformed["binding_commit"] = "UNKNOWN_NOT_ASSERTED"
    elif malformation == "extra":
        malformed["unexpected"] = "closed-roots-reject-extra-fields"
    else:
        malformed["implementation_commit"] = 1
    with pytest.raises(AUTHORITY.ImplementationBindingError):
        AUTHORITY.validate_published_authority(
            output,
            expected_binding_root=malformed,
        )


def test_consumer_rejects_coherent_bundle_forged_under_another_binding_root(
    tmp_path: Path,
) -> None:
    _, mapping, experiment_join, validation = _derived()
    forged_protocol = _fake_protocol_provenance()
    forged_protocol["full_file_sha256"] = "6" * 64
    forged_evidence = _fake_implementation_evidence()
    forged_evidence.update(
        {
            "implementation_commit": "a" * 40,
            "binding_commit": "b" * 40,
            "implementation_script_sha256": "c" * 64,
            "implementation_test_sha256": "d" * 64,
            "protocol_full_sha256": forged_protocol["full_file_sha256"],
        }
    )
    forged_root = AUTHORITY.ExpectedBindingRoot.from_verified_evidence(
        forged_evidence
    )
    forged_provenance = AUTHORITY.build_provenance_document(
        runinfo_provenance={
            "bytes": AUTHORITY.RUNINFO_BYTES,
            "sha256": AUTHORITY.RUNINFO_SHA256,
        },
        soft_provenance={
            "bytes": AUTHORITY.SOFT_BYTES,
            "sha256": AUTHORITY.SOFT_SHA256,
        },
        mapping_payload=mapping,
        experiment_join_payload=experiment_join,
        validation=validation,
        protocol_provenance=forged_protocol,
        implementation_evidence=forged_evidence,
    )
    output = tmp_path / "coherent-forgery"
    result = AUTHORITY.publish_authority(
        output,
        mapping_payload=mapping,
        experiment_join_payload=experiment_join,
        provenance_document=forged_provenance,
        expected_binding_root=forged_root,
    )
    assert result["publication_state"] == AUTHORITY.COMMITTED_AND_ACCEPTED
    with pytest.raises(AUTHORITY.CommittedNotAcceptedError) as raised:
        AUTHORITY.validate_published_authority(
            output,
            expected_binding_root=_expected_binding_root(),
        )
    assert "caller-supplied ExpectedBindingRoot" in str(raised.value.__cause__)


def test_consumer_rechecks_earlier_member_after_later_member_snapshot(
    tmp_path: Path,
) -> None:
    mapping, experiment_join, provenance = _publication_inputs()
    output = tmp_path / "inter-member-toctou"
    AUTHORITY.publish_authority(
        output,
        mapping_payload=mapping,
        experiment_join_payload=experiment_join,
        provenance_document=provenance,
        expected_binding_root=_expected_binding_root(),
    )
    replacement = tmp_path / "replacement-member"
    replacement.write_bytes(mapping)
    mutation_trace: list[tuple[str, int]] = []

    def mutate_first_after_second(name: str, index: int) -> None:
        mutation_trace.append((name, index))
        if index == 1:
            assert name == AUTHORITY.EXPERIMENT_JOIN_FILENAME
            os.replace(replacement, output / AUTHORITY.MAPPING_FILENAME)

    with pytest.raises(AUTHORITY.CommittedNotAcceptedError) as raised:
        AUTHORITY.validate_published_authority(
            output,
            expected_binding_root=_expected_binding_root(),
            _after_member_snapshot=mutate_first_after_second,
        )
    assert mutation_trace[:2] == [
        (AUTHORITY.MAPPING_FILENAME, 0),
        (AUTHORITY.EXPERIMENT_JOIN_FILENAME, 1),
    ]
    assert "identity changed" in str(raised.value.__cause__)


@pytest.mark.parametrize("tamper_target", ["member", "marker"])
def test_post_link_member_or_marker_tamper_is_committed_not_accepted(
    tmp_path: Path,
    tamper_target: str,
) -> None:
    mapping, experiment_join, provenance = _publication_inputs()
    output = tmp_path / tamper_target

    def tamper(path: Path) -> None:
        target = (
            path / AUTHORITY.MAPPING_FILENAME
            if tamper_target == "member"
            else path / AUTHORITY.MARKER_FILENAME
        )
        target.write_bytes(target.read_bytes() + b"tamper")

    with pytest.raises(AUTHORITY.CommittedNotAcceptedError):
        AUTHORITY.publish_authority(
            output,
            mapping_payload=mapping,
            experiment_join_payload=experiment_join,
            provenance_document=provenance,
            expected_binding_root=_expected_binding_root(),
            _post_link_hook=tamper,
        )
    assert (output / AUTHORITY.MARKER_FILENAME).exists()
    with pytest.raises(AUTHORITY.CommittedNotAcceptedError):
        AUTHORITY.validate_published_authority(
            output,
            expected_binding_root=_expected_binding_root(),
        )


def test_post_link_directory_displacement_is_committed_not_accepted(tmp_path: Path) -> None:
    mapping, experiment_join, provenance = _publication_inputs()
    output = tmp_path / "authority"
    displaced = tmp_path / "displaced-authority"

    def displace(path: Path) -> None:
        path.rename(displaced)
        path.mkdir()

    with pytest.raises(AUTHORITY.CommittedNotAcceptedError):
        AUTHORITY.publish_authority(
            output,
            mapping_payload=mapping,
            experiment_join_payload=experiment_join,
            provenance_document=provenance,
            expected_binding_root=_expected_binding_root(),
            _post_link_hook=displace,
        )
    assert (displaced / AUTHORITY.MARKER_FILENAME).exists()
    assert not (output / AUTHORITY.MARKER_FILENAME).exists()


@pytest.mark.parametrize("phase", ["precommit_output_fsync", "marker_stage_fsync", "marker_stage_close"])
def test_precommit_faults_are_partial_not_committed(tmp_path: Path, phase: str) -> None:
    mapping, experiment_join, provenance = _publication_inputs()
    output = tmp_path / phase
    with pytest.raises(AUTHORITY.PartialPublicationError) as raised:
        AUTHORITY.publish_authority(
            output,
            mapping_payload=mapping,
            experiment_join_payload=experiment_join,
            provenance_document=provenance,
            expected_binding_root=_expected_binding_root(),
            _faults={phase: OSError(f"synthetic {phase}")},
        )
    assert raised.value.publication_state == AUTHORITY.PARTIAL_NOT_COMMITTED
    assert output.is_dir()
    assert not (output / AUTHORITY.MARKER_FILENAME).exists()


def test_postcommit_unlink_fsync_and_close_faults_are_visible_warnings(tmp_path: Path) -> None:
    mapping, experiment_join, provenance = _publication_inputs()
    output = tmp_path / "warning-authority"
    phases = {
        "post_link_marker_stage_unlink": OSError("synthetic unlink warning"),
        "post_link_output_fsync": OSError("synthetic output fsync warning"),
        "post_link_parent_fsync": OSError("synthetic parent fsync warning"),
        "postcommit_output_close": OSError("synthetic output close warning"),
        "postcommit_parent_close": OSError("synthetic parent close warning"),
    }
    result = AUTHORITY.publish_authority(
        output,
        mapping_payload=mapping,
        experiment_join_payload=experiment_join,
        provenance_document=provenance,
        expected_binding_root=_expected_binding_root(),
        _faults=phases,
    )
    assert result["publication_state"] == AUTHORITY.PUBLISHED_WITH_DURABILITY_WARNING
    assert result["committed"] is True
    assert result["accepted"] is True
    assert result["durability_warning"] is True
    assert set(result["durability_warning_codes"]) == {
        "MARKER_STAGE_UNLINK_WARNING",
        "POST_LINK_OUTPUT_FSYNC_WARNING",
        "POST_LINK_PARENT_FSYNC_WARNING",
        "POSTCOMMIT_OUTPUT_CLOSE_WARNING",
        "POSTCOMMIT_PARENT_CLOSE_WARNING",
    }
    assert AUTHORITY.validate_published_authority(
        output,
        expected_binding_root=_expected_binding_root(),
    )["accepted"] is True


def test_consumer_rejects_copy_rename_and_duplicate_key_forged_marker(tmp_path: Path) -> None:
    mapping, experiment_join, provenance = _publication_inputs()
    output = tmp_path / "original-authority"
    AUTHORITY.publish_authority(
        output,
        mapping_payload=mapping,
        experiment_join_payload=experiment_join,
        provenance_document=provenance,
        expected_binding_root=_expected_binding_root(),
    )

    copied = tmp_path / "copied-authority"
    shutil.copytree(output, copied)
    with pytest.raises(AUTHORITY.CommittedNotAcceptedError):
        AUTHORITY.validate_published_authority(
            copied,
            expected_binding_root=_expected_binding_root(),
        )

    renamed = tmp_path / "renamed-authority"
    output.rename(renamed)
    with pytest.raises(AUTHORITY.CommittedNotAcceptedError):
        AUTHORITY.validate_published_authority(
            renamed,
            expected_binding_root=_expected_binding_root(),
        )

    forged = tmp_path / "forged-authority"
    shutil.copytree(renamed, forged)
    marker_path = forged / AUTHORITY.MARKER_FILENAME
    marker_text = marker_path.read_text(encoding="ascii")
    marker_path.write_text(
        marker_text.replace("{\n", '{\n  "status": "FORGED_DUPLICATE",\n', 1),
        encoding="ascii",
    )
    with pytest.raises(AUTHORITY.CommittedNotAcceptedError):
        AUTHORITY.validate_published_authority(
            forged,
            expected_binding_root=_expected_binding_root(),
        )
