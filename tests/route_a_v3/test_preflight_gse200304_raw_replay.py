from __future__ import annotations

import copy
import gzip
import hashlib
import inspect
import json
import os
import sys
import types
from pathlib import Path
from typing import Any, Callable, Mapping

import pytest


STAGING = Path(__file__).resolve().parents[2]
SCRIPT = STAGING / "scripts" / "route_a_v3" / "preflight_gse200304_raw_replay.py"
CONFIG = STAGING / "configs" / "route_a_v3_gse200304_raw_replay.json"

# Compile/exec avoids target-tree bytecode and registers the module for dataclasses.
PREFLIGHT = types.ModuleType("preflight_gse200304_raw_replay_test_target")
PREFLIGHT.__file__ = str(SCRIPT)
sys.modules[PREFLIGHT.__name__] = PREFLIGHT
exec(compile(SCRIPT.read_bytes(), str(SCRIPT), "exec"), PREFLIGHT.__dict__)


EXPECTED_RUNS = tuple(
    f"SRR186567{number:02d}"
    for number in (*range(42, 61), *range(66, 71))
)


def _protocol() -> tuple[dict[str, Any], dict[str, Any]]:
    protocol, provenance = PREFLIGHT.load_protocol(CONFIG)
    assert provenance["core_projection_sha256"] == PREFLIGHT.PROTOCOL_CORE_SHA256
    assert provenance["binding_status"] == "UNKNOWN_NOT_ASSERTED"
    return protocol, provenance


def _write_protocol(root: Path, value: Mapping[str, Any] | bytes) -> Path:
    config_directory = root / "configs"
    config_directory.mkdir(parents=True, exist_ok=True)
    path = config_directory / PREFLIGHT.PROTOCOL_BASENAME
    if isinstance(value, bytes):
        payload = value
    else:
        payload = (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    path.write_bytes(payload)
    return path


def _bound_protocol(root: Path, *, script_bytes: bytes | None = None) -> tuple[Path, dict[str, Any]]:
    protocol = json.loads(CONFIG.read_text(encoding="utf-8"))
    if script_bytes is None:
        script_bytes = SCRIPT.read_bytes()
    script_path = root / protocol["implementation_binding"]["script_repo_path"]
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_bytes(script_bytes)
    binding = protocol["implementation_binding"]
    binding["status"] = "BOUND"
    binding["production_implementation_commit"] = "a" * 40
    binding["production_script_sha256"] = hashlib.sha256(script_bytes).hexdigest()
    binding["hard_blocker"] = None
    protocol["hard_unknown_blockers"] = list(PREFLIGHT.NON_BINDING_HARD_BLOCKERS)
    return _write_protocol(root, protocol), protocol


def _sample_sheet() -> dict[str, Any]:
    roles = [
        (family, replicate)
        for family in PREFLIGHT.EXPECTED_SAMPLE_FAMILIES
        for replicate in PREFLIGHT.EXPECTED_REPLICATES
    ]
    return {
        "schema_version": "route_a_v3_gse200302_raw_sample_sheet.v1",
        "rows": [
            {
                "run_accession": run,
                "measurement_family": family,
                "replicate": replicate,
                "mate_1_filename": f"{run}_1.fastq.gz",
                "mate_2_filename": f"{run}_2.fastq.gz",
            }
            for run, (family, replicate) in zip(EXPECTED_RUNS, roles)
        ],
    }


def _count_policy() -> dict[str, Any]:
    return {
        "schema_version": "route_a_v3_gse200302_sam_to_count_policy.v1",
        "policy": {
            "count_increment": 1,
            "count_unit": "PAIRED_FRAGMENT",
            "discordant_pair_handling": "REJECT",
            "duplicate_handling": "REJECT_MARKED_DUPLICATES",
            "excluded_sam_flags": [4, 8, 256, 512, 1024, 2048],
            "identical_reference_tie_handling": "REJECT_AMBIGUOUS",
            "minimum_mapq": 10,
            "multimapping_handling": "REJECT_NONUNIQUE",
            "overlapping_mates_handling": "COUNT_FRAGMENT_ONCE",
            "paired_read_handling": "REQUIRE_CONCORDANT_PAIR",
            "required_sam_flags": [1, 2],
            "secondary_alignment_handling": "REJECT",
            "supplementary_alignment_handling": "REJECT",
            "unmapped_mate_handling": "REJECT_PAIR",
        },
    }


def _success_document() -> dict[str, Any]:
    protocol, provenance = _protocol()
    return PREFLIGHT._build_preflight_document(
        protocol,
        provenance,
        PREFLIGHT.audit_implementation_binding(protocol),
        dict(PREFLIGHT.EXPECTED_ACQUISITION_AUDIT),
        dict(PREFLIGHT.EXPECTED_REFERENCE_AUDIT),
        PREFLIGHT._not_provided_sample_sheet_audit(),
        PREFLIGHT._not_provided_count_policy_audit(),
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert type(value) is dict
    return value


def _acquisition_members() -> list[str]:
    names = [
        "ACQUISITION_BINDING.json",
        "ACQUISITION_STATUS.json",
        "FASTQ_INTEGRITY_MANIFEST.json",
        "SHA256SUMS",
    ]
    for run in EXPECTED_RUNS:
        for mate in (1, 2):
            fastq = f"{run}_{mate}.fastq.gz"
            names.extend((fastq, f"{fastq}.transfer.json"))
    return sorted(names)


def _acquisition_marker(directory: Path) -> dict[str, Any]:
    members = _acquisition_members()
    return {
        "schema_version": "route_a_v3_gse200304_fastq_acquisition.v1",
        "record_type": "GSE200304_FASTQ_ACQUISITION_PUBLICATION_COMMIT",
        "dataset_accession": "GSE200304",
        "bioproject_accession": "PRJNA824033",
        "generated_at": "2026-08-10T00:00:00+00:00",
        "publication_status": "FASTQ_ACQUISITION_COMMITTED",
        "output_directory": str(directory),
        "manifest_sha256": PREFLIGHT.EXPECTED_ACQUISITION_MANIFEST_SHA256,
        "source_terminal_marker_sha256": PREFLIGHT.EXPECTED_ACQUISITION_SOURCE_MARKER_SHA256,
        "implementation_binding": {
            "status": "BOUND",
            "binding_mode": "TWO_COMMIT_NON_SELF_REFERENTIAL",
            "implementation_commit": "1" * 40,
            "implementation_script_sha256": "2" * 64,
            "binding_commit": "3" * 40,
            "protocol_sha256": "4" * 64,
            "worktree_and_index_clean": True,
        },
        "member_set": members,
        "member_sha256": {name: "5" * 64 for name in members},
        "verified_file_count": 48,
        "verified_run_count": 24,
        "verified_total_bytes": 12_738_938_976,
        "repository_md5_verified_count": 48,
        "local_sha256_recorded_count": 48,
        "qualified_study_contribution": 0,
        "training_allowed": False,
        "next_phase_authorized": False,
        "claim_boundary": PREFLIGHT.EXPECTED_ACQUISITION_CLAIM,
    }


def _write_acquisition_marker(directory: Path, marker: Mapping[str, Any]) -> None:
    directory.mkdir(parents=True)
    (directory / PREFLIGHT.PUBLICATION_MARKER_FILENAME).write_text(
        json.dumps(marker, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _source_binding(path: Path, *, expected: str | None = None) -> Any:
    if expected is None:
        expected = hashlib.sha256(path.read_bytes()).hexdigest()
    return PREFLIGHT.HashPinnedLazySource(path, expected, "synthetic_source")


def _dna(index: int) -> str:
    alphabet = "ACGT"
    encoded: list[str] = []
    value = index
    for _ in range(16):
        encoded.append(alphabet[value % 4])
        value //= 4
    return "".join(encoded) + "A" * (250 - len(encoded))


def test_protocol_trust_unknown_to_bound_projection_and_config_only_binding(tmp_path: Path) -> None:
    unknown, unknown_provenance = _protocol()
    bound_path, bound = _bound_protocol(tmp_path / "mirror")
    bound_value, bound_provenance = PREFLIGHT.load_protocol(bound_path)

    unknown_raw = hashlib.sha256(CONFIG.read_bytes()).hexdigest()
    bound_raw = hashlib.sha256(bound_path.read_bytes()).hexdigest()
    assert unknown_raw != bound_raw
    assert unknown_provenance["core_projection_sha256"] == bound_provenance[
        "core_projection_sha256"
    ] == PREFLIGHT.PROTOCOL_CORE_SHA256
    assert PREFLIGHT._canonical_protocol_projection(unknown) == PREFLIGHT._canonical_protocol_projection(
        bound
    )
    assert bound_value["implementation_binding"]["production_script_sha256"] == hashlib.sha256(
        SCRIPT.read_bytes()
    ).hexdigest()
    assert bound_provenance["binding_status"] == "BOUND"
    assert len(unknown["hard_unknown_blockers"]) == 17
    assert len(bound["hard_unknown_blockers"]) == 16


def test_binding_half_states_extra_keys_paths_and_wrong_source_hash_are_rejected(
    tmp_path: Path,
) -> None:
    unknown = json.loads(CONFIG.read_text(encoding="utf-8"))
    invalid_values: list[dict[str, Any]] = []

    half = copy.deepcopy(unknown)
    half["implementation_binding"]["status"] = "BOUND"
    invalid_values.append(half)

    extra = copy.deepcopy(unknown)
    extra["implementation_binding"]["unexpected"] = False
    invalid_values.append(extra)

    path_drift = copy.deepcopy(unknown)
    path_drift["implementation_binding"]["script_repo_path"] = "scripts/other.py"
    invalid_values.append(path_drift)

    reordered = copy.deepcopy(unknown)
    reordered["hard_unknown_blockers"] = list(reversed(reordered["hard_unknown_blockers"]))
    invalid_values.append(reordered)

    for invalid in invalid_values:
        with pytest.raises(PREFLIGHT.ProtocolError):
            PREFLIGHT._validate_protocol(invalid)

    bound_path, bound = _bound_protocol(tmp_path / "wrong_hash")
    bound["implementation_binding"]["production_script_sha256"] = "f" * 64
    bound_path.write_text(json.dumps(bound, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(PREFLIGHT.ProtocolError, match="differs from source bytes"):
        PREFLIGHT.load_protocol(bound_path)

    drift_path, drift_bound = _bound_protocol(tmp_path / "source_drift")
    implementation_path = drift_path.parent.parent / drift_bound["implementation_binding"][
        "script_repo_path"
    ]
    implementation_path.write_bytes(implementation_path.read_bytes() + b"\n")
    with pytest.raises(PREFLIGHT.ProtocolError, match="differs from source bytes"):
        PREFLIGHT.load_protocol(drift_path)


@pytest.mark.parametrize(
    "payload",
    [
        b'{"x":1,"x":2}\n',
        b'{"x":NaN}\n',
        b'{"x":Infinity}\n',
        b'{"x":-Infinity}\n',
        b'{"x":1e400}\n',
    ],
)
def test_protocol_json_rejects_duplicates_and_nonfinite_numbers(
    tmp_path: Path, payload: bytes
) -> None:
    path = _write_protocol(tmp_path / hashlib.sha256(payload).hexdigest()[:8], payload)
    with pytest.raises(PREFLIGHT.ProtocolError, match="strict finite duplicate-free"):
        PREFLIGHT.load_protocol(path)


def test_protocol_json_excessive_nesting_is_a_controlled_error(tmp_path: Path) -> None:
    payload = ('{"x":' + "[" * 1_100 + "0" + "]" * 1_100 + "}\n").encode("ascii")
    path = _write_protocol(tmp_path / "deep", payload)
    with pytest.raises(PREFLIGHT.ProtocolError):
        PREFLIGHT.load_protocol(path)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["denominator_discrepancy"].__setitem__("paper_reported", None),
        lambda value: value["denominator_discrepancy"]["paper_reported"].pop("attrition_count"),
        lambda value: value["denominator_discrepancy"]["paper_reported"].__setitem__(
            "attrition_count", 120.0
        ),
        lambda value: value["sample_sheet_contract"].__setitem__("unexpected", False),
    ],
)
def test_nested_protocol_drift_is_a_controlled_protocol_error(
    tmp_path: Path, mutation: Callable[[dict[str, Any]], Any]
) -> None:
    value = json.loads(CONFIG.read_text(encoding="utf-8"))
    mutation(value)
    path = _write_protocol(tmp_path / hashlib.sha256(repr(mutation).encode()).hexdigest()[:8], value)
    with pytest.raises(PREFLIGHT.ProtocolError):
        PREFLIGHT.load_protocol(path)


def test_production_unknown_binding_publishes_before_any_adapter_or_candidate_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        calls.append("external")
        raise AssertionError("an adapter or candidate loader ran under UNKNOWN binding")

    monkeypatch.setattr(PREFLIGHT, "_default_source_bindings", forbidden)
    monkeypatch.setattr(PREFLIGHT, "_default_acquisition_audit", forbidden)
    monkeypatch.setattr(PREFLIGHT, "_default_reference_audit", forbidden)
    monkeypatch.setattr(PREFLIGHT.HashPinnedLazySource, "verify", forbidden)

    output = tmp_path / "unknown_failure"
    result = PREFLIGHT.run_preflight(protocol_path=CONFIG, output_directory=output)
    assert calls == []
    assert result["status"] == PREFLIGHT.FAILURE_OUTCOME
    assert result["write_trace"] == [
        PREFLIGHT.FAILURE_FILENAME,
        PREFLIGHT.SHA256SUMS_FILENAME,
        PREFLIGHT.PUBLICATION_MARKER_FILENAME,
    ]
    assert _read_json(output / PREFLIGHT.FAILURE_FILENAME) == PREFLIGHT._failure_payload(
        "PRODUCTION_IMPLEMENTATION_BINDING_UNKNOWN"
    )
    assert PREFLIGHT.validate_published_bundle(output)["accepted"] is True


def test_production_interface_has_no_external_or_observer_callbacks(tmp_path: Path) -> None:
    signature = inspect.signature(PREFLIGHT.run_preflight)
    assert list(signature.parameters) == ["protocol_path", "output_directory"]
    for keyword in (
        "acquisition_adapter",
        "reference_adapter",
        "acquisition_directory",
        "reference_source",
        "sample_sheet_document",
        "count_policy_document",
        "write_observer",
    ):
        with pytest.raises(TypeError):
            PREFLIGHT.run_preflight(
                protocol_path=CONFIG,
                output_directory=tmp_path / keyword,
                **{keyword: object()},
            )
    parser_signature = inspect.signature(PREFLIGHT._parse_args)
    assert list(parser_signature.parameters) == ["argv"]


def test_failure_payload_factory_and_consumer_are_exact_type_strict(tmp_path: Path) -> None:
    valid = PREFLIGHT._failure_payload("PRODUCTION_IMPLEMENTATION_BINDING_UNKNOWN")
    PREFLIGHT._validate_failure_payload(valid)
    with pytest.raises(PREFLIGHT.PublicationError):
        PREFLIGHT._failure_payload("NOT_IN_ENUM")

    mutations: list[Callable[[dict[str, Any]], None]] = [
        lambda value: value.__setitem__("status", PREFLIGHT.BLOCKED_OUTCOME),
        lambda value: value.__setitem__("execution_outcome", PREFLIGHT.BLOCKED_OUTCOME),
        lambda value: value.__setitem__("raw_payload_included", True),
        lambda value: value.__setitem__("success_bundle_published", True),
        lambda value: value.__setitem__("qualified", 0),
        lambda value: value.__setitem__("ordinary_study_contribution", False),
        lambda value: value.__setitem__("unexpected", False),
        lambda value: value.pop("training_allowed"),
    ]
    for index, mutate in enumerate(mutations):
        invalid = copy.deepcopy(valid)
        mutate(invalid)
        with pytest.raises(PREFLIGHT.PublicationError):
            PREFLIGHT._validate_failure_payload(invalid)
        target = tmp_path / f"invalid-{index}"
        with pytest.raises(PREFLIGHT.PublicationError):
            PREFLIGHT._publish_closed_bundle(
                target,
                invalid,
                outcome=PREFLIGHT.FAILURE_OUTCOME,
            )
        assert not target.exists()

    output = tmp_path / "consumer"
    PREFLIGHT._publish_closed_bundle(output, valid, outcome=PREFLIGHT.FAILURE_OUTCOME)
    invalid = copy.deepcopy(valid)
    invalid["qualified"] = 0
    document_bytes = PREFLIGHT._json_bytes(invalid)
    (output / PREFLIGHT.FAILURE_FILENAME).write_bytes(document_bytes)
    (output / PREFLIGHT.SHA256SUMS_FILENAME).write_text(
        f"{hashlib.sha256(document_bytes).hexdigest()}  {PREFLIGHT.FAILURE_FILENAME}\n",
        encoding="ascii",
    )
    with pytest.raises(PREFLIGHT.PublicationError):
        PREFLIGHT.validate_published_bundle(output)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["implementation_binding_audit"].__setitem__("run", "SRR99999999"),
        lambda value: value["acquisition_audit"].__setitem__("run_ids", ["SRR99999999"]),
        lambda value: value["reference_audit"].__setitem__("sequence", "ACGTACGT"),
        lambda value: value["sample_sheet_audit"].__setitem__("extra", False),
        lambda value: value["count_policy_audit"].__setitem__("extra", 0),
        lambda value: value["count_policy_audit"].__setitem__("required_flag_count", 999),
        lambda value: value["confirmed_method"]["xtail_endpoints"][0].__setitem__(
            "identifier", "sample-secret"
        ),
        lambda value: value["denominator_discrepancy"]["paper_reported"].__setitem__(
            "extra", 0
        ),
        lambda value: value["execution_policy"].__setitem__("extra", False),
        lambda value: value["gate_truth"].__setitem__("canonical_record_count", False),
    ],
)
def test_success_document_nested_schemas_reject_extras_identifiers_and_type_confusion(
    mutation: Callable[[dict[str, Any]], None]
) -> None:
    invalid = _success_document()
    mutation(invalid)
    with pytest.raises(PREFLIGHT.PreflightError):
        PREFLIGHT._validate_preflight_document(invalid)


def test_default_acquisition_audit_is_hash_pinned_paired_closed_and_aggregate_only(
    tmp_path: Path,
) -> None:
    source = tmp_path / "adapter_source.py"
    source.write_bytes(b"raise RuntimeError('must never execute')\n")
    acquisition = tmp_path / "acquisition"
    _write_acquisition_marker(acquisition, _acquisition_marker(acquisition))
    audit = PREFLIGHT._default_acquisition_audit(acquisition, _source_binding(source))
    assert audit == PREFLIGHT.EXPECTED_ACQUISITION_AUDIT
    assert audit["target_subseries_accession"] == "GSE200302"
    assert audit["superseries_accession"] == "GSE200304"
    encoded = json.dumps(audit, sort_keys=True)
    assert all(run not in encoded for run in EXPECTED_RUNS)
    assert ".fastq.gz" not in encoded

    protocol, _ = _protocol()
    acquisition_source, reference_source = PREFLIGHT._default_source_bindings(CONFIG, protocol)
    assert acquisition_source.source_path == STAGING / protocol["adapter_bindings"]["acquisition"][
        "repo_path"
    ]
    assert reference_source.source_path == STAGING / protocol["adapter_bindings"]["reference"][
        "repo_path"
    ]
    assert acquisition_source.verify() == PREFLIGHT.EXPECTED_ACQUIRER_SHA256
    assert reference_source.verify() == PREFLIGHT.EXPECTED_QUALIFIER_SHA256


def test_adapter_source_drift_fails_before_acquisition_target_path_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "adapter_source.py"
    source.write_bytes(b"synthetic adapter bytes\n")
    acquisition = tmp_path / "target_must_not_be_read"
    acquisition.mkdir()
    original = PREFLIGHT._read_regular_snapshot
    reads: list[Path] = []

    def tracked(path: Path, **kwargs: Any) -> tuple[bytes, str]:
        reads.append(Path(path))
        return original(path, **kwargs)

    monkeypatch.setattr(PREFLIGHT, "_read_regular_snapshot", tracked)
    loader = _source_binding(source, expected="0" * 64)
    with pytest.raises(PREFLIGHT.AdapterSourceError, match="hash drift"):
        PREFLIGHT._default_acquisition_audit(acquisition, loader)
    assert reads == [source]


def test_acquisition_marker_rejects_missing_extra_order_type_and_wrong_run_pair(
    tmp_path: Path,
) -> None:
    source = tmp_path / "adapter.py"
    source.write_bytes(b"pass\n")

    def missing_mate(marker: dict[str, Any]) -> None:
        for name in (
            f"{EXPECTED_RUNS[0]}_1.fastq.gz",
            f"{EXPECTED_RUNS[0]}_1.fastq.gz.transfer.json",
        ):
            marker["member_set"].remove(name)
            marker["member_sha256"].pop(name)

    def extra_member(marker: dict[str, Any]) -> None:
        marker["member_set"].append("rogue.bin")
        marker["member_set"].sort()
        marker["member_sha256"]["rogue.bin"] = "5" * 64

    def wrong_order(marker: dict[str, Any]) -> None:
        marker["member_set"].reverse()

    def type_confusion(marker: dict[str, Any]) -> None:
        marker["verified_file_count"] = True

    def wrong_run_set(marker: dict[str, Any]) -> None:
        old = EXPECTED_RUNS[0]
        new = "SRR99999999"
        replacements = {}
        for mate in (1, 2):
            old_fastq = f"{old}_{mate}.fastq.gz"
            new_fastq = f"{new}_{mate}.fastq.gz"
            replacements[old_fastq] = new_fastq
            replacements[f"{old_fastq}.transfer.json"] = f"{new_fastq}.transfer.json"
        marker["member_set"] = sorted(replacements.get(name, name) for name in marker["member_set"])
        marker["member_sha256"] = {name: "5" * 64 for name in marker["member_set"]}

    mutators = (missing_mate, extra_member, wrong_order, type_confusion, wrong_run_set)
    for index, mutate in enumerate(mutators):
        acquisition = tmp_path / f"invalid-acquisition-{index}"
        marker = _acquisition_marker(acquisition)
        mutate(marker)
        _write_acquisition_marker(acquisition, marker)
        with pytest.raises(PREFLIGHT.AcquisitionError):
            PREFLIGHT._default_acquisition_audit(acquisition, _source_binding(source))


def test_default_reference_audit_and_aggregate_are_exact_identifier_free(tmp_path: Path) -> None:
    unique = [_dna(index).lower() for index in range(13_832)]
    records = [*unique, unique[0], unique[1], unique[2], unique[3]]
    aggregate = PREFLIGHT.audit_reference_records(records)
    assert aggregate == PREFLIGHT.EXPECTED_REFERENCE_AUDIT
    encoded = json.dumps(aggregate, sort_keys=True)
    assert unique[0] not in encoded

    tsv = "identifier\tFull_Oligo\n" + "".join(
        f"row-{index}\t{sequence}\n" for index, sequence in enumerate(records)
    )
    compressed = gzip.compress(tsv.encode("utf-8"), mtime=0)
    reference_path = tmp_path / "design.tsv.gz"
    reference_path.write_bytes(compressed)
    source = tmp_path / "qualifier_source.py"
    source.write_bytes(b"raise RuntimeError('not executed')\n")
    protocol, _ = _protocol()
    synthetic_protocol = copy.deepcopy(protocol)
    synthetic_protocol["reference_contract"]["source_asset_bytes"] = len(compressed)
    synthetic_protocol["reference_contract"]["source_asset_sha256"] = hashlib.sha256(
        compressed
    ).hexdigest()
    assert PREFLIGHT._default_reference_audit(
        reference_path,
        _source_binding(source),
        synthetic_protocol,
    ) == PREFLIGHT.EXPECTED_REFERENCE_AUDIT


def test_reference_corrupt_deflate_and_oversized_csv_field_are_controlled(
    tmp_path: Path,
) -> None:
    source = tmp_path / "qualifier.py"
    source.write_bytes(b"pass\n")
    protocol, _ = _protocol()
    payloads = (
        b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\x03garbage-garbage",
        gzip.compress(("Full_Oligo\n" + "A" * 200_000 + "\n").encode("ascii"), mtime=0),
    )
    for index, payload in enumerate(payloads):
        reference = tmp_path / f"invalid-{index}.tsv.gz"
        reference.write_bytes(payload)
        synthetic = copy.deepcopy(protocol)
        synthetic["reference_contract"]["source_asset_bytes"] = len(payload)
        synthetic["reference_contract"]["source_asset_sha256"] = hashlib.sha256(payload).hexdigest()
        with pytest.raises(PREFLIGHT.ReferenceAuditError):
            PREFLIGHT._default_reference_audit(reference, _source_binding(source), synthetic)


def test_sample_sheet_and_count_policy_are_closed_type_strict_but_not_authority(
) -> None:
    protocol, _ = _protocol()
    sample = _sample_sheet()
    sample_audit = PREFLIGHT.validate_sample_sheet(sample, protocol)
    assert sample_audit == PREFLIGHT._valid_sample_sheet_audit()

    missing = copy.deepcopy(sample)
    missing["rows"].pop()
    duplicate = copy.deepcopy(sample)
    duplicate["rows"][-1]["measurement_family"] = duplicate["rows"][0][
        "measurement_family"
    ]
    duplicate["rows"][-1]["replicate"] = duplicate["rows"][0]["replicate"]
    confused = copy.deepcopy(sample)
    confused["rows"][0]["replicate"] = True
    for invalid in (missing, duplicate, confused):
        with pytest.raises(PREFLIGHT.SampleSheetError):
            PREFLIGHT.validate_sample_sheet(invalid, protocol)

    policy = _count_policy()
    policy_audit = PREFLIGHT.validate_count_policy(policy, protocol)
    assert policy_audit == PREFLIGHT._valid_count_policy_audit(2, 6)
    invalid_policies = []
    unknown = copy.deepcopy(policy)
    unknown["policy"]["multimapping_handling"] = "UNKNOWN_NOT_ASSERTED"
    invalid_policies.append(unknown)
    mapq_bool = copy.deepcopy(policy)
    mapq_bool["policy"]["minimum_mapq"] = True
    invalid_policies.append(mapq_bool)
    increment_bool = copy.deepcopy(policy)
    increment_bool["policy"]["count_increment"] = True
    invalid_policies.append(increment_bool)
    extra = copy.deepcopy(policy)
    extra["policy"]["unexpected"] = "REJECT"
    invalid_policies.append(extra)
    for invalid in invalid_policies:
        with pytest.raises(PREFLIGHT.CountPolicyError):
            PREFLIGHT.validate_count_policy(invalid, protocol)


def test_author_argv_is_exact_ordered_and_permanently_inert() -> None:
    assert PREFLIGHT.build_bowtie2_build_argv("mpra_lib_full.fa", "mpra_lib_full") == (
        "bowtie2-build",
        "mpra_lib_full.fa",
        "mpra_lib_full",
    )
    assert PREFLIGHT.build_bowtie2_alignment_argv(
        "/bowtie2_index_full/mpra_lib_full",
        "sample_L001_R1_001.fastq.gz",
        "sample_L001_R2_001.fastq.gz",
        "sample.sam",
    ) == (
        "bowtie2",
        "-x",
        "/bowtie2_index_full/mpra_lib_full",
        "-1",
        "sample_L001_R1_001.fastq.gz",
        "-2",
        "sample_L001_R2_001.fastq.gz",
        "-N",
        "0",
        "--no-sq",
        "--no-hd",
        "-S",
        "sample.sam",
    )
    source = SCRIPT.read_text(encoding="utf-8")
    for forbidden in ("import subprocess", "from subprocess", "os.system", "Popen("):
        assert forbidden not in source
    for callable_name in ("execute_alignment", "execute_sam_to_count", "execute_xtail"):
        assert not hasattr(PREFLIGHT, callable_name)
    protocol, _ = _protocol()
    assert all(value is False for value in protocol["execution_policy"].values())


def test_all_seventeen_blockers_xtail_unknowns_and_denominator_conflict_are_frozen() -> None:
    protocol, _ = _protocol()
    assert protocol["hard_unknown_blockers"] == list(PREFLIGHT.EXPECTED_HARD_BLOCKERS)
    assert len(protocol["hard_unknown_blockers"]) == len(set(protocol["hard_unknown_blockers"])) == 17
    required = {
        "EXACT_SRR_SAMPLE_ROLES_UNKNOWN",
        "SAM_TO_COUNT_PAIRED_HANDLING_UNKNOWN",
        "SAM_TO_COUNT_MULTIMAP_POLICY_UNKNOWN",
        "SAM_TO_COUNT_FLAG_POLICY_UNKNOWN",
        "SAM_TO_COUNT_MAPQ_POLICY_UNKNOWN",
        "SAM_TO_COUNT_DUPLICATE_POLICY_UNKNOWN",
        "SAM_TO_COUNT_IDENTICAL_REFERENCE_TIE_POLICY_UNKNOWN",
        "XTAIL_6772_INCLUSION_POLICY_UNKNOWN",
        "EDGER_EXACT_VERSION_UNKNOWN",
        "DESEQ2_EXACT_VERSION_UNKNOWN",
        "XTAIL_DEPENDENCY_LOCK_UNKNOWN",
        "XTAIL_RNG_SEED_AND_STATE_UNKNOWN",
        "PRJNA824033_VS_GSE200304_PRJNA824026_IDENTITY_CONFLICT_UNKNOWN",
        "AUTHOR_CODE_REDISTRIBUTION_PERMISSION_UNKNOWN",
        "RAW_FASTQ_REDISTRIBUTION_PERMISSION_UNKNOWN",
    }
    assert required <= set(protocol["hard_unknown_blockers"])
    assert protocol["denominator_discrepancy"] == PREFLIGHT.EXPECTED_DENOMINATOR_DISCREPANCY
    assert protocol["denominator_discrepancy"]["paper_reported"]["attrition_count"] == 120
    assert protocol["denominator_discrepancy"]["current_mechanical_audit"][
        "attrition_count"
    ] == 113
    method = protocol["author_method_contract"]
    assert (method["r_version"], method["xtail_version"], method["xtail_bins"]) == (
        "4.2.0",
        "1.1.15",
        1000,
    )
    assert method["multiple_testing_adjustment"] == "BH"
    assert len(method["xtail_endpoints"]) == 2


@pytest.mark.parametrize(
    "phase",
    ["after_document_write", "precommit_output_fsync", "precommit_parent_fsync", "before_marker_link"],
)
def test_precommit_failures_never_publish_a_terminal_marker(tmp_path: Path, phase: str) -> None:
    output = tmp_path / phase
    with pytest.raises(PREFLIGHT.PreflightError):
        PREFLIGHT._publish_closed_bundle(
            output,
            PREFLIGHT._failure_payload("PUBLICATION_FAILED"),
            outcome=PREFLIGHT.FAILURE_OUTCOME,
            _faults={phase: RuntimeError(phase)},
        )
    assert not (output / PREFLIGHT.PUBLICATION_MARKER_FILENAME).exists()


def test_partial_marker_stage_is_cleaned_without_visibility(tmp_path: Path) -> None:
    output = tmp_path / "partial-marker"
    with pytest.raises(PREFLIGHT.PreflightError):
        PREFLIGHT._publish_closed_bundle(
            output,
            PREFLIGHT._failure_payload("PUBLICATION_FAILED"),
            outcome=PREFLIGHT.FAILURE_OUTCOME,
            _faults={"marker_stage_partial_write": RuntimeError("partial")},
        )
    assert not (output / PREFLIGHT.PUBLICATION_MARKER_FILENAME).exists()
    assert list(tmp_path.glob(".route-a-v3-marker-*.stage")) == []


@pytest.mark.parametrize(
    ("phase", "warning"),
    [
        ("post_marker_output_fsync", "POST_MARKER_OUTPUT_FSYNC_WARNING"),
        ("post_marker_parent_fsync", "POST_MARKER_PARENT_FSYNC_WARNING"),
        ("postcommit_close_output", "POSTCOMMIT_OUTPUT_CLOSE_WARNING"),
        ("postcommit_close_parent", "POSTCOMMIT_PARENT_CLOSE_WARNING"),
    ],
)
def test_postmarker_fsync_and_close_exceptions_never_downgrade_commit(
    tmp_path: Path, phase: str, warning: str
) -> None:
    output = tmp_path / phase
    result = PREFLIGHT._publish_closed_bundle(
        output,
        PREFLIGHT._failure_payload("PUBLICATION_FAILED"),
        outcome=PREFLIGHT.FAILURE_OUTCOME,
        _faults={phase: RuntimeError(phase)},
    )
    assert result["committed"] is True
    assert warning in result["durability_warning_codes"]
    assert PREFLIGHT.validate_published_bundle(output)["accepted"] is True


def test_link_visible_then_exception_and_stage_cleanup_exception_remain_committed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_link = PREFLIGHT.os.link

    def link_then_raise(*args: Any, **kwargs: Any) -> None:
        original_link(*args, **kwargs)
        raise RuntimeError("simulated ambiguous link return")

    monkeypatch.setattr(PREFLIGHT.os, "link", link_then_raise)
    monkeypatch.setattr(PREFLIGHT, "_require_platform_capabilities", lambda: None)
    output = tmp_path / "link-ambiguity"
    result = PREFLIGHT._publish_closed_bundle(
        output,
        PREFLIGHT._failure_payload("PUBLICATION_FAILED"),
        outcome=PREFLIGHT.FAILURE_OUTCOME,
    )
    assert result["committed"] is True
    assert "MARKER_LINK_RETURN_AMBIGUITY_RECOVERED" in result["durability_warning_codes"]
    assert PREFLIGHT.validate_published_bundle(output)["accepted"] is True

    monkeypatch.undo()
    injected_output = tmp_path / "link-window-fault"
    injected = PREFLIGHT._publish_closed_bundle(
        injected_output,
        PREFLIGHT._failure_payload("PUBLICATION_FAILED"),
        outcome=PREFLIGHT.FAILURE_OUTCOME,
        _faults={"after_marker_link_before_visibility": KeyboardInterrupt()},
    )
    assert injected["committed"] is True
    assert "MARKER_LINK_RETURN_AMBIGUITY_RECOVERED" in injected["durability_warning_codes"]

    outer_window_output = tmp_path / "outer-link-window-fault"
    outer_window = PREFLIGHT._publish_closed_bundle(
        outer_window_output,
        PREFLIGHT._failure_payload("PUBLICATION_FAILED"),
        outcome=PREFLIGHT.FAILURE_OUTCOME,
        _faults={"after_link_resolution_before_visibility": KeyboardInterrupt()},
    )
    assert outer_window["committed"] is True
    assert "MARKER_VISIBILITY_RECOVERED_AFTER_EXCEPTION" in outer_window[
        "durability_warning_codes"
    ]

    original_unlink = PREFLIGHT.os.unlink

    def reject_stage_cleanup(path: Any, *args: Any, **kwargs: Any) -> None:
        if isinstance(path, str) and path.startswith(".route-a-v3-marker-"):
            raise RuntimeError("post-marker cleanup observer failure")
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(PREFLIGHT, "_require_platform_capabilities", lambda: None)
    monkeypatch.setattr(PREFLIGHT.os, "unlink", reject_stage_cleanup)
    output_cleanup = tmp_path / "cleanup-warning"
    cleanup_result = PREFLIGHT._publish_closed_bundle(
        output_cleanup,
        PREFLIGHT._failure_payload("PUBLICATION_FAILED"),
        outcome=PREFLIGHT.FAILURE_OUTCOME,
    )
    assert cleanup_result["committed"] is True
    assert "MARKER_STAGE_CLEANUP_WARNING" in cleanup_result["durability_warning_codes"]
    assert (output_cleanup / PREFLIGHT.PUBLICATION_MARKER_FILENAME).is_file()


def test_marker_is_last_atomic_and_no_acceptance_read_occurs_after_link(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = {"linked": False}
    events: list[str] = []
    original_link = PREFLIGHT.os.link
    original_stat = PREFLIGHT.os.stat
    original_fstat = PREFLIGHT.os.fstat
    original_listdir = PREFLIGHT.os.listdir
    original_read_member = PREFLIGHT._read_member_at
    original_fsync = PREFLIGHT.os.fsync

    def guarded_link(*args: Any, **kwargs: Any) -> None:
        original_link(*args, **kwargs)
        state["linked"] = True
        events.append("link")

    def guarded_stat(*args: Any, **kwargs: Any) -> Any:
        if state["linked"]:
            raise AssertionError("stat after terminal marker visibility")
        return original_stat(*args, **kwargs)

    def guarded_fstat(*args: Any, **kwargs: Any) -> Any:
        if state["linked"]:
            raise AssertionError("fstat after terminal marker visibility")
        return original_fstat(*args, **kwargs)

    def guarded_listdir(*args: Any, **kwargs: Any) -> Any:
        if state["linked"]:
            raise AssertionError("listdir after terminal marker visibility")
        return original_listdir(*args, **kwargs)

    def guarded_read_member(*args: Any, **kwargs: Any) -> bytes:
        if state["linked"]:
            raise AssertionError("member read after terminal marker visibility")
        return original_read_member(*args, **kwargs)

    def tracked_fsync(*args: Any, **kwargs: Any) -> None:
        events.append("fsync-after-link" if state["linked"] else "fsync-before-link")
        original_fsync(*args, **kwargs)

    monkeypatch.setattr(PREFLIGHT, "_require_platform_capabilities", lambda: None)
    monkeypatch.setattr(PREFLIGHT.os, "link", guarded_link)
    monkeypatch.setattr(PREFLIGHT.os, "stat", guarded_stat)
    monkeypatch.setattr(PREFLIGHT.os, "fstat", guarded_fstat)
    monkeypatch.setattr(PREFLIGHT.os, "listdir", guarded_listdir)
    monkeypatch.setattr(PREFLIGHT, "_read_member_at", guarded_read_member)
    monkeypatch.setattr(PREFLIGHT.os, "fsync", tracked_fsync)

    output = tmp_path / "atomic"
    result = PREFLIGHT._publish_closed_bundle(
        output,
        PREFLIGHT._failure_payload("PUBLICATION_FAILED"),
        outcome=PREFLIGHT.FAILURE_OUTCOME,
    )
    assert result["write_trace"] == [
        PREFLIGHT.FAILURE_FILENAME,
        PREFLIGHT.SHA256SUMS_FILENAME,
        PREFLIGHT.PUBLICATION_MARKER_FILENAME,
    ]
    assert events.count("link") == 1
    assert events.count("fsync-after-link") == 2
    assert result["committed"] is True


def test_postcommit_observer_is_non_authoritative_and_no_overwrite_is_exact(
    tmp_path: Path,
) -> None:
    output = tmp_path / "observer"

    def raising_observer(_: Mapping[str, Any]) -> None:
        raise RuntimeError("observer is non-authoritative")

    result = PREFLIGHT._publish_closed_bundle(
        output,
        _success_document(),
        outcome=PREFLIGHT.BLOCKED_OUTCOME,
        _postcommit_observer=raising_observer,
    )
    assert result["committed"] is True
    assert "POSTCOMMIT_OBSERVER_WARNING" in result["durability_warning_codes"]
    assert PREFLIGHT.validate_published_bundle(output)["status"] == PREFLIGHT.BLOCKED_OUTCOME
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    with pytest.raises(PREFLIGHT.PublicationContention):
        PREFLIGHT._publish_closed_bundle(
            output,
            PREFLIGHT._failure_payload("PUBLICATION_FAILED"),
            outcome=PREFLIGHT.FAILURE_OUTCOME,
        )
    after = {path.name: path.read_bytes() for path in output.iterdir()}
    assert before == after
    assert PREFLIGHT.FAILURE_FILENAME not in before


def test_success_failure_mutual_exclusion_and_path_identifier_sequence_privacy(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sensitive_parts = (
        "SRR99999999",
        "sampleZZQ49382",
        "barcodeZZQ59381",
        "ACGTACGTACGTACGT",
    )
    parent = tmp_path.joinpath(*sensitive_parts)
    parent.mkdir(parents=True)
    failure = parent / "failure-bundle"
    exit_code = PREFLIGHT.main(
        ["--protocol", str(CONFIG), "--output-directory", str(failure)]
    )
    cli_output = capsys.readouterr().out
    assert exit_code == 2
    assert str(failure) not in cli_output

    success = tmp_path / "success-bundle"
    PREFLIGHT._publish_closed_bundle(
        success,
        _success_document(),
        outcome=PREFLIGHT.BLOCKED_OUTCOME,
    )
    assert (failure / PREFLIGHT.FAILURE_FILENAME).is_file()
    assert not (failure / PREFLIGHT.PREFLIGHT_FILENAME).exists()
    assert (success / PREFLIGHT.PREFLIGHT_FILENAME).is_file()
    assert not (success / PREFLIGHT.FAILURE_FILENAME).exists()

    combined = b"\n".join(path.read_bytes() for path in sorted(failure.iterdir()))
    combined += b"\n" + cli_output.encode("utf-8")
    assert str(failure).encode("utf-8") not in combined
    for sensitive in sensitive_parts:
        assert sensitive.encode("ascii") not in combined
    marker = _read_json(failure / PREFLIGHT.PUBLICATION_MARKER_FILENAME)
    assert set(marker) == set(
        PREFLIGHT._marker_payload(
            outcome=PREFLIGHT.FAILURE_OUTCOME,
            bundle_id=marker["bundle_id"],
            bundle_digest=marker["bundle_digest"],
            member_payloads={
                PREFLIGHT.FAILURE_FILENAME: (failure / PREFLIGHT.FAILURE_FILENAME).read_bytes(),
                PREFLIGHT.SHA256SUMS_FILENAME: (failure / PREFLIGHT.SHA256SUMS_FILENAME).read_bytes(),
            },
        )
    )
    assert marker["ordinary_study_contribution"] == 0
    assert marker["a1_study_contribution"] == 0
    assert marker["true_a2_study_contribution"] == 0


def test_symlink_paths_are_fail_closed_and_required_platform_flags_are_explicit(
    tmp_path: Path,
) -> None:
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    protocol_link = real_parent / PREFLIGHT.PROTOCOL_BASENAME
    protocol_link.symlink_to(CONFIG)
    with pytest.raises(PREFLIGHT.ScopeViolation):
        PREFLIGHT.load_protocol(protocol_link)
    with pytest.raises(PREFLIGHT.ScopeViolation):
        PREFLIGHT._publish_closed_bundle(
            linked_parent / "bundle",
            PREFLIGHT._failure_payload("PUBLICATION_FAILED"),
            outcome=PREFLIGHT.FAILURE_OUTCOME,
        )
    source = SCRIPT.read_text(encoding="utf-8")
    assert "getattr(os," not in source
    assert "os.O_NOFOLLOW" in source
    assert "os.O_DIRECTORY" in source
    assert "os.O_NONBLOCK" in source
    PREFLIGHT._require_platform_capabilities()


def test_fifo_leafs_fail_fast_and_descriptor_close_errors_stay_controlled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fifo_root = tmp_path / "fifo-root"
    fifo_root.mkdir()
    protocol_fifo = fifo_root / PREFLIGHT.PROTOCOL_BASENAME
    os.mkfifo(protocol_fifo)
    with pytest.raises(PREFLIGHT.ScopeViolation):
        PREFLIGHT.load_protocol(protocol_fifo)

    bundle = tmp_path / "fifo-bundle"
    bundle.mkdir()
    os.mkfifo(bundle / PREFLIGHT.PUBLICATION_MARKER_FILENAME)
    with pytest.raises(PREFLIGHT.PublicationError):
        PREFLIGHT.validate_published_bundle(bundle)

    original_close = PREFLIGHT.os.close
    original_fstat = PREFLIGHT.os.fstat
    raised = {"regular": False}

    def regular_close_then_error(descriptor: int) -> None:
        info = original_fstat(descriptor)
        original_close(descriptor)
        if PREFLIGHT.stat.S_ISREG(info.st_mode) and not raised["regular"]:
            raised["regular"] = True
            raise OSError("simulated close EIO")

    monkeypatch.setattr(PREFLIGHT.os, "close", regular_close_then_error)
    with pytest.raises(PREFLIGHT.PreflightError):
        PREFLIGHT.load_protocol(CONFIG)
    assert raised["regular"] is True

    monkeypatch.undo()
    original_close = PREFLIGHT.os.close
    original_fstat = PREFLIGHT.os.fstat
    raised_directory = {"value": False}

    def directory_close_then_error(descriptor: int) -> None:
        info = original_fstat(descriptor)
        original_close(descriptor)
        if PREFLIGHT.stat.S_ISDIR(info.st_mode) and not raised_directory["value"]:
            raised_directory["value"] = True
            raise OSError("simulated directory close EIO")

    monkeypatch.setattr(PREFLIGHT.os, "close", directory_close_then_error)
    with pytest.raises(PREFLIGHT.ScopeViolation):
        PREFLIGHT.load_protocol(CONFIG)
    assert raised_directory["value"] is True


def test_committed_config_json_parses_and_hashes_are_reportable() -> None:
    protocol = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert type(protocol) is dict
    assert hashlib.sha256(CONFIG.read_bytes()).hexdigest() == (
        "4fe57314f51203836527bfdfe0c38f04e75df167cfef5d3fa5468597078b3d68"
    )
    assert PREFLIGHT.PROTOCOL_CORE_SHA256 == (
        "381a65d3070eef00bd4b73a8936fd779a999c2a890c221802fdea772b48a24de"
    )
