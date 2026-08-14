from __future__ import annotations

import copy
import csv
import gzip
import hashlib
import importlib.util
import io
import json
import os
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = (
    ROOT
    / "configs"
    / "route_a_v3_gse207584_aggregate_dense_family_qualification_preflight_v1.json"
)
MODULE_PATH = (
    ROOT
    / "scripts"
    / "route_a_v3"
    / "preflight_gse207584_aggregate_dense_family_qualification.py"
)
SPEC = importlib.util.spec_from_file_location("gse207584_dense_preflight", MODULE_PATH)
assert SPEC and SPEC.loader
PREFLIGHT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PREFLIGHT
SPEC.loader.exec_module(PREFLIGHT)

OWN_BINDING_SCALARS = PREFLIGHT.OWN_BINDING_SCALARS
DEC023_GSE207584_REQUIRED_GATE_IDS = (
    "INTENDED_UNIVERSE_MEMBERSHIP_CLOSED",
    "SOURCE_TO_CANDIDATE_SYNONYMOUS_EDIT_REPLAY_CLOSED",
    "DENSE_FAMILY_AND_CONTEXT_CLOSED",
    "ENDPOINT_DIRECTION_SCALE_AND_SEMANTICS_CLOSED",
    "THREE_BIOLOGICAL_REPLICATE_SLOPE_AND_VALID_STANDARD_ERROR_CLOSED",
    "MISSING_CENSORING_AND_COVERAGE_SELECTION_CLOSED",
    "LICENSE_AND_REUSE_RIGHTS_CLOSED",
    "MODEL_INPUT_ROUTE_AND_SCRATCH_EXPOSURE_CLOSED",
    "OUTCOME_BLIND_SOURCE_GROUP_NEAR_DUPLICATE_SPLIT_AND_ZERO_LEAKAGE_CLOSED",
    "POST_DEDUP_INDEPENDENT_EFFECTIVE_N_CLOSED",
    "PREFROZEN_SOURCE_GROUP_POWER_AND_FULL_CI_WIDTH_CLOSED",
)


def _protocol() -> dict[str, object]:
    value = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    PREFLIGHT._validate_protocol(value)
    return value


def _clean_i_protocol() -> dict[str, object]:
    value = PREFLIGHT._normalise_preflight_binding(_protocol())
    PREFLIGHT._validate_protocol(value)
    return value


def _bound_protocol() -> dict[str, object]:
    value = _clean_i_protocol()
    group = value["implementation_binding"]["preflight_group"]
    group.update(
        {
            "status": PREFLIGHT.BOUND,
            "implementation_commit": "6" * 40,
            "implementation_script_sha256": "c" * 64,
            "implementation_test_sha256": "d" * 64,
        }
    )
    PREFLIGHT._validate_protocol(value)
    return value


def _write_protocol(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return path


def _fixture_binding(*args: object) -> dict[str, str]:
    return {
        "status": "TEST_FIXTURE_BOUND_WITHOUT_GIT",
        "authority_commit": PREFLIGHT.AUTHORITY_COMMIT,
        "runtime_i1_commit": PREFLIGHT.RUNTIME_I1_COMMIT,
        "runtime_i2_commit": PREFLIGHT.RUNTIME_I2_COMMIT,
        "runtime_b2_commit": PREFLIGHT.RUNTIME_B2_COMMIT,
        "gse261709_i1_commit": PREFLIGHT.PREDECESSOR_I1_COMMIT,
        "gse261709_i2_commit": PREFLIGHT.PREDECESSOR_I2_COMMIT,
        "gse261709_b2_commit": PREFLIGHT.PREDECESSOR_B2_COMMIT,
        "gse207584_i1_commit": PREFLIGHT.GSE207_I1_COMMIT,
        "gse207584_i2_commit": PREFLIGHT.GSE207_I2_COMMIT,
        "gse207584_b2_commit": PREFLIGHT.GSE207_B2_COMMIT,
        "gse207584_i3_commit": PREFLIGHT.GSE207_I3_COMMIT,
        "gse207584_b3_commit": PREFLIGHT.GSE207_B3_COMMIT,
        "gse207584_i4_commit": "6" * 40,
        "gse207584_b4_commit": "7" * 40,
    }


def _bind_source_mapping(
    protocol: dict[str, object], source_mapping: Path
) -> None:
    contract = protocol["ordinary_public_asset_contract"][
        "authoritative_source_mapping"
    ]
    contract.update(
        {
            "status": PREFLIGHT.BOUND,
            "official_locator": (
                "https://example.org/GSE207584/authoritative_source_mapping.csv.gz"
            ),
            "official_authority_id": "TEST_AUTHORITATIVE_PUBLIC_MAPPING",
            "filename": source_mapping.name,
            "compressed_bytes": source_mapping.stat().st_size,
            "compressed_sha256": hashlib.sha256(
                source_mapping.read_bytes()
            ).hexdigest(),
            "field_dictionary_locator": (
                "https://example.org/GSE207584/mapping-field-dictionary"
            ),
        }
    )
    PREFLIGHT._validate_protocol(protocol)


def _csv_gzip(path: Path, header: tuple[str, ...], rows: list[list[object]]) -> Path:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    path.write_bytes(gzip.compress(buffer.getvalue().encode("utf-8")))
    return path


def _fasta_gzip(path: Path, records: dict[str, str]) -> Path:
    text = "".join(f">{name}\n{sequence}\n" for name, sequence in records.items())
    path.write_bytes(gzip.compress(text.encode("utf-8")))
    return path


SOURCES = {
    "P1": "ATGGCTGAATTTTAA",
    "P2": "ATGAAAGCTTTCTAA",
    "P3": "ATGCCTCAAGGTTAA",
}
CANDIDATES = {
    "P1": (
        "ATGGCCGAATTTTAA",
        "ATGGCAGAATTCTAA",
        "ATGGCGGAGTTTTAA",
    ),
    "P2": (
        "ATGAAGGCTTTCTAA",
        "ATGAAAGCCTTCTAA",
        "ATGAAGGCATTCTAA",
    ),
    "P3": (
        "ATGCCCCAAGGTTAA",
        "ATGCCTCAGGGTTAA",
        "ATGCCACAGGGCTAA",
    ),
}


def _positive_values(offset: float) -> list[float]:
    return [
        12.0 + offset,
        13.0 + offset,
        14.0 + offset,
        9.0 + offset,
        10.0 + offset,
        11.0 + offset,
        6.0 + offset,
        7.0 + offset,
        8.0 + offset,
    ]


def _synthetic_assets(
    tmp_path: Path,
    *,
    include_mapping: bool,
    poison: bool = False,
    add_missing_intended: bool = False,
    conflicting_duplicate_order: str | None = None,
) -> tuple[Path, Path, Path | None]:
    observed_rows: list[list[object]] = []
    fasta: dict[str, str] = {}
    mapping_rows: list[list[object]] = []
    index = 0
    for protein, sequences in CANDIDATES.items():
        for candidate_index, candidate_sequence in enumerate(sequences):
            candidate_id = f"{protein}_C{candidate_index + 1}"
            if poison:
                candidate_id += "_MEMBER_ID_POISON"
            values = _positive_values(index * 0.1)
            duplicate_values = values
            if protein == "P1" and candidate_index == 0:
                if conflicting_duplicate_order not in {
                    None,
                    "ORIGINAL_FIRST",
                    "CONFLICT_FIRST",
                }:
                    raise AssertionError("invalid synthetic duplicate order")
                if conflicting_duplicate_order is not None:
                    conflicting_values = list(values)
                    conflicting_values[-1] += 5.0
                    if conflicting_duplicate_order == "CONFLICT_FIRST":
                        values, duplicate_values = conflicting_values, values
                    else:
                        duplicate_values = conflicting_values
            observed_rows.append(
                [candidate_id, protein, "DESIGN_A_MEMBER_POISON" if poison else "DESIGN_A", *values]
            )
            if candidate_index == 0:
                observed_rows.append(
                    [candidate_id, protein, "DESIGN_B_MEMBER_POISON" if poison else "DESIGN_B", *duplicate_values]
                )
            fasta[candidate_id] = candidate_sequence
            mapping_rows.append(
                [
                    candidate_id,
                    protein,
                    protein,
                    "DANIO_RERIO_EMBRYO_INJECTED_REPORTER",
                    SOURCES[protein],
                    candidate_sequence,
                ]
            )
            index += 1
    if add_missing_intended:
        fasta["MISSING_MEMBER_POISON" if poison else "MISSING_CANDIDATE"] = (
            "ATGGCTGAGTTTTAA"
        )
    observed = _csv_gzip(
        tmp_path / "GSE207584_Zebrafish-library-perfect.csv.gz",
        PREFLIGHT.OBSERVED_HEADER,
        observed_rows,
    )
    reference = _fasta_gzip(
        tmp_path / "GSE207584_reference.fasta.gz",
        fasta,
    )
    source_mapping = None
    if include_mapping:
        if add_missing_intended:
            mapping_rows.append(
                [
                    "MISSING_MEMBER_POISON" if poison else "MISSING_CANDIDATE",
                    "P1",
                    "P1",
                    "DANIO_RERIO_EMBRYO_INJECTED_REPORTER",
                    SOURCES["P1"],
                    "ATGGCTGAGTTTTAA",
                ]
            )
        source_mapping = _csv_gzip(
            tmp_path / "authoritative_source_mapping.csv.gz",
            PREFLIGHT.MAPPING_HEADER,
            mapping_rows,
        )
    return observed, reference, source_mapping


def _execute_fixture(
    tmp_path: Path,
    *,
    include_mapping: bool,
    poison: bool = False,
    add_missing_intended: bool = False,
    bind_mapping: bool = True,
) -> tuple[dict[str, object], Path]:
    observed, reference, source_mapping = _synthetic_assets(
        tmp_path,
        include_mapping=include_mapping,
        poison=poison,
        add_missing_intended=add_missing_intended,
    )
    protocol_value = _bound_protocol()
    if source_mapping is not None and bind_mapping:
        _bind_source_mapping(protocol_value, source_mapping)
    protocol = _write_protocol(tmp_path / "protocol.json", protocol_value)
    output = tmp_path / "out" / PREFLIGHT.REPORT_FILENAME
    report = PREFLIGHT.execute(
        protocol,
        observed,
        reference,
        output,
        source_mapping=source_mapping,
        repo_root=tmp_path,
        binding_auditor=_fixture_binding,
    )
    return report, output


def test_disk_candidate_is_strict_valid_i_or_b_and_normalizes_to_i() -> None:
    protocol = _protocol()
    group = protocol["implementation_binding"]["preflight_group"]
    own_state = [group[field] for field in OWN_BINDING_SCALARS]
    normalized_i = PREFLIGHT._normalise_preflight_binding(protocol)
    PREFLIGHT._validate_protocol(normalized_i)

    if own_state == [PREFLIGHT.UNKNOWN] * 4:
        assert group["status"] == PREFLIGHT.UNKNOWN
        assert normalized_i == protocol
    else:
        assert group["status"] == PREFLIGHT.BOUND
        assert PREFLIGHT._hex_commit(group["implementation_commit"])
        assert group["implementation_script_sha256"] == hashlib.sha256(
            MODULE_PATH.read_bytes()
        ).hexdigest()
        assert group["implementation_test_sha256"] == hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest()
        restored_b = copy.deepcopy(normalized_i)
        for field in OWN_BINDING_SCALARS:
            restored_b["implementation_binding"]["preflight_group"][field] = group[
                field
            ]
        assert restored_b == protocol

    synthetic_i = _clean_i_protocol()
    synthetic_b = copy.deepcopy(synthetic_i)
    synthetic_b["implementation_binding"]["preflight_group"].update(
        {
            "status": PREFLIGHT.BOUND,
            "implementation_commit": "6" * 40,
            "implementation_script_sha256": hashlib.sha256(
                MODULE_PATH.read_bytes()
            ).hexdigest(),
            "implementation_test_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
        }
    )
    PREFLIGHT._validate_protocol(synthetic_b)
    assert PREFLIGHT._normalise_preflight_binding(synthetic_b) == synthetic_i


def test_protocol_freezes_dec023_gse261_and_gse207_through_b3_for_disk_i4_or_b4() -> None:
    protocol = _protocol()
    assert PROTOCOL_PATH.read_bytes() == PREFLIGHT._protocol_json_bytes(protocol)
    authority = protocol["implementation_binding"]["authority_runtime_group"]
    predecessor = protocol["implementation_binding"]["predecessor_preflight_group"]
    preflight = protocol["implementation_binding"]["preflight_group"]
    assert authority["status"] == PREFLIGHT.BOUND
    assert authority["authority_commit"] == PREFLIGHT.AUTHORITY_COMMIT
    assert authority["runtime_i1_commit"] == PREFLIGHT.RUNTIME_I1_COMMIT
    assert authority["runtime_i2_commit"] == PREFLIGHT.RUNTIME_I2_COMMIT
    assert authority["runtime_b2_commit"] == PREFLIGHT.RUNTIME_B2_COMMIT
    assert authority["authority_blob_sha256_by_path"] == PREFLIGHT.AUTHORITY_BLOBS
    assert authority["runtime_i1_blob_sha256_by_path"] == PREFLIGHT.RUNTIME_I1_BLOBS
    assert authority["runtime_i2_blob_sha256_by_path"] == PREFLIGHT.RUNTIME_I2_BLOBS
    assert authority["runtime_b2_blob_sha256_by_path"] == PREFLIGHT.RUNTIME_B2_BLOBS
    assert predecessor["status"] == PREFLIGHT.PREDECESSOR_FROZEN_STATUS
    assert predecessor["implementation_i1_commit"] == PREFLIGHT.PREDECESSOR_I1_COMMIT
    assert predecessor["implementation_i2_commit"] == PREFLIGHT.PREDECESSOR_I2_COMMIT
    assert predecessor["binding_b2_commit"] == PREFLIGHT.PREDECESSOR_B2_COMMIT
    assert predecessor["implementation_i1_blob_sha256_by_path"] == (
        PREFLIGHT.PREDECESSOR_I1_BLOBS
    )
    assert predecessor["implementation_i2_blob_sha256_by_path"] == (
        PREFLIGHT.PREDECESSOR_I2_BLOBS
    )
    assert predecessor["binding_b2_blob_sha256_by_path"] == (
        PREFLIGHT.PREDECESSOR_B2_BLOBS
    )
    assert preflight["predecessor_implementation_i1"] == {
        "status": PREFLIGHT.GSE207_I1_FROZEN_STATUS,
        "commit": PREFLIGHT.GSE207_I1_COMMIT,
        "expected_parent": PREFLIGHT.PREDECESSOR_B2_COMMIT,
        "exact_changed_paths": list(PREFLIGHT.EXACT3),
        "blob_sha256_by_path": PREFLIGHT.GSE207_I1_BLOBS,
    }
    assert preflight["predecessor_implementation_i2"] == {
        "status": PREFLIGHT.GSE207_I2_FROZEN_STATUS,
        "commit": PREFLIGHT.GSE207_I2_COMMIT,
        "expected_parent": PREFLIGHT.GSE207_I1_COMMIT,
        "exact_changed_paths": list(PREFLIGHT.EXACT3),
        "blob_sha256_by_path": PREFLIGHT.GSE207_I2_BLOBS,
    }
    assert preflight["predecessor_binding_b2"] == {
        "status": PREFLIGHT.GSE207_B2_FROZEN_STATUS,
        "commit": PREFLIGHT.GSE207_B2_COMMIT,
        "expected_parent": PREFLIGHT.GSE207_I2_COMMIT,
        "exact_changed_paths": list(PREFLIGHT.GSE207_B2_EXACT_CHANGED_PATHS),
        "blob_sha256_by_path": PREFLIGHT.GSE207_B2_BLOBS,
    }
    assert preflight["predecessor_implementation_i3"] == {
        "status": PREFLIGHT.GSE207_I3_FROZEN_STATUS,
        "commit": PREFLIGHT.GSE207_I3_COMMIT,
        "expected_parent": PREFLIGHT.GSE207_B2_COMMIT,
        "exact_changed_paths": list(PREFLIGHT.EXACT3),
        "blob_sha256_by_path": PREFLIGHT.GSE207_I3_BLOBS,
    }
    assert preflight["predecessor_binding_b3"] == {
        "status": PREFLIGHT.GSE207_B3_FROZEN_STATUS,
        "commit": PREFLIGHT.GSE207_B3_COMMIT,
        "expected_parent": PREFLIGHT.GSE207_I3_COMMIT,
        "exact_changed_paths": list(PREFLIGHT.GSE207_B3_EXACT_CHANGED_PATHS),
        "blob_sha256_by_path": PREFLIGHT.GSE207_B3_BLOBS,
    }
    assert PREFLIGHT.OBSERVED_HEADER[:3] == ("Name", "Protein_id", "Group")
    assert tuple(
        protocol["ordinary_public_asset_contract"]["observed_perfect_csv"][
            "required_header_exactly"
        ]
    ) == PREFLIGHT.OBSERVED_HEADER
    normalized_disk = PREFLIGHT._normalise_preflight_binding(protocol)
    assert [
        normalized_disk["implementation_binding"]["preflight_group"][field]
        for field in OWN_BINDING_SCALARS
    ] == [
        PREFLIGHT.UNKNOWN
    ] * 4
    assert protocol["claim_boundary"]["current_credit_delta"] == {
        "ordinary": 0,
        "A1": 0,
        "true_A2": 0,
    }
    assert all(
        protocol["claim_boundary"][key] is False
        for key in (
            "qualification_allowed_or_changed",
            "true_a2_status_allowed_or_changed",
            "study_credit_allowed_or_changed",
            "canonical_allowed_or_changed",
            "training_allowed_or_changed",
            "gpu_allowed_or_changed",
            "model_selection_allowed_or_changed",
            "next_phase_allowed_or_changed",
        )
    )
    normalised = PREFLIGHT._normalise_preflight_binding(_bound_protocol())
    assert normalised["implementation_binding"]["authority_runtime_group"][
        "status"
    ] == PREFLIGHT.BOUND
    assert normalised["implementation_binding"]["predecessor_preflight_group"][
        "status"
    ] == PREFLIGHT.PREDECESSOR_FROZEN_STATUS
    assert [
        normalised["implementation_binding"]["preflight_group"][field]
        for field in OWN_BINDING_SCALARS
    ] == [PREFLIGHT.UNKNOWN] * 4


def test_gate_ids_equal_dec023_required_fail_closed_tuple_exactly() -> None:
    protocol = _protocol()
    assert PREFLIGHT.GATE_IDS == DEC023_GSE207584_REQUIRED_GATE_IDS
    assert tuple(protocol["gate_ids"]) == DEC023_GSE207584_REQUIRED_GATE_IDS


def test_frozen_predecessor_drift_and_partial_own4_are_rejected() -> None:
    protocol = _clean_i_protocol()
    protocol["implementation_binding"]["authority_runtime_group"][
        "authority_commit"
    ] = "1" * 40
    with pytest.raises(PREFLIGHT.ProtocolError, match="frozen DEC023"):
        PREFLIGHT._validate_protocol(protocol)

    protocol = _clean_i_protocol()
    protocol["implementation_binding"]["authority_runtime_group"][
        "runtime_i2_blob_sha256_by_path"
    ][PREFLIGHT.RUNTIME_PATHS[2]] = "f" * 64
    with pytest.raises(PREFLIGHT.ProtocolError, match="frozen DEC023"):
        PREFLIGHT._validate_protocol(protocol)

    protocol = _clean_i_protocol()
    protocol["implementation_binding"]["predecessor_preflight_group"][
        "implementation_i2_commit"
    ] = "4" * 40
    with pytest.raises(PREFLIGHT.ProtocolError, match="frozen GSE261709"):
        PREFLIGHT._validate_protocol(protocol)

    protocol = _clean_i_protocol()
    protocol["implementation_binding"]["preflight_group"][
        "predecessor_implementation_i1"
    ]["commit"] = "5" * 40
    with pytest.raises(PREFLIGHT.ProtocolError, match="frozen GSE207 I1"):
        PREFLIGHT._validate_protocol(protocol)

    protocol = _clean_i_protocol()
    protocol["implementation_binding"]["preflight_group"][
        "predecessor_implementation_i2"
    ]["commit"] = "5" * 40
    with pytest.raises(PREFLIGHT.ProtocolError, match="frozen GSE207 I2"):
        PREFLIGHT._validate_protocol(protocol)

    protocol = _clean_i_protocol()
    protocol["implementation_binding"]["preflight_group"][
        "predecessor_binding_b2"
    ]["commit"] = "5" * 40
    with pytest.raises(PREFLIGHT.ProtocolError, match="frozen GSE207 B2"):
        PREFLIGHT._validate_protocol(protocol)

    protocol = _clean_i_protocol()
    protocol["implementation_binding"]["preflight_group"][
        "predecessor_implementation_i3"
    ]["commit"] = "5" * 40
    with pytest.raises(PREFLIGHT.ProtocolError, match="frozen GSE207 I3"):
        PREFLIGHT._validate_protocol(protocol)

    protocol = _clean_i_protocol()
    protocol["implementation_binding"]["preflight_group"][
        "predecessor_binding_b3"
    ]["commit"] = "5" * 40
    with pytest.raises(PREFLIGHT.ProtocolError, match="frozen GSE207 B3"):
        PREFLIGHT._validate_protocol(protocol)

    protocol = _clean_i_protocol()
    protocol["implementation_binding"]["preflight_group"][
        "implementation_commit"
    ] = "4" * 40
    with pytest.raises(PREFLIGHT.ProtocolError, match="own4"):
        PREFLIGHT._validate_protocol(protocol)


def test_repository_audit_proves_dec023_gse261_and_gse207_through_i4_b4(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = _bound_protocol()
    binding = protocol["implementation_binding"]
    authority = binding["authority_runtime_group"]
    predecessor = binding["predecessor_preflight_group"]
    preflight = binding["preflight_group"]
    frozen_gse207_i1 = preflight["predecessor_implementation_i1"]
    frozen_gse207_i2 = preflight["predecessor_implementation_i2"]
    frozen_gse207_b2 = preflight["predecessor_binding_b2"]
    frozen_gse207_i3 = preflight["predecessor_implementation_i3"]
    frozen_gse207_b3 = preflight["predecessor_binding_b3"]
    script_payload = b"synthetic GSE207 implementation script\n"
    test_payload = b"synthetic GSE207 focused test\n"
    preflight["implementation_script_sha256"] = hashlib.sha256(
        script_payload
    ).hexdigest()
    preflight["implementation_test_sha256"] = hashlib.sha256(
        test_payload
    ).hexdigest()
    PREFLIGHT._validate_protocol(protocol)

    root = tmp_path / "repo"
    root.mkdir()
    protocol_path = _write_protocol(root / PREFLIGHT.CONFIG_PATH, protocol)
    protocol_bytes = protocol_path.read_bytes()
    script_path = root / PREFLIGHT.SCRIPT_PATH
    test_path = root / PREFLIGHT.TEST_PATH
    script_path.parent.mkdir(parents=True)
    test_path.parent.mkdir(parents=True)
    script_path.write_bytes(script_payload)
    test_path.write_bytes(test_payload)
    preflight_i4 = preflight["implementation_commit"]
    preflight_b4 = "7" * 40

    authority_payloads = {
        path: f"authority:{path}".encode() for path in PREFLIGHT.AUTHORITY_EXACT10
    }
    runtime_payloads = {
        PREFLIGHT.RUNTIME_I1_COMMIT: {
            path: f"runtime-i1:{path}".encode() for path in PREFLIGHT.RUNTIME_PATHS
        },
        PREFLIGHT.RUNTIME_I2_COMMIT: {
            path: f"runtime-i2:{path}".encode() for path in PREFLIGHT.RUNTIME_PATHS
        },
    }
    runtime_payloads[PREFLIGHT.RUNTIME_B2_COMMIT] = {
        PREFLIGHT.RUNTIME_PATHS[0]: b"runtime-b2:config",
        PREFLIGHT.RUNTIME_PATHS[1]: runtime_payloads[PREFLIGHT.RUNTIME_I2_COMMIT][
            PREFLIGHT.RUNTIME_PATHS[1]
        ],
        PREFLIGHT.RUNTIME_PATHS[2]: runtime_payloads[PREFLIGHT.RUNTIME_I2_COMMIT][
            PREFLIGHT.RUNTIME_PATHS[2]
        ],
    }
    predecessor_payloads = {
        PREFLIGHT.PREDECESSOR_I1_COMMIT: {
            path: f"gse261-i1:{path}".encode()
            for path in PREFLIGHT.PREDECESSOR_PATHS
        },
        PREFLIGHT.PREDECESSOR_I2_COMMIT: {
            path: f"gse261-i2:{path}".encode()
            for path in PREFLIGHT.PREDECESSOR_PATHS
        },
    }
    predecessor_payloads[PREFLIGHT.PREDECESSOR_B2_COMMIT] = {
        PREFLIGHT.PREDECESSOR_PATHS[0]: b"gse261-b2:config",
        PREFLIGHT.PREDECESSOR_PATHS[1]: predecessor_payloads[
            PREFLIGHT.PREDECESSOR_I2_COMMIT
        ][PREFLIGHT.PREDECESSOR_PATHS[1]],
        PREFLIGHT.PREDECESSOR_PATHS[2]: predecessor_payloads[
            PREFLIGHT.PREDECESSOR_I2_COMMIT
        ][PREFLIGHT.PREDECESSOR_PATHS[2]],
    }
    gse207_i1_payloads = {
        path: f"gse207-i1:{path}".encode() for path in PREFLIGHT.EXACT3
    }
    gse207_i2_payloads = {
        path: f"gse207-i2:{path}".encode() for path in PREFLIGHT.EXACT3
    }
    gse207_b2_payloads = {
        PREFLIGHT.CONFIG_PATH: b"gse207-b2:config",
        PREFLIGHT.SCRIPT_PATH: gse207_i2_payloads[PREFLIGHT.SCRIPT_PATH],
        PREFLIGHT.TEST_PATH: gse207_i2_payloads[PREFLIGHT.TEST_PATH],
    }
    gse207_i3_payloads = {
        path: f"gse207-i3:{path}".encode() for path in PREFLIGHT.EXACT3
    }
    gse207_b3_payloads = {
        PREFLIGHT.CONFIG_PATH: b"gse207-b3:config",
        PREFLIGHT.SCRIPT_PATH: gse207_i3_payloads[PREFLIGHT.SCRIPT_PATH],
        PREFLIGHT.TEST_PATH: gse207_i3_payloads[PREFLIGHT.TEST_PATH],
    }

    digest_by_payload: dict[bytes, str] = {}
    digest_by_payload.update(
        {
            authority_payloads[path]: digest
            for path, digest in authority["authority_blob_sha256_by_path"].items()
        }
    )
    for commit, digest_field in (
        (PREFLIGHT.RUNTIME_I1_COMMIT, "runtime_i1_blob_sha256_by_path"),
        (PREFLIGHT.RUNTIME_I2_COMMIT, "runtime_i2_blob_sha256_by_path"),
        (PREFLIGHT.RUNTIME_B2_COMMIT, "runtime_b2_blob_sha256_by_path"),
    ):
        digest_by_payload.update(
            {
                runtime_payloads[commit][path]: digest
                for path, digest in authority[digest_field].items()
            }
        )
    digest_by_payload.update(
        {
            gse207_i1_payloads[path]: digest
            for path, digest in frozen_gse207_i1["blob_sha256_by_path"].items()
        }
    )
    digest_by_payload.update(
        {
            gse207_i2_payloads[path]: digest
            for path, digest in frozen_gse207_i2["blob_sha256_by_path"].items()
        }
    )
    digest_by_payload.update(
        {
            gse207_b2_payloads[path]: digest
            for path, digest in frozen_gse207_b2["blob_sha256_by_path"].items()
        }
    )
    digest_by_payload.update(
        {
            gse207_i3_payloads[path]: digest
            for path, digest in frozen_gse207_i3["blob_sha256_by_path"].items()
        }
    )
    digest_by_payload.update(
        {
            gse207_b3_payloads[path]: digest
            for path, digest in frozen_gse207_b3["blob_sha256_by_path"].items()
        }
    )
    for commit, digest_field in (
        (
            PREFLIGHT.PREDECESSOR_I1_COMMIT,
            "implementation_i1_blob_sha256_by_path",
        ),
        (
            PREFLIGHT.PREDECESSOR_I2_COMMIT,
            "implementation_i2_blob_sha256_by_path",
        ),
        (PREFLIGHT.PREDECESSOR_B2_COMMIT, "binding_b2_blob_sha256_by_path"),
    ):
        digest_by_payload.update(
            {
                predecessor_payloads[commit][path]: digest
                for path, digest in predecessor[digest_field].items()
            }
        )

    real_sha256 = hashlib.sha256

    class FrozenDigest:
        def __init__(self, value: str) -> None:
            self.value = value

        def hexdigest(self) -> str:
            return self.value

    def fake_sha256(payload: bytes) -> object:
        expected = digest_by_payload.get(payload)
        return FrozenDigest(expected) if expected is not None else real_sha256(payload)

    parent_by_commit = {
        preflight_b4: preflight_i4,
        preflight_i4: PREFLIGHT.GSE207_B3_COMMIT,
        PREFLIGHT.GSE207_B3_COMMIT: PREFLIGHT.GSE207_I3_COMMIT,
        PREFLIGHT.GSE207_I3_COMMIT: PREFLIGHT.GSE207_B2_COMMIT,
        PREFLIGHT.GSE207_B2_COMMIT: PREFLIGHT.GSE207_I2_COMMIT,
        PREFLIGHT.GSE207_I2_COMMIT: PREFLIGHT.GSE207_I1_COMMIT,
        PREFLIGHT.GSE207_I1_COMMIT: PREFLIGHT.PREDECESSOR_B2_COMMIT,
        PREFLIGHT.PREDECESSOR_B2_COMMIT: PREFLIGHT.PREDECESSOR_I2_COMMIT,
        PREFLIGHT.PREDECESSOR_I2_COMMIT: PREFLIGHT.PREDECESSOR_I1_COMMIT,
        PREFLIGHT.PREDECESSOR_I1_COMMIT: PREFLIGHT.RUNTIME_B2_COMMIT,
        PREFLIGHT.RUNTIME_B2_COMMIT: PREFLIGHT.RUNTIME_I2_COMMIT,
        PREFLIGHT.RUNTIME_I2_COMMIT: PREFLIGHT.RUNTIME_I1_COMMIT,
        PREFLIGHT.RUNTIME_I1_COMMIT: PREFLIGHT.AUTHORITY_COMMIT,
        PREFLIGHT.AUTHORITY_COMMIT: PREFLIGHT.AUTHORITY_EXPECTED_PARENT,
    }
    parent_drift = {"enabled": False}

    def fake_run_git(_root: Path, *args: str) -> str:
        fixed = {
            ("rev-parse", "HEAD"): preflight_b4,
            ("rev-parse", "@{upstream}"): preflight_b4,
            (
                "rev-parse",
                "--verify",
                f"refs/remotes/origin/{PREFLIGHT.PRODUCTION_BRANCH}",
            ): preflight_b4,
            ("rev-parse", "--abbrev-ref", "HEAD"): PREFLIGHT.PRODUCTION_BRANCH,
            ("rev-parse", "--abbrev-ref", "@{upstream}"): (
                f"origin/{PREFLIGHT.PRODUCTION_BRANCH}"
            ),
            ("status", "--porcelain=v1", "--untracked-files=all"): "",
        }
        if args in fixed:
            return fixed[args]
        if len(args) == 2 and args[0] == "rev-parse" and args[1].endswith("^"):
            commit = args[1][:-1]
            if parent_drift["enabled"] and commit == preflight_i4:
                return PREFLIGHT.GSE207_I1_COMMIT
            return parent_by_commit[commit]
        raise AssertionError(args)

    def fake_changed_paths(_root: Path, commit: str) -> tuple[str, ...]:
        if commit == PREFLIGHT.AUTHORITY_COMMIT:
            return tuple(sorted(PREFLIGHT.AUTHORITY_EXACT10))
        if commit in {PREFLIGHT.RUNTIME_I1_COMMIT, PREFLIGHT.RUNTIME_I2_COMMIT}:
            return tuple(sorted(PREFLIGHT.RUNTIME_PATHS))
        if commit == PREFLIGHT.RUNTIME_B2_COMMIT:
            return tuple(sorted(PREFLIGHT.RUNTIME_B2_EXACT_CHANGED_PATHS))
        if commit in {
            PREFLIGHT.PREDECESSOR_I1_COMMIT,
            PREFLIGHT.PREDECESSOR_I2_COMMIT,
        }:
            return tuple(sorted(PREFLIGHT.PREDECESSOR_PATHS))
        if commit == PREFLIGHT.PREDECESSOR_B2_COMMIT:
            return tuple(sorted(PREFLIGHT.PREDECESSOR_B2_EXACT_CHANGED_PATHS))
        if commit == PREFLIGHT.GSE207_I1_COMMIT:
            return tuple(sorted(PREFLIGHT.EXACT3))
        if commit == PREFLIGHT.GSE207_I2_COMMIT:
            return tuple(sorted(PREFLIGHT.EXACT3))
        if commit == PREFLIGHT.GSE207_B2_COMMIT:
            return tuple(sorted(PREFLIGHT.GSE207_B2_EXACT_CHANGED_PATHS))
        if commit == PREFLIGHT.GSE207_I3_COMMIT:
            return tuple(sorted(PREFLIGHT.EXACT3))
        if commit == PREFLIGHT.GSE207_B3_COMMIT:
            return tuple(sorted(PREFLIGHT.GSE207_B3_EXACT_CHANGED_PATHS))
        if commit == preflight_i4:
            return tuple(sorted(PREFLIGHT.EXACT3))
        if commit == preflight_b4:
            return (PREFLIGHT.CONFIG_PATH,)
        raise AssertionError(commit)

    runtime_drift = {"enabled": False}

    def fake_git_blob(_root: Path, commit: str, path: str) -> bytes:
        if commit == PREFLIGHT.AUTHORITY_COMMIT:
            return authority_payloads[path]
        if commit in runtime_payloads:
            if (
                runtime_drift["enabled"]
                and commit == PREFLIGHT.RUNTIME_I2_COMMIT
                and path == PREFLIGHT.RUNTIME_PATHS[2]
            ):
                return b"drifted runtime I2 focused test"
            return runtime_payloads[commit][path]
        if commit in predecessor_payloads:
            return predecessor_payloads[commit][path]
        if commit == PREFLIGHT.GSE207_I1_COMMIT:
            return gse207_i1_payloads[path]
        if commit == PREFLIGHT.GSE207_I2_COMMIT:
            return gse207_i2_payloads[path]
        if commit == PREFLIGHT.GSE207_B2_COMMIT:
            return gse207_b2_payloads[path]
        if commit == PREFLIGHT.GSE207_I3_COMMIT:
            return gse207_i3_payloads[path]
        if commit == PREFLIGHT.GSE207_B3_COMMIT:
            return gse207_b3_payloads[path]
        if commit == preflight_i4:
            return {
                PREFLIGHT.CONFIG_PATH: PREFLIGHT._protocol_json_bytes(
                    PREFLIGHT._normalise_preflight_binding(protocol)
                ),
                PREFLIGHT.SCRIPT_PATH: script_payload,
                PREFLIGHT.TEST_PATH: test_payload,
            }[path]
        if commit == preflight_b4 and path == PREFLIGHT.CONFIG_PATH:
            return protocol_bytes
        raise AssertionError((commit, path))

    monkeypatch.setattr(PREFLIGHT, "__file__", str(script_path))
    monkeypatch.setattr(PREFLIGHT, "_run_git", fake_run_git)
    monkeypatch.setattr(PREFLIGHT, "_changed_paths", fake_changed_paths)
    monkeypatch.setattr(PREFLIGHT, "_git_blob", fake_git_blob)
    monkeypatch.setattr(PREFLIGHT.hashlib, "sha256", fake_sha256)

    assert PREFLIGHT._default_binding_auditor(
        protocol,
        protocol_path,
        protocol_bytes,
        root,
    ) == {
        "status": "BOUND_DEC023_GSE261_AND_GSE207_I1_I2_B2_I3_B3_I4_B4",
        "authority_commit": PREFLIGHT.AUTHORITY_COMMIT,
        "runtime_i1_commit": PREFLIGHT.RUNTIME_I1_COMMIT,
        "runtime_i2_commit": PREFLIGHT.RUNTIME_I2_COMMIT,
        "runtime_b2_commit": PREFLIGHT.RUNTIME_B2_COMMIT,
        "gse261709_i1_commit": PREFLIGHT.PREDECESSOR_I1_COMMIT,
        "gse261709_i2_commit": PREFLIGHT.PREDECESSOR_I2_COMMIT,
        "gse261709_b2_commit": PREFLIGHT.PREDECESSOR_B2_COMMIT,
        "gse207584_i1_commit": PREFLIGHT.GSE207_I1_COMMIT,
        "gse207584_i2_commit": PREFLIGHT.GSE207_I2_COMMIT,
        "gse207584_b2_commit": PREFLIGHT.GSE207_B2_COMMIT,
        "gse207584_i3_commit": PREFLIGHT.GSE207_I3_COMMIT,
        "gse207584_b3_commit": PREFLIGHT.GSE207_B3_COMMIT,
        "gse207584_i4_commit": preflight_i4,
        "gse207584_b4_commit": preflight_b4,
    }

    runtime_drift["enabled"] = True
    with pytest.raises(PREFLIGHT.ProtocolError, match="frozen lifecycle blob"):
        PREFLIGHT._default_binding_auditor(
            protocol,
            protocol_path,
            protocol_bytes,
            root,
        )

    runtime_drift["enabled"] = False
    parent_drift["enabled"] = True
    with pytest.raises(PREFLIGHT.ProtocolError, match="lifecycle parent"):
        PREFLIGHT._default_binding_auditor(
            protocol,
            protocol_path,
            protocol_bytes,
            root,
        )


def test_unknown_lifecycle_stops_before_git_asset_or_output_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = _write_protocol(
        tmp_path / "protocol.json", _clean_i_protocol()
    )
    output = tmp_path / "must-not-exist" / PREFLIGHT.REPORT_FILENAME
    calls = {"git": 0, "aggregate": 0}

    def forbidden_git(*args: object) -> str:
        calls["git"] += 1
        raise AssertionError("Git must not run while a required group is UNKNOWN")

    def forbidden_aggregate(*args: object) -> dict[str, object]:
        calls["aggregate"] += 1
        raise AssertionError("asset reader must not run")

    monkeypatch.setattr(PREFLIGHT, "_run_git", forbidden_git)
    with pytest.raises(PREFLIGHT.BindingNotFrozen, match="not BOUND"):
        PREFLIGHT.execute(
            protocol,
            tmp_path / "missing-observed.csv.gz",
            tmp_path / "missing-reference.fasta.gz",
            output,
            repo_root=tmp_path,
            aggregator=forbidden_aggregate,
        )
    assert calls == {"git": 0, "aggregate": 0}
    assert not output.exists()
    assert not output.parent.exists()


def test_stale_copy_is_rejected_before_git_asset_or_output_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stale_script = tmp_path / "stale-copy" / MODULE_PATH.name
    stale_script.parent.mkdir()
    stale_script.write_bytes(MODULE_PATH.read_bytes())
    stale_spec = importlib.util.spec_from_file_location(
        "gse207584_dense_preflight_stale_copy", stale_script
    )
    assert stale_spec and stale_spec.loader
    stale_preflight = importlib.util.module_from_spec(stale_spec)
    sys.modules[stale_spec.name] = stale_preflight
    stale_spec.loader.exec_module(stale_preflight)

    repo_root = tmp_path / "production-repo"
    protocol_path = _write_protocol(
        repo_root / stale_preflight.CONFIG_PATH,
        _bound_protocol(),
    )
    output = tmp_path / "must-not-exist" / stale_preflight.REPORT_FILENAME
    calls = {"git": 0, "aggregate": 0}

    def forbidden_git(*args: object) -> str:
        calls["git"] += 1
        raise AssertionError("Git must not run for a copied producer")

    def forbidden_aggregate(*args: object) -> dict[str, object]:
        calls["aggregate"] += 1
        raise AssertionError("assets must not be read for a copied producer")

    monkeypatch.setattr(stale_preflight, "_run_git", forbidden_git)
    with pytest.raises(stale_preflight.ProtocolError, match="executing producer"):
        stale_preflight.execute(
            protocol_path,
            tmp_path / "missing-observed.csv.gz",
            tmp_path / "missing-reference.fasta.gz",
            output,
            repo_root=repo_root,
            aggregator=forbidden_aggregate,
        )
    assert calls == {"git": 0, "aggregate": 0}
    assert not output.exists()
    assert not output.parent.exists()


def test_missing_source_mapping_yields_aggregate_blocker_without_fabrication(
    tmp_path: Path,
) -> None:
    report, output = _execute_fixture(tmp_path, include_mapping=False)
    assert output.is_file()
    assert report["status"] == PREFLIGHT.STATUS_BLOCKED
    assert report["gates"][
        "SOURCE_TO_CANDIDATE_SYNONYMOUS_EDIT_REPLAY_CLOSED"
    ]["status"] == PREFLIGHT.UNKNOWN
    assert report["gates"]["POST_DEDUP_INDEPENDENT_EFFECTIVE_N_CLOSED"][
        "status"
    ] == PREFLIGHT.UNKNOWN
    observation = report["aggregate_observation"]
    assert observation["intended_universe"]["reference_fasta_construct_count"] == 9
    assert observation["sequence_and_edit_replay"]["mapping_row_count"] == 0
    assert report["internal_access_attestation"]["ordinary_public_assets_read_count"] == 2


def test_unknown_mapping_authority_ignores_cli_mapping_without_read_or_output_leak(
    tmp_path: Path,
) -> None:
    protocol = _write_protocol(tmp_path / "protocol.json", _bound_protocol())
    observed, reference, source_mapping = _synthetic_assets(
        tmp_path, include_mapping=True
    )
    assert source_mapping is not None
    source_mapping.write_bytes(b"MAPPING_MEMBER_POISON_NOT_A_GZIP")
    output = tmp_path / "out" / PREFLIGHT.REPORT_FILENAME
    report = PREFLIGHT.execute(
        protocol,
        observed,
        reference,
        output,
        source_mapping=source_mapping,
        repo_root=tmp_path,
        binding_auditor=_fixture_binding,
    )
    intended = report["aggregate_observation"]["intended_universe"]
    assert intended["authoritative_mapping_protocol_status"] == PREFLIGHT.UNKNOWN
    assert intended["source_mapping_input_provided"] is True
    assert intended["source_mapping_read_count"] == 0
    assert report["gates"][
        "SOURCE_TO_CANDIDATE_SYNONYMOUS_EDIT_REPLAY_CLOSED"
    ]["status"] == PREFLIGHT.UNKNOWN
    assert report["gates"]["POST_DEDUP_INDEPENDENT_EFFECTIVE_N_CLOSED"][
        "status"
    ] == PREFLIGHT.UNKNOWN
    assert report["internal_access_attestation"]["source_mapping_read_count"] == 0
    assert "MAPPING_MEMBER_POISON" not in output.read_text(encoding="utf-8")


def test_partial_mapping_authority_group_is_rejected() -> None:
    protocol = _clean_i_protocol()
    protocol["ordinary_public_asset_contract"]["authoritative_source_mapping"][
        "official_locator"
    ] = "https://example.org/partial-mapping.csv.gz"
    with pytest.raises(PREFLIGHT.ProtocolError, match="partial source mapping"):
        PREFLIGHT._validate_protocol(protocol)


def test_bound_mapping_identity_mismatch_stops_before_parse_or_output(
    tmp_path: Path,
) -> None:
    observed, reference, source_mapping = _synthetic_assets(
        tmp_path, include_mapping=True
    )
    assert source_mapping is not None
    protocol_value = _bound_protocol()
    _bind_source_mapping(protocol_value, source_mapping)
    protocol = _write_protocol(tmp_path / "protocol.json", protocol_value)
    source_mapping.write_bytes(source_mapping.read_bytes() + b"identity-drift")
    output = tmp_path / "out" / PREFLIGHT.REPORT_FILENAME
    with pytest.raises(PREFLIGHT.AssetError, match="byte count differs"):
        PREFLIGHT.execute(
            protocol,
            observed,
            reference,
            output,
            source_mapping=source_mapping,
            repo_root=tmp_path,
            binding_auditor=_fixture_binding,
        )
    assert not output.exists()


def test_source_components_join_shared_source_identity_without_candidate_overlap() -> None:
    geometry = PREFLIGHT._source_family_components(
        {"F1", "F2", "F3"},
        source_id_families={"SOURCE_A": {"F1", "F2"}, "SOURCE_B": {"F3"}},
        source_sequence_families={"ATGAAATAA": {"F1", "F2"}, "ATGCCCTAA": {"F3"}},
        candidate_sequence_families={
            "ATGAAGTAA": {"F1"},
            "ATGAAATAA": {"F2"},
            "ATGCCATAA": {"F3"},
        },
    )
    assert geometry["cross_family_shared_source_id_count"] == 1
    assert geometry["cross_family_duplicate_source_sequence_count"] == 1
    assert geometry["exact_cross_family_duplicate_candidate_sequence_count"] == 0
    assert geometry["post_dedup_independent_effective_n"] == 2
    assert geometry["source_family_component_size_histogram"] == {"1": 1, "2": 1}


def test_three_biological_replicates_are_not_nine_independent_units(
    tmp_path: Path,
) -> None:
    report, _ = _execute_fixture(tmp_path, include_mapping=True)
    endpoint = report["aggregate_observation"]["endpoint_and_replicates"]
    assert endpoint["biological_replicate_count"] == 3
    assert endpoint["timepoint_by_replicate_observation_count"] == 9
    assert endpoint["independent_n_per_candidate"] == 3
    power = report["aggregate_observation"]["split_dedup_and_power"]
    assert power["row_count_used_as_power_n"] is False
    assert power["nine_observations_used_as_power_n"] is False
    assert power["analysis_unit"] == "BIOLOGICAL_SOURCE_GROUP"


def test_conflicting_duplicate_is_excluded_and_first_row_order_cannot_change_endpoint_evidence(
    tmp_path: Path,
) -> None:
    reports: list[dict[str, object]] = []
    for order in ("ORIGINAL_FIRST", "CONFLICT_FIRST"):
        fixture_root = tmp_path / order.lower()
        fixture_root.mkdir()
        observed, reference, source_mapping = _synthetic_assets(
            fixture_root,
            include_mapping=True,
            conflicting_duplicate_order=order,
        )
        assert source_mapping is not None
        protocol = _bound_protocol()
        _bind_source_mapping(protocol, source_mapping)
        reports.append(
            PREFLIGHT.aggregate(protocol, observed, reference, source_mapping)
        )

    original_first, conflict_first = reports
    assert original_first["aggregate_observation"]["endpoint_and_replicates"] == (
        conflict_first["aggregate_observation"]["endpoint_and_replicates"]
    )
    assert original_first["aggregate_observation"]["family_and_context"] == (
        conflict_first["aggregate_observation"]["family_and_context"]
    )
    assert original_first["aggregate_observation"]["split_dedup_and_power"] == (
        conflict_first["aggregate_observation"]["split_dedup_and_power"]
    )

    observation = original_first["aggregate_observation"]
    observed_asset = observation["observed_asset"]
    endpoint = observation["endpoint_and_replicates"]
    assert observed_asset["duplicate_measurement_conflict_count"] == 1
    assert observed_asset["unresolved_conflicting_candidate_count"] == 1
    assert endpoint["valid_candidate_endpoint_and_se_count"] == 8
    assert endpoint["missing_or_censored_candidate_endpoint_count"] == 0
    assert sum(endpoint["standard_error_histogram"].values()) == 8
    assert observation["family_and_context"][
        "eligible_family_count_after_endpoint_availability"
    ] == 2

    reason = "DUPLICATE_MEASUREMENT_TUPLE_SEMANTICS_UNRESOLVED"
    for gate_id, expected_status in (
        ("ENDPOINT_DIRECTION_SCALE_AND_SEMANTICS_CLOSED", PREFLIGHT.FAIL),
        (
            "THREE_BIOLOGICAL_REPLICATE_SLOPE_AND_VALID_STANDARD_ERROR_CLOSED",
            PREFLIGHT.FAIL,
        ),
        ("MISSING_CENSORING_AND_COVERAGE_SELECTION_CLOSED", PREFLIGHT.UNKNOWN),
    ):
        assert original_first["gates"][gate_id] == {
            "status": expected_status,
            "reason": reason,
        }
        assert conflict_first["gates"][gate_id] == original_first["gates"][gate_id]


def test_synonymous_replay_family_geometry_and_power_fail_closed(tmp_path: Path) -> None:
    report, _ = _execute_fixture(tmp_path, include_mapping=True)
    gates = report["gates"]
    assert gates["SOURCE_TO_CANDIDATE_SYNONYMOUS_EDIT_REPLAY_CLOSED"][
        "status"
    ] == PREFLIGHT.PASS
    assert gates["DENSE_FAMILY_AND_CONTEXT_CLOSED"]["status"] == PREFLIGHT.PASS
    assert gates[
        "THREE_BIOLOGICAL_REPLICATE_SLOPE_AND_VALID_STANDARD_ERROR_CLOSED"
    ][
        "status"
    ] == PREFLIGHT.PASS
    power = report["aggregate_observation"]["split_dedup_and_power"]
    assert power["eligible_source_family_count_before_dedup"] == 3
    assert power["post_dedup_independent_effective_n"] == 3
    assert power["required_effective_n_for_both_power_and_ci_width"] == 156
    assert power["power_method"] == PREFLIGHT.POWER_METHOD
    assert power["confidence_interval_method"] == PREFLIGHT.CI_METHOD
    assert gates["PREFROZEN_SOURCE_GROUP_POWER_AND_FULL_CI_WIDTH_CLOSED"][
        "status"
    ] == PREFLIGHT.FAIL
    assert report["status"] == PREFLIGHT.STATUS_STOP


def test_prefrozen_power_at_97_source_groups_is_insufficient() -> None:
    assert PREFLIGHT.required_effective_n(
        rho=0.25,
        alpha=0.05,
        target_power=0.80,
        confidence=0.95,
        max_width=0.30,
    ) == 156
    assert PREFLIGHT.fisher_power(97, 0.25, 0.05) == pytest.approx(
        0.69444, abs=1e-5
    )
    assert PREFLIGHT.fisher_ci_width(97, 0.25, 0.95) == pytest.approx(
        0.38057, abs=1e-5
    )


def test_intended_universe_is_not_redefined_by_detected_perfect_subset(
    tmp_path: Path,
) -> None:
    report, _ = _execute_fixture(
        tmp_path,
        include_mapping=True,
        add_missing_intended=True,
    )
    intended = report["aggregate_observation"]["intended_universe"]
    assert intended["authoritative_mapping_candidate_count"] == 10
    assert intended["observed_unique_candidate_count"] == 9
    assert intended["intended_not_observed_count"] == 1
    assert report["gates"]["MISSING_CENSORING_AND_COVERAGE_SELECTION_CLOSED"][
        "status"
    ] == PREFLIGHT.UNKNOWN


def test_member_identifiers_sequences_and_measurements_do_not_reach_output(
    tmp_path: Path,
) -> None:
    _, output = _execute_fixture(
        tmp_path,
        include_mapping=True,
        poison=True,
        add_missing_intended=True,
    )
    rendered = output.read_text(encoding="utf-8")
    assert "POISON" not in rendered
    for source in SOURCES.values():
        assert source not in rendered
    for candidates in CANDIDATES.values():
        for candidate in candidates:
            assert candidate not in rendered
    assert '"split_assignment_output_count": 0' in rendered
    assert '"member_identifier_sequence_or_row_measurement_output_count": 0' in rendered


def test_nonpositive_abundance_is_censored_without_pseudocount(tmp_path: Path) -> None:
    values: tuple[float | None, ...] = (10.0, 11.0, 12.0, 8.0, 9.0, 10.0, 0.0, 7.0, 8.0)
    assert PREFLIGHT.estimate_endpoint(values) == {
        "valid": False,
        "endpoint": None,
        "standard_error": None,
    }


def test_endpoint_direction_is_higher_for_slower_decay() -> None:
    slower = (10.0, 11.0, 12.0, 9.0, 10.0, 11.0, 8.0, 9.0, 10.0)
    faster = (10.0, 11.0, 12.0, 6.0, 7.0, 8.0, 3.0, 4.0, 5.0)
    slower_endpoint = PREFLIGHT.estimate_endpoint(slower)
    faster_endpoint = PREFLIGHT.estimate_endpoint(faster)
    assert slower_endpoint["valid"] is True
    assert faster_endpoint["valid"] is True
    assert slower_endpoint["endpoint"] > faster_endpoint["endpoint"]


def test_replay_rejects_zero_edit_nonsynonymous_and_indel() -> None:
    source = SOURCES["P1"]
    assert PREFLIGHT.replay_synonymous_edit(source, source)["reason"] == "ZERO_EDIT"
    nonsynonymous = "ATGACTGAATTTTAA"
    assert PREFLIGHT.replay_synonymous_edit(source, nonsynonymous)["reason"] == (
        "PROTEIN_IDENTITY_FAIL"
    )
    assert PREFLIGHT.replay_synonymous_edit(source, source[:-3])["reason"] == (
        "LENGTH_OR_INDEL_MISMATCH"
    )


def test_unexpected_observed_header_is_rejected_without_output(tmp_path: Path) -> None:
    protocol = _write_protocol(tmp_path / "protocol.json", _bound_protocol())
    observed = _csv_gzip(
        tmp_path / "bad.csv.gz",
        PREFLIGHT.OBSERVED_HEADER + ("forbidden_effect",),
        [],
    )
    reference = _fasta_gzip(tmp_path / "ref.fasta.gz", {"x": CANDIDATES["P1"][0]})
    output = tmp_path / "out" / PREFLIGHT.REPORT_FILENAME
    with pytest.raises(PREFLIGHT.AssetError, match="header differs"):
        PREFLIGHT.execute(
            protocol,
            observed,
            reference,
            output,
            repo_root=tmp_path,
            binding_auditor=_fixture_binding,
        )
    assert not output.exists()


def test_existing_output_stops_before_asset_aggregation(tmp_path: Path) -> None:
    protocol = _write_protocol(tmp_path / "protocol.json", _bound_protocol())
    output = tmp_path / PREFLIGHT.REPORT_FILENAME
    output.write_text("existing\n", encoding="utf-8")
    calls = {"aggregate": 0}

    def forbidden_aggregate(*args: object) -> dict[str, object]:
        calls["aggregate"] += 1
        raise AssertionError("assets must not be reread when output exists")

    with pytest.raises(PREFLIGHT.OutputError, match="already exists"):
        PREFLIGHT.execute(
            protocol,
            tmp_path / "missing.csv.gz",
            tmp_path / "missing.fasta.gz",
            output,
            repo_root=tmp_path,
            binding_auditor=_fixture_binding,
            aggregator=forbidden_aggregate,
        )
    assert calls == {"aggregate": 0}
    assert output.read_text(encoding="utf-8") == "existing\n"


@pytest.mark.skipif(
    os.environ.get("GSE207584_AUTHORIZED_REAL_ASSET_TEST") != "1",
    reason="real ordinary-public asset access requires the bound production lifecycle",
)
def test_authorized_real_asset_shape_only_when_explicitly_enabled() -> None:
    observed = Path(
        "/mnt/cunyuliu/mrna_xeditflow_v3_1/raw_view/GSE207584/"
        "GSE207584_Zebrafish-library-perfect.csv.gz"
    )
    reference = Path(
        "/mnt/cunyuliu/mrna_xeditflow_v3_1/raw_view/GSE207584/"
        "GSE207584_reference.fasta.gz"
    )
    assert observed.is_file() and reference.is_file()
    result = PREFLIGHT.aggregate(_protocol(), observed, reference, None)
    assert result["aggregate_observation"]["observed_asset"]["body_row_count"] > 0
    assert result["gates"][
        "SOURCE_TO_CANDIDATE_SYNONYMOUS_EDIT_REPLAY_CLOSED"
    ]["status"] == PREFLIGHT.UNKNOWN
