#!/usr/bin/env python3
"""Aggregate-only GSE207584/iCodon dense-family qualification preflight.

The producer is intentionally not a qualifier.  This exact3 is the append-only
GSE207584 I4 duplicate-semantics and gate-ID repair candidate: it freezes the
actual DEC-023 A/runtime I1/I2/B2, GSE261709 I1/I2/B2, and historical GSE207584
I1/I2/B2/I3/B3 identities while retaining exactly four dynamic GSE207584 I4/B4
binding scalars as UNKNOWN.  A later config-only GSE207584 B4 must bind those four scalars and
prove the complete direct-parent chain before touching an input or the output path.  The
producer reads only ordinary-public fields authorized for the preflight,
retains member material in memory, and atomically publishes one aggregate JSON
report.  A missing authoritative source-to-candidate mapping is an expected
fail-closed outcome rather than a reason to invent an edit replay.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gzip
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import tempfile
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist
from typing import Any, Callable, Iterable, Iterator, Mapping, TextIO


SCHEMA_VERSION = (
    "route_a_v3_gse207584_aggregate_dense_family_qualification_preflight.v1"
)
PROTOCOL_ID = "GSE207584_AGGREGATE_DENSE_FAMILY_QUALIFICATION_PREFLIGHT_V1"
REPORT_FILENAME = (
    "GSE207584_AGGREGATE_DENSE_FAMILY_QUALIFICATION_PREFLIGHT_V1.json"
)
DECISION_ID = "V3-DEC-023"
DATASET_ID = "GSE207584"
UNKNOWN = "UNKNOWN_NOT_ASSERTED"
BOUND = "BOUND"

CONFIG_PATH = (
    "configs/"
    "route_a_v3_gse207584_aggregate_dense_family_qualification_preflight_v1.json"
)
SCRIPT_PATH = (
    "scripts/route_a_v3/"
    "preflight_gse207584_aggregate_dense_family_qualification.py"
)
TEST_PATH = (
    "tests/route_a_v3/"
    "test_preflight_gse207584_aggregate_dense_family_qualification.py"
)
EXACT3 = (CONFIG_PATH, SCRIPT_PATH, TEST_PATH)
RUNTIME_PATHS = (
    "configs/route_a_v3_dec023_authority_runtime_sync_v1.json",
    "scripts/route_a_v3/dec023_authority_runtime_sync.py",
    "tests/route_a_v3/test_dec023_authority_runtime_sync.py",
)
PREDECESSOR_PROTOCOL_ID = (
    "GSE261709_PUBLIC_IDENTIFIER_ASSET_SCHEMA_AGGREGATE_GEOMETRY_PREFLIGHT_V1"
)
PREDECESSOR_PATHS = (
    "configs/route_a_v3_gse261709_public_identifier_asset_schema_aggregate_"
    "geometry_preflight_v1.json",
    "scripts/route_a_v3/"
    "preflight_gse261709_public_identifier_asset_schema_aggregate_geometry.py",
    "tests/route_a_v3/"
    "test_preflight_gse261709_public_identifier_asset_schema_aggregate_geometry.py",
)
AUTHORITY_EXACT10 = (
    "configs/route_a_v3.yaml",
    "configs/route_a_v3_a1_qualification.json",
    "docs/contracts/amendments/mrna_xeditflow_route_a_v3_dec023.yaml",
    "docs/contracts/supersession_mrna_xeditflow_v1_1_to_route_a_v3.yaml",
    "docs/execution/route_a_v3_a1_interim.yaml",
    "docs/execution/route_a_v3_data_role_registry.yaml",
    "docs/execution/route_a_v3_decision_log.yaml",
    "docs/execution/route_a_v3_registry_manifest.json",
    "scripts/route_a_v3/validate_a0_bundle.py",
    "tests/route_a_v3/test_a0_integrity_guards.py",
)
AUTHORITY_COMMIT = "f7cfff896a1a30d25a3b73ea7f89957d70d95d39"
AUTHORITY_EXPECTED_PARENT = "ae8e730d726754466e5c914d7ff962377607ac50"
AUTHORITY_BLOBS = {
    "configs/route_a_v3.yaml": (
        "df38455904d67f22a2fea1fb08a3314cd4fb120e91ea711427ad1689653ba8ce"
    ),
    "configs/route_a_v3_a1_qualification.json": (
        "98de408ec423836efac75bcd75b4fd940e9fbd52a0bf1b3c397ea0c67e548740"
    ),
    "docs/contracts/amendments/mrna_xeditflow_route_a_v3_dec023.yaml": (
        "44622c7f589d841105cb21d0b35219aa9163fe4d54350671106408d4c8439e4a"
    ),
    "docs/contracts/supersession_mrna_xeditflow_v1_1_to_route_a_v3.yaml": (
        "0edfddd90ebea11db1cebb7084bdf28dd8a99426c3956a81cdfd7a4a9ccb12e2"
    ),
    "docs/execution/route_a_v3_a1_interim.yaml": (
        "fbeef398ff59764375edf9a35c2c35fd3a93db89eb3b06326ec1c73968646eff"
    ),
    "docs/execution/route_a_v3_data_role_registry.yaml": (
        "bb577d4ce7d7dc673f41bb182b7868f66816c15a3ed4235c98e0839292e75d6b"
    ),
    "docs/execution/route_a_v3_decision_log.yaml": (
        "8e514512ccd63d87a596231b11183c06765ba50ba3736adc165f141da8fa13d0"
    ),
    "docs/execution/route_a_v3_registry_manifest.json": (
        "abebbe62f7b6dbac8e0a7673fc5580a56fd96198a37362ec206519028b457c83"
    ),
    "scripts/route_a_v3/validate_a0_bundle.py": (
        "a6a71a82e90352c6c9bc02fad95c54436e268d1ee58233093dc8412c5e5739bb"
    ),
    "tests/route_a_v3/test_a0_integrity_guards.py": (
        "a0c9b3cc457011e57046506a02e57a4314a26a8885b3d110f88797a000daca0a"
    ),
}
RUNTIME_I1_COMMIT = "b0afa92eea9718c15a5989cfa67bac57036617d9"
RUNTIME_I1_BLOBS = {
    RUNTIME_PATHS[0]: (
        "330a5fceaa97a1c1f16fcb20f1c6e4e35329923a293bcb463f35eaa666cb4701"
    ),
    RUNTIME_PATHS[1]: (
        "3082aa44b70356d0e512fa8d1c92daadd08084a97594ba20976d0b93ca4706bd"
    ),
    RUNTIME_PATHS[2]: (
        "0ad48f947eee136e9104ba1dcdd5c921281ad0379ce2a17dcd4c411861115e65"
    ),
}
RUNTIME_I2_COMMIT = "d125bec7e0d9f28a679ff98c25b4feb70e198034"
RUNTIME_I2_BLOBS = {
    RUNTIME_PATHS[0]: (
        "4baa8ff33938dfc0448fb7e94097d407d8fb60beb122192aafb15eecd4f81c20"
    ),
    RUNTIME_PATHS[1]: (
        "3502227b48a0506a8c2e9c2ddec0a51017d2cb86e68c155fbdc4bccf72fa41be"
    ),
    RUNTIME_PATHS[2]: (
        "e3147049d06536b95298089ef1cd03676d8e477848890b1149dbd66352c44cc7"
    ),
}
RUNTIME_B2_COMMIT = "b225f73b72d73be3380e2d48325c7773a67c0d17"
RUNTIME_B2_EXACT_CHANGED_PATHS = (RUNTIME_PATHS[0],)
RUNTIME_B2_BLOBS = {
    RUNTIME_PATHS[0]: (
        "d1d39a659aa3ccde38d88fadb76c72d77d47585cfd7038e58493eec818b6b280"
    ),
    RUNTIME_PATHS[1]: RUNTIME_I2_BLOBS[RUNTIME_PATHS[1]],
    RUNTIME_PATHS[2]: RUNTIME_I2_BLOBS[RUNTIME_PATHS[2]],
}
PREDECESSOR_FROZEN_STATUS = "FROZEN_BOUND_I1_I2_B2"
PREDECESSOR_I1_COMMIT = "512d9cdf02a3b3826a3a5a0dcd79a7f4ccf88399"
PREDECESSOR_I1_BLOBS = {
    PREDECESSOR_PATHS[0]: (
        "77e22ef8fe145a44887ada0ecda7ac14e3e0fa89b08c4e5772ec5efd18e4e8d4"
    ),
    PREDECESSOR_PATHS[1]: (
        "0bf35b359f083242c2febb819526119191a3dbd92b06fcf9bb224f7d906ea28e"
    ),
    PREDECESSOR_PATHS[2]: (
        "8f69c635f7780135a4c79a6d4f43eb824c507da4e2055393b700642ef8c31d29"
    ),
}
PREDECESSOR_I2_COMMIT = "e8638d541f048361c338948e000bf4d52d8e1654"
PREDECESSOR_I2_BLOBS = {
    PREDECESSOR_PATHS[0]: (
        "85cebe980424a1a3e0df6880fc5f476cf5a4d36f897559f90800088a6bea5b53"
    ),
    PREDECESSOR_PATHS[1]: (
        "b94aeeb98a27ff0b5c5d176c5152a188336adcc1b49beeade9727068101581fe"
    ),
    PREDECESSOR_PATHS[2]: (
        "03b9850326dd14c90a1bbff306b96c780229d649469766cce96da6808316ca97"
    ),
}
PREDECESSOR_B2_COMMIT = "1b6422e2d6b8a77e876a0289a0f18f570fe037d2"
PREDECESSOR_B2_EXACT_CHANGED_PATHS = (PREDECESSOR_PATHS[0],)
PREDECESSOR_B2_BLOBS = {
    PREDECESSOR_PATHS[0]: (
        "24e523a8c327234999594d1019f8e6ffafd0b08330710db2b9f284d26937c50a"
    ),
    PREDECESSOR_PATHS[1]: PREDECESSOR_I2_BLOBS[PREDECESSOR_PATHS[1]],
    PREDECESSOR_PATHS[2]: PREDECESSOR_I2_BLOBS[PREDECESSOR_PATHS[2]],
}
GSE207_I1_FROZEN_STATUS = "FROZEN_BOUND_EXACT3"
GSE207_I1_COMMIT = "21337a0ee240bf469d3036d7718fb069068707a4"
GSE207_I1_BLOBS = {
    CONFIG_PATH: (
        "67c1ea9264e813154f1a4808fe8bad9d535315b9e097440412dcae8b6d4050bf"
    ),
    SCRIPT_PATH: (
        "1c967a3e0fd861579a1adcb0c881dc1ab37725867ccd5cdd35ccc3cecd3e7b16"
    ),
    TEST_PATH: (
        "6b70a439c94a61cd64464d53839b23615b1b21f75f7eb5f6a45bb521a397f070"
    ),
}
GSE207_I2_FROZEN_STATUS = "FROZEN_BOUND_EXACT3"
GSE207_I2_COMMIT = "973e75f1daff45e0087d385c1b015d300b1c3f0f"
GSE207_I2_BLOBS = {
    CONFIG_PATH: (
        "ff1b81bbce7e029d97280dd50be47f5f3073be452fc0a55a4fed256baa34fa97"
    ),
    SCRIPT_PATH: (
        "1a3468c1dfe3628ff95587b5922be25d630ca09e2bb567e8c04b181ff862b1a8"
    ),
    TEST_PATH: (
        "fd9cc811bf9c6b5ce4a4e2a4f99f1aa134adf62361eced53ca927f53576a06a0"
    ),
}
GSE207_B2_FROZEN_STATUS = "FROZEN_BOUND_CONFIG_ONLY"
GSE207_B2_COMMIT = "d8f501ecfafb55a54a23225d7abbe3422a24fcdd"
GSE207_B2_EXACT_CHANGED_PATHS = (CONFIG_PATH,)
GSE207_B2_BLOBS = {
    CONFIG_PATH: (
        "2491af4cc3c54eb5b7219253f377ad85d80e61f536e31e0f5bcba008ccad75d3"
    ),
    SCRIPT_PATH: GSE207_I2_BLOBS[SCRIPT_PATH],
    TEST_PATH: GSE207_I2_BLOBS[TEST_PATH],
}
GSE207_I3_FROZEN_STATUS = "FROZEN_BOUND_EXACT3"
GSE207_I3_COMMIT = "0ebd391f3e7713f7c0564065eab5610aaa9ed65a"
GSE207_I3_BLOBS = {
    CONFIG_PATH: (
        "047a47129d01286f34c72db963563ec69efc2312c87de268986267eaf99feae0"
    ),
    SCRIPT_PATH: (
        "919753efb890efd6cafa41c010563537493cf96238e3a7a33f0351c8758db729"
    ),
    TEST_PATH: (
        "cd5f1a0aa345c769540cfb7833b2af614b5f23a405be1e03c685fded8fc45af8"
    ),
}
GSE207_B3_FROZEN_STATUS = "FROZEN_BOUND_CONFIG_ONLY"
GSE207_B3_COMMIT = "54dc498009a9be09cde5e84c197d17a4182c0b8b"
GSE207_B3_EXACT_CHANGED_PATHS = (CONFIG_PATH,)
GSE207_B3_BLOBS = {
    CONFIG_PATH: (
        "0575ec9fddb0839d494a485b84263ba88dae0ecbbefd3b402ac0bafc61862a62"
    ),
    SCRIPT_PATH: GSE207_I3_BLOBS[SCRIPT_PATH],
    TEST_PATH: GSE207_I3_BLOBS[TEST_PATH],
}
OWN_BINDING_SCALARS = (
    "status",
    "implementation_commit",
    "implementation_script_sha256",
    "implementation_test_sha256",
)
PRODUCTION_REPO_ROOT = (
    "/home/cunyuliu/mrna_editflow_goal/worktrees/routea_v3_a1_20260810"
)
PRODUCTION_BRANCH = "routea-v3-a1-20260810"

OBSERVED_HEADER = (
    "Name",
    "Protein_id",
    "Group",
    "zf_library_2h_1",
    "zf_library_2h_2",
    "zf_library_2h_3",
    "zf_library_5h_1",
    "zf_library_5h_2",
    "zf_library_5h_3",
    "zf_library_8h_1",
    "zf_library_8h_2",
    "zf_library_8h_3",
)
MAPPING_HEADER = (
    "candidate_id",
    "source_id",
    "family_id",
    "context_id",
    "source_sequence",
    "candidate_sequence",
)
MAPPING_AUTHORITY_DYNAMIC_FIELDS = (
    "official_locator",
    "official_authority_id",
    "filename",
    "compressed_bytes",
    "compressed_sha256",
    "field_dictionary_locator",
)
MAPPING_PROVENANCE_CLASS = (
    "PUBLISHER_OR_AUTHOR_ORDINARY_PUBLIC_DESIGNED_SOURCE_CANDIDATE_MAPPING"
)
TIMEPOINTS = (2.0, 5.0, 8.0)
REPLICATES = (1, 2, 3)

GATE_IDS = (
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

PASS = "PASS_PREFLIGHT_ONLY"
FAIL = "FAIL_CLOSED"
STATUS_STOP = "STOP_CURRENT_PROTOCOL_NOT_QUALIFIED"
STATUS_BLOCKED = "BLOCKED_PENDING_EVIDENCE_NOT_QUALIFIED"
STATUS_READY = "READY_TO_REQUEST_FORMAL_QUALIFICATION_AUTHORITY_NOT_QUALIFIED"
POWER_METHOD = "BONETT_WRIGHT_FISHER_Z_ASYMPTOTIC_TWO_SIDED_SPEARMAN"
CI_METHOD = "BONETT_WRIGHT_FISHER_Z_SPEARMAN_AT_PREFROZEN_ALTERNATIVE"
WORKING_DISTRIBUTION_ASSUMPTION = (
    "MONOTONIC_TRANSFORMATION_OF_BIVARIATE_NORMAL_AT_PREFROZEN_SPEARMAN_RHO"
)
HEX40_RE = re.compile(r"^[0-9a-f]{40}$")
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")


class PreflightError(RuntimeError):
    """Base class for preflight failures."""


class ProtocolError(PreflightError):
    """The protocol or its lifecycle binding is malformed."""


class BindingNotFrozen(ProtocolError):
    """The authority/runtime/preflight lifecycle is not fully BOUND."""


class AssetError(PreflightError):
    """An ordinary-public input does not satisfy its frozen schema."""


class OutputError(PreflightError):
    """The sole aggregate output cannot be published safely."""


def _strict_json(payload: bytes, *, label: str) -> dict[str, Any]:
    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate object key")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=no_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON token {token}")
            ),
        )
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"{label} is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ProtocolError(f"{label} must be a JSON object")
    return value


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ProtocolError(f"{label} must be an object")
    return value


def _all_unknown(values: Iterable[Any]) -> bool:
    return all(value == UNKNOWN for value in values)


def _hex_commit(value: Any) -> bool:
    return isinstance(value, str) and HEX40_RE.fullmatch(value) is not None


def _hex_digest(value: Any) -> bool:
    return isinstance(value, str) and HEX64_RE.fullmatch(value) is not None


def _validate_digest_map(value: Any, paths: tuple[str, ...], *, label: str) -> None:
    if not isinstance(value, dict) or set(value) != set(paths):
        raise ProtocolError(f"{label} must bind exactly its frozen paths")
    if not all(_hex_digest(digest) for digest in value.values()):
        raise ProtocolError(f"{label} contains an invalid digest")


def _validate_lifecycle(binding: Mapping[str, Any]) -> None:
    if binding.get("binding_scheme") != (
        "DEC023_A_RUNTIME_I1_I2_B2_THEN_GSE261_I1_I2_B2_THEN_GSE207_"
        "FROZEN_I1_I2_B2_I3_B3_DYNAMIC_I4_B4_V6"
    ):
        raise ProtocolError("binding scheme differs")

    authority = _mapping(
        binding.get("authority_runtime_group"), label="authority/runtime group"
    )
    expected_authority = {
        "status": BOUND,
        "authority_commit": AUTHORITY_COMMIT,
        "authority_expected_parent": AUTHORITY_EXPECTED_PARENT,
        "authority_exact_changed_paths": list(AUTHORITY_EXACT10),
        "authority_blob_sha256_by_path": AUTHORITY_BLOBS,
        "runtime_paths": list(RUNTIME_PATHS),
        "runtime_i1_commit": RUNTIME_I1_COMMIT,
        "runtime_i1_expected_parent": AUTHORITY_COMMIT,
        "runtime_i1_blob_sha256_by_path": RUNTIME_I1_BLOBS,
        "runtime_i2_commit": RUNTIME_I2_COMMIT,
        "runtime_i2_expected_parent": RUNTIME_I1_COMMIT,
        "runtime_i2_blob_sha256_by_path": RUNTIME_I2_BLOBS,
        "runtime_b2_commit": RUNTIME_B2_COMMIT,
        "runtime_b2_expected_parent": RUNTIME_I2_COMMIT,
        "runtime_b2_exact_changed_paths": list(RUNTIME_B2_EXACT_CHANGED_PATHS),
        "runtime_b2_blob_sha256_by_path": RUNTIME_B2_BLOBS,
    }
    if dict(authority) != expected_authority:
        raise ProtocolError("frozen DEC023 A/I1/I2/B2 authority group differs")

    predecessor = _mapping(
        binding.get("predecessor_preflight_group"),
        label="GSE261709 predecessor preflight group",
    )
    expected_predecessor = {
        "status": PREDECESSOR_FROZEN_STATUS,
        "protocol_id": PREDECESSOR_PROTOCOL_ID,
        "paths": list(PREDECESSOR_PATHS),
        "implementation_exact_changed_paths": list(PREDECESSOR_PATHS),
        "implementation_i1_commit": PREDECESSOR_I1_COMMIT,
        "implementation_i1_expected_parent": RUNTIME_B2_COMMIT,
        "implementation_i1_blob_sha256_by_path": PREDECESSOR_I1_BLOBS,
        "implementation_i2_commit": PREDECESSOR_I2_COMMIT,
        "implementation_i2_expected_parent": PREDECESSOR_I1_COMMIT,
        "implementation_i2_blob_sha256_by_path": PREDECESSOR_I2_BLOBS,
        "binding_b2_commit": PREDECESSOR_B2_COMMIT,
        "binding_b2_expected_parent": PREDECESSOR_I2_COMMIT,
        "binding_b2_exact_changed_paths": list(
            PREDECESSOR_B2_EXACT_CHANGED_PATHS
        ),
        "binding_b2_blob_sha256_by_path": PREDECESSOR_B2_BLOBS,
    }
    if dict(predecessor) != expected_predecessor:
        raise ProtocolError("frozen GSE261709 I1/I2/B2 predecessor differs")

    preflight = _mapping(binding.get("preflight_group"), label="preflight group")
    expected_preflight_keys = {
        "predecessor_implementation_i1",
        "predecessor_implementation_i2",
        "predecessor_binding_b2",
        "predecessor_implementation_i3",
        "predecessor_binding_b3",
        "status",
        "implementation_commit",
        "implementation_script_path",
        "implementation_script_sha256",
        "implementation_test_path",
        "implementation_test_sha256",
        "unknown_to_bound_scalar_paths",
        "implementation_exact_changed_paths",
        "binding_exact_changed_paths",
    }
    if set(preflight) != expected_preflight_keys:
        raise ProtocolError("GSE207 preflight binding fields differ")
    if preflight.get("predecessor_implementation_i1") != {
        "status": GSE207_I1_FROZEN_STATUS,
        "commit": GSE207_I1_COMMIT,
        "expected_parent": PREDECESSOR_B2_COMMIT,
        "exact_changed_paths": list(EXACT3),
        "blob_sha256_by_path": GSE207_I1_BLOBS,
    }:
        raise ProtocolError("frozen GSE207 I1 identity differs")
    if preflight.get("predecessor_implementation_i2") != {
        "status": GSE207_I2_FROZEN_STATUS,
        "commit": GSE207_I2_COMMIT,
        "expected_parent": GSE207_I1_COMMIT,
        "exact_changed_paths": list(EXACT3),
        "blob_sha256_by_path": GSE207_I2_BLOBS,
    }:
        raise ProtocolError("frozen GSE207 I2 identity differs")
    if preflight.get("predecessor_binding_b2") != {
        "status": GSE207_B2_FROZEN_STATUS,
        "commit": GSE207_B2_COMMIT,
        "expected_parent": GSE207_I2_COMMIT,
        "exact_changed_paths": list(GSE207_B2_EXACT_CHANGED_PATHS),
        "blob_sha256_by_path": GSE207_B2_BLOBS,
    }:
        raise ProtocolError("frozen GSE207 B2 identity differs")
    if preflight.get("predecessor_implementation_i3") != {
        "status": GSE207_I3_FROZEN_STATUS,
        "commit": GSE207_I3_COMMIT,
        "expected_parent": GSE207_B2_COMMIT,
        "exact_changed_paths": list(EXACT3),
        "blob_sha256_by_path": GSE207_I3_BLOBS,
    }:
        raise ProtocolError("frozen GSE207 I3 identity differs")
    if preflight.get("predecessor_binding_b3") != {
        "status": GSE207_B3_FROZEN_STATUS,
        "commit": GSE207_B3_COMMIT,
        "expected_parent": GSE207_I3_COMMIT,
        "exact_changed_paths": list(GSE207_B3_EXACT_CHANGED_PATHS),
        "blob_sha256_by_path": GSE207_B3_BLOBS,
    }:
        raise ProtocolError("frozen GSE207 B3 identity differs")
    if preflight.get("implementation_script_path") != SCRIPT_PATH:
        raise ProtocolError("GSE207 implementation script path differs")
    if preflight.get("implementation_test_path") != TEST_PATH:
        raise ProtocolError("GSE207 implementation test path differs")
    if preflight.get("unknown_to_bound_scalar_paths") != [
        f"implementation_binding.preflight_group.{field}"
        for field in OWN_BINDING_SCALARS
    ]:
        raise ProtocolError("GSE207 own4 scalar paths differ")
    if tuple(preflight.get("implementation_exact_changed_paths", ())) != EXACT3:
        raise ProtocolError("preflight implementation must be exact3")
    if tuple(preflight.get("binding_exact_changed_paths", ())) != (CONFIG_PATH,):
        raise ProtocolError("preflight binding must be config-only")
    preflight_dynamic = [preflight.get(field) for field in OWN_BINDING_SCALARS]
    if preflight.get("status") == UNKNOWN:
        if preflight_dynamic != [UNKNOWN] * 4:
            raise ProtocolError("GSE207 own4 must remain one UNKNOWN group")
    elif preflight.get("status") == BOUND:
        if not _hex_commit(preflight.get("implementation_commit")):
            raise ProtocolError("preflight implementation commit is invalid")
        for key in ("implementation_script_sha256", "implementation_test_sha256"):
            if not _hex_digest(preflight.get(key)):
                raise ProtocolError(f"preflight {key} is invalid")
    else:
        raise ProtocolError("preflight status is invalid")


def _validate_protocol(protocol: Mapping[str, Any]) -> None:
    expected = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "contract_id": "mrna_xeditflow_route_a_v3",
        "phase_id": "A1",
        "dataset_id": DATASET_ID,
        "bioproject_id": "PRJNA856272",
        "decision_id": DECISION_ID,
        "protocol_status": (
            "AGGREGATE_DENSE_FAMILY_QUALIFICATION_PREFLIGHT_ONLY_NOT_QUALIFICATION"
        ),
    }
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ProtocolError(f"protocol {key} differs")
    _validate_lifecycle(
        _mapping(protocol.get("implementation_binding"), label="binding")
    )

    authority = _mapping(protocol.get("decision_authority"), label="authority")
    if authority.get("role_before_dec023") != "AUDIT_ONLY":
        raise ProtocolError("prior role differs")
    if authority.get("authorized_role") != (
        "AGGREGATE_DENSE_FAMILY_QUALIFICATION_PREFLIGHT_ONLY"
    ):
        raise ProtocolError("authorized role differs")
    if authority.get("allowed_output_class") != (
        "AGGREGATE_COUNTS_HISTOGRAMS_AND_GATE_STATUSES_ONLY"
    ):
        raise ProtocolError("output class differs")

    repository = _mapping(
        protocol.get("repository_authority"), label="repository authority"
    )
    if dict(repository) != {
        "production_repo_root": PRODUCTION_REPO_ROOT,
        "branch": PRODUCTION_BRANCH,
        "upstream_ref": f"origin/{PRODUCTION_BRANCH}",
        "required_state_before_asset_or_output_io": (
            "CLEAN_HEAD_EQUALS_UPSTREAM_EQUALS_ORIGIN"
        ),
    }:
        raise ProtocolError("repository authority differs")

    assets = _mapping(
        protocol.get("ordinary_public_asset_contract"), label="asset contract"
    )
    observed = _mapping(assets.get("observed_perfect_csv"), label="observed CSV")
    if tuple(observed.get("required_header_exactly", ())) != OBSERVED_HEADER:
        raise ProtocolError("observed CSV header differs")
    source_map = _mapping(
        assets.get("authoritative_source_mapping"), label="source mapping"
    )
    if tuple(source_map.get("required_header_exactly", ())) != MAPPING_HEADER:
        raise ProtocolError("source mapping header differs")
    if source_map.get("provenance_class") != MAPPING_PROVENANCE_CLASS:
        raise ProtocolError("source mapping provenance class differs")
    if source_map.get("cli_input_permitted_only_when_bound") is not True:
        raise ProtocolError("source mapping CLI boundary differs")
    if source_map.get("identity_mismatch_action") != (
        "STOP_BEFORE_MAPPING_PARSE_OR_OUTPUT"
    ):
        raise ProtocolError("source mapping identity mismatch action differs")
    mapping_dynamic = tuple(
        source_map.get(field) for field in MAPPING_AUTHORITY_DYNAMIC_FIELDS
    )
    if source_map.get("status") == UNKNOWN:
        if not _all_unknown(mapping_dynamic):
            raise ProtocolError("partial source mapping authority group is forbidden")
    elif source_map.get("status") == BOUND:
        locator = source_map.get("official_locator")
        dictionary = source_map.get("field_dictionary_locator")
        authority_id = source_map.get("official_authority_id")
        filename = source_map.get("filename")
        byte_count = source_map.get("compressed_bytes")
        digest = source_map.get("compressed_sha256")
        if not isinstance(locator, str) or not locator.startswith("https://"):
            raise ProtocolError("bound source mapping locator must be public HTTPS")
        if not isinstance(dictionary, str) or not dictionary.startswith("https://"):
            raise ProtocolError("bound source mapping field dictionary must be HTTPS")
        if (
            not isinstance(authority_id, str)
            or not authority_id
            or authority_id == UNKNOWN
        ):
            raise ProtocolError("bound source mapping authority identity is invalid")
        if (
            not isinstance(filename, str)
            or not filename
            or Path(filename).name != filename
        ):
            raise ProtocolError("bound source mapping filename is invalid")
        if (
            isinstance(byte_count, bool)
            or not isinstance(byte_count, int)
            or byte_count <= 0
        ):
            raise ProtocolError("bound source mapping byte count is invalid")
        if not _hex_digest(digest):
            raise ProtocolError("bound source mapping digest is invalid")
    else:
        raise ProtocolError("source mapping authority status is invalid")

    endpoint = _mapping(protocol.get("endpoint_policy"), label="endpoint")
    expected_endpoint = {
        "biological_replicate_count": 3,
        "timepoint_by_replicate_observation_count": 9,
        "nine_observations_are_nine_independent_replicates": False,
        "per_replicate_estimand": (
            "OLS_SLOPE_OF_NATURAL_LOG_ABUNDANCE_VERSUS_HOUR"
        ),
        "candidate_endpoint": "MEAN_OF_THREE_BIOLOGICAL_REPLICATE_ESTIMATES",
        "candidate_standard_error": (
            "SAMPLE_SD_OF_THREE_REPLICATE_ESTIMATES_DIVIDED_BY_SQRT_THREE"
        ),
        "direction": "HIGHER_IS_SLOWER_DECAY_AND_GREATER_STABILITY",
        "nonfinite_or_nonpositive_abundance_policy": (
            "MISSING_CENSORED_NO_PSEUDOCOUNT_NO_OUTCOME_BASED_UNIVERSE_REDEFINITION"
        ),
    }
    if dict(endpoint) != expected_endpoint:
        raise ProtocolError("endpoint and replicate policy differs")
    if tuple(
        _mapping(
            protocol.get("family_and_context_policy"), label="family/context"
        ).get("timepoints_hours", ())
    ) != TIMEPOINTS:
        raise ProtocolError("timepoints differ")

    split = _mapping(protocol.get("split_and_dedup_policy"), label="split/dedup")
    if dict(split) != {
        "split_unit": "SOURCE_FAMILY_CONNECTED_COMPONENT",
        "same_source_id_connects_source_families": True,
        "exact_source_sequence_duplicates_connect_source_families": True,
        "exact_candidate_sequence_duplicates_connect_source_families": True,
        "member_split_assignments_may_be_output": False,
        "minimum_components_for_three_way_split_feasibility": 3,
    }:
        raise ProtocolError("source-group split and dedup policy differs")

    power = _mapping(protocol.get("prefrozen_power_policy"), label="power")
    expected_power = {
        "analysis_and_bootstrap_unit": "BIOLOGICAL_SOURCE_GROUP",
        "target_metric": "WITHIN_STUDY_SPEARMAN",
        "minimum_effect_at_alternative": 0.25,
        "alpha_two_sided": 0.05,
        "target_power": 0.8,
        "confidence_level": 0.95,
        "maximum_ci_full_width": 0.3,
        "power_method": POWER_METHOD,
        "confidence_interval_method": CI_METHOD,
        "working_distribution_assumption": WORKING_DISTRIBUTION_ASSUMPTION,
    }
    if dict(power) != expected_power:
        raise ProtocolError("prefrozen power policy differs")
    if tuple(protocol.get("gate_ids", ())) != GATE_IDS:
        raise ProtocolError("gate set differs")

    claims = _mapping(protocol.get("claim_boundary"), label="claim boundary")
    forbidden_true = (
        "qualification_allowed_or_changed",
        "true_a2_status_allowed_or_changed",
        "study_credit_allowed_or_changed",
        "canonical_allowed_or_changed",
        "training_allowed_or_changed",
        "gpu_allowed_or_changed",
        "model_selection_allowed_or_changed",
        "next_phase_allowed_or_changed",
    )
    if any(claims.get(key) is not False for key in forbidden_true):
        raise ProtocolError("a forbidden claim boundary was enabled")
    if claims.get("current_credit_delta") != {
        "ordinary": 0,
        "A1": 0,
        "true_A2": 0,
    }:
        raise ProtocolError("current credit delta must remain zero")
    if claims.get("theoretical_maximum_only_if_later_fully_qualified") != {
        "ordinary": 1,
        "A1": 0,
        "true_A2": 1,
    }:
        raise ProtocolError("theoretical maximum differs")
    output = _mapping(protocol.get("output_contract"), label="output")
    if output.get("filename") != REPORT_FILENAME:
        raise ProtocolError("output filename differs")
    if (
        output.get("single_json_only") is not True
        or output.get("atomic_publish") is not True
        or output.get("overwrite_allowed") is not False
        or output.get("row_or_member_material_persisted_count") != 0
    ):
        raise ProtocolError("output boundary differs")


def _normalise_preflight_binding(protocol: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(protocol))
    group = result["implementation_binding"]["preflight_group"]
    for key in OWN_BINDING_SCALARS:
        group[key] = UNKNOWN
    _validate_protocol(result)
    return result


def _json_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=True,
                allow_nan=False,
                sort_keys=True,
                indent=2,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise OutputError("aggregate report is not finite JSON") from exc


def _protocol_json_bytes(value: Any) -> bytes:
    """Canonical lifecycle config bytes; preserves the frozen key order."""
    try:
        return (
            json.dumps(value, ensure_ascii=True, allow_nan=False, indent=2) + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ProtocolError("protocol lifecycle config is not finite JSON") from exc


def _run_git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        raise ProtocolError("Git lifecycle verification failed")
    return result.stdout.strip()


def _git_blob(root: Path, commit: str, path: str) -> bytes:
    result = subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise ProtocolError("a frozen Git blob is missing")
    return result.stdout


def _changed_paths(root: Path, commit: str) -> tuple[str, ...]:
    output = _run_git(
        root,
        "diff-tree",
        "--no-commit-id",
        "--name-only",
        "-r",
        commit,
    )
    return tuple(sorted(line for line in output.splitlines() if line))


def _verify_commit(
    root: Path,
    *,
    commit: str,
    expected_parent: str,
    expected_paths: tuple[str, ...],
    expected_blobs: Mapping[str, str] | None = None,
) -> None:
    if _run_git(root, "rev-parse", f"{commit}^") != expected_parent:
        raise ProtocolError("a lifecycle parent differs")
    if _changed_paths(root, commit) != tuple(sorted(expected_paths)):
        raise ProtocolError("a lifecycle commit changed unexpected paths")
    if expected_blobs is not None:
        for path, expected in expected_blobs.items():
            observed = hashlib.sha256(_git_blob(root, commit, path)).hexdigest()
            if observed != expected:
                raise ProtocolError("a frozen lifecycle blob differs")


def _require_bound_lifecycle(protocol: Mapping[str, Any]) -> None:
    binding = protocol["implementation_binding"]
    if binding["authority_runtime_group"]["status"] != BOUND:
        raise BindingNotFrozen("authority/runtime group is not BOUND")
    if binding["predecessor_preflight_group"]["status"] != (
        PREDECESSOR_FROZEN_STATUS
    ):
        raise BindingNotFrozen("GSE261709 predecessor chain is not frozen")
    if binding["preflight_group"]["status"] != BOUND:
        raise BindingNotFrozen("preflight group is not BOUND")


def _default_binding_auditor(
    protocol: Mapping[str, Any],
    protocol_path: Path,
    protocol_bytes: bytes,
    repo_root: Path,
) -> dict[str, Any]:
    _require_bound_lifecycle(protocol)
    root = repo_root.resolve()
    expected_protocol_path = (root / CONFIG_PATH).resolve()
    if protocol_path.resolve() != expected_protocol_path:
        raise ProtocolError("protocol must be loaded from the production path")
    script_path = (root / SCRIPT_PATH).resolve()
    if Path(__file__).resolve() != script_path:
        raise ProtocolError("executing producer is not the production Git path")

    binding = protocol["implementation_binding"]
    authority = binding["authority_runtime_group"]
    predecessor = binding["predecessor_preflight_group"]
    preflight = binding["preflight_group"]
    head = _run_git(root, "rev-parse", "HEAD")
    upstream = _run_git(root, "rev-parse", "@{upstream}")
    origin = _run_git(
        root,
        "rev-parse",
        "--verify",
        f"refs/remotes/origin/{PRODUCTION_BRANCH}",
    )
    if head != upstream or head != origin:
        raise ProtocolError("HEAD/upstream/origin differ")
    if _run_git(root, "rev-parse", "--abbrev-ref", "HEAD") != PRODUCTION_BRANCH:
        raise ProtocolError("production branch differs")
    if _run_git(root, "rev-parse", "--abbrev-ref", "@{upstream}") != (
        f"origin/{PRODUCTION_BRANCH}"
    ):
        raise ProtocolError("production upstream differs")
    if _run_git(root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise ProtocolError("production worktree is dirty")

    _verify_commit(
        root,
        commit=authority["authority_commit"],
        expected_parent=authority["authority_expected_parent"],
        expected_paths=AUTHORITY_EXACT10,
        expected_blobs=authority["authority_blob_sha256_by_path"],
    )
    _verify_commit(
        root,
        commit=authority["runtime_i1_commit"],
        expected_parent=authority["runtime_i1_expected_parent"],
        expected_paths=RUNTIME_PATHS,
        expected_blobs=authority["runtime_i1_blob_sha256_by_path"],
    )
    _verify_commit(
        root,
        commit=authority["runtime_i2_commit"],
        expected_parent=authority["runtime_i2_expected_parent"],
        expected_paths=RUNTIME_PATHS,
        expected_blobs=authority["runtime_i2_blob_sha256_by_path"],
    )
    _verify_commit(
        root,
        commit=authority["runtime_b2_commit"],
        expected_parent=authority["runtime_b2_expected_parent"],
        expected_paths=RUNTIME_B2_EXACT_CHANGED_PATHS,
        expected_blobs=authority["runtime_b2_blob_sha256_by_path"],
    )
    _verify_commit(
        root,
        commit=predecessor["implementation_i1_commit"],
        expected_parent=predecessor["implementation_i1_expected_parent"],
        expected_paths=PREDECESSOR_PATHS,
        expected_blobs=predecessor["implementation_i1_blob_sha256_by_path"],
    )
    _verify_commit(
        root,
        commit=predecessor["implementation_i2_commit"],
        expected_parent=predecessor["implementation_i2_expected_parent"],
        expected_paths=PREDECESSOR_PATHS,
        expected_blobs=predecessor["implementation_i2_blob_sha256_by_path"],
    )
    _verify_commit(
        root,
        commit=predecessor["binding_b2_commit"],
        expected_parent=predecessor["binding_b2_expected_parent"],
        expected_paths=PREDECESSOR_B2_EXACT_CHANGED_PATHS,
        expected_blobs=predecessor["binding_b2_blob_sha256_by_path"],
    )
    frozen_i1 = preflight["predecessor_implementation_i1"]
    _verify_commit(
        root,
        commit=frozen_i1["commit"],
        expected_parent=frozen_i1["expected_parent"],
        expected_paths=EXACT3,
        expected_blobs=frozen_i1["blob_sha256_by_path"],
    )
    frozen_i2 = preflight["predecessor_implementation_i2"]
    _verify_commit(
        root,
        commit=frozen_i2["commit"],
        expected_parent=frozen_i2["expected_parent"],
        expected_paths=EXACT3,
        expected_blobs=frozen_i2["blob_sha256_by_path"],
    )
    frozen_b2 = preflight["predecessor_binding_b2"]
    _verify_commit(
        root,
        commit=frozen_b2["commit"],
        expected_parent=frozen_b2["expected_parent"],
        expected_paths=GSE207_B2_EXACT_CHANGED_PATHS,
        expected_blobs=frozen_b2["blob_sha256_by_path"],
    )
    frozen_i3 = preflight["predecessor_implementation_i3"]
    _verify_commit(
        root,
        commit=frozen_i3["commit"],
        expected_parent=frozen_i3["expected_parent"],
        expected_paths=EXACT3,
        expected_blobs=frozen_i3["blob_sha256_by_path"],
    )
    frozen_b3 = preflight["predecessor_binding_b3"]
    _verify_commit(
        root,
        commit=frozen_b3["commit"],
        expected_parent=frozen_b3["expected_parent"],
        expected_paths=GSE207_B3_EXACT_CHANGED_PATHS,
        expected_blobs=frozen_b3["blob_sha256_by_path"],
    )
    _verify_commit(
        root,
        commit=preflight["implementation_commit"],
        expected_parent=frozen_b3["commit"],
        expected_paths=EXACT3,
    )
    _verify_commit(
        root,
        commit=head,
        expected_parent=preflight["implementation_commit"],
        expected_paths=(CONFIG_PATH,),
    )

    expected_i = _protocol_json_bytes(_normalise_preflight_binding(protocol))
    if _git_blob(root, preflight["implementation_commit"], CONFIG_PATH) != expected_i:
        raise ProtocolError("preflight implementation config is not canonical I")
    if _git_blob(root, head, CONFIG_PATH) != protocol_bytes:
        raise ProtocolError("preflight binding config differs from disk")
    script_blob = _git_blob(root, preflight["implementation_commit"], SCRIPT_PATH)
    test_blob = _git_blob(root, preflight["implementation_commit"], TEST_PATH)
    if hashlib.sha256(script_blob).hexdigest() != preflight[
        "implementation_script_sha256"
    ]:
        raise ProtocolError("preflight implementation script digest differs")
    if hashlib.sha256(test_blob).hexdigest() != preflight[
        "implementation_test_sha256"
    ]:
        raise ProtocolError("preflight implementation test digest differs")
    if script_path.read_bytes() != script_blob:
        raise ProtocolError("executing producer bytes differ from bound producer")
    if (root / TEST_PATH).read_bytes() != test_blob:
        raise ProtocolError("focused test bytes differ from bound test")
    return {
        "status": "BOUND_DEC023_GSE261_AND_GSE207_I1_I2_B2_I3_B3_I4_B4",
        "authority_commit": authority["authority_commit"],
        "runtime_i1_commit": authority["runtime_i1_commit"],
        "runtime_i2_commit": authority["runtime_i2_commit"],
        "runtime_b2_commit": authority["runtime_b2_commit"],
        "gse261709_i1_commit": predecessor["implementation_i1_commit"],
        "gse261709_i2_commit": predecessor["implementation_i2_commit"],
        "gse261709_b2_commit": predecessor["binding_b2_commit"],
        "gse207584_i1_commit": frozen_i1["commit"],
        "gse207584_i2_commit": frozen_i2["commit"],
        "gse207584_b2_commit": frozen_b2["commit"],
        "gse207584_i3_commit": frozen_i3["commit"],
        "gse207584_b3_commit": frozen_b3["commit"],
        "gse207584_i4_commit": preflight["implementation_commit"],
        "gse207584_b4_commit": head,
    }


@contextmanager
def _open_text(path: Path) -> Iterator[TextIO]:
    try:
        if path.name.endswith(".gz"):
            with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
                yield handle
        else:
            with path.open("r", encoding="utf-8", newline="") as handle:
                yield handle
    except (OSError, UnicodeError) as exc:
        raise AssetError("an ordinary-public text asset is unreadable") from exc


def _parse_float(value: str) -> tuple[float | None, bool]:
    if value.strip() == "":
        return None, False
    try:
        number = float(value)
    except ValueError:
        return None, True
    if not math.isfinite(number):
        return None, True
    return number, False


def _read_observed(path: Path) -> dict[str, Any]:
    records: dict[str, dict[str, Any]] = {}
    conflicted_candidates: set[str] = set()
    body_rows = 0
    malformed_rows = 0
    invalid_numeric_cells = 0
    duplicate_measurement_conflicts = 0
    with _open_text(path) as handle:
        reader = csv.reader(handle)
        try:
            header = tuple(next(reader))
        except StopIteration as exc:
            raise AssetError("observed CSV is empty") from exc
        if header != OBSERVED_HEADER:
            raise AssetError("observed CSV header differs from the frozen schema")
        for row in reader:
            body_rows += 1
            if len(row) != len(OBSERVED_HEADER):
                malformed_rows += 1
                continue
            candidate_id, protein_id, group = row[:3]
            if not protein_id or not group or not candidate_id:
                malformed_rows += 1
                continue
            values: list[float | None] = []
            for cell in row[3:]:
                number, invalid = _parse_float(cell)
                values.append(number)
                invalid_numeric_cells += int(invalid)
            existing = records.get(candidate_id)
            if existing is None:
                records[candidate_id] = {
                    "protein_id": protein_id,
                    "groups": {group},
                    "values": tuple(values),
                }
            else:
                existing["groups"].add(group)
                if (
                    existing["protein_id"] != protein_id
                    or existing["values"] != tuple(values)
                ):
                    duplicate_measurement_conflicts += 1
                    conflicted_candidates.add(candidate_id)
    group_label_histogram = Counter(
        _small_count_bin(len(record["groups"])) for record in records.values()
    )
    return {
        "records": records,
        "body_row_count": body_rows,
        "unique_candidate_count": len(records),
        "malformed_row_count": malformed_rows,
        "invalid_numeric_cell_count": invalid_numeric_cells,
        "duplicate_measurement_conflict_count": duplicate_measurement_conflicts,
        "conflicted_candidate_ids": conflicted_candidates,
        "unresolved_conflicting_candidate_count": len(conflicted_candidates),
        "candidate_design_group_label_count_histogram": dict(
            sorted(group_label_histogram.items())
        ),
    }


def _normalise_sequence(value: str) -> str | None:
    sequence = "".join(value.split()).upper().replace("U", "T")
    if not sequence or set(sequence) - set("ACGT"):
        return None
    return sequence


def _read_fasta(path: Path) -> dict[str, Any]:
    records: dict[str, str] = {}
    duplicate_header_count = 0
    invalid_sequence_count = 0
    current: str | None = None
    pieces: list[str] = []

    def finish() -> None:
        nonlocal duplicate_header_count, invalid_sequence_count
        if current is None:
            return
        sequence = _normalise_sequence("".join(pieces))
        if sequence is None:
            invalid_sequence_count += 1
            return
        if current in records:
            duplicate_header_count += 1
            return
        records[current] = sequence

    with _open_text(path) as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                finish()
                current = line[1:].strip()
                pieces = []
                if not current:
                    raise AssetError("reference FASTA contains an empty header")
            else:
                if current is None:
                    raise AssetError("reference FASTA sequence precedes its header")
                pieces.append(line)
        finish()
    if not records:
        raise AssetError("reference FASTA has no valid records")
    length_histogram = Counter(_length_bin(len(value)) for value in records.values())
    return {
        "records": records,
        "record_count": len(records),
        "duplicate_header_count": duplicate_header_count,
        "invalid_sequence_count": invalid_sequence_count,
        "sequence_length_histogram": dict(sorted(length_histogram.items())),
    }


def _verify_source_mapping_identity(
    path: Path, contract: Mapping[str, Any]
) -> None:
    if contract.get("status") != BOUND:
        raise AssetError("source mapping authority is not BOUND")
    if path.name != contract.get("filename"):
        raise AssetError("source mapping filename differs from the bound identity")
    digest = hashlib.sha256()
    byte_count = 0
    try:
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                byte_count += len(chunk)
                digest.update(chunk)
    except OSError as exc:
        raise AssetError("bound source mapping is unreadable") from exc
    if byte_count != contract.get("compressed_bytes"):
        raise AssetError("source mapping byte count differs from the bound identity")
    if digest.hexdigest() != contract.get("compressed_sha256"):
        raise AssetError("source mapping digest differs from the bound identity")


def _read_source_mapping(
    path: Path, contract: Mapping[str, Any]
) -> dict[str, Any]:
    _verify_source_mapping_identity(path, contract)
    records: dict[str, dict[str, str]] = {}
    malformed_rows = 0
    duplicate_candidate_count = 0
    with _open_text(path) as handle:
        reader = csv.reader(handle)
        try:
            header = tuple(next(reader))
        except StopIteration as exc:
            raise AssetError("source mapping is empty") from exc
        if header != MAPPING_HEADER:
            raise AssetError("source mapping header differs from the frozen schema")
        for row in reader:
            if len(row) != len(MAPPING_HEADER) or any(cell == "" for cell in row):
                malformed_rows += 1
                continue
            candidate_id = row[0]
            if candidate_id in records:
                duplicate_candidate_count += 1
                continue
            records[candidate_id] = dict(zip(MAPPING_HEADER[1:], row[1:]))
    return {
        "records": records,
        "row_count": len(records) + duplicate_candidate_count + malformed_rows,
        "unique_candidate_count": len(records),
        "malformed_row_count": malformed_rows,
        "duplicate_candidate_count": duplicate_candidate_count,
    }


CODON_TABLE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}


def _translate_cds(sequence: str) -> str | None:
    if (
        len(sequence) % 3 != 0
        or len(sequence) < 6
        or sequence[:3] != "ATG"
        or sequence[-3:] not in {"TAA", "TAG", "TGA"}
    ):
        return None
    translated = "".join(
        CODON_TABLE[sequence[index : index + 3]]
        for index in range(0, len(sequence), 3)
    )
    if "*" in translated[:-1] or not translated.endswith("*"):
        return None
    return translated


def replay_synonymous_edit(source_raw: str, candidate_raw: str) -> dict[str, Any]:
    source = _normalise_sequence(source_raw)
    candidate = _normalise_sequence(candidate_raw)
    if source is None or candidate is None:
        return {"valid": False, "reason": "INVALID_ALPHABET_OR_EMPTY", "edits": 0}
    if len(source) != len(candidate):
        return {"valid": False, "reason": "LENGTH_OR_INDEL_MISMATCH", "edits": 0}
    source_protein = _translate_cds(source)
    candidate_protein = _translate_cds(candidate)
    if source_protein is None or candidate_protein is None:
        return {"valid": False, "reason": "CDS_INVARIANT_FAIL", "edits": 0}
    if source_protein != candidate_protein:
        return {"valid": False, "reason": "PROTEIN_IDENTITY_FAIL", "edits": 0}
    edits = sum(left != right for left, right in zip(source, candidate))
    if edits == 0:
        return {"valid": False, "reason": "ZERO_EDIT", "edits": 0}
    return {"valid": True, "reason": "SYNONYMOUS_SUBSTITUTION_ONLY", "edits": edits}


def estimate_endpoint(values: tuple[float | None, ...]) -> dict[str, Any]:
    if len(values) != 9:
        raise ValueError("endpoint requires exactly nine time-by-replicate values")
    replicate_estimates: list[float] = []
    for replicate_index in range(3):
        abundance = [values[replicate_index + 3 * time] for time in range(3)]
        if any(value is None or value <= 0.0 for value in abundance):
            return {"valid": False, "endpoint": None, "standard_error": None}
        logs = [math.log(float(value)) for value in abundance]
        mean_x = statistics.mean(TIMEPOINTS)
        mean_y = statistics.mean(logs)
        denominator = sum((point - mean_x) ** 2 for point in TIMEPOINTS)
        slope = sum(
            (point - mean_x) * (value - mean_y)
            for point, value in zip(TIMEPOINTS, logs)
        ) / denominator
        replicate_estimates.append(slope)
    endpoint = statistics.mean(replicate_estimates)
    standard_error = statistics.stdev(replicate_estimates) / math.sqrt(3.0)
    return {
        "valid": True,
        "endpoint": endpoint,
        "standard_error": standard_error,
    }


def _small_count_bin(value: int) -> str:
    if value <= 0:
        return "0"
    if value <= 2:
        return str(value)
    if value <= 5:
        return "3-5"
    if value <= 10:
        return "6-10"
    return "11+"


def _length_bin(value: int) -> str:
    if value < 300:
        return "LT300"
    if value == 300:
        return "EQ300"
    if value <= 600:
        return "301-600"
    return "GT600"


def _edit_bin(value: int) -> str:
    if value <= 2:
        return str(value)
    if value <= 5:
        return "3-5"
    if value <= 10:
        return "6-10"
    if value <= 25:
        return "11-25"
    return "26+"


def _se_bin(value: float) -> str:
    if value == 0.0:
        return "0"
    if value <= 0.01:
        return "GT0-0.01"
    if value <= 0.05:
        return "GT0.01-0.05"
    if value <= 0.10:
        return "GT0.05-0.10"
    return "GT0.10"


class _UnionFind:
    def __init__(self, members: Iterable[str]):
        self.parent = {member: member for member in members}

    def find(self, member: str) -> str:
        parent = self.parent[member]
        if parent != member:
            self.parent[member] = self.find(parent)
        return self.parent[member]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def _source_family_components(
    eligible_families: set[str],
    *,
    source_id_families: Mapping[str, set[str]],
    source_sequence_families: Mapping[str, set[str]],
    candidate_sequence_families: Mapping[str, set[str]],
) -> dict[str, Any]:
    union = _UnionFind(eligible_families)

    def connect(index: Mapping[str, set[str]]) -> int:
        cross_family_identity_count = 0
        for families in index.values():
            eligible = sorted(families & eligible_families)
            if len(eligible) > 1:
                cross_family_identity_count += 1
                for family in eligible[1:]:
                    union.union(eligible[0], family)
        return cross_family_identity_count

    shared_source_id_count = connect(source_id_families)
    duplicate_source_sequence_count = connect(source_sequence_families)
    duplicate_candidate_sequence_count = connect(candidate_sequence_families)
    component_size_histogram: Counter[str] = Counter()
    if eligible_families:
        component_sizes = Counter(
            union.find(family) for family in eligible_families
        )
        component_size_histogram.update(
            _small_count_bin(size) for size in component_sizes.values()
        )
        effective_n = len(component_sizes)
    else:
        effective_n = 0
    return {
        "cross_family_shared_source_id_count": shared_source_id_count,
        "cross_family_duplicate_source_sequence_count": (
            duplicate_source_sequence_count
        ),
        "exact_cross_family_duplicate_candidate_sequence_count": (
            duplicate_candidate_sequence_count
        ),
        "source_family_component_size_histogram": dict(
            sorted(component_size_histogram.items())
        ),
        "post_dedup_independent_effective_n": effective_n,
    }


def _normal_cdf(value: float) -> float:
    return NormalDist().cdf(value)


def fisher_power(n: int, rho: float, alpha: float) -> float:
    if n <= 3:
        return 0.0
    null_standard_error = 1.0 / math.sqrt(n - 3.0)
    alternative_standard_error = (
        math.sqrt(1.0 + rho**2 / 2.0) * null_standard_error
    )
    alternative_z = math.atanh(rho)
    critical = (
        NormalDist().inv_cdf(1.0 - alpha / 2.0) * null_standard_error
    )
    return (
        1.0
        - _normal_cdf((critical - alternative_z) / alternative_standard_error)
        + _normal_cdf((-critical - alternative_z) / alternative_standard_error)
    )


def fisher_ci_width(n: int, rho: float, confidence: float) -> float:
    if n <= 3:
        return 2.0
    critical = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    center = math.atanh(rho)
    alternative_standard_error = (
        math.sqrt(1.0 + rho**2 / 2.0) / math.sqrt(n - 3.0)
    )
    radius = critical * alternative_standard_error
    return math.tanh(center + radius) - math.tanh(center - radius)


def required_effective_n(
    *, rho: float, alpha: float, target_power: float, confidence: float, max_width: float
) -> int:
    for n in range(4, 10001):
        if fisher_power(n, rho, alpha) >= target_power and fisher_ci_width(
            n, rho, confidence
        ) <= max_width:
            return n
    raise ProtocolError("prefrozen power target has no bounded solution")


def _gate(status: str, reason: str) -> dict[str, str]:
    if status not in {PASS, FAIL, UNKNOWN}:
        raise AssertionError("invalid internal gate status")
    return {"status": status, "reason": reason}


def aggregate(
    protocol: Mapping[str, Any],
    observed_csv: Path,
    reference_fasta: Path,
    source_mapping: Path | None,
) -> dict[str, Any]:
    observed = _read_observed(observed_csv)
    fasta = _read_fasta(reference_fasta)
    mapping_contract = protocol["ordinary_public_asset_contract"][
        "authoritative_source_mapping"
    ]
    mapping_authority_bound = mapping_contract["status"] == BOUND
    mapping_input_provided = source_mapping is not None
    mapping = (
        _read_source_mapping(source_mapping, mapping_contract)
        if source_mapping is not None and mapping_authority_bound
        else None
    )
    mapping_read_count = int(mapping is not None)
    observed_records = observed["records"]
    fasta_records = fasta["records"]
    mapping_records = mapping["records"] if mapping else {}

    observed_ids = set(observed_records)
    fasta_ids = set(fasta_records)
    intended_ids = set(mapping_records) if mapping else fasta_ids
    observed_not_intended = observed_ids - intended_ids
    intended_not_observed = intended_ids - observed_ids
    fasta_not_mapping = fasta_ids - set(mapping_records) if mapping else set()
    mapping_not_fasta = set(mapping_records) - fasta_ids if mapping else set()

    endpoint_by_candidate: dict[str, dict[str, Any]] = {}
    se_histogram: Counter[str] = Counter()
    valid_endpoint_count = 0
    invalid_endpoint_count = 0
    for candidate_id, record in observed_records.items():
        if candidate_id in observed["conflicted_candidate_ids"]:
            continue
        estimate = estimate_endpoint(record["values"])
        endpoint_by_candidate[candidate_id] = estimate
        if estimate["valid"]:
            valid_endpoint_count += 1
            se_histogram[_se_bin(float(estimate["standard_error"]))] += 1
        else:
            invalid_endpoint_count += 1

    replay_valid_ids: set[str] = set()
    replay_reason_counts: Counter[str] = Counter()
    edit_histogram: Counter[str] = Counter()
    candidate_fasta_mismatch_count = 0
    observed_source_id_mismatch_count = 0
    family_candidates: dict[str, set[str]] = defaultdict(set)
    family_sources: dict[str, set[tuple[str, str]]] = defaultdict(set)
    family_contexts: dict[str, set[str]] = defaultdict(set)
    source_id_families: dict[str, set[str]] = defaultdict(set)
    source_sequence_families: dict[str, set[str]] = defaultdict(set)
    candidate_sequence_families: dict[str, set[str]] = defaultdict(set)

    if mapping:
        for candidate_id, row in mapping_records.items():
            replay = replay_synonymous_edit(
                row["source_sequence"], row["candidate_sequence"]
            )
            reason = str(replay["reason"])
            replay_reason_counts[reason] += 1
            if replay["valid"]:
                edit_histogram[_edit_bin(int(replay["edits"]))] += 1
                normal_candidate = _normalise_sequence(row["candidate_sequence"])
                if fasta_records.get(candidate_id) != normal_candidate:
                    candidate_fasta_mismatch_count += 1
                else:
                    replay_valid_ids.add(candidate_id)
                    candidate_sequence_families[str(normal_candidate)].add(
                        row["family_id"]
                    )
            if candidate_id in observed_records and observed_records[candidate_id][
                "protein_id"
            ] != row["source_id"]:
                observed_source_id_mismatch_count += 1
            source_sequence = _normalise_sequence(row["source_sequence"]) or "INVALID"
            family_candidates[row["family_id"]].add(candidate_id)
            family_sources[row["family_id"]].add((row["source_id"], source_sequence))
            family_contexts[row["family_id"]].add(row["context_id"])
            source_id_families[row["source_id"]].add(row["family_id"])
            if source_sequence != "INVALID":
                source_sequence_families[source_sequence].add(row["family_id"])

    minimum_candidates = protocol["family_and_context_policy"][
        "minimum_distinct_valid_candidates_per_family"
    ]
    expected_context = protocol["family_and_context_policy"]["context"]
    structurally_valid_families: set[str] = set()
    eligible_families: set[str] = set()
    family_size_histogram: Counter[str] = Counter()
    inconsistent_family_source_count = 0
    inconsistent_family_context_count = 0
    for family, candidates in family_candidates.items():
        family_size_histogram[_small_count_bin(len(candidates))] += 1
        if len(family_sources[family]) != 1:
            inconsistent_family_source_count += 1
        if family_contexts[family] != {expected_context}:
            inconsistent_family_context_count += 1
        valid_candidates = candidates & replay_valid_ids
        if (
            len(valid_candidates) >= minimum_candidates
            and len(family_sources[family]) == 1
            and family_contexts[family] == {expected_context}
        ):
            structurally_valid_families.add(family)
            complete = {
                candidate
                for candidate in valid_candidates
                if candidate in endpoint_by_candidate
                and endpoint_by_candidate[candidate]["valid"]
            }
            if len(complete) >= minimum_candidates:
                eligible_families.add(family)

    component_geometry = _source_family_components(
        eligible_families,
        source_id_families=source_id_families,
        source_sequence_families=source_sequence_families,
        candidate_sequence_families=candidate_sequence_families,
    )
    effective_n = component_geometry["post_dedup_independent_effective_n"]

    power_policy = protocol["prefrozen_power_policy"]
    required_n = required_effective_n(
        rho=power_policy["minimum_effect_at_alternative"],
        alpha=power_policy["alpha_two_sided"],
        target_power=power_policy["target_power"],
        confidence=power_policy["confidence_level"],
        max_width=power_policy["maximum_ci_full_width"],
    )
    achieved_power = fisher_power(
        effective_n,
        power_policy["minimum_effect_at_alternative"],
        power_policy["alpha_two_sided"],
    )
    achieved_width = fisher_ci_width(
        effective_n,
        power_policy["minimum_effect_at_alternative"],
        power_policy["confidence_level"],
    )
    if effective_n > 3:
        null_fisher_z_standard_error: float | None = 1.0 / math.sqrt(
            effective_n - 3.0
        )
        alternative_fisher_z_standard_error: float | None = (
            math.sqrt(
                1.0
                + power_policy["minimum_effect_at_alternative"] ** 2 / 2.0
            )
            * null_fisher_z_standard_error
        )
    else:
        null_fisher_z_standard_error = None
        alternative_fisher_z_standard_error = None

    gates: dict[str, dict[str, str]] = {}
    membership_fail = (
        fasta["duplicate_header_count"] > 0
        or fasta["invalid_sequence_count"] > 0
        or len(observed_not_intended) > 0
        or (mapping is not None and (len(fasta_not_mapping) + len(mapping_not_fasta) > 0))
        or (mapping is not None and mapping["duplicate_candidate_count"] > 0)
    )
    gates[GATE_IDS[0]] = _gate(
        FAIL if membership_fail else PASS,
        "INTENDED_DENOMINATOR_OR_JOIN_CONFLICT"
        if membership_fail
        else "INTENDED_DENOMINATOR_RETAINED_AND_OBSERVED_SUBSET_JOINED",
    )
    if mapping is None:
        mapping_reason = (
            "AUTHORITATIVE_SOURCE_MAPPING_AUTHORITY_NOT_BOUND"
            if not mapping_authority_bound
            else "BOUND_AUTHORITATIVE_SOURCE_MAPPING_INPUT_MISSING"
        )
        gates[GATE_IDS[1]] = _gate(UNKNOWN, mapping_reason)
        gates[GATE_IDS[2]] = _gate(UNKNOWN, mapping_reason)
    else:
        replay_fail = (
            mapping["malformed_row_count"] > 0
            or mapping["duplicate_candidate_count"] > 0
            or len(replay_valid_ids) != len(mapping_records)
            or candidate_fasta_mismatch_count > 0
        )
        gates[GATE_IDS[1]] = _gate(
            FAIL if replay_fail else PASS,
            "SOURCE_CANDIDATE_REPLAY_OR_FASTA_JOIN_CONFLICT"
            if replay_fail
            else "ALL_INTENDED_CANDIDATES_REPLAY_AS_NONZERO_SYNONYMOUS_SUBSTITUTIONS",
        )
        family_fail = (
            inconsistent_family_source_count > 0
            or inconsistent_family_context_count > 0
            or observed_source_id_mismatch_count > 0
            or not structurally_valid_families
        )
        gates[GATE_IDS[2]] = _gate(
            FAIL if family_fail else PASS,
            "FAMILY_SOURCE_CONTEXT_OR_MINIMUM_CANDIDATE_CONFLICT"
            if family_fail
            else "SOURCE_FAMILIES_HAVE_ONE_SOURCE_ONE_CONTEXT_AND_AT_LEAST_THREE_CANDIDATES",
        )
    unresolved_conflicts = observed["unresolved_conflicting_candidate_count"]
    duplicate_semantics_reason = (
        "DUPLICATE_MEASUREMENT_TUPLE_SEMANTICS_UNRESOLVED"
    )
    gates[GATE_IDS[3]] = _gate(
        FAIL if unresolved_conflicts else PASS,
        duplicate_semantics_reason
        if unresolved_conflicts
        else "LOG_DECAY_SLOPE_DIRECTION_SCALE_AND_SEMANTICS_FROZEN",
    )
    endpoint_fail = (
        observed["malformed_row_count"] > 0
        or observed["invalid_numeric_cell_count"] > 0
        or unresolved_conflicts > 0
        or invalid_endpoint_count > 0
        or valid_endpoint_count == 0
    )
    gates[GATE_IDS[4]] = _gate(
        FAIL if endpoint_fail else PASS,
        duplicate_semantics_reason
        if unresolved_conflicts
        else (
            "THREE_REPLICATE_SLOPE_OR_STANDARD_ERROR_UNAVAILABLE"
            if endpoint_fail
            else "THREE_BIOLOGICAL_REPLICATE_SLOPES_AND_SAMPLE_STANDARD_ERROR_COMPUTABLE"
        ),
    )
    gates[GATE_IDS[5]] = _gate(
        UNKNOWN
        if intended_not_observed or invalid_endpoint_count or unresolved_conflicts
        else PASS,
        duplicate_semantics_reason
        if unresolved_conflicts
        else (
            "PERFECT_DETECTION_MISSINGNESS_MECHANISM_NOT_CLOSED"
            if intended_not_observed or invalid_endpoint_count
            else "NO_INTENDED_MEMBER_MISSING_OR_CENSORED"
        ),
    )
    rights = protocol["rights_policy"]
    rights_pass = (
        rights["private_academic_analysis_allowed"] is True
        and rights["aggregate_derived_reporting_allowed"] is True
    )
    gates[GATE_IDS[6]] = _gate(
        PASS if rights_pass else UNKNOWN,
        "ASSET_SPECIFIC_PRIVATE_ANALYSIS_AND_AGGREGATE_REPORTING_TERMS_BOUND"
        if rights_pass
        else "ASSET_SPECIFIC_RIGHTS_NOT_BOUND",
    )
    scratch = protocol["scratch_exposure_policy"]
    scratch_pass = scratch["dataset_specific_owner_policy_status"] == "BOUND"
    gates[GATE_IDS[7]] = _gate(
        PASS if scratch_pass else UNKNOWN,
        "SCRATCH_ONLY_NO_FOUNDATION_ROUTE_BOUND"
        if scratch_pass
        else "DATASET_SPECIFIC_SCRATCH_EXPOSURE_POLICY_NOT_BOUND",
    )
    if mapping is None or gates[GATE_IDS[1]]["status"] != PASS:
        gates[GATE_IDS[8]] = _gate(UNKNOWN, "SOURCE_FAMILY_COMPONENTS_NOT_IDENTIFIABLE")
        gates[GATE_IDS[9]] = _gate(UNKNOWN, "POST_DEDUP_EFFECTIVE_N_NOT_IDENTIFIABLE")
        gates[GATE_IDS[10]] = _gate(UNKNOWN, "PREFROZEN_POWER_N_NOT_IDENTIFIABLE")
    else:
        split_pass = effective_n >= protocol["split_and_dedup_policy"][
            "minimum_components_for_three_way_split_feasibility"
        ]
        gates[GATE_IDS[8]] = _gate(
            PASS if split_pass else FAIL,
            "SOURCE_COMPONENT_SPLIT_FEASIBLE_WITHOUT_MEMBER_ASSIGNMENT_OUTPUT"
            if split_pass
            else "TOO_FEW_SOURCE_COMPONENTS_FOR_THREE_WAY_SPLIT",
        )
        gates[GATE_IDS[9]] = _gate(
            PASS if effective_n > 0 else FAIL,
            "POST_DEDUP_EFFECTIVE_N_COMPUTED_AT_SOURCE_COMPONENT_UNIT"
            if effective_n > 0
            else "NO_ELIGIBLE_POST_DEDUP_SOURCE_COMPONENT",
        )
        power_pass = (
            effective_n >= required_n
            and achieved_power >= power_policy["target_power"]
            and achieved_width <= power_policy["maximum_ci_full_width"]
        )
        gates[GATE_IDS[10]] = _gate(
            PASS if power_pass else FAIL,
            "PREFROZEN_POWER_AND_CI_WIDTH_MET"
            if power_pass
            else "PREFROZEN_POWER_OR_CI_WIDTH_NOT_MET_AT_SOURCE_GROUP_UNIT",
        )

    if any(gate["status"] == FAIL for gate in gates.values()):
        overall = STATUS_STOP
    elif any(gate["status"] == UNKNOWN for gate in gates.values()):
        overall = STATUS_BLOCKED
    else:
        overall = STATUS_READY

    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "dataset_id": DATASET_ID,
        "bioproject_id": "PRJNA856272",
        "decision_id": DECISION_ID,
        "protocol_status": protocol["protocol_status"],
        "status": overall,
        "aggregate_observation": {
            "intended_universe": {
                "reference_fasta_construct_count": fasta["record_count"],
                "authoritative_mapping_protocol_status": mapping_contract["status"],
                "source_mapping_input_provided": mapping_input_provided,
                "source_mapping_read_count": mapping_read_count,
                "authoritative_mapping_present": mapping is not None,
                "authoritative_mapping_candidate_count": (
                    mapping["unique_candidate_count"] if mapping else 0
                ),
                "observed_unique_candidate_count": observed["unique_candidate_count"],
                "intended_not_observed_count": len(intended_not_observed),
                "observed_not_intended_count": len(observed_not_intended),
                "fasta_not_mapping_count": len(fasta_not_mapping),
                "mapping_not_fasta_count": len(mapping_not_fasta),
                "duplicate_fasta_header_count": fasta["duplicate_header_count"],
                "invalid_fasta_sequence_count": fasta["invalid_sequence_count"],
            },
            "observed_asset": {
                "body_row_count": observed["body_row_count"],
                "unique_candidate_count": observed["unique_candidate_count"],
                "malformed_row_count": observed["malformed_row_count"],
                "invalid_numeric_cell_count": observed["invalid_numeric_cell_count"],
                "duplicate_measurement_conflict_count": observed[
                    "duplicate_measurement_conflict_count"
                ],
                "unresolved_conflicting_candidate_count": observed[
                    "unresolved_conflicting_candidate_count"
                ],
                "candidate_design_group_label_count_histogram": observed[
                    "candidate_design_group_label_count_histogram"
                ],
            },
            "sequence_and_edit_replay": {
                "mapping_row_count": mapping["row_count"] if mapping else 0,
                "valid_synonymous_replay_count": len(replay_valid_ids),
                "invalid_replay_reason_counts": dict(sorted(replay_reason_counts.items())),
                "nucleotide_edit_count_histogram": dict(sorted(edit_histogram.items())),
                "candidate_fasta_mismatch_count": candidate_fasta_mismatch_count,
                "sequence_length_histogram": fasta["sequence_length_histogram"],
            },
            "family_and_context": {
                "family_count": len(family_candidates),
                "family_candidate_count_histogram": dict(
                    sorted(family_size_histogram.items())
                ),
                "structurally_valid_family_count": len(structurally_valid_families),
                "eligible_family_count_after_endpoint_availability": len(
                    eligible_families
                ),
                "inconsistent_family_source_count": inconsistent_family_source_count,
                "inconsistent_family_context_count": inconsistent_family_context_count,
                "observed_source_id_mismatch_count": observed_source_id_mismatch_count,
            },
            "endpoint_and_replicates": {
                "biological_replicate_count": 3,
                "timepoint_by_replicate_observation_count": 9,
                "independent_n_per_candidate": 3,
                "valid_candidate_endpoint_and_se_count": valid_endpoint_count,
                "missing_or_censored_candidate_endpoint_count": invalid_endpoint_count,
                "standard_error_histogram": dict(sorted(se_histogram.items())),
                "endpoint_direction": "HIGHER_IS_SLOWER_DECAY_AND_GREATER_STABILITY",
            },
            "split_dedup_and_power": {
                "eligible_source_family_count_before_dedup": len(eligible_families),
                "cross_family_shared_source_id_count": component_geometry[
                    "cross_family_shared_source_id_count"
                ],
                "cross_family_duplicate_source_sequence_count": component_geometry[
                    "cross_family_duplicate_source_sequence_count"
                ],
                "exact_cross_family_duplicate_candidate_sequence_count": (
                    component_geometry[
                        "exact_cross_family_duplicate_candidate_sequence_count"
                    ]
                ),
                "source_family_component_size_histogram": component_geometry[
                    "source_family_component_size_histogram"
                ],
                "post_dedup_independent_effective_n": effective_n,
                "analysis_unit": "BIOLOGICAL_SOURCE_GROUP",
                "row_count_used_as_power_n": False,
                "nine_observations_used_as_power_n": False,
                "required_effective_n_for_both_power_and_ci_width": required_n,
                "achieved_fisher_z_power": achieved_power,
                "achieved_fisher_z_ci_full_width": achieved_width,
                "power_method": POWER_METHOD,
                "confidence_interval_method": CI_METHOD,
                "working_distribution_assumption": (
                    WORKING_DISTRIBUTION_ASSUMPTION
                ),
                "null_fisher_z_standard_error": null_fisher_z_standard_error,
                "alternative_fisher_z_standard_error": (
                    alternative_fisher_z_standard_error
                ),
                "minimum_effect_at_alternative": power_policy[
                    "minimum_effect_at_alternative"
                ],
                "target_power": power_policy["target_power"],
                "maximum_ci_full_width": power_policy["maximum_ci_full_width"],
            },
        },
        "gates": {gate_id: gates[gate_id] for gate_id in GATE_IDS},
        "internal_access_attestation": {
            "ordinary_public_assets_read_count": 2 + mapping_read_count,
            "source_mapping_input_provided_count": int(mapping_input_provided),
            "source_mapping_read_count": mapping_read_count,
            "private_or_restricted_asset_read_count": 0,
            "sealed_asset_contact_count": 0,
            "member_identifier_sequence_or_row_measurement_output_count": 0,
            "row_or_member_material_persisted_count": 0,
            "split_assignment_output_count": 0,
            "training_run_count": 0,
            "gpu_run_count": 0,
            "model_selection_count": 0,
        },
        "claim_boundary": copy.deepcopy(protocol["claim_boundary"]),
    }


def _atomic_publish(output: Path, payload: bytes) -> None:
    if output.name != REPORT_FILENAME:
        raise OutputError("output basename differs from the frozen sole artifact")
    if output.exists():
        raise OutputError("aggregate output already exists; overwrite is forbidden")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = handle.name
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if output.exists():
            raise OutputError("aggregate output appeared during atomic publish")
        os.replace(temporary, output)
        temporary = None
    finally:
        if temporary is not None:
            try:
                Path(temporary).unlink()
            except FileNotFoundError:
                pass


def _assert_output_target_available(output: Path) -> None:
    if output.name != REPORT_FILENAME:
        raise OutputError("output basename differs from the frozen sole artifact")
    if output.exists():
        raise OutputError("aggregate output already exists; overwrite is forbidden")


def execute(
    protocol_path: Path,
    observed_csv: Path,
    reference_fasta: Path,
    output: Path,
    *,
    source_mapping: Path | None = None,
    repo_root: Path | None = None,
    binding_auditor: Callable[
        [Mapping[str, Any], Path, bytes, Path], dict[str, Any]
    ] = _default_binding_auditor,
    aggregator: Callable[
        [Mapping[str, Any], Path, Path, Path | None], dict[str, Any]
    ] = aggregate,
) -> dict[str, Any]:
    protocol_bytes = protocol_path.read_bytes()
    protocol = _strict_json(protocol_bytes, label="protocol")
    _validate_protocol(protocol)
    root = Path(repo_root or PRODUCTION_REPO_ROOT)
    binding = binding_auditor(protocol, protocol_path, protocol_bytes, root)
    _assert_output_target_available(output)
    report = aggregator(protocol, observed_csv, reference_fasta, source_mapping)
    report["binding"] = binding
    report["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    payload = _json_bytes(report)
    _atomic_publish(output, payload)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--observed-perfect-csv", required=True, type=Path)
    parser.add_argument("--reference-fasta", required=True, type=Path)
    parser.add_argument("--authoritative-source-mapping", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    report = execute(
        arguments.protocol,
        arguments.observed_perfect_csv,
        arguments.reference_fasta,
        arguments.output,
        source_mapping=arguments.authoritative_source_mapping,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "protocol_status": report["protocol_status"],
                "output": str(arguments.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
