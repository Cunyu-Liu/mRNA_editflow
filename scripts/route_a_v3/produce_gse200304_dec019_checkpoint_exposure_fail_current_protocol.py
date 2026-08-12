#!/usr/bin/env python3
"""Produce the aggregate-only GSE200304 checkpoint-exposure protocol stop.

This producer consumes only its frozen JSON configuration and its own
implementation binding.  It does not open dataset payloads, sequences, model
artifacts, or weights, and it does not run a model or emit a consumer gate
record.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence


UNKNOWN = "UNKNOWN_NOT_ASSERTED"
BOUND = "BOUND"
FAIL_CURRENT_PROTOCOL = "FAIL_CURRENT_PROTOCOL"
SCHEMA_VERSION = (
    "route_a_v3_gse200304_dec019_checkpoint_exposure_fail_current_protocol.v1"
)
REPORT_SCHEMA_VERSION = (
    "route_a_v3_gse200304_dec019_checkpoint_exposure_fail_current_protocol_report.v1"
)
PROTOCOL_ID = (
    "ROUTE_A_V3_GSE200304_DEC019_CHECKPOINT_EXPOSURE_FAIL_CURRENT_PROTOCOL_V1"
)
RECORD_TYPE = (
    "GSE200304_DEC019_CHECKPOINT_EXPOSURE_FAIL_CURRENT_PROTOCOL_"
    "AGGREGATE_ONLY_V1"
)
CONTRACT_ID = "mrna_xeditflow_route_a_v3"
DATASET_ID = "GSE200304"
DECISION_ID = "V3-DEC-019"
GATE_ID = "CHECKPOINT_SPECIFIC_EXPOSURE"
EXACT_BLOCKER = "CHECKPOINT_SPECIFIC_EXPOSURE_NOT_PASS"
OUTPUT_BASENAME = (
    "GSE200304_DEC019_CHECKPOINT_EXPOSURE_FAIL_CURRENT_PROTOCOL.json"
)
CONFIG_REPO_PATH = (
    "configs/"
    "route_a_v3_gse200304_dec019_checkpoint_exposure_fail_current_protocol_v1.json"
)
SCRIPT_REPO_PATH = (
    "scripts/route_a_v3/"
    "produce_gse200304_dec019_checkpoint_exposure_fail_current_protocol.py"
)
TEST_REPO_PATH = (
    "tests/route_a_v3/"
    "test_produce_gse200304_dec019_checkpoint_exposure_fail_current_protocol.py"
)
BINDING_SCALAR_PATHS = [
    "implementation_binding.status",
    "implementation_binding.implementation_commit",
    "implementation_binding.implementation_script_sha256",
    "implementation_binding.implementation_test_sha256",
]
REQUIRED_TRUE_FACT_FIELDS = [
    "checkpoint_ids_and_revisions_frozen",
    "checkpoint_artifact_digests_bound",
    "exact_member_exposure_audit_pass",
    "near_duplicate_exposure_audit_pass",
]
UNKNOWN_GATE_FIELDS = [
    "audited_checkpoint_count",
    "checkpoint_artifact_digests_bound",
    "checkpoint_ids_and_revisions_frozen",
    "exact_member_exposure_audit_pass",
    "near_duplicate_exposure_audit_pass",
]
CANDIDATE_FAMILIES = [
    "OPTIMUS_5PRIME",
    "UTR_LM",
    "MRNABERT",
    "ORTHRUS",
]
SOURCE_IDS = [
    "GSE200304_PRIMARY_ARTICLE",
    "OPTIMUS_PRIMARY_ARTICLE",
    "GSE114002_GEO_SERIES",
    "UTR_LM_PRIMARY_ARTICLE",
    "MRNABERT_PRIMARY_ARTICLE",
    "ORTHRUS_PRIMARY_ARTICLE",
    "ORTHRUS_AUTHOR_CODE",
]
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


class ProducerError(RuntimeError):
    """The frozen record contract or implementation binding differs."""


class BindingError(ProducerError):
    """The normal UNKNOWN-I to config-only-B lifecycle is not complete."""


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, indent=2)
        + "\n"
    ).encode("utf-8")


def load_config(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProducerError(f"cannot load config: {path}") from exc
    if type(value) is not dict:
        raise ProducerError("config root is not an object")
    return value


def config_core_projection(config: Mapping[str, Any]) -> dict[str, Any]:
    projected = copy.deepcopy(dict(config))
    projected.pop("implementation_binding", None)
    return projected


def config_core_sha256(config: Mapping[str, Any]) -> str:
    return sha256(json_bytes(config_core_projection(config)))


def _scalar_differences(before: Any, after: Any, prefix: str = "") -> set[str]:
    if type(before) is not type(after):
        return {prefix}
    if isinstance(before, dict):
        if set(before) != set(after):
            return {prefix}
        result: set[str] = set()
        for key in before:
            child = f"{prefix}.{key}" if prefix else key
            result.update(_scalar_differences(before[key], after[key], child))
        return result
    if isinstance(before, list):
        if len(before) != len(after):
            return {prefix}
        result = set()
        for index, (old, new) in enumerate(zip(before, after)):
            result.update(_scalar_differences(old, new, f"{prefix}[{index}]"))
        return result
    return set() if before == after else {prefix}


def _expect(actual: Any, expected: Any, *, label: str) -> None:
    if type(actual) is not type(expected) or actual != expected:
        raise ProducerError(f"{label} differs from the frozen protocol")


def validate_static_config(config: Mapping[str, Any]) -> None:
    for key, expected in {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "contract_id": CONTRACT_ID,
        "phase_id": "A1",
        "dataset_id": DATASET_ID,
        "decision_id": DECISION_ID,
    }.items():
        _expect(config.get(key), expected, label=key)

    binding = config.get("implementation_binding")
    if type(binding) is not dict:
        raise ProducerError("implementation binding is absent")
    _expect(
        binding.get("binding_scheme"),
        "CONFIG_ONLY_POST_IMPLEMENTATION_BINDING_V1",
        label="implementation binding scheme",
    )
    _expect(
        binding.get("blocker_if_unbound"),
        "CHECKPOINT_EXPOSURE_FAIL_RECORD_IMPLEMENTATION_BINDING_UNKNOWN_NOT_ASSERTED",
        label="implementation binding blocker",
    )
    _expect(
        binding.get("implementation_script_path"),
        SCRIPT_REPO_PATH,
        label="implementation script path",
    )
    _expect(
        binding.get("implementation_test_path"),
        TEST_REPO_PATH,
        label="implementation test path",
    )
    _expect(
        binding.get("unknown_to_bound_scalar_paths"),
        BINDING_SCALAR_PATHS,
        label="binding scalar paths",
    )
    if HEX64.fullmatch(str(binding.get("config_core_sha256"))) is None:
        raise BindingError("config core SHA is not bound")
    _expect(
        config_core_sha256(config),
        binding["config_core_sha256"],
        label="config core SHA",
    )
    if binding.get("status") == UNKNOWN:
        for key in (
            "implementation_commit",
            "implementation_script_sha256",
            "implementation_test_sha256",
        ):
            _expect(binding.get(key), UNKNOWN, label=f"unbound {key}")
    elif binding.get("status") == BOUND:
        if HEX40.fullmatch(str(binding.get("implementation_commit"))) is None:
            raise BindingError("implementation commit is not bound")
        for key in (
            "implementation_script_sha256",
            "implementation_test_sha256",
        ):
            if HEX64.fullmatch(str(binding.get(key))) is None:
                raise BindingError(f"{key} is not bound")
    else:
        raise BindingError("implementation binding status is outside the closed enum")

    repository = config.get("repository_authority")
    if type(repository) is not dict:
        raise ProducerError("repository authority is absent")
    _expect(
        repository.get("lifecycle_scheme"),
        "UNKNOWN_I_TO_CONFIG_ONLY_B_V1",
        label="repository lifecycle",
    )
    _expect(
        repository.get("implementation_commit_exact_changed_paths"),
        [CONFIG_REPO_PATH, SCRIPT_REPO_PATH, TEST_REPO_PATH],
        label="implementation exact-three path set",
    )
    _expect(
        repository.get("binding_commit_exact_changed_paths"),
        [CONFIG_REPO_PATH],
        label="binding exact-one path set",
    )

    taxonomy = config.get("evidence_taxonomy")
    if type(taxonomy) is not dict:
        raise ProducerError("evidence taxonomy is absent")
    _expect(
        taxonomy.get("reasoned_inference_may_be_promoted_to_confirmed"),
        False,
        label="inference promotion policy",
    )
    _expect(
        taxonomy.get("unknown_may_be_encoded_as_not_applicable"),
        False,
        label="unknown-to-not-applicable policy",
    )

    target = config.get("target_task")
    if type(target) is not dict:
        raise ProducerError("target task is absent")
    _expect(target.get("evidence_class"), "CONFIRMED_FACT", label="target evidence")
    _expect(target.get("dataset_id"), DATASET_ID, label="target dataset")
    _expect(target.get("reported_endpoint"), "TotalPoly", label="target endpoint")
    _expect(
        target.get("required_candidate_interface"),
        "EXECUTABLE_GSE200304_TOTALPOLY_REGRESSION_HEAD",
        label="target interface",
    )

    sources = config.get("official_primary_sources")
    if type(sources) is not list:
        raise ProducerError("official primary sources are absent")
    _expect(
        [item.get("source_id") for item in sources if type(item) is dict],
        SOURCE_IDS,
        label="official source identities",
    )
    if any(
        type(item) is not dict
        or not str(item.get("url", "")).startswith("https://")
        or item.get("source_type")
        not in {
            "PRIMARY_RESEARCH_ARTICLE",
            "PRIMARY_PUBLIC_SERIES_RECORD",
            "AUTHOR_CODE_REPOSITORY",
        }
        for item in sources
    ):
        raise ProducerError("official source registry contains a non-primary locator")

    reviews = config.get("candidate_task_reviews")
    if type(reviews) is not list:
        raise ProducerError("candidate task reviews are absent")
    _expect(
        [item.get("candidate_family") for item in reviews if type(item) is dict],
        CANDIDATE_FAMILIES,
        label="candidate family set",
    )
    for review in reviews:
        if type(review) is not dict:
            raise ProducerError("candidate review is not an object")
        facts = review.get("confirmed_source_facts")
        if type(facts) is not list or not facts:
            raise ProducerError("candidate confirmed source facts are absent")
        if any(
            type(fact) is not dict
            or fact.get("evidence_class") != "CONFIRMED_FACT"
            or type(fact.get("source_ids")) is not list
            or not fact["source_ids"]
            or any(source_id not in SOURCE_IDS for source_id in fact["source_ids"])
            for fact in facts
        ):
            raise ProducerError("candidate confirmed-source fact classification differs")
        assessment = review.get("task_match_assessment")
        if type(assessment) is not dict:
            raise ProducerError("candidate task-match assessment is absent")
        _expect(
            assessment.get("status"),
            "PAPER_ONLY_TASK_MISMATCH",
            label="candidate task-match status",
        )
        _expect(
            assessment.get("evidence_class"),
            "REASONED_INFERENCE",
            label="candidate task-match evidence class",
        )
        _expect(
            review.get("current_public_executable_checkpoint_selected"),
            False,
            label="candidate checkpoint selection",
        )

    freeze = config.get("checkpoint_set_freeze")
    if type(freeze) is not dict:
        raise ProducerError("checkpoint-set freeze is absent")
    expected_freeze = {
        "freeze_status": "FROZEN_CURRENT_PUBLIC_EXECUTABLE_FOUNDATION_CHECKPOINT_SET",
        "selection_basis_evidence_class": "REASONED_INFERENCE",
        "considered_candidate_family_count": 4,
        "task_mismatch_candidate_family_count": 4,
        "current_public_executable_foundation_checkpoint_ids": [],
        "current_public_executable_foundation_checkpoint_count": 0,
        "audited_checkpoint_count": 0,
    }
    _expect(dict(freeze), expected_freeze, label="checkpoint-set freeze")

    protocol = config.get("current_protocol")
    expected_protocol = {
        "gate_id": GATE_ID,
        "nonwaivable": True,
        "required_true_fact_fields": REQUIRED_TRUE_FACT_FIELDS,
        "minimum_audited_checkpoint_count_for_pass": 1,
        "observed_audited_checkpoint_count": 0,
        "empty_checkpoint_set_can_pass": False,
        "pass_under_current_protocol": False,
        "evidence_record_status": FAIL_CURRENT_PROTOCOL,
    }
    _expect(protocol, expected_protocol, label="current protocol stop")

    projection = config.get("gate_and_authorization_projection")
    if type(projection) is not dict:
        raise ProducerError("gate and authorization projection is absent")
    expected_gate = {
        "gate_id": GATE_ID,
        "status": UNKNOWN,
        "facts": None,
        "unknown_fields": UNKNOWN_GATE_FIELDS,
        "exact_blocker": EXACT_BLOCKER,
    }
    _expect(
        projection.get("current_exposure_gate"),
        expected_gate,
        label="current exposure gate",
    )
    for key, expected in {
        "qualified": False,
        "scientific_claim_status": "NOT_ESTABLISHED",
        "training_allowed": False,
        "model_selection_allowed": False,
        "next_phase_authorized": False,
    }.items():
        _expect(projection.get(key), expected, label=key)

    boundary = config.get("execution_boundary")
    if type(boundary) is not dict:
        raise ProducerError("execution boundary is absent")
    for key in (
        "dataset_payload_opened",
        "sequence_payload_opened",
        "row_level_payload_opened",
        "checkpoint_weights_downloaded",
        "checkpoint_artifact_payload_opened",
        "restricted_or_sealed_contact",
        "gse246381_contact",
    ):
        _expect(boundary.get(key), False, label=f"execution boundary {key}")
    for key in (
        "model_execution_count",
        "training_run_count",
        "model_selection_run_count",
    ):
        _expect(boundary.get(key), 0, label=f"execution boundary {key}")
    for key in (
        "exact_member_exposure_audit_status",
        "near_duplicate_exposure_audit_status",
    ):
        _expect(boundary.get(key), "NOT_RUN", label=f"execution boundary {key}")
    _expect(boundary.get("aggregate_only"), True, label="aggregate-only boundary")

    output = config.get("output_contract")
    expected_output = {
        "allowed_basename": OUTPUT_BASENAME,
        "record_type": RECORD_TYPE,
        "aggregate_only": True,
        "single_json_record": True,
        "consumer_gate_record_emitted": False,
        "amendment_modified": False,
        "activation_modified": False,
        "adjudicator_modified": False,
    }
    _expect(output, expected_output, label="output contract")


def validate_implementation_binding(
    config: Mapping[str, Any], *, repo_root: Path
) -> None:
    binding = config["implementation_binding"]
    if binding["status"] != BOUND:
        raise BindingError(binding["blocker_if_unbound"])
    if HEX40.fullmatch(str(binding["implementation_commit"])) is None:
        raise BindingError("implementation commit is not bound")
    for key, repo_path in (
        ("implementation_script_sha256", SCRIPT_REPO_PATH),
        ("implementation_test_sha256", TEST_REPO_PATH),
    ):
        path = repo_root / repo_path
        try:
            observed = sha256(path.read_bytes())
        except OSError as exc:
            raise BindingError(f"cannot read bound implementation file: {repo_path}") from exc
        _expect(observed, binding[key], label=key)


def validate_i_to_b_transition(
    i_config: Mapping[str, Any],
    b_config: Mapping[str, Any],
    *,
    implementation_commit: str,
    implementation_script_sha256: str,
    implementation_test_sha256: str,
) -> None:
    validate_static_config(i_config)
    validate_static_config(b_config)
    if i_config["implementation_binding"]["status"] != UNKNOWN:
        raise BindingError("I config is not UNKNOWN-bound")
    if b_config["implementation_binding"]["status"] != BOUND:
        raise BindingError("B config is not BOUND")
    if _scalar_differences(i_config, b_config) != set(BINDING_SCALAR_PATHS):
        raise BindingError("I-to-B differences are not the exact four binding scalars")
    expected = {
        "implementation_commit": implementation_commit,
        "implementation_script_sha256": implementation_script_sha256,
        "implementation_test_sha256": implementation_test_sha256,
    }
    for key, value in expected.items():
        _expect(
            b_config["implementation_binding"][key],
            value,
            label=f"B {key}",
        )
    _expect(
        config_core_sha256(i_config),
        config_core_sha256(b_config),
        label="I/B config core",
    )


def build_report(config: Mapping[str, Any]) -> dict[str, Any]:
    """Project only the frozen aggregate evidence; no source payload is opened."""

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "record_id": "GSE200304_DEC019_CHECKPOINT_EXPOSURE_FAIL_CURRENT_PROTOCOL_V1",
        "record_type": RECORD_TYPE,
        "contract_id": CONTRACT_ID,
        "phase_id": "A1",
        "dataset_id": DATASET_ID,
        "decision_id": DECISION_ID,
        "status": FAIL_CURRENT_PROTOCOL,
        "evidence_taxonomy": copy.deepcopy(config["evidence_taxonomy"]),
        "target_task": copy.deepcopy(config["target_task"]),
        "official_primary_sources": copy.deepcopy(config["official_primary_sources"]),
        "candidate_task_reviews": copy.deepcopy(config["candidate_task_reviews"]),
        "checkpoint_set_freeze": copy.deepcopy(config["checkpoint_set_freeze"]),
        "current_protocol": copy.deepcopy(config["current_protocol"]),
        "current_exposure_gate": copy.deepcopy(
            config["gate_and_authorization_projection"]["current_exposure_gate"]
        ),
        "gate_and_authorization_projection": copy.deepcopy(
            config["gate_and_authorization_projection"]
        ),
        "execution_boundary": copy.deepcopy(config["execution_boundary"]),
        "claim_boundary": config["claim_boundary"],
        "producer_binding": {
            "status": BOUND,
            "implementation_commit": config["implementation_binding"][
                "implementation_commit"
            ],
        },
    }


def produce(config: Mapping[str, Any], *, repo_root: Path) -> bytes:
    validate_static_config(config)
    validate_implementation_binding(config, repo_root=repo_root)
    return json_bytes(build_report(config))


def write_report(output_path: Path, payload: bytes) -> None:
    if output_path.name != OUTPUT_BASENAME:
        raise ProducerError(f"output basename must be {OUTPUT_BASENAME}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output_path.open("xb") as handle:
            handle.write(payload)
    except FileExistsError as exc:
        raise ProducerError("output already exists; no overwrite is permitted") from exc
    except OSError as exc:
        raise ProducerError(f"cannot write output: {output_path}") from exc


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    payload = produce(config, repo_root=args.repo_root)
    write_report(args.output, payload)
    print(
        json.dumps(
            {
                "status": FAIL_CURRENT_PROTOCOL,
                "current_exposure_gate_status": UNKNOWN,
                "exact_blocker": EXACT_BLOCKER,
                "audited_checkpoint_count": 0,
                "training_allowed": False,
                "model_selection_allowed": False,
                "next_phase_authorized": False,
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
