from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    ROOT
    / "scripts/route_a_v3/"
    "produce_gse200304_dec019_checkpoint_exposure_fail_current_protocol.py"
)
CONFIG = (
    ROOT
    / "configs/"
    "route_a_v3_gse200304_dec019_checkpoint_exposure_fail_current_protocol_v1.json"
)


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location("g200_checkpoint_exposure_stop", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PRODUCER = _load_module()


def _config() -> dict[str, Any]:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def _unknown_config() -> dict[str, Any]:
    config = copy.deepcopy(_config())
    PRODUCER.validate_static_config(config)
    binding = config["implementation_binding"]
    binding["status"] = PRODUCER.UNKNOWN
    binding["implementation_commit"] = PRODUCER.UNKNOWN
    binding["implementation_script_sha256"] = PRODUCER.UNKNOWN
    binding["implementation_test_sha256"] = PRODUCER.UNKNOWN
    PRODUCER.validate_static_config(config)
    return config


def _bound_config() -> dict[str, Any]:
    config = _unknown_config()
    binding = config["implementation_binding"]
    binding["status"] = PRODUCER.BOUND
    binding["implementation_commit"] = "1" * 40
    binding["implementation_script_sha256"] = PRODUCER.sha256(SCRIPT.read_bytes())
    binding["implementation_test_sha256"] = PRODUCER.sha256(
        Path(__file__).read_bytes()
    )
    PRODUCER.validate_static_config(config)
    return config


def _rehash_core(config: dict[str, Any]) -> None:
    config["implementation_binding"]["config_core_sha256"] = (
        PRODUCER.config_core_sha256(config)
    )


def _string_values(value: Any) -> list[str]:
    if isinstance(value, dict):
        return [item for child in value.values() for item in _string_values(child)]
    if isinstance(value, list):
        return [item for child in value for item in _string_values(child)]
    return [value] if isinstance(value, str) else []


def test_disk_binding_and_normalized_unknown_i_are_valid() -> None:
    disk_config = _config()
    PRODUCER.validate_static_config(disk_config)
    disk_binding = disk_config["implementation_binding"]
    if disk_binding["status"] == PRODUCER.BOUND:
        assert disk_binding["implementation_script_sha256"] == PRODUCER.sha256(
            SCRIPT.read_bytes()
        )
        assert disk_binding["implementation_test_sha256"] == PRODUCER.sha256(
            Path(__file__).read_bytes()
        )
        PRODUCER.validate_implementation_binding(disk_config, repo_root=ROOT)
    else:
        assert disk_binding["status"] == PRODUCER.UNKNOWN
        assert disk_binding["implementation_commit"] == PRODUCER.UNKNOWN
        assert disk_binding["implementation_script_sha256"] == PRODUCER.UNKNOWN
        assert disk_binding["implementation_test_sha256"] == PRODUCER.UNKNOWN

    config = _unknown_config()
    binding = config["implementation_binding"]
    assert binding["status"] == PRODUCER.UNKNOWN
    assert binding["implementation_commit"] == PRODUCER.UNKNOWN
    assert binding["implementation_script_sha256"] == PRODUCER.UNKNOWN
    assert binding["implementation_test_sha256"] == PRODUCER.UNKNOWN
    assert binding["config_core_sha256"] == PRODUCER.config_core_sha256(config)

    with pytest.raises(
        PRODUCER.BindingError,
        match="CHECKPOINT_EXPOSURE_FAIL_RECORD_IMPLEMENTATION_BINDING_UNKNOWN_NOT_ASSERTED",
    ):
        PRODUCER.produce(config, repo_root=ROOT)


def test_i_to_b_transition_is_exactly_config_only_four_scalars() -> None:
    i_config = _unknown_config()
    b_config = _bound_config()
    binding = b_config["implementation_binding"]
    assert PRODUCER._scalar_differences(i_config, b_config) == set(
        PRODUCER.BINDING_SCALAR_PATHS
    )
    PRODUCER.validate_i_to_b_transition(
        i_config,
        b_config,
        implementation_commit=binding["implementation_commit"],
        implementation_script_sha256=binding["implementation_script_sha256"],
        implementation_test_sha256=binding["implementation_test_sha256"],
    )

    changed_science = copy.deepcopy(b_config)
    changed_science["current_protocol"]["empty_checkpoint_set_can_pass"] = True
    _rehash_core(changed_science)
    with pytest.raises(PRODUCER.ProducerError, match="frozen protocol"):
        PRODUCER.validate_i_to_b_transition(
            i_config,
            changed_science,
            implementation_commit=binding["implementation_commit"],
            implementation_script_sha256=binding["implementation_script_sha256"],
            implementation_test_sha256=binding["implementation_test_sha256"],
        )


def test_report_is_exact_aggregate_only_fail_current_protocol() -> None:
    config = _bound_config()
    report = json.loads(PRODUCER.produce(config, repo_root=ROOT))

    assert report["status"] == PRODUCER.FAIL_CURRENT_PROTOCOL
    freeze = report["checkpoint_set_freeze"]
    assert freeze["current_public_executable_foundation_checkpoint_ids"] == []
    assert freeze["current_public_executable_foundation_checkpoint_count"] == 0
    assert freeze["audited_checkpoint_count"] == 0
    assert freeze["considered_candidate_family_count"] == 4
    assert freeze["task_mismatch_candidate_family_count"] == 4

    assert [
        item["candidate_family"] for item in report["candidate_task_reviews"]
    ] == PRODUCER.CANDIDATE_FAMILIES
    assert all(
        item["task_match_assessment"] == {
            "status": "PAPER_ONLY_TASK_MISMATCH",
            "evidence_class": "REASONED_INFERENCE",
            "reason": item["task_match_assessment"]["reason"],
        }
        and item["current_public_executable_checkpoint_selected"] is False
        and all(
            fact["evidence_class"] == "CONFIRMED_FACT"
            for fact in item["confirmed_source_facts"]
        )
        for item in report["candidate_task_reviews"]
    )

    fact_ids = {
        fact["fact_id"]
        for item in report["candidate_task_reviews"]
        for fact in item["confirmed_source_facts"]
    }
    assert {
        "PRIMARY_MODEL_TASK_IS_5PRIME_UTR_TO_MEAN_RIBOSOME_LOADING",
        "PRIMARY_MRL_DATASET_IS_GSE114002",
        "PRIMARY_INPUT_IS_FIXED_100_BP_5PRIME_UTR_FOR_TE_TRAINING",
        "PRIMARY_TE_TARGET_IS_RIBO_SEQ_RPKM_DIVIDED_BY_RNA_SEQ_RPKM",
        "PRIMARY_PUBLIC_SURFACE_DOES_NOT_IDENTIFY_ONE_CANONICAL_GSE200304_TOTALPOLY_REGRESSION_HEAD",
        "PRIMARY_PRETRAINING_INTERFACE_IS_MASKED_LANGUAGE_MODELING_AND_FEATURE_REPRESENTATION",
        "PRIMARY_PUBLIC_SURFACE_DOES_NOT_IDENTIFY_A_TOTALPOLY_REGRESSION_HEAD",
        "PRIMARY_MODEL_SCOPE_IS_FULL_MATURE_RNA",
        "PRIMARY_INTERFACE_TREATS_INCOMPLETE_FRAGMENTS_AS_OUT_OF_DISTRIBUTION",
        "PRIMARY_CHECKPOINT_INTERFACE_RETURNS_REPRESENTATIONS_OR_MLM_LOGITS_ONLY",
    } <= fact_ids

    protocol = report["current_protocol"]
    assert protocol["nonwaivable"] is True
    assert protocol["minimum_audited_checkpoint_count_for_pass"] == 1
    assert protocol["observed_audited_checkpoint_count"] == 0
    assert protocol["empty_checkpoint_set_can_pass"] is False
    assert protocol["pass_under_current_protocol"] is False

    gate = report["current_exposure_gate"]
    assert gate["status"] == PRODUCER.UNKNOWN
    assert gate["facts"] is None
    assert gate["unknown_fields"] == PRODUCER.UNKNOWN_GATE_FIELDS
    assert gate["exact_blocker"] == PRODUCER.EXACT_BLOCKER

    projection = report["gate_and_authorization_projection"]
    assert projection["qualified"] is False
    assert projection["scientific_claim_status"] == "NOT_ESTABLISHED"
    assert projection["training_allowed"] is False
    assert projection["model_selection_allowed"] is False
    assert projection["next_phase_authorized"] is False
    assert "NOT_APPLICABLE" not in _string_values(report)


def test_execution_boundary_is_not_run_and_not_contacted() -> None:
    report = json.loads(PRODUCER.produce(_bound_config(), repo_root=ROOT))
    boundary = report["execution_boundary"]
    assert boundary["aggregate_only"] is True
    assert boundary["ordinary_public_source_locators_only"] is True
    assert boundary["dataset_payload_opened"] is False
    assert boundary["sequence_payload_opened"] is False
    assert boundary["row_level_payload_opened"] is False
    assert boundary["checkpoint_weights_downloaded"] is False
    assert boundary["checkpoint_artifact_payload_opened"] is False
    assert boundary["model_execution_count"] == 0
    assert boundary["exact_member_exposure_audit_status"] == "NOT_RUN"
    assert boundary["near_duplicate_exposure_audit_status"] == "NOT_RUN"
    assert boundary["training_run_count"] == 0
    assert boundary["model_selection_run_count"] == 0
    assert boundary["restricted_or_sealed_contact"] is False
    assert boundary["gse246381_contact"] is False


def test_producer_reads_only_its_bound_script_and_test(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _bound_config()
    observed: list[Path] = []
    real_read_bytes = Path.read_bytes

    def recording_read_bytes(path: Path) -> bytes:
        observed.append(path)
        return real_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", recording_read_bytes)
    PRODUCER.produce(config, repo_root=ROOT)
    assert [path.resolve() for path in observed] == [
        SCRIPT.resolve(),
        Path(__file__).resolve(),
    ]


def test_false_pass_or_not_applicable_rewrite_is_rejected() -> None:
    false_pass = _bound_config()
    false_pass["current_protocol"]["pass_under_current_protocol"] = True
    false_pass["current_protocol"]["empty_checkpoint_set_can_pass"] = True
    false_pass["gate_and_authorization_projection"]["current_exposure_gate"][
        "status"
    ] = "PASS"
    _rehash_core(false_pass)
    with pytest.raises(PRODUCER.ProducerError):
        PRODUCER.validate_static_config(false_pass)

    invented_na = _bound_config()
    invented_na["candidate_task_reviews"][0]["task_match_assessment"][
        "status"
    ] = "NOT_APPLICABLE"
    _rehash_core(invented_na)
    with pytest.raises(PRODUCER.ProducerError, match="task-match status"):
        PRODUCER.validate_static_config(invented_na)


def test_writer_emits_one_normal_json_without_overwrite(tmp_path: Path) -> None:
    payload = PRODUCER.produce(_bound_config(), repo_root=ROOT)
    output = tmp_path / PRODUCER.OUTPUT_BASENAME
    PRODUCER.write_report(output, payload)
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == (
        PRODUCER.FAIL_CURRENT_PROTOCOL
    )
    assert [path.name for path in tmp_path.iterdir()] == [PRODUCER.OUTPUT_BASENAME]
    with pytest.raises(PRODUCER.ProducerError, match="already exists"):
        PRODUCER.write_report(output, payload)
