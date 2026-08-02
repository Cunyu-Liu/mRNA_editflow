"""MK0-08/09/10 CUDA fail-closed, role and acceptance-manifest tests."""

from __future__ import annotations

import hashlib
import inspect
import json
import math
from pathlib import Path
import re

import pytest
import yaml

from mrna_editflow.core.mk0.acceptance import (
    GateResult,
    aggregate_acceptance,
    canonical_json_bytes,
    sha256_bytes,
    verify_artifact_binding,
)
from mrna_editflow.core.mk0.critic_boundary import (
    base_generation_without_critic,
    reject_final_evaluator_as_guidance,
)
from mrna_editflow.core.mk0.foundation_fusion import (
    FoundationFusionRateField,
    OfficialPaperRateAdapter,
    require_neural_cuda,
)
from mrna_editflow.core.mk0.training_boundary import (
    EditFlowTrainingExample,
    canonical_rate_input_bytes,
    rate_input_state,
)
from mrna_editflow.core.mk0.types import ActionType, AtomicAction, EditState

from .conftest import SEED


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "math" / "math_kernel_v1.yaml"
HEX64 = re.compile(r"^[0-9a-f]{64}$")

MANDATORY_STATIC_FILES = (
    "docs/math/mk0_original_vs_extension_matrix.md",
    "docs/math/mk0_state_action_spec.md",
    "docs/math/mk0_coupling_probability_path.md",
    "docs/math/mk0_bregman_derivation.md",
    "docs/math/mk0_stop_semantics.md",
    "docs/math/mk0_sampler_semantics.md",
    "docs/math/mk0_foundation_fusion.md",
    "docs/math/mk0_critic_boundary.md",
    "configs/math/math_kernel_v1.yaml",
    "schemas/edit_state_v1.schema.json",
    "schemas/edit_action_v1.schema.json",
    "schemas/edit_trajectory_v1.schema.json",
    "schemas/termination_event_v1.schema.json",
    "schemas/coupling_manifest_v1.schema.json",
)


def _gate(gate_id: str, *, passed: bool = True) -> GateResult:
    return GateResult(
        gate_id=gate_id,
        name=f"gate_{gate_id}",
        passed=passed,
        test_domain="frozen_tiny_or_sampled_domain",
        exhaustive_or_sampled="exhaustive",
        sample_count=1,
        dtype="float64",
        atol=1.0e-10,
        rtol=1.0e-8,
        seed=SEED,
        failure_count=0 if passed else 1,
        denominator=1,
        artifact_path=f"artifacts/mk0/{gate_id}.json",
        artifact_sha256="a" * 64,
    )


def test_neural_cuda_guard_rejects_explicit_cpu_without_forward() -> None:
    with pytest.raises(RuntimeError, match="GPU-only|CUDA|CPU fallback"):
        require_neural_cuda("cpu")


def test_foundation_rate_interface_is_inference_visible_only_and_dynamic() -> None:
    fields = set(FoundationFusionRateField.inference_signature_fields)
    assert {
        "source",
        "current",
        "M_run",
        "region",
        "context",
        "target_condition",
        "time",
        "remaining_budget",
        "h_run",
    } <= fields
    assert (
        not {
            "Z_aux",
            "target_sequence",
            "target_alignment",
            "remaining_target_edits",
        }
        & fields
    )
    forward_parameters = set(
        inspect.signature(FoundationFusionRateField.forward).parameters
    )
    assert forward_parameters == {"self", "state", "time", "actions"}

    # The reference implementation must explicitly bypass the source cache for
    # dynamic current encoding, preventing a source-only stale state.
    source = inspect.getsource(FoundationFusionRateField._encoded_state)
    assert "state.source, source_cache=True" in source
    assert "state.current, source_cache=False" in source

    local = inspect.getsource(FoundationFusionRateField._local_representations)
    assert "current_tokens[position]" in local
    assert "aligned_tokens[position]" in local
    assert "_gap_representation(current_tokens, gap)" in local
    assert "_gap_representation(aligned_tokens, gap)" in local


def test_training_auxiliary_replacement_cannot_change_canonical_rate_input() -> None:
    state = EditState.initial(
        "ACGU",
        context={"assay": "synthetic", "endpoint": "engineering"},
        target_condition="increase",
    )
    left = EditFlowTrainingExample(
        state,
        training_auxiliary={
            "target_sequence": "UUUU",
            "target_alignment": [["A", "U"]],
            "remaining_target_edits": 4,
        },
    )
    right = EditFlowTrainingExample(
        state,
        training_auxiliary={
            "target_sequence": "CCCC",
            "target_alignment": [["A", "C"]],
            "remaining_target_edits": 1,
        },
    )
    assert rate_input_state(left) is state
    assert rate_input_state(right) is state
    assert canonical_rate_input_bytes(left) == canonical_rate_input_bytes(right)
    assert b"target_sequence" not in canonical_rate_input_bytes(left)


def test_foundation_constructor_requires_a_real_encoder_not_a_placeholder() -> None:
    signature = inspect.signature(FoundationFusionRateField.__init__)
    foundation = signature.parameters["foundation"]
    tokenizer = signature.parameters["tokenizer"]
    assert foundation.default is inspect.Parameter.empty
    assert tokenizer.default is inspect.Parameter.empty
    module_source = inspect.getsource(inspect.getmodule(FoundationFusionRateField))
    assert "class Placeholder" not in module_source
    assert "class Dummy" not in module_source

    paper_source = inspect.getsource(OfficialPaperRateAdapter)
    assert 'OFFICIAL_CLASS = "UtrLmModel"' in paper_source
    assert "placeholder_foundation_forward_calls" in paper_source
    assert "placeholder/project-local foundations are forbidden" in paper_source


def test_all_mandatory_static_mk0_contract_files_exist_and_are_nonempty() -> None:
    for relative in MANDATORY_STATIC_FILES:
        path = ROOT / relative
        assert path.is_file(), relative
        assert path.stat().st_size > 0, relative


def test_base_generation_runs_without_critic_and_records_no_role_queries() -> None:
    def stop_only(_state: EditState, _time: float):
        return {AtomicAction(ActionType.STOP): 8.0}

    result, audit = base_generation_without_critic(
        EditState.initial("AC", budget=2),
        stop_only,
        step_size=0.05,
        stability_hazard=0.05,
        min_length=1,
        max_length=6,
        seed=SEED,
    )
    assert result.final_state.termination_reason is not None
    assert audit.critic_present is False
    assert audit.guidance_queries == 0
    assert audit.final_evaluator_queries == 0
    assert audit.pass_no_final_evaluator_guidance


def test_final_evaluator_cannot_be_guidance_or_selector() -> None:
    evaluator = lambda _sequence: 1.0
    with pytest.raises(PermissionError):
        reject_final_evaluator_as_guidance(evaluator, as_guidance=True)
    reject_final_evaluator_as_guidance(evaluator, as_guidance=False)
    reject_final_evaluator_as_guidance(None, as_guidance=True)


@pytest.mark.parametrize(
    "forbidden_name",
    (
        "critic",
        "evaluator",
        "final_evaluator",
        "guidance",
        "reward",
        "reranker",
        "selector",
        "score_fn",
    ),
)
def test_base_generation_rejects_role_bearing_keyword_injection(
    forbidden_name: str,
) -> None:
    def stop_only(_state: EditState, _time: float):
        return {AtomicAction(ActionType.STOP): 8.0}

    with pytest.raises(PermissionError):
        base_generation_without_critic(
            EditState.initial("AC", budget=2),
            stop_only,
            step_size=0.05,
            stability_hazard=0.05,
            min_length=1,
            max_length=6,
            seed=SEED,
            **{forbidden_name: object()},
        )


def test_math_kernel_config_freezes_required_scientific_and_numerical_boundaries() -> (
    None
):
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["schema_version"] == "math_kernel_v1"
    assert config["phase"] == "MK0"
    assert config["evidence_level"] == "E0_MATH_ENGINEERING_ONLY"
    assert HEX64.fullmatch(config["contract"]["sha256"])
    assert config["time_direction"] == "source_at_0_to_target_at_1"
    assert config["scope"]["performance_claims_allowed"] is False
    assert config["scope"]["biological_trajectory_claims_allowed"] is False
    assert config["scope"]["exact_gillespie_claims_allowed"] is False
    assert config["scope"]["final_label_access_allowed"] is False
    assert config["numeric"]["cpu"]["dtype"] == "float64"
    assert config["numeric"]["cpu"]["atol"] == 1.0e-10
    assert config["numeric"]["cpu"]["rtol"] == 1.0e-8
    assert config["numeric"]["gpu"]["allow_cpu_fallback"] is False
    assert config["numeric"]["gpu"]["amp_enabled"] is False
    assert config["numeric"]["gpu"]["tf32_enabled"] is False
    assert config["schedule"]["primary"]["name"] == "cubic"
    assert config["schedule"]["time_eps"] == 1.0e-4
    assert config["stop"]["gamma_ref"] == 16.0
    assert config["stop"]["quadrature"] == {
        "method": "gauss_legendre",
        "nodes": 64,
        "reference_nodes": 128,
    }
    assert config["samplers"]["paper_reference"]["exact_gillespie"] is False
    assert config["samplers"]["primary"]["exact_gillespie"] is False
    assert config["samplers"]["primary"]["adaptive_max_hazard_product"] == 0.05
    assert config["foundation"]["incremental_update_enabled"] is False
    assert config["critic_boundary"]["base_sampler_requires_critic"] is False
    assert config["critic_boundary"]["final_evaluator_guidance_allowed"] is False


def test_config_defines_exactly_35_ordered_gates_with_complete_bindings() -> None:
    gates = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))["acceptance"][
        "gates"
    ]
    assert [gate["id"] for gate in gates] == [f"M{i:02d}" for i in range(1, 36)]
    required = {
        "id",
        "name",
        "expected",
        "domain",
        "coverage",
        "sample_count",
        "dtype",
        "atol",
        "rtol",
        "seed",
        "failure_denominator",
        "artifact_path",
    }
    for gate in gates:
        assert required <= set(gate), f"incomplete metadata for {gate.get('id')}"
        # Pure static-text/runtime-query gates are preregistered with an
        # explicit placeholder; the formal GateResult must replace it with a
        # positive observed count (tested below).
        assert (
            isinstance(gate["sample_count"], int) and gate["sample_count"] > 0
        ) or gate["sample_count"] == "RUNTIME_REQUIRED"
        assert gate["seed"] == SEED
        assert gate["domain"] and gate["coverage"] and gate["dtype"]
        assert gate["artifact_path"].startswith("artifacts/mk0/")
        if gate["atol"] is not None:
            assert math.isfinite(gate["atol"]) and gate["atol"] >= 0.0
        if gate["rtol"] is not None:
            assert math.isfinite(gate["rtol"]) and gate["rtol"] >= 0.0


def test_gate_result_rejects_invalid_denominator_pass_binding_and_digest() -> None:
    with pytest.raises(ValueError):
        GateResult(**{**_gate("M01").__dict__, "denominator": 0})
    with pytest.raises(ValueError):
        GateResult(**{**_gate("M01").__dict__, "failure_count": 1})
    with pytest.raises(ValueError):
        GateResult(**{**_gate("M01").__dict__, "artifact_sha256": "short"})
    with pytest.raises(ValueError):
        GateResult(**{**_gate("M01").__dict__, "artifact_sha256": "z" * 64})
    with pytest.raises(ValueError):
        GateResult(**{**_gate("M01").__dict__, "sample_count": 0})
    with pytest.raises(ValueError):
        GateResult(**{**_gate("M01").__dict__, "atol": -1.0})


def test_acceptance_requires_all_35_ordered_gates_and_preserves_e0_boundary() -> None:
    gates = [_gate(f"M{i:02d}") for i in range(1, 36)]
    report = aggregate_acceptance(
        gates,
        run_id="MK0_TEST",
        goal_sha256="b" * 64,
    )
    assert report["pass"] is True
    assert report["gate_count"] == 35
    assert report["failed_gate_ids"] == []
    assert report["evidence_level"] == "E0_MATH_ENGINEERING_ONLY"
    assert not any(report["scientific_claims"].values())

    failed = gates.copy()
    failed[16] = _gate("M17", passed=False)
    failure_report = aggregate_acceptance(
        failed,
        run_id="MK0_TEST_FAIL",
        goal_sha256="b" * 64,
    )
    assert failure_report["pass"] is False
    assert failure_report["status"] == "FAILED_WITH_EVIDENCE"
    assert failure_report["failed_gate_ids"] == ["M17"]

    with pytest.raises(ValueError):
        aggregate_acceptance(gates[:-1], run_id="bad", goal_sha256="b" * 64)
    reordered = gates.copy()
    reordered[0], reordered[1] = reordered[1], reordered[0]
    with pytest.raises(ValueError):
        aggregate_acceptance(reordered, run_id="bad", goal_sha256="b" * 64)
    with pytest.raises(ValueError):
        aggregate_acceptance(gates, run_id="", goal_sha256="b" * 64)
    with pytest.raises(ValueError):
        aggregate_acceptance(gates, run_id="MK0_TEST", goal_sha256="not-a-sha256")


def test_canonical_json_and_file_hash_binding_are_exact(tmp_path: Path) -> None:
    payload = {"unicode": "核", "b": [2, 1], "a": {"z": False}}
    encoded = canonical_json_bytes(payload)
    assert encoded.endswith(b"\n")
    assert encoded == (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    expected = hashlib.sha256(encoded).hexdigest()
    assert sha256_bytes(encoded) == expected
    path = tmp_path / "artifact.json"
    path.write_bytes(encoded)
    assert verify_artifact_binding(path, expected)
    path.write_bytes(encoded + b"tamper")
    assert not verify_artifact_binding(path, expected)
