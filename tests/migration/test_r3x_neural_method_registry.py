from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from types import MappingProxyType

import pytest
import torch

import scripts.r3x.neural_method_registry as registry
from scripts.r3x.models import (
    EditAwareFullPairUnrestrictedCritic,
    ExactPotentialCritic,
    FullPairUnrestrictedSiamese,
    ModelConfig,
)
from scripts.r3x.neural_method_registry import (
    A0_CONTROL_METHOD_ID,
    A0_METHOD_ID,
    A1_CONTROL_METHOD_ID,
    A1_METHOD_ID,
    A1_UNK_ONLY_METHOD_ID,
    A1_WRONG_LOCAL_METHOD_ID,
    B_ADDITIVE_METHOD_ID,
    B_MATCHED_CONTROL_METHOD_ID,
    B_UNK_ONLY_METHOD_ID,
    B_WRONG_LOCAL_METHOD_ID,
    EDIT_AWARE_FULL_PAIR_METHOD_ID,
    FULL_PAIR_METHOD_ID,
    NeuralMethodRegistryError,
    canonical_registry_payload,
    compile_registered_development_model,
    registry_sha256,
)


def _cfg(**updates) -> ModelConfig:
    values = {
        "vocab_size": 5,
        "d_model": 16,
        "n_heads": 4,
        "n_layers": 1,
        "ff_dim": 32,
        "max_len": 16,
        "n_endpoints": 3,
        "n_contexts": 4,
        "max_edits": 5,
        "pair_rank": 2,
    }
    values.update(updates)
    return ModelConfig(**values)


def _compile(method_id: str, cfg: ModelConfig | None = None):
    return compile_registered_development_model(
        method_id,
        cfg or _cfg(),
        expected_registry_sha256=registry_sha256(),
    )


def test_registry_has_unique_versioned_method_and_architecture_ids() -> None:
    payload = canonical_registry_payload()
    methods = payload["methods"]
    assert payload["schema_version"] == "r3x_neural_method_architecture_registry_v4"
    assert payload["factory_version"] == "R3X_EXACT_CLASS_FACTORY_V4"
    assert payload["method_order"] == [item["method_id"] for item in methods]
    assert len({item["method_id"] for item in methods}) == len(methods)
    assert len({item["architecture_id"] for item in methods}) == len(methods)
    assert isinstance(registry._METHOD_BY_ID, MappingProxyType)
    assert isinstance(registry._FACTORIES, MappingProxyType)
    assert payload["information_set"] == "PAIRED_SEQUENCE_NO_MEASURED_ANCHOR"
    assert payload["action_budget_enforcement"] == "RUNNER_ENFORCEMENT_REQUIRED"
    assert payload["action_budget_validated_by_compiler"] is False
    assert payload["construction_device"] == "cpu"
    assert payload["floating_dtype"] == "torch.float32"
    assert (
        payload["preallocation_policy"]
        == "CANDIDATE_PLUS_REFERENCE_FLOAT32_PARAMETER_BYTES_V1"
    )
    assert payload["execution_authorized"] is False
    assert payload["training_ready"] is False
    assert payload["publication"] is False
    assert payload["publication_evidence"] is False
    assert payload["b_interaction_status"] == "NOT_RUN_ACTION_SUPPORT"
    assert payload["sealed_access"] is False
    assert not any("INTERACTION" in item["method_id"] for item in methods)
    assert all(item["execution_authorized"] is False for item in methods)
    assert all(item["training_ready"] is False for item in methods)
    assert all(item["publication"] is False for item in methods)
    assert all(item["publication_evidence"] is False for item in methods)
    b_matched = next(
        item for item in methods if item["method_id"] == B_MATCHED_CONTROL_METHOD_ID
    )
    assert b_matched["architecture_id"] == (
        "R3X_B_SPARSE_EDIT_SET_POSITION_AGNOSTIC_MATCHED_CONTROL_V2"
    )
    assert b_matched["training_variant_id"] == (
        "POSITION_AGNOSTIC_ARCHITECTURE_MATCHED_CONTROL_V2"
    )
    assert b_matched["control_intervention"] == (
        "REMOVE_ACTUAL_EDIT_POSITION_FROM_PREDICTION_USE_GLOBAL_POOLED_SOURCE_"
        "AND_REF_ALT_PSEUDOCOORDINATE_USING_SAME_PARAMETERS"
    )
    assert len(registry_sha256()) == 64


def test_v3_position_agnostic_registry_bytes_fail_closed() -> None:
    legacy = copy.deepcopy(canonical_registry_payload())
    legacy["schema_version"] = "r3x_neural_method_architecture_registry_v3"
    legacy["factory_version"] = "R3X_EXACT_CLASS_FACTORY_V3"
    for method in legacy["methods"]:
        method["factory_id"] = "R3X_EXACT_CLASS_FACTORY_V3"
        if method["method_id"] == B_MATCHED_CONTROL_METHOD_ID:
            method["architecture_id"] = (
                "R3X_B_SPARSE_EDIT_SET_POSITION_AGNOSTIC_MATCHED_CONTROL_V1"
            )
            method["training_variant_id"] = (
                "POSITION_AGNOSTIC_ARCHITECTURE_MATCHED_CONTROL_V1"
            )
            method["control_intervention"] = (
                "REPLACE_POSITION_WITH_REF_ALT_PSEUDOCOORDINATE_USING_SAME_PARAMETERS"
            )
    legacy_sha = hashlib.sha256(registry._canonical_bytes(legacy)).hexdigest()
    assert legacy_sha != registry_sha256()
    with pytest.raises(NeuralMethodRegistryError, match="does not match"):
        compile_registered_development_model(
            B_MATCHED_CONTROL_METHOD_ID,
            _cfg(),
            expected_registry_sha256=legacy_sha,
        )


@pytest.mark.parametrize(
    "method_id",
    [
        FULL_PAIR_METHOD_ID,
        EDIT_AWARE_FULL_PAIR_METHOD_ID,
        A0_METHOD_ID,
        A0_CONTROL_METHOD_ID,
        A1_METHOD_ID,
        A1_CONTROL_METHOD_ID,
        A1_WRONG_LOCAL_METHOD_ID,
        A1_UNK_ONLY_METHOD_ID,
        B_ADDITIVE_METHOD_ID,
        B_MATCHED_CONTROL_METHOD_ID,
        B_WRONG_LOCAL_METHOD_ID,
        B_UNK_ONLY_METHOD_ID,
    ],
)
def test_compiler_returns_exact_registered_architecture(method_id: str) -> None:
    model, binding = _compile(method_id)
    assert binding["method_id"] == method_id
    assert binding["architecture_id"] == model.architecture_id
    assert binding["parameter_count"] > 0
    assert binding["model_config_sha256"]
    assert binding["information_set"] == "PAIRED_SEQUENCE_NO_MEASURED_ANCHOR"
    assert binding["allowed_primary_edit_counts"] == [1, 3, 5]
    assert binding["maximum_primary_edits"] == 5
    assert binding["action_budget_enforcement"] == "RUNNER_ENFORCEMENT_REQUIRED"
    assert binding["action_budget_validated_by_compiler"] is False
    assert binding["registry_sha256"] == registry_sha256()
    assert binding["caller_expected_registry_sha256"] == registry_sha256()
    assert binding["registry_code_sha256"] == hashlib.sha256(
        Path(registry.__file__).read_bytes()
    ).hexdigest()
    assert binding["construction_device"] == "cpu"
    assert binding["floating_dtype"] == "torch.float32"
    assert binding["candidate_state_tensor_contract"]["device"] == "cpu"
    assert binding["candidate_state_tensor_contract"]["parameter_dtypes"] == [
        "torch.float32"
    ]
    assert binding["preallocation_model_count"] == (
        2 if binding["matched_reference_method_id"] is not None else 1
    )
    assert binding["preallocation_peak_parameter_bytes_upper_bound"] == (
        binding["preallocation_parameter_upper_bound"]
        * binding["preallocation_model_count"]
        * 4
    )
    assert binding["execution_authorized"] is False
    assert binding["training_ready"] is False
    assert binding["publication"] is False
    assert binding["publication_evidence"] is False
    assert binding["b_interaction_status"] == "NOT_RUN_ACTION_SUPPORT"
    assert binding["training_variant_id"]
    assert binding["control_intervention"]
    if binding["matched_reference_method_id"] is not None:
        assert binding["parameter_match"]["matched"] is True


@pytest.mark.parametrize(
    "control_method_id,reference_method_id",
    [
        (A1_WRONG_LOCAL_METHOD_ID, A1_METHOD_ID),
        (A1_UNK_ONLY_METHOD_ID, A1_METHOD_ID),
        (B_MATCHED_CONTROL_METHOD_ID, B_ADDITIVE_METHOD_ID),
        (B_WRONG_LOCAL_METHOD_ID, B_ADDITIVE_METHOD_ID),
        (B_UNK_ONLY_METHOD_ID, B_ADDITIVE_METHOD_ID),
    ],
)
def test_new_development_controls_are_exact_actual_config_parameter_matches(
    control_method_id: str,
    reference_method_id: str,
) -> None:
    _model, binding = _compile(control_method_id)
    assert binding["matched_reference_method_id"] == reference_method_id
    assert binding["parameter_match"] == {
        "reference_parameters": binding["parameter_count"],
        "candidate_parameters": binding["parameter_count"],
        "gap_fraction": 0.0,
        "maximum_gap_fraction": 0.05,
        "matched": True,
    }


def test_old_full_pair_id_cannot_resolve_to_edit_aware_architecture() -> None:
    old, old_binding = _compile(FULL_PAIR_METHOD_ID)
    new, new_binding = _compile(EDIT_AWARE_FULL_PAIR_METHOD_ID)
    assert old.__class__ is not new.__class__
    assert old_binding["architecture_id"] != new_binding["architecture_id"]


@pytest.mark.parametrize(
    "legacy_or_ambiguous_id",
    [
        "FULL_PAIR_LATEST",
        "FULL_PAIR_UNRESTRICTED_SIAMESE",
        "A_EXACT_POTENTIAL_DIFFERENCE",
        "A_PARAMETER_MATCHED_UNRESTRICTED_CONTROL",
        "A1_BOUNDED_LOCAL_EXACT_POTENTIAL",
    ],
)
def test_unknown_and_legacy_aliases_fail_closed(legacy_or_ambiguous_id: str) -> None:
    with pytest.raises(NeuralMethodRegistryError, match="unregistered"):
        _compile(legacy_or_ambiguous_id)


def test_expected_registry_digest_is_mandatory_and_exact() -> None:
    with pytest.raises(TypeError, match="expected_registry_sha256"):
        compile_registered_development_model(A0_METHOD_ID, _cfg())
    with pytest.raises(NeuralMethodRegistryError, match="lowercase SHA-256"):
        compile_registered_development_model(
            A0_METHOD_ID,
            _cfg(),
            expected_registry_sha256="NOT_A_SHA",
        )
    with pytest.raises(NeuralMethodRegistryError, match="does not match"):
        compile_registered_development_model(
            A0_METHOD_ID,
            _cfg(),
            expected_registry_sha256="0" * 64,
        )


def test_false_parameter_match_fails_closed_for_actual_config() -> None:

    # This legal ModelConfig previously escaped plan validation even though the
    # edit-aware comparator was not within the frozen five-percent budget.
    with pytest.raises(NeuralMethodRegistryError, match="parameter-matched"):
        _compile(
            EDIT_AWARE_FULL_PAIR_METHOD_ID,
            _cfg(d_model=8, n_heads=1, ff_dim=8, max_len=1),
        )


def test_extreme_pair_rank_fails_before_a_false_matched_control_is_returned() -> None:
    with pytest.raises(NeuralMethodRegistryError):
        _compile(
            A1_CONTROL_METHOD_ID,
            _cfg(d_model=64, n_heads=4, ff_dim=128, pair_rank=4096),
        )


def test_candidate_plus_reference_byte_guard_runs_before_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def construction_must_not_run(*args, **kwargs):
        raise AssertionError("model construction occurred before byte preflight")

    monkeypatch.setattr(
        registry, "_construct_cpu_float32", construction_must_not_run
    )
    with pytest.raises(NeuralMethodRegistryError, match="pre-allocation parameter-byte"):
        _compile(
            A1_CONTROL_METHOD_ID,
            _cfg(d_model=64, n_heads=4, ff_dim=128, pair_rank=65_536),
        )


def test_factory_crosswalk_is_immutable_and_not_a_canonical_payload_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before_payload = canonical_registry_payload()
    before_sha = registry_sha256()
    with pytest.raises(TypeError):
        registry._FACTORIES[A0_METHOD_ID] = EditAwareFullPairUnrestrictedCritic
    with pytest.raises(TypeError):
        registry._METHOD_BY_ID[A0_METHOD_ID] = registry._METHOD_BY_ID[
            EDIT_AWARE_FULL_PAIR_METHOD_ID
        ]

    # Rebinding the old diagnostic name cannot redirect compilation or alter
    # canonical registry bytes; the compiler consumes frozen _MethodSpec values.
    rogue_factories = dict(registry._FACTORIES)
    rogue_factories[FULL_PAIR_METHOD_ID] = ExactPotentialCritic
    rogue_factories[A0_METHOD_ID] = EditAwareFullPairUnrestrictedCritic
    monkeypatch.setattr(registry, "_FACTORIES", rogue_factories)
    model, binding = _compile(FULL_PAIR_METHOD_ID)
    assert type(model) is FullPairUnrestrictedSiamese
    assert binding["registry_sha256"] == before_sha
    assert canonical_registry_payload() == before_payload

    # The historical reference-substitution exploit must still encounter the
    # true A0 reference and therefore retain the actual-config failure.
    with pytest.raises(NeuralMethodRegistryError, match="parameter-matched"):
        _compile(
            EDIT_AWARE_FULL_PAIR_METHOD_ID,
            _cfg(d_model=8, n_heads=1, ff_dim=8, max_len=1),
        )


def test_subclass_self_report_and_reference_metadata_tamper_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SelfReportingSubclass(ExactPotentialCritic):
        architecture_id = ExactPotentialCritic.architecture_id
        information_set = registry.PAIRED_INFORMATION_SET

    rogue = SelfReportingSubclass(_cfg())
    with pytest.raises(NeuralMethodRegistryError, match="non-exact registered class"):
        registry._verify_exact_instance(
            rogue,
            registry._METHOD_BY_ID[A0_METHOD_ID],
            role="candidate",
        )

    monkeypatch.setattr(ExactPotentialCritic, "architecture_id", "ROGUE_A0")
    with pytest.raises(
        NeuralMethodRegistryError, match="matched reference.*architecture ID"
    ):
        _compile(FULL_PAIR_METHOD_ID)


def test_global_meta_device_and_float64_dtype_fail_closed_before_construction() -> None:
    previous_device = torch.get_default_device()
    previous_dtype = torch.get_default_dtype()
    expected_sha = registry_sha256()
    try:
        torch.set_default_dtype(torch.float64)
        with pytest.raises(NeuralMethodRegistryError, match="float32 default dtype"):
            compile_registered_development_model(
                A0_METHOD_ID,
                _cfg(),
                expected_registry_sha256=expected_sha,
            )
        torch.set_default_dtype(torch.float32)
        torch.set_default_device("meta")
        with pytest.raises(NeuralMethodRegistryError, match="CPU default device"):
            compile_registered_development_model(
                A0_METHOD_ID,
                _cfg(),
                expected_registry_sha256=expected_sha,
            )
    finally:
        torch.set_default_dtype(previous_dtype)
        torch.set_default_device(previous_device)
