from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from scripts.r3x.development_checkpoint import (
    DevelopmentCheckpointError,
    REQUIRED_DEVELOPMENT_DATA_AUTHORITY_HASHES,
    REQUIRED_DEVELOPMENT_NONDATA_AUTHORITY_HASHES,
    build_development_checkpoint,
    load_development_checkpoint_exact,
    save_development_checkpoint_exclusive,
    validate_development_checkpoint,
)
from scripts.r3x.models import ExactPotentialCritic, ModelConfig


def _cfg() -> ModelConfig:
    return ModelConfig(
        d_model=8,
        n_heads=2,
        n_layers=1,
        ff_dim=16,
        max_len=8,
        n_endpoints=2,
        n_contexts=2,
        max_edits=5,
        pair_rank=2,
    )


def _data_authorities() -> dict[str, str]:
    return {
        name: format(index + 1, "x") * 64
        for index, name in enumerate(REQUIRED_DEVELOPMENT_DATA_AUTHORITY_HASHES)
    }


def _nondata_authorities() -> dict[str, str]:
    return {
        name: format(index + 2, "x") * 64
        for index, name in enumerate(REQUIRED_DEVELOPMENT_NONDATA_AUTHORITY_HASHES)
    }


def _payload() -> dict:
    torch.manual_seed(11)
    return build_development_checkpoint(
        ExactPotentialCritic(_cfg()),
        _cfg(),
        run_start_sha256="1" * 64,
        run_plan_sha256="2" * 64,
        development_activation_sha256="3" * 64,
        development_data_authority_hashes=_data_authorities(),
        development_nondata_authority_hashes=_nondata_authorities(),
        estimand_binding_sha256="4" * 64,
        method_id="A_EXACT_POTENTIAL_DIFFERENCE",
        training_variant_id="PRIMARY",
        training_config_sha256="5" * 64,
        condition_vocab={
            "endpoint": {"__UNK__": 0, "endpoint_a": 1},
            "context": {"__UNK__": 0, "context_a": 1},
        },
        information_set="PAIRED_SEQUENCE_NO_MEASURED_ANCHOR",
        information_set_fingerprint="6" * 64,
        seed=11,
        selected_epoch=1,
        checkpoint_selection_sha256="7" * 64,
        partition_row_set_sha256={"train": "8" * 64, "development": "9" * 64},
    )


def _validate(payload: dict) -> dict:
    return validate_development_checkpoint(
        payload,
        expected_architecture_id="R3X_A_EXACT_POTENTIAL_DIFFERENCE_V1",
        expected_development_data_authority_hashes=_data_authorities(),
        expected_development_nondata_authority_hashes=_nondata_authorities(),
        expected_run_start_sha256="1" * 64,
        expected_run_plan_sha256="2" * 64,
        expected_development_activation_sha256="3" * 64,
        expected_method_id="A_EXACT_POTENTIAL_DIFFERENCE",
        expected_training_variant_id="PRIMARY",
        expected_seed=11,
        expected_estimand_binding_sha256="4" * 64,
        expected_information_set_fingerprint="6" * 64,
        expected_partition_row_set_sha256={"train": "8" * 64, "development": "9" * 64},
    )


def test_development_checkpoint_has_no_test_binding_or_permission() -> None:
    payload = _validate(_payload())
    assert set(payload["partition_row_set_sha256"]) == {"train", "development"}
    assert set(payload["development_data_authority_hashes"]) == {
        "train_development_outcomes"
    }
    assert all(
        "internal_test" not in key
        for key in payload["development_nondata_authority_hashes"]
    )
    assert payload["internal_test_feature_access"] is False
    assert payload["internal_test_outcome_access"] is False
    assert payload["sealed_access"] is False
    assert payload["sign_gate"] == 0.60


def test_checkpoint_exact_write_load_and_no_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "development_checkpoint.pt"
    digest = save_development_checkpoint_exclusive(path, _payload())
    loaded = load_development_checkpoint_exact(path, digest)
    _validate(loaded)
    with pytest.raises(DevelopmentCheckpointError):
        save_development_checkpoint_exclusive(path, _payload())
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(DevelopmentCheckpointError):
        load_development_checkpoint_exact(path, digest)


def test_legacy_internal_test_hash_cannot_enter_development_checkpoint() -> None:
    with pytest.raises(DevelopmentCheckpointError, match="only train/development"):
        build_development_checkpoint(
            ExactPotentialCritic(_cfg()),
            _cfg(),
            run_start_sha256="1" * 64,
            run_plan_sha256="2" * 64,
            development_activation_sha256="3" * 64,
            development_data_authority_hashes=_data_authorities(),
            development_nondata_authority_hashes=_nondata_authorities(),
            estimand_binding_sha256="4" * 64,
            method_id="A_EXACT_POTENTIAL_DIFFERENCE",
            training_variant_id="PRIMARY",
            training_config_sha256="5" * 64,
            condition_vocab={
                "endpoint": {"__UNK__": 0, "endpoint_a": 1},
                "context": {"__UNK__": 0, "context_a": 1},
            },
            information_set="PAIRED_SEQUENCE_NO_MEASURED_ANCHOR",
            information_set_fingerprint="6" * 64,
            seed=11,
            selected_epoch=1,
            checkpoint_selection_sha256="7" * 64,
            partition_row_set_sha256={
                "train": "8" * 64,
                "development": "9" * 64,
                "internal_test": "a" * 64,
            },
        )


@pytest.mark.parametrize(
    "mutator,match",
    [
        (lambda value: value.__setitem__("scientific_scope", "PUBLICATION_ELIGIBLE"), "scope escalated"),
        (lambda value: value.__setitem__("sign_gate", 0.50), "sign gate changed"),
        (lambda value: value.__setitem__("internal_test_outcome_access", True), "boundary changed"),
        (lambda value: value["partition_row_set_sha256"].__setitem__("internal_test", "a" * 64), "only train/development"),
        (lambda value: value["development_data_authority_hashes"].__setitem__("internal_test_outcomes", "a" * 64), "incomplete or unexpected"),
    ],
)
def test_scope_gate_and_test_disclosure_tampering_fail(mutator, match: str) -> None:
    payload = copy.deepcopy(_payload())
    mutator(payload)
    with pytest.raises(DevelopmentCheckpointError, match=match):
        _validate(payload)


def test_nonfinite_state_or_wrong_expected_binding_fails() -> None:
    payload = copy.deepcopy(_payload())
    key = next(iter(payload["state_dict"]))
    state = payload["state_dict"][key]
    if not state.is_floating_point():
        pytest.skip("fixture first state tensor is not floating point")
    state.view(-1)[0] = float("nan")
    with pytest.raises(DevelopmentCheckpointError, match="non-finite"):
        _validate(payload)
    with pytest.raises(DevelopmentCheckpointError, match="authority hashes mismatch"):
        validate_development_checkpoint(
            _payload(),
            expected_architecture_id="R3X_A_EXACT_POTENTIAL_DIFFERENCE_V1",
            expected_development_data_authority_hashes={
                **_data_authorities(),
                "train_development_outcomes": "f" * 64,
            },
            expected_development_nondata_authority_hashes=_nondata_authorities(),
        )
