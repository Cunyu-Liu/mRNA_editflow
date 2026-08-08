from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts.r3x.checkpoint import REQUIRED_AUTHORITY_HASHES
from scripts.r3x.selection_artifact import (
    FALLBACK_PHASE,
    GATE_MET_PHASE,
    SELECTION_RULE,
    SelectionArtifactProtocolError,
    build_checkpoint_selection_artifact,
    canonical_checkpoint_selection_bytes,
    load_checkpoint_selection_artifact_exact,
    validate_checkpoint_selection_artifact,
    write_checkpoint_selection_artifact_exclusive,
)


def _authorities() -> dict[str, str]:
    return {
        name: f"{index + 1:x}" * 64
        for index, name in enumerate(REQUIRED_AUTHORITY_HASHES)
    }


def _bindings() -> dict:
    return {
        "run_start_sha256": "c" * 64,
        "activation_manifest_sha256": "d" * 64,
        "authority_hashes": _authorities(),
        "estimand_binding_sha256": "e" * 64,
        "method_id": "A_EXACT_POTENTIAL_DIFFERENCE",
        "training_variant_id": "PRIMARY",
        "seed": 11,
        "partition_row_set_sha256": {
            "train": "f" * 64,
            "development": "0" * 64,
        },
        "training_config_sha256": "a" * 64,
    }


def _epochs_gate_met() -> list[dict]:
    return [
        {
            "epoch": 1,
            "macro_sign_accuracy": 0.59,
            "macro_delta_spearman": 0.99,
            "macro_gaussian_nll": 0.01,
        },
        {
            "epoch": 2,
            "macro_sign_accuracy": 0.61,
            "macro_delta_spearman": 0.30,
            "macro_gaussian_nll": 0.50,
        },
        {
            "epoch": 3,
            "macro_sign_accuracy": 0.60,
            "macro_delta_spearman": 0.40,
            "macro_gaussian_nll": 0.70,
        },
        {
            "epoch": 4,
            "macro_sign_accuracy": 0.99,
            "macro_delta_spearman": 0.40,
            "macro_gaussian_nll": 0.60,
        },
        {
            "epoch": 5,
            "macro_sign_accuracy": 0.60,
            "macro_delta_spearman": 0.40,
            "macro_gaussian_nll": 0.60,
        },
    ]


def _build(epochs: list[dict] | None = None) -> dict:
    return build_checkpoint_selection_artifact(
        **_bindings(), epoch_candidates=epochs or _epochs_gate_met()
    )


def test_gate_met_selection_is_exact_two_phase_lexicographic_replay() -> None:
    payload = _build()
    assert payload["status"] == "DEV_SIGN_GATE_MET"
    assert payload["selection_phase"] == GATE_MET_PHASE
    assert payload["selection_rule"] == SELECTION_RULE
    # Epoch 1 has the best Spearman but is ineligible.  Epoch 4 beats epoch 3
    # on NLL; epoch 4 beats epoch 5 on the final earliest-epoch tie break.
    assert payload["selected_epoch"] == 4
    assert payload["selected_development_metrics"] == {
        "macro_sign_accuracy": 0.99,
        "macro_delta_spearman": 0.40,
        "macro_gaussian_nll": 0.60,
    }
    assert payload["lexicographic_selection_tuple_fields"] == [
        "sign_gate_met",
        "macro_delta_spearman",
        "negative_macro_gaussian_nll",
        "negative_epoch",
    ]
    assert payload["epoch_selection_records"][0][
        "lexicographic_selection_tuple"
    ][0] == 0.0
    assert payload["epoch_selection_records"][3][
        "lexicographic_selection_tuple"
    ] == [1.0, 0.4, -0.6, -4.0]


def test_no_gate_fallback_prioritizes_sign_then_spearman_nll_and_epoch() -> None:
    payload = _build([
        {
            "epoch": 3,
            "macro_sign_accuracy": 0.55,
            "macro_delta_spearman": 0.10,
            "macro_gaussian_nll": 0.80,
        },
        {
            "epoch": 2,
            "macro_sign_accuracy": 0.55,
            "macro_delta_spearman": 0.10,
            "macro_gaussian_nll": 0.80,
        },
        {
            "epoch": 1,
            "macro_sign_accuracy": 0.54,
            "macro_delta_spearman": 1.00,
            "macro_gaussian_nll": -10.0,
        },
    ])
    assert payload["status"] == "HOLD_DEV_SIGN_GATE_NOT_MET"
    assert payload["selection_phase"] == FALLBACK_PHASE
    assert payload["selected_epoch"] == 2
    assert payload["lexicographic_selection_tuple_fields"][0] == (
        "macro_sign_accuracy"
    )


def test_schema_binds_only_train_and_development_and_asserts_all_boundaries() -> None:
    payload = _build()
    assert set(payload["partition_row_set_sha256"]) == {"train", "development"}
    assert set(payload["authority_hashes"]) == set(REQUIRED_AUTHORITY_HASHES)
    assert payload["selection_data_role"] == "ORDINARY_DEVELOPMENT_ONLY"
    for key in (
        "test_based_selection",
        "best_seed_selection",
        "internal_test_access",
        "predictions_included",
        "row_ids_included",
        "sealed_access",
        "downstream_unlock",
    ):
        assert payload[key] is False
    assert payload["ordinary_only"] is True
    artifact_text = json.dumps(payload, sort_keys=True)
    assert '"internal_test"' not in artifact_text
    assert '"row_id"' not in artifact_text


@pytest.mark.parametrize(
    "forbidden_field",
    [
        "test_metrics",
        "internal_test_metrics",
        "test_predictions",
        "prediction_rows",
        "row_ids",
        "sealed_labels",
        "downstream_permission",
    ],
)
def test_epoch_input_forbids_test_predictions_row_ids_and_permissions(
    forbidden_field: str,
) -> None:
    epochs = _epochs_gate_met()
    epochs[0][forbidden_field] = []
    with pytest.raises(
        SelectionArtifactProtocolError,
        match="forbidden non-development disclosure or permission key",
    ):
        _build(epochs)


def test_internal_test_partition_hash_cannot_enter_selection_artifact() -> None:
    bindings = _bindings()
    bindings["partition_row_set_sha256"]["internal_test"] = "1" * 64
    with pytest.raises(SelectionArtifactProtocolError, match="internal_test"):
        build_checkpoint_selection_artifact(
            **bindings, epoch_candidates=_epochs_gate_met()
        )


@pytest.mark.parametrize(
    "mutator",
    [
        lambda value: value.__setitem__("selected_epoch", 5),
        lambda value: value.__setitem__("sign_gate", 0.59),
        lambda value: value.__setitem__("selection_rule", "POSTHOC_RULE"),
        lambda value: value.__setitem__("sealed_access", True),
        lambda value: value.__setitem__("downstream_unlock", True),
        lambda value: value.__setitem__("internal_test_access", True),
        lambda value: value.__setitem__("predictions_included", True),
        lambda value: value["epoch_selection_records"][3].__setitem__(
            "lexicographic_selection_tuple", [1.0, 0.4, -0.6, -5.0]
        ),
        lambda value: value["selected_development_metrics"].__setitem__(
            "macro_sign_accuracy", 0.60
        ),
        lambda value: value.__setitem__("posthoc_metric", 1.0),
    ],
)
def test_recomputation_or_strict_schema_rejects_any_artifact_drift(mutator) -> None:
    changed = copy.deepcopy(_build())
    mutator(changed)
    with pytest.raises(SelectionArtifactProtocolError):
        validate_checkpoint_selection_artifact(changed)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("macro_sign_accuracy", float("nan"), "finite"),
        ("macro_sign_accuracy", 1.01, r"\[0, 1\]"),
        ("macro_delta_spearman", float("inf"), "finite"),
        ("macro_delta_spearman", -1.01, r"\[-1, 1\]"),
        ("macro_gaussian_nll", True, "finite"),
    ],
)
def test_nonfinite_boolean_and_out_of_range_metrics_fail_closed(
    field: str, value: object, match: str
) -> None:
    epochs = _epochs_gate_met()
    epochs[0][field] = value
    with pytest.raises(SelectionArtifactProtocolError, match=match):
        _build(epochs)


def test_duplicate_epoch_nonfrozen_seed_variant_and_incomplete_authority_fail() -> None:
    epochs = _epochs_gate_met()
    epochs[1]["epoch"] = epochs[0]["epoch"]
    with pytest.raises(SelectionArtifactProtocolError, match="unique"):
        _build(epochs)

    bindings = _bindings()
    bindings["seed"] = 13
    with pytest.raises(SelectionArtifactProtocolError, match="frozen seeds"):
        build_checkpoint_selection_artifact(
            **bindings, epoch_candidates=_epochs_gate_met()
        )

    bindings = _bindings()
    bindings["training_variant_id"] = "POSTHOC_VARIANT"
    with pytest.raises(SelectionArtifactProtocolError, match="not frozen"):
        build_checkpoint_selection_artifact(
            **bindings, epoch_candidates=_epochs_gate_met()
        )

    bindings = _bindings()
    del bindings["authority_hashes"]["ready_dataset_sha256"]
    with pytest.raises(SelectionArtifactProtocolError, match="strict schema"):
        build_checkpoint_selection_artifact(
            **bindings, epoch_candidates=_epochs_gate_met()
        )


def test_exact_writer_loader_canonical_bytes_and_expected_bindings(
    tmp_path: Path,
) -> None:
    payload = _build()
    path = tmp_path / "selection.json"
    digest = write_checkpoint_selection_artifact_exclusive(path, payload)
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    assert path.read_bytes() == canonical_checkpoint_selection_bytes(payload)
    loaded = load_checkpoint_selection_artifact_exact(
        path,
        digest,
        expected_run_start_sha256="c" * 64,
        expected_activation_manifest_sha256="d" * 64,
        expected_authority_hashes=_authorities(),
        expected_estimand_binding_sha256="e" * 64,
        expected_method_id="A_EXACT_POTENTIAL_DIFFERENCE",
        expected_training_variant_id="PRIMARY",
        expected_seed=11,
        expected_partition_row_set_sha256={
            "train": "f" * 64,
            "development": "0" * 64,
        },
        expected_training_config_sha256="a" * 64,
    )
    assert loaded == payload
    with pytest.raises(SelectionArtifactProtocolError, match="already exists"):
        write_checkpoint_selection_artifact_exclusive(path, payload)
    with pytest.raises(SelectionArtifactProtocolError, match="SHA-256 mismatch"):
        load_checkpoint_selection_artifact_exact(path, "0" * 64)
    with pytest.raises(SelectionArtifactProtocolError, match="method_id binding"):
        load_checkpoint_selection_artifact_exact(
            path, digest, expected_method_id="B_SPARSE_EDIT_SET_ADDITIVE"
        )


def test_exact_loader_rejects_duplicate_keys_and_noncanonical_json(
    tmp_path: Path,
) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_bytes(b'{"schema_version":"first","schema_version":"second"}\n')
    duplicate_digest = hashlib.sha256(duplicate.read_bytes()).hexdigest()
    with pytest.raises(SelectionArtifactProtocolError, match="duplicate JSON key"):
        load_checkpoint_selection_artifact_exact(duplicate, duplicate_digest)

    noncanonical = tmp_path / "noncanonical.json"
    noncanonical.write_text(
        json.dumps(_build(), sort_keys=False, separators=(",", ":")),
        encoding="utf-8",
    )
    noncanonical_digest = hashlib.sha256(noncanonical.read_bytes()).hexdigest()
    with pytest.raises(SelectionArtifactProtocolError, match="not canonical"):
        load_checkpoint_selection_artifact_exact(noncanonical, noncanonical_digest)
