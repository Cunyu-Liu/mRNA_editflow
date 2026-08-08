from __future__ import annotations

import copy
from pathlib import Path

import pytest

from scripts.r3x.predictor_provenance import normalize_predictor_input_rows
from scripts.r3x.staged_disclosure import (
    DevelopmentDisclosure,
    FeatureDisclosure,
    StagedActivation,
)
from scripts.r3x.typed_staged_rows import (
    EXACT_FEATURE_FIELDS,
    EXACT_OUTCOME_FIELDS,
    TypedStagedRowError,
    normalize_development_disclosure,
    normalize_test_feature_disclosure,
    predictor_input_from_typed_row,
    target_free_feature_row_id,
)


def _activation() -> StagedActivation:
    return StagedActivation(
        path=Path("/ordinary/activation.json"),
        sha256="a" * 64,
        feature_fields=EXACT_FEATURE_FIELDS,
        outcome_fields=EXACT_OUTCOME_FIELDS,
        partition_row_set_sha256={
            "train": "1" * 64,
            "development": "2" * 64,
            "internal_test": "3" * 64,
        },
        authorities={},
        nondata_authority_sha256={},
        nondata_projections={},
    )


def _features(*, candidate: str = "AGGUAC") -> dict:
    return {
        "endpoint": "half_life",
        "publication_context_id": "hek293_arm_a",
        "region": "5UTR",
        "normalized_source_sequence": "ACGUAC",
        "normalized_candidate_sequence": candidate,
        "edit_list": [{"op": "SUB", "pos": 1, "ref": "C", "alt": "G"}],
        "edit_count": 1,
        "study": "GSE217518",
        "estimand_id": "half_life_identity_difference",
        "source_component_id": "component_a",
        "source_group_id_v2": "pool_a",
    }


def _staged_row(partition: str) -> dict:
    features = _features()
    return {
        "row_id": target_free_feature_row_id(partition, features),
        "partition": partition,
        "features": features,
        "outcomes": {"delta": 0.25 if partition == "train" else -0.1},
    }


def _development_disclosure() -> DevelopmentDisclosure:
    return DevelopmentDisclosure(
        activation=_activation(),
        train_rows=(_staged_row("train"),),
        development_rows=(_staged_row("development"),),
        capability_status="STAGED_DISCLOSURE_SAFETY_VALIDATED_NOT_TRAINING_READY",
        training_ready=False,
    )


def test_exact_development_rows_and_predictor_projection() -> None:
    typed = normalize_development_disclosure(_development_disclosure())
    assert typed.typed_rows_ready is True
    assert typed.internal_test_disclosed is False
    assert typed.sealed_access is False
    assert typed.train[0]["delta"] == 0.25
    assert typed.development[0]["delta"] == -0.1
    assert typed.train[0]["edit_list"] == [
        {"op": "SUB", "pos": 1, "ref": "C", "alt": "G"}
    ]

    train_input = predictor_input_from_typed_row(
        typed.train[0], include_training_target=True
    )
    development_input = predictor_input_from_typed_row(
        typed.development[0], include_training_target=False
    )
    normalized = normalize_predictor_input_rows(
        [train_input, development_input]
    )
    by_partition = {row["partition"]: row for row in normalized}
    assert set(by_partition) == {"ORDINARY_TRAIN", "ORDINARY_DEVELOPMENT"}
    assert by_partition["ORDINARY_DEVELOPMENT"]["training_target"] is None
    assert set(by_partition["ORDINARY_TRAIN"]["predictor_input"]) == {
        "endpoint",
        "publication_context_id",
        "region",
        "normalized_source_sequence",
        "normalized_candidate_sequence",
        "edit_list",
        "edit_count",
    }
    assert "study" not in by_partition["ORDINARY_TRAIN"]["predictor_input"]
    assert (
        "source_component_id"
        not in by_partition["ORDINARY_TRAIN"]["predictor_input"]
    )


@pytest.mark.parametrize(
    "mutator,match",
    [
        (
            lambda disclosure: disclosure.train_rows[0]["features"].__setitem__(
                "study_proxy", "hidden"
            ),
            "exact schema",
        ),
        (
            lambda disclosure: disclosure.train_rows[0]["features"].__setitem__(
                "normalized_candidate_sequence", "ACGUAC"
            ),
            "reproduce candidate",
        ),
        (
            lambda disclosure: disclosure.train_rows[0]["features"].__setitem__(
                "edit_count", 2
            ),
            "1, 3, or 5",
        ),
        (
            lambda disclosure: disclosure.train_rows[0]["outcomes"].__setitem__(
                "label", 1.0
            ),
            "exact schema",
        ),
    ],
)
def test_unknown_proxy_edit_or_outcome_fields_hold(mutator, match: str) -> None:
    disclosure = copy.deepcopy(_development_disclosure())
    mutator(disclosure)
    with pytest.raises(TypedStagedRowError, match=match):
        normalize_development_disclosure(disclosure)


def test_activation_cannot_self_define_a_wider_feature_contract() -> None:
    disclosure = copy.deepcopy(_development_disclosure())
    disclosure = DevelopmentDisclosure(
        activation=StagedActivation(
            **{
                **disclosure.activation.__dict__,
                "feature_fields": tuple(sorted((*EXACT_FEATURE_FIELDS, "study_proxy"))),
            }
        ),
        train_rows=disclosure.train_rows,
        development_rows=disclosure.development_rows,
        capability_status=disclosure.capability_status,
        training_ready=False,
    )
    with pytest.raises(TypedStagedRowError, match="feature_fields differ"):
        normalize_development_disclosure(disclosure)


def test_outcome_or_caller_selected_identity_cannot_drive_row_id() -> None:
    features = _features()
    train_id = target_free_feature_row_id("train", features)
    assert train_id == target_free_feature_row_id("train", copy.deepcopy(features))
    assert train_id != target_free_feature_row_id("development", features)

    disclosure = copy.deepcopy(_development_disclosure())
    disclosure.train_rows[0]["row_id"] = "f" * 64
    with pytest.raises(TypedStagedRowError, match="feature-only identity"):
        normalize_development_disclosure(disclosure)

    changed_outcome = copy.deepcopy(_development_disclosure())
    changed_outcome.train_rows[0]["outcomes"]["delta"] = -999.0
    typed = normalize_development_disclosure(changed_outcome)
    assert typed.train[0]["r3_input_row_sha256"] == train_id


def test_predictor_target_exposure_must_match_partition() -> None:
    typed = normalize_development_disclosure(_development_disclosure())
    with pytest.raises(TypedStagedRowError, match="target exposure"):
        predictor_input_from_typed_row(
            typed.development[0], include_training_target=True
        )
    with pytest.raises(TypedStagedRowError, match="target exposure"):
        predictor_input_from_typed_row(
            typed.train[0], include_training_target=False
        )


def test_feature_stage_contains_no_outcome_and_is_still_exact() -> None:
    features = _features()
    disclosure = FeatureDisclosure(
        activation=_activation(),
        pre_test_freeze_sha256="b" * 64,
        selected_method_id="FULL_PAIR_UNRESTRICTED_SIAMESE",
        predictor_method_ids=("FULL_PAIR_UNRESTRICTED_SIAMESE",),
        internal_test_rows=(
            {
                "row_id": target_free_feature_row_id(
                    "internal_test", features
                ),
                "partition": "internal_test",
                "features": features,
            },
        ),
    )
    rows = normalize_test_feature_disclosure(disclosure)
    assert rows[0]["partition"] == "internal_test"
    assert "delta" not in rows[0]
    assert rows[0]["schema_version"] == "r3x_primary_paired_sequence_test_feature_row_v1"


def test_row_normalizer_never_upgrades_physical_training_readiness() -> None:
    disclosure = _development_disclosure()
    object.__setattr__(disclosure, "training_ready", True)
    with pytest.raises(TypedStagedRowError, match="must not self-assert"):
        normalize_development_disclosure(disclosure)
