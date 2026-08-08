from __future__ import annotations

import copy
import hashlib
import inspect
import json
from dataclasses import replace
from pathlib import Path

import pytest

from scripts.r3x.protocol import canonical_sha256
from scripts.r3x.splits import (
    _ordered_components as authoritative_ordered_components,
    _partition_by_component as authoritative_partition_by_component,
)
from scripts.r3x.staged_disclosure import (
    DevelopmentDisclosure,
    FeatureDisclosure,
    StagedActivation,
)
from scripts.r3x.stage_row_projections import (
    CONTEXT_SOURCE_CROSSWALK,
    DEVELOPMENT_STAGE,
    FirewallReplayContext,
    INCLUSION_MANIFEST,
    INTERNAL_TEST_FEATURE_STAGE,
    ROW_AUTHORITIES,
    SOURCE_COMPONENT_SPLIT_MANIFEST,
    StageRowProjectionError,
    TEST_FEATURE_PROJECTION_HASH_KEYS,
    TrustedFirewallVerifier,
    attest_pretraining_firewall,
    assess_development_runtime_row_authority_capability,
    build_development_row_projections,
    build_internal_test_feature_row_projections,
    canonical_pretraining_firewall_certificate_bytes,
    canonical_stage_row_projection_bytes,
    independent_firewall_verifier,
    load_pretraining_firewall_certificate_exact,
    load_stage_row_projection_exact,
    register_trusted_firewall_verifier,
    replay_component_partition_assignment,
    stage_projection_hashes,
    validate_pretraining_firewall_certificate,
    validate_stage_row_projection,
    verifier_code_sha256,
    verify_offline_row_authority_preparation,
    verify_row_authority_capability,
    write_pretraining_firewall_certificate_exclusive,
    write_stage_row_projection_exclusive,
)
from scripts.r3x.typed_safe_authorities import DEVELOPMENT_PROJECTION_HASH_KEYS
from scripts.r3x.typed_staged_rows import (
    EXACT_FEATURE_FIELDS,
    EXACT_OUTCOME_FIELDS,
    TYPED_FEATURE_STAGE_SCHEMA,
    normalize_development_disclosure,
    normalize_test_feature_disclosure,
    target_free_feature_row_id,
)


SOURCES = {
    INCLUSION_MANIFEST: "a" * 64,
    CONTEXT_SOURCE_CROSSWALK: "b" * 64,
    SOURCE_COMPONENT_SPLIT_MANIFEST: "c" * 64,
}
TRAIN_DATA_SHA = "d" * 64
TEST_FEATURE_SHA = "e" * 64
SPLIT_SEED = 20260808
VERIFIER_ID = "ROW_FIREWALL_REFERENCE_V1"

_COMPONENT_ASSIGNMENT = replay_component_partition_assignment(
    ["component_a", "component_b", "component_c"], SPLIT_SEED
)
TRAIN_COMPONENT = next(
    component
    for component, partition in _COMPONENT_ASSIGNMENT.items()
    if partition == "train"
)
DEVELOPMENT_COMPONENT = next(
    component
    for component, partition in _COMPONENT_ASSIGNMENT.items()
    if partition == "development"
)
TEST_COMPONENT = next(
    component
    for component, partition in _COMPONENT_ASSIGNMENT.items()
    if partition == "internal_test"
)


def _activation() -> StagedActivation:
    return StagedActivation(
        path=Path("/ordinary/activation.json"),
        sha256="f" * 64,
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


def _features(
    *,
    component: str,
    study: str,
    context: str,
    group: str,
    candidate: str = "AGGUAC",
) -> dict:
    return {
        "endpoint": "half_life",
        "publication_context_id": context,
        "region": "5UTR",
        "normalized_source_sequence": "ACGUAC",
        "normalized_candidate_sequence": candidate,
        "edit_list": [{"op": "SUB", "pos": 1, "ref": "C", "alt": "G"}],
        "edit_count": 1,
        "study": study,
        "estimand_id": "half_life_identity_difference",
        "source_component_id": component,
        "source_group_id_v2": group,
    }


def _staged_development_row(
    partition: str,
    *,
    component: str,
    study: str,
    context: str,
    group: str,
) -> dict:
    features = _features(
        component=component,
        study=study,
        context=context,
        group=group,
    )
    return {
        "row_id": target_free_feature_row_id(partition, features),
        "partition": partition,
        "features": features,
        "outcomes": {"delta": 0.25 if partition == "train" else -0.10},
    }


def _typed_development(
    *,
    train_component: str = TRAIN_COMPONENT,
    development_component: str = DEVELOPMENT_COMPONENT,
    train_study: str = "GSE217518",
):
    disclosure = DevelopmentDisclosure(
        activation=_activation(),
        train_rows=(
            _staged_development_row(
                "train",
                component=train_component,
                study=train_study,
                context="hek293_arm_a",
                group="pool_train_a",
            ),
        ),
        development_rows=(
            _staged_development_row(
                "development",
                component=development_component,
                study="GSE217519",
                context="hela_arm_b",
                group="pool_development_a",
            ),
        ),
        capability_status="STAGED_DISCLOSURE_SAFETY_VALIDATED_NOT_TRAINING_READY",
        training_ready=False,
    )
    return normalize_development_disclosure(disclosure)


def _typed_test(
    *,
    component: str = TEST_COMPONENT,
    context: str = "cho_arm_c",
):
    features = _features(
        component=component,
        study="GSE217520",
        context=context,
        group="pool_test_a",
    )
    disclosure = FeatureDisclosure(
        activation=_activation(),
        pre_test_freeze_sha256="4" * 64,
        selected_method_id="FULL_PAIR_UNRESTRICTED_SIAMESE",
        predictor_method_ids=("FULL_PAIR_UNRESTRICTED_SIAMESE",),
        internal_test_rows=(
            {
                "row_id": target_free_feature_row_id("internal_test", features),
                "partition": "internal_test",
                "features": features,
            },
        ),
    )
    return normalize_test_feature_disclosure(disclosure)


def _projections(*, test_component: str = TEST_COMPONENT):
    typed_development = _typed_development()
    typed_test = _typed_test(component=test_component)
    development = build_development_row_projections(
        typed_development,
        source_authority_sha256=SOURCES,
        train_development_outcomes_sha256=TRAIN_DATA_SHA,
        split_seed=SPLIT_SEED,
    )
    test = build_internal_test_feature_row_projections(
        typed_test,
        source_authority_sha256=SOURCES,
        internal_test_features_sha256=TEST_FEATURE_SHA,
        split_seed=SPLIT_SEED,
    )
    return typed_development, typed_test, development, test


def _context(*, test_component: str = TEST_COMPONENT) -> FirewallReplayContext:
    typed_development, typed_test, development, test = _projections(
        test_component=test_component
    )
    return FirewallReplayContext(
        development_projections=development,
        internal_test_feature_projections=test,
        typed_development_rows=typed_development,
        typed_internal_test_feature_rows=typed_test,
        source_authority_sha256=SOURCES,
        train_development_outcomes_sha256=TRAIN_DATA_SHA,
        internal_test_features_sha256=TEST_FEATURE_SHA,
        split_seed=SPLIT_SEED,
    )


def _registry():
    code_sha = verifier_code_sha256(independent_firewall_verifier)
    registration = register_trusted_firewall_verifier(
        VERIFIER_ID,
        independent_firewall_verifier,
        expected_code_sha256=code_sha,
    )
    return {VERIFIER_ID: registration}


def _certificate(context: FirewallReplayContext | None = None):
    context = context or _context()
    return attest_pretraining_firewall(
        context,
        verifier_id=VERIFIER_ID,
        verifier_registry=_registry(),
    )


def _all_keys(value):
    keys = set()
    if isinstance(value, dict):
        for key, nested in value.items():
            keys.add(key)
            keys.update(_all_keys(nested))
    elif isinstance(value, list):
        for nested in value:
            keys.update(_all_keys(nested))
    return keys


def test_stage_projections_are_exact_feature_only_and_hash_key_compatible() -> None:
    _, _, development, test = _projections()

    assert set(stage_projection_hashes(development, stage=DEVELOPMENT_STAGE)) == set(
        DEVELOPMENT_PROJECTION_HASH_KEYS
    )
    assert set(
        stage_projection_hashes(test, stage=INTERNAL_TEST_FEATURE_STAGE)
    ) == set(TEST_FEATURE_PROJECTION_HASH_KEYS)
    for stage, projections in (
        (DEVELOPMENT_STAGE, development),
        (INTERNAL_TEST_FEATURE_STAGE, test),
    ):
        row_sets = set()
        for authority in ROW_AUTHORITIES:
            payload = validate_stage_row_projection(projections[authority])
            assert payload["disclosure_stage"] == stage
            assert payload["authority_name"] == authority
            assert payload["ordinary_only"] is True
            assert payload["sealed_access"] is False
            assert payload["downstream_unlock"] is False
            assert payload["sign_accuracy_gate"] == 0.60
            row_sets.add(payload["row_set_sha256"])
            keys = _all_keys(payload)
            assert not keys.intersection(
                {
                    "delta",
                    "target",
                    "label",
                    "outcome",
                    "outcomes",
                    "normalized_source_sequence",
                    "normalized_candidate_sequence",
                    "path",
                    "payload",
                }
            )
        assert len(row_sets) == 1


def test_crosswalk_and_split_bind_exact_frozen_semantics() -> None:
    _, _, development, _ = _projections()
    crosswalk = development[CONTEXT_SOURCE_CROSSWALK]["entries"][0]
    assert {
        crosswalk["source_mapping_status"],
        crosswalk["conflict_status"],
        crosswalk["component_scope_status"],
    } == {"FROZEN_UNIQUE", "CONFLICT_FREE", "GLOBAL_COMPONENT"}
    assert all(
        crosswalk[field]
        for field in (
            "study",
            "endpoint",
            "estimand_id",
            "publication_context_id",
            "source_group_id_v2",
            "source_component_id",
        )
    )
    split = development[SOURCE_COMPONENT_SPLIT_MANIFEST]
    assert split["split_policy"]["method"] == "SHA256_SEEDED_COMPONENT_ORDER_V1"
    assert split["split_policy"]["seed"] == SPLIT_SEED
    for entry in split["entries"]:
        expected = canonical_sha256(
            {
                "assignment_algorithm_id": "SHA256_SEEDED_COMPONENT_ORDER_V1",
                "seed": SPLIT_SEED,
                "source_component_id": entry["source_component_id"],
            }
        )
        assert entry["component_order_key_sha256"] == expected


def test_component_order_and_partition_replay_have_authoritative_splits_parity() -> None:
    components = [
        "component_z",
        "component_a",
        "component_m",
        "component_b",
        "component_y",
    ]
    expected_order = authoritative_ordered_components(components, SPLIT_SEED)
    actual_assignment = replay_component_partition_assignment(
        components, SPLIT_SEED
    )
    assert list(actual_assignment) == expected_order
    assert actual_assignment == authoritative_partition_by_component(
        components, SPLIT_SEED
    )


def test_disjoint_but_outcome_informed_component_repartition_holds() -> None:
    typed_development = _typed_development(
        train_component=DEVELOPMENT_COMPONENT,
        development_component=TRAIN_COMPONENT,
    )
    typed_test = _typed_test()
    development = build_development_row_projections(
        typed_development,
        source_authority_sha256=SOURCES,
        train_development_outcomes_sha256=TRAIN_DATA_SHA,
        split_seed=SPLIT_SEED,
    )
    test = build_internal_test_feature_row_projections(
        typed_test,
        source_authority_sha256=SOURCES,
        internal_test_features_sha256=TEST_FEATURE_SHA,
        split_seed=SPLIT_SEED,
    )
    context = FirewallReplayContext(
        development_projections=development,
        internal_test_feature_projections=test,
        typed_development_rows=typed_development,
        typed_internal_test_feature_rows=typed_test,
        source_authority_sha256=SOURCES,
        train_development_outcomes_sha256=TRAIN_DATA_SHA,
        internal_test_features_sha256=TEST_FEATURE_SHA,
        split_seed=SPLIT_SEED,
    )
    with pytest.raises(StageRowProjectionError, match="global seeded replay"):
        attest_pretraining_firewall(
            context,
            verifier_id=VERIFIER_ID,
            verifier_registry=_registry(),
        )


def test_offline_full_replay_is_prepared_but_not_runtime_trusted_or_training_ready() -> None:
    context = _context()
    certificate = _certificate(context)
    capability = verify_offline_row_authority_preparation(
        context,
        certificate,
        verifier_registry=_registry(),
    )

    assert capability.status == "PREPARED_OFFLINE_NOT_RUNTIME_TRUSTED"
    assert capability.offline_full_replay_verified is True
    assert capability.row_authorities_ready is False
    assert capability.aggregate_typed_safe_authorities_ready is False
    assert capability.training_ready is False
    assert capability.sealed_access is False
    assert capability.downstream_unlock is False
    assert certificate["row_overlap_count"] == 0
    assert certificate["component_overlap_count"] == 0
    assert certificate["offline_full_replay_verified"] is True
    assert certificate["runtime_trust_status"] == "PREPARED_OFFLINE_NOT_RUNTIME_TRUSTED"
    assert set(certificate["development_projection_hashes"]) == set(
        DEVELOPMENT_PROJECTION_HASH_KEYS
    )
    assert set(certificate["internal_test_feature_projection_hashes"]) == set(
        TEST_FEATURE_PROJECTION_HASH_KEYS
    )
    # Certificates carry aggregate commitments and counts, never record-level
    # support, row IDs, sequences, or outcomes.
    forbidden = {
        "r3_input_row_sha256",
        "entries",
        "sequence",
        "delta",
        "target",
        "label",
        "outcome",
        "outcomes",
    }
    assert not _all_keys(certificate).intersection(forbidden)


@pytest.mark.parametrize("hidden_key", ["delta", "target", "label", "outcomes", "payload"])
def test_hidden_outcome_or_generic_projection_field_holds(hidden_key: str) -> None:
    _, _, development, _ = _projections()
    forged = copy.deepcopy(development[INCLUSION_MANIFEST])
    forged["entries"][0][hidden_key] = 1
    with pytest.raises(StageRowProjectionError, match="exact schema"):
        validate_stage_row_projection(forged)


def test_hidden_outcome_or_extra_typed_feature_field_holds_before_projection() -> None:
    typed = _typed_development()
    forged_train = copy.deepcopy(typed.train[0])
    forged_train["observed_outcome"] = forged_train["delta"]
    forged = replace(typed, train=(forged_train,))
    with pytest.raises(StageRowProjectionError, match="exact schema"):
        build_development_row_projections(
            forged,
            source_authority_sha256=SOURCES,
            train_development_outcomes_sha256=TRAIN_DATA_SHA,
            split_seed=SPLIT_SEED,
        )


def test_direct_builder_rejects_gse246381_defense_in_depth() -> None:
    typed = _typed_development(train_study="GSE246381")
    with pytest.raises(StageRowProjectionError, match="prohibited support"):
        build_development_row_projections(
            typed,
            source_authority_sha256=SOURCES,
            train_development_outcomes_sha256=TRAIN_DATA_SHA,
            split_seed=SPLIT_SEED,
        )


def test_exact_loader_rejects_gse246381_even_in_rehashed_projection(
    tmp_path: Path,
) -> None:
    payload = copy.deepcopy(
        _context().development_projections[CONTEXT_SOURCE_CROSSWALK]
    )
    payload["entries"][0]["study"] = "GSE246381"
    raw = (
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    ).encode()
    path = tmp_path / "development" / "crosswalk.json"
    path.parent.mkdir()
    path.write_bytes(raw)
    with pytest.raises(StageRowProjectionError, match="prohibited support"):
        load_stage_row_projection_exact(
            tmp_path,
            path,
            hashlib.sha256(raw).hexdigest(),
            expected_authority_name=CONTEXT_SOURCE_CROSSWALK,
            expected_stage=DEVELOPMENT_STAGE,
        )


def test_arbitrary_hex_row_identity_is_rejected_even_if_row_set_hash_is_reforged() -> None:
    typed = _typed_development()
    forged_train = {**typed.train[0], "r3_input_row_sha256": "7" * 64}
    forged = replace(
        typed,
        train=(forged_train,),
        train_row_set_sha256=canonical_sha256(["7" * 64]),
    )
    with pytest.raises(StageRowProjectionError, match="feature-only support"):
        build_development_row_projections(
            forged,
            source_authority_sha256=SOURCES,
            train_development_outcomes_sha256=TRAIN_DATA_SHA,
            split_seed=SPLIT_SEED,
        )

    test_row = {**_typed_test()[0], "r3_input_row_sha256": "8" * 64}
    with pytest.raises(StageRowProjectionError, match="feature-only support"):
        build_internal_test_feature_row_projections(
            (test_row,),
            source_authority_sha256=SOURCES,
            internal_test_features_sha256=TEST_FEATURE_SHA,
            split_seed=SPLIT_SEED,
        )


def test_three_projection_row_set_mismatch_holds() -> None:
    context = _context()
    alternate_rows = _typed_test(context="cho_arm_d")
    alternate = build_internal_test_feature_row_projections(
        alternate_rows,
        source_authority_sha256=SOURCES,
        internal_test_features_sha256=TEST_FEATURE_SHA,
        split_seed=SPLIT_SEED,
    )
    mixed = copy.deepcopy(context.internal_test_feature_projections)
    mixed[CONTEXT_SOURCE_CROSSWALK] = alternate[CONTEXT_SOURCE_CROSSWALK]
    forged_context = replace(context, internal_test_feature_projections=mixed)
    with pytest.raises(StageRowProjectionError, match="different row sets"):
        attest_pretraining_firewall(
            forged_context,
            verifier_id=VERIFIER_ID,
            verifier_registry=_registry(),
        )


def test_development_test_component_overlap_holds() -> None:
    context = _context(test_component=TRAIN_COMPONENT)
    with pytest.raises(StageRowProjectionError, match="component overlap"):
        attest_pretraining_firewall(
            context,
            verifier_id=VERIFIER_ID,
            verifier_registry=_registry(),
        )


def test_same_model_row_across_stages_holds_even_if_component_id_is_reforged() -> None:
    context = _context()
    development_row = context.typed_development_rows.development[0]
    features = {field: development_row[field] for field in EXACT_FEATURE_FIELDS}
    features["source_component_id"] = "component_test_reforged"
    test_row = {
        "schema_version": TYPED_FEATURE_STAGE_SCHEMA,
        "r3_input_row_sha256": target_free_feature_row_id(
            "internal_test", features
        ),
        "partition": "internal_test",
        **features,
    }
    test_projections = build_internal_test_feature_row_projections(
        (test_row,),
        source_authority_sha256=SOURCES,
        internal_test_features_sha256=TEST_FEATURE_SHA,
        split_seed=SPLIT_SEED,
    )
    forged_context = replace(
        context,
        typed_internal_test_feature_rows=(test_row,),
        internal_test_feature_projections=test_projections,
    )
    with pytest.raises(StageRowProjectionError, match="row overlap"):
        attest_pretraining_firewall(
            forged_context,
            verifier_id=VERIFIER_ID,
            verifier_registry=_registry(),
        )


def test_crosswalk_status_tamper_holds() -> None:
    _, _, development, _ = _projections()
    forged = copy.deepcopy(development[CONTEXT_SOURCE_CROSSWALK])
    forged["entries"][0]["conflict_status"] = "CONFLICT_ALLOWED"
    with pytest.raises(StageRowProjectionError, match="crosswalk status"):
        validate_stage_row_projection(forged)


def test_feature_commitment_tamper_survives_structural_rehash_but_fails_typed_replay() -> None:
    context = _context()
    forged = copy.deepcopy(context.development_projections)
    projection = forged[INCLUSION_MANIFEST]
    projection["entries"][0]["feature_only_commitment_sha256"] = "9" * 64
    projection["row_commitment_sha256"] = canonical_sha256(
        [
            {
                "r3_input_row_sha256": entry["r3_input_row_sha256"],
                "feature_only_commitment_sha256": entry[
                    "feature_only_commitment_sha256"
                ],
            }
            for entry in projection["entries"]
        ]
    )
    validate_stage_row_projection(projection)
    forged_context = replace(context, development_projections=forged)
    with pytest.raises(StageRowProjectionError, match="row binding|full replay"):
        attest_pretraining_firewall(
            forged_context,
            verifier_id=VERIFIER_ID,
            verifier_registry=_registry(),
        )


@pytest.mark.parametrize("binding", ["source", "data"])
def test_wrong_source_or_data_authority_hash_holds_on_full_replay(binding: str) -> None:
    context = _context()
    if binding == "source":
        wrong_sources = {**SOURCES, INCLUSION_MANIFEST: "8" * 64}
        forged = build_development_row_projections(
            context.typed_development_rows,
            source_authority_sha256=wrong_sources,
            train_development_outcomes_sha256=TRAIN_DATA_SHA,
            split_seed=SPLIT_SEED,
        )
    else:
        forged = build_development_row_projections(
            context.typed_development_rows,
            source_authority_sha256=SOURCES,
            train_development_outcomes_sha256="8" * 64,
            split_seed=SPLIT_SEED,
        )
    forged_context = replace(context, development_projections=forged)
    with pytest.raises(StageRowProjectionError, match="full replay"):
        attest_pretraining_firewall(
            forged_context,
            verifier_id=VERIFIER_ID,
            verifier_registry=_registry(),
        )


def test_wrong_stage_binding_holds() -> None:
    _, _, development, _ = _projections()
    forged = copy.deepcopy(development[INCLUSION_MANIFEST])
    forged["disclosure_stage"] = INTERNAL_TEST_FEATURE_STAGE
    with pytest.raises(StageRowProjectionError, match="schema_version|schema"):
        validate_stage_row_projection(forged)


def test_missing_verifier_id_mismatch_and_code_mismatch_hold() -> None:
    context = _context()
    with pytest.raises(StageRowProjectionError, match="registry"):
        attest_pretraining_firewall(
            context,
            verifier_id=VERIFIER_ID,
            verifier_registry=None,
        )

    good = _registry()[VERIFIER_ID]
    wrong_id = TrustedFirewallVerifier(
        verifier_id="ANOTHER_VERIFIER",
        verifier_code_sha256=good.verifier_code_sha256,
        verifier=good.verifier,
    )
    with pytest.raises(StageRowProjectionError, match="key/ID mismatch"):
        attest_pretraining_firewall(
            context,
            verifier_id=VERIFIER_ID,
            verifier_registry={VERIFIER_ID: wrong_id},
        )

    wrong_code = TrustedFirewallVerifier(
        verifier_id=VERIFIER_ID,
        verifier_code_sha256="0" * 64,
        verifier=good.verifier,
    )
    with pytest.raises(StageRowProjectionError, match="code changed"):
        attest_pretraining_firewall(
            context,
            verifier_id=VERIFIER_ID,
            verifier_registry={VERIFIER_ID: wrong_code},
        )


def test_verifier_disagreement_and_self_reported_certificate_cannot_unlock() -> None:
    context = _context()

    def lying_verifier(replay_context):
        evidence = independent_firewall_verifier(replay_context)
        evidence["development_row_count"] += 1
        return evidence

    lying_id = "LYING_VERIFIER"
    lying_sha = verifier_code_sha256(lying_verifier)
    registry = {
        lying_id: register_trusted_firewall_verifier(
            lying_id,
            lying_verifier,
            expected_code_sha256=lying_sha,
        )
    }
    with pytest.raises(StageRowProjectionError, match="evidence disagrees"):
        attest_pretraining_firewall(
            context,
            verifier_id=lying_id,
            verifier_registry=registry,
        )

    certificate = _certificate(context)
    with pytest.raises(StageRowProjectionError, match="registry"):
        verify_row_authority_capability(context, certificate, verifier_registry=None)


def test_development_runtime_api_replays_only_development_and_stays_hold() -> None:
    context = _context()
    capability = assess_development_runtime_row_authority_capability(
        context.typed_development_rows,
        context.development_projections,
        source_authority_sha256=SOURCES,
        train_development_outcomes_sha256=TRAIN_DATA_SHA,
        split_seed=SPLIT_SEED,
    )
    assert capability.development_projection_verified is True
    assert capability.offline_full_replay_verified is False
    assert capability.row_authorities_ready is False
    assert capability.training_ready is False
    assert capability.certificate_sha256 is None
    assert "ACTIVATION_SCHEMA_DOES_NOT_PREBIND_ROW_FIREWALL_CERTIFICATE_SHA256" in (
        capability.blockers
    )
    parameter_names = set(
        inspect.signature(
            assess_development_runtime_row_authority_capability
        ).parameters
    )
    assert not any("test" in name for name in parameter_names)
    assert "verifier_registry" not in parameter_names
    assert "certificate" not in parameter_names


def test_certificate_structure_rejects_overlap_sign_or_sealed_mutation() -> None:
    certificate = _certificate()
    for field, value in (
        ("component_overlap_count", 1),
        ("sign_accuracy_gate", 0.59),
        ("sealed_access", True),
        ("downstream_unlock", True),
    ):
        forged = copy.deepcopy(certificate)
        forged[field] = value
        with pytest.raises(StageRowProjectionError):
            validate_pretraining_firewall_certificate(forged)


def test_exact_projection_and_certificate_io_no_overwrite(tmp_path: Path) -> None:
    context = _context()
    payload = context.development_projections[INCLUSION_MANIFEST]
    projection_path = tmp_path / "development" / "inclusion.json"
    projection_path.parent.mkdir()
    digest = write_stage_row_projection_exclusive(tmp_path, projection_path, payload)
    assert load_stage_row_projection_exact(
        tmp_path,
        projection_path,
        digest,
        expected_authority_name=INCLUSION_MANIFEST,
        expected_stage=DEVELOPMENT_STAGE,
        expected_source_authority_sha256=SOURCES[INCLUSION_MANIFEST],
        expected_data_authority_sha256=TRAIN_DATA_SHA,
    ) == payload
    with pytest.raises(StageRowProjectionError, match="already exists"):
        write_stage_row_projection_exclusive(tmp_path, projection_path, payload)

    certificate = _certificate(context)
    certificate_path = (
        tmp_path / "authority_preparation" / "firewall" / "firewall.json"
    )
    certificate_path.parent.mkdir(parents=True)
    certificate_sha = write_pretraining_firewall_certificate_exclusive(
        tmp_path, certificate_path, certificate
    )
    assert load_pretraining_firewall_certificate_exact(
        tmp_path, certificate_path, certificate_sha
    ) == certificate
    with pytest.raises(StageRowProjectionError, match="already exists"):
        write_pretraining_firewall_certificate_exclusive(
            tmp_path, certificate_path, certificate
        )


def test_projection_writer_enforces_literal_stage_namespaces(tmp_path: Path) -> None:
    context = _context()
    development_dir = tmp_path / "development"
    development_dir.mkdir()
    offline_test_dir = (
        tmp_path / "authority_preparation" / "internal_test_features"
    )
    offline_test_dir.mkdir(parents=True)

    test_payload = context.internal_test_feature_projections[INCLUSION_MANIFEST]
    with pytest.raises(StageRowProjectionError, match="stage namespace"):
        write_stage_row_projection_exclusive(
            tmp_path,
            development_dir / "test-feature.json",
            test_payload,
        )

    development_payload = context.development_projections[INCLUSION_MANIFEST]
    with pytest.raises(StageRowProjectionError, match="stage namespace"):
        write_stage_row_projection_exclusive(
            tmp_path,
            offline_test_dir / "development.json",
            development_payload,
        )

    test_path = offline_test_dir / "test-feature.json"
    digest = write_stage_row_projection_exclusive(
        tmp_path,
        test_path,
        test_payload,
    )
    assert load_stage_row_projection_exact(
        tmp_path,
        test_path,
        digest,
        expected_authority_name=INCLUSION_MANIFEST,
        expected_stage=INTERNAL_TEST_FEATURE_STAGE,
    ) == test_payload


def test_noncanonical_and_duplicate_json_hold_on_exact_load(tmp_path: Path) -> None:
    payload = _context().development_projections[INCLUSION_MANIFEST]
    development_dir = tmp_path / "development"
    development_dir.mkdir()
    noncanonical = development_dir / "noncanonical.json"
    raw = json.dumps(payload, sort_keys=True).encode()
    noncanonical.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    with pytest.raises(StageRowProjectionError, match="not canonical"):
        load_stage_row_projection_exact(
            tmp_path,
            noncanonical,
            digest,
            expected_authority_name=INCLUSION_MANIFEST,
            expected_stage=DEVELOPMENT_STAGE,
        )

    duplicate = development_dir / "duplicate.json"
    raw = b'{"authority_name":"inclusion_manifest","authority_name":"x"}'
    duplicate.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    with pytest.raises(StageRowProjectionError, match="duplicate JSON key"):
        load_stage_row_projection_exact(
            tmp_path,
            duplicate,
            digest,
            expected_authority_name=INCLUSION_MANIFEST,
            expected_stage=DEVELOPMENT_STAGE,
        )


def test_symlink_input_or_output_path_holds(tmp_path: Path) -> None:
    payload = _context().development_projections[INCLUSION_MANIFEST]
    development_dir = tmp_path / "development"
    development_dir.mkdir()
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    alias_dir = development_dir / "alias"
    alias_dir.symlink_to(real_dir, target_is_directory=True)
    with pytest.raises(StageRowProjectionError, match="symlink"):
        write_stage_row_projection_exclusive(
            tmp_path, alias_dir / "projection.json", payload
        )

    real_file = real_dir / "projection.json"
    raw = canonical_stage_row_projection_bytes(payload)
    real_file.write_bytes(raw)
    alias_file = development_dir / "projection-link.json"
    alias_file.symlink_to(real_file)
    digest = hashlib.sha256(raw).hexdigest()
    with pytest.raises(StageRowProjectionError, match="symlink"):
        load_stage_row_projection_exact(
            tmp_path,
            alias_file,
            digest,
            expected_authority_name=INCLUSION_MANIFEST,
            expected_stage=DEVELOPMENT_STAGE,
        )


def test_published_bytes_are_exact_canonical_json() -> None:
    context = _context()
    projection = context.development_projections[INCLUSION_MANIFEST]
    certificate = _certificate(context)
    assert canonical_stage_row_projection_bytes(projection).endswith(b"\n")
    assert canonical_pretraining_firewall_certificate_bytes(certificate).endswith(b"\n")
