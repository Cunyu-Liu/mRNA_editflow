"""Information-set isolation and dataset materialization tests."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.r3x.dataset import (  # noqa: E402
    build_examples,
    materialize_method_rows,
    preflight_and_load_examples,
)
from scripts.r3x.information_sets import (  # noqa: E402
    FoundationQualification,
    InformationSet,
    InformationSetError,
    assert_same_information_set,
    build_feature_view,
    information_set_allowlist,
    information_set_fingerprint,
)
from scripts.r3x.protocol import (  # noqa: E402
    ProtocolHoldError,
    REQUIRED_STAGED_DISCLOSURE_CAPABILITIES,
    STAGED_DISCLOSURE_HOLD_CODE,
)


def _row(**overrides) -> dict:
    row = {
        "r3_input_row_sha256": "row_1",
        "study": "study_a",
        "endpoint": "endpoint_a",
        "publication_context_id": "cell_a",
        "region": "5UTR",
        "normalized_source_sequence": "AAAA",
        "normalized_candidate_sequence": "CAAA",
        "edit_list": [{"op": "SUB", "pos": 0, "source_nt": "A", "target_nt": "C"}],
        "edit_count": 1,
        "source_value": 1.2,
        "source_anchor_status": "MEASURED_SINGLE_VALUE",
        "delta": 0.4,
        "source_component_id": "component_1",
        "source_group_id_v2": "ranking_pool_1",
        "partition": "development",
        "foundation_embedding": [0.1, 0.2, 0.3],
        "foundation_model_id": "fixture-fm",
        "foundation_exposure_stratum": "NO_CONFIRMED_MEMBER_EXPOSURE",
        "metadata": {"unused": True},
    }
    row.update(overrides)
    return row


def _qualification(**overrides) -> FoundationQualification:
    fields = {
        "status": "QUALIFIED",
        "license_status": "QUALIFIED",
        "exposure_status": "QUALIFIED",
        "exposure_audit_sha256": "a" * 64,
        "license_audit_sha256": "b" * 64,
        "feature_manifest_sha256": "c" * 64,
    }
    fields.update(overrides)
    return FoundationQualification(**fields)


def test_six_information_sets_have_exact_distinct_allowlists():
    assert set(InformationSet) == {
        InformationSet.NO_ANCHOR_ZERO_PRIOR,
        InformationSet.EDIT_ONLY,
        InformationSet.CANDIDATE_ONLY,
        InformationSet.PAIRED_SEQUENCE_NO_MEASURED_ANCHOR,
        InformationSet.MEASURED_SOURCE_ANCHOR_ASSISTED,
        InformationSet.FOUNDATION_EXPOSURE_STRATIFIED,
    }
    assert information_set_allowlist(InformationSet.NO_ANCHOR_ZERO_PRIOR) == (
        "endpoint",
        "publication_context_id",
        "region",
    )
    assert information_set_allowlist(InformationSet.EDIT_ONLY) == (
        "endpoint",
        "publication_context_id",
        "region",
        "edit_list",
        "edit_count",
    )
    assert information_set_allowlist(InformationSet.CANDIDATE_ONLY) == (
        "endpoint",
        "publication_context_id",
        "region",
        "normalized_candidate_sequence",
    )


def test_public_canonical_fingerprints_are_frozen_distinct_and_view_derived():
    expected = {
        InformationSet.NO_ANCHOR_ZERO_PRIOR: (
            "12e9deb0118d8aaa17922a2feaf9b625f2bc41453930431a1abcfa4bee70135d"
        ),
        InformationSet.EDIT_ONLY: (
            "925d74ece68ea7baac7a9a292487f013d9d8e72bfcbcee7114ab58c50a42f6f9"
        ),
        InformationSet.CANDIDATE_ONLY: (
            "21f52095b165e91c01468dc683a24abe8983e2f1818b18aab22e2900a16295c2"
        ),
        InformationSet.PAIRED_SEQUENCE_NO_MEASURED_ANCHOR: (
            "bf369369d9110f28a9f5d648ddf068c79b31ec57675f00d93d9de87c49f1384a"
        ),
        InformationSet.MEASURED_SOURCE_ANCHOR_ASSISTED: (
            "bfa14accf6d089044cc50e9177a311edc3d475388e7100928ce244a573fd0ba9"
        ),
    }
    assert {
        information_set_fingerprint(info) for info in expected
    } == set(expected.values())
    for info, digest in expected.items():
        assert information_set_fingerprint(info) == digest
        assert information_set_fingerprint(info.value) == digest
        assert build_feature_view(_row(), info).fingerprint == digest


def test_public_fingerprint_rejects_unknown_and_unqualified_foundation():
    with pytest.raises(InformationSetError) as unknown:
        information_set_fingerprint("CALLER_SELECTED_INFORMATION_SET")
    assert unknown.value.code == "HOLD_UNKNOWN_INFORMATION_SET"
    with pytest.raises(InformationSetError) as foundation:
        information_set_fingerprint(
            InformationSet.FOUNDATION_EXPOSURE_STRATIFIED
        )
    assert foundation.value.code == "NOT_RUN_FOUNDATION_UNQUALIFIED"


@pytest.mark.parametrize(
    "information_set",
    [
        InformationSet.NO_ANCHOR_ZERO_PRIOR,
        InformationSet.EDIT_ONLY,
        InformationSet.CANDIDATE_ONLY,
        InformationSet.PAIRED_SEQUENCE_NO_MEASURED_ANCHOR,
    ],
)
def test_direct_delta_information_sets_never_expose_measured_source_value(information_set):
    view = build_feature_view(_row(), information_set)
    assert "source_value" not in view.values
    assert "source_anchor_status" not in view.values
    assert "delta" not in view.values
    assert "study" not in view.values
    assert "source_group_id_v2" not in view.values


def test_candidate_only_cannot_inspect_source_sequence_or_edit_script():
    view = build_feature_view(_row(), InformationSet.CANDIDATE_ONLY)
    assert "normalized_source_sequence" not in view.values
    assert "edit_list" not in view.values
    assert view.values["normalized_candidate_sequence"] == "CAAA"


def test_paired_direct_delta_uses_both_sequences_without_measured_anchor():
    view = build_feature_view(
        _row(), InformationSet.PAIRED_SEQUENCE_NO_MEASURED_ANCHOR
    )
    assert view.values["normalized_source_sequence"] == "AAAA"
    assert view.values["normalized_candidate_sequence"] == "CAAA"
    assert "source_value" not in view.values


def test_anchor_assisted_information_set_is_a_separate_leaderboard():
    paired = build_feature_view(
        _row(), InformationSet.PAIRED_SEQUENCE_NO_MEASURED_ANCHOR
    )
    anchor = build_feature_view(
        _row(), InformationSet.MEASURED_SOURCE_ANCHOR_ASSISTED
    )
    assert anchor.values["source_value"] == pytest.approx(1.2)
    assert anchor.values["source_anchor_status"] == "MEASURED_SINGLE_VALUE"
    with pytest.raises(InformationSetError) as exc:
        assert_same_information_set([paired, anchor])
    assert exc.value.code == "HOLD_UNEQUAL_INFORMATION_SET"


@pytest.mark.parametrize(
    "qualification",
    [
        None,
        _qualification(status="PROPOSED"),
        _qualification(license_status="UNKNOWN"),
        _qualification(exposure_status="UNKNOWN"),
        _qualification(exposure_audit_sha256="not-a-hash"),
    ],
)
def test_foundation_features_require_frozen_exposure_and_license_qualification(
    qualification,
):
    with pytest.raises(InformationSetError) as exc:
        build_feature_view(
            _row(),
            InformationSet.FOUNDATION_EXPOSURE_STRATIFIED,
            foundation_qualification=qualification,
        )
    assert exc.value.code == "NOT_RUN_FOUNDATION_UNQUALIFIED"


def test_qualified_foundation_view_binds_audit_hashes_in_fingerprint():
    first = build_feature_view(
        _row(),
        InformationSet.FOUNDATION_EXPOSURE_STRATIFIED,
        foundation_qualification=_qualification(),
    )
    repeat = build_feature_view(
        _row(r3_input_row_sha256="row_2"),
        InformationSet.FOUNDATION_EXPOSURE_STRATIFIED,
        foundation_qualification=_qualification(),
    )
    changed = build_feature_view(
        _row(),
        InformationSet.FOUNDATION_EXPOSURE_STRATIFIED,
        foundation_qualification=_qualification(feature_manifest_sha256="d" * 64),
    )
    assert first.fingerprint == repeat.fingerprint
    assert first.fingerprint == information_set_fingerprint(
        InformationSet.FOUNDATION_EXPOSURE_STRATIFIED,
        foundation_qualification=_qualification(),
    )
    assert first.fingerprint != changed.fingerprint
    assert first.values["foundation_embedding"] == (0.1, 0.2, 0.3)


def test_feature_view_is_immutable_and_carries_per_field_provenance():
    view = build_feature_view(_row(), InformationSet.EDIT_ONLY)
    with pytest.raises(TypeError):
        view.values["edit_count"] = 2
    with pytest.raises(TypeError):
        view.provenance["edit_count"] = "other"
    assert view.provenance["edit_count"] == "ordinary_row:edit_count"
    assert isinstance(view.values["edit_list"], tuple)


def test_same_information_set_views_are_comparable_despite_different_values():
    first = build_feature_view(_row(), InformationSet.EDIT_ONLY)
    second = build_feature_view(
        _row(r3_input_row_sha256="row_2", edit_count=3), InformationSet.EDIT_ONLY
    )
    assert assert_same_information_set([first, second]) == first.fingerprint


def test_target_is_kept_outside_feature_mapping():
    examples = build_examples(
        [_row(), _row(r3_input_row_sha256="row_2", delta=-0.2)],
        InformationSet.PAIRED_SEQUENCE_NO_MEASURED_ANCHOR,
        target_field="delta",
    )
    assert tuple(example.target for example in examples) == (0.4, -0.2)
    assert all("delta" not in example.features.values for example in examples)
    assert all(example.target_field == "delta" for example in examples)


def test_nonfinite_or_missing_target_holds_dataset_materialization():
    with pytest.raises(InformationSetError) as exc:
        build_examples([_row(delta=float("nan"))], InformationSet.EDIT_ONLY)
    assert exc.value.code == "HOLD_INVALID_TARGET"


def test_preflight_failure_precedes_label_loader_and_output_creation(tmp_path: Path):
    # Even a metadata-complete v2 assertion cannot bypass the unimplemented
    # physical disclosure and predictor-provenance validator.
    manifest = {
        "schema_version": "r3_activation_manifest_v2_staged_disclosure",
        "status": "FROZEN",
        "mode": "ORDINARY_ONLY",
        "staged_disclosure": {
            key: True for key in REQUIRED_STAGED_DISCLOSURE_CAPABILITIES
        },
    }
    path = tmp_path / "activation.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    called = []
    output = tmp_path / "future-output"

    def label_loader():
        called.append(True)
        return [_row()]

    with pytest.raises(ProtocolHoldError) as exc:
        preflight_and_load_examples(
            activation_manifest_path=path,
            activation_manifest_sha256=digest,
            prospective_output_dir=output,
            row_loader=label_loader,
            information_set=InformationSet.EDIT_ONLY,
        )
    assert exc.value.code == STAGED_DISCLOSURE_HOLD_CODE
    assert exc.value.details["missing_capabilities"] == []
    assert sorted(exc.value.details["invalid_capabilities"]) == sorted(
        REQUIRED_STAGED_DISCLOSURE_CAPABILITIES
    )
    assert called == []
    assert not output.exists()


def test_guard_return_does_not_fall_through_to_row_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    path = tmp_path / "activation.json"
    path.write_text("{}", encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    called = []
    monkeypatch.setattr(
        "scripts.r3x.dataset.require_staged_disclosure_before_outcome_access",
        lambda *_args, **_kwargs: None,
    )
    with pytest.raises(ProtocolHoldError) as exc:
        preflight_and_load_examples(
            activation_manifest_path=path,
            activation_manifest_sha256=digest,
            prospective_output_dir=tmp_path / "future-output",
            row_loader=lambda: called.append(True) or [_row()],
            information_set=InformationSet.EDIT_ONLY,
        )
    assert exc.value.code == "HOLD_STAGED_DISCLOSURE_STATE_MACHINE_NOT_IMPLEMENTED"
    assert called == []


@pytest.mark.parametrize("materializer", ["examples", "method_rows"])
def test_protected_rows_cannot_enter_direct_materializers(materializer: str):
    protected = _row(
        partition="internal_test",
        is_sealed=True,
        metadata={"role": "sealed_final"},
    )
    with pytest.raises(ProtocolHoldError) as exc:
        if materializer == "examples":
            build_examples([protected], InformationSet.EDIT_ONLY)
        else:
            materialize_method_rows([protected], InformationSet.EDIT_ONLY)
    assert exc.value.code == "HOLD_PROTECTED_INPUT"


@pytest.mark.parametrize("materializer", ["examples", "method_rows"])
def test_real_internal_test_rows_require_staged_release(materializer: str):
    unreleased = _row(partition="internal_test")
    with pytest.raises(ProtocolHoldError) as exc:
        if materializer == "examples":
            build_examples([unreleased], InformationSet.EDIT_ONLY)
        else:
            materialize_method_rows(
                [unreleased],
                InformationSet.PAIRED_SEQUENCE_NO_MEASURED_ANCHOR,
                target_fields=(),
            )
    assert exc.value.code == "HOLD_TEST_OUTCOME_NOT_RELEASED"
    assert exc.value.details["test_feature_release"] is False
    assert exc.value.details["test_outcome_release"] is False


def test_synthetic_engineering_internal_test_fixture_is_explicitly_scoped():
    from scripts.r3x.dataset import _synthetic_engineering_materialization_scope

    synthetic = _row(partition="internal_test")
    with _synthetic_engineering_materialization_scope():
        projected, _ = materialize_method_rows(
            [synthetic], InformationSet.EDIT_ONLY, target_fields=("delta",)
        )
    assert projected[0]["delta"] == pytest.approx(0.4)


def test_synthetic_scope_is_not_a_row_field_bypass():
    forged = _row(
        partition="internal_test",
        synthetic_engineering_fixture=True,
        study="synthetic_study",
        record_id="synthetic_001",
    )
    with pytest.raises(ProtocolHoldError) as exc:
        materialize_method_rows([forged], InformationSet.EDIT_ONLY)
    assert exc.value.code == "HOLD_TEST_OUTCOME_NOT_RELEASED"


def test_missing_or_unknown_partition_cannot_bypass_release():
    for partition in (None, "renamed_test"):
        row = _row(partition=partition)
        with pytest.raises(ProtocolHoldError) as exc:
            materialize_method_rows([row], InformationSet.EDIT_ONLY)
        assert exc.value.code == "HOLD_PARTITION_AUTHORITY_REQUIRED"


def test_study_id_is_never_smuggled_through_extra_metadata():
    view = build_feature_view(
        _row(study="study_secret", metadata={"study": "study_secret"}),
        InformationSet.MEASURED_SOURCE_ANCHOR_ASSISTED,
    )
    serialized = json.dumps(dict(view.values), sort_keys=True)
    assert "study_secret" not in serialized


def test_conflicting_context_or_sequence_aliases_hold_instead_of_silent_choice():
    with pytest.raises(InformationSetError) as exc:
        build_feature_view(
            _row(context_id_v2="different"),
            InformationSet.PAIRED_SEQUENCE_NO_MEASURED_ANCHOR,
        )
    assert exc.value.code == "HOLD_CONFLICTING_INFORMATION_ALIAS"
    with pytest.raises(InformationSetError) as exc:
        build_feature_view(
            _row(source_sequence="CCCC"),
            InformationSet.PAIRED_SEQUENCE_NO_MEASURED_ANCHOR,
        )
    assert exc.value.code == "HOLD_CONFLICTING_INFORMATION_ALIAS"


def test_method_materialization_drops_every_nonallowlisted_feature():
    rows, fingerprint = materialize_method_rows(
        [_row(study="hidden_study", source_value=99.0)],
        InformationSet.EDIT_ONLY,
    )
    assert len(fingerprint) == 64
    assert "source_value" not in rows[0]
    assert "normalized_source_sequence" not in rows[0]
    assert "study" in rows[0]  # audit/sampler metadata, never a feature input
