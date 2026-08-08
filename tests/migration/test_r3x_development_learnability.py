from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts.r3x.development_learnability import (
    COMPLETE_STATUS,
    DIAGNOSTIC_ROLE,
    HOLD_STATUS,
    NOISE_CEILING_STATUS,
    SIGN_GATE,
    build_development_learnability_diagnostic,
    canonical_development_learnability_diagnostic_bytes,
    canonical_estimand_registry_semantic_sha256,
    canonical_registered_cell_universe_semantic_sha256,
    canonical_train_development_outcomes_semantic_sha256,
    load_development_learnability_diagnostic_exact,
    validate_development_learnability_diagnostic,
    write_development_learnability_diagnostic_exclusive,
)
from scripts.r3x.protocol import ProtocolHoldError, canonical_sha256
from scripts.r3x.statistics import _paired_sign_scores


def _id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _registry() -> dict:
    common = {
        "estimand_id": "shared_delta",
        "raw_source_field": "source_value",
        "raw_candidate_field": "candidate_value",
        "raw_unit": "normalized_unit",
        "direction": "HIGHER_IS_MORE_STABLE",
        "transform": "IDENTITY_DIFFERENCE",
        "valid_domain": "FINITE",
        "role": "PRIMARY",
        "replicate_policy": "SINGLE_OBSERVATION_ONLY",
        "source_evidence_sha256": "e" * 64,
    }
    return {
        "schema_version": "r3_endpoint_estimand_registry_v1",
        "status": "FROZEN",
        "entries": [
            {**common, "study": "study_a", "endpoint": "endpoint_a"},
            {**common, "study": "study_b", "endpoint": "endpoint_b"},
        ],
    }


def _row(
    study: str,
    endpoint: str,
    partition: str,
    index: int,
    delta: float,
    component: str,
    *,
    context: str | None = None,
    edit_count: int = 1,
) -> dict:
    source = "ACGUACGU"
    edits = []
    candidate = list(source)
    for offset in range(edit_count):
        pos = offset
        ref = source[pos]
        alt = {"A": "C", "C": "G", "G": "U", "U": "A"}[ref]
        edits.append({"op": "SUB", "pos": pos, "ref": ref, "alt": alt})
        candidate[pos] = alt
    return {
        "r3_input_row_sha256": _id(
            f"{study}/{endpoint}/{partition}/{index}/{delta}/{component}/{context}"
        ),
        "record_id": f"ordinary_{study}_{partition}_{index}",
        "partition": partition,
        "study": study,
        "endpoint": endpoint,
        "estimand_id": "shared_delta",
        "publication_context_id": context or f"context_{study}",
        "region": "5UTR",
        "normalized_source_sequence": source,
        "normalized_candidate_sequence": "".join(candidate),
        "edit_list": edits,
        "edit_count": edit_count,
        "source_component_id": component,
        "source_group_id_v2": f"pool_{component}",
        "source_value": 1.0,
        "candidate_value": 1.0 + delta,
        "delta": delta,
    }


def _rows(*, positive_repeats: int = 100) -> list[dict]:
    rows = []
    for index in range(10):
        train_delta = 1.0 if index < 7 else -1.0
        rows.extend([
            _row(
                "study_a", "endpoint_a", "train", index, train_delta,
                f"train_a_{index}",
            ),
            _row(
                "study_b", "endpoint_b", "train", index, train_delta,
                f"train_b_{index}",
            ),
        ])
    # Keep both signs above the frozen disclosure minimum while retaining a
    # descriptive majority baseline above 0.60 after cell-macro averaging.
    for index in range(23):
        rows.append(_row(
            "study_b",
            "endpoint_b",
            "development",
            index,
            1.0 if index < 20 else -1.0,
            f"dev_b_{index}",
        ))
    rows.extend(
        _row("study_a", "endpoint_a", "development", index, 1.0, "dev_a_pos")
        for index in range(positive_repeats)
    )
    # Four positive and eight negative components give formal accuracy 1/3.
    for index in range(3):
        rows.append(_row(
            "study_a",
            "endpoint_a",
            "development",
            positive_repeats + index,
            1.0,
            f"dev_a_pos_singleton_{index}",
        ))
    for index in range(8):
        rows.append(_row(
            "study_a",
            "endpoint_a",
            "development",
            positive_repeats + 3 + index,
            -1.0,
            f"dev_a_neg_{index}",
        ))
    return rows


def _registered_cells(rows: list[dict]) -> list[dict]:
    fields = ("study", "estimand_id", "publication_context_id", "edit_count")
    unique = {
        tuple(row[field] for field in fields): {field: row[field] for field in fields}
        for row in rows
    }
    return [unique[key] for key in sorted(unique)]


def _bindings(rows: list[dict], registry: dict, registered_cells: list[dict]):
    return {
        "train_development_outcomes_sha256": (
            canonical_train_development_outcomes_semantic_sha256(rows)
        ),
        "estimand_registry_sha256": canonical_estimand_registry_semantic_sha256(
            registry
        ),
        "registered_cells": registered_cells,
        "registered_cell_universe_sha256": (
            canonical_registered_cell_universe_semantic_sha256(registered_cells)
        ),
    }


def _build(rows=None, registry=None, registered_cells=None) -> dict:
    actual_rows = _rows() if rows is None else rows
    actual_registry = _registry() if registry is None else registry
    actual_cells = (
        _registered_cells(actual_rows) if registered_cells is None else registered_cells
    )
    return build_development_learnability_diagnostic(
        actual_rows,
        actual_registry,
        **_bindings(actual_rows, actual_registry, actual_cells),
    )


def _exact_kwargs(rows: list[dict], registry: dict | None = None) -> dict:
    actual_registry = _registry() if registry is None else registry
    cells = _registered_cells(rows)
    bindings = _bindings(rows, actual_registry, cells)
    return {
        "expected_rows": rows,
        "expected_estimand_registry": actual_registry,
        "expected_train_development_outcomes_sha256": bindings[
            "train_development_outcomes_sha256"
        ],
        "expected_estimand_registry_sha256": bindings["estimand_registry_sha256"],
        "expected_registered_cells": cells,
        "expected_registered_cell_universe_sha256": bindings[
            "registered_cell_universe_sha256"
        ],
    }


def _cell(payload: dict, study: str) -> dict:
    return next(cell for cell in payload["cells"] if cell["formal_cell"]["study"] == study)


def test_nested_two_study_scope_is_descriptive_and_never_go() -> None:
    payload = _build()
    assert payload["status"] == COMPLETE_STATUS
    assert payload["diagnostic_role"] == DIAGNOSTIC_ROLE
    assert payload["study_endpoint_scope"]["status"] == (
        "DEVELOPMENT_ONLY_STUDY_ENDPOINT_ONE_TO_ONE_NESTED"
    )
    assert payload["study_endpoint_scope"]["publication_cross_study_claim"] is False
    assert payload["policies"]["sign_gate"] == SIGN_GATE == 0.60
    assert payload["policies"]["noise_ceiling_status"] == NOISE_CEILING_STATUS
    assert payload["boundaries"]["go_effect"] is False
    assert payload["boundaries"]["internal_test_disclosed"] is False
    assert payload["boundaries"]["sealed_access"] is False
    assert payload["boundaries"]["model_training_authorized"] is False
    assert payload["boundaries"]["pretest_freeze_eligible"] is False
    # 1/3 and 20/23 are cell-macro averaged, so the descriptive value just
    # exceeds 0.60.  It still cannot become a gate or scientific GO.
    assert payload["aggregate"][
        "cell_macro_development_component_balanced_majority_sign_accuracy"
    ] == pytest.approx(83.0 / 138.0)
    assert payload["aggregate"]["diagnostic_numeric_threshold_reached"] is True
    assert payload["aggregate"]["go_effect"] is False


def test_component_balancing_is_invariant_to_within_component_replication() -> None:
    one = _build(rows=_rows(positive_repeats=1))
    many = _build(rows=_rows(positive_repeats=100))
    one_a = _cell(one, "study_a")
    many_a = _cell(many, "study_a")
    assert one_a["development_component_balanced_majority_sign_accuracy"] == pytest.approx(
        1.0 / 3.0
    )
    assert many_a["development_component_balanced_majority_sign_accuracy"] == pytest.approx(
        1.0 / 3.0
    )
    assert one_a["development_row_micro_majority_sign_accuracy"] == pytest.approx(1.0 / 3.0)
    assert many_a["development_row_micro_majority_sign_accuracy"] == pytest.approx(103.0 / 111.0)
    assert one["aggregate"][
        "cell_macro_development_component_balanced_majority_sign_accuracy"
    ] == many["aggregate"][
        "cell_macro_development_component_balanced_majority_sign_accuracy"
    ]
    assert one["aggregate"][
        "pooled_row_micro_development_majority_sign_accuracy"
    ] != many["aggregate"][
        "pooled_row_micro_development_majority_sign_accuracy"
    ]


def test_endpoint_and_region_do_not_redefine_the_frozen_primary_cell() -> None:
    registry = _registry()
    extra_entry = copy.deepcopy(registry["entries"][0])
    extra_entry["endpoint"] = "endpoint_a_secondary_alias"
    registry["entries"].insert(1, extra_entry)
    rows = _rows(positive_repeats=1)
    for index, partition in enumerate(("train", "development")):
        extra = _row(
            "study_a",
            "endpoint_a_secondary_alias",
            partition,
            50 + index,
            1.0,
            f"alias_{partition}",
            context="context_study_a",
        )
        extra["region"] = "CDS"
        rows.append(extra)
    payload = _build(rows=rows, registry=registry)
    study_a_cells = [
        cell for cell in payload["cells"] if cell["formal_cell"]["study"] == "study_a"
    ]
    assert len(study_a_cells) == 1
    assert set(study_a_cells[0]["formal_cell"]) == {
        "study",
        "estimand_id",
        "publication_context_id",
        "edit_count",
    }


def test_majority_fraction_is_explicitly_not_a_ceiling() -> None:
    payload = _build()
    assert payload["policies"]["majority_prior_interpretation"] == (
        "NAIVE_BASELINE_NOT_A_LEARNABILITY_UPPER_BOUND"
    )
    assert "ceiling" not in payload["aggregate"]
    assert "replicate" not in payload["aggregate"]


def test_component_balanced_majority_tie_selects_positive() -> None:
    rows = _rows(positive_repeats=1)
    train_a = [
        row
        for row in rows
        if row["study"] == "study_a" and row["partition"] == "train"
    ]
    for row in train_a[5:]:
        row["delta"] = -1.0
        row["candidate_value"] = 0.0
    payload = _build(rows=rows)
    assert _cell(payload, "study_a")["train_majority_sign"] == 1


def test_diagnostic_primary_sign_matches_frozen_statistics_kernel() -> None:
    rows = _rows(positive_repeats=3)
    payload = _build(rows=rows)
    development = [row for row in rows if row["partition"] == "development"]
    predictions = []
    prior_by_cell = {
        (
            cell["formal_cell"]["study"],
            cell["formal_cell"]["estimand_id"],
            cell["formal_cell"]["publication_context_id"],
            cell["formal_cell"]["edit_count"],
        ): float(cell["train_majority_sign"])
        for cell in payload["cells"]
    }
    for row in development:
        key = (
            row["study"],
            row["estimand_id"],
            row["publication_context_id"],
            row["edit_count"],
        )
        predictions.append(prior_by_cell[key])
    formal = _paired_sign_scores(development, predictions, predictions)
    assert formal["candidate"] == pytest.approx(
        payload["aggregate"][
            "cell_macro_development_component_balanced_majority_sign_accuracy"
        ]
    )


@pytest.mark.parametrize("protected", ["partition", "metadata"])
def test_internal_test_and_protected_rows_hold_before_artifact(protected: str) -> None:
    rows = _rows()
    if protected == "partition":
        rows[0]["partition"] = "internal_test"
        expected = "HOLD_TEST_PARTITION_PREMATURELY_DISCLOSED"
    else:
        rows[0]["metadata"] = {"role": "sealed_final"}
        expected = "HOLD_PROTECTED_INPUT"
    with pytest.raises(ProtocolHoldError) as exc:
        _build(rows=rows)
    assert exc.value.code == expected


@pytest.mark.parametrize("edit_count", [0, 2, 4, 6, True])
def test_only_frozen_primary_edit_counts_are_accepted(edit_count) -> None:
    rows = _rows()
    rows[0]["edit_count"] = edit_count
    with pytest.raises(ProtocolHoldError) as exc:
        _build(rows=rows)
    assert exc.value.code == "HOLD_DEVELOPMENT_LEARNABILITY_ACTION_BUDGET"


def test_formal_cell_missing_one_partition_is_a_structured_hold() -> None:
    rows = _rows(positive_repeats=1)
    # Keep study_a/endpoint_a represented in both partitions, but split the
    # publication context so neither formal cell has both partitions.
    for row in rows:
        if row["study"] == "study_a":
            row["publication_context_id"] = (
                "context_a_train_only"
                if row["partition"] == "train"
                else "context_a_development_only"
            )
    payload = _build(rows=rows)
    assert payload["status"] == HOLD_STATUS
    assert payload["aggregate"]["n_undefined_cells"] == 2
    assert payload["aggregate"]["diagnostic_numeric_threshold_reached"] is None
    reasons = {
        tuple(cell["undefined_reasons"])
        for cell in payload["cells"]
        if cell["formal_cell"]["study"] == "study_a"
    }
    assert reasons == {
        ("MISSING_DEVELOPMENT_PARTITION",),
        ("MISSING_TRAIN_PARTITION",),
    }
    assert payload["aggregate"]["go_effect"] is False


@pytest.mark.parametrize("identity_field", ["source_component_id", "source_group_id_v2"])
def test_component_and_ranking_group_cannot_cross_partitions(identity_field: str) -> None:
    rows = _rows(positive_repeats=1)
    train = next(row for row in rows if row["study"] == "study_a" and row["partition"] == "train")
    development = next(
        row for row in rows if row["study"] == "study_a" and row["partition"] == "development"
    )
    development[identity_field] = train[identity_field]
    with pytest.raises(ProtocolHoldError) as exc:
        _build(rows=rows)
    assert exc.value.code == "HOLD_DEVELOPMENT_LEARNABILITY_PARTITION_LEAKAGE"


def test_same_component_can_appear_in_multiple_cells_within_one_partition() -> None:
    rows = _rows(positive_repeats=1)
    sources = [
        row
        for row in rows
        if row["study"] == "study_a" and row["partition"] == "train"
    ]
    for index, source in enumerate(sources):
        extra = copy.deepcopy(source)
        extra["r3_input_row_sha256"] = _id(f"same-component-second-cell-{index}")
        extra["publication_context_id"] = "context_a_second"
        rows.append(extra)
    payload = _build(rows=rows)
    assert payload["status"] == HOLD_STATUS
    assert payload["aggregate"]["n_undefined_cells"] == 1


def test_true_zero_is_excluded_from_sign_accuracy_when_diversity_is_safe() -> None:
    rows = _rows(positive_repeats=1)
    victim = next(
        row
        for row in rows
        if row["study"] == "study_a"
        and row["partition"] == "development"
        and row["source_component_id"] == "dev_a_pos"
    )
    victim["delta"] = 0.0
    victim["candidate_value"] = victim["source_value"]
    payload = _build(rows=rows)
    cell = _cell(payload, "study_a")
    assert payload["status"] == COMPLETE_STATUS
    assert cell["partition_summaries"]["development"]["n_zero_rows"] == 1
    assert cell["development_component_balanced_majority_sign_accuracy"] == pytest.approx(
        3.0 / 11.0
    )


def test_homogeneous_zero_outcomes_are_suppressed_before_artifact() -> None:
    rows = _rows(positive_repeats=1)
    for row in rows:
        if row["study"] == "study_a" and row["partition"] == "development":
            row["delta"] = 0.0
            row["candidate_value"] = row["source_value"]
    with pytest.raises(ProtocolHoldError) as exc:
        _build(rows=rows)
    assert exc.value.code == (
        "HOLD_DEVELOPMENT_LEARNABILITY_ATTRIBUTE_DIVERSITY_SUPPRESSION"
    )
    assert "n_positive_rows" not in exc.value.details
    assert "n_negative_rows" not in exc.value.details


@pytest.mark.parametrize("delta", [-1.0, 1.0])
def test_homogeneous_nonzero_signs_are_suppressed_before_artifact(
    delta: float,
) -> None:
    rows = _rows(positive_repeats=1)
    for row in rows:
        if row["study"] == "study_b" and row["partition"] == "development":
            row["delta"] = delta
            row["candidate_value"] = row["source_value"] + delta
    with pytest.raises(ProtocolHoldError) as exc:
        _build(rows=rows)
    assert exc.value.code == (
        "HOLD_DEVELOPMENT_LEARNABILITY_ATTRIBUTE_DIVERSITY_SUPPRESSION"
    )


def test_delta_and_endpoint_aliases_must_replay_registry_semantics() -> None:
    rows = _rows(positive_repeats=1)
    rows[0]["delta"] = -7.0
    with pytest.raises(ProtocolHoldError) as exc:
        _build(rows=rows)
    assert exc.value.code == "HOLD_DEVELOPMENT_LEARNABILITY_ESTIMAND_REPLAY"

    rows = _rows(positive_repeats=1)
    rows[0]["endpoint_id"] = "different_endpoint"
    with pytest.raises(ProtocolHoldError) as exc:
        _build(rows=rows)
    assert exc.value.code == "HOLD_DEVELOPMENT_LEARNABILITY_ESTIMAND_REPLAY"


def test_semantic_authority_hashes_are_not_arbitrary_caller_strings() -> None:
    rows = _rows(positive_repeats=1)
    registry = _registry()
    cells = _registered_cells(rows)
    good = _bindings(rows, registry, cells)
    with pytest.raises(ProtocolHoldError) as exc:
        build_development_learnability_diagnostic(
            rows,
            registry,
            **{**good, "train_development_outcomes_sha256": "f" * 64},
        )
    assert exc.value.code == "HOLD_DEVELOPMENT_LEARNABILITY_AUTHORITY"
    with pytest.raises(ProtocolHoldError) as exc:
        build_development_learnability_diagnostic(
            rows,
            registry,
            **{**good, "estimand_registry_sha256": "f" * 64},
        )
    assert exc.value.code == "HOLD_DEVELOPMENT_LEARNABILITY_AUTHORITY"


def test_fully_missing_registered_cell_remains_visible_and_holds() -> None:
    rows = _rows(positive_repeats=1)
    registered = _registered_cells(rows)
    registered.append({
        "study": "study_a",
        "estimand_id": "shared_delta",
        "publication_context_id": "frozen_but_missing_context",
        "edit_count": 3,
    })
    payload = _build(rows=rows, registered_cells=registered)
    assert payload["status"] == HOLD_STATUS
    missing = next(
        cell
        for cell in payload["cells"]
        if cell["formal_cell"]["publication_context_id"] == "frozen_but_missing_context"
    )
    assert missing["undefined_reasons"] == [
        "MISSING_TRAIN_PARTITION",
        "MISSING_DEVELOPMENT_PARTITION",
    ]
    assert payload["aggregate"]["diagnostic_numeric_threshold_reached"] is None


def test_cell_and_aggregate_tampering_fail_internal_replay() -> None:
    payload = _build()
    tampered = copy.deepcopy(payload)
    tampered["aggregate"][
        "cell_macro_development_component_balanced_majority_sign_accuracy"
    ] = 1.0
    with pytest.raises(ProtocolHoldError) as exc:
        validate_development_learnability_diagnostic(tampered)
    assert exc.value.code == "HOLD_DEVELOPMENT_LEARNABILITY_REPLAY"

    tampered = copy.deepcopy(payload)
    tampered["cells"][0]["partition_summaries"]["development"][
        "sum_component_positive_fraction_nonzero"
    ] += 1.0
    with pytest.raises(ProtocolHoldError) as exc:
        validate_development_learnability_diagnostic(tampered)
    assert exc.value.code in {
        "HOLD_DEVELOPMENT_LEARNABILITY_SCHEMA",
        "HOLD_DEVELOPMENT_LEARNABILITY_REPLAY",
    }


def test_structural_validator_replays_cell_universe_and_partition_counts() -> None:
    payload = _build()
    tampered = copy.deepcopy(payload)
    tampered["registered_cell_universe_sha256"] = "f" * 64
    with pytest.raises(ProtocolHoldError) as exc:
        validate_development_learnability_diagnostic(tampered)
    assert exc.value.code == "HOLD_DEVELOPMENT_LEARNABILITY_REPLAY"

    tampered = copy.deepcopy(payload)
    tampered["input_binding"]["partitions"]["development"]["n_rows"] += 1
    without_sha = {
        "schema_version": tampered["input_binding"]["schema_version"],
        "partitions": tampered["input_binding"]["partitions"],
    }
    tampered["input_binding"]["overall_binding_sha256"] = canonical_sha256(
        without_sha
    )
    with pytest.raises(ProtocolHoldError) as exc:
        validate_development_learnability_diagnostic(tampered)
    assert exc.value.code == "HOLD_DEVELOPMENT_LEARNABILITY_REPLAY"


def test_builder_is_order_invariant_and_omits_identity_records_and_magnitudes() -> None:
    rows = _rows(positive_repeats=4)
    forward = _build(rows=rows)
    reverse = _build(rows=list(reversed(rows)))
    assert canonical_development_learnability_diagnostic_bytes(forward) == (
        canonical_development_learnability_diagnostic_bytes(reverse)
    )
    artifact_text = canonical_development_learnability_diagnostic_bytes(forward).decode()
    assert rows[0]["r3_input_row_sha256"] not in artifact_text
    assert '"delta"' not in artifact_text
    assert rows[0]["source_component_id"] not in artifact_text
    assert rows[0]["source_group_id_v2"] not in artifact_text
    assert "component_sufficient_statistics_sha256" not in artifact_text
    assert "mean_abs_delta" not in artifact_text
    assert "sum_abs_delta" not in artifact_text
    assert forward["boundaries"][
        "semantic_content_hashes_replayed_by_structure_validator"
    ] is False
    assert forward["boundaries"]["trust_requires_exact_input_replay"] is True
    assert forward["boundaries"]["aggregate_confidentiality_claim"] is False


def test_many_singletons_are_bound_by_inputs_not_emitted_linearly() -> None:
    rows = []
    for index in range(10):
        train_delta = 1.0 if index < 7 else -1.0
        rows.extend([
            _row(
                "study_a", "endpoint_a", "train", index, train_delta,
                f"train_a_{index}",
            ),
            _row(
                "study_b", "endpoint_b", "train", index, train_delta,
                f"train_b_{index}",
            ),
        ])
    for index in range(23):
        rows.append(_row(
            "study_b",
            "endpoint_b",
            "development",
            index,
            1.0 if index < 20 else -1.0,
            f"dev_b_{index}",
        ))
    for index in range(2_000):
        delta = (1.0 if index % 2 == 0 else -1.0) * (index + 1) / 10_000.0
        rows.append(
            _row(
                "study_a",
                "endpoint_a",
                "development",
                index,
                delta,
                f"unique_component_{index}",
            )
        )
    raw = canonical_development_learnability_diagnostic_bytes(_build(rows=rows))
    assert len(raw) < 50_000
    assert b"unique_component_0" not in raw
    assert b"unique_component_1999" not in raw


def test_small_cell_outcome_summary_is_suppressed_before_artifact() -> None:
    rows = _rows(positive_repeats=1)
    rows[:] = [
        row
        for row in rows
        if not (
            row["study"] == "study_b"
            and row["partition"] == "development"
            and int(row["record_id"].rsplit("_", 1)[-1]) >= 9
        )
    ]
    with pytest.raises(ProtocolHoldError) as exc:
        _build(rows=rows)
    assert exc.value.code == "HOLD_DEVELOPMENT_LEARNABILITY_SMALL_CELL_SUPPRESSION"


def test_exclusive_write_and_exact_input_bound_replay(tmp_path: Path) -> None:
    rows = _rows(positive_repeats=3)
    payload = _build(rows=rows)
    path = tmp_path / "development-learnability.json"
    digest = write_development_learnability_diagnostic_exclusive(
        path,
        payload,
        **_exact_kwargs(rows),
    )
    reversed_rows = list(reversed(rows))
    loaded = load_development_learnability_diagnostic_exact(
        path,
        digest,
        **_exact_kwargs(reversed_rows),
    )
    assert loaded == payload
    with pytest.raises(ProtocolHoldError) as exc:
        write_development_learnability_diagnostic_exclusive(
            path,
            payload,
            **_exact_kwargs(rows),
        )
    assert exc.value.code == "HOLD_DEVELOPMENT_LEARNABILITY_ARTIFACT_IO"


def test_exact_loader_rejects_other_inputs_even_with_valid_artifact(tmp_path: Path) -> None:
    rows = _rows(positive_repeats=2)
    payload = _build(rows=rows)
    path = tmp_path / "development-learnability.json"
    digest = write_development_learnability_diagnostic_exclusive(
        path,
        payload,
        **_exact_kwargs(rows),
    )
    other = copy.deepcopy(rows)
    other[2]["delta"] = -1.0
    other[2]["candidate_value"] = 0.0
    with pytest.raises(ProtocolHoldError) as exc:
        load_development_learnability_diagnostic_exact(
            path,
            digest,
            **_exact_kwargs(other),
        )
    assert exc.value.code == "HOLD_DEVELOPMENT_LEARNABILITY_INPUT_BINDING"


def test_exact_loader_rejects_duplicate_keys_and_noncanonical_bytes(tmp_path: Path) -> None:
    rows = _rows(positive_repeats=1)
    payload = _build(rows=rows)
    raw = canonical_development_learnability_diagnostic_bytes(payload)
    parsed = json.loads(raw)
    noncanonical = json.dumps(parsed, separators=(",", ":")).encode("utf-8")
    path = tmp_path / "noncanonical.json"
    path.write_bytes(noncanonical)
    digest = hashlib.sha256(noncanonical).hexdigest()
    with pytest.raises(ProtocolHoldError) as exc:
        load_development_learnability_diagnostic_exact(
            path,
            digest,
            **_exact_kwargs(rows),
        )
    assert exc.value.code == "HOLD_DEVELOPMENT_LEARNABILITY_ARTIFACT_IO"
