from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import pytest

from scripts.r3x.paired_artifact import (
    PairedArtifactProtocolError,
    build_verified_paired_reference,
    build_paired_comparison_artifact,
    load_verified_paired_reference_exact,
    load_paired_comparison_artifact_exact,
    validate_paired_comparison_artifact,
    validate_verified_paired_reference,
    write_paired_comparison_artifact_exclusive,
)
from scripts.r3x.statistics import PRIMARY_PAIRED_METRIC_ID


def _rows() -> list[dict]:
    rows = []
    for index in range(10):
        rows.append({
            "r3_input_row_sha256": f"{index + 1:064x}",
            "record_id": f"row_{index}",
            "study": "study",
            "estimand_id": "estimand",
            "context_id_v2": "ctx",
            "actual_edit_count": 1,
            "source_component_id": f"component_{index}",
            "delta": 1.0 if index % 2 == 0 else -1.0,
        })
    return rows


def _policy() -> dict:
    return {
        "schema_version": "r3x_paired_strongest_baseline_comparison_policy_v1",
        "status": "FROZEN",
        "policy_id": "paired_policy",
        "primary_candidate_model_id": "A_EXACT_POTENTIAL_DIFFERENCE",
        "strongest_baseline_id": "FULL_PAIR_UNRESTRICTED_SIAMESE",
        "information_set": "PAIRED_SEQUENCE_NO_MEASURED_ANCHOR",
        "comparison_family_id": "ordinary_primary_family",
        "comparison_family_member_ids": ["a_vs_fullpair_sign", "secondary_comparison"],
        "primary_comparison_member_id": "a_vs_fullpair_sign",
        "primary_sign_inference": {
            "metric_id": PRIMARY_PAIRED_METRIC_ID,
            "alpha": 0.05,
            "confidence": 0.95,
            "bootstrap": {
                "n_resamples": 2000,
                "seed": 23,
                "min_components_per_cell": 10,
                "max_attempt_multiplier": 20,
            },
            "permutation": {
                "n_permutations": 10000,
                "seed": 42,
                "exact_component_limit": 20,
                "min_components_per_cell": 10,
            },
        },
    }


def _inputs() -> tuple[list[dict], list[float], list[float], dict]:
    rows = _rows()
    candidate = [float(row["delta"]) for row in rows]
    baseline = [-value for value in candidate]
    kwargs = {
        "policy": _policy(),
        "policy_sha256": "a" * 64,
        "authority_sha256": {
            "ready_dataset_sha256": "b" * 64,
            "metric_zero_policy_registry_sha256": "c" * 64,
        },
        "candidate_model_id": "A_EXACT_POTENTIAL_DIFFERENCE",
        "baseline_model_id": "FULL_PAIR_UNRESTRICTED_SIAMESE",
        "candidate_information_set_fingerprint": "d" * 64,
        "baseline_information_set_fingerprint": "d" * 64,
        "candidate_prediction_artifact_sha256": "e" * 64,
        "baseline_prediction_artifact_sha256": "f" * 64,
        "additional_family_raw_p_values": {"secondary_comparison": 0.20},
        "additional_family_evidence_sha256": {"secondary_comparison": "1" * 64},
    }
    return rows, candidate, baseline, kwargs


def _build() -> dict:
    rows, candidate, baseline, kwargs = _inputs()
    return build_paired_comparison_artifact(rows, candidate, baseline, **kwargs)


def test_builds_strict_recomputable_primary_sign_artifact() -> None:
    payload = _build()
    assert payload["schema_version"] == "r3x_paired_comparison_result_v2"
    assert payload["status"] == "PASS_PAIRED_STRONGEST_BASELINE"
    assert payload["sealed_access"] is False
    assert payload["downstream_unlock"] is False
    assert payload["candidate_row_order_sha256"] == payload["baseline_row_order_sha256"]
    assert payload["candidate_row_set_sha256"] == payload["baseline_row_set_sha256"]

    primary = payload["primary_sign_utility"]
    assert primary["candidate_raw_value"] == pytest.approx(1.0)
    assert primary["baseline_raw_value"] == pytest.approx(0.0)
    assert primary["candidate_minus_baseline"] == pytest.approx(1.0)
    assert primary["confidence_interval_candidate_minus_baseline"] == pytest.approx([1.0, 1.0])
    assert primary["raw_p_value_two_sided"] == pytest.approx(2.0 / 1024.0)
    assert primary["holm_adjusted_p_value"] == pytest.approx(4.0 / 1024.0)
    assert primary["bootstrap"]["resampling_unit"] == "GLOBAL_SOURCE_RELATION_COMPONENT"
    assert primary["permutation"]["component_swap_is_synchronized_across_cells"] is True
    cell = primary["cell_support"]["study|estimand|ctx|1"]
    assert cell["n_components"] == 10
    assert cell["n_nonzero_rows"] == 10
    assert cell["candidate_raw_sign_utility"] == pytest.approx(1.0)
    assert cell["baseline_raw_sign_utility"] == pytest.approx(0.0)

    family = payload["multiple_comparison_family"]
    assert family["frozen_member_ids"] == ["a_vs_fullpair_sign", "secondary_comparison"]
    assert set(family["raw_p_values_two_sided"]) == set(family["frozen_member_ids"])
    assert set(family["holm_adjusted_p_values"]) == set(family["frozen_member_ids"])
    assert set(family["member_evidence_sha256"]) == set(family["frozen_member_ids"])


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("additional_family_raw_p_values", {}, "missing=\\['secondary_comparison'\\]"),
        (
            "additional_family_raw_p_values",
            {"secondary_comparison": 0.2, "unfrozen_member": 0.3},
            "extra=\\['unfrozen_member'\\]",
        ),
        ("additional_family_evidence_sha256", {}, "missing=\\['secondary_comparison'\\]"),
        (
            "additional_family_evidence_sha256",
            {"secondary_comparison": "1" * 64, "unfrozen_member": "2" * 64},
            "extra=\\['unfrozen_member'\\]",
        ),
    ],
)
def test_unknown_missing_or_extra_family_members_fail_closed(
    field: str, value: dict, match: str
) -> None:
    rows, candidate, baseline, kwargs = _inputs()
    kwargs[field] = value
    with pytest.raises(PairedArtifactProtocolError, match=match):
        build_paired_comparison_artifact(rows, candidate, baseline, **kwargs)


def test_policy_must_freeze_primary_member_and_all_inference_settings() -> None:
    rows, candidate, baseline, kwargs = _inputs()
    kwargs["policy"] = copy.deepcopy(kwargs["policy"])
    kwargs["policy"]["primary_comparison_member_id"] = "posthoc_member"
    with pytest.raises(PairedArtifactProtocolError, match="absent from the frozen family"):
        build_paired_comparison_artifact(rows, candidate, baseline, **kwargs)

    kwargs["policy"] = _policy()
    del kwargs["policy"]["primary_sign_inference"]["bootstrap"]["seed"]
    with pytest.raises(PairedArtifactProtocolError, match="keys mismatch"):
        build_paired_comparison_artifact(rows, candidate, baseline, **kwargs)


def test_candidate_baseline_and_information_set_are_exactly_policy_bound() -> None:
    rows, candidate, baseline, kwargs = _inputs()
    kwargs["candidate_model_id"] = "B_SPARSE_EDIT_SET_ADDITIVE"
    with pytest.raises(PairedArtifactProtocolError, match="candidate_model_id"):
        build_paired_comparison_artifact(rows, candidate, baseline, **kwargs)

    rows, candidate, baseline, kwargs = _inputs()
    kwargs["baseline_information_set_fingerprint"] = "0" * 64
    with pytest.raises(PairedArtifactProtocolError, match="fingerprints differ"):
        build_paired_comparison_artifact(rows, candidate, baseline, **kwargs)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda value: value["primary_sign_utility"].__setitem__("candidate_raw_value", 0.5),
        lambda value: value["primary_sign_utility"].__setitem__(
            "confidence_interval_candidate_minus_baseline", [0.1, 1.0]
        ),
        lambda value: value["multiple_comparison_family"]["raw_p_values_two_sided"].__setitem__(
            "secondary_comparison", 0.19
        ),
        lambda value: value["bindings"].__setitem__("policy_sha256", "0" * 64),
        lambda value: value.__setitem__("unexpected_posthoc_field", True),
    ],
)
def test_validator_recomputes_and_rejects_any_field_drift(mutator) -> None:
    rows, candidate, baseline, kwargs = _inputs()
    payload = build_paired_comparison_artifact(rows, candidate, baseline, **kwargs)
    changed = copy.deepcopy(payload)
    mutator(changed)
    with pytest.raises(PairedArtifactProtocolError):
        validate_paired_comparison_artifact(
            changed, rows, candidate, baseline, **kwargs
        )


def test_row_order_and_row_set_are_distinct_and_recomputed() -> None:
    rows, candidate, baseline, kwargs = _inputs()
    original = build_paired_comparison_artifact(rows, candidate, baseline, **kwargs)
    order = [9, 0, 1, 2, 3, 4, 5, 6, 7, 8]
    changed_rows = [rows[index] for index in order]
    changed_candidate = [candidate[index] for index in order]
    changed_baseline = [baseline[index] for index in order]
    changed = build_paired_comparison_artifact(
        changed_rows, changed_candidate, changed_baseline, **kwargs
    )
    assert changed["candidate_row_set_sha256"] == original["candidate_row_set_sha256"]
    assert changed["candidate_row_order_sha256"] != original["candidate_row_order_sha256"]
    with pytest.raises(PairedArtifactProtocolError, match="does not exactly match"):
        validate_paired_comparison_artifact(
            original, changed_rows, changed_candidate, changed_baseline, **kwargs
        )


def test_duplicate_row_identity_and_prediction_misalignment_fail_closed() -> None:
    rows, candidate, baseline, kwargs = _inputs()
    rows[1]["r3_input_row_sha256"] = rows[0]["r3_input_row_sha256"]
    with pytest.raises(PairedArtifactProtocolError, match="duplicate"):
        build_paired_comparison_artifact(rows, candidate, baseline, **kwargs)

    rows, candidate, baseline, kwargs = _inputs()
    with pytest.raises(PairedArtifactProtocolError, match="one-to-one"):
        build_paired_comparison_artifact(rows, candidate[:-1], baseline, **kwargs)


def test_exclusive_writer_and_exact_loader_round_trip(tmp_path: Path) -> None:
    rows, candidate, baseline, kwargs = _inputs()
    payload = build_paired_comparison_artifact(rows, candidate, baseline, **kwargs)
    path = tmp_path / "paired.json"
    digest = write_paired_comparison_artifact_exclusive(path, payload)
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    assert path.read_bytes().endswith(b"\n")
    loaded = load_paired_comparison_artifact_exact(
        path, digest, rows, candidate, baseline, **kwargs
    )
    assert loaded == payload
    reference = build_verified_paired_reference(
        artifact_relative_path="paired.json",
        artifact_sha256=digest,
        artifact_payload=loaded,
    )
    assert validate_verified_paired_reference(reference) == reference
    context = {
        "rows": rows,
        "candidate_prediction": candidate,
        "baseline_prediction": baseline,
        "build_kwargs": kwargs,
    }
    assert load_verified_paired_reference_exact(
        tmp_path, reference, context
    ) == reference
    changed_reference = copy.deepcopy(reference)
    changed_reference["artifact_sha256"] = "0" * 64
    with pytest.raises(PairedArtifactProtocolError, match="does not match"):
        validate_verified_paired_reference(changed_reference)
    with pytest.raises(PairedArtifactProtocolError, match="already exists"):
        write_paired_comparison_artifact_exclusive(path, payload)
    with pytest.raises(PairedArtifactProtocolError, match="SHA-256 mismatch"):
        load_paired_comparison_artifact_exact(
            path, "0" * 64, rows, candidate, baseline, **kwargs
        )


def test_verified_reference_loader_rejects_symlink_and_hardlink_aliases(
    tmp_path: Path,
) -> None:
    rows, candidate, baseline, kwargs = _inputs()
    payload = build_paired_comparison_artifact(rows, candidate, baseline, **kwargs)
    context = {
        "rows": rows,
        "candidate_prediction": candidate,
        "baseline_prediction": baseline,
        "build_kwargs": kwargs,
    }

    hard_root = tmp_path / "hard"
    hard_root.mkdir()
    hard_path = hard_root / "paired.json"
    digest = write_paired_comparison_artifact_exclusive(hard_path, payload)
    os.link(hard_path, hard_root / "second-link.json")
    hard_reference = build_verified_paired_reference(
        artifact_relative_path="paired.json",
        artifact_sha256=digest,
        artifact_payload=payload,
    )
    with pytest.raises(PairedArtifactProtocolError, match="single-link"):
        load_verified_paired_reference_exact(hard_root, hard_reference, context)

    symbolic_root = tmp_path / "symbolic"
    symbolic_root.mkdir()
    target = symbolic_root / "target.json"
    digest = write_paired_comparison_artifact_exclusive(target, payload)
    alias = symbolic_root / "paired.json"
    alias.symlink_to(target.name)
    symbolic_reference = build_verified_paired_reference(
        artifact_relative_path="paired.json",
        artifact_sha256=digest,
        artifact_payload=payload,
    )
    with pytest.raises(PairedArtifactProtocolError, match="symlink"):
        load_verified_paired_reference_exact(symbolic_root, symbolic_reference, context)


def test_exact_loader_rejects_validly_hashed_but_tampered_payload(tmp_path: Path) -> None:
    rows, candidate, baseline, kwargs = _inputs()
    payload = build_paired_comparison_artifact(rows, candidate, baseline, **kwargs)
    payload["primary_sign_utility"]["candidate_minus_baseline"] = 0.5
    path = tmp_path / "tampered.json"
    raw = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    path.write_bytes(raw)
    with pytest.raises(PairedArtifactProtocolError, match="inconsistent|does not exactly match"):
        load_paired_comparison_artifact_exact(
            path, hashlib.sha256(raw).hexdigest(), rows, candidate, baseline, **kwargs
        )


def test_writer_refuses_protected_namespace_and_extra_fields(tmp_path: Path) -> None:
    payload = _build()
    protected = tmp_path / "sealed_results"
    protected.mkdir()
    with pytest.raises(PairedArtifactProtocolError, match="HOLD_PROTECTED_INPUT"):
        write_paired_comparison_artifact_exclusive(protected / "paired.json", payload)

    payload["posthoc"] = "not allowed"
    with pytest.raises(PairedArtifactProtocolError, match="keys mismatch"):
        write_paired_comparison_artifact_exclusive(tmp_path / "paired.json", payload)
