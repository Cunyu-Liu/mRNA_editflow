"""Adversarial tests for the physical v2 staged-disclosure boundary."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import scripts.r3x.staged_disclosure as staged_module  # noqa: E402
from scripts.r3x.artifact_graph import make_artifact_ref  # noqa: E402
from scripts.r3x.artifact_io import canonical_json_bytes  # noqa: E402
from scripts.r3x.protocol import REQUIRED_NEGATIVE_CONTROLS, canonical_sha256  # noqa: E402
from scripts.r3x.staged_commits import (  # noqa: E402
    ORDINARY_TEST_OUTCOME_RELEASE_RECORD,
    OUTCOME_RELEASE_CLAIM_RELATIVE_PATH,
    PRE_TEST_FREEZE_COMMIT,
    REQUIRED_NONDATA_AUTHORITY_HASHES,
    STATE_RELATIVE_PATHS,
    TEST_PREDICTION_COMMIT,
    build_staged_bindings,
    required_predictor_unit_ids,
    write_development_commit_exclusive,
    write_pre_test_freeze_commit_exclusive,
    write_test_prediction_commit_exclusive,
)
from scripts.r3x.staged_disclosure import (  # noqa: E402
    ACTIVATION_SCHEMA,
    StagedDisclosureError,
    disclose_internal_test_features,
    disclose_internal_test_outcomes,
    validate_development_stage,
)


TRAIN_ID = hashlib.sha256(b"train-row").hexdigest()
DEVELOPMENT_ID = hashlib.sha256(b"development-row").hexdigest()
TEST_ID_A = hashlib.sha256(b"test-row-a").hexdigest()
TEST_ID_B = hashlib.sha256(b"test-row-b").hexdigest()
OTHER_TEST_ID = hashlib.sha256(b"other-test-row").hexdigest()
FEATURE_FIELDS = ["context_id", "edit_count", "source_sequence", "variant_sequence"]
OUTCOME_FIELDS = ["delta_effect"]
METHOD_ID = "PAIRED_KMER_RIDGE"


def _feature_payload(index: int) -> dict:
    return {
        "context_id": f"ordinary_context_{index}",
        "edit_count": 1,
        "source_sequence": "ACGU",
        "variant_sequence": "AGGU",
    }


def _write_json(path: Path, payload: dict) -> str:
    path.write_text(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonl_bytes(rows: list[dict]) -> bytes:
    return (
        "".join(
            json.dumps(
                row,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
            for row in rows
        )
    ).encode("utf-8")


def _write_bytes(path: Path, raw: bytes) -> str:
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _write_canonical_commit(path: Path, payload: dict) -> str:
    return _write_bytes(path, canonical_json_bytes(payload))


def _row_set(*row_ids: str) -> str:
    return canonical_sha256(sorted(row_ids))


@dataclass
class Fixture:
    root: Path
    manifest_path: Path
    manifest_sha: str
    manifest: dict
    train_development_path: Path
    feature_path: Path
    outcome_path: Path
    selector_path: Path
    predictor_path: Path
    freeze_path: Path
    freeze_sha: str
    freeze: dict
    predictions_path: Path
    predictions_sha: str
    prediction_commit_path: Path
    prediction_commit_sha: str
    prediction_commit: dict


def _build_fixture(
    tmp_path: Path,
    *,
    train_development_rows: list[dict] | None = None,
    feature_rows: list[dict] | None = None,
    outcome_rows: list[dict] | None = None,
    prediction_rows: list[dict] | None = None,
    train_development_raw: bytes | None = None,
    feature_raw: bytes | None = None,
    outcome_raw: bytes | None = None,
    manifest_extra: dict | None = None,
) -> Fixture:
    tmp_path.mkdir(parents=True, exist_ok=True)
    train_development_rows = train_development_rows or [
        {
            "row_id": TRAIN_ID,
            "partition": "train",
            "features": _feature_payload(0),
            "outcomes": {"delta_effect": 0.4},
        },
        {
            "row_id": DEVELOPMENT_ID,
            "partition": "development",
            "features": _feature_payload(1),
            "outcomes": {"delta_effect": -0.2},
        },
    ]
    feature_rows = feature_rows or [
        {
            "row_id": TEST_ID_A,
            "partition": "internal_test",
            "features": _feature_payload(2),
        },
        {
            "row_id": TEST_ID_B,
            "partition": "internal_test",
            "features": _feature_payload(3),
        },
    ]
    outcome_rows = outcome_rows or [
        {
            "row_id": TEST_ID_A,
            "partition": "internal_test",
            "outcomes": {"delta_effect": 0.7},
        },
        {
            "row_id": TEST_ID_B,
            "partition": "internal_test",
            "outcomes": {"delta_effect": -0.5},
        },
    ]
    prediction_rows = prediction_rows or [
        {
            "row_id": TEST_ID_A,
            "partition": "internal_test",
            "predictions": {METHOD_ID: 0.6},
        },
        {
            "row_id": TEST_ID_B,
            "partition": "internal_test",
            "predictions": {METHOD_ID: -0.4},
        },
    ]

    train_development_path = tmp_path / "train_development.jsonl"
    feature_path = tmp_path / "internal_test_features.jsonl"
    outcome_path = tmp_path / "internal_test_outcomes.jsonl"
    train_sha = _write_bytes(
        train_development_path,
        train_development_raw
        if train_development_raw is not None
        else _jsonl_bytes(train_development_rows),
    )
    feature_sha = _write_bytes(
        feature_path,
        feature_raw if feature_raw is not None else _jsonl_bytes(feature_rows),
    )
    outcome_sha = _write_bytes(
        outcome_path,
        outcome_raw if outcome_raw is not None else _jsonl_bytes(outcome_rows),
    )

    nondata_dir = tmp_path / "nondata"
    nondata_dir.mkdir()
    nondata_authorities = {}
    for index, name in enumerate(REQUIRED_NONDATA_AUTHORITY_HASHES):
        if name in staged_module.STAGE_PROJECTED_NONDATA_AUTHORITIES:
            source_sha = hashlib.sha256(
                f"unopened-source:{name}".encode("utf-8")
            ).hexdigest()
            projection_refs = {}
            for stage in ("development", "internal_test_features"):
                projection_path = nondata_dir / f"{index:02d}_{name}_{stage}.json"
                projection_sha = _write_json(
                    projection_path,
                    {
                        "schema_version": (
                            f"r3x_{name}_{stage}_summary_projection_v1"
                        ),
                        "status": "FROZEN",
                        "authority_name": name,
                        "source_authority_sha256": source_sha,
                        "disclosure_stage": stage,
                        "contains_outcomes": False,
                        "partition_scope": (
                            ["development", "train"]
                            if stage == "development"
                            else ["internal_test_features"]
                        ),
                        "record_count": 2,
                        "content_commitment_sha256": hashlib.sha256(
                            f"content:{name}:{stage}".encode("utf-8")
                        ).hexdigest(),
                        "semantic_validation_certificate_sha256": hashlib.sha256(
                            f"certificate:{name}:{stage}".encode("utf-8")
                        ).hexdigest(),
                    },
                )
                projection_refs[stage] = {
                    "path": str(projection_path.relative_to(tmp_path)),
                    "sha256": projection_sha,
                    "status": "FROZEN",
                }
            nondata_authorities[name] = {
                "status": "FROZEN_WITH_STAGE_PROJECTIONS",
                "source_sha256": source_sha,
                "development_projection": projection_refs["development"],
                "internal_test_feature_projection": projection_refs[
                    "internal_test_features"
                ],
            }
            continue
        authority_path = nondata_dir / f"{index:02d}_{name}.json"
        authority_sha = _write_json(
            authority_path,
            {
                "schema_version": f"fixture_{name}_v1",
                "status": "FROZEN",
                "authority_name": name,
            },
        )
        nondata_authorities[name] = {
            "path": str(authority_path.relative_to(tmp_path)),
            "sha256": authority_sha,
            "status": "FROZEN",
        }

    row_sets = {
        "train": _row_set(TRAIN_ID),
        "development": _row_set(DEVELOPMENT_ID),
        "internal_test": _row_set(TEST_ID_A, TEST_ID_B),
    }
    manifest = {
        "schema_version": ACTIVATION_SCHEMA,
        "status": "FROZEN",
        "mode": "ORDINARY_ONLY",
        "selected_route": "BENCHMARK_CRITIC_ROUTE_A_THEN_B_ORDINARY_ONLY",
        "sign_accuracy_gate": 0.60,
        "sign_gate_change_authorized": False,
        "ordinary_only": True,
        "sealed_access": False,
        "downstream_unlock": False,
        "overwrite_authorized": False,
        "model_feature_fields": FEATURE_FIELDS,
        "outcome_fields": OUTCOME_FIELDS,
        "partition_row_set_sha256": row_sets,
        "authorities": {
            "train_development_outcomes": {
                "role": "ORDINARY_TRAIN_DEVELOPMENT_OUTCOMES",
                "path": train_development_path.name,
                "sha256": train_sha,
                "status": "FROZEN",
            },
            "internal_test_features": {
                "role": "ORDINARY_INTERNAL_TEST_FEATURES_ONLY",
                "path": feature_path.name,
                "sha256": feature_sha,
                "status": "FROZEN",
            },
            "internal_test_outcomes": {
                "role": "ORDINARY_INTERNAL_TEST_OUTCOMES_ONLY",
                "path": outcome_path.name,
                "sha256": outcome_sha,
                "status": "FROZEN",
            },
        },
        "nondata_authorities": nondata_authorities,
    }
    if manifest_extra:
        manifest.update(manifest_extra)
    manifest_path = tmp_path / "staged_activation.json"
    manifest_sha = _write_json(manifest_path, manifest)

    def artifact_ref(path: Path, digest: str) -> dict:
        return make_artifact_ref(tmp_path, path, digest)

    selector_path = tmp_path / "development" / "selector.json"
    selector_path.parent.mkdir()
    selector_sha = _write_json(
        selector_path,
        {
            "schema_version": "r3x_development_strongest_baseline_selection_v2",
            "artifact_role": "ORDINARY_DEVELOPMENT_STRONGEST_BASELINE_SELECTION",
            "status": "COMMITTED",
            "selected_method_id": METHOD_ID,
        },
    )
    predictor_path = tmp_path / "development" / "predictor_provenance.json"
    predictor_sha = _write_json(
        predictor_path,
        {
            "schema_version": "r3x_development_predictor_provenance_v2",
            "artifact_role": "ORDINARY_DEVELOPMENT_EXACT_PREDICTOR_PROVENANCE",
            "status": "COMPLETE_EXACT_PREDICTOR_RECOMPUTATION",
            "method_id": METHOD_ID,
        },
    )
    predictions_path = tmp_path / "internal_test_predictions.jsonl"
    predictions_sha = _write_bytes(predictions_path, _jsonl_bytes(prediction_rows))
    dev_audit_path = tmp_path / "audits" / "development.json"
    dev_audit_path.parent.mkdir()
    dev_audit_sha = _write_json(dev_audit_path, {"status": "NO_TEST_ACCESS"})
    freeze_audit_path = tmp_path / "audits" / "pre_test.json"
    freeze_audit_sha = _write_json(freeze_audit_path, {"status": "NO_TEST_ACCESS"})

    bindings = build_staged_bindings(
        run_start_sha256="1" * 64,
        activation_sha256=manifest_sha,
        authority_hashes={
            "train_development_outcomes": train_sha,
            "internal_test_features": feature_sha,
            "internal_test_outcomes": outcome_sha,
        },
        nondata_authority_hashes={
            name: (
                nondata_authorities[name]["source_sha256"]
                if name in staged_module.STAGE_PROJECTED_NONDATA_AUTHORITIES
                else nondata_authorities[name]["sha256"]
            )
            for name in REQUIRED_NONDATA_AUTHORITY_HASHES
        },
        nondata_projection_hashes={
            f"{name}_{stage}_projection": nondata_authorities[name][field][
                "sha256"
            ]
            for name in staged_module.STAGE_PROJECTED_NONDATA_AUTHORITIES
            for stage, field in (
                ("development", "development_projection"),
                (
                    "internal_test_features",
                    "internal_test_feature_projection",
                ),
            )
        },
        n_rows=len(feature_rows),
        ordered_row_ids_sha256=row_sets["internal_test"],
        row_set_sha256=row_sets["internal_test"],
        required_method_ids=[METHOD_ID],
        required_estimand_ids=["estimand_a"],
        required_seeds=[11, 23, 42, 67, 101],
        required_control_ids=REQUIRED_NEGATIVE_CONTROLS,
        code_sha256=nondata_authorities["analysis_code_tree_manifest"][
            "source_sha256"
        ],
        config_sha256=nondata_authorities["model_optimizer_seed_config"][
            "source_sha256"
        ],
    )
    development = write_development_commit_exclusive(
        tmp_path,
        bindings=bindings,
        development_authority_ref=artifact_ref(train_development_path, train_sha),
        development_artifact_refs={
            "PREDICTOR_PROVENANCE": artifact_ref(predictor_path, predictor_sha),
            "SELECTOR": artifact_ref(selector_path, selector_sha),
        },
        access_audit_ref=artifact_ref(dev_audit_path, dev_audit_sha),
        access_assertions={
            "internal_test_feature_accessed": False,
            "internal_test_outcome_accessed": False,
        },
    )
    checkpoint_dir = tmp_path / "frozen" / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    checkpoint_refs = {}
    for index, unit in enumerate(required_predictor_unit_ids(bindings)):
        checkpoint_path = checkpoint_dir / f"{index:03d}.bin"
        checkpoint_sha = _write_bytes(
            checkpoint_path, f"checkpoint:{unit}".encode("utf-8")
        )
        checkpoint_refs[unit] = artifact_ref(checkpoint_path, checkpoint_sha)
    freeze_result = write_pre_test_freeze_commit_exclusive(
        tmp_path,
        bindings=bindings,
        development_commit_ref=development["ref"],
        selector_ref=artifact_ref(selector_path, selector_sha),
        predictor_provenance_refs={
            METHOD_ID: artifact_ref(predictor_path, predictor_sha)
        },
        selected_method_id=METHOD_ID,
        frozen_checkpoint_refs=checkpoint_refs,
        access_audit_ref=artifact_ref(freeze_audit_path, freeze_audit_sha),
        access_assertions={
            "internal_test_feature_accessed": False,
            "internal_test_outcome_accessed": False,
        },
    )
    unit_prediction_dir = tmp_path / "predictions" / "units"
    unit_prediction_dir.mkdir(parents=True)
    unit_prediction_refs = {}
    for index, unit in enumerate(required_predictor_unit_ids(bindings)):
        unit_path = unit_prediction_dir / f"{index:03d}.jsonl"
        unit_sha = _write_bytes(unit_path, f"prediction:{unit}\n".encode("utf-8"))
        unit_prediction_refs[unit] = artifact_ref(unit_path, unit_sha)
    prediction_result = write_test_prediction_commit_exclusive(
        tmp_path,
        bindings=bindings,
        development_commit_ref=development["ref"],
        pre_test_freeze_commit_ref=freeze_result["ref"],
        feature_authority_ref=artifact_ref(feature_path, feature_sha),
        evaluation_predictions_ref=artifact_ref(predictions_path, predictions_sha),
        frozen_prediction_refs=unit_prediction_refs,
        access_assertions={
            "internal_test_feature_accessed": True,
            "internal_test_outcome_accessed": False,
        },
    )
    freeze = freeze_result["payload"]
    freeze_path = Path(freeze_result["path"])
    freeze_sha = freeze_result["sha256"]
    prediction_commit = prediction_result["payload"]
    prediction_commit_path = Path(prediction_result["path"])
    prediction_commit_sha = prediction_result["sha256"]
    return Fixture(
        root=tmp_path,
        manifest_path=manifest_path,
        manifest_sha=manifest_sha,
        manifest=manifest,
        train_development_path=train_development_path,
        feature_path=feature_path,
        outcome_path=outcome_path,
        selector_path=selector_path,
        predictor_path=predictor_path,
        freeze_path=freeze_path,
        freeze_sha=freeze_sha,
        freeze=freeze,
        predictions_path=predictions_path,
        predictions_sha=predictions_sha,
        prediction_commit_path=prediction_commit_path,
        prediction_commit_sha=prediction_commit_sha,
        prediction_commit=prediction_commit,
    )


def _spy_exact_reads(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    reads: list[Path] = []
    original = staged_module.read_exact_bytes

    def spy(path, expected_sha256, *, purpose):
        reads.append(Path(path).resolve())
        return original(path, expected_sha256, purpose=purpose)

    monkeypatch.setattr(staged_module, "read_exact_bytes", spy)
    return reads


def test_each_stage_opens_only_its_authorized_data_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_fixture(tmp_path)
    development_projection_paths = {
        (fixture.root / reference["development_projection"]["path"]).resolve()
        for reference in fixture.manifest["nondata_authorities"].values()
    }
    feature_projection_paths = {
        (
            fixture.root
            / reference["internal_test_feature_projection"]["path"]
        ).resolve()
        for reference in fixture.manifest["nondata_authorities"].values()
    }
    reads = _spy_exact_reads(monkeypatch)

    development = validate_development_stage(
        fixture.manifest_path, fixture.manifest_sha
    )
    assert [row["row_id"] for row in development.train_rows] == [TRAIN_ID]
    assert [row["row_id"] for row in development.development_rows] == [
        DEVELOPMENT_ID
    ]
    assert development.capability_status == (
        "STAGED_DISCLOSURE_SAFETY_VALIDATED_NOT_TRAINING_READY"
    )
    assert development.training_ready is False
    assert fixture.train_development_path.resolve() in reads
    assert fixture.feature_path.resolve() not in reads
    assert fixture.outcome_path.resolve() not in reads
    assert development_projection_paths <= set(reads)
    assert feature_projection_paths.isdisjoint(reads)

    reads.clear()
    features = disclose_internal_test_features(
        fixture.manifest_path,
        fixture.manifest_sha,
        pre_test_freeze_path=fixture.freeze_path,
        pre_test_freeze_sha256=fixture.freeze_sha,
    )
    assert features.selected_method_id == METHOD_ID
    assert features.predictor_method_ids == (METHOD_ID,)
    assert {row["row_id"] for row in features.internal_test_rows} == {
        TEST_ID_A,
        TEST_ID_B,
    }
    assert fixture.feature_path.resolve() in reads
    assert fixture.train_development_path.resolve() not in reads
    assert fixture.outcome_path.resolve() not in reads
    assert feature_projection_paths <= set(reads)
    assert development_projection_paths.isdisjoint(reads)

    reads.clear()
    outcomes = disclose_internal_test_outcomes(
        fixture.manifest_path,
        fixture.manifest_sha,
        test_prediction_commit_path=fixture.prediction_commit_path,
        test_prediction_commit_sha256=fixture.prediction_commit_sha,
    )
    assert outcomes.selected_method_id == METHOD_ID
    assert outcomes.method_ids == (METHOD_ID,)
    assert {row["row_id"] for row in outcomes.outcome_rows} == {
        TEST_ID_A,
        TEST_ID_B,
    }
    assert fixture.train_development_path.resolve() not in reads
    assert fixture.feature_path.resolve() not in reads
    assert development_projection_paths.isdisjoint(reads)
    assert feature_projection_paths.isdisjoint(reads)
    assert fixture.predictions_path.resolve() in reads
    assert (
        fixture.root / STATE_RELATIVE_PATHS[ORDINARY_TEST_OUTCOME_RELEASE_RECORD]
    ).is_file()
    assert (fixture.root / OUTCOME_RELEASE_CLAIM_RELATIVE_PATH).is_file()


def test_invalid_freeze_stops_before_feature_or_outcome_reader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_fixture(tmp_path)
    fixture.freeze["selected_method_id"] = "UNBOUND_METHOD"
    bad_freeze_sha = _write_canonical_commit(fixture.freeze_path, fixture.freeze)
    reads = _spy_exact_reads(monkeypatch)

    with pytest.raises(StagedDisclosureError, match="DAG replay|selected_method"):
        disclose_internal_test_features(
            fixture.manifest_path,
            fixture.manifest_sha,
            pre_test_freeze_path=fixture.freeze_path,
            pre_test_freeze_sha256=bad_freeze_sha,
        )
    assert fixture.feature_path.resolve() not in reads
    assert fixture.outcome_path.resolve() not in reads
    assert fixture.train_development_path.resolve() not in reads


def test_prediction_row_mismatch_stops_before_outcome_reader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_fixture(tmp_path)
    bad_predictions = [
        {
            "row_id": TEST_ID_A,
            "partition": "internal_test",
            "predictions": {METHOD_ID: 0.6},
        },
        {
            "row_id": OTHER_TEST_ID,
            "partition": "internal_test",
            "predictions": {METHOD_ID: -0.4},
        },
    ]
    new_prediction_sha = _write_bytes(
        fixture.predictions_path, _jsonl_bytes(bad_predictions)
    )
    fixture.prediction_commit["evaluation_predictions_ref"][
        "sha256"
    ] = new_prediction_sha
    new_commit_sha = _write_canonical_commit(
        fixture.prediction_commit_path, fixture.prediction_commit
    )
    reads = _spy_exact_reads(monkeypatch)

    with pytest.raises(StagedDisclosureError, match="row set differs"):
        disclose_internal_test_outcomes(
            fixture.manifest_path,
            fixture.manifest_sha,
            test_prediction_commit_path=fixture.prediction_commit_path,
            test_prediction_commit_sha256=new_commit_sha,
        )
    assert fixture.predictions_path.resolve() in reads
    assert fixture.outcome_path.resolve() not in reads
    assert fixture.feature_path.resolve() not in reads
    assert fixture.train_development_path.resolve() not in reads


@pytest.mark.parametrize(
    "bad_feature",
    [
        {
            "row_id": TEST_ID_A,
            "partition": "internal_test",
            "features": {
                **_feature_payload(2),
                "context_id": {"delta_effect": 0.7},
            },
        },
        {
            "row_id": TEST_ID_A,
            "partition": "internal_test",
            "features": {
                **_feature_payload(2),
                "source_sequence": "ordinary sealed member",
            },
        },
        {
            "row_id": TEST_ID_A,
            "partition": "development",
            "features": _feature_payload(2),
        },
        {
            "row_id": TEST_ID_A,
            "partition": "internal_test",
            "features": _feature_payload(2),
            "unknown": 1,
        },
    ],
    ids=["nested-outcome", "protected-content", "wrong-partition", "unknown-field"],
)
def test_feature_authority_rejects_leakage_protection_partition_and_unknown_fields(
    tmp_path: Path, bad_feature: dict
) -> None:
    fixture = _build_fixture(
        tmp_path,
        feature_rows=[
            bad_feature,
            {
                "row_id": TEST_ID_B,
                "partition": "internal_test",
                "features": _feature_payload(3),
            },
        ],
    )
    with pytest.raises((StagedDisclosureError, staged_module.ProtocolHoldError)):
        disclose_internal_test_features(
            fixture.manifest_path,
            fixture.manifest_sha,
            pre_test_freeze_path=fixture.freeze_path,
            pre_test_freeze_sha256=fixture.freeze_sha,
        )


def test_duplicate_json_keys_and_duplicate_row_ids_are_rejected(tmp_path: Path) -> None:
    duplicate_key_raw = (
        "{"
        f'"row_id":"{TRAIN_ID}",'
        '"partition":"train","partition":"development",'
        '"features":{"context_id":"c","edit_count":1,'
        '"source_sequence":"ACGU","variant_sequence":"AGGU"},'
        '"outcomes":{"delta_effect":0.1}}\n'
    ).encode("utf-8")
    fixture = _build_fixture(tmp_path / "duplicate_key", train_development_raw=duplicate_key_raw)
    with pytest.raises(StagedDisclosureError, match="duplicate JSON key"):
        validate_development_stage(fixture.manifest_path, fixture.manifest_sha)

    duplicate_rows = [
        {
            "row_id": TEST_ID_A,
            "partition": "internal_test",
            "features": _feature_payload(2),
        },
        {
            "row_id": TEST_ID_A,
            "partition": "internal_test",
            "features": _feature_payload(3),
        },
    ]
    fixture_two = _build_fixture(tmp_path / "duplicate_rows", feature_rows=duplicate_rows)
    with pytest.raises(StagedDisclosureError, match="not unique"):
        disclose_internal_test_features(
            fixture_two.manifest_path,
            fixture_two.manifest_sha,
            pre_test_freeze_path=fixture_two.freeze_path,
            pre_test_freeze_sha256=fixture_two.freeze_sha,
        )


def test_outcome_row_set_mismatch_is_rejected(tmp_path: Path) -> None:
    fixture = _build_fixture(
        tmp_path,
        outcome_rows=[
            {
                "row_id": TEST_ID_A,
                "partition": "internal_test",
                "outcomes": {"delta_effect": 0.7},
            },
            {
                "row_id": OTHER_TEST_ID,
                "partition": "internal_test",
                "outcomes": {"delta_effect": -0.5},
            },
        ],
    )
    with pytest.raises(StagedDisclosureError, match="row set differs"):
        disclose_internal_test_outcomes(
            fixture.manifest_path,
            fixture.manifest_sha,
            test_prediction_commit_path=fixture.prediction_commit_path,
            test_prediction_commit_sha256=fixture.prediction_commit_sha,
        )


def test_manifest_duplicate_key_unknown_field_and_sign_gate_mutation_fail_closed(
    tmp_path: Path,
) -> None:
    fixture = _build_fixture(tmp_path / "duplicate_manifest")
    raw = fixture.manifest_path.read_text(encoding="utf-8")
    duplicate = raw.replace(
        '"status":"FROZEN"', '"status":"FROZEN","status":"FROZEN"', 1
    ).encode("utf-8")
    duplicate_sha = _write_bytes(fixture.manifest_path, duplicate)
    with pytest.raises(StagedDisclosureError, match="duplicate JSON key"):
        validate_development_stage(fixture.manifest_path, duplicate_sha)

    unknown = _build_fixture(
        tmp_path / "unknown_manifest", manifest_extra={"unregistered": True}
    )
    with pytest.raises(StagedDisclosureError, match="strict schema"):
        validate_development_stage(unknown.manifest_path, unknown.manifest_sha)

    changed_gate = _build_fixture(
        tmp_path / "changed_gate", manifest_extra={"sign_accuracy_gate": 0.59}
    )
    with pytest.raises(StagedDisclosureError, match="exactly 0.60"):
        validate_development_stage(changed_gate.manifest_path, changed_gate.manifest_sha)


def test_stage_projection_is_mandatory_and_cannot_carry_rows_or_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing = _build_fixture(tmp_path / "missing")
    del missing.manifest["nondata_authorities"]["inclusion_manifest"][
        "development_projection"
    ]
    missing_sha = _write_json(missing.manifest_path, missing.manifest)
    reads = _spy_exact_reads(monkeypatch)
    with pytest.raises(
        StagedDisclosureError,
        match="HOLD_REQUIRED_STAGE_SPECIFIC_NONDATA_PROJECTIONS",
    ):
        validate_development_stage(missing.manifest_path, missing_sha)
    assert missing.train_development_path.resolve() not in reads

    injected = _build_fixture(tmp_path / "injected")
    projection_ref = injected.manifest["nondata_authorities"][
        "context_source_crosswalk"
    ]["development_projection"]
    projection_path = injected.root / projection_ref["path"]
    projection = json.loads(projection_path.read_text(encoding="utf-8"))
    projection["rows"] = [TEST_ID_A]
    projection["values"] = [0.7]
    projection_ref["sha256"] = _write_json(projection_path, projection)
    injected_sha = _write_json(injected.manifest_path, injected.manifest)
    reads.clear()
    with pytest.raises(StagedDisclosureError, match="projection payload"):
        validate_development_stage(injected.manifest_path, injected_sha)
    assert injected.train_development_path.resolve() not in reads


def test_same_path_symlink_and_hardlink_aliases_are_rejected(tmp_path: Path) -> None:
    same_path = _build_fixture(tmp_path / "same_path")
    same_path.manifest["authorities"]["internal_test_outcomes"] = dict(
        same_path.manifest["authorities"]["internal_test_features"]
    )
    same_path.manifest["authorities"]["internal_test_outcomes"]["role"] = (
        "ORDINARY_INTERNAL_TEST_OUTCOMES_ONLY"
    )
    same_manifest_sha = _write_json(same_path.manifest_path, same_path.manifest)
    with pytest.raises(StagedDisclosureError, match="distinct paths|distinct byte"):
        validate_development_stage(same_path.manifest_path, same_manifest_sha)

    symlinked = _build_fixture(tmp_path / "symlink")
    feature_target = symlinked.root / "feature_target.jsonl"
    symlinked.feature_path.rename(feature_target)
    symlinked.feature_path.symlink_to(feature_target)
    with pytest.raises(StagedDisclosureError, match="symlink"):
        validate_development_stage(symlinked.manifest_path, symlinked.manifest_sha)

    hardlinked = _build_fixture(tmp_path / "hardlink")
    extra_link = hardlinked.root / "feature_hardlink.jsonl"
    os.link(hardlinked.feature_path, extra_link)
    with pytest.raises(StagedDisclosureError, match="hard link"):
        validate_development_stage(hardlinked.manifest_path, hardlinked.manifest_sha)


def test_existing_output_stops_before_any_authority_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_fixture(tmp_path)
    output = tmp_path / "already_exists.json"
    output.write_text("do not overwrite", encoding="utf-8")
    reads = _spy_exact_reads(monkeypatch)
    with pytest.raises(StagedDisclosureError, match="must not overwrite"):
        disclose_internal_test_outcomes(
            fixture.manifest_path,
            fixture.manifest_sha,
            test_prediction_commit_path=fixture.prediction_commit_path,
            test_prediction_commit_sha256=fixture.prediction_commit_sha,
            prospective_output_path=output,
        )
    assert reads == []
    assert output.read_text(encoding="utf-8") == "do not overwrite"


def test_auxiliary_refs_cannot_smuggle_forbidden_data_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    feature_fixture = _build_fixture(tmp_path / "feature_stage")
    feature_fixture.freeze["predictor_provenance_refs"][METHOD_ID] = {
        "relative_path": feature_fixture.outcome_path.name,
        "sha256": (
        feature_fixture.manifest["authorities"]["internal_test_outcomes"]["sha256"]
        ),
    }
    bad_freeze_sha = _write_canonical_commit(
        feature_fixture.freeze_path, feature_fixture.freeze
    )
    reads = _spy_exact_reads(monkeypatch)
    with pytest.raises(StagedDisclosureError, match="aliases|reuses|cannot bind"):
        disclose_internal_test_features(
            feature_fixture.manifest_path,
            feature_fixture.manifest_sha,
            pre_test_freeze_path=feature_fixture.freeze_path,
            pre_test_freeze_sha256=bad_freeze_sha,
        )
    assert feature_fixture.outcome_path.resolve() not in reads
    assert feature_fixture.feature_path.resolve() not in reads

    outcome_fixture = _build_fixture(tmp_path / "outcome_stage")
    outcome_fixture.prediction_commit["evaluation_predictions_ref"] = {
        "relative_path": outcome_fixture.feature_path.name,
        "sha256": outcome_fixture.manifest["authorities"]
        ["internal_test_features"]["sha256"],
    }
    bad_prediction_commit_sha = _write_canonical_commit(
        outcome_fixture.prediction_commit_path,
        outcome_fixture.prediction_commit,
    )
    reads.clear()
    with pytest.raises(StagedDisclosureError, match="aliases|reuses|cannot bind"):
        disclose_internal_test_outcomes(
            outcome_fixture.manifest_path,
            outcome_fixture.manifest_sha,
            test_prediction_commit_path=outcome_fixture.prediction_commit_path,
            test_prediction_commit_sha256=bad_prediction_commit_sha,
        )
    assert outcome_fixture.feature_path.resolve() not in reads
    assert outcome_fixture.outcome_path.resolve() not in reads
    assert outcome_fixture.train_development_path.resolve() not in reads


def test_outcome_commit_must_remain_ordinary_locked_and_precede_outcome_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_fixture(tmp_path)
    fixture.prediction_commit["sealed"] = True
    bad_commit_sha = _write_canonical_commit(
        fixture.prediction_commit_path, fixture.prediction_commit
    )
    reads = _spy_exact_reads(monkeypatch)
    with pytest.raises(staged_module.ProtocolHoldError, match="protected|sealed"):
        disclose_internal_test_outcomes(
            fixture.manifest_path,
            fixture.manifest_sha,
            test_prediction_commit_path=fixture.prediction_commit_path,
            test_prediction_commit_sha256=bad_commit_sha,
        )
    assert fixture.outcome_path.resolve() not in reads
    assert fixture.feature_path.resolve() not in reads
    assert fixture.train_development_path.resolve() not in reads
