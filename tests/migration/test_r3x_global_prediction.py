from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import pytest

from scripts.r3x.artifact_graph import (
    FROZEN_SEED_SEQUENCE,
    PREDICTION_ROW_SCHEMA,
    build_prediction_manifest,
    build_seed_commit,
    make_artifact_ref,
    write_ensemble_from_seed_commits_exclusive,
    write_prediction_jsonl_exclusive,
    write_prediction_manifest_exclusive,
    write_seed_commit_exclusive,
)
from scripts.r3x.artifact_io import atomic_publish_bytes
from scripts.r3x.estimands import build_primary_estimand_slices
from scripts.r3x.global_prediction import (
    GLOBAL_PREDICTION_COMMIT_SCHEMA,
    GlobalPredictionError,
    replay_global_prediction_commit,
    validate_global_prediction_commit,
    write_global_method_predictions_exclusive,
)


REGISTRY_SHA = "a" * 64
RUN_START_SHA = "b" * 64
INFORMATION_FINGERPRINT = "c" * 64
METHOD_ID = "A_EXACT_POTENTIAL_DIFFERENCE"


def _row_id(estimand_id: str, partition: str, index: int) -> str:
    return hashlib.sha256(
        f"{estimand_id}:{partition}:{index}".encode("utf-8")
    ).hexdigest()


def _registry_and_rows() -> tuple[dict, list[dict]]:
    entries = []
    rows = []
    for suffix in ("a", "b"):
        estimand_id = f"estimand_{suffix}"
        study = f"study_{suffix}"
        endpoint = f"endpoint_{suffix}"
        entries.append({
            "study": study,
            "endpoint": endpoint,
            "estimand_id": estimand_id,
            "raw_source_field": "source_value",
            "raw_candidate_field": "candidate_value",
            "raw_unit": "fixture_unit",
            "direction": "HIGHER_IS_MORE_STABLE",
            "transform": "IDENTITY_DIFFERENCE",
            "valid_domain": "FINITE",
            "role": "PRIMARY",
            "replicate_policy": "SINGLE_OBSERVATION_ONLY",
            "source_evidence_sha256": "d" * 64,
        })
        for partition in ("train", "development", "internal_test"):
            for index in range(2):
                rows.append({
                    "r3_input_row_sha256": _row_id(estimand_id, partition, index),
                    "study": study,
                    "endpoint": endpoint,
                    "endpoint_id": endpoint,
                    "estimand_id": estimand_id,
                    "partition": partition,
                    # These scientific fields are deliberately present in the
                    # input fixture and must never cross the prediction boundary.
                    "delta": 1000.0 + index,
                    "source_sequence": "AAAA",
                    "candidate_sequence": "AAAU",
                })
    return {
        "schema_version": "r3_endpoint_estimand_registry_v1",
        "status": "FROZEN",
        "entries": entries,
    }, rows


def _frozen_inputs():
    registry, rows = _registry_and_rows()
    slices = build_primary_estimand_slices(
        rows, registry, estimand_registry_sha256=REGISTRY_SHA
    )
    global_test_rows = sorted(
        [row for row in rows if row["partition"] == "internal_test"],
        key=lambda row: row["r3_input_row_sha256"],
    )
    return slices, global_test_rows


def _commit_ensemble(
    run_root: Path,
    item,
    *,
    tag: str,
    run_start_sha256: str = RUN_START_SHA,
    method_id: str = METHOD_ID,
    training_variant_id: str = "PRIMARY",
    information_set_fingerprint: str = INFORMATION_FINGERPRINT,
    estimand_binding_sha256: str | None = None,
    row_ids: tuple[str, ...] | None = None,
) -> dict:
    binding = estimand_binding_sha256 or item.partition_binding_sha256
    ids = row_ids or item.row_ids_for("internal_test")
    seed_refs = {}
    for seed_index, seed in enumerate(FROZEN_SEED_SEQUENCE):
        unit = run_root / "units" / tag / str(seed)
        unit.mkdir(parents=True)
        selection_path = unit / "checkpoint_selection.json"
        selection_sha = atomic_publish_bytes(
            selection_path, f"selection:{tag}:{seed}".encode("utf-8")
        )
        selection_ref = make_artifact_ref(run_root, selection_path, selection_sha)
        checkpoint_path = unit / "checkpoint.pt"
        checkpoint_sha = atomic_publish_bytes(
            checkpoint_path, f"checkpoint:{tag}:{seed}".encode("utf-8")
        )
        checkpoint_ref = make_artifact_ref(run_root, checkpoint_path, checkpoint_sha)
        prediction_rows = [
            {
                "schema_version": PREDICTION_ROW_SCHEMA,
                "row_id": row_id,
                "predicted_delta_mean": float(seed_index + row_index / 10.0),
                "predicted_delta_logvar": math.log(1.0 + row_index / 10.0),
            }
            for row_index, row_id in enumerate(ids)
        ]
        prediction_path = unit / "predictions.jsonl"
        prediction_sha = write_prediction_jsonl_exclusive(
            prediction_path, prediction_rows
        )
        prediction_ref = make_artifact_ref(run_root, prediction_path, prediction_sha)
        manifest = build_prediction_manifest(
            run_start_sha256=run_start_sha256,
            estimand_binding_sha256=binding,
            method_id=method_id,
            training_variant_id=training_variant_id,
            seed=seed,
            checkpoint_sha256=checkpoint_sha,
            prediction_jsonl_ref=prediction_ref,
            prediction_rows=prediction_rows,
            information_set_fingerprint=information_set_fingerprint,
        )
        manifest_path = unit / "prediction_manifest.json"
        manifest_sha = write_prediction_manifest_exclusive(manifest_path, manifest)
        manifest_ref = make_artifact_ref(run_root, manifest_path, manifest_sha)
        commit = build_seed_commit(
            checkpoint_selection_ref=selection_ref,
            checkpoint_ref=checkpoint_ref,
            prediction_manifest_ref=manifest_ref,
            prediction_jsonl_ref=prediction_ref,
            prediction_manifest=manifest,
        )
        commit_path = unit / "SEED_COMMIT.json"
        commit_sha = write_seed_commit_exclusive(commit_path, commit)
        seed_refs[str(seed)] = make_artifact_ref(run_root, commit_path, commit_sha)

    ensemble_dir = run_root / "ensembles" / tag
    ensemble_dir.mkdir(parents=True)
    return write_ensemble_from_seed_commits_exclusive(
        run_root=run_root,
        seed_commit_refs=seed_refs,
        prediction_path=ensemble_dir / "predictions.jsonl",
        manifest_path=ensemble_dir / "ENSEMBLE_COMMIT.json",
    )


def _sources(run_root: Path, slices, **second_overrides) -> dict[str, dict]:
    first = _commit_ensemble(run_root, slices[0], tag="estimand_a")
    second = _commit_ensemble(
        run_root, slices[1], tag="estimand_b", **second_overrides
    )
    # Deliberately reverse insertion order; the global commit must canonicalize it.
    return {
        slices[1].estimand_id: second["manifest_ref"],
        slices[0].estimand_id: first["manifest_ref"],
    }


def _write(run_root: Path, slices, global_rows, refs):
    output = run_root / "global"
    output.mkdir(parents=True, exist_ok=True)
    return write_global_method_predictions_exclusive(
        run_root=run_root,
        ensemble_manifest_refs_by_estimand=refs,
        estimand_slices=slices,
        global_test_rows=global_rows,
        prediction_path=output / "predictions.jsonl",
        commit_path=output / "GLOBAL_PREDICTION_COMMIT.json",
    )


def test_global_prediction_commit_is_strict_canonical_and_exactly_replayable(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    slices, global_rows = _frozen_inputs()
    refs = _sources(run_root, slices)

    written = _write(run_root, slices, global_rows, refs)
    replay = replay_global_prediction_commit(
        run_root,
        written["commit_ref"],
        estimand_slices=slices,
        global_test_rows=global_rows,
    )
    commit = replay["commit"]
    assert commit["schema_version"] == GLOBAL_PREDICTION_COMMIT_SCHEMA
    assert commit["primary_estimand_ids"] == ["estimand_a", "estimand_b"]
    assert [source["estimand_id"] for source in commit["source_ensembles"]] == [
        "estimand_a", "estimand_b",
    ]
    assert commit["ordinary_only"] is True
    assert commit["sealed_access"] is False
    assert commit["downstream_unlock"] is False
    assert [row["row_id"] for row in replay["rows"]] == [
        row["r3_input_row_sha256"] for row in global_rows
    ]
    assert all(len(set(row)) == 4 for row in replay["rows"])
    prediction_raw = (run_root / written["prediction_ref"]["relative_path"]).read_bytes()
    commit_raw = (run_root / written["commit_ref"]["relative_path"]).read_bytes()
    for forbidden in (
        b'"delta"', b"source_sequence", b"candidate_sequence", b"source_value",
        b"candidate_value",
    ):
        assert forbidden not in prediction_raw
        assert forbidden not in commit_raw


def test_global_publish_is_exclusive_and_preserves_first_commit(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    slices, global_rows = _frozen_inputs()
    refs = _sources(run_root, slices)
    first = _write(run_root, slices, global_rows, refs)
    first_prediction = (run_root / first["prediction_ref"]["relative_path"]).read_bytes()
    first_commit = (run_root / first["commit_ref"]["relative_path"]).read_bytes()
    with pytest.raises(GlobalPredictionError, match="already exists"):
        _write(run_root, slices, global_rows, refs)
    assert (run_root / first["prediction_ref"]["relative_path"]).read_bytes() == first_prediction
    assert (run_root / first["commit_ref"]["relative_path"]).read_bytes() == first_commit


def test_global_merge_requires_each_primary_estimand_exactly_once(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    slices, global_rows = _frozen_inputs()
    refs = _sources(run_root, slices)
    with pytest.raises(GlobalPredictionError, match="exactly one ensemble"):
        write_global_method_predictions_exclusive(
            run_root=run_root,
            ensemble_manifest_refs_by_estimand={"estimand_a": refs["estimand_a"]},
            estimand_slices=slices,
            global_test_rows=global_rows,
            prediction_path=run_root / "missing_predictions.jsonl",
            commit_path=run_root / "missing_commit.json",
        )
    duplicate_ref = {"estimand_a": refs["estimand_a"], "estimand_b": refs["estimand_a"]}
    with pytest.raises(GlobalPredictionError, match="binding differs"):
        write_global_method_predictions_exclusive(
            run_root=run_root,
            ensemble_manifest_refs_by_estimand=duplicate_ref,
            estimand_slices=slices,
            global_test_rows=global_rows,
            prediction_path=run_root / "duplicate_predictions.jsonl",
            commit_path=run_root / "duplicate_commit.json",
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"run_start_sha256": "e" * 64},
        {"method_id": "B_SPARSE_EDIT_SET_ADDITIVE"},
        {"training_variant_id": "CONTEXT_LABEL_PERMUTATION"},
        {"information_set_fingerprint": "f" * 64},
    ],
)
def test_global_merge_rejects_cross_estimand_context_drift(
    tmp_path: Path, overrides: dict
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    slices, global_rows = _frozen_inputs()
    refs = _sources(run_root, slices, **overrides)
    with pytest.raises(GlobalPredictionError, match="disagree on method, variant, run start"):
        _write(run_root, slices, global_rows, refs)


def test_global_merge_rejects_estimand_binding_row_set_and_global_order_drift(
    tmp_path: Path,
) -> None:
    slices, global_rows = _frozen_inputs()

    binding_root = tmp_path / "binding"
    binding_root.mkdir()
    binding_refs = _sources(
        binding_root,
        slices,
        estimand_binding_sha256=slices[0].partition_binding_sha256,
    )
    with pytest.raises(GlobalPredictionError, match="binding differs"):
        _write(binding_root, slices, global_rows, binding_refs)

    rows_root = tmp_path / "rows"
    rows_root.mkdir()
    wrong_id = hashlib.sha256(b"not-a-frozen-row").hexdigest()
    rows_refs = _sources(rows_root, slices, row_ids=(wrong_id,))
    with pytest.raises(GlobalPredictionError, match="do not exactly match"):
        _write(rows_root, slices, global_rows, rows_refs)

    order_root = tmp_path / "order"
    order_root.mkdir()
    order_refs = _sources(order_root, slices)
    with pytest.raises(GlobalPredictionError, match="canonical row_id order"):
        _write(order_root, slices, list(reversed(global_rows)), order_refs)
    with pytest.raises(GlobalPredictionError, match="exactly cover"):
        _write(order_root, slices, global_rows[:-1], order_refs)


def test_exact_replay_detects_global_prediction_byte_tampering(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    slices, global_rows = _frozen_inputs()
    refs = _sources(run_root, slices)
    written = _write(run_root, slices, global_rows, refs)

    global_path = run_root / written["prediction_ref"]["relative_path"]
    original = global_path.read_bytes()
    global_path.write_bytes(original.replace(b'"predicted_delta_mean":2.0', b'"predicted_delta_mean":9.0'))
    with pytest.raises(GlobalPredictionError, match="SHA-256 mismatch"):
        replay_global_prediction_commit(
            run_root,
            written["commit_ref"],
            estimand_slices=slices,
            global_test_rows=global_rows,
        )


def test_exact_replay_detects_source_ensemble_byte_tampering(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    slices, global_rows = _frozen_inputs()
    refs = _sources(run_root, slices)
    written = _write(run_root, slices, global_rows, refs)

    source_prediction_ref = written["commit"]["source_ensembles"][0][
        "ensemble_prediction_jsonl_ref"
    ]
    source_path = run_root / source_prediction_ref["relative_path"]
    original = source_path.read_bytes()
    source_path.write_bytes(
        original.replace(
            b'"predicted_delta_mean":2.0', b'"predicted_delta_mean":8.0'
        )
    )
    with pytest.raises(GlobalPredictionError, match="SHA-256 mismatch"):
        replay_global_prediction_commit(
            run_root,
            written["commit_ref"],
            estimand_slices=slices,
            global_test_rows=global_rows,
        )


def test_commit_schema_rejects_extra_scientific_fields_and_unlocks(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    slices, global_rows = _frozen_inputs()
    written = _write(run_root, slices, global_rows, _sources(run_root, slices))

    extra = copy.deepcopy(written["commit"])
    extra["delta"] = [1.0]
    with pytest.raises(GlobalPredictionError, match="strict schema"):
        validate_global_prediction_commit(extra)
    source_extra = copy.deepcopy(written["commit"])
    source_extra["source_ensembles"][0]["source_sequence"] = "AAAA"
    with pytest.raises(GlobalPredictionError, match="strict schema"):
        validate_global_prediction_commit(source_extra)
    unlocked = copy.deepcopy(written["commit"])
    unlocked["downstream_unlock"] = True
    with pytest.raises(GlobalPredictionError, match="cannot unlock"):
        validate_global_prediction_commit(unlocked)
    sealed = copy.deepcopy(written["commit"])
    sealed["sealed_access"] = True
    with pytest.raises(GlobalPredictionError, match="sealed access"):
        validate_global_prediction_commit(sealed)
