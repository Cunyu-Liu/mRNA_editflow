from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path

import pytest

from scripts.r3x.artifact_graph import (
    ArtifactGraphError,
    FROZEN_SEED_SEQUENCE,
    PREDICTION_ROW_SCHEMA,
    build_prediction_manifest,
    build_seed_commit,
    make_artifact_ref,
    parse_prediction_jsonl,
    prediction_jsonl_bytes,
    read_artifact_ref,
    replay_ensemble_manifest,
    replay_fixed_ensemble,
    replay_seed_commit,
    resolve_artifact_ref,
    validate_artifact_ref,
    validate_ensemble_manifest,
    validate_prediction_manifest,
    validate_seed_commit,
    write_ensemble_from_seed_commits_exclusive,
    write_prediction_jsonl_exclusive,
    write_prediction_manifest_exclusive,
    write_seed_commit_exclusive,
)
from scripts.r3x.artifact_io import atomic_publish_bytes


RUN_START_SHA = "1" * 64
ESTIMAND_SHA = "2" * 64
INFO_SHA = "3" * 64
METHOD_ID = "A_EXACT_POTENTIAL_DIFFERENCE"
ROW_IDS = [hashlib.sha256(f"row-{index}".encode()).hexdigest() for index in range(3)]


def prediction_rows(means, logvars=None):
    if logvars is None:
        logvars = [0.0] * len(means)
    return [
        {
            "schema_version": PREDICTION_ROW_SCHEMA,
            "row_id": row_id,
            "predicted_delta_mean": mean,
            "predicted_delta_logvar": logvar,
        }
        for row_id, mean, logvar in zip(ROW_IDS, means, logvars)
    ]


def commit_seed(
    root: Path,
    seed: int,
    *,
    means=None,
    logvars=None,
    estimand_sha: str = ESTIMAND_SHA,
    method_id: str = METHOD_ID,
    unit_tag: str | None = None,
) -> dict:
    if means is None:
        offset = FROZEN_SEED_SEQUENCE.index(seed)
        means = [float(offset + index) for index in range(len(ROW_IDS))]
    if logvars is None:
        logvars = [math.log(2.0)] * len(ROW_IDS)
    unit = root / "units" / (unit_tag or method_id) / "PRIMARY" / str(seed)
    unit.mkdir(parents=True)

    selection_path = unit / "checkpoint_selection.json"
    selection_sha = atomic_publish_bytes(
        selection_path, f"development-selection-{seed}-{estimand_sha}-{method_id}".encode()
    )
    selection_ref = make_artifact_ref(root, selection_path, selection_sha)

    checkpoint_path = unit / "checkpoint.pt"
    checkpoint_sha = atomic_publish_bytes(
        checkpoint_path, f"checkpoint-{seed}-{estimand_sha}-{method_id}".encode()
    )
    checkpoint_ref = make_artifact_ref(root, checkpoint_path, checkpoint_sha)

    rows = prediction_rows(means, logvars)
    prediction_path = unit / "predictions.jsonl"
    prediction_sha = write_prediction_jsonl_exclusive(prediction_path, rows)
    prediction_ref = make_artifact_ref(root, prediction_path, prediction_sha)

    manifest = build_prediction_manifest(
        run_start_sha256=RUN_START_SHA,
        estimand_binding_sha256=estimand_sha,
        method_id=method_id,
        training_variant_id="PRIMARY",
        seed=seed,
        checkpoint_sha256=checkpoint_sha,
        prediction_jsonl_ref=prediction_ref,
        prediction_rows=rows,
        information_set_fingerprint=INFO_SHA,
    )
    manifest_path = unit / "prediction_manifest.json"
    manifest_sha = write_prediction_manifest_exclusive(manifest_path, manifest)
    manifest_ref = make_artifact_ref(root, manifest_path, manifest_sha)

    commit = build_seed_commit(
        checkpoint_selection_ref=selection_ref,
        checkpoint_ref=checkpoint_ref,
        prediction_manifest_ref=manifest_ref,
        prediction_jsonl_ref=prediction_ref,
        prediction_manifest=manifest,
    )
    commit_path = unit / "SEED_COMMIT.json"
    commit_sha = write_seed_commit_exclusive(commit_path, commit)
    return make_artifact_ref(root, commit_path, commit_sha)


def five_seed_commits(root: Path) -> dict[str, dict]:
    return {
        str(seed): commit_seed(root, seed) for seed in FROZEN_SEED_SEQUENCE
    }


def test_prediction_jsonl_has_exactly_four_fields_and_canonical_order(tmp_path: Path) -> None:
    rows = list(reversed(prediction_rows([1.0, 2.0, 3.0])))
    raw = prediction_jsonl_bytes(rows)
    parsed = parse_prediction_jsonl(raw)
    assert [row["row_id"] for row in parsed] == sorted(ROW_IDS)
    assert all(set(row) == {
        "schema_version", "row_id", "predicted_delta_mean",
        "predicted_delta_logvar",
    } for row in parsed)
    for forbidden in (
        b'"delta"', b"source_sequence", b"candidate_sequence", b"study",
        b"context", b"metadata", b"edit_list",
    ):
        assert forbidden not in raw

    output = tmp_path / "predictions.jsonl"
    digest = write_prediction_jsonl_exclusive(output, rows)
    assert hashlib.sha256(output.read_bytes()).hexdigest() == digest
    assert parse_prediction_jsonl(output.read_bytes()) == parsed


@pytest.mark.parametrize(
    "extra_field",
    ["delta", "source_sequence", "candidate_sequence", "study", "context", "metadata"],
)
def test_prediction_rows_reject_labels_sequences_metadata_and_every_extra_field(
    tmp_path: Path, extra_field: str
) -> None:
    row = prediction_rows([0.1])[0]
    row[extra_field] = "forbidden"
    output = tmp_path / f"{extra_field}.jsonl"
    with pytest.raises(ArtifactGraphError, match="strict schema"):
        write_prediction_jsonl_exclusive(output, [row])
    assert not output.exists()


def test_prediction_jsonl_rejects_nonfinite_duplicate_noncanonical_and_duplicate_keys() -> None:
    bad = prediction_rows([0.1])[0]
    bad["predicted_delta_mean"] = float("nan")
    with pytest.raises(ArtifactGraphError, match="finite"):
        prediction_jsonl_bytes([bad])
    numeric_string = prediction_rows([0.1])[0]
    numeric_string["predicted_delta_mean"] = "0.1"
    with pytest.raises(ArtifactGraphError, match="JSON number"):
        prediction_jsonl_bytes([numeric_string])
    duplicate = prediction_rows([0.1])[0]
    with pytest.raises(ArtifactGraphError, match="unique"):
        prediction_jsonl_bytes([duplicate, copy.deepcopy(duplicate)])

    noncanonical_rows = prediction_rows([0.0, 1.0, 2.0])
    if [row["row_id"] for row in noncanonical_rows] == sorted(ROW_IDS):
        noncanonical_rows.reverse()
    raw = b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
        for row in noncanonical_rows
    )
    with pytest.raises(ArtifactGraphError, match="canonical row_id order"):
        parse_prediction_jsonl(raw)

    row_id = ROW_IDS[0]
    duplicate_key = (
        '{"schema_version":"r3x_prediction_row_v1","row_id":"%s",'
        '"row_id":"%s","predicted_delta_mean":0.0,'
        '"predicted_delta_logvar":0.0}\n' % (row_id, row_id)
    ).encode()
    with pytest.raises(ArtifactGraphError, match="duplicate JSON key"):
        parse_prediction_jsonl(duplicate_key)


@pytest.mark.parametrize(
    "relative",
    ["../escape.json", "/absolute.json", "a//b.json", "a/./b.json", "a\\b.json", ""],
)
def test_artifact_refs_reject_unsafe_paths(relative: str) -> None:
    with pytest.raises(ArtifactGraphError, match="relative_path|relative path"):
        validate_artifact_ref({"relative_path": relative, "sha256": "a" * 64})


def test_artifact_refs_reject_protected_names_and_extra_fields() -> None:
    with pytest.raises(ArtifactGraphError, match="PROTECTED"):
        validate_artifact_ref(
            {"relative_path": "sealed/predictions.jsonl", "sha256": "a" * 64}
        )
    with pytest.raises(ArtifactGraphError, match="strict schema"):
        validate_artifact_ref(
            {
                "relative_path": "predictions.jsonl",
                "sha256": "a" * 64,
                "metadata": {},
            }
        )


def test_safe_reference_replay_rejects_hash_symlink_and_hardlink(tmp_path: Path) -> None:
    root = tmp_path / "run"
    root.mkdir()
    path = root / "artifact.bin"
    sha = atomic_publish_bytes(path, b"bytes")
    ref = make_artifact_ref(root, path, sha)
    assert resolve_artifact_ref(root, ref) == path.resolve()
    assert read_artifact_ref(root, ref) == b"bytes"
    with pytest.raises(ArtifactGraphError, match="SHA-256 mismatch"):
        read_artifact_ref(root, {**ref, "sha256": "f" * 64})

    symlink = root / "symlink.bin"
    symlink.symlink_to(path)
    with pytest.raises(ArtifactGraphError, match="symlink"):
        read_artifact_ref(
            root, {"relative_path": "symlink.bin", "sha256": sha}
        )
    hardlink = root / "hardlink.bin"
    os.link(path, hardlink)
    with pytest.raises(ArtifactGraphError, match="single-link"):
        read_artifact_ref(root, ref)


def test_prediction_manifest_and_seed_commit_are_strict_and_cross_bound(tmp_path: Path) -> None:
    root = tmp_path / "run"
    root.mkdir()
    ref = commit_seed(root, 11)
    replay = replay_seed_commit(root, ref, expected_seed=11)
    assert replay["commit"]["status"] == "COMMITTED"
    assert replay["manifest"]["checkpoint_sha256"] == replay["commit"]["checkpoint_ref"]["sha256"]
    assert len(replay["rows"]) == len(ROW_IDS)

    extra_manifest = copy.deepcopy(replay["manifest"])
    extra_manifest["metadata"] = {"delta": 1.0}
    with pytest.raises(ArtifactGraphError, match="strict schema"):
        validate_prediction_manifest(extra_manifest)
    extra_commit = copy.deepcopy(replay["commit"])
    extra_commit["target"] = 1.0
    with pytest.raises(ArtifactGraphError, match="strict schema"):
        validate_seed_commit(extra_commit)

    wrong_checkpoint = dict(replay["commit"]["checkpoint_ref"])
    wrong_checkpoint["sha256"] = "f" * 64
    with pytest.raises(ArtifactGraphError, match="checkpoint SHA-256 mismatch"):
        build_seed_commit(
            checkpoint_selection_ref=replay["commit"]["checkpoint_selection_ref"],
            checkpoint_ref=wrong_checkpoint,
            prediction_manifest_ref=replay["commit"]["prediction_manifest_ref"],
            prediction_jsonl_ref=replay["commit"]["prediction_jsonl_ref"],
            prediction_manifest=replay["manifest"],
        )


def test_fixed_ensemble_replays_exact_five_commits_and_total_variance(tmp_path: Path) -> None:
    root = tmp_path / "run"
    root.mkdir()
    refs = five_seed_commits(root)

    rows, context = replay_fixed_ensemble(root, refs)
    assert context["seed_commit_refs"] == refs
    assert len(rows) == len(ROW_IDS)
    # For each row, seed means are x+[0,1,2,3,4] and each aleatoric
    # variance is 2.  Between-seed population variance is 2, hence total 4.
    by_id = {row["row_id"]: row for row in rows}
    for index, row_id in enumerate(ROW_IDS):
        assert by_id[row_id]["predicted_delta_mean"] == pytest.approx(index + 2.0)
        assert by_id[row_id]["predicted_delta_logvar"] == pytest.approx(math.log(4.0))

    ensemble_dir = root / "ensemble"
    ensemble_dir.mkdir()
    result = write_ensemble_from_seed_commits_exclusive(
        run_root=root,
        seed_commit_refs=refs,
        prediction_path=ensemble_dir / "predictions.jsonl",
        manifest_path=ensemble_dir / "ENSEMBLE_COMMIT.json",
    )
    replayed = replay_ensemble_manifest(root, result["manifest_ref"])
    assert replayed["rows"] == rows
    assert replayed["manifest"]["aggregation"] == "FIXED_FIVE_MODEL_TOTAL_VARIANCE_ENSEMBLE"
    assert replayed["manifest"]["n_models"] == 5
    assert replayed["manifest"]["seed_prediction_jsonl_sha256"] == context["seed_prediction_jsonl_sha256"]


def test_ensemble_rejects_missing_seed_and_cross_estimand_commit(tmp_path: Path) -> None:
    root = tmp_path / "run"
    root.mkdir()
    refs = five_seed_commits(root)
    missing = dict(refs)
    missing.pop("101")
    with pytest.raises(ArtifactGraphError, match="exactly the five"):
        replay_fixed_ensemble(root, missing)

    refs["101"] = commit_seed(
        root, 101, estimand_sha="f" * 64, unit_tag="wrong_estimand"
    )
    with pytest.raises(ArtifactGraphError, match="one ensemble unit"):
        replay_fixed_ensemble(root, refs)


def test_ensemble_detects_changed_committed_prediction_bytes(tmp_path: Path) -> None:
    root = tmp_path / "run"
    root.mkdir()
    refs = five_seed_commits(root)
    replay = replay_seed_commit(root, refs["42"])
    prediction_path = resolve_artifact_ref(root, replay["commit"]["prediction_jsonl_ref"])
    original = prediction_path.read_bytes()
    prediction_path.write_bytes(original + b" ")
    with pytest.raises(ArtifactGraphError, match="SHA-256 mismatch"):
        replay_fixed_ensemble(root, refs)


def test_ensemble_manifest_rejects_extra_fields_and_nonfrozen_formula(tmp_path: Path) -> None:
    root = tmp_path / "run"
    root.mkdir()
    refs = five_seed_commits(root)
    ensemble_dir = root / "ensemble"
    ensemble_dir.mkdir()
    result = write_ensemble_from_seed_commits_exclusive(
        run_root=root,
        seed_commit_refs=refs,
        prediction_path=ensemble_dir / "predictions.jsonl",
        manifest_path=ensemble_dir / "ENSEMBLE_COMMIT.json",
    )
    extra = copy.deepcopy(result["manifest"])
    extra["metrics"] = {"sign_accuracy": 1.0}
    with pytest.raises(ArtifactGraphError, match="strict schema"):
        validate_ensemble_manifest(extra)
    changed = copy.deepcopy(result["manifest"])
    changed["variance_formula"] = "AVERAGE_LOGVAR"
    with pytest.raises(ArtifactGraphError, match="variance formula"):
        validate_ensemble_manifest(changed)
