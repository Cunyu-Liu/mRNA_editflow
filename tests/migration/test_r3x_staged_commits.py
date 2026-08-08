from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts.r3x.artifact_graph import make_artifact_ref
from scripts.r3x.artifact_io import atomic_publish_bytes
from scripts.r3x.protocol import REQUIRED_NEGATIVE_CONTROLS
from scripts.r3x.staged_commits import (
    DEVELOPMENT_COMMIT,
    ORDINARY_TEST_OUTCOME_RELEASE_RECORD,
    OUTCOME_RELEASE_CLAIM_RELATIVE_PATH,
    PRE_TEST_FREEZE_COMMIT,
    REQUIRED_NONDATA_AUTHORITY_HASHES,
    REQUIRED_NONDATA_PROJECTION_HASHES,
    STATE_RELATIVE_PATHS,
    TEST_PREDICTION_COMMIT,
    StagedCommitError,
    build_staged_bindings,
    replay_development_commit,
    replay_ordinary_test_outcome_release_record,
    replay_pre_test_freeze_commit,
    replay_test_prediction_commit,
    required_predictor_unit_ids,
    validate_test_prediction_commit,
    write_development_commit_exclusive,
    write_ordinary_test_outcome_release_record_exclusive,
    write_pre_test_freeze_commit_exclusive,
    write_test_prediction_commit_exclusive,
)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def publish_ref(root: Path, relative: str, raw: bytes) -> dict:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = atomic_publish_bytes(path, raw)
    return make_artifact_ref(root, path, digest)


@pytest.fixture
def staged(tmp_path: Path) -> dict:
    root = tmp_path / "run"
    root.mkdir()
    development_raw = b"train-development-authority"
    feature_raw = b"feature-only-internal-test-authority"
    outcome_raw = b"ordinary-internal-test-outcomes"
    development_ref = publish_ref(
        root, "authorities/train_development.bin", development_raw
    )
    feature_ref = publish_ref(root, "authorities/test_features.bin", feature_raw)
    outcome_ref = publish_ref(root, "authorities/test_labels.bin", outcome_raw)
    bindings = build_staged_bindings(
        run_start_sha256="1" * 64,
        activation_sha256="2" * 64,
        authority_hashes={
            "internal_test_features": sha(feature_raw),
            "internal_test_outcomes": sha(outcome_raw),
            "train_development_outcomes": sha(development_raw),
        },
        nondata_authority_hashes={
            name: sha(f"nondata:{name}".encode())
            for name in REQUIRED_NONDATA_AUTHORITY_HASHES
        },
        nondata_projection_hashes={
            name: sha(f"projection:{name}".encode())
            for name in REQUIRED_NONDATA_PROJECTION_HASHES
        },
        n_rows=3,
        ordered_row_ids_sha256="3" * 64,
        row_set_sha256="4" * 64,
        required_method_ids=["FULLPAIR", "RIDGE"],
        required_estimand_ids=["estimand_a"],
        required_seeds=[11, 23, 42, 67, 101],
        required_control_ids=REQUIRED_NEGATIVE_CONTROLS,
        code_sha256="5" * 64,
        config_sha256="6" * 64,
    )
    selector_ref = publish_ref(root, "development/selector.json", b"selector")
    predictor_provenance_refs = {
        method: publish_ref(
            root,
            f"development/provenance/{index:03d}.json",
            f"provenance:{method}".encode(),
        )
        for index, method in enumerate(bindings["required_method_ids"])
    }
    dev_audit_ref = publish_ref(
        root, "audits/development_no_test_access.json", b"dev-audit"
    )
    freeze_audit_ref = publish_ref(
        root, "audits/pre_test_freeze_no_test_access.json", b"freeze-audit"
    )
    units = required_predictor_unit_ids(bindings)
    checkpoint_refs = {
        unit: publish_ref(
            root,
            f"frozen/checkpoints/{index:03d}.bin",
            f"checkpoint:{unit}".encode(),
        )
        for index, unit in enumerate(units)
    }
    prediction_refs = {
        unit: publish_ref(
            root,
            f"predictions/{index:03d}.jsonl",
            (json.dumps({"unit": unit, "prediction": 0.0}) + "\n").encode(),
        )
        for index, unit in enumerate(units)
    }
    evaluation_predictions_ref = publish_ref(
        root,
        "predictions/evaluation_predictions.jsonl",
        b'{"row_id":"fixture","partition":"internal_test","predictions":{}}\n',
    )
    return {
        "root": root,
        "bindings": bindings,
        "development_ref": development_ref,
        "feature_ref": feature_ref,
        "outcome_ref": outcome_ref,
        "outcome_raw": outcome_raw,
        "selector_ref": selector_ref,
        "predictor_provenance_refs": predictor_provenance_refs,
        "dev_audit_ref": dev_audit_ref,
        "freeze_audit_ref": freeze_audit_ref,
        "checkpoint_refs": checkpoint_refs,
        "prediction_refs": prediction_refs,
        "evaluation_predictions_ref": evaluation_predictions_ref,
    }


def commit_development(fixture: dict) -> dict:
    return write_development_commit_exclusive(
        fixture["root"],
        bindings=fixture["bindings"],
        development_authority_ref=fixture["development_ref"],
        development_artifact_refs={"SELECTOR_COMMIT": fixture["selector_ref"]},
        access_audit_ref=fixture["dev_audit_ref"],
        access_assertions={
            "internal_test_feature_accessed": False,
            "internal_test_outcome_accessed": False,
        },
    )


def commit_freeze(fixture: dict, development: dict) -> dict:
    return write_pre_test_freeze_commit_exclusive(
        fixture["root"],
        bindings=fixture["bindings"],
        development_commit_ref=development["ref"],
        selector_ref=fixture["selector_ref"],
        predictor_provenance_refs=fixture["predictor_provenance_refs"],
        selected_method_id=fixture["bindings"]["required_method_ids"][0],
        frozen_checkpoint_refs=fixture["checkpoint_refs"],
        access_audit_ref=fixture["freeze_audit_ref"],
        access_assertions={
            "internal_test_feature_accessed": False,
            "internal_test_outcome_accessed": False,
        },
    )


def commit_predictions(fixture: dict, development: dict, freeze: dict) -> dict:
    return write_test_prediction_commit_exclusive(
        fixture["root"],
        bindings=fixture["bindings"],
        development_commit_ref=development["ref"],
        pre_test_freeze_commit_ref=freeze["ref"],
        feature_authority_ref=fixture["feature_ref"],
        evaluation_predictions_ref=fixture["evaluation_predictions_ref"],
        frozen_prediction_refs=fixture["prediction_refs"],
        access_assertions={
            "internal_test_feature_accessed": True,
            "internal_test_outcome_accessed": False,
        },
    )


def commit_through_predictions(fixture: dict) -> tuple[dict, dict, dict]:
    development = commit_development(fixture)
    freeze = commit_freeze(fixture, development)
    prediction = commit_predictions(fixture, development, freeze)
    return development, freeze, prediction


def release(fixture: dict, development: dict, freeze: dict, prediction: dict, **kwargs):
    return write_ordinary_test_outcome_release_record_exclusive(
        fixture["root"],
        bindings=fixture["bindings"],
        development_commit_ref=development["ref"],
        pre_test_freeze_commit_ref=freeze["ref"],
        test_prediction_commit_ref=prediction["ref"],
        outcome_authority_ref=fixture["outcome_ref"],
        access_assertions={
            "internal_test_feature_accessed": True,
            "internal_test_outcome_accessed": True,
        },
        **kwargs,
    )


def test_four_state_dag_is_canonical_exact_and_replayable(staged: dict) -> None:
    development, freeze, prediction = commit_through_predictions(staged)
    released = release(staged, development, freeze, prediction)

    assert released["outcome_bytes"] == staged["outcome_raw"]
    assert [
        development["payload"]["state"],
        freeze["payload"]["state"],
        prediction["payload"]["state"],
        released["payload"]["state"],
    ] == [
        DEVELOPMENT_COMMIT,
        PRE_TEST_FREEZE_COMMIT,
        TEST_PREDICTION_COMMIT,
        ORDINARY_TEST_OUTCOME_RELEASE_RECORD,
    ]
    assert freeze["payload"]["schema_version"] == "r3x_pre_test_freeze_commit_v2"
    assert prediction["payload"]["schema_version"] == (
        "r3x_test_prediction_commit_v2"
    )
    for result in (development, freeze, prediction, released):
        assert result["payload"]["ordinary_only"] is True
        assert result["payload"]["sealed"] is False
        assert result["payload"]["run_start_sha256"] == "1" * 64
        assert result["payload"]["sign_accuracy_gate"] == 0.60
        assert result["payload"]["sign_gate_change_authorized"] is False
        assert set(result["payload"]["authority_hashes"]) == {
            "train_development_outcomes",
            "internal_test_features",
            "internal_test_outcomes",
        }
        assert set(result["payload"]["nondata_authority_hashes"]) == set(
            REQUIRED_NONDATA_AUTHORITY_HASHES
        )
        assert set(result["payload"]["nondata_projection_hashes"]) == set(
            REQUIRED_NONDATA_PROJECTION_HASHES
        )
        raw = Path(result["path"]).read_bytes()
        assert raw.endswith(b"\n")
        assert hashlib.sha256(raw).hexdigest() == result["sha256"]

    assert replay_development_commit(staged["root"], development["ref"])[
        "payload"
    ] == development["payload"]
    assert replay_pre_test_freeze_commit(staged["root"], freeze["ref"])[
        "payload"
    ] == freeze["payload"]
    assert replay_test_prediction_commit(staged["root"], prediction["ref"])[
        "payload"
    ] == prediction["payload"]
    replayed = replay_ordinary_test_outcome_release_record(
        staged["root"], released["ref"]
    )
    assert replayed["payload"] == released["payload"]


def test_transition_order_and_no_overwrite_are_strict(staged: dict) -> None:
    fake = {
        "relative_path": STATE_RELATIVE_PATHS[DEVELOPMENT_COMMIT],
        "sha256": "a" * 64,
    }
    with pytest.raises(StagedCommitError, match="missing prior DEVELOPMENT_COMMIT"):
        write_pre_test_freeze_commit_exclusive(
            staged["root"],
            bindings=staged["bindings"],
            development_commit_ref=fake,
            selector_ref=staged["selector_ref"],
            predictor_provenance_refs=staged["predictor_provenance_refs"],
            selected_method_id=staged["bindings"]["required_method_ids"][0],
            frozen_checkpoint_refs=staged["checkpoint_refs"],
            access_audit_ref=staged["freeze_audit_ref"],
            access_assertions={
                "internal_test_feature_accessed": False,
                "internal_test_outcome_accessed": False,
            },
        )

    development = commit_development(staged)
    original = Path(development["path"]).read_bytes()
    with pytest.raises(StagedCommitError, match="already exists"):
        commit_development(staged)
    assert Path(development["path"]).read_bytes() == original


def test_freeze_rejects_missing_units_and_any_test_access_assertion(staged: dict) -> None:
    development = commit_development(staged)
    missing = dict(staged["checkpoint_refs"])
    missing.pop(next(iter(missing)))
    with pytest.raises(StagedCommitError, match="exact frozen units"):
        write_pre_test_freeze_commit_exclusive(
            staged["root"],
            bindings=staged["bindings"],
            development_commit_ref=development["ref"],
            selector_ref=staged["selector_ref"],
            predictor_provenance_refs=staged["predictor_provenance_refs"],
            selected_method_id=staged["bindings"]["required_method_ids"][0],
            frozen_checkpoint_refs=missing,
            access_audit_ref=staged["freeze_audit_ref"],
            access_assertions={
                "internal_test_feature_accessed": False,
                "internal_test_outcome_accessed": False,
            },
        )
    with pytest.raises(StagedCommitError, match="requires feature/outcome"):
        write_pre_test_freeze_commit_exclusive(
            staged["root"],
            bindings=staged["bindings"],
            development_commit_ref=development["ref"],
            selector_ref=staged["selector_ref"],
            predictor_provenance_refs=staged["predictor_provenance_refs"],
            selected_method_id=staged["bindings"]["required_method_ids"][0],
            frozen_checkpoint_refs=staged["checkpoint_refs"],
            access_audit_ref=staged["freeze_audit_ref"],
            access_assertions={
                "internal_test_feature_accessed": True,
                "internal_test_outcome_accessed": False,
            },
        )


def test_prediction_rejects_premature_outcome_and_missing_prediction(staged: dict) -> None:
    development = commit_development(staged)
    freeze = commit_freeze(staged, development)
    with pytest.raises(StagedCommitError, match="outcome authority cannot be supplied"):
        write_test_prediction_commit_exclusive(
            staged["root"],
            bindings=staged["bindings"],
            development_commit_ref=development["ref"],
            pre_test_freeze_commit_ref=freeze["ref"],
            feature_authority_ref=staged["feature_ref"],
            evaluation_predictions_ref=staged["evaluation_predictions_ref"],
            frozen_prediction_refs=staged["prediction_refs"],
            access_assertions={
                "internal_test_feature_accessed": True,
                "internal_test_outcome_accessed": False,
            },
            outcome_authority_ref=staged["outcome_ref"],
        )
    missing = dict(staged["prediction_refs"])
    missing.pop(next(iter(missing)))
    with pytest.raises(StagedCommitError, match="every frozen"):
        write_test_prediction_commit_exclusive(
            staged["root"],
            bindings=staged["bindings"],
            development_commit_ref=development["ref"],
            pre_test_freeze_commit_ref=freeze["ref"],
            feature_authority_ref=staged["feature_ref"],
            evaluation_predictions_ref=staged["evaluation_predictions_ref"],
            frozen_prediction_refs=missing,
            access_assertions={
                "internal_test_feature_accessed": True,
                "internal_test_outcome_accessed": False,
            },
        )
    assert not (staged["root"] / STATE_RELATIVE_PATHS[TEST_PREDICTION_COMMIT]).exists()


def test_swapped_and_tampered_prior_refs_fail_exact_replay(staged: dict) -> None:
    development, freeze, prediction = commit_through_predictions(staged)
    swapped = copy.deepcopy(prediction["payload"])
    swapped["prior_commit_refs"] = {
        DEVELOPMENT_COMMIT: freeze["ref"],
        PRE_TEST_FREEZE_COMMIT: development["ref"],
    }
    with pytest.raises(StagedCommitError, match="swapped|canonical"):
        validate_test_prediction_commit(swapped)

    tampered = dict(prediction["ref"])
    tampered["sha256"] = "f" * 64
    with pytest.raises(StagedCommitError, match="SHA-256 mismatch"):
        replay_test_prediction_commit(staged["root"], tampered)


def test_swapped_checkpoint_binding_is_rejected_on_replay(staged: dict) -> None:
    development, freeze, prediction = commit_through_predictions(staged)
    payload = copy.deepcopy(prediction["payload"])
    units = sorted(payload["frozen_prediction_refs"])
    payload["frozen_prediction_refs"][units[0]]["checkpoint_ref"], payload[
        "frozen_prediction_refs"
    ][units[1]]["checkpoint_ref"] = (
        payload["frozen_prediction_refs"][units[1]]["checkpoint_ref"],
        payload["frozen_prediction_refs"][units[0]]["checkpoint_ref"],
    )
    path = staged["root"] / STATE_RELATIVE_PATHS[TEST_PREDICTION_COMMIT]
    path.unlink()
    raw = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode()
    digest = atomic_publish_bytes(path, raw)
    ref = make_artifact_ref(staged["root"], path, digest)
    with pytest.raises(StagedCommitError, match="does not point to its frozen checkpoint"):
        replay_test_prediction_commit(staged["root"], ref)


def test_release_is_once_only_and_completed_record_replays_without_outcome_file(
    staged: dict,
) -> None:
    development, freeze, prediction = commit_through_predictions(staged)
    released = release(staged, development, freeze, prediction)
    with pytest.raises(StagedCommitError, match="already exists"):
        release(staged, development, freeze, prediction)

    # Replay is an access receipt: it recursively verifies every prior commit
    # and the durable claim but intentionally does not reopen outcome bytes.
    Path(staged["root"] / staged["outcome_ref"]["relative_path"]).unlink()
    replayed = replay_ordinary_test_outcome_release_record(
        staged["root"], released["ref"]
    )
    assert replayed["payload"]["consumed_once"] is True


def test_completed_release_replays_after_caller_crashes_before_receiving_result(
    staged: dict,
) -> None:
    development, freeze, prediction = commit_through_predictions(staged)
    # Publication completed, but model a caller that lost the return value in a
    # process crash.  Recovery derives the immutable ref from disk and replays
    # the full DAG without asking for a second outcome release.
    release(staged, development, freeze, prediction)
    final_path = staged["root"] / STATE_RELATIVE_PATHS[
        ORDINARY_TEST_OUTCOME_RELEASE_RECORD
    ]
    recovered_ref = make_artifact_ref(
        staged["root"], final_path, hashlib.sha256(final_path.read_bytes()).hexdigest()
    )
    recovered = replay_ordinary_test_outcome_release_record(
        staged["root"], recovered_ref
    )
    assert recovered["payload"]["state"] == ORDINARY_TEST_OUTCOME_RELEASE_RECORD


def test_crash_after_release_claim_is_permanently_fail_closed(staged: dict) -> None:
    development, freeze, prediction = commit_through_predictions(staged)

    def crash() -> None:
        raise RuntimeError("simulated process crash")

    with pytest.raises(RuntimeError, match="simulated process crash"):
        release(
            staged,
            development,
            freeze,
            prediction,
            _after_claim_hook=crash,
        )
    assert (staged["root"] / OUTCOME_RELEASE_CLAIM_RELATIVE_PATH).is_file()
    assert not (
        staged["root"]
        / STATE_RELATIVE_PATHS[ORDINARY_TEST_OUTCOME_RELEASE_RECORD]
    ).exists()
    with pytest.raises(StagedCommitError, match="partial outcome release claim"):
        release(staged, development, freeze, prediction)


def test_path_escape_symlink_and_hardlink_refs_are_rejected(staged: dict) -> None:
    development = commit_development(staged)
    escaped = {
        "relative_path": "../outside.bin",
        "sha256": "a" * 64,
    }
    checkpoints = dict(staged["checkpoint_refs"])
    checkpoints[next(iter(checkpoints))] = escaped
    with pytest.raises(StagedCommitError, match="relative path"):
        write_pre_test_freeze_commit_exclusive(
            staged["root"],
            bindings=staged["bindings"],
            development_commit_ref=development["ref"],
            selector_ref=staged["selector_ref"],
            predictor_provenance_refs=staged["predictor_provenance_refs"],
            selected_method_id=staged["bindings"]["required_method_ids"][0],
            frozen_checkpoint_refs=checkpoints,
            access_audit_ref=staged["freeze_audit_ref"],
            access_assertions={
                "internal_test_feature_accessed": False,
                "internal_test_outcome_accessed": False,
            },
        )

    link = staged["root"] / "frozen" / "link.bin"
    link.symlink_to(staged["root"] / staged["selector_ref"]["relative_path"])
    link_ref = {"relative_path": "frozen/link.bin", "sha256": staged["selector_ref"]["sha256"]}
    checkpoints = dict(staged["checkpoint_refs"])
    checkpoints[next(iter(checkpoints))] = link_ref
    with pytest.raises(StagedCommitError, match="symlink"):
        write_pre_test_freeze_commit_exclusive(
            staged["root"],
            bindings=staged["bindings"],
            development_commit_ref=development["ref"],
            selector_ref=staged["selector_ref"],
            predictor_provenance_refs=staged["predictor_provenance_refs"],
            selected_method_id=staged["bindings"]["required_method_ids"][0],
            frozen_checkpoint_refs=checkpoints,
            access_audit_ref=staged["freeze_audit_ref"],
            access_assertions={
                "internal_test_feature_accessed": False,
                "internal_test_outcome_accessed": False,
            },
        )


def test_run_root_symlink_is_rejected_before_state_write(staged: dict) -> None:
    link = staged["root"].parent / "run-link"
    link.symlink_to(staged["root"], target_is_directory=True)
    with pytest.raises(StagedCommitError, match="symlink"):
        write_development_commit_exclusive(
            link,
            bindings=staged["bindings"],
            development_authority_ref=staged["development_ref"],
            development_artifact_refs={"SELECTOR_COMMIT": staged["selector_ref"]},
            access_audit_ref=staged["dev_audit_ref"],
            access_assertions={
                "internal_test_feature_accessed": False,
                "internal_test_outcome_accessed": False,
            },
        )
