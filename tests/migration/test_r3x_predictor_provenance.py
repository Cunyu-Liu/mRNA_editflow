from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import pytest

from scripts.r3x.artifact_graph import (
    make_artifact_ref,
    parse_prediction_jsonl,
    read_artifact_ref,
    write_prediction_jsonl_exclusive,
)
from scripts.r3x.artifact_io import atomic_publish_bytes, canonical_json_bytes
from scripts.r3x.predictor_provenance import (
    DEVELOPMENT_PARTITION,
    ENSEMBLE_RULE,
    ENSEMBLE_VARIANCE_FORMULA,
    FROZEN_NEURAL_SEEDS,
    NEURAL_KIND,
    NON_DATA_AUTHORITY_HASH_KEYS,
    PREDICTOR_INPUT_ROW_SCHEMA,
    PREDICTOR_PROVENANCE_SCHEMA,
    SCALAR_KIND,
    TRAIN_PARTITION,
    VARIANCE_FLOOR,
    PredictorProvenanceError,
    PredictorVerifier,
    build_development_checkpoint,
    build_development_checkpoint_selection,
    build_predictor_provenance,
    canonical_predictor_provenance_bytes,
    normalize_predictor_input_rows,
    parse_predictor_input_jsonl,
    replay_predictor_provenance,
    validate_predictor_provenance,
    verifier_identity,
    write_development_checkpoint_exclusive,
    write_development_checkpoint_selection_exclusive,
    write_predictor_input_jsonl_exclusive,
    write_predictor_provenance_exclusive,
)


RUN_SHA = "1" * 64
ACTIVATION_SHA = "2" * 64
INFO_SHA = "3" * 64
ESTIMAND_SHA = "4" * 64
VERIFIER_SHA = "5" * 64
TRAIN_DEVELOPMENT_OUTCOMES_SHA = "6" * 64
METHOD = "PAIRED_LOCAL_GLOBAL_KMER_RIDGE"
ESTIMAND = "ESTIMAND_A"
CELL = "ESTIMAND_A:CELL_1"


def _authorities() -> dict[str, str]:
    return {
        name: format(index + 1, "x") * 64
        for index, name in enumerate(NON_DATA_AUTHORITY_HASH_KEYS)
    }


def _row_id(partition: str, index: int, suffix: str = "") -> str:
    return hashlib.sha256(f"{partition}/{index}/{suffix}".encode()).hexdigest()


def _predictor_input(index: int, *, positions: tuple[int, ...] | None = None) -> dict:
    source = "ACGUACGUACGU"
    if positions is None:
        positions = (index - 1,)
    candidate = list(source)
    edits = []
    for position in positions:
        ref = source[position]
        alt = {"A": "C", "C": "G", "G": "U", "U": "A"}[ref]
        candidate[position] = alt
        edits.append({"op": "SUB", "pos": position, "ref": ref, "alt": alt})
    return {
        "endpoint": "mrna_abundance",
        "publication_context_id": "cell_line_a|assay_a|library_a",
        "region": "5UTR",
        "normalized_source_sequence": source,
        "normalized_candidate_sequence": "".join(candidate),
        "edit_list": edits,
        "edit_count": len(edits),
    }


def _feature_score(row: dict) -> float:
    return float(row["predictor_input"]["edit_list"][0]["pos"] + 1)


def _input_rows(partition: str, *, suffix: str = "") -> list[dict]:
    output = []
    for index in range(1, 4):
        score = float(index)
        output.append({
            "schema_version": PREDICTOR_INPUT_ROW_SCHEMA,
            "row_id": _row_id(partition, index, suffix),
            "partition": partition,
            "estimand_id": ESTIMAND,
            "cell_id": CELL,
            "predictor_input": _predictor_input(index),
            "training_target": (
                2.0 * score if partition == TRAIN_PARTITION else None
            ),
        })
    return output


def _write_bytes(run_root: Path, path: Path, value: object) -> dict:
    raw = canonical_json_bytes(value) if isinstance(value, dict) else bytes(value)
    digest = atomic_publish_bytes(path, raw)
    return make_artifact_ref(run_root, path, digest)


def _row_hashes(rows: list[dict]) -> tuple[str, str]:
    ids = sorted(row["row_id"] for row in rows)
    raw = json.dumps(ids, separators=(",", ":")).encode()
    digest = hashlib.sha256(raw).hexdigest()
    return digest, digest


def _scalar_verifier(
    *,
    code_sha: str,
    config_sha: str,
    verifier_sha: str = VERIFIER_SHA,
) -> PredictorVerifier:
    def recompute(run_root: Path, cell: dict) -> dict:
        state = json.loads(
            read_artifact_ref(
                run_root, cell["predictor_artifacts"]["fitted_state_ref"]
            )
        )
        config = json.loads(
            read_artifact_ref(
                run_root, cell["predictor_artifacts"]["fit_config_ref"]
            )
        )
        rows = parse_predictor_input_jsonl(
            read_artifact_ref(run_root, cell["development_rows_ref"])
        )
        predictions = [{
            "schema_version": "r3x_prediction_row_v1",
            "row_id": row["row_id"],
            "predicted_delta_mean": (
                float(state["weight"]) * _feature_score(row)
                + float(config["bias"])
            ),
            "predicted_delta_logvar": None,
        } for row in rows]
        return {"prediction": predictions}

    return PredictorVerifier(
        verifier_id="SCALAR_EXACT_REPLAY_V1",
        code_sha256=verifier_sha,
        expected_predictor_code_sha256=code_sha,
        expected_predictor_config_sha256=config_sha,
        recompute=recompute,
    )


def _scalar_bundle(tmp_path: Path) -> dict:
    run_root = tmp_path / "ordinary_run"
    dev = run_root / "development" / "scalar"
    dev.mkdir(parents=True)
    train_rows = _input_rows(TRAIN_PARTITION)
    development_rows = _input_rows(DEVELOPMENT_PARTITION)
    train_path = dev / "train_rows.jsonl"
    dev_path = dev / "development_rows.jsonl"
    train_sha = write_predictor_input_jsonl_exclusive(train_path, train_rows)
    development_sha = write_predictor_input_jsonl_exclusive(
        dev_path, development_rows
    )
    state_ref = _write_bytes(run_root, dev / "fitted_state.json", {"weight": 2.0})
    config_ref = _write_bytes(run_root, dev / "fit_config.json", {"bias": 0.25})
    prediction_rows = [{
        "schema_version": "r3x_prediction_row_v1",
        "row_id": row["row_id"],
        "predicted_delta_mean": 2.0 * _feature_score(row) + 0.25,
        "predicted_delta_logvar": None,
    } for row in development_rows]
    prediction_path = dev / "predictions.jsonl"
    prediction_sha = write_prediction_jsonl_exclusive(
        prediction_path, prediction_rows
    )
    authorities = _authorities()
    verifier = _scalar_verifier(
        code_sha=authorities["analysis_code_tree_manifest_sha256"],
        config_sha=authorities["model_optimizer_seed_config_sha256"],
    )
    cell = {
        "cell_id": CELL,
        "train_rows_ref": make_artifact_ref(run_root, train_path, train_sha),
        "development_rows_ref": make_artifact_ref(
            run_root, dev_path, development_sha
        ),
        "prediction_jsonl_ref": make_artifact_ref(
            run_root, prediction_path, prediction_sha
        ),
        "predictor_artifacts": {
            "kind": SCALAR_KIND,
            "fitted_state_ref": state_ref,
            "fit_config_ref": config_ref,
        },
    }
    provenance = build_predictor_provenance(
        run_root,
        run_start_sha256=RUN_SHA,
        activation_manifest_sha256=ACTIVATION_SHA,
        train_development_outcomes_sha256=TRAIN_DEVELOPMENT_OUTCOMES_SHA,
        non_data_authority_hashes=authorities,
        method_id=METHOD,
        information_set_fingerprint=INFO_SHA,
        predictor_code_sha256=authorities[
            "analysis_code_tree_manifest_sha256"
        ],
        predictor_config_sha256=authorities[
            "model_optimizer_seed_config_sha256"
        ],
        verifier=verifier,
        estimands=[{
            "estimand_id": ESTIMAND,
            "estimand_binding_sha256": ESTIMAND_SHA,
            "cells": [cell],
        }],
    )
    provenance_path = dev / "predictor_provenance.json"
    provenance_sha = write_predictor_provenance_exclusive(
        run_root,
        provenance_path,
        provenance,
        verifiers={verifier.verifier_id: verifier},
    )
    return {
        "run_root": run_root,
        "dev": dev,
        "authorities": authorities,
        "verifier": verifier,
        "cell": cell,
        "provenance": provenance,
        "provenance_path": provenance_path,
        "provenance_ref": make_artifact_ref(
            run_root, provenance_path, provenance_sha
        ),
    }


def test_scalar_exact_replay_binds_rows_state_config_code_and_authorities(
    tmp_path: Path,
) -> None:
    bundle = _scalar_bundle(tmp_path)
    verifier = bundle["verifier"]
    replayed = replay_predictor_provenance(
        bundle["run_root"],
        bundle["provenance_ref"],
        verifiers={verifier.verifier_id: verifier},
        expected_run_start_sha256=RUN_SHA,
        expected_activation_manifest_sha256=ACTIVATION_SHA,
        expected_train_development_outcomes_sha256=(
            TRAIN_DEVELOPMENT_OUTCOMES_SHA
        ),
        expected_non_data_authority_hashes=bundle["authorities"],
        expected_method_id=METHOD,
        expected_information_set_fingerprint=INFO_SHA,
    )
    assert replayed == bundle["provenance"]
    assert replayed["schema_version"] == PREDICTOR_PROVENANCE_SCHEMA
    assert replayed["partition"] == DEVELOPMENT_PARTITION
    assert replayed["verifier_identity"] == verifier_identity(verifier)
    assert bundle["provenance_path"].read_bytes() == (
        canonical_predictor_provenance_bytes(replayed)
    )
    serialized = json.dumps(replayed, sort_keys=True).lower()
    assert "internal_test" not in serialized
    assert '"sealed' not in serialized


def test_scalar_swapped_state_prediction_and_rows_fail_exact_replay(
    tmp_path: Path,
) -> None:
    bundle = _scalar_bundle(tmp_path)
    run_root = bundle["run_root"]
    verifier = bundle["verifier"]
    registry = {verifier.verifier_id: verifier}

    swapped_state = _write_bytes(
        run_root, bundle["dev"] / "wrong_state.json", {"weight": -9.0}
    )
    input_estimands = [{
        "estimand_id": ESTIMAND,
        "estimand_binding_sha256": ESTIMAND_SHA,
        "cells": [copy.deepcopy(bundle["cell"])],
    }]
    input_estimands[0]["cells"][0]["predictor_artifacts"][
        "fitted_state_ref"
    ] = swapped_state
    with pytest.raises(PredictorProvenanceError, match="exact predictor recomputation"):
        build_predictor_provenance(
            run_root,
            run_start_sha256=RUN_SHA,
            activation_manifest_sha256=ACTIVATION_SHA,
            train_development_outcomes_sha256=TRAIN_DEVELOPMENT_OUTCOMES_SHA,
            non_data_authority_hashes=bundle["authorities"],
            method_id=METHOD,
            information_set_fingerprint=INFO_SHA,
            predictor_code_sha256=bundle["authorities"][
                "analysis_code_tree_manifest_sha256"
            ],
            predictor_config_sha256=bundle["authorities"][
                "model_optimizer_seed_config_sha256"
            ],
            verifier=verifier,
            estimands=input_estimands,
        )

    wrong_prediction_path = bundle["dev"] / "wrong_prediction.jsonl"
    original_rows = parse_predictor_input_jsonl(
        read_artifact_ref(run_root, bundle["cell"]["development_rows_ref"])
    )
    wrong_prediction_sha = write_prediction_jsonl_exclusive(
        wrong_prediction_path,
        [{
            "schema_version": "r3x_prediction_row_v1",
            "row_id": row["row_id"],
            "predicted_delta_mean": -100.0,
            "predicted_delta_logvar": None,
        } for row in original_rows],
    )
    forged = copy.deepcopy(bundle["provenance"])
    forged["estimands"][0]["cells"][0]["prediction_jsonl_ref"] = (
        make_artifact_ref(run_root, wrong_prediction_path, wrong_prediction_sha)
    )
    with pytest.raises(PredictorProvenanceError, match="exact predictor recomputation"):
        write_predictor_provenance_exclusive(
            run_root,
            bundle["dev"] / "forged_prediction_provenance.json",
            forged,
            verifiers=registry,
        )

    wrong_rows_path = bundle["dev"] / "wrong_rows.jsonl"
    wrong_rows = _input_rows(DEVELOPMENT_PARTITION, suffix="WRONG")
    wrong_rows_sha = write_predictor_input_jsonl_exclusive(
        wrong_rows_path, wrong_rows
    )
    wrong_rows_input = copy.deepcopy(bundle["cell"])
    wrong_rows_input["development_rows_ref"] = make_artifact_ref(
        run_root, wrong_rows_path, wrong_rows_sha
    )
    with pytest.raises(PredictorProvenanceError, match="prediction rows"):
        build_predictor_provenance(
            run_root,
            run_start_sha256=RUN_SHA,
            activation_manifest_sha256=ACTIVATION_SHA,
            train_development_outcomes_sha256=TRAIN_DEVELOPMENT_OUTCOMES_SHA,
            non_data_authority_hashes=bundle["authorities"],
            method_id=METHOD,
            information_set_fingerprint=INFO_SHA,
            predictor_code_sha256=bundle["authorities"][
                "analysis_code_tree_manifest_sha256"
            ],
            predictor_config_sha256=bundle["authorities"][
                "model_optimizer_seed_config_sha256"
            ],
            verifier=verifier,
            estimands=[{
                "estimand_id": ESTIMAND,
                "estimand_binding_sha256": ESTIMAND_SHA,
                "cells": [wrong_rows_input],
            }],
        )


def test_missing_or_mismatched_verifier_and_external_bindings_fail_closed(
    tmp_path: Path,
) -> None:
    bundle = _scalar_bundle(tmp_path)
    with pytest.raises(PredictorProvenanceError, match="registry is required"):
        replay_predictor_provenance(
            bundle["run_root"], bundle["provenance_ref"], verifiers=None
        )
    wrong_verifier = _scalar_verifier(
        code_sha=bundle["authorities"]["analysis_code_tree_manifest_sha256"],
        config_sha=bundle["authorities"]["model_optimizer_seed_config_sha256"],
        verifier_sha="f" * 64,
    )
    with pytest.raises(PredictorProvenanceError, match="identity, schema"):
        replay_predictor_provenance(
            bundle["run_root"],
            bundle["provenance_ref"],
            verifiers={wrong_verifier.verifier_id: wrong_verifier},
        )
    registry = {bundle["verifier"].verifier_id: bundle["verifier"]}
    with pytest.raises(PredictorProvenanceError, match="information_set_fingerprint"):
        replay_predictor_provenance(
            bundle["run_root"],
            bundle["provenance_ref"],
            verifiers=registry,
            expected_information_set_fingerprint="f" * 64,
        )
    wrong_authorities = dict(bundle["authorities"])
    wrong_authorities[NON_DATA_AUTHORITY_HASH_KEYS[0]] = "f" * 64
    with pytest.raises(PredictorProvenanceError, match="non-data authority"):
        replay_predictor_provenance(
            bundle["run_root"],
            bundle["provenance_ref"],
            verifiers=registry,
            expected_non_data_authority_hashes=wrong_authorities,
        )


def _neural_bundle(tmp_path: Path) -> dict:
    run_root = tmp_path / "ordinary_run"
    dev = run_root / "development" / "neural"
    dev.mkdir(parents=True)
    train_rows = _input_rows(TRAIN_PARTITION)
    development_rows = _input_rows(DEVELOPMENT_PARTITION)
    train_path = dev / "train_rows.jsonl"
    development_path = dev / "development_rows.jsonl"
    train_sha = write_predictor_input_jsonl_exclusive(train_path, train_rows)
    development_sha = write_predictor_input_jsonl_exclusive(
        development_path, development_rows
    )
    train_set, _ = _row_hashes(train_rows)
    development_set, _ = _row_hashes(development_rows)
    authorities = _authorities()
    code_sha = authorities["analysis_code_tree_manifest_sha256"]
    config_sha = authorities["model_optimizer_seed_config_sha256"]
    seed_artifacts = []
    state_by_seed = {}
    seed_prediction_rows = []
    for seed_index, seed in enumerate(FROZEN_NEURAL_SEEDS, start=1):
        unit = dev / f"seed_{seed}"
        unit.mkdir()
        state = {"weight": 1.0 + seed_index / 10.0}
        state_ref = _write_bytes(run_root, unit / "state.json", state)
        state_by_seed[str(seed)] = state
        selection = build_development_checkpoint_selection(
            run_start_sha256=RUN_SHA,
            activation_manifest_sha256=ACTIVATION_SHA,
            train_development_outcomes_sha256=TRAIN_DEVELOPMENT_OUTCOMES_SHA,
            non_data_authority_hashes=authorities,
            method_id="FULL_PAIR_UNRESTRICTED_SIAMESE",
            estimand_id=ESTIMAND,
            estimand_binding_sha256=ESTIMAND_SHA,
            cell_id=CELL,
            seed=seed,
            train_row_set_sha256=train_set,
            development_row_set_sha256=development_set,
            predictor_code_sha256=code_sha,
            predictor_config_sha256=config_sha,
            selected_epoch=seed_index,
        )
        selection_path = unit / "checkpoint_selection.json"
        selection_sha = write_development_checkpoint_selection_exclusive(
            selection_path, selection
        )
        selection_ref = make_artifact_ref(run_root, selection_path, selection_sha)
        checkpoint = build_development_checkpoint(
            run_start_sha256=RUN_SHA,
            activation_manifest_sha256=ACTIVATION_SHA,
            train_development_outcomes_sha256=TRAIN_DEVELOPMENT_OUTCOMES_SHA,
            non_data_authority_hashes=authorities,
            method_id="FULL_PAIR_UNRESTRICTED_SIAMESE",
            estimand_id=ESTIMAND,
            estimand_binding_sha256=ESTIMAND_SHA,
            cell_id=CELL,
            seed=seed,
            train_row_set_sha256=train_set,
            development_row_set_sha256=development_set,
            information_set_fingerprint=INFO_SHA,
            predictor_code_sha256=code_sha,
            predictor_config_sha256=config_sha,
            checkpoint_selection_ref=selection_ref,
            state_ref=state_ref,
            architecture_id="FULL_PAIR_SIAMESE_V1",
        )
        checkpoint_path = unit / "checkpoint.json"
        checkpoint_sha = write_development_checkpoint_exclusive(
            checkpoint_path, checkpoint
        )
        checkpoint_ref = make_artifact_ref(
            run_root, checkpoint_path, checkpoint_sha
        )
        predictions = [{
            "schema_version": "r3x_prediction_row_v1",
            "row_id": row["row_id"],
            "predicted_delta_mean": (
                float(state["weight"]) * _feature_score(row)
            ),
            "predicted_delta_logvar": math.log(0.5),
        } for row in development_rows]
        prediction_path = unit / "development_predictions.jsonl"
        prediction_sha = write_prediction_jsonl_exclusive(
            prediction_path, predictions
        )
        prediction_ref = make_artifact_ref(
            run_root, prediction_path, prediction_sha
        )
        seed_prediction_rows.append(predictions)
        seed_artifacts.append({
            "seed": seed,
            "checkpoint_selection_ref": selection_ref,
            "checkpoint_ref": checkpoint_ref,
            "development_prediction_ref": prediction_ref,
        })
    ensemble_rows = []
    for row_index, row in enumerate(development_rows):
        means = [
            seed_rows[row_index]["predicted_delta_mean"]
            for seed_rows in seed_prediction_rows
        ]
        logvars = [
            seed_rows[row_index]["predicted_delta_logvar"]
            for seed_rows in seed_prediction_rows
        ]
        mean = math.fsum(means) / len(means)
        variance = math.fsum(
            math.exp(logvar) + value * value
            for value, logvar in zip(means, logvars)
        ) / len(means) - mean * mean
        ensemble_rows.append({
            "schema_version": "r3x_prediction_row_v1",
            "row_id": row["row_id"],
            "predicted_delta_mean": mean,
            "predicted_delta_logvar": math.log(max(variance, VARIANCE_FLOOR)),
        })
    ensemble_path = dev / "ensemble_predictions.jsonl"
    ensemble_sha = write_prediction_jsonl_exclusive(
        ensemble_path, ensemble_rows
    )
    ensemble_ref = make_artifact_ref(run_root, ensemble_path, ensemble_sha)

    def recompute(root: Path, cell: dict) -> dict:
        rows = parse_predictor_input_jsonl(
            read_artifact_ref(root, cell["development_rows_ref"])
        )
        result = {}
        for item in cell["predictor_artifacts"]["seed_artifacts"]:
            checkpoint = json.loads(read_artifact_ref(root, item["checkpoint_ref"]))
            state = json.loads(read_artifact_ref(root, checkpoint["state_ref"]))
            result[str(item["seed"])] = [{
                "schema_version": "r3x_prediction_row_v1",
                "row_id": row["row_id"],
                "predicted_delta_mean": (
                    float(state["weight"]) * _feature_score(row)
                ),
                "predicted_delta_logvar": math.log(0.5),
            } for row in rows]
        return result

    verifier = PredictorVerifier(
        verifier_id="NEURAL_EXACT_REPLAY_V1",
        code_sha256=VERIFIER_SHA,
        expected_predictor_code_sha256=code_sha,
        expected_predictor_config_sha256=config_sha,
        recompute=recompute,
    )
    cell = {
        "cell_id": CELL,
        "train_rows_ref": make_artifact_ref(run_root, train_path, train_sha),
        "development_rows_ref": make_artifact_ref(
            run_root, development_path, development_sha
        ),
        "prediction_jsonl_ref": ensemble_ref,
        "predictor_artifacts": {
            "kind": NEURAL_KIND,
            "seeds": list(FROZEN_NEURAL_SEEDS),
            "seed_artifacts": seed_artifacts,
            "ensemble_prediction_ref": ensemble_ref,
            "ensemble_rule": ENSEMBLE_RULE,
            "ensemble_variance_formula": ENSEMBLE_VARIANCE_FORMULA,
            "variance_floor": VARIANCE_FLOOR,
        },
    }
    provenance = build_predictor_provenance(
        run_root,
        run_start_sha256=RUN_SHA,
        activation_manifest_sha256=ACTIVATION_SHA,
        train_development_outcomes_sha256=TRAIN_DEVELOPMENT_OUTCOMES_SHA,
        non_data_authority_hashes=authorities,
        method_id="FULL_PAIR_UNRESTRICTED_SIAMESE",
        information_set_fingerprint=INFO_SHA,
        predictor_code_sha256=code_sha,
        predictor_config_sha256=config_sha,
        verifier=verifier,
        estimands=[{
            "estimand_id": ESTIMAND,
            "estimand_binding_sha256": ESTIMAND_SHA,
            "cells": [cell],
        }],
    )
    return {
        "run_root": run_root,
        "dev": dev,
        "authorities": authorities,
        "verifier": verifier,
        "cell": cell,
        "provenance": provenance,
    }


def test_neural_exact_replay_requires_staged_safe_checkpoints_and_five_seeds(
    tmp_path: Path,
) -> None:
    bundle = _neural_bundle(tmp_path)
    assert validate_predictor_provenance(bundle["provenance"]) == bundle["provenance"]
    cell = bundle["provenance"]["estimands"][0]["cells"][0]
    assert cell["predictor_artifacts"]["seeds"] == list(FROZEN_NEURAL_SEEDS)

    swapped = copy.deepcopy(bundle["provenance"])
    artifacts = swapped["estimands"][0]["cells"][0]["predictor_artifacts"][
        "seed_artifacts"
    ]
    artifacts[0]["checkpoint_ref"], artifacts[1]["checkpoint_ref"] = (
        artifacts[1]["checkpoint_ref"],
        artifacts[0]["checkpoint_ref"],
    )
    with pytest.raises(PredictorProvenanceError, match="seed binding mismatch"):
        write_predictor_provenance_exclusive(
            bundle["run_root"],
            bundle["dev"] / "swapped_checkpoint_provenance.json",
            swapped,
            verifiers={bundle["verifier"].verifier_id: bundle["verifier"]},
        )

    legacy_ref = _write_bytes(
        bundle["run_root"],
        bundle["dev"] / "legacy_checkpoint.pt",
        b"legacy r3x_critic_checkpoint_v2 with internal split binding",
    )
    legacy = copy.deepcopy(bundle["provenance"])
    legacy["estimands"][0]["cells"][0]["predictor_artifacts"][
        "seed_artifacts"
    ][0]["checkpoint_ref"] = legacy_ref
    with pytest.raises(
        PredictorProvenanceError, match="not strict development-only JSON"
    ):
        write_predictor_provenance_exclusive(
            bundle["run_root"],
            bundle["dev"] / "legacy_checkpoint_provenance.json",
            legacy,
            verifiers={bundle["verifier"].verifier_id: bundle["verifier"]},
        )


@pytest.mark.parametrize(
    "extra_key",
    [
        "foo",
        "study_proxy",
        "feature_digest",
        "internal_test",
        "true_delta",
        "source_value",
    ],
)
def test_predictor_input_rejects_every_non_primary_feature_key(
    extra_key: str,
) -> None:
    rows = _input_rows(DEVELOPMENT_PARTITION)
    rows[0]["predictor_input"][extra_key] = "opaque-or-protected"
    with pytest.raises(PredictorProvenanceError, match="strict schema"):
        normalize_predictor_input_rows(rows)


def test_predictor_input_requires_exact_keys_and_v2_schema() -> None:
    rows = _input_rows(DEVELOPMENT_PARTITION)
    normalized = normalize_predictor_input_rows(rows)
    assert normalized[0]["schema_version"] == PREDICTOR_INPUT_ROW_SCHEMA
    assert set(normalized[0]["predictor_input"]) == {
        "endpoint",
        "publication_context_id",
        "region",
        "normalized_source_sequence",
        "normalized_candidate_sequence",
        "edit_list",
        "edit_count",
    }

    missing = copy.deepcopy(rows)
    del missing[0]["predictor_input"]["endpoint"]
    with pytest.raises(PredictorProvenanceError, match="strict schema"):
        normalize_predictor_input_rows(missing)

    alias = copy.deepcopy(rows)
    alias[0]["predictor_input"]["endpoint_id"] = alias[0]["predictor_input"].pop(
        "endpoint"
    )
    with pytest.raises(PredictorProvenanceError, match="strict schema"):
        normalize_predictor_input_rows(alias)

    legacy = copy.deepcopy(rows)
    legacy[0]["schema_version"] = "r3x_predictor_input_row_v1"
    with pytest.raises(PredictorProvenanceError, match="unsupported"):
        normalize_predictor_input_rows(legacy)


@pytest.mark.parametrize(
    ("field", "bad_value", "message"),
    [
        ("endpoint", " mrna_abundance", "canonical string"),
        ("publication_context_id", "context\nleak", "canonical string"),
        ("normalized_source_sequence", "acguacguacgu", "uppercase A/C/G/U"),
        ("normalized_source_sequence", "ACGTACGUACGU", "uppercase A/C/G/U"),
        ("normalized_candidate_sequence", "CCGUACGUACG", "equal length"),
    ],
)
def test_predictor_input_rejects_noncanonical_metadata_or_sequences(
    field: str,
    bad_value: object,
    message: str,
) -> None:
    rows = _input_rows(DEVELOPMENT_PARTITION)
    rows[0]["predictor_input"][field] = bad_value
    with pytest.raises(PredictorProvenanceError, match=message):
        normalize_predictor_input_rows(rows)


def test_predictor_input_requires_exact_recursive_edit_schema() -> None:
    row = _input_rows(DEVELOPMENT_PARTITION)[0]

    extra = copy.deepcopy(row)
    extra["predictor_input"]["edit_list"][0]["token"] = "C"
    with pytest.raises(PredictorProvenanceError, match="strict schema"):
        normalize_predictor_input_rows([extra])

    alias = copy.deepcopy(row)
    edit = alias["predictor_input"]["edit_list"][0]
    edit["position"] = edit.pop("pos")
    with pytest.raises(PredictorProvenanceError, match="strict schema"):
        normalize_predictor_input_rows([alias])

    wrong_op = copy.deepcopy(row)
    wrong_op["predictor_input"]["edit_list"][0]["op"] = "sub"
    with pytest.raises(PredictorProvenanceError, match="exactly SUB"):
        normalize_predictor_input_rows([wrong_op])

    bool_position = copy.deepcopy(row)
    bool_position["predictor_input"]["edit_list"][0]["pos"] = True
    with pytest.raises(PredictorProvenanceError, match="zero-based"):
        normalize_predictor_input_rows([bool_position])

    bad_base = copy.deepcopy(row)
    bad_base["predictor_input"]["edit_list"][0]["alt"] = "T"
    with pytest.raises(PredictorProvenanceError, match="distinct uppercase"):
        normalize_predictor_input_rows([bad_base])


def test_predictor_input_requires_frozen_budget_and_exact_replay() -> None:
    row = _input_rows(DEVELOPMENT_PARTITION)[0]

    invalid_budget = copy.deepcopy(row)
    invalid_budget["predictor_input"] = _predictor_input(1, positions=(0, 1))
    with pytest.raises(PredictorProvenanceError, match="frozen budgets"):
        normalize_predictor_input_rows([invalid_budget])

    count_mismatch = copy.deepcopy(row)
    count_mismatch["predictor_input"]["edit_count"] = 3
    with pytest.raises(PredictorProvenanceError, match="exactly equal"):
        normalize_predictor_input_rows([count_mismatch])

    wrong_ref = copy.deepcopy(row)
    wrong_ref["predictor_input"]["edit_list"][0]["ref"] = "U"
    with pytest.raises(PredictorProvenanceError, match="does not match"):
        normalize_predictor_input_rows([wrong_ref])

    wrong_alt = copy.deepcopy(row)
    wrong_alt["predictor_input"]["edit_list"][0]["alt"] = "G"
    with pytest.raises(PredictorProvenanceError, match="does not match"):
        normalize_predictor_input_rows([wrong_alt])

    unregistered_difference = copy.deepcopy(row)
    candidate = list(
        unregistered_difference["predictor_input"][
            "normalized_candidate_sequence"
        ]
    )
    candidate[4] = "C"
    unregistered_difference["predictor_input"][
        "normalized_candidate_sequence"
    ] = "".join(candidate)
    with pytest.raises(PredictorProvenanceError, match="exact replay"):
        normalize_predictor_input_rows([unregistered_difference])

    unsorted = copy.deepcopy(row)
    unsorted["predictor_input"] = _predictor_input(1, positions=(0, 1, 2))
    unsorted["predictor_input"]["edit_list"].reverse()
    with pytest.raises(PredictorProvenanceError, match="strictly increasing"):
        normalize_predictor_input_rows([unsorted])

    duplicate = copy.deepcopy(row)
    duplicate["predictor_input"] = _predictor_input(1, positions=(0, 1, 2))
    duplicate_edit = copy.deepcopy(duplicate["predictor_input"]["edit_list"][0])
    duplicate["predictor_input"]["edit_list"] = [
        duplicate_edit,
        copy.deepcopy(duplicate_edit),
        copy.deepcopy(duplicate_edit),
    ]
    with pytest.raises(PredictorProvenanceError, match="strictly increasing"):
        normalize_predictor_input_rows([duplicate])

    valid_three = copy.deepcopy(row)
    valid_three["predictor_input"] = _predictor_input(1, positions=(0, 1, 2))
    assert normalize_predictor_input_rows([valid_three])[0]["predictor_input"] == (
        valid_three["predictor_input"]
    )

    valid_five = copy.deepcopy(row)
    valid_five["predictor_input"] = _predictor_input(
        1, positions=(0, 1, 2, 3, 4)
    )
    assert normalize_predictor_input_rows([valid_five])[0]["predictor_input"] == (
        valid_five["predictor_input"]
    )


def test_training_target_is_separate_and_development_target_is_none() -> None:
    train = normalize_predictor_input_rows(_input_rows(TRAIN_PARTITION))
    assert all(row["training_target"] is not None for row in train)
    assert all("training_target" not in row["predictor_input"] for row in train)

    target = _input_rows(DEVELOPMENT_PARTITION)
    target[0]["training_target"] = 1.0
    with pytest.raises(PredictorProvenanceError, match="cannot contain"):
        normalize_predictor_input_rows(target)


def test_old_or_extra_provenance_schema_never_validates(tmp_path: Path) -> None:
    payload = copy.deepcopy(_scalar_bundle(tmp_path)["provenance"])
    payload["schema_version"] = "r3x_development_predictor_provenance_v1"
    with pytest.raises(PredictorProvenanceError, match="unsupported"):
        validate_predictor_provenance(payload)
    payload = copy.deepcopy(_scalar_bundle(tmp_path / "extra")["provenance"])
    payload["self_reported_metrics"] = {"sign_accuracy": 1.0}
    with pytest.raises(PredictorProvenanceError, match="strict schema"):
        validate_predictor_provenance(payload)
