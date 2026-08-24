#!/usr/bin/env python3
"""Consume one Critic V4 authorization and atomically score full plus C0 ensembles."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.route2_bottom_encoder_chunk_cache_v4 import (
    assemble_frozen_bottom_encoder_chunk_cache_v4,
)
from core.route2_xeditcritic_batch_v4 import (
    FrozenBottomEncoderChunkCacheViewV4,
    XEditCriticCollatorV4,
    XEditCriticDatasetV4,
)
from core.route2_xeditcritic_gate_v3 import (
    paired_source_group_task_macro_bootstrap_v3,
)
from core.route2_xeditcritic_gate_v4 import adjudicate_critic_frozen_test_v4
from core.route2_xeditcritic_training_data_v3 import records_from_projection_rows
from scripts.route_a_v3.route2_mrnabert_bottom_six_encoder_v4 import (
    FrozenMRNABERTBottomSixEncoderV4,
)
from scripts.route_a_v3.run_route2_xeditcritic_v3_atomic_frozen_test import (
    load_authorized_test_rows_v3,
)
from scripts.route_a_v3.train_route2_xeditcritic_v3 import (
    TaskRobustScalerV3,
    XEditCriticCollatorV3,
    XEditCriticDatasetV3,
    require_cuda,
    validation_metrics,
)
from scripts.route_a_v3.train_route2_xeditcritic_v4 import (
    _build_model,
    _evaluate,
    screen_run_spec_v4,
)


class AtomicFrozenTestV4Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AtomicFrozenTestV4Error(message)


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"artifact is not an object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    _require(bool(rows), f"prediction artifact is empty: {path}")
    return rows


def require_atomic_test_authorization_v4(
    protocol: Mapping[str, Any], three_seed_gate: Mapping[str, Any]
) -> tuple[int, ...]:
    _require(
        protocol.get("schema_version")
        == "route_a_v3_route2_xeditcritic_v4_frozen_test_protocol.v1"
        and protocol.get("status")
        == "FROZEN_PROSPECTIVE_BEFORE_CONFIRMATION_OR_TEST_OUTCOME_READ",
        "Critic V4 frozen TEST protocol is not prospectively frozen",
    )
    seeds = tuple(int(seed) for seed in protocol["required_confirmation_seeds"])
    _require(
        seeds == (20260908, 20260909, 20260910)
        and three_seed_gate.get("required_seeds") == list(seeds)
        and three_seed_gate.get("status") == "XEDITCRITIC_V4_THREE_SEED_PASS"
        and three_seed_gate.get("development_test_authorized") is True
        and three_seed_gate.get("atomic_development_test_only") is True
        and three_seed_gate.get("additional_seed_authorized") is False,
        "Critic V4 three-seed gate does not authorize exact atomic TEST",
    )
    _require(
        protocol.get("single_atomic_access_authorized_only_after_three_seed_pass")
        is True
        and protocol.get("ephemeral_test_rows_only") is True
        and protocol.get("general_test_projection_persisted") is False
        and protocol.get("test_bottom_six_cache_persisted") is False
        and protocol.get("new_final_evaluation_outcomes_accessed") is False,
        "Critic V4 atomic TEST persistence or access policy changed",
    )
    return seeds


def _scaler(payload: Mapping[str, Any]) -> TaskRobustScalerV3:
    _require(
        payload.get("schema_version")
        == "route_a_v3_route2_xeditcritic_task_robust_scaler.v3",
        "Critic V4 checkpoint scaler identity changed",
    )
    return TaskRobustScalerV3(
        scales={str(key): float(value) for key, value in payload["task_scales"].items()},
        region_scales={int(key): float(value) for key, value in payload["region_scales"].items()},
        global_scale=float(payload["global_scale"]),
        floor=float(payload["floor"]),
        training_record_count=int(payload["training_record_count"]),
    )


def _checkpoint_paths(
    protocol: Mapping[str, Any], seeds: Sequence[int]
) -> tuple[dict[int, dict[str, Any]], dict[str, dict[int, Path]]]:
    runtime_root = Path(protocol["confirmation_runtime_config_root"])
    configs = {}
    paths: dict[str, dict[int, Path]] = {"candidate": {}, "baseline": {}}
    for seed in seeds:
        config = _read(runtime_root / f"seed_{seed}.json")
        _require(
            config.get("schema_version")
            == "route_a_v3_route2_xeditcritic_v4_confirmation_runtime.v1"
            and config.get("run_stage") == "CONFIRMATION"
            and int(config.get("training_seed", -1)) == seed
            and config.get("required_confirmation_run_ids")
            == ["v4_full", "c0_v4"],
            f"Critic V4 confirmation runtime changed: {seed}",
        )
        configs[seed] = config
        for role, run_id in (("candidate", "v4_full"), ("baseline", "c0_v4")):
            path = Path(config["output_root"]) / run_id / "final_pass_8_checkpoint.pt"
            _require(path.is_file(), f"Critic V4 frozen checkpoint is absent: {seed}/{run_id}")
            paths[role][seed] = path
    return configs, paths


def _preflight_checkpoints(
    paths: Mapping[str, Mapping[int, Path]], seeds: Sequence[int]
) -> None:
    for role, run_id in (("candidate", "v4_full"), ("baseline", "c0_v4")):
        for seed in seeds:
            checkpoint = torch.load(paths[role][seed], map_location="cpu", weights_only=False)
            _require(
                checkpoint.get("schema_version")
                == "route_a_v3_route2_xeditcritic_v4_confirmation_checkpoint.v1"
                and checkpoint.get("run_stage") == "CONFIRMATION"
                and checkpoint.get("run_id") == run_id
                and int(checkpoint.get("seed", -1)) == seed
                and int(checkpoint.get("selected_pass", -1)) == 8
                and checkpoint.get("candidate_bundle_permutation") is False
                and int(checkpoint.get("development_test_outcome_reads", -1)) == 0
                and int(checkpoint.get("new_final_evaluation_outcome_reads", -1)) == 0,
                f"Critic V4 frozen checkpoint identity changed: {seed}/{run_id}",
            )


def _ephemeral_bottom_six_view(
    rows: Sequence[Mapping[str, Any]],
    *,
    protocol: Mapping[str, Any],
    device: torch.device,
) -> FrozenBottomEncoderChunkCacheViewV4:
    sequences = sorted(
        {
            str(sequence)
            for row in rows
            for sequence in (row["source_sequence"], row["candidate_sequence"])
        }
    )
    sequence_to_index = {sequence: index for index, sequence in enumerate(sequences)}
    encoder = FrozenMRNABERTBottomSixEncoderV4(
        Path(protocol["mrnabert_model_path"]),
        device,
        maximum_sequences_per_batch=int(
            protocol["bottom_six_maximum_sequences_per_batch"]
        ),
        batch_token_budget=int(protocol["bottom_six_batch_token_budget"]),
        attention_backend=str(protocol["attention_backend"]),
    )
    encoded = encoder.encode_online(
        {index: sequence for index, sequence in enumerate(sequences)}
    )
    cache_rows = [{**row, "split": "VALIDATION"} for row in rows]
    payload = assemble_frozen_bottom_encoder_chunk_cache_v4(
        cache_rows,
        sequence_to_index=sequence_to_index,
        encoded=encoded,
        model_id="EPHEMERAL_AUTHORIZED_V4_TEST_BOTTOM_SIX",
        pretrained_parameter_count=encoder.parameter_count,
        attention_backend=encoder.attention_backend,
    )
    del encoder, encoded
    torch.cuda.empty_cache()
    return FrozenBottomEncoderChunkCacheViewV4(
        payload, {str(row["canonical_record_id"]) for row in rows}
    )


def _predict_checkpoint(
    rows: Sequence[Mapping[str, Any]],
    checkpoint: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    run_id: str,
    cache_view: FrozenBottomEncoderChunkCacheViewV4,
    device: torch.device,
    prediction_path: Path,
) -> dict[str, Any]:
    projection_like = [{**row, "split": "VALIDATION"} for row in rows]
    records = [
        replace(record, split="TEST")
        for record in records_from_projection_rows(projection_like)
    ]
    vocabs = checkpoint["vocabs"]
    scaler = _scaler(checkpoint["target_scaler"])
    record_by_id = {record.record_id: record for record in records}
    if run_id == "v4_full":
        dataset = XEditCriticDatasetV4(
            records,
            all_records=record_by_id,
            vocabs=vocabs,
            target_scaler=scaler,
            cache=None,
        )
        collator = XEditCriticCollatorV4(
            cache_view,
            minimum_physical_batch=int(config["memory_preflight"]["minimum_physical_batch"]),
        )
    else:
        dataset = XEditCriticDatasetV3(
            records,
            all_records=record_by_id,
            vocabs=vocabs,
            target_scaler=scaler,
            cache=None,
        )
        collator = XEditCriticCollatorV3(pretrained_width=768)
    model, capacity = _build_model(
        config, screen_run_spec_v4(config, run_id), vocabs, device=device
    )
    _require(capacity == checkpoint["capacity"], "Critic V4 TEST checkpoint capacity changed")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    metrics = _evaluate(
        model,
        dataset,
        collator,
        physical_batch_size=int(checkpoint["physical_batch_size"]),
        device=device,
        prediction_path=prediction_path,
    )
    del model
    torch.cuda.empty_cache()
    return metrics


def _ensemble_prediction_rows(
    per_seed_rows: Mapping[int, Sequence[Mapping[str, Any]]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    seeds = tuple(sorted(per_seed_rows))
    by_seed = {
        seed: {str(row["record_id"]): row for row in per_seed_rows[seed]}
        for seed in seeds
    }
    record_ids = sorted(by_seed[seeds[0]])
    _require(
        all(set(by_seed[seed]) == set(record_ids) for seed in seeds),
        "Critic V4 TEST seed prediction records differ",
    )
    output = []
    for record_id in record_ids:
        rows = [by_seed[seed][record_id] for seed in seeds]
        for field in ("source_group_id", "task_id", "target", "scaled_target"):
            _require(
                all(row[field] == rows[0][field] for row in rows),
                f"Critic V4 TEST seed field differs: {record_id}/{field}",
            )
        predictions = [float(row["prediction"]) for row in rows]
        scaled_predictions = [float(row["scaled_prediction"]) for row in rows]
        output.append(
            {
                "record_id": record_id,
                "source_group_id": rows[0]["source_group_id"],
                "task_id": rows[0]["task_id"],
                "target": float(rows[0]["target"]),
                "scaled_target": float(rows[0]["scaled_target"]),
                "prediction": float(np.mean(predictions)),
                "scaled_prediction": float(np.mean(scaled_predictions)),
                "ensemble_sd": float(np.std(predictions)),
                "per_seed_predictions": {
                    str(seed): predictions[index] for index, seed in enumerate(seeds)
                },
            }
        )
    metrics = validation_metrics(
        [row["target"] for row in output],
        [row["prediction"] for row in output],
        [row["scaled_target"] for row in output],
        [row["scaled_prediction"] for row in output],
        [row["task_id"] for row in output],
    )
    return output, metrics


def run(protocol: Mapping[str, Any]) -> dict[str, Any]:
    output_directory = Path(protocol["output_directory"])
    _require(not output_directory.exists(), f"Critic V4 atomic TEST is already consumed: {output_directory}")
    three_seed_gate = _read(Path(protocol["three_seed_gate_path"]))
    seeds = require_atomic_test_authorization_v4(protocol, three_seed_gate)
    configs, checkpoint_paths = _checkpoint_paths(protocol, seeds)
    _preflight_checkpoints(checkpoint_paths, seeds)
    for path in (
        Path(protocol["development_manifest"]),
        Path(protocol["endpoint_descriptor_registry"]),
        Path(protocol["mrnabert_model_path"]),
        *(Path(value) for value in protocol["canonical_paths"]),
    ):
        _require(path.exists(), f"Critic V4 atomic TEST preflight input is absent: {path}")
    device = require_cuda(int(protocol["physical_gpu_index"]))
    output_directory.mkdir(parents=True)
    (output_directory / "authorization_consumed.json").write_text(
        json.dumps(
            {
                "schema_version": "route_a_v3_route2_xeditcritic_v4_atomic_test_authorization.v1",
                "status": "ATOMIC_TEST_AUTHORIZATION_CONSUMED_NO_RETRY",
                "required_seeds": list(seeds),
                "consumed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "development_test_access_event_count": 1,
                "general_test_projection_persisted": False,
                "test_bottom_six_cache_persisted": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    access_started = False
    try:
        access_started = True
        rows = load_authorized_test_rows_v3(
            manifest_path=Path(protocol["development_manifest"]),
            canonical_paths=[Path(value) for value in protocol["canonical_paths"]],
            endpoint_descriptor_path=Path(protocol["endpoint_descriptor_registry"]),
            authorization_consumed=True,
        )
        _require(
            len(rows) == int(protocol["expected_test_record_count"]) == 18_292,
            "Critic V4 atomic TEST record count changed",
        )
        cache_view = _ephemeral_bottom_six_view(rows, protocol=protocol, device=device)
        per_role_seed_rows: dict[str, dict[int, list[dict[str, Any]]]] = {
            "candidate": {},
            "baseline": {},
        }
        for role, run_id in (("candidate", "v4_full"), ("baseline", "c0_v4")):
            for seed in seeds:
                checkpoint = torch.load(
                    checkpoint_paths[role][seed], map_location="cpu", weights_only=False
                )
                prediction_path = output_directory / f"{role}_seed_{seed}_test_predictions.private.jsonl"
                _predict_checkpoint(
                    rows,
                    checkpoint,
                    configs[seed],
                    run_id=run_id,
                    cache_view=cache_view,
                    device=device,
                    prediction_path=prediction_path,
                )
                per_role_seed_rows[role][seed] = _read_jsonl(prediction_path)
        candidate_rows, candidate_metrics = _ensemble_prediction_rows(
            per_role_seed_rows["candidate"]
        )
        baseline_rows, baseline_metrics = _ensemble_prediction_rows(
            per_role_seed_rows["baseline"]
        )
        for role, values in (("candidate", candidate_rows), ("baseline", baseline_rows)):
            (output_directory / f"{role}_ensemble_test_predictions.private.jsonl").write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in values),
                encoding="utf-8",
            )
        bootstrap = paired_source_group_task_macro_bootstrap_v3(
            candidate_rows,
            baseline_rows,
            iterations=int(protocol["bootstrap_iterations"]),
            seed=int(protocol["bootstrap_seed"]),
        )
        common = {
            "status": "ATOMIC_FROZEN_DEVELOPMENT_TEST_EVALUATION_COMPLETE",
            "test_record_count": len(rows),
            "development_test_outcomes_accessed": True,
            "development_test_access_event_count": 1,
            "new_final_evaluation_outcomes_accessed": False,
            "general_test_projection_persisted": False,
            "test_bottom_six_cache_persisted": False,
        }
        candidate = {**common, "model": "V4-FULL", "test_metrics": candidate_metrics}
        baseline = {**common, "model": "C0-V4", "test_metrics": baseline_metrics}
        gate = adjudicate_critic_frozen_test_v4(candidate, baseline, bootstrap)
        result = {
            "schema_version": "route_a_v3_route2_xeditcritic_v4_atomic_frozen_test.v1",
            "status": "ATOMIC_FROZEN_DEVELOPMENT_TEST_TERMINAL",
            "required_seeds": list(seeds),
            "candidate": candidate,
            "baseline": baseline,
            "paired_bootstrap": bootstrap,
            "frozen_test_gate": gate,
            "general_test_projection_persisted": False,
            "test_bottom_six_cache_persisted": False,
            "new_final_evaluation_outcomes_accessed": False,
        }
        (output_directory / "atomic_frozen_test.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return result
    except Exception as exc:
        failure = {
            "schema_version": "route_a_v3_route2_xeditcritic_v4_atomic_frozen_test_failure.v1",
            "status": "ATOMIC_FROZEN_TEST_TERMINAL_FAILURE_NO_AUTOMATIC_RETRY",
            "development_test_access_started": access_started,
            "development_test_access_event_count": int(access_started),
            "general_test_projection_persisted": False,
            "test_bottom_six_cache_persisted": False,
            "new_final_evaluation_outcomes_accessed": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        (output_directory / "failure.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    arguments = parser.parse_args()
    print(json.dumps(run(_read(arguments.protocol)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
