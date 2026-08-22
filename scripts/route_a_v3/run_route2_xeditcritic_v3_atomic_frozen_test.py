#!/usr/bin/env python3
"""Consume one Critic V3 TEST authorization and atomically score model plus C0."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.route2_development_projection_v3 import (
    extract_canonical_record_id,
    load_development_manifest,
    load_endpoint_descriptors,
    project_canonical_row,
)
from core.route2_xeditcritic_gate_v3 import (
    adjudicate_critic_frozen_test_v3,
    paired_source_group_task_macro_bootstrap_v3,
)
from core.route2_xeditcritic_training_data_v3 import records_from_projection_rows
from core.route2_xeditcritic_v3 import XEditCriticV3
from scripts.route_a_v3.route2_mrnabert_lora_edit_site_encoder_v3 import (
    TrainableMRNABERTEditSiteEncoderV3,
)
from scripts.route_a_v3.route2_mrnabert_edit_site_encoder_v3 import (
    FrozenMRNABERTEditSiteEncoderV3,
)
from scripts.route_a_v3.train_route2_xeditcritic_v3 import (
    TaskRobustScalerV3,
    XEditCriticCollatorV3,
    XEditCriticDatasetV3,
    _move,
    require_cuda,
    validation_metrics,
)
from scripts.route_a_v3.train_route2_xeditcritic_v3_c3_online import (
    microbatch_indices,
    select_batch_rows,
)


class AtomicFrozenTestV3Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AtomicFrozenTestV3Error(message)


def require_atomic_test_authorization_v3(
    protocol: Mapping[str, Any], three_seed_gate: Mapping[str, Any]
) -> tuple[str, tuple[int, ...]]:
    _require(
        protocol.get("schema_version")
        == "route_a_v3_route2_xeditcritic_v3_frozen_test_protocol.v1",
        "unexpected Critic V3 frozen TEST protocol",
    )
    _require(
        protocol.get("status")
        == "FROZEN_PROSPECTIVE_BEFORE_CONFIRMATION_OR_TEST_OUTCOME_READ",
        "Critic V3 frozen TEST protocol is not prospective",
    )
    _require(
        three_seed_gate.get("status") == "XEDITCRITIC_V3_THREE_SEED_PASS"
        and three_seed_gate.get("development_test_authorized") is True,
        "Critic V3 three-seed gate does not authorize Development TEST",
    )
    selected = str(three_seed_gate.get("selected_arm"))
    _require(selected in {"C2", "C3"}, "Critic V3 frozen TEST selected arm differs")
    seeds = tuple(int(seed) for seed in protocol["required_confirmation_seeds"])
    _require(
        seeds == (20260831, 20260901, 20260902)
        and three_seed_gate.get("required_seeds") == list(seeds),
        "Critic V3 frozen TEST seed cohort differs",
    )
    _require(
        protocol.get("single_atomic_access_authorized_only_after_three_seed_pass")
        is True
        and protocol.get("general_test_projection_persisted") is False
        and protocol.get("new_final_evaluation_outcomes_accessed") is False,
        "Critic V3 frozen TEST access policy differs",
    )
    return selected, seeds


def load_authorized_test_rows_v3(
    *,
    manifest_path: Path,
    canonical_paths: Sequence[Path],
    endpoint_descriptor_path: Path,
    authorization_consumed: bool,
) -> list[dict[str, Any]]:
    _require(authorization_consumed, "Development TEST authorization was not consumed")
    manifest = load_development_manifest(manifest_path)
    descriptors = load_endpoint_descriptors(endpoint_descriptor_path)
    expected = {record_id for record_id, row in manifest.items() if row["split"] == "TEST"}
    rows = []
    seen = set()
    for canonical_path in canonical_paths:
        with canonical_path.open(encoding="utf-8") as handle:
            for raw_line in handle:
                record_id = extract_canonical_record_id(raw_line)
                if record_id not in expected:
                    continue
                _require(record_id not in seen, f"authorized TEST record is duplicated: {record_id}")
                seen.add(record_id)
                canonical = json.loads(raw_line)
                modeling_manifest_row = {**manifest[record_id], "split": "VALIDATION"}
                projected = project_canonical_row(
                    canonical, modeling_manifest_row, descriptors
                )
                projected["split"] = "TEST"
                rows.append(projected)
    _require(seen == expected, "authorized TEST rows do not exactly cover the manifest")
    return sorted(rows, key=lambda row: str(row["canonical_record_id"]))


def _scaler(payload: Mapping[str, Any]) -> TaskRobustScalerV3:
    _require(
        payload.get("schema_version")
        == "route_a_v3_route2_xeditcritic_task_robust_scaler.v3",
        "Critic V3 checkpoint scaler differs",
    )
    return TaskRobustScalerV3(
        scales={str(key): float(value) for key, value in payload["task_scales"].items()},
        region_scales={int(key): float(value) for key, value in payload["region_scales"].items()},
        global_scale=float(payload["global_scale"]),
        floor=float(payload["floor"]),
        training_record_count=int(payload["training_record_count"]),
    )


def _load_lora_state(
    encoder: TrainableMRNABERTEditSiteEncoderV3, state: Mapping[str, torch.Tensor]
) -> None:
    parameters = dict(encoder.named_parameters())
    _require(
        set(state) == {name for name, parameter in parameters.items() if parameter.requires_grad},
        "Critic V3 C3 frozen TEST LoRA state differs",
    )
    with torch.no_grad():
        for name, value in state.items():
            parameters[name].copy_(value.to(parameters[name]))


class _AuthorizedTestFeatureViewV3:
    """Ephemeral frozen features; never serialized as a general TEST cache."""

    def __init__(self, rows: Sequence[Mapping[str, Any]], encoded, sequence_index) -> None:
        self.rows = {str(row["canonical_record_id"]): row for row in rows}
        self.encoded = encoded
        self.sequence_index = sequence_index

    def bundle(self, record_id: str) -> dict[str, torch.Tensor]:
        row = self.rows[record_id]
        source = self.encoded[self.sequence_index[str(row["source_sequence"])]]
        candidate = self.encoded[self.sequence_index[str(row["candidate_sequence"])]]
        positions = [int(edit["position"]) for edit in row["source_relative_edits"]]

        def stack(features, name):
            values = [getattr(features.positions[position], name) for position in positions]
            return torch.stack(values).to(torch.float16) if values else torch.empty((0, 768), dtype=torch.float16)

        return {
            "edit_positions": torch.tensor(positions, dtype=torch.int32),
            "source_site": stack(source, "site"),
            "candidate_site": stack(candidate, "site"),
            "source_window_mean": stack(source, "window_mean"),
            "candidate_window_mean": stack(candidate, "window_mean"),
            "source_window_max": stack(source, "window_max"),
            "candidate_window_max": stack(candidate, "window_max"),
            "source_global": source.global_residual.to(torch.float16),
            "candidate_global": candidate.global_residual.to(torch.float16),
        }


def _build_authorized_test_feature_view_v3(
    rows: Sequence[Mapping[str, Any]],
    *,
    protocol: Mapping[str, Any],
    device: torch.device,
) -> _AuthorizedTestFeatureViewV3:
    sequences = sorted(
        {str(sequence) for row in rows for sequence in (row["source_sequence"], row["candidate_sequence"])}
    )
    sequence_index = {sequence: index for index, sequence in enumerate(sequences)}
    positions: dict[int, set[int]] = defaultdict(set)
    for row in rows:
        requested = {int(edit["position"]) for edit in row["source_relative_edits"]}
        positions[sequence_index[str(row["source_sequence"])]].update(requested)
        positions[sequence_index[str(row["candidate_sequence"])]].update(requested)
    for index in range(len(sequences)):
        positions[index]
    encoder = FrozenMRNABERTEditSiteEncoderV3(
        Path(protocol["mrnabert_model_path"]),
        device,
        chunk_nucleotides=int(protocol["chunk_nucleotides"]),
        chunk_overlap=int(protocol["chunk_overlap"]),
        local_radius=int(protocol["local_radius"]),
        maximum_sequences_per_batch=int(protocol["maximum_sequences_per_batch"]),
        batch_token_budget=int(protocol["batch_token_budget"]),
        attention_backend=str(protocol["attention_backend"]),
    )
    encoded = encoder.encode_requested_features(
        {index: sequence for index, sequence in enumerate(sequences)}, positions
    )
    del encoder
    torch.cuda.empty_cache()
    return _AuthorizedTestFeatureViewV3(rows, encoded, sequence_index)


def _preflight_checkpoint(path: Path, *, arm: str, seed: int) -> None:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    _require(checkpoint.get("arm") == arm, f"frozen TEST checkpoint arm differs: {seed}/{arm}")
    _require(int(checkpoint.get("seed", -1)) == seed, f"frozen TEST checkpoint seed differs: {seed}/{arm}")
    required = {"model_state_dict", "vocabs", "target_scaler"}
    if arm == "C3":
        required.add("lora_state_dict")
    _require(required <= set(checkpoint), f"frozen TEST checkpoint payload is incomplete: {seed}/{arm}")
    del checkpoint


def _predict_one_seed(
    rows: Sequence[Mapping[str, Any]],
    checkpoint: Mapping[str, Any],
    *,
    arm: str,
    model_path: Path,
    device: torch.device,
    batch_size: int,
    online_microbatch_size: int,
    feature_view: _AuthorizedTestFeatureViewV3 | None,
) -> tuple[list[float], dict[str, Any]]:
    projection_like = [{**row, "split": "VALIDATION"} for row in rows]
    records = [
        replace(record, split="TEST")
        for record in records_from_projection_rows(projection_like)
    ]
    vocabs = checkpoint["vocabs"]
    scaler = _scaler(checkpoint["target_scaler"])
    dataset = XEditCriticDatasetV3(
        records,
        all_records={record.record_id: record for record in records},
        vocabs=vocabs,
        target_scaler=scaler,
        cache=feature_view if arm in {"C2", "C3"} else None,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=XEditCriticCollatorV3(pretrained_width=768),
        num_workers=0,
        pin_memory=True,
    )
    model = XEditCriticV3(
        arm=arm,
        control_mode="NONE",
        study_count=len(vocabs["study"]),
        assay_count=len(vocabs["assay"]),
        context_count=len(vocabs["context"]),
        quantity_count=len(vocabs["quantity"]),
        measurement_count=len(vocabs["measurement"]),
        numerator_count=len(vocabs["numerator"]),
        denominator_count=len(vocabs["denominator"]),
        pretrained_width=768,
        dropout=0.10,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    encoder = None
    if arm == "C3":
        encoder = TrainableMRNABERTEditSiteEncoderV3(
            model_path, device, rank=16, alpha=32.0, dropout=0.05
        )
        _load_lora_state(encoder, checkpoint["lora_state_dict"])
        encoder.eval()
    predictions = []
    with torch.inference_mode():
        for raw_batch in loader:
            batches = (
                [list(range(len(raw_batch["record_ids"])))]
                if arm != "C3"
                else microbatch_indices(
                    len(raw_batch["record_ids"]), online_microbatch_size
                )
            )
            for indices in batches:
                batch = _move(select_batch_rows(raw_batch, indices), device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    if arm == "C3":
                        assert encoder is not None
                        feature_batch = encoder.forward_cache_anchored(batch)
                    else:
                        feature_batch = batch
                    scaled = model(feature_batch)["mean"].float()
                predictions.extend(
                    (scaled * batch["target_scale"]).cpu().tolist()
                )
    _require(len(predictions) == len(records), "Critic V3 frozen TEST prediction count differs")
    del model, encoder
    torch.cuda.empty_cache()
    return predictions, scaler.to_dict()


def _ensemble_rows(
    rows: Sequence[Mapping[str, Any]],
    per_seed: Mapping[int, Sequence[float]],
    scaler_payload: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    scaler = _scaler(scaler_payload)
    predictions = np.asarray([per_seed[seed] for seed in sorted(per_seed)], dtype=float)
    mean = predictions.mean(axis=0)
    sd = predictions.std(axis=0)
    targets = [float(row["direction_normalized_delta"]) for row in rows]
    tasks = [str(row["task_id"]) for row in rows]
    scales = [scaler.scale(task, int(row["region_id"])) for task, row in zip(tasks, rows, strict=True)]
    scaled_targets = [target / scale for target, scale in zip(targets, scales, strict=True)]
    scaled_predictions = [prediction / scale for prediction, scale in zip(mean, scales, strict=True)]
    metrics = validation_metrics(
        targets, mean.tolist(), scaled_targets, scaled_predictions, tasks
    )
    output = []
    for index, row in enumerate(rows):
        output.append(
            {
                "record_id": str(row["canonical_record_id"]),
                "source_group_id": str(row["source_group_id"]),
                "task_id": str(row["task_id"]),
                "target": targets[index],
                "scaled_target": scaled_targets[index],
                "prediction": float(mean[index]),
                "ensemble_sd": float(sd[index]),
                "per_seed_predictions": {
                    str(seed): float(per_seed[seed][index]) for seed in sorted(per_seed)
                },
            }
        )
    return output, metrics


def run(protocol: Mapping[str, Any]) -> dict[str, Any]:
    output_directory = Path(protocol["output_directory"])
    _require(not output_directory.exists(), f"atomic frozen TEST is already consumed: {output_directory}")
    three_seed_gate = json.loads(
        Path(protocol["three_seed_gate_path"]).read_text(encoding="utf-8")
    )
    selected, seeds = require_atomic_test_authorization_v3(protocol, three_seed_gate)
    runtime_root = Path(protocol["confirmation_runtime_config_root"])
    checkpoints = {"candidate": {}, "baseline": {}}
    for seed in seeds:
        runtime = json.loads((runtime_root / f"seed{seed}.json").read_text(encoding="utf-8"))
        _require(
            int(runtime.get("seed", -1)) == seed
            and runtime.get("selected_arm") == selected,
            f"frozen TEST confirmation runtime differs: {seed}",
        )
        root = Path(runtime["output_root"])
        for role, arm in (("candidate", selected.lower()), ("baseline", "c0")):
            path = root / arm / "final_pass_checkpoint.pt"
            _require(path.is_file(), f"frozen TEST checkpoint is absent: {seed}/{arm}")
            checkpoints[role][seed] = path
    device = require_cuda(int(protocol["physical_gpu_index"]))
    for role, arm in (("candidate", selected), ("baseline", "C0")):
        for seed in seeds:
            _preflight_checkpoint(checkpoints[role][seed], arm=arm, seed=seed)
    for path in (
        Path(protocol["development_manifest"]),
        Path(protocol["endpoint_descriptor_registry"]),
        Path(protocol["mrnabert_model_path"]),
        *(Path(path) for path in protocol["canonical_paths"]),
    ):
        _require(path.exists(), f"atomic frozen TEST preflight input is absent: {path}")
    output_directory.mkdir(parents=True)
    consumed = {
        "schema_version": "route_a_v3_route2_xeditcritic_v3_atomic_test_authorization.v1",
        "status": "ATOMIC_TEST_AUTHORIZATION_CONSUMED_NO_RETRY",
        "selected_arm": selected,
        "required_seeds": list(seeds),
        "consumed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "development_test_access_event_count": 1,
        "general_test_projection_persisted": False,
    }
    (output_directory / "authorization_consumed.json").write_text(
        json.dumps(consumed, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    access_started = False
    try:
        access_started = True
        rows = load_authorized_test_rows_v3(
            manifest_path=Path(protocol["development_manifest"]),
            canonical_paths=[Path(path) for path in protocol["canonical_paths"]],
            endpoint_descriptor_path=Path(protocol["endpoint_descriptor_registry"]),
            authorization_consumed=True,
        )
        _require(
            len(rows) == int(protocol["expected_test_record_count"]) == 18292,
            "atomic frozen TEST record count differs",
        )
        feature_view = _build_authorized_test_feature_view_v3(
            rows, protocol=protocol, device=device
        )
        predictions: dict[str, dict[int, list[float]]] = {"candidate": {}, "baseline": {}}
        scalers: dict[str, dict[int, Mapping[str, Any]]] = {"candidate": {}, "baseline": {}}
        for role, arm in (("candidate", selected), ("baseline", "C0")):
            for seed in seeds:
                checkpoint = torch.load(
                    checkpoints[role][seed], map_location="cpu", weights_only=False
                )
                prediction, scaler = _predict_one_seed(
                    rows,
                    checkpoint,
                    arm=arm,
                    model_path=Path(protocol["mrnabert_model_path"]),
                    device=device,
                    batch_size=int(protocol["batch_size"]),
                    online_microbatch_size=int(protocol["online_encoder_microbatch_size"]),
                    feature_view=feature_view,
                )
                predictions[role][seed] = prediction
                scalers[role][seed] = scaler
            _require(
                all(
                    scalers[role][seed] == scalers[role][seeds[0]]
                    for seed in seeds
                ),
                f"frozen TEST scaler differs across {role} seeds",
            )
        candidate_rows, candidate_metrics = _ensemble_rows(
            rows, predictions["candidate"], scalers["candidate"][seeds[0]]
        )
        baseline_rows, baseline_metrics = _ensemble_rows(
            rows, predictions["baseline"], scalers["baseline"][seeds[0]]
        )
        for name, values in (("candidate", candidate_rows), ("baseline", baseline_rows)):
            (output_directory / f"{name}_test_predictions.private.jsonl").write_text(
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
        }
        candidate_summary = {**common, "arm": selected, "test_metrics": candidate_metrics}
        baseline_summary = {**common, "arm": "C0", "test_metrics": baseline_metrics}
        gate = adjudicate_critic_frozen_test_v3(
            candidate_summary, baseline_summary, bootstrap
        )
        result = {
            "schema_version": "route_a_v3_route2_xeditcritic_v3_atomic_frozen_test.v1",
            "status": "ATOMIC_FROZEN_DEVELOPMENT_TEST_TERMINAL",
            "selected_arm": selected,
            "required_seeds": list(seeds),
            "candidate": candidate_summary,
            "baseline": baseline_summary,
            "paired_bootstrap": bootstrap,
            "frozen_test_gate": gate,
            "general_test_projection_persisted": False,
            "new_final_evaluation_outcomes_accessed": False,
        }
        (output_directory / "atomic_frozen_test.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return result
    except Exception as exc:
        failure = {
            "schema_version": "route_a_v3_route2_xeditcritic_v3_atomic_frozen_test_failure.v1",
            "status": "ATOMIC_FROZEN_TEST_TERMINAL_FAILURE_NO_AUTOMATIC_RETRY",
            "development_test_access_started": access_started,
            "development_test_access_event_count": int(access_started),
            "general_test_projection_persisted": False,
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
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    print(json.dumps(run(protocol), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
