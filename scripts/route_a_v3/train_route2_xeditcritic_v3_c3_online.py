#!/usr/bin/env python3
"""Train one C3 last-four-LoRA Critic V3 arm with effective batch size 32."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.route2_development_projection_v3 import load_projection_rows
from core.route2_edit_site_token_cache_v3 import load_edit_site_token_cache_v3
from core.route2_xeditcritic_training_data_v3 import (
    SqrtTaskStudySourcePassSamplerV3,
    build_exact_source_task_candidate_bundle_permutation,
    build_vocabs,
    different_source_group_pair_indices,
    records_from_projection_rows,
)
from core.route2_xeditcritic_v3 import XEditCriticV3
from scripts.route_a_v3.route2_mrnabert_lora_edit_site_encoder_v3 import (
    TrainableMRNABERTEditSiteEncoderV3,
)
from scripts.route_a_v3.train_route2_xeditcritic_v3 import (
    EditSiteCacheViewV3,
    XEditCriticCollatorV3,
    XEditCriticDatasetV3,
    _move,
    _require,
    _set_seed,
    fit_task_robust_scaler,
    require_cuda,
    validation_metrics,
)


def select_batch_rows(batch: Mapping[str, Any], indices: Sequence[int]) -> dict[str, Any]:
    selected: dict[str, Any] = {}
    tensor_index = torch.tensor(indices, dtype=torch.long)
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            selected[key] = value[tensor_index]
        elif isinstance(value, list):
            selected[key] = [value[index] for index in indices]
        else:
            selected[key] = value
    return selected


def microbatch_indices(batch_size: int, physical_microbatch_size: int) -> list[list[int]]:
    _require(batch_size > 0 and physical_microbatch_size > 0, "microbatch geometry is invalid")
    return [
        list(range(start, min(batch_size, start + physical_microbatch_size)))
        for start in range(0, batch_size, physical_microbatch_size)
    ]


def online_evaluate(
    model: XEditCriticV3,
    encoder: TrainableMRNABERTEditSiteEncoderV3,
    loader: DataLoader,
    device: torch.device,
    *,
    physical_microbatch_size: int,
    prediction_output_path: Path | None = None,
) -> dict[str, Any]:
    targets: list[float] = []
    predictions: list[float] = []
    scaled_targets: list[float] = []
    scaled_predictions: list[float] = []
    tasks: list[str] = []
    record_ids: list[str] = []
    source_groups: list[str] = []
    model.eval()
    encoder.eval()
    with torch.inference_mode():
        for raw_batch in loader:
            for indices in microbatch_indices(
                len(raw_batch["record_ids"]), physical_microbatch_size
            ):
                batch = _move(select_batch_rows(raw_batch, indices), device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    feature_batch = encoder.forward_cache_anchored(batch)
                    output = model(feature_batch)
                scaled_prediction = output["mean"].float()
                prediction = scaled_prediction * batch["target_scale"]
                targets.extend(batch["target"].float().cpu().tolist())
                predictions.extend(prediction.cpu().tolist())
                scaled_targets.extend(batch["scaled_target"].float().cpu().tolist())
                scaled_predictions.extend(scaled_prediction.cpu().tolist())
                tasks.extend(batch["task_ids"])
                record_ids.extend(batch["record_ids"])
                source_groups.extend(batch["source_groups"])
    metrics = validation_metrics(
        targets, predictions, scaled_targets, scaled_predictions, tasks
    )
    if prediction_output_path is not None:
        _require(not prediction_output_path.exists(), "Validation prediction artifact already exists")
        with prediction_output_path.open("w", encoding="utf-8") as handle:
            for values in zip(
                record_ids,
                source_groups,
                tasks,
                targets,
                predictions,
                scaled_targets,
                scaled_predictions,
            ):
                record_id, source_group, task, target, prediction, scaled_target, scaled_prediction = values
                handle.write(
                    json.dumps(
                        {
                            "record_id": record_id,
                            "source_group_id": source_group,
                            "task_id": task,
                            "target": target,
                            "prediction": prediction,
                            "scaled_target": scaled_target,
                            "scaled_prediction": scaled_prediction,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
    return metrics


def run(
    config: Mapping[str, Any],
    *,
    control_mode: str,
    candidate_bundle_permutation: bool,
    physical_gpu_index: int,
) -> dict[str, Any]:
    _require(not (control_mode != "NONE" and candidate_bundle_permutation), "candidate controls cannot be combined")
    seed = int(config["screen_seed"])
    _set_seed(seed)
    device = require_cuda(physical_gpu_index)
    run_id = "c3"
    if control_mode != "NONE":
        run_id += "_" + control_mode.lower()
    if candidate_bundle_permutation:
        run_id += "_candidate_bundle_permutation"
    output_directory = Path(config["output_root"]) / run_id
    _require(not output_directory.exists(), f"terminal run directory already exists: {output_directory}")
    output_directory.mkdir(parents=True)
    started = time.time()
    try:
        records = records_from_projection_rows(
            load_projection_rows([Path(path) for path in config["projection_paths"]])
        )
        _require(len(records) == int(config["expected_record_count"]), "projection count changed")
        train_records = [record for record in records if record.split == "TRAIN"]
        validation_records = [record for record in records if record.split == "VALIDATION"]
        record_by_id = {record.record_id: record for record in records}
        vocabs = build_vocabs(records)
        scaler = fit_task_robust_scaler(
            train_records, floor=float(config["target_scale_floor"])
        )
        cache = EditSiteCacheViewV3(
            load_edit_site_token_cache_v3(Path(config["edit_site_cache"])),
            set(record_by_id),
        )
        if candidate_bundle_permutation:
            overrides, permutation_summary = build_exact_source_task_candidate_bundle_permutation(
                train_records, seed=seed
            )
        else:
            overrides, permutation_summary = {}, {
                "complete_candidate_bundle_permuted": False,
                "recipient_count": 0,
                "eligible_task_count": 0,
            }
        train_dataset = XEditCriticDatasetV3(
            train_records,
            all_records=record_by_id,
            vocabs=vocabs,
            target_scaler=scaler,
            cache=cache,
            candidate_bundle_overrides=overrides,
        )
        validation_dataset = XEditCriticDatasetV3(
            validation_records,
            all_records=record_by_id,
            vocabs=vocabs,
            target_scaler=scaler,
            cache=cache,
        )
        collator = XEditCriticCollatorV3(
            pretrained_width=int(config["pretrained_width"])
        )
        sampler = SqrtTaskStudySourcePassSamplerV3(
            train_records,
            batch_size=int(config["batch_size"]),
            seed=seed,
            repeat_cap=int(config["maximum_record_repeats_per_pass"]),
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=int(config["batch_size"]),
            shuffle=False,
            collate_fn=collator,
            num_workers=0,
            pin_memory=True,
        )
        model = XEditCriticV3(
            arm="C3",
            control_mode=control_mode,
            study_count=len(vocabs["study"]),
            assay_count=len(vocabs["assay"]),
            context_count=len(vocabs["context"]),
            quantity_count=len(vocabs["quantity"]),
            measurement_count=len(vocabs["measurement"]),
            numerator_count=len(vocabs["numerator"]),
            denominator_count=len(vocabs["denominator"]),
            pretrained_width=int(config["pretrained_width"]),
            dropout=float(config["dropout"]),
        ).to(device)
        encoder = TrainableMRNABERTEditSiteEncoderV3(
            Path(config["mrnabert_model_path"]),
            device,
            rank=int(config["lora_rank"]),
            alpha=float(config["lora_alpha"]),
            dropout=float(config["lora_dropout"]),
        )
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": [parameter for parameter in model.parameters() if parameter.requires_grad],
                    "lr": float(config["head_learning_rate"]),
                },
                {
                    "params": encoder.lora_parameters(),
                    "lr": float(config["lora_learning_rate"]),
                },
            ],
            weight_decay=float(config["weight_decay"]),
        )
        physical_microbatch_size = int(config["c3_physical_microbatch_records"])
        pass_rows = []
        update_count = 0
        torch.cuda.reset_peak_memory_stats(device)
        for pass_index in range(int(config["passes"])):
            sampler.set_pass(pass_index)
            train_loader = DataLoader(
                train_dataset,
                batch_sampler=sampler,
                collate_fn=collator,
                num_workers=0,
                pin_memory=True,
            )
            model.train()
            encoder.train()
            losses = []
            regression_losses = []
            ranking_losses = []
            for raw_batch in train_loader:
                optimizer.zero_grad(set_to_none=True)
                batch_weight = float(raw_batch["sample_weight"].sum().item())
                _require(batch_weight > 0.0, "effective batch has zero weight")
                regression_value = 0.0
                for indices in microbatch_indices(
                    len(raw_batch["record_ids"]), physical_microbatch_size
                ):
                    batch = _move(select_batch_rows(raw_batch, indices), device)
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        output = model(encoder.forward_cache_anchored(batch))
                        per_record = F.huber_loss(
                            output["mean"],
                            batch["scaled_target"],
                            reduction="none",
                            delta=float(config["huber_delta"]),
                        )
                        regression = (
                            per_record * batch["sample_weight"]
                        ).sum() / batch_weight
                    _require(torch.isfinite(regression).item(), "C3 regression loss is nonfinite")
                    regression.backward()
                    regression_value += float(regression.detach().cpu())

                ranking_value = None
                if pass_index == int(config["passes"]) - 1:
                    pairs = different_source_group_pair_indices(
                        raw_batch["scaled_target"],
                        raw_batch["source_groups"],
                        raw_batch["task_ids"],
                    )
                    if pairs:
                        pair_losses = []
                        pairs_per_microbatch = max(1, physical_microbatch_size // 2)
                        for start in range(0, len(pairs), pairs_per_microbatch):
                            selected_pairs = pairs[start : start + pairs_per_microbatch]
                            indices = [index for pair in selected_pairs for index in pair]
                            pair_batch = _move(select_batch_rows(raw_batch, indices), device)
                            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                                pair_output = model(
                                    encoder.forward_cache_anchored(pair_batch)
                                )["mean"]
                                pair_target = pair_batch["scaled_target"]
                                target_delta = pair_target[0::2] - pair_target[1::2]
                                prediction_delta = pair_output[0::2] - pair_output[1::2]
                                pair_sum = F.softplus(
                                    -target_delta.sign() * prediction_delta
                                ).sum()
                                weighted_pair_loss = (
                                    float(config["ranking_loss_weight"])
                                    * pair_sum
                                    / len(pairs)
                                )
                            _require(torch.isfinite(weighted_pair_loss).item(), "C3 ranking loss is nonfinite")
                            weighted_pair_loss.backward()
                            pair_losses.append(float(pair_sum.detach().cpu()))
                        ranking_value = sum(pair_losses) / len(pairs)
                trainable_parameters = [
                    parameter
                    for group in optimizer.param_groups
                    for parameter in group["params"]
                ]
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    trainable_parameters, float(config["gradient_clip_norm"])
                )
                _require(torch.isfinite(gradient_norm).item(), "C3 gradient norm is nonfinite")
                optimizer.step()
                update_count += 1
                total_loss = regression_value + (
                    0.0
                    if ranking_value is None
                    else float(config["ranking_loss_weight"]) * ranking_value
                )
                losses.append(total_loss)
                regression_losses.append(regression_value)
                if ranking_value is not None:
                    ranking_losses.append(ranking_value)

            metrics = online_evaluate(
                model,
                encoder,
                validation_loader,
                device,
                physical_microbatch_size=physical_microbatch_size,
                prediction_output_path=(
                    output_directory / "final_validation_predictions.jsonl"
                    if pass_index == int(config["passes"]) - 1
                    else None
                ),
            )
            pass_row = {
                "pass": pass_index + 1,
                "update_count_cumulative": update_count,
                "mean_loss": float(np.mean(losses)),
                "mean_regression_loss": float(np.mean(regression_losses)),
                "mean_ranking_loss": None if not ranking_losses else float(np.mean(ranking_losses)),
                "validation": metrics,
            }
            pass_rows.append(pass_row)
            print(
                json.dumps(
                    {"event": "XEDITCRITIC_V3_C3_PASS_COMPLETE", "run_id": run_id, **pass_row},
                    sort_keys=True,
                ),
                flush=True,
            )

        lora_state = {
            name: parameter.detach().cpu()
            for name, parameter in encoder.named_parameters()
            if parameter.requires_grad
        }
        checkpoint_path = output_directory / "final_pass_checkpoint.pt"
        torch.save(
            {
                "schema_version": "route_a_v3_route2_xeditcritic_v3_c3_checkpoint.v1",
                "arm": "C3",
                "control_mode": control_mode,
                "candidate_bundle_permutation": candidate_bundle_permutation,
                "seed": seed,
                "selected_pass": int(config["passes"]),
                "selection_policy": "FINAL_PASS_FIXED_NO_RANKING_PHASE_RESELECTION",
                "model_state_dict": model.state_dict(),
                "lora_state_dict": lora_state,
                "lora_installation": encoder.lora_installation.__dict__,
                "vocabs": vocabs,
                "target_scaler": scaler.to_dict(),
                "validation_metrics": pass_rows[-1]["validation"],
            },
            checkpoint_path,
        )
        summary = {
            "schema_version": "route_a_v3_route2_xeditcritic_v3_c3_screen_run.v1",
            "status": "TERMINAL_SCREEN_ARM_COMPLETE",
            "run_id": run_id,
            "arm": "C3",
            "control_mode": control_mode,
            "candidate_bundle_permutation": candidate_bundle_permutation,
            "candidate_permutation_summary": permutation_summary,
            "seed": seed,
            "physical_gpu_index": physical_gpu_index,
            "cuda_device_name": torch.cuda.get_device_name(device),
            "precision": "BF16",
            "effective_batch_size": int(config["batch_size"]),
            "physical_microbatch_records": physical_microbatch_size,
            "head_trainable_parameter_count": model.trainable_parameter_count,
            "lora_trainable_parameter_count": encoder.trainable_parameter_count,
            "total_trainable_parameter_count": model.trainable_parameter_count + encoder.trainable_parameter_count,
            "pass_count": len(pass_rows),
            "update_count": update_count,
            "peak_vram_bytes": torch.cuda.max_memory_allocated(device),
            "passes": pass_rows,
            "final_validation": pass_rows[-1]["validation"],
            "checkpoint_path": str(checkpoint_path),
            "validation_prediction_path": str(
                output_directory / "final_validation_predictions.jsonl"
            ),
            "elapsed_seconds": time.time() - started,
            "development_test_outcomes_accessed": False,
            "new_final_evaluation_outcomes_accessed": False,
        }
        (output_directory / "run_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return summary
    except Exception as exc:
        failure = {
            "schema_version": "route_a_v3_route2_xeditcritic_v3_c3_screen_failure.v1",
            "status": "TERMINAL_IMPLEMENTATION_OR_RUNTIME_FAILURE",
            "arm": "C3",
            "control_mode": control_mode,
            "candidate_bundle_permutation": candidate_bundle_permutation,
            "seed": seed,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "elapsed_seconds": time.time() - started,
            "development_test_outcomes_accessed": False,
            "new_final_evaluation_outcomes_accessed": False,
        }
        (output_directory / "failure.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--control-mode",
        default="NONE",
        choices=("NONE", "SOURCE_ONLY", "EDIT_METADATA_ONLY", "NO_CANDIDATE_SEQUENCE"),
    )
    parser.add_argument("--candidate-bundle-permutation", action="store_true")
    parser.add_argument("--physical-gpu-index", required=True, type=int)
    arguments = parser.parse_args()
    config = json.loads(arguments.config.read_text(encoding="utf-8"))
    print(
        json.dumps(
            run(
                config,
                control_mode=arguments.control_mode,
                candidate_bundle_permutation=arguments.candidate_bundle_permutation,
                physical_gpu_index=arguments.physical_gpu_index,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
