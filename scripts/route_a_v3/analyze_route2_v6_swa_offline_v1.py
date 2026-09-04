#!/usr/bin/env python3
"""Task 3.5 (SPECS_CRITIC_V6): offline SWA analysis for the V6 first training.

Averages the pass_6 / pass_7 / pass_8 checkpoints of the v6_full run
(runner_7815fdeb) per state-dict key (SWA arithmetic mean), saves the averaged
checkpoint into the ANALYSIS directory (never into the run family directory),
then runs a full VALIDATION inference pass on CUDA (BF16) with the exact
training-time data pipeline.

Reported metrics (identical calibers to the V6 adjudication):
- task-macro + per-task Spearman / standardized MAE (validation_metrics)
- MPRAU variant pair-mean rho, the preregistered criterion: variants = record
  id before ':context:' over VALIDATION manifest ids of ENCSR854RUF, >=2
  contexts, per-variant means of target and prediction, Spearman over
  variants (identical to analysis_w_ladder_adjudication_20260903 /
  adjudicate_v7_mprau_vs_v5_v1.py).  The run_summary extended
  pair_mean_spearman is ALL-TASK POOLED and is NOT used here.
- comparison vs FINAL_PASS_8_FIXED with paired bootstrap over variants
  (2,000 iters, seed 20260816).

Reads: VALIDATION split only. TEST/Eval outcomes are never touched.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.route2_bottom_encoder_chunk_cache_v4 import (  # noqa: E402
    load_frozen_bottom_encoder_chunk_cache_v4,
)
from core.route2_development_projection_v3 import load_projection_rows  # noqa: E402
from core.route2_xeditcritic_batch_v4 import (  # noqa: E402
    FrozenBottomEncoderChunkCacheViewV4,
    XEditCriticCollatorV4,
    XEditCriticDatasetV4,
)
from core.route2_xeditcritic_pair_mean_v6 import (  # noqa: E402
    apply_pair_mean_targets_v6,
    apply_rank_gaussian_targets_v6,
)
from core.route2_xeditcritic_training_data_v3 import (  # noqa: E402
    build_vocabs,
    records_from_projection_rows,
)
from core.route2_xeditcritic_v4 import XEditCriticV4  # noqa: E402
from scripts.route_a_v3.route2_mrnabert_upper_six_encoder_v4 import (  # noqa: E402
    TrainableMRNABERTUpperSixEncoderV4,
)
from scripts.route_a_v3.train_route2_xeditcritic_v3 import (  # noqa: E402
    fit_task_robust_scaler,
    require_cuda,
    validation_metrics,
)
from scripts.route_a_v3.train_route2_xeditcritic_v4 import (  # noqa: E402
    evaluation_index_batches_v4,
)

MNT = "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2"
FAMILY_DIR = Path(
    f"{MNT}/experiments/xeditcritic_v6/v6_screen_seed_20260907_runner_7815fdeb/v6_full"
)
CONFIG_PATH = REPO_ROOT / "configs/route_a_v3_route2_xeditcritic_v6_screen_v1.json"
MANIFEST = Path(f"{MNT}/manifests/route2_development_frozen_v1/development_manifest.jsonl")
DEFAULT_OUTPUT_DIR = Path(f"{MNT}/experiments/analysis_v6_swa_offline_20260904")
SWA_PASSES = (6, 7, 8)
BOOT_ITERS = 2000
BOOT_SEED = 20260816
MPRAU_CEILING = 0.683
MPRAU_STUDY = "ENCSR854RUF"
FOLLOWUP_DELTA_THRESHOLD = 0.02


def _move(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def manifest_validation_ids(study: str = MPRAU_STUDY) -> set[str]:
    ids: set[str] = set()
    with MANIFEST.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["split"] == "VALIDATION" and row["study_unit_id"] == study:
                ids.add(str(row["canonical_record_id"]))
    return ids


def mprau_variant_pair_mean(
    rows: Sequence[Mapping[str, Any]], manifest_ids: set[str]
) -> dict[str, Any]:
    """Preregistered MPRAU caliber (variant pair-mean rho, 2,008 variants)."""
    by_variant: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        rid = str(row["record_id"])
        if rid in manifest_ids and rid.startswith(f"{MPRAU_STUDY}:"):
            by_variant[rid.split(":context:")[0]].append(row)
    variants: dict[str, tuple[float, float]] = {}
    for variant, members in by_variant.items():
        if len(members) >= 2:
            variants[variant] = (
                float(np.mean([float(m["target"]) for m in members])),
                float(np.mean([float(m["prediction"]) for m in members])),
            )
    targets = [value[0] for value in variants.values()]
    predictions = [value[1] for value in variants.values()]
    rho = float(spearmanr(targets, predictions).statistic)
    return {
        "variant_count": len(variants),
        "cell_record_count": sum(len(m) for m in by_variant.values()),
        "pair_mean_spearman": rho,
        "ceiling": MPRAU_CEILING,
        "ceiling_ratio": rho / MPRAU_CEILING,
    }


def paired_bootstrap(
    reference_variants: dict[str, tuple[float, float]],
    arm_variants: dict[str, tuple[float, float]],
) -> dict[str, Any]:
    shared = sorted(set(reference_variants) & set(arm_variants))
    target = np.array([reference_variants[v][0] for v in shared])
    reference_prediction = np.array([reference_variants[v][1] for v in shared])
    arm_prediction = np.array([arm_variants[v][1] for v in shared])
    rho_reference = float(spearmanr(target, reference_prediction).statistic)
    rho_arm = float(spearmanr(target, arm_prediction).statistic)
    rng = np.random.default_rng(BOOT_SEED)
    count = len(shared)
    deltas: list[float] = []
    for _ in range(BOOT_ITERS):
        index = rng.integers(0, count, count)
        try:
            left = float(spearmanr(target[index], reference_prediction[index]).statistic)
            right = float(spearmanr(target[index], arm_prediction[index]).statistic)
            if np.isfinite(left) and np.isfinite(right):
                deltas.append(right - left)
        except Exception:
            continue
    deltas_array = np.asarray(deltas)
    ci = [
        float(np.percentile(deltas_array, 2.5)),
        float(np.percentile(deltas_array, 97.5)),
    ]
    return {
        "bootstrap_iterations": BOOT_ITERS,
        "bootstrap_seed": BOOT_SEED,
        "shared_variant_count": count,
        "reference_pair_mean_spearman": rho_reference,
        "arm_pair_mean_spearman": rho_arm,
        "delta_pair_mean_spearman": rho_arm - rho_reference,
        "bootstrap_ci_95": ci,
        "ci_excludes_zero": bool(ci[0] > 0 or ci[1] < 0),
    }


def read_prediction_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def metrics_from_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return validation_metrics(
        [float(row["target"]) for row in rows],
        [float(row["prediction"]) for row in rows],
        [float(row["scaled_target"]) for row in rows],
        [float(row["scaled_prediction"]) for row in rows],
        [str(row["task_id"]) for row in rows],
    )


def _load_validation_pipeline(
    config: Mapping[str, Any],
) -> tuple[XEditCriticDatasetV4, XEditCriticCollatorV4, dict[str, Any]]:
    projection_rows = load_projection_rows(
        [Path(path) for path in config["projection_paths"]]
    )
    all_records = records_from_projection_rows(projection_rows)
    geometry = config["data_geometry"]
    assert len(all_records) == int(geometry["expected_record_count"])
    train_records = [r for r in all_records if r.split == "TRAIN"]
    validation_records = [r for r in all_records if r.split == "VALIDATION"]
    assert len(train_records) == int(geometry["expected_train_count"])
    assert len(validation_records) == int(geometry["expected_validation_count"])
    if bool(config["training"].get("pair_mean_targets", False)):
        train_records, _ = apply_pair_mean_targets_v6(train_records, pair_tasks=None)
        validation_records, _ = apply_pair_mean_targets_v6(
            validation_records, pair_tasks=None
        )
    if bool(config["training"].get("per_task_rank_gaussian", False)):
        train_records, _ = apply_rank_gaussian_targets_v6(train_records, rank_tasks=None)
    vocabs = build_vocabs(all_records)
    scaler = fit_task_robust_scaler(
        train_records,
        floor=float(config["training"]["target_scale_floor"]),
    )
    record_by_id = {record.record_id: record for record in all_records}
    cache_payload = load_frozen_bottom_encoder_chunk_cache_v4(
        Path(config["bottom_six_cache"])
    )
    cache = FrozenBottomEncoderChunkCacheViewV4(
        cache_payload, set(record_by_id), validate_payload=False
    )
    dataset = XEditCriticDatasetV4(
        validation_records,
        all_records=record_by_id,
        vocabs=vocabs,
        target_scaler=scaler,
        cache=None,
    )
    collator = XEditCriticCollatorV4(
        cache,
        minimum_physical_batch=int(config["memory_preflight"]["minimum_physical_batch"]),
    )
    return dataset, collator, vocabs


def _build_model(
    config: Mapping[str, Any], vocabs: Mapping[str, Mapping[str, int]], device: torch.device
) -> XEditCriticV4:
    architecture = config["architecture"]
    upper = TrainableMRNABERTUpperSixEncoderV4(
        Path(config["mrnabert_model_path"]),
        device,
        attention_backend=str(config["memory_preflight"]["attention_backend"]),
        activation_checkpointing=bool(
            config["memory_preflight"]["activation_checkpointing"]
        ),
    )
    return XEditCriticV4(
        upper_encoder=upper,
        study_count=len(vocabs["study"]),
        assay_count=len(vocabs["assay"]),
        context_count=len(vocabs["context"]),
        quantity_count=len(vocabs["quantity"]),
        measurement_count=len(vocabs["measurement"]),
        numerator_count=len(vocabs["numerator"]),
        denominator_count=len(vocabs["denominator"]),
        region_count=2,
        control_mode="NONE",
        mechanism_mode="FULL",
        pretrained_width=int(architecture["pretrained_width"]),
        model_width=int(architecture["model_width"]),
        block_count=int(architecture["edit_block_count"]),
        heads=int(architecture["attention_heads"]),
        ffn_width=int(architecture["ffn_width"]),
        expert_count=int(architecture["semantic_expert_count"]),
        expert_bottleneck_width=int(architecture["semantic_expert_bottleneck_width"]),
        expert_top_k=int(architecture["semantic_router_top_k"]),
        raw_hidden_dim=int(architecture["raw_hidden_dim"]),
        raw_depth=int(architecture["raw_depth"]),
        readout_hidden_width=int(architecture["readout_hidden_width"]),
        dropout=float(architecture["dropout"]),
        minimum_physical_batch=int(config["memory_preflight"]["minimum_physical_batch"]),
        activation_checkpointing=bool(
            config["memory_preflight"]["activation_checkpointing"]
        ),
        cell_offset_head=bool(architecture.get("cell_offset_head", False)),
        cell_offset_hidden_width=int(architecture.get("cell_offset_hidden_width", 256)),
    ).to(device)


def average_state_dicts(
    payloads: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    key_sets = [set(p["model_state_dict"]) for p in payloads]
    assert all(keys == key_sets[0] for keys in key_sets), "pass checkpoints differ in keys"
    non_float_keys = [
        key
        for key in payloads[0]["model_state_dict"]
        if payloads[0]["model_state_dict"][key].dtype
        not in (torch.float32, torch.float16, torch.bfloat16)
    ]
    averaged: dict[str, torch.Tensor] = {}
    for key, first in payloads[0]["model_state_dict"].items():
        stack = torch.stack(
            [p["model_state_dict"][key].float() for p in payloads], dim=0
        )
        averaged[key] = stack.mean(dim=0).to(dtype=first.dtype)
    notes = {
        "averaging": "ARITHMETIC_MEAN_PER_KEY_OVER_SAME_FAMILY_PASS_CHECKPOINTS",
        "state_dict_entry_count": len(averaged),
        "non_float_entries": non_float_keys,
        "buffer_handling": (
            "no BN running-stats or integer buffers exist in the V4/V6 state dict "
            "(all entries are float32 parameters), so plain arithmetic mean applies"
        ),
        "dtypes": sorted(
            {str(value.dtype) for value in payloads[0]["model_state_dict"].values()}
        ),
    }
    return averaged, notes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=6)
    parser.add_argument("--config", type=str, default=str(CONFIG_PATH))
    parser.add_argument("--family-dir", type=str, default=str(FAMILY_DIR))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    family = Path(args.family_dir)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    started = time.time()

    pass_paths = [family / f"pass_{number}_checkpoint.pt" for number in SWA_PASSES]
    final_predictions_path = family / "final_validation_predictions.jsonl"
    for path in pass_paths + [final_predictions_path]:
        assert path.is_file(), f"per-pass artifact is absent: {path}"

    print("[1/6] loading pass checkpoints and averaging weights...", flush=True)
    payloads = [
        torch.load(path, map_location="cpu", weights_only=False) for path in pass_paths
    ]
    averaged_state, averaging_notes = average_state_dicts(payloads)
    swa_checkpoint_path = output_dir / "swa_pass678_checkpoint.pt"
    swa_payload = dict(payloads[-1])
    swa_payload.update(
        {
            "schema_version": "route_a_v3_route2_xeditcritic_v6_swa_offline_checkpoint.v1",
            "selection_policy": "SWA_OFFLINE_PASS_6_7_8_ARITHMETIC_MEAN",
            "swa_source_passes": list(SWA_PASSES),
            "swa_source_paths": [str(path) for path in pass_paths],
            "model_state_dict": averaged_state,
            "validation_metrics": None,
            "development_test_outcome_reads": 0,
            "new_final_evaluation_outcome_reads": 0,
        }
    )
    torch.save(swa_payload, swa_checkpoint_path)
    print(f"  swa checkpoint written: {swa_checkpoint_path}", flush=True)

    manifest_ids = manifest_validation_ids()
    print(f"  manifest VALIDATION ids for {MPRAU_STUDY}: {len(manifest_ids)}", flush=True)

    print("[2/6] reference metrics from FINAL_PASS_8 predictions...", flush=True)
    final_rows = read_prediction_rows(final_predictions_path)
    final_metrics = metrics_from_rows(final_rows)
    final_mprau = mprau_variant_pair_mean(final_rows, manifest_ids)
    final_variants = {
        key: value
        for key, value in _variant_table(final_rows, manifest_ids).items()
    }

    pass_trajectory: dict[str, Any] = {}
    for number in SWA_PASSES:
        path = family / f"pass_{number}_validation_predictions.jsonl"
        rows = read_prediction_rows(path)
        pass_trajectory[f"pass_{number}"] = mprau_variant_pair_mean(rows, manifest_ids)

    print("[3/6] building validation pipeline (projection + bottom-six cache)...", flush=True)
    dataset, collator, vocabs = _load_validation_pipeline(config)
    device = require_cuda(args.gpu)

    print("[4/6] building model and loading SWA weights...", flush=True)
    model = _build_model(config, vocabs, device)
    model.load_state_dict(averaged_state, strict=True)
    model.eval()

    print("[5/6] full VALIDATION inference with SWA weights (CUDA BF16)...", flush=True)
    swa_rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for batch_index, (indices, valid_count) in enumerate(
            evaluation_index_batches_v4(len(dataset), 32)
        ):
            batch = _move(collator([dataset[i] for i in indices]), device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(batch)
            scaled_prediction = output["mean"].float()[:valid_count]
            prediction = scaled_prediction * batch["target_scale"][:valid_count]
            for index in range(valid_count):
                swa_rows.append(
                    {
                        "record_id": batch["record_ids"][index],
                        "source_group_id": batch["source_groups"][index],
                        "task_id": batch["task_ids"][index],
                        "target": float(batch["target"][index]),
                        "prediction": float(prediction[index]),
                        "scaled_target": float(batch["scaled_target"][index]),
                        "scaled_prediction": float(scaled_prediction[index]),
                    }
                )
            if (batch_index + 1) % 100 == 0:
                print(
                    f"  batch {batch_index + 1}: rows={len(swa_rows)} "
                    f"elapsed={time.time() - started:.0f}s",
                    flush=True,
                )
    assert len(swa_rows) == len(dataset), "padded rows entered the SWA cohort"
    swa_predictions_path = output_dir / "swa_validation_predictions.jsonl"
    with swa_predictions_path.open("w", encoding="utf-8") as handle:
        for row in swa_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    print("[6/6] computing SWA metrics and comparison...", flush=True)
    swa_metrics = metrics_from_rows(swa_rows)
    swa_mprau = mprau_variant_pair_mean(swa_rows, manifest_ids)
    swa_variants = _variant_table(swa_rows, manifest_ids)
    bootstrap = paired_bootstrap(final_variants, swa_variants)

    delta_mprau = swa_mprau["pair_mean_spearman"] - final_mprau["pair_mean_spearman"]
    result = {
        "schema_version": "route_a_v3_v6_swa_offline_analysis.v1",
        "task": "SPECS_CRITIC_V6 Task 3.5 SWA offline post-hoc analysis",
        "swa_definition": {
            "passes": list(SWA_PASSES),
            "source_paths": [str(path) for path in pass_paths],
            **averaging_notes,
        },
        "swa_checkpoint_path": str(swa_checkpoint_path),
        "swa_validation_predictions_path": str(swa_predictions_path),
        "split": "VALIDATION_ONLY_PROTECTED_READS_ZERO",
        "final_reference": {
            "path": str(final_predictions_path),
            "selection_policy": "FINAL_PASS_8_FIXED_NO_VALIDATION_PEAK_RESELECTION",
            "task_macro_spearman": final_metrics["task_macro_spearman"],
            "task_macro_standardized_mae": final_metrics["task_macro_standardized_mae"],
            "per_task": final_metrics["tasks"],
            "mprau_variant_pair_mean": final_mprau,
        },
        "pass_trajectory_mprau_variant_pair_mean": pass_trajectory,
        "swa": {
            "task_macro_spearman": swa_metrics["task_macro_spearman"],
            "task_macro_standardized_mae": swa_metrics["task_macro_standardized_mae"],
            "per_task": swa_metrics["tasks"],
            "mprau_variant_pair_mean": swa_mprau,
        },
        "comparison": {
            "delta_task_macro_spearman": swa_metrics["task_macro_spearman"]
            - final_metrics["task_macro_spearman"],
            "delta_mprau_variant_pair_mean_spearman": delta_mprau,
            "paired_bootstrap_swa_minus_final": bootstrap,
            "followup_delta_threshold": FOLLOWUP_DELTA_THRESHOLD,
            "worth_followup": bool(delta_mprau >= FOLLOWUP_DELTA_THRESHOLD),
        },
        "caliber_warnings": [
            "MPRAU criterion = variant pair-mean rho (2,008 variants), identical to "
            "the W-ladder/V6-H3 adjudication; the run_summary extended "
            "pair_mean_spearman is ALL-TASK POOLED and is not used here.",
            "SWA is a post-hoc offline analysis outside the frozen V6 protocol; it "
            "never replaces the FINAL_PASS_8_FIXED terminal selection.",
        ],
        "elapsed_seconds": time.time() - started,
    }
    output_path = output_dir / "swa_offline_results.json"
    output_path.write_text(json.dumps(result, indent=1, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "final_mprau": final_mprau["pair_mean_spearman"],
                "swa_mprau": swa_mprau["pair_mean_spearman"],
                "delta": delta_mprau,
                "worth_followup": result["comparison"]["worth_followup"],
                "final_task_macro": final_metrics["task_macro_spearman"],
                "swa_task_macro": swa_metrics["task_macro_spearman"],
            },
            indent=1,
        ),
        flush=True,
    )
    print(f"written: {output_path}", flush=True)
    return 0


def _variant_table(
    rows: Sequence[Mapping[str, Any]], manifest_ids: set[str]
) -> dict[str, tuple[float, float]]:
    by_variant: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        rid = str(row["record_id"])
        if rid in manifest_ids and rid.startswith(f"{MPRAU_STUDY}:"):
            by_variant[rid.split(":context:")[0]].append(row)
    return {
        variant: (
            float(np.mean([float(m["target"]) for m in members])),
            float(np.mean([float(m["prediction"]) for m in members])),
        )
        for variant, members in by_variant.items()
        if len(members) >= 2
    }


if __name__ == "__main__":
    raise SystemExit(main())
