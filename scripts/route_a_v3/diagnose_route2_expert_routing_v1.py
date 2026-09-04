#!/usr/bin/env python3
"""Task 2.6 (SPECS_CRITIC_V6): expert routing utilization diagnosis.

Loads the Critic V6 first-training terminal checkpoint (FINAL_PASS_8_FIXED,
runner_7815fdeb/v6_full), rebuilds the exact VALIDATION data pipeline, runs the
full model forward on CUDA (BF16 autocast, inference only), and reports the
endpoint-semantic MoE router utilization of the terminal model:

- per-expert top-1 share, soft-weighted share, and edit-token-weighted share
- max share, effective expert count (share > 5%), routing entropies
- per-task and per-study routing distributions (do tasks occupy different
  expert subspaces, or do they all pile onto the same experts?)

Routing caliber note: the V4/V6 endpoint-semantic router routes per SAMPLE
(one route-weight vector from the endpoint-condition metadata), and the
selected experts are broadcast to every edit token of that sample.  Sample
share is therefore the primary unit; the edit-token-weighted share is reported
as a secondary view.  The route weights are read from the model forward
output (the same tensor the training loss consumed), not re-implemented.

Reads: VALIDATION split only. TEST/Eval outcomes are never touched.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

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
)
from scripts.route_a_v3.train_route2_xeditcritic_v4 import (  # noqa: E402
    evaluation_index_batches_v4,
)

CONFIG_PATH = (
    REPO_ROOT
    / "configs/route_a_v3_route2_xeditcritic_v6_screen_v1.json"
)
CHECKPOINT_PATH = Path(
    "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/experiments/xeditcritic_v6/"
    "v6_screen_seed_20260907_runner_7815fdeb/v6_full/final_pass_8_checkpoint.pt"
)
DEFAULT_OUTPUT_DIR = Path(
    "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/experiments/"
    "analysis_v6_expert_routing_20260904"
)


def _move(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def _load_validation_pipeline(
    config: Mapping[str, Any],
) -> tuple[XEditCriticDatasetV4, XEditCriticCollatorV4, dict[str, Any]]:
    """Rebuild the exact training-time VALIDATION pipeline (no TEST reads)."""
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
    model = XEditCriticV4(
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
    return model


def _entropy(shares: np.ndarray) -> float:
    positive = shares[shares > 0.0]
    return float(-(positive * np.log(positive)).sum())


def _route_statistics(
    weights: np.ndarray, edit_counts: np.ndarray, *, expert_count: int
) -> dict[str, Any]:
    sample_count = int(weights.shape[0])
    top1 = weights.argmax(axis=1)
    top1_counts = np.bincount(top1, minlength=expert_count)
    top1_share = top1_counts / max(1, sample_count)
    weighted_share = weights.mean(axis=0)
    token_total = float(edit_counts.sum())
    token_share = np.zeros(expert_count, dtype=np.float64)
    if token_total > 0:
        for expert in range(expert_count):
            mask = top1 == expert
            token_share[expert] = float(edit_counts[mask].sum()) / token_total
    rounded = np.round(weights, 6)
    distinct_patterns = {
        tuple(row.tolist()) for row in rounded
    }
    pair_counts = Counter(
        tuple(sorted(np.argsort(-row)[:2].tolist())) for row in weights
    )
    return {
        "sample_count": sample_count,
        "edit_token_count": int(token_total),
        "top1_share": {str(e): float(top1_share[e]) for e in range(expert_count)},
        "weighted_share": {str(e): float(weighted_share[e]) for e in range(expert_count)},
        "edit_token_weighted_share": {
            str(e): float(token_share[e]) for e in range(expert_count)
        },
        "max_top1_share": float(top1_share.max()),
        "effective_expert_count_share_gt_5pct": int((top1_share > 0.05).sum()),
        "entropy_top1_nats": _entropy(top1_share),
        "entropy_weighted_nats": _entropy(weighted_share),
        "mean_top1_gate_weight": float(weights.max(axis=1).mean()) if sample_count else None,
        "distinct_route_pattern_count": len(distinct_patterns),
        "top2_pair_frequencies": {
            f"{pair[0]}+{pair[1]}": int(count)
            for pair, count in sorted(pair_counts.items(), key=lambda kv: -kv[1])
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=6)
    parser.add_argument("--config", type=str, default=str(CONFIG_PATH))
    parser.add_argument("--checkpoint", type=str, default=str(CHECKPOINT_PATH))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    expert_count = int(config["architecture"]["semantic_expert_count"])
    top_k = int(config["architecture"]["semantic_router_top_k"])

    device = require_cuda(args.gpu)
    started = time.time()

    print("[1/4] building validation pipeline (projection + bottom-six cache)...", flush=True)
    dataset, collator, vocabs = _load_validation_pipeline(config)
    id_to_study = {index: name for name, index in vocabs["study"].items()}
    physical_batch_size = 32

    print("[2/4] building model and loading terminal checkpoint...", flush=True)
    model = _build_model(config, vocabs, device)
    payload = torch.load(Path(args.checkpoint), map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    print(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "selection_policy": payload.get("selection_policy"),
                "selected_pass": payload.get("selected_pass"),
            }
        ),
        flush=True,
    )

    print("[3/4] full VALIDATION forward (CUDA BF16, inference only)...", flush=True)
    weight_rows: list[list[float]] = []
    edit_count_rows: list[int] = []
    task_rows: list[str] = []
    study_rows: list[str] = []
    batch_balance_losses: list[float] = []
    with torch.inference_mode():
        for batch_index, (indices, valid_count) in enumerate(
            evaluation_index_batches_v4(len(dataset), physical_batch_size)
        ):
            batch = _move(collator([dataset[i] for i in indices]), device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(batch)
            weights = output["route_weights"].float()[:valid_count]
            weight_rows.extend(weights.cpu().tolist())
            edit_count_rows.extend(
                ((~batch["edit_padding_mask"]).sum(dim=1)[:valid_count])
                .int()
                .cpu()
                .tolist()
            )
            task_rows.extend(batch["task_ids"][:valid_count])
            study_rows.extend(
                id_to_study[int(value)] for value in batch["study_ids"][:valid_count].tolist()
            )
            batch_balance_losses.append(float(output["router_balance_loss"]))
            if (batch_index + 1) % 100 == 0:
                elapsed = time.time() - started
                print(
                    f"  batch {batch_index + 1}: rows={len(weight_rows)} elapsed={elapsed:.0f}s",
                    flush=True,
                )
    assert len(weight_rows) == len(dataset), "padded rows entered the routing cohort"

    weights = np.asarray(weight_rows, dtype=np.float64)
    edit_counts = np.asarray(edit_count_rows, dtype=np.int64)
    tasks = np.asarray(task_rows)
    studies = np.asarray(study_rows)

    print("[4/4] computing routing statistics...", flush=True)
    overall = _route_statistics(weights, edit_counts, expert_count=expert_count)
    per_task: dict[str, Any] = {}
    for task in sorted(set(task_rows)):
        mask = tasks == task
        per_task[task] = _route_statistics(
            weights[mask], edit_counts[mask], expert_count=expert_count
        )
    per_study: dict[str, Any] = {}
    for study in sorted(set(study_rows)):
        mask = studies == study
        per_study[study] = _route_statistics(
            weights[mask], edit_counts[mask], expert_count=expert_count
        )
    task_expert_matrix = {
        task: per_task[task]["top1_share"] for task in sorted(per_task)
    }

    result = {
        "schema_version": "route_a_v3_v6_expert_routing_diagnosis.v1",
        "task": "SPECS_CRITIC_V6 Task 2.6 expert routing utilization diagnosis",
        "checkpoint_path": str(args.checkpoint),
        "config_path": str(args.config),
        "selection_policy": payload.get("selection_policy"),
        "selected_pass": payload.get("selected_pass"),
        "split": "VALIDATION_ONLY_PROTECTED_READS_ZERO",
        "router": {
            "expert_count": expert_count,
            "top_k": top_k,
            "routing_unit": "SAMPLE_ENDPOINT_CONDITION_BROADCAST_TO_EDIT_TOKENS",
            "router_parameter_names": ["router.router.weight", "router.router.bias"],
            "mean_router_balance_loss": float(np.mean(batch_balance_losses)),
        },
        "validation_record_count": int(len(dataset)),
        "covered_task_count": int(len(set(task_rows))),
        "covered_study_count": int(len(set(study_rows))),
        "overall": overall,
        "per_task": per_task,
        "per_study": per_study,
        "task_expert_top1_share_matrix": task_expert_matrix,
        "elapsed_seconds": time.time() - started,
    }
    output_path = output_dir / "expert_routing_results.json"
    output_path.write_text(
        json.dumps(result, indent=1, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(result["overall"], indent=1, sort_keys=True), flush=True)
    print(f"written: {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
