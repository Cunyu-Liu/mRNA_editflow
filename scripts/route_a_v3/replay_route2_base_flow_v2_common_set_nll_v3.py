#!/usr/bin/env python3
"""Replay frozen Base Flow V2 on SetFlow V3's common Validation objective."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.route2_base_flow_model import Route2BaseFlowModel
from core.route2_development_projection_v3 import load_projection_rows
from core.route2_source_token_cache_v3 import SourceTokenCacheIndexV3, load_source_token_cache_v3
from core.route2_xeditsetflow_f0_replay_v3 import validate_frozen_f0_for_common_replay_v3
from core.route2_xeditsetflow_training_v3 import SetMarginalStateDatasetV3, setflow_records_from_projection_rows, setflow_vocabs
from scripts.route_a_v3.train_route2_xeditsetflow_v3 import evaluate_common_validation_nll_v3


class F0CommonReplayError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise F0CommonReplayError(message)


def replay(config: Mapping[str, Any]) -> dict[str, Any]:
    output_path = Path(config["frozen_f0_common_nll_output_path"])
    _require(not output_path.exists(), f"terminal F0 common-NLL replay already exists: {output_path}")
    _require(torch.cuda.is_available(), "CUDA is unavailable for common-NLL replay")
    _require(not os.environ.get("CUDA_VISIBLE_DEVICES"), "CUDA_VISIBLE_DEVICES remapping is forbidden")
    device = torch.device(str(config["device"]))
    _require(device == torch.device(f"cuda:{int(config['physical_gpu_index'])}"), "F0 replay device provenance changed")
    torch.cuda.set_device(device)
    checkpoint = torch.load(Path(config["frozen_f0_checkpoint_path"]), map_location=device, weights_only=False)
    training_summary = json.loads(Path(config["frozen_f0_training_summary_path"]).read_text(encoding="utf-8"))
    frozen = validate_frozen_f0_for_common_replay_v3(checkpoint, training_summary)
    train_rows = load_projection_rows([Path(config["train_projection_path"])], allowed_splits=("TRAIN",))
    validation_rows = load_projection_rows([Path(config["validation_projection_path"])], allowed_splits=("VALIDATION",))
    train_records, _train = setflow_records_from_projection_rows(train_rows)
    validation_records, validation_eligibility = setflow_records_from_projection_rows(validation_rows)
    _require(len(train_records) == int(config["expected_train_record_count"]), "F0 replay TRAIN cohort changed")
    _require(len(validation_records) == int(config["expected_validation_record_count"]), "F0 replay Validation cohort changed")
    vocabs = setflow_vocabs(train_records)
    _require(checkpoint["assay_vocab"] == vocabs["assay"], "F0 assay vocabulary differs from projection")
    _require(checkpoint["context_vocab"] == vocabs["context"], "F0 context vocabulary differs from projection")
    model = Route2BaseFlowModel(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    cache = SourceTokenCacheIndexV3(load_source_token_cache_v3(Path(config["source_token_cache_path"])))
    dataset = SetMarginalStateDatasetV3(
        validation_records,
        vocabs,
        seed=int(config["common_validation_state_seed"]),
        states_per_record=int(config["states_per_record"]),
    )
    common_nll, active_count = evaluate_common_validation_nll_v3(
        model,
        "f1",
        dataset,
        source_cache=cache,
        batch_size=int(config["batch_size"]),
        device=device,
        bf16=True,
    )
    _require(
        all(torch.equal(before[name], value.detach()) for name, value in model.state_dict().items()),
        "F0 parameter changed during read-only replay",
    )
    result = {
        "schema_version": "route_a_v3_route2_f0_common_set_nll_replay.v3",
        "status": "FROZEN_BASE_FLOW_V2_COMMON_SET_NLL_REPLAY_COMPLETE",
        **frozen,
        "validation_record_count": len(validation_records),
        "validation_states_per_record": int(config["states_per_record"]),
        "validation_active_state_count": active_count,
        "validation_structural_budget_exhausted_state_count": len(validation_records) * int(config["states_per_record"]) - active_count,
        "common_validation_set_marginal_nll": common_nll,
        "common_validation_state_seed": int(config["common_validation_state_seed"]),
        "over_budget_excluded_validation_record_count": validation_eligibility["skipped_over_budget_count"],
        "parameter_update_count": 0,
        "parameter_changed_during_replay": False,
        "development_test_outcomes_accessed": False,
        "evaluation_records_read": 0,
        "evaluation_outcomes_accessed": False,
        "critic_score_used": False,
        "independent_evaluator_used": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_suffix(output_path.suffix + ".partial")
    _require(not partial.exists(), f"partial F0 replay output exists: {partial}")
    partial.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(partial, output_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    print(json.dumps(replay(config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
