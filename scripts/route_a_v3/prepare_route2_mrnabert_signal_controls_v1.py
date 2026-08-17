#!/usr/bin/env python3
"""Prepare matched mRNABERT candidate-permutation and source-only controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


PRIMARY_MODEL_KIND = "delta_pretrained_mrnabert_edit_centered_antisymmetric"
SOURCE_ONLY_MODEL_KIND = "delta_pretrained_mrnabert_edit_centered_source_only_control"
PERMUTATION_CONTROL = "WITHIN_EXACT_SOURCE_TASK_TRAIN_CANDIDATE_PERMUTATION"


class ControlPreparationError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ControlPreparationError(message)


def _control_config(
    selected: Mapping[str, Any],
    *,
    role: str,
    model_kind: str,
    candidate_control: str,
    gpu: int,
    output_directory: Path,
) -> dict[str, Any]:
    result = dict(selected)
    result.update({
        "scientific_role": role,
        "result_stage": "HPO_VALIDATION_ONLY",
        "device": f"cuda:{gpu}",
        "physical_gpu_index": gpu,
        "model_kind": model_kind,
        "candidate_control": candidate_control,
        "baseline_id": f"{selected['baseline_id']}__{role.lower()}",
        "attempt_purpose": f"MRNABERT_SELECTED_LOSS_{role}",
        "output_directory": str(output_directory),
    })
    return result


def prepare(
    selected_config: Mapping[str, Any],
    *,
    candidate_permutation_gpu: int,
    source_only_gpu: int,
    candidate_permutation_run_dir: Path,
    source_only_run_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    _require(
        selected_config.get("model_kind") == PRIMARY_MODEL_KIND,
        "selected config is not the primary pretrained edit-centered critic",
    )
    _require(
        selected_config.get("result_stage") == "HPO_VALIDATION_ONLY",
        "controls must match Development VALIDATION selection",
    )
    _require(selected_config.get("candidate_control") == "NONE", "selected config is already a control")
    _require(selected_config.get("evaluation_outcomes_accessed") is False, "Evaluation entered selected config")
    _require(selected_config.get("development_test_outcomes_accessed") is False, "Development TEST entered selected config")
    for gpu in (candidate_permutation_gpu, source_only_gpu):
        _require(0 <= gpu <= 5, "controls must use a physical A100 GPU 0-5")
    permutation = _control_config(
        selected_config,
        role="MATCHED_TRAIN_CANDIDATE_PERMUTATION_CONTROL",
        model_kind=PRIMARY_MODEL_KIND,
        candidate_control=PERMUTATION_CONTROL,
        gpu=candidate_permutation_gpu,
        output_directory=candidate_permutation_run_dir,
    )
    source_only = _control_config(
        selected_config,
        role="PARAMETER_MATCHED_PRETRAINED_SOURCE_ONLY_CONTROL",
        model_kind=SOURCE_ONLY_MODEL_KIND,
        candidate_control="NONE",
        gpu=source_only_gpu,
        output_directory=source_only_run_dir,
    )
    return permutation, source_only


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-config", type=Path, required=True)
    parser.add_argument("--candidate-permutation-config", type=Path, required=True)
    parser.add_argument("--source-only-config", type=Path, required=True)
    parser.add_argument("--candidate-permutation-run-dir", type=Path, required=True)
    parser.add_argument("--source-only-run-dir", type=Path, required=True)
    parser.add_argument("--candidate-permutation-gpu", type=int, required=True)
    parser.add_argument("--source-only-gpu", type=int, required=True)
    args = parser.parse_args()
    for path in (
        args.candidate_permutation_config,
        args.source_only_config,
        args.candidate_permutation_run_dir,
        args.source_only_run_dir,
    ):
        _require(not path.exists(), f"control target already exists: {path}")
    selected = json.loads(args.selected_config.read_text(encoding="utf-8"))
    permutation, source_only = prepare(
        selected,
        candidate_permutation_gpu=args.candidate_permutation_gpu,
        source_only_gpu=args.source_only_gpu,
        candidate_permutation_run_dir=args.candidate_permutation_run_dir,
        source_only_run_dir=args.source_only_run_dir,
    )
    for path, payload in (
        (args.candidate_permutation_config, permutation),
        (args.source_only_config, source_only),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "candidate_permutation_config": str(args.candidate_permutation_config),
        "source_only_config": str(args.source_only_config),
        "loss_kind": selected["loss_kind"],
        "seed": selected["seed"],
        "development_test_opened": False,
        "evaluation_opened": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
