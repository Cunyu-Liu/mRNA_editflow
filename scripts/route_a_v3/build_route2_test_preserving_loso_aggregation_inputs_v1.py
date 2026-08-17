#!/usr/bin/env python3
"""Build aligned mRNABERT/global-scaled LOSO aggregation inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


HOLDOUT_STUDIES = (
    "GSE200304",
    "GSE114002",
    "GSE149487",
    "GSE217518",
    "ENCSR854RUF",
    "GSE186455",
    "GSE269595",
)
SEED_GPU_PAIRS = ((20260822, 0), (20260823, 3), (20260824, 5))
ZERO_RECORD_STUDIES = ("GSE256185",)


class LosoInputError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LosoInputError(message)


def _read_summary(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"training summary is absent: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"training summary root differs: {path}")
    _require(value.get("development_test_outcomes_evaluated") is False, "Development TEST entered LOSO")
    _require(value.get("test_metrics") is None, "Development TEST metrics entered LOSO")
    _require(value.get("evaluation_outcomes_read") == 0, "Evaluation entered LOSO")
    _require(isinstance(value.get("validation_metrics"), Mapping), "LOSO validation metrics are absent")
    return value


def build_inputs(
    model_run_root: Path,
    baseline_run_root: Path,
    *,
    loss_kind: str,
) -> dict[int, dict[str, Any]]:
    outputs = {}
    for seed, gpu in SEED_GPU_PAIRS:
        model_results = []
        baseline_results = []
        for study in HOLDOUT_STUDIES:
            model_summary = _read_summary(
                model_run_root / study / f"seed{seed}_gpu{gpu}_{loss_kind}_v1/training_summary.json"
            )
            baseline_summary = _read_summary(
                baseline_run_root / study / f"seed{seed}_gpu{gpu}_global_scaled_v1/training_summary.json"
            )
            _require(model_summary.get("seed") == baseline_summary.get("seed") == seed, "LOSO seed differs")
            _require(
                model_summary.get("loso_holdout_study_unit_id")
                == baseline_summary.get("loso_holdout_study_unit_id")
                == study,
                "LOSO holdout differs",
            )
            model_results.append({
                "study_unit_id": study,
                "training_summary": model_summary,
                "evaluation": {
                    "split": f"LOSO::{study}",
                    "metrics": model_summary["validation_metrics"],
                },
            })
            baseline_results.append({
                "study_unit_id": study,
                "training_summary": baseline_summary,
                "evaluation": {
                    "split": f"LOSO::{study}",
                    "metrics": baseline_summary["validation_metrics"],
                },
            })
        outputs[seed] = {
            "schema_version": "route_a_v3_route2_loso_aggregation_input.v1",
            "seed": seed,
            "development_inventory_studies": [*HOLDOUT_STUDIES, *ZERO_RECORD_STUDIES],
            "expected_loso_studies": list(HOLDOUT_STUDIES),
            "zero_record_development_studies": list(ZERO_RECORD_STUDIES),
            "model_results": model_results,
            "baseline_results": baseline_results,
            "development_test_opened": False,
            "evaluation_opened": False,
        }
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-run-root", type=Path, required=True)
    parser.add_argument("--baseline-run-root", type=Path, required=True)
    parser.add_argument("--loss-kind", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    _require(not args.output_dir.exists(), f"LOSO input output directory already exists: {args.output_dir}")
    payloads = build_inputs(
        args.model_run_root,
        args.baseline_run_root,
        loss_kind=args.loss_kind,
    )
    args.output_dir.mkdir(parents=True)
    paths = []
    for seed, payload in sorted(payloads.items()):
        path = args.output_dir / f"test_preserving_loso_aggregation_input_seed{seed}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        paths.append(str(path))
    print(json.dumps({
        "status": "THREE_TEST_PRESERVING_LOSO_AGGREGATION_INPUTS_BUILT",
        "paths": paths,
        "development_test_opened": False,
        "evaluation_opened": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
