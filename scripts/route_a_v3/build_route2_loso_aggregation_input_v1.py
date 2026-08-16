#!/usr/bin/env python3
"""Build one seed-aligned Route 2 LOSO aggregation input."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEVELOPMENT_INVENTORY_STUDIES = (
    "GSE200304",
    "GSE114002",
    "GSE149487",
    "GSE217518",
    "ENCSR854RUF",
    "GSE186455",
    "GSE256185",
    "GSE269595",
)
EXPECTED_LOSO_STUDIES = tuple(
    study for study in DEVELOPMENT_INVENTORY_STUDIES if study != "GSE256185"
)
ZERO_RECORD_DEVELOPMENT_STUDIES = ("GSE256185",)


class LosoInputBuildError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LosoInputBuildError(message)


def _load(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"required JSON is absent: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"required JSON is not an object: {path}")
    return payload


def build(seed: int, model_run_root: Path, evaluation_root: Path) -> dict[str, Any]:
    model_results = []
    baseline_results = []
    for study in EXPECTED_LOSO_STUDIES:
        lower = study.lower()
        summaries = sorted(
            path / "training_summary.json"
            for path in model_run_root.glob(f"delta_main_2m_lr3e4_loso_{lower}_seed{seed}*")
            if (path / "training_summary.json").is_file()
        )
        _require(len(summaries) == 1, f"expected one completed LOSO training summary for {study}, found {len(summaries)}")
        model_evaluation = evaluation_root / f"{lower}_model_vs_strongest_evaluation_v1.json"
        baseline_evaluation = evaluation_root / f"{lower}_strongest_only_evaluation_v1.json"
        model_results.append({
            "study_unit_id": study,
            "training_summary": _load(summaries[0]),
            "evaluation": _load(model_evaluation),
        })
        baseline_results.append({
            "study_unit_id": study,
            "evaluation": _load(baseline_evaluation),
        })
    return {
        "schema_version": "route_a_v3_route2_loso_aggregation_input.v1",
        "seed": seed,
        "development_inventory_studies": list(DEVELOPMENT_INVENTORY_STUDIES),
        "expected_loso_studies": list(EXPECTED_LOSO_STUDIES),
        "zero_record_development_studies": list(ZERO_RECORD_DEVELOPMENT_STUDIES),
        "model_results": model_results,
        "baseline_results": baseline_results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--model-run-root", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _require(not args.output.exists(), f"output already exists: {args.output}")
    payload = build(args.seed, args.model_run_root, args.evaluation_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "LOSO_AGGREGATION_INPUT_BUILT",
        "seed": args.seed,
        "model_result_count": len(payload["model_results"]),
        "baseline_result_count": len(payload["baseline_results"]),
        "evaluation_studies_included": 0,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
