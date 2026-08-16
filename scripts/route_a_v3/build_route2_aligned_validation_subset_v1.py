#!/usr/bin/env python3
"""Build aligned Development-validation manifests and prediction files for ablation comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


class AlignedSubsetError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AlignedSubsetError(message)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    _require(rows, f"input is empty: {path}")
    return rows


def build(config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]], dict[str, Any]]:
    _require(
        config["schema_version"] == "route_a_v3_route2_aligned_validation_subset_config.v1",
        "unexpected config schema",
    )
    _require(config["evaluation_outcomes_accessed"] is False, "aligned subset accessed Evaluation")
    included_studies = [str(value) for value in config["included_study_unit_ids"]]
    _require(included_studies and len(included_studies) == len(set(included_studies)), "included studies are empty or duplicated")
    included = set(included_studies)

    included_regions = [str(value).replace("′", "").replace("'", "") for value in config.get("included_regions", [])]
    _require(len(included_regions) == len(set(included_regions)), "included regions are duplicated")
    _require(set(included_regions) <= {"3UTR", "5UTR"}, "included region is unsupported")

    manifest_rows = _read_jsonl(Path(config["development_manifest_path"]))
    _require(all(row["pool_assignment"] == "DEVELOPMENT" for row in manifest_rows), "manifest contains non-Development rows")
    validation_rows = [row for row in manifest_rows if row["split"] == "VALIDATION"]
    validation_ids = {str(row["canonical_record_id"]) for row in validation_rows}
    region_by_id: dict[str, str] = {}
    if included_regions:
        canonical_paths = config.get("canonical_paths", [])
        _require(canonical_paths, "canonical paths are required for a region-filtered subset")
        for path_value in canonical_paths:
            path = Path(path_value)
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                record_id = str(row["canonical_record_id"])
                if record_id not in validation_ids:
                    continue
                _require(record_id not in region_by_id, f"canonical record is duplicated: {record_id}")
                region_by_id[record_id] = str(row["region"]).replace("′", "").replace("'", "")
        _require(set(region_by_id) == validation_ids, "canonical inputs do not cover Development validation")
    selected_rows = [
        row for row in validation_rows
        if str(row["study_unit_id"]) in included
        and (not included_regions or region_by_id[str(row["canonical_record_id"])] in set(included_regions))
    ]
    selected_ids = {str(row["canonical_record_id"]) for row in selected_rows}
    _require(selected_rows, "aligned validation subset is empty")
    _require(len(selected_ids) == len(selected_rows), "aligned validation subset has duplicated ids")
    _require({str(row["study_unit_id"]) for row in selected_rows} == included, "an included study has no validation records")

    specs = config["prediction_inputs"]
    prediction_ids = [str(spec["prediction_id"]) for spec in specs]
    _require(prediction_ids and len(prediction_ids) == len(set(prediction_ids)), "prediction inputs are empty or duplicated")
    filtered: dict[str, list[dict[str, Any]]] = {}
    input_counts: dict[str, int] = {}
    for spec in specs:
        prediction_id = str(spec["prediction_id"])
        rows = _read_jsonl(Path(spec["path"]))
        by_id: dict[str, dict[str, Any]] = {}
        for row in rows:
            record_id = str(row["canonical_record_id"])
            _require(record_id in validation_ids, f"prediction is outside Development validation: {prediction_id}/{record_id}")
            _require(record_id not in by_id, f"prediction is duplicated: {prediction_id}/{record_id}")
            by_id[record_id] = row
        _require(selected_ids <= set(by_id), f"prediction does not cover aligned subset: {prediction_id}")
        filtered[prediction_id] = [by_id[record_id] for record_id in sorted(selected_ids)]
        input_counts[prediction_id] = len(rows)

    summary = {
        "schema_version": "route_a_v3_route2_aligned_validation_subset.v1",
        "status": "DEVELOPMENT_VALIDATION_ABLATION_SUBSET_ALIGNED",
        "included_study_unit_ids": sorted(included),
        "included_regions": sorted(included_regions),
        "record_count": len(selected_rows),
        "prediction_ids": sorted(filtered),
        "prediction_input_record_counts": input_counts,
        "prediction_output_record_count_each": len(selected_rows),
        "evaluation_outcomes_accessed": False,
        "scientific_claim_status": "NOT_ESTABLISHED",
    }
    return sorted(selected_rows, key=lambda row: str(row["canonical_record_id"])), filtered, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    _require(not args.output_directory.exists(), f"output directory already exists: {args.output_directory}")
    manifest_rows, predictions, summary = build(json.loads(args.config.read_text(encoding="utf-8")))
    args.output_directory.mkdir(parents=True)
    (args.output_directory / "validation_manifest.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in manifest_rows), encoding="utf-8"
    )
    for prediction_id, rows in predictions.items():
        (args.output_directory / f"{prediction_id}.validation_predictions.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
        )
    (args.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
