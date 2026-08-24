#!/usr/bin/env python3
"""Prepare authorized fixed-pass Critic V4 refit or paired LOSO configs."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.route2_development_projection_v3 import load_projection_rows
from core.route2_xeditcritic_gate_v4 import (
    CONFIRMATION_SEEDS_V4,
    LOSO_STUDIES_V4,
)
from core.route2_xeditcritic_training_data_v3 import (
    XEditCriticRecordV3,
    records_from_projection_rows,
)
from core.route2_xeditcritic_training_v4 import FixedEffectiveTaskBatchSamplerV4


class CriticPosttestPrepareV4Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CriticPosttestPrepareV4Error(message)


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"artifact is not an object: {path}")
    return payload


def require_v4_posttest_authority(protocol: Mapping[str, Any]) -> None:
    _require(
        protocol.get("schema_version")
        == "route_a_v3_route2_xeditcritic_v4_posttest_protocol.v1"
        and protocol.get("status")
        == "FROZEN_PROSPECTIVE_BEFORE_V4_POSTTEST_EXECUTION",
        "Critic V4 posttest protocol changed",
    )
    three = _read(Path(protocol["three_seed_gate_path"]))
    atomic = _read(Path(protocol["atomic_frozen_test_path"]))
    frozen_gate = atomic.get("frozen_test_gate")
    _require(
        three.get("status") == "XEDITCRITIC_V4_THREE_SEED_PASS"
        and three.get("required_seeds") == list(CONFIRMATION_SEEDS_V4)
        and three.get("development_test_authorized") is True
        and three.get("atomic_development_test_only") is True,
        "Critic V4 posttest lacks exact three-seed PASS",
    )
    _require(
        atomic.get("status") == "ATOMIC_FROZEN_DEVELOPMENT_TEST_TERMINAL"
        and isinstance(frozen_gate, Mapping)
        and frozen_gate.get("status") == "XEDITCRITIC_V4_FROZEN_TEST_PASS"
        and frozen_gate.get("all_development_refit_authorized") is True
        and atomic.get("general_test_projection_persisted") is False
        and atomic.get("test_bottom_six_cache_persisted") is False
        and atomic.get("new_final_evaluation_outcomes_accessed") is False,
        "Critic V4 posttest lacks exact frozen TEST PASS",
    )


def _load_records(protocol: Mapping[str, Any]) -> list[XEditCriticRecordV3]:
    records = records_from_projection_rows(
        load_projection_rows([Path(value) for value in protocol["projection_paths"]])
    )
    _require(
        len(records) == int(protocol["all_development_refit"]["record_count"])
        == 107_873,
        "Critic V4 all-Development record count changed",
    )
    _require(
        set(Counter(record.study for record in records)) == set(LOSO_STUDIES_V4),
        "Critic V4 LOSO study inventory changed",
    )
    return records


def _updates_per_pass(records: Sequence[XEditCriticRecordV3], *, seed: int) -> int:
    sampler = FixedEffectiveTaskBatchSamplerV4(
        records,
        seed=seed,
        repeat_cap=4,
        effective_batch=32,
    )
    sampler.set_pass(0)
    count = len(sampler.batches_for_pass())
    _require(count > 0, "Critic V4 posttest sampler emitted no update")
    return count


def _runtime(
    base: Mapping[str, Any],
    protocol: Mapping[str, Any],
    *,
    stage: str,
    seed: int,
    output_root: Path,
    train_count: int,
    validation_count: int,
    updates_per_pass: int,
    held_out_study: str | None = None,
) -> dict[str, Any]:
    run_ids = ["v4_full"] if stage == "REFIT" else ["v4_full", "c0_v4"]
    geometry = {
        **dict(base["data_geometry"]),
        "expected_record_count": 107_873,
        "expected_train_count": train_count,
        "expected_validation_count": validation_count,
        "pass_count": 8,
        "updates_per_pass": updates_per_pass,
        "total_optimizer_updates": updates_per_pass * 8,
    }
    result = {
        **dict(base),
        "schema_version": "route_a_v3_route2_xeditcritic_v4_posttest_runtime.v1",
        "status": "FROZEN_POSTTEST_CONFIG_NOT_STARTED",
        "run_stage": stage,
        "training_seed": seed,
        "required_posttest_run_ids": run_ids,
        "posttest_protocol_path": str(
            REPO_ROOT
            / "configs/route_a_v3_route2_xeditcritic_v4_posttest_protocol_v1.json"
        ),
        "three_seed_gate_path": str(protocol["three_seed_gate_path"]),
        "atomic_frozen_test_path": str(protocol["atomic_frozen_test_path"]),
        "projection_paths": list(protocol["projection_paths"]),
        "bottom_six_cache": str(protocol["bottom_six_cache"]),
        "preflight_output": str(protocol["formal_preflight_path"]),
        "output_root": str(output_root),
        "data_geometry": geometry,
        "selected_model": "V4-FULL",
        "held_out_study": held_out_study,
        "held_out_study_scale_policy": (
            "NOT_APPLICABLE_ALL_DEVELOPMENT_REFIT"
            if held_out_study is None
            else "UNKNOWN_STUDY_SCALE_FIXED_1"
        ),
        "checkpoint_selection": "FINAL_PASS_8_FIXED_NO_TEST_OR_VALIDATION_SELECTION",
        "development_test_outcomes_accessed_during_posttest": False,
        "new_final_evaluation_outcomes_accessed": False,
    }
    return result


def prepare_refit_configs_v4(
    protocol: Mapping[str, Any], base: Mapping[str, Any]
) -> dict[str, Any]:
    require_v4_posttest_authority(protocol)
    records = _load_records(protocol)
    root = Path(protocol["all_development_refit"]["run_root"])
    jobs = []
    for seed in CONFIRMATION_SEEDS_V4:
        updates = _updates_per_pass(records, seed=seed)
        output_root = root / f"seed_{seed}"
        config = _runtime(
            base,
            protocol,
            stage="REFIT",
            seed=seed,
            output_root=output_root,
            train_count=len(records),
            validation_count=0,
            updates_per_pass=updates,
        )
        jobs.append(
            {
                "seed": seed,
                "run_id": "v4_full",
                "physical_gpu_index": (seed - CONFIRMATION_SEEDS_V4[0]) % 6,
                "config": config,
                "summary_path": str(output_root / "v4_full" / "run_summary.json"),
                "failure_path": str(output_root / "v4_full" / "failure.json"),
            }
        )
    return {
        "schema_version": "route_a_v3_route2_xeditcritic_v4_refit_job_manifest.v1",
        "status": "XEDITCRITIC_V4_REFIT_CONFIGS_PREPARED_NOT_STARTED",
        "required_seeds": list(CONFIRMATION_SEEDS_V4),
        "refit_pass_count": 8,
        "job_count": 3,
        "jobs": jobs,
        "development_test_outcomes_accessed_during_refit": False,
        "new_final_evaluation_outcomes_accessed": False,
        "loso_authorized": False,
    }


def prepare_loso_configs_v4(
    protocol: Mapping[str, Any],
    base: Mapping[str, Any],
    refit: Mapping[str, Any],
) -> dict[str, Any]:
    require_v4_posttest_authority(protocol)
    _require(
        refit.get("status") == "XEDITCRITIC_V4_ALL_DEVELOPMENT_REFIT_COMPLETE"
        and refit.get("required_seeds") == list(CONFIRMATION_SEEDS_V4)
        and int(refit.get("completed_refit_count", -1)) == 3
        and int(refit.get("refit_pass_count", -1)) == 8
        and refit.get("loso_authorized") is True,
        "Critic V4 LOSO requires all three refits",
    )
    records = _load_records(protocol)
    study_counts = Counter(record.study for record in records)
    root = Path(protocol["test_preserving_loso"]["run_root"])
    jobs = []
    for seed in CONFIRMATION_SEEDS_V4:
        for study in LOSO_STUDIES_V4:
            train_records = [record for record in records if record.study != study]
            updates = _updates_per_pass(train_records, seed=seed)
            output_root = root / f"seed_{seed}" / study
            config = _runtime(
                base,
                protocol,
                stage="LOSO",
                seed=seed,
                output_root=output_root,
                train_count=len(train_records),
                validation_count=study_counts[study],
                updates_per_pass=updates,
                held_out_study=study,
            )
            for run_id in ("v4_full", "c0_v4"):
                jobs.append(
                    {
                        "seed": seed,
                        "held_out_study": study,
                        "run_id": run_id,
                        "physical_gpu_index": len(jobs) % 6,
                        "config": config,
                        "summary_path": str(output_root / run_id / "run_summary.json"),
                        "failure_path": str(output_root / run_id / "failure.json"),
                    }
                )
    _require(len(jobs) == 42, "Critic V4 LOSO paired job count changed")
    return {
        "schema_version": "route_a_v3_route2_xeditcritic_v4_loso_job_manifest.v1",
        "status": "XEDITCRITIC_V4_LOSO_CONFIGS_PREPARED_NOT_STARTED",
        "required_seeds": list(CONFIRMATION_SEEDS_V4),
        "held_out_studies": list(LOSO_STUDIES_V4),
        "study_record_counts": dict(sorted(study_counts.items())),
        "refit_pass_count": 8,
        "job_count": 42,
        "jobs": jobs,
        "development_test_outcomes_accessed_during_loso": False,
        "new_final_evaluation_outcomes_accessed": False,
    }


def write_manifest_v4(payload: Mapping[str, Any], output_dir: Path) -> None:
    _require(not output_dir.exists(), f"Critic V4 posttest config root exists: {output_dir}")
    output_dir.mkdir(parents=True)
    written: dict[tuple[int, str | None], str] = {}
    for job in payload["jobs"]:
        identity = (int(job["seed"]), job.get("held_out_study"))
        if identity in written:
            continue
        label = f"seed_{identity[0]}"
        if identity[1] is not None:
            label += f"_{identity[1]}"
        path = output_dir / f"{label}.json"
        path.write_text(
            json.dumps(job["config"], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        written[identity] = str(path)
    manifest = {
        **dict(payload),
        "runtime_config_paths": list(written.values()),
        "jobs": [
            {key: value for key, value in job.items() if key != "config"}
            | {"config_path": written[(int(job["seed"]), job.get("held_out_study"))]}
            for job in payload["jobs"]
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=("REFIT", "LOSO"))
    parser.add_argument("--refit-manifest", type=Path)
    arguments = parser.parse_args()
    protocol = _read(arguments.protocol)
    base = _read(REPO_ROOT / protocol["screen_config"])
    if arguments.mode == "REFIT":
        payload = prepare_refit_configs_v4(protocol, base)
        output = Path(protocol["all_development_refit"]["runtime_config_root"])
    else:
        _require(arguments.refit_manifest is not None, "LOSO refit manifest is absent")
        payload = prepare_loso_configs_v4(
            protocol, base, _read(arguments.refit_manifest)
        )
        output = Path(protocol["test_preserving_loso"]["runtime_config_root"])
    write_manifest_v4(payload, output)
    print(json.dumps({key: value for key, value in payload.items() if key != "jobs"}, sort_keys=True))


if __name__ == "__main__":
    main()
