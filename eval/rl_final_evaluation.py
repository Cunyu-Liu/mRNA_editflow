"""Stage 6 final-evaluation contract and report builder.

This module deliberately separates *collecting evidence* from *claiming a
result*.  It consumes immutable, per-method evaluation-result JSON files and
refuses to label the comparison complete when a test split, independent oracle,
or requested comparator/ablation is missing.  In particular, it will not
accept train-role teacher results as a final evaluation.

The evaluator is intentionally model-agnostic: generation and scoring stay in
the existing Stage A--D entry points.  Those entry points must write a result
record using the schema checked here.  This keeps the final table auditable
without changing a policy, decoder, or biological constraint.

Usage (preflight of the repository's currently declared assets)::

    python -m mrna_editflow.eval.rl_final_evaluation \
      --project-root . \
      --out-json docs/rl_final_evaluation.json \
      --out-md docs/rl_final_evaluation.md \
      --out-ablation-md docs/rl_ablation_table.md \
      --out-failure-md docs/rl_failure_cases.md

For a real final evaluation, provide ``--inventory``.  It is a JSON object
with ``splits``, ``oracles``, ``methods``, ``ablations`` and ``result_paths``.
Each referenced result is immutable JSON and must identify its input dataset,
test-role split manifest, checkpoint, code revision, seed, hardware, runtime,
oracle, decoder and reward-schema version.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


_REPO_ROOT = Path(__file__).resolve().parents[1]
_PACKAGE_PARENT = _REPO_ROOT.parent
for _path in (str(_PACKAGE_PARENT), str(_REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from mrna_editflow.data.split_contract import load_and_verify_split_manifest, sha256_file


SCHEMA_VERSION = 1

REQUIRED_METHODS = (
    "stage_a_editflow_only",
    "stage_b_single_objective_ranker",
    "stage_b_vector_reward_ranker",
    "stage_b_stop",
    "stage_b_dagger",
    "grpo_without_kl",
    "grpo_with_kl",
    "grpo_with_kl_editflow_replay",
    "oracle_guided_local_search",
    "random_safe_editing",
    "external_baselines",
)

REQUIRED_ABLATIONS = (
    "no_stop",
    "no_cycle_prevention",
    "raw_intensity_softmax",
    "log_intensity_softmax",
    "single_step_teacher",
    "dagger_teacher",
    "scalar_reward",
    "vector_reward",
    "fixed_preference",
    "random_preference",
    "no_uncertainty_penalty",
    "no_edit_cost",
    "no_kl",
    "no_editflow_replay",
)

REQUIRED_SPLITS = (
    "family_disjoint_train",
    "family_disjoint_validation",
    "family_disjoint_test",
    "ood_family_test",
    "length_shift_test",
    "gc_shift_test",
)
OPTIONAL_SPLITS = ("species_shift_test",)
# Train and validation contracts are mandatory reproducibility evidence, but a
# final-comparison result must never be computed on either role.
EVALUATION_RESULT_SPLITS = (
    "family_disjoint_test",
    "ood_family_test",
    "length_shift_test",
    "gc_shift_test",
)

CONSTRAINT_METRICS = (
    "valid_cds_fraction",
    "protein_identity_fraction",
    "frame_preservation_fraction",
    "region_constraint_satisfaction",
    "edit_budget_satisfaction",
    "stop_rate",
    "cycle_rejection_count",
)
RANKING_METRICS = (
    "mean_regret",
    "recall_at_1",
    "recall_at_8",
    "recall_at_32",
    "ndcg_at_k",
    "positive_edit_precision",
)
DESIGN_METRICS = (
    "raw_te_proxy_delta",
    "risk_adjusted_te_proxy_delta",
    "accessibility_delta",
    "gc_violation_rate",
    "uaug_violation_rate",
    "cai_delta_cds_tasks",
    "edit_distance",
    "reward_per_edit",
    "pareto_hypervolume",
    "pareto_front_coverage",
    "diversity",
)
STABILITY_METRICS = (
    "seed_variance",
    "bootstrap_confidence_interval",
    "paired_significance_test",
    "kl_trajectory",
    "reward_variance",
    "entropy",
    "gradient_norm",
    "failure_rate",
    "runtime",
    "gpu_memory",
)
REQUIRED_METRICS = (
    CONSTRAINT_METRICS + RANKING_METRICS + DESIGN_METRICS + STABILITY_METRICS
)
RESULT_METADATA_FIELDS = (
    "dataset_hash",
    "split_manifest",
    "checkpoint_hash",
    "code_commit",
    "seed",
    "hardware",
    "runtime",
    "oracle_metadata",
    "decoder_type",
    "reward_schema_version",
)
REQUIRED_FAILURE_CASES = (
    "forced_harmful_edits",
    "premature_stop",
    "over_editing",
    "reward_hacking_motifs",
    "extreme_gc",
    "oracle_disagreement",
    "ood_failures",
    "local_search_beats_grpo",
    "grpo_beats_ranker",
)


class FinalEvaluationContractError(ValueError):
    """Raised for malformed Stage 6 evidence rather than silently accepting it."""


def _sha256(path: Path) -> str:
    return sha256_file(path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FinalEvaluationContractError(f"invalid JSON artifact: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise FinalEvaluationContractError(f"JSON artifact must be an object: {path}")
    return payload


def _resolve(root: Path, raw_path: object) -> Path | None:
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None
    path = Path(raw_path)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _git_commit(project_root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _hardware_identity() -> dict[str, object]:
    """Best-effort immutable hardware description; absence is recorded, not guessed."""
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL,
        )
        gpus = [line.strip() for line in output.splitlines() if line.strip()]
    except (OSError, subprocess.CalledProcessError):
        gpus = []
    return {"platform": platform.platform(), "python": sys.version.split()[0], "gpus": gpus or None}


def _status_row(status: str, *, reason: str | None = None, **extra: object) -> dict[str, object]:
    row: dict[str, object] = {"status": status, **extra}
    if reason:
        row["reason"] = reason
    return row


def inspect_split(project_root: Path, item: Mapping[str, object]) -> dict[str, object]:
    """Verify a split contract and record only immutable, observed facts."""
    path = _resolve(project_root, item.get("manifest_path"))
    if path is None:
        return _status_row("missing", reason="manifest_path_not_supplied")
    if not path.is_file():
        return _status_row("missing", reason="manifest_not_found", manifest_path=str(path))
    try:
        contract = load_and_verify_split_manifest(str(path))
    except Exception as exc:  # contract errors must reach the report, not be hidden
        return _status_row("invalid", reason=str(exc), manifest_path=str(path), manifest_sha256=_sha256(path))
    roles = {name: int(role.count) for name, role in contract.roles.items()}
    return _status_row(
        "available",
        manifest_path=str(path),
        manifest_sha256=contract.manifest_sha256,
        dataset_id=contract.dataset_id,
        dataset_hash=contract.records_sha256,
        roles=roles,
        family_disjoint=bool(contract.family_disjoint),
        near_neighbor_threshold_passed=bool(contract.near_neighbor_threshold_passed),
        paper_eligible=bool(contract.paper_eligible),
        block_reasons=list(contract.block_reasons),
    )


def inspect_oracle(project_root: Path, item: Mapping[str, object]) -> dict[str, object]:
    path = _resolve(project_root, item.get("manifest_path"))
    if path is None:
        return _status_row("missing", reason="oracle_manifest_not_supplied")
    if not path.is_file():
        return _status_row("missing", reason="oracle_manifest_not_found", manifest_path=str(path))
    try:
        payload = _read_json(path)
    except FinalEvaluationContractError as exc:
        return _status_row("invalid", reason=str(exc), manifest_path=str(path))
    artifact = _resolve(path.parent, payload.get("artifact_path"))
    expected_sha = payload.get("artifact_sha256")
    artifact_ok = bool(artifact and artifact.is_file() and isinstance(expected_sha, str) and _sha256(artifact) == expected_sha)
    independent = payload.get("independent") is True
    heuristic = "heuristic" in str(payload.get("oracle_type", "")).lower()
    if not artifact_ok:
        return _status_row(
            "invalid", reason="oracle_artifact_missing_or_sha_mismatch", manifest_path=str(path), manifest_sha256=_sha256(path)
        )
    return _status_row(
        "available",
        manifest_path=str(path),
        manifest_sha256=_sha256(path),
        artifact_path=str(artifact),
        artifact_sha256=str(expected_sha),
        oracle_type=payload.get("oracle_type"),
        source=payload.get("source"),
        independent=independent,
        heuristic=heuristic,
        independence_statement=payload.get("independence_statement"),
    )


def inspect_method(project_root: Path, item: Mapping[str, object]) -> dict[str, object]:
    executable = item.get("executable") is True
    if item.get("status") == "not_executable":
        return _status_row("not_executable", reason=str(item.get("reason", "no_executable_baseline_evidence")))
    source = _resolve(project_root, item.get("checkpoint_path"))
    source_kind = "checkpoint"
    if source is None:
        source = _resolve(project_root, item.get("artifact_path"))
        source_kind = "artifact"
    if source is None or not source.is_file():
        return _status_row("missing", reason=f"{source_kind}_not_found", executable=executable)
    return _status_row(
        "available",
        executable=executable,
        source_kind=source_kind,
        source_path=str(source),
        source_hash=_sha256(source),
    )


def _is_nonempty(value: object) -> bool:
    return value is not None and value != "" and value != [] and value != {}


def validate_result(
    project_root: Path,
    path: Path,
    *,
    expected_method: str,
    split_name: str,
    training_oracle_sha: str | None,
    heldout_oracle_sha: str | None,
    alternative_oracle_sha: str | None,
) -> dict[str, object]:
    """Validate one method/split result without interpreting its numerical outcome."""
    if not path.is_file():
        return _status_row("missing", reason="result_not_found", result_path=str(path))
    try:
        payload = _read_json(path)
    except FinalEvaluationContractError as exc:
        return _status_row("invalid", reason=str(exc), result_path=str(path))
    metadata = payload.get("metadata")
    metrics = payload.get("metrics")
    if payload.get("schema_version") != SCHEMA_VERSION or not isinstance(metadata, Mapping) or not isinstance(metrics, Mapping):
        return _status_row("invalid", reason="result_requires_schema_version_1_metadata_and_metrics", result_path=str(path))
    missing_meta = [name for name in RESULT_METADATA_FIELDS if not _is_nonempty(metadata.get(name))]
    missing_metrics = [name for name in REQUIRED_METRICS if not _is_nonempty(metrics.get(name))]
    role = metadata.get("split_role")
    oracle = metadata.get("oracle_metadata")
    oracle_sha = oracle.get("manifest_sha256") if isinstance(oracle, Mapping) else None
    reasons: list[str] = []
    if payload.get("method") != expected_method:
        reasons.append("method_name_mismatch")
    if role != "test":
        reasons.append("final_result_must_use_test_role")
    if metadata.get("evaluation_split") != split_name:
        reasons.append("evaluation_split_name_mismatch")
    if training_oracle_sha and oracle_sha == training_oracle_sha:
        reasons.append("training_oracle_cannot_be_final_evaluation_oracle")
    if split_name == "family_disjoint_test" and oracle_sha not in {heldout_oracle_sha, alternative_oracle_sha}:
        reasons.append("family_test_requires_heldout_or_alternative_oracle")
    if missing_meta:
        reasons.append("missing_metadata:" + ",".join(missing_meta))
    if missing_metrics:
        reasons.append("missing_metrics:" + ",".join(missing_metrics))
    return _status_row(
        "available" if not reasons else "invalid",
        reason="; ".join(reasons) if reasons else None,
        result_path=str(path),
        result_sha256=_sha256(path),
        method=payload.get("method"),
        split=split_name,
        metrics={name: metrics.get(name) for name in REQUIRED_METRICS if name in metrics},
        metadata=dict(metadata),
    )


def default_inventory(project_root: Path) -> dict[str, Any]:
    """Small, explicit inventory of known repository assets; no guessed results."""
    return {
        "schema_version": SCHEMA_VERSION,
        "splits": {
            "family_disjoint_train": {"manifest_path": "benchmark/dev/gencode_family_leakage_protocol/split_manifest.json"},
            "family_disjoint_validation": {"manifest_path": "benchmark/dev/gencode_family_leakage_protocol/split_manifest.json"},
            "family_disjoint_test": {"manifest_path": "benchmark/dev/gencode_family_leakage_protocol/split_manifest.json"},
            "ood_family_test": {"manifest_path": "benchmark/dev/rl_stage6_assets/splits/ood_family_test/split_manifest.json"},
            "length_shift_test": {"manifest_path": "benchmark/dev/rl_stage6_assets/splits/length_shift_test/split_manifest.json"},
            "gc_shift_test": {"manifest_path": "benchmark/dev/rl_stage6_assets/splits/gc_shift_test/split_manifest.json"},
        },
        "oracles": {
            "training": {"manifest_path": "benchmark/dev/rl_stage6_assets/oracles/training_oracle_manifest.json"},
            "heldout": {"manifest_path": "benchmark/paper/leakage_free_headline/oracle_manifest.json"},
            "alternative": {"manifest_path": "benchmark/dev/rl_stage6_assets/oracles/alternative_heuristic_oracle_manifest.json"},
            "public_experimental": {},
        },
        "methods": {
            "stage_a_editflow_only": {"checkpoint_path": "ckpts/stage_a_public_full_1k/stage_a_best.pt"},
            "stage_b_single_objective_ranker": {"checkpoint_path": "ckpts/proposal_ranker_t5_mo_te_only_head256/proposal_ranker_best.pt"},
            "stage_b_vector_reward_ranker": {"checkpoint_path": "ckpts/proposal_ranker_t5_mo_pareto_head256/proposal_ranker_best.pt"},
            "stage_b_stop": {}, "stage_b_dagger": {}, "grpo_without_kl": {}, "grpo_with_kl": {}, "grpo_with_kl_editflow_replay": {},
            "oracle_guided_local_search": {"artifact_path": "benchmark/utr_local_search_head1024.json"},
            "random_safe_editing": {},
            "external_baselines": {"artifact_path": "docs/external_sota_real_run_audit.json", "executable": True},
        },
        "ablations": {name: {} for name in REQUIRED_ABLATIONS},
        "result_paths": {},
        "environment": {},
    }


def _availability(rows: Mapping[str, Mapping[str, object]], required: Iterable[str]) -> list[str]:
    return [name for name in required if rows.get(name, {}).get("status") != "available"]


def build_report(project_root: str | Path, inventory: Mapping[str, object] | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    root = Path(project_root).resolve()
    inv = dict(default_inventory(root) if inventory is None else inventory)
    if inv.get("schema_version") != SCHEMA_VERSION:
        raise FinalEvaluationContractError("inventory schema_version must equal 1")

    raw_splits = inv.get("splits") if isinstance(inv.get("splits"), Mapping) else {}
    splits = {name: inspect_split(root, dict(raw_splits.get(name, {}))) for name in REQUIRED_SPLITS + OPTIONAL_SPLITS}
    raw_oracles = inv.get("oracles") if isinstance(inv.get("oracles"), Mapping) else {}
    oracles = {name: inspect_oracle(root, dict(raw_oracles.get(name, {}))) for name in ("training", "heldout", "alternative", "public_experimental")}
    raw_methods = inv.get("methods") if isinstance(inv.get("methods"), Mapping) else {}
    methods = {name: inspect_method(root, dict(raw_methods.get(name, {}))) for name in REQUIRED_METHODS}
    raw_ablations = inv.get("ablations") if isinstance(inv.get("ablations"), Mapping) else {}
    ablations = {}
    for name in REQUIRED_ABLATIONS:
        path = _resolve(root, dict(raw_ablations.get(name, {})).get("result_path"))
        ablations[name] = (
            _status_row("available", result_path=str(path), result_sha256=_sha256(path))
            if path is not None and path.is_file()
            else _status_row("missing", reason="ablation_result_not_supplied_or_not_found")
        )
    training_sha = oracles["training"].get("manifest_sha256") if oracles["training"].get("status") == "available" else None
    heldout_sha = oracles["heldout"].get("manifest_sha256") if oracles["heldout"].get("status") == "available" else None
    alternative_sha = oracles["alternative"].get("manifest_sha256") if oracles["alternative"].get("status") == "available" else None
    result_paths = inv.get("result_paths") if isinstance(inv.get("result_paths"), Mapping) else {}
    results: dict[str, dict[str, object]] = {}
    for method in REQUIRED_METHODS:
        per_split = result_paths.get(method, {}) if isinstance(result_paths.get(method, {}), Mapping) else {}
        results[method] = {}
        for split_name in EVALUATION_RESULT_SPLITS + OPTIONAL_SPLITS:
            raw = per_split.get(split_name)
            path = _resolve(root, raw)
            results[method][split_name] = _status_row("missing", reason="result_not_supplied") if path is None else validate_result(
                root, path, expected_method=method, split_name=split_name, training_oracle_sha=training_sha, heldout_oracle_sha=heldout_sha, alternative_oracle_sha=alternative_sha
            )

    blockers: list[str] = []
    missing_splits = _availability(splits, REQUIRED_SPLITS)
    if missing_splits:
        blockers.append("required_splits_unavailable:" + ",".join(missing_splits))
    for name in ("family_disjoint_train", "family_disjoint_validation", "family_disjoint_test"):
        if splits[name].get("family_disjoint") is not True:
            blockers.append("family_disjoint_contract_not_verified:" + name)
    missing_oracles = _availability(oracles, ("training", "heldout", "alternative"))
    if missing_oracles:
        blockers.append("required_oracles_unavailable:" + ",".join(missing_oracles))
    if heldout_sha and heldout_sha == training_sha:
        blockers.append("heldout_oracle_matches_training_oracle")
    if alternative_sha and alternative_sha == training_sha:
        blockers.append("alternative_oracle_matches_training_oracle")
    if oracles["heldout"].get("independent") is not True:
        blockers.append("heldout_oracle_not_independent")
    # The alternative Oracle is deliberately allowed to be a heuristic.  It
    # supports disagreement and reward-hacking diagnostics, never a claim of
    # experimental performance or an independent held-out result.
    # The protocol requires an external baseline only where an executable
    # implementation is actually available.  ``not_executable`` is therefore
    # auditable evidence of absence, not a fabricated comparator; a silent
    # missing entry remains a blocker.
    required_method_evidence = tuple(
        name for name in REQUIRED_METHODS
        if not (name == "external_baselines" and methods[name].get("status") == "not_executable")
    )
    missing_methods = _availability(methods, required_method_evidence)
    if missing_methods:
        blockers.append("required_methods_unavailable:" + ",".join(missing_methods))
    missing_ablations = _availability(ablations, REQUIRED_ABLATIONS)
    if missing_ablations:
        blockers.append("required_ablations_unavailable:" + ",".join(missing_ablations))
    invalid_results = []
    for method, rows in results.items():
        for split_name in EVALUATION_RESULT_SPLITS:
            if rows[split_name].get("status") != "available":
                invalid_results.append(method + ":" + split_name)
    if invalid_results:
        blockers.append("final_result_matrix_incomplete:" + ",".join(invalid_results))

    env = inv.get("environment") if isinstance(inv.get("environment"), Mapping) else {}
    status = "complete" if not blockers else "incomplete_preflight"
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "rl_final_evaluation",
        "status": status,
        "claim_tier": "paper" if status == "complete" else "development_only",
        "paper_eligible": status == "complete" and all(splits[n].get("paper_eligible") is True for n in REQUIRED_SPLITS),
        "conclusion": "No method comparison is claimed until all required final-evaluation evidence is available." if blockers else "Results are evidence-complete; interpret numerical metrics from the immutable result matrix.",
        "block_reasons": blockers,
        "project_root": str(root),
        "code_commit": _git_commit(root),
        "hardware": {**_hardware_identity(), **dict(env)},
        "report_runtime_s": round(time.perf_counter() - started, 6),
        "result_metadata_contract": {
            field: "required for every immutable method/split result; unavailable entries are rejected" for field in RESULT_METADATA_FIELDS
        },
        "required_methods": list(REQUIRED_METHODS),
        "required_ablations": list(REQUIRED_ABLATIONS),
        "required_splits": list(REQUIRED_SPLITS),
        "evaluation_result_splits": list(EVALUATION_RESULT_SPLITS),
        "optional_splits": list(OPTIONAL_SPLITS),
        "required_metrics": {
            "constraints": list(CONSTRAINT_METRICS), "ranking": list(RANKING_METRICS),
            "design": list(DESIGN_METRICS), "stability": list(STABILITY_METRICS),
        },
        "splits": splits, "oracles": oracles, "methods": methods, "ablations": ablations, "results": results,
        "failure_case_categories": list(REQUIRED_FAILURE_CASES),
    }


def write_json(report: Mapping[str, object], path: str | Path) -> None:
    target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _md_table(rows: Sequence[tuple[str, Mapping[str, object]]], columns: Sequence[str]) -> str:
    table_columns = ("item",) + tuple(columns)
    header = "| " + " | ".join(table_columns) + " |\n"
    divider = "| " + " | ".join("---" for _ in table_columns) + " |\n"
    body = "".join(
        "| " + " | ".join((name,) + tuple(str(row.get(column, "")) for column in columns)) + " |\n"
        for name, row in rows
    )
    return header + divider + body


def write_markdown(report: Mapping[str, object], path: str | Path) -> None:
    target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# RL final evaluation", "", f"Status: `{report['status']}`.", "", str(report["conclusion"]), "", "## Evidence gates", ""]
    reasons = report.get("block_reasons", [])
    lines.extend([f"- `{reason}`" for reason in reasons] or ["- All required evidence gates passed."])
    lines += ["", "## Split and Oracle status", ""]
    split_rows = [(name, row) for name, row in dict(report["splits"]).items()]
    lines.append(_md_table(split_rows, ("status", "manifest_path", "family_disjoint", "paper_eligible", "reason")))
    oracle_rows = [(name, row) for name, row in dict(report["oracles"]).items()]
    lines.append(_md_table(oracle_rows, ("status", "oracle_type", "independent", "manifest_sha256", "reason")))
    lines += ["", "## Comparator status", ""]
    method_rows = [(name, row) for name, row in dict(report["methods"]).items()]
    lines.append(_md_table(method_rows, ("status", "source_kind", "source_path", "source_hash", "reason")))
    lines += ["", "## Required per-result provenance", "", "Each admissible row must record: `dataset_hash`, `split_manifest`, `checkpoint_hash`, `code_commit`, `seed`, `hardware`, `runtime`, `oracle_metadata`, `decoder_type`, and `reward_schema_version`. Missing fields reject the result rather than receiving an inferred value.", "", "## Scientific interpretation boundary", "", "This report records heuristic-proxy outputs only as model/Oracle scores. It does not claim measured translation efficiency, half-life, or experimental effects. A result generated on the training Oracle or train-role teacher data is rejected as a final comparison.", ""]
    target.write_text("\n".join(lines), encoding="utf-8")


def write_ablation_markdown(report: Mapping[str, object], path: str | Path) -> None:
    target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
    rows = [(name, row) for name, row in dict(report["ablations"]).items()]
    text = "# RL ablation table\n\n" + _md_table(rows, ("status", "reason"))
    text += "\nNumerical ablation deltas are intentionally omitted until immutable test-role result artifacts satisfy the final-evaluation contract.\n"
    target.write_text(text, encoding="utf-8")


def write_failure_markdown(report: Mapping[str, object], path: str | Path) -> None:
    target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# RL failure cases", "", "No failure-rate or qualitative case claim is made before the required independent test results are present.", ""]
    labels = {
        "forced_harmful_edits": "Forced harmful edits", "premature_stop": "Premature STOP", "over_editing": "Over-editing",
        "reward_hacking_motifs": "Reward-hacking motifs", "extreme_gc": "Extreme GC", "oracle_disagreement": "Oracle disagreement",
        "ood_failures": "OOD failures", "local_search_beats_grpo": "Cases where local search beats GRPO", "grpo_beats_ranker": "Cases where GRPO beats ranker",
    }
    for key in REQUIRED_FAILURE_CASES:
        lines += [f"## {labels[key]}", "", "Pending: no admissible independent test result artifact supplied.", ""]
    target.write_text("\n".join(lines), encoding="utf-8")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--inventory", help="JSON inventory/result manifest; omit for audited preflight")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    parser.add_argument("--out-ablation-md", required=True)
    parser.add_argument("--out-failure-md", required=True)
    parser.add_argument("--require-complete", action="store_true", help="exit non-zero unless every final-evaluation gate passes")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    root = Path(args.project_root).resolve()
    inventory = _read_json(Path(args.inventory).resolve()) if args.inventory else None
    report = build_report(root, inventory)
    write_json(report, args.out_json)
    write_markdown(report, args.out_md)
    write_ablation_markdown(report, args.out_ablation_md)
    write_failure_markdown(report, args.out_failure_md)
    if args.require_complete and report["status"] != "complete":
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
