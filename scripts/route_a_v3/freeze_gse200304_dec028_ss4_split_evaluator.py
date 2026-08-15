#!/usr/bin/env python3
"""Freeze the one DEC028 SS4 component-disjoint split and evaluator contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs/route_a_v3_dec028_gse200304_ss4_split_evaluator_v1.json"
PROTOCOL_ID = "ROUTE_A_V3_DEC028_GSE200304_SS4_SPLIT_EVALUATOR_V1"
JOIN_DOMAIN = b"route-a-v3/gse200304/dec019/join-key/v1"
LOCATOR_DOMAIN = b"route-a-v3/gse200304/dec019/canonical-row-locator/v1"


class SplitError(RuntimeError):
    pass


def _require(value: bool, message: str) -> None:
    if not value:
        raise SplitError(message)


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    validate_config(value)
    return value


def validate_config(config: Mapping[str, Any]) -> None:
    _require(config.get("protocol_id") == PROTOCOL_ID, "protocol differs")
    _require(config.get("decision_id") == "V3-DEC-028", "decision differs")
    _require(config.get("document_status") == "ACTIVE_FOR_ONE_SS4_SPLIT_AND_FREEZE_ONLY", "SS4 inactive")
    authority = config["authority"]
    _require(authority["authorized_execution_count"] == 1, "execution count differs")
    for key in ("endpoint_or_model_result_allowed_for_assignment", "model_authorized", "cuda_authorized", "optimizer_authorized", "training_authorized", "g1_authorized", "sealed_access_authorized"):
        _require(authority[key] is False, f"forbidden authority enabled: {key}")
    split = config["split_contract"]
    _require(split["roles"] == ["TRAIN", "CALIBRATION", "TEST"], "roles differ")
    _require(split["split_salt"] == "GSE200304_DEC028_SINGLE_STUDY_SPLIT_V1", "salt differs")
    _require(sum(split["target_group_proportions"].values()) == 1.0, "proportions differ")
    _require(split["historical_fold_roles_reused"] is False and split["outcome_columns_used"] == [] and split["retry_or_resalt_allowed"] is False, "outcome or retry enabled")
    evaluator = config["evaluator_contract"]
    _require(evaluator["primary_metric"] == "WITHIN_STUDY_SOURCE_GROUP_EQUAL_WEIGHT_SPEARMAN", "primary metric differs")
    _require(evaluator["guide_or_model_selection_output_allowed"] is False and evaluator["checkpoint_selection_allowed"] is False, "evaluator feedback enabled")
    _require(len(evaluator["baseline_set"]) == 4, "baseline set differs")


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True).stdout.strip()


def audit_repository(config: Mapping[str, Any]) -> None:
    binding = config["implementation_binding"]
    _require(binding["status"] == "BOUND", "implementation binding is not BOUND")
    _require(_git("status", "--porcelain") == "", "repository is not clean")
    head = _git("rev-parse", "HEAD"); implementation = binding["implementation_commit"]
    _require(_git("rev-parse", f"{head}^") == implementation, "binding parent differs")
    _require(sorted(_git("diff-tree", "--no-commit-id", "--name-only", "-r", head).splitlines()) == sorted(binding["binding_exact_changed_paths"]), "binding paths differ")
    _require(_git("rev-parse", f"{implementation}^") == binding["implementation_expected_parent"], "implementation parent differs")
    _require(sorted(_git("diff-tree", "--no-commit-id", "--name-only", "-r", implementation).splitlines()) == sorted(binding["implementation_exact_changed_paths"]), "implementation paths differ")


def _identity(path: Path, spec: Mapping[str, Any]) -> bytes:
    payload = path.read_bytes()
    _require(len(payload) == spec["bytes"] and hashlib.sha256(payload).hexdigest() == spec["sha256"], f"input identity differs: {path.name}")
    return payload


def _framed(parts: tuple[bytes, ...]) -> bytes:
    return b"".join(len(item).to_bytes(8, "big") + item for item in parts)


def _domain_hash(domain: bytes, parts: tuple[bytes, ...]) -> bytes:
    return hashlib.sha256(_framed((domain, *parts))).digest()


def canonical_locator(record_key: str) -> str:
    join = _domain_hash(JOIN_DOMAIN, (record_key.encode("utf-8"),))
    return _domain_hash(LOCATOR_DOMAIN, (b"GSE200304", join, b"TotalPoly:RNA")).hex()


def assign_components(component_sizes: Mapping[str, int], salt: str, target: Mapping[str, float], roles: list[str]) -> dict[str, str]:
    total = sum(component_sizes.values())
    counts = {role: 0 for role in roles}
    ordered = sorted(component_sizes, key=lambda item: (-component_sizes[item], hashlib.sha256(salt.encode() + b"\0" + item.encode()).hexdigest(), item))
    result: dict[str, str] = {}
    for component in ordered:
        deficits = {role: target[role] * total - counts[role] for role in roles}
        role = max(roles, key=lambda item: (deficits[item], -roles.index(item)))
        result[component] = role
        counts[role] += component_sizes[component]
    return result


def execute(config: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    _require(not output_dir.exists(), "output directory exists; one-shot SS4 exhausted")
    ss3 = Path(config["inputs"]["ss3_directory"])
    conformance = json.loads((ss3 / "GSE200304_DEC028_SS3_MATERIALIZATION_CONFORMANCE.json").read_text())
    _require(conformance["overall_status"] == config["authority"]["materialization_conformance_required"], "SS3 conformance is not PASS")
    manifest = json.loads((ss3 / "GSE200304_DEC028_SS3_MATERIALIZATION_AGGREGATE_MANIFEST.json").read_text())
    _require(manifest["private_row_count"] == 6547 and manifest["split_assignment_count"] == 0, "SS3 manifest geometry differs")
    _require(manifest["required_fields_exactly"] == ["record_key", "source_group", "source_sequence", "candidate_sequence", "context_vector", "edit_features", "direction_normalized_effect", "biological_standard_error"], "SS3 row schema differs")
    rows_path = ss3 / "GSE200304_SINGLE_STUDY_DEVELOPMENT_ROWS_PRIVATE.jsonl"
    rows_payload = rows_path.read_bytes()
    _require(len(rows_payload) == manifest["private_rows_bytes"] and hashlib.sha256(rows_payload).hexdigest() == manifest["private_rows_sha256"], "SS3 private rows identity differs")
    mapping_spec = config["inputs"]["historical_group_mapping"]
    component_spec = config["inputs"]["historical_component_assignment"]
    mapping = json.loads(_identity(Path(mapping_spec["path"]), mapping_spec))
    historical = json.loads(_identity(Path(component_spec["path"]), component_spec))
    locator_to_group = {item["canonical_locator"]: item["biological_group_id"] for item in mapping["mappings"]}
    group_to_component = {item["biological_group_id"]: item["component_id"] for item in historical["assignments"]}
    _require((len(locator_to_group), len(group_to_component), historical["component_count"]) == (6547, 6544, 1936), "historical graph geometry differs")
    rows = [json.loads(line) for line in rows_payload.splitlines()]
    _require(len(rows) == 6547, "SS3 row count differs")
    record_to_group: dict[str, str] = {}
    record_to_component: dict[str, str] = {}
    new_to_old: dict[str, set[str]] = defaultdict(set); old_to_new: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        locator = canonical_locator(row["record_key"])
        _require(locator in locator_to_group, "record lacks historical canonical locator")
        group = locator_to_group[locator]
        _require(group in group_to_component, "group lacks historical component")
        record_to_group[row["record_key"]] = group
        record_to_component[row["record_key"]] = group_to_component[group]
        new_to_old[row["source_group"]].add(group); old_to_new[group].add(row["source_group"])
        _require(math.isfinite(row["direction_normalized_effect"]), "nonfinite effect")
        _require(math.isfinite(row["biological_standard_error"]) and row["biological_standard_error"] > 0, "invalid SE")
    _require(len(record_to_group) == 6547 and len(set(record_to_group.values())) == 6544, "record/group closure differs")
    _require(all(len(value) == 1 for value in new_to_old.values()) and all(len(value) == 1 for value in old_to_new.values()), "SS3 and historical group partitions differ")
    groups_by_component: dict[str, set[str]] = defaultdict(set)
    for key, group in record_to_group.items():
        groups_by_component[record_to_component[key]].add(group)
    _require(len(groups_by_component) == 1936, "component closure differs")
    component_role = assign_components({key: len(value) for key, value in groups_by_component.items()}, config["split_contract"]["split_salt"], config["split_contract"]["target_group_proportions"], config["split_contract"]["roles"])
    assignments = {key: component_role[component] for key, component in record_to_component.items()}
    row_counts = Counter(assignments.values())
    group_roles: dict[str, set[str]] = defaultdict(set); component_roles: dict[str, set[str]] = defaultdict(set)
    for key, role in assignments.items():
        group_roles[record_to_group[key]].add(role); component_roles[record_to_component[key]].add(role)
    _require(set(row_counts) == set(config["split_contract"]["roles"]), "one or more split roles empty")
    _require(all(len(value) == 1 for value in group_roles.values()), "group leakage")
    _require(all(len(value) == 1 for value in component_roles.values()), "component leakage")
    group_counts = Counter(next(iter(value)) for value in group_roles.values())
    component_counts = Counter(next(iter(value)) for value in component_roles.values())
    assignment_payload = (json.dumps(assignments, sort_keys=True) + "\n").encode("utf-8")
    audit = {
        "protocol_id": PROTOCOL_ID,
        "overall_status": "PASS_SS4_SPLIT_EVALUATOR_BASELINE_FREEZE",
        "row_count": len(assignments), "group_count": len(group_roles), "component_count": len(component_roles),
        "row_counts_by_role": dict(row_counts), "group_counts_by_role": dict(group_counts), "component_counts_by_role": dict(component_counts),
        "group_cross_role_count": 0, "component_cross_role_count": 0,
        "private_assignment_bytes": len(assignment_payload),
        "private_assignment_sha256": hashlib.sha256(assignment_payload).hexdigest(),
        "historical_fold_roles_reused": False, "outcome_columns_used": [],
        "primary_metric": config["evaluator_contract"]["primary_metric"],
        "baseline_count": len(config["evaluator_contract"]["baseline_set"]),
        "metric_execution_count": 0, "baseline_fit_count": 0, "model_construction_count": 0,
        "cuda_touch_count": 0, "optimizer_construction_count": 0, "parameter_update_count": 0, "g1_launched": False,
        "qualified_counts": {"ordinary": 1, "a1": 1, "true_a2": 0, "canonical_records": 6547},
        "scientific_claim_status": "NOT_ESTABLISHED"
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        (temporary / config["outputs"]["private_assignment_filename"]).write_bytes(assignment_payload)
        (temporary / config["outputs"]["aggregate_audit_filename"]).write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
        os.rename(temporary, output_dir)
    finally:
        if temporary.exists():
            for child in temporary.iterdir(): child.unlink()
            temporary.rmdir()
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(); config = load_config(args.config); audit_repository(config)
    result = execute(config, args.output_dir); print(json.dumps(result, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
