#!/usr/bin/env python3
"""Build Route 2 data tables and a source/gene/sequence-grouped Development split."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs/route_a_v3_route2_manifest_split_v1.json"
SPLITS = ("TRAIN", "VALIDATION", "TEST")
COMPLETE_SCOPE = "COMPLETE_DEVELOPMENT_AND_EVALUATION"
DEVELOPMENT_ONLY_SCOPE = "DEVELOPMENT_ONLY_PRE_EVALUATION_CLOSE"


class ManifestError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ManifestError(message)


class UnionFind:
    def __init__(self, members: Iterable[str]):
        self.parent = {member: member for member in members}
        self.size = {member: 1 for member in members}

    def find(self, member: str) -> str:
        parent = self.parent[member]
        if parent != member:
            self.parent[member] = self.find(parent)
        return self.parent[member]

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if self.size[left_root] < self.size[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        self.size[left_root] += self.size[right_root]


@dataclass(frozen=True)
class RecordMetadata:
    canonical_record_id: str
    study_unit_id: str
    pool_assignment: str
    source_group_key: str
    source_id: str
    source_sequence: str
    candidate_sequence: str
    gene_tokens: tuple[str, ...]
    stratum: tuple[str, str, str]


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    validate_config(config)
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    _require(config["schema_version"] == "route_a_v3_route2_manifest_split.v1", "unexpected schema version")
    scope = str(config.get("materialization_scope", COMPLETE_SCOPE))
    _require(
        scope in {COMPLETE_SCOPE, DEVELOPMENT_ONLY_SCOPE},
        f"unsupported materialization scope: {scope}",
    )
    studies = config["studies"]
    expected_study_count = 10 if scope == COMPLETE_SCOPE else 8
    _require(len(studies) == expected_study_count, "converted study inventory count differs from materialization scope")
    _require(len({study["study_unit_id"] for study in studies}) == len(studies), "study inventory is duplicated")
    _require(sum(study["pool_assignment"] == "DEVELOPMENT" for study in studies) == 8, "Development study count changed")
    expected_evaluation_count = 2 if scope == COMPLETE_SCOPE else 0
    _require(
        sum(study["pool_assignment"] == "EVALUATION" for study in studies) == expected_evaluation_count,
        "Evaluation study count differs from materialization scope",
    )
    for study in studies:
        _require(study["expected_canonical_record_count"] >= 0, "canonical count is not frozen")
        _require(Path(study["canonical_records_path"]).is_absolute(), "canonical path is not absolute")
    split = config["split_policy"]
    _require(split["unit"] == "CONNECTED_SOURCE_COMPONENT", "split unit changed")
    _require(split["group_by_source_id"] is True, "source grouping disabled")
    _require(split["group_by_gene_within_study"] is True, "gene grouping disabled")
    _require(split["group_by_exact_source_or_candidate_sequence"] is True, "exact sequence grouping disabled")
    _require(split["group_by_near_duplicate_source_sequence"] is True, "near-duplicate grouping disabled")
    _require(split["near_duplicate_same_length_identity"] == 0.95, "near-duplicate identity changed")
    ratios = split["ratios"]
    _require(set(ratios) == set(SPLITS) and math.isclose(sum(ratios.values()), 1.0), "split ratios changed")
    _require(all(value > 0 for value in ratios.values()), "split ratio is nonpositive")
    evaluation = config["evaluation_policy"]
    _require(evaluation["training_eligible"] is False, "Evaluation enabled for training")
    _require(evaluation["model_selection_eligible"] is False, "Evaluation enabled for model selection")
    _require(evaluation["outcome_metrics_computed"] is False, "Evaluation metrics were prematurely enabled")
    output = config["output"]
    _require(output["overwrite_allowed"] is False, "successful output overwrite enabled")
    _require(output["directory"].startswith("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/"), "output leaves Route 2 root")


def _gene_tokens(record: Mapping[str, Any], fields: list[str]) -> tuple[str, ...]:
    tokens: set[str] = set()
    for field in fields:
        value = record.get(field)
        values = value if isinstance(value, list) else [value]
        for item in values:
            if item is not None and str(item).strip():
                tokens.add(str(item).strip())
    return tuple(sorted(tokens))


def _load_study(spec: Mapping[str, Any]) -> tuple[list[RecordMetadata], Counter[str]]:
    path = Path(spec["canonical_records_path"])
    _require(path.is_file(), f"canonical input absent: {path}")
    result: list[RecordMetadata] = []
    stats: Counter[str] = Counter()
    seen_ids: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ManifestError(f"invalid canonical JSON: {path}:{line_number}") from exc
            _require(record["study_unit_id"] == spec["study_unit_id"], f"study id changed in {path.name}")
            _require(record["pool_assignment"] == spec["pool_assignment"], f"pool changed in {path.name}")
            canonical_id = str(record["canonical_record_id"])
            _require(canonical_id not in seen_ids, f"canonical id duplicated within {spec['study_unit_id']}")
            seen_ids.add(canonical_id)
            source_id = str(record["source_id"])
            source = str(record["source_sequence"]).upper()
            candidate = str(record["candidate_sequence"]).upper()
            _require(source and candidate and len(source) == len(candidate), f"invalid sequence pair in {canonical_id}")
            _require(not ((set(source) | set(candidate)) - set("ACGT")), f"invalid sequence alphabet in {canonical_id}")
            gene_tokens = _gene_tokens(record, spec["gene_group_fields"])
            if not gene_tokens:
                stats["records_without_gene_token"] += 1
            result.append(RecordMetadata(
                canonical_record_id=canonical_id,
                study_unit_id=spec["study_unit_id"],
                pool_assignment=spec["pool_assignment"],
                source_group_key=f"{spec['study_unit_id']}::{source_id}",
                source_id=source_id,
                source_sequence=source,
                candidate_sequence=candidate,
                gene_tokens=gene_tokens,
                stratum=(spec["study_unit_id"], str(record["region"]), str(record["endpoint_id"])),
            ))
            stats["canonical_record_count"] += 1
    _require(stats["canonical_record_count"] == spec["expected_canonical_record_count"], f"canonical count changed for {spec['study_unit_id']}")
    stats["source_group_count"] = len({record.source_group_key for record in result})
    return result, stats


def _partition_blocks(sequence: str, maximum_mismatches: int) -> list[tuple[int, str]]:
    count = maximum_mismatches + 1
    return [
        (index, sequence[(index * len(sequence)) // count:((index + 1) * len(sequence)) // count])
        for index in range(count)
    ]


def _near_duplicate_sequence_pairs(sequences: list[str], identity: float) -> Iterable[tuple[str, str]]:
    by_length: dict[int, list[str]] = defaultdict(list)
    for sequence in sequences:
        by_length[len(sequence)].append(sequence)
    for length, same_length in by_length.items():
        maximum_mismatches = math.floor((1.0 - identity) * length + 1e-9)
        block_index: dict[tuple[int, str], list[int]] = defaultdict(list)
        compared: set[tuple[int, int]] = set()
        for right_index, sequence in enumerate(same_length):
            candidate_indices: set[int] = set()
            for block in _partition_blocks(sequence, maximum_mismatches):
                candidate_indices.update(block_index[block])
            for left_index in candidate_indices:
                pair = (left_index, right_index)
                if pair in compared:
                    continue
                compared.add(pair)
                left = same_length[left_index]
                if sum(a != b for a, b in zip(left, sequence)) <= maximum_mismatches:
                    yield left, sequence
            for block in _partition_blocks(sequence, maximum_mismatches):
                block_index[block].append(right_index)


def _build_development_components(
    records: list[RecordMetadata],
    identity: float,
) -> tuple[dict[str, str], dict[str, int]]:
    source_groups = sorted({record.source_group_key for record in records})
    union = UnionFind(source_groups)
    gene_owner: dict[tuple[str, str], str] = {}
    exact_sequence_owner: dict[str, str] = {}
    source_sequence_owners: dict[str, set[str]] = defaultdict(set)
    for record in records:
        group = record.source_group_key
        for gene_token in record.gene_tokens:
            key = (record.study_unit_id, gene_token)
            owner = gene_owner.setdefault(key, group)
            union.union(owner, group)
        for sequence in (record.source_sequence, record.candidate_sequence):
            owner = exact_sequence_owner.setdefault(sequence, group)
            union.union(owner, group)
        source_sequence_owners[record.source_sequence].add(group)

    near_pair_count = 0
    for left_sequence, right_sequence in _near_duplicate_sequence_pairs(sorted(source_sequence_owners), identity):
        near_pair_count += 1
        left_owner = next(iter(source_sequence_owners[left_sequence]))
        right_owner = next(iter(source_sequence_owners[right_sequence]))
        union.union(left_owner, right_owner)
    component_by_group = {group: union.find(group) for group in source_groups}
    component_sizes = Counter(component_by_group.values())
    stats = {
        "source_group_count": len(source_groups),
        "connected_component_count": len(component_sizes),
        "largest_component_source_group_count": max(component_sizes.values()),
        "near_duplicate_source_sequence_pair_count": near_pair_count,
    }
    return component_by_group, stats


def _assign_components(
    records: list[RecordMetadata],
    component_by_group: Mapping[str, str],
    ratios: Mapping[str, float],
    seed: int,
) -> dict[str, str]:
    component_record_counts: Counter[str] = Counter()
    component_strata: dict[str, Counter[tuple[str, str, str]]] = defaultdict(Counter)
    total_strata: Counter[tuple[str, str, str]] = Counter()
    for record in records:
        component = component_by_group[record.source_group_key]
        component_record_counts[component] += 1
        component_strata[component][record.stratum] += 1
        total_strata[record.stratum] += 1
    total_records = len(records)
    randomizer = random.Random(seed)
    components = list(component_record_counts)
    randomizer.shuffle(components)
    components.sort(key=component_record_counts.__getitem__, reverse=True)
    assigned_records: Counter[str] = Counter()
    assigned_strata: dict[str, Counter[tuple[str, str, str]]] = {split: Counter() for split in SPLITS}
    assignment: dict[str, str] = {}
    for component in components:
        scores: dict[str, float] = {}
        for split in SPLITS:
            overall_deficit = ratios[split] * total_records - assigned_records[split]
            relevant_deficits = [
                ratios[split] * total_strata[stratum] - assigned_strata[split][stratum]
                for stratum in component_strata[component]
            ]
            scores[split] = overall_deficit + sum(relevant_deficits) / len(relevant_deficits)
        chosen = max(SPLITS, key=lambda split: (scores[split], -SPLITS.index(split)))
        assignment[component] = chosen
        assigned_records[chosen] += component_record_counts[component]
        assigned_strata[chosen].update(component_strata[component])
    _require(all(assigned_records[split] > 0 for split in SPLITS), "grouped split produced an empty partition")
    return assignment


def _loso_definitions(
    records: list[RecordMetadata],
    component_by_group: Mapping[str, str],
) -> list[dict[str, Any]]:
    studies = sorted({record.study_unit_id for record in records})
    result = []
    for holdout in studies:
        holdout_components = {
            component_by_group[record.source_group_key]
            for record in records if record.study_unit_id == holdout
        }
        evaluation = [record for record in records if record.study_unit_id == holdout]
        training = [
            record for record in records
            if record.study_unit_id != holdout
            and component_by_group[record.source_group_key] not in holdout_components
        ]
        excluded_bridge = [
            record for record in records
            if record.study_unit_id != holdout
            and component_by_group[record.source_group_key] in holdout_components
        ]
        _require(evaluation, f"LOSO holdout study is empty: {holdout}")
        _require(training, f"LOSO training set is empty after component exclusion: {holdout}")
        result.append({
            "schema_version": "route_a_v3_route2_loso_fold.v1",
            "fold_id": f"LOSO::{holdout}",
            "holdout_study_unit_id": holdout,
            "evaluation_role": "HELD_OUT_STUDY",
            "training_role": "OTHER_STUDIES_EXCLUDING_CONNECTED_HOLDOUT_COMPONENTS",
            "evaluation_record_count": len(evaluation),
            "training_record_count": len(training),
            "excluded_connected_other_study_record_count": len(excluded_bridge),
            "holdout_connected_component_count": len(holdout_components),
            "evaluation_outcomes_used_for_training_or_selection": False,
        })
    return result


def _evaluation_exposure(
    development: list[RecordMetadata],
    evaluation: list[RecordMetadata],
    identity: float,
) -> tuple[dict[str, dict[str, bool]], dict[str, int]]:
    development_sequences = {sequence for record in development for sequence in (record.source_sequence, record.candidate_sequence)}
    development_sources = {record.source_sequence for record in development}
    evaluation_sources = {record.source_sequence for record in evaluation}
    near_evaluation_sources: set[str] = set()
    tagged_sequences = [f"D:{sequence}" for sequence in development_sources] + [f"E:{sequence}" for sequence in evaluation_sources]
    raw_sequences = [item[2:] for item in tagged_sequences]
    origin_by_sequence: dict[str, set[str]] = defaultdict(set)
    for item in tagged_sequences:
        origin_by_sequence[item[2:]].add(item[0])
    for left, right in _near_duplicate_sequence_pairs(sorted(set(raw_sequences)), identity):
        if origin_by_sequence[left] != origin_by_sequence[right] and {"D", "E"} <= (origin_by_sequence[left] | origin_by_sequence[right]):
            if "E" in origin_by_sequence[left]:
                near_evaluation_sources.add(left)
            if "E" in origin_by_sequence[right]:
                near_evaluation_sources.add(right)
    result: dict[str, dict[str, bool]] = {}
    for record in evaluation:
        result[record.canonical_record_id] = {
            "exact_sequence_seen_in_development": record.source_sequence in development_sequences or record.candidate_sequence in development_sequences,
            "near_duplicate_source_seen_in_development": record.source_sequence in near_evaluation_sources,
        }
    stats = {
        "evaluation_record_count": len(evaluation),
        "exact_sequence_overlap_record_count": sum(value["exact_sequence_seen_in_development"] for value in result.values()),
        "near_duplicate_source_overlap_record_count": sum(value["near_duplicate_source_seen_in_development"] for value in result.values()),
    }
    return result, stats


def execute(config: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    _require(not output_dir.exists(), f"output directory already exists: {output_dir}")
    scope = str(config.get("materialization_scope", COMPLETE_SCOPE))
    records: list[RecordMetadata] = []
    study_stats: dict[str, dict[str, int]] = {}
    for spec in config["studies"]:
        study_records, stats = _load_study(spec)
        records.extend(study_records)
        study_stats[spec["study_unit_id"]] = dict(stats)
    canonical_ids = [record.canonical_record_id for record in records]
    _require(len(canonical_ids) == len(set(canonical_ids)), "canonical id is duplicated across studies")
    development = [record for record in records if record.pool_assignment == "DEVELOPMENT"]
    evaluation = [record for record in records if record.pool_assignment == "EVALUATION"]
    identity = config["split_policy"]["near_duplicate_same_length_identity"]
    component_by_group, component_stats = _build_development_components(development, identity)
    component_assignment = _assign_components(
        development,
        component_by_group,
        config["split_policy"]["ratios"],
        config["split_policy"]["seed"],
    )
    evaluation_exposure, exposure_stats = _evaluation_exposure(development, evaluation, identity)
    loso_definitions = _loso_definitions(development, component_by_group)
    development_inventory = [
        spec["study_unit_id"] for spec in config["studies"]
        if spec["pool_assignment"] == "DEVELOPMENT"
    ]
    nonempty_development_studies = sorted({record.study_unit_id for record in development})
    zero_record_development_studies = sorted(set(development_inventory) - set(nonempty_development_studies))
    _require(len(development_inventory) == 8, "Development inventory count changed")
    _require(
        len(loso_definitions) == len(nonempty_development_studies),
        "LOSO fold count differs from nonempty Development study count",
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        development_path = temporary / config["output"]["development_manifest_filename"]
        loso_path = temporary / config["output"]["loso_fold_definitions_filename"]
        split_counts: Counter[str] = Counter()
        evaluation_handle = None
        if scope == COMPLETE_SCOPE:
            evaluation_path = temporary / config["output"]["evaluation_manifest_filename"]
            evaluation_handle = evaluation_path.open("w", encoding="utf-8")
        with development_path.open("w", encoding="utf-8") as development_handle:
            for record in records:
                if record.pool_assignment == "DEVELOPMENT":
                    component = component_by_group[record.source_group_key]
                    split = component_assignment[component]
                    split_counts[split] += 1
                    payload = {
                        "canonical_record_id": record.canonical_record_id,
                        "study_unit_id": record.study_unit_id,
                        "pool_assignment": record.pool_assignment,
                        "split": split,
                        "connected_source_component_id": component,
                        "source_group_key": record.source_group_key,
                        "stratum": list(record.stratum),
                    }
                    development_handle.write(json.dumps(payload, sort_keys=True) + "\n")
                else:
                    _require(evaluation_handle is not None, "Evaluation record entered Development-only materialization")
                    exposure = evaluation_exposure[record.canonical_record_id]
                    payload = {
                        "canonical_record_id": record.canonical_record_id,
                        "study_unit_id": record.study_unit_id,
                        "pool_assignment": record.pool_assignment,
                        "split": "EVALUATION_ZERO_SHOT",
                        **exposure,
                        "headline_zero_shot_overlap_eligible": not any(exposure.values()),
                    }
                    evaluation_handle.write(json.dumps(payload, sort_keys=True) + "\n")
        if evaluation_handle is not None:
            evaluation_handle.close()
        with loso_path.open("w", encoding="utf-8") as loso_handle:
            for payload in loso_definitions:
                loso_handle.write(json.dumps(payload, sort_keys=True) + "\n")
        summary = {
            "schema_version": config["schema_version"],
            "status": (
                "ROUTE2_MANIFEST_AND_GROUPED_SPLIT_MATERIALIZED"
                if scope == COMPLETE_SCOPE
                else "ROUTE2_DEVELOPMENT_ONLY_MANIFEST_AND_GROUPED_SPLIT_MATERIALIZED"
            ),
            "materialization_scope": scope,
            "evaluation_manifest_materialized": scope == COMPLETE_SCOPE,
            "study_stats": study_stats,
            "development_record_count": len(development),
            "evaluation_record_count": len(evaluation),
            "development_split_record_counts": dict(sorted(split_counts.items())),
            "component_stats": component_stats,
            "evaluation_exposure_stats": exposure_stats,
            "loso_fold_count": len(loso_definitions),
            "development_inventory_study_ids": sorted(development_inventory),
            "nonempty_development_loso_study_ids": nonempty_development_studies,
            "zero_record_development_study_ids": zero_record_development_studies,
            "loso_folds": loso_definitions,
            "evaluation_outcome_metrics_computed": False,
            "scientific_claim_status": "NOT_ESTABLISHED",
        }
        (temporary / config["output"]["summary_filename"]).write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.rename(temporary, output_dir)
        return summary
    finally:
        if temporary.exists():
            for child in temporary.iterdir():
                child.unlink()
            temporary.rmdir()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    summary = execute(config, args.output_dir or Path(config["output"]["directory"]))
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
