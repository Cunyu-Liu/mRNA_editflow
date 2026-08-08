"""Adversarial tests for the V6 development-only, no-execution run plan."""

from __future__ import annotations

import ast
import copy
import hashlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import scripts.r3x.development_run_plan as plan_module  # noqa: E402
import scripts.r3x.typed_safe_authorities as typed_module  # noqa: E402
from scripts.r3x.artifact_io import canonical_json_bytes  # noqa: E402
from scripts.r3x.baseline_selection import SELECTION_SCHEMA  # noqa: E402
from scripts.r3x.protocol import canonical_sha256  # noqa: E402
from scripts.r3x.typed_safe_authorities import (  # noqa: E402
    ENDPOINT_ESTIMAND_REGISTRY,
    MODEL_OPTIMIZER_SEED_CONFIG,
    PAIRED_BASELINE_POLICY,
    TYPED_DEVELOPMENT_SAFE,
    write_typed_safe_authority_exclusive,
)


BASELINE_IDS = list(typed_module.FROZEN_BASELINE_METHOD_ORDER)
PROJECTION_NAMES = (
    "inclusion_manifest_development_projection",
    "context_source_crosswalk_development_projection",
    "source_component_split_manifest_development_projection",
)
ROW_SOURCE_NAMES = (
    "inclusion_manifest",
    "context_source_crosswalk",
    "source_component_split_manifest",
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _ref(root: Path, path: Path) -> dict[str, str]:
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _write_canonical(path: Path, payload: Mapping) -> str:
    path.write_bytes(canonical_json_bytes(payload))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _common(
    authority_name: str,
    *,
    data_sha: str,
    projection_hashes: Mapping[str, str],
    source_sha: str,
) -> dict:
    return {
        "schema_version": typed_module.SCHEMAS[authority_name],
        "artifact_role": typed_module.ROLES[authority_name],
        "status": TYPED_DEVELOPMENT_SAFE,
        "authority_name": authority_name,
        "source_authority_sha256": source_sha,
        "train_development_outcomes_sha256": data_sha,
        "development_projection_hashes": dict(projection_hashes),
        "sign_accuracy_gate": 0.60,
        "sign_gate_change_authorized": False,
        "ordinary_only": True,
        "training_ready": False,
        "sealed_access": False,
        "downstream_unlock": False,
    }


def _estimand_payload(
    *, data_sha: str, projection_hashes: Mapping[str, str]
) -> dict:
    return _common(
        ENDPOINT_ESTIMAND_REGISTRY,
        data_sha=data_sha,
        projection_hashes=projection_hashes,
        source_sha=_sha("source:endpoint-estimand-registry"),
    ) | {
        "entries": [
            {
                "study": "ordinary_study_a",
                "endpoint": "half_life",
                "estimand_id": "variant_minus_source_half_life",
                "raw_source_field": "source_half_life",
                "raw_candidate_field": "candidate_half_life",
                "raw_unit": "hours",
                "direction": "HIGHER_IS_MORE_STABLE",
                "transform": "IDENTITY_DIFFERENCE",
                "valid_domain": "FINITE",
                "role": "PRIMARY",
                "replicate_policy": "SINGLE_OBSERVATION_ONLY",
            },
            {
                "study": "ordinary_study_b",
                "endpoint": "translation_efficiency",
                "estimand_id": "variant_minus_source_translation_efficiency",
                "raw_source_field": "source_translation_efficiency",
                "raw_candidate_field": "candidate_translation_efficiency",
                "raw_unit": "relative_units",
                "direction": "HIGHER_IS_MORE_STABLE",
                "transform": "IDENTITY_DIFFERENCE",
                "valid_domain": "FINITE",
                "role": "PRIMARY",
                "replicate_policy": "SINGLE_OBSERVATION_ONLY",
            },
        ]
    }


def _paired_payload(
    *,
    data_sha: str,
    projection_hashes: Mapping[str, str],
    hgb_sha: str,
) -> dict:
    family = ["primary_candidate_vs_development_selected_baseline_sign"]
    return _common(
        PAIRED_BASELINE_POLICY,
        data_sha=data_sha,
        projection_hashes=projection_hashes,
        source_sha=_sha("source:paired-baseline-policy"),
    ) | {
        "policy_id": "R3X_PAIRED_STRONGEST_BASELINE_POLICY_V1",
        "baseline_suite_status": "PRETEST_POLICY_FROZEN",
        "primary_candidate_model_id": "A_EXACT_POTENTIAL_DIFFERENCE",
        "information_set": "PAIRED_SEQUENCE_NO_MEASURED_ANCHOR",
        "comparison_family_id": "R3X_PRIMARY_PAIRED_FAMILY_V1",
        "comparison_family_member_ids": family,
        "primary_comparison_member_id": family[0],
        "eligible_baseline_method_ids": list(BASELINE_IDS),
        "required_baseline_method_ids": list(BASELINE_IDS),
        "development_selector_required": True,
        "development_selector_schema": SELECTION_SCHEMA,
        "frozen_environment_execution_qualification_sha256": hgb_sha,
        "histogram_gradient_boosting_execution_qualification_asserted": True,
        "execution_qualification_replay_status": (
            "CERTIFICATE_HASH_BOUND_NOT_REPLAYED"
        ),
        "foundation_baseline_status": (
            "NOT_RUN_EXPOSURE_LICENSE_QUALIFICATION_UNKNOWN"
        ),
        "primary_sign_inference": {
            "metric_id": "MACRO_COMPONENT_CELL_BALANCED_SIGN_ACCURACY",
            "alpha": 0.05,
            "confidence": 0.95,
            "bootstrap": {
                "n_resamples": 2000,
                "seed": 23,
                "min_components_per_cell": 10,
                "max_attempt_multiplier": 100,
            },
            "permutation": {
                "n_permutations": 10000,
                "seed": 42,
                "exact_component_limit": 20,
                "min_components_per_cell": 10,
            },
        },
    }


def _hgb_payload(*, code_tree_sha: str) -> dict:
    return {
        "schema_version": plan_module.HGB_QUALIFICATION_SCHEMA,
        "artifact_role": plan_module.HGB_QUALIFICATION_ROLE,
        "status": plan_module.HGB_QUALIFICATION_STATUS,
        "qualification_id": "HGB_CPU_EXACT_REPLAY_V2",
        "method_id": "HISTOGRAM_GRADIENT_BOOSTING",
        "qualification_scope": "FROZEN_HGB_POLICY_ONLY",
        "device_class": "CPU",
        "environment_sha256": _sha("environment:hgb-cpu"),
        "code_tree_sha256": code_tree_sha,
        "verifier_code_sha256": _sha("verifier:hgb-v2"),
        "synthetic_smoke_sha256": _sha("synthetic:hgb-smoke"),
        "execution_replay_status": "HASH_DECLARATION_ONLY_NOT_REPLAYED",
        "ordinary_only": True,
        "network_access": False,
        "gpu_execution": False,
    }


def _config_v4_payload(
    *,
    data_sha: str,
    projection_hashes: Mapping[str, str],
    hgb_sha: str,
    code_tree_sha: str,
) -> dict:
    return {
        "schema_version": plan_module.EXECUTABLE_CONFIG_SCHEMA,
        "artifact_role": plan_module.EXECUTABLE_CONFIG_ROLE,
        "status": plan_module.EXECUTABLE_CONFIG_STATUS,
        "source_authority_sha256": _sha("source:executable-config-v4"),
        "train_development_outcomes_sha256": data_sha,
        "development_projection_hashes": dict(projection_hashes),
        "code_tree_sha256": code_tree_sha,
        "ordinary_only": True,
        "seeds": [11, 23, 42, 67, 101],
        "seed_aggregation": "FIXED_PREDICTION_ENSEMBLE",
        "estimand_training_scope": "INDEPENDENT_MODEL_PER_PRIMARY_ESTIMAND",
        "model": {
            "vocab_size": 5,
            "d_model": 64,
            "n_heads": 4,
            "n_layers": 2,
            "ff_dim": 128,
            "max_len": 512,
            "n_endpoints": 3,
            "n_contexts": 3,
            "max_edits": 5,
            "pair_rank": 4,
            "dropout": 0.0,
            "dropout_consumption_status": (
                "EXPLICIT_ZERO_MATCHES_BOUND_MODEL_IMPLEMENTATION"
            ),
        },
        "optimizer": {
            "name": "Adam",
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
        },
        "loss": {
            "huber_weight": 1.0,
            "nll_weight": 1.0,
            "sign_weight": 1.0,
            "sign_temperature": 1.0,
        },
        "checkpoint_selection": {
            "rule": "SIGN_GATE_THEN_SPEARMAN_THEN_NLL_THEN_EARLIEST_EPOCH",
            "selection_partition": "ORDINARY_DEVELOPMENT",
        },
        "training": {
            "epochs": 20,
            "batch_size": 64,
            "device_class": "CPU",
            "gradient_clip_norm": 5.0,
            "condition_unknown_dropout": 0.15,
        },
        "prior_policy": {
            "training_weight_policy": plan_module.FORMAL_SCALAR_WEIGHT_POLICY,
            "weighted_risk_normalization": "SUM_TO_ONE",
            "aggregation_policy": plan_module.PRIOR_TRAINING_POLICY,
        },
        "ridge_policy": {
            "feature_schema_by_method": {
                "EDIT_ONLY_RIDGE": plan_module.EDIT_ONLY_FEATURE_SCHEMA,
                "CANDIDATE_KMER_RIDGE": plan_module.CANDIDATE_KMER_FEATURE_SCHEMA,
                "PAIRED_LOCAL_GLOBAL_KMER_RIDGE": (
                    plan_module.PAIRED_KMER_FEATURE_SCHEMA
                ),
            },
            "solver_policy": plan_module.RIDGE_SOLVER_POLICY,
            "training_weight_policy": plan_module.FORMAL_SCALAR_WEIGHT_POLICY,
            "weighted_risk_normalization": "SUM_TO_ONE",
            "standardization_policy": (
                "FORMAL_WEIGHTED_TRAIN_ONLY_MEAN_VARIANCE_V2"
            ),
            "intercept_policy": (
                "FORMAL_WEIGHTED_TARGET_MEAN_UNPENALIZED_V1"
            ),
            "objective_policy": plan_module.RIDGE_TRAINING_WEIGHT_POLICY,
            "alpha_grid": [0.01, 0.1, 1.0, 10.0, 100.0],
            "max_component_folds": 5,
            "minimum_components_for_cv": 3,
            "standardization_floor": 1e-12,
            "selection_metric": plan_module.RIDGE_CV_SCORING_POLICY,
            "tie_rule": "LOWEST_ALPHA",
        },
        "hgb_policy": {
            "feature_set": "PAIRED_LOCAL_GLOBAL_KMER_WITH_CONDITION_UNK",
            "feature_schema": plan_module.PAIRED_KMER_FEATURE_SCHEMA,
            "training_weight_policy": plan_module.HGB_TRAINING_WEIGHT_POLICY,
            "weight_normalization": plan_module.HGB_WEIGHT_NORMALIZATION,
            "learning_rate": 0.05,
            "max_iter": 200,
            "max_leaf_nodes": 15,
            "l2_regularization": 1.0,
            "random_state": 42,
            "minimum_train_rows": 100,
            "early_stopping": False,
            "qualification_sha256": hgb_sha,
        },
        "nearest_neighbor_policy": {
            "feature_set": "PAIRED_LOCAL_GLOBAL_KMER_WITH_CONDITION_UNK",
            "feature_schema": plan_module.PAIRED_KMER_FEATURE_SCHEMA,
            "distance": "SQUARED_EUCLIDEAN",
            "scaling": plan_module.NEAREST_NEIGHBOR_SCALING_POLICY,
            "training_weight_policy": plan_module.FORMAL_SCALAR_WEIGHT_POLICY,
            "weighted_risk_normalization": "SUM_TO_ONE",
            "standardization_floor": 1e-12,
            "tie_absolute_tolerance": 1e-12,
            "tie_relative_tolerance": 0.0,
            "tie_aggregation": (
                plan_module.NEAREST_NEIGHBOR_TIE_AGGREGATION_POLICY
            ),
            "search_policy": plan_module.NEAREST_NEIGHBOR_SEARCH_POLICY,
            "train_chunk_size": 4096,
        },
        "resource_budget": {
            "max_wall_time_seconds_per_fit": 7200,
            "max_peak_memory_bytes_per_fit": 8_000_000_000,
            "max_forward_equivalents_per_fit": 10_000_000,
            "max_total_fit_units": 256,
            "max_estimands": 32,
            "max_rows": 1_000_000,
            "max_sequence_tokens_per_batch": 262_144,
        },
    }


def _config_v1_payload(
    *, data_sha: str, projection_hashes: Mapping[str, str], hgb_sha: str
) -> dict:
    return _common(
        MODEL_OPTIMIZER_SEED_CONFIG,
        data_sha=data_sha,
        projection_hashes=projection_hashes,
        source_sha=_sha("source:model-config-v1"),
    ) | {
        "seeds": [11, 23, 42, 67, 101],
        "seed_aggregation": "FIXED_PREDICTION_ENSEMBLE",
        "estimand_training_scope": "INDEPENDENT_MODEL_PER_PRIMARY_ESTIMAND",
        "condition_unknown_dropout": 0.15,
        "optimizer": {
            "name": "Adam",
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
        },
        "loss": {
            "huber_weight": 1.0,
            "nll_weight": 1.0,
            "sign_weight": 1.0,
            "sign_temperature": 1.0,
        },
        "checkpoint_selection": {
            "rule": "SIGN_GATE_THEN_SPEARMAN_THEN_NLL_THEN_EARLIEST_EPOCH",
            "selection_partition": "development",
        },
        "model": {
            "architecture_id": "PAIRED_POTENTIAL_DIFFERENCE_V1",
            "max_sequence_length": 4096,
            "embedding_dim": 256,
            "hidden_dim": 512,
            "num_layers": 4,
            "dropout": 0.10,
        },
        "frozen_environment_execution_qualification_sha256": hgb_sha,
        "histogram_gradient_boosting_execution_qualification_asserted": True,
        "execution_qualification_replay_status": (
            "CERTIFICATE_HASH_BOUND_NOT_REPLAYED"
        ),
        "foundation_baseline_status": (
            "NOT_RUN_EXPOSURE_LICENSE_QUALIFICATION_UNKNOWN"
        ),
    }


def _row_inventory_payload(
    *, data_sha: str, projection_hashes: Mapping[str, str]
) -> dict:
    rows = []
    studies = (
        (
            "a",
            "ordinary_study_a",
            "half_life",
            "variant_minus_source_half_life",
        ),
        (
            "b",
            "ordinary_study_b",
            "translation_efficiency",
            "variant_minus_source_translation_efficiency",
        ),
    )
    for short, study, endpoint, estimand_id in studies:
        for partition, count in (
            ("ORDINARY_TRAIN", 100),
            ("ORDINARY_DEVELOPMENT", 10),
        ):
            for index in range(count):
                suffix = f"{short}:{partition}:{index:03d}"
                rows.append(
                    {
                        "row_id": _sha(f"row:{suffix}"),
                        "partition": partition,
                        "study": study,
                        "endpoint": endpoint,
                        "estimand_id": estimand_id,
                        "source_component_id": f"COMPONENT_{suffix}",
                        "source_group_id_v2": f"POOL_{suffix}",
                    }
                )
    rows.sort(key=lambda row: row["row_id"])
    partition_hashes = {
        partition: canonical_sha256(
            sorted(row["row_id"] for row in rows if row["partition"] == partition)
        )
        for partition in ("ORDINARY_TRAIN", "ORDINARY_DEVELOPMENT")
    }
    return {
        "schema_version": plan_module.ROW_INVENTORY_SCHEMA,
        "artifact_role": plan_module.ROW_INVENTORY_ROLE,
        "status": plan_module.ROW_INVENTORY_STATUS,
        "capability_scope": "ORDINARY_TRAIN_AND_ORDINARY_DEVELOPMENT",
        "ordinary_only": True,
        "train_development_outcomes_sha256": data_sha,
        "source_authority_sha256": {
            name: _sha(f"source:row:{name}") for name in ROW_SOURCE_NAMES
        },
        "development_projection_hashes": dict(projection_hashes),
        "partition_row_set_sha256": partition_hashes,
        "rows": rows,
    }


def _release_payload(
    *,
    refs: Mapping[str, Mapping[str, str]],
    projection_hashes: Mapping[str, str],
    code_tree_sha: str,
) -> dict:
    approved = {
        name: refs[name]["sha256"]
        for name in (
            "paired_baseline_policy",
            "endpoint_estimand_registry",
            "executable_config",
            "hgb_qualification",
            "row_inventory",
        )
    }
    return {
        "schema_version": plan_module.DEVELOPMENT_RELEASE_SCHEMA,
        "artifact_role": plan_module.DEVELOPMENT_RELEASE_ROLE,
        "status": plan_module.DEVELOPMENT_RELEASE_STATUS,
        "release_id": "INDEPENDENT_DEVELOPMENT_PLAN_RELEASE_V1",
        "release_scope": "ORDINARY_TRAIN_AND_ORDINARY_DEVELOPMENT_PLAN_ONLY",
        "producer_id": "PRIVILEGED_BROKER_A",
        "verifier_id": "INDEPENDENT_VERIFIER_B",
        "verifier_code_sha256": _sha("verifier:development-release"),
        "approved_code_tree_sha256": code_tree_sha,
        "approved_artifact_sha256": approved,
        "projection_binding_sha256": canonical_sha256(dict(projection_hashes)),
        "independence_status": "SELF_REPORTED_DISTINCT_IDS_NOT_TRUST_ROOT_VERIFIED",
        "plan_construction_authorized": True,
        "training_execution_authorized": False,
        "gpu_execution_authorized": False,
        "future_stage_access_authorized": False,
        "ordinary_only": True,
    }


@dataclass
class PreparedInputs:
    root: Path
    refs: dict[str, dict[str, str]]
    payloads: dict[str, dict]
    code_tree_sha: str

    @property
    def expected_release_sha(self) -> str:
        return self.refs["independent_development_release"]["sha256"]

    def build(self) -> dict:
        return plan_module.build_development_run_plan(
            self.root,
            source_refs=self.refs,
            expected_development_release_sha256=self.expected_release_sha,
        )

    def replace(self, name: str, payload: Mapping, *, refresh_release: bool = True) -> None:
        path = self.root / self.refs[name]["relative_path"]
        _write_canonical(path, payload)
        self.payloads[name] = copy.deepcopy(dict(payload))
        self.refs[name] = _ref(self.root, path)
        if refresh_release and name != "independent_development_release":
            self.refresh_release()

    def refresh_release(self) -> None:
        release = _release_payload(
            refs=self.refs,
            projection_hashes=self.payloads["row_inventory"][
                "development_projection_hashes"
            ],
            code_tree_sha=self.code_tree_sha,
        )
        path = self.root / self.refs["independent_development_release"][
            "relative_path"
        ]
        _write_canonical(path, release)
        self.payloads["independent_development_release"] = release
        self.refs["independent_development_release"] = _ref(self.root, path)


@pytest.fixture
def prepared(tmp_path: Path) -> PreparedInputs:
    root = tmp_path / "ordinary_run"
    for relative in (
        "development/authorities",
        "development/run_plan_inputs",
        "development/releases",
        "development/qualifications",
        "development/run_plans",
    ):
        (root / relative).mkdir(parents=True, exist_ok=True)

    data_sha = _sha("ordinary-train-development-authority")
    projection_hashes = {
        name: _sha(f"projection:{name}") for name in PROJECTION_NAMES
    }
    code_tree_sha = _sha("approved-code-tree-v6")
    payloads: dict[str, dict] = {}
    refs: dict[str, dict[str, str]] = {}

    hgb_path = root / "development/qualifications/hgb_qualification_v2.json"
    payloads["hgb_qualification"] = _hgb_payload(code_tree_sha=code_tree_sha)
    _write_canonical(hgb_path, payloads["hgb_qualification"])
    refs["hgb_qualification"] = _ref(root, hgb_path)
    hgb_sha = refs["hgb_qualification"]["sha256"]

    paired_path = root / "development/authorities/paired_policy.json"
    payloads["paired_baseline_policy"] = _paired_payload(
        data_sha=data_sha,
        projection_hashes=projection_hashes,
        hgb_sha=hgb_sha,
    )
    paired_sha = write_typed_safe_authority_exclusive(
        root, paired_path, payloads["paired_baseline_policy"]
    )
    refs["paired_baseline_policy"] = {
        "relative_path": paired_path.relative_to(root).as_posix(),
        "sha256": paired_sha,
    }

    estimand_path = root / "development/authorities/estimand_registry.json"
    payloads["endpoint_estimand_registry"] = _estimand_payload(
        data_sha=data_sha,
        projection_hashes=projection_hashes,
    )
    estimand_sha = write_typed_safe_authority_exclusive(
        root, estimand_path, payloads["endpoint_estimand_registry"]
    )
    refs["endpoint_estimand_registry"] = {
        "relative_path": estimand_path.relative_to(root).as_posix(),
        "sha256": estimand_sha,
    }

    config_path = root / "development/run_plan_inputs/executable_config_v2.json"
    payloads["executable_config"] = _config_v4_payload(
        data_sha=data_sha,
        projection_hashes=projection_hashes,
        hgb_sha=hgb_sha,
        code_tree_sha=code_tree_sha,
    )
    _write_canonical(config_path, payloads["executable_config"])
    refs["executable_config"] = _ref(root, config_path)

    inventory_path = root / "development/run_plan_inputs/row_inventory.json"
    payloads["row_inventory"] = _row_inventory_payload(
        data_sha=data_sha,
        projection_hashes=projection_hashes,
    )
    _write_canonical(inventory_path, payloads["row_inventory"])
    refs["row_inventory"] = _ref(root, inventory_path)

    release_path = root / "development/releases/development_release.json"
    payloads["independent_development_release"] = _release_payload(
        refs=refs,
        projection_hashes=projection_hashes,
        code_tree_sha=code_tree_sha,
    )
    _write_canonical(release_path, payloads["independent_development_release"])
    refs["independent_development_release"] = _ref(root, release_path)

    return PreparedInputs(root, refs, payloads, code_tree_sha)


def _assert_hold(callable_, code: str) -> plan_module.DevelopmentRunPlanError:
    with pytest.raises(plan_module.DevelopmentRunPlanError) as caught:
        callable_()
    assert caught.value.code == code
    return caught.value


def test_module_has_no_legacy_training_or_accelerator_imports() -> None:
    tree = ast.parse(Path(plan_module.__file__).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    assert not any(name.endswith("run_ordinary") for name in imported)
    assert not any(name.endswith("models") for name in imported)
    assert not ({"torch", "numpy", "sklearn"} & imported)


def test_complete_v4_build_is_deterministic_exact_eight_and_no_execution(
    prepared: PreparedInputs,
) -> None:
    first = prepared.build()
    second = prepared.build()
    assert first == second
    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    assert first["eligible_baseline_method_order"] == BASELINE_IDS
    assert first["seeds"] == [11, 23, 42, 67, 101]
    assert first["capability_scope"] == (
        "ORDINARY_TRAIN_AND_ORDINARY_DEVELOPMENT"
    )
    assert first["execution_state"] == {
        "ordinary_only": True,
        "training_ready": False,
        "execution_authorized": False,
        "external_trust_root_verified": False,
        "development_capability_bound": False,
        "training_executed": False,
        "gpu_used": False,
        "network_used": False,
        "unreleased_partition_access": False,
        "unreleased_outcome_access": False,
        "future_stage_access": False,
        "publication_claim_authorized": False,
    }
    assert first["status"] == "DEVELOPMENT_PLAN_SCAFFOLD_NOT_AUTHORIZED"
    assert first["planning_budget_summary"]["total_fit_unit_upper_bound"] == 174
    assert first["planning_budget_summary"]["implementation_dry_run_completed"] is False
    assert {
        partition["partition"]
        for item in first["estimand_slices"]
        for partition in item["partitions"]
    } == {"ORDINARY_TRAIN", "ORDINARY_DEVELOPMENT"}
    core = dict(first)
    internal_sha = core.pop("plan_sha256")
    assert internal_sha == canonical_sha256(core)


def test_paired_policy_exact_order_is_authoritative_not_set_or_alphabetical(
    prepared: PreparedInputs,
) -> None:
    assert BASELINE_IDS == [
        "ZERO_DELTA",
        "SOURCE_COMPONENT_BALANCED_TRAIN_PRIOR",
        "EDIT_ONLY_RIDGE",
        "CANDIDATE_KMER_RIDGE",
        "PAIRED_LOCAL_GLOBAL_KMER_RIDGE",
        "PAIRED_FEATURE_NEAREST_NEIGHBOR",
        "HISTOGRAM_GRADIENT_BOOSTING",
        "FULL_PAIR_UNRESTRICTED_SIAMESE",
    ]
    assert BASELINE_IDS != sorted(BASELINE_IDS)
    reordered = copy.deepcopy(prepared.payloads["paired_baseline_policy"])
    reordered["eligible_baseline_method_ids"] = sorted(BASELINE_IDS)
    reordered["required_baseline_method_ids"] = sorted(BASELINE_IDS)
    prepared.replace("paired_baseline_policy", reordered)
    _assert_hold(prepared.build, "HOLD_DEVELOPMENT_RUN_PLAN_DEPENDENCY")


def test_writer_is_exclusive_and_loader_exact_rebuilds_sources(
    prepared: PreparedInputs,
) -> None:
    plan = prepared.build()
    path = prepared.root / "development/run_plans/RUN_PLAN.json"
    digest = plan_module.write_development_run_plan_exclusive(
        prepared.root, path, plan
    )
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    assert plan_module.load_development_run_plan_exact(
        prepared.root, path, digest
    ) == plan
    _assert_hold(
        lambda: plan_module.write_development_run_plan_exclusive(
            prepared.root, path, plan
        ),
        "HOLD_DEVELOPMENT_RUN_PLAN_WRITE",
    )


def test_v1_config_holds_without_using_defaults_and_creates_no_output(
    prepared: PreparedInputs,
) -> None:
    v1 = _config_v1_payload(
        data_sha=prepared.payloads["row_inventory"][
            "train_development_outcomes_sha256"
        ],
        projection_hashes=prepared.payloads["row_inventory"][
            "development_projection_hashes"
        ],
        hgb_sha=prepared.refs["hgb_qualification"]["sha256"],
    )
    prepared.replace("executable_config", v1)
    output = prepared.root / "development/run_plans/v1-must-not-exist.json"
    _assert_hold(
        prepared.build,
        "HOLD_DEVELOPMENT_EXECUTABLE_CONFIG_V1_INCOMPLETE",
    )
    assert not output.exists()


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload["model"].__setitem__("unknown_dimension", 7),
        lambda payload: payload["resource_budget"].__setitem__(
            "unknown_budget", 1
        ),
    ],
)
def test_unknown_config_keys_fail_closed(
    prepared: PreparedInputs, mutation
) -> None:
    payload = copy.deepcopy(prepared.payloads["executable_config"])
    mutation(payload)
    prepared.replace("executable_config", payload)
    _assert_hold(prepared.build, "HOLD_DEVELOPMENT_RUN_PLAN_SCHEMA")


def test_v4_config_freezes_distinct_scalar_and_hgb_weight_normalizations(
    prepared: PreparedInputs,
) -> None:
    config = prepared.payloads["executable_config"]
    assert config["schema_version"] == "r3x_development_executable_config_v4"
    assert plan_module.RUN_PLAN_SCHEMA == "r3x_development_run_plan_v4"
    assert config["ridge_policy"]["training_weight_policy"] == (
        plan_module.FORMAL_SCALAR_WEIGHT_POLICY
    )
    assert config["ridge_policy"]["weighted_risk_normalization"] == (
        "SUM_TO_ONE"
    )
    assert config["ridge_policy"]["selection_metric"] == (
        plan_module.RIDGE_CV_SCORING_POLICY
    )
    assert config["prior_policy"] == {
        "training_weight_policy": plan_module.FORMAL_SCALAR_WEIGHT_POLICY,
        "weighted_risk_normalization": "SUM_TO_ONE",
        "aggregation_policy": plan_module.PRIOR_TRAINING_POLICY,
    }
    assert config["nearest_neighbor_policy"]["training_weight_policy"] == (
        plan_module.FORMAL_SCALAR_WEIGHT_POLICY
    )
    assert config["nearest_neighbor_policy"]["weighted_risk_normalization"] == (
        "SUM_TO_ONE"
    )
    assert config["hgb_policy"]["training_weight_policy"] == (
        plan_module.HGB_TRAINING_WEIGHT_POLICY
    )
    assert config["hgb_policy"]["weight_normalization"] == (
        plan_module.HGB_WEIGHT_NORMALIZATION
    )


@pytest.mark.parametrize(
    ("section", "field", "forged"),
    [
        ("ridge_policy", "training_weight_policy", "ROW_WEIGHTED"),
        ("prior_policy", "training_weight_policy", "ROW_WEIGHTED"),
        ("prior_policy", "aggregation_policy", "GLOBAL_ROW_MEAN"),
        ("ridge_policy", "weighted_risk_normalization", "SUM_TO_N_ROWS"),
        ("ridge_policy", "selection_metric", "EQUAL_FOLD_MEAN_MSE"),
        ("nearest_neighbor_policy", "scaling", "TRAIN_ONLY_STANDARDIZATION"),
        (
            "nearest_neighbor_policy",
            "tie_aggregation",
            "MATH_FSUM_CANONICAL_TRAIN_ROW_ORDER_MEAN_TARGET",
        ),
        ("hgb_policy", "weight_normalization", "SUM_TO_ONE"),
    ],
)
def test_weighting_policy_drift_fails_closed(
    prepared: PreparedInputs,
    section: str,
    field: str,
    forged: str,
) -> None:
    payload = copy.deepcopy(prepared.payloads["executable_config"])
    payload[section][field] = forged
    prepared.replace("executable_config", payload)
    _assert_hold(prepared.build, "HOLD_DEVELOPMENT_RUN_PLAN_SCHEMA")


def test_v3_config_schema_cannot_silently_acquire_v4_weighting_semantics(
    prepared: PreparedInputs,
) -> None:
    payload = copy.deepcopy(prepared.payloads["executable_config"])
    payload["schema_version"] = "r3x_development_executable_config_v3"
    prepared.replace("executable_config", payload)
    _assert_hold(prepared.build, "HOLD_DEVELOPMENT_RUN_PLAN_SCHEMA")


def test_zero_primary_sign_loss_and_dropout_one_fail_closed(
    prepared: PreparedInputs,
) -> None:
    payload = copy.deepcopy(prepared.payloads["executable_config"])
    payload["loss"]["sign_weight"] = 0.0
    prepared.replace("executable_config", payload)
    _assert_hold(prepared.build, "HOLD_DEVELOPMENT_RUN_PLAN_CONFIG")

    payload = copy.deepcopy(prepared.payloads["executable_config"])
    payload["loss"]["sign_weight"] = 1.0
    payload["training"]["condition_unknown_dropout"] = 1.0
    prepared.replace("executable_config", payload)
    _assert_hold(prepared.build, "HOLD_DEVELOPMENT_RUN_PLAN_SCHEMA")


def test_fit_unit_budget_and_hgb_per_estimand_support_are_enforced(
    prepared: PreparedInputs,
) -> None:
    payload = copy.deepcopy(prepared.payloads["executable_config"])
    payload["resource_budget"]["max_total_fit_units"] = 1
    prepared.replace("executable_config", payload)
    _assert_hold(prepared.build, "HOLD_DEVELOPMENT_RUN_PLAN_RESOURCE_BUDGET")

    payload = copy.deepcopy(prepared.payloads["executable_config"])
    payload["resource_budget"]["max_total_fit_units"] = 256
    prepared.replace("executable_config", payload)
    inventory = copy.deepcopy(prepared.payloads["row_inventory"])
    removable = next(
        row
        for row in inventory["rows"]
        if row["study"] == "ordinary_study_a"
        and row["partition"] == "ORDINARY_TRAIN"
    )
    inventory["rows"].remove(removable)
    inventory["partition_row_set_sha256"]["ORDINARY_TRAIN"] = canonical_sha256(
        sorted(
            row["row_id"]
            for row in inventory["rows"]
            if row["partition"] == "ORDINARY_TRAIN"
        )
    )
    prepared.replace("row_inventory", inventory)
    _assert_hold(prepared.build, "HOLD_DEVELOPMENT_RUN_PLAN_HGB_SUPPORT")


def test_source_group_cannot_cross_train_and_development(
    prepared: PreparedInputs,
) -> None:
    inventory = copy.deepcopy(prepared.payloads["row_inventory"])
    train_group = next(
        row["source_group_id_v2"]
        for row in inventory["rows"]
        if row["partition"] == "ORDINARY_TRAIN"
    )
    development_row = next(
        row
        for row in inventory["rows"]
        if row["partition"] == "ORDINARY_DEVELOPMENT"
    )
    development_row["source_group_id_v2"] = train_group
    prepared.replace("row_inventory", inventory)
    _assert_hold(prepared.build, "HOLD_DEVELOPMENT_RUN_PLAN_BINDING")


def test_standalone_validator_requires_frozen_order_and_selector_schema(
    prepared: PreparedInputs,
) -> None:
    plan = prepared.build()
    plan["eligible_baseline_method_order"] = list(
        reversed(plan["eligible_baseline_method_order"])
    )
    core = dict(plan)
    core.pop("plan_sha256")
    plan["plan_sha256"] = canonical_sha256(core)
    _assert_hold(
        lambda: plan_module.validate_development_run_plan(plan),
        "HOLD_DEVELOPMENT_RUN_PLAN_BASELINE_SET",
    )

    plan = prepared.build()
    plan["scientific_policy"]["selector_schema"] = "forged_selector_v999"
    core = dict(plan)
    core.pop("plan_sha256")
    plan["plan_sha256"] = canonical_sha256(core)
    _assert_hold(
        lambda: plan_module.validate_development_run_plan(plan),
        "HOLD_DEVELOPMENT_RUN_PLAN_SCHEMA",
    )


@pytest.mark.parametrize(
    ("location", "key", "value"),
    [
        ("resource_budget", "internal\u200b_test_support", 0),
        ("resource_budget", "ordinary_note", "ＳＥＡＬＥＤ"),
        ("resource_budget", "down\u200bstream_hash", False),
        ("resource_budget", "protec\u200bted_path", "none"),
    ],
)
def test_unicode_and_nested_protected_tokens_fail_closed(
    prepared: PreparedInputs,
    location: str,
    key: str,
    value: object,
) -> None:
    payload = copy.deepcopy(prepared.payloads["executable_config"])
    payload[location][key] = value
    prepared.replace("executable_config", payload)
    _assert_hold(
        prepared.build, "HOLD_DEVELOPMENT_RUN_PLAN_PROTECTED_CONTENT"
    )


def test_protected_or_escaping_reference_path_fails_before_open(
    prepared: PreparedInputs,
) -> None:
    refs = copy.deepcopy(prepared.refs)
    refs["row_inventory"]["relative_path"] = (
        "development/run_plan_inputs/internal\u200b_test_support.json"
    )
    _assert_hold(
        lambda: plan_module.build_development_run_plan(
            prepared.root,
            source_refs=refs,
            expected_development_release_sha256=prepared.expected_release_sha,
        ),
        "HOLD_DEVELOPMENT_RUN_PLAN_PROTECTED_CONTENT",
    )

    refs = copy.deepcopy(prepared.refs)
    refs["row_inventory"]["relative_path"] = "../escape.json"
    _assert_hold(
        lambda: plan_module.build_development_run_plan(
            prepared.root,
            source_refs=refs,
            expected_development_release_sha256=prepared.expected_release_sha,
        ),
        "HOLD_DEVELOPMENT_RUN_PLAN_PATH",
    )


def test_duplicate_json_keys_are_rejected(prepared: PreparedInputs) -> None:
    path = prepared.root / prepared.refs["executable_config"]["relative_path"]
    raw = (
        b'{"schema_version":"r3x_development_executable_config_v2",'
        b'"schema_version":"r3x_development_executable_config_v2"}'
    )
    path.write_bytes(raw)
    prepared.refs["executable_config"] = _ref(prepared.root, path)
    prepared.refresh_release()
    _assert_hold(prepared.build, "HOLD_DEVELOPMENT_RUN_PLAN_JSON")

    duplicate_plan = prepared.root / "development/run_plans/duplicate.json"
    duplicate_raw = b'{"schema_version":"a","schema_version":"b"}'
    duplicate_plan.write_bytes(duplicate_raw)
    _assert_hold(
        lambda: plan_module.load_development_run_plan_exact(
            prepared.root,
            duplicate_plan,
            hashlib.sha256(duplicate_raw).hexdigest(),
        ),
        "HOLD_DEVELOPMENT_RUN_PLAN_JSON",
    )


@pytest.mark.parametrize("kind", ["symlink", "hardlink"])
def test_input_symlink_and_hardlink_are_rejected(
    prepared: PreparedInputs, kind: str
) -> None:
    original = prepared.root / prepared.refs["executable_config"]["relative_path"]
    alias = original.with_name(f"config-{kind}.json")
    if kind == "symlink":
        alias.symlink_to(original.name)
        prepared.refs["executable_config"] = {
            "relative_path": alias.relative_to(prepared.root).as_posix(),
            "sha256": prepared.refs["executable_config"]["sha256"],
        }
        prepared.refresh_release()
    else:
        os.link(original, alias)
    _assert_hold(prepared.build, "HOLD_DEVELOPMENT_RUN_PLAN_PATH")


def test_output_symlink_parent_is_rejected(prepared: PreparedInputs) -> None:
    plan = prepared.build()
    real_parent = prepared.root / "development/run_plans"
    pinned = prepared.root / "development/run_plans-pinned"
    outside = prepared.root.parent / "outside-run-plans"
    outside.mkdir()
    real_parent.rename(pinned)
    real_parent.symlink_to(outside, target_is_directory=True)
    output = real_parent / "RUN_PLAN.json"
    _assert_hold(
        lambda: plan_module.write_development_run_plan_exclusive(
            prepared.root, output, plan
        ),
        "HOLD_DEVELOPMENT_RUN_PLAN_PATH",
    )
    assert not (outside / output.name).exists()


@pytest.mark.parametrize(
    "missing_source",
    ["independent_development_release", "hgb_qualification"],
)
def test_missing_release_or_hgb_holds_before_output_creation(
    prepared: PreparedInputs, missing_source: str
) -> None:
    plan = prepared.build()
    (prepared.root / prepared.refs[missing_source]["relative_path"]).unlink()
    output = prepared.root / f"development/run_plans/no-{missing_source}.json"
    _assert_hold(
        lambda: plan_module.write_development_run_plan_exclusive(
            prepared.root, output, plan
        ),
        "HOLD_DEVELOPMENT_RUN_PLAN_DEPENDENCY",
    )
    assert not output.exists()


def test_source_projection_alias_is_rejected(prepared: PreparedInputs) -> None:
    inventory = copy.deepcopy(prepared.payloads["row_inventory"])
    inventory["source_authority_sha256"][ROW_SOURCE_NAMES[0]] = inventory[
        "development_projection_hashes"
    ][PROJECTION_NAMES[0]]
    prepared.replace("row_inventory", inventory)
    _assert_hold(prepared.build, "HOLD_DEVELOPMENT_RUN_PLAN_BINDING")


@pytest.mark.parametrize("changed", ["row", "config", "authority"])
def test_bound_input_change_changes_plan_sha(
    prepared: PreparedInputs, changed: str
) -> None:
    before = prepared.build()
    if changed == "row":
        inventory = copy.deepcopy(prepared.payloads["row_inventory"])
        inventory["rows"][0]["source_group_id_v2"] = "POOL_CHANGED"
        prepared.replace("row_inventory", inventory)
    elif changed == "config":
        config = copy.deepcopy(prepared.payloads["executable_config"])
        config["training"]["epochs"] += 1
        prepared.replace("executable_config", config)
    else:
        paired = copy.deepcopy(prepared.payloads["paired_baseline_policy"])
        paired["source_authority_sha256"] = _sha("source:paired-policy-changed")
        prepared.replace("paired_baseline_policy", paired)
    after = prepared.build()
    assert before["plan_sha256"] != after["plan_sha256"]


def test_projection_change_without_all_authority_rebinding_holds(
    prepared: PreparedInputs,
) -> None:
    inventory = copy.deepcopy(prepared.payloads["row_inventory"])
    inventory["development_projection_hashes"][PROJECTION_NAMES[0]] = _sha(
        "projection:changed"
    )
    prepared.replace("row_inventory", inventory)
    _assert_hold(prepared.build, "HOLD_DEVELOPMENT_RUN_PLAN_BINDING")


def test_plan_validator_rejects_unknown_or_protected_output_fields(
    prepared: PreparedInputs,
) -> None:
    plan = prepared.build()
    plan["unknown"] = False
    _assert_hold(
        lambda: plan_module.validate_development_run_plan(plan),
        "HOLD_DEVELOPMENT_RUN_PLAN_SCHEMA",
    )

    plan = prepared.build()
    plan["protec\u200bted_hash"] = False
    _assert_hold(
        lambda: plan_module.validate_development_run_plan(plan),
        "HOLD_DEVELOPMENT_RUN_PLAN_PROTECTED_CONTENT",
    )
