from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts.r3x.action_support import (
    ActionSupportHoldError,
    load_action_support_policy,
    validate_action_support_policy,
)
from scripts.r3x.models import ModelConfig, SparseEditSetCritic


def policy() -> dict:
    return {
        "schema_version": "r3_action_support_policy_v1",
        "status": "FROZEN",
        "qualification_status": "PUBLICATION_QUALIFIED",
        "authority_hashes": {
            "ready_dataset_sha256": "1" * 64,
            "inclusion_manifest_sha256": "2" * 64,
            "context_source_crosswalk_sha256": "3" * 64,
            "endpoint_estimand_registry_sha256": "4" * 64,
            "source_component_split_manifest_sha256": "5" * 64,
        },
        "thresholds": {
            "min_independent_source_components_per_action": 20,
            "min_study_context_cells_per_action": 2,
            "max_single_component_dominance": 0.20,
        },
        "observed": {
            "min_independent_source_components_per_action": 25,
            "min_study_context_cells_per_action": 3,
            "max_single_component_dominance": 0.10,
            "qualified_action_strata": 8,
        },
    }


def test_publication_policy_must_be_frozen_hash_bound_and_meet_thresholds() -> None:
    out = validate_action_support_policy(policy())
    assert out["status"] == "PUBLICATION_QUALIFIED"
    for mutation in (
        lambda p: p.__setitem__("status", "PROPOSED"),
        lambda p: p["authority_hashes"].__setitem__("ready_dataset_sha256", "bad"),
        lambda p: p["observed"].__setitem__("min_independent_source_components_per_action", 1),
        lambda p: p["observed"].__setitem__("max_single_component_dominance", 0.9),
    ):
        bad = copy.deepcopy(policy())
        mutation(bad)
        with pytest.raises(ActionSupportHoldError):
            validate_action_support_policy(bad)


def test_policy_loader_binds_exact_bytes(tmp_path: Path) -> None:
    path = tmp_path / "ordinary_action_support.json"
    path.write_text(json.dumps(policy(), sort_keys=True), encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert load_action_support_policy(path, digest)["policy_sha256"] == digest
    with pytest.raises(ActionSupportHoldError, match="SHA-256"):
        load_action_support_policy(path, "0" * 64)


def test_legacy_interaction_model_is_unconditionally_not_constructible() -> None:
    cfg = ModelConfig(d_model=16, n_heads=4, n_layers=1, ff_dim=32, max_len=16,
                      n_endpoints=3, n_contexts=3)
    with pytest.raises(ActionSupportHoldError, match="NOT_RUN_ACTION_SUPPORT"):
        SparseEditSetCritic(cfg, interaction=True, action_support_policy=None)
    bad = copy.deepcopy(policy())
    bad["qualification_status"] = "DEV_AVAILABLE"
    with pytest.raises(ActionSupportHoldError):
        SparseEditSetCritic(cfg, interaction=True, action_support_policy=bad)
    with pytest.raises(ActionSupportHoldError, match="not registered"):
        SparseEditSetCritic(cfg, interaction=True, action_support_policy=policy())
    with pytest.raises(ActionSupportHoldError, match="additive B does not consume"):
        SparseEditSetCritic(cfg, interaction=False, action_support_policy=policy())
