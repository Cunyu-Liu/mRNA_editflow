from __future__ import annotations

import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/route_a_v3/build_route2_mrnabert_guidance_readiness_input_v1.py"
POLICY = ROOT / "configs/route_a_v3_route2_mrnabert_guidance_reward_policy_v1.json"


def _load():
    spec = importlib.util.spec_from_file_location("readiness_input_builder_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _kwargs(tmp_path: Path):
    checkpoint = tmp_path / "delta_predictor_checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    return {
        "validation_training_summary": {"result_stage": "FROZEN_DEVELOPMENT_VALIDATION"},
        "final_refit_summary": {"result_stage": "FINAL_ALL_DEVELOPMENT_REFIT"},
        "final_refit_checkpoint": checkpoint,
        "signal_adjudication": {"status": "signal"},
        "loso_results": [
            {"seed": seed, "status": "LOSO_MODEL_BASELINE_ALIGNED_COMPLETE"}
            for seed in (20260822, 20260823, 20260824)
        ],
        "flow_training_summary": {"status": "flow training"},
        "flow_validation_summary": {"status": "flow validation"},
        "reward_policy": json.loads(POLICY.read_text(encoding="utf-8")),
        "online_encoder_validation": {
            "schema_version": "route_a_v3_route2_mrnabert_online_encoder_validation.v1",
            "status": "ONLINE_FROZEN_MRNABERT_MATCHES_CANONICAL_CACHE",
            "novel_candidate_encoding_supported": True,
            "frozen_parameter_count": 113389056,
            "evaluation_records_read": 0,
            "maximum_absolute_difference": 0.004,
            "absolute_tolerance": 0.01,
        },
    }


def test_builds_complete_readiness_input_without_evaluation(tmp_path: Path) -> None:
    module = _load()
    payload = module.build_input(**_kwargs(tmp_path))
    critic = payload["critic"]
    assert critic["critic_checkpoint_frozen"] is True
    assert critic["reward_calibration_policy_frozen"] is True
    assert critic["generated_candidate_online_encoder_ready"] is True
    assert critic["evaluation_records_used_for_training_hpo_threshold_or_reward"] == 0
    assert [row["seed"] for row in critic["loso_seed_results"]] == [
        20260822, 20260823, 20260824
    ]


def test_learned_uncertainty_or_mutable_critic_is_not_a_frozen_policy(
    tmp_path: Path,
) -> None:
    module = _load()
    for key, value in (
        ("uncertainty_in_guidance", "LEARNED_LOG_VARIANCE"),
        ("critic_parameter_update_during_generation", True),
    ):
        kwargs = _kwargs(tmp_path)
        kwargs["reward_policy"] = deepcopy(kwargs["reward_policy"])
        kwargs["reward_policy"][key] = value
        payload = module.build_input(**kwargs)
        assert payload["critic"]["reward_calibration_policy_frozen"] is False


def test_online_encoder_must_match_cache_and_support_novel_candidates(
    tmp_path: Path,
) -> None:
    module = _load()
    for key, value in (
        ("novel_candidate_encoding_supported", False),
        ("maximum_absolute_difference", 0.02),
        ("evaluation_records_read", 1),
    ):
        kwargs = _kwargs(tmp_path)
        kwargs["online_encoder_validation"] = deepcopy(
            kwargs["online_encoder_validation"]
        )
        kwargs["online_encoder_validation"][key] = value
        payload = module.build_input(**kwargs)
        assert payload["critic"]["generated_candidate_online_encoder_ready"] is False


def test_missing_checkpoint_or_wrong_seed_set_fails(tmp_path: Path) -> None:
    module = _load()
    kwargs = _kwargs(tmp_path)
    kwargs["final_refit_checkpoint"].unlink()
    try:
        module.build_input(**kwargs)
    except module.ReadinessInputError as exc:
        assert "checkpoint is absent" in str(exc)
    else:
        raise AssertionError("missing checkpoint was accepted")

    kwargs = _kwargs(tmp_path)
    kwargs["loso_results"][2]["seed"] = 999
    try:
        module.build_input(**kwargs)
    except module.ReadinessInputError as exc:
        assert "seed set differs" in str(exc)
    else:
        raise AssertionError("wrong LOSO seed set was accepted")
