from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
SCRIPT = ROOT / "scripts/route_a_v3/run_route2_base_flow_g0_validation_v1.py"


def _load():
    spec = importlib.util.spec_from_file_location("run_route2_base_flow_g0_validation_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _model_and_checkpoint(module):
    model = module.Route2BaseFlowModel(hidden_dim=16, assay_count=1, context_count=1)
    checkpoint = {"assay_vocab": {"__UNK__": 0}, "context_vocab": {"__UNK__": 0}}
    return model, checkpoint


def _cuda_model(module):
    if not torch.cuda.is_available():
        pytest.skip("learned G0 validation requires CUDA")
    device = torch.device(f"cuda:{int(os.environ.get('ROUTE2_TEST_CUDA_INDEX', '0'))}")
    model, checkpoint = _model_and_checkpoint(module)
    return model.to(device), checkpoint, device


def test_fixed_seed_trajectory_replays_and_is_legal() -> None:
    module = _load()
    model, checkpoint, device = _cuda_model(module)
    root = module.initial_state("ACG", budget=2, assay_id="a", context_id="c")
    rate_function = module.learned_rate_function(
        model, region_id=0, assay_id=0, context_id=0, device=device
    )
    first = module.sample_one(root, rate_function, seed=11, device=device)
    second = module.sample_one(root, rate_function, seed=11, device=device)
    assert first == second
    terminal, actions, forwards = first
    assert terminal.terminal_cause in {"EXPLICIT_STOP", "BUDGET_EXHAUSTED"}
    assert terminal.edit_count <= 2
    assert forwards == len(actions)


def test_learned_small_graph_matches_complete_path_enumeration() -> None:
    module = _load()
    model, checkpoint, device = _cuda_model(module)
    result = module.learned_small_graph_check(model, checkpoint, device)
    assert result["status"] == "PASS"
    assert result["total_variation"] <= 1e-12
    assert result["terminal_state_count"] > 0


def test_source_eligibility_refuses_credit_or_evaluation_outcomes(tmp_path: Path) -> None:
    module = _load()
    path = tmp_path / "sources.jsonl"
    base = {
        "source_key": "s",
        "source_sequence": "AAAA",
        "edit_budget": 1,
        "candidate_budget": 2,
        "region": "5UTR",
        "assay_id": "a",
        "biological_context_id": "c",
        "generated_candidates_grant_canonical_credit": False,
        "evaluation_outcomes_included": False,
    }
    import json
    path.write_text(json.dumps({**base, "generated_candidates_grant_canonical_credit": True}) + "\n")
    with pytest.raises(module.G0ValidationError, match="credit"):
        module.load_sources(path)
    path.write_text(json.dumps({**base, "evaluation_outcomes_included": True}) + "\n")
    with pytest.raises(module.G0ValidationError, match="Evaluation outcome"):
        module.load_sources(path)


def test_random_checkpoint_cannot_be_presented_as_learned_gpu_evidence(tmp_path: Path, monkeypatch) -> None:
    module = _load()
    model, checkpoint = _model_and_checkpoint(module)
    checkpoint["model_config"] = {"hidden_dim": 16, "assay_count": 1, "context_count": 1}
    checkpoint["model_state"] = model.state_dict()
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)
    sources = tmp_path / "sources.jsonl"
    import json
    sources.write_text(json.dumps({
        "source_key": "s", "source_sequence": "AAAA", "edit_budget": 1, "candidate_budget": 1,
        "region": "5UTR", "assay_id": "a", "biological_context_id": "c",
        "generated_candidates_grant_canonical_credit": False, "evaluation_outcomes_included": False,
    }) + "\n")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(module, "cuda_device_observation", lambda *args, **kwargs: {})
    monkeypatch.setattr(module, "load_model", lambda path, device: (model, checkpoint))
    with pytest.raises(module.G0ValidationError, match="does not prove"):
        module.execute({
            "device": "cuda:0", "physical_gpu_index": 0, "source_eligibility_manifest": str(sources),
            "checkpoint_path": str(checkpoint_path), "seed": 1,
        }, tmp_path / "out")


def test_checkpoint_requires_observed_cuda_training_provenance() -> None:
    module = _load()
    provenance = {
        "seed": 1,
        "optimizer_steps": 10,
        "parameter_changed": True,
        "cuda_training_tensors_verified": True,
        "physical_gpu_index": 2,
        "torch_device": "cuda:2",
        "cpu_fallback_used": False,
        "cuda_device_index": 2,
        "cuda_device_uuid": "GPU-train",
        "cuda_total_memory_mb": 40960.0,
    }
    assert module.validate_checkpoint_training_provenance(provenance) == provenance
    for field, value in (
        ("cuda_training_tensors_verified", False),
        ("cuda_device_index", 1),
        ("cuda_device_uuid", None),
        ("cuda_total_memory_mb", 0.0),
    ):
        invalid = dict(provenance)
        invalid[field] = value
        with pytest.raises(module.G0ValidationError, match="observed CUDA provenance"):
            module.validate_checkpoint_training_provenance(invalid)


def test_g0_validation_reports_empirical_generation_score() -> None:
    module = _load()
    model, checkpoint, device = _cuda_model(module)
    sources = [{
        "source_key": "s", "source_sequence": "AAAA", "edit_budget": 1, "candidate_budget": 6,
        "region": "5UTR", "assay_id": "a", "biological_context_id": "c",
    }]
    progress_rows = []
    rows, summary = module.validate(
        model, checkpoint, sources, device=device, seed=7, progress=progress_rows.append
    )
    assert summary["trajectory_sampling_device"] == str(device)
    assert all(row["critic_forwards"] == 0 for row in rows)
    assert all(row["generation_score"] <= 0.0 for row in rows)
    assert progress_rows == [{
        "event": "SOURCE_COHORT_COMPLETED",
        "completed_source_cohort_count": 1,
        "total_source_cohort_count": 1,
        "source_key": "s",
        "trajectory_count": 6,
        "generator_nfe": summary["generator_nfe"],
    }]
    by_sequence = {}
    for row in rows:
        by_sequence.setdefault(row["candidate_sequence"], set()).add(row["generation_score"])
    assert all(len(scores) == 1 for scores in by_sequence.values())
