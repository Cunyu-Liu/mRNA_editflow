from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/route_a_v3/run_route2_search_generation_baselines_v1.py"


def _load():
    spec = importlib.util.spec_from_file_location("route2_search_baselines_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _score(sequence: str) -> float:
    weights = {"A": 0.0, "C": 1.0, "G": 2.0, "U": 3.0}
    return sum(weights[base] for base in sequence)


@pytest.mark.parametrize(
    "method",
    ["random_legal", "greedy", "beam", "genetic", "local_search", "generate_then_rerank"],
)
def test_search_adapters_are_deterministic_legal_and_budgeted(method: str) -> None:
    module = _load()
    kwargs = dict(
        edit_budget=2,
        candidate_budget=4,
        max_critic_forwards=40,
        score_function=_score,
        seed=17,
        beam_width=3,
        population_size=6,
        oversample_factor=3,
    )
    first = module.run_search_method(method, "AAAA", **kwargs)
    second = module.run_search_method(method, "AAAA", **kwargs)
    assert first == second
    assert 1 <= len(first.candidates) <= 4
    assert len(first.candidates) == len(set(first.candidates))
    assert all(module.legal_candidate("AAAA", sequence, 2) for sequence in first.candidates)
    assert first.critic_forwards <= 40
    assert first.generator_nfe == 0
    assert first.proposal_count >= first.critic_forwards - 1
    assert first.source_score == _score("AAAA")


def test_exhaustive_finds_global_optimum_when_space_is_covered() -> None:
    module = _load()
    size = module.legal_space_size(2, 2)
    result = module.run_search_method(
        "exhaustive",
        "AA",
        edit_budget=2,
        candidate_budget=2,
        max_critic_forwards=size,
        score_function=_score,
        seed=1,
    )
    assert result.candidates[0] == "UU"
    assert result.critic_forwards == size == 16


def test_exhaustive_refuses_unmatched_partial_coverage() -> None:
    module = _load()
    with pytest.raises(module.SearchBaselineError, match="requires a forward budget"):
        module.run_search_method(
            "exhaustive",
            "AAAA",
            edit_budget=2,
            candidate_budget=2,
            max_critic_forwards=10,
            score_function=_score,
            seed=1,
        )


def test_monotone_neighbors_never_reedit_or_revert() -> None:
    module = _load()
    neighbors = module.monotone_neighbors("AAAA", "CAAA", 2)
    assert len(neighbors) == 9
    assert all(sequence[0] == "C" for sequence in neighbors)
    assert all(module.edit_count("AAAA", sequence) == 2 for sequence in neighbors)


def test_checkpoint_scorer_rejects_random_untrained_checkpoint(tmp_path: Path, monkeypatch) -> None:
    module = _load()
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    path = tmp_path / "checkpoint.pt"
    torch.save({"training_provenance": {}}, path)
    with pytest.raises(module.SearchBaselineError, match="does not prove"):
        module.TorchCheckpointScorer(path, "cuda:0")


def test_scoring_execution_provenance_distinguishes_cuda_from_precomputed_scores(
    monkeypatch,
) -> None:
    module = _load()
    monkeypatch.setattr(module.torch.cuda, "device_count", lambda: 8)
    monkeypatch.setattr(
        module,
        "cuda_device_observation",
        lambda index, require_physical_index_match: {
            "cuda_device_index": index,
            "cuda_device_uuid": "GPU-test",
            "cuda_total_memory_mb": 40960.0,
        },
    )
    checkpoint = module.scoring_execution_provenance(True, "cuda:2", 2)
    assert checkpoint["critic_scoring_execution"] == "CUDA_CHECKPOINT"
    assert checkpoint["device"] == "cuda:2"
    assert checkpoint["physical_gpu_index"] == 2
    assert checkpoint["cpu_fallback_used"] is False
    assert checkpoint["cuda_device_index"] == 2
    assert checkpoint["cuda_total_memory_mb"] > 0
    assert checkpoint["cuda_device_uuid"]
    precomputed = module.scoring_execution_provenance(False, None, None)
    assert precomputed == {
        "critic_scoring_execution": "PRECOMPUTED_SCORE_TABLE",
        "device": None,
        "physical_gpu_index": None,
        "cpu_fallback_used": None,
    }
    with pytest.raises(module.SearchBaselineError, match="requires CUDA"):
        module.scoring_execution_provenance(True, "cpu", 0)


def test_checkpoint_search_requires_gpu_training_provenance_and_preserves_failure_evidence() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'provenance.get("cuda_training_tensors_verified") is True' in source
    assert "write_gpu_failure_evidence(" in source


def test_checkpoint_search_requires_frozen_observed_cuda_provenance() -> None:
    module = _load()
    provenance = {
        "result_stage": "FROZEN_DEVELOPMENT_VALIDATION",
        "optimizer_steps": 10,
        "parameter_changed": True,
        "cpu_fallback_used": False,
        "cuda_training_tensors_verified": True,
        "physical_gpu_index": 2,
        "device": "cuda:2",
        "cuda_device_index": 2,
        "cuda_device_uuid": "GPU-frozen",
        "cuda_total_memory_mb": 40960.0,
    }
    module.validate_frozen_checkpoint_provenance(provenance)
    for field, value in (("result_stage", "FROZEN_DEVELOPMENT_TEST"), ("cuda_device_uuid", None)):
        invalid = dict(provenance)
        invalid[field] = value
        with pytest.raises(module.SearchBaselineError, match="frozen learned GPU"):
            module.validate_frozen_checkpoint_provenance(invalid)


def test_search_hyperparameters_are_explicit_and_positive() -> None:
    module = _load()
    values = module.validated_search_hyperparameters(
        beam_width=16,
        genetic_population_size=32,
        oversample_factor=8,
        exhaustive_space_limit=4096,
    )
    assert values == {
        "beam_width": 16,
        "genetic_population_size": 32,
        "oversample_factor": 8,
        "exhaustive_space_limit": 4096,
    }
    with pytest.raises(module.SearchBaselineError, match="positive integers"):
        module.validated_search_hyperparameters(
            beam_width=0,
            genetic_population_size=32,
            oversample_factor=8,
            exhaustive_space_limit=4096,
        )


def test_budgeted_scorer_batches_missing_candidates_and_counts_equivalents() -> None:
    module = _load()

    class BatchedScore:
        def __init__(self):
            self.batches = []

        def __call__(self, sequence):
            raise AssertionError("single scoring should not be used")

        def score_many(self, sequences):
            self.batches.append(tuple(sequences))
            return [_score(sequence) for sequence in sequences]

    function = BatchedScore()
    scorer = module.BudgetedScorer(function, max_forwards=3)
    values = scorer.score_available(["AAAA", "CAAA", "GAAA", "UAAA"])
    assert [sequence for sequence, _score_value in values] == ["AAAA", "CAAA", "GAAA"]
    assert function.batches == [("AAAA", "CAAA", "GAAA")]
    assert scorer.forward_count == 3
    assert scorer.score_available(["AAAA", "CAAA"])[0][1] == _score("AAAA")
    assert len(function.batches) == 1


def test_guided_per_source_budget_table_is_exact_and_positive(tmp_path: Path) -> None:
    module = _load()
    path = tmp_path / "guided_compute.jsonl"
    path.write_text(
        '{"source_key":"S1","matched_search_critic_forward_budget":17}\n'
        '{"source_key":"S2","matched_search_critic_forward_budget":29}\n',
        encoding="utf-8",
    )
    assert module.load_critic_budgets_by_source(path) == {"S1": 17, "S2": 29}
    path.write_text(
        '{"source_key":"S1","matched_search_critic_forward_budget":0}\n',
        encoding="utf-8",
    )
    with pytest.raises(module.SearchBaselineError, match="budget is invalid"):
        module.load_critic_budgets_by_source(path)
