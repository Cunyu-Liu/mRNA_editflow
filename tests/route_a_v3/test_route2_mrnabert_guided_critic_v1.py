from __future__ import annotations

from pathlib import Path

import pytest
import torch

from core.route2_delta_predictor import Route2PretrainedEditCenteredDeltaPredictor
from core.route2_legal_xeditflow import apply_action, initial_state, legal_actions
from scripts.route_a_v3.route2_mrnabert_guided_critic_v1 import (
    FrozenRoute2MRNABERTCritic,
    GuidedCriticError,
)


class FakeEncoder:
    embedding_width = 8
    parameter_count = 113_389_056

    def __init__(
        self,
        _model_path,
        _device,
        *,
        attention_backend="OFFICIAL_PYTORCH_FALLBACK",
    ):
        self.calls = 0
        self.attention_backend = attention_backend

    def encode_sequences(self, sequences):
        self.calls += 1
        rows = []
        for sequence in sequences:
            counts = [sequence.count(base) for base in "ACGU"]
            rows.append(torch.tensor(counts + counts, dtype=torch.float32))
        return torch.stack(rows)

    def clear_cache(self):
        return None


def _checkpoint(path: Path, *, result_stage="FINAL_ALL_DEVELOPMENT_REFIT") -> None:
    model_config = {
        "hidden_dim": 16,
        "depth": 1,
        "study_count": 1,
        "assay_count": 2,
        "context_count": 2,
        "endpoint_count": 2,
        "pretrained_width": 8,
        "learned_uncertainty": False,
        "source_only_control": False,
    }
    model = Route2PretrainedEditCenteredDeltaPredictor(**model_config)
    torch.save(
        {
            "model_kind": "delta_pretrained_mrnabert_edit_centered_antisymmetric",
            "model_config": model_config,
            "model_state": model.state_dict(),
            "vocabs": {
                "study": {"__UNK__": 0},
                "assay": {"__UNK__": 0, "A": 1},
                "context": {"__UNK__": 0, "C": 1},
                "endpoint": {"__UNK__": 0, "E": 1},
            },
            "training_provenance": {
                "result_stage": result_stage,
                "optimizer_steps": 10,
                "parameter_changed": True,
                "cuda_training_tensors_verified": True,
                "cpu_fallback_used": False,
            },
            "selection_provenance": {"checkpoint_selection": "FINAL_EPOCH"},
        },
        path,
    )


def test_scores_novel_sub_candidates_and_identity_is_zero(tmp_path: Path) -> None:
    checkpoint = tmp_path / "critic.pt"
    _checkpoint(checkpoint)
    critic = FrozenRoute2MRNABERTCritic(
        checkpoint,
        tmp_path,
        torch.device("cpu"),
        encoder_class=FakeEncoder,
    )
    values = critic.score_candidates(
        "AAAA",
        ["AAAA", "CAAA", "GAAA"],
        assay_id="A",
        context_id="C",
        endpoint_id="E",
        region="3UTR",
    )
    assert values[0] == pytest.approx(0.0, abs=1e-6)
    assert len(values) == 3
    assert all(-5.0 <= value <= 5.0 for value in values)


def test_state_potential_is_memoized(tmp_path: Path) -> None:
    checkpoint = tmp_path / "critic.pt"
    _checkpoint(checkpoint)
    critic = FrozenRoute2MRNABERTCritic(
        checkpoint, tmp_path, torch.device("cpu"), encoder_class=FakeEncoder
    )
    state = initial_state("AAAA", budget=1, assay_id="A", context_id="C")
    first = critic.potential(state, endpoint_id="E", region="3UTR")
    second = critic.potential(state, endpoint_id="E", region="3UTR")
    assert first == second
    assert critic.cached_potential_count == 1
    assert critic.encoder.calls == 1
    assert critic.model_batch_forward_count == 1
    assert critic.candidate_forward_equivalent_count == 1
    critic.clear_source_caches()
    assert critic.cached_potential_count == 0
    assert critic.encoder.calls == 1
    third = critic.potential(state, endpoint_id="E", region="3UTR")
    assert third == first
    assert critic.model_batch_forward_count == 2
    assert critic.candidate_forward_equivalent_count == 2


def test_multiple_child_potentials_are_scored_in_one_batch(tmp_path: Path) -> None:
    checkpoint = tmp_path / "critic.pt"
    _checkpoint(checkpoint)
    critic = FrozenRoute2MRNABERTCritic(
        checkpoint, tmp_path, torch.device("cpu"), encoder_class=FakeEncoder
    )
    root = initial_state("AA", budget=1, assay_id="A", context_id="C")
    children = [
        apply_action(root, action) for action in legal_actions(root)
    ]
    values = critic.potentials(
        [root, *children], endpoint_id="E", region="3UTR"
    )
    assert len(values) == 1 + len(children)
    assert critic.model_batch_forward_count == 1
    assert critic.candidate_forward_equivalent_count == len({
        root.current_sequence, *(child.current_sequence for child in children)
    })


def test_nonfinal_or_length_changing_candidate_is_rejected(tmp_path: Path) -> None:
    checkpoint = tmp_path / "critic.pt"
    _checkpoint(checkpoint, result_stage="FROZEN_DEVELOPMENT_VALIDATION")
    with pytest.raises(GuidedCriticError, match="final all-Development"):
        FrozenRoute2MRNABERTCritic(
            checkpoint, tmp_path, torch.device("cpu"), encoder_class=FakeEncoder
        )

    _checkpoint(checkpoint)
    critic = FrozenRoute2MRNABERTCritic(
        checkpoint, tmp_path, torch.device("cpu"), encoder_class=FakeEncoder
    )
    with pytest.raises(GuidedCriticError, match="SUB candidates only"):
        critic.score_candidates(
            "AAAA",
            ["AAA"],
            assay_id="A",
            context_id="C",
            endpoint_id="E",
            region="3UTR",
        )
