"""Frozen final-refit mRNABERT critic for Route 2 generated candidates."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Mapping

import torch

from core.route2_delta_predictor import (
    ROUTE2_PRETRAINED_EDIT_CENTERED_MODEL_KIND,
    Route2PretrainedEditCenteredDeltaPredictor,
)
from core.route2_legal_xeditflow import FlowState
from scripts.route_a_v3.route2_mrnabert_online_encoder_v1 import (
    FrozenMRNABERTOnlineEncoder,
    normalize_rna,
)


TOKEN = {"A": 0, "C": 1, "G": 2, "U": 3}
REGION = {"5UTR": 0, "3UTR": 1}


class GuidedCriticError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GuidedCriticError(message)


def _category_id(vocab: Mapping[str, int], value: str) -> int:
    _require("__UNK__" in vocab, "critic category vocabulary lacks __UNK__")
    return int(vocab.get(value, vocab["__UNK__"]))


class FrozenRoute2MRNABERTCritic:
    """Score source/current pairs while keeping critic and encoder frozen."""

    def __init__(
        self,
        checkpoint_path: Path,
        model_path: Path,
        device: torch.device,
        *,
        potential_minimum: float = -5.0,
        potential_maximum: float = 5.0,
        encoder_attention_backend: str = "OFFICIAL_PYTORCH_FALLBACK",
        encoder_class=FrozenMRNABERTOnlineEncoder,
    ) -> None:
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        _require(
            checkpoint.get("model_kind") == ROUTE2_PRETRAINED_EDIT_CENTERED_MODEL_KIND,
            "checkpoint is not the final mRNABERT edit-centered critic",
        )
        provenance = checkpoint.get("training_provenance", {})
        selection = checkpoint.get("selection_provenance", {})
        _require(
            provenance.get("result_stage") == "FINAL_ALL_DEVELOPMENT_REFIT",
            "critic is not the final all-Development refit",
        )
        _require(
            selection.get("checkpoint_selection") == "FINAL_EPOCH",
            "critic checkpoint was not frozen at the final refit epoch",
        )
        _require(
            provenance.get("optimizer_steps", 0) > 0
            and provenance.get("parameter_changed") is True
            and provenance.get("cuda_training_tensors_verified") is True
            and provenance.get("cpu_fallback_used") is False,
            "critic checkpoint lacks GPU parameter-update provenance",
        )
        minimum = float(potential_minimum)
        maximum = float(potential_maximum)
        _require(
            math.isfinite(minimum) and math.isfinite(maximum) and minimum < maximum,
            "potential clip is invalid",
        )
        self.minimum = minimum
        self.maximum = maximum
        self.device = device
        self.model = Route2PretrainedEditCenteredDeltaPredictor(
            **checkpoint["model_config"]
        ).to(device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval().requires_grad_(False)
        self.vocabs = checkpoint["vocabs"]
        self.encoder = encoder_class(
            model_path,
            device,
            attention_backend=encoder_attention_backend,
        )
        _require(
            int(checkpoint["model_config"]["pretrained_width"])
            == int(self.encoder.embedding_width),
            "online mRNABERT width differs from critic input width",
        )
        self._potential_cache: dict[tuple[str, str, str, str, str, str], float] = {}
        self.model_batch_forward_count = 0
        self.candidate_forward_equivalent_count = 0

    @property
    def cached_potential_count(self) -> int:
        return len(self._potential_cache)

    def clear_source_caches(self) -> None:
        """Start an independently budgeted source cohort without stale scores."""

        self._potential_cache.clear()
        self.encoder.clear_cache()

    def score_candidates(
        self,
        source_sequence: str,
        candidate_sequences: Iterable[str],
        *,
        assay_id: str,
        context_id: str,
        endpoint_id: str,
        region: str,
    ) -> list[float]:
        source = normalize_rna(source_sequence)
        candidates = [normalize_rna(sequence) for sequence in candidate_sequences]
        _require(bool(candidates), "critic candidate batch is empty")
        _require(
            all(len(candidate) == len(source) for candidate in candidates),
            "critic supports source-relative SUB candidates only",
        )
        region_key = str(region).replace("′", "").replace("'", "")
        _require(region_key in REGION, "critic region is unsupported")
        batch = len(candidates)
        source_tokens = torch.tensor(
            [[TOKEN[base] for base in source]] * batch,
            dtype=torch.long,
            device=self.device,
        )
        candidate_tokens = torch.tensor(
            [[TOKEN[base] for base in sequence] for sequence in candidates],
            dtype=torch.long,
            device=self.device,
        )
        padding = torch.zeros_like(source_tokens, dtype=torch.bool)
        pretrained = self.encoder.encode_sequences([source, *candidates])
        source_pretrained = pretrained[0].repeat(batch, 1).to(self.device)
        candidate_pretrained = pretrained[1:].to(self.device)
        assay = _category_id(self.vocabs["assay"], str(assay_id))
        context = _category_id(self.vocabs["context"], str(context_id))
        endpoint = _category_id(self.vocabs["endpoint"], str(endpoint_id))
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.device.type == "cuda",
        ):
            output = self.model(
                source_tokens,
                candidate_tokens,
                padding,
                torch.zeros(batch, dtype=torch.long, device=self.device),
                torch.full((batch,), assay, dtype=torch.long, device=self.device),
                torch.full((batch,), context, dtype=torch.long, device=self.device),
                torch.full((batch,), endpoint, dtype=torch.long, device=self.device),
                torch.full(
                    (batch,), REGION[region_key], dtype=torch.long, device=self.device
                ),
                source_pretrained,
                candidate_pretrained,
            )
        self.model_batch_forward_count += 1
        self.candidate_forward_equivalent_count += batch
        values = output["mean"].float().cpu().tolist()
        _require(all(math.isfinite(float(value)) for value in values), "critic mean is nonfinite")
        return [min(self.maximum, max(self.minimum, float(value))) for value in values]

    def potential(
        self,
        state: FlowState,
        *,
        endpoint_id: str,
        region: str,
    ) -> float:
        return self.potentials(
            [state], endpoint_id=endpoint_id, region=region
        )[0]

    def potentials(
        self,
        states: Iterable[FlowState],
        *,
        endpoint_id: str,
        region: str,
    ) -> list[float]:
        ordered = list(states)
        _require(bool(ordered), "potential state batch is empty")
        source = ordered[0].source_sequence
        assay = ordered[0].assay_id
        context = ordered[0].context_id
        _require(
            all(
                state.source_sequence == source
                and state.assay_id == assay
                and state.context_id == context
                for state in ordered
            ),
            "batched potentials must share source and biological context",
        )
        keys = [
            (
                state.source_sequence,
                state.current_sequence,
                state.assay_id,
                state.context_id,
                str(endpoint_id),
                str(region),
            )
            for state in ordered
        ]
        missing_keys = list(dict.fromkeys(
            key for key in keys if key not in self._potential_cache
        ))
        if missing_keys:
            values = self.score_candidates(
                source,
                [key[1] for key in missing_keys],
                assay_id=assay,
                context_id=context,
                endpoint_id=endpoint_id,
                region=region,
            )
            self._potential_cache.update(zip(missing_keys, values))
        return [self._potential_cache[key] for key in keys]
