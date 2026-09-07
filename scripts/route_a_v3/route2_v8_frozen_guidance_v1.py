#!/usr/bin/env python3
"""Frozen V8 critic for SetFlow guided generation (Stage 3).

V8 Stage 3 (SPECS_CRITIC_V6 spec "V8 攻坚线" 9.5): rerun SetFlow B2 with a
frozen V8 model as the guidance critic, then compare V8-critic vs V5-critic
guided delta-recovery per task family.

Potential of a candidate sequence = clip(f(candidate) - f(source), min, max),
f = frozen V8 regressor under the source's task domain + cell context (the
frozen-delta calibre, identical to the benchmark evaluation). Because the
readout is a single linear head on the pooled representation, both the domain
and cell embeddings cancel exactly in the delta (they add a constant to f of
both sequences), so scores are invariant to the conditioning choice - we still
pass the correct domain/cell for provenance accounting.

Interface mirrors FrozenXEditCriticV5 (route2_xeditcritic_v5_frozen_guidance_v1):
potentials(states, *, endpoint_id, region, source_row) -> list[float],
clear_source_caches(), guidance_provenance(), counters for budget accounting.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

# V8 backbone lives in the V8 stage-1/2 worktree.
V8_WT_CORE = "/home/cunyuliu/mrna_editflow_goal/worktrees/route_a_v3_v8_stage1_prep_20260904/core"
if V8_WT_CORE not in sys.path:
    sys.path.insert(0, V8_WT_CORE)

from route2_v8_hybrid_backbone_v1 import (  # noqa: E402
    CELL_IDS,
    DOMAIN_IDS,
    build_v8_regressor,
)

# study -> domain name (delta is conditioning-invariant; kept for provenance)
STUDY_DOMAIN = {
    "GSE114002": "mrl",
    "GSE269595": "polya",
    "ENCSR854RUF": "mprau",
    "GSE149487": "gse149487",
    "GSE186455": "gse186455",
    "GSE200304": "gse200304",
    "GSE217518": "gse217518",
    "GSE256185": "gse256185",
}

_NUM_DOMAINS = len(DOMAIN_IDS)
_NUM_CELLS = max(CELL_IDS.values()) + 1


class FrozenV8CriticError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FrozenV8CriticError(message)


class FrozenV8Critic:
    """Frozen V8 sequence regressor adapted to the guided-generation critic API."""

    def __init__(
        self,
        v8_checkpoint: Path,
        mrnabert_path: Path,
        device: torch.device,
        *,
        potential_minimum: float,
        potential_maximum: float,
        arch: str | None = None,
        scoring_batch_size: int = 64,
        round_chunk_size: int = 32,
    ) -> None:
        v8_checkpoint = Path(v8_checkpoint)
        _require(v8_checkpoint.is_file(), f"V8 checkpoint absent: {v8_checkpoint}")
        _require(Path(mrnabert_path).is_dir(), f"mRNABERT model dir absent: {mrnabert_path}")
        self.device = device
        self.minimum = float(potential_minimum)
        self.maximum = float(potential_maximum)
        self.scoring_batch_size = int(scoring_batch_size)
        self.round_chunk_size = int(round_chunk_size)
        raw = torch.load(v8_checkpoint, map_location="cpu", weights_only=False)
        state = raw["model_state_dict"]
        self.arch = str(arch or raw.get("arch") or "s")
        if self.arch not in ("s", "h"):
            raise FrozenV8CriticError(f"bad V8 arch {self.arch!r}")
        model = build_v8_regressor(
            Path(mrnabert_path), self.arch, num_domains=_NUM_DOMAINS, num_cells=_NUM_CELLS
        )
        model_state = model.state_dict()
        compatible = {
            k: v for k, v in state.items()
            if k in model_state and model_state[k].shape == v.shape
        }
        model.load_state_dict(compatible, strict=False)
        dom = state.get("domain_embeddings.weight")
        if dom is not None:
            n = min(dom.shape[0], model.domain_embeddings.weight.shape[0])
            model.domain_embeddings.weight.data[:n] = dom[:n]
        model.to(device)
        model.eval()
        self.model = model
        from transformers import AutoTokenizer  # noqa: PLC0415

        self.tokenizer = AutoTokenizer.from_pretrained(
            Path(mrnabert_path), local_files_only=True
        )
        self._potential_memo: dict[str, float] = {}
        self.potential_query_count = 0
        self.potential_newly_scored_count = 0
        self.model_batch_forward_count = 0
        self.candidate_forward_equivalent_count = 0

    # ------------------------------------------------------------------ API
    def clear_source_caches(self) -> None:
        self._potential_memo.clear()

    def guidance_provenance(self) -> dict[str, Any]:
        return {
            "critic_family": "V8_FROZEN_GUIDANCE",
            "arch": self.arch,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "potential_memo_size": len(self._potential_memo),
            "potential_query_count": self.potential_query_count,
            "potential_newly_scored_count": self.potential_newly_scored_count,
            "model_batch_forward_count": self.model_batch_forward_count,
            "candidate_forward_equivalent_count": self.candidate_forward_equivalent_count,
        }

    def potentials(
        self,
        states: Iterable[Any],
        *,
        endpoint_id: str,
        region: str,
        source_row: Mapping[str, Any] | None = None,
    ) -> list[float]:
        ordered = list(states)
        _require(bool(ordered), "potential state batch is empty")
        _require(source_row is not None, "V8 critic potentials requires the runner source row")
        source = ordered[0].source_sequence
        assay = ordered[0].assay_id
        context = ordered[0].context_id
        for state in ordered:
            _require(
                state.source_sequence == source
                and state.assay_id == assay
                and state.context_id == context,
                "batched potentials must share source and biological context",
            )
        _require(
            str(source_row["source_sequence"]).upper().replace("T", "U") == source
            and str(source_row["assay_id"]) == assay
            and str(source_row["biological_context_id"]) == context,
            "source row does not match the queried flow states",
        )
        sequences = [state.current_sequence for state in ordered]
        self.potential_query_count += len(ordered)
        if source not in self._potential_memo:
            # identity (candidate == source) potential is exactly zero.
            self._potential_memo[source] = 0.0
        missing = [
            seq for seq in dict.fromkeys(sequences) if seq not in self._potential_memo
        ]
        self.potential_newly_scored_count += len(missing)
        for start in range(0, len(missing), self.round_chunk_size):
            self._score_candidate_group(
                source, source_row, missing[start : start + self.round_chunk_size]
            )
        return [self._potential_memo[seq] for seq in sequences]

    def potential(
        self,
        state: Any,
        *,
        endpoint_id: str,
        region: str,
        source_row: Mapping[str, Any] | None = None,
    ) -> float:
        return self.potentials(
            [state], endpoint_id=endpoint_id, region=region, source_row=source_row
        )[0]

    # ------------------------------------------------------------- scoring
    def _score_candidate_group(
        self, source: str, source_row: Mapping[str, Any], candidates: Sequence[str]
    ) -> None:
        _require(
            all(len(candidate) == len(source) for candidate in candidates),
            "V8 critic supports source-relative SUB candidates only",
        )
        study = str(source_row.get("study_unit_id"))
        domain_name = STUDY_DOMAIN.get(study, "mrl")
        domain_id = DOMAIN_IDS[domain_name]
        context = str(source_row.get("biological_context_id"))
        cell_id = CELL_IDS.get(context, 0)
        tokenizer = self.tokenizer
        all_seqs = [source, *candidates]
        encoded = tokenizer(
            [str(s).upper().replace("U", "T") for s in all_seqs],
            add_special_tokens=True, padding=True, truncation=True,
            max_length=512, return_tensors="pt",
        )
        ids = encoded["input_ids"].to(self.device)
        mask = encoded["attention_mask"].to(self.device)
        domain_t = torch.full((len(all_seqs),), domain_id, dtype=torch.long, device=self.device)
        cell_t = torch.full((len(all_seqs),), cell_id, dtype=torch.long, device=self.device)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            values = self.model(ids, mask, domain_t, cell_t).float().cpu().numpy()
        self.model_batch_forward_count += 1
        self.candidate_forward_equivalent_count += len(candidates)
        source_value = float(values[0])
        for candidate, value in zip(candidates, values[1:], strict=True):
            _require(math.isfinite(float(value)), "V8 critic mean is nonfinite")
            delta = float(value) - source_value
            self._potential_memo[candidate] = min(self.maximum, max(self.minimum, delta))
