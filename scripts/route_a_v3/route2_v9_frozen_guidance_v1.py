#!/usr/bin/env python3
"""Frozen V9 adapter-zoo critic adapted to the guided-generation critic API.

Mirrors FrozenV8Critic (route2_v8_frozen_guidance_v1) with the V9 model
geometry: frozen trunk + multi-task LoRA + per-task heads + polyA CNN stem,
loaded from a run_route2_v9_adapter_zoo_v1 checkpoint. The potentials()
contract, memoisation, provenance counters and the link-integrity assertion
(distinct candidates must never receive identical scores) are identical.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

_V8_GUIDANCE = Path(__file__).resolve().parent / "route2_v8_frozen_guidance_v1.py"
_V9_RUNNER = Path(
    "/home/cunyuliu/mrna_editflow_goal/worktrees/route_a_v3_v8_stage1_prep_20260904"
    "/scripts/route_a_v3/run_route2_v9_adapter_zoo_v1.py"
)

import importlib.util
import sys


def _load_module(name: str, path: Path):
    # force the V9 runner's own repo root to WIN all `core.*` imports: move it to
    # sys.path[0] and drop any previously resolved `core` package cache (the probe
    # evaluator may have anchored `core` to a different worktree).
    repo_root = Path(path).resolve().parents[2]
    rp = str(repo_root)
    if rp in sys.path:
        sys.path.remove(rp)
    sys.path.insert(0, rp)
    sys.modules.pop("core", None)
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_v8g = _load_module("v8g_local", _V8_GUIDANCE)
_require = _v8g._require
STUDY_DOMAIN = _v8g.STUDY_DOMAIN
DOMAIN_IDS = _v8g.DOMAIN_IDS
CELL_IDS = _v8g.CELL_IDS

_NUM_DOMAINS = len(DOMAIN_IDS)
_NUM_CELLS = len(CELL_IDS)


class FrozenV9Critic:
    """Frozen V9 adapter-zoo regressor for guidance / mixed-pool probing."""

    def __init__(
        self,
        v9_checkpoint: Path,
        mrnabert_path: Path,
        device: torch.device,
        *,
        potential_minimum: float,
        potential_maximum: float,
        scoring_batch_size: int = 64,
        round_chunk_size: int = 32,
    ) -> None:
        v9_checkpoint = Path(v9_checkpoint)
        _require(v9_checkpoint.is_file(), f"V9 checkpoint absent: {v9_checkpoint}")
        _require(Path(mrnabert_path).is_dir(), f"mRNABERT model dir absent: {mrnabert_path}")
        self.device = device
        self.minimum = float(potential_minimum)
        self.maximum = float(potential_maximum)
        self.scoring_batch_size = int(scoring_batch_size)
        self.round_chunk_size = int(round_chunk_size)

        v9 = _load_module("v9_runner_local", _V9_RUNNER)
        model = v9.build_v9(Path(mrnabert_path), num_cells=_NUM_CELLS)
        v9.load_v9_init(model)
        # freeze trunk BEFORE wrap (runner discipline)
        for p in model.base.parameters():
            p.requires_grad_(False)
        model.wrap_multitask_lora()
        raw = torch.load(v9_checkpoint, map_location="cpu", weights_only=False)
        state = raw["model_state_dict"]
        model_state = model.state_dict()
        compatible = {k: v for k, v in state.items()
                      if k in model_state and model_state[k].shape == v.shape}
        missing = [k for k in model_state if k not in compatible]
        _require(len(missing) == 0, f"V9 checkpoint missing keys: {missing[:5]}")
        model.load_state_dict(compatible, strict=True)
        model.to(device)
        model.eval()
        self.model = model
        from transformers import AutoTokenizer  # noqa: PLC0415

        self.tokenizer = AutoTokenizer.from_pretrained(Path(mrnabert_path), local_files_only=True)
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
            "critic_family": "V9_FROZEN_ADAPTER_ZOO_GUIDANCE",
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
        _require(source_row is not None, "V9 critic potentials requires the runner source row")
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
            self._potential_memo[source] = 0.0
        missing = [seq for seq in dict.fromkeys(sequences) if seq not in self._potential_memo]
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
            "V9 critic supports source-relative SUB candidates only",
        )
        study = str(source_row.get("study_unit_id"))
        domain_name = STUDY_DOMAIN.get(study, "mrl")
        domain_id = DOMAIN_IDS[domain_name]
        context = str(source_row.get("biological_context_id"))
        cell_id = CELL_IDS.get(context, 0)
        tokenizer = self.tokenizer
        all_seqs = [source, *candidates]
        # per-nucleotide space-join (link-integrity contract, spec N.4.4)
        encoded = tokenizer(
            [" ".join(str(s).upper().replace("U", "T")) for s in all_seqs],
            add_special_tokens=True, padding=True, truncation=True,
            max_length=512, return_tensors="pt",
        )
        ids = encoded["input_ids"].to(self.device)
        mask = encoded["attention_mask"].to(self.device)
        # task-homogeneous scoring batch: duplicate rows are all the same domain
        domain_t = torch.full((len(all_seqs),), domain_id, dtype=torch.long, device=self.device)
        cell_t = torch.full((len(all_seqs),), cell_id, dtype=torch.long, device=self.device)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            values = self.model(ids, mask, domain_t, cell_t).float().cpu().numpy()
        self.model_batch_forward_count += 1
        self.candidate_forward_equivalent_count += len(candidates)
        source_value = float(values[0])
        # link-integrity assertion (N5 lesson; distinct candidates must differ)
        distinct = {seq for seq in candidates if seq != source}
        if len(distinct) >= 2:
            scored = [float(v) for v in values[1:]]
            _require(
                len(set(scored)) > 1,
                "V9 critic link failure: distinct candidates received identical "
                "potentials (tokenizer/encoding degenerate)",
            )
        for candidate, value in zip(candidates, values[1:], strict=True):
            _require(math.isfinite(float(value)), "V9 critic mean is nonfinite")
            delta = float(value) - source_value
            self._potential_memo[candidate] = min(self.maximum, max(self.minimum, delta))
