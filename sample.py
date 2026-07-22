"""Region-aware sampling and constrained editing for mRNA-EditFlow.

The sampler is deliberately offline-capable: it can run without a trained
checkpoint and still enforce the mRNA edit grammar needed by Task 4. When a
trained model is available, callers may pass it in and use this module as the
constraint-preserving outer loop; the deterministic operators here remain the
source of safety for frame/protein constraints.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from dataclasses import dataclass
from typing import List, Mapping, Optional, Sequence, Tuple, Union

from mrna_editflow.core.constants import (
    CODON_TABLE,
    ID_TO_NUC,
    NUC_VOCAB,
    NUC_TO_ID,
    REGION_3UTR,
    REGION_5UTR,
    SYNONYMOUS_CODONS,
    VOCAB_MODEL_SIZE,
    is_valid_cds,
    translate,
)
from mrna_editflow.core.config import BackboneConfig, DataConfig, MEFConfig, ModelConfig, CouplingConfig, TrainConfig
from mrna_editflow.core.schema import MRNARecord
from mrna_editflow.data.clean_mrna import clean_record
from mrna_editflow.data.download_mrna import load_records_jsonl, synthesize_corpus, write_records_jsonl
from mrna_editflow.rl.action_scoring import action_log_score_float
from mrna_editflow.rl.decoder_state import DecoderAction, DecoderState, choose_stop_aware_action, sequence_hash

SampleOutput = Union[MRNARecord, str]


@dataclass
class SamplingConfig:
    task_id: str = "T1"
    steps: int = 8
    edit_budget: Optional[int] = None
    target_length: Optional[int] = None
    motif: Optional[str] = None
    motif_action: str = "insert"
    motif_region: str = "5utr"
    guidance_scale: float = 0.0
    target_te: Optional[float] = None
    target_start_accessibility: Optional[float] = None
    guidance_candidates: int = 4
    return_record: bool = True
    seed: int = 0
    allow_stop: bool = True
    stop_logit_bias: float = 0.0
    min_action_margin: float = 0.0


def _normalise_nt(seq: str) -> str:
    seq = seq.upper().replace("T", "U")
    if any(ch not in NUC_VOCAB for ch in seq):
        raise ValueError(f"sequence contains non-ACGU characters: {seq!r}")
    return seq


def _copy_record(rec: MRNARecord, transcript_id: Optional[str] = None) -> MRNARecord:
    return MRNARecord(
        transcript_id=transcript_id or rec.transcript_id,
        five_utr=rec.five_utr,
        cds=rec.cds,
        three_utr=rec.three_utr,
        species=rec.species,
        metadata=dict(rec.metadata),
    )


def _normalise_region(region: str) -> str:
    key = region.lower().replace("'", "").replace("-", "").replace("_", "")
    if key in {"5utr", "fiveutr", "5"}:
        return "five_utr"
    if key in {"3utr", "threeutr", "3"}:
        return "three_utr"
    if key == "cds":
        return "cds"
    raise ValueError("region must be one of 5utr, cds, 3utr")


def _finish(rec: MRNARecord, return_record: bool) -> SampleOutput:
    checked = MRNARecord(
        transcript_id=rec.transcript_id,
        five_utr=_normalise_nt(rec.five_utr),
        cds=_normalise_nt(rec.cds),
        three_utr=_normalise_nt(rec.three_utr),
        species=rec.species,
        metadata=dict(rec.metadata),
    )
    if not is_valid_cds(checked.cds):
        raise ValueError("sampling produced an invalid mRNA record")
    return checked if return_record else checked.seq


def _random_nt_except(rng: random.Random, old: str) -> str:
    choices = [ch for ch in NUC_VOCAB if ch != old]
    return rng.choice(choices)


def _mutate_utr_substitutions(rec: MRNARecord, rng: random.Random, budget: int) -> MRNARecord:
    out = _copy_record(rec, transcript_id=f"{rec.transcript_id}_utr_refined")
    if budget <= 0:
        return out
    five = list(out.five_utr)
    three = list(out.three_utr)
    slots = [("five", i) for i in range(len(five))] + [("three", i) for i in range(len(three))]
    rng.shuffle(slots)
    for region, idx in slots[:budget]:
        if region == "five":
            five[idx] = _random_nt_except(rng, five[idx])
        else:
            three[idx] = _random_nt_except(rng, three[idx])
    out.five_utr = "".join(five)
    out.three_utr = "".join(three)
    return out


def _synonymous_optimize(rec: MRNARecord, rng: random.Random, budget: int) -> MRNARecord:
    """Apply <= budget nucleotide substitutions while preserving protein."""
    out = _copy_record(rec, transcript_id=f"{rec.transcript_id}_syn")
    if budget <= 0:
        return out
    original_protein = translate(out.cds)
    codons = [out.cds[i : i + 3] for i in range(0, len(out.cds), 3)]
    remaining = budget
    indices = list(range(1, max(1, len(codons) - 1)))  # skip AUG and terminal stop
    rng.shuffle(indices)
    for idx in indices:
        codon = codons[idx]
        aa = CODON_TABLE[codon]
        synonyms = [c for c in SYNONYMOUS_CODONS.get(aa, [codon]) if c != codon]
        rng.shuffle(synonyms)
        for cand in synonyms:
            diff = sum(a != b for a, b in zip(codon, cand))
            if 0 < diff <= remaining:
                codons[idx] = cand
                remaining -= diff
                break
        if remaining <= 0:
            break
    out.cds = "".join(codons)
    if translate(out.cds) != original_protein:
        raise AssertionError("synonymous optimization changed protein identity")
    if len(out.cds) != len(rec.cds) or not is_valid_cds(out.cds):
        raise AssertionError("synonymous optimization broke CDS validity")
    return out


def _move_toward_length(
    rec: MRNARecord,
    rng: random.Random,
    target_length: int,
    edit_budget: Optional[int],
) -> MRNARecord:
    out = _copy_record(rec, transcript_id=f"{rec.transcript_id}_len{target_length}")
    current = len(out.seq)
    delta = int(target_length) - current
    if delta == 0:
        return out
    max_change = abs(delta) if edit_budget is None else min(abs(delta), max(0, int(edit_budget)))
    if max_change == 0:
        return out
    if delta > 0:
        insert = "".join(rng.choice(NUC_VOCAB) for _ in range(max_change))
        out.three_utr = out.three_utr + insert
        return out

    remove = max_change
    if len(out.three_utr) >= remove:
        out.three_utr = out.three_utr[:-remove]
        return out
    remove -= len(out.three_utr)
    out.three_utr = ""
    if len(out.five_utr) >= remove:
        out.five_utr = out.five_utr[remove:]
    else:
        out.five_utr = ""
    return out


def _insert_motif(
    rec: MRNARecord,
    motif: str,
    region: str,
    rng: random.Random,
) -> MRNARecord:
    out = _copy_record(rec, transcript_id=f"{rec.transcript_id}_insert")
    motif = _normalise_nt(motif)
    target = _normalise_region(region)
    if target == "cds":
        raise ValueError("CDS motif insertion is not allowed by the frame/protein-safe sampler")
    seq = getattr(out, target)
    pos = len(seq) if target == "five_utr" else 0
    if len(seq) > 0 and target == "five_utr":
        pos = rng.randint(0, len(seq))
    setattr(out, target, seq[:pos] + motif + seq[pos:])
    return out


def _excise_motif(rec: MRNARecord, motif: str, region: str) -> MRNARecord:
    out = _copy_record(rec, transcript_id=f"{rec.transcript_id}_excise")
    motif = _normalise_nt(motif)
    target = _normalise_region(region)
    if target == "cds":
        raise ValueError("CDS motif excision is not allowed by the frame/protein-safe sampler")
    seq = getattr(out, target)
    pos = seq.find(motif)
    if pos >= 0:
        setattr(out, target, seq[:pos] + seq[pos + len(motif) :])
    return out


def _default_record(seed: int) -> MRNARecord:
    raw = synthesize_corpus(8, seed=seed)
    raw.sort(key=lambda r: len(r.seq))
    for rec in raw:
        cleaned = clean_record(rec)
        if cleaned is not None:
            return cleaned
    raise ValueError("could not synthesize a valid mRNA record")


def _apply_task_edit(
    tid: str,
    rec: MRNARecord,
    rng: random.Random,
    *,
    record_was_supplied: bool,
    steps: int,
    edit_budget: Optional[int],
    target_length: Optional[int],
    motif: Optional[str],
    motif_action: str,
    motif_region: str,
) -> MRNARecord:
    if tid == "T1":
        if target_length is not None:
            rec = _move_toward_length(rec, rng, target_length, edit_budget)
        return rec

    if not record_was_supplied:
        raise ValueError(f"{tid} requires an input record")

    if tid in {"T2", "T3"}:
        budget = edit_budget if edit_budget is not None else max(1, min(steps, 3))
        return _mutate_utr_substitutions(rec, rng, int(budget))
    if tid == "T4":
        budget = edit_budget if edit_budget is not None else max(1, min(steps, 6))
        return _synonymous_optimize(rec, rng, int(budget))
    if tid == "T5":
        budget = max(0, int(edit_budget if edit_budget is not None else steps))
        return _mutate_utr_substitutions(rec, rng, budget)
    if tid == "T6":
        if target_length is None:
            raise ValueError("T6 requires target_length")
        return _move_toward_length(rec, rng, int(target_length), edit_budget)
    if tid == "T7":
        if not motif:
            raise ValueError("T7 requires motif")
        action = motif_action.lower()
        if action in {"insert", "insertion"}:
            return _insert_motif(rec, motif, motif_region, rng)
        if action in {"excise", "excision", "delete", "remove"}:
            return _excise_motif(rec, motif, motif_region)
        raise ValueError("motif_action must be insert or excise")
    raise ValueError(f"unknown task_id {tid!r}; expected T1-T7")


def _guidance_is_active(
    guidance_scale: float,
    target_te: Optional[float],
    target_start_accessibility: Optional[float],
) -> bool:
    return (
        float(guidance_scale) != 0.0
        or target_te is not None
        or target_start_accessibility is not None
    )


def _guidance_oracle(oracle: Optional[object]) -> object:
    if oracle is not None:
        return oracle
    from mrna_editflow.eval.oracle import LocalTranslationOracle

    return LocalTranslationOracle()


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _guidance_score(
    rec: MRNARecord,
    oracle: object,
    *,
    guidance_scale: float,
    target_te: Optional[float],
    target_start_accessibility: Optional[float],
) -> float:
    if not hasattr(oracle, "score_record"):
        raise TypeError("oracle must provide a score_record(record) method")
    score = oracle.score_record(rec)  # type: ignore[attr-defined]
    if not isinstance(score, Mapping):
        raise TypeError("oracle.score_record(record) must return a mapping")
    features = score.get("features", {})
    if not isinstance(features, Mapping):
        features = {}
    te = _as_float(score.get("ensemble_te", score.get("te", 0.0)))
    access = _as_float(features.get("start_accessibility", 0.0))
    scale = float(guidance_scale)
    target_weight = max(1.0, abs(scale))
    objective = 0.0
    if target_te is None:
        objective += scale * te
    else:
        objective -= target_weight * abs(te - float(target_te))
    if target_start_accessibility is not None:
        objective -= target_weight * abs(access - float(target_start_accessibility))
    return float(objective)


def _select_guided_candidate(
    candidates: Sequence[MRNARecord],
    *,
    guidance_scale: float,
    target_te: Optional[float],
    target_start_accessibility: Optional[float],
    oracle: Optional[object],
) -> MRNARecord:
    if not candidates:
        raise ValueError("guidance requires at least one candidate")
    if not _guidance_is_active(guidance_scale, target_te, target_start_accessibility):
        return candidates[0]
    pred = _guidance_oracle(oracle)
    return max(
        candidates,
        key=lambda rec: _guidance_score(
            rec,
            pred,
            guidance_scale=guidance_scale,
            target_te=target_te,
            target_start_accessibility=target_start_accessibility,
        ),
    )


def sample_mrna(
    task_id: str = "T1",
    record: Optional[MRNARecord] = None,
    *,
    steps: int = 8,
    edit_budget: Optional[int] = None,
    target_length: Optional[int] = None,
    motif: Optional[str] = None,
    motif_action: str = "insert",
    motif_region: str = "5utr",
    guidance_scale: float = 0.0,
    target_te: Optional[float] = None,
    target_start_accessibility: Optional[float] = None,
    oracle: Optional[object] = None,
    guidance_candidates: int = 4,
    return_record: bool = True,
    seed: int = 0,
    model: Optional[object] = None,
    backbone: Optional[object] = None,
    device: Optional[str] = None,
    proposal_top_k: int = 8,
    proposal_temperature: float = 1.0,
    editable_regions: Optional[Sequence[str]] = None,
    allow_stop: bool = True,
    stop_logit_bias: float = 0.0,
    min_action_margin: float = 0.0,
    allow_action_space_expansion: bool = False,
) -> SampleOutput:
    """Sample or edit an mRNA for tasks T1-T7.

    With both ``model`` and ``backbone``, route explicitly to the constrained
    model-guided decoder. With neither, retain the random-safe decoder.
    """
    if (model is None) != (backbone is None):
        raise ValueError("sample_mrna requires both model and backbone, or neither")
    tid = task_id.upper()
    base = _copy_record(record) if record is not None else _default_record(seed)
    if model is not None and backbone is not None:
        decoded = model_guided_edit_record(
            base,
            model,
            backbone,
            task_id=tid,
            edit_budget=int(edit_budget if edit_budget is not None else steps),
            target_length=target_length,
            seed=seed,
            device=device,
            proposal_top_k=proposal_top_k,
            proposal_temperature=proposal_temperature,
            guidance_scale=guidance_scale,
            target_te=target_te,
            target_start_accessibility=target_start_accessibility,
            oracle=oracle,
            editable_regions=editable_regions,
            allow_stop=allow_stop,
            stop_logit_bias=stop_logit_bias,
            min_action_margin=min_action_margin,
            allow_action_space_expansion=allow_action_space_expansion,
        )
        return _finish(decoded, return_record)
    guided = _guidance_is_active(guidance_scale, target_te, target_start_accessibility)
    n_candidates = 1 if not guided else max(1, min(32, int(guidance_candidates)))
    candidates = []
    for idx in range(n_candidates):
        rec = _apply_task_edit(
            tid,
            _copy_record(base),
            random.Random(seed + idx),
            record_was_supplied=record is not None,
            steps=steps,
            edit_budget=edit_budget,
            target_length=target_length,
            motif=motif,
            motif_action=motif_action,
            motif_region=motif_region,
        )
        candidates.append(_finish(rec, return_record=True))

    rec = _select_guided_candidate(
        candidates,
        guidance_scale=guidance_scale,
        target_te=target_te,
        target_start_accessibility=target_start_accessibility,
        oracle=oracle,
    )
    rec.metadata.update(
        {
            "decoder_type": "random_safe",
            "checkpoint_path": None,
            "checkpoint_sha256": None,
            "oracle_guidance_used": bool(guided),
            "terminated_by_stop": False,
            "applied_edit_count": int(levenshtein_distance(base.seq, rec.seq)),
            "max_edit_budget": int(edit_budget if edit_budget is not None else steps),
            "cycle_rejections": 0,
            "out_of_training_action_space": False,
        }
    )
    return _finish(rec, return_record)


def sample_sequence(*args, **kwargs) -> str:
    """Convenience wrapper returning a sequence string."""
    kwargs["return_record"] = False
    out = sample_mrna(*args, **kwargs)
    return str(out)


def levenshtein_distance(a: str, b: str) -> int:
    """Small stdlib edit-distance helper used by tests and budget checks."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


class RegionAwareSampler:
    """Thin object wrapper around :func:`sample_mrna` for experiment code."""

    def __init__(self, seed: int = 0, model: Optional[object] = None, backbone: Optional[object] = None):
        self.seed = seed
        self.model = model
        self.backbone = backbone

    def sample(self, task_id: str = "T1", record: Optional[MRNARecord] = None, **kwargs) -> SampleOutput:
        kwargs.setdefault("seed", self.seed)
        kwargs.setdefault("model", self.model)
        kwargs.setdefault("backbone", self.backbone)
        return sample_mrna(task_id=task_id, record=record, **kwargs)


def _coerce_checkpoint_config(payload: Mapping[str, object]) -> MEFConfig:
    """Reconstruct :class:`MEFConfig` from a checkpoint payload.

    Stage A checkpoints store ``config`` as nested plain dictionaries. This
    helper mirrors the training entry point's config contract without importing
    private functions, so sampling remains a stable public API. Complexity is
    ``O(number_of_config_fields)``.
    """
    data = payload.get("config", {})
    if not isinstance(data, Mapping):
        return MEFConfig()
    return MEFConfig(
        data=DataConfig(**dict(data.get("data", {}))),  # type: ignore[arg-type]
        backbone=BackboneConfig(**dict(data.get("backbone", {}))),  # type: ignore[arg-type]
        model=ModelConfig(**dict(data.get("model", {}))),  # type: ignore[arg-type]
        coupling=CouplingConfig(**dict(data.get("coupling", {}))),  # type: ignore[arg-type]
        train=TrainConfig(**dict(data.get("train", {}))),  # type: ignore[arg-type]
    )


def load_stage_a_checkpoint(checkpoint_path: str, device: Optional[str] = None):
    """Load a Stage A or region-adapter checkpoint into ``(cfg, backbone, model)``.

    Stage A checkpoints contain frozen-backbone and MEF-head state dicts. Region
    Stage B checkpoints wrap the same MEF head with
    ``RegionSpecializedEditFormer`` adapters while preserving the public forward
    signature used by the sampler. The sampler uses the model only for ranking
    legal edit proposals; all hard biological constraints are still enforced by
    deterministic operators. Complexity is ``O(number_of_parameters)``.
    """
    import torch

    from mrna_editflow.models.backbones import FrozenBackbone
    from mrna_editflow.models.mrna_editformer import MRNAEditFormer
    from mrna_editflow.models.region_adapters import RegionSpecializedEditFormer

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    try:
        payload = torch.load(checkpoint_path, map_location=dev, weights_only=False)
    except TypeError:  # pragma: no cover - older torch without weights_only
        payload = torch.load(checkpoint_path, map_location=dev)
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint must contain a mapping payload")
    cfg = _coerce_checkpoint_config(payload)
    backbone = FrozenBackbone(cfg.backbone).to(dev)
    base_model = MRNAEditFormer(cfg.model, backbone_dim=backbone.out_dim).to(dev)
    if "backbone_state" in payload:
        backbone.load_state_dict(payload["backbone_state"], strict=False)  # type: ignore[arg-type]
    stage = str(payload.get("stage", "A"))
    if stage == "B_region":
        region_ids = payload.get("region_ids")
        if not isinstance(region_ids, Sequence) or isinstance(region_ids, (str, bytes)):
            region_ids = None
        bottleneck = payload.get("adapter_bottleneck")
        if not isinstance(bottleneck, int):
            model_state = payload.get("model_state", {})
            if isinstance(model_state, Mapping):
                for key, value in model_state.items():
                    if key.startswith("adapters.") and key.endswith(".down.weight") and hasattr(value, "shape"):
                        bottleneck = int(value.shape[0])
                        break
        if not isinstance(bottleneck, int):
            bottleneck = 32
        model = RegionSpecializedEditFormer(
            base_model,
            bottleneck=int(bottleneck),
            regions=[int(x) for x in region_ids] if region_ids is not None else None,
            freeze_base=True,
        ).to(dev)
    else:
        model = base_model
    if "model_state" in payload:
        model.load_state_dict(payload["model_state"], strict=False)  # type: ignore[arg-type]
    action_space = payload.get("trained_action_space")
    if action_space is None and any(
        key in payload
        for key in ("trained_task", "trained_editable_regions", "trained_operations")
    ):
        action_space = {
            key: payload.get(key)
            for key in ("trained_task", "trained_editable_regions", "trained_operations")
        }
    if action_space is not None and not isinstance(action_space, Mapping):
        raise ValueError("trained_action_space checkpoint metadata must be a mapping")
    digest = hashlib.sha256()
    with open(checkpoint_path, "rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    # Preserve the established three-value return while exposing checkpoint
    # semantics to model-guided entry points without a second loader API.
    model._trained_action_space = dict(action_space) if isinstance(action_space, Mapping) else None  # type: ignore[attr-defined]
    model._checkpoint_path = str(checkpoint_path)  # type: ignore[attr-defined]
    model._checkpoint_sha256 = digest.hexdigest()  # type: ignore[attr-defined]
    model._checkpoint_stage = stage  # type: ignore[attr-defined]
    backbone.freeze()
    backbone.eval()
    model.eval()
    return cfg, backbone, model


def _record_tensors(record: MRNARecord, device):
    """Convert a record to model tensors ``[1, L]`` without padding.

    Region ids and codon phases are exactly aligned to nucleotide positions.
    Complexity is ``O(L)``.
    """
    import torch

    token_ids = torch.tensor([record.token_ids()], dtype=torch.long, device=device)
    region_ids = torch.tensor([record.region_ids()], dtype=torch.long, device=device)
    phase_ids = torch.tensor([record.codon_phases()], dtype=torch.long, device=device)
    padding_mask = torch.zeros_like(token_ids, dtype=torch.bool)
    return token_ids, region_ids, phase_ids, padding_mask


def _model_out_for_record(record: MRNARecord, model, backbone, device, time_value: float = 0.5):
    """Run one model forward pass for edit-proposal scoring.

    We evaluate the CTMC field at a fixed mid-bridge time ``t=0.5``. For a
    source sequence ``x``, candidate operation score is the model intensity
    ``lambda_op(i) p_op(a|i)``. Complexity is one transformer forward pass:
    ``O(layers * L^2 * dim)``.
    """
    import torch

    token_ids, region_ids, phase_ids, padding_mask = _record_tensors(record, device)
    t = torch.full((1, 1), float(time_value), dtype=torch.float32, device=device)
    with torch.no_grad():
        return model.forward(token_ids, region_ids, phase_ids, t, padding_mask, backbone)


def _replace_nt(record: MRNARecord, pos: int, nt: str, transcript_id: str) -> MRNARecord:
    """Return a copy with one full-sequence nucleotide substituted."""
    five_len = len(record.five_utr)
    cds_len = len(record.cds)
    out = _copy_record(record, transcript_id=transcript_id)
    if pos < five_len:
        chars = list(out.five_utr)
        chars[pos] = nt
        out.five_utr = "".join(chars)
    elif pos < five_len + cds_len:
        idx = pos - five_len
        chars = list(out.cds)
        chars[idx] = nt
        out.cds = "".join(chars)
    else:
        idx = pos - five_len - cds_len
        chars = list(out.three_utr)
        chars[idx] = nt
        out.three_utr = "".join(chars)
    return out


def _insert_nt_after(record: MRNARecord, pos: int, nt: str, transcript_id: str) -> MRNARecord:
    """Insert one nucleotide after a UTR position, never inside CDS."""
    five_len = len(record.five_utr)
    cds_len = len(record.cds)
    out = _copy_record(record, transcript_id=transcript_id)
    if pos < five_len:
        idx = pos + 1
        out.five_utr = out.five_utr[:idx] + nt + out.five_utr[idx:]
    elif pos >= five_len + cds_len:
        idx = pos - five_len - cds_len + 1
        out.three_utr = out.three_utr[:idx] + nt + out.three_utr[idx:]
    else:
        raise ValueError("model-guided insertion inside CDS is forbidden")
    return out


def _delete_nt(record: MRNARecord, pos: int, transcript_id: str) -> MRNARecord:
    """Delete one UTR nucleotide, never inside CDS."""
    five_len = len(record.five_utr)
    cds_len = len(record.cds)
    out = _copy_record(record, transcript_id=transcript_id)
    if pos < five_len:
        out.five_utr = out.five_utr[:pos] + out.five_utr[pos + 1:]
    elif pos >= five_len + cds_len:
        idx = pos - five_len - cds_len
        out.three_utr = out.three_utr[:idx] + out.three_utr[idx + 1:]
    else:
        raise ValueError("model-guided deletion inside CDS is forbidden")
    return out


def _normalise_editable_utr_regions(
    editable_regions: Optional[Sequence[str]],
) -> Tuple[str, ...]:
    """Validate and canonicalize the UTR regions available to the decoder."""
    if editable_regions is None:
        return ("utr5", "utr3")
    aliases = {
        "5utr": "utr5",
        "5'utr": "utr5",
        "utr5": "utr5",
        "3utr": "utr3",
        "3'utr": "utr3",
        "utr3": "utr3",
    }
    values = (
        [editable_regions]
        if isinstance(editable_regions, str)
        else list(editable_regions)
    )
    normalized: List[str] = []
    for value in values:
        key = str(value).strip().lower().replace("_", "")
        if key not in aliases:
            raise ValueError(
                "editable_regions must contain only utr5/utr3 aliases"
            )
        region = aliases[key]
        if region not in normalized:
            normalized.append(region)
    if not normalized:
        raise ValueError("editable_regions must not be empty")
    return tuple(normalized)


def _task_operations(task_id: str) -> tuple[str, ...]:
    tid = str(task_id).upper()
    if tid in {"T2", "T3", "T4", "T5"}:
        return ("sub",)
    if tid == "T6":
        return ("ins", "del")
    return ()


def _normalise_task_editable_regions(
    task_id: str,
    editable_regions: Optional[Sequence[str]],
) -> tuple[str, ...]:
    """Validate regions against the task's hard editing grammar."""
    if str(task_id).upper() == "T4":
        values = [editable_regions] if isinstance(editable_regions, str) else list(editable_regions or ("cds",))
        if not values or any(str(value).strip().lower() != "cds" for value in values):
            raise ValueError("T4 editable_regions must be exactly ('cds',)")
        return ("cds",)
    return _normalise_editable_utr_regions(editable_regions)


def _resolve_decoder_action_space(
    model: object,
    *,
    task_id: str,
    requested_regions: Optional[Sequence[str]],
    allow_action_space_expansion: bool,
) -> tuple[tuple[str, ...], bool]:
    """Resolve the trained action domain without silently widening it.

    Legacy Stage A checkpoints have no teacher action-domain metadata and keep
    the historical UTR default. Newly saved proposal-ranker checkpoints carry
    an explicit domain and fail closed unless expansion is opted into.
    """
    metadata = getattr(model, "_trained_action_space", None)
    if not isinstance(metadata, Mapping):
        return _normalise_task_editable_regions(task_id, requested_regions), False
    trained_task = str(metadata.get("trained_task", "")).upper()
    raw_regions = metadata.get("trained_editable_regions", ())
    raw_ops = {str(op).lower() for op in metadata.get("trained_operations", ())}
    if not isinstance(raw_regions, Sequence) or isinstance(raw_regions, (str, bytes)):
        raise ValueError("trained_action_space.trained_editable_regions must be a sequence")
    trained_regions = _normalise_task_editable_regions(task_id, [str(region) for region in raw_regions])
    requested = (
        trained_regions
        if requested_regions is None
        else _normalise_task_editable_regions(task_id, requested_regions)
    )
    task_mismatch = bool(trained_task and trained_task != str(task_id).upper())
    operation_mismatch = not set(_task_operations(task_id)).issubset(raw_ops)
    region_mismatch = not set(requested).issubset(set(trained_regions))
    if (task_mismatch or operation_mismatch or region_mismatch) and not allow_action_space_expansion:
        raise ValueError(
            "requested decoder action space is outside checkpoint training domain; "
            "set allow_action_space_expansion=True to opt in explicitly"
        )
    if requested_regions is None and not (task_mismatch or operation_mismatch):
        return trained_regions, False
    return requested, bool(task_mismatch or operation_mismatch or region_mismatch)


def _attach_decoder_metadata(
    record: MRNARecord,
    *,
    state: DecoderState,
    model: object,
    decoder_type: str,
    oracle_guidance_used: bool,
) -> MRNARecord:
    record.metadata.update(
        {
            "decoder_type": decoder_type,
            "checkpoint_path": getattr(model, "_checkpoint_path", None),
            "checkpoint_sha256": getattr(model, "_checkpoint_sha256", None),
            "oracle_guidance_used": bool(oracle_guidance_used),
            **state.to_metadata(),
        }
    )
    return record


def _utr_positions(
    record: MRNARecord,
    editable_regions: Optional[Sequence[str]] = None,
) -> List[int]:
    """Return only UTR positions explicitly enabled for candidate generation."""
    regions = _normalise_editable_utr_regions(editable_regions)
    positions: List[int] = []
    if "utr5" in regions:
        positions.extend(range(len(record.five_utr)))
    if "utr3" in regions:
        offset = len(record.five_utr) + len(record.cds)
        positions.extend(offset + i for i in range(len(record.three_utr)))
    return positions


def _synonymous_substitution_candidates(record: MRNARecord, out) -> List[Tuple[float, int, str]]:
    """Enumerate protein-preserving CDS substitutions scored by shared log score.

    Score for replacing position ``i`` with nucleotide ``a`` is
    ``lambda_sub(i) * p_sub_i(a)``. Start and terminal stop codons are skipped.
    A candidate is emitted only when the one-base edited codon still translates
    to the original amino acid:

    ``AA(c_0 c_1 c_2) = AA(c_0 ... a_j ... c_2)``.

    This avoids unsafe intermediate edits from multi-base synonymous codons
    such as Arg ``CGU -> AGA``. Complexity is ``O(N_codon * 3 * |V|)``.
    """
    candidates: List[Tuple[float, int, str]] = []
    five_len = len(record.five_utr)
    codons = [record.cds[i:i + 3] for i in range(0, len(record.cds), 3)]
    for codon_idx, codon in enumerate(codons):
        if codon_idx == 0 or codon_idx == len(codons) - 1:
            continue
        aa = CODON_TABLE.get(codon)
        if aa is None or aa == "*":
            continue
        for offset, old in enumerate(codon):
            pos = five_len + codon_idx * 3 + offset
            for new in NUC_VOCAB:
                if new == old:
                    continue
                edited = codon[:offset] + new + codon[offset + 1:]
                if CODON_TABLE.get(edited) != aa:
                    continue
                score = action_log_score_float(out, "sub", pos, new)
                candidates.append((score, pos, new))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates


def _utr_substitution_candidates(
    record: MRNARecord,
    out,
    editable_regions: Optional[Sequence[str]] = None,
) -> List[Tuple[float, int, str]]:
    """Enumerate UTR substitutions scored by the shared CTMC log score."""
    seq = record.seq
    candidates: List[Tuple[float, int, str]] = []
    for pos in _utr_positions(record, editable_regions):
        old = seq[pos]
        for nt in NUC_VOCAB:
            if nt == old:
                continue
            score = action_log_score_float(out, "sub", pos, nt)
            candidates.append((score, pos, nt))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates


def _utr_insert_candidates(
    record: MRNARecord,
    out,
    editable_regions: Optional[Sequence[str]] = None,
) -> List[Tuple[float, int, str]]:
    """Enumerate UTR insertions scored by the shared CTMC log score."""
    candidates: List[Tuple[float, int, str]] = []
    for pos in _utr_positions(record, editable_regions):
        for nt in NUC_VOCAB:
            score = action_log_score_float(out, "ins", pos, nt)
            candidates.append((score, pos, nt))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates


def _utr_delete_candidates(
    record: MRNARecord,
    out,
    editable_regions: Optional[Sequence[str]] = None,
) -> List[Tuple[float, int, str]]:
    """Enumerate UTR deletions scored by the shared CTMC log score."""
    candidates = [
        (action_log_score_float(out, "del", pos, None), pos, "")
        for pos in _utr_positions(record, editable_regions)
    ]
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates


def _model_score_for_choice(out, op: str, pos: int, nt: str) -> float:
    """Backward-compatible wrapper around the shared action log-score."""
    return action_log_score_float(out, op, pos, nt or None)


def _choose_scored_candidate(
    candidates: Sequence[Tuple[float, int, str]],
    rng: random.Random,
    *,
    top_k: int,
    temperature: float,
) -> Tuple[float, int, str]:
    """Choose one legal edit proposal from model-scored candidates.

    ``top_k=1`` gives deterministic greedy decoding. ``top_k<=0`` evaluates the
    full legal candidate pool. For ``top_k>1`` we sample from the top-k proposals
    with softmax weights

    ``p_i ∝ exp((score_i - max_score) / temperature)``.

    Subtracting ``max_score`` keeps the exponent finite. If all weights collapse
    to zero/non-finite values we fall back to a uniform choice over top-k. The
    operation is seed-dependent through ``rng`` and never introduces illegal
    edits because the candidate list was pre-filtered by region/codon grammar.
    Complexity is ``O(top_k)``.
    """
    if not candidates:
        raise ValueError("cannot choose from an empty candidate list")
    # ``top_k <= 0`` means "use the full legal candidate pool"; this is useful
    # for oracle-guided ablations where we want to test whether the model's
    # narrow top-k locality is the limiting factor.
    k = len(candidates) if int(top_k) <= 0 else max(1, min(int(top_k), len(candidates)))
    top = list(candidates[:k])
    if k == 1 or float(temperature) <= 0.0:
        return top[0]
    max_score = max(float(item[0]) for item in top)
    temp = max(float(temperature), 1e-8)
    weights = []
    for score, _pos, _nt in top:
        try:
            weights.append(math.exp((float(score) - max_score) / temp))
        except OverflowError:
            weights.append(0.0)
    total = sum(w for w in weights if math.isfinite(w) and w > 0.0)
    if total <= 0.0:
        return top[rng.randrange(k)]
    threshold = rng.random() * total
    acc = 0.0
    for item, weight in zip(top, weights):
        if not math.isfinite(weight) or weight <= 0.0:
            continue
        acc += weight
        if acc >= threshold:
            return item
    return top[-1]


def _choose_guided_proposal(
    choices: Sequence[Tuple[float, int, str]],
    rng: random.Random,
    *,
    current: MRNARecord,
    make_record,
    guidance_scale: float,
    target_te: Optional[float],
    target_start_accessibility: Optional[float],
    oracle: Optional[object],
    top_k: int,
    temperature: float,
) -> Tuple[float, int, str]:
    """Choose a legal proposal using model score plus optional oracle guidance.

    Candidate generation is two-stage:

    1. The MEF rate field ranks all legal edits by CTMC intensity. We keep only
       the top-k local proposals (or all proposals when ``top_k<=0``),
       preserving the model's learned locality unless full-pool ablation is
       requested.
    2. If guidance is active, each proposal is materialised as a valid
       :class:`MRNARecord` and scored by
       ``combined = model_score + guidance_objective(record)``.

    The returned proposal is sampled from the combined scores with the same
    softmax logic as :func:`_choose_scored_candidate`, so seeds affect proposal
    choice while every candidate remains grammar-valid. Complexity is
    ``O(top_k * oracle_score_cost)`` when guidance is active, otherwise
    ``O(top_k)``.
    """
    if not _guidance_is_active(guidance_scale, target_te, target_start_accessibility):
        return _choose_scored_candidate(choices, rng, top_k=top_k, temperature=temperature)
    if not choices:
        raise ValueError("cannot choose from an empty candidate list")
    pred = _guidance_oracle(oracle)
    k = len(choices) if int(top_k) <= 0 else max(1, min(int(top_k), len(choices)))
    guided: List[Tuple[float, int, str]] = []
    for model_score, pos, nt in choices[:k]:
        rec = make_record(pos, nt)
        guide = _guidance_score(
            rec,
            pred,
            guidance_scale=guidance_scale,
            target_te=target_te,
            target_start_accessibility=target_start_accessibility,
        )
        guided.append((float(model_score) + float(guide), pos, nt))
    guided.sort(key=lambda item: item[0], reverse=True)
    return _choose_scored_candidate(guided, rng, top_k=k, temperature=temperature)


def model_guided_edit_record(
    record: MRNARecord,
    model,
    backbone,
    *,
    task_id: str = "T5",
    edit_budget: int = 3,
    target_length: Optional[int] = None,
    seed: int = 0,
    device: Optional[str] = None,
    proposal_top_k: int = 8,
    proposal_temperature: float = 1.0,
    guidance_scale: float = 0.0,
    target_te: Optional[float] = None,
    target_start_accessibility: Optional[float] = None,
    oracle: Optional[object] = None,
    editable_regions: Optional[Sequence[str]] = None,
    allow_stop: bool = True,
    stop_logit_bias: float = 0.0,
    min_action_margin: float = 0.0,
    allow_action_space_expansion: bool = False,
) -> MRNARecord:
    """Decode grammar-valid edits with shared log scores and explicit STOP.

    The requested budget is an upper bound. STOP is a configurable baseline,
    not a learned head, and cycle/reverse actions are rejected before sampling.
    """
    import torch

    tid = task_id.upper()
    enabled_utr_regions, expanded = _resolve_decoder_action_space(
        model,
        task_id=tid,
        requested_regions=editable_regions,
        allow_action_space_expansion=allow_action_space_expansion,
    )
    rng = random.Random(int(seed))
    dev = torch.device(device or next(model.parameters()).device)
    current = _copy_record(record, transcript_id=f"{record.transcript_id}_{tid.lower()}_model")
    budget = max(0, int(edit_budget))
    if tid == "T6":
        if target_length is None:
            target_length = len(record.seq)
        budget = abs(int(target_length) - len(record.seq)) if edit_budget is None else budget
    state = DecoderState(current.seq, budget, out_of_training_action_space=expanded)
    guidance_active = _guidance_is_active(guidance_scale, target_te, target_start_accessibility)
    for _step in range(budget):
        out = _model_out_for_record(current, model, backbone, dev)
        if tid == "T4":
            choices = _synonymous_substitution_candidates(current, out)
            op = "sub"
            make_record = lambda pos, nt: _replace_nt(current, pos, nt, f"{record.transcript_id}_{tid.lower()}_model")
        elif tid in {"T2", "T3", "T5"}:
            choices = _utr_substitution_candidates(current, out, enabled_utr_regions)
            op = "sub"
            make_record = lambda pos, nt: _replace_nt(current, pos, nt, f"{record.transcript_id}_{tid.lower()}_model")
        elif tid == "T6":
            delta = int(target_length) - len(current.seq)
            if delta == 0:
                state.terminated_by_stop = True
                break
            choices = (
                _utr_insert_candidates(current, out, enabled_utr_regions)
                if delta > 0
                else _utr_delete_candidates(current, out, enabled_utr_regions)
            )
            if delta > 0:
                op = "ins"
                make_record = lambda pos, nt: _insert_nt_after(current, pos, nt, f"{record.transcript_id}_{tid.lower()}_model")
            else:
                op = "del"
                make_record = lambda pos, nt: _delete_nt(current, pos, f"{record.transcript_id}_{tid.lower()}_model")
        else:
            raise ValueError("model-guided decoder supports T2/T3/T4/T5/T6")
        actions: list[DecoderAction] = []
        for log_score, pos, nt in choices:
            candidate = make_record(pos, nt)
            score = float(log_score)
            if guidance_active:
                score += _guidance_score(
                    candidate,
                    _guidance_oracle(oracle),
                    guidance_scale=guidance_scale,
                    target_te=target_te,
                    target_start_accessibility=target_start_accessibility,
                )
            actions.append(
                DecoderAction(
                    op=op,
                    pos=int(pos),
                    nt=str(nt) or None,
                    log_score=score,
                    next_sequence_hash=sequence_hash(candidate.seq),
                    old_nt=current.seq[int(pos)] if op == "sub" else None,
                )
            )
        chosen = choose_stop_aware_action(
            actions,
            state,
            rng,
            top_k=proposal_top_k,
            temperature=proposal_temperature,
            allow_stop=allow_stop,
            stop_logit_bias=stop_logit_bias,
            min_action_margin=min_action_margin,
        )
        if chosen is None:
            break
        current = make_record(int(chosen.pos), str(chosen.nt or ""))
        _finish(current, return_record=True)
    return _attach_decoder_metadata(
        _finish(current, return_record=True),  # type: ignore[arg-type]
        state=state,
        model=model,
        decoder_type="model_guided",
        oracle_guidance_used=guidance_active,
    )


def cascade_model_guided_edit_record(
    record: MRNARecord,
    recall_model,
    recall_backbone,
    precision_model,
    precision_backbone,
    *,
    task_id: str = "T5",
    edit_budget: int = 3,
    target_length: Optional[int] = None,
    seed: int = 0,
    device: Optional[str] = None,
    recall_top_k: int = 64,
    proposal_temperature: float = 1.0,
    guidance_scale: float = 0.0,
    target_te: Optional[float] = None,
    target_start_accessibility: Optional[float] = None,
    oracle: Optional[object] = None,
    editable_regions: Optional[Sequence[str]] = None,
    allow_stop: bool = True,
    stop_logit_bias: float = 0.0,
    min_action_margin: float = 0.0,
    allow_action_space_expansion: bool = False,
) -> MRNARecord:
    """Generate one candidate with recall-then-precision cascade decoding.

    At each edit step, the recall model ranks the full legal candidate set and
    keeps ``K`` candidates. The precision model then rescales only that retained
    set and the decoder samples/chooses from the precision scores:

    ``C_K = TopK_recall(C(x))``

    ``c* ~ softmax(s_precision(c) / temperature), c in C_K``.

    Hard biological constraints remain identical to
    :func:`model_guided_edit_record`: T4 uses synonymous CDS substitutions,
    T2/T3/T5 use UTR substitutions, and T6 uses UTR indels toward the target
    length. Complexity is ``O(E * (2*model_forward + |C|*V))`` for edit budget
    ``E``.
    """
    import torch

    tid = task_id.upper()
    enabled_utr_regions, expanded = _resolve_decoder_action_space(
        precision_model,
        task_id=tid,
        requested_regions=editable_regions,
        allow_action_space_expansion=allow_action_space_expansion,
    )
    rng = random.Random(int(seed))
    dev = torch.device(device or next(recall_model.parameters()).device)
    current = _copy_record(record, transcript_id=f"{record.transcript_id}_{tid.lower()}_cascade")
    budget = max(0, int(edit_budget))
    if tid == "T6":
        if target_length is None:
            target_length = len(record.seq)
        budget = abs(int(target_length) - len(record.seq)) if edit_budget is None else budget
    state = DecoderState(current.seq, budget, out_of_training_action_space=expanded)
    guidance_active = _guidance_is_active(guidance_scale, target_te, target_start_accessibility)

    for step in range(budget):
        recall_out = _model_out_for_record(current, recall_model, recall_backbone, dev)
        precision_out = _model_out_for_record(current, precision_model, precision_backbone, dev)
        make_record = None
        op = "sub"
        if tid == "T4":
            choices = _synonymous_substitution_candidates(current, recall_out)
            make_record = lambda pos, nt: _replace_nt(  # noqa: E731
                current, pos, nt, f"{record.transcript_id}_{tid.lower()}_cascade"
            )
            op = "sub"
        elif tid in {"T2", "T3", "T5"}:
            choices = _utr_substitution_candidates(
                current,
                recall_out,
                enabled_utr_regions,
            )
            make_record = lambda pos, nt: _replace_nt(  # noqa: E731
                current, pos, nt, f"{record.transcript_id}_{tid.lower()}_cascade"
            )
            op = "sub"
        elif tid == "T6":
            delta = int(target_length) - len(current.seq)
            if delta == 0:
                break
            if delta > 0:
                choices = _utr_insert_candidates(
                    current,
                    recall_out,
                    enabled_utr_regions,
                )
                make_record = lambda pos, nt: _insert_nt_after(  # noqa: E731
                    current, pos, nt, f"{record.transcript_id}_{tid.lower()}_cascade"
                )
                op = "ins"
            else:
                choices = _utr_delete_candidates(
                    current,
                    recall_out,
                    enabled_utr_regions,
                )
                make_record = lambda pos, nt: _delete_nt(  # noqa: E731
                    current, pos, f"{record.transcript_id}_{tid.lower()}_cascade"
                )
                op = "del"
        else:
            return model_guided_edit_record(
                record,
                precision_model,
                precision_backbone,
                task_id=tid,
                edit_budget=edit_budget,
                target_length=target_length,
                seed=seed,
                device=device,
                proposal_top_k=recall_top_k,
                proposal_temperature=proposal_temperature,
                guidance_scale=guidance_scale,
                target_te=target_te,
                target_start_accessibility=target_start_accessibility,
                oracle=oracle,
                editable_regions=enabled_utr_regions,
                allow_stop=allow_stop,
                stop_logit_bias=stop_logit_bias,
                min_action_margin=min_action_margin,
                allow_action_space_expansion=allow_action_space_expansion,
            )
        k = len(choices) if int(recall_top_k) <= 0 else max(1, min(int(recall_top_k), len(choices)))
        retained = choices[:k]
        precision_scored = [
            (_model_score_for_choice(precision_out, op, pos, nt), pos, nt)
            for _recall_score, pos, nt in retained
        ]
        precision_scored.sort(key=lambda item: item[0], reverse=True)
        actions: list[DecoderAction] = []
        for log_score, pos, nt in precision_scored:
            candidate = make_record(pos, nt)
            score = float(log_score)
            if guidance_active:
                score += _guidance_score(
                    candidate, _guidance_oracle(oracle), guidance_scale=guidance_scale,
                    target_te=target_te, target_start_accessibility=target_start_accessibility,
                )
            actions.append(DecoderAction(op, int(pos), str(nt) or None, score, sequence_hash(candidate.seq), current.seq[int(pos)] if op == "sub" else None))
        chosen = choose_stop_aware_action(
            actions, state, rng, top_k=k, temperature=proposal_temperature,
            allow_stop=allow_stop, stop_logit_bias=stop_logit_bias,
            min_action_margin=min_action_margin,
        )
        if chosen is None:
            break
        current = make_record(int(chosen.pos), str(chosen.nt or ""))
        _finish(current, return_record=True)
    return _attach_decoder_metadata(
        _finish(current, return_record=True),  # type: ignore[arg-type]
        state=state,
        model=precision_model,
        decoder_type="cascade_model_guided",
        oracle_guidance_used=guidance_active,
    )


def generate_candidate_records(
    records: Sequence[MRNARecord],
    *,
    task_id: str = "T5",
    checkpoint_path: Optional[str] = None,
    cascade_recall_checkpoint_path: Optional[str] = None,
    model: Optional[object] = None,
    backbone: Optional[object] = None,
    limit: Optional[int] = None,
    edit_budget: int = 3,
    target_length_delta: int = 0,
    seed: int = 0,
    device: Optional[str] = None,
    proposal_top_k: int = 8,
    cascade_recall_top_k: int = 64,
    proposal_temperature: float = 1.0,
    guidance_scale: float = 0.0,
    target_te: Optional[float] = None,
    target_start_accessibility: Optional[float] = None,
    oracle: Optional[object] = None,
    editable_regions: Optional[Sequence[str]] = None,
    allow_stop: bool = True,
    stop_logit_bias: float = 0.0,
    min_action_margin: float = 0.0,
    allow_action_space_expansion: bool = False,
) -> List[MRNARecord]:
    """Generate benchmark candidates from records, optionally checkpoint-guided.

    When ``checkpoint_path`` or ``model/backbone`` is supplied, candidates are
    ranked by the trained MEF CTMC field. Otherwise the deterministic safe
    sampler is used. Outputs are canonical :class:`MRNARecord` objects suitable
    for :mod:`mrna_editflow.eval.run_eval`.
    """
    if checkpoint_path is not None:
        _cfg, backbone, model = load_stage_a_checkpoint(checkpoint_path, device=device)
    recall_model = None
    recall_backbone = None
    if cascade_recall_checkpoint_path is not None:
        _recall_cfg, recall_backbone, recall_model = load_stage_a_checkpoint(
            cascade_recall_checkpoint_path,
            device=device,
        )
    selected = list(records[: int(limit)]) if limit is not None else list(records)
    out: List[MRNARecord] = []
    for i, rec in enumerate(selected):
        target_length = len(rec.seq) + int(target_length_delta) if task_id.upper() == "T6" else None
        if recall_model is not None and recall_backbone is not None and model is not None and backbone is not None:
            cand = cascade_model_guided_edit_record(
                rec,
                recall_model,
                recall_backbone,
                model,
                backbone,
                task_id=task_id,
                edit_budget=edit_budget,
                target_length=target_length,
                seed=seed + i,
                device=device,
                recall_top_k=cascade_recall_top_k,
                proposal_temperature=proposal_temperature,
                guidance_scale=guidance_scale,
                target_te=target_te,
                target_start_accessibility=target_start_accessibility,
                oracle=oracle,
                editable_regions=editable_regions,
                allow_stop=allow_stop,
                stop_logit_bias=stop_logit_bias,
                min_action_margin=min_action_margin,
                allow_action_space_expansion=allow_action_space_expansion,
            )
        elif model is not None and backbone is not None:
            cand = model_guided_edit_record(
                rec,
                model,
                backbone,
                task_id=task_id,
                edit_budget=edit_budget,
                target_length=target_length,
                seed=seed + i,
                device=device,
                proposal_top_k=proposal_top_k,
                proposal_temperature=proposal_temperature,
                guidance_scale=guidance_scale,
                target_te=target_te,
                target_start_accessibility=target_start_accessibility,
                oracle=oracle,
                editable_regions=editable_regions,
                allow_stop=allow_stop,
                stop_logit_bias=stop_logit_bias,
                min_action_margin=min_action_margin,
                allow_action_space_expansion=allow_action_space_expansion,
            )
        else:
            if editable_regions is not None:
                raise ValueError(
                    "editable_regions requires checkpoint-guided generation"
                )
            cand = sample_mrna(
                task_id=task_id,
                record=rec,
                edit_budget=edit_budget,
                target_length=target_length,
                guidance_scale=guidance_scale,
                target_te=target_te,
                target_start_accessibility=target_start_accessibility,
                oracle=oracle,
                guidance_candidates=max(1, int(proposal_top_k)),
                return_record=True,
                seed=seed + i,
            )
        out.append(cand)  # type: ignore[arg-type]
    return out


def write_candidates_jsonl(records: Sequence[MRNARecord], path: str) -> None:
    """Write generated candidates as canonical JSONL. Complexity: O(total nt)."""
    write_records_jsonl(records, path)


def _load_record(path: Optional[str]) -> Optional[MRNARecord]:
    if path is None:
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return MRNARecord.from_dict(json.load(fh))


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample/edit mRNA records with MEF constraints")
    parser.add_argument("--task-id", default="T1")
    parser.add_argument("--record-json", default=None)
    parser.add_argument("--records-jsonl", default=None, help="Batch source records JSONL")
    parser.add_argument("--output-jsonl", default=None, help="Write batch candidates JSONL")
    parser.add_argument("--checkpoint", default=None, help="Optional Stage A checkpoint for model-guided edits")
    parser.add_argument("--cascade-recall-checkpoint", default=None, help="Optional recall checkpoint for two-stage cascade decoding")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--proposal-top-k", type=int, default=8, help="Top-k legal proposals; <=0 evaluates all")
    parser.add_argument("--cascade-recall-top-k", type=int, default=64, help="Recall-stage top-k for cascade decoding")
    parser.add_argument("--proposal-temperature", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--edit-budget", type=int, default=None)
    parser.add_argument("--target-length", type=int, default=None)
    parser.add_argument("--target-length-delta", type=int, default=0)
    parser.add_argument("--motif", default=None)
    parser.add_argument("--motif-action", default="insert")
    parser.add_argument("--motif-region", default="5utr")
    parser.add_argument("--guidance-scale", type=float, default=0.0)
    parser.add_argument("--target-te", type=float, default=None)
    parser.add_argument("--target-start-accessibility", type=float, default=None)
    parser.add_argument("--guidance-candidates", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--return-seq", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    if args.records_jsonl is not None:
        candidates = generate_candidate_records(
            load_records_jsonl(args.records_jsonl),
            task_id=args.task_id,
            checkpoint_path=args.checkpoint,
            cascade_recall_checkpoint_path=args.cascade_recall_checkpoint,
            limit=args.limit,
            edit_budget=args.edit_budget if args.edit_budget is not None else max(1, args.steps),
            target_length_delta=args.target_length_delta,
            seed=args.seed,
            device=args.device,
            proposal_top_k=args.proposal_top_k,
            cascade_recall_top_k=args.cascade_recall_top_k,
            proposal_temperature=args.proposal_temperature,
            guidance_scale=args.guidance_scale,
            target_te=args.target_te,
            target_start_accessibility=args.target_start_accessibility,
        )
        if args.output_jsonl:
            write_candidates_jsonl(candidates, args.output_jsonl)
        else:
            for rec in candidates:
                print(json.dumps(rec.to_dict(), sort_keys=True))
        return 0

    out = sample_mrna(
        task_id=args.task_id,
        record=_load_record(args.record_json),
        steps=args.steps,
        edit_budget=args.edit_budget,
        target_length=args.target_length,
        motif=args.motif,
        motif_action=args.motif_action,
        motif_region=args.motif_region,
        guidance_scale=args.guidance_scale,
        target_te=args.target_te,
        target_start_accessibility=args.target_start_accessibility,
        guidance_candidates=args.guidance_candidates,
        return_record=not args.return_seq,
        seed=args.seed,
    )
    if isinstance(out, MRNARecord):
        print(json.dumps(out.to_dict(), sort_keys=True))
    else:
        print(out)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
