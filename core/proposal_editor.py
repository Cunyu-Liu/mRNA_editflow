"""ProposalEditor: constrained proposal ranker / sequential editor (P0-4).

This module renames and wraps the existing sequential decoding logic
(:func:`mrna_editflow.sample.model_guided_edit_record`) behind a clear, honest
interface. :class:`ProposalEditor` is a **constrained proposal ranker**, NOT a
true Continuous-Time Markov Chain (CTMC) sampler:

* it queries the model at a **fixed** bridge time ``t`` (default ``0.5``) at
  every step, ignoring the time-varying rate field that a faithful CTMC
  integrator must follow;
* it enumerates all legal single-step edits (sub / ins / del) for the current
  region grammar, scores each by the CTMC intensity
  ``lambda_{i,op} * Q_{i,op}(token)``, and keeps the top-k (or STOP);
* it applies **one** edit per step and repeats until the edit budget is
  exhausted or STOP is chosen.

Because the time step never advances, the trajectories produced here are not
samples from the Edit-Flow path measure. For faithful flow integration use
:class:`mrna_editflow.core.ctmc_sampler.CTMCSampler` (tau-leaping over a
time-varying rate field). :class:`ProposalEditor` remains useful as a fast,
grammar-safe, deterministic-ish proposal ranker and as the historical baseline
against which the true CTMC sampler is compared.

The editor is model-agnostic: it accepts any callable with the same signature
as :class:`~mrna_editflow.core.ctmc_sampler.CTMCSampler`'s ``model_fn``::

    model_fn(token_ids, region_ids, phase_ids, time_step, padding_mask, backbone)
        -> {"rates": [B,L,3], "ins_probs": [B,L,V], "sub_probs": [B,L,V], "aux": ...}

Grammar safety is identical to ``sample.py``: T4 uses synonymous CDS
substitutions, T2/T3/T5 use UTR substitutions, T6 uses UTR indels toward a
target length; CDS indels and nonsynonymous substitutions are never proposed.

Complexity: ``O(E * (F + |C|*V))`` for edit budget ``E``, one model forward
``F`` per step, and a legal candidate pool of size ``|C|*V``.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

from .constants import (
    CODON_TABLE,
    NUC_TO_ID,
    NUC_VOCAB,
    REGION_CDS,
    SYNONYMOUS_CODONS,
    is_valid_cds,
    translate,
)
from .schema import MRNARecord
from ..rl.decoder_state import (
    DecoderAction,
    DecoderState,
    choose_stop_aware_action,
    sequence_hash,
)

# A model callable returning the standard Edit-Flow output dict.
ModelFn = Callable[..., Dict[str, torch.Tensor]]

# Candidate tuple: (rate, position, nucleotide) where ``rate = lambda * Q``.
_Candidate = Tuple[float, int, str]


@dataclass
class ProposalEditorConfig:
    """Configuration for :class:`ProposalEditor`.

    Attributes
    ----------
    time_step : float
        Fixed bridge time ``t`` used for scoring. NOT integrated over time.
    top_k : int
        Top-k legal proposals retained per step; ``<=0`` keeps the full pool.
    temperature : float
        Softmax temperature for stochastic top-k selection; ``<=0`` gives
        deterministic greedy decoding.
    allow_stop : bool
        Whether STOP is a selectable action.
    stop_logit_bias : float
        Explicit STOP baseline (log-score). STOP is not a learned head.
    min_action_margin : float
        Minimum margin of the best edit log-score over STOP to avoid stopping.
    seed : int
        RNG seed for reproducible stochastic selection.
    """

    time_step: float = 0.5
    top_k: int = 8
    temperature: float = 1.0
    allow_stop: bool = True
    stop_logit_bias: float = 0.0
    min_action_margin: float = 0.0
    seed: int = 0


class ProposalEditor:
    """Constrained proposal ranker / sequential editor.

    .. warning::

        This is **NOT a true CTMC sampler**. The model is always queried at the
        same fixed time ``t`` (default ``0.5``) and only one edit is applied per
        step. Use :class:`mrna_editflow.core.ctmc_sampler.CTMCSampler` for
        faithful flow integration with a time-varying rate field.

    Parameters
    ----------
    model_fn : callable
        Any callable with the ``CTMCSampler`` model interface
        ``(token_ids, region_ids, phase_ids, time_step, padding_mask, backbone)
        -> dict``. Typically ``model.forward``.
    config : ProposalEditorConfig
        Decoder hyper-parameters (fixed time step, top-k, STOP policy, seed).

    Notes
    -----
    The editor reuses :class:`~mrna_editflow.rl.decoder_state.DecoderState` for
    cycle detection / budget / STOP bookkeeping and
    :func:`~mrna_editflow.rl.decoder_state.choose_stop_aware_action` for the
    STOP-aware top-k selection, so its selection semantics match the historical
    ``model_guided_edit_record`` decoder exactly.
    """

    def __init__(
        self,
        model_fn: ModelFn,
        config: ProposalEditorConfig = ProposalEditorConfig(),
    ) -> None:
        self.model_fn = model_fn
        self.config = config

    # ------------------------------------------------------------------
    @torch.no_grad()
    def edit(
        self,
        record: MRNARecord,
        *,
        task_id: str = "T5",
        edit_budget: int = 3,
        backbone: Any = None,
        device: Optional[str] = None,
        target_length: Optional[int] = None,
        editable_regions: Optional[Sequence[str]] = None,
    ) -> MRNARecord:
        """Run the sequential proposal editor on ``record``.

        Parameters
        ----------
        record : MRNARecord
            Source transcript to edit (copied; never mutated).
        task_id : str
            One of ``T2/T3/T4/T5/T6``. T4 = synonymous CDS substitutions,
            T2/T3/T5 = UTR substitutions, T6 = UTR indels toward
            ``target_length``.
        edit_budget : int
            Upper bound on the number of applied edits.
        backbone : object, optional
            Frozen backbone passed through to ``model_fn``.
        device : str, optional
            Torch device (defaults to CPU; tests need no GPU).
        target_length : int, optional
            Target full-sequence length for T6.
        editable_regions : sequence of str, optional
            UTR regions available to the decoder (``utr5``/``utr3`` aliases).
            Ignored for T4, which is always CDS-only.

        Returns
        -------
        MRNARecord
            The edited record with ``decoder_type="proposal_editor"`` and the
            :class:`DecoderState` metadata attached.

        Complexity: ``O(E * (F + |C|*V))`` for budget ``E`` and one forward
        ``F`` per step.
        """
        cfg = self.config
        tid = task_id.upper()
        if tid not in {"T2", "T3", "T4", "T5", "T6"}:
            raise ValueError("ProposalEditor supports T2/T3/T4/T5/T6")
        rng = random.Random(int(cfg.seed))
        dev = torch.device(device or "cpu")
        base_tid = record.transcript_id
        current = _copy_record(record, transcript_id=f"{base_tid}_{tid.lower()}_proposal")
        budget = max(0, int(edit_budget))
        if tid == "T6" and target_length is None:
            target_length = len(record.seq)
        state = DecoderState(current.seq, budget)

        for _step in range(budget):
            out = _model_out_for_record(current, self.model_fn, backbone, dev, cfg.time_step)
            choices, op, make_record = self._candidates(
                current, out, tid, editable_regions, target_length, base_tid
            )
            if not choices:
                break
            actions = self._build_actions(current, choices, op, make_record)
            chosen = choose_stop_aware_action(
                actions,
                state,
                rng,
                top_k=cfg.top_k,
                temperature=cfg.temperature,
                allow_stop=cfg.allow_stop,
                stop_logit_bias=cfg.stop_logit_bias,
                min_action_margin=cfg.min_action_margin,
            )
            if chosen is None:
                break
            current = make_record(int(chosen.pos), str(chosen.nt or ""))
            _validate(current)

        return _attach_metadata(
            _validate(current),
            state=state,
            time_step=float(cfg.time_step),
        )

    # ------------------------------------------------------------------
    def _candidates(
        self,
        current: MRNARecord,
        out: Dict[str, torch.Tensor],
        tid: str,
        editable_regions: Optional[Sequence[str]],
        target_length: Optional[int],
        base_tid: str,
    ) -> Tuple[List[_Candidate], str, Callable[[int, str], MRNARecord]]:
        """Enumerate grammar-legal candidates for the current task.

        Returns ``(choices, op, make_record)`` where ``choices`` is sorted by
        descending rate ``lambda * Q`` and ``make_record(pos, nt)`` materialises
        one candidate into a valid :class:`MRNARecord`.
        """
        tid_tag = f"{base_tid}_{tid.lower()}_proposal"
        if tid == "T4":
            make = lambda pos, nt: _replace_nt(current, pos, nt, tid_tag)  # noqa: E731
            return _synonymous_substitution_candidates(current, out), "sub", make
        if tid in {"T2", "T3", "T5"}:
            make = lambda pos, nt: _replace_nt(current, pos, nt, tid_tag)  # noqa: E731
            return _utr_substitution_candidates(current, out, editable_regions), "sub", make
        if tid == "T6":
            delta = int(target_length) - len(current.seq)  # type: ignore[arg-type]
            if delta == 0:
                return [], "sub", lambda pos, nt: current  # noqa: E731
            if delta > 0:
                make = lambda pos, nt: _insert_nt_after(current, pos, nt, tid_tag)  # noqa: E731
                return _utr_insert_candidates(current, out, editable_regions), "ins", make
            make = lambda pos, nt: _delete_nt(current, pos, tid_tag)  # noqa: E731
            return _utr_delete_candidates(current, out, editable_regions), "del", make
        raise ValueError(f"unsupported task {tid!r}")  # pragma: no cover - guarded above

    # ------------------------------------------------------------------
    def _build_actions(
        self,
        current: MRNARecord,
        choices: Sequence[_Candidate],
        op: str,
        make_record: Callable[[int, str], MRNARecord],
    ) -> List[DecoderAction]:
        """Wrap scored candidates as :class:`DecoderAction` objects.

        The CTMC intensity ``lambda * Q`` (the candidate rate) is converted to a
        log-score ``log(max(rate, eps))`` so the shared STOP-aware selector
        (:func:`choose_stop_aware_action`) ranks identically to the historical
        decoder. ``eps = 1e-20`` matches :mod:`mrna_editflow.rl.action_scoring`.
        """
        actions: List[DecoderAction] = []
        for rate, pos, nt in choices:
            candidate = make_record(int(pos), str(nt))
            log_score = math.log(max(float(rate), 1e-20))
            actions.append(
                DecoderAction(
                    op=op,
                    pos=int(pos),
                    nt=str(nt) or None,
                    log_score=log_score,
                    next_sequence_hash=sequence_hash(candidate.seq),
                    old_nt=current.seq[int(pos)] if op == "sub" else None,
                )
            )
        return actions


# ===========================================================================
# Record helpers (ported from sample.py, kept self-contained)
# ===========================================================================
def _copy_record(rec: MRNARecord, transcript_id: Optional[str] = None) -> MRNARecord:
    return MRNARecord(
        transcript_id=transcript_id or rec.transcript_id,
        five_utr=rec.five_utr,
        cds=rec.cds,
        three_utr=rec.three_utr,
        species=rec.species,
        metadata=dict(rec.metadata),
    )


def _normalise_nt(seq: str) -> str:
    seq = seq.upper().replace("T", "U")
    if any(ch not in NUC_VOCAB for ch in seq):
        raise ValueError(f"sequence contains non-ACGU characters: {seq!r}")
    return seq


def _validate(rec: MRNARecord) -> MRNARecord:
    """Normalise to RNA alphabet and assert CDS validity post-edit."""
    checked = MRNARecord(
        transcript_id=rec.transcript_id,
        five_utr=_normalise_nt(rec.five_utr),
        cds=_normalise_nt(rec.cds),
        three_utr=_normalise_nt(rec.three_utr),
        species=rec.species,
        metadata=dict(rec.metadata),
    )
    if not is_valid_cds(checked.cds):
        raise ValueError("proposal editor produced an invalid mRNA record")
    return checked


def _attach_metadata(
    record: MRNARecord,
    *,
    state: DecoderState,
    time_step: float,
) -> MRNARecord:
    record.metadata.update(
        {
            "decoder_type": "proposal_editor",
            "is_ctmc_sampler": False,
            "time_step": float(time_step),
            "checkpoint_path": None,
            "checkpoint_sha256": None,
            "oracle_guidance_used": False,
            **state.to_metadata(),
        }
    )
    return record


# ===========================================================================
# Tensor conversion + model forward (fixed time step)
# ===========================================================================
def _record_tensors(record: MRNARecord, device: torch.device):
    """Convert a record to model tensors ``[1, L]`` without padding.

    Region ids and codon phases are aligned to nucleotide positions.
    Complexity: O(L).
    """
    token_ids = torch.tensor([record.token_ids()], dtype=torch.long, device=device)
    region_ids = torch.tensor([record.region_ids()], dtype=torch.long, device=device)
    phase_ids = torch.tensor([record.codon_phases()], dtype=torch.long, device=device)
    padding_mask = torch.zeros_like(token_ids, dtype=torch.bool)
    return token_ids, region_ids, phase_ids, padding_mask


def _model_out_for_record(
    record: MRNARecord,
    model_fn: ModelFn,
    backbone: Any,
    device: torch.device,
    time_value: float = 0.5,
) -> Dict[str, torch.Tensor]:
    """One model forward at the FIXED bridge time ``t=time_value``.

    Complexity: one model forward ``O(layers * L^2 * dim)``.
    """
    token_ids, region_ids, phase_ids, padding_mask = _record_tensors(record, device)
    t = torch.full((1, 1), float(time_value), dtype=torch.float32, device=device)
    return model_fn(token_ids, region_ids, phase_ids, t, padding_mask, backbone)


# ===========================================================================
# Rate scoring: lambda_{i,op} * Q_{i,op}(token)
# ===========================================================================
def _rate_score(
    out: Dict[str, torch.Tensor], op: str, pos: int, nt: Optional[str]
) -> float:
    """Return the CTMC intensity ``lambda_{i,op} * Q_{i,op}(token)``.

    * ``sub``: ``rates[pos, sub] * sub_probs[pos, nt]``
    * ``ins``: ``rates[pos, ins] * ins_probs[pos, nt]``
    * ``del``: ``rates[pos, del]`` (no token distribution)

    Complexity: O(1).
    """
    rates = out["rates"][0]  # [L, 3]
    pos_i = int(pos)
    if op == "sub":
        if nt is None or nt not in NUC_TO_ID:
            raise ValueError(f"sub requires an A/C/G/U nucleotide, got {nt!r}")
        return float(rates[pos_i, 1].item()) * float(
            out["sub_probs"][0, pos_i, NUC_TO_ID[nt]].item()
        )
    if op == "ins":
        if nt is None or nt not in NUC_TO_ID:
            raise ValueError(f"ins requires an A/C/G/U nucleotide, got {nt!r}")
        return float(rates[pos_i, 0].item()) * float(
            out["ins_probs"][0, pos_i, NUC_TO_ID[nt]].item()
        )
    if op == "del":
        return float(rates[pos_i, 2].item())
    raise ValueError(f"unsupported op {op!r}; expected sub/ins/del")


# ===========================================================================
# Region-aware edit appliers (full-sequence positions; CDS indels forbidden)
# ===========================================================================
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


# ===========================================================================
# Candidate enumeration (grammar-legal; sorted by descending rate)
# ===========================================================================
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
        [editable_regions] if isinstance(editable_regions, str) else list(editable_regions)
    )
    normalized: List[str] = []
    for value in values:
        key = str(value).strip().lower().replace("_", "")
        if key not in aliases:
            raise ValueError("editable_regions must contain only utr5/utr3 aliases")
        region = aliases[key]
        if region not in normalized:
            normalized.append(region)
    if not normalized:
        raise ValueError("editable_regions must not be empty")
    return tuple(normalized)


def _utr_positions(
    record: MRNARecord,
    editable_regions: Optional[Sequence[str]] = None,
) -> List[int]:
    """Return UTR positions enabled for candidate generation.

    Complexity: O(L).
    """
    regions = _normalise_editable_utr_regions(editable_regions)
    positions: List[int] = []
    if "utr5" in regions:
        positions.extend(range(len(record.five_utr)))
    if "utr3" in regions:
        offset = len(record.five_utr) + len(record.cds)
        positions.extend(offset + i for i in range(len(record.three_utr)))
    return positions


def _synonymous_substitution_candidates(
    record: MRNARecord, out: Dict[str, torch.Tensor]
) -> List[_Candidate]:
    """Enumerate protein-preserving CDS substitutions scored by ``lambda*Q``.

    Start (AUG) and terminal stop codons are skipped. A candidate is emitted
    only when the one-base edited codon still translates to the original amino
    acid, so every proposed intermediate is itself a legal synonymous state.

    Note: this enumerates single-nucleotide synonymous subs only (the states
    reachable by the historical nt-level head). Multi-nucleous synonymous codon
    transitions (e.g. Arg ``CGU -> AGA``) are unreachable here and require the
    atomic codon head in :mod:`mrna_editflow.core.codon_head`.

    Complexity: ``O(N_codon * 3 * |V|)``.
    """
    candidates: List[_Candidate] = []
    five_len = len(record.five_utr)
    codons = [record.cds[i : i + 3] for i in range(0, len(record.cds), 3)]
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
                edited = codon[:offset] + new + codon[offset + 1 :]
                if CODON_TABLE.get(edited) != aa:
                    continue
                rate = _rate_score(out, "sub", pos, new)
                candidates.append((rate, pos, new))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates


def _utr_substitution_candidates(
    record: MRNARecord,
    out: Dict[str, torch.Tensor],
    editable_regions: Optional[Sequence[str]] = None,
) -> List[_Candidate]:
    """Enumerate UTR substitutions scored by ``lambda*Q``.

    Complexity: ``O(|UTR| * |V|)``.
    """
    seq = record.seq
    candidates: List[_Candidate] = []
    for pos in _utr_positions(record, editable_regions):
        old = seq[pos]
        for nt in NUC_VOCAB:
            if nt == old:
                continue
            rate = _rate_score(out, "sub", pos, nt)
            candidates.append((rate, pos, nt))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates


def _utr_insert_candidates(
    record: MRNARecord,
    out: Dict[str, torch.Tensor],
    editable_regions: Optional[Sequence[str]] = None,
) -> List[_Candidate]:
    """Enumerate UTR insertions scored by ``lambda*Q``.

    Complexity: ``O(|UTR| * |V|)``.
    """
    candidates: List[_Candidate] = []
    for pos in _utr_positions(record, editable_regions):
        for nt in NUC_VOCAB:
            rate = _rate_score(out, "ins", pos, nt)
            candidates.append((rate, pos, nt))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates


def _utr_delete_candidates(
    record: MRNARecord,
    out: Dict[str, torch.Tensor],
    editable_regions: Optional[Sequence[str]] = None,
) -> List[_Candidate]:
    """Enumerate UTR deletions scored by ``lambda_del``.

    Complexity: ``O(|UTR|)``.
    """
    candidates: List[_Candidate] = [
        (_rate_score(out, "del", pos, None), pos, "") for pos in _utr_positions(record, editable_regions)
    ]
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates


__all__ = [
    "ProposalEditor",
    "ProposalEditorConfig",
    "ModelFn",
]
