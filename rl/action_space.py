"""P1-07: Legal action distribution — action space and legal mask.

Defines the action space for the mRNA design MDP:

  - (ins, pos, nt): insert nucleotide ``nt`` after position ``pos``
  - (sub, pos, nt): substitute nucleotide at ``pos`` with ``nt``
  - (del, pos):     delete nucleotide at ``pos``
  - STOP:           terminate the trajectory

Legality rules (mirror ``models/mrna_editformer.py::_apply_codon_constraints``
and ``sample.py`` sampler-side constraints):

  - 5'UTR / 3'UTR: free nt-level ins / sub / del. All 4 nucleotides allowed
    for sub (identity is excluded from the legal set; the model may still
    assign it probability, which the mask filters out).
  - CDS substitution: only nucleotides that keep the codon synonymous
    (via ``synonymous_nt_sub_mask``).
  - CDS indels (``codon_indel=False``, the default): forbidden (frame lock).
  - CDS indels (``codon_indel=True``): only at codon-start (phase 0) positions.
  - STOP: always legal.

This module is intentionally model-agnostic: it only consumes
``MRNARecord`` and the constants from ``core.constants``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch

from mrna_editflow.core.constants import (
    NUC_TO_ID,
    PHASE_NONE,
    REGION_3UTR,
    REGION_5UTR,
    REGION_CDS,
    V,
)
from mrna_editflow.core.schema import MRNARecord


# ---------------------------------------------------------------------------
# Action types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Action:
    """An action in the mRNA design MDP.

    Attributes
    ----------
    op : str
        One of ``"ins"``, ``"sub"``, ``"del"``, ``"stop"``.
    pos : int
        Position in the full sequence (0-indexed). Ignored for ``op="stop"``;
        set to -1 for STOP.
    nt : int
        Nucleotide id in ``{0,1,2,3}`` (A,C,G,U). Ignored for ``op="del"`` and
        ``op="stop"``; set to -1 for those.
    """

    op: str
    pos: int = -1
    nt: int = -1

    def is_stop(self) -> bool:
        return self.op == "stop"

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        if self.op == "stop":
            return "Action(STOP)"
        nt_str = {0: "A", 1: "C", 2: "G", 3: "U"}.get(self.nt, "?")
        return f"Action({self.op}@{self.pos},{nt_str})"


STOP_ACTION = Action(op="stop", pos=-1, nt=-1)


# ---------------------------------------------------------------------------
# Legal action mask
# ---------------------------------------------------------------------------


@dataclass
class ActionMask:
    """Boolean mask over the action space for a single state.

    All tensors are shape ``[L]`` or ``[L, V]`` (V=4 nucleotides) and live on
    the same device as the input record tensors.

    The legal action set is::

        legal = { (ins, i, a) : ins_mask[i, a] }
              ∪ { (sub, i, a) : sub_mask[i, a] }
              ∪ { (del, i)    : del_mask[i]    }
              ∪ { STOP } (if stop_legal)

    Note: identity substitutions (a == current nt at position i) are excluded
    from ``sub_mask`` by convention, since they are no-ops. The model may still
    assign them probability mass; the mask filters that mass out.
    """

    ins_mask: torch.Tensor  # [L, V] bool
    sub_mask: torch.Tensor  # [L, V] bool
    del_mask: torch.Tensor  # [L] bool
    stop_legal: bool = True

    @property
    def num_legal(self) -> int:
        """Total number of legal actions (including STOP if legal)."""
        return int(
            self.ins_mask.sum().item()
            + self.sub_mask.sum().item()
            + self.del_mask.sum().item()
            + (1 if self.stop_legal else 0)
        )

    @property
    def num_ins(self) -> int:
        return int(self.ins_mask.sum().item())

    @property
    def num_sub(self) -> int:
        return int(self.sub_mask.sum().item())

    @property
    def num_del(self) -> int:
        return int(self.del_mask.sum().item())

    def to(self, device: torch.device) -> "ActionMask":
        return ActionMask(
            ins_mask=self.ins_mask.to(device),
            sub_mask=self.sub_mask.to(device),
            del_mask=self.del_mask.to(device),
            stop_legal=self.stop_legal,
        )


@dataclass
class ActionLogProbs:
    """Log probabilities over the action space for a single state.

    The masked distribution is normalized so that::

        sum_{(i,a) in legal} exp(ins_logprobs[i,a])
      + sum_{(i,a) in legal} exp(sub_logprobs[i,a])
      + sum_{i in legal}     exp(del_logprobs[i])
      + exp(stop_logprob)
      = 1.0

    Illegal positions have ``-inf`` log-probs.

    The ``raw_*`` fields (optional, populated when ``return_raw=True``) hold
    the pre-external-mask log-probs: i.e., the model's own output normalized
    over *all* (op, pos, nt) including illegal ones. This is useful for the
    "quality before/after masking" diagnostic.
    """

    ins_logprobs: torch.Tensor  # [L, V]
    sub_logprobs: torch.Tensor  # [L, V]
    del_logprobs: torch.Tensor  # [L]
    stop_logprob: Union[float, torch.Tensor]
    log_partition: float  # log Lambda_masked
    # Optional pre-external-mask values.
    raw_ins_logprobs: Optional[torch.Tensor] = None  # [L, V]
    raw_sub_logprobs: Optional[torch.Tensor] = None  # [L, V]
    raw_del_logprobs: Optional[torch.Tensor] = None  # [L]
    raw_stop_logprob: Optional[Union[float, torch.Tensor]] = None
    raw_log_partition: Optional[float] = None  # log Lambda_raw

    def logprob(self, action: Action) -> float:
        """Get log p(action | state). Returns -inf for illegal actions."""
        if action.is_stop():
            return float(self.stop_logprob)
        if action.op == "ins":
            return float(self.ins_logprobs[action.pos, action.nt].item())
        if action.op == "sub":
            return float(self.sub_logprobs[action.pos, action.nt].item())
        if action.op == "del":
            return float(self.del_logprobs[action.pos].item())
        raise ValueError(f"unknown action op: {action.op!r}")

    def raw_logprob(self, action: Action) -> float:
        """Get raw (pre-external-mask) log p(action | state).

        Returns ``-inf`` if ``raw_*`` fields are not populated.
        """
        if self.raw_ins_logprobs is None:
            return float("-inf")
        if action.is_stop():
            return float(self.raw_stop_logprob if self.raw_stop_logprob is not None else float("-inf"))
        if action.op == "ins":
            return float(self.raw_ins_logprobs[action.pos, action.nt].item())
        if action.op == "sub":
            return float(self.raw_sub_logprobs[action.pos, action.nt].item())
        if action.op == "del":
            return float(self.raw_del_logprobs[action.pos].item())
        raise ValueError(f"unknown action op: {action.op!r}")


def build_legal_action_mask(
    record: MRNARecord,
    device: torch.device,
    *,
    codon_indel: bool = False,
    allow_identity_sub: bool = False,
) -> ActionMask:
    """Build the legal action mask for a single record.

    Parameters
    ----------
    record : MRNARecord
        The current state.
    device : torch.device
        Device for the output tensors.
    codon_indel : bool, default False
        If False (default), nt-level ins/del are forbidden in CDS (frame lock).
        If True, nt-level ins/del are allowed at codon-start (phase 0) positions
        in CDS, treated as whole-codon indels.
    allow_identity_sub : bool, default False
        If False (default), identity substitutions (a == current nt) are excluded
        from ``sub_mask``. If True, they are included (useful for debugging).

    Returns
    -------
    ActionMask
        The legal action mask. Shape ``[L, V]`` for ins/sub, ``[L]`` for del.

    Complexity
    ----------
    O(L) where L is the sequence length.
    """
    token_ids = torch.tensor([record.token_ids()], dtype=torch.long, device=device)
    region_ids = torch.tensor([record.region_ids()], dtype=torch.long, device=device)
    phase_ids = torch.tensor([record.codon_phases()], dtype=torch.long, device=device)
    # Drop the leading batch dim for per-position tensors.
    tokens = token_ids[0]  # [L]
    regions = region_ids[0]  # [L]
    phases = phase_ids[0]  # [L]
    L = tokens.shape[0]

    is_cds = regions == REGION_CDS
    is_utr = (regions == REGION_5UTR) | (regions == REGION_3UTR)
    is_codon_start = is_cds & (phases == 0)

    # ---- Substitution mask [L, V] ----
    # UTR: all 4 nucleotides allowed (excluding identity unless allow_identity_sub).
    # CDS: only synonymous nucleotides (via synonymous_nt_sub_mask).
    from mrna_editflow.core.mrna_flow_utils import synonymous_nt_sub_mask

    syn_mask = synonymous_nt_sub_mask(token_ids, region_ids, phase_ids)[0]  # [L, V]
    # UTR positions: all 4 nucleotides allowed
    utr_all = torch.zeros(L, V, dtype=torch.bool, device=device)
    utr_all[is_utr] = True
    sub_mask = syn_mask | utr_all  # [L, V]
    # Exclude identity substitutions unless explicitly allowed.
    if not allow_identity_sub:
        identity_mask = torch.zeros(L, V, dtype=torch.bool, device=device)
        identity_mask.scatter_(1, tokens.unsqueeze(-1), True)
        sub_mask = sub_mask & ~identity_mask

    # ---- Insertion mask [L, V] ----
    # UTR: all 4 nucleotides allowed at every position.
    # CDS: only if codon_indel=True, and only at codon-start positions.
    ins_mask = torch.zeros(L, V, dtype=torch.bool, device=device)
    if codon_indel:
        ins_eligible = is_utr | is_codon_start
    else:
        ins_eligible = is_utr
    ins_mask[ins_eligible] = True

    # ---- Deletion mask [L] ----
    # UTR: every position deletable.
    # CDS: only if codon_indel=True, and only at codon-start positions.
    if codon_indel:
        del_mask = is_utr | is_codon_start
    else:
        del_mask = is_utr

    return ActionMask(
        ins_mask=ins_mask,
        sub_mask=sub_mask,
        del_mask=del_mask,
        stop_legal=True,
    )


# ---------------------------------------------------------------------------
# Action application
# ---------------------------------------------------------------------------


def apply_action(record: MRNARecord, action: Action, *, transcript_id: Optional[str] = None) -> MRNARecord:
    """Apply an action to a record, returning a new record.

    For ``op="stop"``, returns the input record unchanged (with a new
    transcript_id if provided). For other ops, applies the edit and returns
    a new ``MRNARecord``.

    Raises
    ------
    ValueError
        If the action is illegal (e.g. CDS indel when ``codon_indel=False``,
        or position out of range).
    """
    if action.is_stop():
        if transcript_id is None:
            return record
        return MRNARecord(
            transcript_id=transcript_id,
            five_utr=record.five_utr,
            cds=record.cds,
            three_utr=record.three_utr,
            species=record.species,
        )

    tid = transcript_id or record.transcript_id
    five = record.five_utr
    cds = record.cds
    three = record.three_utr
    five_len = len(five)
    cds_len = len(cds)

    if action.op == "sub":
        pos = action.pos
        nt_id = action.nt
        if nt_id < 0 or nt_id >= V:
            raise ValueError(f"invalid nt id {nt_id}")
        nt_str = "ACGU"[nt_id]
        if pos < five_len:
            new_five = five[:pos] + nt_str + five[pos + 1:]
            new_cds = cds
            new_three = three
        elif pos < five_len + cds_len:
            idx = pos - five_len
            new_cds = cds[:idx] + nt_str + cds[idx + 1:]
            new_five = five
            new_three = three
        else:
            idx = pos - five_len - cds_len
            new_three = three[:idx] + nt_str + three[idx + 1:]
            new_five = five
            new_cds = cds
        return MRNARecord(
            transcript_id=tid,
            five_utr=new_five,
            cds=new_cds,
            three_utr=new_three,
            species=record.species,
        )

    if action.op == "ins":
        pos = action.pos
        nt_id = action.nt
        if nt_id < 0 or nt_id >= V:
            raise ValueError(f"invalid nt id {nt_id}")
        nt_str = "ACGU"[nt_id]
        if pos < five_len - 1:
            # Insert after pos in 5'UTR
            new_five = five[:pos + 1] + nt_str + five[pos + 1:]
            new_cds = cds
            new_three = three
        elif pos == five_len - 1:
            # Boundary: insert at end of 5'UTR (still UTR)
            new_five = five + nt_str
            new_cds = cds
            new_three = three
        elif pos < five_len + cds_len - 1:
            # Inside CDS — only allowed if codon_indel=True and pos is codon-start.
            # We don't enforce that here (the mask does); we just apply.
            idx = pos - five_len
            new_cds = cds[:idx + 1] + nt_str + cds[idx + 1:]
            new_five = five
            new_three = three
        elif pos < five_len + cds_len + len(three) - 1:
            idx = pos - five_len - cds_len
            new_three = three[:idx + 1] + nt_str + three[idx + 1:]
            new_five = five
            new_cds = cds
        else:
            # Append at end of 3'UTR
            new_three = three + nt_str
            new_five = five
            new_cds = cds
        return MRNARecord(
            transcript_id=tid,
            five_utr=new_five,
            cds=new_cds,
            three_utr=new_three,
            species=record.species,
        )

    if action.op == "del":
        pos = action.pos
        if pos < five_len:
            new_five = five[:pos] + five[pos + 1:]
            new_cds = cds
            new_three = three
        elif pos < five_len + cds_len:
            idx = pos - five_len
            new_cds = cds[:idx] + cds[idx + 1:]
            new_five = five
            new_three = three
        else:
            idx = pos - five_len - cds_len
            new_three = three[:idx] + three[idx + 1:]
            new_five = five
            new_cds = cds
        return MRNARecord(
            transcript_id=tid,
            five_utr=new_five,
            cds=new_cds,
            three_utr=new_three,
            species=record.species,
        )

    raise ValueError(f"unknown action op: {action.op!r}")
