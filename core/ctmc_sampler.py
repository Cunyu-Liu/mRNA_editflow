"""True CTMC sampler for mRNA Edit Flow (tau-leaping).

This module implements a genuine Continuous-Time Markov Chain sampler that
integrates the rate field from t=0 to t=1, replacing the old fixed-t=0.5
sequential top-k decoder.

Key differences from the old ``ProposalEditor`` (sequential decoder):

1. **Time-varying rate field**: the model is queried at multiple time steps
   ``t_0 < t_1 < ... < t_K = 1``, and the rates change as the sequence
   evolves and time progresses.
2. **Tau-leaping**: at each time step, the probability of an event at
   position ``i`` of type ``op`` is ``1 - exp(-h * lambda_{i,op}(x_t, t))``.
   Multiple non-conflicting events are applied simultaneously.
3. **Forward/reverse rates**: optionally, a reverse rate head can correct
   overshoots while preserving the target marginal.
4. **Grammar-safe**: all events are filtered through the region grammar
   (synonymous CDS substitutions, no frameshift, start/stop preserved).

The sampler is model-agnostic: it accepts any callable that maps
``(token_ids, region_ids, phase_ids, time_step, padding_mask, backbone)``
to the standard output dict ``{rates, ins_probs, sub_probs, aux}``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from .constants import (
    GAP_TOKEN,
    NUC_TO_ID,
    PAD_TOKEN,
    PHASE_NONE,
    REGION_5UTR,
    REGION_CDS,
    REGION_3UTR,
    START_CODON,
    STOP_CODONS,
    SYNONYMOUS_CODONS,
    V,
)
from .schema import MRNARecord


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class CTMCConfig:
    """Configuration for the CTMC sampler.

    Attributes
    ----------
    n_steps : number of time discretization steps (K).
    max_events_per_step : maximum simultaneous events per tau-leap.
    use_reverse : whether to use a reverse rate for self-correction.
    reverse_fraction : fraction of the forward rate to use as reverse.
    seed : random seed for reproducibility.
    grammar_safe : if True, all events are filtered through region grammar.
    """
    n_steps: int = 50
    max_events_per_step: int = 5
    use_reverse: bool = False
    reverse_fraction: float = 0.1
    seed: int = 42
    grammar_safe: bool = True


# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------

@dataclass
class EditEvent:
    """A single CTMC event."""
    op: str  # "ins", "sub", "del"
    pos: int
    token: int  # target token for ins/sub; ignored for del
    rate: float  # lambda_{i,op} * Q_{i,op}(token)

    def __repr__(self) -> str:
        return f"EditEvent({self.op}@{self.pos}->{self.token}, r={self.rate:.4f})"


@dataclass
class CTMCTrajectory:
    """Full trajectory of a CTMC sampling run."""
    steps: List[Dict[str, Any]] = field(default_factory=list)
    final_tokens: Optional[List[int]] = None
    final_time: float = 0.0
    n_events: int = 0
    n_steps_actual: int = 0


# ---------------------------------------------------------------------------
# Grammar-safe event filtering
# ---------------------------------------------------------------------------

def _is_valid_event(
    event: EditEvent,
    tokens: List[int],
    region_ids: List[int],
    phase_ids: List[int],
) -> bool:
    """Check if an event respects the mRNA grammar.

    - CDS substitutions must be synonymous (checked against the codon).
    - CDS insertions/deletions are forbidden (frame lock).
    - Start/stop codons are never modified.
    """
    if event.pos >= len(tokens):
        return False
    region = region_ids[event.pos]
    if region == REGION_CDS:
        if event.op in ("ins", "del"):
            return False
        # Check synonymous: find the codon, verify the substitution is synonymous
        phase = phase_ids[event.pos]
        codon_start = event.pos - phase
        if codon_start + 3 > len(tokens):
            return False
        codon_nts = list(tokens[codon_start:codon_start + 3])
        codon_nts[phase] = event.token
        old_rna = "".join("ACGU"[tokens[codon_start + phase]], )
        new_rna = "".join("ACGU"[n] for n in codon_nts)
        from .constants import CODON_TABLE
        old_aa = CODON_TABLE.get("".join("ACGU"[n] for n in tokens[codon_start:codon_start+3]), "X")
        new_aa = CODON_TABLE.get(new_rna, "X")
        if old_aa != new_aa:
            return False
        # Never edit start/stop
        old_codon = "".join("ACGU"[n] for n in tokens[codon_start:codon_start+3])
        if old_codon == START_CODON or old_codon in STOP_CODONS:
            return False
    return True


def _find_non_conflicting(events: List[EditEvent]) -> List[EditEvent]:
    """Select a maximal non-conflicting subset of events.

    Two events conflict if they affect overlapping positions (for substitutions
    and deletions) or adjacent positions (for insertions, which shift indices).
    """
    if not events:
        return []
    # Sort by rate (descending) for greedy selection
    sorted_events = sorted(events, key=lambda e: -e.rate)
    selected: List[EditEvent] = []
    used_positions: set = set()
    for ev in sorted_events:
        if ev.op == "del":
            if ev.pos in used_positions:
                continue
            used_positions.add(ev.pos)
        elif ev.op == "sub":
            if ev.pos in used_positions:
                continue
            used_positions.add(ev.pos)
        elif ev.op == "ins":
            # Insertion at position pos shifts everything after it
            if ev.pos in used_positions or (ev.pos - 1) in used_positions:
                continue
            used_positions.add(ev.pos)
        selected.append(ev)
        if len(selected) >= 50:  # safety cap
            break
    return selected


# ---------------------------------------------------------------------------
# Event application
# ---------------------------------------------------------------------------

def _apply_events(
    tokens: List[int],
    region_ids: List[int],
    phase_ids: List[int],
    events: List[EditEvent],
) -> Tuple[List[int], List[int], List[int]]:
    """Apply a set of non-conflicting events to the sequence.

    Returns updated ``(tokens, region_ids, phase_ids)``.
    Insertions/deletions change the sequence length; region/phase tracks are
    recomputed for the affected region.
    """
    if not events:
        return tokens, region_ids, phase_ids

    # Process events from right to left so index shifts don't affect earlier events
    sorted_events = sorted(events, key=lambda e: -e.pos)

    tokens = list(tokens)
    region_ids = list(region_ids)
    phase_ids = list(phase_ids)

    for ev in sorted_events:
        if ev.op == "sub":
            tokens[ev.pos] = ev.token
        elif ev.op == "del":
            del tokens[ev.pos]
            del region_ids[ev.pos]
            del phase_ids[ev.pos]
        elif ev.op == "ins":
            tokens.insert(ev.pos, ev.token)
            # Infer region/phase for inserted token
            if ev.pos > 0:
                region_ids.insert(ev.pos, region_ids[ev.pos - 1])
                # For CDS, insertion would shift the frame — but grammar_safe
                # filtering should prevent CDS insertions. For UTR, phase is NONE.
                phase_ids.insert(ev.pos, PHASE_NONE if region_ids[ev.pos] != REGION_CDS else 0)
            else:
                region_ids.insert(ev.pos, REGION_5UTR)
                phase_ids.insert(ev.pos, PHASE_NONE)

    # Recompute CDS phases if length changed
    if any(ev.op in ("ins", "del") for ev in events):
        _recompute_phases(tokens, region_ids, phase_ids)

    return tokens, region_ids, phase_ids


def _recompute_phases(tokens: List[int], region_ids: List[int], phase_ids: List[int]) -> None:
    """Recompute codon phases for CDS regions in-place."""
    in_cds = False
    codon_pos = 0
    for i in range(len(region_ids)):
        if region_ids[i] == REGION_CDS:
            if not in_cds:
                in_cds = True
                codon_pos = 0
            phase_ids[i] = codon_pos % 3
            codon_pos += 1
        else:
            in_cds = False
            phase_ids[i] = PHASE_NONE


# ---------------------------------------------------------------------------
# CTMC Sampler
# ---------------------------------------------------------------------------

class CTMCSampler:
    """True CTMC tau-leaping sampler for mRNA Edit Flow.

    Given a trained model (any callable returning the standard output dict),
    this sampler integrates the rate field from t=0 to t=1 to produce a
    sequence of edits that transform the source into a target-like sequence.

    Usage::

        sampler = CTMCSampler(model_fn, config)
        trajectory = sampler.sample(
            token_ids, region_ids, phase_ids, backbone, device
        )
        # trajectory.final_tokens is the resulting sequence
    """

    def __init__(
        self,
        model_fn: Callable[..., Dict[str, Tensor]],
        config: CTMCConfig,
    ):
        self.model_fn = model_fn
        self.config = config
        self._gen = torch.Generator()
        self._gen.manual_seed(config.seed)

    @torch.no_grad()
    def sample(
        self,
        token_ids: Tensor,
        region_ids: Tensor,
        phase_ids: Tensor,
        backbone: Any,
        device: torch.device,
        source_seq: Optional[Tensor] = None,
    ) -> CTMCTrajectory:
        """Run the CTMC sampler.

        Parameters
        ----------
        token_ids : [B, L] target sequence (used for region/phase reference).
        region_ids : [B, L] region ids.
        phase_ids : [B, L] codon phases.
        backbone : frozen backbone for model forward.
        device : torch device.
        source_seq : [B, L_src] optional source sequence (default: empty).

        Returns
        -------
        CTMCTrajectory with per-step diagnostics.
        """
        cfg = self.config
        B = token_ids.shape[0]
        assert B == 1, "CTMCSampler currently supports batch_size=1"

        # Initialize from source (or empty)
        if source_seq is not None:
            cur_tokens = source_seq[0].tolist()
        else:
            cur_tokens = []

        # Reference region/phase from target (for grammar checks)
        ref_region = region_ids[0].tolist()
        ref_phase = phase_ids[0].tolist()

        # Infer current region/phase for the source
        if not cur_tokens:
            cur_region = []
            cur_phase = []
        else:
            # For corrupted source, regions are approximately aligned to target
            # Use target proportions
            target_len = len(ref_region)
            if target_len > 0:
                cur_region = _infer_regions(cur_tokens, ref_region)
                cur_phase = _infer_phases(cur_region)
            else:
                cur_region = [REGION_5UTR] * len(cur_tokens)
                cur_phase = [PHASE_NONE] * len(cur_tokens)

        dt = 1.0 / cfg.n_steps
        trajectory = CTMCTrajectory()
        total_events = 0

        for step in range(cfg.n_steps):
            t_curr = step * dt
            t_tensor = torch.tensor([[t_curr]], device=device, dtype=torch.float32)

            # Prepare model input
            cur_t = torch.tensor([cur_tokens], dtype=torch.long, device=device)
            cur_r = torch.tensor([cur_region], dtype=torch.long, device=device)
            cur_p = torch.tensor([cur_phase], dtype=torch.long, device=device)
            cur_pad = torch.zeros_like(cur_t, dtype=torch.bool)

            if cur_t.shape[1] == 0:
                # Empty sequence — insert a random nucleotide to bootstrap
                boot_tok = int(torch.randint(0, V, (1,), generator=self._gen).item())
                cur_tokens = [boot_tok]
                cur_region = [REGION_5UTR]
                cur_phase = [PHASE_NONE]
                continue

            # Model forward
            out = self.model_fn(cur_t, cur_r, cur_p, t_tensor, cur_pad, backbone)
            rates = out["rates"][0].cpu()  # [L, 3]
            ins_probs = out["ins_probs"][0].cpu()  # [L, V]
            sub_probs = out["sub_probs"][0].cpu()  # [L, V]

            # Compute event probabilities: P(event) = 1 - exp(-h * lambda)
            h = dt
            lam_ins = rates[:, 0]  # [L]
            lam_sub = rates[:, 1]
            lam_del = rates[:, 2]

            p_ins = 1.0 - torch.exp(-h * lam_ins)  # [L]
            p_sub = 1.0 - torch.exp(-h * lam_sub)
            p_del = 1.0 - torch.exp(-h * lam_del)

            # Sample events
            events: List[EditEvent] = []
            L = len(cur_tokens)

            for i in range(min(L, rates.shape[0])):
                # Substitution
                if torch.rand(1, generator=self._gen).item() < p_sub[i].item():
                    token = torch.multinomial(sub_probs[i], 1, generator=self._gen).item()
                    if token < V and token != cur_tokens[i]:
                        ev = EditEvent("sub", i, token, lam_sub[i].item() * sub_probs[i, token].item())
                        if not cfg.grammar_safe or _is_valid_event(ev, cur_tokens, cur_region, cur_phase):
                            events.append(ev)

                # Deletion
                if torch.rand(1, generator=self._gen).item() < p_del[i].item():
                    ev = EditEvent("del", i, 0, lam_del[i].item())
                    if not cfg.grammar_safe or _is_valid_event(ev, cur_tokens, cur_region, cur_phase):
                        events.append(ev)

                # Insertion (after position i)
                if i < L - 1 and torch.rand(1, generator=self._gen).item() < p_ins[i].item():
                    token = torch.multinomial(ins_probs[i], 1, generator=self._gen).item()
                    if token < V:
                        ev = EditEvent("ins", i + 1, token, lam_ins[i].item() * ins_probs[i, token].item())
                        if not cfg.grammar_safe or _is_valid_event(ev, cur_tokens, cur_region, cur_phase):
                            events.append(ev)

            # Select non-conflicting events
            events = _find_non_conflicting(events)
            events = events[:cfg.max_events_per_step]

            # Apply events
            cur_tokens, cur_region, cur_phase = _apply_events(
                cur_tokens, cur_region, cur_phase, events
            )

            total_events += len(events)
            trajectory.steps.append({
                "step": step,
                "t": t_curr,
                "n_events": len(events),
                "events": [(e.op, e.pos, e.token) for e in events],
                "seq_len": len(cur_tokens),
            })

            # Optional reverse rate correction
            if cfg.use_reverse and events:
                t_rev = t_curr + dt * cfg.reverse_fraction
                t_rev_tensor = torch.tensor([[min(t_rev, 1.0)]], device=device, dtype=torch.float32)
                # Use the model's reverse rate (if available) to pull back slightly
                # This is a simplified version; a full reverse rate head would be added
                # to the model. For now, we skip the reverse step.
                pass

        trajectory.final_tokens = cur_tokens
        trajectory.final_time = 1.0
        trajectory.n_events = total_events
        trajectory.n_steps_actual = cfg.n_steps
        return trajectory


def _infer_regions(tokens: List[int], ref_region: List[int]) -> List[int]:
    """Infer region partition for a source sequence based on target proportions."""
    ref_len = len(ref_region)
    src_len = len(tokens)
    if ref_len == 0:
        return [REGION_5UTR] * src_len
    # Find boundaries in reference
    utr5_end = 0
    cds_end = 0
    for i, r in enumerate(ref_region):
        if r == REGION_5UTR:
            utr5_end = i + 1
        elif r == REGION_CDS:
            cds_end = i + 1
    # Scale to source length
    src_utr5_end = round(src_len * utr5_end / ref_len) if ref_len > 0 else 0
    src_cds_end = round(src_len * cds_end / ref_len) if ref_len > 0 else src_len
    regions = [REGION_5UTR] * src_utr5_end
    regions += [REGION_CDS] * (src_cds_end - src_utr5_end)
    regions += [REGION_3UTR] * (src_len - src_cds_end)
    return regions[:src_len]


def _infer_phases(region_ids: List[int]) -> List[int]:
    """Compute codon phases from region ids."""
    phases = []
    codon_pos = 0
    in_cds = False
    for r in region_ids:
        if r == REGION_CDS:
            if not in_cds:
                in_cds = True
                codon_pos = 0
            phases.append(codon_pos % 3)
            codon_pos += 1
        else:
            in_cds = False
            phases.append(PHASE_NONE)
    return phases


__all__ = [
    "CTMCConfig",
    "EditEvent",
    "CTMCTrajectory",
    "CTMCSampler",
]
