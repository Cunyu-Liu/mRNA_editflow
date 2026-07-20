"""P1-07: Legal action distribution — Policy API.

Wraps :class:`MRNAEditFormer` into a stochastic policy over the mRNA design
MDDP action space (ins / sub / del / STOP).

The action distribution is the normalized CTMC intensity field:

    q(ins, i, a | s) = rates[i, 0] * ins_probs[i, a]    (masked to legal)
    q(sub, i, a | s) = rates[i, 1] * sub_probs[i, a]    (masked to legal)
    q(del, i   | s) = rates[i, 2]                       (masked to legal)
    q(STOP      | s) = stop_rate(s)
    Lambda(s)        = sum of all q(a | s)
    p(a | s)         = q(a | s) / Lambda(s)

The model already applies ``_apply_codon_constraints`` internally, so
``out["rates"]`` and ``out["sub_probs"]`` are partially legality-masked. This
Policy adds an *external* legal mask (sampler-specific constraints: skip codon
0 / terminal stop / non-identity substitutions) on top, and exposes both the
"raw" (model-output) and "masked" (post-external-mask) distributions for
quality comparison.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch

from mrna_editflow.core.constants import V
from mrna_editflow.core.schema import MRNARecord
from mrna_editflow.rl.action_space import (
    STOP_ACTION,
    Action,
    ActionMask,
    ActionLogProbs,
    apply_action,
    build_legal_action_mask,
)


# ---------------------------------------------------------------------------
# Policy config
# ---------------------------------------------------------------------------


@dataclass
class PolicyConfig:
    """Configuration for :class:`Policy`.

    Attributes
    ----------
    stop_rate_strategy : str
        - ``"constant"``: ``stop_rate = stop_rate_value``
        - ``"budget_aware"``: ``stop_rate = stop_rate_value * (1 - budget_remaining/budget_total)``
          (encourages STOP when budget is exhausted)
    stop_rate_value : float
        Base stop rate. Must be > 0.
    temperature : float
        Temperature for sampling. ``1.0`` = exact CTMC sampling; ``>1`` = more
        uniform; ``<1`` = more greedy.
    time_step : float
        Flow time at which to evaluate the model (default 0.5, matching
        ``sample.py``).
    codon_indel : bool
        Whether nt-level indels are allowed in CDS at codon-start positions.
        Default ``False`` (frame lock).
    """

    stop_rate_strategy: str = "constant"
    stop_rate_value: float = 1.0
    temperature: float = 1.0
    time_step: float = 0.5
    codon_indel: bool = False

    def __post_init__(self) -> None:
        if self.stop_rate_strategy not in ("constant", "budget_aware"):
            raise ValueError(
                f"stop_rate_strategy must be 'constant' or 'budget_aware', got {self.stop_rate_strategy!r}"
            )
        if self.stop_rate_value <= 0:
            raise ValueError(f"stop_rate_value must be > 0, got {self.stop_rate_value}")
        if self.temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {self.temperature}")
        if not (0.0 <= self.time_step <= 1.0):
            raise ValueError(f"time_step must be in [0, 1], got {self.time_step}")


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


class Policy:
    """mRNA design policy wrapping :class:`MRNAEditFormer`.

    The policy exposes a normalized action distribution over the legal action
    set, supports STOP, and computes trajectory log-probabilities.

    Parameters
    ----------
    model : MRNAEditFormer
        The trained (or training) model. Must have a ``forward`` method
        returning ``{"rates": [B,L,3], "ins_probs": [B,L,V], "sub_probs": [B,L,V], ...}``.
    backbone : FrozenBackbone
        The frozen backbone, used by ``model.forward``.
    cfg : PolicyConfig
        Policy configuration.
    device : torch.device
        Device for tensors.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        backbone: torch.nn.Module,
        cfg: PolicyConfig,
        device: torch.device,
    ) -> None:
        self.model = model
        self.backbone = backbone
        self.cfg = cfg
        self.device = device

    # ------------------------------------------------------------------
    # State encoding
    # ------------------------------------------------------------------

    def _record_tensors(self, record: MRNARecord) -> Tuple[torch.Tensor, ...]:
        """Convert a record to model tensors ``[1, L]`` without padding."""
        token_ids = torch.tensor([record.token_ids()], dtype=torch.long, device=self.device)
        region_ids = torch.tensor([record.region_ids()], dtype=torch.long, device=self.device)
        phase_ids = torch.tensor([record.codon_phases()], dtype=torch.long, device=self.device)
        padding_mask = torch.zeros_like(token_ids, dtype=torch.bool)
        return token_ids, region_ids, phase_ids, padding_mask

    def _model_forward(self, record: MRNARecord, no_grad: bool = True) -> dict:
        """Run one model forward pass at ``cfg.time_step``.

        Parameters
        ----------
        no_grad : bool, default True
            If True (default), run under ``torch.no_grad()`` for inference.
            If False, run with grad enabled (for REINFORCE policy gradient).
        """
        token_ids, region_ids, phase_ids, padding_mask = self._record_tensors(record)
        t = torch.full((1, 1), float(self.cfg.time_step), dtype=torch.float32, device=self.device)
        if no_grad:
            with torch.no_grad():
                out = self.model.forward(token_ids, region_ids, phase_ids, t, padding_mask, self.backbone)
        else:
            out = self.model.forward(token_ids, region_ids, phase_ids, t, padding_mask, self.backbone)
        return out

    # ------------------------------------------------------------------
    # Stop rate
    # ------------------------------------------------------------------

    def _stop_rate(
        self,
        record: MRNARecord,
        budget_remaining: Optional[int] = None,
        budget_total: Optional[int] = None,
    ) -> float:
        """Compute the STOP rate for the current state."""
        if self.cfg.stop_rate_strategy == "constant":
            return float(self.cfg.stop_rate_value)
        if self.cfg.stop_rate_strategy == "budget_aware":
            if budget_total is None or budget_total <= 0 or budget_remaining is None:
                return float(self.cfg.stop_rate_value)
            frac_remaining = max(0.0, min(1.0, budget_remaining / budget_total))
            # When budget is full, stop_rate = stop_rate_value.
            # When budget is exhausted, stop_rate = stop_rate_value * 2 (double).
            return float(self.cfg.stop_rate_value * (2.0 - frac_remaining))
        raise ValueError(f"unknown stop_rate_strategy: {self.cfg.stop_rate_strategy!r}")

    # ------------------------------------------------------------------
    # Legal action mask
    # ------------------------------------------------------------------

    def legal_action_mask(self, record: MRNARecord) -> ActionMask:
        """Build the legal action mask for ``record``."""
        return build_legal_action_mask(
            record,
            self.device,
            codon_indel=self.cfg.codon_indel,
            allow_identity_sub=False,
        )

    # ------------------------------------------------------------------
    # Action log-probabilities
    # ------------------------------------------------------------------

    def action_logprobs(
        self,
        record: MRNARecord,
        *,
        budget_remaining: Optional[int] = None,
        budget_total: Optional[int] = None,
        return_raw: bool = False,
        no_grad: bool = True,
    ) -> ActionLogProbs:
        """Compute log p(action | record) for all legal actions.

        Parameters
        ----------
        record : MRNARecord
            Current state.
        budget_remaining, budget_total : optional
            For ``stop_rate_strategy="budget_aware"``.
        return_raw : bool, default False
            If True, also populate ``raw_*`` fields with the *pre-external-mask*
            log-probs (i.e., the model output as-is, which already includes the
            internal ``_apply_codon_constraints`` mask).
        no_grad : bool, default True
            If True (default), run model forward under ``torch.no_grad()``.
            If False, run with grad enabled (for REINFORCE policy gradient).
            When ``no_grad=False``, the returned log-prob tensors will have
            grad history and can be used in ``loss.backward()``.

        Returns
        -------
        ActionLogProbs
            Log probabilities over the action space. The masked distribution
            is normalized so that ``sum of exp(logprob) over legal actions = 1``.

        Notes
        -----
        The "raw" distribution (``return_raw=True``) is the model's own output
        normalized over all (op, pos, nt) including illegal ones. Comparing
        ``raw_log_partition`` vs ``log_partition`` and the per-action raw vs
        masked log-probs gives the "quality before/after masking" diagnostic.
        """
        out = self._model_forward(record, no_grad=no_grad)
        rates = out["rates"][0].float()  # [L, 3]
        # ins_probs/sub_probs have V_dim=VOCAB_MODEL_SIZE=6 (A,C,G,U,BOS,PAD).
        # The legal mask only covers nucleotides 0..3, so slice to V=4.
        ins_probs = out["ins_probs"][0, :, :V].float()  # [L, V]
        sub_probs = out["sub_probs"][0, :, :V].float()  # [L, V]

        # Raw unnormalized q(action) — model output as-is.
        q_raw_ins = rates[:, 0:1] * ins_probs  # [L, V]
        q_raw_sub = rates[:, 1:2] * sub_probs  # [L, V]
        q_raw_del = rates[:, 2]  # [L]
        q_raw_stop = self._stop_rate(record, budget_remaining, budget_total)

        # Total raw mass (over all actions, legal or not).
        Lambda_raw = float(
            q_raw_ins.sum().item() + q_raw_sub.sum().item() + q_raw_del.sum().item() + q_raw_stop
        )
        if Lambda_raw <= 0 or not math.isfinite(Lambda_raw):
            # Degenerate: model produced all-zero rates. Fall back to uniform-over-STOP.
            return self._degenerate_logprobs(record, return_raw)

        log_Lambda_raw = math.log(Lambda_raw)

        # Raw log-probs (pre-external-mask).
        raw_ins_logprobs = (q_raw_ins / Lambda_raw).clamp_min(1e-30).log()
        raw_sub_logprobs = (q_raw_sub / Lambda_raw).clamp_min(1e-30).log()
        raw_del_logprobs = (q_raw_del / Lambda_raw).clamp_min(1e-30).log()
        raw_stop_logprob = math.log(max(q_raw_stop, 1e-30)) - log_Lambda_raw

        # External legal mask.
        mask = self.legal_action_mask(record)

        # Masked unnormalized q(action) — zero out illegal actions.
        q_masked_ins = q_raw_ins * mask.ins_mask.float()
        q_masked_sub = q_raw_sub * mask.sub_mask.float()
        q_masked_del = q_raw_del * mask.del_mask.float()
        q_masked_stop = q_raw_stop if mask.stop_legal else 0.0

        Lambda_masked = float(
            q_masked_ins.sum().item()
            + q_masked_sub.sum().item()
            + q_masked_del.sum().item()
            + q_masked_stop
        )
        if Lambda_masked <= 0 or not math.isfinite(Lambda_masked):
            # All legal actions have zero rate. Force STOP to be the only legal action.
            return self._force_stop_logprobs(record, return_raw, raw_ins_logprobs, raw_sub_logprobs, raw_del_logprobs, raw_stop_logprob, log_Lambda_raw)

        log_Lambda_masked = math.log(Lambda_masked)

        # Masked log-probs (normalized over legal actions only).
        masked_ins_logprobs = (q_masked_ins / Lambda_masked).clamp_min(1e-30).log()
        masked_sub_logprobs = (q_masked_sub / Lambda_masked).clamp_min(1e-30).log()
        masked_del_logprobs = (q_masked_del / Lambda_masked).clamp_min(1e-30).log()
        masked_stop_logprob = math.log(max(q_masked_stop, 1e-30)) - log_Lambda_masked
        # If stop was illegal (shouldn't happen by default), set logprob to -inf.
        if not mask.stop_legal:
            masked_stop_logprob = float("-inf")

        # Zero out illegal positions in the masked log-probs for clarity.
        masked_ins_logprobs = masked_ins_logprobs.masked_fill(~mask.ins_mask, float("-inf"))
        masked_sub_logprobs = masked_sub_logprobs.masked_fill(~mask.sub_mask, float("-inf"))
        masked_del_logprobs = masked_del_logprobs.masked_fill(~mask.del_mask, float("-inf"))

        return ActionLogProbs(
            ins_logprobs=masked_ins_logprobs,
            sub_logprobs=masked_sub_logprobs,
            del_logprobs=masked_del_logprobs,
            stop_logprob=masked_stop_logprob,
            log_partition=log_Lambda_masked,
            raw_ins_logprobs=raw_ins_logprobs if return_raw else None,
            raw_sub_logprobs=raw_sub_logprobs if return_raw else None,
            raw_del_logprobs=raw_del_logprobs if return_raw else None,
            raw_stop_logprob=raw_stop_logprob if return_raw else None,
            raw_log_partition=log_Lambda_raw if return_raw else None,
        )

    def _degenerate_logprobs(self, record: MRNARecord, return_raw: bool) -> ActionLogProbs:
        """All-zero model output: put all mass on STOP."""
        L = len(record.token_ids())
        V = 4
        device = self.device
        neg_inf = float("-inf")
        ins_lp = torch.full((L, V), neg_inf, device=device)
        sub_lp = torch.full((L, V), neg_inf, device=device)
        del_lp = torch.full((L,), neg_inf, device=device)
        return ActionLogProbs(
            ins_logprobs=ins_lp,
            sub_logprobs=sub_lp,
            del_logprobs=del_lp,
            stop_logprob=0.0,
            log_partition=0.0,
            raw_ins_logprobs=ins_lp.clone() if return_raw else None,
            raw_sub_logprobs=sub_lp.clone() if return_raw else None,
            raw_del_logprobs=del_lp.clone() if return_raw else None,
            raw_stop_logprob=0.0 if return_raw else None,
            raw_log_partition=0.0 if return_raw else None,
        )

    def _force_stop_logprobs(
        self,
        record: MRNARecord,
        return_raw: bool,
        raw_ins_logprobs: torch.Tensor,
        raw_sub_logprobs: torch.Tensor,
        raw_del_logprobs: torch.Tensor,
        raw_stop_logprob: float,
        log_Lambda_raw: float,
    ) -> ActionLogProbs:
        """Legal action set is empty (except STOP): put all mass on STOP."""
        L = raw_ins_logprobs.shape[0]
        V = raw_ins_logprobs.shape[1]
        device = raw_ins_logprobs.device
        neg_inf = float("-inf")
        ins_lp = torch.full((L, V), neg_inf, device=device)
        sub_lp = torch.full((L, V), neg_inf, device=device)
        del_lp = torch.full((L,), neg_inf, device=device)
        return ActionLogProbs(
            ins_logprobs=ins_lp,
            sub_logprobs=sub_lp,
            del_logprobs=del_lp,
            stop_logprob=0.0,
            log_partition=0.0,
            raw_ins_logprobs=raw_ins_logprobs if return_raw else None,
            raw_sub_logprobs=raw_sub_logprobs if return_raw else None,
            raw_del_logprobs=raw_del_logprobs if return_raw else None,
            raw_stop_logprob=raw_stop_logprob if return_raw else None,
            raw_log_partition=log_Lambda_raw if return_raw else None,
        )

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(
        self,
        record: MRNARecord,
        *,
        budget_remaining: Optional[int] = None,
        budget_total: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[Action, float]:
        """Sample an action from the policy.

        Returns
        -------
        action : Action
            The sampled action.
        log_prob : float
            The log-probability of the sampled action under the (masked,
            temperature-adjusted) policy.

        Notes
        -----
        Temperature ``T`` rescales the log-probs as ``logprob / T`` before
        sampling. This changes the distribution; the returned ``log_prob`` is
        under the temperature-adjusted distribution.
        """
        lps = self.action_logprobs(
            record, budget_remaining=budget_remaining, budget_total=budget_total, return_raw=False
        )
        T = float(self.cfg.temperature)

        # Build flat (logprob, action_index) list.
        # Action indexing:
        #   [0, L*V)        -> (ins, i, a)   with i = idx // V, a = idx % V
        #   [L*V, 2*L*V)    -> (sub, i, a)
        #   [2*L*V, 2*L*V+L) -> (del, i)
        #   2*L*V + L       -> STOP
        ins_lp = lps.ins_logprobs  # [L, V]
        sub_lp = lps.sub_logprobs  # [L, V]
        del_lp = lps.del_logprobs  # [L]
        stop_lp = lps.stop_logprob

        L, V = ins_lp.shape
        device = ins_lp.device

        # Flatten and apply temperature.
        flat_ins = (ins_lp.view(-1) / T).clone()
        flat_sub = (sub_lp.view(-1) / T).clone()
        flat_del = (del_lp.view(-1) / T).clone()
        flat_stop = torch.tensor([stop_lp / T], device=device, dtype=flat_ins.dtype)

        # Replace -inf with very negative finite for softmax stability.
        neg_large = torch.finfo(flat_ins.dtype).min
        flat_ins = flat_ins.masked_fill(~torch.isfinite(flat_ins), neg_large)
        flat_sub = flat_sub.masked_fill(~torch.isfinite(flat_sub), neg_large)
        flat_del = flat_del.masked_fill(~torch.isfinite(flat_del), neg_large)
        if not math.isfinite(flat_stop.item()):
            flat_stop = torch.tensor([neg_large], device=device, dtype=flat_ins.dtype)

        flat_all = torch.cat([flat_ins, flat_sub, flat_del, flat_stop])
        # Softmax to get sampling probabilities.
        probs = torch.softmax(flat_all, dim=0)
        # Sample.
        if generator is not None:
            sample_idx = torch.multinomial(probs, num_samples=1, generator=generator).item()
        else:
            sample_idx = torch.multinomial(probs, num_samples=1).item()
        sample_logprob = float(torch.log(probs[sample_idx] + 1e-30).item())

        # Decode sample_idx to Action.
        if sample_idx < L * V:
            i = sample_idx // V
            a = sample_idx % V
            return Action(op="ins", pos=i, nt=a), sample_logprob
        elif sample_idx < 2 * L * V:
            idx = sample_idx - L * V
            i = idx // V
            a = idx % V
            return Action(op="sub", pos=i, nt=a), sample_logprob
        elif sample_idx < 2 * L * V + L:
            i = sample_idx - 2 * L * V
            return Action(op="del", pos=i, nt=-1), sample_logprob
        else:
            return STOP_ACTION, sample_logprob

    # ------------------------------------------------------------------
    # Trajectory log-prob
    # ------------------------------------------------------------------

    def trajectory_logprob(
        self,
        trajectory: List[Tuple[MRNARecord, Action]],
        *,
        budget_total: Optional[int] = None,
    ) -> float:
        """Compute log p(trajectory) = sum_t log p(a_t | s_t).

        Parameters
        ----------
        trajectory : list of (state, action) pairs
            The trajectory. The last action should typically be STOP; the state
            for the STOP action is the state *before* STOP (i.e., the final
            edited record).
        budget_total : optional
            For ``stop_rate_strategy="budget_aware"``. If provided,
            ``budget_remaining`` at step ``t`` is ``budget_total - t``.

        Returns
        -------
        float
            The trajectory log-probability. ``-inf`` if any action has zero
            probability under the policy.
        """
        total = 0.0
        for t, (state, action) in enumerate(trajectory):
            budget_remaining = (
                max(0, budget_total - t) if budget_total is not None else None
            )
            lps = self.action_logprobs(
                state,
                budget_remaining=budget_remaining,
                budget_total=budget_total,
                return_raw=False,
            )
            lp = lps.logprob(action)
            if not math.isfinite(lp):
                return float("-inf")
            total += lp
        return total

    # ------------------------------------------------------------------
    # Quality diagnostics
    # ------------------------------------------------------------------

    def quality_before_after_masking(
        self,
        record: MRNARecord,
        *,
        budget_remaining: Optional[int] = None,
        budget_total: Optional[int] = None,
    ) -> dict:
        """Compute quality metrics before and after external legality masking.

        Returns a dict with:
            raw_mass_on_legal: float — total probability mass on legal actions
                                        under the raw (pre-external-mask) distribution.
            raw_mass_on_illegal: float — mass on illegal actions.
            masked_mass_on_legal: float — should always be 1.0.
            num_legal_actions: int — number of legal actions.
            log_partition_raw: float — log Lambda_raw (pre-external-mask).
            log_partition_masked: float — log Lambda_masked (post-external-mask).
            mass_loss: float — raw_mass_on_illegal (== 1 - raw_mass_on_legal).
        """
        lps = self.action_logprobs(
            record,
            budget_remaining=budget_remaining,
            budget_total=budget_total,
            return_raw=True,
        )
        mask = self.legal_action_mask(record)

        # Raw mass on legal actions.
        raw_ins = torch.exp(lps.raw_ins_logprobs) * mask.ins_mask.float()
        raw_sub = torch.exp(lps.raw_sub_logprobs) * mask.sub_mask.float()
        raw_del = torch.exp(lps.raw_del_logprobs) * mask.del_mask.float()
        raw_stop = math.exp(lps.raw_stop_logprob) if mask.stop_legal else 0.0

        raw_mass_on_legal = float(
            raw_ins.sum().item() + raw_sub.sum().item() + raw_del.sum().item() + raw_stop
        )
        raw_mass_on_illegal = 1.0 - raw_mass_on_legal

        return {
            "raw_mass_on_legal": raw_mass_on_legal,
            "raw_mass_on_illegal": raw_mass_on_illegal,
            "masked_mass_on_legal": 1.0,
            "num_legal_actions": mask.num_legal,
            "num_ins": mask.num_ins,
            "num_sub": mask.num_sub,
            "num_del": mask.num_del,
            "log_partition_raw": lps.raw_log_partition,
            "log_partition_masked": lps.log_partition,
            "mass_loss": raw_mass_on_illegal,
        }
