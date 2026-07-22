"""P2-05: Group Relative Policy Optimization (GRPO) for mRNA design.

Unlike the EMA-baseline REINFORCE in :mod:`mrna_editflow.rl.tiny_mdp`, GRPO
normalizes advantages *within a group* of N trajectories sampled from the
SAME starting state:

    A_{b,i} = (G_{b,i} - mean_i(G_{b,*})) / (std_i(G_{b,*}) + eps)

This removes the need for a learned value function and reduces variance when
the reward distribution is heavy-tailed (common for predicted-TE oracles).

Reward signal
-------------
The intended reward for the P2-05 pilot is the **delta predicted TE** from
Oracle #3 (P1-05 GBT regressor), which is a *predicted / internal proxy* for
translation efficiency. Until P2-01 multi-region oracle validation completes,
any claim that GRPO "improves TE" MUST be qualified as "improves predicted TE
(internal proxy)".

Blocker
-------
The full pilot is BLOCKED until P2-02 produces a fixed checkpoint to use as
the policy backbone. The advantage-computation core and the tiny-MDP trainer
are provided here so they can be unit-tested independently.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from mrna_editflow.rl.tiny_mdp import (
    REINFORCE,
    TinyMDP,
    Trajectory,
    compute_returns,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class GRPOConfig:
    """Configuration for GRPO.

    Attributes
    ----------
    group_size : int
        Number of trajectories per starting state (N). Must be >= 2 for
        meaningful normalization.
    eps : float
        Numerical stability for std normalization. Must be > 0.
    clip_advantage : float
        Optional clipping of normalized advantages to [-clip, +clip].
        0.0 means no clipping.
    kl_coef : float
        Coefficient for the KL penalty term ``KL(pi || pi_ref)``. The
        penalty keeps the trained policy close to the reference policy
        (typically the warm-start checkpoint). 0.0 disables the penalty.
    entropy_coef : float
        Coefficient for the entropy bonus ``-coef * H(pi)``. The bonus
        encourages exploration. 0.0 disables the bonus.
    """

    group_size: int = 8
    eps: float = 1e-8
    clip_advantage: float = 0.0
    kl_coef: float = 0.0
    entropy_coef: float = 0.0

    def __post_init__(self) -> None:
        if self.group_size < 2:
            raise ValueError(
                f"group_size must be >= 2 for normalization, got {self.group_size}"
            )
        if self.eps <= 0:
            raise ValueError(f"eps must be > 0, got {self.eps}")
        if self.clip_advantage < 0:
            raise ValueError(
                f"clip_advantage must be >= 0, got {self.clip_advantage}"
            )
        if self.kl_coef < 0:
            raise ValueError(f"kl_coef must be >= 0, got {self.kl_coef}")
        if self.entropy_coef < 0:
            raise ValueError(
                f"entropy_coef must be >= 0, got {self.entropy_coef}"
            )


# ---------------------------------------------------------------------------
# Core: group-normalized advantages
# ---------------------------------------------------------------------------


def group_normalized_advantages(
    returns: torch.Tensor,
    cfg: Optional[GRPOConfig] = None,
) -> torch.Tensor:
    """Compute group-normalized advantages.

    Parameters
    ----------
    returns : torch.Tensor
        Shape ``[N]`` (single group) or ``[B, N]`` (batch of groups). Each
        row / vector is the returns of N trajectories sampled from the same
        starting state. Normalization is per-row (per starting state).
    cfg : GRPOConfig, optional
        Configuration. If None, defaults are used and ``group_size`` is
        inferred from ``returns.shape[-1]``.

    Returns
    -------
    advantages : torch.Tensor
        Same shape as ``returns``. Each row has mean ~0 and std ~1.

    Raises
    ------
    ValueError
        If ``returns`` is not 1D or 2D, or if its last dim does not match
        ``cfg.group_size`` (when ``cfg`` is provided).
    """
    if cfg is None:
        cfg = GRPOConfig(group_size=int(returns.shape[-1]))
    if returns.shape[-1] != cfg.group_size:
        raise ValueError(
            f"returns last dim {returns.shape[-1]} != cfg.group_size "
            f"{cfg.group_size}"
        )
    if returns.dim() == 1:
        mean = returns.mean()
        std = returns.std(unbiased=False)
    elif returns.dim() == 2:
        mean = returns.mean(dim=-1, keepdim=True)
        std = returns.std(dim=-1, keepdim=True, unbiased=False)
    else:
        raise ValueError(
            f"returns must be 1D or 2D, got {returns.dim()}D"
        )
    adv = (returns - mean) / (std + cfg.eps)
    if cfg.clip_advantage > 0:
        adv = adv.clamp(-cfg.clip_advantage, cfg.clip_advantage)
    return adv


# ---------------------------------------------------------------------------
# Trainer: GRPO-style REINFORCE on TinyMDP
# ---------------------------------------------------------------------------


class GRPOREINFORCE(REINFORCE):
    """GRPO-style REINFORCE with group-normalized advantages.

    Subclasses :class:`mrna_editflow.rl.tiny_mdp.REINFORCE` and overrides
    :meth:`compute_loss` to use group-normalized advantages instead of an
    EMA baseline.

    The policy gradient is:

        loss = -mean_{b,i}( A_{b,i} * sum_t log pi(a_{b,i,t} | s_{b,i,t}) )

    where ``A_{b,i} = (G_{b,i} - mean_i(G_{b,*})) / (std_i(G_{b,*}) + eps)``
    and ``b`` indexes the starting state, ``i`` indexes the trajectory within
    the group.
    """

    def __init__(
        self,
        policy,  # rl.policy.Policy
        mdp: TinyMDP,
        cfg: GRPOConfig,
        lr: float = 0.01,
        ref_policy=None,  # Optional[rl.policy.Policy]
    ) -> None:
        # Skip REINFORCE.__init__ baseline setup; we use group normalization.
        self.policy = policy
        self.mdp = mdp
        self.cfg = cfg
        self.lr = lr
        self.use_baseline = False
        self.baseline_decay = 0.0
        self.baseline = 0.0
        self.ref_policy = ref_policy
        self.optimizer = torch.optim.SGD(
            self.policy.model.parameters(),
            lr=lr,
        )

    # ------------------------------------------------------------------
    # Group collection
    # ------------------------------------------------------------------

    def collect_group(
        self,
        generator: Optional[torch.Generator] = None,
    ) -> List[Trajectory]:
        """Collect a group of N trajectories from the SAME initial state.

        The MDP is reset to the same initial state for each of the N
        trajectories so that the only source of variation is the policy's
        stochastic sampling.
        """
        if self.cfg.group_size < 2:
            raise ValueError("group_size must be >= 2")
        group: List[Trajectory] = []
        for _ in range(self.cfg.group_size):
            group.append(self.collect_trajectory(generator=generator))
        return group

    # ------------------------------------------------------------------
    # Loss with group-normalized advantages
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        groups: List[List[Trajectory]],
    ) -> Tuple[torch.Tensor, dict]:
        """Compute the GRPO loss over a batch of trajectory groups.

        Parameters
        ----------
        groups : List[List[Trajectory]]
            ``B`` groups, each with ``N`` trajectories (``N = cfg.group_size``).

        Returns
        -------
        loss : torch.Tensor (scalar)
            The GRPO loss (ready for ``.backward()``).
        metrics : dict
            Diagnostic metrics.
        """
        B = len(groups)
        N = self.cfg.group_size
        for g in groups:
            if len(g) != N:
                raise ValueError(
                    f"each group must have {N} trajectories, got {len(g)}"
                )

        # Returns matrix [B, N] — first-step return G_0 = full discounted
        # return (episodic, no bootstrap). We use G_0 as the group signal.
        returns_mat = torch.zeros(B, N, device=self.policy.device)
        for b, group in enumerate(groups):
            for i, traj in enumerate(group):
                rets = compute_returns(traj.transitions, self.mdp.gamma)
                returns_mat[b, i] = float(rets[0]) if rets else 0.0

        adv_mat = group_normalized_advantages(returns_mat, self.cfg)

        total_loss = torch.zeros((), device=self.policy.device)
        total_logprob = 0.0
        total_return = 0.0
        total_advantage_sq = 0.0
        total_kl = 0.0
        total_entropy = 0.0
        n_steps = 0

        for b, group in enumerate(groups):
            for i, traj in enumerate(group):
                adv = float(adv_mat[b, i].item())
                for t, transition in enumerate(traj.transitions):
                    lps = self.policy.action_logprobs(
                        transition.state,
                        budget_remaining=self.mdp.max_steps - transition.step,
                        budget_total=self.mdp.max_steps,
                        no_grad=False,
                    )
                    if transition.action.is_stop():
                        diff_lp = self._differentiable_stop_logprob(lps)
                    elif transition.action.op == "ins":
                        diff_lp = lps.ins_logprobs[transition.action.pos, transition.action.nt]
                    elif transition.action.op == "sub":
                        diff_lp = lps.sub_logprobs[transition.action.pos, transition.action.nt]
                    elif transition.action.op == "del":
                        diff_lp = lps.del_logprobs[transition.action.pos]
                    else:
                        continue
                    if not math.isfinite(float(diff_lp.item())):
                        continue
                    # Policy gradient term: -advantage * log pi(a|s)
                    total_loss = total_loss - adv * diff_lp
                    total_logprob += float(diff_lp.item())
                    total_return += float(rets[t]) if t < len(rets) else 0.0
                    total_advantage_sq += adv * adv

                    # KL penalty term: +kl_coef * (log pi(a|s) - log pi_ref(a|s))
                    if self.cfg.kl_coef > 0 and self.ref_policy is not None:
                        ref_lps = self.ref_policy.action_logprobs(
                            transition.state,
                            budget_remaining=self.mdp.max_steps - transition.step,
                            budget_total=self.mdp.max_steps,
                            no_grad=True,
                        )
                        if transition.action.is_stop():
                            ref_lp = float(ref_lps.stop_logprob)
                        elif transition.action.op == "ins":
                            ref_lp = float(ref_lps.ins_logprobs[transition.action.pos, transition.action.nt].item())
                        elif transition.action.op == "sub":
                            ref_lp = float(ref_lps.sub_logprobs[transition.action.pos, transition.action.nt].item())
                        elif transition.action.op == "del":
                            ref_lp = float(ref_lps.del_logprobs[transition.action.pos].item())
                        else:
                            ref_lp = 0.0
                        kl_sample = float(diff_lp.item()) - ref_lp
                        total_kl += kl_sample
                        total_loss = total_loss + self.cfg.kl_coef * kl_sample

                    # Entropy bonus term: -entropy_coef * H(pi)
                    if self.cfg.entropy_coef > 0:
                        entropy = self._distribution_entropy(lps)
                        total_entropy += float(entropy.item())
                        total_loss = total_loss - self.cfg.entropy_coef * entropy

                    n_steps += 1

        if n_steps > 0:
            total_loss = total_loss / n_steps
            total_logprob /= n_steps
            total_return /= n_steps
            total_advantage_sq /= n_steps
            total_kl /= n_steps
            total_entropy /= n_steps

        metrics = {
            "loss": float(total_loss.item()),
            "mean_logprob": total_logprob,
            "mean_return": float(returns_mat.mean().item()),
            "mean_advantage": float(adv_mat.mean().item()),
            "mean_advantage_sq": float((adv_mat ** 2).mean().item()),
            "mean_kl": total_kl,
            "mean_entropy": total_entropy,
            "n_steps": n_steps,
            "n_groups": B,
            "group_size": N,
            "return_std_mean": float(returns_mat.std(dim=-1).mean().item()),
        }
        return total_loss, metrics

    # ------------------------------------------------------------------
    # Entropy of the action distribution
    # ------------------------------------------------------------------

    @staticmethod
    def _distribution_entropy(lps) -> torch.Tensor:
        """Compute the entropy H(pi) = -sum_a pi(a) * log pi(a).

        Uses the masked log-probs (post-external-mask, normalized over legal
        actions). Returns a scalar tensor on the same device as ``lps``.
        """
        device = lps.ins_logprobs.device if hasattr(lps.ins_logprobs, "device") else torch.device("cpu")
        eps = 1e-12
        # Gather all log-probs into a flat tensor, then compute entropy.
        logprobs = []
        if hasattr(lps, "ins_logprobs") and lps.ins_logprobs is not None:
            logprobs.append(lps.ins_logprobs.reshape(-1))
        if hasattr(lps, "sub_logprobs") and lps.sub_logprobs is not None:
            logprobs.append(lps.sub_logprobs.reshape(-1))
        if hasattr(lps, "del_logprobs") and lps.del_logprobs is not None:
            logprobs.append(lps.del_logprobs.reshape(-1))
        if hasattr(lps, "stop_logprob") and lps.stop_logprob is not None:
            # stop_logprob may be a Python float
            try:
                stop_lp = torch.tensor(float(lps.stop_logprob), device=device)
            except (TypeError, ValueError):
                stop_lp = None
            if stop_lp is not None:
                logprobs.append(stop_lp.reshape(-1))
        if not logprobs:
            return torch.zeros((), device=device)
        all_lp = torch.cat(logprobs)
        # Only finite log-probs contribute (illegal actions have -inf).
        finite_mask = torch.isfinite(all_lp)
        if not finite_mask.any():
            return torch.zeros((), device=device)
        finite_lp = all_lp[finite_mask]
        probs = torch.exp(finite_lp)
        # Renormalize over finite actions (should already sum to ~1, but guard
        # against numerical drift).
        probs = probs / (probs.sum() + eps)
        entropy = -(probs * finite_lp).sum()
        return entropy

    # ------------------------------------------------------------------
    # Memory-efficient loss with per-trajectory backward (gradient accum)
    # ------------------------------------------------------------------

    def _compute_loss_grad_accum(
        self,
        groups: List[List[Trajectory]],
    ) -> dict:
        """Memory-efficient loss computation with per-trajectory backward.

        Computes the same loss as :meth:`compute_loss` but calls
        ``.backward()`` on each trajectory's contribution immediately,
        freeing the computation graph. This avoids accumulating
        ``B * N * T`` forward-pass graphs simultaneously, which causes CUDA
        OOM on long sequences (e.g. 981 nt CDS, 116M-param EditFormer).

        MUST be called after ``self.optimizer.zero_grad()`` — gradients are
        accumulated into ``.grad`` directly via per-trajectory backward.

        Returns
        -------
        metrics : dict
            Diagnostic metrics. The loss tensor is NOT returned (backward
            is already called). Use ``metrics["loss"]`` for the scalar value.

        Notes
        -----
        - Pre-counts total transitions ``n_steps_estimated`` as the divisor
          for per-trajectory backward, so accumulated gradients match the
          original ``total_loss / n_steps`` semantics in the common case
          (no transitions skipped due to non-finite log-probs).
        - If some transitions are skipped (rare), gradients are scaled by
          ``1/n_steps_estimated`` instead of ``1/actual_n_steps``, a minor
          approximation acceptable for memory efficiency.
        - Preserves the original KL-term behavior: ``kl_sample`` is a Python
          float, so the KL term contributes to loss VALUE only (no gradient).
        """
        B = len(groups)
        N = self.cfg.group_size
        for g in groups:
            if len(g) != N:
                raise ValueError(
                    f"each group must have {N} trajectories, got {len(g)}"
                )

        # Returns matrix [B, N] and advantages (no grad).
        returns_mat = torch.zeros(B, N, device=self.policy.device)
        for b, group in enumerate(groups):
            for i, traj in enumerate(group):
                rets = compute_returns(traj.transitions, self.mdp.gamma)
                returns_mat[b, i] = float(rets[0]) if rets else 0.0
        adv_mat = group_normalized_advantages(returns_mat, self.cfg)

        # Pre-count total transitions as the divisor for per-trajectory
        # backward. This matches `n_steps` in compute_loss when no
        # transitions are skipped (the common case).
        n_steps_estimated = 0
        for b, group in enumerate(groups):
            for i, traj in enumerate(group):
                n_steps_estimated += len(traj.transitions)

        if n_steps_estimated == 0:
            return {
                "loss": 0.0,
                "mean_logprob": 0.0,
                "mean_return": float(returns_mat.mean().item()),
                "mean_advantage": float(adv_mat.mean().item()),
                "mean_advantage_sq": float((adv_mat ** 2).mean().item()),
                "mean_kl": 0.0,
                "mean_entropy": 0.0,
                "n_steps": 0,
                "n_groups": B,
                "group_size": N,
                "return_std_mean": float(returns_mat.std(dim=-1).mean().item()),
            }

        total_logprob = 0.0
        total_return = 0.0
        total_advantage_sq = 0.0
        total_kl = 0.0
        total_entropy = 0.0
        total_loss_value = 0.0
        actual_n_steps = 0

        for b, group in enumerate(groups):
            for i, traj in enumerate(group):
                adv = float(adv_mat[b, i].item())
                rets = compute_returns(traj.transitions, self.mdp.gamma)
                traj_loss = torch.zeros((), device=self.policy.device)
                for t, transition in enumerate(traj.transitions):
                    lps = self.policy.action_logprobs(
                        transition.state,
                        budget_remaining=self.mdp.max_steps - transition.step,
                        budget_total=self.mdp.max_steps,
                        no_grad=False,
                    )
                    if transition.action.is_stop():
                        diff_lp = self._differentiable_stop_logprob(lps)
                    elif transition.action.op == "ins":
                        diff_lp = lps.ins_logprobs[transition.action.pos, transition.action.nt]
                    elif transition.action.op == "sub":
                        diff_lp = lps.sub_logprobs[transition.action.pos, transition.action.nt]
                    elif transition.action.op == "del":
                        diff_lp = lps.del_logprobs[transition.action.pos]
                    else:
                        continue
                    if not math.isfinite(float(diff_lp.item())):
                        continue
                    # PG term (differentiable)
                    traj_loss = traj_loss - adv * diff_lp
                    total_logprob += float(diff_lp.item())
                    total_return += float(rets[t]) if t < len(rets) else 0.0
                    total_advantage_sq += adv * adv

                    # KL penalty term: kl_sample is a Python float, so this
                    # contributes to loss VALUE only (no gradient). Preserves
                    # original compute_loss behavior.
                    if self.cfg.kl_coef > 0 and self.ref_policy is not None:
                        ref_lps = self.ref_policy.action_logprobs(
                            transition.state,
                            budget_remaining=self.mdp.max_steps - transition.step,
                            budget_total=self.mdp.max_steps,
                            no_grad=True,
                        )
                        if transition.action.is_stop():
                            ref_lp = float(ref_lps.stop_logprob)
                        elif transition.action.op == "ins":
                            ref_lp = float(ref_lps.ins_logprobs[transition.action.pos, transition.action.nt].item())
                        elif transition.action.op == "sub":
                            ref_lp = float(ref_lps.sub_logprobs[transition.action.pos, transition.action.nt].item())
                        elif transition.action.op == "del":
                            ref_lp = float(ref_lps.del_logprobs[transition.action.pos].item())
                        else:
                            ref_lp = 0.0
                        kl_sample = float(diff_lp.item()) - ref_lp
                        total_kl += kl_sample
                        traj_loss = traj_loss + self.cfg.kl_coef * kl_sample

                    # Entropy bonus term (differentiable)
                    if self.cfg.entropy_coef > 0:
                        entropy = self._distribution_entropy(lps)
                        total_entropy += float(entropy.item())
                        traj_loss = traj_loss - self.cfg.entropy_coef * entropy

                    actual_n_steps += 1

                # Backward on per-trajectory loss, scaled by 1/n_steps_estimated.
                # This frees the computation graph immediately, keeping peak
                # memory bounded to ONE trajectory's forward-pass activations
                # instead of B*N*T simultaneously.
                if traj_loss.requires_grad:
                    (traj_loss / n_steps_estimated).backward()
                total_loss_value += float(traj_loss.item()) / n_steps_estimated

        # Use actual_n_steps for metrics (consistent with original compute_loss
        # when transitions are skipped). Use 1 to avoid div-by-zero.
        metrics_n = actual_n_steps if actual_n_steps > 0 else 1
        metrics = {
            "loss": total_loss_value,
            "mean_logprob": total_logprob / metrics_n,
            "mean_return": float(returns_mat.mean().item()),
            "mean_advantage": float(adv_mat.mean().item()),
            "mean_advantage_sq": float((adv_mat ** 2).mean().item()),
            "mean_kl": total_kl / metrics_n,
            "mean_entropy": total_entropy / metrics_n,
            "n_steps": actual_n_steps,
            "n_groups": B,
            "group_size": N,
            "return_std_mean": float(returns_mat.std(dim=-1).mean().item()),
        }
        return metrics

    # ------------------------------------------------------------------
    # One gradient step on a batch of groups
    # ------------------------------------------------------------------

    def step(self, groups: List[List[Trajectory]]) -> dict:
        """One gradient step on a batch of trajectory groups.

        Uses gradient accumulation (per-trajectory backward) for memory
        efficiency. See :meth:`_compute_loss_grad_accum` for details.
        """
        self.optimizer.zero_grad()
        metrics = self._compute_loss_grad_accum(groups)
        self.optimizer.step()
        return metrics


# ---------------------------------------------------------------------------
# Convergence check (tiny MDP)
# ---------------------------------------------------------------------------


def grpo_convergence_check(
    trainer: GRPOREINFORCE,
    n_iters: int = 200,
    n_groups: int = 4,
    generator: Optional[torch.Generator] = None,
    target_reward: Optional[float] = None,
    tol: float = 1e-2,
) -> dict:
    """Run GRPO on a tiny MDP and check that mean return improves.

    Parameters
    ----------
    trainer : GRPOREINFORCE
        The trainer (policy + mdp + cfg).
    n_iters : int
        Number of gradient steps.
    n_groups : int
        Number of groups per gradient step.
    generator : torch.Generator, optional
        RNG for reproducibility.
    target_reward : float, optional
        If provided, require final mean return >= target_reward - tol.
    tol : float
        Tolerance for the target check.

    Returns
    -------
    result : dict
        ``{"converged": bool, "initial_return": float, "final_return": float,
        "history": List[float]}``.
    """
    history: List[float] = []
    initial_return = float("nan")
    for it in range(n_iters):
        groups = [trainer.collect_group(generator=generator) for _ in range(n_groups)]
        metrics = trainer.step(groups)
        history.append(metrics["mean_return"])
        if it == 0:
            initial_return = metrics["mean_return"]
    final_return = history[-1] if history else float("nan")
    improved = final_return > initial_return
    meets_target = True
    if target_reward is not None:
        meets_target = final_return >= target_reward - tol
    return {
        "converged": bool(improved and meets_target),
        "initial_return": initial_return,
        "final_return": final_return,
        "history": history,
    }
