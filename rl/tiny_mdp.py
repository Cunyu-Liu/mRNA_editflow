"""P1-08: Tiny enumerable MDP for RL correctness testing.

Provides:
  - ``TinyTrainableModel``: small differentiable nn.Module matching the
    MRNAEditFormer forward signature (returns rates/ins_probs/sub_probs).
  - ``TinyMDP``: enumerable mRNA design MDP with a reward function.
  - ``REINFORCE``: policy-gradient algorithm with optional baseline.
  - ``compute_returns``: discounted return computation.
  - ``numerical_gradient_check``: finite-difference gradient verification.

The tiny MDP uses short sequences (e.g. 5-10 nt) so the state/action space
is enumerable for correctness testing.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from mrna_editflow.core.constants import NUC_TO_ID, V
from mrna_editflow.core.schema import MRNARecord
from mrna_editflow.rl.action_space import STOP_ACTION, Action, apply_action
from mrna_editflow.rl.policy import Policy, PolicyConfig


# ---------------------------------------------------------------------------
# Tiny trainable model
# ---------------------------------------------------------------------------


class TinyTrainableModel(nn.Module):
    """Small differentiable model matching MRNAEditFormer forward signature.

    Per-position MLP that outputs rates[B,L,3] and token distributions.
    Used for REINFORCE policy-gradient testing on tiny MDPs.

    The model is intentionally simple (2-layer MLP per position, no
    transformer) so that gradient correctness can be verified analytically.
    """

    def __init__(self, vocab_dim: int = V, hidden: int = 16, rates_init: float = 0.5):
        super().__init__()
        self.vocab_dim = vocab_dim
        # Per-position shared MLP.
        self.rates_head = nn.Sequential(
            nn.Linear(vocab_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 3),
            nn.Softplus(),
        )
        self.ins_head = nn.Sequential(
            nn.Linear(vocab_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, vocab_dim),
        )
        self.sub_head = nn.Sequential(
            nn.Linear(vocab_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, vocab_dim),
        )
        # Initialize bias so initial rates are ~rates_init.
        with torch.no_grad():
            self.rates_head[-2].bias.fill_(math.log(max(rates_init, 1e-3)))

    def forward(
        self,
        token_ids: torch.Tensor,  # [B, L]
        region_ids: torch.Tensor,  # [B, L]
        phase_ids: torch.Tensor,  # [B, L]
        time_step: torch.Tensor,  # [B, 1]
        padding_mask: torch.Tensor,  # [B, L]
        backbone,  # unused
    ) -> dict:
        x = F.one_hot(token_ids.clamp(0, self.vocab_dim - 1), self.vocab_dim).float()  # [B, L, V]
        rates = self.rates_head(x)  # [B, L, 3]
        ins_logits = self.ins_head(x)  # [B, L, V]
        sub_logits = self.sub_head(x)  # [B, L, V]
        ins_probs = F.softmax(ins_logits, dim=-1)  # [B, L, V]
        sub_probs = F.softmax(sub_logits, dim=-1)  # [B, L, V]
        # Zero out PAD positions.
        valid = (~padding_mask).unsqueeze(-1).to(rates.dtype)
        rates = rates * valid
        ins_probs = ins_probs * valid
        sub_probs = sub_probs * valid
        return {
            "rates": rates,
            "ins_probs": ins_probs,
            "sub_probs": sub_probs,
            "aux": None,
        }


class _NullBackbone:
    """Backbone stub that does nothing (TinyTrainableModel ignores it)."""

    out_dim = 0

    def freeze(self) -> None:
        pass

    def to(self, device):
        return self


# ---------------------------------------------------------------------------
# Tiny MDP
# ---------------------------------------------------------------------------


@dataclass
class TinyMDP:
    """A tiny enumerable mRNA design MDP.

    The agent starts from ``initial_record`` and edits it toward ``target_seq``
    (the full ``record.seq`` string). Reward is given at STOP (or when
    ``max_steps`` is reached) as ``-hamming(state, target) / len(target) + bonus``
    where ``bonus`` is awarded only if the final state matches the target.

    Parameters
    ----------
    target_seq : str
        Target full sequence (5'UTR + CDS + 3'UTR).
    initial_record : MRNARecord
        Starting state.
    max_steps : int
        Maximum trajectory length (after which STOP is forced).
    stop_bonus : float
        Bonus reward for stopping (encourages termination).
    target_bonus : float
        Bonus reward for matching the target at termination.
    gamma : float
        Discount factor.
    """

    target_seq: str
    initial_record: MRNARecord
    max_steps: int = 5
    stop_bonus: float = 0.1
    target_bonus: float = 1.0
    gamma: float = 0.99

    def initial_state(self) -> MRNARecord:
        return self.initial_record

    def reward(
        self,
        state: MRNARecord,
        action: Action,
        next_state: MRNARecord,
        step: int,
    ) -> float:
        """Reward function.

        - Per-step reward: 0 (sparse reward).
        - Terminal reward (STOP or max_steps): -hamming/L + bonuses.
        """
        is_terminal = action.is_stop() or (step + 1) >= self.max_steps
        if not is_terminal:
            return 0.0
        # Hamming distance to target.
        L = len(self.target_seq)
        seq = next_state.seq
        if len(seq) != L:
            # Length mismatch: heavy penalty.
            return -1.0 + (self.target_bonus if seq == self.target_seq else 0.0)
        hamming = sum(1 for a, b in zip(seq, self.target_seq) if a != b)
        r = -hamming / L
        if action.is_stop():
            r += self.stop_bonus
        if seq == self.target_seq:
            r += self.target_bonus
        return r

    def is_terminal(self, action: Action, step: int) -> bool:
        return action.is_stop() or (step + 1) >= self.max_steps


# ---------------------------------------------------------------------------
# REINFORCE
# ---------------------------------------------------------------------------


@dataclass
class Transition:
    """A single (s, a, r, s') transition."""
    state: MRNARecord
    action: Action
    reward: float
    next_state: MRNARecord
    step: int


@dataclass
class Trajectory:
    """A full trajectory of transitions."""
    transitions: List[Transition] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.transitions)

    def total_reward(self) -> float:
        return sum(t.reward for t in self.transitions)


def compute_returns(transitions: List[Transition], gamma: float) -> List[float]:
    """Compute discounted returns G_t = sum_{k>=t} gamma^(k-t) * r_k.

    Complexity: O(T) where T = len(transitions).
    """
    T = len(transitions)
    returns = [0.0] * T
    running = 0.0
    for t in reversed(range(T)):
        running = transitions[t].reward + gamma * running
        returns[t] = running
    return returns


class REINFORCE:
    """REINFORCE policy-gradient algorithm with optional baseline.

    Parameters
    ----------
    policy : Policy
        The policy to optimize (wraps a trainable model).
    mdp : TinyMDP
        The MDP to train on.
    lr : float
        Learning rate.
    use_baseline : bool
        If True, subtract a moving-average baseline from returns.
    baseline_decay : float
        EMA decay for the baseline.
    """

    def __init__(
        self,
        policy: Policy,
        mdp: TinyMDP,
        lr: float = 0.01,
        use_baseline: bool = True,
        baseline_decay: float = 0.9,
    ) -> None:
        self.policy = policy
        self.mdp = mdp
        self.lr = lr
        self.use_baseline = use_baseline
        self.baseline_decay = baseline_decay
        self.baseline = 0.0
        self.optimizer = torch.optim.SGD(
            self.policy.model.parameters(),
            lr=lr,
        )

    def collect_trajectory(self, generator: Optional[torch.Generator] = None) -> Trajectory:
        """Collect one trajectory by rolling out the policy.

        Sampling is non-differentiable (uses ``torch.no_grad`` internally).
        """
        traj = Trajectory()
        state = self.mdp.initial_state()
        for step in range(self.mdp.max_steps):
            action, _ = self.policy.sample(
                state,
                budget_remaining=self.mdp.max_steps - step,
                budget_total=self.mdp.max_steps,
                generator=generator,
            )
            next_state = apply_action(state, action)
            reward = self.mdp.reward(state, action, next_state, step)
            traj.transitions.append(
                Transition(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    step=step,
                )
            )
            if self.mdp.is_terminal(action, step):
                break
            state = next_state
        return traj

    def compute_loss(
        self,
        trajectories: List[Trajectory],
    ) -> Tuple[torch.Tensor, dict]:
        """Compute the REINFORCE loss over a batch of trajectories.

        Loss = -mean_t( (G_t - b) * log p(a_t | s_t) )

        Returns
        -------
        loss : torch.Tensor (scalar)
            The REINFORCE loss (ready for ``.backward()``).
        metrics : dict
            Diagnostic metrics.
        """
        total_loss = torch.zeros((), device=self.policy.device)
        total_logprob = 0.0
        total_return = 0.0
        total_advantage_sq = 0.0
        n_steps = 0

        for traj in trajectories:
            returns = compute_returns(traj.transitions, self.mdp.gamma)
            for t, transition in enumerate(traj.transitions):
                # Recompute log-prob with grad enabled.
                lps = self.policy.action_logprobs(
                    transition.state,
                    budget_remaining=self.mdp.max_steps - transition.step,
                    budget_total=self.mdp.max_steps,
                    no_grad=False,
                )
                lp = lps.logprob(transition.action)
                if not math.isfinite(lp):
                    # Skip illegal actions (shouldn't happen if sampling is correct).
                    continue
                lp_tensor = torch.tensor(lp, device=self.policy.device, requires_grad=False)
                # We need a differentiable log-prob. The lps tensors have grad history.
                # Extract the specific log-prob as a differentiable scalar.
                if transition.action.is_stop():
                    diff_lp = lps.stop_logprob  # this is a Python float, not differentiable!
                    # Need to recompute stop_logprob as a differentiable tensor.
                    # ... (see below)
                    diff_lp_t = self._differentiable_stop_logprob(lps)
                elif transition.action.op == "ins":
                    diff_lp_t = lps.ins_logprobs[transition.action.pos, transition.action.nt]
                elif transition.action.op == "sub":
                    diff_lp_t = lps.sub_logprobs[transition.action.pos, transition.action.nt]
                elif transition.action.op == "del":
                    diff_lp_t = lps.del_logprobs[transition.action.pos]
                else:
                    continue

                G = returns[t]
                if self.use_baseline:
                    advantage = G - self.baseline
                else:
                    advantage = G
                total_loss = total_loss - advantage * diff_lp_t
                total_logprob += float(diff_lp_t.item()) if hasattr(diff_lp_t, "item") else float(diff_lp_t)
                total_return += G
                total_advantage_sq += advantage * advantage
                n_steps += 1

        if n_steps > 0:
            total_loss = total_loss / n_steps
            total_logprob /= n_steps
            total_return /= n_steps
            total_advantage_sq /= n_steps

        # Update baseline (EMA of mean return).
        if self.use_baseline and n_steps > 0:
            self.baseline = (
                self.baseline_decay * self.baseline
                + (1 - self.baseline_decay) * total_return
            )

        metrics = {
            "loss": float(total_loss.item()),
            "mean_logprob": total_logprob,
            "mean_return": total_return,
            "mean_advantage_sq": total_advantage_sq,
            "baseline": self.baseline,
            "n_steps": n_steps,
        }
        return total_loss, metrics

    def _differentiable_stop_logprob(self, lps) -> torch.Tensor:
        """Recompute stop_logprob as a differentiable tensor.

        The ``lps.stop_logprob`` is a Python float (not differentiable) because
        it was computed via ``math.log``. We need to recompute it from the
        differentiable ``ins_logprobs`` / ``sub_logprobs`` / ``del_logprobs``
        tensors.

        p(STOP) = 1 - sum of exp(ins_lp) - sum of exp(sub_lp) - sum of exp(del_lp)
        log p(STOP) = log(p(STOP))
        """
        ins_mass = torch.exp(lps.ins_logprobs).nansum()
        sub_mass = torch.exp(lps.sub_logprobs).nansum()
        del_mass = torch.exp(lps.del_logprobs).nansum()
        stop_prob = (1.0 - ins_mass - sub_mass - del_mass).clamp_min(1e-30)
        return torch.log(stop_prob)

    def update(self, trajectories: List[Trajectory]) -> dict:
        """One REINFORCE update step.

        Returns metrics dict.
        """
        self.optimizer.zero_grad()
        loss, metrics = self.compute_loss(trajectories)
        loss.backward()
        # Gradient clipping for stability.
        torch.nn.utils.clip_grad_norm_(self.policy.model.parameters(), max_norm=10.0)
        self.optimizer.step()
        metrics["loss"] = float(loss.item())
        return metrics

    def train(
        self,
        n_episodes: int,
        batch_size: int = 8,
        generator: Optional[torch.Generator] = None,
        log_every: int = 0,
    ) -> List[dict]:
        """Train for ``n_episodes`` episodes.

        Returns a list of per-batch metrics.
        """
        all_metrics: List[dict] = []
        n_batches = max(1, n_episodes // batch_size)
        for batch_idx in range(n_batches):
            trajectories = [
                self.collect_trajectory(generator=generator) for _ in range(batch_size)
            ]
            metrics = self.update(trajectories)
            metrics["batch_idx"] = batch_idx
            all_metrics.append(metrics)
            if log_every > 0 and (batch_idx + 1) % log_every == 0:
                print(
                    f"  batch {batch_idx + 1}/{n_batches}: "
                    f"loss={metrics['loss']:.4f}, "
                    f"mean_return={metrics['mean_return']:.4f}, "
                    f"baseline={metrics['baseline']:.4f}"
                )
        return all_metrics


# ---------------------------------------------------------------------------
# Gradient correctness check
# ---------------------------------------------------------------------------


def numerical_gradient_check(
    policy: Policy,
    mdp: TinyMDP,
    trajectories: List[Trajectory],
    param_idx: int = 0,
    eps: float = 1e-4,
    atol: float = 1e-2,
) -> Tuple[float, float, bool]:
    """Verify analytic gradient matches numerical gradient for one parameter.

    Picks the ``param_idx``-th parameter of ``policy.model``, computes the
    analytic gradient via autograd, then computes the numerical gradient via
    finite differences (central difference).

    Returns
    -------
    analytic_grad : float
    numerical_grad : float
    matches : bool
        True if ``|analytic - numerical| < atol``.
    """
    params = list(policy.model.parameters())
    if param_idx >= len(params):
        raise IndexError(f"param_idx {param_idx} out of range (model has {len(params)} params)")
    param = params[param_idx]

    # Flatten for easy indexing.
    flat_param = param.view(-1)
    if flat_param.numel() == 0:
        return 0.0, 0.0, True
    # Check gradient at index 0 of the flat parameter.
    idx = 0

    # --- Analytic gradient ---
    policy.model.zero_grad()
    loss, _ = REINFORCE(policy, mdp, use_baseline=False).compute_loss(trajectories)
    loss.backward()
    # Access grad on the leaf param (flat_param is a view, its .grad is None).
    if param.grad is None:
        return 0.0, 0.0, True
    analytic_grad = float(param.grad.view(-1)[idx].item())

    # --- Numerical gradient (central difference) ---
    orig = float(flat_param[idx].item())
    with torch.no_grad():
        flat_param[idx] = orig + eps
    loss_plus, _ = REINFORCE(policy, mdp, use_baseline=False).compute_loss(trajectories)
    with torch.no_grad():
        flat_param[idx] = orig - eps
    loss_minus, _ = REINFORCE(policy, mdp, use_baseline=False).compute_loss(trajectories)
    with torch.no_grad():
        flat_param[idx] = orig  # restore
    numerical_grad = float((loss_plus.item() - loss_minus.item()) / (2 * eps))

    matches = abs(analytic_grad - numerical_grad) < atol
    return analytic_grad, numerical_grad, matches


# ---------------------------------------------------------------------------
# Profile helper
# ---------------------------------------------------------------------------


def profile_reinforce(
    policy: Policy,
    mdp: TinyMDP,
    n_episodes: int = 16,
    batch_size: int = 4,
) -> dict:
    """Run a small REINFORCE training loop and collect profile metrics.

    Returns a dict with timing and convergence diagnostics.
    """
    import time

    agent = REINFORCE(policy, mdp, lr=0.01, use_baseline=True)
    t0 = time.perf_counter()
    metrics_list = agent.train(n_episodes, batch_size=batch_size)
    elapsed = time.perf_counter() - t0

    returns = [m["mean_return"] for m in metrics_list]
    return {
        "n_episodes": n_episodes,
        "batch_size": batch_size,
        "n_batches": len(metrics_list),
        "elapsed_seconds": elapsed,
        "seconds_per_batch": elapsed / max(1, len(metrics_list)),
        "initial_return": returns[0] if returns else 0.0,
        "final_return": returns[-1] if returns else 0.0,
        "return_improvement": (returns[-1] - returns[0]) if len(returns) >= 2 else 0.0,
        "max_return": max(returns) if returns else 0.0,
        "min_return": min(returns) if returns else 0.0,
        "final_baseline": agent.baseline,
        "final_loss": metrics_list[-1]["loss"] if metrics_list else 0.0,
    }
