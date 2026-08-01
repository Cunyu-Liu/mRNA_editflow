"""Auditable group-relative policy optimisation primitives (GRPO)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import torch


@dataclass(frozen=True)
class GRPOConfig:
    # Legacy tiny-MDP API retained for compatibility with P2-05 tests/artifacts.
    group_size: int = 8
    eps: float = 1e-8
    clip_advantage: float = 0.0
    kl_coef: float = 0.0
    entropy_coef: float = 0.0
    clip_epsilon: float = 0.2
    reward_clip: float = 20.0
    advantage_clip: float = 10.0
    minimum_group_variance: float = 1e-8

    def __post_init__(self) -> None:
        if self.group_size < 2 or self.eps <= 0 or self.clip_advantage < 0 or self.kl_coef < 0 or self.entropy_coef < 0:
            raise ValueError("invalid legacy GRPO configuration")


@dataclass(frozen=True)
class GroupAdvantages:
    objectives: tuple[str, ...]
    per_objective: Mapping[str, torch.Tensor]
    aggregated: torch.Tensor
    constant_objective_count: int

    def audit_dict(self) -> dict[str, object]:
        return {
            "objectives": list(self.objectives),
            "per_objective_advantages": {key: value.detach().cpu().tolist() for key, value in self.per_objective.items()},
            "aggregated_advantages": self.aggregated.detach().cpu().tolist(),
            "constant_objective_count": int(self.constant_objective_count),
        }


def group_advantages(
    reward_vectors: Sequence[Mapping[str, float]],
    preference: Mapping[str, float],
    *, config: GRPOConfig = GRPOConfig(), device: torch.device | None = None,
) -> GroupAdvantages:
    """Standardize each objective within one source group, then aggregate.

    A near-constant objective produces an exactly zero advantage rather than a
    noisy division or NaN. This function is used for sampled trajectories, not
    teacher export scalarization.
    """
    if not reward_vectors:
        raise ValueError("GRPO requires at least one trajectory per group")
    objectives = tuple(sorted({str(key) for row in reward_vectors for key in row}))
    if not objectives:
        raise ValueError("trajectory reward vectors are empty")
    pref = {key: max(0.0, float(preference.get(key, 0.0))) for key in objectives}
    total = sum(pref.values())
    if total <= 0.0:
        raise ValueError("preference must give positive weight to at least one present objective")
    pref = {key: value / total for key, value in pref.items()}
    per: dict[str, torch.Tensor] = {}
    constant = 0
    for key in objectives:
        values = torch.tensor([float(row.get(key, 0.0)) for row in reward_vectors], dtype=torch.float32, device=device)
        values = values.clamp(-float(config.reward_clip), float(config.reward_clip))
        variance = values.var(unbiased=False)
        if not torch.isfinite(variance) or float(variance.detach().cpu()) <= float(config.minimum_group_variance):
            per[key] = torch.zeros_like(values)
            constant += 1
        else:
            per[key] = ((values - values.mean()) / variance.sqrt()).clamp(
                -float(config.advantage_clip), float(config.advantage_clip)
            )
    aggregate = sum(pref[key] * per[key] for key in objectives)
    return GroupAdvantages(objectives, per, aggregate, constant)


def clipped_policy_loss(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    *, clip_epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PPO/GRPO clipped surrogate loss, with trajectory advantage per action."""
    if new_log_probs.shape != old_log_probs.shape or new_log_probs.shape != advantages.shape:
        raise ValueError("new_log_probs, old_log_probs and advantages must share shape")
    ratio = torch.exp(new_log_probs - old_log_probs)
    clipped = ratio.clamp(1.0 - float(clip_epsilon), 1.0 + float(clip_epsilon))
    surrogate = torch.minimum(ratio * advantages, clipped * advantages)
    return -surrogate.mean(), ratio


def group_normalized_advantages(returns: torch.Tensor, cfg: Optional[GRPOConfig] = None) -> torch.Tensor:
    """Backward-compatible scalar group normalization used by old tiny-MDP tests."""
    if returns.ndim not in {1, 2}:
        raise ValueError("returns must be one or two dimensional")
    cfg = cfg or GRPOConfig(group_size=int(returns.shape[-1]))
    if returns.shape[-1] != cfg.group_size:
        raise ValueError("returns group size differs from configuration")
    mean = returns.mean(dim=-1, keepdim=True)
    std = returns.std(dim=-1, keepdim=True, unbiased=False)
    output = (returns - mean) / (std + cfg.eps)
    return output.clamp(-cfg.clip_advantage, cfg.clip_advantage) if cfg.clip_advantage > 0 else output


class GRPOREINFORCE:
    """Compatibility tiny-MDP trainer; production Stage 5 uses clipped loss above."""
    def __init__(self, policy, mdp, cfg: GRPOConfig, lr: float = 0.01, ref_policy=None) -> None:
        from mrna_editflow.rl.tiny_mdp import REINFORCE
        self.policy, self.mdp, self.cfg, self.lr, self.ref_policy = policy, mdp, cfg, lr, ref_policy
        self.use_baseline, self.baseline, self.baseline_decay = False, 0.0, 0.0
        self.optimizer = torch.optim.SGD(self.policy.model.parameters(), lr=lr)
        self._base = REINFORCE(policy, mdp, lr=lr, use_baseline=False)

    def collect_trajectory(self, generator=None): return self._base.collect_trajectory(generator=generator)
    def collect_group(self, generator=None): return [self.collect_trajectory(generator=generator) for _ in range(self.cfg.group_size)]

    @staticmethod
    def _distribution_entropy(lps) -> torch.Tensor:
        values = [item.reshape(-1) for item in (getattr(lps, "ins_logprobs", None), getattr(lps, "sub_logprobs", None), getattr(lps, "del_logprobs", None)) if item is not None]
        if getattr(lps, "stop_logprob", None) is not None: values.append(torch.as_tensor([lps.stop_logprob], device=values[0].device if values else "cpu"))
        if not values: return torch.tensor(0.0)
        lp = torch.cat(values); lp = lp[torch.isfinite(lp)]; p = torch.exp(lp); return -(p / p.sum().clamp_min(1e-12) * lp).sum()

    def compute_loss(self, groups):
        from mrna_editflow.rl.tiny_mdp import compute_returns
        for group in groups:
            if len(group) != self.cfg.group_size: raise ValueError("wrong group size")
        if not groups:
            return torch.zeros((), device=self.policy.device, requires_grad=True), {
                "loss": 0.0, "mean_return": 0.0, "mean_advantage": 0.0,
                "mean_advantage_sq": 0.0, "mean_logprob": 0.0,
                "mean_kl": 0.0, "mean_entropy": 0.0, "n_steps": 0,
                "n_groups": 0, "group_size": self.cfg.group_size,
                "return_std_mean": 0.0,
            }
        returns = torch.tensor([[compute_returns(traj.transitions, self.mdp.gamma)[0] if traj.transitions else 0.0 for traj in group] for group in groups], device=self.policy.device)
        adv = group_normalized_advantages(returns, self.cfg); terms=[]; ent=[]; logprob_terms=[]
        for b, group in enumerate(groups):
            for i, traj in enumerate(group):
                for transition in traj.transitions:
                    lps = self.policy.action_logprobs(transition.state, budget_remaining=self.mdp.max_steps-transition.step, budget_total=self.mdp.max_steps, no_grad=False)
                    lp = lps.logprob(transition.action)
                    if math.isfinite(lp):
                        # recover differentiable selected value
                        if transition.action.is_stop(): value = torch.as_tensor(lps.stop_logprob, device=self.policy.device)
                        elif transition.action.op == "ins": value = lps.ins_logprobs[transition.action.pos, transition.action.nt]
                        elif transition.action.op == "sub": value = lps.sub_logprobs[transition.action.pos, transition.action.nt]
                        else: value = lps.del_logprobs[transition.action.pos]
                        terms.append(-adv[b, i].detach() * value); logprob_terms.append(value); ent.append(self._distribution_entropy(lps))
        loss = torch.stack(terms).mean() if terms else torch.zeros((), device=self.policy.device, requires_grad=True)
        if ent: loss = loss - self.cfg.entropy_coef * torch.stack(ent).mean()
        return loss, {"loss": float(loss.detach()), "mean_return": float(returns.mean()), "mean_advantage": float(adv.mean()), "mean_advantage_sq": float((adv**2).mean()), "mean_logprob": float(torch.stack(logprob_terms).detach().mean()) if logprob_terms else 0.0, "mean_kl": 0.0, "mean_entropy": float(torch.stack(ent).mean()) if ent else 0.0, "n_steps": len(terms), "n_groups": len(groups), "group_size": self.cfg.group_size, "return_std_mean": float(returns.std(dim=-1).mean())}

    def step(self, groups):
        self.optimizer.zero_grad(); loss, metrics = self.compute_loss(groups); loss.backward(); self.optimizer.step(); return metrics

    def _compute_loss_grad_accum(self, groups):
        """Compatibility gradient-accumulation entrypoint for tiny-MDP runs."""
        loss, metrics = self.compute_loss(groups)
        loss.backward()
        return metrics


def grpo_convergence_check(
    trainer, n_iters: int, n_groups: int, generator=None, *, evaluation_trajectories: int = 256,
) -> dict[str, object]:
    """Train a tiny MDP and compare fixed-seed Monte Carlo evaluations.

    Training-batch means are intentionally retained as ``history`` for
    diagnostics, but are not a valid convergence metric: each batch is
    on-policy and stochastic.  The reported before/after values instead use
    the same independent generator seed and a larger sample, so a pass is
    evidence of an actual policy-return change rather than a lucky last group.
    """
    from mrna_editflow.rl.tiny_mdp import compute_returns

    if int(evaluation_trajectories) < 1:
        raise ValueError("evaluation_trajectories must be positive")

    def evaluate() -> float:
        eval_generator = torch.Generator(device=trainer.policy.device)
        eval_generator.manual_seed(918273)
        values = []
        for _ in range(int(evaluation_trajectories)):
            trajectory = trainer.collect_trajectory(generator=eval_generator)
            values.append(compute_returns(trajectory.transitions, trainer.mdp.gamma)[0] if trajectory.transitions else 0.0)
        return float(torch.tensor(values).mean())

    initial_return = evaluate()
    history=[]
    for _ in range(n_iters):
        metrics = trainer.step([trainer.collect_group(generator=generator) for _ in range(n_groups)]); history.append(metrics["mean_return"])
    final_return = evaluate()
    return {"history": history, "initial_return": initial_return, "final_return": final_return, "converged": final_return >= initial_return}


__all__ = ["GRPOConfig", "GroupAdvantages", "group_advantages", "clipped_policy_loss", "group_normalized_advantages", "GRPOREINFORCE", "grpo_convergence_check"]
