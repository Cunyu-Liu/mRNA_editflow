"""mRNA-EditFlow reinforcement learning subpackage.

Modules
-------
action_space
    Action types, legal action masks, and action application.
policy
    Stochastic policy wrapping :class:`MRNAEditFormer` with a normalized
    CTMC action distribution, STOP action, and trajectory log-probabilities.
tiny_mdp
    Tiny enumerable MDP for RL correctness testing (P1-08).
cto
    Innovation 1: Constrained Trajectory Optimization (CTO) — hard
    constraint via rejection sampling + feasibility-masked REINFORCE (P1-12).
synergy
    Innovation 2: Counterfactual Cross-Region Synergy RL — synergy reward
    shaping via 4 counterfactual rollouts + lambda schedule (P1-12).
real_mdp
    Real mRNA design MDP backed by Oracle #3 (P2-05). Implements the
    TinyMDP interface but operates on real MRNARecord sequences with
    a predictive oracle providing the reward signal.
"""
from mrna_editflow.rl.action_space import (
    STOP_ACTION,
    Action,
    ActionLogProbs,
    ActionMask,
    apply_action,
    build_legal_action_mask,
)
from mrna_editflow.rl.policy import Policy, PolicyConfig
from mrna_editflow.rl.tiny_mdp import (
    REINFORCE,
    TinyMDP,
    TinyTrainableModel,
    Trajectory,
    Transition,
    compute_returns,
)
from mrna_editflow.rl.cto import (
    CTOREINFORCE,
    ConstraintConfig,
    SoftPenaltyREINFORCE,
    cto_convergence_check,
    is_feasible,
    trajectory_cost,
)
from mrna_editflow.rl.grpo import (
    GRPOConfig,
    GRPOREINFORCE,
    group_normalized_advantages,
    grpo_convergence_check,
)
from mrna_editflow.rl.synergy import (
    SynergyConfig,
    SynergyREINFORCE,
    LambdaSchedule,
    make_tiny_synergy_mdp,
    synergy_convergence_check,
)
from mrna_editflow.rl.real_mdp import OracleLike, RealMRNAMDP

__all__ = [
    # action_space
    "STOP_ACTION",
    "Action",
    "ActionLogProbs",
    "ActionMask",
    "apply_action",
    "build_legal_action_mask",
    # policy
    "Policy",
    "PolicyConfig",
    # tiny_mdp
    "REINFORCE",
    "TinyMDP",
    "TinyTrainableModel",
    "Trajectory",
    "Transition",
    "compute_returns",
    # cto (Innovation 1)
    "CTOREINFORCE",
    "ConstraintConfig",
    "SoftPenaltyREINFORCE",
    "cto_convergence_check",
    "is_feasible",
    "trajectory_cost",
    # synergy (Innovation 2)
    "SynergyConfig",
    "SynergyREINFORCE",
    "LambdaSchedule",
    "make_tiny_synergy_mdp",
    "synergy_convergence_check",
    # grpo (P2-05)
    "GRPOConfig",
    "GRPOREINFORCE",
    "group_normalized_advantages",
    "grpo_convergence_check",
    # real_mdp (P2-05)
    "OracleLike",
    "RealMRNAMDP",
]
