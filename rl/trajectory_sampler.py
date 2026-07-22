"""Constrained on-policy trajectory sampling for Stage 5 GRPO."""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence

import torch

from mrna_editflow.core.constants import ID_TO_NUC, REGION_CDS, REGION_5UTR, STOP_CODONS, is_valid_cds, translate
from mrna_editflow.core.schema import MRNARecord
from mrna_editflow.eval.metrics import edit_distance, gc_fraction
from mrna_editflow.eval.oracle import LocalTranslationOracle
from mrna_editflow.rl.action_space import Action, STOP_ACTION, apply_action
from mrna_editflow.rl.policy import Policy


@dataclass(frozen=True)
class SamplerConfig:
    task_id: str = "T5"
    max_edits: int = 3
    temperature: float = 1.0
    gc_min: float = 0.25
    gc_max: float = 0.75
    max_length_delta: int = 30
    repeated_motif_length: int = 4


@dataclass
class PolicyStep:
    state: MRNARecord
    action: Action
    old_log_prob: float
    reference_log_prob: float
    action_mask_hash: str
    budget_remaining: int
    termination_reason: str = ""
    reward_components: Mapping[str, float] = field(default_factory=dict)


@dataclass
class SampledTrajectory:
    source: MRNARecord
    steps: list[PolicyStep]
    final_record: MRNARecord
    termination_reason: str
    reward_components: dict[str, float] = field(default_factory=dict)
    reward_audit: dict[str, object] = field(default_factory=dict)


def _hash_mask(actions: Sequence[Action]) -> str:
    payload = "|".join(f"{item.op}:{item.pos}:{item.nt}" for item in actions)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _task_legal(action: Action, record: MRNARecord, task_id: str) -> bool:
    task = task_id.upper()
    if action.is_stop():
        return True
    if task == "T5":
        return action.op == "sub" and 0 <= action.pos < len(record.five_utr)
    if task == "T4":
        # Preserve start and terminal stop codons even if they are synonymous.
        return action.op == "sub" and record.cds_start + 3 <= action.pos < record.cds_end - 3
    if task == "T6":
        return action.op in {"ins", "del"} and not (record.cds_start <= action.pos < record.cds_end)
    raise ValueError("constrained GRPO supports T4/T5/T6 task masks")


def _enumerate_distribution(
    policy: Policy, record: MRNARecord, *, cfg: SamplerConfig, visited: set[str], budget_remaining: int,
    no_grad: bool,
) -> tuple[list[Action], torch.Tensor]:
    """Return a differentiable masked distribution; illegal mass is exactly 0."""
    lps = policy.action_logprobs(record, budget_remaining=budget_remaining, budget_total=cfg.max_edits, no_grad=no_grad)
    choices: list[tuple[Action, torch.Tensor]] = []
    length, vocab = lps.ins_logprobs.shape
    if budget_remaining > 0:
        for pos in range(length):
            for nt in range(vocab):
                for op, tensor in (("ins", lps.ins_logprobs), ("sub", lps.sub_logprobs)):
                    value = tensor[pos, nt]
                    action = Action(op, pos, nt)
                    if torch.isfinite(value) and _task_legal(action, record, cfg.task_id):
                        candidate = apply_action(record, action)
                        if candidate.seq not in visited:
                            choices.append((action, value))
            value = lps.del_logprobs[pos]
            action = Action("del", pos, -1)
            if torch.isfinite(value) and _task_legal(action, record, cfg.task_id):
                candidate = apply_action(record, action)
                if candidate.seq not in visited:
                    choices.append((action, value))
    # STOP is explicit and is the only option once budget is exhausted.
    choices.append((STOP_ACTION, torch.as_tensor(lps.stop_logprob, device=policy.device, dtype=lps.ins_logprobs.dtype)))
    actions, base_log_probs = zip(*choices)
    logits = torch.stack(list(base_log_probs)) / float(cfg.temperature)
    return list(actions), torch.log_softmax(logits, dim=0)


def constrained_distribution(
    policy: Policy, record: MRNARecord, *, cfg: SamplerConfig, visited: Optional[set[str]] = None,
    budget_remaining: Optional[int] = None, no_grad: bool = True,
) -> tuple[list[Action], torch.Tensor, str]:
    """Public policy distribution after all region/protein/task/cycle/budget masks."""
    visited = set(visited or {record.seq})
    actions, log_probs = _enumerate_distribution(
        policy, record, cfg=cfg, visited=visited,
        budget_remaining=cfg.max_edits if budget_remaining is None else int(budget_remaining), no_grad=no_grad,
    )
    return actions, log_probs, _hash_mask(actions)


def _action_logprob(actions: Sequence[Action], log_probs: torch.Tensor, action: Action) -> torch.Tensor:
    for index, candidate in enumerate(actions):
        if candidate == action:
            return log_probs[index]
    return log_probs.new_tensor(float("-inf"))


def sample_trajectory(
    source: MRNARecord, policy: Policy, reference_policy: Policy, *, cfg: SamplerConfig,
    generator: Optional[torch.Generator] = None,
) -> SampledTrajectory:
    """Sample one bounded constrained trajectory and record behavior/reference log-probs."""
    current, visited, steps = source, {source.seq}, []
    for step in range(max(0, int(cfg.max_edits)) + 1):
        remaining = max(0, int(cfg.max_edits) - step)
        actions, old_lps, mask_hash = constrained_distribution(policy, current, cfg=cfg, visited=visited, budget_remaining=remaining, no_grad=True)
        with torch.no_grad():
            ref_actions, ref_lps, ref_hash = constrained_distribution(reference_policy, current, cfg=cfg, visited=visited, budget_remaining=remaining, no_grad=True)
        if actions != ref_actions or mask_hash != ref_hash:
            raise RuntimeError("current and frozen reference policy masks diverged")
        index = int(torch.multinomial(torch.exp(old_lps), 1, generator=generator).item())
        action = actions[index]
        steps.append(PolicyStep(current, action, float(old_lps[index]), float(ref_lps[index]), mask_hash, remaining))
        if action.is_stop():
            return SampledTrajectory(source, steps, current, "stop")
        current = apply_action(current, action)
        visited.add(current.seq)
    # Defensive forced STOP after exactly max_edits action steps.
    steps.append(PolicyStep(current, STOP_ACTION, 0.0, 0.0, _hash_mask([STOP_ACTION]), 0, "edit_budget"))
    return SampledTrajectory(source, steps, current, "edit_budget")


def sample_group(source: MRNARecord, policy: Policy, reference_policy: Policy, *, group_size: int, cfg: SamplerConfig, seed: int) -> list[SampledTrajectory]:
    generator = torch.Generator(device=policy.device).manual_seed(int(seed))
    return [sample_trajectory(source, policy, reference_policy, cfg=cfg, generator=generator) for _ in range(max(1, int(group_size)))]


def _motif_penalty(sequence: str, length: int) -> float:
    if len(sequence) < length:
        return 0.0
    motifs = [sequence[i:i + length] for i in range(len(sequence) - length + 1)]
    return -float(sum(max(0, motifs.count(motif) - 1) for motif in set(motifs)))


def score_trajectory_reward(trajectory: SampledTrajectory, *, oracle: Optional[LocalTranslationOracle] = None, cfg: SamplerConfig) -> SampledTrajectory:
    """Attach raw/risk-adjusted heuristic reward and required safety audit."""
    pred = oracle or LocalTranslationOracle()
    def properties(record: MRNARecord) -> dict[str, float]:
        result = pred.score_utr(record.five_utr, record.cds[:12])
        features = result.get("features", {}) if isinstance(result.get("features", {}), Mapping) else {}
        return {"te": float(result.get("ensemble_te", result.get("te", 0.0))), "access": float(features.get("start_accessibility", 0.0)), "agreement": float(result.get("agreement", 1.0)), "uncertainty": float(result.get("uncertainty", 0.0)), "gc": float(gc_fraction(record.seq))}
    base, final = properties(trajectory.source), properties(trajectory.final_record)
    protein_ok = is_valid_cds(trajectory.final_record.cds) and translate(trajectory.final_record.cds) == translate(trajectory.source.cds)
    edit_cost = float(edit_distance(trajectory.source.seq, trajectory.final_record.seq))
    motif = _motif_penalty(trajectory.final_record.seq, cfg.repeated_motif_length)
    gc_ok = cfg.gc_min <= final["gc"] <= cfg.gc_max
    length_ok = abs(len(trajectory.final_record.seq) - len(trajectory.source.seq)) <= cfg.max_length_delta
    raw = {"te": final["te"] - base["te"], "access": final["access"] - base["access"], "agreement": final["agreement"] - 1.0, "uncertainty": -final["uncertainty"], "edit_cost": -edit_cost, "novelty": -edit_cost / max(1, len(trajectory.source.seq)), "repeated_motif": motif}
    all_negative = all(raw[key] <= 0.0 for key in ("te", "access"))
    hard_ok = bool(protein_ok and gc_ok and length_ok)
    risk_adjusted = sum(raw.values()) - final["uncertainty"]
    if all_negative or not hard_ok:
        risk_adjusted = min(0.0, risk_adjusted)
    trajectory.reward_components = raw
    for step in trajectory.steps:
        step.reward_components = dict(raw)
    trajectory.reward_audit = {"raw_reward": dict(raw), "risk_adjusted_reward": risk_adjusted, "constraint_status": {"protein_identity": protein_ok, "extreme_gc": gc_ok, "extreme_length": length_ok}, "oracle_agreement": final["agreement"], "edit_distance": edit_cost, "action_trajectory": [{"op": step.action.op, "pos": step.action.pos, "nt": step.action.nt} for step in trajectory.steps], "all_negative_stop_preferred": all_negative}
    return trajectory


def trajectory_audit(trajectory: SampledTrajectory) -> dict[str, object]:
    """JSON-ready per-step audit required for a policy-gradient update."""
    return {"termination_reason": trajectory.termination_reason, "reward": trajectory.reward_audit, "steps": [{"state": step.state.to_dict(), "action": {"op": step.action.op, "pos": step.action.pos, "nt": step.action.nt}, "old_log_prob": step.old_log_prob, "reference_log_prob": step.reference_log_prob, "action_mask_hash": step.action_mask_hash, "budget_remaining": step.budget_remaining, "reward_components": dict(step.reward_components), "termination_reason": step.termination_reason} for step in trajectory.steps]}


__all__ = ["SamplerConfig", "PolicyStep", "SampledTrajectory", "constrained_distribution", "sample_trajectory", "sample_group", "score_trajectory_reward", "trajectory_audit"]
