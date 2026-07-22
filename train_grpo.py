"""Stage 5 constrained multi-objective GRPO training (development mode)."""
from __future__ import annotations

import copy
import argparse
import json
import math
import os
import random
from dataclasses import asdict, dataclass, field
from typing import Mapping, Optional, Sequence

import torch

from mrna_editflow.core.schema import MRNARecord
from mrna_editflow.data.download_mrna import load_records_jsonl
from mrna_editflow.eval.artifact_contract import OracleContractError, normalize_run_mode
from mrna_editflow.rl.grpo import GRPOConfig, clipped_policy_loss, group_advantages
from mrna_editflow.rl.kl_regularization import AdaptiveKLController, categorical_kl
from mrna_editflow.rl.policy import Policy, PolicyConfig
from mrna_editflow.rl.trajectory_sampler import SamplerConfig, constrained_distribution, sample_group, score_trajectory_reward, trajectory_audit
from mrna_editflow.sample import load_stage_a_checkpoint
from mrna_editflow.train_backbone import _flow_batch_loss
from mrna_editflow.core import mrna_flow_utils as U


@dataclass(frozen=True)
class TrainGRPOConfig:
    steps: int = 100
    group_size: int = 4
    max_edits: int = 3
    task_id: str = "T5"
    temperature: float = 1.0
    lr: float = 1e-5
    clip_epsilon: float = 0.2
    beta_kl: float = 0.01
    beta_flow: float = 0.0
    beta_entropy: float = 0.0
    flow_replay_ratio: float = 0.0
    flow_loss_weight: float = 1.0
    gradient_clip: float = 1.0
    gradient_accumulation: int = 1
    mixed_precision: bool = False
    max_kl: float = 0.25
    seed: int = 0
    checkpoint_interval: int = 25
    preference: Mapping[str, float] = field(default_factory=lambda: {"te": 0.55, "access": 0.15, "uncertainty": 0.10, "edit_cost": 0.10, "novelty": 0.05, "repeated_motif": 0.05})


def _seed(seed: int) -> None:
    random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def _save(path: str, *, model, backbone, optimizer, controller, config, step, reference_checkpoint, profile_path) -> None:
    torch.save({"stage": "constrained_grpo", "model_state": model.state_dict(), "backbone_state": backbone.state_dict(), "optimizer_state": optimizer.state_dict(), "kl_controller": asdict(controller), "config": asdict(config), "step": step, "reference_checkpoint": reference_checkpoint, "profile_path": profile_path}, path)


def train_grpo(
    *, records: Sequence[MRNARecord], base_checkpoint: str, save_dir: str, profile_path: str,
    config: TrainGRPOConfig = TrainGRPOConfig(), reference_checkpoint: Optional[str] = None,
    resume_checkpoint: Optional[str] = None, device: Optional[str] = None, run_mode: str = "development",
) -> dict[str, object]:
    """Perform true clipped policy updates against a permanently frozen reference."""
    if normalize_run_mode(run_mode) == "paper":
        raise OracleContractError("paper-mode GRPO functional claims require an independent real Oracle; heuristic development oracle is rejected")
    if not records: raise ValueError("GRPO requires non-empty training records")
    if config.gradient_accumulation < 1: raise ValueError("gradient_accumulation must be >= 1")
    _seed(config.seed)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    _cfg, backbone, model = load_stage_a_checkpoint(base_checkpoint, device=str(dev))
    ref_path = reference_checkpoint or base_checkpoint
    _ref_cfg, ref_backbone, reference_model = load_stage_a_checkpoint(ref_path, device=str(dev))
    reference_model.eval(); ref_backbone.eval()
    for parameter in reference_model.parameters(): parameter.requires_grad_(False)
    for parameter in ref_backbone.parameters(): parameter.requires_grad_(False)
    reference_before = [parameter.detach().clone() for parameter in reference_model.parameters()]
    policy_cfg = PolicyConfig(temperature=config.temperature, mixed_precision=config.mixed_precision)
    policy, reference = Policy(model, backbone, policy_cfg, dev), Policy(reference_model, ref_backbone, policy_cfg, dev)
    sampler_cfg = SamplerConfig(task_id=config.task_id, max_edits=config.max_edits, temperature=config.temperature)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    controller = AdaptiveKLController(coefficient=config.beta_kl, max_kl=config.max_kl)
    start = 0
    if resume_checkpoint:
        payload = torch.load(resume_checkpoint, map_location=dev, weights_only=False)
        model.load_state_dict(payload["model_state"]); backbone.load_state_dict(payload["backbone_state"], strict=False)
        optimizer.load_state_dict(payload["optimizer_state"]); controller = AdaptiveKLController(**dict(payload["kl_controller"])); start = int(payload["step"])
    os.makedirs(save_dir, exist_ok=True); os.makedirs(os.path.dirname(os.path.abspath(profile_path)), exist_ok=True)
    checkpoint_path = os.path.join(save_dir, "grpo_last.pt")
    with open(profile_path, "a", encoding="utf-8") as profile:
        for step in range(start + 1, int(config.steps) + 1):
            source = records[(step - 1) % len(records)]
            model.eval()
            trajectories = sample_group(source, policy, reference, group_size=config.group_size, cfg=sampler_cfg, seed=config.seed + step)
            for trajectory in trajectories: score_trajectory_reward(trajectory, cfg=sampler_cfg)
            advantages = group_advantages([trajectory.reward_components for trajectory in trajectories], config.preference, config=GRPOConfig(clip_epsilon=config.clip_epsilon))
            model.train()
            accumulation_index = step - start - 1
            if accumulation_index % int(config.gradient_accumulation) == 0:
                optimizer.zero_grad(set_to_none=True)
            new_lps: list[torch.Tensor] = []; old_lps: list[torch.Tensor] = []; advs: list[torch.Tensor] = []; kls: list[torch.Tensor] = []; entropies: list[torch.Tensor] = []
            for index, trajectory in enumerate(trajectories):
                for action_step in trajectory.steps:
                    actions, current_lps, mask_hash = constrained_distribution(policy, action_step.state, cfg=sampler_cfg, budget_remaining=action_step.budget_remaining, no_grad=False)
                    ref_actions, reference_lps, reference_hash = constrained_distribution(reference, action_step.state, cfg=sampler_cfg, budget_remaining=action_step.budget_remaining, no_grad=True)
                    if actions != ref_actions or mask_hash != reference_hash or mask_hash != action_step.action_mask_hash:
                        raise RuntimeError("action-mask provenance mismatch during GRPO update")
                    selected = next(i for i, action in enumerate(actions) if action == action_step.action)
                    new_lps.append(current_lps[selected]); old_lps.append(torch.tensor(action_step.old_log_prob, device=dev)); advs.append(advantages.aggregated[index])
                    probs = torch.exp(current_lps); kls.append(categorical_kl(current_lps, reference_lps)); entropies.append(-(probs * current_lps).sum())
            new_t, old_t, adv_t = torch.stack(new_lps), torch.stack(old_lps), torch.stack(advs)
            policy_loss, ratios = clipped_policy_loss(new_t, old_t, adv_t, clip_epsilon=config.clip_epsilon)
            observed_kl = torch.stack(kls).mean(); entropy = torch.stack(entropies).mean()
            _, skip = controller.update(float(observed_kl.detach().cpu()))
            flow_loss = new_t.sum() * 0.0
            if config.flow_replay_ratio > 0.0 and random.Random(config.seed + step).random() < config.flow_replay_ratio:
                flow_loss = _flow_batch_loss(model, backbone, [source], _cfg, dev, U.CubicScheduler(), config.seed + step)["loss"]
            loss = policy_loss + controller.coefficient * observed_kl + config.beta_flow * config.flow_loss_weight * flow_loss - config.beta_entropy * entropy
            finite = bool(torch.isfinite(loss).all())
            should_step = ((accumulation_index + 1) % int(config.gradient_accumulation) == 0) or step == int(config.steps)
            if finite and not skip:
                (loss / int(config.gradient_accumulation)).backward()
                if should_step:
                    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip).detach().cpu()); optimizer.step(); updated = True
                else:
                    grad_norm, updated = 0.0, False
            else:
                grad_norm, updated = 0.0, False
                if skip:
                    optimizer.zero_grad(set_to_none=True)
                    for group in optimizer.param_groups: group["lr"] = float(group["lr"]) * 0.5
            stats = {"stage": "constrained_grpo", "step": step, "loss": float(loss.detach().cpu()), "policy_loss": float(policy_loss.detach().cpu()), "observed_kl": float(observed_kl.detach().cpu()), "kl_coefficient": controller.coefficient, "entropy": float(entropy.detach().cpu()), "flow_loss": float(flow_loss.detach().cpu()), "updated": updated, "skip_kl_guard": skip, "grad_norm": grad_norm, "constant_objective_count": advantages.constant_objective_count, "advantages": advantages.audit_dict(), "trajectory_rewards": [item.reward_audit for item in trajectories], "sampled_trajectories": [trajectory_audit(item) for item in trajectories]}
            profile.write(json.dumps(stats, sort_keys=True) + "\n"); profile.flush()
            if step % max(1, config.checkpoint_interval) == 0 or step == config.steps: _save(checkpoint_path, model=model, backbone=backbone, optimizer=optimizer, controller=controller, config=config, step=step, reference_checkpoint=ref_path, profile_path=profile_path)
    reference_unchanged = all(torch.equal(before, after.detach()) for before, after in zip(reference_before, reference_model.parameters()))
    return {"checkpoint_path": checkpoint_path, "profile_path": profile_path, "reference_checkpoint": ref_path, "reference_frozen": True, "reference_unchanged": reference_unchanged, "stage": "constrained_grpo"}


__all__ = ["TrainGRPOConfig", "train_grpo"]


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-jsonl", required=True); parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--save-dir", required=True); parser.add_argument("--profile-path", required=True)
    parser.add_argument("--steps", type=int, default=100); parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--flow-replay-ratio", type=float, default=0.0); parser.add_argument("--flow-loss-weight", type=float, default=1.0)
    parser.add_argument("--beta-kl", type=float, default=0.01); parser.add_argument("--beta-flow", type=float, default=0.0)
    parser.add_argument("--beta-entropy", type=float, default=0.0); parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--mixed-precision", action="store_true"); parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("--device", default=None); parser.add_argument("--seed", type=int, default=0); parser.add_argument("--run-mode", choices=("development", "paper"), default="development")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    result = train_grpo(records=load_records_jsonl(args.records_jsonl), base_checkpoint=args.base_checkpoint, save_dir=args.save_dir, profile_path=args.profile_path, resume_checkpoint=args.resume_checkpoint, device=args.device, run_mode=args.run_mode, config=TrainGRPOConfig(steps=args.steps, group_size=args.group_size, flow_replay_ratio=args.flow_replay_ratio, flow_loss_weight=args.flow_loss_weight, beta_kl=args.beta_kl, beta_flow=args.beta_flow, beta_entropy=args.beta_entropy, gradient_accumulation=args.gradient_accumulation, mixed_precision=args.mixed_precision, seed=args.seed))
    print(json.dumps(result, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
