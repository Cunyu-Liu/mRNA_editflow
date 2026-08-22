"""Soft-value guidance and matched-compute SMC primitives for XEditFlow V3."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from core.route2_xeditcritic_v3 import EndpointConditionerV1
from core.route2_xeditsetflow_v3 import HybridSetFlowBlockV3


class XEditFlowGuidanceV3Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditFlowGuidanceV3Error(message)


def uncertainty_penalized_reward_v3(
    calibrated_seed_predictions: torch.Tensor, *, kappa: float
) -> torch.Tensor:
    _require(
        calibrated_seed_predictions.ndim == 2
        and calibrated_seed_predictions.shape[1] == 3,
        "Critic V3 reward requires exactly three seed predictions",
    )
    _require(kappa in {0.0, 0.5, 1.0}, "uncertainty penalty is outside the frozen grid")
    _require(
        bool(torch.isfinite(calibrated_seed_predictions).all().item()),
        "Critic V3 reward input is nonfinite",
    )
    return calibrated_seed_predictions.mean(dim=1) - float(kappa) * calibrated_seed_predictions.std(
        dim=1, unbiased=False
    )


def soft_value_target_v3(rollout_rewards: torch.Tensor, *, temperature: float) -> torch.Tensor:
    _require(
        rollout_rewards.ndim == 2 and rollout_rewards.shape[1] == 8,
        "soft-value target requires exactly K=8 base-flow rollouts",
    )
    _require(temperature in {0.5, 1.0}, "soft-value temperature is outside the frozen grid")
    _require(bool(torch.isfinite(rollout_rewards).all().item()), "soft-value rewards are nonfinite")
    return float(temperature) * (
        torch.logsumexp(rollout_rewards / float(temperature), dim=1)
        - math.log(rollout_rewards.shape[1])
    )


def beta_schedule_v3(progress: torch.Tensor | float, *, beta_max: float) -> torch.Tensor:
    _require(beta_max in {0.5, 1.0, 2.0}, "maximum guidance strength is outside the frozen grid")
    values = torch.as_tensor(progress)
    _require(
        bool(torch.isfinite(values).all().item())
        and bool(((values >= 0.0) & (values <= 1.0)).all().item()),
        "guidance progress is outside [0,1]",
    )
    return float(beta_max) * (0.25 + 0.75 * values)


def potential_guided_rates_v3(
    base_rates: torch.Tensor,
    legal_mask: torch.Tensor,
    current_potential: torch.Tensor,
    child_potential: torch.Tensor,
    *,
    progress: torch.Tensor,
    beta_max: float,
) -> torch.Tensor:
    _require(base_rates.shape == legal_mask.shape == child_potential.shape, "guided action tensors differ")
    _require(
        current_potential.shape == progress.shape == (base_rates.shape[0],),
        "guided state tensors differ",
    )
    _require(
        bool(torch.isfinite(base_rates[legal_mask]).all().item())
        and bool((base_rates[legal_mask] > 0.0).all().item()),
        "guided legal base rate is invalid",
    )
    _require(
        bool((base_rates[~legal_mask] == 0.0).all().item()),
        "illegal base action has nonzero rate",
    )
    _require(
        bool(torch.isfinite(current_potential).all().item())
        and bool(torch.isfinite(child_potential[legal_mask]).all().item()),
        "guided scalar potential is nonfinite",
    )
    beta = beta_schedule_v3(progress, beta_max=beta_max).to(base_rates).unsqueeze(1)
    tilted = base_rates * torch.exp(
        beta * (child_potential - current_potential.unsqueeze(1))
    )
    _require(bool(torch.isfinite(tilted[legal_mask]).all().item()), "guided legal rate is nonfinite")
    return torch.where(legal_mask, tilted, torch.zeros_like(tilted))


class XEditValueV3(nn.Module):
    """Six-block, width-384 scalar state potential; no free action-ratio head."""

    def __init__(
        self,
        *,
        assay_count: int,
        context_count: int,
        quantity_count: int,
        measurement_count: int,
        numerator_count: int,
        denominator_count: int,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        width = 384
        condition_width = 256
        nucleotide_width = 48
        self.endpoint_conditioner = EndpointConditionerV1(
            quantity_count=quantity_count,
            measurement_count=measurement_count,
            numerator_count=numerator_count,
            denominator_count=denominator_count,
            assay_count=assay_count,
            context_count=context_count,
            region_count=2,
            output_width=condition_width,
            category_width=32,
        )
        self.source_nucleotide = nn.Embedding(5, nucleotide_width, padding_idx=4)
        self.current_nucleotide = nn.Embedding(5, nucleotide_width, padding_idx=4)
        self.source_pretrained_projection = nn.Linear(768, width)
        self.state_projection = nn.Linear(nucleotide_width * 2 + 4, width)
        self.input_norm = nn.LayerNorm(width)
        self.blocks = nn.ModuleList(
            HybridSetFlowBlockV3(
                width=width,
                heads=8,
                ffn_width=1536,
                window=64,
                dilation=2 ** (index % 4),
                shifted=bool(index % 2),
                dropout=dropout,
                condition_width=condition_width,
            )
            for index in range(6)
        )
        self.pool_attention = nn.Linear(width, 1)
        self.scalar_head = nn.Sequential(
            nn.Linear(width + condition_width + 2, width),
            nn.GELU(),
            nn.Linear(width, 1),
        )

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def forward(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        source = batch["source_tokens"]
        current = batch["current_tokens"]
        padding = batch["padding_mask"]
        pretrained = batch["source_pretrained_tokens"]
        _require(source.shape == current.shape == padding.shape, "value sequence tensors differ")
        _require(pretrained.shape[:2] == source.shape, "value pretrained tokens do not align")
        valid = ~padding
        edited = (source != current) & valid
        length = valid.sum(dim=1, keepdim=True)
        position_index = torch.arange(source.shape[1], device=source.device).unsqueeze(0)
        position = (
            position_index / (length - 1).clamp_min(1)
        ).to(pretrained.dtype) * valid
        edited_count = edited.sum(dim=1).to(pretrained.dtype)
        remaining = batch["remaining_budget"].to(pretrained.dtype)
        assigned = edited_count + remaining
        progress = edited_count / assigned.clamp_min(1)
        state = torch.cat(
            (
                self.source_nucleotide(source),
                self.current_nucleotide(current),
                edited.to(pretrained.dtype).unsqueeze(-1),
                position.unsqueeze(-1),
                remaining.log1p().view(-1, 1, 1).expand(-1, source.shape[1], -1),
                progress.view(-1, 1, 1).expand(-1, source.shape[1], -1),
            ),
            dim=-1,
        )
        condition = self.endpoint_conditioner(
            {
                "quantity": batch["quantity_ids"],
                "measurement": batch["measurement_ids"],
                "numerator": batch["numerator_ids"],
                "denominator": batch["denominator_ids"],
                "assay": batch["assay_ids"],
                "context": batch["context_ids"],
                "region": batch["region_ids"],
            }
        )
        hidden = self.input_norm(
            self.source_pretrained_projection(
                pretrained.to(self.source_pretrained_projection.weight.dtype)
            )
            + self.state_projection(state)
        ) * valid.unsqueeze(-1)
        for block in self.blocks:
            hidden = block(hidden, padding, condition)
        attention_logits = self.pool_attention(hidden).squeeze(-1).masked_fill(
            ~valid, -torch.inf
        )
        attention = torch.softmax(attention_logits, dim=1)
        pooled = (hidden * attention.unsqueeze(-1)).sum(dim=1)
        value = self.scalar_head(
            torch.cat(
                (
                    pooled,
                    condition,
                    remaining.log1p().unsqueeze(-1),
                    progress.unsqueeze(-1),
                ),
                dim=-1,
            )
        ).squeeze(-1)
        _require(value.shape == (source.shape[0],), "value output is not one scalar per state")
        return value


def normalized_particle_weights_v3(log_weights: Sequence[float]) -> np.ndarray:
    values = np.asarray(log_weights, dtype=float)
    _require(values.shape == (32,) and np.all(np.isfinite(values)), "SMC requires 32 finite particle weights")
    shifted = values - float(np.max(values))
    weights = np.exp(shifted)
    total = float(np.sum(weights))
    _require(math.isfinite(total) and total > 0.0, "SMC particle weight total is invalid")
    return weights / total


def effective_sample_size_v3(log_weights: Sequence[float]) -> float:
    weights = normalized_particle_weights_v3(log_weights)
    return float(1.0 / np.sum(np.square(weights)))


def stratified_resample_v3(
    log_weights: Sequence[float], *, seed: int, threshold: float = 16.0
) -> dict[str, Any]:
    _require(threshold == 16.0, "SMC ESS threshold differs from the freeze")
    weights = normalized_particle_weights_v3(log_weights)
    ess = float(1.0 / np.sum(np.square(weights)))
    if ess >= threshold:
        return {
            "resampled": False,
            "ess_before": ess,
            "ancestor_indices": list(range(32)),
            "log_weights_after": list(np.log(weights)),
        }
    rng = np.random.default_rng(seed)
    positions = (np.arange(32, dtype=float) + rng.random(32)) / 32.0
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0
    ancestors = np.searchsorted(cumulative, positions, side="right")
    return {
        "resampled": True,
        "ess_before": ess,
        "ancestor_indices": ancestors.astype(int).tolist(),
        "log_weights_after": [float(-math.log(32.0))] * 32,
    }


def smc_importance_transition_v3(
    base_rates: torch.Tensor,
    legal_mask: torch.Tensor,
    current_potential: torch.Tensor,
    child_potential: torch.Tensor,
    *,
    progress: torch.Tensor,
    beta_max: float,
    uniforms: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Propose from base rates and weight by the frozen scalar potential.

    Multiplying the base proposal by ``exp(beta * (V(child)-V(state)))``
    recovers the exact unnormalised guided rate without a free action-ratio.
    """

    guided_rates = potential_guided_rates_v3(
        base_rates,
        legal_mask,
        current_potential,
        child_potential,
        progress=progress,
        beta_max=beta_max,
    )
    _require(uniforms.shape == current_potential.shape, "SMC uniforms differ")
    _require(
        bool(torch.isfinite(uniforms).all().item())
        and bool(((uniforms >= 0.0) & (uniforms < 1.0)).all().item()),
        "SMC uniform is outside [0,1)",
    )
    proposal = torch.where(legal_mask, base_rates, torch.zeros_like(base_rates))
    proposal = proposal / proposal.sum(dim=1, keepdim=True)
    cumulative = proposal.double().cumsum(dim=1)
    choices = (cumulative < uniforms.double().unsqueeze(1)).sum(dim=1).clamp_max(
        base_rates.shape[1] - 1
    )
    rows = torch.arange(base_rates.shape[0], device=base_rates.device)
    _require(bool(legal_mask[rows, choices].all().item()), "SMC proposal selected an illegal action")
    beta = beta_schedule_v3(progress, beta_max=beta_max).to(base_rates)
    increments = beta * (child_potential[rows, choices] - current_potential)
    guided_probability = guided_rates / guided_rates.sum(dim=1, keepdim=True)
    _require(
        bool(torch.isfinite(increments).all().item())
        and bool(torch.isfinite(guided_probability[legal_mask]).all().item()),
        "SMC potential weight is nonfinite",
    )
    return {
        "choice_indices": choices,
        "log_weight_increment": increments,
        "base_proposal_probability": proposal,
        "guided_action_probability": guided_probability,
        "guided_rates": guided_rates,
    }


@dataclass(frozen=True)
class PotentialTransitionSetV3:
    actions: tuple[Any, ...]
    children: tuple[Any, ...]
    base_rates: tuple[float, ...]
    current_potential: float
    child_potentials: tuple[float, ...]
    progress: float


def run_scalar_potential_smc_v3(
    root_state: Any,
    transition_provider: Callable[[Any], PotentialTransitionSetV3],
    *,
    particle_seeds: Sequence[int],
    resampling_seed: int,
    beta_max: float,
    is_terminal: Callable[[Any], bool],
    candidate_sequence: Callable[[Any], str],
) -> dict[str, Any]:
    """Replayable 32-particle SMC over an arbitrary hard-legal state graph."""

    _require(len(particle_seeds) == 32, "formal SMC requires exactly 32 particle seeds")
    _require(len(set(int(seed) for seed in particle_seeds)) == 32, "SMC particle seed stream is duplicated")
    generators = [np.random.default_rng(int(seed)) for seed in particle_seeds]
    particles = [root_state for _ in range(32)]
    trajectories: list[list[Any]] = [[] for _ in range(32)]
    log_weights = np.full(32, -math.log(32.0), dtype=float)
    resampling_events = []
    step = 0
    while not all(is_terminal(state) for state in particles):
        active = [index for index, state in enumerate(particles) if not is_terminal(state)]
        _require(step <= 5, "SMC trajectory exceeded five edits plus structural termination")
        for index in active:
            transition = transition_provider(particles[index])
            count = len(transition.actions)
            _require(
                count > 0
                and len(transition.children) == count
                and len(transition.base_rates) == count
                and len(transition.child_potentials) == count,
                "SMC transition geometry differs",
            )
            rates = torch.tensor([transition.base_rates], dtype=torch.float64)
            legal = torch.ones_like(rates, dtype=torch.bool)
            current = torch.tensor([transition.current_potential], dtype=torch.float64)
            child = torch.tensor([transition.child_potentials], dtype=torch.float64)
            progress = torch.tensor([transition.progress], dtype=torch.float64)
            uniform = torch.tensor([generators[index].random()], dtype=torch.float64)
            sampled = smc_importance_transition_v3(
                rates,
                legal,
                current,
                child,
                progress=progress,
                beta_max=beta_max,
                uniforms=uniform,
            )
            choice = int(sampled["choice_indices"][0])
            particles[index] = transition.children[choice]
            trajectories[index].append(transition.actions[choice])
            log_weights[index] += float(sampled["log_weight_increment"][0])
        step += 1
        resampling = stratified_resample_v3(
            log_weights,
            seed=int(resampling_seed) + step,
            threshold=16.0,
        )
        resampling_events.append(
            {
                "step": step,
                "ess_before": resampling["ess_before"],
                "resampled": resampling["resampled"],
            }
        )
        if resampling["resampled"]:
            ancestors = resampling["ancestor_indices"]
            particles = [particles[index] for index in ancestors]
            trajectories = [list(trajectories[index]) for index in ancestors]
            log_weights = np.asarray(resampling["log_weights_after"], dtype=float)
    terminals = [
        {
            "candidate_sequence": candidate_sequence(state),
            "log_weight": float(log_weights[index]),
            "trajectory_actions": list(trajectories[index]),
            "particle_slot": index,
        }
        for index, state in enumerate(particles)
    ]
    candidates = deduplicate_terminal_candidates_v3(terminals, candidate_cap=32)
    return {
        "schema_version": "route_a_v3_route2_xeditflow_potential_smc.v3",
        "particle_count": 32,
        "candidate_cap": 32,
        "completed_steps": step,
        "resampling_events": resampling_events,
        "terminal_particle_count": 32,
        "unique_candidate_count": len(candidates),
        "candidates": candidates,
        "fixed_seed_replayable": True,
        "proposal": "BASE_FLOW_TRANSITION",
        "incremental_importance_weight": "EXP_BETA_TIMES_SCALAR_POTENTIAL_DIFFERENCE",
        "free_action_ratio_head_used": False,
    }


def deduplicate_terminal_candidates_v3(
    particles: Sequence[Mapping[str, Any]], *, candidate_cap: int = 32
) -> list[dict[str, Any]]:
    _require(candidate_cap == 32, "SMC candidate cap differs from the freeze")
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for particle in particles:
        sequence = str(particle["candidate_sequence"])
        grouped.setdefault(sequence, []).append(particle)
    result = []
    for sequence, members in grouped.items():
        log_weights = np.asarray([float(row["log_weight"]) for row in members])
        _require(np.all(np.isfinite(log_weights)), "terminal particle weight is nonfinite")
        maximum = float(np.max(log_weights))
        merged = maximum + math.log(float(np.exp(log_weights - maximum).sum()))
        representative = dict(max(members, key=lambda row: float(row["log_weight"])))
        representative["candidate_sequence"] = sequence
        representative["merged_log_weight"] = merged
        representative["particle_multiplicity"] = len(members)
        result.append(representative)
    result.sort(key=lambda row: (-float(row["merged_log_weight"]), row["candidate_sequence"]))
    return result[:candidate_cap]


@dataclass
class MatchedComputeRecordV2:
    source_key: str
    base_flow_forwards: int = 0
    value_forwards: int = 0
    critic_forwards_by_member: list[int] = field(default_factory=lambda: [0, 0, 0])
    candidate_count: int = 0
    wall_time_seconds: float = 0.0
    peak_vram_mb: float = 0.0
    edit_budget_violation_count: int = 0
    candidate_budget_violation_count: int = 0
    replay_failure_count: int = 0
    numerical_failure_count: int = 0

    def add_critic_forwards(self, member_index: int, count: int = 1) -> None:
        _require(member_index in {0, 1, 2}, "critic ensemble member index differs")
        _require(count >= 0, "critic forward count is negative")
        self.critic_forwards_by_member[member_index] += int(count)

    @property
    def total_forward_equivalents(self) -> int:
        return int(
            self.base_flow_forwards
            + self.value_forwards
            + sum(self.critic_forwards_by_member)
        )

    def to_dict(self) -> dict[str, Any]:
        _require(self.candidate_count <= 32, "matched-compute candidate cap exceeded")
        _require(self.total_forward_equivalents <= 320, "matched-compute forward ceiling exceeded")
        _require(
            min(
                self.base_flow_forwards,
                self.value_forwards,
                *self.critic_forwards_by_member,
                self.candidate_count,
                self.edit_budget_violation_count,
                self.candidate_budget_violation_count,
                self.replay_failure_count,
                self.numerical_failure_count,
            )
            >= 0,
            "matched-compute counter is negative",
        )
        _require(
            math.isfinite(self.wall_time_seconds)
            and self.wall_time_seconds >= 0.0
            and math.isfinite(self.peak_vram_mb)
            and self.peak_vram_mb >= 0.0,
            "matched-compute resource measurement is invalid",
        )
        return {
            "schema_version": "MatchedComputeRecordV2",
            "source_key": self.source_key,
            "base_flow_forwards": self.base_flow_forwards,
            "value_forwards": self.value_forwards,
            "critic_forwards_by_member": list(self.critic_forwards_by_member),
            "critic_forward_total": sum(self.critic_forwards_by_member),
            "total_forward_equivalents": self.total_forward_equivalents,
            "forward_equivalent_ceiling": 320,
            "candidate_count": self.candidate_count,
            "candidate_cap": 32,
            "wall_time_seconds": self.wall_time_seconds,
            "peak_vram_mb": self.peak_vram_mb,
            "failure_counters": {
                "edit_budget_violation_count": self.edit_budget_violation_count,
                "candidate_budget_violation_count": self.candidate_budget_violation_count,
                "replay_failure_count": self.replay_failure_count,
                "numerical_failure_count": self.numerical_failure_count,
            },
        }
