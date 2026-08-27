"""Cross-state mode-responsibility successor objective for XEditSetFlow S1."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import torch

from core.route2_xeditsetflow_runtime_v4 import build_setflow_screen_model_v4
from core.route2_xeditsetflow_v4 import (
    MixtureSetFlowLossV4,
    XEditSetFlowV4,
    mixture_setflow_loss_v4,
)


S1_RUN_ROLE_TO_V4 = {
    "v4_s1_full": "v4_full",
    "v4_s1_single_mode": "v4_single_mode",
}


class XEditSetFlowS1Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise XEditSetFlowS1Error(message)


@dataclass(frozen=True)
class SetFlowScreenRunSpecS1:
    run_id: str
    mode_count: int
    mode_information_weight: float
    selectable: bool


@dataclass(frozen=True)
class MixtureSetFlowLossS1:
    total: torch.Tensor
    common_set_marginal: torch.Tensor
    source_candidate_coverage: torch.Tensor
    remaining_count: torch.Tensor
    mode_information: torch.Tensor
    active_state_count: int
    active_candidate_constraint_count: int
    base_v4: MixtureSetFlowLossV4
    cross_state_candidate_mode_responsibility: torch.Tensor
    active_responsibility_constraint_count: int
    active_responsibility_candidate_count: int
    active_responsibility_occurrence_count: int


def screen_run_spec_s1(
    config: Mapping[str, Any], run_id: str
) -> SetFlowScreenRunSpecS1:
    """Validate one S1 arm without relabeling its external run identity as V4."""

    _require(run_id in S1_RUN_ROLE_TO_V4, "run id is not one frozen SetFlow S1 screen run")
    rows = config.get("required_screen_runs")
    _require(isinstance(rows, list), "SetFlow S1 required screen runs are absent")
    matches = [
        row
        for row in rows
        if isinstance(row, Mapping) and str(row.get("run_id")) == run_id
    ]
    _require(len(matches) == 1, "run id is not one unique frozen SetFlow S1 screen run")
    row = matches[0]
    spec = SetFlowScreenRunSpecS1(
        run_id=str(row["run_id"]),
        mode_count=int(row["mode_count"]),
        mode_information_weight=float(row["mode_information_weight"]),
        selectable=bool(row["selectable"]),
    )
    _require(
        (
            spec.run_id,
            spec.mode_count,
            spec.mode_information_weight,
            spec.selectable,
        )
        in {
            ("v4_s1_full", 8, 0.05, True),
            ("v4_s1_single_mode", 1, 0.0, False),
        },
        "SetFlow S1 screen role changed",
    )
    return spec


def build_setflow_screen_model_s1(
    config: Mapping[str, Any],
    vocabs: Mapping[str, Mapping[str, int]],
    *,
    run_id: str,
) -> tuple[XEditSetFlowV4, dict[str, Any]]:
    """Build the frozen V4 architecture while retaining the distinct S1 arm role."""

    spec = screen_run_spec_s1(config, run_id)
    architecture_role = S1_RUN_ROLE_TO_V4[spec.run_id]
    scoped_role = {
        "run_id": architecture_role,
        "mode_count": spec.mode_count,
        "mode_information_weight": spec.mode_information_weight,
        "selectable": spec.selectable,
    }
    scoped_config = {
        **dict(config),
        "required_screen_runs": [scoped_role],
    }
    return build_setflow_screen_model_v4(
        scoped_config,
        vocabs,
        run_id=architecture_role,
    )


def _candidate_mode_responsibilities_s1(
    output: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    rates = output["mode_rates"].float()
    prior = output["mode_prior"].float()
    candidate_positive = batch["candidate_positive_action_mask"]
    candidate_valid = batch["candidate_valid_mask"]
    _require(rates.ndim == 3, "SetFlow S1 rates are not batch x mode x action")
    _require(
        prior.shape == rates.shape[:2],
        "SetFlow S1 mode priors do not align with rates",
    )
    _require(
        candidate_positive.ndim == 3
        and candidate_positive.shape[0] == rates.shape[0]
        and candidate_positive.shape[2] == rates.shape[2]
        and candidate_valid.shape == candidate_positive.shape[:2],
        "SetFlow S1 candidate target geometry changed",
    )
    mode_probabilities = rates / rates.sum(dim=2, keepdim=True).clamp_min(1e-20)
    candidate_mode_mass = torch.einsum(
        "bma,bca->bcm",
        mode_probabilities,
        candidate_positive.to(mode_probabilities.dtype),
    )
    joint = candidate_mode_mass * prior[:, None, :]
    active = candidate_valid & ~batch["structural_budget_exhausted"][:, None]
    _require(
        bool(torch.isfinite(joint[active]).all().item()),
        "SetFlow S1 active candidate-mode mass is nonfinite",
    )
    _require(
        bool((joint.sum(dim=2)[active] > 0).all().item()),
        "SetFlow S1 active candidate has no mode responsibility mass",
    )
    return joint / joint.sum(dim=2, keepdim=True).clamp_min(1e-20)


def mixture_setflow_loss_s1(
    output: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
    *,
    coverage_weight: float = 0.50,
    remaining_count_weight: float = 0.20,
    mode_information_weight: float = 0.05,
    state_slots: torch.Tensor,
    source_occurrence_ids: torch.Tensor,
    canonical_candidate_indices: torch.Tensor,
    cross_state_candidate_mode_responsibility_weight: float,
) -> MixtureSetFlowLossS1:
    """Add detached-root forward KL without changing the frozen V4 objective."""

    base_v4 = mixture_setflow_loss_v4(
        output,
        batch,
        coverage_weight=coverage_weight,
        remaining_count_weight=remaining_count_weight,
        mode_information_weight=mode_information_weight,
    )
    rates = output["mode_rates"]
    candidate_valid = batch["candidate_valid_mask"]
    structural = batch["structural_budget_exhausted"]
    batch_size = rates.shape[0]
    _require(
        state_slots.shape == source_occurrence_ids.shape == (batch_size,),
        "SetFlow S1 state and occurrence identities do not align",
    )
    _require(
        canonical_candidate_indices.shape == candidate_valid.shape,
        "SetFlow S1 canonical candidate identities do not align",
    )
    _require(
        state_slots.dtype == torch.long
        and source_occurrence_ids.dtype == torch.long
        and canonical_candidate_indices.dtype == torch.long,
        "SetFlow S1 identities must use torch.long",
    )
    _require(
        state_slots.device
        == source_occurrence_ids.device
        == canonical_candidate_indices.device
        == rates.device,
        "SetFlow S1 identities are not on the model-output device",
    )
    _require(
        bool(torch.equal(canonical_candidate_indices >= 0, candidate_valid)),
        "SetFlow S1 canonical candidate identities do not align with validity",
    )
    _require(
        batch_size % 4 == 0
        and torch.equal(
            state_slots,
            torch.arange(batch_size, device=state_slots.device) % 4,
        )
        and torch.equal(
            source_occurrence_ids,
            torch.arange(batch_size, device=source_occurrence_ids.device) // 4,
        ),
        "SetFlow S1 rows are not contiguous four-state source occurrences",
    )
    weight = float(cross_state_candidate_mode_responsibility_weight)
    _require(
        math.isfinite(weight) and weight >= 0.0,
        "SetFlow S1 cross-state responsibility weight is invalid",
    )
    responsibilities = _candidate_mode_responsibilities_s1(output, batch)
    occurrence_losses: list[torch.Tensor] = []
    active_constraint_count = 0
    active_candidate_count = 0
    for occurrence_id in range(batch_size // 4):
        root_row = occurrence_id * 4
        _require(
            not bool(structural[root_row].item()),
            "SetFlow S1 root state is structural",
        )
        root_positions = torch.nonzero(
            candidate_valid[root_row], as_tuple=False
        ).flatten()
        root_by_canonical: dict[int, int] = {}
        for position in root_positions.tolist():
            canonical_index = int(
                canonical_candidate_indices[root_row, position].item()
            )
            _require(
                canonical_index not in root_by_canonical,
                "SetFlow S1 root has duplicate canonical candidate identity",
            )
            root_by_canonical[canonical_index] = int(position)
        _require(bool(root_by_canonical), "SetFlow S1 root has no compatible candidate")
        per_candidate_state_losses: dict[int, list[torch.Tensor]] = {}
        for row in range(root_row + 1, root_row + 4):
            if bool(structural[row].item()):
                continue
            seen_nonroot: set[int] = set()
            for position in torch.nonzero(
                candidate_valid[row], as_tuple=False
            ).flatten().tolist():
                canonical_index = int(
                    canonical_candidate_indices[row, position].item()
                )
                _require(
                    canonical_index not in seen_nonroot,
                    "SetFlow S1 non-root state has duplicate canonical candidate identity",
                )
                seen_nonroot.add(canonical_index)
                _require(
                    canonical_index in root_by_canonical,
                    "SetFlow S1 non-root candidate has no root posterior target",
                )
                root_position = root_by_canonical[canonical_index]
                target = responsibilities[root_row, root_position].detach()
                current = responsibilities[row, position]
                forward_kl = (
                    target
                    * (
                        torch.log(target.clamp_min(1e-20))
                        - torch.log(current.clamp_min(1e-20))
                    )
                ).sum()
                per_candidate_state_losses.setdefault(canonical_index, []).append(
                    forward_kl
                )
                active_constraint_count += 1
        candidate_losses = [
            torch.stack(per_candidate_state_losses[canonical_index]).mean()
            for canonical_index in sorted(per_candidate_state_losses)
        ]
        if candidate_losses:
            occurrence_losses.append(torch.stack(candidate_losses).mean())
            active_candidate_count += len(candidate_losses)
    cross_state_loss = (
        torch.stack(occurrence_losses).mean()
        if occurrence_losses and rates.shape[1] > 1
        else rates.new_zeros((), dtype=torch.float32)
    )
    _require(
        bool(torch.isfinite(cross_state_loss).item()),
        "SetFlow S1 cross-state responsibility loss is nonfinite",
    )
    total = (
        base_v4.total
        if rates.shape[1] == 1 or not occurrence_losses
        else base_v4.total + weight * cross_state_loss
    )
    _require(bool(torch.isfinite(total).item()), "SetFlow S1 total loss is nonfinite")
    return MixtureSetFlowLossS1(
        total=total,
        common_set_marginal=base_v4.common_set_marginal,
        source_candidate_coverage=base_v4.source_candidate_coverage,
        remaining_count=base_v4.remaining_count,
        mode_information=base_v4.mode_information,
        active_state_count=base_v4.active_state_count,
        active_candidate_constraint_count=base_v4.active_candidate_constraint_count,
        base_v4=base_v4,
        cross_state_candidate_mode_responsibility=cross_state_loss,
        active_responsibility_constraint_count=active_constraint_count,
        active_responsibility_candidate_count=active_candidate_count,
        active_responsibility_occurrence_count=len(occurrence_losses),
    )
