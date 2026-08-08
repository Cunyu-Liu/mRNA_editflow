"""Route A and B ordinary-only source-relative effect critics."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn

from .action_support import ActionSupportHoldError


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 5  # PAD=0, A/C/G/U=1..4
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    ff_dim: int = 128
    max_len: int = 512
    n_endpoints: int = 2  # includes explicit UNK=0
    n_contexts: int = 2  # includes explicit UNK=0
    max_edits: int = 5
    pair_rank: int = 4

    def __post_init__(self) -> None:
        integer_fields = (
            self.vocab_size, self.d_model, self.n_heads, self.n_layers,
            self.ff_dim, self.max_len, self.n_endpoints, self.n_contexts,
            self.max_edits, self.pair_rank,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in integer_fields):
            raise ValueError("all ModelConfig dimensions must be positive integers")
        if self.vocab_size != 5:
            raise ValueError("R3X nucleotide vocabulary is frozen as PAD+A/C/G/U (size 5)")
        if self.d_model < 8:
            raise ValueError("R3X publication route requires d_model >= 8")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.max_edits != 5:
            raise ValueError("R3X primary edit budget uses a max of five substitutions")


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def require_parameter_match(
    reference: nn.Module,
    candidate: nn.Module,
    *,
    maximum_gap_fraction: float = 0.05,
) -> dict[str, float | int | bool]:
    """Fail closed when a matched-control claim is false for the actual config."""

    if (
        isinstance(maximum_gap_fraction, bool)
        or not isinstance(maximum_gap_fraction, (int, float))
        or not math.isfinite(float(maximum_gap_fraction))
        or not 0.0 <= float(maximum_gap_fraction) <= 1.0
    ):
        raise ValueError("maximum_gap_fraction must be finite in [0,1]")
    reference_count = count_parameters(reference)
    candidate_count = count_parameters(candidate)
    if reference_count <= 0:
        raise ValueError("reference model has no trainable parameters")
    gap = abs(candidate_count - reference_count) / reference_count
    if gap > float(maximum_gap_fraction):
        raise ValueError(
            "parameter-matched claim fails for the actual frozen configuration"
        )
    return {
        "reference_parameters": reference_count,
        "candidate_parameters": candidate_count,
        "gap_fraction": float(gap),
        "maximum_gap_fraction": float(maximum_gap_fraction),
        "matched": True,
    }


def derive_replay_edit_tensors(
    source_tokens: torch.Tensor,
    candidate_tokens: torch.Tensor,
    source_mask: torch.Tensor,
    candidate_mask: torch.Tensor,
    *,
    max_edits: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Derive the exact sparse SUB action from an aligned sequence pair.

    This is intentionally a parser of an already replay-qualified pair, not a
    second action authority.  It admits only the frozen primary budgets and
    orders edits by their actual sequence position, so a B critic cannot obtain
    a caller-selected edit ordering or a dense candidate feature branch.
    """

    if isinstance(max_edits, bool) or not isinstance(max_edits, int) or max_edits < 5:
        raise ValueError("max_edits must cover the frozen five-edit budget")
    if (
        source_tokens.ndim != 2
        or candidate_tokens.ndim != 2
        or source_mask.ndim != 2
        or candidate_mask.ndim != 2
        or source_tokens.shape != candidate_tokens.shape
        or source_mask.shape != source_tokens.shape
        or candidate_mask.shape != source_tokens.shape
    ):
        raise ValueError(
            "replay-derived edits require aligned source/candidate [batch,length] tensors"
        )
    if source_mask.dtype is not torch.bool or candidate_mask.dtype is not torch.bool:
        raise ValueError("replay-derived edit masks must be boolean")
    if not torch.equal(source_mask, candidate_mask):
        raise ValueError("replay-derived SUB edits require an exact equal sequence mask")
    if torch.any(source_tokens[~source_mask] != 0) or torch.any(
        candidate_tokens[~candidate_mask] != 0
    ):
        raise ValueError("masked replay-derived sequence positions must carry PAD=0")

    changed = (source_tokens != candidate_tokens) & source_mask
    counts = changed.sum(dim=1)
    if any(int(value) not in {1, 3, 5} for value in counts.tolist()):
        raise ValueError("replay-derived edit budget must be exactly one of {1,3,5}")
    batch = source_tokens.shape[0]
    positions = torch.zeros(
        (batch, max_edits), dtype=torch.long, device=source_tokens.device
    )
    ref = torch.zeros_like(positions)
    alt = torch.zeros_like(positions)
    edit_mask = torch.zeros(
        (batch, max_edits), dtype=torch.bool, device=source_tokens.device
    )
    for row in range(batch):
        active_positions = torch.nonzero(changed[row], as_tuple=False).flatten()
        count = int(active_positions.numel())
        positions[row, :count] = active_positions
        ref[row, :count] = source_tokens[row, active_positions]
        alt[row, :count] = candidate_tokens[row, active_positions]
        edit_mask[row, :count] = True
    if torch.any(ref[edit_mask] == alt[edit_mask]):
        raise ValueError("replay-derived edits cannot contain null substitutions")
    return positions, ref, alt, edit_mask


def derive_wrong_local_feature_positions(
    source_tokens: torch.Tensor,
    source_mask: torch.Tensor,
    edit_positions: torch.Tensor,
    edit_ref: torch.Tensor,
    edit_mask: torch.Tensor,
) -> torch.Tensor:
    """Relocate edit-local features to compatible, unedited source positions.

    Actual actions and the candidate replay are never changed.  For every SUB,
    the control selects the first still-unused visible position carrying the
    same reference nucleotide, excluding all actual edit positions.  Processing
    is sorted by actual position, making the intervention invariant to an input
    edit-list permutation.  Absence of compatible support is an explicit HOLD,
    not an identity fallback.
    """

    if (
        source_tokens.ndim != 2
        or source_mask.ndim != 2
        or source_tokens.shape != source_mask.shape
        or edit_positions.ndim != 2
        or edit_ref.shape != edit_positions.shape
        or edit_mask.shape != edit_positions.shape
        or edit_mask.dtype is not torch.bool
    ):
        raise ValueError("wrong-local position tensors are not aligned")
    relocated = edit_positions.masked_fill(~edit_mask, 0).clone()
    for row in range(source_tokens.shape[0]):
        active_columns = torch.nonzero(edit_mask[row], as_tuple=False).flatten().tolist()
        active_columns.sort(key=lambda column: int(edit_positions[row, column]))
        actual = {int(edit_positions[row, column]) for column in active_columns}
        chosen: set[int] = set()
        visible_length = int(source_mask[row].sum().item())
        for column in active_columns:
            ref_token = int(edit_ref[row, column])
            candidates = [
                position
                for position in range(visible_length)
                if position not in actual
                and position not in chosen
                and int(source_tokens[row, position]) == ref_token
            ]
            if not candidates:
                raise ValueError(
                    "HOLD_WRONG_LOCAL_INSUFFICIENT_COMPATIBLE_POSITION_SUPPORT"
                )
            selected = candidates[0]
            relocated[row, column] = selected
            chosen.add(selected)
    return relocated


class SharedSequenceEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=0)
        self.position_embedding = nn.Embedding(cfg.max_len, cfg.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.ff_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # Padding is already handled explicitly; disabling nested tensors keeps
        # behavior/warnings stable across supported PyTorch builds.
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=cfg.n_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if tokens.ndim != 2 or mask.ndim != 2 or tokens.shape != mask.shape:
            raise ValueError("sequence tokens/mask must be aligned [batch,length]")
        if tokens.shape[1] > self.cfg.max_len:
            raise ValueError("sequence exceeds frozen max_len; silent truncation is prohibited")
        if mask.dtype is not torch.bool:
            raise ValueError("sequence mask must be boolean with True for valid tokens")
        if not torch.all(mask.any(dim=1)):
            raise ValueError("each sequence requires at least one visible token")
        valid = tokens[mask]
        if torch.any(valid < 1) or torch.any(valid >= self.cfg.vocab_size):
            raise ValueError("visible nucleotide token is outside A/C/G/U vocabulary")
        if torch.any(tokens[~mask] != 0):
            raise ValueError("masked sequence positions must carry PAD=0")
        positions = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0)
        hidden = self.token_embedding(tokens) + self.position_embedding(positions)
        hidden = self.transformer(hidden, src_key_padding_mask=~mask)
        hidden = self.norm(hidden)
        hidden = hidden * mask.unsqueeze(-1)
        pooled = hidden.sum(dim=1) / mask.sum(dim=1, keepdim=True).to(hidden.dtype)
        return hidden, pooled


class _ContextConditioning(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.endpoint_embedding = nn.Embedding(cfg.n_endpoints, cfg.d_model)
        self.context_embedding = nn.Embedding(cfg.n_contexts, cfg.d_model)
        self.condition_norm = nn.LayerNorm(cfg.d_model)

    def condition(self, endpoint_ids: torch.Tensor, context_ids: torch.Tensor) -> torch.Tensor:
        if endpoint_ids.ndim != 1 or context_ids.ndim != 1 or endpoint_ids.shape != context_ids.shape:
            raise ValueError("endpoint/context IDs must be aligned vectors")
        if torch.any(endpoint_ids < 0) or torch.any(endpoint_ids >= self.cfg.n_endpoints):
            raise ValueError("endpoint ID out of vocabulary; unseen values must use UNK=0")
        if torch.any(context_ids < 0) or torch.any(context_ids >= self.cfg.n_contexts):
            raise ValueError("context ID out of vocabulary; unseen values must use UNK=0")
        return self.condition_norm(
            self.endpoint_embedding(endpoint_ids) + self.context_embedding(context_ids)
        )


class ExactPotentialCritic(_ContextConditioning):
    """A: exact endpoint/context-conditioned scalar potential difference."""

    architecture_id = "R3X_A_EXACT_POTENTIAL_DIFFERENCE_V1"
    information_set = "PAIRED_SEQUENCE_NO_MEASURED_ANCHOR"

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        self.encoder = SharedSequenceEncoder(cfg)
        self.potential_head = nn.Sequential(
            nn.Linear(2 * cfg.d_model, cfg.d_model),
            nn.SiLU(),
            nn.Linear(cfg.d_model, 1),
        )
        self.logvar_head = nn.Sequential(
            nn.Linear(3 * cfg.d_model, cfg.d_model),
            nn.SiLU(),
            nn.Linear(cfg.d_model, 1),
        )

    def _represent(
        self,
        source_tokens: torch.Tensor,
        candidate_tokens: torch.Tensor,
        source_mask: torch.Tensor,
        candidate_mask: torch.Tensor,
        endpoint_ids: torch.Tensor,
        context_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, source = self.encoder(source_tokens, source_mask)
        _, candidate = self.encoder(candidate_tokens, candidate_mask)
        condition = self.condition(endpoint_ids, context_ids)
        if source.shape[0] != condition.shape[0]:
            raise ValueError("sequence and context batch sizes must align")
        return source, candidate, condition

    def forward(
        self,
        source_tokens: torch.Tensor,
        candidate_tokens: torch.Tensor,
        source_mask: torch.Tensor,
        candidate_mask: torch.Tensor,
        endpoint_ids: torch.Tensor,
        context_ids: torch.Tensor,
    ) -> dict:
        source, candidate, condition = self._represent(
            source_tokens, candidate_tokens, source_mask, candidate_mask,
            endpoint_ids, context_ids,
        )
        source_potential = self.potential_head(torch.cat([source, condition], dim=-1)).squeeze(-1)
        candidate_potential = self.potential_head(torch.cat([candidate, condition], dim=-1)).squeeze(-1)
        mean = candidate_potential - source_potential
        symmetric = torch.cat([
            torch.abs(candidate - source),
            0.5 * (candidate + source),
            condition,
        ], dim=-1)
        logvar = torch.clamp(self.logvar_head(symmetric).squeeze(-1), -12.0, 8.0)
        return {
            "mean": mean,
            "logvar": logvar,
            "sign": mean,
            "rank": mean,
            "architecture_id": self.architecture_id,
            "sign_decision_rule": "RAW_PREDICTED_DELTA_MEAN_GE_ZERO",
        }


class SiteAdditiveLocalGlobalExactPotentialCritic(ExactPotentialCritic):
    """Historical global-context site-sum exact-potential diagnostic.

    The site head consumes Transformer hidden states, so a single edit can
    change every site's energy.  This class therefore does not assert bounded
    locality and is not the V6 primary sparse-edit mechanism candidate.
    """

    architecture_id = "R3X_A1_SITE_ADDITIVE_LOCAL_GLOBAL_EXACT_POTENTIAL_V1"

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        site_hidden = max(4, cfg.d_model // 2)
        self.site_potential_head = nn.Sequential(
            nn.Linear(2 * cfg.d_model, site_hidden),
            nn.SiLU(),
            nn.Linear(site_hidden, 1),
        )

    def _local_global_potential(
        self,
        hidden: torch.Tensor,
        pooled: torch.Tensor,
        mask: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        global_potential = self.potential_head(
            torch.cat([pooled, condition], dim=-1)
        ).squeeze(-1)
        expanded_condition = condition.unsqueeze(1).expand(-1, hidden.shape[1], -1)
        site_energy = self.site_potential_head(
            torch.cat([hidden, expanded_condition], dim=-1)
        ).squeeze(-1)
        visible = mask.to(site_energy.dtype)
        site_potential = (site_energy * visible).sum(dim=1) / torch.sqrt(
            visible.sum(dim=1).clamp_min(1.0)
        )
        return global_potential + site_potential

    def _local_global_represent(
        self,
        source_tokens: torch.Tensor,
        candidate_tokens: torch.Tensor,
        source_mask: torch.Tensor,
        candidate_mask: torch.Tensor,
        endpoint_ids: torch.Tensor,
        context_ids: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        source_hidden, source = self.encoder(source_tokens, source_mask)
        candidate_hidden, candidate = self.encoder(candidate_tokens, candidate_mask)
        condition = self.condition(endpoint_ids, context_ids)
        if source.shape[0] != condition.shape[0]:
            raise ValueError("sequence and context batch sizes must align")
        source_potential = self._local_global_potential(
            source_hidden, source, source_mask, condition
        )
        candidate_potential = self._local_global_potential(
            candidate_hidden, candidate, candidate_mask, condition
        )
        return source, candidate, condition, source_potential, candidate_potential

    def forward(
        self,
        source_tokens: torch.Tensor,
        candidate_tokens: torch.Tensor,
        source_mask: torch.Tensor,
        candidate_mask: torch.Tensor,
        endpoint_ids: torch.Tensor,
        context_ids: torch.Tensor,
    ) -> dict:
        source, candidate, condition, source_potential, candidate_potential = (
            self._local_global_represent(
                source_tokens,
                candidate_tokens,
                source_mask,
                candidate_mask,
                endpoint_ids,
                context_ids,
            )
        )
        mean = candidate_potential - source_potential
        symmetric = torch.cat(
            [
                torch.abs(candidate - source),
                0.5 * (candidate + source),
                condition,
            ],
            dim=-1,
        )
        logvar = torch.clamp(
            self.logvar_head(symmetric).squeeze(-1), -12.0, 8.0
        )
        return {
            "mean": mean,
            "logvar": logvar,
            "sign": mean,
            "rank": mean,
            "architecture_id": self.architecture_id,
            "sign_decision_rule": "RAW_PREDICTED_DELTA_MEAN_GE_ZERO",
            "potential_constraint": True,
            "local_site_branch": True,
            "bounded_local_support": False,
            "mechanism_scope": "GLOBAL_CONTEXT_SITE_SUM_DIAGNOSTIC",
            "site_normalization": "SUM_OVER_SQRT_VISIBLE_LENGTH",
        }


class BoundedLocalGlobalExactPotentialCritic(ExactPotentialCritic):
    """V6 A1: exact global potential plus a finite-receptive-field site sum.

    The local branch is a shared depthwise-separable convolution over each
    sequence independently.  With kernel width five, changing one nucleotide
    can alter local hidden states only within radius two.  The final effect is
    still exactly ``phi(candidate, condition) - phi(source, condition)``.
    """

    architecture_id = "R3X_A1_BOUNDED_LOCAL_GLOBAL_EXACT_POTENTIAL_V2"
    local_receptive_field_radius = 2

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        kernel_size = 2 * self.local_receptive_field_radius + 1
        self.local_depthwise = nn.Conv1d(
            cfg.d_model,
            cfg.d_model,
            kernel_size=kernel_size,
            padding=self.local_receptive_field_radius,
            groups=cfg.d_model,
            bias=False,
        )
        self.local_pointwise = nn.Conv1d(
            cfg.d_model, cfg.d_model, kernel_size=1, bias=False
        )
        self.local_norm = nn.LayerNorm(cfg.d_model)
        site_hidden = max(4, cfg.d_model // 2)
        self.local_site_head = nn.Sequential(
            nn.Linear(2 * cfg.d_model, site_hidden),
            nn.SiLU(),
            nn.Linear(site_hidden, 1, bias=False),
        )
        self.local_uncertainty_projection = nn.Linear(1, cfg.d_model, bias=False)

    def _bounded_local_hidden(
        self, tokens: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        if tokens.ndim != 2 or mask.ndim != 2 or tokens.shape != mask.shape:
            raise ValueError("local sequence tokens/mask must align [batch,length]")
        if tokens.shape[1] > self.cfg.max_len:
            raise ValueError("sequence exceeds frozen max_len")
        if mask.dtype is not torch.bool or not torch.all(mask.any(dim=1)):
            raise ValueError("local sequence mask must be valid boolean support")
        padded_prefix = (~mask).cumsum(dim=1) > 0
        if torch.any(mask & padded_prefix):
            raise ValueError("local sequence mask must be right-contiguous")
        valid = tokens[mask]
        if torch.any(valid < 1) or torch.any(valid >= self.cfg.vocab_size):
            raise ValueError("visible local token is outside A/C/G/U vocabulary")
        if torch.any(tokens[~mask] != 0):
            raise ValueError("masked local sequence positions must carry PAD=0")
        positions = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0)
        embedded = self.encoder.token_embedding(tokens) + self.encoder.position_embedding(
            positions
        )
        embedded = embedded * mask.unsqueeze(-1)
        convolved = self.local_depthwise(embedded.transpose(1, 2))
        convolved = self.local_pointwise(torch.nn.functional.gelu(convolved))
        hidden = self.local_norm(convolved.transpose(1, 2))
        return hidden * mask.unsqueeze(-1)

    def _bounded_local_site_energies(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self._bounded_local_hidden(tokens, mask)
        expanded_condition = condition.unsqueeze(1).expand(-1, hidden.shape[1], -1)
        energy = self.local_site_head(
            torch.cat([hidden, expanded_condition], dim=-1)
        ).squeeze(-1)
        return energy * mask.to(energy.dtype)

    def _bounded_local_potential_and_energy(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        pooled: torch.Tensor,
        condition: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        global_potential = self.potential_head(
            torch.cat([pooled, condition], dim=-1)
        ).squeeze(-1)
        local_energy = self._bounded_local_site_energies(tokens, mask, condition)
        local_potential = local_energy.sum(dim=1) / torch.sqrt(
            mask.sum(dim=1).to(local_energy.dtype).clamp_min(1.0)
        )
        return global_potential + local_potential, local_energy

    def _candidate_local_tokens(
        self,
        source_tokens: torch.Tensor,
        candidate_tokens: torch.Tensor,
        source_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return the candidate tensor consumed only by the local branch."""

        return candidate_tokens

    def _bounded_represent(
        self,
        source_tokens: torch.Tensor,
        candidate_tokens: torch.Tensor,
        source_mask: torch.Tensor,
        candidate_mask: torch.Tensor,
        endpoint_ids: torch.Tensor,
        context_ids: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if source_tokens.shape != candidate_tokens.shape:
            raise ValueError(
                "bounded-local substitution route requires aligned equal-shape tensors"
            )
        if source_mask.shape != candidate_mask.shape or not torch.equal(
            source_mask, candidate_mask
        ):
            raise ValueError(
                "bounded-local substitution route requires an exact equal mask"
            )
        _, source = self.encoder(source_tokens, source_mask)
        _, candidate = self.encoder(candidate_tokens, candidate_mask)
        condition = self.condition(endpoint_ids, context_ids)
        if source.shape[0] != condition.shape[0]:
            raise ValueError("sequence and context batch sizes must align")
        source_potential, source_local_energy = self._bounded_local_potential_and_energy(
            source_tokens, source_mask, source, condition
        )
        candidate_local_tokens = self._candidate_local_tokens(
            source_tokens, candidate_tokens, source_mask
        )
        candidate_potential, candidate_local_energy = self._bounded_local_potential_and_energy(
            candidate_local_tokens, candidate_mask, candidate, condition
        )
        visible = source_mask.to(source_local_energy.dtype)
        local_uncertainty_summary = (
            torch.abs(candidate_local_energy - source_local_energy) * visible
        ).sum(dim=1) / torch.sqrt(visible.sum(dim=1).clamp_min(1.0))
        return (
            source,
            candidate,
            condition,
            source_potential,
            candidate_potential,
            local_uncertainty_summary,
        )

    def forward(
        self,
        source_tokens: torch.Tensor,
        candidate_tokens: torch.Tensor,
        source_mask: torch.Tensor,
        candidate_mask: torch.Tensor,
        endpoint_ids: torch.Tensor,
        context_ids: torch.Tensor,
    ) -> dict:
        (
            source,
            candidate,
            condition,
            source_potential,
            candidate_potential,
            local_uncertainty_summary,
        ) = (
            self._bounded_represent(
                source_tokens,
                candidate_tokens,
                source_mask,
                candidate_mask,
                endpoint_ids,
                context_ids,
            )
        )
        mean = candidate_potential - source_potential
        symmetric = torch.cat(
            [
                torch.abs(candidate - source),
                0.5 * (candidate + source),
                condition
                + self.local_uncertainty_projection(
                    local_uncertainty_summary.unsqueeze(-1)
                ),
            ],
            dim=-1,
        )
        logvar = torch.clamp(
            self.logvar_head(symmetric).squeeze(-1), -12.0, 8.0
        )
        return {
            "mean": mean,
            "logvar": logvar,
            "sign": mean,
            "rank": mean,
            "architecture_id": self.architecture_id,
            "sign_decision_rule": "RAW_PREDICTED_DELTA_MEAN_GE_ZERO",
            "potential_constraint": True,
            "bounded_local_branch_support": True,
            "overall_effect_support": "GLOBAL_PLUS_FINITE_RF_LOCAL_NOT_STRICTLY_BOUNDED",
            "whole_mean_bounded_support": False,
            "uncertainty_bounded_support": False,
            "global_branch_present": True,
            "local_receptive_field_radius": self.local_receptive_field_radius,
            "local_receptive_field_overlap_possible_through_distance": 2
            * self.local_receptive_field_radius,
            "strict_local_interaction_exclusion_minimum_distance": 2
            * self.local_receptive_field_radius
            + 1,
            "length_extension_invariant": False,
            "site_normalization": "SUM_OVER_SQRT_VISIBLE_LENGTH",
            "local_uncertainty_summary": "SWAP_SYMMETRIC_ABSOLUTE_SITE_ENERGY_DELTA",
        }


class BoundedLocalWrongPositionControl(BoundedLocalGlobalExactPotentialCritic):
    """A1 same-capacity control with only the local edit position made wrong.

    The global source/candidate potential still receives the exact observed
    pair.  The candidate local branch instead receives the same ref-to-alt
    substitutions relocated to compatible unedited source positions.  Thus
    row identity, label, action replay, length, and nucleotide composition stay
    fixed while the hypothesized local mechanism is falsifiably disrupted.
    """

    architecture_id = "R3X_A1_BOUNDED_LOCAL_WRONG_POSITION_CONTROL_V1"

    def _candidate_local_tokens(
        self,
        source_tokens: torch.Tensor,
        candidate_tokens: torch.Tensor,
        source_mask: torch.Tensor,
    ) -> torch.Tensor:
        positions, ref, alt, edit_mask = derive_replay_edit_tensors(
            source_tokens,
            candidate_tokens,
            source_mask,
            source_mask,
            max_edits=self.cfg.max_edits,
        )
        relocated = derive_wrong_local_feature_positions(
            source_tokens, source_mask, positions, ref, edit_mask
        )
        wrong_candidate = source_tokens.clone()
        for row in range(source_tokens.shape[0]):
            active = edit_mask[row]
            wrong_candidate[row, relocated[row, active]] = alt[row, active]
        return wrong_candidate

    def forward(self, *args, **kwargs) -> dict:
        output = super().forward(*args, **kwargs)
        output.update(
            {
                "potential_constraint": False,
                "wrong_local_position_control": True,
                "wrong_local_scope": "LOCAL_CANDIDATE_BRANCH_ONLY",
                "actual_action_replay_changed": False,
                "scientific_control_only": True,
            }
        )
        return output


class BoundedLocalConditionUnknownControl(BoundedLocalGlobalExactPotentialCritic):
    """A1 same-capacity condition ablation using only the explicit UNK IDs."""

    architecture_id = "R3X_A1_BOUNDED_LOCAL_CONDITION_UNK_ONLY_V1"

    def forward(
        self,
        source_tokens: torch.Tensor,
        candidate_tokens: torch.Tensor,
        source_mask: torch.Tensor,
        candidate_mask: torch.Tensor,
        endpoint_ids: torch.Tensor,
        context_ids: torch.Tensor,
    ) -> dict:
        output = super().forward(
            source_tokens,
            candidate_tokens,
            source_mask,
            candidate_mask,
            torch.zeros_like(endpoint_ids),
            torch.zeros_like(context_ids),
        )
        output.update(
            {
                "condition_unk_only": True,
                "observed_endpoint_or_context_consumed": False,
                "scientific_control_only": True,
            }
        )
        return output


class BoundedLocalParameterMatchedUnrestrictedCritic(
    BoundedLocalGlobalExactPotentialCritic
):
    """Condition-aware non-integrable control for bounded-local A1 V2."""

    architecture_id = "R3X_A1_BOUNDED_LOCAL_PARAMETER_MATCHED_CONTROL_V2"

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        self.left_pair_projection = nn.Linear(
            cfg.d_model, cfg.pair_rank, bias=False
        )
        self.right_pair_projection = nn.Linear(
            cfg.d_model, cfg.pair_rank, bias=False
        )
        self.condition_pair_projection = nn.Linear(
            cfg.d_model, cfg.pair_rank, bias=False
        )
        self.pair_bias = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        source_tokens: torch.Tensor,
        candidate_tokens: torch.Tensor,
        source_mask: torch.Tensor,
        candidate_mask: torch.Tensor,
        endpoint_ids: torch.Tensor,
        context_ids: torch.Tensor,
    ) -> dict:
        (
            source,
            candidate,
            condition,
            source_potential,
            candidate_potential,
            local_uncertainty_summary,
        ) = (
            self._bounded_represent(
                source_tokens,
                candidate_tokens,
                source_mask,
                candidate_mask,
                endpoint_ids,
                context_ids,
            )
        )
        residual = (
            self.left_pair_projection(source)
            * self.right_pair_projection(candidate)
            * self.condition_pair_projection(condition)
        ).sum(dim=-1) / math.sqrt(self.cfg.pair_rank) + self.pair_bias
        mean = candidate_potential - source_potential + residual
        symmetric = torch.cat(
            [
                torch.abs(candidate - source),
                0.5 * (candidate + source),
                condition
                + self.local_uncertainty_projection(
                    local_uncertainty_summary.unsqueeze(-1)
                ),
            ],
            dim=-1,
        )
        logvar = torch.clamp(
            self.logvar_head(symmetric).squeeze(-1), -12.0, 8.0
        )
        return {
            "mean": mean,
            "logvar": logvar,
            "sign": mean,
            "rank": mean,
            "architecture_id": self.architecture_id,
            "sign_decision_rule": "RAW_PREDICTED_DELTA_MEAN_GE_ZERO",
            "potential_constraint": False,
            "bounded_local_branch_support": True,
            "overall_effect_support": "GLOBAL_PLUS_FINITE_RF_LOCAL_NOT_STRICTLY_BOUNDED",
            "whole_mean_bounded_support": False,
            "uncertainty_bounded_support": False,
            "global_branch_present": True,
            "local_receptive_field_radius": self.local_receptive_field_radius,
            "local_receptive_field_overlap_possible_through_distance": 2
            * self.local_receptive_field_radius,
            "strict_local_interaction_exclusion_minimum_distance": 2
            * self.local_receptive_field_radius
            + 1,
            "length_extension_invariant": False,
            "site_normalization": "SUM_OVER_SQRT_VISIBLE_LENGTH",
            "local_uncertainty_summary": "SWAP_SYMMETRIC_ABSOLUTE_SITE_ENERGY_DELTA",
            "condition_aware_nonintegrable_residual": True,
        }


class SiteAdditiveParameterMatchedUnrestrictedCritic(
    SiteAdditiveLocalGlobalExactPotentialCritic
):
    """A1-matched condition-aware non-integrable residual control."""

    architecture_id = "R3X_A1_SITE_ADDITIVE_PARAMETER_MATCHED_CONTROL_V1"

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        self.left_pair_projection = nn.Linear(cfg.d_model, cfg.pair_rank, bias=False)
        self.right_pair_projection = nn.Linear(cfg.d_model, cfg.pair_rank, bias=False)
        self.condition_pair_projection = nn.Linear(
            cfg.d_model, cfg.pair_rank, bias=False
        )
        self.pair_bias = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        source_tokens: torch.Tensor,
        candidate_tokens: torch.Tensor,
        source_mask: torch.Tensor,
        candidate_mask: torch.Tensor,
        endpoint_ids: torch.Tensor,
        context_ids: torch.Tensor,
    ) -> dict:
        source, candidate, condition, source_potential, candidate_potential = (
            self._local_global_represent(
                source_tokens,
                candidate_tokens,
                source_mask,
                candidate_mask,
                endpoint_ids,
                context_ids,
            )
        )
        residual = (
            self.left_pair_projection(source)
            * self.right_pair_projection(candidate)
            * self.condition_pair_projection(condition)
        ).sum(dim=-1) / math.sqrt(self.cfg.pair_rank) + self.pair_bias
        mean = candidate_potential - source_potential + residual
        symmetric = torch.cat(
            [
                torch.abs(candidate - source),
                0.5 * (candidate + source),
                condition,
            ],
            dim=-1,
        )
        logvar = torch.clamp(
            self.logvar_head(symmetric).squeeze(-1), -12.0, 8.0
        )
        return {
            "mean": mean,
            "logvar": logvar,
            "sign": mean,
            "rank": mean,
            "architecture_id": self.architecture_id,
            "sign_decision_rule": "RAW_PREDICTED_DELTA_MEAN_GE_ZERO",
            "potential_constraint": False,
            "local_site_branch": True,
            "site_normalization": "SUM_OVER_SQRT_VISIBLE_LENGTH",
            "condition_aware_nonintegrable_residual": True,
        }


class ParameterMatchedUnrestrictedSiamese(ExactPotentialCritic):
    """Historical A0-matched low-rank unconstrained pair residual."""

    architecture_id = "R3X_A_PARAMETER_MATCHED_UNRESTRICTED_SIAMESE_V1"

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        self.left_pair_projection = nn.Linear(cfg.d_model, cfg.pair_rank, bias=False)
        self.right_pair_projection = nn.Linear(cfg.d_model, cfg.pair_rank, bias=False)
        self.pair_bias = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        source_tokens: torch.Tensor,
        candidate_tokens: torch.Tensor,
        source_mask: torch.Tensor,
        candidate_mask: torch.Tensor,
        endpoint_ids: torch.Tensor,
        context_ids: torch.Tensor,
    ) -> dict:
        source, candidate, condition = self._represent(
            source_tokens, candidate_tokens, source_mask, candidate_mask,
            endpoint_ids, context_ids,
        )
        source_potential = self.potential_head(torch.cat([source, condition], dim=-1)).squeeze(-1)
        candidate_potential = self.potential_head(torch.cat([candidate, condition], dim=-1)).squeeze(-1)
        pair_residual = (
            self.left_pair_projection(source) * self.right_pair_projection(candidate)
        ).sum(dim=-1) / math.sqrt(self.cfg.pair_rank) + self.pair_bias
        mean = candidate_potential - source_potential + pair_residual
        symmetric = torch.cat([
            torch.abs(candidate - source), 0.5 * (candidate + source), condition,
        ], dim=-1)
        logvar = torch.clamp(self.logvar_head(symmetric).squeeze(-1), -12.0, 8.0)
        return {
            "mean": mean,
            "logvar": logvar,
            "sign": mean,
            "rank": mean,
            "architecture_id": self.architecture_id,
            "sign_decision_rule": "RAW_PREDICTED_DELTA_MEAN_GE_ZERO",
        }


class ConditionAwareParameterMatchedUnrestrictedSiamese(ExactPotentialCritic):
    """V2 A0 control with a condition-aware non-integrable residual."""

    architecture_id = "R3X_A_PARAMETER_MATCHED_CONDITION_AWARE_CONTROL_V2"

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        self.left_pair_projection = nn.Linear(cfg.d_model, cfg.pair_rank, bias=False)
        self.right_pair_projection = nn.Linear(cfg.d_model, cfg.pair_rank, bias=False)
        self.condition_pair_projection = nn.Linear(
            cfg.d_model, cfg.pair_rank, bias=False
        )
        self.pair_bias = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        source_tokens: torch.Tensor,
        candidate_tokens: torch.Tensor,
        source_mask: torch.Tensor,
        candidate_mask: torch.Tensor,
        endpoint_ids: torch.Tensor,
        context_ids: torch.Tensor,
    ) -> dict:
        source, candidate, condition = self._represent(
            source_tokens,
            candidate_tokens,
            source_mask,
            candidate_mask,
            endpoint_ids,
            context_ids,
        )
        source_potential = self.potential_head(
            torch.cat([source, condition], dim=-1)
        ).squeeze(-1)
        candidate_potential = self.potential_head(
            torch.cat([candidate, condition], dim=-1)
        ).squeeze(-1)
        residual = (
            self.left_pair_projection(source)
            * self.right_pair_projection(candidate)
            * self.condition_pair_projection(condition)
        ).sum(dim=-1) / math.sqrt(self.cfg.pair_rank) + self.pair_bias
        mean = candidate_potential - source_potential + residual
        symmetric = torch.cat(
            [
                torch.abs(candidate - source),
                0.5 * (candidate + source),
                condition,
            ],
            dim=-1,
        )
        logvar = torch.clamp(
            self.logvar_head(symmetric).squeeze(-1), -12.0, 8.0
        )
        return {
            "mean": mean,
            "logvar": logvar,
            "sign": mean,
            "rank": mean,
            "architecture_id": self.architecture_id,
            "sign_decision_rule": "RAW_PREDICTED_DELTA_MEAN_GE_ZERO",
            "potential_constraint": False,
            "condition_aware_nonintegrable_residual": True,
        }


class FullPairUnrestrictedSiamese(_ContextConditioning):
    """Historical pooled order-sensitive pair control with a shared encoder."""

    architecture_id = "R3X_FULL_PAIR_UNRESTRICTED_SIAMESE_V1"
    information_set = "PAIRED_SEQUENCE_NO_MEASURED_ANCHOR"

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        self.encoder = SharedSequenceEncoder(cfg)
        hidden = max(4, cfg.d_model // 3)
        self.pair_head = nn.Sequential(
            nn.Linear(6 * cfg.d_model, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        self.logvar_head = nn.Sequential(
            nn.Linear(3 * cfg.d_model, cfg.d_model),
            nn.SiLU(),
            nn.Linear(cfg.d_model, 1),
        )

    def forward(
        self,
        source_tokens: torch.Tensor,
        candidate_tokens: torch.Tensor,
        source_mask: torch.Tensor,
        candidate_mask: torch.Tensor,
        endpoint_ids: torch.Tensor,
        context_ids: torch.Tensor,
    ) -> dict:
        _, source = self.encoder(source_tokens, source_mask)
        _, candidate = self.encoder(candidate_tokens, candidate_mask)
        condition = self.condition(endpoint_ids, context_ids)
        ordered = torch.cat([
            source, candidate, candidate - source, torch.abs(candidate - source),
            candidate * source, condition,
        ], dim=-1)
        mean = self.pair_head(ordered).squeeze(-1)
        symmetric = torch.cat([
            torch.abs(candidate - source), 0.5 * (candidate + source), condition,
        ], dim=-1)
        logvar = torch.clamp(self.logvar_head(symmetric).squeeze(-1), -12.0, 8.0)
        return {
            "mean": mean,
            "logvar": logvar,
            "sign": mean,
            "rank": mean,
            "architecture_id": self.architecture_id,
            "sign_decision_rule": "RAW_PREDICTED_DELTA_MEAN_GE_ZERO",
            "potential_constraint": False,
        }


class EditAwareFullPairUnrestrictedCritic(FullPairUnrestrictedSiamese):
    """V2 comparator adding parameter-light changed-token pair pooling."""

    architecture_id = "R3X_FULL_PAIR_EDIT_AWARE_UNRESTRICTED_SIAMESE_V2"

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        hidden = max(4, cfg.d_model // 4)
        self.pair_head = nn.Sequential(
            nn.Linear(9 * cfg.d_model, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        source_tokens: torch.Tensor,
        candidate_tokens: torch.Tensor,
        source_mask: torch.Tensor,
        candidate_mask: torch.Tensor,
        endpoint_ids: torch.Tensor,
        context_ids: torch.Tensor,
    ) -> dict:
        if source_tokens.shape != candidate_tokens.shape or not torch.equal(
            source_mask, candidate_mask
        ):
            raise ValueError(
                "edit-aware FullPair requires aligned equal-length substitution tensors and masks"
            )
        source_hidden, source = self.encoder(source_tokens, source_mask)
        candidate_hidden, candidate = self.encoder(candidate_tokens, candidate_mask)
        changed = (source_tokens != candidate_tokens) & source_mask
        changed_counts = changed.sum(dim=1)
        if any(int(value) not in {0, 1, 3, 5} for value in changed_counts.tolist()):
            raise ValueError(
                "edit-aware FullPair requires primary changed-token count in {1,3,5}; "
                "zero is admitted only for identity diagnostics"
            )
        changed_float = changed.unsqueeze(-1).to(source_hidden.dtype)
        denominator = changed_float.sum(dim=1).clamp_min(1.0)
        changed_source = (source_hidden * changed_float).sum(dim=1) / denominator
        changed_candidate = (candidate_hidden * changed_float).sum(dim=1) / denominator
        condition = self.condition(endpoint_ids, context_ids)
        ordered = torch.cat(
            [
                source,
                candidate,
                candidate - source,
                torch.abs(candidate - source),
                candidate * source,
                changed_source,
                changed_candidate,
                changed_candidate - changed_source,
                condition,
            ],
            dim=-1,
        )
        mean = self.pair_head(ordered).squeeze(-1)
        symmetric = torch.cat(
            [
                torch.abs(candidate - source),
                0.5 * (candidate + source),
                condition,
            ],
            dim=-1,
        )
        logvar = torch.clamp(
            self.logvar_head(symmetric).squeeze(-1), -12.0, 8.0
        )
        return {
            "mean": mean,
            "logvar": logvar,
            "sign": mean,
            "rank": mean,
            "architecture_id": self.architecture_id,
            "sign_decision_rule": "RAW_PREDICTED_DELTA_MEAN_GE_ZERO",
            "potential_constraint": False,
            "changed_position_token_pooling": True,
            "token_level_cross_attention": False,
            "frozen_edit_budget": [1, 3, 5],
        }


class R2StyleNoExplicitStudyCrossAttentionCritic(_ContextConditioning):
    """R2-style comparator without an explicit study-ID model input.

    This is an adaptation, not a single-variable ablation of the historical
    SparseEditFormer.  Endpoint/context can still encode study-specific
    information, so absence of a semantic study proxy requires a separate,
    hash-bound data audit and is deliberately not asserted by this class.
    """

    architecture_id = "R3X_R2_STYLE_NO_EXPLICIT_STUDY_CROSS_ATTENTION_ADAPTATION_V1"
    information_set = "PAIRED_SEQUENCE_NO_MEASURED_ANCHOR"

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        self.encoder = SharedSequenceEncoder(cfg)
        self.cross_attention = nn.MultiheadAttention(
            cfg.d_model, cfg.n_heads, dropout=0.0, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(cfg.d_model)
        self.mean_head = nn.Sequential(
            nn.Linear(3 * cfg.d_model, cfg.d_model),
            nn.SiLU(),
            nn.Linear(cfg.d_model, 1),
        )
        self.logvar_head = nn.Sequential(
            nn.Linear(3 * cfg.d_model, cfg.d_model),
            nn.SiLU(),
            nn.Linear(cfg.d_model, 1),
        )

    def forward(
        self,
        source_tokens: torch.Tensor,
        candidate_tokens: torch.Tensor,
        source_mask: torch.Tensor,
        candidate_mask: torch.Tensor,
        endpoint_ids: torch.Tensor,
        context_ids: torch.Tensor,
    ) -> dict:
        if source_tokens.shape != candidate_tokens.shape or not torch.equal(
            source_mask, candidate_mask
        ):
            raise ValueError(
                "R2-style primary route requires aligned equal-length substitution tensors and masks"
            )
        source_hidden, source_pooled = self.encoder(source_tokens, source_mask)
        candidate_hidden, _ = self.encoder(candidate_tokens, candidate_mask)
        cross, _ = self.cross_attention(
            candidate_hidden,
            source_hidden,
            source_hidden,
            key_padding_mask=~source_mask,
            need_weights=False,
        )
        cross = self.cross_norm(candidate_hidden + cross)
        valid = candidate_mask.unsqueeze(-1).to(cross.dtype)
        cross_pooled = (cross * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        changed = ((source_tokens != candidate_tokens) & source_mask & candidate_mask).unsqueeze(-1)
        changed_float = changed.to(cross.dtype)
        difference = candidate_hidden - source_hidden
        edit_summary = (difference * changed_float).sum(dim=1) / changed_float.sum(dim=1).clamp_min(1.0)
        condition = self.condition(endpoint_ids, context_ids)
        representation = torch.cat([cross_pooled, edit_summary, condition], dim=-1)
        mean = self.mean_head(representation).squeeze(-1)
        uncertainty = torch.cat([
            torch.abs(cross_pooled - source_pooled),
            0.5 * (cross_pooled + source_pooled),
            condition,
        ], dim=-1)
        logvar = torch.clamp(self.logvar_head(uncertainty).squeeze(-1), -12.0, 8.0)
        return {
            "mean": mean,
            "logvar": logvar,
            "sign": mean,
            "rank": mean,
            "architecture_id": self.architecture_id,
            "sign_decision_rule": "RAW_PREDICTED_DELTA_MEAN_GE_ZERO",
            "explicit_study_id_input_or_embedding": False,
            "semantic_study_proxy_exclusion_established": False,
            "historical_r2_single_variable_ablation": False,
            "fresh_training_required": True,
        }


class SparseEditSetCritic(_ContextConditioning):
    """B additive operator; the legacy interaction constructor is locked.

    Interaction support must eventually be introduced as a new exact registry
    method/class after independently issued action-pair support evidence is
    qualified.  A caller-supplied mapping cannot unlock it through this class.
    """

    information_set = "PAIRED_SEQUENCE_NO_MEASURED_ANCHOR"
    feature_position_policy = "ACTUAL_EDIT_POSITION"
    position_coordinate_policy = "NORMALIZED_ACTUAL_OR_CONTROL_FEATURE_POSITION"
    source_hidden_feature_policy = "GATHER_ENCODER_HIDDEN_AT_FEATURE_POSITION"

    def __init__(
        self,
        cfg: ModelConfig,
        *,
        interaction: bool,
        action_support_policy: Mapping | None = None,
    ):
        if bool(interaction):
            raise ActionSupportHoldError(
                "NOT_RUN_ACTION_SUPPORT",
                "B interaction is not registered and cannot be unlocked by a "
                "caller-supplied action-support mapping; a future versioned exact "
                "interaction class and authority are required",
            )
        if action_support_policy is not None:
            raise ActionSupportHoldError(
                "NOT_RUN_ACTION_SUPPORT",
                "additive B does not consume an interaction action-support policy",
            )
        super().__init__(cfg)
        self.interaction = bool(interaction)
        self.action_support_status = "NOT_REQUIRED_FOR_ADDITIVE"
        self.architecture_id = "R3X_B_SPARSE_EDIT_SET_ADDITIVE_V1"
        self.encoder = SharedSequenceEncoder(cfg)
        edit_dim = max(4, cfg.d_model // 4)
        self.ref_embedding = nn.Embedding(cfg.vocab_size, edit_dim, padding_idx=0)
        self.alt_embedding = nn.Embedding(cfg.vocab_size, edit_dim, padding_idx=0)
        self.position_projection = nn.Sequential(nn.Linear(1, edit_dim), nn.SiLU())
        self.edit_token_mlp = nn.Sequential(
            nn.Linear(2 * cfg.d_model + 3 * edit_dim, cfg.d_model),
            nn.SiLU(),
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.SiLU(),
        )
        self.additive_head = nn.Linear(cfg.d_model, 1)
        if self.interaction:
            self.interaction_head = nn.Sequential(
                nn.Linear(3 * cfg.d_model, cfg.d_model),
                nn.SiLU(),
                nn.Linear(cfg.d_model, 1),
            )
        self.logvar_head = nn.Sequential(
            nn.Linear(3 * cfg.d_model, cfg.d_model),
            nn.SiLU(),
            nn.Linear(cfg.d_model, 1),
        )

    def _validate_edits(
        self,
        source_tokens: torch.Tensor,
        source_mask: torch.Tensor,
        positions: torch.Tensor,
        ref: torch.Tensor,
        alt: torch.Tensor,
        edit_mask: torch.Tensor,
        candidate_tokens: torch.Tensor | None,
    ) -> None:
        expected_shape = (source_tokens.shape[0], self.cfg.max_edits)
        if any(value.shape != expected_shape for value in (positions, ref, alt, edit_mask)):
            raise ValueError("edit tensors must have shape [batch,max_edits]")
        if edit_mask.dtype is not torch.bool:
            raise ValueError("edit mask must be boolean")
        counts = edit_mask.sum(dim=1)
        if any(int(value) not in {1, 3, 5} for value in counts.tolist()):
            raise ValueError("edit budget must be exactly one of {1,3,5}")
        safe_positions = positions.masked_fill(~edit_mask, 0)
        if torch.any(safe_positions < 0) or torch.any(safe_positions >= source_tokens.shape[1]):
            raise ValueError("edit position is outside the encoded source")
        visible = source_mask.gather(1, safe_positions)
        if not torch.all(visible[edit_mask]):
            raise ValueError("every edit position must be visible; truncation is prohibited")
        source_at_edit = source_tokens.gather(1, safe_positions)
        if not torch.equal(source_at_edit[edit_mask], ref[edit_mask]):
            raise ValueError("edit ref does not match the source token")
        if torch.any(ref[edit_mask] < 1) or torch.any(ref[edit_mask] >= self.cfg.vocab_size):
            raise ValueError("edit ref token is invalid")
        if torch.any(alt[edit_mask] < 1) or torch.any(alt[edit_mask] >= self.cfg.vocab_size):
            raise ValueError("edit alt token is invalid")
        if torch.any(ref[edit_mask] == alt[edit_mask]):
            raise ValueError("null substitutions are prohibited")
        for row in range(source_tokens.shape[0]):
            active = positions[row][edit_mask[row]].tolist()
            if len(set(active)) != len(active):
                raise ValueError("duplicate edit positions are prohibited")
        if candidate_tokens is not None:
            if candidate_tokens.shape != source_tokens.shape:
                raise ValueError("candidate replay tensor must align with source")
            expected = source_tokens.clone()
            for row in range(source_tokens.shape[0]):
                active = edit_mask[row]
                expected[row, positions[row, active]] = alt[row, active]
            if not torch.equal(expected[source_mask], candidate_tokens[source_mask]):
                raise ValueError("candidate replay does not match exact edit tokens")

    def _edit_feature_positions(
        self,
        source_tokens: torch.Tensor,
        source_mask: torch.Tensor,
        positions: torch.Tensor,
        ref: torch.Tensor,
        edit_mask: torch.Tensor,
    ) -> torch.Tensor:
        return positions.masked_fill(~edit_mask, 0)

    def _position_coordinates(
        self,
        normalized_position: torch.Tensor,
        edit_ref: torch.Tensor,
        edit_alt: torch.Tensor,
        edit_mask: torch.Tensor,
    ) -> torch.Tensor:
        return normalized_position

    def _prediction_edit_features(
        self,
        hidden: torch.Tensor,
        pooled: torch.Tensor,
        source_tokens: torch.Tensor,
        source_mask: torch.Tensor,
        edit_positions: torch.Tensor,
        edit_ref: torch.Tensor,
        edit_alt: torch.Tensor,
        edit_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return source-hidden and coordinate inputs used by the predictor."""

        feature_positions = self._edit_feature_positions(
            source_tokens, source_mask, edit_positions, edit_ref, edit_mask
        )
        if feature_positions.shape != edit_positions.shape:
            raise ValueError("edit feature positions must align with actual edits")
        if torch.any(feature_positions < 0) or torch.any(
            feature_positions >= source_tokens.shape[1]
        ):
            raise ValueError("edit feature position is outside the encoded source")
        feature_visible = source_mask.gather(1, feature_positions)
        if not torch.all(feature_visible[edit_mask]):
            raise ValueError("every edit feature position must be visible")
        feature_ref = source_tokens.gather(1, feature_positions)
        if not torch.equal(feature_ref[edit_mask], edit_ref[edit_mask]):
            raise ValueError("edit feature position is incompatible with the action ref")
        gather_index = feature_positions.unsqueeze(-1).expand(
            -1, -1, self.cfg.d_model
        )
        source_hidden = hidden.gather(1, gather_index)
        lengths = source_mask.sum(dim=1, keepdim=True).clamp_min(2).to(hidden.dtype)
        normalized_position = feature_positions.to(hidden.dtype).unsqueeze(-1) / (
            lengths.unsqueeze(-1) - 1.0
        )
        position_coordinates = self._position_coordinates(
            normalized_position, edit_ref, edit_alt, edit_mask
        )
        return source_hidden, position_coordinates

    def forward(
        self,
        source_tokens: torch.Tensor,
        source_mask: torch.Tensor,
        edit_positions: torch.Tensor,
        edit_ref: torch.Tensor,
        edit_alt: torch.Tensor,
        edit_mask: torch.Tensor,
        endpoint_ids: torch.Tensor,
        context_ids: torch.Tensor,
        *,
        candidate_tokens: torch.Tensor | None = None,
    ) -> dict:
        hidden, pooled = self.encoder(source_tokens, source_mask)
        self._validate_edits(
            source_tokens, source_mask, edit_positions, edit_ref, edit_alt,
            edit_mask, candidate_tokens,
        )
        condition = self.condition(endpoint_ids, context_ids)
        source_hidden, position_coordinates = self._prediction_edit_features(
            hidden,
            pooled,
            source_tokens,
            source_mask,
            edit_positions,
            edit_ref,
            edit_alt,
            edit_mask,
        )
        position_embedding = self.position_projection(position_coordinates)
        condition_tokens = condition.unsqueeze(1).expand(-1, self.cfg.max_edits, -1)
        edit_token = self.edit_token_mlp(torch.cat([
            source_hidden,
            self.ref_embedding(edit_ref),
            self.alt_embedding(edit_alt),
            position_embedding,
            condition_tokens,
        ], dim=-1))
        mask_float = edit_mask.unsqueeze(-1).to(edit_token.dtype)
        edit_token = edit_token * mask_float
        mean = (self.additive_head(edit_token).squeeze(-1) * edit_mask.to(edit_token.dtype)).sum(dim=1)
        if self.interaction:
            interaction = torch.zeros_like(mean)
            for left in range(self.cfg.max_edits):
                for right in range(left + 1, self.cfg.max_edits):
                    pair_mask = edit_mask[:, left] & edit_mask[:, right]
                    if not torch.any(pair_mask):
                        continue
                    a, b = edit_token[:, left], edit_token[:, right]
                    pair = torch.cat([a + b, torch.abs(a - b), a * b], dim=-1)
                    score = self.interaction_head(pair).squeeze(-1)
                    interaction = interaction + score * pair_mask.to(score.dtype)
            mean = mean + interaction
        count = edit_mask.sum(dim=1, keepdim=True).to(edit_token.dtype)
        edit_summary = edit_token.sum(dim=1) / count
        uncertainty = torch.cat([pooled, edit_summary, condition], dim=-1)
        logvar = torch.clamp(self.logvar_head(uncertainty).squeeze(-1), -12.0, 8.0)
        return {
            "mean": mean,
            "logvar": logvar,
            "sign": mean,
            "rank": mean,
            "architecture_id": self.architecture_id,
            "action_support_status": self.action_support_status,
            "candidate_dense_branch_used": False,
            "feature_position_policy": self.feature_position_policy,
            "position_coordinate_policy": self.position_coordinate_policy,
            "source_hidden_feature_policy": self.source_hidden_feature_policy,
            "sign_decision_rule": "RAW_PREDICTED_DELTA_MEAN_GE_ZERO",
        }


class ReplayDerivedSparseEditSetAdditiveCritic(SparseEditSetCritic):
    """B-additive with edit tensors derived only from exact pair replay."""

    architecture_id = "R3X_B_REPLAY_DERIVED_SPARSE_EDIT_SET_ADDITIVE_V1"

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg, interaction=False)
        self.architecture_id = type(self).architecture_id

    def forward(
        self,
        source_tokens: torch.Tensor,
        candidate_tokens: torch.Tensor,
        source_mask: torch.Tensor,
        candidate_mask: torch.Tensor,
        endpoint_ids: torch.Tensor,
        context_ids: torch.Tensor,
    ) -> dict:
        positions, ref, alt, edit_mask = derive_replay_edit_tensors(
            source_tokens,
            candidate_tokens,
            source_mask,
            candidate_mask,
            max_edits=self.cfg.max_edits,
        )
        output = super().forward(
            source_tokens,
            source_mask,
            positions,
            ref,
            alt,
            edit_mask,
            endpoint_ids,
            context_ids,
            candidate_tokens=candidate_tokens,
        )
        output.update(
            {
                "edit_tensors_derived_from_exact_pair_replay": True,
                "interaction_branch_present": False,
            }
        )
        return output


class ReplayDerivedSparseEditSetPositionAgnosticControl(
    ReplayDerivedSparseEditSetAdditiveCritic
):
    """Same-parameter B control with no edit-position prediction feature.

    Actual positions remain available only to derive and validate the exact
    replayed SUB action.  The prediction branch repeats the global pooled
    source representation for every edit and uses only ref/alt content in its
    one-dimensional matched projection branch.
    """

    architecture_id = "R3X_B_SPARSE_EDIT_SET_POSITION_AGNOSTIC_MATCHED_CONTROL_V2"
    feature_position_policy = "ACTUAL_EDIT_POSITION_EXCLUDED_FROM_PREDICTION"
    position_coordinate_policy = "REF_ALT_PSEUDOCOORDINATE_NO_ACTUAL_POSITION"
    source_hidden_feature_policy = (
        "GLOBAL_POOLED_SOURCE_REPEATED_PER_EDIT_NO_POSITION_GATHER"
    )

    def _prediction_edit_features(
        self,
        hidden: torch.Tensor,
        pooled: torch.Tensor,
        source_tokens: torch.Tensor,
        source_mask: torch.Tensor,
        edit_positions: torch.Tensor,
        edit_ref: torch.Tensor,
        edit_alt: torch.Tensor,
        edit_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # The four discarded tensors contain either position-indexed state or
        # the position index itself.  SparseEditSetCritic._validate_edits has
        # already used them to validate the exact action/candidate replay; none
        # enters this control's prediction features.
        del hidden, source_tokens, source_mask, edit_positions
        source_hidden = pooled.unsqueeze(1).expand(-1, self.cfg.max_edits, -1)
        position_placeholder = pooled.new_zeros((*edit_ref.shape, 1))
        position_coordinates = self._position_coordinates(
            position_placeholder, edit_ref, edit_alt, edit_mask
        )
        return source_hidden, position_coordinates

    def _position_coordinates(
        self,
        normalized_position: torch.Tensor,
        edit_ref: torch.Tensor,
        edit_alt: torch.Tensor,
        edit_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Keep the one-dimensional projection branch active without leaking
        # actual position.  Ref/alt are already admitted B inputs; this scalar
        # pseudo-coordinate changes with the action tokens and is masked for
        # padding, so raw and effective capacity are not padded by a dead zero
        # feature merely to manufacture a parameter-count match.
        scale = float(2 * (self.cfg.vocab_size - 1))
        pseudo = (
            edit_ref.to(normalized_position.dtype)
            + edit_alt.to(normalized_position.dtype)
        ).unsqueeze(-1) / scale
        return pseudo * edit_mask.unsqueeze(-1).to(pseudo.dtype)

    def forward(self, *args, **kwargs) -> dict:
        output = super().forward(*args, **kwargs)
        output.update(
            {
                "position_agnostic_matched_control": True,
                "actual_edit_position_consumed_by_prediction_branch": False,
                "actual_edit_position_usage": (
                    "EXACT_ACTION_AND_CANDIDATE_REPLAY_VALIDATION_ONLY"
                ),
                "global_pooled_source_repeated_per_edit": True,
                "scientific_control_only": True,
            }
        )
        return output


class ReplayDerivedSparseEditSetWrongPositionControl(
    ReplayDerivedSparseEditSetAdditiveCritic
):
    """Same-parameter B control gathering features at compatible wrong loci."""

    architecture_id = "R3X_B_SPARSE_EDIT_SET_WRONG_POSITION_CONTROL_V1"
    feature_position_policy = "COMPATIBLE_UNEDITED_WRONG_SOURCE_POSITION"

    def _edit_feature_positions(
        self,
        source_tokens: torch.Tensor,
        source_mask: torch.Tensor,
        positions: torch.Tensor,
        ref: torch.Tensor,
        edit_mask: torch.Tensor,
    ) -> torch.Tensor:
        return derive_wrong_local_feature_positions(
            source_tokens, source_mask, positions, ref, edit_mask
        )

    def forward(self, *args, **kwargs) -> dict:
        output = super().forward(*args, **kwargs)
        output.update(
            {
                "wrong_local_position_control": True,
                "actual_action_replay_changed": False,
                "scientific_control_only": True,
            }
        )
        return output


class ReplayDerivedSparseEditSetConditionUnknownControl(
    ReplayDerivedSparseEditSetAdditiveCritic
):
    """Same-parameter B condition ablation using only explicit UNK IDs."""

    architecture_id = "R3X_B_SPARSE_EDIT_SET_CONDITION_UNK_ONLY_V1"

    def forward(
        self,
        source_tokens: torch.Tensor,
        candidate_tokens: torch.Tensor,
        source_mask: torch.Tensor,
        candidate_mask: torch.Tensor,
        endpoint_ids: torch.Tensor,
        context_ids: torch.Tensor,
    ) -> dict:
        output = super().forward(
            source_tokens,
            candidate_tokens,
            source_mask,
            candidate_mask,
            torch.zeros_like(endpoint_ids),
            torch.zeros_like(context_ids),
        )
        output.update(
            {
                "condition_unk_only": True,
                "observed_endpoint_or_context_consumed": False,
                "scientific_control_only": True,
            }
        )
        return output
