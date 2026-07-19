"""Edit-Flow CTMC kernels adapted to mRNA (MEF).

This module re-implements the original Edit Flow discrete flow-matching math and
specialises it to the mRNA edit grammar (region-aware three-way coupling +
codon-lattice constraints). It is intentionally self-contained (only torch /
numpy) so the package runs offline.

CTMC / Edit-Flow loss math
--------------------------
Edit Flow models a **continuous-time Markov chain (CTMC)** over sequences whose
transitions are single edit operations: *insert* token ``a`` after a position,
*substitute* a position to token ``a``, or *delete* a position. The generator
(rate matrix) is parameterised by per-position, per-operation rates
``lambda_ins, lambda_sub, lambda_del >= 0`` and token distributions
``ins_probs, sub_probs``.

Conditional path. Given a source ``x0`` and target ``x1`` we compute an optimal
edit alignment (edit-distance DP) producing gap-padded, equal-length sequences
``z0, z1``. A time ``t ~ U[0,1]`` and a scheduler ``kappa(t)`` (CubicScheduler)
define the conditional bridge, sampled *independently per aligned position*::

    z_t^i ~ (1 - kappa(t)) * onehot(z0^i) + kappa(t) * onehot(z1^i)

The (gap-free) model input ``x_t`` is ``z_t`` with GAP tokens removed.

Training objective. For each operation type the loss is the CTMC flow-matching
Bregman/ELBO term used by the reference ``train_mix.py``::

    L_op = mean_b [ sum_i lambda_op(x_t)_i
                    - sum_i sum_a u*_op(z_t -> z1)_{i,a} * log lambda_op(x_t)_{i,a}
                      * sched_coeff(t) ]

with the target field ``u*`` given by :func:`make_ut_mask_from_z` and

    sched_coeff(t) = kappa'(t) / (1 - kappa(t)).

The first term is the total outgoing rate (mass leaving the current state); the
second is the cross-entropy pulling the predicted rate toward the ground-truth
edit that moves ``z_t`` onto ``z1``. Logs are clamped at ``min=-20`` and rates
pass through softplus for numerical stability.

Complexity
----------
* Optimal alignment DP: ``O(m * n)`` time / space per pair (m=len(x0),
  n=len(x1)); backtrack ``O(m + n)``. A batch costs ``sum_i O(m_i n_i)``.
* Coupling construction, gap removal and mask building are all linear in the
  padded batch size ``O(B * L)``.
* Codon-lattice masks are ``O(64)`` table lookups, precomputed once.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from .constants import (
    AA_TO_CODON_INDICES,
    BOS_TOKEN,
    CODON_INDEX_TO_AA,
    GAP_TOKEN,
    NUM_PHASES,
    NUM_REGIONS,
    PAD_TOKEN,
    PHASE_NONE,
    REGION_CDS,
    V,
    VOCAB_ALIGN_SIZE,
    VOCAB_MODEL_SIZE,
    codon_to_index,
    index_to_codon,
)
from .config import CouplingConfig
from .schema import MRNARecord

# Sentinel region id for gap / BOS / pad positions in z-space region tracks.
REGION_SENTINEL: int = NUM_REGIONS  # == 3


# ===========================================================================
# Scheduler + probability helpers (reused Edit-Flow math)
# ===========================================================================
class CubicScheduler:
    """Cubic interpolation scheduler ``kappa(t)`` (reused from Edit Flow).

    ``kappa(0)=0``, ``kappa(1)=1``; ``a``/``b`` shape the acceleration profile.
    """

    def __init__(self, a: float = 1.0, b: float = 1.0) -> None:
        self.a = a
        self.b = b

    def __call__(self, t: Tensor) -> Tensor:
        return (
            -2 * (t ** 3)
            + 3 * (t ** 2)
            + self.a * (t ** 3 - 2 * t ** 2 + t)
            + self.b * (t ** 3 - t ** 2)
        )

    def derivative(self, t: Tensor) -> Tensor:
        return (
            -6 * (t ** 2)
            + 6 * t
            + self.a * (3 * t ** 2 - 4 * t + 1)
            + self.b * (3 * t ** 2 - 2 * t)
        )


def x2prob(x: Tensor, vocab_size: int = VOCAB_ALIGN_SIZE) -> Tensor:
    """One-hot class-distribution representation of a token tensor."""
    return F.one_hot(x.long(), num_classes=vocab_size).float()


def sample_p(pt: Tensor, temperature: float = 1.0) -> Tensor:
    """Sample a token sequence from a class-distribution tensor ``[B, L, C]``."""
    b, l, _ = pt.shape
    flat = pt.reshape(b * l, -1)
    # guard degenerate rows so multinomial never sees an all-zero distribution.
    row_sums = flat.sum(dim=-1, keepdim=True)
    flat = torch.where(row_sums > 0, flat, torch.ones_like(flat))
    xt = torch.multinomial(flat / temperature, 1)
    return xt.reshape(b, l)


def sample_cond_pt(p0: Tensor, p1: Tensor, t: Tensor, kappa: CubicScheduler) -> Tensor:
    """Sample the conditional bridge ``z_t`` from endpoint one-hots.

    ``pt = (1 - kappa(t)) * p0 + kappa(t) * p1`` sampled per position.
    """
    t = t.reshape(-1, 1, 1)
    pt = (1 - kappa(t)) * p0 + kappa(t) * p1
    return sample_p(pt)


# ===========================================================================
# Optimal edit alignment (edit-distance DP) + gap utilities
# ===========================================================================
def _align_pair(seq_0: np.ndarray, seq_1: np.ndarray) -> Tuple[List[int], List[int]]:
    """Levenshtein DP + backtrack -> gap-padded aligned pair.

    Complexity: O(m*n) time/space, ``m=len(seq_0)``, ``n=len(seq_1)``.
    """
    m, n = len(seq_0), len(seq_1)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq_0[i - 1] == seq_1[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    aligned_0: List[int] = []
    aligned_1: List[int] = []
    i, j = m, n
    while i or j:
        if i and j and seq_0[i - 1] == seq_1[j - 1]:
            aligned_0.append(int(seq_0[i - 1]))
            aligned_1.append(int(seq_1[j - 1]))
            i, j = i - 1, j - 1
        elif i and j and dp[i][j] == dp[i - 1][j - 1] + 1:
            aligned_0.append(int(seq_0[i - 1]))
            aligned_1.append(int(seq_1[j - 1]))
            i, j = i - 1, j - 1
        elif i and dp[i][j] == dp[i - 1][j] + 1:
            aligned_0.append(int(seq_0[i - 1]))
            aligned_1.append(GAP_TOKEN)
            i -= 1
        else:
            aligned_0.append(GAP_TOKEN)
            aligned_1.append(int(seq_1[j - 1]))
            j -= 1
    return aligned_0[::-1], aligned_1[::-1]


def opt_align_xs_to_zs(x0: Tensor, x1: Tensor) -> Tuple[Tensor, Tensor]:
    """Optimal-alignment DP wrapper returning aligned z0, z1 tensors."""
    a0, a1 = _align_pair(x0.cpu().numpy(), x1.cpu().numpy())
    return (
        torch.tensor(a0, dtype=torch.long, device=x0.device),
        torch.tensor(a1, dtype=torch.long, device=x1.device),
    )


def rm_gap_tokens(z: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Remove GAP tokens from aligned rows, re-pad to a rectangular batch.

    Returns ``(x, x_pad_mask, z_gap_mask, z_pad_mask)``.
    """
    z_gap_mask = z == GAP_TOKEN
    z_pad_mask = z == PAD_TOKEN
    rows = [z[i][~z_gap_mask[i]] for i in range(z.shape[0])]
    x_max_len = max((len(r) for r in rows), default=1)
    x_max_len = max(x_max_len, 1)
    x = torch.stack(
        [F.pad(r, (0, x_max_len - len(r)), value=PAD_TOKEN) for r in rows], dim=0
    )
    x_pad_mask = x == PAD_TOKEN
    return x, x_pad_mask, z_gap_mask, z_pad_mask


def rm_gap_tokens_with_aux(
    z_t: Tensor, region_z: Tensor, phase_z: Tensor
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Gap-strip ``z_t`` while gathering the aligned region/phase tracks.

    Guarantees ``region_x`` / ``phase_x`` share ``x_t``'s length and padding, so
    the model always receives per-position region/phase for the *gap-free* input
    it actually consumes. Pads region with :data:`REGION_SENTINEL`, phase with
    :data:`PHASE_NONE`.
    """
    z_gap_mask = z_t == GAP_TOKEN
    z_pad_mask = z_t == PAD_TOKEN
    x_rows, r_rows, p_rows = [], [], []
    for i in range(z_t.shape[0]):
        keep = ~z_gap_mask[i]
        x_rows.append(z_t[i][keep])
        r_rows.append(region_z[i][keep])
        p_rows.append(phase_z[i][keep])
    x_max_len = max((len(r) for r in x_rows), default=1)
    x_max_len = max(x_max_len, 1)
    x = torch.stack([F.pad(r, (0, x_max_len - len(r)), value=PAD_TOKEN) for r in x_rows], 0)
    region_x = torch.stack(
        [F.pad(r, (0, x_max_len - len(r)), value=REGION_SENTINEL) for r in r_rows], 0
    )
    phase_x = torch.stack(
        [F.pad(r, (0, x_max_len - len(r)), value=PHASE_NONE) for r in p_rows], 0
    )
    x_pad_mask = x == PAD_TOKEN
    return x, x_pad_mask, z_gap_mask, z_pad_mask, region_x, phase_x


# ===========================================================================
# Ground-truth operation field + z-space reprojection (reused Edit-Flow math)
# ===========================================================================
def make_ut_mask_from_z(z_t: Tensor, z_1: Tensor, vocab_size: int = VOCAB_MODEL_SIZE) -> Tensor:
    """Build the ground-truth operation target ``u*`` in z-space.

    Layout of the last axis (size ``2*vocab_size + 1``):
    ``[0:V]`` insertion token, ``[V:2V]`` substitution token, ``[-1]`` deletion.
    PAD/GAP bookkeeping matches the reference implementation.
    """
    batch_size, z_seq_len = z_t.shape
    n_ops = 2 * vocab_size + 1
    z_neq = (z_t != z_1) & (z_t != PAD_TOKEN) & (z_1 != PAD_TOKEN)
    z_ins = (z_t == GAP_TOKEN) & (z_1 != GAP_TOKEN) & z_neq
    z_del = (z_t != GAP_TOKEN) & (z_1 == GAP_TOKEN) & z_neq
    z_sub = z_neq & ~z_ins & ~z_del

    u_mask = torch.zeros((batch_size, z_seq_len, n_ops), dtype=torch.bool, device=z_t.device)
    # Only route valid model-vocab targets (nucleotides/BOS/PAD < vocab_size).
    ins_ok = z_ins & (z_1 < vocab_size)
    sub_ok = z_sub & (z_1 < vocab_size)
    u_mask[ins_ok, z_1[ins_ok]] = True
    u_mask[sub_ok, z_1[sub_ok] + vocab_size] = True
    u_mask[:, :, -1][z_del] = True
    return u_mask


def fill_gap_tokens_with_repeats(x_ut: Tensor, z_gap_mask: Tensor, z_pad_mask: Tensor) -> Tensor:
    """Scatter x-space predictions back onto z-space positions for the loss."""
    batch_size = z_gap_mask.shape[0]
    x_seq_len = x_ut.shape[1]
    indices = (~z_gap_mask).cumsum(dim=1) - 1
    indices = indices.clamp(min=0, max=x_seq_len - 1)
    batch_indices = torch.arange(batch_size, device=x_ut.device).unsqueeze(1)
    result = x_ut[batch_indices, indices]
    result[z_pad_mask] = 0
    return result


def edit_flow_loss(
    out: Dict[str, Tensor],
    z_t: Tensor,
    z_1: Tensor,
    x_pad_mask: Tensor,
    z_gap_mask: Tensor,
    z_pad_mask: Tensor,
    t: Tensor,
    scheduler: "CubicScheduler",
    vocab_size: int = VOCAB_MODEL_SIZE,
    time_eps: float = 1e-4,
    sched_coeff_clip: float = 100.0,
) -> Dict[str, Tensor]:
    """Compute the Edit-Flow CTMC training loss (ins/sub/del) for mRNA.

    Faithful port of the reference ``train_mix.py`` objective with the model's
    head-output dict (``rates``/``ins_probs``/``sub_probs``). See the module
    docstring for the math. Returns ``{loss, loss_ins, loss_sub, loss_del}``.

    Numerical stability (spec requirement): rates are non-negative (softplus).
    The region-conditioned codon mask can drive a predicted rate to exactly 0
    while its ground-truth op-mask is set, which would make ``0*log(0)`` produce
    a NaN gradient. We therefore ``clamp_min`` the reprojected field to a tiny
    epsilon before ``log`` and then clamp the log at ``min=-20`` exactly as the
    reference does. For long public mRNAs under AMP, the scheduler coefficient
    ``kappa'(t)/(1-kappa(t))`` can explode as ``t -> 1``; ``time_eps`` and
    ``sched_coeff_clip`` implement the documented bounded-hazard variant.

    Complexity: O(B * Lz * vocab_size).
    """
    # Keep the loss in fp32 even when the forward pass uses AMP/fp16. Casts are
    # differentiable and prevent tiny clamp epsilons from underflowing.
    rates = out["rates"].float()
    ins_probs = out["ins_probs"].float()
    sub_probs = out["sub_probs"].float()
    lam_ins, lam_sub, lam_del = rates[:, :, 0], rates[:, :, 1], rates[:, :, 2]
    valid_x = (~x_pad_mask).to(dtype=rates.dtype)
    lam_ins = lam_ins * valid_x
    lam_sub = lam_sub * valid_x
    lam_del = lam_del * valid_x

    u_ins = lam_ins.unsqueeze(-1) * ins_probs
    u_sub = lam_sub.unsqueeze(-1) * sub_probs
    ux = torch.cat([u_ins, u_sub, lam_del.unsqueeze(-1)], dim=-1)
    uz = fill_gap_tokens_with_repeats(ux, z_gap_mask, z_pad_mask)

    uz_mask = make_ut_mask_from_z(z_t, z_1, vocab_size=vocab_size).to(dtype=uz.dtype)
    uz_mask = uz_mask * (~z_pad_mask).to(dtype=uz.dtype).unsqueeze(-1)

    t_safe = t.float().clamp(float(time_eps), 1.0 - float(time_eps))
    denom = (1 - scheduler(t_safe)).clamp_min(float(time_eps))
    sched_coeff = scheduler.derivative(t_safe) / denom
    sched_coeff = sched_coeff.clamp(max=float(sched_coeff_clip)).to(dtype=uz.dtype)
    log_uz = torch.clamp(uz.clamp_min(1e-20).log(), min=-20)
    ce = log_uz * uz_mask * sched_coeff.unsqueeze(-1)

    loss_ins = (lam_ins.sum(1) - ce[:, :, :vocab_size].sum((1, 2))).mean()
    loss_sub = (lam_sub.sum(1) - ce[:, :, vocab_size : 2 * vocab_size].sum((1, 2))).mean()
    loss_del = (lam_del.sum(1) - ce[:, :, -1].sum(1)).mean()
    loss = loss_ins + loss_sub + loss_del
    return {"loss": loss, "loss_ins": loss_ins, "loss_sub": loss_sub, "loss_del": loss_del}


# ===========================================================================
# Region / phase tracks aligned to z1
# ===========================================================================
def _region_phase_for_z1(
    z1_row: Tensor, region_seq: Sequence[int], phase_seq: Sequence[int]
) -> Tuple[List[int], List[int]]:
    """Scatter a record's per-nt region/phase onto the non-gap slots of ``z1``.

    Gap slots carry the previous region (forward-fill) with ``PHASE_NONE`` so the
    track stays smooth; positions before the first token use the sentinels.
    Complexity: O(len(z1_row)).
    """
    region_out: List[int] = []
    phase_out: List[int] = []
    ptr = 0
    last_region = REGION_SENTINEL
    for tok in z1_row.tolist():
        if tok == GAP_TOKEN:
            region_out.append(last_region)
            phase_out.append(PHASE_NONE)
        else:
            if ptr < len(region_seq):
                last_region = int(region_seq[ptr])
                region_out.append(last_region)
                phase_out.append(int(phase_seq[ptr]))
            else:
                region_out.append(REGION_SENTINEL)
                phase_out.append(PHASE_NONE)
            ptr += 1
    return region_out, phase_out


# ===========================================================================
# Three-way hybrid coupling
# ===========================================================================
@dataclass
class HybridBatch:
    """Output of :func:`make_hybrid_batch` (all tensors on the target device).

    All sequence tensors have a prepended ``BOS_TOKEN`` (Edit-Flow convention).

    * ``x0``/``x1``            : ``[B, Lx0+1]`` / ``[B, Lx1+1]`` gap-free tokens.
    * ``z0``/``z1``            : ``[B, Lz+1]`` aligned (gap-padded) tokens.
    * ``region_ids``/``phase_ids`` : ``[B, Lz+1]`` z-space tracks aligned to ``z1``.
    * ``t``                    : ``[B, 1]`` sampled times.
    * ``padding_mask``         : ``[B, Lx1+1]`` (``x1 == PAD``) coarse mask.
    * ``route``                : per-sample coupling name.
    """

    x0: Tensor
    x1: Tensor
    z0: Tensor
    z1: Tensor
    t: Tensor
    region_ids: Tensor
    phase_ids: Tensor
    padding_mask: Tensor
    route: List[str] = field(default_factory=list)


def _corrupt_tokens(tokens: List[int], sub_p: float, ins_p: float, del_p: float, rng: np.random.Generator) -> List[int]:
    """Random sub/ins/del corruption over the nucleotide vocabulary (0..V-1)."""
    out: List[int] = []
    for tok in tokens:
        if rng.random() < del_p:
            continue
        cur = tok
        if rng.random() < sub_p:
            cur = int(rng.integers(0, V))
        out.append(cur)
        if rng.random() < ins_p:
            out.append(int(rng.integers(0, V)))
    if not out:
        out = [int(rng.integers(0, V))]
    return out


def _choose_route(cfg: CouplingConfig, has_ortholog: bool, rng: np.random.Generator) -> str:
    """Sample a coupling route, renormalising over available routes."""
    routes = ["empty", "corruption"]
    weights = [max(cfg.empty_prob, 0.0), max(cfg.corruption_prob, 0.0)]
    if has_ortholog:
        routes.append("ortholog")
        weights.append(max(cfg.ortholog_prob, 0.0))
    total = sum(weights)
    if total <= 0:  # degenerate config -> fall back to empty growth.
        return "empty"
    probs = [w / total for w in weights]
    return str(rng.choice(routes, p=probs))


def make_hybrid_batch(
    records: Sequence[MRNARecord],
    cfg: CouplingConfig,
    device: torch.device,
    ortholog_map: Optional[Dict[str, List[MRNARecord]]] = None,
    seed: Optional[int] = None,
) -> HybridBatch:
    """Region-aware three-way hybrid coupling for mRNA Edit Flow.

    For every target record ``x1`` a source ``x0`` is drawn from one of three
    couplings, then optimally aligned to ``(z0, z1)``:

    1. **empty-growth**       -- ``x0 = []`` (pure generation from nothing).
    2. **corruption-refine**  -- ``x0`` = randomly sub/ins/del-corrupted ``x1``.
    3. **ortholog**           -- ``x0`` = a homologous transcript (evolution-aware
       editing); only offered when an ortholog is supplied in ``ortholog_map``,
       otherwise its probability mass is renormalised onto the other routes.

    Region/phase tracks are computed from the target record and carried in
    z-space aligned to ``z1`` (see :func:`_region_phase_for_z1`).

    Complexity: ``sum_i O(m_i * n_i)`` for the alignments (dominant term).
    """
    rng = np.random.default_rng(seed)
    ortholog_map = ortholog_map or {}
    x0_list, x1_list, z0_list, z1_list = [], [], [], []
    rz_list, pz_list, routes = [], [], []

    for rec in records:
        x1_tokens = rec.token_ids()
        region_seq = rec.region_ids()
        phase_seq = rec.codon_phases()
        _x1 = torch.tensor(x1_tokens, dtype=torch.long, device=device)

        orths = ortholog_map.get(rec.transcript_id, [])
        route = _choose_route(cfg, has_ortholog=len(orths) > 0, rng=rng)

        if route == "empty":
            _x0 = torch.empty((0,), dtype=torch.long, device=device)
        elif route == "ortholog":
            orth = orths[int(rng.integers(0, len(orths)))]
            o_tokens = orth.token_ids()
            if not o_tokens:  # empty ortholog -> degrade to corruption.
                o_tokens = _corrupt_tokens(x1_tokens, cfg.sub_prob, cfg.ins_prob, cfg.del_prob, rng)
                route = "corruption"
            _x0 = torch.tensor(o_tokens, dtype=torch.long, device=device)
        else:  # corruption
            c_tokens = _corrupt_tokens(x1_tokens, cfg.sub_prob, cfg.ins_prob, cfg.del_prob, rng)
            _x0 = torch.tensor(c_tokens, dtype=torch.long, device=device)

        _z0, _z1 = opt_align_xs_to_zs(_x0, _x1)
        r_z, p_z = _region_phase_for_z1(_z1, region_seq, phase_seq)

        x0_list.append(_x0)
        x1_list.append(_x1)
        z0_list.append(_z0)
        z1_list.append(_z1)
        rz_list.append(torch.tensor(r_z, dtype=torch.long, device=device))
        pz_list.append(torch.tensor(p_z, dtype=torch.long, device=device))
        routes.append(route)

    batch_size = len(records)
    x0_max = max((len(x) for x in x0_list), default=0)
    x1_max = max((len(x) for x in x1_list), default=1)
    z_max = max((len(z) for z in z1_list), default=1)

    def _pad_stack(rows, max_len, value):
        return torch.stack([F.pad(r, (0, max_len - len(r)), value=value) for r in rows], dim=0)

    x1_t = _pad_stack(x1_list, x1_max, PAD_TOKEN)
    if x0_max > 0:
        x0_t = _pad_stack(x0_list, x0_max, PAD_TOKEN)
    else:
        x0_t = torch.full((batch_size, 0), PAD_TOKEN, dtype=torch.long, device=device)
    z0_t = _pad_stack(z0_list, z_max, PAD_TOKEN)
    z1_t = _pad_stack(z1_list, z_max, PAD_TOKEN)
    rz_t = _pad_stack(rz_list, z_max, REGION_SENTINEL)
    pz_t = _pad_stack(pz_list, z_max, PHASE_NONE)

    # Prepend BOS (and matching sentinels for the region/phase tracks).
    x1_t = F.pad(x1_t, (1, 0), value=BOS_TOKEN)
    x0_t = F.pad(x0_t, (1, 0), value=BOS_TOKEN)
    z0_t = F.pad(z0_t, (1, 0), value=BOS_TOKEN)
    z1_t = F.pad(z1_t, (1, 0), value=BOS_TOKEN)
    rz_t = F.pad(rz_t, (1, 0), value=REGION_SENTINEL)
    pz_t = F.pad(pz_t, (1, 0), value=PHASE_NONE)

    t = torch.rand(batch_size, 1, device=device)
    padding_mask = x1_t == PAD_TOKEN
    return HybridBatch(
        x0=x0_t, x1=x1_t, z0=z0_t, z1=z1_t, t=t,
        region_ids=rz_t, phase_ids=pz_t, padding_mask=padding_mask, route=routes,
    )


# ===========================================================================
# Codon-lattice constraint utilities (mRNA-specific)
# ===========================================================================
def _build_synonymous_matrix() -> Tensor:
    """Precompute ``[64, 64]`` bool matrix: ``M[i, j]`` iff codon j ~ codon i.

    "~" means "encodes the same amino acid" (stop group included). Complexity:
    O(64^2) one-time.
    """
    mat = torch.zeros((64, 64), dtype=torch.bool)
    for i in range(64):
        aa = CODON_INDEX_TO_AA[i]
        for j in AA_TO_CODON_INDICES[aa]:
            mat[i, j] = True
    return mat


# Module-level constant tables (built once).
SYNONYMOUS_CODON_MATRIX: Tensor = _build_synonymous_matrix()
CODON_INDEX_TO_AA_ID: Dict[int, int] = {}
_AA_ORDER = sorted(set(CODON_INDEX_TO_AA.values()))
_AA_TO_AAID = {aa: k for k, aa in enumerate(_AA_ORDER)}
for _ci, _aa in CODON_INDEX_TO_AA.items():
    CODON_INDEX_TO_AA_ID[_ci] = _AA_TO_AAID[_aa]


def synonymous_codon_mask(aa: str) -> Tensor:
    """Boolean ``[64]`` mask: ``True`` at codon indices that encode ``aa``.

    Complexity: O(#synonyms) <= O(6).
    """
    mask = torch.zeros(64, dtype=torch.bool)
    for idx in AA_TO_CODON_INDICES.get(aa, []):
        mask[idx] = True
    return mask


def _build_allowed_nt_sub_table() -> Tensor:
    """Precompute ``[64, 3, 4]`` bool table of frame-safe single-nt substitutions.

    ``T[c, p, n]`` is ``True`` iff replacing nucleotide at codon position ``p``
    (0/1/2) of codon ``c`` with nucleotide ``n`` (A/C/G/U -> 0..3) yields a codon
    that is *synonymous* to ``c`` (encodes the same amino acid). The current
    nucleotide is always allowed (identity is trivially synonymous).

    This is the nucleotide-resolution projection of the codon lattice used to
    mask the substitution head's logits inside CDS. Complexity: O(64*3*4).
    """
    table = torch.zeros((64, 3, 4), dtype=torch.bool)
    for c in range(64):
        aa = CODON_INDEX_TO_AA[c]
        base = [(c // 16) % 4, (c // 4) % 4, c % 4]
        for p in range(3):
            for n in range(4):
                cand = list(base)
                cand[p] = n
                cand_idx = cand[0] * 16 + cand[1] * 4 + cand[2]
                if CODON_INDEX_TO_AA[cand_idx] == aa:
                    table[c, p, n] = True
    return table


# ``[64, 3, 4]`` frame-safe single-nt substitution table (built once).
ALLOWED_NT_SUB_TABLE: Tensor = _build_allowed_nt_sub_table()


def synonymous_nt_sub_mask(
    token_ids: Tensor, region_ids: Tensor, phase_ids: Tensor
) -> Tensor:
    """Per-position allowed substitution nucleotides ``[B, L, 4]`` (bool).

    * UTR / non-CDS positions: all four nucleotides allowed (free nt-level sub).
    * CDS positions with a *complete* in-frame codon: only nucleotides that keep
      the codon synonymous (via :data:`ALLOWED_NT_SUB_TABLE`).
    * CDS positions with an incomplete codon (near a boundary): restricted to the
      current nucleotide (identity) so protein content can never be altered.

    ``token_ids``/``region_ids``/``phase_ids`` are x-space tensors of shape
    ``[B, L]`` (BOS prepended). Phase encodes the codon offset so
    ``codon_start = pos - phase`` recovers each codon regardless of the 5'UTR
    length. Complexity: O(B * L).
    """
    b, length = token_ids.shape
    device = token_ids.device
    is_cds = region_ids == REGION_CDS
    phase = phase_ids.clamp(0, 2)
    pos = torch.arange(length, device=device).unsqueeze(0).expand(b, -1)
    codon_start = pos - phase  # [B, L]

    i0 = codon_start.clamp(0, length - 1)
    i1 = (codon_start + 1).clamp(0, length - 1)
    i2 = (codon_start + 2).clamp(0, length - 1)
    nt = token_ids.clamp(0, V - 1)
    n0 = torch.gather(nt, 1, i0)
    n1 = torch.gather(nt, 1, i1)
    n2 = torch.gather(nt, 1, i2)
    codon_idx = (n0 * 16 + n1 * 4 + n2).clamp(0, 63)  # [B, L]

    table = ALLOWED_NT_SUB_TABLE.to(device)  # [64, 3, 4]
    # gather allowed-nt row for (codon_idx, phase): -> [B, L, 4]
    flat = table.reshape(64 * 3, 4)
    lut_idx = (codon_idx * 3 + phase).reshape(-1)
    allowed_cds = flat[lut_idx].reshape(b, length, 4)

    # completeness: whole codon inside sequence and all 3 nt are CDS.
    complete = (codon_start >= 0) & (codon_start + 2 < length)
    cds_all3 = (
        torch.gather(is_cds, 1, i0)
        & torch.gather(is_cds, 1, i1)
        & torch.gather(is_cds, 1, i2)
    )
    complete = complete & cds_all3

    all_true = torch.ones((b, length, 4), dtype=torch.bool, device=device)
    identity = F.one_hot(nt, num_classes=4).bool()  # only current nt

    out = torch.where(is_cds.unsqueeze(-1), identity, all_true)
    out = torch.where((is_cds & complete).unsqueeze(-1), allowed_cds, out)
    return out


def synonymous_mask_for_codon_index(codon_idx: Tensor) -> Tensor:
    """Vectorised synonymous mask for a tensor of codon indices.

    ``codon_idx``: any shape of longs in ``[0, 64)``.
    Returns bool tensor ``codon_idx.shape + (64,)`` where the last axis marks the
    allowed (synonymous) target codons. Complexity: O(N * 64).
    """
    mat = SYNONYMOUS_CODON_MATRIX.to(codon_idx.device)
    return mat[codon_idx.clamp(0, 63)]


def tokens_to_codon_indices(nt_tokens: Tensor) -> Tensor:
    """Map contiguous nt-token triplets ``[..., 3L]`` -> codon indices ``[..., L]``.

    Uses base-4 place value (matches :func:`constants.codon_to_index`). Non
    ``A/C/G/U`` tokens (BOS/PAD/GAP) are clamped so the call never errors; callers
    should only trust CDS positions. Complexity: O(N).
    """
    t = nt_tokens.clamp(0, V - 1)
    if t.shape[-1] % 3 != 0:
        raise ValueError("last dim must be a multiple of 3 to form codons")
    t = t.reshape(*t.shape[:-1], -1, 3)
    return t[..., 0] * 16 + t[..., 1] * 4 + t[..., 2]


def codon_index_to_tokens(codon_idx: Tensor) -> Tensor:
    """Inverse of :func:`tokens_to_codon_indices`: ``[..., L]`` -> ``[..., 3L]``."""
    c = codon_idx.clamp(0, 63)
    n0 = (c // 16) % 4
    n1 = (c // 4) % 4
    n2 = c % 4
    stacked = torch.stack([n0, n1, n2], dim=-1)  # [..., L, 3]
    return stacked.reshape(*codon_idx.shape[:-1], -1)


def cds_position_mask(region_ids: Tensor) -> Tensor:
    """Boolean mask of CDS positions (``region == REGION_CDS``)."""
    return region_ids == REGION_CDS


def build_indel_forbid_mask(region_ids: Tensor, codon_indel: bool) -> Tensor:
    """Positions where nt-level insert/delete must be forbidden.

    When ``codon_indel`` is ``False`` all CDS positions forbid indels (frame
    lock). When ``True`` nt-level indels are still forbidden inside CDS (only
    whole-codon indels are allowed, handled by the operator), so the per-nt
    forbid mask is identical. UTR positions never forbid indels.
    Returns bool ``[B, L]`` (``True`` = forbid nt indel here).
    """
    return cds_position_mask(region_ids)


__all__ = [
    "CubicScheduler", "x2prob", "sample_p", "sample_cond_pt",
    "opt_align_xs_to_zs", "rm_gap_tokens", "rm_gap_tokens_with_aux",
    "make_ut_mask_from_z", "fill_gap_tokens_with_repeats", "edit_flow_loss",
    "HybridBatch", "make_hybrid_batch",
    "REGION_SENTINEL",
    "SYNONYMOUS_CODON_MATRIX", "synonymous_codon_mask",
    "synonymous_mask_for_codon_index", "tokens_to_codon_indices",
    "codon_index_to_tokens", "cds_position_mask", "build_indel_forbid_mask",
    "ALLOWED_NT_SUB_TABLE", "synonymous_nt_sub_mask",
]
