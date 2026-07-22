"""P2-01: Counterfactual cross-region synergy panel v2 (multi-region oracle).

Replaces the v1 LocalTranslationOracle with MultiRegionOracle (P1-04 CNN-50mer
ensemble + CAI + 3'UTR stability + cross-region coupling terms).

Runs 8 arms per wild-type (vs v1's 5 arms) to enable region-pair decomposition:
  1. wild_type      : no edits
  2. single_5utr    : 5'UTR only
  3. single_cds     : CDS only
  4. single_3utr    : 3'UTR only
  5. pair_5_cds     : 5'UTR + CDS (no 3'UTR)
  6. pair_c_3       : CDS + 3'UTR (no 5'UTR)
  7. pair_5_3       : 5'UTR + 3'UTR (no CDS)
  8. joint          : all regions

Synergy scores:
  - syn_sum  = delta_joint - (delta_5 + delta_c + delta_3)          (3-region)
  - syn_5c   = delta_{5+c} - (delta_5 + delta_c)                    (5'UTR x CDS)
  - syn_c3   = delta_{c+3} - (delta_c + delta_3)                    (CDS x 3'UTR)
  - syn_53   = delta_{5+3} - (delta_5 + delta_3)                    (5'UTR x 3'UTR)
  - syn_5c3  = delta_joint - (delta_5 + delta_c + delta_3 + syn_5c + syn_c3 + syn_53)  (triple/3-way)

All ``improves TE/stability/expression`` claims are ``predicted/internal proxy``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from mrna_editflow.core.constants import REGION_3UTR, REGION_5UTR, REGION_CDS
from mrna_editflow.core.schema import MRNARecord
from mrna_editflow.rl.action_space import apply_action, build_legal_action_mask
from mrna_editflow.rl.policy import Policy, PolicyConfig
from mrna_editflow.rl.tiny_mdp import TinyMDP, TinyTrainableModel, Trajectory

# Import MultiRegionOracle from eval/multi_region_oracle.py.
from mrna_editflow.eval.multi_region_oracle import (
    MultiRegionOracle,
    MultiRegionOracleConfig,
    build_default_multi_region_oracle,
)

# Reuse the original policy_sample_trajectory (verified interface).
_SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)
from run_counterfactual_panel import policy_sample_trajectory  # noqa: E402


# ---------------------------------------------------------------------------
# Multi-region action mask (allows a SET of regions)
# ---------------------------------------------------------------------------

def build_multi_region_mask(
    record: MRNARecord,
    device: torch.device,
    allowed_regions: Optional[Sequence[int]],
    *,
    codon_indel: bool = False,
):
    """Build an action mask allowing edits in a SET of regions.

    If ``allowed_regions is None``, all regions are allowed (joint).
    """
    full_mask = build_legal_action_mask(
        record, device, codon_indel=codon_indel, allow_identity_sub=False
    )
    if allowed_regions is None:
        return full_mask

    region_ids = torch.tensor(
        record.region_ids(), dtype=torch.long, device=device
    )  # [L]
    in_region = torch.zeros_like(region_ids, dtype=torch.bool)
    for r in allowed_regions:
        in_region = in_region | (region_ids == r)

    full_mask.ins_mask = full_mask.ins_mask & in_region.unsqueeze(-1)
    full_mask.sub_mask = full_mask.sub_mask & in_region.unsqueeze(-1)
    full_mask.del_mask = full_mask.del_mask & in_region
    return full_mask


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ArmResultV2:
    """Result of one arm for one wild-type."""
    arm_name: str
    allowed_regions: Optional[List[int]]  # None=joint, [0]=5UTR, etc.
    n_edits: int
    final_seq: str
    oracle_score: Dict[str, Any]


@dataclass
class WildTypePanelV2:
    """8 arms for one wild-type record."""
    transcript_id: str
    wild_type: ArmResultV2
    single_5utr: ArmResultV2
    single_cds: ArmResultV2
    single_3utr: ArmResultV2
    pair_5_cds: ArmResultV2
    pair_c_3: ArmResultV2
    pair_5_3: ArmResultV2
    joint: ArmResultV2
    synergy_scores: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Oracle-based MDP (uses MultiRegionOracle)
# ---------------------------------------------------------------------------

class OracleMDPV2(TinyMDP):
    """TinyMDP with MultiRegionOracle reward.

    Reward = oracle_score(final_state) - oracle_score(initial_state).
    Uses ``ensemble_te`` as the scalar reward signal (bounded [0, 1]).
    """

    def __init__(
        self,
        initial_record: MRNARecord,
        oracle: MultiRegionOracle,
        max_steps: int = 8,
        gamma: float = 1.0,
    ) -> None:
        super().__init__(
            target_seq=initial_record.seq,
            initial_record=initial_record,
            max_steps=max_steps,
            stop_bonus=0.0,
            target_bonus=0.0,
            gamma=gamma,
        )
        self.oracle = oracle
        self._baseline_score = self._score(initial_record)

    def _score(self, record: MRNARecord) -> float:
        s = self.oracle.score_record(record)
        return float(s.get("ensemble_te", 0.0))

    def reward(self, state, action, next_state, step):  # type: ignore[override]
        is_terminal = action.is_stop() or (step + 1) >= self.max_steps
        if not is_terminal:
            return 0.0
        return self._score(next_state) - self._baseline_score


# ---------------------------------------------------------------------------
# Rollout collection
# ---------------------------------------------------------------------------

def _collect_arm_v2(
    policy: Policy,
    mdp: OracleMDPV2,
    allowed_regions: Optional[Sequence[int]],
    generator: torch.Generator,
) -> Tuple[Trajectory, MRNARecord]:
    """Collect one rollout, optionally restricted to a set of regions."""
    original_mask_fn = policy.legal_action_mask
    if allowed_regions is not None:
        def restricted_mask(rec, _rr=tuple(allowed_regions)):
            return build_multi_region_mask(
                rec, policy.device, _rr,
                codon_indel=policy.cfg.codon_indel,
            )
        policy.legal_action_mask = restricted_mask  # type: ignore[assignment]

    try:
        traj = policy_sample_trajectory(policy, mdp, generator)
    finally:
        policy.legal_action_mask = original_mask_fn  # type: ignore[assignment]

    record = mdp.initial_state()
    for t in traj.transitions:
        record = apply_action(record, t.action)
    return traj, record


# ---------------------------------------------------------------------------
# Panel execution
# ---------------------------------------------------------------------------

# 8 arms: (name, allowed_regions)
ARM_SPECS: List[Tuple[str, Optional[List[int]]]] = [
    ("single_5utr", [REGION_5UTR]),
    ("single_cds", [REGION_CDS]),
    ("single_3utr", [REGION_3UTR]),
    ("pair_5_cds", [REGION_5UTR, REGION_CDS]),
    ("pair_c_3", [REGION_CDS, REGION_3UTR]),
    ("pair_5_3", [REGION_5UTR, REGION_3UTR]),
    ("joint", None),
]


def run_panel_for_wild_type_v2(
    record: MRNARecord,
    oracle: MultiRegionOracle,
    max_steps: int,
    seed: int,
    device: torch.device,
) -> WildTypePanelV2:
    """Run the 8-arm panel for one wild-type record."""
    model = TinyTrainableModel(vocab_dim=4, hidden=16, rates_init=0.5)
    model.to(device)
    backbone = type(
        "B", (), {"out_dim": 0, "freeze": lambda self: None, "to": lambda self, d: self}
    )()
    cfg = PolicyConfig(
        stop_rate_strategy="constant",
        stop_rate_value=1.0,
        temperature=1.0,
        time_step=0.5,
        codon_indel=False,
    )
    policy = Policy(model=model, backbone=backbone, cfg=cfg, device=device)
    mdp = OracleMDPV2(
        initial_record=record,
        oracle=oracle,
        max_steps=max_steps,
        gamma=1.0,
    )

    # Wild-type arm.
    wt_score = oracle.score_record(record)
    wt_arm = ArmResultV2(
        arm_name="wild_type",
        allowed_regions=None,
        n_edits=0,
        final_seq=record.seq,
        oracle_score=wt_score,
    )

    # 7 rollout arms.
    arm_results: Dict[str, ArmResultV2] = {}
    for arm_name, allowed_regions in ARM_SPECS:
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed + hash(arm_name) % (2**31 - 1)))
        traj, final_record = _collect_arm_v2(policy, mdp, allowed_regions, gen)
        n_edits = sum(1 for t in traj.transitions if not t.action.is_stop())
        score = oracle.score_record(final_record)
        arm_results[arm_name] = ArmResultV2(
            arm_name=arm_name,
            allowed_regions=list(allowed_regions) if allowed_regions is not None else None,
            n_edits=n_edits,
            final_seq=final_record.seq,
            oracle_score=score,
        )

    panel = WildTypePanelV2(
        transcript_id=record.transcript_id,
        wild_type=wt_arm,
        single_5utr=arm_results["single_5utr"],
        single_cds=arm_results["single_cds"],
        single_3utr=arm_results["single_3utr"],
        pair_5_cds=arm_results["pair_5_cds"],
        pair_c_3=arm_results["pair_c_3"],
        pair_5_3=arm_results["pair_5_3"],
        joint=arm_results["joint"],
    )

    # Compute synergy scores using ensemble_te as scalar reward.
    r_wt = float(wt_score.get("ensemble_te", 0.0))
    r_5 = float(arm_results["single_5utr"].oracle_score.get("ensemble_te", 0.0))
    r_c = float(arm_results["single_cds"].oracle_score.get("ensemble_te", 0.0))
    r_3 = float(arm_results["single_3utr"].oracle_score.get("ensemble_te", 0.0))
    r_5c = float(arm_results["pair_5_cds"].oracle_score.get("ensemble_te", 0.0))
    r_c3 = float(arm_results["pair_c_3"].oracle_score.get("ensemble_te", 0.0))
    r_53 = float(arm_results["pair_5_3"].oracle_score.get("ensemble_te", 0.0))
    r_j = float(arm_results["joint"].oracle_score.get("ensemble_te", 0.0))

    # Deltas (improvement over wild-type).
    d_5 = r_5 - r_wt
    d_c = r_c - r_wt
    d_3 = r_3 - r_wt
    d_5c = r_5c - r_wt
    d_c3 = r_c3 - r_wt
    d_53 = r_53 - r_wt
    d_j = r_j - r_wt

    # Synergy scores.
    # 3-region: syn_sum = delta_joint - (delta_5 + delta_c + delta_3)
    syn_sum = d_j - (d_5 + d_c + d_3)
    # Pairwise: syn_5c = delta_{5+c} - (delta_5 + delta_c)
    syn_5c = d_5c - (d_5 + d_c)
    syn_c3 = d_c3 - (d_c + d_3)
    syn_53 = d_53 - (d_5 + d_3)
    # Triple (3-way interaction): residual after removing main + pairwise.
    # syn_5c3 = delta_joint - (delta_5 + delta_c + delta_3 + syn_5c + syn_c3 + syn_53)
    syn_5c3 = d_j - (d_5 + d_c + d_3 + syn_5c + syn_c3 + syn_53)

    panel.synergy_scores = {
        # Absolute scores.
        "r_wild_type": r_wt,
        "r_single_5utr_abs": r_5,
        "r_single_cds_abs": r_c,
        "r_single_3utr_abs": r_3,
        "r_pair_5_cds_abs": r_5c,
        "r_pair_c_3_abs": r_c3,
        "r_pair_5_3_abs": r_53,
        "r_joint_abs": r_j,
        # Deltas.
        "delta_5utr_vs_wt": d_5,
        "delta_cds_vs_wt": d_c,
        "delta_3utr_vs_wt": d_3,
        "delta_pair_5_cds_vs_wt": d_5c,
        "delta_pair_c_3_vs_wt": d_c3,
        "delta_pair_5_3_vs_wt": d_53,
        "delta_joint_vs_wt": d_j,
        # 3-region synergy.
        "syn_sum": syn_sum,
        "syn_mean": d_j - (d_5 + d_c + d_3) / 3.0,
        "syn_best": d_j - max(d_5, d_c, d_3),
        "syn_vs_wt": d_j,
        # Pairwise synergy (region-pair decomposition).
        "syn_5c": syn_5c,  # 5'UTR x CDS
        "syn_c3": syn_c3,  # CDS x 3'UTR
        "syn_53": syn_53,  # 5'UTR x 3'UTR
        # Triple (3-way) interaction.
        "syn_5c3": syn_5c3,
    }
    return panel


def panel_to_dict_v2(panel: WildTypePanelV2) -> Dict[str, Any]:
    """Serialize panel to dict (JSON-safe)."""
    def _arm(a: ArmResultV2) -> Dict[str, Any]:
        # Deep-copy oracle_score, ensure JSON-serializable.
        score = dict(a.oracle_score)
        # Remove nested weights dict (keep top-level only for size).
        if "weights" in score:
            score["weights"] = "omitted"
        return {
            "arm_name": a.arm_name,
            "allowed_regions": a.allowed_regions,
            "n_edits": a.n_edits,
            "final_seq_len": len(a.final_seq),
            "oracle_score": score,
        }
    return {
        "transcript_id": panel.transcript_id,
        "wild_type": _arm(panel.wild_type),
        "single_5utr": _arm(panel.single_5utr),
        "single_cds": _arm(panel.single_cds),
        "single_3utr": _arm(panel.single_3utr),
        "pair_5_cds": _arm(panel.pair_5_cds),
        "pair_c_3": _arm(panel.pair_c_3),
        "pair_5_3": _arm(panel.pair_5_3),
        "joint": _arm(panel.joint),
        "synergy_scores": panel.synergy_scores,
    }


# ---------------------------------------------------------------------------
# Records loading
# ---------------------------------------------------------------------------

def load_wild_type_records_v2(
    path: str,
    max_utr_len: int = 500,
    max_cds_len: int = 3000,
    max_3utr_len: int = 1000,
) -> List[MRNARecord]:
    """Load wild-type records from JSONL.

    Filters: 5'UTR in [50, max_utr_len], CDS multiple of 3 in [100, max_cds_len],
    3'UTR in [50, max_3utr_len], no ambiguous bases.
    """
    records: List[MRNARecord] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            try:
                rec = MRNARecord.from_dict(d)
            except Exception:
                continue
            # Length filters.
            if not (50 <= len(rec.five_utr) <= max_utr_len):
                continue
            if len(rec.cds) % 3 != 0 or not (100 <= len(rec.cds) <= max_cds_len):
                continue
            if not (50 <= len(rec.three_utr) <= max_3utr_len):
                continue
            # No ambiguous bases.
            full = rec.seq.upper()
            if any(c not in "ACGU" for c in full):
                continue
            records.append(rec)
    return records


# ---------------------------------------------------------------------------
# Main (serial)
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="P2-01 counterfactual synergy panel v2 (multi-region oracle)"
    )
    parser.add_argument(
        "--records",
        default="data/processed/gencode_human_transcripts.records.jsonl",
    )
    parser.add_argument("--n-wild-types", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--max-utr-len", type=int, default=500)
    parser.add_argument("--max-cds-len", type=int, default=3000)
    parser.add_argument("--max-3utr-len", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-cnn", action="store_true",
                        help="Skip CNN ensemble (fast testing).")
    parser.add_argument("--repo-root", default=_REPO_ROOT)
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    print(f"[panel_v2] Loading records from {args.records}")
    records = load_wild_type_records_v2(
        args.records,
        max_utr_len=args.max_utr_len,
        max_cds_len=args.max_cds_len,
        max_3utr_len=args.max_3utr_len,
    )
    print(f"[panel_v2] {len(records)} valid records after filters")

    if len(records) < args.n_wild_types:
        print(f"[panel_v2] Warning: only {len(records)} records, using all.")
        args.n_wild_types = len(records)
    records = records[: args.n_wild_types]

    print(f"[panel_v2] Building MultiRegionOracle (skip_cnn={args.skip_cnn})...")
    oracle = build_default_multi_region_oracle(
        repo_root=args.repo_root,
        device=args.device,
        skip_cnn=args.skip_cnn,
    )
    print(f"[panel_v2] Oracle ready. Running {len(records)} wild-types x 8 arms...")

    t0 = time.time()
    panels: List[WildTypePanelV2] = []
    for i, record in enumerate(records):
        seed_i = args.seed + i * 1000
        torch.manual_seed(seed_i)
        panel = run_panel_for_wild_type_v2(
            record=record,
            oracle=oracle,
            max_steps=args.max_steps,
            seed=seed_i,
            device=device,
        )
        panels.append(panel)
        if (i + 1) <= 5 or (i + 1) % 50 == 0 or (i + 1) == len(records):
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 1e-6)
            eta = (len(records) - i - 1) / max(rate, 1e-6)
            syn = panel.synergy_scores["syn_sum"]
            print(f"[panel_v2] {i+1}/{len(records)} "
                  f"syn_sum={syn:+.6f} ({elapsed:.1f}s, {rate:.1f}/s, ETA {eta:.0f}s)")

    # Save panels.
    output = {
        "config": {
            "records_path": args.records,
            "n_wild_types": args.n_wild_types,
            "max_steps": args.max_steps,
            "max_utr_len": args.max_utr_len,
            "max_cds_len": args.max_cds_len,
            "max_3utr_len": args.max_3utr_len,
            "seed": args.seed,
            "skip_cnn": args.skip_cnn,
            "oracle_version": "multi_region_v2_p2_01",
            "n_arms": 8,
        },
        "panels": [panel_to_dict_v2(p) for p in panels],
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"[panel_v2] Wrote {args.output} ({len(panels)} panels)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
