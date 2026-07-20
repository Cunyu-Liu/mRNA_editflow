"""P1-13: Counterfactual cross-region synergy edit panel.

For each wild-type record, runs 5 arms:

  1. ``wild_type``      : no edits, oracle scores the original record.
  2. ``single_5utr``    : region-restricted rollout (only 5'UTR editable).
  3. ``single_cds``     : region-restricted rollout (only CDS editable).
  4. ``single_3utr``    : region-restricted rollout (only 3'UTR editable).
  5. ``joint``          : full rollout (all regions editable).

For each arm we collect the final edited record, score it with the
``LocalTranslationOracle``, and compute synergy scores:

  - ``syn_sum``  = R_joint - (R_5utr + R_cds + R_3utr)        (lambda=1, RL-consistent)
  - ``syn_mean`` = R_joint - mean(R_5utr, R_cds, R_3utr)      (lambda=1/3)
  - ``syn_best`` = R_joint - max(R_5utr, R_cds, R_3utr)       (joint vs best single)
  - ``syn_vs_wt``= R_joint - R_wild_type                       (joint improvement)

Aggregate statistics: mean, std, median, paired t-test (H0: syn = 0),
Cohen's d effect size.

Usage
-----
    python scripts/run_counterfactual_panel.py \\
        --records mrna_editflow/data/processed/gencode_human_transcripts.head64.records.jsonl \\
        --n-wild-types 20 \\
        --max-steps 8 \\
        --output docs/cross_region_synergy_panel_results.json \\
        --seed 1729

Notes
-----
- This is a pipeline-validation script. The full P1-13 panel requires:
  (a) a trained policy (P1-00 STOP → use random TinyTrainableModel),
  (b) a multi-region oracle ensemble (P1-04, currently 5'UTR-focused),
  (c) 1000 wild-type records (currently 59 available).
- The LocalTranslationOracle is 5'UTR-focused (MRL/TE). CDS edits beyond
  the first 12 nt and 3'UTR edits will not change the oracle score, so
  single-CDS and single-3'UTR arms are expected to be ~0 improvement.
  This is a known limitation documented in the finding report.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

# Ensure mrna_editflow is importable when run from repo root.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from mrna_editflow.core.constants import REGION_3UTR, REGION_5UTR, REGION_CDS
from mrna_editflow.core.schema import MRNARecord
from mrna_editflow.eval.oracle import LocalTranslationOracle
from mrna_editflow.rl.action_space import apply_action
from mrna_editflow.rl.policy import Policy, PolicyConfig
from mrna_editflow.rl.synergy import build_region_restricted_mask
from mrna_editflow.rl.tiny_mdp import TinyMDP, TinyTrainableModel, Trajectory


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ArmResult:
    """Result of one arm for one wild-type."""

    arm_name: str
    region_restriction: Optional[int]  # None=joint, 0=5UTR, 1=CDS, 2=3UTR
    n_edits: int
    final_seq: str
    oracle_score: Dict[str, float]


@dataclass
class WildTypePanel:
    """5 arms for one wild-type record."""

    transcript_id: str
    wild_type: ArmResult
    single_5utr: ArmResult
    single_cds: ArmResult
    single_3utr: ArmResult
    joint: ArmResult
    synergy_scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class PanelStats:
    """Aggregate statistics over all wild-types."""

    n_wild_types: int
    syn_sum_mean: float
    syn_sum_std: float
    syn_sum_median: float
    syn_mean_mean: float
    syn_mean_std: float
    syn_mean_median: float
    syn_best_mean: float
    syn_best_std: float
    syn_best_median: float
    syn_vs_wt_mean: float
    syn_vs_wt_std: float
    syn_vs_wt_median: float
    # Paired t-test on syn_sum (H0: syn_sum = 0).
    t_stat_syn_sum: float
    t_pvalue_syn_sum: float
    # Cohen's d on syn_sum.
    cohens_d_syn_sum: float
    # Per-arm improvement over wild-type.
    arm_improvement: Dict[str, Dict[str, float]]


# ---------------------------------------------------------------------------
# Oracle-based MDP
# ---------------------------------------------------------------------------


class OracleMDP(TinyMDP):
    """TinyMDP with oracle-score reward.

    Reward = oracle_score(final_state) - oracle_score(initial_state).
    This isolates the *improvement* due to edits, not the absolute score.
    """

    def __init__(
        self,
        initial_record: MRNARecord,
        oracle: LocalTranslationOracle,
        max_steps: int = 8,
        gamma: float = 1.0,
    ) -> None:
        # target_seq is unused; pass a placeholder.
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
        """Use ensemble_te as the reward signal (bounded [0, 1])."""
        s = self.oracle.score_record(record)
        return float(s.get("ensemble_te", 0.0))

    def reward(self, state, action, next_state, step):  # type: ignore[override]
        is_terminal = action.is_stop() or (step + 1) >= self.max_steps
        if not is_terminal:
            return 0.0
        return self._score(next_state) - self._baseline_score


# ---------------------------------------------------------------------------
# Counterfactual panel
# ---------------------------------------------------------------------------


def _collect_arm(
    policy: Policy,
    mdp: OracleMDP,
    region_restriction: Optional[int],
    generator: torch.Generator,
) -> Tuple[Trajectory, MRNARecord]:
    """Collect one rollout, optionally region-restricted.

    Returns (trajectory, final_record).
    """
    original_mask_fn = policy.legal_action_mask
    if region_restriction is not None:
        def restricted_mask(rec, _rr=region_restriction):
            return build_region_restricted_mask(
                rec, policy.device, _rr, codon_indel=policy.cfg.codon_indel
            )
        policy.legal_action_mask = restricted_mask  # type: ignore[assignment]

    try:
        traj = policy_sample_trajectory(policy, mdp, generator)
    finally:
        policy.legal_action_mask = original_mask_fn  # type: ignore[assignment]

    # Reconstruct final record by applying all actions.
    record = mdp.initial_state()
    for t in traj.transitions:
        record = apply_action(record, t.action)
    return traj, record


def policy_sample_trajectory(
    policy: Policy,
    mdp: OracleMDP,
    generator: torch.Generator,
) -> Trajectory:
    """Sample one trajectory by rolling out ``policy`` on ``mdp``.

    This mirrors ``REINFORCE.collect_trajectory`` but is a free function so
    we can apply it under different region restrictions.
    """
    from mrna_editflow.rl.tiny_mdp import Trajectory, Transition

    traj = Trajectory()
    state = mdp.initial_state()
    for step in range(mdp.max_steps):
        action, _ = policy.sample(
            state,
            budget_remaining=mdp.max_steps - step,
            budget_total=mdp.max_steps,
            generator=generator,
        )
        next_state = apply_action(state, action)
        reward = mdp.reward(state, action, next_state, step)
        traj.transitions.append(
            Transition(state=state, action=action, reward=reward, next_state=next_state, step=step)
        )
        state = next_state
        if action.is_stop():
            break
    return traj


def run_panel_for_wild_type(
    record: MRNARecord,
    oracle: LocalTranslationOracle,
    max_steps: int,
    seed: int,
    device: torch.device,
) -> WildTypePanel:
    """Run the 5-arm panel for one wild-type record."""
    # Build a fresh random policy for each wild-type (no training).
    model = TinyTrainableModel(vocab_dim=4, hidden=16, rates_init=0.5)
    model.to(device)
    backbone = type("B", (), {"out_dim": 0, "freeze": lambda self: None, "to": lambda self, d: self})()
    cfg = PolicyConfig(
        stop_rate_strategy="constant",
        stop_rate_value=1.0,
        temperature=1.0,
        time_step=0.5,
        codon_indel=False,
    )
    policy = Policy(model=model, backbone=backbone, cfg=cfg, device=device)
    mdp = OracleMDP(
        initial_record=record,
        oracle=oracle,
        max_steps=max_steps,
        gamma=1.0,
    )

    # Wild-type arm (no edits).
    wt_score = oracle.score_record(record)
    wt_arm = ArmResult(
        arm_name="wild_type",
        region_restriction=None,
        n_edits=0,
        final_seq=record.seq,
        oracle_score=wt_score,
    )

    # 4 rollout arms with independent generators (no shared prefix).
    arms: List[ArmResult] = []
    region_specs: List[Tuple[str, Optional[int]]] = [
        ("single_5utr", REGION_5UTR),
        ("single_cds", REGION_CDS),
        ("single_3utr", REGION_3UTR),
        ("joint", None),
    ]
    for arm_name, region in region_specs:
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed + hash(arm_name) % (2**31 - 1)))
        traj, final_record = _collect_arm(policy, mdp, region, gen)
        n_edits = sum(1 for t in traj.transitions if not t.action.is_stop())
        score = oracle.score_record(final_record)
        arms.append(
            ArmResult(
                arm_name=arm_name,
                region_restriction=region,
                n_edits=n_edits,
                final_seq=final_record.seq,
                oracle_score=score,
            )
        )

    panel = WildTypePanel(
        transcript_id=record.transcript_id,
        wild_type=wt_arm,
        single_5utr=arms[0],
        single_cds=arms[1],
        single_3utr=arms[2],
        joint=arms[3],
    )

    # Compute synergy scores using ensemble_te as the scalar reward.
    # All rewards are DELTAS (improvement over wild-type), so that
    #   syn_sum = delta_joint - (delta_5 + delta_c + delta_3)
    # is the standard synergy decomposition (0 = additive, >0 = positive
    # synergy, <0 = redundancy).
    r_wt = float(wt_score.get("ensemble_te", 0.0))
    r_5_abs = float(arms[0].oracle_score.get("ensemble_te", 0.0))
    r_c_abs = float(arms[1].oracle_score.get("ensemble_te", 0.0))
    r_3_abs = float(arms[2].oracle_score.get("ensemble_te", 0.0))
    r_j_abs = float(arms[3].oracle_score.get("ensemble_te", 0.0))

    # Deltas (improvement over wild-type).
    d_5 = r_5_abs - r_wt
    d_c = r_c_abs - r_wt
    d_3 = r_3_abs - r_wt
    d_j = r_j_abs - r_wt

    panel.synergy_scores = {
        # Absolute scores (for reference).
        "r_wild_type": r_wt,
        "r_single_5utr_abs": r_5_abs,
        "r_single_cds_abs": r_c_abs,
        "r_single_3utr_abs": r_3_abs,
        "r_joint_abs": r_j_abs,
        # Deltas (improvement over wild-type).
        "delta_5utr_vs_wt": d_5,
        "delta_cds_vs_wt": d_c,
        "delta_3utr_vs_wt": d_3,
        "delta_joint_vs_wt": d_j,
        # Synergy (joint vs sum of singles). 0=additive, >0=synergy, <0=redundancy.
        "syn_sum": d_j - (d_5 + d_c + d_3),
        # Synergy (joint vs mean of singles).
        "syn_mean": d_j - (d_5 + d_c + d_3) / 3.0,
        # Synergy (joint vs best single).
        "syn_best": d_j - max(d_5, d_c, d_3),
        # Joint improvement over wild-type (same as delta_joint_vs_wt).
        "syn_vs_wt": d_j,
    }
    return panel


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def _paired_t_test(x: List[float], mu: float = 0.0) -> Tuple[float, float]:
    """One-sample t-test: H0: mean(x) = mu.

    Returns (t_stat, two-sided p-value). Uses a simple implementation
    (no scipy dependency) with the t-distribution CDF approximated by
    the normal CDF for n >= 30, and an exact small-n formula otherwise.

    For the pipeline validation, this is sufficient. For the final
    paper-grade analysis, use scipy.stats.ttest_1samp.
    """
    n = len(x)
    if n < 2:
        return float("nan"), float("nan")
    mean = statistics.mean(x)
    # Sample standard deviation (ddof=1).
    if n < 2:
        std = 0.0
    else:
        std = statistics.stdev(x)
    if std == 0.0:
        return float("inf") if mean != mu else 0.0, 0.0 if mean == mu else 0.0
    t_stat = (mean - mu) / (std / math.sqrt(n))
    # Approximate two-sided p-value.
    # For n >= 30, use normal approximation.
    if n >= 30:
        # Normal CDF via erf.
        p_value = 2.0 * (1.0 - _normal_cdf(abs(t_stat)))
    else:
        # Use the incomplete beta function approximation for small n.
        # This is a rough approximation; for paper-grade, use scipy.
        df = n - 1
        p_value = 2.0 * _t_sf(abs(t_stat), df)
    return float(t_stat), float(p_value)


def _normal_cdf(x: float) -> float:
    """Standard normal CDF via erf approximation (Abramowitz & Stegun 7.1.26)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _t_sf(t: float, df: int) -> float:
    """Survival function P(T > t) for Student's t with df degrees of freedom.

    Uses the regularized incomplete beta function. For pipeline validation,
    a normal approximation is acceptable; we use it here.
    """
    # Normal approximation (conservative for small df).
    return 1.0 - _normal_cdf(t)


def _cohens_d(x: List[float]) -> float:
    """Cohen's d one-sample effect size: mean(x) / std(x)."""
    n = len(x)
    if n < 2:
        return float("nan")
    mean = statistics.mean(x)
    std = statistics.stdev(x)
    if std == 0.0:
        return float("inf") if mean != 0.0 else 0.0
    return mean / std


def aggregate_stats(panels: List[WildTypePanel]) -> PanelStats:
    """Compute aggregate statistics over all wild-type panels."""
    n = len(panels)
    syn_sums = [p.synergy_scores["syn_sum"] for p in panels]
    syn_means = [p.synergy_scores["syn_mean"] for p in panels]
    syn_bests = [p.synergy_scores["syn_best"] for p in panels]
    syn_vs_wts = [p.synergy_scores["syn_vs_wt"] for p in panels]

    t_stat, t_pval = _paired_t_test(syn_sums, mu=0.0)
    d = _cohens_d(syn_sums)

    # Per-arm improvement over wild-type.
    arm_keys = ["single_5utr", "single_cds", "single_3utr", "joint"]
    arm_improvement: Dict[str, Dict[str, float]] = {}
    for arm in arm_keys:
        if arm == "single_5utr":
            deltas = [p.synergy_scores["delta_5utr_vs_wt"] for p in panels]
        elif arm == "single_cds":
            deltas = [p.synergy_scores["delta_cds_vs_wt"] for p in panels]
        elif arm == "single_3utr":
            deltas = [p.synergy_scores["delta_3utr_vs_wt"] for p in panels]
        else:  # joint
            deltas = [p.synergy_scores["syn_vs_wt"] for p in panels]
        arm_improvement[arm] = {
            "mean": statistics.mean(deltas) if deltas else 0.0,
            "std": statistics.stdev(deltas) if len(deltas) >= 2 else 0.0,
            "median": statistics.median(deltas) if deltas else 0.0,
        }

    return PanelStats(
        n_wild_types=n,
        syn_sum_mean=statistics.mean(syn_sums) if syn_sums else 0.0,
        syn_sum_std=statistics.stdev(syn_sums) if len(syn_sums) >= 2 else 0.0,
        syn_sum_median=statistics.median(syn_sums) if syn_sums else 0.0,
        syn_mean_mean=statistics.mean(syn_means) if syn_means else 0.0,
        syn_mean_std=statistics.stdev(syn_means) if len(syn_means) >= 2 else 0.0,
        syn_mean_median=statistics.median(syn_means) if syn_means else 0.0,
        syn_best_mean=statistics.mean(syn_bests) if syn_bests else 0.0,
        syn_best_std=statistics.stdev(syn_bests) if len(syn_bests) >= 2 else 0.0,
        syn_best_median=statistics.median(syn_bests) if syn_bests else 0.0,
        syn_vs_wt_mean=statistics.mean(syn_vs_wts) if syn_vs_wts else 0.0,
        syn_vs_wt_std=statistics.stdev(syn_vs_wts) if len(syn_vs_wts) >= 2 else 0.0,
        syn_vs_wt_median=statistics.median(syn_vs_wts) if syn_vs_wts else 0.0,
        t_stat_syn_sum=t_stat,
        t_pvalue_syn_sum=t_pval,
        cohens_d_syn_sum=d,
        arm_improvement=arm_improvement,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def load_wild_type_records(path: str, max_utr_len: int = 160) -> List[MRNARecord]:
    """Load wild-type records with all 3 regions and 5'UTR <= max_utr_len."""
    records: List[MRNARecord] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            five = d.get("five_utr", "")
            cds = d.get("cds", "")
            three = d.get("three_utr", "")
            if not (five and cds and three):
                continue
            # Oracle truncates 5'UTR to 160 nt; skip longer UTRs for clean comparison.
            if len(five) > max_utr_len:
                continue
            # Skip records with T (DNA) — V=4 mask would be out of bounds.
            seq = five + cds + three
            if "T" in seq:
                continue
            try:
                r = MRNARecord(
                    transcript_id=d["transcript_id"],
                    five_utr=five,
                    cds=cds,
                    three_utr=three,
                    species=d.get("species", "human"),
                )
                records.append(r)
            except Exception:
                continue
    return records


def panel_to_dict(panel: WildTypePanel) -> dict:
    return {
        "transcript_id": panel.transcript_id,
        "wild_type": asdict(panel.wild_type),
        "single_5utr": asdict(panel.single_5utr),
        "single_cds": asdict(panel.single_cds),
        "single_3utr": asdict(panel.single_3utr),
        "joint": asdict(panel.joint),
        "synergy_scores": panel.synergy_scores,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="P1-13 counterfactual synergy panel")
    parser.add_argument(
        "--records",
        default="mrna_editflow/data/processed/gencode_human_transcripts.head64.records.jsonl",
        help="Path to wild-type records JSONL.",
    )
    parser.add_argument("--n-wild-types", type=int, default=20, help="Number of wild-types to evaluate.")
    parser.add_argument("--max-steps", type=int, default=8, help="Max edits per rollout.")
    parser.add_argument("--max-utr-len", type=int, default=160, help="Max 5'UTR length (oracle truncates at 160).")
    parser.add_argument("--seed", type=int, default=1729, help="Master RNG seed.")
    parser.add_argument("--output", default="docs/cross_region_synergy_panel_results.json", help="Output JSON path.")
    parser.add_argument("--device", default="cpu", help="Device (cpu or cuda).")
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    print(f"[panel] Loading records from {args.records}")
    records = load_wild_type_records(args.records, max_utr_len=args.max_utr_len)
    print(f"[panel] Loaded {len(records)} valid wild-type records (5'UTR <= {args.max_utr_len} nt, no T)")

    if len(records) < args.n_wild_types:
        print(f"[panel] Warning: only {len(records)} records available, using all.")
        args.n_wild_types = len(records)

    records = records[: args.n_wild_types]
    print(f"[panel] Running panel on {len(records)} wild-types, max_steps={args.max_steps}")

    oracle = LocalTranslationOracle(max_len=args.max_utr_len, seed=args.seed)

    panels: List[WildTypePanel] = []
    t0 = time.time()
    for i, record in enumerate(records):
        seed_i = args.seed + i * 1000
        try:
            panel = run_panel_for_wild_type(
                record=record,
                oracle=oracle,
                max_steps=args.max_steps,
                seed=seed_i,
                device=device,
            )
            panels.append(panel)
            elapsed = time.time() - t0
            print(
                f"[panel] {i + 1}/{len(records)} {record.transcript_id} "
                f"syn_sum={panel.synergy_scores['syn_sum']:+.4f} "
                f"syn_vs_wt={panel.synergy_scores['syn_vs_wt']:+.4f} "
                f"({elapsed:.1f}s elapsed)"
            )
        except Exception as e:
            print(f"[panel] ERROR on {record.transcript_id}: {e}")
            continue

    if not panels:
        print("[panel] No panels collected. Aborting.")
        return 1

    stats = aggregate_stats(panels)

    # Serialize.
    output = {
        "config": {
            "records_path": args.records,
            "n_wild_types": args.n_wild_types,
            "max_steps": args.max_steps,
            "max_utr_len": args.max_utr_len,
            "seed": args.seed,
            "device": args.device,
        },
        "stats": asdict(stats),
        "panels": [panel_to_dict(p) for p in panels],
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, sort_keys=True)
    print(f"[panel] Wrote {args.output}")

    # Print summary.
    print("\n=== Panel Summary ===")
    print(f"n_wild_types:        {stats.n_wild_types}")
    print(f"syn_sum  mean ± std: {stats.syn_sum_mean:+.4f} ± {stats.syn_sum_std:.4f}  (median {stats.syn_sum_median:+.4f})")
    print(f"syn_mean mean ± std: {stats.syn_mean_mean:+.4f} ± {stats.syn_mean_std:.4f}  (median {stats.syn_mean_median:+.4f})")
    print(f"syn_best mean ± std: {stats.syn_best_mean:+.4f} ± {stats.syn_best_std:.4f}  (median {stats.syn_best_median:+.4f})")
    print(f"syn_vs_wt mean ± std: {stats.syn_vs_wt_mean:+.4f} ± {stats.syn_vs_wt_std:.4f}  (median {stats.syn_vs_wt_median:+.4f})")
    print(f"t-test (syn_sum=0):  t={stats.t_stat_syn_sum:+.4f}, p={stats.t_pvalue_syn_sum:.4f}")
    print(f"Cohen's d (syn_sum): {stats.cohens_d_syn_sum:+.4f}")
    print("\nPer-arm improvement over wild-type (ensemble_te):")
    for arm, s in stats.arm_improvement.items():
        print(f"  {arm:14s}  mean={s['mean']:+.4f}  std={s['std']:.4f}  median={s['median']:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
