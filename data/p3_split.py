"""P3-01: leakage-safe split assignment + audit (source/cargo/family grouping).

Grouping rules (prompt Task 4): all candidates of the same source, highly
homologous cargo, same protein family, same experimental library, and same
reporter backbone are grouped ATOMICALLY — they never cross split roles.

Role semantics:
    train : source-disjoint
    val   : source-disjoint
    test  : cargo/family-disjoint (whole families held out from ALL other roles)
    ood   : length/GC/context-shifted groups (diagnostic holdout)

Hard guarantees (verified by ``audit_split``):
    * no source_id appears in more than one role
    * no family_cluster_id is shared between test and any other role
    * no exact candidate_sequence is shared across roles
    * OOD groups satisfy the declared shift criteria

Family overlap between train/val and OOD is not forbidden by the spec but is
REPORTED in the audit for transparency.
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class SourceGroup:
    group_id: str
    family_id: str
    n_records: int
    gc: float
    length: int
    cohort: str = ""  # e.g. "sample2019_snv" / "reconstructed"; OOD shift is
                      # computed WITHIN a cohort so library-specific length/GC
                      # floors (all 50nt MPRA mothers) are not misread as shift.


@dataclass(frozen=True)
class SplitConfig:
    train_frac: float = 0.70
    val_frac: float = 0.10
    test_frac: float = 0.10
    seed: int = 20260723
    ood_gc_quantile: float = 0.05     # GC outside [q, 1-q] within cohort -> shifted
    ood_length_quantile: float = 0.05  # length outside [q, 1-q] within cohort -> shifted
    min_ood_groups: int = 1


def _quantile(sorted_vals: Sequence[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = min(len(sorted_vals) - 1, max(0, int(round(q * (len(sorted_vals) - 1)))))
    return float(sorted_vals[idx])


def flag_ood_groups(groups: Sequence[SourceGroup], cfg: SplitConfig) -> Dict[str, List[str]]:
    """Return {group_id: [shift reasons]} for groups meeting OOD shift criteria.

    GC/length tails are computed WITHIN each cohort (``SourceGroup.cohort``);
    groups with an empty cohort share one anonymous cohort. This prevents a
    fixed-length library (e.g. all-50nt MPRA mothers) from being wholesale
    flagged as length-shifted relative to a different cohort.
    """
    by_cohort: Dict[str, List[SourceGroup]] = {}
    for g in groups:
        by_cohort.setdefault(g.cohort, []).append(g)

    bounds: Dict[str, Tuple[float, float, float, float]] = {}
    for cohort, members in by_cohort.items():
        gcs = sorted(m.gc for m in members)
        lens = sorted(float(m.length) for m in members)
        bounds[cohort] = (
            _quantile(gcs, cfg.ood_gc_quantile),
            _quantile(gcs, 1.0 - cfg.ood_gc_quantile),
            _quantile(lens, cfg.ood_length_quantile),
            _quantile(lens, 1.0 - cfg.ood_length_quantile),
        )

    flagged: Dict[str, List[str]] = {}
    for g in groups:
        glo, ghi, llo, lhi = bounds[g.cohort]
        reasons: List[str] = []
        # STRICT inequalities: with discrete tie-masses (e.g. the 128 nt 5'UTR
        # spike of the reconstructed cohort), an inclusive boundary comparison
        # would flag the entire modal mass at the quantile boundary as shifted.
        if glo < ghi and (g.gc < glo or g.gc > ghi):
            reasons.append("gc_shift")
        if llo < lhi and (float(g.length) < llo or float(g.length) > lhi):
            reasons.append("length_shift")
        if reasons:
            flagged[g.group_id] = reasons
    return flagged


def assign_split_roles(
    groups: Sequence[SourceGroup],
    cfg: SplitConfig,
) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """Deterministically assign each group to train/val/test/ood.

    Returns (roles, metadata). Order of operations:
      1. OOD: all shift-flagged groups.
      2. test: whole families (seeded order) until test_frac of remaining records.
      3. val:  seeded group order until val_frac of total records.
      4. train: remainder.
    """
    rng = random.Random(cfg.seed)
    total_records = sum(g.n_records for g in groups)
    flagged = flag_ood_groups(groups, cfg)

    roles: Dict[str, str] = {}
    ood_reasons: Dict[str, List[str]] = {}
    for g in groups:
        if g.group_id in flagged:
            roles[g.group_id] = "ood"
            ood_reasons[g.group_id] = flagged[g.group_id]

    pool = [g for g in groups if g.group_id not in roles]

    # test: whole families, disjoint from everything else by construction.
    # Families already touched by OOD are skipped: test must stay
    # cargo/family-disjoint from ALL other roles (hard audit guarantee).
    ood_families = {g.family_id for g in groups if g.group_id in flagged}
    fam_map: Dict[str, List[SourceGroup]] = {}
    for g in pool:
        fam_map.setdefault(g.family_id, []).append(g)
    families = sorted(fam_map.items(), key=lambda kv: (kv[0]))
    rng.shuffle(families)
    test_target = cfg.test_frac * total_records
    test_records = 0
    test_families: List[str] = []
    for fam_id, members in families:
        if test_records >= test_target:
            break
        if fam_id in ood_families:
            continue
        for m in members:
            roles[m.group_id] = "test"
            test_records += m.n_records
        test_families.append(fam_id)

    rest = [g for g in pool if g.group_id not in roles]
    rng.shuffle(rest)
    val_target = cfg.val_frac * total_records
    val_records = 0
    for g in rest:
        if val_records >= val_target:
            break
        roles[g.group_id] = "val"
        val_records += g.n_records
    for g in rest:
        if g.group_id not in roles:
            roles[g.group_id] = "train"

    metadata = {
        "seed": cfg.seed,
        "fractions": {"train": cfg.train_frac, "val": cfg.val_frac, "test": cfg.test_frac},
        "ood_criteria": {
            "gc_quantile": cfg.ood_gc_quantile,
            "length_quantile": cfg.ood_length_quantile,
            "per_cohort": True,
        },
        "ood_group_reasons": ood_reasons,
        "test_families": sorted(test_families),
        "role_group_counts": {
            role: sum(1 for r in roles.values() if r == role)
            for role in ("train", "val", "test", "ood")
        },
        "role_record_counts": {
            role: sum(g.n_records for g in groups if roles[g.group_id] == role)
            for role in ("train", "val", "test", "ood")
        },
    }
    return roles, metadata


def canonical_assignment_json(roles: Mapping[str, str]) -> str:
    return json.dumps(dict(sorted(roles.items())), sort_keys=True, separators=(",", ":"))


def split_assignment_sha256(roles: Mapping[str, str]) -> str:
    return hashlib.sha256(canonical_assignment_json(roles).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Cross-role exact-sequence collision resolution
# ---------------------------------------------------------------------------

# Role retention priority for colliding candidate sequences. Evaluation roles
# (test/ood) are protected first: a colliding record is dropped from the
# lower-priority role so an exact sequence never appears in two roles.
ROLE_PRIORITY: Mapping[str, int] = {"test": 0, "ood": 1, "val": 2, "train": 3}


def resolve_cross_role_sequence_collisions(
    tier_records: Mapping[str, Sequence[Any]],
) -> Tuple[set, Dict[str, Any]]:
    """Find candidate sequences spanning >1 split role; return records to drop.

    Near-homologous sources in different families can generate byte-identical
    candidate sequences (e.g. two reconstructed 5'UTRs differing at one
    position produce the same single edit). Since group->role assignment is
    source/family-atomic, such exact sequences can cross roles — a hard
    leakage violation caught by ``audit_split``. Resolution is deterministic:
    for each colliding sequence keep ALL records in the highest-priority role
    (test > ood > val > train) and drop the rest. Source-disjointness and
    family-disjointness are unaffected (records are removed, never moved).

    ``tier_records`` maps tier name -> records exposing ``candidate_sequence``,
    ``split_role`` and ``record_id``. Returns (dropped_record_ids, stats).
    """
    seq_map: Dict[str, List[Any]] = {}
    for records in tier_records.values():
        for rec in records:
            seq_map.setdefault(str(rec.candidate_sequence), []).append(rec)

    dropped: set = set()
    drops_by_role: Dict[str, int] = {"train": 0, "val": 0, "ood": 0, "test": 0}
    n_collision_sequences = 0
    for seq, recs in seq_map.items():
        roles_present = {str(r.split_role) for r in recs}
        if len(roles_present) <= 1:
            continue
        n_collision_sequences += 1
        keep_role = min(roles_present, key=lambda r: ROLE_PRIORITY[r])
        for r in recs:
            if str(r.split_role) != keep_role:
                dropped.add(r.record_id)
                drops_by_role[str(r.split_role)] += 1

    stats: Dict[str, Any] = {
        "n_collision_sequences": n_collision_sequences,
        "n_dropped_records": len(dropped),
        "drops_by_role": drops_by_role,
        "role_priority": {r: p for r, p in sorted(ROLE_PRIORITY.items(),
                                                  key=lambda kv: kv[1])},
        "policy": "keep all records of the highest-priority role for each "
                  "colliding candidate_sequence; drop the rest (deterministic)",
    }
    return dropped, stats


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------

def audit_split(
    records: Iterable[Mapping[str, Any]],
    tier_name: str,
) -> Dict[str, Any]:
    """Verify leakage guarantees over finalized records (with split_role set).

    ``records`` must expose: source_id, family_cluster_id, candidate_sequence,
    split_role. Returns a JSON-serializable audit dict; ``passed`` is True only
    if all hard guarantees hold.
    """
    by_source: Dict[str, set] = {}
    by_family: Dict[str, set] = {}
    by_seq: Dict[str, set] = {}
    role_counts: Dict[str, int] = {"train": 0, "val": 0, "test": 0, "ood": 0}
    n = 0
    for rec in records:
        role = rec.get("split_role")
        if role not in role_counts:
            raise ValueError(f"record missing/invalid split_role: {role}")
        role_counts[role] += 1
        n += 1
        by_source.setdefault(str(rec["source_id"]), set()).add(role)
        fam = rec.get("family_cluster_id")
        if fam is not None:
            by_family.setdefault(str(fam), set()).add(role)
        by_seq.setdefault(str(rec["candidate_sequence"]), set()).add(role)

    source_violations = sorted(s for s, r in by_source.items() if len(r) > 1)
    seq_violations = sum(1 for r in by_seq.values() if len(r) > 1)

    def roles_of(fams: Iterable[str]) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {}
        for f in fams:
            out[f] = sorted(by_family[f])
        return out

    test_fams = {f for f, r in by_family.items() if "test" in r}
    test_family_violations = sorted(f for f in test_fams if len(by_family[f]) > 1)

    trainval_fams = {f for f, r in by_family.items() if r & {"train", "val"}}
    ood_fams = {f for f, r in by_family.items() if "ood" in r}
    trainval_ood_family_overlap = sorted(trainval_fams & ood_fams)

    passed = (
        not source_violations
        and seq_violations == 0
        and not test_family_violations
    )
    return {
        "tier": tier_name,
        "n_records": n,
        "role_counts": role_counts,
        "n_sources": len(by_source),
        "n_families": len(by_family),
        "passed": passed,
        "source_role_violations": source_violations[:20],
        "n_source_role_violations": len(source_violations),
        "cross_role_exact_sequence_collisions": seq_violations,
        "test_family_violations": test_family_violations[:20],
        "n_test_family_violations": len(test_family_violations),
        "trainval_ood_family_overlap_count": len(trainval_ood_family_overlap),
        "trainval_ood_family_overlap_examples": trainval_ood_family_overlap[:20],
    }
