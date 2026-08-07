"""E0-X sealed-final access protocol (pure, unit-testable, no remote data).

Phase E0-X operationally seals the GSE246381 external.  This module enforces
the sealed access protocol that `run_e0x_final.py` must obey:

  * append an ACCESS_INTENT event before any final evaluation;
  * compare-and-append reservation (authenticated, atomic append against the
    running hash chain head);
  * exactly one terminal completion or abort;
  * an abort/crash invalidates the v1 final and is NOT retryable;
  * the only output is the pre-registered aggregate (never row-level
    label / ID / order).

The module is pure (filesystem + hashing only) so it can be unit-tested
without the restricted store or the frozen models.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# hashing / chain
# ---------------------------------------------------------------------------

def sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _canonical_json(obj: Dict) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def hash_event(event: Dict) -> str:
    """Deterministic event hash: over the event excluding the event_sha256 field."""
    body = {k: v for k, v in event.items() if k != "event_sha256"}
    return sha256_hex(_canonical_json(body).encode("utf-8"))


def make_event(access_id: str, object_id: str, intent: str, status: str,
               prev_event_sha256: Optional[str], **extra) -> Dict:
    """Build a hash-linked event (event_sha256 computed over the body)."""
    ev = {"access_id": access_id, "object_id": object_id, "intent": intent,
          "status": status, "prev_event_sha256": prev_event_sha256, **extra}
    ev["event_sha256"] = hash_event(ev)
    return ev


def read_chain(access_log: Path) -> List[Dict]:
    events: List[Dict] = []
    if access_log.exists():
        with access_log.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    events.append(json.loads(line))
    return events


def verify_chain(events: List[Dict]) -> bool:
    """The hash chain is valid iff every event's prev == prior event_sha256 and
    every event_sha256 matches its own body hash."""
    prev = None
    for ev in events:
        if ev.get("prev_event_sha256") != prev:
            return False
        if ev.get("event_sha256") != hash_event(ev):
            return False
        prev = ev.get("event_sha256")
    return True


def chain_head(events: List[Dict]) -> Optional[str]:
    return events[-1].get("event_sha256") if events else None


# ---------------------------------------------------------------------------
# atomic compare-and-append
# ---------------------------------------------------------------------------

def _exclusive_lock(path: Path, timeout_s: float = 10.0) -> bool:
    """Gain an exclusive lock by O_EXCL file creation; returns True if acquired."""
    deadline = time.time() + timeout_s
    while True:
        try:
            fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            return True
        except FileExistsError:
            if time.time() > deadline:
                return False
            time.sleep(0.05)


def compare_and_append(access_log: Path, event: Dict, timeout_s: float = 10.0) -> Tuple[bool, str]:
    """Append `event` only if its prev_event_sha256 matches the current chain head.

    Returns (ok, reason).  On success the event is appended and the new head is
    the event's event_sha256.  On failure nothing is written and the caller must
    abort (no retry of a sealed final).
    """
    access_log = Path(access_log)
    access_log.parent.mkdir(parents=True, exist_ok=True)
    lock = access_log.with_suffix(".lock")
    if not _exclusive_lock(lock, timeout_s):
        return False, "could not acquire access-log lock (concurrent writer)"
    try:
        events = read_chain(access_log)
        # compare-and-append precondition: chain must be valid and head must match
        if not verify_chain(events):
            return False, "existing access-log hash chain is invalid"
        if event.get("prev_event_sha256") != chain_head(events):
            return False, ("stale reservation: expected head %s, found %s"
                           % (chain_head(events), event.get("prev_event_sha256")))
        with access_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
        return True, "appended"
    finally:
        try:
            lock.unlink()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# sealed access state machine
# ---------------------------------------------------------------------------

class SealedAccessError(RuntimeError):
    """Raised when the sealed access protocol is violated."""


class SealedAccessState:
    """Tracks the sealed final lifecycle.

    States:
      UNSEALED      : no ACCESS_INTENT yet.
      INTENT_APPENDED: ACCESS_INTENT written, reservation not yet reserved.
      RESERVED      : compare-and-append reservation confirmed.
      COMPLETED     : exactly one terminal completion event appended.
      ABORTED       : a terminal abort event appended (v1 final invalidated).
      INVALIDATED   : a crash/abort occurred; the v1 final is NOT retryable.
    """
    UNSEALED = "UNSEALED"
    INTENT_APPENDED = "INTENT_APPENDED"
    RESERVED = "RESERVED"
    COMPLETED = "COMPLETED"
    ABORTED = "ABORTED"
    INVALIDATED = "INVALIDATED"

    def __init__(self, access_log: Path):
        self.access_log = Path(access_log)
        self.reload()

    def reload(self) -> None:
        self._events = read_chain(self.access_log)
        self._state = self._derive_state()

    def _derive_state(self) -> str:
        if not verify_chain(self._events):
            return self.INVALIDATED
        if not self._events:
            return self.UNSEALED
        statuses = [ev.get("status") for ev in self._events
                    if ev.get("intent", "").startswith("e0x_")]
        # status values used in the append-only chain:
        #   ACCESS_INTENT -> RESERVED -> COMPLETION | ABORT
        if "COMPLETION" in statuses:
            return self.COMPLETED
        if "ABORT" in statuses:
            return self.ABORTED
        if "RESERVED" in statuses:
            return self.RESERVED
        if "ACCESS_INTENT" in statuses:
            return self.INTENT_APPENDED
        return self.UNSEALED

    @property
    def state(self) -> str:
        return self._state

    @property
    def events(self) -> List[Dict]:
        return list(self._events)

    # -- transitions ------------------------------------------------------

    def append_intent(self, access_id: str, object_id: str, intent: str,
                      preregistration_id: str) -> str:
        """Append the ACCESS_INTENT. Returns the new chain head."""
        if self._state != self.UNSEALED:
            raise SealedAccessError(
                "cannot append ACCESS_INTENT from state %s" % self._state)
        ev = make_event(access_id, object_id, intent, "ACCESS_INTENT",
                        chain_head(self._events),
                        preregistration_id=preregistration_id)
        ok, reason = compare_and_append(self.access_log, ev)
        if not ok:
            raise SealedAccessError("ACCESS_INTENT append failed: " + reason)
        self.reload()
        return ev["event_sha256"]

    def reserve(self, access_id: str, object_id: str, intent: str,
                preregistration_id: str) -> str:
        """Compare-and-append the reservation. Returns the new chain head."""
        if self._state != self.INTENT_APPENDED:
            raise SealedAccessError(
                "cannot reserve from state %s (must be INTENT_APPENDED)" % self._state)
        ev = make_event(access_id, object_id, intent, "RESERVED",
                        chain_head(self._events),
                        preregistration_id=preregistration_id)
        ok, reason = compare_and_append(self.access_log, ev)
        if not ok:
            raise SealedAccessError("reservation append failed: " + reason)
        self.reload()
        return ev["event_sha256"]

    def complete(self, access_id: str, object_id: str, intent: str,
                 preregistration_id: str, result_sha256: str) -> str:
        """Exactly one terminal COMPLETION. Returns the new chain head."""
        if self._state != self.RESERVED:
            raise SealedAccessError(
                "cannot COMPLETE from state %s (must be RESERVED)" % self._state)
        ev = make_event(access_id, object_id, intent, "COMPLETION",
                        chain_head(self._events),
                        preregistration_id=preregistration_id,
                        result_sha256=result_sha256)
        ok, reason = compare_and_append(self.access_log, ev)
        if not ok:
            raise SealedAccessError("completion append failed: " + reason)
        self.reload()
        return ev["event_sha256"]

    def abort(self, access_id: str, object_id: str, intent: str,
              preregistration_id: str, reason: str) -> str:
        """Exactly one terminal ABORT; invalidates the v1 final (not retryable)."""
        if self._state not in (self.INTENT_APPENDED, self.RESERVED):
            raise SealedAccessError(
                "cannot ABORT from state %s" % self._state)
        ev = make_event(access_id, object_id, intent, "ABORT",
                        chain_head(self._events),
                        preregistration_id=preregistration_id,
                        abort_reason=reason)
        ok, msg = compare_and_append(self.access_log, ev)
        if not ok:
            raise SealedAccessError("abort append failed: " + msg)
        self.reload()
        return ev["event_sha256"]


# ---------------------------------------------------------------------------
# aggregate-only output
# ---------------------------------------------------------------------------

def apply_holm(prereg: Dict, pvalues: List[float]) -> List[Optional[float]]:
    """Apply the frozen Holm-Bonferroni family to the hypothesis p-values.

    Delegates to prereg.holm_bonferroni (imported lazily to avoid a cycle).
    """
    from scripts.e0x import prereg as _prereg
    alpha = (prereg.get("metrics", {}).get("holm_family", {}) or {}).get("alpha", 0.05)
    return _prereg.holm_bonferroni(pvalues, alpha=alpha)


def build_aggregate(prereg: Dict, per_hypothesis: List[Dict]) -> Dict:
    """Build the pre-registered aggregate-only result.

    `per_hypothesis` is a list of {id, metric, stat, pvalue, n} aggregate
    entries (NO row-level fields).  Enforces the frozen output schema fields.
    """
    fields = (prereg.get("execution", {}).get("output_schema", {}) or {}).get(
        "fields", ["phase", "preregistration_id", "status", "sealed_access_state",
                   "per_hypothesis", "holm_adjusted_pvalues", "go_nogo_verdict"])
    pvalues = [h.get("pvalue") for h in per_hypothesis]
    non_null = [p for p in pvalues if p is not None]
    adj_non_null = apply_holm(prereg, non_null)
    it = iter(adj_non_null)
    holm_adj = [next(it) if p is not None else None for p in pvalues]
    out = {
        "phase": prereg.get("phase"),
        "preregistration_id": prereg.get("preregistration_id"),
        "status": "COMPLETED",
        "sealed_access_state": "COMPLETED",
        "per_hypothesis": per_hypothesis,
        "holm_adjusted_pvalues": holm_adj,
    }
    # keep only the frozen fields actually present
    return {k: out[k] for k in out if k in fields}


# ---------------------------------------------------------------------------
# hypothesis / verdict (pure, testable, no remote data)
# ---------------------------------------------------------------------------

def build_hypothesis(hid: str, metric: str, stat: float, pvalue: Optional[float],
                     n: int) -> Dict:
    return {"id": hid, "metric": metric, "stat": stat, "pvalue": pvalue, "n": n}


def permutation_pvalue(true_delta: "np.ndarray", pred: "np.ndarray",
                       contexts: List[str], n_perm: int = 2000,
                       seed: int = 42,
                       groups: Optional[List[str]] = None) -> Optional[float]:
    """One-sided permutation p-value (positive macro Spearman association).

    Null: the order of predicted deltas is exchangeable with the measured deltas
    within each context.  We shuffle predicted deltas within each context,
    recompute the context Spearman, macro-average, and count how often the
    permuted macro Spearman is >= the observed.  Returns (count+1)/(n_perm+1),
    or None if no context has a well-defined Spearman.
    """
    import numpy as np
    from scipy.stats import spearmanr
    true_delta = np.asarray(true_delta, float)
    pred = np.asarray(pred, float)
    if len(true_delta) < 3:
        return None
    rng = np.random.RandomState(seed)
    ctx_arr = np.asarray(contexts)
    group_arr = np.asarray(groups) if groups is not None else None
    if group_arr is not None and group_arr.shape != ctx_arr.shape:
        raise ValueError("groups must align with contexts")

    def _macro_spearman(p):
        rho_by_study = {}
        for c in set(ctx_arr):
            idx = np.where(ctx_arr == c)[0]
            if len(idx) < 3:
                continue
            a, b = true_delta[idx], p[idx]
            if np.std(a) == 0 or np.std(b) == 0:
                continue
            r, _ = spearmanr(a, b)
            if not np.isnan(r):
                study = str(c).split("|", 1)[0]
                rho_by_study.setdefault(study, []).append(float(r))
        study_macro = [float(np.mean(values))
                       for values in rho_by_study.values() if values]
        return float(np.mean(study_macro)) if study_macro else None

    obs = _macro_spearman(pred)
    if obs is None:
        return None
    count = 0
    permutable = False
    if group_arr is not None:
        permutable = any(
            len(np.where(group_arr == group)[0]) >= 2
            for group in set(group_arr)
        )
        if not permutable:
            return None
    for _ in range(n_perm):
        p_perm = pred.copy()
        if group_arr is None:
            for c in set(ctx_arr):
                idx = np.where(ctx_arr == c)[0]
                p_perm[idx] = rng.permutation(p_perm[idx])
        else:
            # Candidate labels are exchangeable only within one conservative
            # source/context cluster.  Singleton groups stay fixed and cannot
            # manufacture row-level pseudo-replication significance.
            for group in set(group_arr):
                idx = np.where(group_arr == group)[0]
                if len(idx) >= 2:
                    p_perm[idx] = rng.permutation(p_perm[idx])
        pm = _macro_spearman(p_perm)
        if pm is not None and pm >= obs:
            count += 1
    return (count + 1) / (n_perm + 1)


def verdict_from_aggregate(prereg: Dict, per_hypothesis: List[Dict],
                           holm_adj: List[Optional[float]]) -> Dict:
    """Compute a fail-closed complete-family verdict.

    ``GO`` and ``NO_GO`` are scientific decisions and are emitted only when
    the exact ordered H1/H2/H3 family, every required scalar, the full Holm
    vector, and every frozen threshold are structurally present.  Missing or
    malformed decision inputs produce ``HOLD_REPAIR`` rather than a false
    scientific decision.  Once the record is complete, any failed statistical,
    effect, guidance, OOD, legality, length, or budget gate produces ``NO_GO``.
    """
    gates = (prereg.get("go_nogo") or {})
    eg = gates.get("effect_gate") or {}
    gg = gates.get("guidance_gate") or {}
    hc = gates.get("hard_constraints") or {}
    expected_family = (
        ("H1_EFFECT_TRANSFER", "macro_delta_spearman"),
        ("H2_GUIDANCE_VALUE", "generation_quality_mean_delta"),
        ("H3_LEGALITY", "legality_rate"),
    )
    prereg_family = tuple(
        (h.get("id"), h.get("metric"))
        for h in ((((prereg.get("metrics") or {}).get("holm_family") or {}).get(
            "hypotheses")) or [])
        if isinstance(h, dict)
    )
    observed_family = tuple(
        (h.get("id"), h.get("metric"))
        for h in per_hypothesis if isinstance(h, dict)
    ) if isinstance(per_hypothesis, list) else ()

    repair_reasons = []
    if prereg_family != expected_family:
        repair_reasons.append("preregistration_family_not_exact_H1_H2_H3")
    if observed_family != expected_family:
        repair_reasons.append("aggregate_family_not_exact_ordered_H1_H2_H3")
    if not isinstance(holm_adj, list) or len(holm_adj) != len(expected_family):
        repair_reasons.append("holm_vector_missing_or_wrong_length")

    required_gates = {
        "effect_gate": (
            "macro_delta_spearman_ge", "macro_sign_accuracy_ge",
            "top10_enrichment_ge", "beat_strongest_nonfoundation_baseline",
        ),
        "guidance_gate": (
            "primary_strategy", "vs", "generation_quality_mean_delta_gt",
            "generation_quality_frac_beneficial_gt", "ndcg_gap_ge",
            "top10_recall_ge", "normalized_regret_le",
        ),
        "hard_constraints": (
            "legality_ge", "length_preservation_ge", "budget_violation_le",
        ),
    }
    for block_name, keys in required_gates.items():
        block = gates.get(block_name)
        if not isinstance(block, dict) or any(key not in block for key in keys):
            repair_reasons.append("missing_" + block_name + "_threshold")
    def finite_number(value: object) -> bool:
        import math
        return (isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(float(value)))

    numeric_gate_fields = (
        (eg, "macro_delta_spearman_ge"),
        (eg, "macro_sign_accuracy_ge"),
        (eg, "top10_enrichment_ge"),
        (gg, "generation_quality_mean_delta_gt"),
        (gg, "generation_quality_frac_beneficial_gt"),
        (gg, "ndcg_gap_ge"),
        (gg, "top10_recall_ge"),
        (gg, "normalized_regret_le"),
        (hc, "legality_ge"),
        (hc, "length_preservation_ge"),
        (hc, "budget_violation_le"),
    )
    if any(key in block and not finite_number(block[key])
           for block, key in numeric_gate_fields):
        repair_reasons.append("invalid_frozen_gate_threshold_type")
    if "beat_strongest_nonfoundation_baseline" in eg and (
            eg["beat_strongest_nonfoundation_baseline"] is not True):
        repair_reasons.append("invalid_effect_baseline_gate")
    ood = gates.get("ood_margin")
    if not isinstance(ood, dict) or ood.get("fail_if_beyond") is not True:
        repair_reasons.append("invalid_ood_margin_gate")
    alpha = (((prereg.get("metrics") or {}).get("holm_family") or {}).get("alpha"))
    if (not isinstance(alpha, (int, float)) or isinstance(alpha, bool)
            or not 0.0 < float(alpha) <= 1.0):
        repair_reasons.append("invalid_holm_alpha")

    required_evidence = {
        "H1_EFFECT_TRANSFER": {
            "stat", "sign_accuracy", "top10pct_enrichment",
            "abs_candidate_spearman",
            "top10pct_enrichment_definition_status",
        },
        "H2_GUIDANCE_VALUE": {
            "stat", "generation_quality_mean_delta_difference",
            "generation_quality_frac_beneficial_difference", "ndcg_gap",
            "top10_recall", "normalized_regret", "ood_margin_pass",
        },
        "H3_LEGALITY": {
            "stat", "length_preservation_rate", "budget_violation_count",
        },
    }
    if observed_family == expected_family:
        for hypothesis in per_hypothesis:
            missing = required_evidence[hypothesis["id"]] - set(hypothesis)
            if missing:
                repair_reasons.append(
                    "%s_missing_evidence_%s" %
                    (hypothesis["id"], "_".join(sorted(missing))))
        numeric_evidence = {
            "H1_EFFECT_TRANSFER": (
                "stat", "sign_accuracy", "top10pct_enrichment",
                "abs_candidate_spearman",
            ),
            "H2_GUIDANCE_VALUE": (
                "stat", "generation_quality_mean_delta_difference",
                "generation_quality_frac_beneficial_difference", "ndcg_gap",
                "top10_recall", "normalized_regret",
            ),
            "H3_LEGALITY": (
                "stat", "length_preservation_rate", "budget_violation_count",
            ),
        }
        for hypothesis in per_hypothesis:
            fields = numeric_evidence[hypothesis["id"]]
            if any(field in hypothesis and not finite_number(hypothesis[field])
                   for field in fields):
                repair_reasons.append(
                    hypothesis["id"] + "_invalid_evidence_scalar")
        h2_candidate = per_hypothesis[1]
        if ("ood_margin_pass" in h2_candidate
                and not isinstance(h2_candidate["ood_margin_pass"], bool)):
            repair_reasons.append("H2_GUIDANCE_VALUE_invalid_evidence_ood_flag")

    checks = {
        "complete_frozen_hypothesis_family": not any(
            reason in {
                "preregistration_family_not_exact_H1_H2_H3",
                "aggregate_family_not_exact_ordered_H1_H2_H3",
            } for reason in repair_reasons),
        "complete_holm_vector": "holm_vector_missing_or_wrong_length"
        not in repair_reasons,
        "complete_frozen_gate_configuration": not any(
            reason.startswith("missing_") or reason.startswith("invalid_")
            for reason in repair_reasons),
        "complete_aggregate_evidence": not any(
            "_missing_evidence_" in reason or "_invalid_evidence_" in reason
            for reason in repair_reasons),
    }
    if repair_reasons:
        verdict = "HOLD_REPAIR"
    else:
        stats = {h["id"]: h for h in per_hypothesis}
        adj = {h["id"]: float(value)
               for h, value in zip(per_hypothesis, holm_adj)
               if isinstance(value, (int, float)) and not isinstance(value, bool)}
        if len(adj) != len(expected_family) or any(
                not 0.0 <= value <= 1.0 for value in adj.values()):
            repair_reasons.append("holm_vector_contains_invalid_value")
            checks["complete_holm_vector"] = False
            verdict = "HOLD_REPAIR"
        else:
            h1 = stats["H1_EFFECT_TRANSFER"]
            h2 = stats["H2_GUIDANCE_VALUE"]
            h3 = stats["H3_LEGALITY"]
            checks.update({
                "H1_macro_delta_spearman_ge": (
                    h1["stat"] >= eg["macro_delta_spearman_ge"]),
                "H1_holm_significant": (
                    adj["H1_EFFECT_TRANSFER"] <= float(alpha)),
                "H1_macro_sign_accuracy_ge": (
                    h1["sign_accuracy"] >= eg["macro_sign_accuracy_ge"]),
                "H1_top10pct_enrichment_ge": (
                    h1["top10pct_enrichment"] >= eg["top10_enrichment_ge"]),
                "H1_enrichment_definition_preregistered_stable": (
                    h1["top10pct_enrichment_definition_status"]
                    == "PREREGISTERED_STABLE"),
                "H1_beat_abs_candidate": (
                    eg["beat_strongest_nonfoundation_baseline"] is True
                    and h1["stat"] > h1["abs_candidate_spearman"]),
                "H2_holm_significant": (
                    adj["H2_GUIDANCE_VALUE"] <= float(alpha)),
                "H2_generation_quality_mean_delta_gt": (
                    h2["generation_quality_mean_delta_difference"]
                    > gg["generation_quality_mean_delta_gt"]),
                "H2_generation_quality_frac_beneficial_gt": (
                    h2["generation_quality_frac_beneficial_difference"]
                    > gg["generation_quality_frac_beneficial_gt"]),
                "H2_ndcg_gap_ge": h2["ndcg_gap"] >= gg["ndcg_gap_ge"],
                "H2_top10_recall_ge": (
                    h2["top10_recall"] >= gg["top10_recall_ge"]),
                "H2_normalized_regret_le": (
                    h2["normalized_regret"] <= gg["normalized_regret_le"]),
                "H2_ood_margin_pass": h2["ood_margin_pass"] is True,
                "H3_holm_significant": (
                    adj["H3_LEGALITY"] <= float(alpha)),
                "H3_legality_ge": h3["stat"] >= hc["legality_ge"],
                "H3_length_preservation_ge": (
                    h3["length_preservation_rate"]
                    >= hc["length_preservation_ge"]),
                "H3_budget_violation_le": (
                    h3["budget_violation_count"]
                    <= hc["budget_violation_le"]),
            })
            verdict = "GO" if all(checks.values()) else "NO_GO"
    return {
        "verdict": verdict,
        "checks": checks,
        "repair_reasons": repair_reasons,
        "effect_gate": {k: v for k, v in eg.items() if k in
                        ("macro_delta_spearman_ge", "macro_sign_accuracy_ge",
                         "top10_enrichment_ge")},
        "hard_constraints": dict(hc),
        "guidance_gate": dict(gg),
    }


ROWL_KEYS = {"sequence_id", "source_id", "pair_id", "sequence_sha256",
             "access_id", "row_id", "candidate_id", "observation_id"}


def assert_no_row_level(result: Dict, path: str = "") -> None:
    """Guard: the aggregate must never leak row-level label / ID / order.

    Raises SealedAccessError if any nested dict contains a row-level key or any
    nested list is a raw vector of per-row values (lengths > 1 with scalar
    leafs that are not aggregate stats).
    """
    if isinstance(result, dict):
        for k, v in result.items():
            if k in ROWL_KEYS:
                raise SealedAccessError(
                    "row-level key leaked at %s.%s" % (path, k))
            assert_no_row_level(v, "%s.%s" % (path, k))
    elif isinstance(result, list):
        # aggregate per-hypothesis entries or adjusted p-vectors are allowed;
        # a raw per-row vector of scalars length > 1 is not.
        if all(isinstance(x, (int, float)) and not isinstance(x, bool)
               for x in result) and len(result) > 1 and path.endswith("pvalues") is False:
            # only short aggregate vectors (e.g. holm_adjusted_pvalues) allowed
            if "holm_adjusted" not in path and "folds" not in path:
                raise SealedAccessError(
                    "raw per-row value vector at %s (len %d)" % (path, len(result)))
        else:
            # recurse into structured list elements (hypothesis dicts, etc.)
            for i, item in enumerate(result):
                assert_no_row_level(item, "%s[%d]" % (path, i))
