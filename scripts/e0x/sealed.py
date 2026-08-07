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
                       seed: int = 42) -> Optional[float]:
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

    def _macro_spearman(p):
        rho = []
        for c in set(ctx_arr):
            idx = np.where(ctx_arr == c)[0]
            if len(idx) < 3:
                continue
            a, b = true_delta[idx], p[idx]
            if np.std(a) == 0 or np.std(b) == 0:
                continue
            r, _ = spearmanr(a, b)
            if not np.isnan(r):
                rho.append(r)
        return float(np.mean(rho)) if rho else None

    obs = _macro_spearman(pred)
    if obs is None:
        return None
    count = 0
    for _ in range(n_perm):
        p_perm = pred.copy()
        for c in set(ctx_arr):
            idx = np.where(ctx_arr == c)[0]
            p_perm[idx] = rng.permutation(p_perm[idx])
        pm = _macro_spearman(p_perm)
        if pm is not None and pm >= obs:
            count += 1
    return (count + 1) / (n_perm + 1)


def verdict_from_aggregate(prereg: Dict, per_hypothesis: List[Dict],
                           holm_adj: List[Optional[float]]) -> Dict:
    """Compute the pre-registered GO/NO-GO verdict from the aggregate stats."""
    gates = (prereg.get("go_nogo") or {})
    eg = gates.get("effect_gate") or {}
    hc = gates.get("hard_constraints") or {}
    stats = {h["id"]: h for h in per_hypothesis}
    adj = {h["id"]: a for h, a in zip(per_hypothesis, holm_adj)}

    h1 = stats.get("H1_EFFECT_TRANSFER")
    h3 = stats.get("H3_LEGALITY")
    checks = {}
    if h1 is not None:
        checks["H1_macro_delta_spearman_ge"] = (
            h1.get("stat") is not None
            and h1["stat"] >= eg.get("macro_delta_spearman_ge", 0.25))
        checks["H1_holm_significant"] = (adj.get("H1_EFFECT_TRANSFER") is not None
                                         and adj["H1_EFFECT_TRANSFER"] <= 0.05)
        # Full pre-registered effect gate: sign accuracy AND top-10% enrichment
        # AND beat-the-strongest-nonfoundation-baseline must ALL hold, not just
        # the spearman + Holm pair.  These stats are carried on the H1 aggregate
        # by run_internal / the sealed evaluator.
        checks["H1_macro_sign_accuracy_ge"] = (
            h1.get("sign_accuracy") is not None
            and h1["sign_accuracy"] >= eg.get("macro_sign_accuracy_ge", 0.60))
        checks["H1_top10pct_enrichment_ge"] = (
            h1.get("top10pct_enrichment") is not None
            and h1["top10pct_enrichment"] >= eg.get("top10_enrichment_ge", 1.50))
        if eg.get("beat_strongest_nonfoundation_baseline", False):
            checks["H1_beat_abs_candidate"] = (
                h1.get("stat") is not None and h1.get("abs_candidate_spearman") is not None
                and h1["stat"] > h1["abs_candidate_spearman"])
    if h3 is not None:
        checks["H3_legality_ge_1"] = (h3.get("stat") is not None
                                      and h3["stat"] >= hc.get("legality_ge", 1.0))
    go = all(checks.values()) if checks else False
    return {
        "verdict": "GO" if go else "NO_GO",
        "checks": checks,
        "effect_gate": {k: v for k, v in eg.items() if k in
                        ("macro_delta_spearman_ge", "macro_sign_accuracy_ge",
                         "top10_enrichment_ge")},
        "hard_constraints": dict(hc),
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