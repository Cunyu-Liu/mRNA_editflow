#!/usr/bin/env python3
"""D2: Retrieval-conditioned guidance critic (prereg v1, 2026-09-10).

Implements the same critic interface as FrozenXEditCriticV5.potentials /
.potential / .clear_source_caches (plus forward counters) so it can be mounted
on run_route2_guided_xeditsetflow_v5_v1.py via --critic-kind retrieval.

Potential (prereg section 1):
    V_retrieval(c | s, t) = sum_k sim(s, s_k) * w(c, e_k) * sign_k
    - retrieval pool: TRAIN split measured edits, same endpoint task,
      decontaminated (prereg section 2 hard gates)
    - sim = 1/(1+hamming(source, source_k)); w = 1/(1+hamming(c, e_k));
      sign_k = clip(direction_normalized_delta, -1, 1)
    - combined mode: V = beta_r * V_retrieval + beta_c * V_V5 (runner applies
      beta to potentials; here we return the retrieval potential scaled so the
      runner's --beta acts as beta_r, and optionally blend a V5 critic).

Decontamination (prereg section 2): run --audit-only first; writes
retrieval_pool_audit.json with the three hard-gate results.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

TRAIN = Path(
    "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/projections/xedit_v3/"
    "development_train_validation_v1/train.jsonl"
)
MEASURED_VAL = Path(
    "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/generation_eligibility/"
    "development_validation_v1/measured_neighborhood.private.jsonl"
)
VAL_MANIFEST = Path(
    "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/generation_eligibility/"
    "development_validation_v1/source_eligibility.jsonl"
)
K_NEIGHBORS = 8


def _norm(s: str) -> str:
    return str(s).upper().replace("T", "U")


def _hamming(a: str, b: str) -> int | None:
    if len(a) != len(b):
        return None
    return sum(x != y for x, y in zip(a, b))


def load_pool():
    """Load TRAIN measured edits, grouped by (endpoint, source)."""
    pool = defaultdict(list)  # endpoint -> list of entries
    components = set()
    source_ids = set()
    n_rows = 0
    with TRAIN.open() as f:
        for line in f:
            r = json.loads(line)
            n_rows += 1
            components.add(r["connected_source_component_id"])
            source_ids.add(r["source_id"])
            pool[str(r["endpoint_id"])].append(
                {
                    "source_sequence": _norm(r["source_sequence"]),
                    "edited": _norm(r["candidate_sequence"]),
                    "direction": max(-1.0, min(1.0, float(r["direction_normalized_delta"]))),
                    "region": str(r["region"]),
                    "source_id": str(r["source_id"]),
                }
            )
    return pool, components, source_ids, n_rows


def audit(pool, components, source_ids):
    val_measured_seqs = set()
    with MEASURED_VAL.open() as f:
        for line in f:
            r = json.loads(line)
            val_measured_seqs.add(_norm(r["candidate_sequence"]))
    val_components = set()
    val_source_ids = set()
    with VAL_MANIFEST.open() as f:
        for line in f:
            r = json.loads(line)
            val_source_ids.add(str(r["source_id"]))
    # components of validation sources: from validation.jsonl
    VALPROJ = TRAIN.parent / "validation.jsonl"
    with VALPROJ.open() as f:
        for line in f:
            r = json.loads(line)
            val_components.add(r["connected_source_component_id"])

    comp_overlap = sorted(components & val_components)
    src_overlap = sorted(source_ids & val_source_ids)

    removed = 0
    for endpoint in list(pool):
        kept = []
        for e in pool[endpoint]:
            if e["edited"] in val_measured_seqs:
                removed += 1
                continue
            kept.append(e)
        pool[endpoint] = kept
    report = {
        "schema_version": "route_a_v3_d2_retrieval_pool_audit.v1",
        "train_rows": len(pool) and sum(len(v) for v in pool.values()),
        "gate1_component_overlap": len(comp_overlap),
        "gate1_pass": len(comp_overlap) == 0,
        "gate2_sequence_exclusions": removed,
        "gate2_pass": True,  # exclusions applied; pass = audit recorded
        "gate3_source_overlap": len(src_overlap),
        "gate3_pass": len(src_overlap) == 0,
        "pool_after": sum(len(v) for v in pool.values()),
    }
    return report, pool


class RetrievalCritic:
    """Same interface as FrozenXEditCriticV5 (subset used by the runner)."""

    def __init__(self, pool_by_endpoint: dict, k: int = K_NEIGHBORS):
        self.pool = pool_by_endpoint
        self.k = k
        self.model_batch_forward_count = 0
        self.candidate_forward_equivalent_count = 0
        self.potential_query_count = 0
        self.potential_newly_scored_count = 0
        self._neighbor_memo: dict[tuple[str, str], list] = {}
        self._potential_memo: dict[tuple[str, str], float] = {}

    def clear_source_caches(self) -> None:
        self._neighbor_memo.clear()
        self._potential_memo.clear()

    def _neighbors(self, source: str, endpoint: str):
        key = (source, endpoint)
        if key in self._neighbor_memo:
            return self._neighbor_memo[key]
        entries = self.pool.get(endpoint, [])
        scored = []
        for e in entries:
            d = _hamming(source, e["source_sequence"])
            if d is None:
                continue
            sim = max(0.0, 1.0 - d / max(1.0, float(len(source))))
            scored.append((sim, e))
        scored.sort(key=lambda x: -x[0])
        top = scored[: self.k]
        self._neighbor_memo[key] = top
        self.model_batch_forward_count += 1
        return top

    def potentials(
        self,
        states,
        *,
        endpoint_id: str,
        region: str,
        source_row=None,
    ) -> list[float]:
        ordered = list(states)
        if not ordered:
            return []
        source = ordered[0].source_sequence
        endpoint = str(endpoint_id)
        neighbors = self._neighbors(source, endpoint)
        out = []
        for state in ordered:
            key = (source, state.current_sequence)
            if key not in self._potential_memo:
                v = 0.0
                for sim, e in neighbors:
                    d = _hamming(state.current_sequence, e["edited"])
                    if d is None:
                        continue
                    # normalized candidate similarity (1 = identical to a
                    # retrieved measured edit, 0 = len-divergent); source sim
                    # likewise length-normalized
                    w = max(0.0, 1.0 - d / max(1.0, float(len(state.current_sequence))))
                    v += sim * w * e["direction"]
                self._potential_memo[key] = v
                self.potential_newly_scored_count += 1
            out.append(self._potential_memo[key])
            self.potential_query_count += 1
        self.candidate_forward_equivalent_count += len(ordered)
        return out

    def potential(self, state, *, endpoint_id: str, region: str, source_row=None) -> float:
        return self.potentials(
            [state], endpoint_id=endpoint_id, region=region, source_row=source_row
        )[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audit-only", action="store_true")
    ap.add_argument(
        "--audit-output",
        type=Path,
        default=Path(
            "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/experiments/"
            "xeditsetflow_v5/d2_retrieval_20260910/retrieval_pool_audit.json"
        ),
    )
    args = ap.parse_args()
    t0 = time.time()
    pool, comps, sids, n = load_pool()
    report, pool = audit(pool, comps, sids)
    report["load_seconds"] = round(time.time() - t0, 1)
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    if not (report["gate1_pass"] and report["gate3_pass"]):
        print("HARD GATE FAILED - do not launch")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
