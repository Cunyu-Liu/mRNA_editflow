"""O0-X closed measured-space optimization with R1-safe pool semantics.

The original O0-X implementation grouped rows by ``source_id`` alone.  Some
assets emit one row per endpoint for the *same* candidate, so that convention
could manufacture a size-two candidate pool out of two endpoint observations.
This module uses an explicitly versioned, endpoint-aware pool key and requires
at least two distinct measured candidate identities before ranking.

All metrics are ordinary-data diagnostics.  They do not read, infer, or score
sealed-final records.  Historical O0 artifacts remain historical; the R1
runner writes a new result namespace and refuses to overwrite an existing
result file.
"""
from __future__ import annotations

import json
import math
import hashlib
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from scripts.m4_sparse.ordinary_io import assert_ordinary_rows


METRIC_CONTRACT_VERSION = "o0x_r1_pool_safe_v1"
POOL_KEY_DEFINITION = (
    "benchmark|study|endpoint|context_id_or_assay_context|source_id"
)
CANDIDATE_IDENTITY_DEFINITION = (
    "normalized_candidate_sequence_else_canonical_edit_script_v1"
)
NDCG_GAIN_DEFINITION = (
    "per_pool_minmax_signed_delta_to_unit_interval_then_exp2_gain_v1"
)
TOPK_GAIN_DEFINITION = (
    "predicted_top_k_mean_minmax_gain_divided_by_oracle_top_k_mean_minmax_gain_v1"
)
SIGNED_MEAN_ENRICHMENT_STATUS = (
    "HOLD_UNSTABLE_FOR_SIGNED_EFFECTS_REQUIRES_PREREGISTERED_DEFINITION"
)
RANDOM_SEED_DEFINITION = "sha256_json_global_seed_and_canonical_pool_key_v1"


# ---------------------------------------------------------------------------
# pool identity and eligibility
# ---------------------------------------------------------------------------

def _text(value) -> str:
    return str(value).strip() if value is not None else ""


def _normalize_sequence(value) -> str:
    return _text(value).upper().replace("T", "U")


def _edit_signature(edit_list) -> str:
    """Canonicalize an edit script without assigning biological equivalence.

    Edit order is retained because a general insertion/deletion script can be
    order-sensitive.  Dictionary keys are sorted solely to make serialization
    deterministic.
    """
    if not isinstance(edit_list, list) or not edit_list:
        return ""
    return json.dumps(edit_list, sort_keys=True, separators=(",", ":"))


def candidate_identity(row: dict) -> Optional[Tuple[str, str]]:
    """Return the conservative identity used to count distinct candidates.

    A final candidate sequence is the primary identity.  This prevents two
    observations or two alternative edit-script encodings of the same final
    sequence from inflating pool size.  A canonical edit script is only a
    fallback for legacy rows without candidate sequence text.  ``candidate_id``
    alone is deliberately insufficient because it may identify an observation
    rather than a distinct intervention.
    """
    sequence = _normalize_sequence(row.get("candidate_sequence"))
    if sequence:
        return ("sequence", sequence)
    edits = _edit_signature(row.get("edit_list"))
    if edits:
        return ("edit_script", edits)
    return None


def pool_key_missing_atoms(row: dict) -> Tuple[str, ...]:
    """Return missing/conflicting atoms that make a safe pool key impossible."""
    missing = []
    if not _text(row.get("benchmark")):
        missing.append("benchmark")
    if not _text(row.get("study")):
        missing.append("study")
    if not _text(row.get("endpoint")):
        missing.append("endpoint")
    if not _text(row.get("source_id")):
        missing.append("source_id")
    context_id = _text(row.get("context_id"))
    assay_context = _text(row.get("assay_context"))
    if not context_id and not assay_context:
        missing.append("context")
    elif context_id and assay_context and context_id != assay_context:
        missing.append("conflicting_context")
    return tuple(missing)


def pool_key(row: dict) -> Optional[Tuple[str, str, str, str, str]]:
    """Return the R1 pool key, or ``None`` when a safe key is unavailable.

    The key is endpoint-aware and study-aware by construction.  Missing
    ``source_id`` or endpoint rows are not coalesced under a synthetic key;
    they are excluded and counted by :func:`run_benchmark_o0x`.
    """
    if pool_key_missing_atoms(row):
        return None
    source_id = _text(row.get("source_id"))
    endpoint = _text(row.get("endpoint"))
    study = _text(row.get("study"))
    benchmark = _text(row.get("benchmark"))
    context = _text(row.get("context_id") or row.get("assay_context"))
    return benchmark, study, endpoint, context, source_id


def _canonical_row_sort_key(row: dict) -> str:
    """Stable duplicate representative independent of incoming row order."""
    return json.dumps(row, sort_keys=True, separators=(",", ":"), default=str)


def _delta_values(rows: Sequence[dict]) -> np.ndarray:
    values = np.asarray([r.get("delta") for r in rows], dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("O0-X ranking requires finite measured delta values")
    return values


def _prepare_pool(entries: Sequence[Tuple[int, dict]]) -> dict:
    """Validate and de-duplicate one endpoint-aware measured candidate pool.

    Byte-equivalent duplicate candidate observations with the same measured
    delta are collapsed deterministically.  Conflicting deltas for one
    candidate identity, inconsistent source sequences, or missing candidate
    identities invalidate the pool rather than silently averaging provenance-
    ambiguous records.
    """
    raw_n = len(entries)
    assert_ordinary_rows(
        (row for _, row in entries), purpose="O0 candidate pool"
    )
    source_sequence_values = [
        _normalize_sequence(row.get("source_sequence")) for _, row in entries
    ]
    if any(not sequence for sequence in source_sequence_values):
        return {
            "status": "EXCLUDED_MISSING_SOURCE_SEQUENCE",
            "raw_n": raw_n, "distinct_n": 0,
            "duplicate_rows": 0, "entries": [],
        }
    source_sequences = set(source_sequence_values)
    if len(source_sequences) > 1:
        return {
            "status": "EXCLUDED_SOURCE_SEQUENCE_MISMATCH",
            "raw_n": raw_n,
            "distinct_n": 0,
            "duplicate_rows": 0,
            "entries": [],
        }

    try:
        source_values = np.asarray(
            [float(row.get("source_value")) for _, row in entries], dtype=float
        )
    except (TypeError, ValueError):
        source_values = np.asarray([], dtype=float)
    if (source_values.size != raw_n or not np.all(np.isfinite(source_values))):
        return {
            "status": "EXCLUDED_MISSING_OR_NONFINITE_SOURCE_VALUE",
            "raw_n": raw_n, "distinct_n": 0,
            "duplicate_rows": 0, "entries": [],
        }
    if not np.allclose(source_values, source_values[0], rtol=1e-7, atol=1e-10):
        return {
            "status": "EXCLUDED_SOURCE_VALUE_CONFLICT",
            "raw_n": raw_n, "distinct_n": 0,
            "duplicate_rows": 0, "entries": [],
        }

    by_candidate: Dict[Tuple[str, str], List[Tuple[int, dict]]] = defaultdict(list)
    for index, row in entries:
        identity = candidate_identity(row)
        if identity is None:
            return {
                "status": "EXCLUDED_MISSING_CANDIDATE_IDENTITY",
                "raw_n": raw_n,
                "distinct_n": 0,
                "duplicate_rows": 0,
                "entries": [],
            }
        by_candidate[identity].append((index, row))

    unique_entries: List[Tuple[int, dict]] = []
    duplicate_rows = raw_n - len(by_candidate)
    for identity in sorted(by_candidate):
        identity_entries = sorted(
            by_candidate[identity],
            key=lambda item: (_canonical_row_sort_key(item[1]), item[0]),
        )
        duplicate_deltas = _delta_values([row for _, row in identity_entries])
        if not np.allclose(
            duplicate_deltas,
            duplicate_deltas[0],
            rtol=1e-7,
            atol=1e-10,
            equal_nan=False,
        ):
            return {
                "status": "EXCLUDED_CONFLICTING_DUPLICATE_CANDIDATE",
                "raw_n": raw_n,
                "distinct_n": len(by_candidate),
                "duplicate_rows": duplicate_rows,
                "entries": [],
            }
        unique_entries.append(identity_entries[0])

    distinct_n = len(unique_entries)
    if distinct_n < 2:
        return {
            "status": "EXCLUDED_SINGLETON_DISTINCT_CANDIDATE",
            "raw_n": raw_n,
            "distinct_n": distinct_n,
            "duplicate_rows": duplicate_rows,
            "entries": unique_entries,
        }

    deltas = _delta_values([row for _, row in unique_entries])
    if float(np.max(deltas) - np.min(deltas)) <= 0.0:
        return {
            "status": "EXCLUDED_DEGENERATE_EQUAL_RELEVANCE",
            "raw_n": raw_n,
            "distinct_n": distinct_n,
            "duplicate_rows": duplicate_rows,
            "entries": unique_entries,
        }

    return {
        "status": "ELIGIBLE",
        "raw_n": raw_n,
        "distinct_n": distinct_n,
        "duplicate_rows": duplicate_rows,
        "entries": unique_entries,
    }


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def minmax_relevance_gain(values: np.ndarray) -> Optional[np.ndarray]:
    """Map signed deltas to a prospective, bounded within-pool gain.

    This is intentionally *not* raw signed relevance in the standard NDCG
    gain.  Equal-relevance pools have no identifiable ranking task and return
    ``None`` instead of an artificial score of zero or one.
    """
    values = np.asarray(values, dtype=float)
    if values.size < 2 or not np.all(np.isfinite(values)):
        return None
    lo, hi = float(values.min()), float(values.max())
    if hi <= lo:
        return None
    return (values - lo) / (hi - lo)


def ndcg_at_k(rel: np.ndarray, k: int) -> Optional[float]:
    """NDCG@k using the explicitly versioned within-pool gain transform.

    ``rel`` is the signed measured delta in predicted rank order.  The values
    are transformed to unit-interval gains before applying exponential NDCG.
    ``None`` means the metric is scientifically undefined (fewer than two
    records, non-finite values, equal relevance, or non-positive ``k``).
    """
    rel = np.asarray(rel, dtype=float)
    effective_k = min(int(k), len(rel))
    gains = minmax_relevance_gain(rel)
    if effective_k <= 0 or gains is None:
        return None
    dcg = sum(
        (2 ** gains[i] - 1) / math.log2(i + 2)
        for i in range(effective_k)
    )
    ideal = np.sort(gains)[::-1]
    idcg = sum(
        (2 ** ideal[i] - 1) / math.log2(i + 2)
        for i in range(effective_k)
    )
    return float(dcg / idcg) if idcg > 0 else None


def top_decile_recall(deltas: np.ndarray, order: np.ndarray) -> Optional[float]:
    n = len(deltas)
    if n < 2:
        return None
    k = max(1, int(math.ceil(n * 0.10)))
    true_top = set(np.argsort(-np.asarray(deltas), kind="stable")[:k].tolist())
    pred_top = set(order[:k].tolist())
    return len(pred_top & true_top) / k


def _validate_order(order: np.ndarray, n: int) -> np.ndarray:
    order = np.asarray(order, dtype=int)
    if order.shape != (n,) or set(order.tolist()) != set(range(n)):
        raise ValueError("order must be a complete permutation of candidate indices")
    return order


def rank_metrics(deltas: np.ndarray, order: np.ndarray, k: int = 10) -> dict:
    """Compute versioned metrics for one eligible, non-degenerate pool."""
    deltas = np.asarray(deltas, dtype=float)
    n = len(deltas)
    order = _validate_order(order, n)
    gains = minmax_relevance_gain(deltas)
    if gains is None:
        raise ValueError("rank_metrics requires >=2 finite, non-degenerate deltas")

    requested_k = int(k)
    effective_k = min(max(requested_k, 0), n)
    k_covers_pool = effective_k >= n
    rel_pred = deltas[order]
    ndcg = ndcg_at_k(rel_pred, effective_k)
    tdr = top_decile_recall(deltas, order)

    # A top-k selection gain is uninformative when k covers the entire pool.
    # Otherwise normalize bounded gains by the oracle top-k mean.  This avoids
    # division by a near-zero or negative signed-delta mean.
    topk_gain = None
    topk_gain_status = "UNDEFINED_K_COVERS_POOL" if k_covers_pool else "DEFINED"
    if effective_k <= 0:
        topk_gain_status = "UNDEFINED_NONPOSITIVE_K"
    elif not k_covers_pool:
        oracle_gain = np.sort(gains)[::-1][:effective_k]
        denom = float(np.mean(oracle_gain))
        if denom > 0:
            topk_gain = float(np.mean(gains[order[:effective_k]]) / denom)
            topk_gain = float(np.clip(topk_gain, 0.0, 1.0))
        else:
            topk_gain_status = "UNDEFINED_ZERO_ORACLE_GAIN"

    mn, mx = float(deltas.min()), float(deltas.max())
    best_pred_true = float(deltas[order[0]])
    regret = (mx - best_pred_true) / (mx - mn)
    return {
        "n": n,
        "requested_k": requested_k,
        "effective_k": effective_k,
        "k_covers_pool": k_covers_pool,
        "ndcg_at_k": ndcg,
        "ndcg_gain_definition": NDCG_GAIN_DEFINITION,
        "top_decile_recall": tdr,
        "oracle_normalized_gain_at_k": topk_gain,
        "oracle_normalized_gain_status": topk_gain_status,
        # Compatibility alias.  Its R1 definition is explicit in every result
        # and must not be compared numerically with the historical signed-mean
        # ratio called enrichment_at_k.
        "enrichment_at_k": topk_gain,
        "enrichment_at_k_definition": TOPK_GAIN_DEFINITION,
        # Raw mean(selected signed delta) / mean(all signed delta) reverses
        # direction on all-negative pools and is singular near a zero mean.
        # It is retained only as an explicit HOLD marker, never a number.
        "signed_mean_enrichment_at_k": None,
        "signed_mean_enrichment_status": SIGNED_MEAN_ENRICHMENT_STATUS,
        "normalized_regret": regret,
    }


# ---------------------------------------------------------------------------
# search strategies over a single source pool (closed, measured)
# ---------------------------------------------------------------------------

def _greedy_edit_order(records: List[dict]) -> np.ndarray:
    """Greedy distance heuristic: closest edits (fewest edit_count) first."""
    return np.argsort(
        np.asarray([r["edit_count"] for r in records]), kind="stable"
    ).astype(int)


def _beam_order(records: List[dict], delta_score: np.ndarray,
                beam_width: int = 8) -> np.ndarray:
    """Return a deterministic scorer order over the closed measured pool."""
    n = len(records)
    if n == 0:
        return np.array([], dtype=int)
    if len(delta_score) != n:
        raise ValueError("delta_score length must equal the distinct candidate count")
    bw = max(1, min(beam_width, n))
    order = np.argsort(-np.asarray(delta_score), kind="stable")
    kept = list(order[:bw])
    rest = list(order[bw:])
    return np.array(kept + rest, dtype=int)


def _random_order(records: List[dict], seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.permutation(len(records))


def pool_seed(global_seed: int, key: Tuple[str, ...]) -> int:
    """Derive an independent deterministic seed from a canonical pool key."""
    payload = json.dumps(
        [int(global_seed), *key], ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def run_source_headline(records: List[dict], pred_score: Optional[np.ndarray],
                        k: int = 10, beam_width: int = 8,
                        random_seed: int = 0) -> Dict:
    """Run O0-X strategies on one already-qualified distinct candidate pool.

    Direct singleton or equal-relevance calls return an explicit exclusion
    status and no strategy metrics.  The benchmark runner performs the fuller
    identity/provenance validation before calling this function.
    """
    assert_ordinary_rows(records, purpose="O0 source headline")
    if pred_score is not None and len(pred_score) != len(records):
        raise ValueError("pred_score length must equal distinct candidate count")

    # Canonicalize before any tied sort or random permutation.  This makes all
    # reported metrics invariant to JSONL row order while preserving alignment
    # with externally supplied prediction scores.
    canonical = sorted(
        range(len(records)),
        key=lambda i: (
            candidate_identity(records[i]) or ("missing", ""),
            _canonical_row_sort_key(records[i]),
        ),
    )
    records = [records[i] for i in canonical]
    if pred_score is not None:
        pred_score = np.asarray(pred_score, dtype=float)[canonical]

    deltas = _delta_values(records) if records else np.asarray([], dtype=float)
    n = len(records)
    if n < 2:
        return {
            "status": "EXCLUDED_SINGLETON_DISTINCT_CANDIDATE",
            "n_candidates": n,
            "strategies": {},
        }
    if minmax_relevance_gain(deltas) is None:
        return {
            "status": "EXCLUDED_DEGENERATE_EQUAL_RELEVANCE",
            "n_candidates": n,
            "strategies": {},
        }
    out = {"status": "ELIGIBLE", "n_candidates": n, "strategies": {}}

    exact = np.argsort(-deltas, kind="stable").astype(int)
    out["strategies"]["exact_enumeration"] = {
        **rank_metrics(deltas, exact, k),
        "query_count": n, "forward_equivalents": 0,
    }

    rnd = _random_order(records, random_seed)
    out["strategies"]["random_legal"] = {
        **rank_metrics(deltas, rnd, k),
        "query_count": n, "forward_equivalents": 0,
    }

    greedy = _greedy_edit_order(records)
    out["strategies"]["greedy"] = {
        **rank_metrics(deltas, greedy, k),
        "query_count": n, "forward_equivalents": 0,
    }

    beam_scorer = (
        np.asarray(pred_score, dtype=float)
        if pred_score is not None
        else -np.asarray([r["edit_count"] for r in records], dtype=float)
    )
    beam = _beam_order(records, beam_scorer, beam_width)
    out["strategies"]["beam"] = {
        **rank_metrics(deltas, beam, k),
        "query_count": n, "forward_equivalents": 0,
    }

    if pred_score is not None:
        rerank = np.argsort(-pred_score, kind="stable").astype(int)
        out["strategies"]["sparseeditformer_rerank"] = {
            **rank_metrics(deltas, rerank, k),
            "query_count": 0, "forward_equivalents": n,
        }

    return out


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------

def _macro(key: str, metrics: List[dict]) -> Optional[float]:
    vals = [m[key] for m in metrics if m.get(key) is not None]
    return float(np.mean(vals)) if vals else None


def aggregate_strategy(strategy_metrics: List[dict]) -> dict:
    """Source-macro over a list of eligible per-source strategy metrics."""
    return {
        "n_sources": len(strategy_metrics),
        "n_records": int(sum(m["n"] for m in strategy_metrics)),
        "macro_ndcg_at_k": _macro("ndcg_at_k", strategy_metrics),
        "macro_top_decile_recall": _macro("top_decile_recall", strategy_metrics),
        "macro_oracle_normalized_gain_at_k": _macro(
            "oracle_normalized_gain_at_k", strategy_metrics
        ),
        "macro_enrichment_at_k": _macro("enrichment_at_k", strategy_metrics),
        "macro_signed_mean_enrichment_at_k": None,
        "signed_mean_enrichment_status": SIGNED_MEAN_ENRICHMENT_STATUS,
        "macro_normalized_regret": _macro("normalized_regret", strategy_metrics),
        "n_pools_where_k_covers_pool": int(
            sum(bool(m.get("k_covers_pool")) for m in strategy_metrics)
        ),
        "total_query_count": int(
            sum(m.get("query_count", 0) for m in strategy_metrics)
        ),
        "total_forward_equivalents": int(
            sum(m.get("forward_equivalents", 0) for m in strategy_metrics)
        ),
    }


def _legacy_endpoint_collision_count(rows: Sequence[dict]) -> int:
    """Count source IDs that the historical source-only key mixed by endpoint."""
    endpoints_by_source = defaultdict(set)
    for row in rows:
        source_id = _text(row.get("source_id"))
        if source_id:
            endpoints_by_source[(
                _text(row.get("benchmark")),
                _text(row.get("study")),
                source_id,
            )].add((
                _text(row.get("endpoint")),
                _text(row.get("context_id") or row.get("assay_context")),
            ))
    return int(sum(len(contexts) > 1 for contexts in endpoints_by_source.values()))


def run_benchmark_o0x(rows: List[dict], folds: List[dict],
                      predict_fn=None, k: int = 10,
                      beam_width: int = 8, random_seed: int = 0,
                      device=None) -> dict:
    """Run endpoint-safe O0-X over S4 folds for one ordinary benchmark.

    The optional ``device`` parameter is retained for API compatibility; all
    measured-pool metrics themselves are device-independent.
    """
    del device
    assert_ordinary_rows(rows, purpose="O0 benchmark")
    for fold in sorted(folds, key=lambda item: _text(item.get("held_out_study"))):
        assert_ordinary_rows(fold.get("test", []), purpose="O0 fold test")
    bench = rows[0]["benchmark"] if rows else "?"
    per_study = {}
    for fold in folds:
        held = fold["held_out_study"]
        test = fold["test"]
        pools: Dict[Tuple[str, str, str, str, str], List[Tuple[int, dict]]] = defaultdict(list)
        rows_missing_pool_key = 0
        missing_pool_atom_counts = defaultdict(int)
        for index, row in enumerate(test):
            key = pool_key(row)
            if key is None:
                rows_missing_pool_key += 1
                for atom in pool_key_missing_atoms(row):
                    missing_pool_atom_counts[atom] += 1
                continue
            pools[key].append((index, row))

        pred_all = None
        if predict_fn is not None and test:
            predicted = predict_fn(test)
            if predicted is not None:
                pred_all = np.asarray(predicted, dtype=float)
                if pred_all.shape != (len(test),):
                    raise ValueError("predict_fn must return one scalar per test row")

        strategy_buckets = defaultdict(list)
        status_counts = defaultdict(int)
        duplicate_rows_collapsed = 0
        headline_sources = 0
        headline_records = 0
        for key in sorted(pools):
            entries = pools[key]
            prepared = _prepare_pool(entries)
            status_counts[prepared["status"]] += 1
            duplicate_rows_collapsed += prepared["duplicate_rows"]
            if prepared["status"] != "ELIGIBLE":
                continue

            unique_entries = prepared["entries"]
            records = [row for _, row in unique_entries]
            pred_score = None
            if pred_all is not None:
                pred_score = np.asarray(
                    [pred_all[index] for index, _ in unique_entries], dtype=float
                )
            source = run_source_headline(
                records, pred_score, k, beam_width,
                pool_seed(random_seed, key),
            )
            if source["status"] != "ELIGIBLE":
                # Defensive: _prepare_pool should already have classified this.
                status_counts[source["status"]] += 1
                status_counts["ELIGIBLE"] -= 1
                continue
            headline_sources += 1
            headline_records += len(records)
            for strategy, metrics in source["strategies"].items():
                strategy_buckets[strategy].append(metrics)

        per_study[held] = {
            "study": held,
            "pool_key_definition": POOL_KEY_DEFINITION,
            "candidate_identity_definition": CANDIDATE_IDENTITY_DEFINITION,
            "n_pools_total": len(pools),
            "n_rows_missing_pool_key": rows_missing_pool_key,
            "missing_pool_atom_counts": dict(sorted(missing_pool_atom_counts.items())),
            "n_singleton_sources": status_counts[
                "EXCLUDED_SINGLETON_DISTINCT_CANDIDATE"
            ],
            "n_degenerate_equal_relevance_sources": status_counts[
                "EXCLUDED_DEGENERATE_EQUAL_RELEVANCE"
            ],
            "n_invalid_source_sequence_mismatch": status_counts[
                "EXCLUDED_SOURCE_SEQUENCE_MISMATCH"
            ],
            "n_invalid_missing_candidate_identity": status_counts[
                "EXCLUDED_MISSING_CANDIDATE_IDENTITY"
            ],
            "n_invalid_conflicting_duplicate_candidate": status_counts[
                "EXCLUDED_CONFLICTING_DUPLICATE_CANDIDATE"
            ],
            "n_invalid_missing_source_sequence": status_counts[
                "EXCLUDED_MISSING_SOURCE_SEQUENCE"
            ],
            "n_invalid_source_value": (
                status_counts["EXCLUDED_MISSING_OR_NONFINITE_SOURCE_VALUE"]
                + status_counts["EXCLUDED_SOURCE_VALUE_CONFLICT"]
            ),
            "n_invalid_missing_or_nonfinite_source_value": status_counts[
                "EXCLUDED_MISSING_OR_NONFINITE_SOURCE_VALUE"
            ],
            "n_invalid_source_value_conflict": status_counts[
                "EXCLUDED_SOURCE_VALUE_CONFLICT"
            ],
            "n_duplicate_candidate_rows_collapsed": duplicate_rows_collapsed,
            "n_legacy_source_ids_spanning_endpoints": _legacy_endpoint_collision_count(test),
            "n_headline_sources": headline_sources,
            "n_headline_distinct_candidate_records": headline_records,
            "pool_status_counts": dict(sorted(status_counts.items())),
            "strategies": {
                strategy: aggregate_strategy(metrics)
                for strategy, metrics in strategy_buckets.items()
            },
        }

    strategy_names = sorted({
        strategy
        for study_result in per_study.values()
        for strategy in study_result["strategies"]
    })
    study_macro = {}
    for strategy in strategy_names:
        aggregates = [
            study_result["strategies"][strategy]
            for study_result in per_study.values()
            if strategy in study_result["strategies"]
        ]
        study_macro[strategy] = {
            "macro_over_studies_ndcg_at_k": _macro("macro_ndcg_at_k", aggregates),
            "macro_over_studies_top_decile_recall": _macro(
                "macro_top_decile_recall", aggregates
            ),
            "macro_over_studies_oracle_normalized_gain_at_k": _macro(
                "macro_oracle_normalized_gain_at_k", aggregates
            ),
            "macro_over_studies_enrichment_at_k": _macro(
                "macro_enrichment_at_k", aggregates
            ),
            "macro_over_studies_normalized_regret": _macro(
                "macro_normalized_regret", aggregates
            ),
            "n_studies_with_strategy": len(aggregates),
            "n_pools_where_k_covers_pool": int(sum(
                aggregate.get("n_pools_where_k_covers_pool", 0)
                for aggregate in aggregates
            )),
            "total_query_count": int(sum(
                aggregate.get("total_query_count", 0)
                for aggregate in aggregates
            )),
            "total_forward_equivalents": int(sum(
                aggregate.get("total_forward_equivalents", 0)
                for aggregate in aggregates
            )),
        }

    return {
        "benchmark": bench,
        "metric_contract_version": METRIC_CONTRACT_VERSION,
        "pool_key_definition": POOL_KEY_DEFINITION,
        "candidate_identity_definition": CANDIDATE_IDENTITY_DEFINITION,
        "ndcg_gain_definition": NDCG_GAIN_DEFINITION,
        "topk_gain_definition": TOPK_GAIN_DEFINITION,
        "signed_mean_enrichment_status": SIGNED_MEAN_ENRICHMENT_STATUS,
        "random_seed_definition": RANDOM_SEED_DEFINITION,
        "publication_grouping_qualified": False,
        "qualification_status": "DEVELOPMENT_ONLY_UNQUALIFIED_GROUPING",
        "k": k,
        "n_records": len(rows),
        "per_study": per_study,
        "study_macro": study_macro,
        "summary": {
            "n_pools_total": int(sum(
                result["n_pools_total"] for result in per_study.values()
            )),
            "n_rows_missing_pool_key": int(sum(
                result["n_rows_missing_pool_key"] for result in per_study.values()
            )),
            "missing_pool_atom_counts": {
                atom: int(sum(
                    result["missing_pool_atom_counts"].get(atom, 0)
                    for result in per_study.values()
                ))
                for atom in sorted({
                    atom
                    for result in per_study.values()
                    for atom in result["missing_pool_atom_counts"]
                })
            },
            "n_invalid_missing_source_sequence": int(sum(
                result["n_invalid_missing_source_sequence"]
                for result in per_study.values()
            )),
            "n_invalid_source_sequence_mismatch": int(sum(
                result["n_invalid_source_sequence_mismatch"]
                for result in per_study.values()
            )),
            "n_invalid_missing_or_nonfinite_source_value": int(sum(
                result["n_invalid_missing_or_nonfinite_source_value"]
                for result in per_study.values()
            )),
            "n_invalid_source_value_conflict": int(sum(
                result["n_invalid_source_value_conflict"]
                for result in per_study.values()
            )),
            "n_singleton_sources": int(sum(
                result["n_singleton_sources"] for result in per_study.values()
            )),
            "n_degenerate_equal_relevance_sources": int(sum(
                result["n_degenerate_equal_relevance_sources"]
                for result in per_study.values()
            )),
            "n_duplicate_candidate_rows_collapsed": int(sum(
                result["n_duplicate_candidate_rows_collapsed"]
                for result in per_study.values()
            )),
            "n_legacy_source_ids_spanning_endpoints": int(sum(
                result["n_legacy_source_ids_spanning_endpoints"]
                for result in per_study.values()
            )),
            "n_headline_sources": int(sum(
                result["n_headline_sources"] for result in per_study.values()
            )),
            "n_headline_distinct_candidate_records": int(sum(
                result["n_headline_distinct_candidate_records"]
                for result in per_study.values()
            )),
        },
    }
