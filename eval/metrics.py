"""Offline evaluation metrics for mRNA-EditFlow.

All metrics are implemented with the Python standard library and numpy only.
They accept ``MRNARecord`` instances, dictionaries with the same fields, or raw
RNA strings where a full-sequence metric is meaningful.
"""
from __future__ import annotations

import math
from collections import Counter
from itertools import product
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np

from mrna_editflow.core.constants import (
    CODON_TABLE,
    START_CODON,
    STOP_CODONS,
    SYNONYMOUS_CODONS,
    is_valid_cds,
    translate,
)

RNA_ALPHABET = "ACGU"
RNA_SET = set(RNA_ALPHABET)
ALL_CODONS = ["".join(p) for p in product(RNA_ALPHABET, repeat=3)]
DEFAULT_MAX_PAIRWISE_DIVERSITY_PAIRS = 64


def normalize_rna(seq: str) -> str:
    return "".join(str(seq or "").upper().replace("T", "U").split())


def _valid_only(seq: str) -> str:
    return "".join(ch for ch in normalize_rna(seq) if ch in RNA_SET)


def _field(record: object, name: str, default: object = "") -> object:
    if isinstance(record, Mapping):
        return record.get(name, default)
    return getattr(record, name, default)


def five_utr_of(record: object) -> str:
    if isinstance(record, str):
        return ""
    return normalize_rna(str(_field(record, "five_utr", "")))


def cds_of(record: object) -> str:
    if isinstance(record, str):
        return ""
    return normalize_rna(str(_field(record, "cds", "")))


def three_utr_of(record: object) -> str:
    if isinstance(record, str):
        return ""
    return normalize_rna(str(_field(record, "three_utr", "")))


def sequence_of(record: object) -> str:
    if isinstance(record, str):
        return normalize_rna(record)
    seq = _field(record, "seq", None)
    if isinstance(seq, str):
        return normalize_rna(seq)
    return five_utr_of(record) + cds_of(record) + three_utr_of(record)


def _safe_mean(values: Sequence[float], default: float = 0.0) -> float:
    if not values:
        return float(default)
    return float(np.mean(np.asarray(values, dtype=float)))


def _safe_fraction(flags: Sequence[bool]) -> float:
    if not flags:
        return 0.0
    return float(np.mean(np.asarray(flags, dtype=float)))


def gc_fraction(seq_or_record: object) -> float:
    seq = _valid_only(sequence_of(seq_or_record))
    if not seq:
        return 0.0
    return float((seq.count("G") + seq.count("C")) / len(seq))


def _kozak_score(record: object) -> float:
    five = _valid_only(five_utr_of(record))
    cds = _valid_only(cds_of(record))
    context = five[-6:] + cds[:6]
    aug = context.find(START_CODON, max(0, len(five[-6:]) - 1))
    if aug < 0:
        aug = len(five[-6:])
    score = 0.0
    denom = 0.0
    if aug >= 3:
        score += 1.0 if context[aug - 3] in ("A", "G") else 0.0
        denom += 1.0
    if aug + 3 < len(context):
        score += 1.0 if context[aug + 3] == "G" else 0.0
        denom += 1.0
    return float(score / denom) if denom else 0.0


def _count_overlapping(seq: str, motif: str) -> int:
    count = 0
    start = 0
    while motif:
        idx = seq.find(motif, start)
        if idx < 0:
            return count
        count += 1
        start = idx + 1
    return count


def codons_of(cds: str, include_incomplete: bool = False) -> list[str]:
    cds = _valid_only(cds)
    limit = len(cds) if include_incomplete else len(cds) - len(cds) % 3
    return [cds[i:i + 3] for i in range(0, limit, 3) if len(cds[i:i + 3]) == 3]


def legality_metrics(records: Iterable[object]) -> dict:
    rows = list(records)
    seq_valid = []
    invalid_fracs = []
    valid_cds = []
    frame_len = []
    start_ok = []
    terminal_stop = []
    internal_stop = []
    for r in rows:
        raw_seq = sequence_of(r)
        clean_seq = _valid_only(raw_seq)
        seq_valid.append(bool(raw_seq) and len(clean_seq) == len(raw_seq))
        invalid_fracs.append(1.0 - len(clean_seq) / max(1, len(raw_seq)))
        cds = _valid_only(cds_of(r))
        has_cds = bool(cds)
        frame_len.append(has_cds and len(cds) % 3 == 0)
        start_ok.append(has_cds and cds.startswith(START_CODON))
        terminal_stop.append(has_cds and len(cds) >= 3 and cds[-3:] in STOP_CODONS)
        prot = translate(cds) if has_cds else ""
        internal_stop.append("*" in prot[:-1])
        valid_cds.append(is_valid_cds(cds) if has_cds else False)
    legal = [a and b for a, b in zip(seq_valid, valid_cds)]
    return {
        "n": len(rows),
        "valid_sequence_fraction": _safe_fraction(seq_valid),
        "valid_cds_fraction": _safe_fraction(valid_cds),
        "frame_intact_fraction": _safe_fraction(valid_cds),
        "frame_length_fraction": _safe_fraction(frame_len),
        "start_codon_fraction": _safe_fraction(start_ok),
        "terminal_stop_fraction": _safe_fraction(terminal_stop),
        "internal_stop_fraction": _safe_fraction(internal_stop),
        "invalid_char_fraction": _safe_mean(invalid_fracs),
        "legal_fraction": _safe_fraction(legal),
    }


def kozak_uaug_stats(records: Iterable[object]) -> dict:
    rows = list(records)
    scores = [_kozak_score(r) for r in rows]
    uaug = [_count_overlapping(_valid_only(five_utr_of(r)), START_CODON) for r in rows]
    return {
        "n": len(rows),
        "mean_kozak_score": _safe_mean(scores),
        "kozak_strong_fraction": _safe_fraction([s >= 1.0 for s in scores]),
        "kozak_partial_fraction": _safe_fraction([s >= 0.5 for s in scores]),
        "mean_uaug_count": _safe_mean([float(x) for x in uaug]),
        "uaug_fraction": _safe_fraction([x > 0 for x in uaug]),
    }


def start_accessibility_proxy(record: object) -> float:
    explicit = _field(record, "start_accessibility", None)
    if explicit is not None:
        try:
            value = float(explicit)
            if math.isfinite(value):
                return max(0.0, min(1.0, value))
        except (TypeError, ValueError):
            pass
    five = _valid_only(five_utr_of(record))
    cds = _valid_only(cds_of(record))
    window = (five[-30:] + cds[:12]) or _valid_only(sequence_of(record))[:42]
    if not window:
        return 0.0
    gc = (window.count("G") + window.count("C")) / len(window)
    complement = {"A": "U", "U": "A", "C": "G", "G": "C"}
    pairs = 0
    for i in range(len(window) // 2):
        if complement.get(window[i]) == window[-1 - i]:
            pairs += 1
    pair_frac = pairs / max(1.0, len(window) / 2.0)
    return float(max(0.0, min(1.0, 1.0 - 0.50 * gc - 0.30 * pair_frac)))


def mfe_proxy(record: object) -> float:
    explicit = _field(record, "mfe", None)
    if explicit is not None:
        try:
            value = float(explicit)
            if math.isfinite(value):
                return value
        except (TypeError, ValueError):
            pass
    seq = _valid_only(sequence_of(record))
    if not seq:
        return 0.0
    gc = seq.count("G") + seq.count("C")
    au = seq.count("A") + seq.count("U")
    complement = {"A": "U", "U": "A", "C": "G", "G": "C"}
    local_pairs = 0
    for i in range(len(seq) - 5):
        if complement.get(seq[i]) == seq[i + 5]:
            local_pairs += 1
    return float(-(0.025 * au + 0.055 * gc + 0.015 * local_pairs))


def start_accessibility_mfe_metrics(
    records: Iterable[object],
    accessibility_min: float = 0.35,
    mfe_range: tuple[float, float] = (-120.0, 0.0),
) -> dict:
    rows = list(records)
    access = [start_accessibility_proxy(r) for r in rows]
    mfes = [mfe_proxy(r) for r in rows]
    lo, hi = mfe_range
    return {
        "n": len(rows),
        "mean_start_accessibility": _safe_mean(access),
        "start_accessible_fraction": _safe_fraction([x >= accessibility_min for x in access]),
        "mean_mfe_proxy": _safe_mean(mfes),
        "mfe_in_range_fraction": _safe_fraction([lo <= x <= hi for x in mfes]),
        "mfe_range_low": float(lo),
        "mfe_range_high": float(hi),
    }


def _kmer_distribution(records: Iterable[object], k: int = 3) -> dict[str, float]:
    counts: Counter[str] = Counter()
    for r in records:
        seq = _valid_only(sequence_of(r))
        if len(seq) < k:
            continue
        for i in range(len(seq) - k + 1):
            counts[seq[i:i + k]] += 1
    keys = ["".join(p) for p in product(RNA_ALPHABET, repeat=k)]
    total = float(sum(counts.values()))
    if total == 0.0:
        return {key: 1.0 / len(keys) for key in keys}
    return {key: float(counts.get(key, 0) / total) for key in keys}


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    mask = p > 0
    if not np.any(mask):
        return 0.0
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))


def kmer_js_distance(candidates: Iterable[object], sources: Iterable[object], k: int = 3) -> float:
    p_dist = _kmer_distribution(list(candidates), k=k)
    q_dist = _kmer_distribution(list(sources), k=k)
    keys = sorted(set(p_dist) | set(q_dist))
    p = np.asarray([p_dist.get(key, 0.0) for key in keys], dtype=float)
    q = np.asarray([q_dist.get(key, 0.0) for key in keys], dtype=float)
    p = p / max(float(p.sum()), 1e-12)
    q = q / max(float(q.sum()), 1e-12)
    m = 0.5 * (p + q)
    return float(max(0.0, 0.5 * _kl(p, m) + 0.5 * _kl(q, m)))


def _codon_distribution(records: Iterable[object], eps: float = 1e-8) -> np.ndarray:
    counts = np.full(len(ALL_CODONS), eps, dtype=float)
    idx = {codon: i for i, codon in enumerate(ALL_CODONS)}
    for r in records:
        for codon in codons_of(cds_of(r)):
            if codon in idx:
                counts[idx[codon]] += 1.0
    return counts / counts.sum()


def codon_usage_kl(candidates: Iterable[object], sources: Iterable[object], eps: float = 1e-8) -> float:
    p = _codon_distribution(list(candidates), eps=eps)
    q = _codon_distribution(list(sources), eps=eps)
    return _kl(p, q)


def _quantile_distance(a: Sequence[float], b: Sequence[float], n: int = 21) -> float:
    if not a and not b:
        return 0.0
    if not a or not b:
        return float("inf")
    qs = np.linspace(0.0, 1.0, n)
    qa = np.quantile(np.asarray(a, dtype=float), qs)
    qb = np.quantile(np.asarray(b, dtype=float), qs)
    return float(np.mean(np.abs(qa - qb)))


def gc_length_distribution_distance(candidates: Iterable[object], sources: Iterable[object]) -> dict:
    cand = list(candidates)
    src = list(sources)
    cand_gc = [gc_fraction(r) for r in cand]
    src_gc = [gc_fraction(r) for r in src]
    cand_len = [float(len(_valid_only(sequence_of(r)))) for r in cand]
    src_len = [float(len(_valid_only(sequence_of(r)))) for r in src]
    len_scale = max(1.0, _safe_mean(src_len, 1.0))
    gc_dist = _quantile_distance(cand_gc, src_gc)
    len_dist = _quantile_distance(cand_len, src_len) / len_scale
    return {
        "gc_quantile_distance": float(gc_dist),
        "length_quantile_distance": float(len_dist),
        "combined_gc_length_distance": float(gc_dist + len_dist),
        "candidate_mean_gc": _safe_mean(cand_gc),
        "source_mean_gc": _safe_mean(src_gc),
        "candidate_mean_length": _safe_mean(cand_len),
        "source_mean_length": _safe_mean(src_len),
    }


def sequence_embeddings(records: Iterable[object], k: int = 3) -> np.ndarray:
    rows = []
    keys = ["".join(p) for p in product(RNA_ALPHABET, repeat=k)]
    for r in records:
        seq = _valid_only(sequence_of(r))
        counts = Counter(seq[i:i + k] for i in range(max(0, len(seq) - k + 1)))
        total = max(1.0, float(sum(counts.values())))
        kmer = [counts.get(key, 0) / total for key in keys]
        rows.append(
            kmer
            + [
                len(seq) / 1000.0,
                gc_fraction(seq),
                start_accessibility_proxy(r),
                max(0.0, min(1.0, -mfe_proxy(r) / max(1.0, len(seq) * 0.08))),
            ]
        )
    if not rows:
        return np.zeros((0, len(keys) + 4), dtype=float)
    return np.asarray(rows, dtype=float)


def _covariance(x: np.ndarray) -> np.ndarray:
    if x.shape[0] <= 1:
        return np.zeros((x.shape[1], x.shape[1]), dtype=float)
    return np.cov(x, rowvar=False)


def _sqrtm_psd(mat: np.ndarray) -> np.ndarray:
    sym = np.nan_to_num(0.5 * (mat + mat.T), nan=0.0, posinf=0.0, neginf=0.0)
    vals, vecs = np.linalg.eigh(sym)
    vals = np.clip(vals, 0.0, None)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        root = (vecs * np.sqrt(vals).reshape(1, -1)) @ vecs.T
    return np.nan_to_num(0.5 * (root + root.T), nan=0.0, posinf=0.0, neginf=0.0)


def embedding_frechet_proxy(candidates: Iterable[object], sources: Iterable[object]) -> float:
    x = sequence_embeddings(list(candidates))
    y = sequence_embeddings(list(sources))
    if x.shape[0] == 0 or y.shape[0] == 0:
        return 0.0
    mu_x = x.mean(axis=0)
    mu_y = y.mean(axis=0)
    cov_x = _covariance(x)
    cov_y = _covariance(y)
    sqrt_x = _sqrtm_psd(cov_x)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        prod = sqrt_x @ cov_y @ sqrt_x
    middle = _sqrtm_psd(np.nan_to_num(prod, nan=0.0, posinf=0.0, neginf=0.0))
    val = float(np.sum((mu_x - mu_y) ** 2) + np.trace(cov_x + cov_y - 2.0 * middle))
    return float(max(0.0, val))


def _trim_common_affixes(a: str, b: str) -> tuple[str, str]:
    """Remove identical prefixes/suffixes before edit-distance DP.

    Levenshtein distance is invariant to deleting a shared prefix and suffix:

    ``d(p + x + s, p + y + s) = d(x, y)``.

    Public mRNA comparisons often differ by a few UTR/codon edits inside long
    shared transcripts, so this reduces the DP rectangle from ``m*n`` to
    ``m' * n'`` without changing the exact answer. Complexity is ``O(m+n)``.
    """
    n = min(len(a), len(b))
    start = 0
    while start < n and a[start] == b[start]:
        start += 1
    a_tail = len(a)
    b_tail = len(b)
    while a_tail > start and b_tail > start and a[a_tail - 1] == b[b_tail - 1]:
        a_tail -= 1
        b_tail -= 1
    return a[start:a_tail], b[start:b_tail]


def edit_distance(a: str, b: str) -> int:
    """Exact Levenshtein distance with prefix/suffix trimming.

    The DP recurrence is

    ``D[i,j]=min(D[i-1,j]+1, D[i,j-1]+1, D[i-1,j-1]+1[a_i!=b_j])``.

    After trimming shared affixes, time is ``O(m' n')`` and memory
    ``O(min(m', n'))`` where ``m'``/``n'`` are the remaining lengths.
    """
    a = normalize_rna(a)
    b = normalize_rna(b)
    if a == b:
        return 0
    a, b = _trim_common_affixes(a, b)
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(
                min(
                    prev[j] + 1,
                    cur[j - 1] + 1,
                    prev[j - 1] + (0 if ca == cb else 1),
                )
            )
        prev = cur
    return prev[-1]


def _edit_distance_at_most(a: str, b: str, max_distance: int) -> int:
    """Exact Levenshtein distance capped at ``max_distance + 1``.

    The recurrence is identical to :func:`edit_distance`, but the DP evaluates
    only the diagonal band ``|i-j| <= k`` for ``k=max_distance``. Any path with
    at most ``k`` edits must stay inside that band because each insertion or
    deletion changes ``i-j`` by one. The result is therefore exact when the true
    distance is ``<= k``; otherwise returning ``k + 1`` safely proves that this
    source cannot improve a current best threshold. Complexity is
    ``O(k * min(m,n))`` time and ``O(k)`` memory after shared-affix trimming.
    """
    k = int(max_distance)
    if k < 0:
        return 0
    a = normalize_rna(a)
    b = normalize_rna(b)
    if a == b:
        return 0
    a, b = _trim_common_affixes(a, b)
    if not a:
        return len(b) if len(b) <= k else k + 1
    if not b:
        return len(a) if len(a) <= k else k + 1
    if abs(len(a) - len(b)) > k:
        return k + 1
    if len(a) < len(b):
        a, b = b, a
    m, n = len(a), len(b)
    inf = k + 1
    prev = {j: j for j in range(0, min(n, k) + 1)}
    for i, ca in enumerate(a, start=1):
        j_start = max(1, i - k)
        j_end = min(n, i + k)
        cur: dict[int, int] = {}
        if j_start == 1:
            cur[0] = i if i <= k else inf
        row_min = inf
        for j in range(j_start, j_end + 1):
            deletion = prev.get(j, inf) + 1
            insertion = cur.get(j - 1, inf) + 1
            substitution = prev.get(j - 1, inf) + (0 if ca == b[j - 1] else 1)
            value = min(deletion, insertion, substitution)
            cur[j] = value
            if value < row_min:
                row_min = value
        if row_min > k:
            return k + 1
        prev = cur
    dist = prev.get(n, inf)
    return dist if dist <= k else k + 1


def normalized_edit_distance(a: str, b: str) -> float:
    denom = max(len(normalize_rna(a)), len(normalize_rna(b)), 1)
    return float(edit_distance(a, b) / denom)


def _rank_to_pair(rank: int, n: int, cumulative: np.ndarray) -> tuple[int, int]:
    """Map a linear upper-triangle rank to pair ``(i, j)`` with ``i < j``."""
    i = int(np.searchsorted(cumulative, int(rank), side="right"))
    prev = int(cumulative[i - 1]) if i > 0 else 0
    j = i + 1 + int(rank) - prev
    return i, j


def _sample_pair_indices(
    n: int,
    max_pairs: int,
) -> tuple[list[tuple[int, int]], int, bool]:
    """Deterministically choose candidate pairs for diversity estimation.

    For ``N`` candidates there are ``N(N-1)/2`` exact pairs. If this count is
    below ``max_pairs`` we return every pair. Otherwise we take evenly-spaced
    ranks across the upper triangle, which is deterministic and avoids storing
    all pairs. Complexity is ``O(min(total_pairs, max_pairs) log N)``.
    """
    total = n * (n - 1) // 2
    if total <= 0:
        return [], total, True
    if max_pairs <= 0 or total <= max_pairs:
        return [(i, j) for i in range(n) for j in range(i + 1, n)], total, True
    counts = np.asarray([n - i - 1 for i in range(n - 1)], dtype=np.int64)
    cumulative = np.cumsum(counts)
    step = total / float(max_pairs)
    ranks = [min(total - 1, int((k + 0.5) * step)) for k in range(max_pairs)]
    pairs = [_rank_to_pair(rank, n, cumulative) for rank in ranks]
    return pairs, total, False


def _normalised_edit_for_sequences(a: str, b: str) -> float:
    denom = max(len(a), len(b), 1)
    return float(edit_distance(a, b) / denom)


def diversity_novelty_metrics(
    candidates: Iterable[object],
    sources: Optional[Iterable[object]] = None,
    max_pairwise_pairs: int = DEFAULT_MAX_PAIRWISE_DIVERSITY_PAIRS,
    max_novelty_sources: int = 0,
) -> dict:
    """Diversity and novelty with exact small-N behavior and scalable large-N mode.

    ``pairwise_diversity`` is the mean normalized Levenshtein distance over all
    candidate pairs when ``N(N-1)/2 <= max_pairwise_pairs``. For larger public
    corpora, the function evaluates a deterministic uniform subsample and sets
    ``pairwise_diversity_exact=False`` while reporting both total and evaluated
    pair counts. This preserves a bounded, reproducible metric instead of
    silently launching an impractical ``O(N^2 L^2)`` job.

    Novelty is exact by default. For each candidate, exact source matches are
    detected by hash lookup, and non-matches are searched with a length lower
    bound ``|len(a)-len(b)|/max(len(a),len(b))`` that safely prunes sources that
    cannot beat the current best distance. When candidates and sources are
    paired by index, exact mode first scores the paired source as a warm-start
    upper bound. This preserves the exact nearest-source metric because the
    paired source is part of the same source set; it only tightens the bound
    before the remaining sources are scanned.

    If ``max_novelty_sources > 0``, only that many length-prioritized source
    records are evaluated per candidate. This changes the metric into a
    deterministic approximation and the output sets ``novelty_exact=False``.
    The time complexity becomes

    ``O(P * D + N * min(M, K) * D)``

    where ``P`` is evaluated candidate pairs, ``D`` is the trimmed edit-distance
    cost, ``N`` candidates, ``M`` sources and ``K=max_novelty_sources``. Exact
    mode uses the same bound with ``K=M`` but replaces full DP by banded DP once
    a finite best distance is known, giving ``O(kL)`` checks for sources that
    only need to prove they are not closer than the current ``k``-edit bound.
    """
    cand = list(candidates)
    seqs = [sequence_of(r) for r in cand]
    unique_fraction = len(set(seqs)) / max(1, len(seqs))
    pair_dists = []
    pair_indices, total_pairs, exact_pairs = _sample_pair_indices(
        len(seqs), int(max_pairwise_pairs)
    )
    for i, j in pair_indices:
        pair_dists.append(_normalised_edit_for_sequences(seqs[i], seqs[j]))

    source_seqs = [sequence_of(r) for r in sources] if sources is not None else []
    source_set = set(source_seqs)
    source_by_len = [(idx, src, len(src)) for idx, src in enumerate(source_seqs)]
    novelty_source_cap = int(max_novelty_sources)
    novelty_exact = novelty_source_cap <= 0 or len(source_by_len) <= novelty_source_cap
    novelty = []
    exact = []
    novelty_comparisons = 0
    paired_exact_warm_start = novelty_source_cap <= 0 and len(source_seqs) == len(seqs)
    for cand_idx, seq in enumerate(seqs):
        if source_seqs:
            if seq in source_set:
                novelty.append(0.0)
                exact.append(True)
                continue
            seq_len = len(seq)
            best = float("inf")
            paired_source_idx: Optional[int] = None
            if paired_exact_warm_start:
                paired_source_idx = cand_idx
                paired_src = source_seqs[cand_idx]
                novelty_comparisons += 1
                paired_denom = max(len(seq), len(paired_src), 1)
                if len(seq) == len(paired_src):
                    hamming_upper = sum(a != b for a, b in zip(seq, paired_src))
                    paired_dist = _edit_distance_at_most(seq, paired_src, hamming_upper)
                else:
                    paired_dist = edit_distance(seq, paired_src)
                best = float(paired_dist / paired_denom)
                if best == 0.0:
                    novelty.append(0.0)
                    exact.append(False)
                    continue
            ordered = sorted(
                source_by_len,
                key=lambda item: abs(seq_len - item[2]) / max(seq_len, item[2], 1),
            )
            if novelty_source_cap > 0:
                ordered = ordered[:novelty_source_cap]
            for src_idx, src, src_len in ordered:
                if src_idx == paired_source_idx:
                    continue
                lower = abs(seq_len - src_len) / max(seq_len, src_len, 1)
                if lower >= best:
                    break
                novelty_comparisons += 1
                denom = max(seq_len, src_len, 1)
                if math.isfinite(best):
                    max_improving_edits = int(math.ceil(best * denom - 1e-12)) - 1
                    if max_improving_edits < 0:
                        break
                    raw_dist = _edit_distance_at_most(seq, src, max_improving_edits)
                    if raw_dist > max_improving_edits:
                        continue
                    dist = float(raw_dist / denom)
                else:
                    dist = _normalised_edit_for_sequences(seq, src)
                if dist < best:
                    best = dist
                    if best == 0.0:
                        break
            novelty.append(best if math.isfinite(best) else 0.0)
            exact.append(False)
    return {
        "unique_fraction": float(unique_fraction),
        "pairwise_diversity": _safe_mean(pair_dists),
        "pairwise_diversity_exact": bool(exact_pairs),
        "pairwise_pairs_total": int(total_pairs),
        "pairwise_pairs_evaluated": int(len(pair_indices)),
        "mean_novelty": _safe_mean(novelty),
        "exact_source_match_fraction": _safe_fraction(exact),
        "novelty_source_comparisons": int(novelty_comparisons),
        "novelty_exact": bool(novelty_exact),
        "novelty_sources_total": int(len(source_seqs)),
        "novelty_sources_evaluated_cap": int(novelty_source_cap),
    }


def protein_identity(candidate: object, source: object) -> float:
    cand = translate(_valid_only(cds_of(candidate))).rstrip("*")
    src = translate(_valid_only(cds_of(source))).rstrip("*")
    if not cand and not src:
        return 1.0
    denom = max(len(cand), len(src), 1)
    matches = sum(1 for a, b in zip(cand, src) if a == b)
    return float(matches / denom)


def protein_identity_metrics(candidates: Iterable[object], sources: Iterable[object]) -> dict:
    vals = [protein_identity(c, s) for c, s in zip(list(candidates), list(sources))]
    return {
        "n": len(vals),
        "mean_protein_identity": _safe_mean(vals),
        "min_protein_identity": float(min(vals)) if vals else 0.0,
        "protein_identity_ge_0_99_fraction": _safe_fraction([v >= 0.99 for v in vals]),
        "per_record": vals,
    }


def _default_codon_weights() -> dict[str, float]:
    weights = {}
    raw_by_aa: dict[str, dict[str, float]] = {}
    for codon, aa in CODON_TABLE.items():
        if aa == "*":
            continue
        raw = 1.0
        raw += 0.40 if codon[2] in ("G", "C") else 0.05
        raw += 0.10 if codon[1] in ("G", "C") else 0.0
        raw_by_aa.setdefault(aa, {})[codon] = raw
    for aa, rows in raw_by_aa.items():
        max_v = max(rows.values())
        for codon, raw in rows.items():
            weights[codon] = raw / max_v
    return weights


DEFAULT_CODON_WEIGHTS = _default_codon_weights()


def codon_weights_from_reference(records: Iterable[object], pseudocount: float = 1.0) -> dict[str, float]:
    by_aa: dict[str, Counter[str]] = {}
    for r in records:
        for codon in codons_of(cds_of(r)):
            aa = CODON_TABLE.get(codon)
            if aa is None or aa == "*":
                continue
            by_aa.setdefault(aa, Counter())[codon] += 1
    weights = {}
    for aa, codons in SYNONYMOUS_CODONS.items():
        if aa == "*":
            continue
        counts = by_aa.get(aa, Counter())
        vals = {codon: counts.get(codon, 0.0) + pseudocount for codon in codons}
        denom = max(vals.values())
        for codon, val in vals.items():
            weights[codon] = float(val / denom)
    return weights


def cai(cds: str, codon_weights: Optional[Mapping[str, float]] = None) -> float:
    weights = codon_weights or DEFAULT_CODON_WEIGHTS
    logs = []
    codons = codons_of(cds)
    for i, codon in enumerate(codons):
        aa = CODON_TABLE.get(codon)
        if aa is None:
            continue
        if aa == "*":
            if i == len(codons) - 1:
                break
            return 0.0
        logs.append(math.log(max(float(weights.get(codon, 1e-6)), 1e-6)))
    if not logs:
        return 0.0
    return float(math.exp(sum(logs) / len(logs)))


def cai_metrics(records: Iterable[object], reference_records: Optional[Iterable[object]] = None) -> dict:
    weights = (
        codon_weights_from_reference(list(reference_records))
        if reference_records is not None
        else DEFAULT_CODON_WEIGHTS
    )
    vals = [cai(cds_of(r), weights) for r in records]
    return {
        "mean_cai": _safe_mean(vals),
        "min_cai": float(min(vals)) if vals else 0.0,
        "per_record": vals,
    }


def _budget_for(
    candidate: object,
    index: int,
    budgets: Optional[int | Sequence[int] | Mapping[str, int]],
    max_edits: Optional[int],
) -> int:
    if isinstance(budgets, int):
        return budgets
    if isinstance(budgets, Sequence) and not isinstance(budgets, (str, bytes)):
        return int(budgets[index])
    if isinstance(budgets, Mapping):
        tid = str(_field(candidate, "transcript_id", index))
        return int(budgets.get(tid, max_edits if max_edits is not None else 0))
    explicit = _field(candidate, "edit_budget", None)
    if explicit is not None:
        return int(explicit)
    return int(max_edits if max_edits is not None else 0)


def edit_budget_metrics(
    candidates: Iterable[object],
    sources: Iterable[object],
    budgets: Optional[int | Sequence[int] | Mapping[str, int]] = None,
    max_edits: Optional[int] = None,
) -> dict:
    """Paired edit-budget metrics with exact distances.

    For each paired candidate/source sequence ``(x_i, s_i)`` and budget
    ``b_i``, the metric reports

    ``d_i = Levenshtein(x_i, s_i)``

    and ``within_i = 1[d_i <= b_i]``. Most constrained T5 runs are designed to
    stay inside a small budget (typically three edits), so we first compute
    ``_edit_distance_at_most(x_i, s_i, b_i)``. If the returned distance is
    ``<= b_i`` it is exact by the Ukkonen band argument in
    :func:`_edit_distance_at_most`; otherwise we fall back to full DP so
    ``mean_edit_distance`` remains exact even for over-budget failures.

    Complexity is ``O(BL)`` for within-budget pairs with budget ``B`` and
    sequence length ``L``; over-budget pairs pay the full trimmed
    ``O(m'n')`` Levenshtein cost.
    """
    cand = list(candidates)
    src = list(sources)
    dists = []
    budget_vals = []
    within = []
    for i, (c, s) in enumerate(zip(cand, src)):
        budget = _budget_for(c, i, budgets, max_edits)
        cand_seq = sequence_of(c)
        src_seq = sequence_of(s)
        dist = _edit_distance_at_most(cand_seq, src_seq, max(0, int(budget)))
        if dist > budget:
            dist = edit_distance(cand_seq, src_seq)
        dists.append(float(dist))
        budget_vals.append(float(budget))
        within.append(dist <= budget)
    margins = [b - d for b, d in zip(budget_vals, dists)]
    return {
        "n": len(dists),
        "within_budget_fraction": _safe_fraction(within),
        "over_budget_count": int(sum(not x for x in within)),
        "mean_edit_distance": _safe_mean(dists),
        "mean_budget": _safe_mean(budget_vals),
        "mean_budget_margin": _safe_mean(margins),
        "per_record_edit_distance": dists,
        "per_record_within_budget": [1.0 if x else 0.0 for x in within],
    }


def _target_length_for(candidate: object, source: Optional[object]) -> Optional[int]:
    for key in ("target_length", "target_len", "desired_length"):
        value = _field(candidate, key, None)
        if value is not None:
            return int(value)
    if source is not None:
        return len(_valid_only(sequence_of(source)))
    return None


def length_control_curve(
    candidates: Iterable[object],
    target_lengths: Optional[Sequence[int]] = None,
    sources: Optional[Iterable[object]] = None,
    bins: Optional[Sequence[float]] = None,
) -> dict:
    cand = list(candidates)
    src = list(sources) if sources is not None else [None] * len(cand)
    targets = []
    lengths = []
    for i, c in enumerate(cand):
        target = int(target_lengths[i]) if target_lengths is not None else _target_length_for(c, src[i])
        if target is None:
            target = len(_valid_only(sequence_of(c)))
        targets.append(float(target))
        lengths.append(float(len(_valid_only(sequence_of(c)))))
    errors = [obs - tgt for obs, tgt in zip(lengths, targets)]
    abs_errors = [abs(x) for x in errors]
    if bins is None:
        if targets:
            lo, hi = min(targets), max(targets)
            if lo == hi:
                bins = [lo - 0.5, hi + 0.5]
            else:
                bins = list(np.linspace(lo, hi, min(4, len(set(targets))) + 1))
        else:
            bins = [0.0, 1.0]
    curve = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        idxs = [i for i, t in enumerate(targets) if lo <= t <= hi]
        if not idxs:
            continue
        curve.append(
            {
                "bin_start": float(lo),
                "bin_end": float(hi),
                "n": len(idxs),
                "target_mean": _safe_mean([targets[i] for i in idxs]),
                "observed_mean": _safe_mean([lengths[i] for i in idxs]),
                "error_mean": _safe_mean([errors[i] for i in idxs]),
                "abs_error_mean": _safe_mean([abs_errors[i] for i in idxs]),
            }
        )
    return {
        "n": len(cand),
        "mean_length": _safe_mean(lengths),
        "mean_target_length": _safe_mean(targets),
        "mean_length_error": _safe_mean(errors),
        "mean_abs_length_error": _safe_mean(abs_errors),
        "length_within_3nt_fraction": _safe_fraction([abs(x) <= 3 for x in errors]),
        "curve": curve,
        "per_record_abs_error": abs_errors,
    }


DEFAULT_MOTIFS = {
    "polyA_signal": {"region": "three_utr", "patterns": ["AAUAAA", "AUUAAA"]},
    "uAUG": {"region": "five_utr", "patterns": [START_CODON]},
    "miR_seed_like": {"region": "three_utr", "patterns": ["ACUG", "UGCA"]},
    "TOP": {"region": "five_utr_prefix", "patterns": []},
    "strong_kozak": {"region": "kozak", "patterns": []},
}


def _motif_region(record: object, region: str) -> str:
    if region == "five_utr":
        return _valid_only(five_utr_of(record))
    if region == "three_utr":
        return _valid_only(three_utr_of(record))
    if region == "cds":
        return _valid_only(cds_of(record))
    return _valid_only(sequence_of(record))


def detect_motifs(
    record: object,
    motifs: Optional[Mapping[str, Mapping[str, object]]] = None,
) -> dict[str, int]:
    motif_spec = motifs or DEFAULT_MOTIFS
    found: dict[str, int] = {}
    for name, spec in motif_spec.items():
        region = str(spec.get("region", "seq"))
        patterns = [normalize_rna(str(p)) for p in spec.get("patterns", [])]  # type: ignore[arg-type]
        if region == "kozak":
            found[name] = 1 if _kozak_score(record) >= 1.0 else 0
            continue
        if region == "five_utr_prefix":
            five = _valid_only(five_utr_of(record))
            prefix = five[: min(8, len(five))]
            found[name] = 1 if prefix and all(ch in ("C", "U") for ch in prefix) else 0
            continue
        seq = _motif_region(record, region)
        found[name] = int(sum(_count_overlapping(seq, p) for p in patterns))
    return found


def reading_frame_metrics(records: Iterable[object]) -> dict:
    rows = list(records)
    frame = []
    start = []
    stop = []
    no_internal = []
    intact = []
    for r in rows:
        cds = _valid_only(cds_of(r))
        frame.append(bool(cds) and len(cds) % 3 == 0)
        start.append(bool(cds) and cds.startswith(START_CODON))
        stop.append(len(cds) >= 3 and cds[-3:] in STOP_CODONS)
        prot = translate(cds) if cds else ""
        no_internal.append("*" not in prot[:-1])
        intact.append(is_valid_cds(cds) if cds else False)
    return {
        "frame_length_fraction": _safe_fraction(frame),
        "start_codon_fraction": _safe_fraction(start),
        "terminal_stop_fraction": _safe_fraction(stop),
        "no_internal_stop_fraction": _safe_fraction(no_internal),
        "reading_frame_intact_fraction": _safe_fraction(intact),
    }


def motif_metrics(
    records: Iterable[object],
    motifs: Optional[Mapping[str, Mapping[str, object]]] = None,
) -> dict:
    rows = list(records)
    motif_spec = motifs or DEFAULT_MOTIFS
    per = [detect_motifs(r, motif_spec) for r in rows]
    out = {"n": len(rows)}
    for name in motif_spec:
        counts = [row.get(name, 0) for row in per]
        out[f"{name}_presence_fraction"] = _safe_fraction([x > 0 for x in counts])
        out[f"{name}_mean_count"] = _safe_mean([float(x) for x in counts])
    frame = reading_frame_metrics(rows)
    out.update(frame)
    if motif_spec:
        pres = []
        for row in per:
            pres.extend([row.get(name, 0) > 0 for name in motif_spec])
        out["any_motif_presence_fraction"] = _safe_fraction(pres)
    else:
        out["any_motif_presence_fraction"] = 0.0
    return out


def distribution_metrics(candidates: Iterable[object], sources: Iterable[object], k: int = 3) -> dict:
    cand = list(candidates)
    src = list(sources)
    gc_len = gc_length_distribution_distance(cand, src)
    return {
        "kmer_js": kmer_js_distance(cand, src, k=k),
        "codon_usage_kl": codon_usage_kl(cand, src),
        "embedding_frechet_proxy": embedding_frechet_proxy(cand, src),
        **gc_len,
    }


def compute_all_metrics(
    candidates: Iterable[object],
    sources: Optional[Iterable[object]] = None,
    budgets: Optional[int | Sequence[int] | Mapping[str, int]] = None,
    target_lengths: Optional[Sequence[int]] = None,
    max_pairwise_pairs: int = DEFAULT_MAX_PAIRWISE_DIVERSITY_PAIRS,
    max_novelty_sources: int = 0,
) -> dict:
    cand = list(candidates)
    src = list(sources) if sources is not None else []
    out = {
        "legality": legality_metrics(cand),
        "kozak_uaug": kozak_uaug_stats(cand),
        "structure": start_accessibility_mfe_metrics(cand),
        "diversity_novelty": diversity_novelty_metrics(
            cand,
            src if src else None,
            max_pairwise_pairs=max_pairwise_pairs,
            max_novelty_sources=max_novelty_sources,
        ),
        "cai": cai_metrics(cand, src if src else None),
        "length_control": length_control_curve(
            cand, target_lengths=target_lengths, sources=src if src else None
        ),
        "motifs": motif_metrics(cand),
    }
    if src:
        out["distribution"] = distribution_metrics(cand, src)
        out["protein_identity"] = protein_identity_metrics(cand, src)
        out["edit_budget"] = edit_budget_metrics(cand, src, budgets=budgets)
    return out


__all__ = [
    "normalize_rna",
    "sequence_of",
    "five_utr_of",
    "cds_of",
    "three_utr_of",
    "gc_fraction",
    "legality_metrics",
    "kozak_uaug_stats",
    "start_accessibility_proxy",
    "mfe_proxy",
    "start_accessibility_mfe_metrics",
    "kmer_js_distance",
    "codon_usage_kl",
    "gc_length_distribution_distance",
    "sequence_embeddings",
    "embedding_frechet_proxy",
    "edit_distance",
    "normalized_edit_distance",
    "DEFAULT_MAX_PAIRWISE_DIVERSITY_PAIRS",
    "diversity_novelty_metrics",
    "protein_identity",
    "protein_identity_metrics",
    "DEFAULT_CODON_WEIGHTS",
    "codon_weights_from_reference",
    "cai",
    "cai_metrics",
    "edit_budget_metrics",
    "length_control_curve",
    "DEFAULT_MOTIFS",
    "detect_motifs",
    "reading_frame_metrics",
    "motif_metrics",
    "distribution_metrics",
    "compute_all_metrics",
]
