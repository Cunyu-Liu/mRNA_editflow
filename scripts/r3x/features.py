"""Deterministic, information-set-specific feature maps for R3X baselines."""
from __future__ import annotations

import itertools
from typing import Mapping, Sequence

import numpy as np


NUCLEOTIDES = "ACGU"
NUC_TO_INDEX = {nt: i for i, nt in enumerate(NUCLEOTIDES)}
EDIT_ONLY_FEATURE_SCHEMA = "R3X_EDIT_ONLY_FEATURES_V1"
CANDIDATE_KMER_FEATURE_SCHEMA = "R3X_CANDIDATE_KMER_FREQUENCIES_V1"
PAIRED_KMER_FEATURE_SCHEMA = (
    "R3X_PAIRED_LOCAL_GLOBAL_KMER_V2_PER_EDIT_WINDOW_MEAN_NO_CROSS_WINDOW_KMERS"
)


def normalize_sequence(sequence: str) -> str:
    if not isinstance(sequence, str):
        raise ValueError("sequence must be a string")
    sequence = sequence.upper().replace("T", "U")
    if not sequence or any(nt not in NUCLEOTIDES for nt in sequence):
        raise ValueError("sequence must be non-empty A/C/G/U")
    return sequence


def _edit_field(edit: Mapping, *names: str):
    for name in names:
        if name in edit and edit[name] is not None:
            return edit[name]
    return None


def validated_substitutions(row: Mapping) -> list[tuple[int, str, str]]:
    source = normalize_sequence(
        row.get("source_sequence") or row.get("normalized_source_sequence")
    )
    candidate = normalize_sequence(
        row.get("candidate_sequence") or row.get("normalized_candidate_sequence")
    )
    if len(source) != len(candidate):
        raise ValueError("edit-only features require equal-length source/candidate")
    raw = row.get("edit_list")
    if not isinstance(raw, list) or not raw:
        raise ValueError("edit_list must be a non-empty list")
    parsed: list[tuple[int, str, str]] = []
    for edit in raw:
        if not isinstance(edit, Mapping) or str(edit.get("op", "")).upper() != "SUB":
            raise ValueError("edit-only features accept SUB operations only")
        pos_raw = _edit_field(edit, "pos", "position")
        if isinstance(pos_raw, bool) or not isinstance(pos_raw, int):
            raise ValueError("SUB position must be an integer")
        pos = int(pos_raw)
        if not 0 <= pos < len(source):
            raise ValueError("SUB position is out of range")
        ref = str(
            _edit_field(edit, "ref", "from", "source_token", "source_nt")
            or source[pos]
        ).upper().replace("T", "U")
        alt = str(
            _edit_field(edit, "alt", "to", "token", "target_token", "target_nt")
            or ""
        ).upper().replace("T", "U")
        if ref not in NUCLEOTIDES or alt not in NUCLEOTIDES or ref == alt:
            raise ValueError("SUB ref/alt must be distinct A/C/G/U tokens")
        parsed.append((pos, ref, alt))
    parsed.sort()
    if len({pos for pos, _, _ in parsed}) != len(parsed):
        raise ValueError("duplicate SUB positions are prohibited")
    replay = list(source)
    for pos, ref, alt in parsed:
        if replay[pos] != ref:
            raise ValueError("edit replay ref does not match source")
        replay[pos] = alt
    if "".join(replay) != candidate:
        raise ValueError("edit replay does not reproduce candidate sequence")
    actual_count = row.get("actual_edit_count", row.get("edit_count"))
    if actual_count is not None and int(actual_count) != len(parsed):
        raise ValueError("actual_edit_count disagrees with exact replay")
    return parsed


def edit_only_features(row: Mapping) -> np.ndarray:
    raw = row.get("edit_list")
    if not isinstance(raw, list) or not raw:
        raise ValueError("edit_list must be a non-empty list")
    edits = []
    for edit in raw:
        if not isinstance(edit, Mapping) or str(edit.get("op", "")).upper() != "SUB":
            raise ValueError("edit-only features accept SUB operations only")
        pos = _edit_field(edit, "pos", "position")
        if isinstance(pos, bool) or not isinstance(pos, int) or not 0 <= pos < 512:
            raise ValueError("edit-only SUB position must be an integer in [0,511]")
        ref = str(_edit_field(edit, "ref", "from", "source_token", "source_nt") or "").upper().replace("T", "U")
        alt = str(_edit_field(edit, "alt", "to", "token", "target_token", "target_nt") or "").upper().replace("T", "U")
        if ref not in NUCLEOTIDES or alt not in NUCLEOTIDES or ref == alt:
            raise ValueError("edit-only SUB ref/alt must be distinct A/C/G/U")
        edits.append((int(pos), ref, alt))
    edits.sort()
    if len({pos for pos, _, _ in edits}) != len(edits):
        raise ValueError("duplicate SUB positions are prohibited")
    declared_count = row.get("edit_count", row.get("actual_edit_count"))
    if isinstance(declared_count, bool) or not isinstance(declared_count, int) or declared_count != len(edits):
        raise ValueError("edit_count must exactly equal the SUB list length")
    positions = np.asarray([pos / 511.0 for pos, _, _ in edits], dtype=np.float64)
    transitions = np.zeros((4, 4), dtype=np.float64)
    for _, ref, alt in edits:
        transitions[NUC_TO_INDEX[ref], NUC_TO_INDEX[alt]] += 1.0
    transitions /= len(edits)
    return np.concatenate(
        [
            np.asarray(
                [
                    len(edits) / 5.0,
                    float(np.mean(positions)),
                    float(np.std(positions)),
                    float(np.min(positions)),
                    float(np.max(positions)),
                    float(np.max(positions) - np.min(positions)),
                ],
                dtype=np.float64,
            ),
            transitions.reshape(-1),
        ]
    )


def _vocabulary(k: int) -> tuple[str, ...]:
    return tuple("".join(parts) for parts in itertools.product(NUCLEOTIDES, repeat=k))


def kmer_frequencies(sequence: str, *, max_k: int = 3) -> np.ndarray:
    sequence = normalize_sequence(sequence)
    if not isinstance(max_k, int) or not 1 <= max_k <= 4:
        raise ValueError("max_k must be an integer in [1, 4]")
    blocks = []
    for k in range(1, max_k + 1):
        vocabulary = _vocabulary(k)
        index = {token: i for i, token in enumerate(vocabulary)}
        counts = np.zeros(len(vocabulary), dtype=np.float64)
        total = max(0, len(sequence) - k + 1)
        for pos in range(total):
            counts[index[sequence[pos : pos + k]]] += 1.0
        if total:
            counts /= total
        blocks.append(counts)
    gc = (sequence.count("G") + sequence.count("C")) / len(sequence)
    return np.concatenate([np.asarray([len(sequence) / 512.0, gc]), *blocks])


def candidate_kmer_features(row: Mapping, *, max_k: int = 3) -> np.ndarray:
    return kmer_frequencies(
        row.get("candidate_sequence") or row.get("normalized_candidate_sequence"),
        max_k=max_k,
    )


def _local_edit_sequences(
    row: Mapping,
    radius: int = 12,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    source = normalize_sequence(
        row.get("source_sequence") or row.get("normalized_source_sequence")
    )
    candidate = normalize_sequence(
        row.get("candidate_sequence") or row.get("normalized_candidate_sequence")
    )
    edits = validated_substitutions(row)
    src_chunks: list[str] = []
    cand_chunks: list[str] = []
    for pos, _, _ in edits:
        lo = max(0, pos - radius)
        hi = min(len(source), pos + radius + 1)
        src_chunks.append(source[lo:hi])
        cand_chunks.append(candidate[lo:hi])
    return tuple(src_chunks), tuple(cand_chunks)


def _mean_window_kmer_frequencies(
    windows: Sequence[str],
    *,
    max_k: int,
) -> np.ndarray:
    """Average per-edit windows without inventing cross-window k-mers."""

    if not windows:
        raise ValueError("local k-mer features require at least one edit window")
    values = [kmer_frequencies(window, max_k=max_k) for window in windows]
    return np.mean(np.vstack(values), axis=0)


def paired_kmer_features(row: Mapping, *, max_k: int = 3) -> np.ndarray:
    source = normalize_sequence(
        row.get("source_sequence") or row.get("normalized_source_sequence")
    )
    candidate = normalize_sequence(
        row.get("candidate_sequence") or row.get("normalized_candidate_sequence")
    )
    validated_substitutions(row)
    src = kmer_frequencies(source, max_k=max_k)
    cand = kmer_frequencies(candidate, max_k=max_k)
    local_source, local_candidate = _local_edit_sequences(row)
    local_src = _mean_window_kmer_frequencies(local_source, max_k=max_k)
    local_cand = _mean_window_kmer_frequencies(local_candidate, max_k=max_k)
    return np.concatenate([src, cand, cand - src, local_cand - local_src, edit_only_features(row)])


def matrix(rows: Sequence[Mapping], feature_fn) -> np.ndarray:
    if not rows:
        raise ValueError("feature matrix refuses empty rows")
    values = [np.asarray(feature_fn(row), dtype=np.float64) for row in rows]
    width = values[0].shape
    if any(value.ndim != 1 or value.shape != width for value in values):
        raise ValueError("feature map returned inconsistent dimensions")
    out = np.vstack(values)
    if not np.all(np.isfinite(out)):
        raise ValueError("feature matrix contains non-finite values")
    return out
