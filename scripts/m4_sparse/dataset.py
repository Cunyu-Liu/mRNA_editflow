"""PyTorch Dataset for the M4 effect dataset (artifacts/b0x/effect_dataset.jsonl).

Encoding conventions mirror scripts/b0x/features.py (NUC_ORDER="ACGU",
MAX_SEQ_LEN=100, 12-dim edit features) so M4 is comparable to B0-X.  Unlike the
legacy first-100 truncation, source and candidate use the same edit-centred
window, which keeps distal edits visible without changing tensor size.

The dataset keeps the candidate value, source value, delta, delta sign and
source/study/endpoint rank group as separate labels.  It also exposes explicit
padding masks and a reserved, trainable unknown-context id for honest S4 use.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import EDIT_FEAT_DIM, MAX_SEQ_LEN, NUC_ORDER

NUC_TO_IDX = {ch: i for i, ch in enumerate(NUC_ORDER)}
GUESS = "N"
UNKNOWN_CONTEXT = "__UNK__"


class IncompatibleEditEncodingError(ValueError):
    """Raised when a row cannot be represented by the frozen R2 input schema."""


def normalize(seq: Optional[str]) -> str:
    return (seq or "").upper().replace("T", "U")


def one_hot(seq: Optional[str], max_len: int = MAX_SEQ_LEN) -> np.ndarray:
    """(max_len, 4) ACGU one-hot of the first max_len nucleotides."""
    s = normalize(seq)
    arr = np.zeros((max_len, len(NUC_ORDER)), dtype=np.float32)
    for i, ch in enumerate(s[:max_len]):
        idx = NUC_TO_IDX.get(ch, -1)
        if idx >= 0:
            arr[i, idx] = 1.0
    return arr


def edit_window_start(source_seq: Optional[str], candidate_seq: Optional[str],
                      edit_list: Optional[List[Dict]],
                      max_len: int = MAX_SEQ_LEN) -> int:
    """Choose one shared, deterministic window that makes distal edits visible.

    If all edit coordinates fit in one window, every edit is included.  For an
    edit span wider than ``max_len`` (rare multi-edit scripts), the window is
    centred on the median edit.  Rows without an edit retain the legacy prefix
    window.  Coordinates are interpreted in source space, matching the dataset.
    """
    source = normalize(source_seq)
    candidate = normalize(candidate_seq)
    if not source or not candidate:
        raise IncompatibleEditEncodingError(
            "R2 edit-centred encoding requires non-empty source and candidate"
        )
    if len(source) != len(candidate):
        raise IncompatibleEditEncodingError(
            "R2 shared-coordinate encoding supports equal-length sequences only"
        )
    total_len = len(source)
    positions = []
    seen_positions = set()
    for edit_index, edit in enumerate(edit_list or []):
        if not isinstance(edit, dict):
            raise IncompatibleEditEncodingError(
                "R2 edit %d is not an object" % edit_index
            )
        if edit.get("op") != "SUB":
            raise IncompatibleEditEncodingError(
                "R2 shared-coordinate encoding supports SUB edits only"
            )
        try:
            raw_position = edit["pos"]
            if (isinstance(raw_position, bool)
                    or not isinstance(raw_position, (int, np.integer))):
                raise TypeError
            position = int(raw_position)
        except (TypeError, ValueError):
            raise IncompatibleEditEncodingError(
                "R2 edit %d has an invalid position" % edit_index
            ) from None
        except KeyError:
            raise IncompatibleEditEncodingError(
                "R2 edit %d has no position" % edit_index
            ) from None
        if position < 0 or position >= total_len:
            raise IncompatibleEditEncodingError(
                "R2 edit position %r is outside source length %d"
                % (position, total_len)
            )
        if position in seen_positions:
            raise IncompatibleEditEncodingError(
                "R2 edit script repeats source position %d" % position
            )
        raw_token = edit.get("token")
        if not isinstance(raw_token, str):
            raise IncompatibleEditEncodingError(
                "R2 SUB edit %d has a non-ACGU target token" % edit_index
            )
        token = normalize(raw_token)
        if len(token) != 1 or token not in NUC_ORDER:
            raise IncompatibleEditEncodingError(
                "R2 SUB edit %d has a non-ACGU target token" % edit_index
            )
        if candidate[position] != token:
            raise IncompatibleEditEncodingError(
                "R2 SUB token disagrees with candidate sequence at position %d"
                % position
            )
        if source[position] == candidate[position]:
            raise IncompatibleEditEncodingError(
                "R2 edit script contains a no-op SUB at position %d" % position
            )
        seen_positions.add(position)
        positions.append(position)
    positions.sort()
    actual_positions = {
        index for index, (source_nt, candidate_nt) in enumerate(
            zip(source, candidate)) if source_nt != candidate_nt
    }
    if actual_positions != seen_positions:
        raise IncompatibleEditEncodingError(
            "R2 edit script does not exactly match source/candidate differences"
        )
    if total_len <= max_len:
        return 0
    if not positions:
        return 0
    max_start = max(total_len - max_len, 0)
    centred = int(round(float(np.median(positions)) - (max_len - 1) / 2.0))
    if positions[-1] - positions[0] < max_len:
        # Feasible interval for a window containing every edit coordinate.
        low = max(0, positions[-1] - max_len + 1)
        high = min(positions[0], max_start)
        return int(min(max(centred, low), high))
    raise IncompatibleEditEncodingError(
        "R2 single-window encoder cannot represent edit span %d with MAX_SEQ_LEN=%d"
        % (positions[-1] - positions[0], max_len)
    )


def _one_hot_window(seq: Optional[str], start: int,
                    max_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """Encode ``seq[start:start+max_len]`` and return a true-on-padding mask."""
    s = normalize(seq)
    fragment = s[start:start + max_len]
    arr = np.zeros((max_len, len(NUC_ORDER)), dtype=np.float32)
    pad_mask = np.ones(max_len, dtype=np.bool_)
    for i, ch in enumerate(fragment):
        pad_mask[i] = False  # an unknown nucleotide is content, not padding
        idx = NUC_TO_IDX.get(ch, -1)
        if idx >= 0:
            arr[i, idx] = 1.0
    return arr, pad_mask


def encode_pair(source_seq: Optional[str], candidate_seq: Optional[str],
                edit_list: Optional[List[Dict]], max_len: int = MAX_SEQ_LEN
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """One-hot encode an aligned source/candidate window plus padding masks."""
    start = edit_window_start(source_seq, candidate_seq, edit_list, max_len)
    src, src_pad = _one_hot_window(source_seq, start, max_len)
    cand, cand_pad = _one_hot_window(candidate_seq, start, max_len)
    return src, cand, src_pad, cand_pad, start


def edit_features(edit_list: Optional[List[Dict]], seq_len: int) -> np.ndarray:
    """12-dim edit feature vector (mirrors scripts/b0x/features.py edit_features).

    [n_edits, pos_mean, pos_min, pos_max, n_sub, n_ins, n_del,
     token_freqs(ACGU + other)].
    """
    edits = edit_list or []
    if not edits:
        return np.zeros(EDIT_FEAT_DIM, dtype=np.float32)
    posns = [e.get("pos", 0) for e in edits]
    denom = max(seq_len, 1)
    pos_mean = float(np.mean(posns)) / denom
    pos_min = float(np.min(posns)) / denom
    pos_max = float(np.max(posns)) / denom
    nsub = sum(1 for e in edits if e.get("op") == "SUB")
    nins = sum(1 for e in edits if e.get("op") == "INS")
    ndel = sum(1 for e in edits if e.get("op") == "DEL")
    tokens = [(str(e.get("token", GUESS)) or GUESS)[0].upper() for e in edits]
    tok = np.zeros(len(NUC_ORDER) + 1, dtype=np.float32)
    for t in tokens:
        idx = NUC_TO_IDX.get(t, -1)
        if idx >= 0:
            tok[idx] += 1.0
        else:
            tok[-1] += 1.0
    tok /= max(len(tokens), 1)
    return np.concatenate([
        np.array([len(edits), pos_mean, pos_min, pos_max, nsub, nins, ndel],
                 dtype=np.float32), tok]).astype(np.float32)


def invert_edits(edit_list: Optional[List[Dict]], source_seq: Optional[str],
                 candidate_seq: Optional[str]) -> List[Dict]:
    """Invert a source->candidate edit list into candidate->source edits.

    Assumes edit positions are in source coordinates (as in the effect dataset).
    SUB token is inverted to the source nucleotide at that position; INS<->DEL
    are swapped.  Exact for single-position point edits; a faithful approximation
    for larger edit scripts.
    """
    src = normalize(source_seq)
    cand = normalize(candidate_seq)
    inv: List[Dict] = []
    for e in (edit_list or []):
        op = e.get("op")
        pos = int(e.get("pos", 0))
        tok = str(e.get("token", "") or "")
        if op == "SUB":
            orig = (src[pos] if pos < len(src) else
                    (cand[pos] if pos < len(cand) else GUESS))
            inv.append({"op": "SUB", "pos": pos, "token": orig})
        elif op == "INS":
            inv.append({"op": "DEL", "pos": pos, "token": tok})
        elif op == "DEL":
            orig = src[pos] if pos < len(src) else GUESS
            inv.append({"op": "INS", "pos": pos, "token": orig})
        else:
            inv.append(dict(e))
    return inv


def build_vocab(rows: List[Dict]) -> Dict[str, Dict[str, int]]:
    """Build context maps with a reserved trainable unknown id.

    Known ids keep the legacy sorted ordering; ``UNKNOWN_CONTEXT`` is appended
    so newly trained R2 checkpoints can use context dropout while old known-id
    ordering remains stable.  S4 evaluation explicitly routes the held-out
    study to this id instead of consulting an untrained held-out embedding.
    """
    studies = sorted({r.get("study") for r in rows
                      if r.get("study") not in (None, UNKNOWN_CONTEXT)})
    endpoints = sorted({r.get("endpoint") for r in rows
                        if r.get("endpoint") not in (None, UNKNOWN_CONTEXT)})
    benchmarks = sorted({r.get("benchmark") for r in rows
                         if r.get("benchmark") not in (None, UNKNOWN_CONTEXT)})

    def _map(values):
        out = {v: i for i, v in enumerate(values)}
        out[UNKNOWN_CONTEXT] = len(out)
        return out

    return {
        "study": _map(studies),
        "endpoint": _map(endpoints),
        "benchmark": _map(benchmarks),
    }


class EffectDataset(Dataset):
    def __init__(self, rows: List[Dict], vocab: Dict[str, Dict[str, int]],
                 target: str = "delta", max_len: int = MAX_SEQ_LEN,
                 force_unk_study: bool = False,
                 force_unk_endpoint: bool = False):
        if not rows:
            raise ValueError("empty EffectDataset")
        for namespace in ("study", "endpoint", "benchmark"):
            mapping = vocab.get(namespace)
            if not isinstance(mapping, dict) or UNKNOWN_CONTEXT not in mapping:
                raise ValueError(
                    "R2 EffectDataset requires a reserved __UNK__ %s id"
                    % namespace
                )
        self.rows = rows
        self.vocab = vocab
        self.target = target
        self.max_len = max_len
        encoded = [encode_pair(r.get("source_sequence"), r.get("candidate_sequence"),
                               r.get("edit_list"), max_len) for r in rows]
        self.src = np.stack([x[0] for x in encoded])
        self.cand = np.stack([x[1] for x in encoded])
        self.src_pad_mask = np.stack([x[2] for x in encoded])
        self.cand_pad_mask = np.stack([x[3] for x in encoded])
        self.window_start = np.array([x[4] for x in encoded], dtype=np.int64)
        edit_positions = [
            [int(e.get("pos", 0)) for e in (r.get("edit_list") or [])]
            for r in rows
        ]
        self.n_edits_total = np.array([len(p) for p in edit_positions], dtype=np.int64)
        self.n_edits_visible = np.array([
            sum(start <= p < start + max_len for p in positions)
            for start, positions in zip(self.window_start, edit_positions)
        ], dtype=np.int64)
        self.edit_span = np.array([
            (max(p) - min(p)) if p else 0 for p in edit_positions
        ], dtype=np.int64)
        self.all_edits_visible = self.n_edits_visible == self.n_edits_total
        if not np.all(self.all_edits_visible):
            raise IncompatibleEditEncodingError(
                "R2 encoder produced a row with invisible edits"
            )
        self.edit = np.stack([
            edit_features(r.get("edit_list"), len(r.get("source_sequence") or ""))
            for r in rows]).astype(np.float32)
        self.inv_edit = np.stack([
            edit_features(invert_edits(r.get("edit_list"), r.get("source_sequence"),
                                       r.get("candidate_sequence")),
                          len(r.get("candidate_sequence") or ""))
            for r in rows]).astype(np.float32)

        study_unk = vocab["study"].get(UNKNOWN_CONTEXT)
        endpoint_unk = vocab["endpoint"].get(UNKNOWN_CONTEXT)
        bench_unk = vocab["benchmark"].get(UNKNOWN_CONTEXT)
        if force_unk_study and study_unk is None:
            raise ValueError("vocab has no reserved unknown study id")
        if force_unk_endpoint and endpoint_unk is None:
            raise ValueError("vocab has no reserved unknown endpoint id")
        self.study = np.array([
            study_unk if force_unk_study else vocab["study"].get(r.get("study"), study_unk)
            for r in rows], dtype=np.int64)
        self.endpoint = np.array([
            endpoint_unk if force_unk_endpoint
            else vocab["endpoint"].get(r.get("endpoint"), endpoint_unk)
            for r in rows], dtype=np.int64)
        self.bench = np.array([
            vocab["benchmark"].get(r.get("benchmark"), bench_unk) for r in rows
        ], dtype=np.int64)

        def _numeric(field: str) -> np.ndarray:
            return np.array([
                float(r[field]) if r.get(field) is not None else np.nan for r in rows
            ], dtype=np.float32)

        self.candidate_value = _numeric("candidate_value")
        self.source_value = _numeric("source_value")
        self.delta = _numeric("delta")
        if not np.all(np.isfinite(self.delta)):
            raise ValueError("EffectDataset requires finite delta labels")
        self.sign = (self.delta > 0).astype(np.float32)

        # Only R1-qualified candidate pools may contribute rank pairs.  Rows in
        # ambiguous/singleton/duplicate-conflict pools retain their regression
        # and sign labels but receive unique singleton rank groups.
        from .o0x_search import _prepare_pool, pool_key
        pool_entries = {}
        for index, row in enumerate(rows):
            key = pool_key(row)
            if key is not None:
                pool_entries.setdefault(key, []).append((index, row))
        eligible_index_to_key = {}
        for key, entries in pool_entries.items():
            prepared = _prepare_pool(entries)
            if prepared["status"] == "ELIGIBLE":
                for index, _ in prepared["entries"]:
                    eligible_index_to_key[index] = key
        rank_keys = [
            ("eligible",) + eligible_index_to_key[index]
            if index in eligible_index_to_key else ("ineligible_row", str(index))
            for index in range(len(rows))
        ]
        group_vocab = {key: i for i, key in enumerate(sorted(set(rank_keys)))}
        self.source_group = np.array([group_vocab[key] for key in rank_keys],
                                     dtype=np.int64)
        self.rank_group_eligible = np.array([
            index in eligible_index_to_key for index in range(len(rows))
        ], dtype=np.bool_)

        if target == "delta":
            self.y = self.delta.copy()
        elif target == "candidate_value":
            if not np.all(np.isfinite(self.candidate_value)):
                raise ValueError("candidate_value target requires finite candidate values")
            if not np.all(np.isfinite(self.source_value)):
                raise ValueError("candidate_value target requires finite source anchors")
            self.y = self.candidate_value.copy()
        else:
            raise ValueError(f"unknown target {target}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        return {
            "src": torch.from_numpy(self.src[i]),
            "cand": torch.from_numpy(self.cand[i]),
            "src_pad_mask": torch.from_numpy(self.src_pad_mask[i]),
            "cand_pad_mask": torch.from_numpy(self.cand_pad_mask[i]),
            "edit": torch.from_numpy(self.edit[i]),
            "inv_edit": torch.from_numpy(self.inv_edit[i]),
            "study": torch.tensor(self.study[i], dtype=torch.long),
            "endpoint": torch.tensor(self.endpoint[i], dtype=torch.long),
            "bench": torch.tensor(self.bench[i], dtype=torch.long),
            "candidate_value": torch.tensor(self.candidate_value[i], dtype=torch.float32),
            "source_value": torch.tensor(self.source_value[i], dtype=torch.float32),
            "delta": torch.tensor(self.delta[i], dtype=torch.float32),
            "sign": torch.tensor(self.sign[i], dtype=torch.float32),
            "source_group": torch.tensor(self.source_group[i], dtype=torch.long),
            "rank_group_eligible": torch.tensor(
                bool(self.rank_group_eligible[i]), dtype=torch.bool),
            "window_start": torch.tensor(self.window_start[i], dtype=torch.long),
            "n_edits_total": torch.tensor(self.n_edits_total[i], dtype=torch.long),
            "n_edits_visible": torch.tensor(self.n_edits_visible[i], dtype=torch.long),
            "edit_span": torch.tensor(self.edit_span[i], dtype=torch.long),
            "all_edits_visible": torch.tensor(
                bool(self.all_edits_visible[i]), dtype=torch.bool),
            "y": torch.tensor(self.y[i], dtype=torch.float32),
        }


def assay_context(row: Dict) -> str:
    return str(row.get("context_id") or row.get("assay_context") or "")


def rank_group_key(row: Dict) -> Tuple[str, str, str, str, str]:
    """R1/R2 endpoint- and context-aware candidate-pool identity."""
    source = str(row.get("source_id") or normalize(row.get("source_sequence")))
    return (
        str(row.get("benchmark") or ""),
        str(row.get("study") or ""),
        str(row.get("endpoint") or ""),
        assay_context(row),
        source,
    )


def split_group_key(row: Dict) -> Tuple[str, str, str, str]:
    """Conservative group used to keep a source out of both train and dev."""
    source = str(row.get("source_id") or normalize(row.get("source_sequence")))
    return (
        str(row.get("benchmark") or ""),
        str(row.get("endpoint") or ""),
        assay_context(row),
        source,
    )
