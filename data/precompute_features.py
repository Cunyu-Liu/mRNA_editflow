"""Offline thermodynamic / structural feature precomputation for mRNA-EditFlow.

Produces one :class:`~mrna_editflow.core.schema.PrecomputedFeatures` per
transcript (global MFE estimate, start-codon accessibility, per-nt pairing
propensity), writes them as NPZ shards, and records a ``manifest.jsonl`` with
per-shard SHA256 and corpus statistics used for provenance / auditing.

Backends (auto-selected, most accurate first)
---------------------------------------------
1. **ViennaRNA python module** (``import RNA``) if importable.
2. **RNAfold binary** on ``PATH`` (parsed dot-bracket + MFE).
3. **Pure-python fallback** (default offline path):
   * *MFE proxy*: a Nussinov-style base-pairing DP that maximises total
     stacking-weighted pair energy (GC=3, AU=2, GU=1 kcal/mol) and reports
     ``mfe = -(max pair energy)``. To stay strictly **O(L^2)** in time and
     memory we omit the multi-loop *bifurcation* term of full Nussinov (a
     documented simplification: nested/stem structures are captured, arbitrary
     branching is not), and additionally **cap the folded length** to
     ``mfe_cap_len`` (5'-proximal window) so per-record cost is bounded.
   * *Start accessibility*: ``1 - <local pairing propensity>`` averaged over a
     window straddling the start codon, where the propensity at position ``i``
     is the fraction of nearby positions that could Watson-Crick / wobble pair
     with ``i`` (O(L * window), linear in L).
   * *pairing_prob*: the full-length per-nt pairing propensity vector.

Complexity per record: O(min(L, cap)^2) for the MFE proxy + O(L * window) for
accessibility / propensity.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from typing import Dict, List, Optional, Sequence

import numpy as np

from mrna_editflow.core.config import DataConfig
from mrna_editflow.core.schema import MRNARecord, PrecomputedFeatures

_MIN_LOOP = 3  # minimum hairpin loop (nt between paired bases)
_PAIR_ENERGY = {
    ("G", "C"): 3.0, ("C", "G"): 3.0,
    ("A", "U"): 2.0, ("U", "A"): 2.0,
    ("G", "U"): 1.0, ("U", "G"): 1.0,
}

# Integer nucleotide codes (unknown -> 4) and a 5x5 pair-energy lookup used by
# the vectorised Nussinov DP. Index 4 is the "other/unknown" sink (0 energy).
_NUC_CODE = {"A": 0, "C": 1, "G": 2, "U": 3}
_ENERGY_MAT = np.zeros((5, 5), dtype=np.float64)
_ENERGY_MAT[2, 1] = _ENERGY_MAT[1, 2] = 3.0  # G-C
_ENERGY_MAT[0, 3] = _ENERGY_MAT[3, 0] = 2.0  # A-U
_ENERGY_MAT[2, 3] = _ENERGY_MAT[3, 2] = 1.0  # G-U wobble


def _can_pair(a: str, b: str) -> bool:
    return (a, b) in _PAIR_ENERGY


# ---------------------------------------------------------------------------
# Backend availability
# ---------------------------------------------------------------------------
def _vienna_module():
    try:
        import RNA  # type: ignore

        return RNA
    except Exception:
        return None


def _rnafold_binary() -> Optional[str]:
    return shutil.which("RNAfold")


# ---------------------------------------------------------------------------
# Pure-python fallback
# ---------------------------------------------------------------------------
def _nussinov_mfe_proxy(seq: str, cap_len: int) -> float:
    """MFE proxy via an O(L^2) nested base-pairing DP (no multiloop term).

    The sequence is capped to its 5'-proximal ``cap_len`` window (initiation
    context matters most and this bounds the quadratic cost). Returns a negative
    ``mfe`` estimate: more/stronger pairs -> more negative. Always finite.

    Recurrence (``e`` = pair energy, ``0`` for non-pairs)::

        dp[i][j] = max( dp[i+1][j],            # i unpaired
                        dp[i][j-1],            # j unpaired
                        dp[i+1][j-1] + e(i,j)  # i,j paired (loop >= _MIN_LOOP)
                      )

    Implemented by filling anti-diagonals of increasing span; each span is
    vectorised over ``i`` with numpy so the constant factor is small while the
    asymptotics remain O(L^2) time / O(L^2) memory (L = min(len(seq), cap_len)).
    """
    s = seq[:cap_len]
    L = len(s)
    if L < 2:
        return 0.0
    codes = np.fromiter((_NUC_CODE.get(c, 4) for c in s), dtype=np.int64, count=L)
    dp = np.zeros((L, L), dtype=np.float64)
    for span in range(1, L):
        i = np.arange(0, L - span)
        j = i + span
        # i unpaired: dp[i+1, j]; j unpaired: dp[i, j-1]
        cand = np.maximum(dp[i + 1, j], dp[i, j - 1])
        # i,j paired (only when the loop is long enough)
        if span > _MIN_LOOP:
            e = _ENERGY_MAT[codes[i], codes[j]]
            paired = dp[i + 1, j - 1] + e
            cand = np.maximum(cand, paired)
        dp[i, j] = cand
    return float(-dp[0, L - 1])


def _pairing_propensity(seq: str, window: int) -> np.ndarray:
    """Per-nt local pairing propensity in ``[0, 1]``.

    ``propensity[i]`` = fraction of positions ``j`` in ``[i-window, i+window]``
    (excluding ``|i-j| <= _MIN_LOOP``) that can Watson-Crick / wobble pair with
    ``seq[i]``. A crude but genuine proxy for how likely ``i`` is base-paired.

    Complexity: O(L * window) time, O(L) memory.
    """
    L = len(seq)
    prop = np.zeros(L, dtype=np.float32)
    if L == 0:
        return prop
    for i in range(L):
        lo = max(0, i - window)
        hi = min(L, i + window + 1)
        cnt = 0
        tot = 0
        ci = seq[i]
        for j in range(lo, hi):
            if abs(i - j) <= _MIN_LOOP:
                continue
            tot += 1
            if _can_pair(ci, seq[j]):
                cnt += 1
        prop[i] = (cnt / tot) if tot > 0 else 0.0
    return prop


def _fallback_features(
    record: MRNARecord, cap_len: int, start_window: int, pair_window: int
) -> PrecomputedFeatures:
    seq = record.seq
    mfe = _nussinov_mfe_proxy(seq, cap_len)
    prop = _pairing_propensity(seq, pair_window)
    start = record.cds_start
    lo = max(0, start - start_window)
    hi = min(len(seq), start + start_window + 1)
    local = prop[lo:hi]
    access = 1.0 - float(local.mean()) if local.size else 1.0
    access = min(1.0, max(0.0, access))
    return PrecomputedFeatures(
        transcript_id=record.transcript_id,
        mfe=float(mfe),
        start_accessibility=float(access),
        pairing_prob=[float(x) for x in prop.tolist()],
    )


# ---------------------------------------------------------------------------
# ViennaRNA backends (used only when available; not exercised in offline tests)
# ---------------------------------------------------------------------------
def _paired_flags_from_dotbracket(structure: str) -> np.ndarray:
    return np.array([0.0 if c == "." else 1.0 for c in structure], dtype=np.float32)


def _vienna_features(record: MRNARecord, RNA, start_window: int) -> PrecomputedFeatures:
    seq = record.seq
    structure, mfe = RNA.fold(seq)  # type: ignore[attr-defined]
    paired = _paired_flags_from_dotbracket(structure)
    start = record.cds_start
    lo = max(0, start - start_window)
    hi = min(len(seq), start + start_window + 1)
    local = paired[lo:hi]
    access = 1.0 - float(local.mean()) if local.size else 1.0
    return PrecomputedFeatures(
        transcript_id=record.transcript_id,
        mfe=float(mfe),
        start_accessibility=float(min(1.0, max(0.0, access))),
        pairing_prob=[float(x) for x in paired.tolist()],
    )


def _rnafold_features(record: MRNARecord, binary: str, start_window: int) -> PrecomputedFeatures:
    seq = record.seq
    proc = subprocess.run(
        [binary, "--noPS"], input=seq + "\n", capture_output=True, text=True
    )
    if proc.returncode != 0:
        raise RuntimeError(f"RNAfold failed: {proc.stderr[-500:]}")
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    struct_line = lines[1]  # "structure ( -12.30)"
    structure = struct_line.split()[0]
    mfe = float(struct_line[struct_line.rfind("(") + 1: struct_line.rfind(")")])
    paired = _paired_flags_from_dotbracket(structure)
    start = record.cds_start
    lo = max(0, start - start_window)
    hi = min(len(seq), start + start_window + 1)
    local = paired[lo:hi]
    access = 1.0 - float(local.mean()) if local.size else 1.0
    return PrecomputedFeatures(
        transcript_id=record.transcript_id,
        mfe=float(mfe),
        start_accessibility=float(min(1.0, max(0.0, access))),
        pairing_prob=[float(x) for x in paired.tolist()],
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def detect_backend() -> str:
    """Return the backend that :func:`compute_features` will use.

    One of ``"vienna_module"``, ``"rnafold_binary"`` or ``"fallback"``.
    """
    if _vienna_module() is not None:
        return "vienna_module"
    if _rnafold_binary() is not None:
        return "rnafold_binary"
    return "fallback"


def compute_features(
    record: MRNARecord,
    mfe_cap_len: int = 600,
    start_window: int = 20,
    pair_window: int = 40,
) -> PrecomputedFeatures:
    """Compute features for one record using the best available backend.

    ``mfe_cap_len`` bounds the fallback MFE proxy's folded length. ``start_window``
    is the +/- nt window around the start codon for accessibility; ``pair_window``
    is the local window for the fallback pairing propensity. Complexity: see
    module docstring.
    """
    RNA = _vienna_module()
    if RNA is not None:
        return _vienna_features(record, RNA, start_window)
    binary = _rnafold_binary()
    if binary is not None:
        return _rnafold_features(record, binary, start_window)
    return _fallback_features(record, mfe_cap_len, start_window, pair_window)


def _sha256_file(path: str) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _dist(values: Sequence[float]) -> Dict[str, float]:
    """Summary statistics for a list of floats (empty -> zeros)."""
    if len(values) == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "p50": 0.0, "p90": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
    }


def _write_shard(
    path: str, feats: List[PrecomputedFeatures], records: List[MRNARecord]
) -> None:
    """Write one NPZ shard (no pickle: strings as ``<U`` dtype, ragged arrays
    flattened + offsets). Complexity: O(sum of transcript lengths)."""
    ids = np.array([f.transcript_id for f in feats])
    mfe = np.array([f.mfe for f in feats], dtype=np.float64)
    access = np.array([f.start_accessibility for f in feats], dtype=np.float64)
    lengths = np.array([len(f.pairing_prob or []) for f in feats], dtype=np.int64)
    flat = (
        np.concatenate([np.asarray(f.pairing_prob, dtype=np.float32) for f in feats])
        if feats else np.empty(0, dtype=np.float32)
    )
    five = np.array([len(r.five_utr) for r in records], dtype=np.int64)
    cds = np.array([len(r.cds) for r in records], dtype=np.int64)
    three = np.array([len(r.three_utr) for r in records], dtype=np.int64)
    np.savez(
        path,
        transcript_ids=ids,
        mfe=mfe,
        start_accessibility=access,
        pairing_prob_flat=flat,
        pairing_prob_lengths=lengths,
        five_len=five,
        cds_len=cds,
        three_len=three,
    )


def precompute_corpus(
    records: Sequence[MRNARecord],
    out_dir: str,
    cfg: Optional[DataConfig] = None,
    drop_stats: Optional[Dict[str, int]] = None,
    shard_size: int = 64,
    mfe_cap_len: int = 600,
    start_window: int = 20,
    pair_window: int = 40,
) -> str:
    """Compute features for a corpus, write NPZ shards + ``manifest.jsonl``.

    Returns the manifest path. The manifest is JSON-lines:

    * **line 0** — a *summary* object::

        {"type": "summary", "backend": str, "n_total": int, "n_shards": int,
         "cleaning_drop_counts": {reason: count, ...} | null,
         "length_distribution": {min,max,mean,p50,p90},
         "region_fractions": {"5utr": f, "cds": f, "3utr": f}}

    * **lines 1..n_shards** — one *shard* object each::

        {"type": "shard", "shard": "features_00000.npz", "sha256": hex,
         "n_seqs": int,
         "length_distribution": {min,max,mean,p50,p90},
         "region_fractions": {"5utr": f, "cds": f, "3utr": f},
         "mfe": {min,max,mean,p50,p90},
         "start_accessibility": {min,max,mean,p50,p90}}

    Complexity: O(sum over records of per-record feature cost).
    """
    if cfg is None:
        cfg = DataConfig()
    os.makedirs(out_dir, exist_ok=True)
    backend = detect_backend()

    manifest_path = os.path.join(out_dir, "manifest.jsonl")
    shard_lines: List[Dict] = []
    all_lengths: List[int] = []
    tot_five = tot_cds = tot_three = 0

    n = len(records)
    n_shards = 0
    for start_idx in range(0, n, shard_size):
        chunk = list(records[start_idx:start_idx + shard_size])
        feats = [
            compute_features(r, mfe_cap_len, start_window, pair_window) for r in chunk
        ]
        shard_name = f"features_{n_shards:05d}.npz"
        shard_path = os.path.join(out_dir, shard_name)
        _write_shard(shard_path, feats, chunk)

        lengths = [len(r.seq) for r in chunk]
        five = sum(len(r.five_utr) for r in chunk)
        cds = sum(len(r.cds) for r in chunk)
        three = sum(len(r.three_utr) for r in chunk)
        total_nt = max(1, five + cds + three)
        all_lengths.extend(lengths)
        tot_five += five
        tot_cds += cds
        tot_three += three

        shard_lines.append({
            "type": "shard",
            "shard": shard_name,
            "sha256": _sha256_file(shard_path),
            "n_seqs": len(chunk),
            "length_distribution": _dist(lengths),
            "region_fractions": {
                "5utr": five / total_nt,
                "cds": cds / total_nt,
                "3utr": three / total_nt,
            },
            "mfe": _dist([f.mfe for f in feats]),
            "start_accessibility": _dist([f.start_accessibility for f in feats]),
        })
        n_shards += 1

    overall_total = max(1, tot_five + tot_cds + tot_three)
    summary = {
        "type": "summary",
        "backend": backend,
        "n_total": n,
        "n_shards": n_shards,
        "cleaning_drop_counts": drop_stats,
        "length_distribution": _dist(all_lengths),
        "region_fractions": {
            "5utr": tot_five / overall_total,
            "cds": tot_cds / overall_total,
            "3utr": tot_three / overall_total,
        },
    }
    with open(manifest_path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(summary, ensure_ascii=False) + "\n")
        for line in shard_lines:
            fh.write(json.dumps(line, ensure_ascii=False) + "\n")
    return manifest_path


def load_features(out_dir: str) -> Dict[str, PrecomputedFeatures]:
    """Load all features written by :func:`precompute_corpus` into a dict.

    Reads every shard listed in ``manifest.jsonl`` (verifying nothing here; use
    :func:`verify_manifest` for integrity). Complexity: O(sum of nt).
    """
    manifest_path = os.path.join(out_dir, "manifest.jsonl")
    features: Dict[str, PrecomputedFeatures] = {}
    with open(manifest_path, "r", encoding="utf-8") as fh:
        for line in fh:
            obj = json.loads(line)
            if obj.get("type") != "shard":
                continue
            shard_path = os.path.join(out_dir, obj["shard"])
            data = np.load(shard_path, allow_pickle=False)
            ids = data["transcript_ids"]
            mfe = data["mfe"]
            access = data["start_accessibility"]
            flat = data["pairing_prob_flat"]
            lengths = data["pairing_prob_lengths"]
            offset = 0
            for i, tid in enumerate(ids):
                L = int(lengths[i])
                pp = flat[offset:offset + L].astype(np.float32).tolist()
                offset += L
                features[str(tid)] = PrecomputedFeatures(
                    transcript_id=str(tid),
                    mfe=float(mfe[i]),
                    start_accessibility=float(access[i]),
                    pairing_prob=[float(x) for x in pp],
                )
    return features


def verify_manifest(out_dir: str) -> bool:
    """Recompute each shard's SHA256 and confirm it matches the manifest.

    Complexity: O(bytes on disk).
    """
    manifest_path = os.path.join(out_dir, "manifest.jsonl")
    with open(manifest_path, "r", encoding="utf-8") as fh:
        for line in fh:
            obj = json.loads(line)
            if obj.get("type") != "shard":
                continue
            shard_path = os.path.join(out_dir, obj["shard"])
            if not os.path.exists(shard_path):
                return False
            if _sha256_file(shard_path) != obj["sha256"]:
                return False
    return True


__all__ = [
    "detect_backend",
    "compute_features",
    "precompute_corpus",
    "load_features",
    "verify_manifest",
]
