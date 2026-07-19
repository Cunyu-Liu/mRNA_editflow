"""Family-disjoint train/val/test splitting for mRNA-EditFlow.

Redundant transcripts (paralogues, isoforms, near-duplicate synthetic variants)
must not straddle the train/test boundary or evaluation leaks. We therefore
cluster sequences into *families* and assign whole clusters to a single split,
which makes train/test cluster-disjointness true **by construction** (and we
additionally assert it).

Two clustering backends
-----------------------
* **MMseqs2** (used automatically iff the ``mmseqs`` binary is on ``PATH``):
  shells out to ``mmseqs easy-cluster --min-seq-id cfg.mmseqs_min_seq_id`` and
  parses the ``*_cluster.tsv`` representative/member table.
* **Pure-python MinHash-LSH fallback** (default offline path): a real MinHash
  sketch over nucleotide k-mers with LSH banding for candidate generation and a
  union-find over pairs whose estimated Jaccard >= ``cfg.mmseqs_min_seq_id``.
  This is a genuine approximate-Jaccard clustering, documented as a fallback for
  when MMseqs2 is unavailable.

Determinism: MinHash permutations and the cluster->split assignment are seeded
from ``cfg.seed``, so splits are fully reproducible.

Complexity: MinHash sketching is O(N * L) for N sequences of length L (fixed
``num_perm``); LSH candidate generation is ~O(N) buckets; union-find is near
O(pairs). MMseqs2 backend is delegated to the external binary.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from mrna_editflow.core.config import DataConfig
from mrna_editflow.core.schema import MRNARecord

_MERSENNE_P = (1 << 31) - 1  # 31-bit Mersenne prime for the (a*x+b) mod p family
# With a,b < 2**31 and x reduced mod _MERSENNE_P (< 2**31), the product a*x is
# < 2**62 and a*x+b < 2**63, so all MinHash arithmetic stays within uint64.


@dataclass
class SplitResult:
    """Outcome of a family-disjoint split.

    Attributes
    ----------
    train / val / test: sorted lists of record indices (into the input list).
    clusters: list of clusters, each a sorted list of record indices.
    method: ``"mmseqs"`` or ``"minhash"``.
    n_clusters: convenience ``len(clusters)``.
    paths: written ``{split: idx_file_path}`` (empty if not written).
    """
    train: List[int]
    val: List[int]
    test: List[int]
    clusters: List[List[int]]
    method: str
    n_clusters: int = 0
    paths: Dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# MinHash-LSH fallback clustering
# ---------------------------------------------------------------------------
def _kmer_codes(seq: str, k: int) -> np.ndarray:
    """Deterministic uint64 codes for the distinct k-mers of ``seq``.

    Uses a fixed polynomial rolling hash (base-5 over ``ACGU`` + sentinel) so
    codes are reproducible across processes (Python's builtin ``hash`` is
    salted and must not be used). Sequences shorter than ``k`` hash as a single
    whole-sequence token. Complexity: O(len(seq)).
    """
    n = len(seq)
    if n == 0:
        return np.empty(0, dtype=np.uint64)
    base = np.uint64(131)
    code_map = {"A": 1, "C": 2, "G": 3, "U": 4, "N": 0}
    if n < k:
        h = np.uint64(0)
        for ch in seq:
            h = h * base + np.uint64(code_map.get(ch, 0))
        return np.array([h], dtype=np.uint64)
    codes = np.empty(n - k + 1, dtype=np.uint64)
    for i in range(n - k + 1):
        h = np.uint64(0)
        for ch in seq[i:i + k]:
            h = h * base + np.uint64(code_map.get(ch, 0))
        codes[i] = h
    return np.unique(codes)


def _minhash_signature(codes: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """MinHash signature: for each permutation ``j`` take ``min_x (a_j*x+b_j) mod p``.

    ``a`` (in ``[1, 2**31)``) and ``b`` (in ``[0, 2**31)``) are kept small so the
    product with 64-bit k-mer codes stays within int64 before the mod. Empty
    ``codes`` (only for empty sequences) yield an all-``p`` signature so such
    items never collide with real ones. Complexity: O(num_perm * |codes|).
    """
    num_perm = a.shape[0]
    if codes.size == 0:
        return np.full(num_perm, _MERSENNE_P, dtype=np.uint64)
    # (num_perm, m) modular hashes; reduce k-mer codes mod p first to bound size.
    x = (codes % np.uint64(_MERSENNE_P)).astype(np.uint64)
    hashed = (np.outer(a, x) + b[:, None]) % np.uint64(_MERSENNE_P)
    return hashed.min(axis=1).astype(np.uint64)


def _optimal_bands(num_perm: int, threshold: float) -> Tuple[int, int]:
    """Pick ``(bands, rows)`` with ``bands*rows == num_perm`` whose LSH knee
    ``(1/bands)**(1/rows)`` is closest to ``threshold``.

    Complexity: O(num_perm) over the divisors scanned.
    """
    best = (num_perm, 1)
    best_err = float("inf")
    for rows in range(1, num_perm + 1):
        if num_perm % rows != 0:
            continue
        bands = num_perm // rows
        knee = (1.0 / bands) ** (1.0 / rows)
        err = abs(knee - threshold)
        if err < best_err:
            best_err = err
            best = (bands, rows)
    return best


def _jaccard_threshold_from_identity(identity: float, k: int) -> float:
    """Convert a sequence-identity threshold to an equivalent k-mer Jaccard.

    Uses the Mash relationship (Ondov et al. 2016) between the Jaccard index
    ``J`` of two sequences' k-mer sets and their mutation distance
    ``D = 1 - identity``::

        2J / (1 + J) = exp(-k * D)   =>   J = m / (2 - m),  m = exp(-k * D)

    This lets the pure-python fallback use the *same* ``min_seq_id`` scale as
    MMseqs2 (which thresholds on sequence identity), instead of the much
    stricter raw k-mer Jaccard. Complexity: O(1).
    """
    import math

    d = max(0.0, 1.0 - identity)
    m = math.exp(-k * d)
    j = m / (2.0 - m)
    # Clamp away from the degenerate extremes for numerical safety.
    return min(0.999, max(1e-4, j))


def cluster_sequences(
    seqs: Sequence[str],
    min_seq_id: float,
    seed: int,
    k: int = 8,
    num_perm: int = 128,
) -> List[List[int]]:
    """Cluster sequences into families via MinHash-LSH + union-find.

    ``min_seq_id`` is a *sequence-identity* threshold (same scale as MMseqs2's
    ``--min-seq-id``); it is converted to the corresponding k-mer Jaccard
    threshold via :func:`_jaccard_threshold_from_identity`. Two sequences are
    unioned when their MinHash-estimated Jaccard (fraction of agreeing signature
    entries) is ``>=`` that converted threshold. LSH banding restricts candidate
    pairs so we never materialise the full O(N^2) matrix. Returns a list of
    clusters (each a sorted list of indices) covering all inputs.

    Deterministic in ``seed``. Complexity: O(N * L) sketching + ~O(candidates).
    """
    n = len(seqs)
    if n == 0:
        return []
    jacc_thr = _jaccard_threshold_from_identity(min_seq_id, k)
    rng = np.random.RandomState(seed & 0x7FFFFFFF)
    a = rng.randint(1, 1 << 31, size=num_perm).astype(np.uint64)
    b = rng.randint(0, 1 << 31, size=num_perm).astype(np.uint64)

    signatures = np.empty((n, num_perm), dtype=np.uint64)
    for i, s in enumerate(seqs):
        signatures[i] = _minhash_signature(_kmer_codes(s, k), a, b)

    # Union-Find over LSH candidate pairs confirmed by estimated Jaccard.
    parent = list(range(n))

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:  # path compression
            parent[x], x = root, parent[x]
        return root

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[max(rx, ry)] = min(rx, ry)

    bands, rows = _optimal_bands(num_perm, jacc_thr)
    checked: set = set()
    for band in range(bands):
        buckets: Dict[bytes, List[int]] = {}
        cols = signatures[:, band * rows:(band + 1) * rows]
        for i in range(n):
            key = cols[i].tobytes()
            buckets.setdefault(key, []).append(i)
        for members in buckets.values():
            if len(members) < 2:
                continue
            for ai in range(len(members)):
                for bi in range(ai + 1, len(members)):
                    u, v = members[ai], members[bi]
                    if find(u) == find(v):
                        continue
                    pair = (u, v) if u < v else (v, u)
                    if pair in checked:
                        continue
                    checked.add(pair)
                    est = float(np.mean(signatures[u] == signatures[v]))
                    if est >= jacc_thr:
                        union(u, v)

    groups: Dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [sorted(members) for members in groups.values()]


# ---------------------------------------------------------------------------
# MMseqs2 backend (used only when the binary is present)
# ---------------------------------------------------------------------------
def _mmseqs_available() -> bool:
    return shutil.which("mmseqs") is not None


def _cluster_with_mmseqs(
    seqs: Sequence[str], min_seq_id: float, coverage: float = 0.8
) -> List[List[int]]:
    """Cluster via ``mmseqs easy-cluster``; parse the representative/member TSV.

    Sequence index is used as the FASTA id so we can map members back to input
    positions. Raises ``RuntimeError`` if the binary fails. Complexity: external.
    """
    n = len(seqs)
    with tempfile.TemporaryDirectory() as tmp:
        fasta = os.path.join(tmp, "in.fasta")
        with open(fasta, "w", encoding="utf-8") as fh:
            for i, s in enumerate(seqs):
                fh.write(f">{i}\n{s}\n")
        out_prefix = os.path.join(tmp, "clu")
        cmd = [
            "mmseqs", "easy-cluster", fasta, out_prefix, os.path.join(tmp, "mmtmp"),
            "--min-seq-id", str(min_seq_id), "-c", str(coverage),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"mmseqs easy-cluster failed: {proc.stderr[-2000:]}")
        tsv = out_prefix + "_cluster.tsv"
        rep_to_members: Dict[str, List[int]] = {}
        with open(tsv, "r", encoding="utf-8") as fh:
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) != 2:
                    continue
                rep, member = parts
                rep_to_members.setdefault(rep, []).append(int(member))
    clusters = [sorted(m) for m in rep_to_members.values()]
    seen = {i for c in clusters for i in c}
    for i in range(n):  # singletons mmseqs may omit
        if i not in seen:
            clusters.append([i])
    return clusters


# ---------------------------------------------------------------------------
# Cluster -> split assignment
# ---------------------------------------------------------------------------
def _assign_clusters(
    clusters: List[List[int]], cfg: DataConfig, n_records: int
) -> Tuple[List[int], List[int], List[int]]:
    """Assign whole clusters to train/val/test to approximate the target fractions.

    Clusters are shuffled deterministically then each is placed in whichever
    split is currently furthest *below* its target count (largest deficit),
    which keeps fractions close while never splitting a cluster. Complexity:
    O(n_clusters).
    """
    rng = np.random.RandomState(cfg.seed & 0x7FFFFFFF)
    order = list(range(len(clusters)))
    rng.shuffle(order)
    # Largest clusters first (within the shuffled order) improves balance.
    order.sort(key=lambda ci: -len(clusters[ci]))

    targets = {
        "train": cfg.train_frac * n_records,
        "val": cfg.val_frac * n_records,
        "test": cfg.test_frac * n_records,
    }
    counts = {"train": 0, "val": 0, "test": 0}
    buckets: Dict[str, List[int]] = {"train": [], "val": [], "test": []}
    for ci in order:
        members = clusters[ci]
        # deficit = how far below target each split is (skip zero-target splits).
        deficits = {
            s: targets[s] - counts[s]
            for s in ("train", "val", "test")
            if targets[s] > 0
        }
        if not deficits:
            chosen = "train"
        else:
            chosen = max(deficits, key=lambda s: deficits[s])
        buckets[chosen].extend(members)
        counts[chosen] += len(members)
    return (sorted(buckets["train"]), sorted(buckets["val"]), sorted(buckets["test"]))


def _write_idx(out_dir: str, name: str, indices: List[int]) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.idx")
    with open(path, "w", encoding="utf-8") as fh:
        for i in indices:
            fh.write(f"{i}\n")
    return path


def read_idx(path: str) -> List[int]:
    """Read an ``.idx`` file back into a list of ints. Complexity: O(lines)."""
    with open(path, "r", encoding="utf-8") as fh:
        return [int(line.strip()) for line in fh if line.strip()]


def family_disjoint_split(
    records: Sequence[MRNARecord],
    cfg: Optional[DataConfig] = None,
    out_dir: Optional[str] = None,
    use_mmseqs: str = "auto",
    write: bool = True,
) -> SplitResult:
    """Produce a family-disjoint split and (optionally) write ``.idx`` files.

    Parameters
    ----------
    records: cleaned records to split.
    cfg: :class:`DataConfig` (defaults to ``DataConfig()``); supplies fractions,
        ``mmseqs_min_seq_id`` and ``seed``.
    out_dir: directory for ``{train,val,test}.idx`` (defaults to
        ``{cfg.data_dir}/splits``). Only used when ``write=True``.
    use_mmseqs: ``"auto"`` (use the binary iff present), ``"never"`` (force the
        MinHash fallback), or ``"force"`` (require the binary).
    write: whether to write the ``.idx`` files.

    Guarantees & asserts that train/val/test clusters are pairwise disjoint,
    i.e. no family straddles splits. Reproducible via ``cfg.seed``.
    """
    if cfg is None:
        cfg = DataConfig()
    seqs = [r.seq for r in records]
    n = len(seqs)

    want_mmseqs = (use_mmseqs == "force") or (use_mmseqs == "auto" and _mmseqs_available())
    if use_mmseqs == "force" and not _mmseqs_available():
        raise RuntimeError("use_mmseqs='force' but the mmseqs binary is not on PATH")
    if want_mmseqs:
        clusters = _cluster_with_mmseqs(seqs, cfg.mmseqs_min_seq_id)
        method = "mmseqs"
    else:
        clusters = cluster_sequences(seqs, cfg.mmseqs_min_seq_id, cfg.seed)
        method = "minhash"

    train, val, test = _assign_clusters(clusters, cfg, n)

    # ---- Hard guarantee: family-disjointness across splits ----
    s_train, s_val, s_test = set(train), set(val), set(test)
    assert not (s_train & s_test), "train/test index overlap detected"
    assert not (s_train & s_val), "train/val index overlap detected"
    assert not (s_val & s_test), "val/test index overlap detected"
    assert len(s_train) + len(s_val) + len(s_test) == n, "split does not cover all records"
    # Every cluster lands entirely in one split (the real leakage guarantee).
    for cl in clusters:
        cset = set(cl)
        in_train = bool(cset & s_train)
        in_val = bool(cset & s_val)
        in_test = bool(cset & s_test)
        assert (in_train + in_val + in_test) <= 1, "a family/cluster was split across sets"

    paths: Dict[str, str] = {}
    if write:
        target_dir = out_dir if out_dir is not None else os.path.join(cfg.data_dir, "splits")
        paths = {
            "train": _write_idx(target_dir, "train", train),
            "val": _write_idx(target_dir, "val", val),
            "test": _write_idx(target_dir, "test", test),
        }

    return SplitResult(
        train=train, val=val, test=test, clusters=clusters,
        method=method, n_clusters=len(clusters), paths=paths,
    )


__all__ = [
    "SplitResult",
    "family_disjoint_split",
    "cluster_sequences",
    "read_idx",
]
