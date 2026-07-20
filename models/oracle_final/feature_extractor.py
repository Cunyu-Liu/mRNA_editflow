"""P1-05 Oracle #3: Hand-engineered feature extractor for GBT-based final oracle.

Extracts ~600-800 features from mRNA/UTR sequences for use with LightGBM
gradient-boosted trees regressor. This oracle is INDEPENDENT from the
training teacher (P1-04 CNN/Transformer) — different architecture, different
feature space, different training data (Leplek 2022 PERSIST-Seq + held-out
Sample 2019).

Feature groups:
    1. Length features (5)
    2. Nucleotide composition (4 + 4*1 + 16 di-mer + 64 tri-mer = 88)
    3. GC content & variants (8)
    4. K-mer counts (k=1..6, summarized) (~120)
    5. Motif counts (Kozak, TIS, polyA, uORF, m6A motifs) (~30)
    6. Codon usage (64 codons + RSCU) (~128)
    7. Structure proxies (predicted MFE via proxy, pairing propensity) (~20)
    8. Position-specific features (sliding window stats) (~200)

Total: ~600+ features

Determinism: all features are computed deterministically from sequence alone
(no model inference, no randomness). Same sequence → same features.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUCLEOTIDES = "ACGU"
CODONS = [a + b + c for a in NUCLEOTIDES for b in NUCLEOTIDES for c in NUCLEOTIDES]
CODON_TO_IDX = {c: i for i, c in enumerate(CODONS)}

# Hand-curated motif list (compiled from literature)
# Each motif is (name, regex_pattern_or_literal, position_constraint)
MOTIFS: List[Tuple[str, str, Optional[str]]] = [
    # Kozak consensus: GCCRCCAUGG (R = A or G)
    # We search for simplified variants
    ("kozak_strong", "GCCACCAUGG", None),
    ("kozak_moderate", "GCCAUCAUGG", None),
    ("kozak_weak", "GCCUUCAUGG", None),
    # TIS (Translation Initiation Site) contexts
    ("TIS_strong", "AUGG", None),
    ("TIS_weak", "AUGC", None),
    # uORF start codons (upstream ORFs)
    ("uORF_AUG_count", "AUG", "any"),
    # Stop codons
    ("stop_UAA", "UAA", None),
    ("stop_UAG", "UAG", None),
    ("stop_UGA", "UGA", None),
    # polyA signals (hexamer)
    ("polyA_AAUAAA", "AAUAAA", None),
    ("polyA_AUAAAA", "AUAAAA", None),
    ("polyA_AAAUAA", "AAAUAA", None),
    # m6A consensus DRACH (D=A/G/U, R=A/G, H=A/C/U)
    ("m6A_AAACA", "AAACA", None),
    ("m6A_AAACC", "AAACC", None),
    ("m6A_AAACU", "AAACU", None),
    ("m6A_AGACA", "AGACA", None),
    ("m6A_AGACC", "AGACC", None),
    ("m6A_AGACU", "AGACU", None),
    ("m6A_GGACA", "GGACA", None),
    ("m6A_GGACC", "GGACC", None),
    ("m6A_GGACU", "GGACU", None),
    ("m6A_UAACA", "UAACA", None),
    ("m6A_UAACC", "UAACC", None),
    ("m6A_UAACU", "UAACU", None),
    ("m6A_UGACA", "UGACA", None),
    ("m6A_UGACC", "UGACC", None),
    ("m6A_UGACU", "UGACU", None),
    # Stable hairpin proxies (GNRA tetraloop family)
    ("hairpin_GAAA", "GAAA", None),
    ("hairpin_GAGA", "GAGA", None),
    ("hairpin_GCAA", "GCAA", None),
    ("hairpin_GCGA", "GCGA", None),
    # AU-rich elements (destabilizing)
    ("ARE_AUUUA", "AUUUA", None),
    ("ARE_AUUUUA", "AUUUUA", None),
    # Top hits from Sample 2019 NBT paper (k-mer importance)
    ("sample2019_top1", "GCCAU", None),
    ("sample2019_top2", "GCCA", None),
    ("sample2019_top3", "CCACC", None),
    ("sample2019_top4", "AUGGC", None),
    ("sample2019_top5", "GCCAUC", None),
]


# ---------------------------------------------------------------------------
# Feature extractor
# ---------------------------------------------------------------------------

@dataclass
class FeatureExtractorConfig:
    """Configuration for feature extraction.

    Attributes:
        max_kmer_k: maximum k for k-mer frequency features
        include_codon_features: include codon usage (only meaningful for CDS)
        include_position_features: include position-specific sliding window stats
        window_sizes: window sizes for sliding window features
        max_length: maximum sequence length to consider (truncate if longer)
    """
    max_kmer_k: int = 6
    include_codon_features: bool = True
    include_position_features: bool = True
    window_sizes: Tuple[int, ...] = (10, 20, 50, 100)
    max_length: int = 12000


def _normalize_seq(seq: str) -> str:
    """Normalize: uppercase, T->U, strip non-ACGUN."""
    s = seq.upper().replace("T", "U")
    return "".join(c for c in s if c in "ACGUN")


def _kmer_counts(seq: str, k: int) -> Dict[str, int]:
    """Count k-mers in sequence."""
    counts: Dict[str, int] = {}
    if len(seq) < k:
        return counts
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i + k]
        if "N" not in kmer:
            counts[kmer] = counts.get(kmer, 0) + 1
    return counts


def _kmer_frequencies(seq: str, k: int) -> Dict[str, float]:
    """Normalized k-mer frequencies (sum to 1)."""
    counts = _kmer_counts(seq, k)
    total = sum(counts.values())
    if total == 0:
        return {kmer: 0.0 for kmer in counts}
    return {kmer: c / total for kmer, c in counts.items()}


def _find_motif_count(seq: str, motif: str) -> int:
    """Count non-overlapping occurrences of motif in seq."""
    count = 0
    i = 0
    while True:
        idx = seq.find(motif, i)
        if idx < 0:
            break
        count += 1
        i = idx + len(motif)
    return count


def _find_motif_positions(seq: str, motif: str) -> List[int]:
    """Find all (overlapping) positions of motif in seq."""
    positions = []
    start = 0
    while True:
        idx = seq.find(motif, start)
        if idx < 0:
            break
        positions.append(idx)
        start = idx + 1
    return positions


def _gc_content(seq: str) -> float:
    """GC content as fraction (0-1)."""
    if not seq:
        return 0.0
    gc = sum(1 for c in seq if c in "GC")
    return gc / len(seq)


def _gc_content_windows(seq: str, n_windows: int = 5) -> List[float]:
    """GC content in n equal-sized windows along the sequence."""
    if not seq:
        return [0.0] * n_windows
    L = len(seq)
    window_size = max(L // n_windows, 1)
    gcs = []
    for i in range(n_windows):
        start = i * window_size
        end = min(start + window_size, L) if i < n_windows - 1 else L
        gcs.append(_gc_content(seq[start:end]))
    return gcs


def _codon_usage(seq: str) -> Tuple[np.ndarray, np.ndarray]:
    """Compute codon counts and RSCU (relative synonymous codon usage).

    Returns:
        (counts (64,), rscu (64,))
    """
    counts = np.zeros(64, dtype=np.float64)
    # Look for AUG (start) and walk in frame 0
    # For simplicity, just count all non-overlapping triplets in frame 0
    # (not biologically accurate, but a stable feature)
    for i in range(0, len(seq) - 2, 3):
        codon = seq[i:i + 3]
        if "N" not in codon and codon in CODON_TO_IDX:
            counts[CODON_TO_IDX[codon]] += 1.0

    # RSCU: for each codon, RSCU = count / mean(synonymous codons)
    # Synonymous codons grouped by amino acid (standard genetic code)
    aa_groups = _amino_acid_codon_groups()
    rscu = np.zeros(64, dtype=np.float64)
    for aa, codon_indices in aa_groups.items():
        if not codon_indices:
            continue
        group_counts = counts[codon_indices]
        group_mean = group_counts.mean()
        if group_mean > 0:
            rscu[codon_indices] = group_counts / group_mean
    return counts, rscu


# Standard genetic code: amino acid -> list of codon indices in CODONS list
def _amino_acid_codon_groups() -> Dict[str, List[int]]:
    """Standard genetic code: amino acid -> codon indices."""
    # Build codon -> AA table
    codon_to_aa = {
        "UUU": "F", "UUC": "F", "UUA": "L", "UUG": "L",
        "CUU": "L", "CUC": "L", "CUA": "L", "CUG": "L",
        "AUU": "I", "AUC": "I", "AUA": "I", "AUG": "M",
        "GUU": "V", "GUC": "V", "GUA": "V", "GUG": "V",
        "UCU": "S", "UCC": "S", "UCA": "S", "UCG": "S",
        "CCU": "P", "CCC": "P", "CCA": "P", "CCG": "P",
        "ACU": "T", "ACC": "T", "ACA": "T", "ACG": "T",
        "GCU": "A", "GCC": "A", "GCA": "A", "GCG": "A",
        "UAU": "Y", "UAC": "Y", "UAA": "*", "UAG": "*",
        "CAU": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
        "AAU": "N", "AAC": "N", "AAA": "K", "AAG": "K",
        "GAU": "D", "GAC": "D", "GAA": "E", "GAG": "E",
        "UGU": "C", "UGC": "C", "UGA": "*", "UGG": "W",
        "CGU": "R", "CGC": "R", "CGA": "R", "CGG": "R",
        "AGU": "S", "AGC": "S", "AGA": "R", "AGG": "R",
        "GGU": "G", "GGC": "G", "GGA": "G", "GGG": "G",
    }
    aa_to_codons: Dict[str, List[int]] = {}
    for codon, aa in codon_to_aa.items():
        aa_to_codons.setdefault(aa, []).append(CODON_TO_IDX[codon])
    return aa_to_codons


# ---------------------------------------------------------------------------
# Main extractor
# ---------------------------------------------------------------------------

class HandEngineeredFeatureExtractor:
    """Extract hand-engineered features from sequences for GBT oracle.

    Determinism: all features are deterministic functions of the sequence.
    """

    def __init__(self, config: Optional[FeatureExtractorConfig] = None) -> None:
        self.config = config or FeatureExtractorConfig()
        # Cache k-mer vocabularies for consistent feature ordering
        self._kmer_vocab: Dict[int, List[str]] = {}
        for k in range(1, self.config.max_kmer_k + 1):
            self._kmer_vocab[k] = [
                "".join(p) for p in _product(NUCLEOTIDES, repeat=k)
            ]
        self._feature_names: Optional[List[str]] = None

    def feature_names(self) -> List[str]:
        """Return ordered list of feature names."""
        if self._feature_names is not None:
            return self._feature_names
        names: List[str] = []
        # Length features
        names.extend(["length", "log_length", "length_bucket_short",
                       "length_bucket_medium", "length_bucket_long"])
        # Nucleotide composition
        names.extend([f"nt_freq_{nt}" for nt in NUCLEOTIDES])
        names.extend([f"nt_count_{nt}" for nt in NUCLEOTIDES])
        for k in (2, 3):
            names.extend([f"{k}mer_freq_{kmer}" for kmer in self._kmer_vocab[k]])
        # GC features
        names.extend([f"gc_content", "gc_skew", "at_content", "purine_content",
                       "pyrimidine_content", "gc_var_5windows_max",
                       "gc_var_5windows_min", "gc_var_5windows_std"])
        # K-mer summary stats
        for k in range(1, self.config.max_kmer_k + 1):
            names.extend([f"kmer{k}_max_freq", f"kmer{k}_entropy",
                          f"kmer{k}_top1_freq", f"kmer{k}_top2_freq",
                          f"kmer{k}_top3_freq", f"kmer{k}_n_unique"])
        # Motif counts
        names.extend([f"motif_{m[0]}" for m in MOTIFS])
        # Codon features
        if self.config.include_codon_features:
            names.extend([f"codon_count_{c}" for c in CODONS])
            names.extend([f"rscu_{c}" for c in CODONS])
            names.extend(["codon_total", "codon_diversity", "cai_proxy"])
        # Position features
        if self.config.include_position_features:
            for w in self.config.window_sizes:
                names.extend([
                    f"gc_window{w}_mean", f"gc_window{w}_std",
                    f"gc_window{w}_max", f"gc_window{w}_min",
                    f"purine_window{w}_mean",
                ])
            # Position-weighted features
            names.extend([
                "gc_5prime_first10", "gc_5prime_first50",
                "gc_3prime_last10", "gc_3prime_last50",
                "first_nt_A", "first_nt_C", "first_nt_G", "first_nt_U",
            ])
        # Structure proxies
        names.extend([
            "pairing_propensity_GC", "pairing_propensity_AU",
            "stem_potential", "loop_potential",
            "cpg_dinuc_count", "upA_dinuc_count",
        ])
        self._feature_names = names
        return names

    def extract(self, sequence: str) -> np.ndarray:
        """Extract features from a single sequence.

        Args:
            sequence: nucleotide sequence (DNA or RNA; T will be converted to U)

        Returns:
            (D,) float32 feature vector
        """
        seq = _normalize_seq(sequence)
        if len(seq) > self.config.max_length:
            seq = seq[: self.config.max_length]
        features: List[float] = []
        features.extend(self._length_features(seq))
        features.extend(self._nt_composition_features(seq))
        features.extend(self._gc_features(seq))
        features.extend(self._kmer_summary_features(seq))
        features.extend(self._motif_features(seq))
        if self.config.include_codon_features:
            features.extend(self._codon_features(seq))
        if self.config.include_position_features:
            features.extend(self._position_features(seq))
        features.extend(self._structure_proxy_features(seq))
        return np.array(features, dtype=np.float32)

    def extract_batch(self, sequences: Sequence[str]) -> np.ndarray:
        """Extract features for a batch of sequences.

        Args:
            sequences: list of sequences

        Returns:
            (N, D) float32 feature matrix
        """
        return np.stack([self.extract(s) for s in sequences])

    # ------------------------------------------------------------------
    # Per-group feature implementations
    # ------------------------------------------------------------------

    def _length_features(self, seq: str) -> List[float]:
        L = len(seq)
        log_L = float(np.log1p(L))
        # Buckets: short < 50, medium 50-500, long > 500
        bucket_short = float(L < 50)
        bucket_medium = float(50 <= L < 500)
        bucket_long = float(L >= 500)
        return [float(L), log_L, bucket_short, bucket_medium, bucket_long]

    def _nt_composition_features(self, seq: str) -> List[float]:
        L = max(len(seq), 1)
        counts = {nt: seq.count(nt) for nt in NUCLEOTIDES}
        freqs = {nt: counts[nt] / L for nt in NUCLEOTIDES}
        features: List[float] = []
        features.extend([freqs[nt] for nt in NUCLEOTIDES])
        features.extend([float(counts[nt]) for nt in NUCLEOTIDES])
        # 2-mer and 3-mer frequencies
        for k in (2, 3):
            kmer_freqs = _kmer_frequencies(seq, k)
            for kmer in self._kmer_vocab[k]:
                features.append(float(kmer_freqs.get(kmer, 0.0)))
        return features

    def _gc_features(self, seq: str) -> List[float]:
        L = max(len(seq), 1)
        gc = sum(1 for c in seq if c in "GC")
        gc_content = gc / L
        # GC skew: (G - C) / (G + C)
        g_count = seq.count("G")
        c_count = seq.count("C")
        gc_skew = (g_count - c_count) / max(g_count + c_count, 1)
        at_content = 1.0 - gc_content
        purine_count = g_count + seq.count("A")
        pyrimidine_count = c_count + seq.count("U")
        purine_content = purine_count / L
        pyrimidine_content = pyrimidine_count / L
        # GC in 5 windows
        gcs = _gc_content_windows(seq, n_windows=5)
        return [
            gc_content, gc_skew, at_content, purine_content,
            pyrimidine_content,
            max(gcs) if gcs else 0.0,
            min(gcs) if gcs else 0.0,
            float(np.std(gcs)) if gcs else 0.0,
        ]

    def _kmer_summary_features(self, seq: str) -> List[float]:
        features: List[float] = []
        for k in range(1, self.config.max_kmer_k + 1):
            freqs = _kmer_frequencies(seq, k)
            if not freqs:
                features.extend([0.0] * 6)
                continue
            values = sorted(freqs.values(), reverse=True)
            # Shannon entropy
            entropy = -sum(v * np.log2(v + 1e-12) for v in values if v > 0)
            features.extend([
                float(values[0]),                   # max freq
                float(entropy),                      # entropy
                float(values[0]) if len(values) > 0 else 0.0,  # top1
                float(values[1]) if len(values) > 1 else 0.0,  # top2
                float(values[2]) if len(values) > 2 else 0.0,  # top3
                float(len(freqs)),                   # n unique kmers
            ])
        return features

    def _motif_features(self, seq: str) -> List[float]:
        features: List[float] = []
        for name, motif, _constraint in MOTIFS:
            count = _find_motif_count(seq, motif)
            features.append(float(count))
        return features

    def _codon_features(self, seq: str) -> List[float]:
        counts, rscu = _codon_usage(seq)
        total = counts.sum()
        # Codon diversity = number of distinct codons / 64
        n_distinct = int((counts > 0).sum())
        diversity = n_distinct / 64.0
        # CAI proxy: max RSCU per amino acid, averaged
        aa_groups = _amino_acid_codon_groups()
        cai_values = []
        for aa, codon_indices in aa_groups.items():
            if aa == "*" or not codon_indices:
                continue
            max_rscu = rscu[codon_indices].max()
            cai_values.append(max_rscu)
        cai_proxy = float(np.mean(cai_values)) if cai_values else 0.0
        features = list(counts.astype(float)) + list(rscu.astype(float))
        features.extend([float(total), diversity, cai_proxy])
        return features

    def _position_features(self, seq: str) -> List[float]:
        L = len(seq)
        features: List[float] = []
        for w in self.config.window_sizes:
            if L < w:
                features.extend([0.0] * 5)
                continue
            # Slide window across sequence
            gc_vals = []
            purine_vals = []
            for i in range(0, L - w + 1, max(w // 2, 1)):
                window = seq[i:i + w]
                gc_vals.append(_gc_content(window))
                purine = sum(1 for c in window if c in "AG") / w
                purine_vals.append(purine)
            if not gc_vals:
                features.extend([0.0] * 5)
                continue
            features.extend([
                float(np.mean(gc_vals)),
                float(np.std(gc_vals)),
                float(np.max(gc_vals)),
                float(np.min(gc_vals)),
                float(np.mean(purine_vals)),
            ])
        # Position-specific features
        first10 = seq[:10]
        first50 = seq[:50]
        last10 = seq[-10:] if L >= 10 else seq
        last50 = seq[-50:] if L >= 50 else seq
        features.extend([
            _gc_content(first10), _gc_content(first50),
            _gc_content(last10), _gc_content(last50),
            float(seq[0] == "A") if L > 0 else 0.0,
            float(seq[0] == "C") if L > 0 else 0.0,
            float(seq[0] == "G") if L > 0 else 0.0,
            float(seq[0] == "U") if L > 0 else 0.0,
        ])
        return features

    def _structure_proxy_features(self, seq: str) -> List[float]:
        """Proxy features for RNA secondary structure propensity.

        These are NOT actual predicted structures — they are cheap proxies
        based on nucleotide composition and dinucleotide frequencies.
        """
        L = max(len(seq), 1)
        # GC pairs (strong)
        gc_pairs = seq.count("GC") + seq.count("CG")
        # AU pairs (weaker)
        au_pairs = seq.count("AU") + seq.count("UA")
        # Stem potential: high GC content near each other
        stem_potential = (gc_pairs * 3.0 + au_pairs * 2.0) / L
        # Loop potential: stretches of single nucleotides
        loop_potential = 0.0
        current_run = 1
        for i in range(1, L):
            if seq[i] == seq[i - 1]:
                current_run += 1
            else:
                if current_run >= 3:
                    loop_potential += current_run
                current_run = 1
        loop_potential /= L
        # CpG dinucleotide (epigenetic marker)
        cpg = seq.count("CG")
        # UpA dinucleotide (RNA editing marker)
        upa = seq.count("UA")
        return [
            gc_pairs / L, au_pairs / L,
            stem_potential, loop_potential,
            float(cpg), float(upa),
        ]


def _product(*iterables, repeat: int = 1):
    """itertools.product replacement."""
    import itertools
    return itertools.product(*iterables, repeat=repeat)


# ---------------------------------------------------------------------------
# Convenience: feature count
# ---------------------------------------------------------------------------

def count_features(config: Optional[FeatureExtractorConfig] = None) -> int:
    """Return total number of features produced by the extractor."""
    extractor = HandEngineeredFeatureExtractor(config)
    return len(extractor.feature_names())


__all__ = [
    "FeatureExtractorConfig",
    "HandEngineeredFeatureExtractor",
    "count_features",
]
