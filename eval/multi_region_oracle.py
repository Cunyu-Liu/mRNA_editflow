"""P2-01: Multi-region oracle with cross-region coupling.

This oracle replaces ``LocalTranslationOracle`` for the P2-01 cross-region
synergy go/no-go gate. It combines:

1. 5'UTR MRL from the P1-04 CNN-50mer cross-fitted ensemble (15 checkpoints:
   5 folds x 3 seeds). Uses the first 50 nt of the 5'UTR (matches training).
2. CDS CAI (codon adaptation index, Sharp & Li 1987) computed deterministically
   from the human codon usage table.
3. 3'UTR stability proxy: AU-rich element (ARE) count + GC content + length.
4. Cross-region coupling terms (NON-ADDITIVE) that enable synergy detection:
   - 5'UTR x CDS: multiplicative MRL*CAI (ribosome recruitment x elongation)
   - CDS x 3'UTR: multiplicative CAI*stability (translation x degradation)
   - 5'UTR x 3'UTR: multiplicative MRL*stability (long-range coupling)
   - Triple coupling: MRL*CAI*stability
   - Structural compatibility: GC-variance across regions (global structure proxy)

The non-additive cross-region terms are ESSENTIAL for synergy detection.
If the oracle were purely additive (score = f(5utr) + g(cds) + h(3utr)),
then syn_sum = delta_joint - (delta_5 + delta_c + delta_3) = 0 by construction.
The multiplicative and global-structure terms break additivity so that
genuine cross-region synergy can be detected.

All ``improves TE/stability/expression`` claims based on this oracle are
``predicted/internal proxy`` until P2-01 multi-region oracle validation
completes (per project constraint).
"""
from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Canonical AU-rich element motifs (conserved ARE core).
_ARE_MOTIFS: Tuple[str, ...] = ("AUUUA", "AUUUUA", "UUUAUUUAUU")

# Default oracle weights (frozen for P2-01; recorded in finding doc).
# Weights sum to 1.0. Cross-region terms total 0.40 (sufficient for synergy).
DEFAULT_WEIGHTS: Dict[str, float] = {
    "mrl_5utr": 0.25,        # main: 5'UTR MRL
    "cai_cds": 0.20,         # main: CDS CAI
    "stab_3utr": 0.15,       # main: 3'UTR stability
    "struct_compat": 0.10,   # global structure (non-additive via GC variance)
    "coupling_5c": 0.10,     # 5'UTR x CDS
    "coupling_c3": 0.08,     # CDS x 3'UTR
    "coupling_53": 0.07,     # 5'UTR x 3'UTR
    "coupling_5c3": 0.05,    # triple
}

# Nearest-neighbor dinucleotide free energy parameters (kcal/mol per step).
# Source: Turner 2004 RNA parameters (simplified set for RNA, 37C).
# Used for the local MFE proxy. Negative = more stable.
# This is a SIMPLIFIED model; for production use ViennaRNA.
_NN_ENERGY: Dict[str, float] = {
    "AA": -1.0, "AC": -2.4, "AG": -2.1, "AU": -1.2,
    "CA": -2.1, "CC": -3.3, "CG": -2.4, "CU": -2.1,
    "GA": -2.2, "GC": -3.6, "GG": -3.3, "GU": -2.5,
    "UA": -1.3, "UC": -2.4, "UG": -2.1, "UU": -1.0,
}


# ---------------------------------------------------------------------------
# Codon usage table + CAI
# ---------------------------------------------------------------------------

def _load_codon_usage_table(path: str) -> Dict[str, float]:
    """Load codon usage table from CSV (codon, aa, frequency).

    Returns dict mapping uppercase RNA codon -> relative frequency.
    """
    table: Dict[str, float] = {}
    with open(path, "r") as f:
        reader = csv.DictReader(f) if _has_header(f) else csv.reader(f)
        f.seek(0)
        for row in csv.reader(f):
            if not row or row[0].startswith("#"):
                continue
            if len(row) < 3:
                continue
            codon = row[0].strip().upper().replace("T", "U")
            if len(codon) != 3:
                continue
            try:
                freq = float(row[2])
            except (ValueError, IndexError):
                continue
            table[codon] = freq
    return table


def _has_header(f) -> bool:
    f.seek(0)
    first = f.readline()
    f.seek(0)
    return first.lower().startswith("codon") or first.startswith("#,,")


def _build_cai_reference(table: Dict[str, float]) -> Dict[str, float]:
    """Build CAI reference: for each amino acid, find max-frequency codon.

    Returns dict mapping codon -> relative adaptiveness w_i = f_i / f_max.
    Stop codons are excluded from CAI computation.
    """
    # Group codons by amino acid (using standard genetic code).
    aa_table = _STANDARD_GENETIC_CODE
    by_aa: Dict[str, List[Tuple[str, float]]] = {}
    for codon, freq in table.items():
        aa = aa_table.get(codon, "?")
        if aa == "*":  # skip stop codons
            continue
        by_aa.setdefault(aa, []).append((codon, freq))

    reference: Dict[str, float] = {}
    for aa, codons in by_aa.items():
        max_freq = max(f for _, f in codons) if codons else 1.0
        if max_freq <= 0:
            continue
        for codon, freq in codons:
            reference[codon] = freq / max_freq
    return reference


# Standard genetic code (RNA). Stop codons -> "*".
_STANDARD_GENETIC_CODE: Dict[str, str] = {}
def _init_genetic_code() -> None:
    bases = "UCAG"
    aas = (
        # U
        "FFLL"  # UU_
        "SSSS"  # UC_
        "YY**"  # UA_
        "CCWW"  # UG_  (W for UGA in standard code? No: UGA=*)
        # Actually let me be precise:
    )
    # Build the standard code explicitly.
    code = {
        # U
        "UUU": "F", "UUC": "F", "UUA": "L", "UUG": "L",
        "UCU": "S", "UCC": "S", "UCA": "S", "UCG": "S",
        "UAU": "Y", "UAC": "Y", "UAA": "*", "UAG": "*",
        "UGU": "C", "UGC": "C", "UGA": "*", "UGG": "W",
        # C
        "CUU": "L", "CUC": "L", "CUA": "L", "CUG": "L",
        "CCU": "P", "CCC": "P", "CCA": "P", "CCG": "P",
        "CAU": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
        "CGU": "R", "CGC": "R", "CGA": "R", "CGG": "R",
        # A
        "AUU": "I", "AUC": "I", "AUA": "I", "AUG": "M",
        "ACU": "T", "ACC": "T", "ACA": "T", "ACG": "T",
        "AAU": "N", "AAC": "N", "AAA": "K", "AAG": "K",
        "AGU": "S", "AGC": "S", "AGA": "R", "AGG": "R",
        # G
        "GUU": "V", "GUC": "V", "GUA": "V", "GUG": "V",
        "GCU": "A", "GCC": "A", "GCA": "A", "GCG": "A",
        "GAU": "D", "GAC": "D", "GAA": "E", "GAG": "E",
        "GGU": "G", "GGC": "G", "GGA": "G", "GGG": "G",
    }
    _STANDARD_GENETIC_CODE.update(code)


_init_genetic_code()


def compute_cai(cds: str, reference: Dict[str, float]) -> float:
    """Compute Codon Adaptation Index (Sharp & Li 1987).

    CAI = exp( (1/L) * sum( ln(w_i) ) )

    where w_i = relative adaptiveness of codon i, L = number of valid codons.
    Returns 0.0 if no valid codons. Result is in [0, 1].
    """
    if not cds or len(cds) < 3:
        return 0.0
    seq = cds.upper().replace("T", "U")
    # Ensure length is multiple of 3 (trim trailing).
    trim = len(seq) - (len(seq) % 3)
    seq = seq[:trim]
    log_sum = 0.0
    n_valid = 0
    for i in range(0, len(seq), 3):
        codon = seq[i : i + 3]
        if codon not in reference:
            continue
        w = reference[codon]
        if w <= 0:
            continue
        log_sum += math.log(w)
        n_valid += 1
    if n_valid == 0:
        return 0.0
    cai = math.exp(log_sum / n_valid)
    # Clamp to [0, 1] (numerical safety).
    return float(max(0.0, min(1.0, cai)))


# ---------------------------------------------------------------------------
# Sequence utilities
# ---------------------------------------------------------------------------

def _gc_content(seq: str) -> float:
    """GC content fraction in [0, 1]. Empty seq -> 0.0."""
    if not seq:
        return 0.0
    s = seq.upper().replace("T", "U")
    gc = sum(1 for c in s if c in ("G", "C"))
    return gc / len(s)


def _count_overlapping(seq: str, motif: str) -> int:
    """Count overlapping occurrences of motif in seq."""
    if not motif or len(motif) > len(seq):
        return 0
    count = 0
    start = 0
    while True:
        idx = seq.find(motif, start)
        if idx == -1:
            break
        count += 1
        start = idx + 1
    return count


def _are_count(three_utr: str) -> int:
    """Count AU-rich elements (AREs) in 3'UTR."""
    s = three_utr.upper().replace("T", "U")
    total = 0
    for motif in _ARE_MOTIFS:
        total += _count_overlapping(s, motif)
    return total


def _local_mfe_proxy(seq: str) -> float:
    """Local MFE proxy via nearest-neighbor dinucleotide energies.

    Returns total energy (kcal/mol). More negative = more stable.
    This is a SIMPLIFIED model (no secondary structure); used only as a
    deterministic sequence-dependent feature. For production MFE, use ViennaRNA.
    """
    if len(seq) < 2:
        return 0.0
    s = seq.upper().replace("T", "U")
    total = 0.0
    for i in range(len(s) - 1):
        di = s[i : i + 2]
        total += _NN_ENERGY.get(di, 0.0)
    return total


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


# ---------------------------------------------------------------------------
# 3'UTR stability proxy
# ---------------------------------------------------------------------------

def compute_stability_proxy(three_utr: str) -> float:
    """3'UTR stability proxy in [0, 1].

    Combines:
    - ARE count (fewer = more stable)
    - GC content (higher = more stable, resistant to nucleases)
    - Length penalty (very long 3'UTRs have more degradation targets)

    All claims are ``predicted/internal proxy`` per project constraint.
    """
    if not three_utr:
        return 0.5
    are = _are_count(three_utr)
    gc = _gc_content(three_utr)
    length = len(three_utr)

    # ARE penalty: each ARE reduces stability.
    are_penalty = _sigmoid(are * 0.5)  # [0, 1], higher ARE -> lower stability

    # GC bonus: higher GC -> more stable.
    gc_bonus = gc  # [0, 1]

    # Length factor: optimal around 200-400 nt, penalize extremes.
    if length < 100:
        length_factor = length / 100.0
    elif length > 800:
        length_factor = max(0.3, 1.0 - (length - 800) / 2000.0)
    else:
        length_factor = 1.0

    # Weighted combination.
    stability = (
        0.40 * (1.0 - are_penalty)   # ARE (stability decreases with ARE)
        + 0.35 * gc_bonus            # GC content
        + 0.25 * length_factor       # Length
    )
    return float(max(0.0, min(1.0, stability)))


# ---------------------------------------------------------------------------
# CNN-50mer ensemble wrapper
# ---------------------------------------------------------------------------

class CNN50merEnsemble:
    """Wrapper for the P1-04 CNN-50mer cross-fitted ensemble.

    Loads all 15 checkpoints (5 folds x 3 seeds) and averages predictions.
    Uses the first 50 nt of the 5'UTR (matches training distribution).
    """

    def __init__(
        self,
        ckpt_dir: str,
        device: str = "cpu",
        arch_name: str = "cnn_50mer",
        dataset_name: str = "sample2019_mpra",
    ) -> None:
        self.ckpt_dir = Path(ckpt_dir)
        self.device = device
        self.arch_name = arch_name
        self.dataset_name = dataset_name
        self._models: List[Any] = []
        self._loaded = False

    def load(self) -> int:
        """Load all matching checkpoints. Returns number loaded."""
        import sys
        # Ensure mrna_editflow is importable.
        # The CNN50merPredictor is in models/predictors/cnn_50mer.py.
        # We import it lazily to avoid torch dependency at module load.
        try:
            from mrna_editflow.models.predictors.cnn_50mer import CNN50merPredictor
        except Exception:
            # Fallback: add repo root to path.
            repo_root = str(self.ckpt_dir.parents[2])
            if repo_root not in sys.path:
                sys.path.insert(0, repo_root)
            from mrna_editflow.models.predictors.cnn_50mer import CNN50merPredictor

        pattern = f"{self.arch_name}__{self.dataset_name}__fold*_seed*.pt"
        ckpt_paths = sorted(self.ckpt_dir.glob(pattern))
        self._models = []
        for p in ckpt_paths:
            # ckpt path without .pt extension (PredictorBase.load adds .pt)
            stem = str(p)[:-3] if str(p).endswith(".pt") else str(p)
            try:
                predictor = CNN50merPredictor.load(
                    Path(stem), device=self.device
                )
                self._models.append(predictor)
            except Exception as e:
                # Skip broken checkpoints.
                continue
        self._loaded = True
        return len(self._models)

    def predict_mrl(self, five_utr: str) -> Tuple[float, float]:
        """Predict MRL for a 5'UTR sequence.

        Uses the first 50 nt (matches CNN-50mer training).
        Returns (mean, std) across the ensemble.
        """
        if not self._loaded:
            n = self.load()
            if n == 0:
                # Fallback: return a neutral MRL.
                return (5.0, 1.0)
        seq = five_utr.upper().replace("T", "U")[:50]
        if not seq:
            return (5.0, 1.0)
        preds: List[float] = []
        for model in self._models:
            try:
                arr = model.predict([seq])
                preds.append(float(arr[0]))
            except Exception:
                continue
        if not preds:
            return (5.0, 1.0)
        mean = float(np.mean(preds))
        std = float(np.std(preds))
        return (mean, std)


# ---------------------------------------------------------------------------
# MultiRegionOracle
# ---------------------------------------------------------------------------

@dataclass
class MultiRegionOracleConfig:
    """Configuration for MultiRegionOracle."""
    cnn_ckpt_dir: str = ""
    codon_usage_path: str = ""
    weights: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    device: str = "cpu"
    # If True, skip CNN ensemble (use neutral MRL=5.0). For fast testing.
    skip_cnn: bool = False
    # Normalization ranges (for clipping).
    mrl_min: float = 0.0
    mrl_max: float = 10.0


class MultiRegionOracle:
    """Composite oracle with cross-region coupling for P2-01.

    This oracle produces a scalar ``ensemble_te`` score (compatible with
    the existing OracleMDP interface) that responds to edits in all three
    regions (5'UTR, CDS, 3'UTR) AND has non-additive cross-region coupling
    terms, enabling cross-region synergy detection.

    All ``improves TE/stability/expression`` claims are ``predicted/internal
    proxy`` until P2-01 validation completes.
    """

    def __init__(self, config: MultiRegionOracleConfig) -> None:
        self.config = config
        self._codon_table: Dict[str, float] = {}
        self._cai_reference: Dict[str, float] = {}
        self._cnn_ensemble: Optional[CNN50merEnsemble] = None
        self._loaded = False

    def load(self) -> None:
        """Load codon usage table and CNN ensemble."""
        # Codon usage table.
        if self.config.codon_usage_path and os.path.exists(
            self.config.codon_usage_path
        ):
            self._codon_table = _load_codon_usage_table(
                self.config.codon_usage_path
            )
            self._cai_reference = _build_cai_reference(self._codon_table)
        # CNN ensemble.
        if not self.config.skip_cnn and self.config.cnn_ckpt_dir:
            self._cnn_ensemble = CNN50merEnsemble(
                ckpt_dir=self.config.cnn_ckpt_dir,
                device=self.config.device,
            )
            n = self._cnn_ensemble.load()
            if n == 0:
                print(
                    f"[MultiRegionOracle] Warning: 0 CNN checkpoints loaded "
                    f"from {self.config.cnn_ckpt_dir}. Using fallback MRL=5.0."
                )
        self._loaded = True

    # ------------------------------------------------------------------
    # Per-region scoring
    # ------------------------------------------------------------------

    def _score_mrl_5utr(self, five_utr: str) -> float:
        """5'UTR MRL from CNN-50mer ensemble (predicted/internal proxy)."""
        if self._cnn_ensemble is None:
            return 5.0  # neutral
        mean, _ = self._cnn_ensemble.predict_mrl(five_utr)
        return mean

    def _score_cai_cds(self, cds: str) -> float:
        """CDS CAI (deterministic, [0, 1])."""
        if not self._cai_reference:
            return 0.5  # neutral
        return compute_cai(cds, self._cai_reference)

    def _score_stab_3utr(self, three_utr: str) -> float:
        """3'UTR stability proxy ([0, 1], predicted/internal proxy)."""
        return compute_stability_proxy(three_utr)

    def _score_struct_compat(
        self, five_utr: str, cds: str, three_utr: str
    ) -> float:
        """Structural compatibility proxy via GC variance across regions.

        Low GC variance = regions are structurally compatible (predicted/internal
        proxy). Returns [0, 1], higher = more compatible.
        """
        gc_5 = _gc_content(five_utr)
        gc_c = _gc_content(cds)
        gc_3 = _gc_content(three_utr)
        gcs = [gc_5, gc_c, gc_3]
        mean_gc = sum(gcs) / 3.0
        variance = sum((g - mean_gc) ** 2 for g in gcs) / 3.0
        # Normalize: variance in [0, 0.25] (max when one is 1, others 0).
        norm_var = min(variance / 0.25, 1.0)
        return 1.0 - norm_var

    # ------------------------------------------------------------------
    # Full record scoring
    # ------------------------------------------------------------------

    def score_record(self, record: object) -> Dict[str, Any]:
        """Score an MRNARecord. Returns dict with ``ensemble_te`` scalar.

        The ``ensemble_te`` field is the composite oracle score in [0, 1],
        compatible with the existing OracleMDP interface.
        """
        five_utr = _field(record, "five_utr", "")
        cds = _field(record, "cds", "")
        three_utr = _field(record, "three_utr", "")
        tid = _field(record, "transcript_id", None)

        # Per-region scores.
        mrl_raw = self._score_mrl_5utr(five_utr)         # ~[0, 10]
        cai = self._score_cai_cds(cds)                   # [0, 1]
        stab = self._score_stab_3utr(three_utr)          # [0, 1]
        struct = self._score_struct_compat(five_utr, cds, three_utr)  # [0, 1]

        # Normalize MRL to [0, 1].
        mrl_norm = (mrl_raw - self.config.mrl_min) / max(
            self.config.mrl_max - self.config.mrl_min, 1e-8
        )
        mrl_norm = max(0.0, min(1.0, mrl_norm))

        # Cross-region coupling terms (NON-ADDITIVE).
        coupling_5c = mrl_norm * cai          # 5'UTR x CDS
        coupling_c3 = cai * stab              # CDS x 3'UTR
        coupling_53 = mrl_norm * stab         # 5'UTR x 3'UTR
        coupling_5c3 = mrl_norm * cai * stab  # triple

        # Weighted combination.
        w = self.config.weights
        score = (
            w["mrl_5utr"] * mrl_norm
            + w["cai_cds"] * cai
            + w["stab_3utr"] * stab
            + w["struct_compat"] * struct
            + w["coupling_5c"] * coupling_5c
            + w["coupling_c3"] * coupling_c3
            + w["coupling_53"] * coupling_53
            + w["coupling_5c3"] * coupling_5c3
        )
        # Clamp to [0, 1].
        score = max(0.0, min(1.0, score))

        # Additional per-region features for analysis.
        gc_5 = _gc_content(five_utr)
        gc_c = _gc_content(cds)
        gc_3 = _gc_content(three_utr)
        are_count = _are_count(three_utr)

        result: Dict[str, Any] = {
            # Composite score (used by OracleMDP).
            "ensemble_te": score,
            "mrl": mrl_raw,
            "ensemble_mrl": mrl_raw,
            "te": score,
            # Per-region normalized scores.
            "mrl_5utr_norm": mrl_norm,
            "cai_cds": cai,
            "stab_3utr": stab,
            "struct_compat": struct,
            # Cross-region coupling (non-additive).
            "coupling_5c": coupling_5c,
            "coupling_c3": coupling_c3,
            "coupling_53": coupling_53,
            "coupling_5c3": coupling_5c3,
            # Raw features.
            "gc_5utr": gc_5,
            "gc_cds": gc_c,
            "gc_3utr": gc_3,
            "are_count_3utr": are_count,
            "five_utr_len": len(five_utr),
            "cds_len": len(cds),
            "three_utr_len": len(three_utr),
            # Metadata.
            "agreement": 1.0,
            "uncertainty": 0.0,
            "oracle_version": "multi_region_v2_p2_01",
            "weights": dict(w),
        }
        if tid is not None:
            result["transcript_id"] = tid
        return result

    def batch_score(self, records: Sequence[object]) -> List[Dict[str, Any]]:
        """Score a batch of records."""
        return [self.score_record(r) for r in records]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _field(record: object, name: str, default: object = None) -> object:
    """Get a field from a record (MRNARecord or dict)."""
    if isinstance(record, dict):
        return record.get(name, default)
    return getattr(record, name, default)


def build_default_multi_region_oracle(
    repo_root: str,
    device: str = "cpu",
    skip_cnn: bool = False,
) -> MultiRegionOracle:
    """Build a MultiRegionOracle with default paths.

    Args:
        repo_root: Path to the mrna_editflow repo root.
        device: torch device.
        skip_cnn: If True, skip CNN ensemble loading (fast testing).
    """
    ckpt_dir = os.path.join(repo_root, "ckpts", "p1_04_predictors")
    codon_path = os.path.join(
        repo_root, "external_tools", "EnsembleDesign",
        "codon_usage_freq_table_human.csv",
    )
    config = MultiRegionOracleConfig(
        cnn_ckpt_dir=ckpt_dir,
        codon_usage_path=codon_path,
        device=device,
        skip_cnn=skip_cnn,
    )
    oracle = MultiRegionOracle(config)
    oracle.load()
    return oracle
