"""V8 Stage 1 joint external-library pipeline.

Unified record format: (sequence [+cell_context_id] -> standardised activity,
domain_label, per-study leakage flags). Domains registered for Stage 1:

- mrl   (domain id 0): Sample-2019 280K random 5'UTR library
        (external_model_assets/sample280k, 677,608 replicate-merged rows),
        target = mean rl, z-scored on the clean library.
- polya (domain id 1): APARENT APA 3' UTR library (GSE113849 isoform table,
        external_model_assets/aparent_apa_3p5m, 2,740,320 rows with
        total_count_vs_distal >= 10), target = proximal usage log2 odds,
        z-scored on the clean library.
- cms   (domain id 2): STUB. The ENCODE CMS array (MPRAU-domain P0 prior) is
        not yet downloaded (ENCODE portal requires user browser relay). The
        domain slot, loader interface and CSV schema are frozen NOW
        (columns: sequence, activity, optional cell_context) so the data can be
        dropped in without touching any other code; until the file exists the
        domain resolves to "skipped" and never blocks the other domains.

Leakage audit: every library row carries per-protected-study boolean flags,
computed with the SAME 3-block pigeonhole rules as the W0 reference scripts
(rule: exact 17bp block collision -> full position-wise comparison over the
zipped (shorter) length -> flag if <= 2 mismatches):
- vs GSE114002: blocks (s[:17], s[17:34], s[34:51]) -- consecutive-thirds scheme
  of the 280K script (covers a full 50-mer).
- vs GSE269595: blocks (s[:17], s[len//2:len//2+17], s[-17:]) -- first/mid/last
  scheme of the APA script.
Both audits are applied to EVERY library row regardless of domain
(cross-domain protection), mirroring the per-study reference protocols.

Sampling: DomainBalancedSampler draws every batch with equal domain quotas
(batch split evenly among active domains; remainder rotated by batch index so
no domain is systematically favoured), each domain iterating its own shuffled
order with cycling. One epoch := ceil(sum(domain sizes) / batch) batches, i.e.
one pass worth of samples over the joint library in balanced proportions.
"""
from __future__ import annotations

import gzip
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import torch

from core.route2_v8_hybrid_backbone_v1 import DOMAIN_IDS, NUM_DOMAINS

MNT = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2")
MRL_LIB_DIR = MNT / "external_model_assets/sample280k"
POLYA_LIB_GZ = MNT / "external_model_assets/aparent_apa_3p5m/GSE113849_data_isoforms.csv.gz"
CMS_ARRAY_CSV = MNT / "external_model_assets/cms_array/cms_array_activity.csv"
CANONICAL_GSE114002 = MNT / "canonical/GSE114002/v1/canonical_records.private.jsonl"
CANONICAL_GSE269595 = MNT / "canonical/GSE269595/v1/canonical_records.private.jsonl"

POLYA_MIN_TOTAL_COUNT = 10
POLYA_P_CLIP = 1e-4

# Protected-study block schemes (identical to the W0 reference scripts).
STUDY_BLOCK_SCHEMES = {
    "GSE114002": "consecutive_thirds",
    "GSE269595": "first_mid_last",
}


def format_sequence(sequence: str) -> str:
    """mRNABERT input convention: uppercase DNA alphabet, space-joined."""
    return " ".join(str(sequence).upper().replace("U", "T"))


# ---------------------------------------------------------------------------
# Library loaders (raw: sequences + unstandardised activities, no tokenizer).
# ---------------------------------------------------------------------------

def load_mrl_library(lib_dir: Path = MRL_LIB_DIR) -> tuple[list[str], np.ndarray]:
    """Sample-2019 280K library: merge the two egfp_unmod replicates by UTR (mean rl)."""
    merged: dict[str, list[float]] = {}
    for name in ("GSM3130435_egfp_unmod_1.csv.gz", "GSM3130436_egfp_unmod_2.csv.gz"):
        with gzip.open(Path(lib_dir) / name, "rt") as handle:
            header = handle.readline().strip().split(",")
            utr_index = header.index("utr")
            rl_index = header.index("rl")
            for line in handle:
                fields = line.rstrip("\n").split(",")
                merged.setdefault(fields[utr_index], []).append(float(fields[rl_index]))
    sequences = list(merged)
    activities = np.asarray([float(np.mean(values)) for values in merged.values()], dtype=np.float64)
    return sequences, activities


def load_polya_library(lib_gz: Path = POLYA_LIB_GZ, min_total_count: int = POLYA_MIN_TOTAL_COUNT, p_clip: float = POLYA_P_CLIP) -> tuple[list[str], np.ndarray]:
    """APARENT APA 3' UTR library: proximal usage log2 odds, count-filtered.

    p = clip(proximal_count / total_count_vs_distal); target = log2(p/(1-p)),
    matching the GSE269595 endpoint PROXIMAL_POLYA_SITE_USAGE_LOG2_ODDS.
    """
    sequences: list[str] = []
    activities: list[float] = []
    with gzip.open(Path(lib_gz), "rt") as handle:
        header = handle.readline().strip().split(",")
        seq_i = header.index("seq")
        prox_i = header.index("proximal_count")
        tot_i = header.index("total_count_vs_distal")
        for line in handle:
            fields = line.rstrip("\n").split(",")
            total = float(fields[tot_i])
            if total < min_total_count:
                continue
            prox = float(fields[prox_i])
            p = min(max(prox / total, p_clip), 1.0 - p_clip)
            sequences.append(fields[seq_i])
            activities.append(float(np.log2(p / (1.0 - p))))
    return sequences, np.asarray(activities, dtype=np.float64)


def cms_library_available(path: Path = CMS_ARRAY_CSV) -> bool:
    return Path(path).exists()


def load_cms_library(path: Path = CMS_ARRAY_CSV) -> tuple[list[str], np.ndarray, list[int]]:
    """CMS array loader (STUB interface, frozen schema).

    Expected CSV columns: sequence, activity, optional cell_context. The file
    is not yet on disk (ENCODE portal download pending user browser relay);
    until then this raises FileNotFoundError and the cms domain resolves to
    skipped, never blocking the other domains.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            "CMS array data not yet downloaded (ENCODE portal requires user "
            f"browser relay). Expected at {path} with columns "
            "sequence,activity[,cell_context]. See "
            "docs/paper/route2_v8_stage1_prereg_v1.md for the frozen stub contract."
        )
    sequences: list[str] = []
    activities: list[float] = []
    contexts: list[int] = []
    with path.open() as handle:
        header = handle.readline().strip().split(",")
        seq_i = header.index("sequence")
        act_i = header.index("activity")
        ctx_i = header.index("cell_context") if "cell_context" in header else None
        for line in handle:
            fields = line.rstrip("\n").split(",")
            sequences.append(fields[seq_i])
            activities.append(float(fields[act_i]))
            contexts.append(int(fields[ctx_i]) if ctx_i is not None else 0)
    return sequences, np.asarray(activities, dtype=np.float64), contexts


def resolve_libraries(requested: list[str]) -> tuple[list[str], list[dict]]:
    """Split requested library names into (active, skipped-with-reason).

    cms is skipped (not an error) while its CSV is absent, so the joint run
    proceeds with the remaining domains. Unknown names are hard errors.
    """
    active: list[str] = []
    skipped: list[dict] = []
    for name in requested:
        if name not in DOMAIN_IDS:
            raise ValueError(f"unknown library domain {name!r}; expected one of {sorted(DOMAIN_IDS)}")
        if name == "cms" and not cms_library_available():
            skipped.append({
                "domain": name,
                "reason": "cms_array CSV not yet downloaded (ENCODE portal pending); stub slot stays reserved (domain id 2)",
            })
            continue
        active.append(name)
    return active, skipped


# ---------------------------------------------------------------------------
# Leakage audit (3-block pigeonhole, per-study reference schemes).
# ---------------------------------------------------------------------------

def _blocks(sequence: str, scheme: str) -> tuple[str, ...]:
    if scheme == "consecutive_thirds":
        return (sequence[:17], sequence[17:34], sequence[34:51])
    if scheme == "first_mid_last":
        mid = len(sequence) // 2
        return (sequence[:17], sequence[mid : mid + 17], sequence[-17:])
    raise ValueError(f"unknown block scheme {scheme!r}")


def build_protected_index(canonical_paths: Optional[dict[str, Path]] = None) -> dict[str, dict[str, set[str]]]:
    """Load protected benchmark sequences (ALL splits, source+candidate) per study."""
    if canonical_paths is None:
        canonical_paths = {"GSE114002": CANONICAL_GSE114002, "GSE269595": CANONICAL_GSE269595}
    import json

    index: dict[str, dict[str, set[str]]] = {}
    for study, path in canonical_paths.items():
        scheme = STUDY_BLOCK_SCHEMES[study]
        block_index: dict[str, set[str]] = {}
        protected: set[str] = set()
        with Path(path).open() as handle:
            for line in handle:
                row = json.loads(line)
                for key in ("source_sequence", "candidate_sequence"):
                    sequence = row[key]
                    protected.add(sequence)
                    for block in _blocks(sequence, scheme):
                        block_index.setdefault(block, set()).add(sequence)
        index[study] = block_index
    return index


def audit_leak_flags(sequences: list[str], protected_index: dict[str, dict[str, set[str]]]) -> dict[str, np.ndarray]:
    """Per-study boolean flag array (one entry per library row)."""
    flags: dict[str, np.ndarray] = {}
    for study, block_index in protected_index.items():
        scheme = STUDY_BLOCK_SCHEMES[study]
        flagged = np.zeros(len(sequences), dtype=bool)
        for i, sequence in enumerate(sequences):
            for block in _blocks(sequence, scheme):
                for candidate in block_index.get(block, ()):
                    if sum(a != b for a, b in zip(sequence, candidate)) <= 2:
                        flagged[i] = True
                        break
                if flagged[i]:
                    break
        flags[study] = flagged
    return flags


# ---------------------------------------------------------------------------
# Standardisation + tokenisation -> training-ready DomainLibrary.
# ---------------------------------------------------------------------------

def standardize(activities: np.ndarray) -> tuple[np.ndarray, float, float]:
    mean = float(activities.mean())
    std = float(activities.std())
    if std <= 0.0:
        raise ValueError("cannot standardise a constant activity vector")
    return ((activities - mean) / std).astype(np.float32), mean, std


@dataclass
class DomainLibrary:
    """Tokenised, standardised, audit-flagged single-domain library.

    Tensors cover the CLEAN rows only (all leak flags False); the full raw
    sequence list and per-study flag arrays are retained for provenance.
    """

    domain: str
    domain_id: int
    sequences: list[str]
    activities_raw: np.ndarray
    leak_flags: dict[str, np.ndarray]
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    targets: torch.Tensor
    target_mean: float
    target_std: float

    @property
    def n_raw(self) -> int:
        return len(self.sequences)

    @property
    def n_clean(self) -> int:
        return int(self.input_ids.shape[0])

    @property
    def flagged_mask(self) -> np.ndarray:
        if not self.leak_flags:
            return np.zeros(self.n_raw, dtype=bool)
        return np.logical_or.reduce(list(self.leak_flags.values()))

    def audit_summary(self) -> dict:
        return {
            "domain": self.domain,
            "n_raw": self.n_raw,
            "n_clean": self.n_clean,
            "n_flagged": int(self.flagged_mask.sum()),
            "per_study_flagged": {study: int(flags.sum()) for study, flags in self.leak_flags.items()},
        }


def prepare_domain_library(
    domain: str,
    sequences: list[str],
    activities: np.ndarray,
    leak_flags: dict[str, np.ndarray],
    tokenizer,
    max_length: int = 512,
) -> DomainLibrary:
    """Tokenise and standardise the clean subset of one domain."""
    clean = ~np.logical_or.reduce(list(leak_flags.values())) if leak_flags else np.ones(len(sequences), dtype=bool)
    clean_sequences = [sequences[i] for i in np.nonzero(clean)[0]]
    z, mean, std = standardize(activities[np.asarray(clean, dtype=bool)])
    encoded = tokenizer(
        [format_sequence(s) for s in clean_sequences],
        add_special_tokens=True,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return DomainLibrary(
        domain=domain,
        domain_id=DOMAIN_IDS[domain],
        sequences=sequences,
        activities_raw=activities,
        leak_flags=leak_flags,
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        targets=torch.tensor(z, dtype=torch.float32),
        target_mean=mean,
        target_std=std,
    )


# ---------------------------------------------------------------------------
# Domain-balanced batch sampling.
# ---------------------------------------------------------------------------

class DomainBalancedSampler:
    """Equal domain quotas per batch; per-domain shuffled cycling order.

    steps_per_epoch = ceil(sum(sizes) / batch). Quotas: base = batch // D per
    domain, the batch % D remainder is distributed round-robin (rotated by
    batch index) so every domain receives an equal share over an epoch.
    """

    def __init__(self, domain_sizes: dict[str, int], batch_size: int, seed: int, domain_order: Optional[list[str]] = None) -> None:
        if not domain_sizes:
            raise ValueError("domain_sizes must not be empty")
        if batch_size < len(domain_sizes):
            raise ValueError(f"batch_size {batch_size} smaller than domain count {len(domain_sizes)}")
        self.domain_sizes = dict(domain_sizes)
        self.batch_size = batch_size
        self.seed = seed
        self.domain_order = domain_order or sorted(domain_sizes, key=lambda d: DOMAIN_IDS.get(d, 0))
        self.steps_per_epoch = math.ceil(sum(domain_sizes.values()) / batch_size)

    def _quotas(self, batch_index: int) -> dict[str, int]:
        n_domains = len(self.domain_order)
        base = self.batch_size // n_domains
        remainder = self.batch_size % n_domains
        quotas = {d: base for d in self.domain_order}
        for offset in range(remainder):
            quotas[self.domain_order[(batch_index + offset) % n_domains]] += 1
        return quotas

    def _domain_epoch_order(self, domain: str, needed: int, rng: np.random.Generator) -> np.ndarray:
        size = self.domain_sizes[domain]
        if size == 0:
            return np.zeros(0, dtype=np.int64)
        n_cycles = math.ceil(needed / size)
        order = np.concatenate([rng.permutation(size) for _ in range(n_cycles)])
        return order[:needed].astype(np.int64)

    def domain_draws_per_epoch(self) -> dict[str, int]:
        """Total per-domain row draws over one epoch (for budget accounting)."""
        all_quotas = [self._quotas(b) for b in range(self.steps_per_epoch)]
        return {d: int(sum(q[d] for q in all_quotas)) for d in self.domain_order}

    def epoch_batches(self, epoch: int) -> Iterator[dict[str, np.ndarray]]:
        """Yield per-epoch batches as {domain: row indices into that domain}."""
        rng = np.random.default_rng([self.seed, epoch])
        needed = self.domain_draws_per_epoch()
        orders = {d: self._domain_epoch_order(d, needed[d], rng) for d in self.domain_order}
        cursors = {d: 0 for d in self.domain_order}
        for b in range(self.steps_per_epoch):
            quotas = self._quotas(b)
            batch: dict[str, np.ndarray] = {}
            for domain in self.domain_order:
                quota = quotas[domain]
                if quota:
                    batch[domain] = orders[domain][cursors[domain] : cursors[domain] + quota]
                    cursors[domain] += quota
            yield batch
