"""Torch dataset, length-bucketing sampler and region-aware collate for MEF.

Bridges the cleaned corpus + precomputed features to the training loop. Each
item exposes the model's full conditioning signal: nucleotide ``token_ids``, the
region partition (``region_ids``), the codon ``phase_ids`` and the scalar
thermodynamic features (``mfe``, ``start_accessibility``) plus the per-nt
``pairing_prob`` vector.

Padding sentinels (exported so model embedding tables can be sized correctly):

* ``token_ids`` pad with :data:`PAD_TOKEN` (embedding size ``VOCAB_MODEL_SIZE``).
* ``region_ids`` pad with :data:`REGION_PAD` = ``NUM_REGIONS`` (embed size
  ``NUM_REGIONS + 1``).
* ``phase_ids`` pad with :data:`PHASE_PAD` = ``PHASE_NONE`` (the existing non-CDS
  sentinel; embed size ``NUM_PHASES + 1``). Real vs padded positions are always
  disambiguated by ``padding_mask``.

``padding_mask`` follows the ``nn.Transformer`` ``key_padding_mask`` convention:
**True marks a padded position to ignore**, False marks a real token.

Complexity: dataset access is O(len(seq)); collate is O(batch * max_len).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from mrna_editflow.core.config import DataConfig
from mrna_editflow.core.constants import (
    NUM_PHASES,
    NUM_REGIONS,
    PAD_TOKEN,
    PHASE_NONE,
)
from mrna_editflow.core.schema import MRNARecord, PrecomputedFeatures

# Padding sentinels (see module docstring).
REGION_PAD: int = NUM_REGIONS   # 3 (extra index beyond 5UTR/CDS/3UTR)
PHASE_PAD: int = PHASE_NONE     # 3 (reuse the non-CDS sentinel)


class MRNADataset(Dataset):
    """Region-annotated mRNA dataset yielding per-record tensors + features.

    Parameters
    ----------
    records: cleaned :class:`MRNARecord` list.
    features: optional ``{transcript_id: PrecomputedFeatures}`` mapping. When a
        record has no entry, ``mfe`` / ``start_accessibility`` default to ``0.0``
        and ``pairing_prob`` to a zero vector of the sequence length.
    indices: optional subset (e.g. a split's ``.idx`` list); defaults to all.

    Each ``__getitem__`` returns a dict of CPU tensors::

        {
          "transcript_id": str,
          "token_ids":  LongTensor (L,),
          "region_ids": LongTensor (L,),
          "phase_ids":  LongTensor (L,),
          "pairing_prob": FloatTensor (L,),
          "mfe": FloatTensor (),          # scalar
          "start_accessibility": FloatTensor (),
          "length": int,
        }

    Complexity: O(L) per item.
    """

    def __init__(
        self,
        records: Sequence[MRNARecord],
        features: Optional[Dict[str, PrecomputedFeatures]] = None,
        indices: Optional[Sequence[int]] = None,
    ) -> None:
        self._records: List[MRNARecord] = list(records)
        self._features = features or {}
        if indices is None:
            self._indices = list(range(len(self._records)))
        else:
            self._indices = list(indices)

    def __len__(self) -> int:
        return len(self._indices)

    def record_at(self, i: int) -> MRNARecord:
        """Return the underlying record for local index ``i``."""
        return self._records[self._indices[i]]

    def seq_length(self, i: int) -> int:
        """Length of the transcript at local index ``i`` (for the sampler)."""
        return len(self.record_at(i).seq)

    def __getitem__(self, i: int) -> Dict[str, object]:
        rec = self.record_at(i)
        token_ids = torch.tensor(rec.token_ids(), dtype=torch.long)
        region_ids = torch.tensor(rec.region_ids(), dtype=torch.long)
        phase_ids = torch.tensor(rec.codon_phases(), dtype=torch.long)
        L = token_ids.shape[0]

        feat = self._features.get(rec.transcript_id)
        if feat is not None:
            mfe = float(feat.mfe)
            access = float(feat.start_accessibility)
            pp = feat.pairing_prob if feat.pairing_prob is not None else [0.0] * L
            pairing = torch.tensor(pp, dtype=torch.float32)
            if pairing.shape[0] != L:  # defensive re-alignment
                pairing = torch.zeros(L, dtype=torch.float32)
        else:
            mfe = 0.0
            access = 0.0
            pairing = torch.zeros(L, dtype=torch.float32)

        return {
            "transcript_id": rec.transcript_id,
            "token_ids": token_ids,
            "region_ids": region_ids,
            "phase_ids": phase_ids,
            "pairing_prob": pairing,
            "mfe": torch.tensor(mfe, dtype=torch.float32),
            "start_accessibility": torch.tensor(access, dtype=torch.float32),
            "length": L,
        }


def collate_fn(batch: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """Pad a list of dataset items into a region-aware batch of tensors.

    Pads ``token_ids`` with :data:`PAD_TOKEN`, ``region_ids`` with
    :data:`REGION_PAD`, ``phase_ids`` with :data:`PHASE_PAD`, and
    ``pairing_prob`` with ``0.0`` to the batch's max length. Returns::

        {
          "transcript_ids": List[str],
          "token_ids":   LongTensor (B, Lmax),
          "region_ids":  LongTensor (B, Lmax),
          "phase_ids":   LongTensor (B, Lmax),
          "pairing_prob": FloatTensor (B, Lmax),
          "padding_mask": BoolTensor (B, Lmax),   # True == pad (ignore)
          "lengths": LongTensor (B,),
          "mfe": FloatTensor (B,),
          "start_accessibility": FloatTensor (B,),
        }

    All per-position tensors share identical (B, Lmax) shape so region/phase are
    aligned to tokens position-for-position. Complexity: O(B * Lmax).
    """
    b = len(batch)
    lengths = [int(item["length"]) for item in batch]  # type: ignore[index]
    max_len = max(lengths) if lengths else 0

    token_ids = torch.full((b, max_len), PAD_TOKEN, dtype=torch.long)
    region_ids = torch.full((b, max_len), REGION_PAD, dtype=torch.long)
    phase_ids = torch.full((b, max_len), PHASE_PAD, dtype=torch.long)
    pairing = torch.zeros((b, max_len), dtype=torch.float32)
    padding_mask = torch.ones((b, max_len), dtype=torch.bool)  # True everywhere, clear reals
    mfe = torch.zeros(b, dtype=torch.float32)
    access = torch.zeros(b, dtype=torch.float32)
    transcript_ids: List[str] = []

    for r, item in enumerate(batch):
        L = lengths[r]
        token_ids[r, :L] = item["token_ids"]          # type: ignore[index]
        region_ids[r, :L] = item["region_ids"]        # type: ignore[index]
        phase_ids[r, :L] = item["phase_ids"]           # type: ignore[index]
        pairing[r, :L] = item["pairing_prob"]          # type: ignore[index]
        padding_mask[r, :L] = False
        mfe[r] = item["mfe"]                            # type: ignore[index]
        access[r] = item["start_accessibility"]        # type: ignore[index]
        transcript_ids.append(str(item["transcript_id"]))  # type: ignore[index]

    return {
        "transcript_ids": transcript_ids,
        "token_ids": token_ids,
        "region_ids": region_ids,
        "phase_ids": phase_ids,
        "pairing_prob": pairing,
        "padding_mask": padding_mask,
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "mfe": mfe,
        "start_accessibility": access,
    }


class LengthBucketBatchSampler(Sampler):
    """Yield batches of indices whose sequences share a length bucket.

    Records are assigned to the smallest ``cfg.length_buckets`` boundary ``>=``
    their length (over-length records fall into a final open bucket). Within each
    bucket, indices are shuffled (seeded from ``cfg.seed`` + ``epoch``) and cut
    into contiguous ``batch_size`` batches; bucket order is likewise shuffled.
    This keeps padding low without cross-bucket length skew.

    Setting ``epoch`` via :meth:`set_epoch` reshuffles reproducibly. With
    ``shuffle=False`` iteration is fully deterministic. Complexity: O(N log N).
    """

    def __init__(
        self,
        dataset: MRNADataset,
        batch_size: int,
        cfg: Optional[DataConfig] = None,
        shuffle: bool = True,
        drop_last: bool = False,
    ) -> None:
        if cfg is None:
            cfg = DataConfig()
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.buckets = tuple(cfg.length_buckets)
        self.seed = int(cfg.seed)
        self.shuffle = shuffle
        self.drop_last = drop_last
        self._epoch = 0
        # Precompute (bucket_id -> [dataset indices]).
        self._bucket_members: Dict[int, List[int]] = {}
        for i in range(len(dataset)):
            bid = self._bucket_of(dataset.seq_length(i))
            self._bucket_members.setdefault(bid, []).append(i)

    def _bucket_of(self, length: int) -> int:
        for bid, boundary in enumerate(self.buckets):
            if length <= boundary:
                return bid
        return len(self.buckets)  # open-ended final bucket

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def _make_batches(self) -> List[List[int]]:
        rng = np.random.RandomState((self.seed + self._epoch) & 0x7FFFFFFF)
        batches: List[List[int]] = []
        for bid in sorted(self._bucket_members):
            members = list(self._bucket_members[bid])
            if self.shuffle:
                rng.shuffle(members)
            for start in range(0, len(members), self.batch_size):
                batch = members[start:start + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                batches.append(batch)
        if self.shuffle:
            order = np.arange(len(batches))
            rng.shuffle(order)
            batches = [batches[i] for i in order]
        return batches

    def __iter__(self):
        return iter(self._make_batches())

    def __len__(self) -> int:
        total = 0
        for members in self._bucket_members.values():
            if self.drop_last:
                total += len(members) // self.batch_size
            else:
                total += (len(members) + self.batch_size - 1) // self.batch_size
        return total


__all__ = [
    "MRNADataset",
    "collate_fn",
    "LengthBucketBatchSampler",
    "REGION_PAD",
    "PHASE_PAD",
]
