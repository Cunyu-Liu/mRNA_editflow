"""Offline per-token embedding cache for MEF Stage-A training.

Runs a :class:`~mrna_editflow.models.backbones.FrozenBackbone` over a corpus of
:class:`~mrna_editflow.core.schema.MRNARecord` and writes per-token embeddings to
sharded ``.npz`` files plus a JSON manifest recording, for each shard, its record
ids, row layout and a SHA256 checksum for integrity/reproducibility.

Because the backbone is frozen, embeddings only need to be computed once; the
generation head can then train against cached features (huge CPU speed-up, and
essential when a real foundation model is expensive to run).

Layout
------
``<out_dir>/shard_00000.npz`` contains, per record ``k`` in the shard:

* ``emb_k``    : ``float16 [L_k, out_dim]`` per-token embedding.
* ``tokens_k`` : ``int16   [L_k]`` nucleotide token ids.
* ``region_k`` : ``int8    [L_k]`` region ids.

``<out_dir>/manifest.json`` records backbone config, ``out_dim``, dtype, and a
list of shard descriptors ``{path, sha256, records:[{transcript_id, index,
length}]}``.

CLI
---
``python -m mrna_editflow.data.cache_embeddings --jsonl records.jsonl \
      --out-dir cache/ --backbone none --hidden-dim 64 --shard-size 256``

Complexity: O(N * Lmax * out_dim) compute; memory bounded by one batch.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from dataclasses import asdict
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
from tqdm import tqdm

from ..core.config import BackboneConfig
from ..core.schema import MRNARecord
from ..models.backbones import FrozenBackbone


def _sha256_of_file(path: str, chunk: int = 1 << 20) -> str:
    """Stream a file through SHA256. Complexity: O(filesize)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def load_records_jsonl(path: str) -> List[MRNARecord]:
    """Load MRNARecords from a JSONL file (one record dict per line)."""
    records: List[MRNARecord] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(MRNARecord.from_dict(json.loads(line)))
    return records


def _pad_batch(
    records: Sequence[MRNARecord], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
    """Right-pad a group of records into token/region tensors + true lengths."""
    from ..core.constants import PAD_TOKEN

    tok_rows = [r.token_ids() for r in records]
    reg_rows = [r.region_ids() for r in records]
    lengths = [len(t) for t in tok_rows]
    max_len = max(lengths) if lengths else 1
    max_len = max(max_len, 1)
    tok = torch.full((len(records), max_len), PAD_TOKEN, dtype=torch.long)
    reg = torch.zeros((len(records), max_len), dtype=torch.long)
    for i, (t, rr) in enumerate(zip(tok_rows, reg_rows)):
        if t:
            tok[i, : len(t)] = torch.tensor(t, dtype=torch.long)
            reg[i, : len(rr)] = torch.tensor(rr, dtype=torch.long)
    pad_mask = tok == PAD_TOKEN
    return tok.to(device), reg.to(device), pad_mask.to(device), lengths


@torch.no_grad()
def cache_embeddings(
    records: Sequence[MRNARecord],
    backbone_cfg: BackboneConfig,
    out_dir: str,
    shard_size: int = 256,
    batch_size: int = 16,
    device: Optional[torch.device] = None,
    show_progress: bool = True,
) -> str:
    """Compute + cache per-token backbone embeddings; return manifest path.

    Parameters
    ----------
    records : sequence of MRNARecord
        Corpus to encode.
    backbone_cfg : BackboneConfig
        Backbone selection (frozen); ``hidden_dim`` fixes ``out_dim``.
    out_dir : str
        Output directory (created if missing).
    shard_size : int
        Records per ``.npz`` shard.
    batch_size : int
        Records per forward pass (padded).
    """
    device = device or torch.device("cpu")
    os.makedirs(out_dir, exist_ok=True)
    backbone = FrozenBackbone(backbone_cfg).to(device)
    backbone.eval()

    manifest: Dict = {
        "backbone": asdict(backbone_cfg),
        "out_dim": backbone.out_dim,
        "is_real_backbone": backbone.is_real,
        "emb_dtype": "float16",
        "num_records": len(records),
        "shards": [],
    }

    n_shards = (len(records) + shard_size - 1) // max(shard_size, 1)
    shard_iter = range(n_shards)
    if show_progress:
        shard_iter = tqdm(shard_iter, desc="caching shards", unit="shard")

    for shard_id in shard_iter:
        lo = shard_id * shard_size
        hi = min(lo + shard_size, len(records))
        shard_records = records[lo:hi]

        arrays: Dict[str, np.ndarray] = {}
        rec_meta: List[Dict] = []
        for b_lo in range(0, len(shard_records), batch_size):
            batch = shard_records[b_lo : b_lo + batch_size]
            tok, reg, pad_mask, lengths = _pad_batch(batch, device)
            emb = backbone.embed(tok, reg, pad_mask)  # [b, max_len, out_dim]
            emb_np = emb.to(torch.float16).cpu().numpy()
            tok_np = tok.cpu().numpy().astype(np.int16)
            reg_np = reg.cpu().numpy().astype(np.int8)
            for j, rec in enumerate(batch):
                idx = len(rec_meta)
                length = lengths[j]
                arrays[f"emb_{idx}"] = emb_np[j, :length]
                arrays[f"tokens_{idx}"] = tok_np[j, :length]
                arrays[f"region_{idx}"] = reg_np[j, :length]
                rec_meta.append(
                    {"transcript_id": rec.transcript_id, "index": idx, "length": int(length)}
                )

        shard_path = os.path.join(out_dir, f"shard_{shard_id:05d}.npz")
        # Write deterministically so the SHA256 is reproducible.
        buf = io.BytesIO()
        np.savez(buf, **arrays)
        with open(shard_path, "wb") as fh:
            fh.write(buf.getvalue())
        manifest["shards"].append(
            {
                "path": os.path.basename(shard_path),
                "sha256": _sha256_of_file(shard_path),
                "records": rec_meta,
            }
        )

    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest_path


def verify_manifest(out_dir: str) -> bool:
    """Re-hash every shard and check it matches the manifest. O(total bytes)."""
    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    for shard in manifest["shards"]:
        path = os.path.join(out_dir, shard["path"])
        if not os.path.exists(path):
            return False
        if _sha256_of_file(path) != shard["sha256"]:
            return False
    return True


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Cache MEF frozen-backbone embeddings.")
    p.add_argument("--jsonl", required=True, help="Input JSONL of MRNARecord dicts.")
    p.add_argument("--out-dir", required=True, help="Output cache directory.")
    p.add_argument("--backbone", default="none", help="Backbone name.")
    p.add_argument("--hidden-dim", type=int, default=64, help="Backbone output width.")
    p.add_argument("--granularity", default="nt", choices=["nt", "dual", "codon"])
    p.add_argument("--weights-path", default=None)
    p.add_argument("--shard-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--no-progress", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    records = load_records_jsonl(args.jsonl)
    cfg = BackboneConfig(
        name=args.backbone,
        hidden_dim=args.hidden_dim,
        freeze=True,
        weights_path=args.weights_path,
        granularity=args.granularity,
    )
    manifest_path = cache_embeddings(
        records,
        cfg,
        out_dir=args.out_dir,
        shard_size=args.shard_size,
        batch_size=args.batch_size,
        show_progress=not args.no_progress,
    )
    print(f"[cache_embeddings] wrote manifest: {manifest_path}")
    ok = verify_manifest(args.out_dir)
    print(f"[cache_embeddings] integrity check: {'OK' if ok else 'FAILED'}")


__all__ = ["cache_embeddings", "load_records_jsonl", "verify_manifest", "main"]


if __name__ == "__main__":
    main()
