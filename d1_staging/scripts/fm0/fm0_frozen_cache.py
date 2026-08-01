#!/usr/bin/env python
"""FM0-01: Precompute & cache frozen UTR-LM embeddings for D_C sources.

For every source_sequence (and candidate_sequence) in the canonical records
of the D_C datasets (GSE114002, GSE200304, GSE217518, GSE149487), compute
UTR-LM frozen embeddings under three pooling modes (cls, mean, max) and
write them to /mnt/cunyuliu/fm0_cache/utrlm_frozen_embeddings/.

This is the "frozen foundation" arm of the H7 ablation. Downstream MK0/EF0
read these cached embeddings instead of re-running the encoder on every step.

Output layout:
  {frozen_cache_dir}/{accession}_{region}.{pooling}.npz
    -> arrays: 'record_id', 'sequence_kind' (source|candidate),
       'embedding' [N, H], 'hidden_size', 'pooling', 'model_id', 'revision'

Acceptance (FM0-01): frozen cache (also a contract H7 must_validate arm).

Usage:
    python scripts/fm0/fm0_frozen_cache.py \
        [--canonical-records data/d1_canonical_records.jsonl] \
        [--cache-dir /mnt/cunyuliu/fm0_cache/utrlm_frozen_embeddings] \
        [--device cuda:5] [--batch-size 64] [--max-records N]

Contract: utr_editflow_contract_v2 (FROZEN)
Task: FM0-01
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from fm0_common import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    ensure_offline_env,
    get_model_id,
    load_config,
    load_model,
    load_tokenizer,
    pick_gpu_device,
    pool_embeddings,
    tokenize_sequences,
    write_json,
)


# Per contract §7: D_C datasets = source-matched measured interventions
DC_ACCESSIONS = {"GSE114002", "GSE200304", "GSE217518", "GSE149487"}


def iter_canonical_records(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def collect_sequences(records_path: Path, max_records: int = 0):
    """Collect unique (record_id, accession, region, kind, sequence) tuples
    from canonical records. Yields de-duplicated sequences for efficiency."""
    seen_seq = {}
    items = []
    for rec in iter_canonical_records(records_path):
        acc = rec.get("accession", "?")
        if acc not in DC_ACCESSIONS:
            continue
        region = rec.get("region", "?")
        rid = rec.get("record_id", "?")
        for kind in ("source", "candidate"):
            seq = (rec.get(f"{kind}_sequence") or "").upper().replace("U", "T")
            if not seq:
                continue
            # De-dup by sequence; keep first record_id for traceability
            if seq not in seen_seq:
                seen_seq[seq] = (rid, kind)
            items.append({
                "record_id": rid,
                "accession": acc,
                "region": region,
                "kind": kind,
                "sequence": seq,
            })
        if max_records and len(items) >= max_records:
            break
    return items, seen_seq


def run_frozen_cache(
    records_path: Path,
    cache_dir: Path,
    device_str: str,
    batch_size: int,
    max_records: int,
) -> dict:
    cfg = load_config()
    ensure_offline_env()
    import numpy as np
    import torch

    if device_str == "auto":
        device = pick_gpu_device()
    else:
        if device_str.startswith("cuda") and not torch.cuda.is_available():
            sys.exit(f"[FM0] FATAL: device={device_str} but CUDA unavailable.")
        device = torch.device(device_str)

    tok = load_tokenizer()
    model = load_model(device=str(device))

    items, seen_seq = collect_sequences(records_path, max_records=max_records)
    unique_seqs = list(seen_seq.keys())
    print(f"[FM0-01] {len(items)} (record,kind) pairs; "
          f"{len(unique_seqs)} unique sequences")

    # Group items by (accession, region, kind) for output files
    # but compute embeddings per unique sequence, then broadcast.
    # To keep it simple & robust, we embed ALL items (with dup) —
    # de-dup would require extra bookkeeping; the cache is for read-only
    # downstream use, so we accept slight recompute. If too slow, switch
    # to unique-seq embedding + index map.
    # For 30k D_C records x 2 kinds = 60k forward positions at B=64, L~50,
    # this is ~1000 batches → minutes, acceptable.

    # Group by (accession, region, kind)
    groups = {}
    for it in items:
        key = (it["accession"], it["region"], it["kind"])
        groups.setdefault(key, []).append(it)

    max_pos = cfg["model"]["max_position_embeddings"]
    poolings = cfg["adaptation"]["frozen"]["pooling"]  # [cls, mean, max]

    written_files = []
    total_embedded = 0
    for (acc, region, kind), group in sorted(groups.items()):
        seqs = [it["sequence"] for it in group]
        rids = [it["record_id"] for it in group]

        # Compute embeddings for each pooling mode
        embeds_by_pool = {p: [] for p in poolings}
        for i in range(0, len(seqs), batch_size):
            batch_seqs = seqs[i : i + batch_size]
            enc = tokenize_sequences(batch_seqs, tok, max_length=max_pos)
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                out = model(**enc)
            h = out.last_hidden_state
            for p in poolings:
                v = pool_embeddings(h, enc["attention_mask"], mode=p)
                embeds_by_pool[p].append(v.cpu().numpy())

        for p in poolings:
            arr = np.concatenate(embeds_by_pool[p], axis=0)  # [N, H]
            safe_region = region.replace("'", "").replace(" ", "")
            out_path = cache_dir / f"{acc}_{safe_region}_{kind}.{p}.npz"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                out_path,
                record_id=np.array(rids, dtype=object),
                embedding=arr.astype(np.float32),
                hidden_size=np.int32(arr.shape[1]),
                pooling=np.array(p),
                model_id=np.array(cfg["model"]["model_id"]),
                revision=np.array(cfg["model"]["revision"]),
                accession=np.array(acc),
                region=np.array(region),
                kind=np.array(kind),
                num_records=np.int32(arr.shape[0]),
            )
            written_files.append({
                "path": str(out_path),
                "accession": acc,
                "region": region,
                "kind": kind,
                "pooling": p,
                "num_records": arr.shape[0],
                "hidden_size": int(arr.shape[1]),
                "size_bytes": out_path.stat().st_size,
            })
        total_embedded += len(seqs)

    report = {
        "task_id": "FM0-01",
        "acceptance": "frozen cache",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "device": str(device),
        "model_id": get_model_id(),
        "revision": cfg["model"]["revision"],
        "records_path": str(records_path),
        "cache_dir": str(cache_dir),
        "dc_accessions": sorted(DC_ACCESSIONS),
        "poolings": poolings,
        "num_total_items": len(items),
        "num_unique_sequences": len(unique_seqs),
        "num_total_embedded": total_embedded,
        "batch_size": batch_size,
        "written_files": written_files,
        "pass": len(written_files) > 0 and total_embedded > 0,
    }
    return report


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--canonical-records",
        default=str(Path(cfg["storage"].get("canonical_records",
                                              "data/d1_canonical_records.jsonl"))),
        help="Path to d1_canonical_records.jsonl",
    )
    ap.add_argument(
        "--cache-dir",
        default=cfg["storage"]["frozen_cache_dir"],
        help="Output cache directory.",
    )
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-records", type=int, default=0,
                    help="If >0, stop after this many records (for smoke runs).")
    ap.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_DIR / "frozen_cache_report.json"),
    )
    args = ap.parse_args()

    report = run_frozen_cache(
        Path(args.canonical_records),
        Path(args.cache_dir),
        args.device,
        args.batch_size,
        args.max_records,
    )
    out = Path(args.output)
    write_json(out, report)

    print(f"[FM0-01] Frozen cache report -> {out}")
    print(f"  cache_dir: {report['cache_dir']}")
    print(f"  device: {report['device']}")
    print(f"  total items embedded: {report['num_total_embedded']}")
    print(f"  unique sequences:     {report['num_unique_sequences']}")
    print(f"  files written:       {len(report['written_files'])}")
    for w in report["written_files"][:6]:
        print(f"    {w['path']}  N={w['num_records']}  H={w['hidden_size']}  "
              f"{w['size_bytes']:,}B")
    if len(report["written_files"]) > 6:
        print(f"    ... ({len(report['written_files']) - 6} more)")
    print(f"  PASS: {report['pass']}")

    if not report["pass"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
