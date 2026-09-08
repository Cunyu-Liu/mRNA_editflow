#!/usr/bin/env python3
"""Track B: freeze the S1/M6 holdout splits (V9-1b prereg §5).

Stratified 10% holdout: S1 by (utr_group, sig_sh!='') / M6 by family.
One-shot freeze, seed 20260908; outputs {s1,m6}_holdout_manifest.json listing
canonical TRAIN_SPLIT / VALIDATION_SPLIT variant ids. Training may only consume
TRAIN_SPLIT rows; gate D3 evaluates on VALIDATION_SPLIT.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

OUT = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/experiments/analysis_track_b_20260908")
S1_LIB = OUT / "s1_stability_pairs.jsonl"
M6_LIB = OUT / "m6_ndd_translation_pairs.jsonl"
SEED = 20260908
HOLDOUT_FRAC = 0.10


def stratified_holdout(rows: list[dict], key_fn) -> tuple[set[str], set[str]]:
    buckets: dict[str, list[str]] = {}
    for r in rows:
        buckets.setdefault(key_fn(r), []).append(r["variant_id"])
    holdout: set[str] = set()
    rng = random.Random(SEED)
    for key, ids in sorted(buckets.items()):
        n_hold = max(1, round(len(ids) * HOLDOUT_FRAC)) if len(ids) >= 5 else 0
        if n_hold:
            holdout.update(rng.sample(sorted(ids), n_hold))
    return {r["variant_id"] for r in rows} - holdout, holdout


def main() -> int:
    manifests = {}
    for name, path, key_fn in (
        ("s1", S1_LIB, lambda r: f"{r.get('utr_group','?')}|{str(r.get('sig_sh','')).strip()!=''}"),
        ("m6", M6_LIB, lambda r: str(r.get("chrom", "?"))),
    ):
        rows = [json.loads(line) for line in path.open()]
        train_ids, val_ids = stratified_holdout(rows, key_fn)
        manifests[name] = {
            "schema_version": "route_a_v3_track_b_holdout_freeze.v1",
            "seed": SEED,
            "holdout_fraction": HOLDOUT_FRAC,
            "n_total": len(rows),
            "n_train": len(train_ids),
            "n_validation": len(val_ids),
            "train_split_ids": sorted(train_ids),
            "validation_split_ids": sorted(val_ids),
            "frozen_at": "2026-09-09 (V9-1b prereg §5, frozen before any training)",
        }
        out = OUT / f"{name}_holdout_manifest.json"
        if out.exists():
            raise SystemExit(f"holdout manifest already frozen: {out} (one-shot freeze violated)")
        out.write_text(json.dumps(manifests[name], indent=1))
        print(f"[{name}] total {len(rows)} -> train {len(train_ids)} / val {len(val_ids)}; wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
