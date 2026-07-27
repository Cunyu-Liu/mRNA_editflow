#!/usr/bin/env python3
"""Small GPU training run for the mixed-resolution action objective."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from mrna_editflow.core.mixed_resolution_state import MixedAction, MixedResolutionState
from mrna_editflow.data.nmi_benchmark_v2 import iter_role_records
from mrna_editflow.models.mixed_resolution_editformer import MixedResolutionEditFormer


def rows_for_training(root: Path, limit: int) -> list[dict]:
    rows = []
    for row in iter_role_records(root / "manifests" / "train.json"):
        edits = row.get("edit_list", [])
        if row.get("delta") is None or not edits or edits[0].get("region") != "five_utr":
            continue
        rows.append(row)
    random.Random(20260727).shuffle(rows)
    return rows[:limit]


def make_state(row: dict) -> tuple[MixedResolutionState, MixedAction]:
    state = MixedResolutionState(
        five_utr=str(row["source_sequence"]),
        cds="AUGUAA",  # synthetic valid coding scaffold; no biological label is inferred from it
        three_utr="",
        cargo_id=str(row.get("cargo_id", "")), cell_context=str(row.get("cell_context", "")),
        transcript_id=str(row.get("source_id", "")),
    )
    e = row["edit_list"][0]
    return state, MixedAction("UTR_SUB", int(e["pos"]), str(e["alt"]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark-root", default="data/nmi_benchmark_v2")
    ap.add_argument("--out", default="artifacts/phase3_mixed_resolution/checkpoint.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--hidden-dim", type=int, default=64)
    args = ap.parse_args()
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("Phase 3 training must run on GPU")
    device = torch.device(args.device)
    rows = rows_for_training(Path(args.benchmark_root), args.limit)
    if not rows:
        raise RuntimeError("no labeled UTR action rows available")
    model = MixedResolutionEditFormer(hidden_dim=args.hidden_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    history = []
    for epoch in range(args.epochs):
        random.Random(epoch + 7).shuffle(rows)
        losses = []
        for row in rows:
            state, target = make_state(row)
            logp, actions = model.log_probs(state)
            try:
                target_idx = actions.index(target)
            except ValueError:
                continue
            loss = -logp[target_idx]
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite mixed-resolution loss")
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            losses.append(float(loss.detach()))
        history.append({"epoch": epoch, "n": len(losses), "mean_nll": sum(losses) / max(1, len(losses)), "device": str(device)})
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "history": history, "device": str(device), "final_test_used": False}, out)
    report = out.with_suffix(".json"); report.write_text(json.dumps({"history": history, "checkpoint": str(out), "final_test_used": False}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"history": history, "checkpoint": str(out), "device": str(device)}, indent=2))


if __name__ == "__main__":
    main()
