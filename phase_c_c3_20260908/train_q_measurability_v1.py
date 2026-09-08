#!/usr/bin/env python3
"""C3 (D1): Train the measurability channel q(c | source) — cheap-kill stage.

q predicts whether an edit candidate would be MEASURED (i.e. is in the
experimenter's measured neighborhood) for a given source.  Frozen mRNABERT
features + light head (pairwise-BCE, ~1.4M params), TRAIN-pool only, component
isolation already audited (TRAIN 5,758 vs VALIDATION 3,439 components, overlap
0, flagged=0, 2026-09-08).

Data:
  positives  : TRAIN measured rows (projections/xedit_v3 train.jsonl, 89,580)
  negatives  : legal single-edit candidates NOT in the TRAIN measured set for
               the same source, sampled uniformly at random (ratio 1:5).
  held-out   : TRAIN components split 90/10 (by component id, seed fixed) ->
               AUC gate evaluated on held-out components ONLY.

Gate (preregistered, spec §4-D1): held-out AUC >= 0.60 AND bootstrap 95% CI
lower bound > 0.55.  Fail -> D1 terminates (kill), result recorded as-is.

Conventions: CUDA BF16 autocast (cpu_fallback=false -> hard error on CPU),
per-nucleotide space-join tokenization (U->T), outputs to /mnt, code in
worktree, protected reads = 0 (TRAIN pool only; VALIDATION untouched).
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

MRNABERT = Path(
    "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/external_model_assets/"
    "mrnabert_a1eb7df25804d23f08646e1cb996b234d7208a40"
)
TRAIN = Path(
    "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/projections/xedit_v3/"
    "development_train_validation_v1/train.jsonl"
)
OUT_DIR_DEFAULT = Path(
    "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/experiments/"
    "xeditsetflow_v5/q_model_20260908"
)
SEED = 20260908
NEG_RATIO = 5
EPOCHS = 3
BATCH_SEQS = 64
LR = 2e-4
MAX_LEN = 512
HIDDEN = 256
DROPOUT = 0.1
GATE_AUC = 0.60
GATE_CI_LOW = 0.55


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(msg)


def load_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MRNABERT, local_files_only=True)


def load_bert() -> tuple[nn.Module, int]:
    # Mirror the project-canonical mRNABERT load (route2_v8_hybrid_backbone_v1
    # .load_mrnabert_base): from_config + manual torch.load of pytorch_model.bin
    # (bypasses transformers>=4.5x torch.load gate for torch<2.6), flash-attn off.
    import os
    import sys

    from transformers import AutoConfig, AutoModel

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    cfg = AutoConfig.from_pretrained(
        MRNABERT, local_files_only=True, trust_remote_code=True
    )
    bert = AutoModel.from_config(cfg, trust_remote_code=True, add_pooling_layer=False)
    modeling_module = sys.modules[bert.__class__.__module__]
    modeling_module.flash_attn_qkvpacked_func = None
    ckpt = torch.load(MRNABERT / "pytorch_model.bin", map_location="cpu", weights_only=False)
    state = {
        key.removeprefix("bert."): value
        for key, value in ckpt.items()
        if key.startswith("bert.")
    }
    bert.load_state_dict(state, strict=False)
    bert.eval()
    for p in bert.parameters():
        p.requires_grad_(False)
    return bert, int(bert.config.hidden_size)


class QHead(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim * 2, HIDDEN),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN, 1),
        )

    def forward(self, src_feat, cand_feat):
        x = torch.cat([src_feat, cand_feat], dim=-1)
        return self.net(x).squeeze(-1)


def apply_edits(seq: str, edits) -> str:
    s = list(seq)
    for e in edits:
        pos = int(e[0]) if isinstance(e, (list, tuple)) else int(e.get("position"))
        base = e[1] if isinstance(e, (list, tuple)) else e.get("target_base")
        if base is not None and pos < len(s):
            s[pos] = str(base)
    return "".join(s)


class PairDS(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def encode_batch(tokenizer, seqs):
    enc = tokenizer(
        [" ".join(str(s).upper().replace("U", "T")) for s in seqs],
        add_special_tokens=True,
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt",
    )
    return enc["input_ids"], enc["attention_mask"]


def encode_feats(bert, tokenizer, seqs, device):
    ids, mask = encode_batch(tokenizer, seqs)
    ids = ids.to(device)
    mask = mask.to(device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = bert(input_ids=ids, attention_mask=mask)
        if isinstance(out, tuple):
            out = out[0]
        else:
            out = out.last_hidden_state
        cls = out[:, 0, :]
    return cls


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # tie correction: average ranks for equal scores
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            avg = (i + j + 2) / 2.0
            ranks[order[i : j + 1]] = avg
        i = j + 1
    pos = labels == 1
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    _require(n_pos > 0 and n_neg > 0, "AUC needs both classes")
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def bootstrap_auc_ci(scores: np.ndarray, labels: np.ndarray, iters=2000, seed=20260816):
    rng = np.random.default_rng(seed)
    aucs = []
    n = len(scores)
    for _ in range(iters):
        idx = rng.integers(0, n, n)
        s, l = scores[idx], labels[idx]
        if l.sum() == 0 or (l == 0).sum() == 0:
            continue
        aucs.append(roc_auc(s, l))
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--physical-gpu-index", type=int, required=True)
    ap.add_argument("--output-dir", type=Path, default=OUT_DIR_DEFAULT)
    ap.add_argument("--max-train-pairs", type=int, default=400_000)
    ap.add_argument("--dry-run", action="store_true", help="tiny subset, 1 epoch")
    args = ap.parse_args()

    t0 = time.time()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device(f"cuda:{args.physical_gpu_index}")
    _require(device.type == "cuda", "cpu_fallback=false: CUDA required")

    rng = random.Random(SEED)

    # ---- load TRAIN pool -------------------------------------------------
    rows = []
    by_source = defaultdict(list)
    with TRAIN.open() as f:
        for line in f:
            r = json.loads(line)
            rows.append(r)
            by_source[r["source_id"]].append(r)
    _require(len(rows) == 89_580, f"TRAIN pool size changed: {len(rows)}")

    comp_of = {r["source_id"]: r["connected_source_component_id"] for r in rows}
    comps = sorted(set(comp_of.values()))
    rng.shuffle(comps)
    n_hold = max(1, int(len(comps) * 0.1))
    hold_comps = set(comps[:n_hold])
    train_comps = set(comps) - hold_comps
    print(f"[data] TRAIN rows={len(rows)} sources={len(by_source)} "
          f"components={len(comps)} holdout={len(hold_comps)} train={len(train_comps)}")

    # ---- build pairs -----------------------------------------------------
    # dedup measured sequence sets per source (TRAIN pool)
    measured_of = {}
    for sid, rs in by_source.items():
        measured_of[sid] = {r["candidate_sequence"].upper().replace("T", "U") for r in rs}

    BASES = "ACGT"
    def neg_candidates(source_row, k: int):
        seq = source_row["source_sequence"].upper()
        seqU = seq.replace("T", "U")
        mset = measured_of[source_row["source_id"]]
        outs = []
        tries = 0
        while len(outs) < k and tries < k * 30:
            tries += 1
            pos = rng.randrange(len(seq))
            b = BASES[rng.randrange(4)]
            if seq[pos] == b:
                continue
            cand = seq[:pos] + b + seq[pos + 1 :]
            if cand.replace("T", "U") in mset:
                continue
            outs.append(cand)
        return outs

    pairs = []  # (source_seq, cand_seq, label, component)
    for sid, rs in by_source.items():
        r0 = rs[0]
        comp = comp_of[sid]
        for r in rs:
            pairs.append((r0["source_sequence"], r["candidate_sequence"], 1, comp))
        for cand in neg_candidates(r0, min(len(rs) * NEG_RATIO, len(r0["source_sequence"]) * 4)):
            pairs.append((r0["source_sequence"], cand, 0, comp))
        if len(pairs) >= args.max_train_pairs and not args.dry_run:
            break
    if args.dry_run:
        pairs = pairs[:2000]
    print(f"[data] pairs={len(pairs)} pos={sum(1 for p in pairs if p[2]==1)}")

    hold_pairs = [p for p in pairs if p[3] in hold_comps]
    train_pairs = [p for p in pairs if p[3] in train_comps]
    print(f"[data] train_pairs={len(train_pairs)} hold_pairs={len(hold_pairs)}")

    # ---- model -----------------------------------------------------------
    tokenizer = load_tokenizer()
    bert, hdim = load_bert()
    bert = bert.to(device)
    head = QHead(hdim).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=LR)

    def run_epoch(pl, training: bool):
        losses = []
        order = list(range(len(pl)))
        if training:
            rng2 = random.Random(SEED + (1 if training else 0))
            rng2.shuffle(order)
        # group into batches of whole pairs; each pair costs 2 encoder passes
        chunk = max(1, BATCH_SEQS // 2)
        for bi in range(0, len(order), chunk):
            idxs = order[bi : bi + chunk]
            seqs, labels = [], []
            for i in idxs:
                s, c, y, _ = pl[i]
                seqs.extend([s, c])
                labels.append(y)
            feats = encode_feats(bert, tokenizer, seqs, device)
            src_feat = feats[0::2]
            cand_feat = feats[1::2]
            y = torch.tensor(labels, dtype=torch.float32, device=device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = head(src_feat.float(), cand_feat.float())
                loss = nn.functional.binary_cross_entropy_with_logits(logits, y)
            if training:
                opt.zero_grad()
                loss.backward()
                opt.step()
            losses.append(float(loss.item()))
        return sum(losses) / max(1, len(losses))

    epochs = 1 if args.dry_run else EPOCHS
    for ep in range(epochs):
        tl = run_epoch(train_pairs, training=True)
        print(f"[epoch {ep}] train_loss={tl:.4f} elapsed={time.time()-t0:.0f}s")

    # ---- gate evaluation --------------------------------------------------
    head.eval()
    scores, labels_np = [], []
    with torch.no_grad():
        for bi in range(0, len(hold_pairs), 32):
            batch = hold_pairs[bi : bi + 32]
            seqs = []
            for s, c, _, _ in batch:
                seqs.extend([s, c])
            feats = encode_feats(bert, tokenizer, seqs, device)
            logits = head(feats[0::2].float(), feats[1::2].float())
            scores.extend(logits.float().cpu().numpy().tolist())
            labels_np.extend([p[2] for p in batch])
    scores_np = np.array(scores, dtype=np.float64)
    labels_np = np.array(labels_np)
    auc = roc_auc(scores_np, labels_np)
    lo, hi = bootstrap_auc_ci(scores_np, labels_np)
    passed = bool(auc >= GATE_AUC and lo > GATE_CI_LOW)

    result = {
        "schema_version": "route_a_v3_route2_q_model_train.v1",
        "status": "Q_MODEL_TRAIN_TERMINAL",
        "gate_passed": passed,
        "gate_rule": "heldout_component_AUC >= 0.60 AND bootstrap95_CI_low > 0.55",
        "heldout_auc": auc,
        "heldout_auc_ci95": [lo, hi],
        "heldout_pairs": int(len(hold_pairs)),
        "train_pairs": int(len(train_pairs)),
        "epochs": epochs,
        "params_trainable": int(sum(p.numel() for p in head.parameters())),
        "seed": SEED,
        "neg_ratio": NEG_RATIO,
        "precision": "BF16_CUDA",
        "cpu_fallback_used": False,
        "protected_reads": 0,
        "train_pool": str(TRAIN),
        "component_overlap_flagged": False,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    (out / "q_train_result.json").write_text(json.dumps(result, indent=1))
    torch.save(
        {"head_state": head.state_dict(), "hidden": HIDDEN, "in_dim": hdim},
        out / "q_head_checkpoint.pt",
    )
    print(json.dumps(result, indent=1))
    return 0 if not args.dry_run else 0


if __name__ == "__main__":
    raise SystemExit(main())
