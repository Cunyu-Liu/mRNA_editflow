#!/usr/bin/env python3
"""Route 2 Baseline P0: MPRAU matched fine-tuning external arms (RNA-FM / UTR-LM).

Preregistered experiment (docs/paper/route2_mprau_matched_ft_external_prereg_v1.md).
Upgrades the R5 "structural blank" claim for MPRAU (ENCSR854RUF, 3'UTR allelic
skew, 6 contexts) into a testable statement: external general-purpose RNA LMs,
fine-tuned under the SAME task data and update budget as our per-task arm
(directft Arm A), still cannot move the task -> blank claim stands; a win is
reported honestly and enters the W ladder.

Per arm (--model rnafm|utrlm), single-task MPRAU:
- Init: official pretrained weights (multimolecule RNA-FM conversion / official
  UTR-LM SISS checkpoint via the vendored esm package).
- Fine-tuning mode (preregistered): FULL-PARAMETER fine-tuning for both arms --
  each model's officially recommended downstream mode (multimolecule README
  HF-Trainer usage; UTR-LM --finetune branch trains the whole ESM2 + head).
- Readout: RNA-FM = masked-mean-pool over non-special tokens + fresh linear
  head; UTR-LM = BOS representation (official --bos_emb) + fresh linear head.
  prediction = f(candidate) - f(source) (frozen-delta, train & eval isomorphic).
- Objective/loss/budget: EXACT mirror of directft Arm A
  (run_route2_mrnabert_directft_arm_a_v1.py): direction_normalized_delta
  regression + huber(delta=1.0) + cross-source pairwise softplus + soft-Spearman
  (temp 0.2) + within-source 0.5; pass 1-2 {1.0, 0.25, 0.0} / pass 3-8
  {1.0, 0.5, 0.25}; 12,048 rows/pass (uniform subsample of the 55,704-row TRAIN
  pool) x 8 passes x batch 32 = 3,016 updates; AdamW wd 1e-4, cosine to 10%
  after 5% warmup, grad clip 1.0, BF16 autocast, seed 20260907.
- Selection: FINAL-PASS-8-FIXED (no peak-picking); per-pass VALIDATION metrics
  are logged for the record only.
- MPRAU MAIN criterion: variant pair-mean rho (record_id minus ":context:"
  suffix, >=2 contexts, per-variant means; NEVER the pooled run_summary
  pair_mean). Paired bootstrap (2,000 iters, seed 20260816) vs V5 multitask
  0.1025 AND vs Saluki frozen weak control 0.1205.
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EVAL_REPO = "/home/cunyuliu/mrna_editflow_goal/worktrees/route_a_v3_setflow_v5_base_fix_20260901"
_ev_spec = importlib.util.spec_from_file_location(
    "ev", EVAL_REPO + "/scripts/route_a_v3/evaluate_route2_prediction_v1.py"
)
ev = importlib.util.module_from_spec(_ev_spec)
sys.modules["ev"] = ev
_ev_spec.loader.exec_module(ev)

MNT = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2")
STUDY = "ENCSR854RUF"
RNAFM_PATH = MNT / "external_model_assets/rnafm"
UTRLM_SCRIPTS = MNT / "external_models/utrlm/Scripts"
UTRLM_CKPT = (
    MNT / "external_model_assets/utrlm/Model/Pretrained/"
    "ESM2SISS_FS4.1_fiveSpeciesCao_6layers_16heads_128embedsize_4096batchToks_"
    "lr1e-05_supervisedweight1.0_structureweight1.0_MLMLossMin_epoch93.pkl"
)
PROJECTION_DIR = MNT / "projections/xedit_v3/development_train_validation_v1"
MANIFEST = MNT / "manifests/route2_development_frozen_v1/development_manifest.jsonl"
CANONICAL = MNT / "canonical/ENCSR854RUF/v1/canonical_records.private.jsonl"
V5_PRED_GLOB = str(MNT / "experiments/xeditcritic_v5/*/v5_full/final_validation_predictions.jsonl")
SALUKI_PRED = MNT / "experiments/analysis_saluki_frozen_mprau_20260903/predictions.jsonl"
OUT_ROOT = MNT / "experiments/analysis_mprau_matched_ft_external_20260904"

BATCH = 32
PASSES = 8
WEIGHT_DECAY = 1e-4
SEED = 20260907
HUBER_DELTA = 1.0
SOFT_RANK_TEMPERATURE = 0.2
WITHIN_SOURCE_WEIGHT = 0.5
GRAD_CLIP = 1.0
K = 10
BOOT_ITERS = 2000
BOOT_SEED = 20260816
MPRAU_ROWS_PER_PASS = 12048  # registered budget: ceil(12048/32)=377 updates/pass
EXPECTED_TRAIN = 55704
EXPECTED_VALIDATION = 12048

ARMS = {
    "rnafm": {
        "backbone_lr": 2e-5,
        "head_lr": 1e-4,
        "label": "multimolecule RNA-FM (99.5M) full-parameter fine-tune",
    },
    "utrlm": {
        "backbone_lr": 1e-4,
        "head_lr": 2e-4,
        "label": "UTR-LM SISS (1.2M) full-parameter fine-tune (official --finetune mode)",
    },
}

REFERENCE = {
    "v5_multitask_pair_mean": 0.1025,
    "saluki_frozen_weak_control_pair_mean": 0.1205,
    "w_ladder_arms_pair_mean": "0.0510-0.0883",
    "mrnabert_directft_arm_a_pair_mean": -0.0908,
    "ceiling": 0.683,
}


# ----------------------------------------------------------------------------
# data (mirror directft Arm A)
# ----------------------------------------------------------------------------

def load_projection_rows(split: str, limit: int | None = None) -> list[dict]:
    path = PROJECTION_DIR / f"{split.lower()}.jsonl"
    rows = []
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("study_unit_id") == STUDY and row.get("split") == split:
                rows.append(row)
                if limit is not None and len(rows) >= limit:
                    break
    return rows


def manifest_validation_ids() -> set[str]:
    ids = set()
    with MANIFEST.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row["study_unit_id"] == STUDY and row["split"] == "VALIDATION":
                ids.add(str(row["canonical_record_id"]))
    return ids


# ----------------------------------------------------------------------------
# loss (exact mirror of directft Arm A)
# ----------------------------------------------------------------------------

def loss_weights(pass_number: int) -> dict[str, float]:
    if pass_number <= 2:
        return {"huber": 1.0, "pairwise": 0.25, "soft_spearman": 0.0}
    return {"huber": 1.0, "pairwise": 0.5, "soft_spearman": 0.25}


def soft_ranks(values: torch.Tensor, temperature: float) -> torch.Tensor:
    return torch.sigmoid((values.unsqueeze(0) - values.unsqueeze(1)) / temperature).sum(dim=1)


def target_midranks(values: torch.Tensor) -> torch.Tensor:
    order = values.argsort()
    ranks = torch.empty_like(values, dtype=torch.float32)
    ranks[order] = torch.arange(1, values.numel() + 1, device=values.device, dtype=torch.float32)
    return ranks


def pairwise_softplus(predictions, targets, pairs):
    left = torch.tensor([p[0] for p in pairs], device=predictions.device)
    right = torch.tensor([p[1] for p in pairs], device=predictions.device)
    target_delta = targets[left] - targets[right]
    prediction_delta = predictions[left] - predictions[right]
    return F.softplus(-target_delta.sign() * prediction_delta).mean()


def soft_spearman(predictions, targets, temperature):
    soft = soft_ranks(predictions, temperature)
    target = target_midranks(targets)
    soft_c = soft - soft.mean()
    target_c = target - target.mean()
    denom = torch.linalg.vector_norm(soft_c) * torch.linalg.vector_norm(target_c)
    if not bool((denom > 0).item()):
        return predictions.new_zeros(())
    return 1.0 - (soft_c * target_c).sum() / denom


# ----------------------------------------------------------------------------
# MPRAU variant pair-mean caliber (mirror directft Arm A / W-ladder adjudication)
# ----------------------------------------------------------------------------

def mprau_variant_table(predictions: dict[str, float], targets: dict[str, float]) -> dict[str, tuple[float, float]]:
    by_variant: dict[str, list[str]] = defaultdict(list)
    for rid in predictions:
        by_variant[rid.split(":context:")[0]].append(rid)
    variants = {}
    for variant, rids in by_variant.items():
        if len(rids) >= 2:
            variants[variant] = (
                float(np.mean([targets[r] for r in rids])),
                float(np.mean([predictions[r] for r in rids])),
            )
    return variants


def pair_mean_rho(variants: dict[str, tuple[float, float]]) -> float:
    from scipy.stats import spearmanr

    t = np.asarray([v[0] for v in variants.values()], dtype=np.float64)
    p = np.asarray([v[1] for v in variants.values()], dtype=np.float64)
    value = spearmanr(t, p).statistic
    return float(value)


def paired_bootstrap_vs_reference(ref_variants, arm_variants, label, iters=BOOT_ITERS, seed=BOOT_SEED):
    from scipy.stats import spearmanr

    shared = sorted(set(ref_variants) & set(arm_variants))
    if len(shared) < 10:
        return {"reference": label, "shared_variant_count": len(shared), "skipped": "too few shared variants"}
    t = np.array([ref_variants[v][0] for v in shared])
    pr = np.array([ref_variants[v][1] for v in shared])
    pa = np.array([arm_variants[v][1] for v in shared])
    rhor = float(spearmanr(t, pr).statistic)
    rhoa = float(spearmanr(t, pa).statistic)
    rng = np.random.default_rng(seed)
    n = len(shared)
    deltas = []
    for _ in range(iters):
        idx = rng.integers(0, n, n)
        try:
            rr = spearmanr(t[idx], pr[idx]).statistic
            ra = spearmanr(t[idx], pa[idx]).statistic
            if np.isfinite(rr) and np.isfinite(ra):
                deltas.append(float(ra) - float(rr))
        except Exception:
            continue
    deltas = np.asarray(deltas)
    ci = [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))]
    return {
        "reference": label,
        "shared_variant_count": n,
        "reference_pair_mean_spearman": rhor,
        "arm_pair_mean_spearman": rhoa,
        "delta_pair_mean_spearman": rhoa - rhor,
        "bootstrap_ci_95": ci,
        "bootstrap_iterations": int(len(deltas)),
        "ci_excludes_zero": bool(ci[0] > 0 or ci[1] < 0),
        "ceiling_ratio_0_683": rhoa / 0.683,
    }


def load_v5_variants():
    paths = sorted(glob.glob(V5_PRED_GLOB))
    if not paths:
        return None
    rows = {}
    with open(paths[0]) as handle:
        for line in handle:
            r = json.loads(line)
            rows[str(r["record_id"])] = r
    by_variant = defaultdict(list)
    for rid, row in rows.items():
        if rid.startswith("ENCSR854RUF:"):
            by_variant[rid.split(":context:")[0]].append(row)
    variants = {}
    for variant, rs in by_variant.items():
        if len(rs) >= 2:
            variants[variant] = (
                float(np.mean([r["target"] for r in rs])),
                float(np.mean([r["prediction"] for r in rs])),
            )
    return variants


def load_saluki_variants(v_targets: dict[str, float]):
    if not SALUKI_PRED.is_file():
        return None
    preds = {}
    with SALUKI_PRED.open() as handle:
        for line in handle:
            r = json.loads(line)
            rid = str(r["canonical_record_id"])
            if rid in v_targets:
                preds[rid] = float(r["predicted_direction_normalized_delta"])
    return mprau_variant_table(preds, v_targets)


# ----------------------------------------------------------------------------
# models
# ----------------------------------------------------------------------------

class RnaFmRegressor(nn.Module):
    """multimolecule RNA-FM + fresh linear head; masked-mean-pool over non-special tokens."""

    def __init__(self, base: nn.Module, width: int):
        super().__init__()
        self.base = base
        self.head = nn.Linear(width, 1)

    def forward(self, input_ids, attention_mask):
        hidden = self.base(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        keep = attention_mask.bool()
        special = torch.zeros_like(keep)
        special[:, 0] = True  # BOS
        lengths = keep.sum(dim=1)
        special[torch.arange(keep.size(0), device=keep.device), lengths - 1] = True  # EOS
        keep = keep & ~special
        pooled = (hidden * keep.unsqueeze(-1)).sum(dim=1) / keep.sum(dim=1, keepdim=True).clamp_min(1)
        return self.head(pooled).squeeze(-1)


class UtrLmRegressor(nn.Module):
    """Official UTR-LM SISS ESM2 + fresh linear head on the BOS representation."""

    def __init__(self, base: nn.Module, width: int):
        super().__init__()
        self.base = base
        self.head = nn.Linear(width, 1)

    def forward(self, tokens):
        out = self.base(
            tokens,
            [6],
            need_head_weights=False,
            return_contacts=False,
            return_representation=True,
        )
        bos = out["representations"][6][:, 0]
        return self.head(bos).squeeze(-1)


def build_rnafm(device: torch.device):
    import os as _os
    _os.environ.setdefault("HF_HUB_OFFLINE", "1")
    _os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from multimolecule.models.rnafm import RnaFmModel, RnaTokenizer

    tokenizer = RnaTokenizer.from_pretrained(RNAFM_PATH, local_files_only=True)
    base = RnaFmModel.from_pretrained(RNAFM_PATH, local_files_only=True)
    backbone_params = sum(p.numel() for p in base.parameters())
    if backbone_params < 90_000_000:
        raise SystemExit(f"RNA-FM pretrained geometry changed: {backbone_params:,} params")
    model = RnaFmRegressor(base, base.config.hidden_size)
    # full-parameter fine-tune (preregistered): everything trainable; the
    # fresh-initialized pooler is unused by the readout and receives no
    # loss gradient, so it is excluded for an honest trainable count
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    if hasattr(model.base, "pooler") and model.base.pooler is not None:
        model.base.pooler.requires_grad_(False)
    model.to(device)
    return model, tokenizer, backbone_params


def build_utrlm(device: torch.device):
    if not (UTRLM_SCRIPTS / "esm/model/esm2_secondarystructure.py").is_file():
        raise SystemExit("official modified UTR-LM ESM source is absent")
    sys.path.insert(0, str(UTRLM_SCRIPTS))
    from esm.data import Alphabet
    from esm.model.esm2_secondarystructure import ESM2

    alphabet = Alphabet(mask_prob=0.0, standard_toks="AGCT")
    if alphabet.tok_to_idx != {
        "<pad>": 0, "<eos>": 1, "<unk>": 2, "A": 3, "G": 4,
        "C": 5, "T": 6, "<cls>": 7, "<mask>": 8, "<sep>": 9,
    }:
        raise SystemExit("official UTR-LM vocabulary changed")
    base = ESM2(num_layers=6, embed_dim=128, attention_heads=16, alphabet=alphabet)
    raw_state = torch.load(UTRLM_CKPT, map_location="cpu", weights_only=True)
    if not (raw_state and all(str(key).startswith("module.") for key in raw_state)):
        raise SystemExit("UTR-LM checkpoint DDP key format changed")
    state = {str(key).removeprefix("module."): value for key, value in raw_state.items()}
    base.load_state_dict(state, strict=True)
    backbone_params = sum(p.numel() for p in base.parameters())
    if backbone_params != 1_208_559:
        raise SystemExit(f"UTR-LM parameter geometry changed: {backbone_params:,}")
    model = UtrLmRegressor(base, base.embed_dim)
    # full-parameter fine-tune (preregistered, official --finetune mode);
    # pretraining artifact heads that receive no loss gradient are excluded,
    # but lm_head.weight is TIED to embed_tokens.weight and must stay
    # trainable (freezing lm_head wholesale would freeze the embeddings)
    model.base.contact_head.requires_grad_(False)
    model.base.supervised_linear.requires_grad_(False)
    model.base.structure_linear.requires_grad_(False)
    tied = model.base.embed_tokens.weight
    for name, parameter in model.base.lm_head.named_parameters():
        if parameter is not tied:
            parameter.requires_grad_(False)
    if not model.base.embed_tokens.weight.requires_grad:
        raise SystemExit("UTR-LM embedding table accidentally frozen (weight-tying guard failed)")
    model.to(device)
    return model, alphabet, backbone_params


def tokenize_rnafm(tokenizer, sequences: list[str]) -> dict[str, torch.Tensor]:
    # projection sequences are RNA alphabet (U); multimolecule tokenizer expects U
    enc = tokenizer([str(s).upper() for s in sequences], add_special_tokens=True, padding=True, truncation=False, return_tensors="pt")
    return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}


def tokenize_utrlm(alphabet, sequences: list[str]) -> dict[str, torch.Tensor]:
    converter = alphabet.get_batch_converter()
    raw = [(str(i), str(s).upper().replace("U", "T"), str(s).upper().replace("U", "T"), []) for i, s in enumerate(sequences)]
    converted = converter(raw)
    tokens = converted[3]
    return {"input_ids": tokens, "attention_mask": tokens.ne(alphabet.padding_idx)}


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=sorted(ARMS))
    parser.add_argument("--physical-gpu-index", required=True, type=int)
    parser.add_argument("--smoke", action="store_true", help="dry-run ~100 updates; writes /tmp only")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable - GPU required (BF16-only discipline)")
    device = torch.device(f"cuda:{args.physical_gpu_index}")
    torch.manual_seed(SEED)
    arm_cfg = ARMS[args.model]

    if args.smoke:
        out_dir = Path(f"/tmp/matched_ft_smoke/{args.model}")
        passes = 2
        train_limit, val_limit = 1600, 512
        budget_rows = 1600  # 50 updates/pass x 2 passes = ~100 updates
        batch = BATCH
    else:
        out_dir = OUT_ROOT / args.model
        passes = PASSES
        train_limit, val_limit = None, None
        budget_rows = MPRAU_ROWS_PER_PASS
        batch = BATCH
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- data ----
    rows = load_projection_rows("TRAIN", limit=train_limit)
    if not args.smoke and len(rows) != EXPECTED_TRAIN:
        raise SystemExit(f"expected {EXPECTED_TRAIN} TRAIN rows for {STUDY}, got {len(rows)}")
    vrows = load_projection_rows("VALIDATION", limit=val_limit)
    if not args.smoke and len(vrows) != EXPECTED_VALIDATION:
        raise SystemExit(f"expected {EXPECTED_VALIDATION} VALIDATION rows for {STUDY}, got {len(vrows)}")
    if budget_rows > len(rows):
        budget_rows = len(rows)
    print(f"model={args.model} train_rows={len(rows)} validation_rows={len(vrows)} "
          f"budget_rows_per_pass={budget_rows} passes={passes} batch={batch} seed={SEED}", flush=True)

    targets = torch.tensor([float(row["direction_normalized_delta"]) for row in rows], dtype=torch.float32)
    groups = [str(row["source_group_id"]) for row in rows]
    vrecords = {str(row["canonical_record_id"]): row for row in vrows}
    vids = sorted(vrecords)
    v_targets = {rid: float(vrecords[rid]["direction_normalized_delta"]) for rid in vids}

    # ---- model + tokenization ----
    if args.model == "rnafm":
        model, tokenizer, backbone_params = build_rnafm(device)
        encode = lambda seqs: tokenize_rnafm(tokenizer, seqs)
    else:
        model, alphabet, backbone_params = build_utrlm(device)
        encode = lambda seqs: tokenize_utrlm(alphabet, seqs)

    enc_src = encode([row["source_sequence"] for row in rows])
    enc_cnd = encode([row["candidate_sequence"] for row in rows])
    v_enc_src = encode([vrecords[rid]["source_sequence"] for rid in vids])
    v_enc_cnd = encode([vrecords[rid]["candidate_sequence"] for rid in vids])

    head_params = list(model.head.parameters())
    head_ids = {id(p) for p in head_params}
    backbone_trainable = [p for p in model.parameters() if id(p) not in head_ids and p.requires_grad]
    trainable = backbone_trainable + head_params
    trainable_count = sum(p.numel() for p in trainable)
    print(f"backbone params: {backbone_params:,}; trainable (full-FT + head): {trainable_count:,}", flush=True)

    # ---- validation helpers ----
    validation_ids = manifest_validation_ids() if not args.smoke else set(vids)
    observations = ev.load_observations([Path(CANONICAL)], validation_ids)

    def forward_batch(enc, sl):
        if args.model == "rnafm":
            return model(enc["input_ids"][sl].to(device), enc["attention_mask"][sl].to(device))
        return model(enc["input_ids"][sl].to(device))

    def predict_validation() -> dict[str, float]:
        model.eval()
        values = []
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for start in range(0, len(vids), 256):
                sl = slice(start, start + 256)
                s = forward_batch(v_enc_src, sl).float()
                c = forward_batch(v_enc_cnd, sl).float()
                values.append((c - s).cpu().numpy())
        model.train()
        delta = np.concatenate(values)
        return {rid: float(delta[i]) for i, rid in enumerate(vids)}

    def validation_report(preds: dict[str, float]) -> dict:
        report: dict = {}
        variants = mprau_variant_table(preds, v_targets)
        report["variant_pair_mean_spearman"] = pair_mean_rho(variants) if variants else None
        report["variant_count"] = len(variants)
        if observations is not None:
            sub = {k: v for k, v in preds.items() if k in validation_ids}
            if len(sub) == len(validation_ids):
                metrics = ev.evaluate(observations, sub, K)
                report["task_macro_spearman"] = metrics.get("task_macro_spearman")
                report["top_1"] = metrics.get("source_macro_top_1_accuracy")
                report["ndcg_at_10"] = metrics.get("source_macro_ndcg_at_k")
            else:
                report["aligned_eval"] = f"skipped: {len(validation_ids) - len(sub)} predictions missing"
        return report

    # ---- optimizer (preregistered full-FT LRs) ----
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_trainable, "lr": arm_cfg["backbone_lr"]},
            {"params": head_params, "lr": arm_cfg["head_lr"]},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    updates_per_pass = (budget_rows + batch - 1) // batch
    total_steps = updates_per_pass * passes
    warmup_steps = max(1, int(total_steps * 0.05))

    def lr_multiplier(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)

    print(f"pass 0 (pre-training) validation: {json.dumps(validation_report(predict_validation()))}", flush=True)
    history = []
    step = 0
    for pass_number in range(1, passes + 1):
        weights = loss_weights(pass_number)
        pool = torch.randperm(len(rows))
        order = pool[:budget_rows]
        losses, hubers, pairs_n, within_n = [], [], 0, 0
        for start in range(0, len(order), batch):
            idx = order[start: start + batch]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                src_pred = forward_batch(enc_src, idx).float()
                cnd_pred = forward_batch(enc_cnd, idx).float()
                prediction = cnd_pred - src_pred

            batch_targets = targets[idx].to(device)
            batch_groups = [groups[i] for i in idx.tolist()]
            huber = F.huber_loss(prediction, batch_targets, delta=HUBER_DELTA)
            n = len(idx)
            different_pairs, same_pairs = [], []
            for i in range(n):
                for j in range(i + 1, n):
                    (same_pairs if batch_groups[i] == batch_groups[j] else different_pairs).append((i, j))
            pairwise = (
                pairwise_softplus(prediction, batch_targets, different_pairs)
                if different_pairs else prediction.new_zeros(())
            )
            within = (
                pairwise_softplus(prediction, batch_targets, same_pairs)
                if same_pairs else prediction.new_zeros(())
            )
            soft = (
                soft_spearman(prediction, batch_targets, SOFT_RANK_TEMPERATURE)
                if weights["soft_spearman"] > 0 else prediction.new_zeros(())
            )
            loss = (
                weights["huber"] * huber
                + weights["pairwise"] * pairwise
                + weights["soft_spearman"] * soft
                + WITHIN_SOURCE_WEIGHT * within
            )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            losses.append(float(loss))
            hubers.append(float(huber))
            pairs_n += len(different_pairs)
            within_n += len(same_pairs)
            step += 1
        val = validation_report(predict_validation())
        history.append({
            "pass": pass_number,
            "mean_loss": float(np.mean(losses)),
            "mean_huber": float(np.mean(hubers)),
            "different_source_pairs": pairs_n,
            "within_source_pairs": within_n,
            "validation": val,
        })
        print(f"pass {pass_number}/{passes}: loss {np.mean(losses):.4f} huber {np.mean(hubers):.4f} "
              f"val {json.dumps(val)} lr {scheduler.get_last_lr()[0]:.2e}", flush=True)

    # ---- final frozen-delta evaluation (FINAL-PASS-8-FIXED) ----
    model.eval()
    preds = predict_validation()
    final_metrics = validation_report(preds)

    torch.save({
        "schema_version": "route_a_v3_route2_mprau_matched_ft_external.v1",
        "model_state_dict": {k: v for k, v in model.state_dict().items()},
        "model": args.model,
        "backbone_lr": arm_cfg["backbone_lr"],
        "head_lr": arm_cfg["head_lr"],
        "seed": SEED, "passes": passes,
        "train_pool_rows": len(rows), "budget_rows_per_pass": budget_rows,
    }, out_dir / "matched_ft_checkpoint.pt")

    variants = mprau_variant_table(preds, v_targets)
    v5_variants = load_v5_variants()
    saluki_variants = load_saluki_variants(v_targets)
    bootstrap_v5 = paired_bootstrap_vs_reference(v5_variants, variants, "v5_multitask") if v5_variants else None
    bootstrap_saluki = paired_bootstrap_vs_reference(saluki_variants, variants, "saluki_frozen_weak_control") if saluki_variants else None

    report = {
        "schema_version": "route_a_v3_route2_mprau_matched_ft_external.v1",
        "mode": "MPRAU_MATCHED_FT_EXTERNAL_FULL_PARAMETER",
        "model": args.model,
        "model_label": arm_cfg["label"],
        "init": "official pretrained weights (no task-agnostic pre-finetune)",
        "smoke": bool(args.smoke),
        "backbone_parameter_count": backbone_params,
        "trainable_parameter_count": trainable_count,
        "finetune_mode": "full-parameter (preregistered; official recommended downstream mode)",
        "readout": "masked-mean-pool + linear head" if args.model == "rnafm" else "BOS representation + linear head",
        "budget": {
            "passes": passes,
            "updates_per_pass": updates_per_pass,
            "total_updates": total_steps,
            "batch": batch,
            "train_pool_rows": len(rows),
            "budget_rows_per_pass": budget_rows,
            "mprau_subsampling": "uniform without replacement per pass" if budget_rows < len(rows) else None,
        },
        "seed": SEED,
        "selection_rule": "FINAL-PASS-8-FIXED_NO_VALIDATION_PEAK_RESELECTION",
        "pass_history": history,
        "final_metrics": final_metrics,
        "reference": REFERENCE,
        "mprau_paired_bootstrap_vs_v5": bootstrap_v5,
        "mprau_paired_bootstrap_vs_saluki_frozen": bootstrap_saluki,
        "pooled_pair_mean_warning": "run_summary pooled pair_mean_spearman is never the criterion",
    }
    (out_dir / "matched_ft_results.json").write_text(json.dumps(report, indent=1, sort_keys=True))

    with (out_dir / "predictions.jsonl").open("w") as handle:
        for rid in vids:
            handle.write(json.dumps({
                "record_id": rid, "prediction": preds[rid], "target": v_targets[rid],
                "source_group_id": vrecords[rid]["source_group_id"],
                "task_id": vrecords[rid]["task_id"],
            }) + "\n")
    print(json.dumps(final_metrics, indent=1))
    print("wrote", out_dir / "matched_ft_results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
