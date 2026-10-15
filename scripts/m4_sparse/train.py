"""S4 leave-one-study-out training for SparseEditFormer.

For each benchmark (5U-A1, 3U-A1) the held-out study is left out; the model is
trained on the remaining studies (A1-NATURAL; no A2 dense pretraining exists
because EditBench-5U-A2-Dense is DORMANT by governance).  A seeded dev split of
the train fold is used for early stopping and sign-head calibration.  The test
fold (held-out study) is never used for training or early stopping.
"""
from __future__ import annotations

import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler
from scipy.stats import spearmanr

from .checkpoint import build_checkpoint_payload
from .config import (
    CHECKPOINT_SELECTION_POLICY,
    DEV_SIGN_ACCURACY_GATE,
    RANK_PAIR_DEFINITION,
    RANK_SCORE_DEFINITION,
)
from .dataset import (
    EffectDataset,
    UNKNOWN_CONTEXT,
    normalize,
)
from .model import (
    SparseEditFormer, calibrated_sign_probability, standardized_delta_logit,
)


def build_folds(rows: List[dict], benchmark: str, split: str = "S4") -> List[Dict]:
    """S4 leave-one-study-out folds for a benchmark.  Train=all other studies,
    test=held-out study (no source/study overlap)."""
    bench_rows = [r for r in rows if r["benchmark"] == benchmark]
    studies = sorted({r["study"] for r in bench_rows})
    folds = []
    for held in studies:
        test = [r for r in bench_rows if r["study"] == held]
        held_source_ids = {str(r.get("source_id")) for r in test if r.get("source_id")}
        held_sequences = {normalize(r.get("source_sequence")) for r in test
                          if normalize(r.get("source_sequence"))}
        train_candidates = [r for r in bench_rows if r["study"] != held]
        train = [
            r for r in train_candidates
            if not (
                (r.get("source_id") and str(r.get("source_id")) in held_source_ids)
                or (normalize(r.get("source_sequence")) in held_sequences)
            )
        ]
        folds.append({"held_out_study": held, "train": train, "test": test,
                      "split": split,
                      "n_train_excluded_source_overlap": (
                          len(train_candidates) - len(train))})
    return folds


def grouped_train_dev_split(rows: List[dict], dev_frac: float,
                            seed: int) -> tuple[List[dict], List[dict]]:
    """Split conservative source components, never candidate observations.

    Rows are linked when either their non-empty ``source_id`` or normalized
    ``source_sequence`` agrees (within a benchmark).  Taking transitive
    components handles aliases such as two ids for one sequence without
    allowing the same biological source to leak across train and dev merely
    because its endpoint or assay context differs.
    """
    if not rows:
        raise ValueError("grouped train/dev split requires non-empty rows")
    parent = list(range(len(rows)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left, right):
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    alias_owner = {}
    for index, row in enumerate(rows):
        benchmark = str(row.get("benchmark") or "")
        aliases = []
        source_id = str(row.get("source_id") or "").strip()
        source_sequence = normalize(row.get("source_sequence"))
        if source_id:
            aliases.append((benchmark, "source_id", source_id))
        if source_sequence:
            aliases.append((benchmark, "source_sequence", source_sequence))
        if not aliases:
            aliases.append((benchmark, "unidentified_row", str(index)))
        for alias in aliases:
            if alias in alias_owner:
                union(index, alias_owner[alias])
            else:
                alias_owner[alias] = index

    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[find(index)].append(row)
    group_entries = sorted(groups.values(), key=lambda group: min(
        str(row.get("record_id") or row.get("pair_id") or "") for row in group
    ))
    if len(group_entries) < 2:
        raise ValueError("grouped train/dev split requires at least two source groups")
    rng = np.random.RandomState(int(seed))
    shuffled_indices = rng.permutation(len(group_entries)).tolist()
    n_dev_groups = min(
        max(int(round(len(group_entries) * float(dev_frac))), 1),
        len(group_entries) - 1,
    )
    dev_indices = set(shuffled_indices[:n_dev_groups])
    train = [row for index, group in enumerate(group_entries)
             if index not in dev_indices for row in group]
    dev = [row for index, group in enumerate(group_entries)
           if index in dev_indices for row in group]
    if not train or not dev:
        raise ValueError("grouped train/dev split produced an empty partition")
    return train, dev


class GroupBatchSampler(Sampler[List[int]]):
    """Pack complete rank groups into batches so supervision is not shuffled away."""
    def __init__(self, group_ids, batch_size: int, seed: int):
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0
        groups = defaultdict(list)
        for index, group in enumerate(np.asarray(group_ids).tolist()):
            groups[int(group)].append(index)
        self.groups = [groups[key] for key in sorted(groups)]

    def __iter__(self):
        rng = np.random.RandomState(self.seed + self.epoch)
        self.epoch += 1
        batch = []
        for group_index in rng.permutation(len(self.groups)):
            indices = list(self.groups[int(group_index)])
            # A rank group is the supervision unit.  Never split an oversized
            # group merely to satisfy the nominal batch size; emit it intact.
            if batch and len(batch) + len(indices) > self.batch_size:
                yield batch
                batch = []
            if len(indices) >= self.batch_size:
                if batch:
                    yield batch
                    batch = []
                yield indices
            else:
                batch.extend(indices)
        if batch:
            yield batch

    def __len__(self):
        # Group packing depends on the per-epoch shuffle.  This conservative
        # upper bound is honest and never under-reports loader length.
        return max(1, len(self.groups))


def rank_supervision_summary(dataset: EffectDataset) -> dict:
    by_group = defaultdict(list)
    for index, group in enumerate(dataset.source_group.tolist()):
        if bool(dataset.rank_group_eligible[index]):
            by_group[int(group)].append(float(dataset.delta[index]))
    informative_pairs = 0
    for values in by_group.values():
        for left in range(len(values)):
            for right in range(left + 1, len(values)):
                informative_pairs += int(values[left] != values[right])
    return {
        "status": ("QUALIFIED" if informative_pairs > 0
                   else "NOT_RUN_ZERO_QUALIFIED_PAIRS"),
        "eligible_rank_group_count": len(by_group),
        "qualified_rank_pair_count": informative_pairs,
        "pair_definition": RANK_PAIR_DEFINITION,
        "score_definition": RANK_SCORE_DEFINITION,
    }


def _pairwise_rank_loss(score: torch.Tensor, target: torch.Tensor,
                        margin: float, device,
                        groups: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Deterministic all-pairs ranking loss within a biological source group.

    A valid pair must share source, study and endpoint (encoded by
    ``EffectDataset.source_group``).  The optional ``groups=None`` path treats
    the input as one group for backwards-compatible unit use.
    """
    if score.numel() < 2:
        return score.sum() * 0.0
    if groups is None:
        groups = torch.zeros(score.size(0), dtype=torch.long, device=score.device)
    losses = []
    for group in torch.unique(groups):
        idx = torch.nonzero(groups == group, as_tuple=False).flatten()
        if idx.numel() < 2:
            continue
        pairs = torch.triu_indices(idx.numel(), idx.numel(), offset=1,
                                   device=score.device)
        left, right = idx[pairs[0]], idx[pairs[1]]
        target_diff = target[left] - target[right]
        informative = target_diff != 0
        if not informative.any():
            continue
        score_diff = score[left] - score[right]
        direction = torch.sign(target_diff[informative])
        losses.append(F.relu(margin - direction * score_diff[informative]))
    if not losses:
        return score.sum() * 0.0
    return torch.cat(losses).mean()


def _drop_context(ids: torch.Tensor, unk_id: int, probability: float,
                  enabled: bool) -> torch.Tensor:
    """Randomly replace known contexts by a reserved trainable unknown id."""
    if not enabled or probability <= 0.0:
        return ids
    if probability >= 1.0:
        return torch.full_like(ids, int(unk_id))
    mask = torch.rand(ids.shape, device=ids.device) < probability
    if mask.numel() and not mask.any():
        mask = mask.clone()
        mask[torch.randint(mask.numel(), (1,), device=ids.device)] = True
    return torch.where(mask, torch.full_like(ids, int(unk_id)), ids)


def _predicted_delta(out: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor],
                     cfg) -> torch.Tensor:
    """Apply the frozen target/anchor semantics to the predicted mean."""
    if cfg.TARGET == "delta":
        return out["mean"]
    if cfg.TARGET == "candidate_value":
        return out["mean"] - batch["source_value"]
    raise ValueError("unsupported regression TARGET %r" % cfg.TARGET)


def compute_loss(model, batch, cfg, device) -> Dict[str, torch.Tensor]:
    context_dropout = float(getattr(cfg, "CONTEXT_DROPOUT", 0.0))
    if not hasattr(cfg, "STUDY_UNK_ID") or not hasattr(cfg, "ENDPOINT_UNK_ID"):
        raise ValueError("R2 training requires explicit UNK ids from the bound vocab")
    study_unk = int(cfg.STUDY_UNK_ID)
    endpoint_unk = int(cfg.ENDPOINT_UNK_ID)
    study = _drop_context(batch["study"], study_unk, context_dropout,
                          enabled=model.training)
    endpoint = _drop_context(batch["endpoint"], endpoint_unk, context_dropout,
                             enabled=model.training)
    out = model(batch["src"], batch["cand"], batch["edit"],
                study, endpoint, batch["bench"],
                src_pad_mask=batch.get("src_pad_mask"),
                cand_pad_mask=batch.get("cand_pad_mask"))
    regression_target = batch["y"]
    delta = batch["delta"]
    predicted_delta = _predicted_delta(out, batch, cfg)
    sign_logit = standardized_delta_logit(predicted_delta, out["logvar"])

    # The mean head is trained once with a robust Huber objective.  The
    # variance calibration term uses a detached residual, so it trains log-var
    # without silently counting the same mean regression error a second time.
    huber = F.huber_loss(out["mean"], regression_target,
                         delta=float(getattr(cfg, "HUBER_DELTA", 1.0)))
    logvar = out["logvar"].clamp(min=-10.0, max=10.0)
    squared_residual = (regression_target - out["mean"]).detach().pow(2)
    nll = 0.5 * (torch.exp(-logvar) * squared_residual + logvar)
    nll = nll.mean()

    # sign/beneficial supervision is always a delta label, even when the mean
    # head is trained on absolute candidate value.
    mask = delta != 0
    if mask.sum() > 0:
        sign_probability = calibrated_sign_probability(
            predicted_delta, out["logvar"], temperature=1.0)
        bce = F.binary_cross_entropy(sign_probability[mask],
                                     batch["sign"][mask])
    else:
        bce = sign_logit.sum() * 0.0

    # Antisymmetry is defined only for a native delta model.  Absolute
    # candidate values are not antisymmetric under source/candidate reversal.
    if cfg.TARGET == "delta":
        out_inv = model(batch["cand"], batch["src"], batch["inv_edit"],
                        study, endpoint, batch["bench"],
                        src_pad_mask=batch.get("cand_pad_mask"),
                        cand_pad_mask=batch.get("src_pad_mask"))
        consis = torch.mean((out["mean"] + out_inv["mean"]) ** 2)
    else:
        consis = out["mean"].sum() * 0.0

    # Ranking directly constrains predicted delta mean among candidates of one
    # qualified source/study/endpoint/context pool.  There is no rank head.
    rank_mask = batch["rank_group_eligible"].bool()
    rank_loss = _pairwise_rank_loss(
        predicted_delta[rank_mask], delta[rank_mask], cfg.MARGIN, device,
        groups=batch["source_group"][rank_mask],
    )

    loss = (cfg.LAMBDA_MEAN * huber + cfg.LAMBDA_VAR * nll
            + cfg.LAMBDA_SIGN * bce + cfg.LAMBDA_CONSISTENCY * consis
            + cfg.LAMBDA_RANK * rank_loss)
    return {"loss": loss, "huber": huber, "nll": nll, "bce": bce,
            "consis": consis, "rank": rank_loss}


def _train_epoch(model, loader, opt, cfg, device) -> float:
    model.train()
    tot, nb = 0.0, 0
    for batch in loader:
        b = {k: v.to(device) for k, v in batch.items()}
        opt.zero_grad()
        losses = compute_loss(model, b, cfg, device)
        losses["loss"].backward()
        opt.step()
        tot += losses["loss"].item()
        nb += 1
    return tot / max(nb, 1)


def _finite_number(value) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and np.isfinite(float(value)))


@torch.no_grad()
def evaluate_dev_selection_metrics(model, dev_rows, vocab, cfg, device,
                                   epoch: int) -> dict:
    """Study-balanced dev metrics for the frozen prospective selector.

    Each study must supply sign accuracy, delta Spearman and Gaussian NLL.
    An epoch with incomplete study coverage is not selectable.
    """
    model.eval()
    ds = EffectDataset(dev_rows, vocab, target=cfg.TARGET)
    loader = DataLoader(ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
                        num_workers=0)
    pred_delta, true_delta, logvars = [], [], []
    for batch in loader:
        b = {k: v.to(device) for k, v in batch.items()}
        out = model(b["src"], b["cand"], b["edit"], b["study"], b["endpoint"],
                    b["bench"], src_pad_mask=b.get("src_pad_mask"),
                    cand_pad_mask=b.get("cand_pad_mask"))
        pred_delta.append(_predicted_delta(out, b, cfg).cpu().numpy())
        true_delta.append(b["delta"].cpu().numpy())
        logvars.append(out["logvar"].clamp(-10.0, 10.0).cpu().numpy())
    pred_delta = np.concatenate(pred_delta)
    true_delta = np.concatenate(true_delta)
    logvars = np.concatenate(logvars)
    if len(pred_delta) != len(dev_rows):
        raise RuntimeError("dev prediction rows do not align with dev metadata")

    per_study = {}
    for study in sorted({str(row.get("study") or "") for row in dev_rows}):
        indices = np.asarray([
            index for index, row in enumerate(dev_rows)
            if str(row.get("study") or "") == study
        ], dtype=int)
        truth = true_delta[indices]
        pred = pred_delta[indices]
        lv = logvars[indices]
        nonzero = truth != 0
        sign_accuracy = (
            float(np.mean((pred[nonzero] >= 0.0) == (truth[nonzero] > 0.0)))
            if nonzero.any() else None
        )
        spearman = None
        if len(indices) >= 3 and np.std(truth) > 0 and np.std(pred) > 0:
            value = spearmanr(pred, truth).statistic
            if np.isfinite(value):
                spearman = float(value)
        nll_values = 0.5 * (np.exp(-lv) * (truth - pred) ** 2 + lv)
        gaussian_nll = (
            float(np.mean(nll_values))
            if len(nll_values) and np.all(np.isfinite(nll_values)) else None
        )
        complete = all(_finite_number(value) for value in (
            sign_accuracy, spearman, gaussian_nll
        ))
        per_study[study] = {
            "n_rows": int(len(indices)),
            "n_nonzero_sign": int(nonzero.sum()),
            "sign_accuracy": sign_accuracy,
            "delta_spearman": spearman,
            "gaussian_nll": gaussian_nll,
            "complete": complete,
        }

    complete = [entry for entry in per_study.values() if entry["complete"]]
    coverage = float(len(complete) / len(per_study)) if per_study else 0.0
    metrics = {
        "epoch": int(epoch),
        "study_macro_sign_accuracy": (
            float(np.mean([entry["sign_accuracy"] for entry in complete]))
            if coverage == 1.0 else None
        ),
        "study_macro_delta_spearman": (
            float(np.mean([entry["delta_spearman"] for entry in complete]))
            if coverage == 1.0 else None
        ),
        "study_macro_gaussian_nll": (
            float(np.mean([entry["gaussian_nll"] for entry in complete]))
            if coverage == 1.0 else None
        ),
        "study_coverage": coverage,
        "n_dev_studies": int(len(per_study)),
        "n_complete_studies": int(len(complete)),
        "n_dev_rows": int(len(dev_rows)),
        "per_study": per_study,
    }
    metrics["status"] = (
        "COMPLETE" if coverage == 1.0 else "INELIGIBLE_MISSING_STUDY_METRIC"
    )
    return metrics


def checkpoint_selection_key(metrics: Dict) -> Optional[tuple]:
    """Return the frozen lexicographic key, or ``None`` below the sign gate."""
    required = (
        "study_macro_sign_accuracy", "study_macro_delta_spearman",
        "study_macro_gaussian_nll", "study_coverage", "epoch",
    )
    missing = [name for name in required if not _finite_number(metrics.get(name))]
    if missing or float(metrics.get("study_coverage", 0.0)) != 1.0:
        raise ValueError(
            "dev checkpoint candidate missing complete study metrics: "
            + ", ".join(missing or ["study_coverage"])
        )
    sign = float(metrics["study_macro_sign_accuracy"])
    if sign < DEV_SIGN_ACCURACY_GATE:
        return None
    # Maximize sign, then Spearman, then minimize NLL, then prefer the earliest
    # epoch.  The final epoch term makes ties independent of input ordering.
    return (
        sign,
        float(metrics["study_macro_delta_spearman"]),
        -float(metrics["study_macro_gaussian_nll"]),
        -int(metrics["epoch"]),
    )


def select_dev_checkpoint(history: List[Dict]) -> Dict:
    """Select deterministically under the frozen sign>=0.60 hard gate."""
    if not history:
        raise RuntimeError("checkpoint selection requires dev metric history")
    epochs = [entry.get("epoch") for entry in history]
    if len(set(epochs)) != len(epochs):
        raise RuntimeError("checkpoint selection history has duplicate epochs")
    eligible = []
    invalid = []
    below_gate = []
    for metrics in history:
        try:
            key = checkpoint_selection_key(metrics)
        except ValueError as exc:
            invalid.append((metrics.get("epoch"), str(exc)))
            continue
        if key is None:
            below_gate.append(metrics.get("epoch"))
            continue
        eligible.append((key, metrics))
    if not eligible:
        if invalid:
            raise RuntimeError(
                "no selectable checkpoint: missing/non-finite study-balanced "
                "dev metrics; %r" % invalid
            )
        raise RuntimeError(
            "no selectable checkpoint satisfies frozen study-macro sign "
            "accuracy >= %.2f; epochs=%r" % (
                DEV_SIGN_ACCURACY_GATE, below_gate)
        )
    return max(eligible, key=lambda item: item[0])[1]


def train_fold(fold: Dict, cfg, vocab: Dict, device, out_path: Optional[Path] = None,
               seed: Optional[int] = None) -> SparseEditFormer:
    torch.manual_seed(int(seed if seed is not None else cfg.SEED))
    np.random.seed(int(seed if seed is not None else cfg.SEED))
    cfg.STUDY_UNK_ID = vocab["study"][UNKNOWN_CONTEXT]
    cfg.ENDPOINT_UNK_ID = vocab["endpoint"][UNKNOWN_CONTEXT]
    run_seed = int(seed if seed is not None else cfg.SEED)
    train_rows = fold["train"]
    if cfg.MAX_TRAIN_CAP and len(train_rows) > cfg.MAX_TRAIN_CAP:
        rng = np.random.RandomState(run_seed)
        keep = rng.choice(len(train_rows), cfg.MAX_TRAIN_CAP, replace=False)
        train_rows = [train_rows[i] for i in keep]

    tt, dev = grouped_train_dev_split(train_rows, cfg.DEV_FRAC, run_seed)

    train_ds = EffectDataset(tt, vocab, target=cfg.TARGET)
    train_loader = DataLoader(
        train_ds,
        batch_sampler=GroupBatchSampler(train_ds.source_group, cfg.BATCH_SIZE,
                                        run_seed),
        num_workers=0,
    )
    model = SparseEditFormer(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)

    rank_summary = rank_supervision_summary(train_ds)
    if (cfg.LAMBDA_RANK > 0
            and rank_summary["qualified_rank_pair_count"] == 0):
        warnings.warn(
            "R3-P0 rank objective has no qualified informative pairs; effect "
            "training may proceed but every ranking/G1 consumer is blocked",
            RuntimeWarning,
        )
    history = []
    states_by_epoch = {}
    for epoch in range(cfg.EPOCHS):
        _train_epoch(model, train_loader, opt, cfg, device)
        metrics = evaluate_dev_selection_metrics(
            model, dev, vocab, cfg, device, epoch=epoch)
        history.append(metrics)
        states_by_epoch[epoch] = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }

    selected = select_dev_checkpoint(history)
    selected_epoch = int(selected["epoch"])
    best_state = states_by_epoch[selected_epoch]
    model.load_state_dict(best_state)
    cfg.SIGN_TEMPERATURE = calibrate_sign_temperature(
        model, dev, vocab, cfg, device)
    # Positive temperature calibrates probabilities only.  The hard decision
    # remains predicted_delta_mean >= 0 and the 0.60 selection gate is frozen.
    dev_selection = {
        "policy": CHECKPOINT_SELECTION_POLICY,
        "sign_accuracy_gate": DEV_SIGN_ACCURACY_GATE,
        "selected_epoch": selected_epoch,
        "selected_metrics": selected,
        "history": history,
    }
    model.rank_supervision = dict(rank_summary)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = build_checkpoint_payload(
            best_state, cfg, fold["held_out_study"], vocab,
            rank_supervision=rank_summary,
            dev_selection=dev_selection,
        )
        payload["training_semantics"] = {
            "rank_supervision": rank_summary,
            "dev_split": "grouped_source_alias_component_v1",
            "rank_batching": "group_preserving_v1",
            "checkpoint_selection": dev_selection,
            "source_overlap_rows_excluded": int(
                fold.get("n_train_excluded_source_overlap", 0)),
        }
        torch.save(payload, out_path)
    return model


def calibrate_sign_temperature(model, dev_rows, vocab, cfg, device) -> float:
    """Fit a single temperature on the sign logits (dev fold) so the sign
    probabilities are calibrated (Platt/temperature).  Returns temperature T."""
    model.eval()
    ds = EffectDataset(dev_rows, vocab, target=cfg.TARGET)
    loader = DataLoader(ds, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0)
    logits, labels = [], []
    with torch.no_grad():
        for b in loader:
            bb = {k: v.to(device) for k, v in b.items()}
            out = model(bb["src"], bb["cand"], bb["edit"], bb["study"],
                        bb["endpoint"], bb["bench"],
                        src_pad_mask=bb.get("src_pad_mask"),
                        cand_pad_mask=bb.get("cand_pad_mask"))
            mask = bb["delta"] != 0
            if mask.sum() > 0:
                predicted_delta = _predicted_delta(out, bb, cfg)
                sign_logit = standardized_delta_logit(
                    predicted_delta, out["logvar"])
                logits.append(sign_logit[mask].cpu())
                labels.append(bb["sign"][mask].float().cpu())
    if not logits:
        return 1.0
    logits = torch.cat(logits)
    labels = torch.cat(labels)
    # Optimize log-temperature so every intermediate LBFGS evaluation remains
    # finite and positive; clamping only after unconstrained optimization can
    # otherwise divide by zero or a negative temperature during the closure.
    log_T = torch.tensor(0.0, requires_grad=True, device=device)
    opt = torch.optim.LBFGS([log_T], lr=0.05, max_iter=30)
    def closure():
        opt.zero_grad()
        temperature = torch.exp(log_T).clamp(min=0.1, max=10.0)
        z = logits.to(device) / temperature
        probability = 0.5 * (1.0 + torch.erf(z / (2.0 ** 0.5)))
        eps = torch.finfo(probability.dtype).eps
        loss = F.binary_cross_entropy(
            probability.clamp(min=eps, max=1.0 - eps), labels.to(device))
        loss.backward()
        return loss
    opt.step(closure)
    return float(torch.exp(log_T).clamp(min=0.1, max=10.0).item())
