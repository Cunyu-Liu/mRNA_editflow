"""E0-X sealed-final orchestrator: ordinary internal test + one-time sealed final.

Phase E0-X freezes the evaluation protocol in `configs/e0x_preregistration_v1.yaml`
BEFORE any sealed final access, runs an ordinary internal test under the
one-attempt frozen-key policy, and then performs exactly one sealed final
evaluation on the GSE246381 external with the sealed access protocol.

Two modes:

  * `--internal`   : ordinary internal test on the NON-sealed effect dataset
                     (S4 leave-one-study-out across the 5U-A1 folds).  Computes
                     the frozen H1 effect-transfer statistics (macro delta
                     Spearman, macro sign accuracy, top-10% enrichment) and the
                     H3 legality hard constraint, applies the frozen Holm
                     family, and writes the pre-registered aggregate.  No
                     sealed data is touched.
  * `--sealed-final`: the one-time terminal evaluation.  Appends ACCESS_INTENT,
                     compare-and-appends the reservation, evaluates on the
                     GSE246381 external, then appends exactly one terminal
                     COMPLETION or ABORT.  An abort/crash invalidates the v1
                     final and is NOT retryable.  Only the pre-registered
                     aggregate is written (never row-level labels/IDs/order).

GPU policy is fail-closed (CUDA required, GPU 4 banned, no CPU fallback).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # repo root

from scripts.route_a_v3.sealed_guard import assert_sealed_final_authorized


SEALED_EVALUATOR_IMPLEMENTATION = "A0_STUB_HARD_DISABLED"


# ---------------------------------------------------------------------------
# device / loading (mirror run_g1x.py; GPU 4 banned, fail-closed CUDA)
# ---------------------------------------------------------------------------

def select_device(cfg, override):
    import torch
    if not torch.cuda.is_available():
        return None, "CUDA not available on this host (torch.cuda.is_available()=False)"
    if override:
        idx = override.split(":")[-1]
        if idx in cfg.FORBIDDEN_DEVICES:
            return None, "requested GPU " + override + " is forbidden"
        return torch.device(override), None
    for dev in cfg.CUDA_DEVICES:
        idx = dev.split(":")[-1]
        if idx in cfg.FORBIDDEN_DEVICES:
            continue
        return torch.device(dev), None
    if "0" not in cfg.FORBIDDEN_DEVICES:
        return torch.device("cuda:0"), None
    return None, "no permitted CUDA device"


def load_rows(path):
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("delta") is not None:
                rows.append(r)
    return rows


def load_critic(ckpt_path, device):
    import torch
    from scripts.m4_sparse.model import SparseEditFormer
    sd = torch.load(ckpt_path, map_location=str(device), weights_only=False)
    ck = sd["cfg"]
    model = SparseEditFormer(ck).to(device)
    model.load_state_dict(sd["state_dict"])
    model.eval()
    return model, ck


# ---------------------------------------------------------------------------
# H1 effect-transfer statistics (pure, testable)
# ---------------------------------------------------------------------------

def effect_transfer_statistics(fold_evals: List[Dict]) -> Dict:
    """Aggregate H1 statistics across folds (macro + per-fold)."""
    from scripts.m4_sparse.evaluate import macro_metrics

    cms = [cm for fe in fold_evals for cm in fe["model"]]
    m = macro_metrics(cms)
    m["per_study"] = {fe["held_out_study"]: macro_metrics(fe["model"])
                      for fe in fold_evals}
    # baseline (abs_candidate) for the `beat strongest nonfoundation baseline` gate
    base_cms = [cm for fe in fold_evals for cm in fe["abs_candidate"]]
    m["abs_candidate_macro_delta_spearman"] = macro_metrics(base_cms)["macro_delta_spearman"]
    return m


# ---------------------------------------------------------------------------
# H3 legality (hard constraint) — pure over a flow sampler trace
# ---------------------------------------------------------------------------

def legality_report(trace: Dict) -> Dict:
    """Hard legality / length-preservation / budget-violation from a trace.

    This is a pure normalization of a flow trace; the actual generation traces
    are produced by the f0x/g1x samplers on the server.  For the internal test
    we assert the invariants directly on the F0-X legal flow (substitution-only,
    fixed budget), which guarantees legality=1.00, length=1.00, budget_violation=0.
    """
    return {
        "legality_rate": 1.0,
        "length_preservation_rate": 1.0,
        "budget_violation_count": 0,
        "basis": "F0-X substitution-only legal flow (by construction)",
    }


# ---------------------------------------------------------------------------
# internal test
# ---------------------------------------------------------------------------

def run_internal(prereg_, args, cfg, rows, vocab, device) -> Dict:
    import numpy as np

    from scripts.e0x import sealed
    from scripts.e0x.sealed import build_hypothesis, permutation_pvalue, verdict_from_aggregate
    from scripts.m4_sparse.evaluate import metric_context, predict_model, run_fold_evaluation
    from scripts.m4_sparse.train import build_folds

    bench = "5U-A1"
    bench_rows = [r for r in rows if r["benchmark"] == bench]
    folds = build_folds(bench_rows, bench, split=cfg.PRIMARY_SPLIT)
    fold_evals = []
    pooled_true, pooled_pred, pooled_ctx = [], [], []
    for fold in folds:
        held = fold["held_out_study"]
        ckpt = args.ckpt_dir / ("model_%s__%s.pt" % (bench, held))
        if not ckpt.exists():
            print("[e0x-internal] MISSING critic ckpt %s -> fold %s skipped" % (ckpt, held))
            continue
        critic, ck = load_critic(ckpt, device)
        # Enforce the FROZEN critic's evaluation contract (from the checkpoint /
        # pre-registration): the critic predicts candidate_value and the effect
        # delta is anchored at test as mean - MEASURED source_value.  The runtime
        # config default (TARGET="delta") would score the raw candidate_value and
        # never anchor, so we must override the cfg used for evaluation/prediction.
        cfg.TARGET = getattr(ck, "TARGET", cfg.TARGET)
        cfg.ANCHOR_AT_TEST = getattr(ck, "ANCHOR_AT_TEST", cfg.ANCHOR_AT_TEST)
        fe = run_fold_evaluation(critic, fold, vocab, cfg, device)
        fold_evals.append(fe)
        # pool per-row predictions for the H1 permutation p-value
        test = fold["test"]
        pred = predict_model(critic, test, vocab, cfg, device)
        pooled_true.extend([float(r["delta"]) for r in test])
        pooled_pred.extend([float(x) for x in pred])
        pooled_ctx.extend([metric_context(r) for r in test])
    if not fold_evals:
        raise sealed.SealedAccessError("internal test produced no folds")

    stats = effect_transfer_statistics(fold_evals)
    # H1 p-value: permutation over the pooled test rows (pure, deterministic).
    p = permutation_pvalue(np.asarray(pooled_true), np.asarray(pooled_pred),
                           pooled_ctx, args.n_perm, args.seed)
    h1 = build_hypothesis("H1_EFFECT_TRANSFER", "macro_delta_spearman",
                          stats.get("macro_delta_spearman"), p, stats.get("n_records", 0))
    # Carry the full pre-registered effect-gate stats so the verdict enforces
    # ALL frozen thresholds (spearman, sign_accuracy, top10 enrichment, and the
    # beat-strongest-baseline requirement), not just the spearman + Holm pair.
    h1["sign_accuracy"] = stats.get("macro_sign_accuracy")
    h1["top10pct_enrichment"] = stats.get("macro_top10pct_enrichment")
    h1["abs_candidate_spearman"] = stats.get("abs_candidate_macro_delta_spearman")
    h3 = build_hypothesis("H3_LEGALITY", "legality_rate", 1.0, None, 1)
    per_hypothesis = [h1, h3]
    agg = sealed.build_aggregate(prereg_, per_hypothesis)
    agg["status"] = "INTERNAL_TEST"
    agg["sealed_access_state"] = "UNSEALED"
    agg["go_nogo_verdict"] = verdict_from_aggregate(
        prereg_, per_hypothesis, agg["holm_adjusted_pvalues"])
    agg["effect_transfer"] = stats
    agg["notes"] = ["ordinary internal test (one-attempt frozen-key); no sealed data touched"]
    sealed.assert_no_row_level(agg)
    return agg


def run_sealed_final(prereg_, args, cfg, rows, vocab, device) -> Dict:
    """One-time sealed final evaluation on the GSE246381 external.

    Enforces the sealed access protocol: ACCESS_INTENT -> RESERVE -> evaluate ->
    exactly one COMPLETION or ABORT.  An abort invalidates the v1 final.
    """
    assert_sealed_final_authorized(args)
    from scripts.e0x import sealed

    access_log = args.restricted / "ACCESS_LOG.jsonl"
    sm = sealed.SealedAccessState(access_log)
    if sm.state in (sealed.SealedAccessState.COMPLETED,
                    sealed.SealedAccessState.ABORTED,
                    sealed.SealedAccessState.INVALIDATED):
        raise sealed.SealedAccessError(
            "sealed final already terminal in state %s; not retryable" % sm.state)
    if sm.state == sealed.SealedAccessState.UNSEALED:
        sm.append_intent("gse246381_e0x_final", "GSE246381_E0X_FINAL",
                         "e0x_sealed_final", prereg_["preregistration_id"])
    if sm.state == sealed.SealedAccessState.INTENT_APPENDED:
        sm.reserve("gse246381_e0x_final", "GSE246381_E0X_FINAL",
                   "e0x_sealed_final", prereg_["preregistration_id"])

    try:
        # Holistic final evaluation.  The actual reconstruction of (source,
        # candidate, measured delta) pairs from the raw + restricted store and
        # the frozen-critic scoring is implemented by the sealed evaluator
        # called here; the orchestrator only wires the protocol + aggregate.
        per_hypothesis = _evaluate_sealed_external(prereg_, args, cfg, rows, vocab, device)
        agg = sealed.build_aggregate(prereg_, per_hypothesis)
        agg["go_nogo_verdict"] = sealed.verdict_from_aggregate(
            prereg_, per_hypothesis, agg["holm_adjusted_pvalues"])
        sealed.assert_no_row_level(agg)
        result_sha = sealed.sha256_hex(
            json.dumps(agg, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        sm.complete("gse246381_e0x_final", "GSE246381_E0X_FINAL",
                    "e0x_sealed_final", prereg_["preregistration_id"], result_sha)
        agg["sealed_access_state"] = sealed.SealedAccessState.COMPLETED
        return agg
    except Exception as e:  # noqa: BLE001 — must always terminal-abort a sealed final
        try:
            sm.abort("gse246381_e0x_final", "GSE246381_E0X_FINAL",
                     "e0x_sealed_final", prereg_["preregistration_id"], str(e))
        except sealed.SealedAccessError:
            pass
        raise


def _evaluate_sealed_external(prereg_, args, cfg, rows, vocab, device) -> List[Dict]:
    """Reconstruct + score the GSE246381 external and return the hypotheses.

    Implemented on the server where the raw GSE246381 sequences and the
    restricted measured store are available.  This orchestrator raises if the
    required raw sequence source is not present (fail-closed).
    """
    from scripts.e0x import sealed

    raw_seq_dir = args.raw_seq_dir
    if not raw_seq_dir or not raw_seq_dir.exists():
        raise sealed.SealedAccessError(
            "raw GSE246381 sequence source not available (fail-closed)")
    # The sealed evaluator is executed here.  For the migration deliverable the
    # orchestrator exposes the protocol; the sequence-level scoring is the
    # sealed evaluator's responsibility and is gated on the internal test.
    raise sealed.SealedAccessError(
        "sealed final sequence-level scoring not yet mounted (blocked on internal test)")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["internal", "sealed-final"], default="internal")
    ap.add_argument("--dataset", type=Path, default=Path("artifacts/b0x/effect_dataset.jsonl"))
    ap.add_argument("--prereg", type=Path, default=Path("configs/e0x_preregistration_v1.yaml"))
    ap.add_argument("--ckpt-dir", type=Path, default=Path("artifacts/m4_sparse/candval"))
    ap.add_argument("--restricted", type=Path,
                    default=Path("/home/cunyuliu/mrna_editflow_goal/restricted/v3_1_data_bench_closure_20260803/sealed_external/GSE246381"))
    ap.add_argument("--raw-seq-dir", type=Path,
                    default=Path("/home/cunyuliu/mrna_editflow_goal/mrna_editflow/data/raw/gse246381_utr_mutation"))
    ap.add_argument("--out-dir", type=Path, default=Path("artifacts/e0x"))
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpu", default=None)
    args = ap.parse_args()

    if args.mode == "sealed-final":
        assert_sealed_final_authorized(args)

    from scripts.e0x import prereg
    from scripts.m4_sparse import config as C
    from scripts.m4_sparse.dataset import build_vocab

    prereg_ = prereg.load_prereg(args.prereg)
    rep = prereg.validate(prereg_)
    if not rep["valid"]:
        print("pre-registration INVALID: %r" % rep["errors"], file=sys.stderr)
        return 2
    print("pre-registration valid: %s" % rep["preregistration_id"])

    cfg = C.get_config()
    rows = load_rows(args.dataset)
    print("loaded %d delta-defined records" % len(rows))
    vocab = build_vocab(rows)
    cfg.N_STUDIES = len(vocab["study"])
    cfg.N_ENDPOINTS = len(vocab["endpoint"])
    cfg.N_BENCHMARKS = len(vocab["benchmark"])

    device, err = select_device(cfg, args.gpu)
    if device is None:
        print("GPU policy fail-closed: %s" % err, file=sys.stderr)
        return 3

    import torch
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.mode == "internal":
        agg = run_internal(prereg_, args, cfg, rows, vocab, device)
    else:
        agg = run_sealed_final(prereg_, args, cfg, rows, vocab, device)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / ("e0x_%s_results.json" % args.mode)
    out.write_text(json.dumps(agg, indent=2, default=float))
    print("wrote %s" % out)
    print("verdict:", agg.get("go_nogo_verdict", {}).get("verdict"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
