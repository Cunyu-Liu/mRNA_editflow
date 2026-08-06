"""O0-X runner: closed measured-space optimization over real measured pools.

Loads the trained anchored SparseEditFormer checkpoints (candidate_value target,
delta = cand_pred - MEASURED source) produced by the M4 phase, and runs the O0-X
closed measured-space optimization on each benchmark's S4 held-out study pools.

Strategies (all on the *measured* candidate pool, NO Flow):
  - exact_enumeration     (oracle ceiling)
  - random_legal          (baseline)
  - greedy                (edit-distance heuristic)
  - beam                  (beam over measured pool)
  - sparseeditformer_rerank (model-predicted delta)

Metrics: NDCG@k, top-decile recall, enrichment@k, normalized regret, with
query count and forward equivalents accounting.  Aggregation: source-macro ->
study-macro.  Singleton pools excluded from ranking headline but counted.

Usage:
    python -m scripts.m4_sparse.run_o0x --dataset artifacts/b0x/effect_dataset.jsonl \
        --ckpt-dir artifacts/m4_sparse/candval --gpu cuda:1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo/scripts

from scripts.m4_sparse import config as C
from scripts.m4_sparse.dataset import build_vocab
from scripts.m4_sparse.evaluate import predict_model
from scripts.m4_sparse.o0x_search import run_benchmark_o0x
from scripts.m4_sparse.train import build_folds


def load_rows(path):
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("delta") is not None:
                rows.append(r)
    return rows


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


def load_model(ckpt_path, device):
    import torch
    from scripts.m4_sparse.model import SparseEditFormer
    sd = torch.load(ckpt_path, map_location=str(device), weights_only=False)
    cfg = sd["cfg"]
    model = SparseEditFormer(cfg).to(device)
    model.load_state_dict(sd["state_dict"])
    model.eval()
    return model, cfg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=Path(C.DATASET))
    ap.add_argument("--ckpt-dir", type=Path, default=Path("artifacts/m4_sparse/candval"))
    ap.add_argument("--out-dir", type=Path, default=Path("artifacts/o0x"))
    ap.add_argument("--benchmarks", nargs="+", default=["5U-A1", "3U-A1"])
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--beam-width", type=int, default=8)
    ap.add_argument("--gpu", default=None)
    args = ap.parse_args()

    cfg = C.get_config()
    rows = load_rows(args.dataset)
    print("loaded %d delta-defined records" % len(rows))
    vocab = build_vocab(rows)
    cfg.N_STUDIES = len(vocab["study"])
    cfg.N_ENDPOINTS = len(vocab["endpoint"])
    cfg.N_BENCHMARKS = len(vocab["benchmark"])

    device, err = select_device(cfg, args.gpu)
    results = {
        "phase": "O0-X", "split": cfg.PRIMARY_SPLIT, "k": args.k,
        "beam_width": args.beam_width, "n_records": len(rows),
        "benchmarks": {}, "gpu": None, "cuda_available": (device is not None),
    }
    if device is None:
        results["gpu_status"] = "not_run:" + err
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "o0x_results.json").write_text(json.dumps(results, indent=2))
        print("CUDA unavailable -> SparseEditFormer rerank not_run: " + err)
        return 0
    results["gpu"] = str(device)
    print("using device " + str(device))

    for benchmark in args.benchmarks:
        bench_rows = [r for r in rows if r["benchmark"] == benchmark]
        folds = build_folds(bench_rows, benchmark, split=cfg.PRIMARY_SPLIT)

        def make_predict_fn(held):
            ckpt = args.ckpt_dir / ("model_%s__%s.pt" % (benchmark, held))
            if not ckpt.exists():
                return None, ("missing_ckpt", str(ckpt))
            model, mcfg = load_model(ckpt, device)
            return (lambda test_rows, m=model, mc=mcfg: predict_model(m, test_rows, vocab, mc, device)), None

        # Build per-fold predict functions; run no-model if a fold lacks a ckpt.
        fold_predicts = {}
        missing = []
        for fold in folds:
            pf, err = make_predict_fn(fold["held_out_study"])
            if pf is None:
                missing.append((fold["held_out_study"], err))
            fold_predicts[fold["held_out_study"]] = pf

        t0 = time.time()
        # Run O0-X with a predict_fn that dispatches per-held-out-study.
        def dispatch_predict(test_rows):
            held = test_rows[0]["study"]
            pf = fold_predicts.get(held)
            if pf is None:
                raise RuntimeError("no predict fn for " + held)
            return pf(test_rows)

        res = run_benchmark_o0x(
            bench_rows, folds, predict_fn=dispatch_predict,
            k=args.k, beam_width=args.beam_width,
            random_seed=cfg.SEED, device=device)
        res["wall_time_s"] = time.time() - t0
        res["missing_checkpoints"] = [{"study": s, "reason": e} for s, e in missing]
        results["benchmarks"][benchmark] = res

        print("\n=== %s ===" % benchmark)
        sm = res["study_macro"]
        for s, m in sm.items():
            print("%-26s ndcg=%s tdr=%s enrich=%s regret=%s queries=%d feq=%d"
                  % (s, _fmt(m["macro_over_studies_ndcg_at_k"]),
                     _fmt(m["macro_over_studies_top_decile_recall"]),
                     _fmt(m["macro_over_studies_enrichment_at_k"]),
                     _fmt(m["macro_over_studies_normalized_regret"]),
                     m["total_query_count"], m["total_forward_equivalents"]))
        print("summary:", res["summary"])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "o0x_results.json"
    out_path.write_text(json.dumps(results, indent=2, default=float))
    print("wrote %s" % out_path)
    return 0


def _fmt(v):
    return "n/a" if v is None else ("%.4f" % v)


if __name__ == "__main__":
    raise SystemExit(main())