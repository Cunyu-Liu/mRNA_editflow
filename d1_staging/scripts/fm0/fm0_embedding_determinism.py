#!/usr/bin/env python
"""FM0-01: Embedding determinism test for UTR-LM.

Verifies that running the same input through the frozen encoder twice produces
BIT-IDENTICAL output (determinism). This is a contract-level requirement: a
non-deterministic frozen backbone would invalidate frozen-cache reuse and
make downstream delta comparisons unreliable.

Acceptance (FM0-01): embedding determinism.

Usage:
    python scripts/fm0/fm0_embedding_determinism.py [--device cuda:5] \
        [--output data/fm0/embedding_determinism_report.json]

Contract: utr_editflow_contract_v2 (FROZEN)
Task: FM0-01
"""

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from fm0_common import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    ensure_offline_env,
    load_config,
    load_model,
    load_tokenizer,
    pick_gpu_device,
    pool_embeddings,
    tokenize_sequences,
    write_json,
)


def run_determinism(device_str: str) -> dict:
    cfg = load_config()
    ensure_offline_env()
    import torch

    if device_str == "auto":
        device = pick_gpu_device()
    else:
        if device_str.startswith("cuda") and not torch.cuda.is_available():
            sys.exit(f"[FM0] FATAL: device={device_str} but CUDA unavailable.")
        device = torch.device(device_str)

    tok = load_tokenizer()
    model = load_model(device=str(device))

    # Use a representative set of sequences (varied lengths, varied alphabet)
    sequences = [
        "ACGU",
        "ACGTACGTACGT",
        "GCCAUUACGGCCAAUUGGACCUUAGGCCAAUU",
        "A" * 50,
        "ACGU" * 100,                                  # 400 nt
    ]

    enc = tokenize_sequences(sequences, tok, max_length=cfg["model"]["max_position_embeddings"])
    enc = {k: v.to(device) for k, v in enc.items()}

    # Run forward TWICE with identical inputs (no randomness in eval mode)
    with torch.no_grad():
        out1 = model(**enc)
        out2 = model(**enc)

    h1 = out1.last_hidden_state
    h2 = out2.last_hidden_state

    # Bit-exact comparison
    bit_exact = bool(torch.equal(h1, h2))
    # Numeric max abs diff (should be exactly 0.0 for a frozen eval-mode model)
    max_abs_diff = float((h1 - h2).abs().max().item())
    mean_abs_diff = float((h1 - h2).abs().mean().item())

    # Per-pooling determinism check
    pooling_results = {}
    for mode in ["cls", "mean", "max"]:
        v1 = pool_embeddings(h1, enc["attention_mask"], mode=mode)
        v2 = pool_embeddings(h2, enc["attention_mask"], mode=mode)
        pooling_results[mode] = {
            "bit_exact": bool(torch.equal(v1, v2)),
            "max_abs_diff": float((v1 - v2).abs().max().item()),
            "shape": list(v1.shape),
        }

    # Also check a third run for extra confidence
    with torch.no_grad():
        out3 = model(**enc)
    h3 = out3.last_hidden_state
    three_run_consistent = bool(torch.equal(h1, h3) and torch.equal(h2, h3))

    report = {
        "task_id": "FM0-01",
        "acceptance": "embedding determinism",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "device": str(device),
        "num_sequences": len(sequences),
        "input_lengths_nt": [len(s) for s in sequences],
        "hidden_state_shape": list(h1.shape),
        "bit_exact_two_runs": bit_exact,
        "three_run_consistent": three_run_consistent,
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "pooling_results": pooling_results,
        "pass": (
            bit_exact
            and three_run_consistent
            and max_abs_diff == 0.0
            and all(p["bit_exact"] for p in pooling_results.values())
        ),
    }
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="auto")
    ap.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_DIR / "embedding_determinism_report.json"),
    )
    args = ap.parse_args()

    report = run_determinism(args.device)
    out = Path(args.output)
    write_json(out, report)

    print(f"[FM0-01] Embedding determinism report -> {out}")
    print(f"  device: {report['device']}")
    print(f"  hidden_state_shape: {report['hidden_state_shape']}")
    print(f"  bit_exact (2 runs): {report['bit_exact_two_runs']}")
    print(f"  3-run consistent:   {report['three_run_consistent']}")
    print(f"  max_abs_diff: {report['max_abs_diff']}")
    for mode, r in report["pooling_results"].items():
        print(f"  pool[{mode}]: bit_exact={r['bit_exact']}  shape={r['shape']}")
    print(f"  PASS: {report['pass']}")

    if not report["pass"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
