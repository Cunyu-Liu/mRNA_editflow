#!/usr/bin/env python
"""FM0-01: Input-length behavior test for UTR-LM.

Probes how the encoder behaves across the full range of input lengths:
  - very short (1 nt, 2 nt, 4 nt)
  - short (10 nt, 30 nt — typical minimal 5'UTR)
  - medium (50 nt, 100 nt, 200 nt — typical MPRA library length)
  - long (500 nt, 800 nt — long endogenous 5'UTRs)
  - max supported (1022 nt — UTR-LM training cap)
  - over-max (1100 nt, 2000 nt — must be truncated, NOT error)

Verifies:
  1. Lengths <= max_position_embeddings - 4 produce sensible hidden states.
  2. Over-max inputs are TRUNCATED to max_position_embeddings (no crash).
  3. Token count == input_nt_count + 2 (BOS + EOS) for unpadded sequences
     shorter than max_length.
  4. Hidden state seq dim matches token count.

Acceptance (FM0-01): input-length behavior.

Usage:
    python scripts/fm0/fm0_input_length_behavior.py [--device cuda:5] \
        [--output data/fm0/input_length_behavior_report.json]

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
    tokenize_sequences,
    write_json,
)


def _make_seq(n: int) -> str:
    """Build a deterministic ACGT sequence of length n (cycling)."""
    base = "ACGT"
    return (base * (n // 4 + 1))[:n]


def run_input_length(device_str: str) -> dict:
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

    max_pos = cfg["model"]["max_position_embeddings"]  # 1026
    # UTR-LM training cap is 1022 nt (per README) — sequences truncated to 30..1022 bp.
    training_max_nt = 1022

    test_lengths = [1, 2, 4, 10, 30, 50, 100, 200, 500, 800,
                    training_max_nt, training_max_nt + 50, 2000]

    results = []
    for n in test_lengths:
        seq = _make_seq(n)
        # padding=False so we observe the natural token count for this sequence
        enc = tokenize_sequences([seq], tok, max_length=max_pos, padding=False)
        ids = enc["input_ids"].tolist()[0]
        n_special = 2  # BOS + EOS (UTR-LM uses <cls> ... <eos>)
        n_content = len(ids) - n_special
        # For over-max input, tokenizer should have truncated to max_pos tokens.
        truncated = n + n_special > max_pos

        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
        h = out.last_hidden_state

        results.append({
            "input_nt_count": n,
            "token_count": len(ids),
            "content_tokens": n_content,
            "expected_content_tokens": min(n, max_pos - n_special),
            "truncated_by_tokenizer": truncated,
            "hidden_state_shape": list(h.shape),
            "hidden_seq_dim_matches_tokens": h.shape[1] == len(ids),
            "mean_abs_hidden": float(h.abs().mean().item()),
            "std_hidden": float(h.std().item()),
            "any_nan": bool(torch.isnan(h).any().item()),
            "any_inf": bool(torch.isinf(h).any().item()),
        })

    all_ok = all(
        not r["any_nan"]
        and not r["any_inf"]
        and r["hidden_seq_dim_matches_tokens"]
        and r["content_tokens"] == r["expected_content_tokens"]
        for r in results
    )

    report = {
        "task_id": "FM0-01",
        "acceptance": "input-length behavior",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "device": str(device),
        "max_position_embeddings": max_pos,
        "training_max_nt": training_max_nt,
        "note": (
            "UTR-LM was pretrained on 5'UTR sequences truncated to 30..1022 bp. "
            "Inputs shorter than 30 nt or longer than 1022 nt are out-of-distribution "
            "but must still produce a forward pass without crashing (truncation applies)."
        ),
        "test_lengths": test_lengths,
        "results": results,
        "pass": all_ok,
    }
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="auto")
    ap.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_DIR / "input_length_behavior_report.json"),
    )
    args = ap.parse_args()

    report = run_input_length(args.device)
    out = Path(args.output)
    write_json(out, report)

    print(f"[FM0-01] Input-length behavior report -> {out}")
    print(f"  device: {report['device']}  max_pos={report['max_position_embeddings']}")
    print(f"  {'nt':>6s}  {'toks':>5s}  {'trunc':>5s}  {'hidden_shape':>20s}  nan/inf")
    for r in report["results"]:
        print(f"  {r['input_nt_count']:>6d}  {r['token_count']:>5d}  "
              f"{str(r['truncated_by_tokenizer']):>5s}  {str(r['hidden_state_shape']):>20s}  "
              f"{r['any_nan']}/{r['any_inf']}")
    print(f"  PASS: {report['pass']}")

    if not report["pass"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
