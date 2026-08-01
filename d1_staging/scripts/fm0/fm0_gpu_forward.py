#!/usr/bin/env python
"""FM0-01: GPU forward test for UTR-LM.

Loads UTR-LM on the chosen GPU device, runs forward passes on batches of
varying sizes, and verifies output shapes + dtype + device placement.

Contract: training_device = GPU_only. This script STOPS (sys.exit non-zero)
if CUDA is unavailable — we do NOT fall back to CPU (forward-only principle).

Acceptance (FM0-01): GPU forward test.

Usage:
    python scripts/fm0/fm0_gpu_forward.py [--device cuda:5] \
        [--output data/fm0/gpu_forward_report.json]

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
    summarize_model,
    tokenize_sequences,
    write_json,
)


def run_gpu_forward(device_str: str) -> dict:
    cfg = load_config()
    ensure_offline_env()
    import torch

    # Resolve device: if caller passes cuda:N, honor it; else auto-pick.
    if device_str == "auto":
        device = pick_gpu_device()
    else:
        if device_str.startswith("cuda") and not torch.cuda.is_available():
            sys.exit(f"[FM0] FATAL: device={device_str} but CUDA unavailable.")
        device = torch.device(device_str)

    tok = load_tokenizer()
    model = load_model(device=str(device))
    summary = summarize_model(model)

    # Test matrix: (batch_size, seq_len_nt)
    test_cases = [
        (1, 10),
        (1, 50),
        (8, 50),
        (4, 200),
        (2, 1000),       # close to max_position_embeddings - 4
    ]

    results = []
    for batch_size, seq_len in test_cases:
        seqs = ["ACGU" * (seq_len // 4) + "A" * (seq_len % 4)] * batch_size
        enc = tokenize_sequences(seqs, tok, max_length=cfg["model"]["max_position_embeddings"])
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
        h = out.last_hidden_state
        results.append({
            "batch_size": batch_size,
            "seq_len_nt": seq_len,
            "input_ids_shape": list(enc["input_ids"].shape),
            "last_hidden_state_shape": list(h.shape),
            "output_dtype": str(h.dtype),
            "output_device": str(h.device),
            "has_pooler_output": out.pooler_output is not None,
        })

    # Validate dtype/device consistency
    all_ok = all(
        r["output_dtype"] == "torch.float32"
        and r["output_device"] == str(device)
        and r["last_hidden_state_shape"][-1] == cfg["model"]["hidden_size"]
        and r["last_hidden_state_shape"][0] == r["batch_size"]
        for r in results
    )

    report = {
        "task_id": "FM0-01",
        "acceptance": "GPU forward test",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "device_count": torch.cuda.device_count(),
        "cuda_version": torch.version.cuda,
        "torch_version": torch.__version__,
        "model_summary": summary,
        "test_cases": results,
        "pass": all_ok,
    }
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="auto",
                    help="cuda:N or 'auto' (auto-pick GPU with most free mem).")
    ap.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_DIR / "gpu_forward_report.json"),
        help="Output JSON report path.",
    )
    args = ap.parse_args()

    report = run_gpu_forward(args.device)
    out = Path(args.output)
    write_json(out, report)

    print(f"[FM0-01] GPU forward report -> {out}")
    print(f"  device: {report['device']} ({report['device_name']})")
    print(f"  torch={report['torch_version']}  cuda={report['cuda_version']}")
    print(f"  test cases: {len(report['test_cases'])}")
    for r in report["test_cases"]:
        print(f"    B={r['batch_size']:>3d} L={r['seq_len_nt']:>5d}  "
              f"hidden={r['last_hidden_state_shape']}  {r['output_dtype']}  "
              f"on {r['output_device']}")
    print(f"  PASS: {report['pass']}")

    if not report["pass"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
