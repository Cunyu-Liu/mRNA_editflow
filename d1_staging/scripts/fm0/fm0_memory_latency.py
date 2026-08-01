#!/usr/bin/env python
"""FM0-01: Memory + latency profile for UTR-LM.

Profiles peak GPU memory and per-batch latency across a grid of
(batch_size, seq_len) values, in eval mode. This informs:
  - which batch sizes are feasible for frozen-cache precomputation
  - which device to use for LoRA / partial-unfreeze training
  - whether real-time inference is feasible for downstream search

Acceptance (FM0-01): memory/latency profile.

Usage:
    python scripts/fm0/fm0_memory_latency.py [--device cuda:5] \
        [--output data/fm0/memory_latency_report.json]

Contract: utr_editflow_contract_v2 (FROZEN)
Task: FM0-01
"""

import argparse
import os
import sys
import time
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
    base = "ACGT"
    return (base * (n // 4 + 1))[:n]


def run_memory_latency(device_str: str, warmup: int = 3, iters: int = 10) -> dict:
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

    # Pre-tokenize all sequences we'll reuse (no padding within a batch,
    # but each test case uses its own batch).
    # Grid: realistic MPRA-style sizes + a few stress cases.
    grid = [
        (1, 50),
        (1, 500),
        (8, 50),
        (8, 500),
        (32, 50),
        (32, 500),
        (64, 100),
        (128, 100),
        (64, 1000),       # long-seq stress
    ]

    results = []
    for batch_size, seq_len in grid:
        seqs = [_make_seq(seq_len)] * batch_size
        enc = tokenize_sequences(seqs, tok,
                                 max_length=cfg["model"]["max_position_embeddings"])
        enc = {k: v.to(device) for k, v in enc.items()}

        # Warmup
        for _ in range(warmup):
            with torch.no_grad():
                _ = model(**enc)
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)

        # Timed iters
        t0 = time.perf_counter()
        for _ in range(iters):
            with torch.no_grad():
                out = model(**enc)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        total_s = t1 - t0
        per_batch_ms = (total_s / iters) * 1000.0
        per_seq_ms = per_batch_ms / batch_size
        peak_mem_mb = (
            torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            if device.type == "cuda" else None
        )
        peak_mem_reserved_mb = (
            torch.cuda.max_memory_reserved(device) / (1024 ** 2)
            if device.type == "cuda" else None
        )

        results.append({
            "batch_size": batch_size,
            "seq_len_nt": seq_len,
            "input_ids_shape": list(enc["input_ids"].shape),
            "iters": iters,
            "warmup": warmup,
            "total_time_s": round(total_s, 6),
            "per_batch_ms": round(per_batch_ms, 4),
            "per_sequence_ms": round(per_seq_ms, 4),
            "throughput_seqs_per_s": round(batch_size / (per_batch_ms / 1000.0), 2),
            "peak_allocated_mb": round(peak_mem_mb, 2) if peak_mem_mb is not None else None,
            "peak_reserved_mb": round(peak_mem_reserved_mb, 2) if peak_mem_reserved_mb is not None else None,
            "output_last_hidden_shape": list(out.last_hidden_state.shape),
        })

    report = {
        "task_id": "FM0-01",
        "acceptance": "memory/latency profile",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "model_dtype": str(next(model.parameters()).dtype),
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "grid": [list(g) for g in grid],
        "results": results,
        "pass": all(r["per_batch_ms"] > 0 for r in results),
    }
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_DIR / "memory_latency_report.json"),
    )
    args = ap.parse_args()

    report = run_memory_latency(args.device, warmup=args.warmup, iters=args.iters)
    out = Path(args.output)
    write_json(out, report)

    print(f"[FM0-01] Memory/latency report -> {out}")
    print(f"  device: {report['device']} ({report['device_name']})")
    print(f"  {'B':>4s} {'L':>5s}  {'per_batch_ms':>12s} {'per_seq_ms':>11s} "
          f"{'toks/s':>10s} {'alloc_MB':>10s} {'res_MB':>10s}")
    for r in report["results"]:
        print(f"  {r['batch_size']:>4d} {r['seq_len_nt']:>5d}  "
              f"{r['per_batch_ms']:>12.3f} {r['per_sequence_ms']:>11.3f} "
              f"{r['throughput_seqs_per_s']:>10.1f} "
              f"{str(r['peak_allocated_mb']):>10s} "
              f"{str(r['peak_reserved_mb']):>10s}")
    print(f"  PASS: {report['pass']}")

    if not report["pass"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
