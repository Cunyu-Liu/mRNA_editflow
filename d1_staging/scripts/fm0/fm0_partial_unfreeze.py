#!/usr/bin/env python
"""FM0-01: Partial unfreeze arm of H7 ablation.

Unfreezes the last N encoder layers of UTR-LM (while keeping embeddings and
earlier layers frozen). This is the "partial_full_finetune" arm of contract
§H7 must_validate.

Verifies:
  1. Only the last N encoder layers' params are trainable.
  2. Embeddings and pooler (if present) remain frozen.
  3. Trainable param count matches expectation.
  4. Forward still works.

Acceptance (FM0-01): partial unfreeze.

Usage:
    python scripts/fm0/fm0_partial_unfreeze.py [--device cuda:5] \
        [--unfreeze-last-n 2] \
        [--output data/fm0/partial_unfreeze_report.json]

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


def _find_encoder_layer_param_names(model) -> list:
    """Identify encoder layer parameter name prefixes.

    For UtrLmModel, params are like 'encoder.layer.0.attention.self.query.weight'.
    Returns a sorted list of layer indices found.
    """
    layer_ids = set()
    for name, _ in model.named_parameters():
        # Look for the pattern 'encoder.layer.{N}.'
        if ".layer." in name:
            try:
                # extract integer after 'layer.'
                idx = int(name.split(".layer.")[1].split(".")[0])
                layer_ids.add(idx)
            except (ValueError, IndexError):
                pass
    return sorted(layer_ids)


def run_partial_unfreeze(device_str: str, unfreeze_last_n: int) -> dict:
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

    # First, set all params to frozen (model.eval() doesn't freeze requires_grad;
    # from_pretrained loads with requires_grad=True by default for the encoder).
    for p in model.parameters():
        p.requires_grad_(False)

    # Identify encoder layers
    layer_ids = _find_encoder_layer_param_names(model)
    if not layer_ids:
        sys.exit("[FM0] FATAL: could not locate encoder.layer.N parameters in model.")
    total_layers = len(layer_ids)
    unfreeze_ids = set(layer_ids[-unfreeze_last_n:])

    # Unfreeze last N layers
    trainable_param_names = []
    for name, p in model.named_parameters():
        if ".layer." in name:
            try:
                idx = int(name.split(".layer.")[1].split(".")[0])
                if idx in unfreeze_ids:
                    p.requires_grad_(True)
                    trainable_param_names.append(name)
            except (ValueError, IndexError):
                pass

    # Count
    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Sanity: embeddings still frozen?
    embedding_trainable = [
        n for n, p in model.named_parameters()
        if p.requires_grad and ("embeddings" in n.lower() or ".word_embeddings" in n.lower())
    ]
    pooler_trainable = [
        n for n, p in model.named_parameters()
        if p.requires_grad and "pooler" in n.lower()
    ]

    # Forward check
    seqs = ["ACGU" * 12, "GCCAUUACGGCCAAUUGGACCUUAGGCCAAUU"]
    enc = tokenize_sequences(seqs, tok, max_length=cfg["model"]["max_position_embeddings"])
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc)
    h = out.last_hidden_state

    # Compute expected trainable params (approximate: count params in last N layers)
    expected_trainable = 0
    for name, p in model.named_parameters():
        if ".layer." in name:
            try:
                idx = int(name.split(".layer.")[1].split(".")[0])
                if idx in unfreeze_ids:
                    expected_trainable += p.numel()
            except (ValueError, IndexError):
                pass

    report = {
        "task_id": "FM0-01",
        "acceptance": "partial unfreeze",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "device": str(device),
        "unfreeze_last_n_layers": unfreeze_last_n,
        "total_encoder_layers": total_layers,
        "unfrozen_layer_indices": sorted(unfreeze_ids),
        "frozen_layer_indices": sorted(set(layer_ids) - unfreeze_ids),
        "num_parameters_total": n_total,
        "num_parameters_trainable": n_trainable,
        "num_parameters_frozen": n_total - n_trainable,
        "trainable_fraction": round(n_trainable / n_total, 6),
        "expected_trainable": expected_trainable,
        "trainable_count_matches_expectation": n_trainable == expected_trainable,
        "embedding_trainable_count": len(embedding_trainable),
        "pooler_trainable_count": len(pooler_trainable),
        "num_trainable_param_groups": len(trainable_param_names),
        "trainable_param_name_samples": trainable_param_names[:8],
        "forward_check": {
            "input_shape": list(enc["input_ids"].shape),
            "hidden_state_shape": list(h.shape),
            "any_nan": bool(torch.isnan(h).any().item()),
            "any_inf": bool(torch.isinf(h).any().item()),
        },
        "pass": (
            n_trainable == expected_trainable
            and n_trainable > 0
            and n_trainable < n_total
            and len(embedding_trainable) == 0
            and len(pooler_trainable) == 0
            and not report_forward_nan(h)
        ),
    }
    return report


def report_forward_nan(h):
    import torch
    return bool(torch.isnan(h).any().item() or torch.isinf(h).any().item())


def main():
    cfg = load_config()
    default_n = cfg["adaptation"]["partial_unfreeze"]["config"]["unfreeze_last_n_layers"]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--unfreeze-last-n", type=int, default=default_n)
    ap.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_DIR / "partial_unfreeze_report.json"),
    )
    args = ap.parse_args()

    report = run_partial_unfreeze(args.device, args.unfreeze_last_n)
    out = Path(args.output)
    write_json(out, report)

    print(f"[FM0-01] Partial unfreeze report -> {out}")
    print(f"  device: {report['device']}")
    print(f"  unfrozen layers: {report['unfrozen_layer_indices']} (of {report['total_encoder_layers']})")
    print(f"  params total:     {report['num_parameters_total']:,}")
    print(f"  params trainable: {report['num_parameters_trainable']:,} "
          f"(expected {report['expected_trainable']:,})")
    print(f"  trainable fraction: {report['trainable_fraction']}")
    print(f"  embedding trainable: {report['embedding_trainable_count']} (must be 0)")
    print(f"  pooler trainable:    {report['pooler_trainable_count']} (must be 0)")
    print(f"  forward shape:    {report['forward_check']['hidden_state_shape']}")
    print(f"  PASS: {report['pass']}")

    if not report["pass"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
