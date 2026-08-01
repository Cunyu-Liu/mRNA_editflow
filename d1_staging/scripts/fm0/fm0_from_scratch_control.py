#!/usr/bin/env python
"""FM0-01: From-scratch control arm of H7 ablation.

Builds a random-initialized UtrLmModel with the SAME architecture as the
pretrained checkpoint but WITHOUT loading pretrained weights. This is the
"small_from_scratch_control" arm of contract §H7 must_validate, used to
isolate the value of the pretrained initialization (vs architecture alone).

Verifies:
  1. Architecture matches the frozen config (same num_layers, hidden, etc.).
  2. No checkpoint weights are loaded (param stats differ from pretrained).
  3. Forward pass works.
  4. Output is deterministic given a fixed seed.

Acceptance (FM0-01): from-scratch control.

Usage:
    python scripts/fm0/fm0_from_scratch_control.py [--device cuda:5] \
        [--seed 20260801] \
        [--output data/fm0/from_scratch_control_report.json]

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
    load_config_obj,
    load_model,
    load_model_from_scratch,
    load_tokenizer,
    pick_gpu_device,
    summarize_model,
    tokenize_sequences,
    write_json,
)


def run_from_scratch(device_str: str, seed: int) -> dict:
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
    config_obj = load_config_obj()

    # 1. Random-init model
    model_scratch = load_model_from_scratch(seed=seed).to(device)
    model_scratch.eval()

    # 2. Pretrained model for comparison (same arch, loaded weights)
    model_pretrained = load_model(device=str(device))
    model_pretrained.eval()

    # 3. Param stats comparison
    s_scratch = summarize_model(model_scratch)
    s_pretrained = summarize_model(model_pretrained)

    # 4. Forward determinism for scratch model (same seed -> same weights)
    enc = tokenize_sequences(
        ["ACGUACGUAC", "GCCAUUACGGCCAAUUGGACCUUAGGCCAAUU"],
        tok,
        max_length=cfg["model"]["max_position_embeddings"],
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out1 = model_scratch(**enc)
        out2 = model_scratch(**enc)
    h1 = out1.last_hidden_state
    h2 = out2.last_hidden_state
    bit_exact = bool(torch.equal(h1, h2))

    # 5. Compare scratch vs pretrained outputs (should differ — proves weights differ)
    with torch.no_grad():
        out_pre = model_pretrained(**enc)
    h_pre = out_pre.last_hidden_state
    max_diff_vs_pretrained = float((h1 - h_pre).abs().max().item())

    # 6. Weight stats comparison
    scratch_flat = torch.cat([p.flatten() for p in model_scratch.parameters()])
    pretrained_flat = torch.cat([p.flatten() for p in model_pretrained.parameters()])
    scratch_param_mean = float(scratch_flat.mean().item())
    pretrained_param_mean = float(pretrained_flat.mean().item())
    # Max abs element-wise diff: robust check that weights truly differ
    # (means can coincide by chance; max-diff cannot for distinct init schemes).
    max_abs_weight_diff = float((scratch_flat - pretrained_flat).abs().max().item())

    report = {
        "task_id": "FM0-01",
        "acceptance": "from-scratch control",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "device": str(device),
        "random_seed": seed,
        "architecture": {
            "class": type(config_obj).__name__,
            "model_type": config_obj.model_type,
            "num_hidden_layers": config_obj.num_hidden_layers,
            "hidden_size": config_obj.hidden_size,
            "num_attention_heads": config_obj.num_attention_heads,
            "intermediate_size": config_obj.intermediate_size,
            "vocab_size": config_obj.vocab_size,
            "max_position_embeddings": config_obj.max_position_embeddings,
        },
        "scratch_model": s_scratch,
        "pretrained_model": s_pretrained,
        "architecture_matches": (
            s_scratch["num_parameters_total"] == s_pretrained["num_parameters_total"]
            and config_obj.num_hidden_layers == cfg["model"]["num_hidden_layers"]
            and config_obj.hidden_size == cfg["model"]["hidden_size"]
            and config_obj.vocab_size == cfg["model"]["vocab_size"]
        ),
        "weights_differ": max_abs_weight_diff > 1e-6,
        "scratch_param_mean": scratch_param_mean,
        "pretrained_param_mean": pretrained_param_mean,
        "max_abs_weight_diff_vs_pretrained": max_abs_weight_diff,
        "forward_check": {
            "input_shape": list(enc["input_ids"].shape),
            "hidden_state_shape": list(h1.shape),
            "deterministic_two_runs": bit_exact,
            "max_abs_diff_vs_pretrained": max_diff_vs_pretrained,
            "any_nan": bool(torch.isnan(h1).any().item()),
            "any_inf": bool(torch.isinf(h1).any().item()),
        },
        "pass": (
            s_scratch["num_parameters_total"] == s_pretrained["num_parameters_total"]
            and abs(scratch_param_mean - pretrained_param_mean) > 1e-6
            and bit_exact
            and max_diff_vs_pretrained > 1e-3
            and not bool(torch.isnan(h1).any().item())
            and not bool(torch.isinf(h1).any().item())
        ),
    }
    return report


def main():
    cfg = load_config()
    default_seed = cfg["adaptation"]["from_scratch"]["config"]["random_seed"]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=default_seed)
    ap.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_DIR / "from_scratch_control_report.json"),
    )
    args = ap.parse_args()

    report = run_from_scratch(args.device, args.seed)
    out = Path(args.output)
    write_json(out, report)

    print(f"[FM0-01] From-scratch control report -> {out}")
    print(f"  device: {report['device']}  seed: {report['random_seed']}")
    print(f"  arch matches pretrained:  {report['architecture_matches']}")
    print(f"  params (scratch):         {report['scratch_model']['num_parameters_total']:,}")
    print(f"  params (pretrained):      {report['pretrained_model']['num_parameters_total']:,}")
    print(f"  weights differ:           {report['weights_differ']}  "
          f"(scratch mean={report['scratch_param_mean']:.6f}, "
          f"pretrained mean={report['pretrained_param_mean']:.6f})")
    print(f"  deterministic (2 runs):   {report['forward_check']['deterministic_two_runs']}")
    print(f"  max diff vs pretrained:   {report['forward_check']['max_abs_diff_vs_pretrained']:.6f}")
    print(f"  PASS: {report['pass']}")

    if not report["pass"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
