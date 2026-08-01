#!/usr/bin/env python
"""FM0-01: LoRA (Low-Rank Adaptation) arm of H7 ablation.

Wraps the frozen UTR-LM encoder with PEFT LoRA adapters on attention Q/V.
Verifies:
  1. Only LoRA adapter params are trainable (base encoder fully frozen).
  2. Forward pass still works with adapters active.
  3. Adapter output differs from frozen-base output (adapters are not identity).
  4. Trainable param count matches expectation (2 x r x (d_model + d_head*H)).

This is the "adapter_LoRA" arm of contract §H7 must_validate.

Acceptance (FM0-01): LoRA.

Usage:
    python scripts/fm0/fm0_lora.py [--device cuda:5] \
        [--output data/fm0/lora_report.json]

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


def run_lora(device_str: str) -> dict:
    cfg = load_config()
    ensure_offline_env()
    import torch

    try:
        import peft
    except ImportError:
        sys.exit("[FM0] FATAL: peft not installed. Run: pip install peft==0.20.0")

    if device_str == "auto":
        device = pick_gpu_device()
    else:
        if device_str.startswith("cuda") and not torch.cuda.is_available():
            sys.exit(f"[FM0] FATAL: device={device_str} but CUDA unavailable.")
        device = torch.device(device_str)

    tok = load_tokenizer()

    # 1. Load frozen base model
    base_model = load_model(device=str(device))
    base_params_total = sum(p.numel() for p in base_model.parameters())
    base_params_trainable_before = sum(
        p.numel() for p in base_model.parameters() if p.requires_grad
    )

    # 2. Apply PEFT LoRA
    lora_cfg = cfg["adaptation"]["lora"]["config"]
    peft_config = peft.LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
        target_modules=lora_cfg["target_modules"],
        task_type=getattr(peft.TaskType, lora_cfg["task_type"]),
    )
    peft_model = peft.get_peft_model(base_model, peft_config)

    # 3. Verify trainable params
    trainable_after = 0
    total_after = 0
    trainable_param_names = []
    for name, p in peft_model.named_parameters():
        total_after += p.numel()
        if p.requires_grad:
            trainable_after += p.numel()
            trainable_param_names.append(name)

    # Sanity: only lora params should be trainable
    non_lora_trainable = [
        n for n in trainable_param_names
        if "lora_" not in n.lower()
    ]

    # 4. Forward pass with adapters
    seqs = ["ACGU" * 12, "GCCAUUACGGCCAAUUGGACCUUAGGCCAAUU"]
    enc = tokenize_sequences(seqs, tok, max_length=cfg["model"]["max_position_embeddings"])
    enc = {k: v.to(device) for k, v in enc.items()}

    # Adapter forward (eval mode for stable comparison)
    peft_model.eval()
    with torch.no_grad():
        out_adapter_init = peft_model(**enc)
    h_adapter_init = out_adapter_init.last_hidden_state

    # 5. Base frozen forward (without adapters) for comparison
    with torch.no_grad():
        out_base = base_model(**enc)
    h_base = out_base.last_hidden_state

    # Fresh LoRA adapters have lora_B zero-initialized (PEFT convention) so the
    # adapter delta is exactly zero at init -> output equals base. This is BY
    # DESIGN (model starts at pretrained behavior). Verify this holds:
    max_diff_init = float((h_adapter_init - h_base).abs().max().item())

    # 5b. Prove the adapter path is actually wired: perturb lora_B to non-zero
    # and verify the output changes away from base. This confirms LoRA math is
    # hooked up correctly (not just registered as no-op modules).
    with torch.no_grad():
        for n, p in peft_model.named_parameters():
            if "lora_B" in n and p.requires_grad:
                p.add_(0.1)  # add a small non-zero perturbation
    with torch.no_grad():
        out_adapter_perturbed = peft_model(**enc)
    h_adapter_perturbed = out_adapter_perturbed.last_hidden_state
    max_diff_after_perturb = float(
        (h_adapter_perturbed - h_base).abs().max().item()
    )

    # 5c. Verify lora_A is non-zero (Kaiming init) and lora_B was zero at init
    lora_A_nonzero_count = 0
    lora_B_init_zero_count = 0
    for n, p in peft_model.named_parameters():
        if "lora_A" in n and p.requires_grad:
            if float(p.abs().sum().item()) > 0:
                lora_A_nonzero_count += 1
        # Note: lora_B is now perturbed, so we can't check init here; we verified
        # via max_diff_init == 0 above that lora_B was zero at init.
    lora_A_total = sum(1 for n, p in peft_model.named_parameters()
                       if "lora_A" in n and p.requires_grad)
    lora_B_total = sum(1 for n, p in peft_model.named_parameters()
                       if "lora_B" in n and p.requires_grad)

    # 6. Expected trainable param count:
    # For each target module (query, value): 2 matrices (A: r x in, B: out x r)
    # UTR-LM: hidden=128, no per-head separate Q/V projection (BERT-style fused QKV)
    # so per layer per module: lora_A (r x 128) + lora_B (128 x r) = 2 * r * 128
    # 6 layers x 2 modules x 2 * r * 128 = 6 * 2 * 2 * 8 * 128 = 24576
    H = cfg["model"]["hidden_size"]
    L = cfg["model"]["num_hidden_layers"]
    expected_trainable = L * len(lora_cfg["target_modules"]) * 2 * lora_cfg["r"] * H

    report = {
        "task_id": "FM0-01",
        "acceptance": "LoRA",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "device": str(device),
        "peft_version": peft.__version__,
        "lora_config": lora_cfg,
        "base_model": {
            "num_parameters_total": base_params_total,
            "num_parameters_trainable_before_peft": base_params_trainable_before,
        },
        "after_peft": {
            "num_parameters_total": total_after,
            "num_parameters_trainable": trainable_after,
            "num_parameters_frozen": total_after - trainable_after,
            "trainable_fraction": round(trainable_after / total_after, 6),
            "expected_trainable": expected_trainable,
            "trainable_count_matches_expectation": trainable_after == expected_trainable,
        },
        "non_lora_trainable_params": non_lora_trainable,
        "num_trainable_param_groups": len(trainable_param_names),
        "trainable_param_name_samples": trainable_param_names[:8],
        "forward_check": {
            "input_shape": list(enc["input_ids"].shape),
            "adapter_hidden_shape": list(h_adapter_init.shape),
            "base_hidden_shape": list(h_base.shape),
            "max_abs_diff_at_init_vs_base": max_diff_init,
            "max_abs_diff_after_loraB_perturb_vs_base": max_diff_after_perturb,
            "lora_A_nonzero_count": lora_A_nonzero_count,
            "lora_A_total": lora_A_total,
            "lora_B_total": lora_B_total,
            "init_is_identity": max_diff_init == 0.0,
            "perturb_perturbs_output": max_diff_after_perturb > 0.0,
        },
        "pass": (
            # Base model loads with all params trainable (normal for from_pretrained)
            base_params_trainable_before == base_params_total
            and len(non_lora_trainable) == 0   # after PEFT, only lora_ params trainable
            and trainable_after > 0
            and trainable_after == expected_trainable
            and lora_A_nonzero_count == lora_A_total  # all lora_A non-zero (Kaiming)
            and max_diff_init == 0.0                   # fresh adapter = identity (lora_B zero)
            and max_diff_after_perturb > 0.0           # perturbing lora_B changes output
            and h_adapter_init.shape == h_base.shape
        ),
    }
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="auto")
    ap.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_DIR / "lora_report.json"),
    )
    args = ap.parse_args()

    report = run_lora(args.device)
    out = Path(args.output)
    write_json(out, report)

    print(f"[FM0-01] LoRA report -> {out}")
    print(f"  device: {report['device']}  peft={report['peft_version']}")
    print(f"  base params total:    {report['base_model']['num_parameters_total']:,}")
    print(f"  base trainable before PEFT: {report['base_model']['num_parameters_trainable_before_peft']}")
    print(f"  after PEFT trainable: {report['after_peft']['num_parameters_trainable']:,} "
          f"(expected {report['after_peft']['expected_trainable']:,})")
    print(f"  trainable fraction:   {report['after_peft']['trainable_fraction']}")
    print(f"  non-lora trainable:   {len(report['non_lora_trainable_params'])} (must be 0)")
    print(f"  lora_A nonzero:       {report['forward_check']['lora_A_nonzero_count']}/{report['forward_check']['lora_A_total']}")
    print(f"  init max_diff vs base:    {report['forward_check']['max_abs_diff_at_init_vs_base']:.6f} (must be 0 — lora_B zero init)")
    print(f"  after perturb max_diff:   {report['forward_check']['max_abs_diff_after_loraB_perturb_vs_base']:.6f} (must be >0 — adapter wired)")
    print(f"  PASS: {report['pass']}")

    if not report["pass"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
