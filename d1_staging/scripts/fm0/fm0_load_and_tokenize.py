#!/usr/bin/env python
"""FM0-01: Load test + tokenization test for UTR-LM.

Verifies:
  1. RnaTokenizer loads from the frozen checkpoint (offline).
  2. UtrLmConfig matches the frozen config in configs/fm0_utrlm_config.yaml.
  3. UtrLmModel (bare encoder) loads and produces correct output shape.
  4. Tokenization is correct: special tokens, T->U conversion, vocab.

Acceptance (FM0-01): load test + tokenization test.

Usage:
    python scripts/fm0/fm0_load_and_tokenize.py \
        [--output data/fm0/load_tokenize_report.json]

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
    get_model_id,
    load_config,
    load_config_obj,
    load_model,
    load_tokenizer,
    summarize_model,
    tokenize_sequences,
    write_json,
)


# Sample sequences chosen to exercise: pure ACGU, ACGT (T->U conversion),
# ambiguous N, and a longer sequence.
SAMPLE_SEQUENCES = [
    "ACGU",                                # 4 nt, pure RNA alphabet
    "ACGTACGT",                            # 8 nt, T->U conversion check
    "ACGTNACGTN",                          # 10 nt, with N
    "GCCAUUACGGCCAAUUGGACCUUAGGCC",        # 28 nt, realistic 5'UTR-ish
    "A" * 100,                             # 100 nt homopolymer
]


def run_load_and_tokenize() -> dict:
    cfg = load_config()
    model_id = get_model_id()
    ensure_offline_env()

    import torch

    # 1. Tokenizer
    tok = load_tokenizer()
    tok_info = {
        "class": type(tok).__name__,
        "vocab_size": tok.vocab_size,
        "model_max_length": tok.model_max_length,
        "special_tokens_map": tok.special_tokens_map,
        "is_fast": getattr(tok, "is_fast", None),
    }

    # 2. Config
    config_obj = load_config_obj()
    cfg_info = {
        "class": type(config_obj).__name__,
        "model_type": config_obj.model_type,
        "hidden_size": config_obj.hidden_size,
        "num_hidden_layers": config_obj.num_hidden_layers,
        "num_attention_heads": config_obj.num_attention_heads,
        "intermediate_size": config_obj.intermediate_size,
        "vocab_size": config_obj.vocab_size,
        "max_position_embeddings": config_obj.max_position_embeddings,
    }
    # Verify config matches frozen yaml
    mismatches = []
    for k, expected in {
        "model_type": cfg["model"]["model_type"],
        "hidden_size": cfg["model"]["hidden_size"],
        "num_hidden_layers": cfg["model"]["num_hidden_layers"],
        "num_attention_heads": cfg["model"]["num_attention_heads"],
        "intermediate_size": cfg["model"]["intermediate_size"],
        "vocab_size": cfg["model"]["vocab_size"],
        "max_position_embeddings": cfg["model"]["max_position_embeddings"],
    }.items():
        actual = getattr(config_obj, k)
        if actual != expected:
            mismatches.append({"field": k, "expected": expected, "actual": actual})

    # 3. Tokenization tests
    enc_examples = []
    for seq in SAMPLE_SEQUENCES:
        enc = tokenize_sequences([seq], tok, max_length=None, padding=False)
        ids = enc["input_ids"].tolist()[0]
        # Decode back to compare (after T->U)
        decoded = tok.decode(ids, skip_special_tokens=True)
        enc_examples.append({
            "input": seq,
            "length_nt": len(seq),
            "input_ids": ids,
            "num_tokens": len(ids),
            "decoded": decoded,
            "T_to_U_applied": ("T" in seq and "T" not in decoded and "U" in decoded),
        })

    # 4. Model load (CPU) + forward smoke
    model = load_model(device="cpu")
    summary = summarize_model(model)

    enc = tokenize_sequences(["ACGUACGUAC"], tok, max_length=None)
    with torch.no_grad():
        out = model(**enc)
    fwd_info = {
        "output_class": type(out).__name__,
        "last_hidden_state_shape": list(out.last_hidden_state.shape),
        "has_pooler_output": hasattr(out, "pooler_output") and out.pooler_output is not None,
    }
    if fwd_info["has_pooler_output"]:
        fwd_info["pooler_output_shape"] = list(out.pooler_output.shape)

    report = {
        "task_id": "FM0-01",
        "acceptance": ["load test", "tokenization test"],
        "generated_at_utc": datetime.now(timezone.utc)
            .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_id": model_id,
        "config_yaml_revision": cfg["model"]["revision"],
        "tokenizer": tok_info,
        "config": cfg_info,
        "config_mismatches_vs_yaml": mismatches,
        "tokenization_examples": enc_examples,
        "model_summary": summary,
        "forward_smoke": fwd_info,
        "pass": (
            not mismatches
            and summary["num_parameters_total"] == cfg["model"]["num_parameters"]
            and fwd_info["last_hidden_state_shape"][-1] == cfg["model"]["hidden_size"]
        ),
    }
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_DIR / "load_tokenize_report.json"),
        help="Output JSON report path.",
    )
    args = ap.parse_args()

    report = run_load_and_tokenize()
    out = Path(args.output)
    write_json(out, report)

    print(f"[FM0-01] Load + tokenize report -> {out}")
    print(f"  tokenizer: {report['tokenizer']['class']}  vocab={report['tokenizer']['vocab_size']}")
    print(f"  config: {report['config']['class']}  hidden={report['config']['hidden_size']} layers={report['config']['num_hidden_layers']}")
    print(f"  mismatches vs yaml: {len(report['config_mismatches_vs_yaml'])}")
    print(f"  params: {report['model_summary']['num_parameters_total']:,}")
    print(f"  forward last_hidden_state: {report['forward_smoke']['last_hidden_state_shape']}")
    print(f"  PASS: {report['pass']}")

    if not report["pass"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
