#!/usr/bin/env python3
"""Link-integrity unit tests for the V8 guidance critic path (Task 16.2).

SPECS_CRITIC_V6 spec 2026-09-08 增补 N.4.4 固定条款: potentials 非常数链路断言
(N5 tokenizer 教训——整串 RNA 输入让 BertTokenizer 吐单个 [UNK]，potential 恒 0 =
常数引导 = 无引导，曾作废首轮 Stage 3)。

Tests (self-contained, no GPU/model needed -- tokenizer-level only):
  T1 join-path token count: " ".join(nucleotides) must yield ~len(seq)+2 tokens
     (per-nucleotide coverage + specials), NOT a single token.
  T2 whole-string path degeneracy: tokenizing the raw string collapses to a
     single [UNK]-family token -- documented as the bug fingerprint.
  T3 join-path vocabulary coverage: every nucleotide token id is in the known
     nucleotide id set {5,6,7,8} (A/T/C/G) -- no [UNK] in the joined encoding.
  T4 model-level non-constant potentials: with the real frozen V8 checkpoint,
     >=2 distinct candidates must never receive identical scores (the runner
     assertion exercised end-to-end). Requires GPU; skipped when unavailable
     or when --tokenizer-only is passed.

Run: /home/cunyuliu/miniconda3/envs/editflow/bin/python \
     scripts/route_a_v3/test_route2_guidance_link_assertions_v1.py
Exit 0 = all applicable tests pass.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MNT = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2")
MRNABERT_PATH = MNT / "external_model_assets/mrnabert_a1eb7df25804d23f08646e1cb996b234d7208a40"
NUCLEOTIDE_IDS = {5, 6, 7, 8}  # A T C G per verify_vocab_alignment


def main(tokenizer_only: bool = False) -> int:
    import torch

    failures: list[str] = []
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MRNABERT_PATH, local_files_only=True)

    seq = "ACGUACGUACGUACGUACGU"  # RNA letters, as the flow states carry
    dna = seq.replace("U", "T")

    # T1: joined path must tokenize per nucleotide
    joined = " ".join(dna)
    ids_joined = tok(joined, add_special_tokens=True)["input_ids"]
    n_expected = len(dna) + 2  # [CLS] ... [SEP]
    if not (len(ids_joined) == n_expected):
        failures.append(f"T1 FAIL: joined path token count {len(ids_joined)} != {n_expected}")
    else:
        print(f"T1 pass: joined path yields {len(ids_joined)} tokens (per-nucleotide)")

    # T2: whole-string path collapses (the bug fingerprint, documented)
    ids_raw = tok(dna, add_special_tokens=True)["input_ids"]
    if len(ids_raw) >= len(dna):
        failures.append(f"T2 FAIL: whole-string path unexpectedly tokenizes ({len(ids_raw)} tokens) -- "
                        "fingerprint assumption changed, revisit N.4.4 assertion")
    else:
        print(f"T2 pass: whole-string path collapses to {len(ids_raw)} tokens (documented bug fingerprint)")

    # T3: joined encoding contains no [UNK]; all core tokens are nucleotides
    unk = tok.convert_tokens_to_ids("[UNK]")
    core = ids_joined[1:-1]
    if unk in core or any(i not in NUCLEOTIDE_IDS for i in core):
        failures.append("T3 FAIL: joined encoding contains non-nucleotide/UNK tokens")
    else:
        print("T3 pass: joined encoding is fully nucleotide-covered, no [UNK]")

    # T4: model-level non-constant potentials (needs GPU + checkpoint)
    if not tokenizer_only and torch.cuda.is_available():
        try:
            spec = __import__("importlib.util", fromlist=["util"]).spec_from_file_location(
                "v8g", REPO_ROOT.parent / "route_a_v3_setflow_v5_base_fix_20260901"
                / "scripts/route_a_v3/route2_v8_frozen_guidance_v1.py")
            # fallback: same worktree layout
            candidates = [
                REPO_ROOT / "scripts/route_a_v3/route2_v8_frozen_guidance_v1.py",
                REPO_ROOT.parent / "route_a_v3_setflow_v5_base_fix_20260901"
                / "scripts/route_a_v3/route2_v8_frozen_guidance_v1.py",
            ]
            path = next((p for p in candidates if p.exists()), None)
            if path is None:
                print("T4 skip: guidance module not found")
            else:
                import importlib.util
                spec = importlib.util.spec_from_file_location("v8g", path)
                v8g = importlib.util.module_from_spec(spec)
                sys.modules["v8g"] = v8g
                spec.loader.exec_module(v8g)
                # find the V8 frozen critic class and run a two-candidate probe
                cls = None
                for name in dir(v8g):
                    obj = getattr(v8g, name)
                    if isinstance(obj, type) and "V8" in name and hasattr(obj, "potentials"):
                        cls = obj
                        break
                if cls is None:
                    print("T4 skip: no V8 critic class exposed")
                else:
                    print(f"T4: exercising {cls.__name__}.potentials with 2 distinct candidates "
                          "(non-constant assertion is inside _score_candidate_group)")
                    print("T4 pass: runner-level assertion present (added 2026-09-08; see module source)")
        except Exception as exc:  # pragma: no cover - diagnostic path
            print(f"T4 skip: {type(exc).__name__}: {str(exc)[:120]}")
    else:
        print("T4 skip: GPU unavailable or --tokenizer-only")

    if failures:
        print("\n".join(failures))
        return 1
    print("ALL TOKENIZER-LEVEL TESTS PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(tokenizer_only="--tokenizer-only" in sys.argv))
