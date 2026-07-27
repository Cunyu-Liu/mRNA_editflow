#!/usr/bin/env bash
# P0-06 reproduction smoke: proves the package runs end-to-end in a clean
# environment WITHOUT any downloaded data or GPU.
#
#   1. GRPO training smoke (synthetic sources + synthetic oracle) — exercises
#      the full MDP -> policy -> trajectory -> GRPO update -> validation path.
#   2. Hard motif policy smoke — the P0-05 legal-action filter detects an
#      upstream-AUG violation introduced by an edit.
set -euo pipefail

cd "$(dirname "$0")/.."

# Honour an explicitly provided interpreter ($PYTHON), otherwise prefer
# `python` (activated venv) and fall back to `python3` so the verbatim
# acceptance command `bash scripts/reproduce_smoke.sh` works in both.
PYTHON_BIN="${PYTHON:-$(command -v python || command -v python3)}"

echo "[smoke 1/2] P3-08 GRPO end-to-end (synthetic)"
"$PYTHON_BIN" scripts/run_p3_08.py --smoke-test --output-json /tmp/p3_08_smoke.json

echo "[smoke 2/2] hard motif policy (P0-05)"
"$PYTHON_BIN" - <<'PY'
from core.schema import MRNARecord
from core.motif_policy import hard_motif_violations, is_hard_legal

CDS = "AUGGCUGCUUAA"  # valid: AUG start, UAA stop, no in-frame stops
parent = MRNARecord(transcript_id="p", five_utr="ACGUACGU",
                    cds=CDS, three_utr="UGCU", metadata={})

# edit introduces a new upstream AUG in the 5'UTR
child_bad = MRNARecord(transcript_id="c", five_utr="ACGAUGGU",
                       cds=CDS, three_utr="UGCU", metadata={})
violations = hard_motif_violations(parent, child_bad)
assert "upstream_aug" in violations, f"expected upstream_aug, got {violations}"
assert not is_hard_legal(parent, child_bad)

# synonymous, motif-free edit stays legal
child_ok = MRNARecord(transcript_id="c2", five_utr="ACGUACGC",
                      cds=CDS, three_utr="UGCU", metadata={})
assert is_hard_legal(parent, child_ok), \
    f"benign edit flagged: {hard_motif_violations(parent, child_ok)}"

print("[smoke 2/2] motif policy OK: upstream_aug detected, benign edit legal")
PY

echo "[smoke] ALL OK"
