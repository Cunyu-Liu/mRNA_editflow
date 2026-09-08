#!/usr/bin/env python3
"""Baseline Task 6.5.8 (P1-4): UTailoR / UTR-Insight R3 attribution audit.

Candidates for the MRL 2025 SOTA row (bound to Table 1 four-axis positioning;
condition: audit pass AND availability).

Findings (2026-09-09, batch 66):
- UTailoR: no public code/weights located (web survey) -> NOT AVAILABLE.
- UTR-Insight (Pan et al., BMC Genomics 2025:26:107, repo pansaichao/UTR_Insight,
  weights Model/utr_insight/model_epoch199.pkl shipped): training notebook
  loads `4.1_train_data_GSM3130435_egfp_unmod_1_BiologyFeatures.csv` —
  GSM3130435 (egfp_unmod_1) belongs to GSE114002, the MRL benchmark source
  study, with NO exclusion of our protected VALIDATION/TEST records (3-block
  pigeonhole cannot be satisfied; the authors' own train/test split is
  independent of ours).

R3 verdict per the UTR-STCNet precedent (commit 08ff6c2f, all three arms
excluded as same-study sources without protected-record exclusion):
UTR-Insight = INVALID for the MRL row; UTailoR = unavailable. The MRL 2025
SOTA named row therefore has NO claimable candidate — registered as-is.
"""
from __future__ import annotations

import json
from pathlib import Path

MNT = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2")
OUT = MNT / "experiments/analysis_utr_insight_r3_audit_20260909"
REPO = MNT / "external_model_assets/utr_insight"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    # verify the training-source strings inside the shipped notebook
    nb = json.loads((REPO / "1.UTR_Insight_train.ipynb").read_text())
    hits = []
    for cell in nb["cells"]:
        src = "".join(cell["source"])
        for token in ("GSM3130435", "egfp_unmod_1", "GSE114002", "designed_library", "GSM3130443", "GSM4084997"):
            if token in src:
                hits.append(token)
    tokens = sorted(set(hits))
    weights = sorted(str(p.relative_to(REPO)) for p in REPO.glob("Model/**/*.pkl"))

    payload = {
        "schema_version": "route_a_v3_utr_insight_r3_audit_v1",
        "candidates": {
            "UTailoR": {"verdict": "NOT_AVAILABLE", "reason": "no public code/weights located (web survey 2026-09-09)"},
            "UTR-Insight": {
                "paper": "BMC Genomics 2025:26:107 (Pan et al.)",
                "repo": "pansaichao/UTR_Insight",
                "weights_shipped": weights,
                "training_source_tokens_in_repo": tokens,
                "training_source": "GSM3130435 egfp_unmod_1 (280K random 5'UTR library)",
                "attribution": "GSM3130435 is a sample of GSE114002 — the MRL benchmark source study",
                "protected_record_exclusion": "none (authors' own train/test split; our 3-block pigeonhole VALIDATION/TEST exclusion not applied)",
                "r3_verdict": "INVALID",
                "precedent": "UTR-STCNet (commit 08ff6c2f): same-study training sources without protected-record exclusion are excluded from the leaderboard",
            },
        },
        "task_6_5_8_verdict": "CLOSED — no claimable MRL 2025 SOTA named row (UTR-Insight INVALID, UTailoR unavailable); MRL external rows remain frozen-Optimus 0.3132 / Route A full-FT ensemble 0.3158 / V9-1a ensemble 0.3217",
    }
    (OUT / "utr_insight_r3_audit.json").write_text(json.dumps(payload, indent=1))
    print(json.dumps(payload["candidates"]["UTR-Insight"], indent=1))
    print(f"task 6.5.8: {payload['task_6_5_8_verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
