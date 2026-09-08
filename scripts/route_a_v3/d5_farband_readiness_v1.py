#!/usr/bin/env python3
"""Critic spec Task 15.6: D5 far-band (40% = 0.27) data-readiness assessment.

D5 amendment v1 keeps 0.27 as the far target (no deadline); 15.6 asks for a
data-readiness report: power re-estimation under ~10x variant-level
supervision + inventory of candidate data sources.

Power math (same Fisher-z conservative basis as the Task 12.2 statement):
  dz = atanh(rho_new) - atanh(rho_ref); n(80%, two-sided 0.05) = 2/(dz/2.80)^2 + 3
Key distinctions:
  - VERIFICATION power: records needed in the validation pool to adjudicate a
    given jump (evaluation-side).
  - LEARNING budget: training variants needed to reach a level (unknown
    scaling law; only empirical anchors on file).

Empirical anchors (all on file): MPRAU train variants ~6.0K pairs (ENCSR854RUF
DEVELOPMENT), validation 2,008 pairs; s_mprau_in 0.1351 = best at this scale;
D5 amendment fact 2: CI width is noise-dominated (more seeds do not shrink it).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

MNT = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2")
OUT = MNT / "experiments/analysis_d5_farband_readiness_20260909"

Z_STAR = 2.80  # 80% power, two-sided 0.05
REF = 0.1351   # s_mprau_in 5-seed ensemble (current best, CI crossing zero)
CEIL = 0.683   # MPRAU ceiling
FAR = 0.27     # 40% completion


def n_for(rho_new: float, rho_ref: float) -> int:
    dz = math.atanh(rho_new) - math.atanh(rho_ref)
    if abs(dz) < 1e-9:
        return -1
    return int(math.ceil(2.0 / (dz / Z_STAR) ** 2 + 3))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    # verification power for candidate jumps
    jumps = {
        "D5 near band: 0.1351 -> 0.17 (+0.035)": n_for(0.17, REF),
        "0.1351 -> 0.20 (29% completion)": n_for(0.20, REF),
        "D5 far band: 0.1351 -> 0.27 (40%)": n_for(FAR, REF),
        "0.20 -> 0.27 (far-band increment after mid-scale data)": n_for(FAR, 0.20),
    }
    current_n = 2008

    sources = {
        "ENCSR854RUF (MPRAU benchmark, same assay)": {
            "scale": "~6.0K variant pairs DEVELOPMENT + 2,008 VALIDATION",
            "status": "fully consumed — s_mprau_in 0.1351 is the empirically observed ceiling at this scale (5/5 seeds > V5; CI noise-dominated)",
        },
        "CMS array (ENCSR854RUF-same-lineage, 85,475 rows)": {
            "scale": "~74.8% variant coverage of MPRAU space (7,284/9,740)",
            "status": "FAILED as prior injection (3 arms 0.033/0.039/-0.007); never evaluated as plain training augmentation — the only on-hand >10x-scale same-assay pool; candidate for one preregistered augmentation arm IF the far band is pursued",
        },
        "S1/M6 new-domain data (Track B, 4,540+916 pairs)": {
            "scale": "5,456 pairs, different assays (stability/translation)",
            "status": "not generalizable (V9-1b D3 FAIL both arms) — no MPRAU supervision value",
        },
        "MPRAVarDB-style external MPRA variant pools": {
            "scale": "242,818 variants across 18 experiments (mixed assays/designs)",
            "status": "not on disk; heterogeneity across assays/designs makes same-lineage supervision unlikely without a per-study audit pipeline (R3 + effect-size regime per source)",
        },
    }

    payload = {
        "schema_version": "route_a_v3_d5_farband_readiness_v1",
        "basis": "Fisher-z conservative power (same as Task 12.2 statement); verification power only — learning-side scaling is empirical",
        "verification_power": {
            "current_validation_pairs": current_n,
            "pairs_needed_80pct": jumps,
            "reading": (
                f"Adjudicating the far band itself (0.1351->0.27) needs only ~{jumps['D5 far band: 0.1351 -> 0.27 (40%)']} "
                f"validation pairs — the CURRENT 2,008-pair pool already suffices to verify a far-band model "
                "if one existed. The binding constraint is LEARNING, not verification: no model at the ~6K-pair "
                "training scale has exceeded 0.1351 (multi-arm evidence: unified/joint/prior/CMS/new-data all fail)."
            ),
        },
        "near_band_power_caveat": (
            f"Paradoxically the D5 NEAR band (+0.035 over 0.1351) needs ~{jumps['D5 near band: 0.1351 -> 0.17 (+0.035)']} validation "
            "pairs for 80% power at the same conservative basis — the current pool cannot confirm a near-band "
            "jump with CI excluding zero (consistent with D5 amendment fact 2: CI noise-dominated; more seeds "
            "do not help; only a larger validation pool would)."
        ),
        "data_sources": sources,
        "readiness_verdict": (
            "FAR BAND NOT READY as a training-side target: (a) verification pool is already sufficient "
            "(2,008 >= ~795 needed for the 0.27 adjudication), so no validation-data wait is required; "
            "(b) the ~10x variant-level supervision does not exist on disk in same-assay form — the only "
            "candidate is the CMS array (85K rows, same lineage) which failed as prior injection but has "
            "never been tried as plain training augmentation; (c) external pools (MPRAVarDB-class) would "
            "require a per-study R3 + effect-size audit pipeline before any use. RECOMMENDATION: far band "
            "starts only if (i) a preregistered CMS-as-training-augmentation arm is run and shows the "
            "s_mprau_in recipe scaling beyond 0.1351 on the existing pool, or (ii) a new same-assay "
            "variant MPRA is deposited; otherwise the near band remains the actionable target and its "
            "significance ceiling stays honestly CI-crossing-zero at n=2,008."
        ),
    }
    (OUT / "d5_farband_readiness.json").write_text(json.dumps(payload, indent=1))
    print(json.dumps(payload["verification_power"], indent=1))
    print(payload["readiness_verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
