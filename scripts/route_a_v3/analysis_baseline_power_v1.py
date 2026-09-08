#!/usr/bin/env python3
"""Baseline Task 12.2: small-sample power statement for the bottom-line family.

SPECS_BASELINE_LEADERBOARD spec Task 12.2: formal power analysis text for the
n=48/274/730 tasks, extending the MRL 730-record insufficiency note (already on
file for the Route A vs frozen-Optimus tie) into a standard clause covering
every small-sample adjudication in the bottom-line family.

Method: Fisher z approximation for the difference of two dependent Spearman
correlations, conservative independence-based SE = sqrt(2/(n-3)) (pairing
reduces true SE, so stated power is a lower bound). MDE at 80% power, two-sided
alpha=0.05 -> z* = 1.96 + 0.84 = 2.80. Also reported: n required for 80% power
at the observed effect (Fisher-z scale), and whether each stored adjudication
is power-limited (observed |dz| < MDE_z).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

MNT = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2")
OUT = MNT / "experiments/analysis_baseline_power_20260909"
Z_STAR = 1.96 + 0.84  # two-sided 0.05 + 80% power

# Bottom-line family rows (aligned with baseline_holm_v1.py Task 12.1).
# n = VALIDATION records actually adjudicated (pair-level where applicable).
ROWS = [
    # name, n, our_rho, their_rho, note
    ("MRL: Route A ens vs frozen-Optimus", 730, 0.3158, 0.3132,
     "stored tie (delta CI crosses zero) - the original 730-record clause"),
    ("polyA: critic V5 vs APARENT frozen", 2628, 0.8219, 0.7343,
     "stored win (delta CI excludes zero)"),
    ("MPRAU: s_mprau_in ens vs Saluki frozen", 2008, 0.1351, 0.1205,
     "stored marginal (delta CI crosses zero)"),
    ("TE200304: critic V5 vs internal target", 1614, 0.0579, -0.0266,
     "internal-target win"),
    ("GSE149487-TE: critic V5 vs internal target", 48, 0.1953, 0.1747,
     "n=48 small-sample win"),
    ("GSE149487-RNA: critic V5 vs RNA-FM frozen", 48, 0.0500, 0.2958,
     "n=48 registered loss"),
    ("GSE186455: critic V5 vs RNA-FM frozen", 274, 0.0639, 0.1043,
     "n=274 loss vs best external row"),
    ("GSE186455: critic V5 vs internal target", 274, 0.0639, -0.0052,
     "internal-target win"),
]


def fisher_z(rho: float) -> float:
    return math.atanh(max(-0.999999, min(0.999999, rho)))


def analyze(name: str, n: int, ours: float, theirs: float, note: str) -> dict:
    dz = fisher_z(ours) - fisher_z(theirs)
    se = math.sqrt(2.0 / (n - 3))
    mde_z = Z_STAR * se
    power_obs = _power(dz / se) if se > 0 else float("nan")
    n_req = int(math.ceil(2.0 / (dz / Z_STAR) ** 2 + 3)) if abs(dz) > 1e-9 else None
    return {
        "comparison": name,
        "n_validation": n,
        "our_spearman": ours,
        "their_spearman": theirs,
        "delta_spearman": round(ours - theirs, 4),
        "delta_fisher_z": round(dz, 4),
        "se_fisher_z": round(se, 4),
        "mde_fisher_z_80pct": round(mde_z, 4),
        "power_at_observed_effect": round(power_obs, 3),
        "n_required_80pct_at_observed_effect": n_req,
        "power_limited": abs(dz) < mde_z,
        "note": note,
    }


def _power(zstat: float) -> float:
    """Approximate two-sided power at |zstat| via normal CDF."""
    from statistics import NormalDist
    nd = NormalDist()
    return nd.cdf(abs(zstat) - 1.96) + nd.cdf(-abs(zstat) - 1.96)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [analyze(*r) for r in ROWS]
    payload = {
        "schema_version": "route_a_v3_baseline_power_statement_v1",
        "method": {
            "approximation": "Fisher z difference of two Spearman correlations",
            "se": "sqrt(2/(n-3)), independence-based (conservative lower bound on power)",
            "alpha": 0.05,
            "target_power": 0.80,
            "z_star": Z_STAR,
        },
        "rows": rows,
    }
    (OUT / "power_statement_v1.json").write_text(json.dumps(payload, indent=1))

    # Markdown statement
    lines = [
        "# Small-sample power statement (bottom-line family) — Task 12.2",
        "",
        "Scope: every adjudicated comparison in the baseline bottom-line family",
        "(frozen evaluator, VALIDATION splits, same rows as the Task 12.1 Holm table).",
        "",
        "## Method",
        "",
        "- Fisher z approximation of the difference of two Spearman correlations:",
        "  dz = arctanh(rho_ours) - arctanh(rho_theirs); SE(dz) = sqrt(2/(n-3)).",
        "- SE is independence-based; paired bootstrap SEs are smaller, so every",
        "  power figure below is a conservative lower bound.",
        "- MDE = minimum detectable Fisher-z difference at 80% power, two-sided",
        "  alpha=0.05 (z* = 2.80). 'power_limited' = observed |dz| < MDE.",
        "",
        "## Table",
        "",
        "| comparison | n | ours | theirs | dSpearman | dz | MDE_z | power@obs | n needed | power-limited |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            "| {cmp} | {n} | {o:.4f} | {t:.4f} | {d:+.4f} | {dz:+.4f} | {mde:.4f} |"
            " {p:.0%} | {nreq} | {pl} |".format(
                cmp=r["comparison"], n=r["n_validation"], o=r["our_spearman"],
                t=r["their_spearman"], d=r["delta_spearman"], dz=r["delta_fisher_z"],
                mde=r["mde_fisher_z_80pct"], p=r["power_at_observed_effect"],
                nreq=r["n_required_80pct_at_observed_effect"] or "n/a",
                pl="YES" if r["power_limited"] else "no",
            )
        )
    lines += [
        "",
        "## Standard clause (applies to all power-limited rows)",
        "",
        "1. **MRL (n=730)**: the Route A ensemble vs frozen-Optimus tie",
        "   (delta +0.0027, CI [-0.045, +0.048]) is consistent with an effect up to",
        "   ~0.15 Spearman in either direction; the 730-record VALIDATION split has",
        "   80% power only for differences >= ~0.15 (Fisher-z MDE 0.147). The tie is",
        "   reported as 'statistically indistinguishable, point estimate ahead',",
        "   never as 'no difference exists'.",
        "2. **GSE149487 TE/RNA (n=48 each)**: at n=48 the MDE is ~0.59 Fisher-z",
        "   (~0.55 Spearman at these baselines). Both the TE win (+0.021) and the",
        "   RNA loss (-0.246) are below MDE and must be reported as",
        "   'direction-only, power-limited'; the RNA loss to RNA-FM (0.050 vs 0.296)",
        "   reaches only ~23% power and would need n~245 for confirmation.",
        "3. **GSE186455 (n=274)**: MDE ~0.24 Fisher-z (~0.23 Spearman). The",
        "   internal-target win (+0.069) is power-limited; the deficit vs RNA-FM",
        "   frozen (-0.040) is likewise not confirmable at this n.",
        "4. **MPRAU (n=2,008 variant pairs)**: MDE ~0.09 Fisher-z (~0.09 Spearman).",
        "   The s_mprau_in ensemble edge over Saluki (+0.0146, CI crossing zero) is",
        "   power-limited - consistent with the D5 amendment keeping 0.1351 as a",
        "   point-estimate-only in-house row.",
        "5. **TE200304 (n=1,614)**: the internal-target win (+0.085) sits just",
        "   below the independent-approximation MDE (0.099 Fisher-z, 67% power) -",
        "   borderline power-limited; the stored paired-bootstrap CI governs the",
        "   formal adjudication. polyA (+0.088 at n=2,628) is the only row that",
        "   exceeds its MDE with ~100% power and a zero-excluding stored CI.",
        "",
        "Reading rule for the paper: any bottom-line row flagged power-limited is",
        "reported with its MDE and required n, and claims are phrased as",
        "directional unless the stored paired-bootstrap CI excludes zero.",
        "",
        "Generated by analysis_baseline_power_20260909/power_statement_v1.json.",
    ]
    (OUT / "power_statement_v1.md").write_text("\n".join(lines))
    print("\n".join(lines[10:24]))
    print(f"wrote {OUT}/power_statement_v1.md and .json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
