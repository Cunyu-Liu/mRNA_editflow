#!/usr/bin/env python3
"""M2: build docs/execution/xeditflow_asset_role_assignment.yaml.

Maps every old P0/P1 asset to exactly one of
    ACCEPTED_FOR_NEW_ROLE / EXCLUDED_WITH_EVIDENCE / REFERENCE_ONLY / PENDING_BLOCKED
and assigns the orthogonal axes (scientific_track / intervention_evidence_grade /
method_training_role / various eligibility booleans) with evidence IDs.

Evidence source: data/v3_1/registry/dataset_assets.jsonl + license_matrix.csv
+ dataset_decisions.jsonl (all already frozen under v3.1, contract SHA
ecc6c635...). This file only adds the new orthogonal axes; it never rewrites
the frozen v3.1 registry.
"""

import csv
import json
from pathlib import Path

OUT = Path("docs/execution/xeditflow_asset_role_assignment.yaml")
ASSETS = Path("data/v3_1/registry/dataset_assets.jsonl")
LICENSE = Path("data/v3_1/registry/license_matrix.csv")
DECISIONS = Path("data/v3_1/registry/dataset_decisions.jsonl")

ACCEPTED = "ACCEPTED_FOR_NEW_ROLE"
EXCLUDED = "EXCLUDED_WITH_EVIDENCE"
REF_ONLY = "REFERENCE_ONLY"
PENDING = "PENDING_BLOCKED"

# role -> evidence-needed mapping (per M2 special rules + d0 state).
def classify(acc, d0, acquisition, mapping, priority, se):
    """Return (role, grade, method_role, endpoint_role, reason, eligibility)."""
    decis = se.get(acc, {})
    d0 = decis.get("d0_decision", d0)
    acq = decis.get("acquisition_status", acquisition)
    lic_ok = se.get(acc, {}).get("permitted_model_training", None)

    # Hard special rules from M2.
    if acc == "GSE246381":
        return (
            ACCEPTED,
            "A2",
            "CRITIC_AUX",
            "DELTA",
            "Sealed external final candidate; restricted shard, one final only; "
            "never enters activation/calibration/model-selection.",
            {"critic_eligibility": "NO", "flow_base_eligibility": "NO",
             "guidance_training_eligibility": "NO",
             "measured_optimization_eligibility": "NO",
             "transfer_eligibility": "NO", "sealed": "SEALED_EXTERNAL_FINAL_CANDIDATE"},
        )
    if acc == "GSE207584":
        return (
            PENDING,
            "D",
            "EXCLUDED",
            "RECONSTRUCTION",
            "Legacy CDS liability; only enters B1 after sequence/family/label rebuild. "
            "Must not auto-unlock as B1.",
            {"critic_eligibility": "NO", "flow_base_eligibility": "NO",
             "guidance_training_eligibility": "NO",
             "measured_optimization_eligibility": "NO",
             "transfer_eligibility": "NO"},
        )
    if acc == "GSE173083":
        return (
            PENDING,
            "C",
            "CRITIC_AUX",
            "ABSOLUTE_PROPERTY",
            "Provenance not fully closed; keep BLOCKED/AUX until evidence chain closes.",
            {"critic_eligibility": "NO", "flow_base_eligibility": "NO",
             "guidance_training_eligibility": "NO",
             "measured_optimization_eligibility": "NO",
             "transfer_eligibility": "NO"},
        )
    if acc == "GSE145046":
        return (
            PENDING,
            "C",
            "CRITIC_AUX",
            "ABSOLUTE_PROPERTY",
            "Input/support rows only; must complete label join before counting as "
            "functional example.",
            {"critic_eligibility": "NO", "flow_base_eligibility": "NO",
             "guidance_training_eligibility": "NO",
             "measured_optimization_eligibility": "NO",
             "transfer_eligibility": "NO"},
        )
    if acc == "GSE114002":
        return (
            ACCEPTED,
            "A1",
            "EFFECT_PRIMARY",
            "DELTA",
            "Full-file inventory + identity/no-edit recovery; primary 5'UTR natural pairs.",
            {"critic_eligibility": "YES", "flow_base_eligibility": "YES",
             "guidance_training_eligibility": "YES",
             "measured_optimization_eligibility": "YES",
             "transfer_eligibility": "YES"},
        )
    if acc == "GSE217518":
        return (
            ACCEPTED,
            "A1",
            "EFFECT_PRIMARY",
            "DELTA",
            "Attrition/window/assay join closure; 5'UTR natural pairs (eLife CC BY 4.0).",
            {"critic_eligibility": "YES", "flow_base_eligibility": "YES",
             "guidance_training_eligibility": "YES",
             "measured_optimization_eligibility": "YES",
             "transfer_eligibility": "YES"},
        )
    if acc == "ENCSR854RUF":
        return (
            ACCEPTED,
            "A1",
            "EFFECT_PRIMARY",
            "DELTA",
            "3'UTR variant; transfer-track A1.",
            {"critic_eligibility": "YES", "flow_base_eligibility": "NO",
             "guidance_training_eligibility": "NO",
             "measured_optimization_eligibility": "YES",
             "transfer_eligibility": "YES"},
        )

    # Default: metadata-only / unresolved.
    if d0 == "MAPPING_UNRESOLVED" or mapping == "MAPPING_UNRESOLVED":
        return (
            PENDING,
            "D",
            "EXCLUDED",
            "NOT_APPLICABLE",
            "SubSeries/asset mapping unresolved; blocked until member-accession closure.",
            {"critic_eligibility": "NO", "flow_base_eligibility": "NO",
             "guidance_training_eligibility": "NO",
             "measured_optimization_eligibility": "NO",
             "transfer_eligibility": "NO"},
        )
    if acq == "NOT_PRESENT":
        return (
            PENDING,
            "D",
            "EXCLUDED",
            "NOT_APPLICABLE",
            "Raw not present; metadata-only. Re-acquisition required before role.",
            {"critic_eligibility": "NO", "flow_base_eligibility": "NO",
             "guidance_training_eligibility": "NO",
             "measured_optimization_eligibility": "NO",
             "transfer_eligibility": "NO"},
        )

    # Downloaded + acquired for rebuild.
    track = se.get(acc, {}).get("scientific_priority", priority)
    if track in ("E", "E_DELTA", "E_LINK"):
        grade = "A1"
        method = "EFFECT_PRIMARY"
        endpoint = "DELTA"
    elif track == "E_DENSE":
        grade = "A2"
        method = "EFFECT_PRIMARY"
        endpoint = "DELTA"
    elif track in ("F", "E_F"):
        grade = "C"
        method = "CRITIC_AUX"
        endpoint = "ABSOLUTE_PROPERTY"
    else:  # AUX_QC, OUT_OF_SCOPE, P2_AUX
        grade = "C"
        method = "DIAGNOSTIC"
        endpoint = "NOT_APPLICABLE"

    elig = {
        "critic_eligibility": "YES" if method == "EFFECT_PRIMARY" else "NO",
        "flow_base_eligibility": "YES" if method == "EFFECT_PRIMARY" else "NO",
        "guidance_training_eligibility": "YES" if method == "EFFECT_PRIMARY" else "NO",
        "measured_optimization_eligibility": "YES" if endpoint == "DELTA" else "NO",
        "transfer_eligibility": "YES" if track == "E_LINK" else "NO",
    }
    return (ACCEPTED, grade, method, endpoint, "Downloaded+verified, acquired for rebuild.", elig)


def main() -> None:
    se = {}
    for line in ASSETS.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        se[d["accession"]] = d
    # Load decisions for d0/acquisition overrides.
    # Skip member rows (they carry member_accession and share the parent's
    # asset_group_id, e.g. GSE200302/303/217530 under GSE200304); only the
    # parent row's own status may override the parent asset.
    for line in DECISIONS.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        if d.get("asset_group_id") in se and not d.get("member_accession"):
            se[d["asset_group_id"]]["d0_decision"] = d.get("d0_decision")
            se[d["asset_group_id"]]["acquisition_status"] = d.get("acquisition_status")
    # Load license training permission.
    lic = {}
    with LICENSE.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            lic[row["asset_group_id"]] = row.get("training", "").upper() == "YES"

    assets = []
    for acc in sorted(se):
        d = se[acc]
        role, grade, method, endpoint, reason, elig = classify(
            acc,
            d.get("d0_decision", "UNKNOWN"),
            d.get("acquisition_status", "UNKNOWN"),
            d.get("mapping_status", "UNKNOWN"),
            d.get("scientific_priority", "UNKNOWN"),
            se,
        )
        assets.append(
            {
                "asset_id": acc,
                "audit_priority": d.get("audit_priority", "P0"),
                "role": role,
                "reason": reason,
                "evidence": {
                    "license_matrix_sha256": "preserved_v3_1",
                    "dataset_decisions_sha256": "preserved_v3_1",
                    "source_evidence": d.get("file_inventory_evidence_id", "EVIDENCE::v3_1"),
                },
                "orthogonal_axes": {
                    "scientific_track": (d.get("scientific_priority") or "E").split("_")[0]
                    if (d.get("scientific_priority") or "E").split("_")[0] in ("E", "F", "AUX", "REFERENCE")
                    else "E",
                    "intervention_evidence_grade": grade,
                    "method_training_role": method,
                    "endpoint_role": endpoint,
                    "critic_eligibility": elig["critic_eligibility"],
                    "flow_base_eligibility": elig["flow_base_eligibility"],
                    "guidance_training_eligibility": elig["guidance_training_eligibility"],
                    "measured_optimization_eligibility": elig["measured_optimization_eligibility"],
                    "transfer_eligibility": elig["transfer_eligibility"],
                },
                "sealed_scope": lic.get(acc, False),
            }
        )

    # Deterministic YAML emission.
    lines = [
        "# M2: P0/P1 asset role assignment under mrna_xeditflow_goal_v1_1",
        "# Every old P0/P1 asset is classified to exactly one role below.",
        "# Must be one of: ACCEPTED_FOR_NEW_ROLE / EXCLUDED_WITH_EVIDENCE /",
        "# REFERENCE_ONLY / PENDING_BLOCKED.",
        "contract_id: mrna_xeditflow_goal_v1_1",
        "schema_version: 1.1",
        "source_namespace: v3_1_preserved",
        "assets:",
    ]
    for a in assets:
        lines.append(f"  - asset_id: {a['asset_id']}")
        lines.append(f"    audit_priority: {a['audit_priority']}")
        lines.append(f"    role: {a['role']}")
        lines.append(f'    reason: "{a["reason"]}"')
        lines.append("    evidence:")
        lines.append(f"      source_evidence: {a['evidence']['source_evidence']}")
        lines.append("    orthogonal_axes:")
        oa = a["orthogonal_axes"]
        for k, v in oa.items():
            # Quote string values so YAML 1.1 does not coerce NO/YES to booleans.
            lines.append(f'      {k}: "{v}"')
        lines.append(f"    sealed_scope: {a['sealed_scope']}")

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT} with {len(assets)} assets")
    from collections import Counter
    print(Counter(a["role"] for a in assets))


if __name__ == "__main__":
    main()