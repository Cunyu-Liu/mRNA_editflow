#!/usr/bin/env python3
"""Audit Benchmark v2 role isolation, task semantics and provenance."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Mapping

ROLES = ["train", "val", "test_id", "test_family", "test_context", "test_assay", "test_ood"]
FINAL_ROLES = set(ROLES[2:])
REQUIRED_SOURCE_MATCHED_FIELDS = [
    "source_id", "candidate_id", "source_sequence", "candidate_sequence",
    "edit_list", "edit_count", "measured_source", "measured_candidate",
    "measured_delta", "cargo", "cell_context", "assay", "batch", "replicate",
]
LOCAL_ROLES = {"train", "val", "test_id", "test_family", "test_ood"}


def _manifest(root: Path, role: str) -> Dict:
    return json.loads((root / "manifests" / f"{role}.json").read_text())


def audit(root: Path, *, allow_final_labels: bool) -> Dict:
    manifests = {role: _manifest(root, role) for role in ROLES}
    record_to_roles: Dict[str, set] = defaultdict(set)
    for role, manifest in manifests.items():
        index_path = root / str(manifest["index_path"])
        with index_path.open() as fh:
            for line in fh:
                rid = line.strip()
                if rid:
                    record_to_roles[rid].add(role)

    source_roles: Dict[tuple, set] = defaultdict(set)
    mother_roles: Dict[tuple, set] = defaultdict(set)
    family_roles: Dict[tuple, set] = defaultdict(set)
    seq_roles: Dict[tuple, set] = defaultdict(set)
    context_roles: Dict[tuple, set] = defaultdict(set)
    assay_roles: Dict[tuple, set] = defaultdict(set)
    cargo_roles: Dict[tuple, set] = defaultdict(set)
    local_family_roles: Dict[tuple, set] = defaultdict(set)
    ood_dimension_counts = Counter()
    counts = Counter()
    confidence = {role: Counter() for role in ROLES}
    task_kinds = {role: Counter() for role in ROLES}
    required_missing = Counter()
    measured_missing = Counter()
    records_path = root / str(next(iter(manifests.values()))["records_path"])
    with records_path.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            rid = str(rec.get("record_id"))
            for role in record_to_roles.get(rid, ()):
                counts[role] += 1
                confidence[role][str(rec.get("confidence"))] += 1
                task_kind = str(rec.get("task_kind"))
                task_kinds[role][task_kind] += 1
                missing = [f for f in REQUIRED_SOURCE_MATCHED_FIELDS if f not in rec]
                required_missing.update(missing)
                if task_kind in {"local_delta", "context_delta", "assay_delta"} and any(rec.get(f) is None for f in ("measured_source", "measured_candidate", "measured_delta")):
                    measured_missing[role] += 1
                # Leakage checks cover local-delta and absolute family
                # tasks; context/assay axis records remain separate tasks and
                # are not silently treated as nucleotide interventions.
                if task_kind in {"local_delta", "absolute_property_regression", "absolute_property_family_shift", "absolute_property_length_shift"}:
                    source_roles[(task_kind, str(rec.get("source_id")))].add(role)
                    mother_roles[(task_kind, str(rec.get("mother_id", rec.get("source_id"))))].add(role)
                    family_roles[(task_kind, str(rec.get("family_cluster_id")))].add(role)
                    seq_roles[(task_kind, str(rec.get("candidate_sequence_sha256")).strip())].add(role)
                context_roles[(task_kind, str(rec.get("cell_context")))].add(role)
                assay_roles[(task_kind, str(rec.get("assay")))].add(role)
                cargo_roles[(task_kind, str(rec.get("cargo")))].add(role)
                if task_kind == "local_delta":
                    local_family_roles[(
                        task_kind,
                        str(rec.get("cargo") or "").strip().casefold(),
                        str(rec.get("protein_family_id") or "").strip().casefold(),
                    )].add(role)
                for dimension in rec.get("ood_dimensions") or []:
                    ood_dimension_counts[(str(dimension), task_kind, role)] += 1

    source_cross = {str(k): sorted(v) for k, v in source_roles.items() if len(v) > 1}
    mother_cross = {str(k): sorted(v) for k, v in mother_roles.items() if len(v) > 1}
    seq_cross = {str(k): sorted(v) for k, v in seq_roles.items() if len(v) > 1}
    family_cross = {
        str(k): sorted(v) for k, v in family_roles.items()
        if "test_family" in v and any(r in LOCAL_ROLES - {"test_family"} for r in v)
    }
    # OOD is a distribution-shift diagnostic; family overlap is reported but
    # does not violate the stricter independent-family contract.
    ood_family_overlap = {
        str(k): sorted(v) for k, v in family_roles.items()
        if "test_ood" in v and any(r in {"train", "val", "test_id", "test_family"} for r in v)
    }
    nonempty = {role: int(counts[role]) > 0 for role in ROLES}
    local_delta_counts = {
        role: int(task_kinds[role].get("local_delta", 0)) for role in ROLES
    }
    context_delta_counts = {
        role: int(task_kinds[role].get("context_delta", 0)) for role in ROLES
    }
    assay_delta_counts = {
        role: int(task_kinds[role].get("assay_delta", 0)) for role in ROLES
    }
    context_test = sum(task_kinds["test_context"].values())
    assay_test = sum(task_kinds["test_assay"].values())
    normalize_cargo = lambda value: str(value).strip().casefold()
    family_absolute_cargos = {
        normalize_cargo(key[1]) for key, roles in cargo_roles.items()
        if key[0] == "absolute_property_family_shift" and "test_family" in roles
    }
    train_absolute_cargos = {
        normalize_cargo(key[1]) for key, roles in cargo_roles.items()
        if key[0] == "absolute_property_regression" and "train" in roles
    }
    family_absolute_cargo_overlap = family_absolute_cargos & train_absolute_cargos
    family_absolute_cargo_holdout_ready = bool(family_absolute_cargos) and not family_absolute_cargo_overlap
    family_local_delta_cargos = {
        normalize_cargo(key[1]) for key, roles in cargo_roles.items()
        if key[0] == "local_delta" and "test_family" in roles
    }
    train_local_delta_cargos = {
        normalize_cargo(key[1]) for key, roles in cargo_roles.items()
        if key[0] == "local_delta" and "train" in roles
    }
    # The independent-family contract compares test_family against the
    # development/test_id pool. test_ood may intentionally reuse a cargo
    # family while remaining sequence/family-cluster disjoint and is reported
    # as a distribution-shift diagnostic instead of train leakage.
    local_family_cross = {
        str(k): sorted(v) for k, v in local_family_roles.items()
        if "test_family" in v and any(r in {"train", "val", "test_id"} for r in v)
    }
    local_family_test_keys = {
        (key[1], key[2]) for key, roles in local_family_roles.items()
        if "test_family" in roles
    }
    local_family_train_keys = {
        (key[1], key[2]) for key, roles in local_family_roles.items()
        if "train" in roles
    }
    species_local_delta_ready = ood_dimension_counts[("species_tail", "local_delta", "test_ood")] > 0
    length_local_delta_ready = ood_dimension_counts[("length_tail", "local_delta", "test_ood")] > 0
    prospective = json.loads((root / "manifests" / "prospective.json").read_text())
    strict_acceptance_reasons = []
    if local_family_test_keys & local_family_train_keys:
        strict_acceptance_reasons.append("test_family local_delta is not cargo/protein-family disjoint")
    if not species_local_delta_ready:
        strict_acceptance_reasons.append("species OOD has observational mouse 5UTRs but no source-matched local_delta labels")
    if not length_local_delta_ready:
        strict_acceptance_reasons.append("length OOD is measured absolute-only; no source-matched local_delta length tail")
    strict_acceptance_reasons.append("full legal DP/beam measured reference requires labels for unobserved legal actions")
    strict_p1_acceptance = {"passed": not strict_acceptance_reasons, "reasons": strict_acceptance_reasons}
    structural_pass = (
        not source_cross and not seq_cross and not family_cross
        and not mother_cross and not local_family_cross
        and not required_missing and not measured_missing
        and all(nonempty.values())
        and context_test > 0 and assay_test > 0
        and family_absolute_cargo_holdout_ready
    )
    report = {
        "schema_version": "nmi_benchmark_v2_audit_v2",
        "passed": structural_pass,
        "allow_final_labels": allow_final_labels,
        "final_label_gate": "open" if allow_final_labels else "closed",
        "counts": {role: int(counts[role]) for role in ROLES},
        "local_delta_counts": local_delta_counts,
        "source_matched_axis_counts": {
            "context_delta": context_delta_counts,
            "assay_delta": assay_delta_counts,
        },
        "confidence_by_role": {role: dict(confidence[role]) for role in ROLES},
        "task_kinds_by_role": {role: dict(task_kinds[role]) for role in ROLES},
        "source_cross_role_count": len(source_cross),
        "mother_cross_role_count": len(mother_cross),
        "candidate_cross_role_count": len(seq_cross),
        "family_train_final_overlap_count": len(family_cross),
        "local_family_train_final_overlap_count": len(local_family_cross),
        "ood_family_overlap_count": len(ood_family_overlap),
        "ood_dimension_counts": {
            f"{dimension}:{task_kind}:{role}": int(count)
            for (dimension, task_kind, role), count in sorted(ood_dimension_counts.items())
        },
        "required_field_missing_counts": dict(required_missing),
        "local_delta_measured_field_missing_by_role": dict(measured_missing),
        "nonempty_final_roles": {role: nonempty[role] for role in FINAL_ROLES},
        "context_axis": {
            "records_present": context_test > 0,
            "scope": manifests["test_context"].get("role_scope"),
            "source_matched_context_delta_ready": context_delta_counts["test_context"] > 0,
            "local_delta_ready": local_delta_counts["test_context"] > 0,
            "absolute_property_only_records": int(task_kinds["test_context"].get("absolute_property_context_shift", 0)),
            "contexts": sorted({k[1] for k, v in context_roles.items() if "test_context" in v}),
        },
        "assay_axis": {
            "records_present": assay_test > 0,
            "scope": manifests["test_assay"].get("role_scope"),
            "source_matched_assay_delta_ready": assay_delta_counts["test_assay"] > 0,
            "local_delta_ready": local_delta_counts["test_assay"] > 0,
            "absolute_property_only_records": int(task_kinds["test_assay"].get("absolute_property_assay_shift", 0)),
            "assays": sorted({k[1] for k, v in assay_roles.items() if "test_assay" in v}),
        },
        "strict_p1_acceptance": strict_p1_acceptance,
        "family_axis": {
            "cargo_absolute_holdout_ready": family_absolute_cargo_holdout_ready,
            "absolute_test_cargo_count": len(family_absolute_cargos),
            "absolute_test_cargo_examples": sorted(family_absolute_cargos)[:10],
            "absolute_train_cargo_count": len(train_absolute_cargos),
            "absolute_train_cargo_examples": sorted(train_absolute_cargos)[:10],
            "absolute_cargo_overlap": sorted(family_absolute_cargo_overlap),
            "local_delta_cargo_family_disjoint": not bool(local_family_test_keys & local_family_train_keys),
            "local_delta_test_cargos": sorted(family_local_delta_cargos),
            "local_delta_train_cargos": sorted(train_local_delta_cargos),
            "local_delta_test_family_keys": sorted([list(key) for key in local_family_test_keys]),
            "local_delta_train_family_keys": sorted([list(key) for key in local_family_train_keys]),
            "scope": manifests["test_family"].get("role_scope"),
        },
        "prospective": prospective,
        "examples": {
            "source_cross_role": dict(list(source_cross.items())[:10]),
            "mother_cross_role": dict(list(mother_cross.items())[:10]),
            "candidate_cross_role": dict(list(seq_cross.items())[:10]),
            "family_train_final_overlap": dict(list(family_cross.items())[:10]),
            "local_family_train_final_overlap": dict(list(local_family_cross.items())[:10]),
            "ood_family_overlap": dict(list(ood_family_overlap.items())[:10]),
        },
        "claim_policy": {
            "proxy_is_biological_ground_truth": False,
            "absolute_property_context_assay_not_local_delta": True,
            "empty_or_proxy_only_roles_block_sota": True,
        },
    }
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/nmi_benchmark_v2")
    ap.add_argument("--out", default="artifacts/phase1/nmi_benchmark_v2_audit.json")
    ap.add_argument("--allow-final-labels", action="store_true")
    args = ap.parse_args()
    report = audit(Path(args.root), allow_final_labels=args.allow_final_labels)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["passed"] else 2)


if __name__ == "__main__":
    main()
