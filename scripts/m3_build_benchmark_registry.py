#!/usr/bin/env python3
"""M3: build docs/execution/xeditflow_benchmark_registry.yaml.

Defines mRNA-EditBench v2 sub-benchmarks and binds each to the M1 task/split
registry and the M2 accepted asset roles. Sub-benchmarks that have no qualified
data (EditBench-CDS-B1-Synonymous) are marked DORMANT and must NOT fabricate a
PASS (per M3: '不得对不存在合格数据的子 benchmark 造空 PASS').

Expected-set / FK closure is enforced by tests/migration/test_m3_migration.py.
"""
from pathlib import Path

import yaml

EXEC = Path("docs/execution")
ASSET_ROLE = EXEC / "xeditflow_asset_role_assignment.yaml"
TASK_REG = EXEC / "xeditflow_task_registry.yaml"
SPLIT_REG = EXEC / "xeditflow_split_registry.yaml"
TASK_SPLIT_MATRIX = EXEC / "xeditflow_task_split_matrix.yaml"
OUT = EXEC / "xeditflow_benchmark_registry.yaml"

NEW_CONTRACT_ID = "mrna_xeditflow_goal_v1_1"


def load(path: Path):
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    assets = load(ASSET_ROLE)["assets"]
    task_reg = load(TASK_REG)
    split_reg = load(SPLIT_REG)
    matrix = load(TASK_SPLIT_MATRIX)["matrix"]

    by_acc = {a["asset_id"]: a for a in assets}
    valid_task_ids = {
        t["id"] for t in task_reg["primary_tasks"] + task_reg["secondary_tasks"] + task_reg["theory_tasks"]
    }
    valid_split_ids = {s["id"] for s in split_reg["splits"]}

    def accepted(acc):
        return acc in by_acc and by_acc[acc]["role"] == "ACCEPTED_FOR_NEW_ROLE"

    def grade(acc):
        return by_acc[acc]["orthogonal_axes"]["intervention_evidence_grade"]

    def by_id(a):
        return a["asset_id"]

    def region(acc):
        return by_acc[acc]["region"]

    # 5'UTR A1 natural-pair benchmark: accepted EFFECT_PRIMARY A1 assets whose
    # genomic region is 5'UTR. 3'UTR assets (ENCSR854RUF, GSE186455, GSE200304,
    # GSE232571, GSE232572, GSE261709, GSE298114) and non-variant MPRA assets
    # (GSE288185, GSE330741 -> PENDING_BLOCKED) are excluded here so that the
    # 5'UTR / 3'UTR pools stay independent endpoint heads.
    five_u_a1 = sorted(
        (a for a in assets
        if a["role"] == "ACCEPTED_FOR_NEW_ROLE"
        and region(a["asset_id"]) == "5UTR"
        and a["orthogonal_axes"]["intervention_evidence_grade"] == "A1"
        and a["orthogonal_axes"]["method_training_role"] == "EFFECT_PRIMARY"
        and a["orthogonal_axes"]["critic_eligibility"] == "YES"),
        key=by_id,
    )
    # 5'UTR A2 dense benchmark: accepted A2 EFFECT_PRIMARY 5'UTR assets.
    five_u_a2 = sorted(
        (a for a in assets
        if a["role"] == "ACCEPTED_FOR_NEW_ROLE"
        and region(a["asset_id"]) == "5UTR"
        and a["orthogonal_axes"]["intervention_evidence_grade"] == "A2"
        and a["orthogonal_axes"]["method_training_role"] == "EFFECT_PRIMARY"),
        key=by_id,
    )
    # 3'UTR A1 variant transfer benchmark: accepted A1 3'UTR assets (independent
    # endpoint heads; transfer track). Includes ENCSR854RUF + the 3'UTR A1 assets
    # that were previously (incorrectly) pooled into the 5'UTR benchmark.
    three_u_a1 = sorted(
        (a for a in assets
        if a["role"] == "ACCEPTED_FOR_NEW_ROLE"
        and region(a["asset_id"]) == "3UTR"
        and a["orthogonal_axes"]["intervention_evidence_grade"] == "A1"),
        key=by_id,
    )
    # CDS B1 synonymous: none accepted yet (GSE207584 is PENDING_BLOCKED).
    cds_b1 = []
    for a in assets:
        if a["orthogonal_axes"]["intervention_evidence_grade"] == "B1":
            cds_b1.append(a["asset_id"])

    benchmarks = [
        {
            "id": "EditBench-5U-A1-Natural",
            "region": "5UTR",
            "evidence_grade": "A1",
            "description": "5'UTR intentionally-assayed natural source/ref-candidate/alt pairs.",
            "status": "ACTIVE" if five_u_a1 else "DORMANT",
            "status_reason": None if five_u_a1 else (
                "No qualified 5'UTR A1 natural-pair asset remains after region correction "
                "(GSE186455/GSE200304/GSE232571/GSE232572/GSE261709/GSE298114 are 3'UTR; "
                "GSE288185 is a non-variant MPRA). Must not fabricate a PASS until a "
                "qualified 5'UTR A1 asset is accepted."
            ),
            "asset_ids": [a["asset_id"] for a in five_u_a1],
            "primary_tasks": [
                "T5_SOURCE_RELATIVE_EFFECT",
                "T5_SELECTIVE_EFFECT",
                "T5_MEASURED_NEIGHBORHOOD_OPTIMIZATION",
                "T5_FIXED_BUDGET_MULTI_STEP_OPTIMIZATION",
            ],
            "splits": ["S1", "S2", "S3", "S4", "S5"],
            "sealed_external": "S6",
        },
        {
            "id": "EditBench-5U-A2-Dense",
            "region": "5UTR",
            "evidence_grade": "A2",
            "description": "5'UTR controlled dense measured landscape.",
            "status": "ACTIVE" if five_u_a2 else "DORMANT",
            "status_reason": None if five_u_a2 else (
                "No qualified 5'UTR A2 dense asset remains after region correction: "
                "GSE330741 is a non-variant in-vivo localization MPRA (PENDING_BLOCKED). "
                "Must not fabricate a PASS until a qualified 5'UTR A2 asset is accepted."
            ),
            "asset_ids": [a["asset_id"] for a in five_u_a2],
            "primary_tasks": [
                "T5_SOURCE_RELATIVE_EFFECT",
                "T5_MEASURED_NEIGHBORHOOD_OPTIMIZATION",
            ],
            "splits": ["S1", "S2", "S3", "S4"],
            "sealed_external": "S6",
        },
        {
            "id": "EditBench-3U-A1-Variant",
            "region": "3UTR",
            "evidence_grade": "A1",
            "description": "3'UTR variant transfer-track A1 (independent endpoint heads).",
            "status": "ACTIVE" if three_u_a1 else "DORMANT",
            "status_reason": None if three_u_a1 else (
                "No qualified 3'UTR A1 variant asset remains after region correction."
            ),
            "asset_ids": [a["asset_id"] for a in three_u_a1],
            "primary_tasks": ["T3_EFFECT_TRANSFER", "CROSS_REGION_TRANSFER"],
            "splits": ["S4", "S5"],
            "sealed_external": "S6",
        },
        {
            "id": "EditBench-CDS-B1-Synonymous",
            "region": "CDS",
            "evidence_grade": "B1",
            "description": "same-protein synonymous-codon family ranking (transfer).",
            "status": "DORMANT",
            "status_reason": (
                "No qualified B1 data accepted yet; GSE207584 is PENDING_BLOCKED legacy CDS "
                "liability. Must not fabricate a PASS until sequence/family/label rebuild."
            ),
            "asset_ids": cds_b1,
            "primary_tasks": ["CDS_SYNONYMOUS_FAMILY_RANKING"],
            "splits": ["S7"],
            "sealed_external": "S6",
        },
    ]

    # expected-set closure.
    expected_benchmark_ids = [b["id"] for b in benchmarks]
    for b in benchmarks:
        for tid in b["primary_tasks"]:
            assert tid in valid_task_ids, f"{b['id']} unknown task {tid}"
            assert tid in matrix, f"{b['id']} task {tid} missing from task-split matrix"
        allowed_task_splits = {sid for tid in b["primary_tasks"] for sid in matrix[tid]}
        for sid in b["splits"]:
            assert sid in valid_split_ids, f"{b['id']} unknown split {sid}"
            assert sid in allowed_task_splits, \
                f"{b['id']} split {sid} not allowed by any of its primary tasks"
        for acc in b["asset_ids"]:
            assert accepted(acc), f"{b['id']} asset {acc} not ACCEPTED_FOR_NEW_ROLE"

    data = {
        "contract_id": NEW_CONTRACT_ID,
        "version": "2.0",
        "date": "2026-08-06",
        "benchmark_name": "mRNA-EditBench v2",
        "expected_benchmark_ids": expected_benchmark_ids,
        "sub_benchmarks": benchmarks,
        "gse246381": {
            "role": "SEALED_EXTERNAL_FINAL_CANDIDATE",
            "in_task_activation": False,
            "in_metric_branch": False,
            "in_calibration": False,
            "in_model_selection": False,
            "ordinary_loader_returns_zero_rows_before_final": True,
            "final_evaluator_count_max": 1,
        },
    }
    OUT.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True) + "# EOF\n",
        encoding="utf-8",
    )
    for b in benchmarks:
        print(f"{b['id']}: status={b['status']} assets={b['asset_ids']} tasks={b['primary_tasks']}")


if __name__ == "__main__":
    main()