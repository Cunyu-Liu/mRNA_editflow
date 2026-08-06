"""M3 migration gates: mRNA-EditBench v2 benchmark registry integrity.

Enforces the migration principle for the benchmark layer:
- expected-set closure for sub-benchmark ids;
- task/split FK closure (every bound task/split exists in the v4 registries);
- asset binding (every bound asset is ACCEPTED_FOR_NEW_ROLE);
- no cross-region mixing (5'UTR and 3'UTR pools are independent endpoint heads);
- DORMANT sub-benchmarks must not fabricate a PASS and must carry a status_reason;
- sealed-external isolation (S6 and GSE246381 never enter training branches).
"""
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

EXEC_DIR = REPO_ROOT / "docs" / "execution"

BENCH_REG = EXEC_DIR / "xeditflow_benchmark_registry.yaml"
ASSET_ROLE = EXEC_DIR / "xeditflow_asset_role_assignment.yaml"
TASK_REG = EXEC_DIR / "xeditflow_task_registry.yaml"
SPLIT_REG = EXEC_DIR / "xeditflow_split_registry.yaml"
TASK_SPLIT_MATRIX = EXEC_DIR / "xeditflow_task_split_matrix.yaml"

NEW_CONTRACT_ID = "mrna_xeditflow_goal_v1_1"


def _load_yaml(path: Path):
    assert path.exists(), f"missing {path}"
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _bench():
    d = _load_yaml(BENCH_REG)
    return d, {b["id"]: b for b in d["sub_benchmarks"]}


def test_contract_tie():
    d, _ = _bench()
    assert d["contract_id"] == NEW_CONTRACT_ID
    assert d["benchmark_name"] == "mRNA-EditBench v2"


def test_expected_benchmark_id_closure():
    d, by_id = _bench()
    assert set(d["expected_benchmark_ids"]) == set(by_id.keys())
    assert len(by_id) == 4


def test_each_subbenchmark_has_required_fields():
    d, by_id = _bench()
    for bid, b in by_id.items():
        for field in ["id", "region", "evidence_grade", "status", "asset_ids",
                      "primary_tasks", "splits", "sealed_external"]:
            assert field in b, f"{bid} missing field {field}"
        assert b["status"] in {"ACTIVE", "DORMANT"}


def test_task_fk_closure():
    d, by_id = _bench()
    task_reg = _load_yaml(TASK_REG)
    valid = {
        t["id"]
        for t in task_reg["primary_tasks"] + task_reg["secondary_tasks"] + task_reg["theory_tasks"]
    }
    for bid, b in by_id.items():
        for tid in b["primary_tasks"]:
            assert tid in valid, f"{bid} references unknown task {tid}"


def test_split_fk_closure():
    d, by_id = _bench()
    split_reg = _load_yaml(SPLIT_REG)
    valid = {s["id"] for s in split_reg["splits"]}
    matrix = _load_yaml(TASK_SPLIT_MATRIX)["matrix"]
    for bid, b in by_id.items():
        for tid in b["primary_tasks"]:
            assert tid in matrix, f"{bid} task {tid} missing from task-split matrix"
        allowed = {sid for tid in b["primary_tasks"] for sid in matrix[tid]}
        for sid in b["splits"]:
            assert sid in valid, f"{bid} references unknown split {sid}"
            assert sid in allowed, f"{bid} split {sid} not allowed by its primary tasks"
        assert b["sealed_external"] == "S6"


def test_asset_binding_all_accepted():
    asset_role = _load_yaml(ASSET_ROLE)
    by_acc = {a["asset_id"]: a for a in asset_role["assets"]}
    d, by_id = _bench()
    for bid, b in by_id.items():
        for acc in b["asset_ids"]:
            assert acc in by_acc, f"{bid} asset {acc} not in asset role registry"
            assert by_acc[acc]["role"] == "ACCEPTED_FOR_NEW_ROLE", \
                f"{bid} binds non-accepted asset {acc}"


def test_no_cross_region_mixing():
    """5'UTR and 3'UTR pools must be independent endpoint heads (no overlap)."""
    d, by_id = _bench()
    five_u = set(by_id["EditBench-5U-A1-Natural"]["asset_ids"]) | \
        set(by_id["EditBench-5U-A2-Dense"]["asset_ids"])
    three_u = set(by_id["EditBench-3U-A1-Variant"]["asset_ids"])
    overlap = five_u & three_u
    assert not overlap, f"cross-region pool mixing: {overlap}"
    # ENCSR854RUF is the 3'UTR asset and must NOT leak into the 5'UTR pool.
    assert "ENCSR854RUF" not in five_u, "3'UTR ENT asset leaked into 5'UTR pool"


def test_dormant_does_not_fabricate_pass():
    d, by_id = _bench()
    cds = by_id["EditBench-CDS-B1-Synonymous"]
    assert cds["status"] == "DORMANT"
    assert "status_reason" in cds and cds["status_reason"]
    assert cds["asset_ids"] == [], \
        "DORMANT sub-benchmark must not bind assets until qualified data accepted"
    # No sub-benchmark may claim a PASS status; a DORMANT benchmark with no
    # qualified data must not be counted as an active/achieved result.
    for b in by_id.values():
        assert b["status"] != "PASS", f"{b['id']} fabricated a PASS"
    assert cds["status"] == "DORMANT" and not cds["asset_ids"]


def test_gse246381_sealed_excluded_everywhere():
    """GSE246381 is a sealed external final candidate: never in any benchmark pool,
    never in activation/metric/calibration/model-selection branches."""
    d, by_id = _bench()
    all_assets = set()
    for b in by_id.values():
        all_assets |= set(b["asset_ids"])
    assert "GSE246381" not in all_assets, "sealed GSE246381 leaked into a benchmark pool"
    g = d["gse246381"]
    assert g["role"] == "SEALED_EXTERNAL_FINAL_CANDIDATE"
    for branch in ["in_task_activation", "in_metric_branch", "in_calibration", "in_model_selection"]:
        assert g[branch] is False, f"GSE246381 must be excluded from {branch}"
    assert g["ordinary_loader_returns_zero_rows_before_final"] is True
    assert g["final_evaluator_count_max"] == 1


def test_sealed_external_split_not_in_activation():
    matrix = _load_yaml(TASK_SPLIT_MATRIX)
    assert matrix["sealed_split"] == "S6"
    assert matrix["sealed_split_in_task_activation"] is False


def test_asset_role_axes_consistent_with_benchmark_pool():
    """Every asset bound to a primary benchmark must carry the expected axes
    (EFFECT_PRIMARY for 5U/3U pools, critic eligible for A1 natural)."""
    asset_role = _load_yaml(ASSET_ROLE)
    by_acc = {a["asset_id"]: a for a in asset_role["assets"]}
    d, by_id = _bench()
    for bid in ["EditBench-5U-A1-Natural", "EditBench-5U-A2-Dense", "EditBench-3U-A1-Variant"]:
        for acc in by_id[bid]["asset_ids"]:
            oa = by_acc[acc]["orthogonal_axes"]
            assert oa["method_training_role"] == "EFFECT_PRIMARY", \
                f"{bid} asset {acc} not EFFECT_PRIMARY"
            assert oa["intervention_evidence_grade"] in {"A1", "A2"}, \
                f"{bid} asset {acc} not A-grade"