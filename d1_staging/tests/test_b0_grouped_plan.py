from pathlib import Path
import sys

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "scripts" / "b0"))

from b0_v3_1_builder import _build_grouped_plan  # noqa: E402


PARTITIONS = [
    {"partition_id": "s::development", "partition_role": "DEVELOPMENT"},
    {"partition_id": "s::internal_test", "partition_role": "INTERNAL_TEST"},
    {"partition_id": "s::train", "partition_role": "TRAIN"},
]


def obj(oid, source, gene="gene:g"):
    return {
        "object_id": oid,
        "object_type": "PAIR",
        "scientific_track": "E",
        "region_scope": "5UTR",
        "study": "gse_test",
        "group_ids_by_atom": {
            "PAIR": [f"pair:{oid}"],
            "SOURCE": [source],
            "GENE": [gene],
        },
    }


def test_shared_group_is_transitively_kept_in_one_partition():
    objects = {
        "p1": obj("p1", "source:shared"),
        "p2": obj("p2", "source:shared"),
        "p3": obj("p3", "source:other", gene="gene:other"),
    }
    split = {
        "split_contract_id": "s",
        "object_scope": "PAIR",
        "region_scope": "5UTR",
        "grouping_atoms_by_object_type": {"PAIR": ["PAIR", "SOURCE", "GENE"]},
        "partitions": PARTITIONS,
    }
    plan, missing = _build_grouped_plan(objects, split, False)
    assert not missing
    assert plan["p1"]["partition_id"] == plan["p2"]["partition_id"]
    assert set(plan) == {"p1", "p2", "p3"}


def test_missing_atom_is_fail_closed():
    objects = {"p1": obj("p1", "source:shared")}
    del objects["p1"]["group_ids_by_atom"]["GENE"]
    split = {
        "split_contract_id": "s",
        "object_scope": "PAIR",
        "region_scope": "5UTR",
        "grouping_atoms_by_object_type": {"PAIR": ["PAIR", "SOURCE", "GENE"]},
        "partitions": PARTITIONS,
    }
    plan, missing = _build_grouped_plan(objects, split, False)
    assert plan == {}
    assert any(reason == "missing_required_grouping_atom:GENE" for (_oid, reason) in missing)
