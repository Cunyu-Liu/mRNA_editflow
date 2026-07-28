from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INVENTORY = (
    ROOT
    / "artifacts/stages/D1_B0_20260728T160012Z_8862125/D1/input_inventory.json"
)
SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _inventory() -> dict:
    return json.loads(INVENTORY.read_text(encoding="utf-8"))


def test_input_snapshot_has_exact_identity_for_every_copied_byte() -> None:
    inventory = _inventory()
    files = inventory["files"]
    paths = [row["path"] for row in files]
    assert len(paths) == len(set(paths)) == 9
    for row in files:
        assert row["path"].startswith(inventory["root"] + "/")
        assert row["bytes"] > 0
        assert SHA256.fullmatch(row["sha256"])
        assert row["source"]
        assert row["role"]
        assert row["label_access"]


def test_input_inventory_never_uses_labels_for_role_selection() -> None:
    inventory = _inventory()
    assert inventory["selection_is_label_independent"] is True
    assert all(
        "SELECTION_ALLOWED" not in row["label_access"]
        for row in inventory["files"]
    )


def test_production_inputs_cover_each_admitted_or_conditional_dataset() -> None:
    inventory = _inventory()
    counts = inventory["dataset_input_counts"]
    assert counts == {
        "GSE114002": 2,
        "GSE200304": 3,
        "GSE217518": 2,
        "GSE246381": 1,
        "MPRAu_processed_ENCSR854RUF": 1,
    }
    assert sum(counts.values()) == len(inventory["files"])


def test_gse217518_official_code_revision_is_full_commit_sha() -> None:
    revision = _inventory()["official_code_revisions"]["GSE217518"]
    assert revision["repository"].startswith("https://github.com/")
    assert re.fullmatch(r"[0-9a-f]{40}", revision["commit"])


def test_input_copy_and_overwrite_counters_are_zero() -> None:
    gate = _inventory()["preflight_gate"]
    assert gate["all_copied_files_have_sha256"] is True
    assert gate["all_copied_files_have_byte_size"] is True
    assert gate["unknown_input_hashes"] == 0
    assert gate["raw_source_files_modified"] == 0
    assert gate["existing_results_overwritten"] == 0
