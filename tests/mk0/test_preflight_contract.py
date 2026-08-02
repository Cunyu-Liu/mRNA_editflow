"""Integrity tests for the metadata-only MK0 preflight inventory."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, relative: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PREFLIGHT = _load("mk0_preflight_contract_test", "scripts/mk0/record_mk0_preflight.py")
FINALIZER = _load(
    "mk0_preflight_finalizer_contract_test",
    "scripts/mk0/finalize_mk0_acceptance.py",
)


def test_preflight_inventory_digest_is_accepted_and_tamper_fails(
    tmp_path: Path,
) -> None:
    (tmp_path / "alpha").write_text("one\n", encoding="utf-8")
    (tmp_path / "beta").mkdir()
    inventory = PREFLIGHT.top_level_inventory(tmp_path)
    FINALIZER._validate_inventory_block(
        inventory,
        expected_root=tmp_path,
        require_nonempty=True,
        label="test",
    )

    digest_tamper = dict(inventory)
    digest_tamper["inventory_sha256"] = "0" * 64
    with pytest.raises(FINALIZER.FinalizeFailure, match="inventory digest drift"):
        FINALIZER._validate_inventory_block(
            digest_tamper,
            expected_root=tmp_path,
            require_nonempty=True,
            label="test",
        )

    inventory["entries"][0]["size_bytes"] += 1
    with pytest.raises(FINALIZER.FinalizeFailure, match="live metadata drift"):
        FINALIZER._validate_inventory_block(
            inventory,
            expected_root=tmp_path,
            require_nonempty=True,
            label="test",
        )


def test_preflight_inventory_rejects_missing_live_entry_and_prefix_drift(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "mrna_editflow_valid"
    artifact.write_text("evidence\n", encoding="utf-8")
    inventory = PREFLIGHT.top_level_inventory(tmp_path)
    artifact.unlink()
    with pytest.raises(FINALIZER.FinalizeFailure, match="entry no longer exists"):
        FINALIZER._validate_inventory_block(
            inventory,
            expected_root=tmp_path,
            require_nonempty=True,
            label="test",
        )

    (tmp_path / "unrelated-root").mkdir()
    prefix_inventory = PREFLIGHT.top_level_inventory(tmp_path)
    with pytest.raises(FINALIZER.FinalizeFailure, match="entry prefix drift"):
        FINALIZER._validate_inventory_block(
            prefix_inventory,
            expected_root=tmp_path,
            require_nonempty=True,
            label="test",
            required_name_prefix="mrna_editflow",
        )
