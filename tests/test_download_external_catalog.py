"""Unit tests for data.download_external_catalog (no network access)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from data.download_external_catalog import (
    EXTERNAL_CATALOG,
    PRIORITIES,
    main,
    select_datasets,
    write_manifest,
)

REQUIRED_ENTRY_KEYS = {
    "priority", "description", "citation", "license", "cleaning_spec", "files",
}


class TestRegistryIntegrity:
    def test_all_entries_have_required_keys(self):
        for name, entry in EXTERNAL_CATALOG.items():
            missing = REQUIRED_ENTRY_KEYS - set(entry)
            assert not missing, f"{name} missing keys: {missing}"

    def test_priorities_valid(self):
        for name, entry in EXTERNAL_CATALOG.items():
            assert entry["priority"] in PRIORITIES, f"{name}: {entry['priority']}"

    def test_every_file_has_url_and_filename(self):
        for name, entry in EXTERNAL_CATALOG.items():
            assert entry["files"], f"{name} has no files"
            for f in entry["files"]:
                assert set(f) == {"url", "filename"}, f"{name}: bad file record {f}"
                assert f["url"].startswith("https://"), f"{name}: non-https {f['url']}"
                assert f["filename"] and "/" not in f["filename"]

    def test_filenames_unique_within_entry(self):
        for name, entry in EXTERNAL_CATALOG.items():
            fns = [f["filename"] for f in entry["files"]]
            assert len(fns) == len(set(fns)), f"{name}: duplicate filenames"

    def test_expected_shard_counts(self):
        assert len(EXTERNAL_CATALOG["refseq_human_mrna_prot"]["files"]) == 30
        assert len(EXTERNAL_CATALOG["refseq_mammalian_cds"]["files"]) == 278
        assert len(EXTERNAL_CATALOG["mrnabert_downstream_zenodo"]["files"]) == 7

    def test_each_priority_tier_nonempty(self):
        for tier in PRIORITIES:
            assert any(e["priority"] == tier for e in EXTERNAL_CATALOG.values())


class TestSelectDatasets:
    def test_all_returns_everything(self):
        assert select_datasets(["all"]) == list(EXTERNAL_CATALOG)

    def test_priority_tier(self):
        got = select_datasets(["p0"])
        expected = [n for n, e in EXTERNAL_CATALOG.items() if e["priority"] == "P0"]
        assert got == expected and got

    def test_tier_case_insensitive(self):
        assert select_datasets(["P1"]) == select_datasets(["p1"])

    def test_explicit_names(self):
        got = select_datasets(["rfam_seed", "bprna_hf"])
        assert got == ["rfam_seed", "bprna_hf"]

    def test_mixed_tier_and_name_dedup(self):
        got = select_datasets(["p2", "rfam_seed"])
        assert got == [n for n, e in EXTERNAL_CATALOG.items() if e["priority"] == "P2"]

    def test_empty_selector_raises(self):
        with pytest.raises(ValueError):
            select_datasets([])

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            select_datasets(["not_a_dataset"])


class TestWriteManifest:
    def _entry(self):
        return {
            "priority": "P9",
            "description": "d",
            "citation": "c",
            "license": "l",
            "cleaning_spec": "s",
        }

    def test_round_trip(self, tmp_path: Path):
        records = [
            {"filename": "a.gz", "url": "https://x/a.gz", "sha256": "0" * 64, "byte_size": 3},
            {"filename": "b.gz", "url": "https://x/b.gz", "sha256": "1" * 64, "byte_size": 5},
        ]
        path = write_manifest(tmp_path, "ds", self._entry(), records)
        data = json.loads(path.read_text())
        assert data["dataset_name"] == "ds"
        assert data["n_files"] == 2
        assert data["total_bytes"] == 8
        assert data["acquisition_date_utc"].endswith("Z")
        for key in ("citation", "license", "cleaning_spec", "priority", "description"):
            assert key in data
        assert all({"filename", "url", "sha256", "byte_size"} <= set(r) for r in data["files"])

    def test_empty_file_list(self, tmp_path: Path):
        path = write_manifest(tmp_path, "ds", self._entry(), [])
        data = json.loads(path.read_text())
        assert data["n_files"] == 0 and data["total_bytes"] == 0


class TestCli:
    def test_list_outputs_all_datasets(self, capsys):
        assert main(["--list"]) == 0
        out = capsys.readouterr().out
        for name in EXTERNAL_CATALOG:
            assert name in out

    def test_missing_datasets_errors(self):
        with pytest.raises(SystemExit):
            main([])
