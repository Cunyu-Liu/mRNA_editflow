from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/route_a_v3/count_route2_emtab10902_sample_v1.py"


def _load():
    spec = importlib.util.spec_from_file_location("count_route2_emtab10902_sample_v1_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_successful_slice_becomes_visible_only_after_counter_exit(tmp_path: Path, monkeypatch) -> None:
    module = _load()

    def fake_run(command, **_kwargs):
        partial = Path(command[-2])
        assert partial.name.endswith(".txt.partial")
        assert not (tmp_path / "S.counts.7.txt").exists()
        partial.write_text("NZSEQ00001\t1\t1\n", encoding="utf-8")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    result = module._run_slice("cp", tmp_path / "library.fa", tmp_path / "reads.fastq.gz", tmp_path, "S", 100, 7)
    assert result == "S.counts.7.txt"
    assert (tmp_path / result).read_text(encoding="utf-8") == "NZSEQ00001\t1\t1\n"
    assert not (tmp_path / "S.counts.7.txt.partial").exists()


def test_failed_slice_keeps_partial_evidence_without_final_name(tmp_path: Path, monkeypatch) -> None:
    module = _load()

    def fake_run(command, **_kwargs):
        Path(command[-2]).write_text("partial evidence\n", encoding="utf-8")
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    with pytest.raises(subprocess.CalledProcessError):
        module._run_slice("cp", tmp_path / "library.fa", tmp_path / "reads.fastq.gz", tmp_path, "S", 100, 7)
    assert (tmp_path / "S.counts.7.txt.partial").read_text(encoding="utf-8") == "partial evidence\n"
    assert not (tmp_path / "S.counts.7.txt").exists()
