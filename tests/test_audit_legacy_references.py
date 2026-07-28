"""Unit tests for scripts/contracts/audit_legacy_references.py."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.contracts.audit_legacy_references import (  # noqa: E402
    LEGACY_PATTERNS, _is_excluded, iter_active_files, main, scan)

# Build legacy-looking strings dynamically so this test file does not itself
# textually contain a live legacy-contract reference.
LEGACY_DOC = "docs/" + "p3_" + "00_frozen_scientific_question.md"
LEGACY_CFG = "configs/" + "p3_" + "frozen_research_contract.yaml"


def _write(root: Path, rel: str, text: str):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def test_detects_reference_in_active_code(tmp_path):
    _write(tmp_path, "scripts/train.py", f"path = '{LEGACY_DOC}'\n")
    _write(tmp_path, "docs/new_contract.md", f"see {LEGACY_DOC}\n")
    violations = scan(tmp_path)
    assert len(violations) == 2
    files = {v[0] for v in violations}
    assert files == {"scripts/train.py", "docs/new_contract.md"}


def test_config_reference_matches_two_patterns(tmp_path):
    """A legacy config path hits both the path and the name pattern."""
    _write(tmp_path, "scripts/train.py", f"path = '{LEGACY_CFG}'\n")
    violations = scan(tmp_path)
    assert len(violations) == 2
    assert {v[2] for v in violations} == {"configs/p3_", "p3_frozen_research_contract"}


def test_archive_and_data_dirs_are_excluded(tmp_path):
    _write(tmp_path, "docs/archive/p3_legacy/old.md", f"{LEGACY_DOC}\n")
    _write(tmp_path, "configs/archive/p3_legacy/old.yaml", f"{LEGACY_CFG}\n")
    _write(tmp_path, "scripts/archive/p3_legacy/old.py", f"{LEGACY_CFG}\n")
    _write(tmp_path, "artifacts/audit.json", f'{{"artifact": "{LEGACY_DOC}"}}\n')
    _write(tmp_path, "data/p3/manifest.json", f'{{"p": "{LEGACY_CFG}"}}\n')
    _write(tmp_path, "results/p3/r.json", f'{{"p": "{LEGACY_CFG}"}}\n')
    assert scan(tmp_path) == []


def test_clean_tree_has_zero_violations(tmp_path):
    _write(tmp_path, "configs/public_intervention_contract.yaml", "contract_id: x\n")
    _write(tmp_path, "docs/public_intervention_scientific_question.md", "q\n")
    assert scan(tmp_path) == []


def test_exclusion_helper(tmp_path):
    assert _is_excluded("docs/archive/p3_legacy/p3_00_x.md")
    assert _is_excluded("artifacts/nmi_artifact_audit.json")
    assert not _is_excluded("scripts/train.py")


def test_patterns_match_expected_forms():
    pats = [p.pattern for p in LEGACY_PATTERNS]
    assert any("configs/" in p for p in pats)
    assert any("docs/" in p for p in pats)


def test_main_strict_exit_codes(tmp_path, monkeypatch, capsys):
    import scripts.contracts.audit_legacy_references as audit

    monkeypatch.setattr(audit, "REPO_ROOT", tmp_path)
    _write(tmp_path, "scripts/ok.py", "x = 1\n")
    assert main(["--strict"]) == 0
    out = capsys.readouterr().out
    assert "active paper code references to legacy contract = 0" in out

    _write(tmp_path, "scripts/bad.py", f"p = '{LEGACY_DOC}'\n")
    assert main(["--strict"]) == 1
    out = capsys.readouterr().out
    assert "active paper code references to legacy contract = 1" in out


def test_repo_active_tree_is_clean():
    """The real repository must satisfy the R0-01 acceptance."""
    assert main(["--strict"]) == 0
