"""Pure-Python GP0 protocol tests; no CUDA or controlled data required."""

from __future__ import annotations

import json
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    GP0GateError,
    canonical_region,
    canonical_sequence,
    load_split_binding,
)


def test_region_display_forms_are_canonicalised() -> None:
    assert canonical_region("5'UTR") == "5UTR"
    assert canonical_region("3′UTR") == "3UTR"
    assert canonical_region("5UTR") == "5UTR"


def test_dna_to_rna_policy_is_explicit_and_counted() -> None:
    sequence, converted = canonical_sequence("actt", policy="dna_t_to_rna_u")
    assert sequence == "ACUU"
    assert converted == 2


def test_strict_rna_rejects_thymine() -> None:
    try:
        canonical_sequence("AC T".replace(" ", ""), policy="strict_rna")
    except GP0GateError as error:
        assert "canonical RNA alphabet" in str(error)
    else:  # pragma: no cover
        raise AssertionError("strict_rna accepted thymine")


def test_split_binding_rejects_conflicting_roles(tmp_path: Path) -> None:
    split = tmp_path / "split.jsonl"
    rows = [
        {"record_id": "r1", "accession": "GSE1", "region": "5'UTR", "split": "train", "split_type": "study"},
        {"record_id": "r1", "accession": "GSE1", "region": "5UTR", "split": "test", "split_type": "source"},
    ]
    split.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    try:
        load_split_binding([split])
    except GP0GateError as error:
        assert "conflicting train/val/test" in str(error)
    else:  # pragma: no cover
        raise AssertionError("conflicting split roles were accepted")
