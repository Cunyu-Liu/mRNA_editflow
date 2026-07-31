"""Unit tests for D0-04 missing-dataset management artifacts."""
from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DOC = REPO_ROOT / "docs/data/missing_dataset_acquisition.md"
REGISTRY = REPO_ROOT / "data_registry/unavailable.yaml"

ALLOWED_STATUSES = {
    "retry_pending",
    "mirror_pending",
    "archive_pending",
    "reconstruction_pending",
    "author_contact_pending",
    "documented_unavailable",
}

REQUIRED_FIELDS = {
    "dataset_id",
    "paper",
    "searched_locations",
    "supplemental",
    "author_code",
    "archive",
    "needs_author_contact",
    "reconstructable_from_raw",
    "current_substitute",
    "status",
    "last_checked_utc",
}

ESCALATION_ORDER = [
    "retry",
    "alternate mirror",
    "archive",
    "raw reconstruction",
    "author contact",
    "documented unavailable",
]


def test_protocol_doc_exists_and_documents_full_ladder():
    text = DOC.read_text(encoding="utf-8")
    for rung in ESCALATION_ORDER:
        assert rung in text, f"protocol doc missing escalation rung: {rung}"
    # ladder order must be exactly retry -> ... -> documented unavailable
    positions = [text.index(rung) for rung in ESCALATION_ORDER]
    assert positions == sorted(positions), "escalation rungs out of order"


def test_registry_is_valid_yaml_with_required_keys():
    data = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    assert data["registry_version"]
    assert data["contract_id"] == "mrna_editflow_single_active_contract"
    assert isinstance(data["datasets"], list)


def test_every_record_has_required_fields_and_legal_status():
    data = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    for rec in data["datasets"]:
        missing = REQUIRED_FIELDS - set(rec)
        assert not missing, f"{rec.get('dataset_id')}: missing fields {missing}"
        assert rec["status"] in ALLOWED_STATUSES, (
            f"{rec['dataset_id']}: illegal status {rec['status']}")
        assert isinstance(rec["searched_locations"], list)
        assert rec["searched_locations"], (
            f"{rec['dataset_id']}: searched_locations must be non-empty")
        assert isinstance(rec["needs_author_contact"], bool)
        assert isinstance(rec["reconstructable_from_raw"], bool)


def test_documented_unavailable_requires_author_contact_evidence():
    data = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    for rec in data["datasets"]:
        if rec["status"] == "documented_unavailable":
            assert rec["needs_author_contact"], (
                f"{rec['dataset_id']}: documented_unavailable without author contact")
