from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

from core.route2_mrnabert_edit_site_features_v3 import (
    ChunkSpan,
    EncodedSequenceFeaturesV3,
    PositionFeature,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts/route_a_v3/run_route2_xeditcritic_v3_atomic_frozen_test.py"
PROTOCOL = REPO_ROOT / "configs/route_a_v3_route2_xeditcritic_v3_frozen_test_protocol_v1.json"
DESCRIPTORS = REPO_ROOT / "configs/route_a_v3_route2_endpoint_descriptors_v1.json"


def _module():
    spec = importlib.util.spec_from_file_location("atomic_xeditcritic_v3_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_atomic_authorization_requires_three_seed_pass() -> None:
    module = _module()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    gate = {
        "status": "XEDITCRITIC_V3_THREE_SEED_PASS",
        "development_test_authorized": True,
        "selected_arm": "C2",
        "required_seeds": [20260831, 20260901, 20260902],
    }
    assert module.require_atomic_test_authorization_v3(protocol, gate) == (
        "C2", (20260831, 20260901, 20260902)
    )
    gate["status"] = "XEDITCRITIC_V3_THREE_SEED_NO_GO"
    with pytest.raises(Exception, match="does not authorize"):
        module.require_atomic_test_authorization_v3(protocol, gate)


def test_authorized_loader_decodes_only_test_rows_and_persists_no_projection(tmp_path: Path) -> None:
    module = _module()
    manifest = tmp_path / "manifest.jsonl"
    canonical = tmp_path / "canonical.jsonl"
    manifest.write_text(
        "".join(
            json.dumps({
                "canonical_record_id": record_id,
                "pool_assignment": "DEVELOPMENT",
                "split": split,
                "study_unit_id": "GSE149487",
                "connected_source_component_id": f"component::{record_id}",
            }) + "\n"
            for record_id, split in (("train", "TRAIN"), ("test", "TEST"))
        )
    )
    test_row = {
        "canonical_record_id": "test",
        "pool_assignment": "DEVELOPMENT",
        "study_unit_id": "GSE149487",
        "source_id": "source",
        "source_sequence": "AAAA",
        "candidate_sequence": "ACAA",
        "region": "5UTR",
        "assay_id": "PLUMAGE_BARCODE_MPRA",
        "biological_context_id": "293T",
        "endpoint_id": "te_log2_polysome_over_totalrna",
        "direction_normalized_delta": 1.25,
    }
    canonical.write_text(
        '{"canonical_record_id":"train", "direction_normalized_delta":NOT_JSON}\n'
        + json.dumps(test_row) + "\n"
    )
    with pytest.raises(Exception, match="not consumed"):
        module.load_authorized_test_rows_v3(
            manifest_path=manifest, canonical_paths=[canonical],
            endpoint_descriptor_path=DESCRIPTORS, authorization_consumed=False,
        )
    rows = module.load_authorized_test_rows_v3(
        manifest_path=manifest, canonical_paths=[canonical],
        endpoint_descriptor_path=DESCRIPTORS, authorization_consumed=True,
    )
    assert len(rows) == 1
    assert rows[0]["split"] == "TEST"
    assert rows[0]["direction_normalized_delta"] == 1.25
    assert list(tmp_path.glob("*projection*")) == []


def test_authorized_feature_view_is_ephemeral_and_record_aligned() -> None:
    module = _module()
    feature_a = PositionFeature(
        site=torch.ones(768), window_mean=torch.ones(768) * 2,
        window_max=torch.ones(768) * 3, chunk=ChunkSpan(0, 4),
    )
    feature_c = PositionFeature(
        site=torch.ones(768) * 4, window_mean=torch.ones(768) * 5,
        window_max=torch.ones(768) * 6, chunk=ChunkSpan(0, 4),
    )
    encoded = {
        0: EncodedSequenceFeaturesV3(torch.ones(768) * 7, {1: feature_a}, (ChunkSpan(0, 4),)),
        1: EncodedSequenceFeaturesV3(torch.ones(768) * 8, {1: feature_c}, (ChunkSpan(0, 4),)),
    }
    row = {
        "canonical_record_id": "test", "source_sequence": "AAAA",
        "candidate_sequence": "ACAA", "source_relative_edits": [{"position": 1}],
    }
    view = module._AuthorizedTestFeatureViewV3([row], encoded, {"AAAA": 0, "ACAA": 1})
    bundle = view.bundle("test")
    assert bundle["edit_positions"].tolist() == [1]
    assert bundle["source_site"].shape == bundle["candidate_site"].shape == (1, 768)
    assert bundle["source_site"][0, 0].item() == 1.0
    assert bundle["candidate_site"][0, 0].item() == 4.0
    assert bundle["source_global"][0].item() == 7.0
    assert bundle["candidate_global"][0].item() == 8.0
