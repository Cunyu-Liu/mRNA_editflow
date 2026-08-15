from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
from openpyxl import Workbook


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts/route_a_v3/convert_route2_encsr854ruf_v1.py"
CONFIG_PATH = REPO_ROOT / "configs/route_a_v3_route2_encsr854ruf_converter_v1.json"


def _module():
    spec = importlib.util.spec_from_file_location("convert_route2_encsr854ruf_v1", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_fasta_slash_alias_expansion_keeps_valid_records_only(tmp_path: Path) -> None:
    module = _module()
    fasta = tmp_path / "input.fasta"
    fasta.write_text(">ref/ref_alias\n" + "A" * 133 + "\n>bad\n" + "N" * 133 + "\n", encoding="utf-8")
    aliases, stats, conflicts = module._load_fasta_aliases(fasta, 133)
    assert aliases["ref"] == "A" * 133 and aliases["ref_alias"] == "A" * 133
    assert "bad" not in aliases
    assert stats["header_count"] == 2 and stats["valid_record_count"] == 1 and stats["invalid_record_count"] == 1
    assert stats["expanded_alias_token_count"] == 3 and stats["valid_alias_token_count"] == 2
    assert conflicts == 0


def test_pair_geometry_replays_declared_sub_and_marks_publisher_index_error() -> None:
    module = _module()
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Oligo Variant Info"
    header = ["mpra_variant_id", "tag", "oligo_id", "variant_id", "chrom", "oligo_starts", "oligo_ends", "strand", "var_start", "var_end", "ref_allele", "alt_allele", "genes", "transcripts", "gene_symbols", "other_var_in_oligo_window"]
    sheet.append(header)
    common = ["pair1", None, None, "v1", 1, 1, 133, "+", 10, 10, "A", "C", "g", "t", "GENE", "index_error"]
    ref = list(common); ref[1] = "ref"; ref[2] = "ref_id"
    alt = list(common); alt[1] = "alt"; alt[2] = "alt_id"
    sheet.append(ref); sheet.append(alt)
    source = "A" * 133
    candidate = source[:9] + "C" + source[10:]
    pairs, index_errors, stats = module._pair_geometry(workbook, {"ref_id": source, "alt_id": candidate})
    assert set(pairs) == {"pair1"}
    assert index_errors == {"pair1"}
    assert stats["action_SUB"] == 1
    assert stats["sub_declared_allele_replay_mismatch_count"] == 0


def test_exact_duplicate_units_collapse_but_conflicting_units_are_rejected() -> None:
    module = _module()
    source = "A" * 133
    candidate = source[:5] + "C" + source[6:]
    base = {"source_sequence": source, "candidate_sequence": candidate, "edit_position_zero_based": 5, "supplement_row_number": 2, "metadata": {}, "action": "SUB"}
    pair1 = dict(base, pair_id="pair1", endpoints={"HEK293FT": {"effect": 1.0, "standard_error": 0.2}, "HEPG2": {"effect": 2.0, "standard_error": 0.3}})
    pair2 = dict(base, pair_id="pair2", supplement_row_number=3, endpoints={"HEK293FT": {"effect": 1.0, "standard_error": 0.2}, "HEPG2": {"effect": 9.0, "standard_error": 0.3}})
    resolved, stats = module._resolve_units([pair1, pair2])
    assert stats["endpoint_slot_count_before_exact_dedup"] == 4
    assert stats["exact_duplicate_unit_count"] == 1
    assert stats["conflicting_exact_unit_count"] == 1
    assert stats["conflicting_endpoint_row_count"] == 2
    assert len(resolved) == 1 and resolved[0]["biological_context_id"] == "HEK293FT"


def test_canonical_preserves_formal_skew_and_se_without_fake_absolute_endpoints() -> None:
    module = _module()
    config = _config()
    source = "A" * 133
    candidate = source[:5] + "C" + source[6:]
    resolved = [{
        "pair_id": "pair1", "source_sequence": source, "candidate_sequence": candidate,
        "edit_position_zero_based": 5, "supplement_row_number": 2,
        "metadata": {"variant_id": "v1", "gene_symbols": "GENE", "strand": "+"},
        "biological_context_id": "HEK293FT", "effect": 1.25, "standard_error": 0.4,
    }]
    record = module._canonical_records(config, resolved)[0]
    assert record["direction_normalized_delta"] == 1.25
    assert record["biological_standard_error"] == 0.4
    assert record["source_endpoint_value"] is None and record["candidate_endpoint_value"] is None
    assert record["historical_exposure_status"] == "KNOWN_EXPOSED_DEVELOPMENT_ONLY"
    assert record["edit_operations"] == [{"type": "SUB", "position_zero_based": 5, "ref": "A", "alt": "C"}]


def test_config_preserves_exposure_no_credit_private_output_and_no_overwrite(tmp_path: Path) -> None:
    module = _module()
    config = _config()
    module.validate_config(config)
    assert config["development_policy"]["historical_exposure_status"] == "KNOWN_EXPOSED_DEVELOPMENT_ONLY"
    assert config["output"]["public_redistribution_allowed"] is False
    assert config["development_policy"]["near_duplicate_split_status"] == "NOT_RUN"
    assert not any(config["credit_policy"]["qualified_credit_delta"].values())
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    marker = output_dir / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(module.ConversionError, match="already exists"):
        module.execute(config, tmp_path / "missing.xlsx", tmp_path / "missing.fasta", output_dir)
    assert marker.read_text(encoding="utf-8") == "keep"
