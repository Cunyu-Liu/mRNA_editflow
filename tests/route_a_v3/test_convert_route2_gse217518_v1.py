from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts/route_a_v3/convert_route2_gse217518_v1.py"
CONFIG_PATH = REPO_ROOT / "configs/route_a_v3_route2_gse217518_converter_v1.json"


def _module():
    spec = importlib.util.spec_from_file_location("convert_route2_gse217518_v1", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _construct(row: int, source: str, candidate: str, hek_wt=10.0, hek_mt=12.0, sh_wt=None, sh_mt=None) -> dict:
    return {
        "supplement_row_number": row,
        "mutant_identifier": f"mutant-{row}",
        "variant_name": "NM_1.1(GENE):c.-1A>C",
        "gene": "GENE",
        "transcript_accession": "NM_1",
        "region": "5UTR",
        "source_sequence": source,
        "candidate_sequence": candidate,
        "window_start_zero_based": 0,
        "edit_position_zero_based": 1,
        "edit_ref": source[1],
        "edit_alt": candidate[1],
        "t05_WT_HEK": hek_wt,
        "t05_mt_HEK": hek_mt,
        "t05_WT_SH": sh_wt,
        "t05_mt_SH": sh_mt,
    }


def test_construct_crop_centers_variant_and_clamps_at_boundary() -> None:
    module = _module()
    source = "A" * 200
    middle_candidate = source[:100] + "C" + source[101:]
    middle = module._crop_construct(source, middle_candidate, 115, 57)
    assert middle is not None
    middle_source, middle_mutant, middle_start = middle
    assert middle_start == 43
    assert len(middle_source) == len(middle_mutant) == 115
    assert next(i for i, (a, b) in enumerate(zip(middle_source, middle_mutant)) if a != b) == 57
    boundary_candidate = source[:2] + "C" + source[3:]
    boundary = module._crop_construct(source, boundary_candidate, 115, 57)
    assert boundary is not None and boundary[2] == 0


def test_gc_fraction_must_match_fragment_plus_publisher_primers() -> None:
    module = _module()
    config = _config()
    fragment = "A" * 115
    primers = config["input"]["primer_sequences"]["5UTR"]
    primer_gc = sum(base in "GC" for primer in primers for base in primer)
    exact_fraction = primer_gc / 155
    assert module._gc_construct_exact(fragment, exact_fraction, primers) == (True, "COMPOSITION")
    assert module._gc_construct_exact(fragment, (primer_gc + 1) / 155, primers) == (False, "COMPOSITION")
    assert module._gc_construct_exact(fragment, 0.123, primers) == (False, "DENOMINATOR")


def test_conflicting_exact_sequence_context_units_are_rejected_not_averaged() -> None:
    module = _module()
    source = "AAGT"
    candidate = "ACGT"
    constructs = [
        _construct(2, source, candidate, hek_wt=10.0, hek_mt=12.0),
        _construct(3, source, candidate, hek_wt=10.0, hek_mt=30.0),
        _construct(4, "TTAA", "TCAA", hek_wt=5.0, hek_mt=6.0),
    ]
    units, slots = module._endpoint_units(constructs)
    resolved, conflict_units, conflict_rows, exact_duplicates = module._resolve_units(units)
    assert slots == 3
    assert conflict_units == 1 and conflict_rows == 2
    assert exact_duplicates == 0
    assert len(resolved) == 1
    assert resolved[0]["direction_normalized_delta"] == 1.0


def test_missing_endpoint_is_absent_not_zero_filled() -> None:
    module = _module()
    constructs = [_construct(2, "AAGT", "ACGT", hek_wt=10.0, hek_mt=12.0, sh_wt=None, sh_mt=None)]
    units, slots = module._endpoint_units(constructs)
    resolved, conflict_units, _, _ = module._resolve_units(units)
    assert slots == 1 and conflict_units == 0 and len(resolved) == 1
    assert resolved[0]["biological_context_id"] == "HEK293T"
    assert resolved[0]["source_endpoint_value"] == 10.0
    assert resolved[0]["candidate_endpoint_value"] == 12.0


def test_config_preserves_sub_only_partial_scope_no_credit_and_no_overwrite(tmp_path: Path) -> None:
    module = _module()
    config = _config()
    module.validate_config(config)
    assert config["action_policy"] == {"allowed_candidate_action": "SUB", "ins_supported": False, "del_supported": False}
    assert config["study"]["conversion_scope"] == "STRICT_PUBLIC_RECONSTRUCTION_SUB_ONLY_PARTIAL"
    assert config["development_policy"]["near_duplicate_split_status"] == "NOT_RUN"
    assert config["development_policy"]["missing_standard_error_representation"] is None
    assert not any(config["credit_policy"]["qualified_credit_delta"].values())
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    marker = output_dir / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(module.ConversionError, match="already exists"):
        module.execute(config, tmp_path / "missing.xlsx", tmp_path / "missing.gb", output_dir)
    assert marker.read_text(encoding="utf-8") == "keep"
