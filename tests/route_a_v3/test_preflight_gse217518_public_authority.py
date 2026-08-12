from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = (
    ROOT
    / "configs"
    / "route_a_v3_gse217518_public_authority_preflight_v1.json"
)
MODULE_PATH = (
    ROOT
    / "scripts"
    / "route_a_v3"
    / "preflight_gse217518_public_authority.py"
)
SPEC = importlib.util.spec_from_file_location(
    "gse217518_public_authority_preflight", MODULE_PATH
)
assert SPEC and SPEC.loader
PREFLIGHT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PREFLIGHT
SPEC.loader.exec_module(PREFLIGHT)


def _protocol() -> dict[str, object]:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    PREFLIGHT._validate_protocol(protocol)
    return protocol


def _fixture_binding(*args: object) -> dict[str, str]:
    return {
        "status": "TEST_FIXTURE_BOUND_WITHOUT_GIT",
        "base_commit": "0" * 40,
        "implementation_commit": "1" * 40,
        "binding_commit": "2" * 40,
    }


class StaticFetcher:
    def __init__(self, geo: str, article: str) -> None:
        self.geo = geo
        self.article = article
        self.urls: list[str] = []

    def fetch_text(self, url: str) -> str:
        self.urls.append(url)
        if "ncbi.nlm.nih.gov/geo/" in url:
            return self.geo
        if "elifesciences.org/articles/97682" in url:
            return self.article
        raise AssertionError(f"unexpected URL: {url}")


def _official_geo_text() -> str:
    protocol = _protocol()
    filenames = [item["filename"] for item in protocol["official_processed_assets"]]
    return " ".join(
        [
            "GSE217518 wild-type mutant half-life weighted linear regression",
            *filenames,
            *PREFLIGHT.EXPECTED_SAMPLE_NAMES,
        ]
    )


def _official_article_text() -> str:
    return (
        "eLife 97682 Half-life estimation used mean squared error and linear "
        "models. For each variant a 115 bp UTR fragment was built."
    )


def _live_observation() -> dict[str, object]:
    protocol = _protocol()
    observation, source_results = PREFLIGHT.build_live_observation(
        protocol, StaticFetcher(_official_geo_text(), _official_article_text())
    )
    assert {source["status"] for source in source_results} == {PREFLIGHT.PASS}
    return observation


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def test_protocol_freezes_exact_four_assets_current_stop_and_normal_binding() -> None:
    protocol = _protocol()
    binding = protocol["implementation_binding"]
    assert binding["base_commit"] == "7b9a9531d44a8398153cc3900bb738e92730af99"
    assert binding["status"] == PREFLIGHT.UNKNOWN
    assert {
        path.rsplit(".", 1)[1]
        for path in binding["unknown_to_bound_scalar_paths"]
    } == set(PREFLIGHT.UNKNOWN_BINDING_SCALARS)
    assert len(protocol["official_processed_assets"]) == 4
    assert {
        (item["region"], item["cell_line"])
        for item in protocol["official_processed_assets"]
    } == {
        ("3UTR", "HEK293T"),
        ("5UTR", "HEK293T"),
        ("3UTR", "SH-SY5Y"),
        ("5UTR", "SH-SY5Y"),
    }
    assert protocol["endpoint_authority"]["status"] == PREFLIGHT.PASS
    assert protocol["author_defined_outlier_policy"]["status"] == PREFLIGHT.BLOCKED
    assert protocol["replicate_authority"]["status"] == PREFLIGHT.PASS
    assert protocol["biological_grouping_authority"]["status"] == PREFLIGHT.BLOCKED
    assert protocol["full_context_reconstruction_authority"]["status"] == (
        PREFLIGHT.BLOCKED
    )
    assert protocol["current_authority_assessment"] == {
        "official_asset_listing": "PASS",
        "endpoint_identity_direction_raw_scale": "PASS",
        "author_defined_outlier_policy": "BLOCKED",
        "replicate_authority": "PASS",
        "biological_grouping_fields": "BLOCKED",
        "full_context_reconstruction": "BLOCKED",
        "ready_for_ordinary_public_row_level_producer": False,
        "terminal_status": "STOP_BEFORE_ORDINARY_PUBLIC_ROW_LEVEL_PRODUCER",
    }


def test_live_official_metadata_stops_before_any_payload(tmp_path: Path) -> None:
    fetcher = StaticFetcher(_official_geo_text(), _official_article_text())
    output_dir = tmp_path / "output"
    report = PREFLIGHT.execute(
        PROTOCOL_PATH,
        output_dir,
        fetcher=fetcher,
        binding_auditor=_fixture_binding,
        recorded_at="2026-08-12T12:00:00Z",
    )

    assert len(fetcher.urls) == 2
    assert all(
        "GSE217518_" not in url or "acc=GSE217518" in url
        for url in fetcher.urls
    )
    assert report["status"] == "STOP_BEFORE_ORDINARY_PUBLIC_ROW_LEVEL_PRODUCER"
    assert report["ready_for_ordinary_public_row_level_producer"] is False
    statuses = {gate["gate_id"]: gate["status"] for gate in report["gates"]}
    assert statuses == {
        "ORDINARY_PUBLIC_AGGREGATE_SCOPE": "PASS",
        "OFFICIAL_U3_U5_BY_HEK_SH_ASSET_LISTING": "PASS",
        "ENDPOINT_IDENTITY_DIRECTION_AND_RAW_SCALE": "PASS",
        "AUTHOR_DEFINED_OUTLIER_POLICY": "BLOCKED",
        "REPLICATE_AUTHORITY": "PASS",
        "BIOLOGICAL_GROUPING_FIELDS": "BLOCKED",
        "MEASURED_FULL_CONTEXT_RECONSTRUCTION": "BLOCKED",
    }
    assert report["gate_counts"] == {"PASS": 4, "BLOCKED": 3, "UNKNOWN_NOT_ASSERTED": 0}
    assert report["scope_attestation"] == {
        "ordinary_public_only": True,
        "aggregate_only": True,
        "row_values_read": False,
        "sequence_values_read": False,
        "effect_values_read": False,
        "processed_asset_download_count": 0,
        "processed_asset_open_count": 0,
        "supplement_payload_open_count": 0,
        "restricted_or_sealed_required": False,
        "restricted_or_sealed_contact": False,
        "GSE246381_contact": False,
        "reconstruction_run_count": 0,
        "qualifier_run_count": 0,
        "canonical_materialization_count": 0,
        "training_run_count": 0,
        "model_selection_run_count": 0,
    }
    files = list(output_dir.iterdir())
    assert [path.name for path in files] == [PREFLIGHT.REPORT_FILENAME]
    serialized = files[0].read_text(encoding="utf-8")
    assert "source_sequence" not in serialized
    assert "candidate_sequence" not in serialized
    assert '"effect_value":' not in serialized
    assert report["terminal_truth"]["qualified"] is False
    assert report["terminal_truth"]["training_allowed"] is False


def test_existing_aggregate_go_requires_explicit_public_rule_group_and_115bp_context(
    tmp_path: Path,
) -> None:
    protocol = _protocol()
    observation = _live_observation()
    observation["header_names"] = [
        "variant_pair_id",
        "transcript_accession_version",
        "gene_or_locus_id",
        "utr_region",
    ]
    observation["author_defined_outlier_policy"] = {
        "status": "PASS",
        "author_defined_rule": "AUTHOR_RULE_WITH_EXPLICIT_MEMBERSHIP_CRITERION",
        "applies_unambiguously_to_ordinary_row_level_effect_production": True,
        "requires_row_value_inference": False,
        "public_authority_ids": ["ELIFE_97682_VERSION_OF_RECORD"],
    }
    observation["biological_grouping_authority"] = {
        "status": "PASS",
        "authoritative_fields": protocol["biological_grouping_authority"][
            "required_source_level_fields"
        ],
        "executable_source_group_definition": (
            "GROUP_BY_VERSIONED_TRANSCRIPT_AND_EXACT_WT_ASSAYED_FRAGMENT"
        ),
        "public_authority_ids": [
            "ELIFE_97682_SUPPLEMENTARY_FILE_1",
            "NCBI_REFSEQ_NUCCORE",
        ],
    }
    observation["full_context_reconstruction_authority"] = {
        "status": "PASS",
        "verified_public_assets": protocol[
            "full_context_reconstruction_authority"
        ]["required_public_assets"],
        "measured_construct_context": protocol[
            "full_context_reconstruction_authority"
        ]["measured_construct_context"],
        "reconstruction_rule": (
            "RECONSTRUCT_THE_AUTHOR_DEFINED_115_BP_BOUNDARY_AWARE_WT_AND_MUTANT_FRAGMENTS"
        ),
        "source_candidate_crosswalk_rule": (
            "JOIN_THE_VARIANT_PAIR_TO_ONE_VERSIONED_TRANSCRIPT_AND_EXACT_WT_FRAGMENT"
        ),
        "produces_entire_refseq_utr_as_measured_context": False,
        "public_authority_ids": [
            "ELIFE_97682_VERSION_OF_RECORD",
            "ELIFE_97682_SUPPLEMENTARY_FILE_1",
            "NCBI_REFSEQ_NUCCORE",
        ],
    }
    aggregate = tmp_path / "authority.json"
    _write_json(aggregate, observation)

    report = PREFLIGHT.execute(
        PROTOCOL_PATH,
        tmp_path / "go-output",
        authority_aggregate_path=aggregate,
        binding_auditor=_fixture_binding,
        recorded_at="2026-08-12T12:00:00Z",
    )

    assert report["status"] == "READY_FOR_ORDINARY_PUBLIC_ROW_LEVEL_PRODUCER"
    assert report["ready_for_ordinary_public_row_level_producer"] is True
    assert report["gate_counts"] == {"PASS": 7, "BLOCKED": 0, "UNKNOWN_NOT_ASSERTED": 0}
    assert report["terminal_truth"]["qualified"] is False
    assert report["terminal_truth"]["canonical_record_count"] == 0
    assert report["terminal_truth"]["next_phase_authorized"] is False
    assert report["sole_next_action"] == (
        "WRITE_THE_ORDINARY_PUBLIC_ROW_LEVEL_PRODUCER_UNDER_A_SEPARATE_AUTHORIZED_TASK"
    )


@pytest.mark.parametrize("stop_reason", ["ROW_INFERRED_POLICY", "RESTRICTED_REQUIRED"])
def test_go_boundary_stops_on_row_inference_or_nonpublic_dependency(
    stop_reason: str,
) -> None:
    protocol = _protocol()
    observation = _live_observation()
    observation["author_defined_outlier_policy"] = {
        "status": "PASS",
        "author_defined_rule": "VALUE_SELECTED_THRESHOLD",
        "applies_unambiguously_to_ordinary_row_level_effect_production": True,
        "requires_row_value_inference": stop_reason == "ROW_INFERRED_POLICY",
        "public_authority_ids": ["ELIFE_97682_VERSION_OF_RECORD"],
    }
    observation["biological_grouping_authority"] = {
        "status": "PASS",
        "authoritative_fields": protocol["biological_grouping_authority"][
            "required_source_level_fields"
        ],
        "executable_source_group_definition": "EXACT_PUBLIC_SOURCE_GROUP_RULE",
        "public_authority_ids": ["ELIFE_97682_SUPPLEMENTARY_FILE_1"],
    }
    observation["full_context_reconstruction_authority"] = {
        "status": "PASS",
        "verified_public_assets": protocol[
            "full_context_reconstruction_authority"
        ]["required_public_assets"],
        "measured_construct_context": protocol[
            "full_context_reconstruction_authority"
        ]["measured_construct_context"],
        "reconstruction_rule": "EXACT_PUBLIC_115_BP_RULE",
        "source_candidate_crosswalk_rule": "EXACT_PUBLIC_CROSSWALK_RULE",
        "produces_entire_refseq_utr_as_measured_context": False,
        "public_authority_ids": ["ELIFE_97682_VERSION_OF_RECORD"],
    }
    if stop_reason == "RESTRICTED_REQUIRED":
        observation["scope"]["restricted_or_sealed_required"] = True

    report = PREFLIGHT.evaluate_observation(
        protocol,
        observation,
        binding=_fixture_binding(),
        source_mode="EXISTING_OFFICIAL_PUBLIC_AUTHORITY_AGGREGATE_ONLY",
        source_results=[],
        recorded_at="2026-08-12T12:00:00Z",
    )
    assert report["ready_for_ordinary_public_row_level_producer"] is False
    assert report["status"] == "STOP_BEFORE_ORDINARY_PUBLIC_ROW_LEVEL_PRODUCER"


def test_unknown_binding_stops_before_aggregate_or_output(tmp_path: Path) -> None:
    missing_aggregate = tmp_path / "must-not-be-read.json"
    output_dir = tmp_path / "must-not-exist"
    with pytest.raises(PREFLIGHT.BindingNotFrozen, match="config-only-B"):
        PREFLIGHT.execute(
            PROTOCOL_PATH,
            output_dir,
            authority_aggregate_path=missing_aggregate,
        )
    assert not output_dir.exists()
