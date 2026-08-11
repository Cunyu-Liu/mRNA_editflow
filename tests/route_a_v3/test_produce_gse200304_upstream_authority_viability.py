from __future__ import annotations

import copy
import gzip
import hashlib
import importlib.util
import io
import json
import os
import sys
import zipfile
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import pytest


STAGING_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    STAGING_ROOT
    / "scripts/route_a_v3/produce_gse200304_upstream_authority_viability.py"
)
CONFIG = (
    STAGING_ROOT
    / "configs/route_a_v3_gse200304_upstream_authority_viability_v1.json"
)

SPEC = importlib.util.spec_from_file_location("g200_upstream_viability_under_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
producer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = producer
SPEC.loader.exec_module(producer)


ENDPOINT_PARAGRAPH = (
    "The plasmid library was transfected into PC3 cells, chosen to represent the "
    "mCRPC cellular environment similar to that of patient samples. Polysome "
    "profiling was performed on the transfected cells to fractionate mRNA by the "
    "number of attached ribosomes, resulting in distinct monosome-, low polysome-, "
    "and high polysome-associated pools of mRNA. Six biological replicates of the "
    "polysome-associated mRNA pools, plus total mRNA and plasmid DNA extracted from "
    "each sample, were sequenced. TE was calculated on the basis of the ratio of "
    "total polysome- or high polysome-associated mRNA to total mRNA for each 3′ UTR "
    "insert, and wild-type and mutant pairs of 3′ UTR inserts were analyzed for "
    "significant TE changes (false discovery rate [FDR] < 0.10) caused by each "
    "mutation using xtail (Table S3)."
)
METHODS_PARAGRAPH = (
    "For polysome MPRA statistical analysis, xtail103 was used to identify "
    "differentially regulated 3′ UTRs. Translation efficiency was calculated by "
    "total polysome (high polysome + low polysome) to total RNA and high polysome "
    "to total RNA ratios for each 3′ UTR. RNA expression changes, used for internal "
    "control validation, were calculated by total RNA to plasmid DNA ratios. A "
    "ratio of ratios was then calculated to compare Mutant TE to WT TE for each 3′ "
    "UTR mutation and an FDR<0.10 in this comparison was considered significant."
)


def frozen_config() -> dict[str, Any]:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def exact_unknown_i(config: dict[str, Any] | None = None) -> dict[str, Any]:
    result = copy.deepcopy(config or frozen_config())
    result["implementation_binding"].update(
        {
            "status": producer.UNKNOWN,
            "implementation_commit": producer.UNKNOWN,
            "implementation_script_sha256": producer.UNKNOWN,
            "implementation_test_sha256": producer.UNKNOWN,
        }
    )
    producer.validate_static_config(result)
    return result


def bound_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    result = exact_unknown_i(config)
    result["implementation_binding"].update(
        {
            "status": producer.BOUND,
            "implementation_commit": "1" * 40,
            "implementation_script_sha256": "2" * 64,
            "implementation_test_sha256": "3" * 64,
        }
    )
    producer.validate_static_config(result)
    return result


def dummy_predecessor(s3: producer.S3SelectiveState) -> producer.PredecessorSummary:
    return producer.PredecessorSummary(
        published_endpoint_config_sha256="4" * 64,
        published_endpoint_trio_manifest_sha256="5" * 64,
        source_exact7_manifest_sha256="6" * 64,
        published_endpoint_bundle_manifest_sha256="7" * 64,
        source_exact7_member_count=7,
        published_endpoint_bundle_member_count=5,
        s3=s3,
    )


def make_jats() -> bytes:
    article = ET.Element("article")
    front = ET.SubElement(article, "front")
    article_meta = ET.SubElement(front, "article-meta")
    for kind, value in (
        ("pmcid", "PMC10540565"),
        ("pmid", "37516102"),
        ("doi", "10.1016/j.celrep.2023.112840"),
    ):
        element = ET.SubElement(article_meta, "article-id", {"pub-id-type": kind})
        element.text = value
    license_element = ET.SubElement(article_meta, "license")
    license_ref = ET.SubElement(license_element, "license_ref")
    license_ref.text = "https://creativecommons.org/licenses/by/4.0/"
    license_text = ET.SubElement(license_element, "license-p")
    license_text.text = (
        "This is an open access article under the CC BY license "
        "(http://creativecommons.org/licenses/by/4.0/)."
    )

    body = ET.SubElement(article, "body")
    paragraph = ET.SubElement(body, "p")
    paragraph.text = ENDPOINT_PARAGRAPH
    methods = ET.SubElement(body, "p")
    methods.text = METHODS_PARAGRAPH
    for label, rid in (("Table S2", "SD3"), ("Table S3", "SD4")):
        xref = ET.SubElement(
            body,
            "xref",
            {"ref-type": "supplementary-material", "rid": rid},
        )
        xref.text = label
    back = ET.SubElement(article, "back")
    for supplement_id, label, href in (
        ("SD3", "3", "NIHMS1928233-supplement-3.csv"),
        ("SD4", "4", "NIHMS1928233-supplement-4.xlsx"),
    ):
        supplement = ET.SubElement(
            back, "supplementary-material", {"id": supplement_id}
        )
        supplement_label = ET.SubElement(supplement, "label")
        supplement_label.text = label
        ET.SubElement(supplement, "media", {"href": href})
    return ET.tostring(article, encoding="utf-8", xml_declaration=True)


def make_soft_plain() -> bytes:
    roles = ("High_Poly", "Low_Poly", "pDNA", "Total_RNA")
    sample_names = [
        f"GSM{6_030_613 + index}" for index in range(len(roles) * 6)
    ]
    lines = [
        "^SERIES = GSE200302",
        "!Series_geo_accession = GSE200302",
        "!Series_relation = SubSeries of: GSE200304",
        "!Series_relation = BioProject: https://example.invalid/PRJNA824033",
        "!Series_supplementary_file = ftp://example.invalid/GSE200302_Twist_Oligo_Order_with_merged_ids.txt.gz",
        "!Series_supplementary_file = ftp://example.invalid/GSE200302_log2_cpm_counts_all_samples.txt.gz",
    ]
    lines.extend(f"!Series_sample_id = {name}" for name in sample_names)
    sample_index = 0
    for role in roles:
        for replicate in range(1, 7):
            name = sample_names[sample_index]
            sample_index += 1
            lines.extend(
                [
                    f"^SAMPLE = {name}",
                    f"!Sample_geo_accession = {name}",
                    f"!Sample_title = {role}_{replicate}_S{sample_index}",
                    "!Sample_supplementary_file_1 = NONE",
                ]
            )
    return ("\n".join(lines) + "\n").encode("utf-8")


def make_s3_xlsx(
    *, pair_count: int, finite_totalpoly_count: int
) -> tuple[bytes, list[str]]:
    keys = [f"chr1:{100_000 + index}_A-C" for index in range(pair_count)]
    shared = [
        "barcode",
        "Comparison",
        "xtail_log2FC_TE",
        "xtail_pvalue",
        "xtail_FDR",
        "HighPoly:RNA",
        "TotalPoly:RNA",
        "NA",
        *keys,
    ]
    shared_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        + "".join(f"<si><t>{value}</t></si>" for value in shared)
        + "</sst>"
    ).encode("utf-8")
    rows = [
        '<row r="1">'
        '<c r="A1" t="s"><v>0</v></c>'
        '<c r="C1" t="s"><v>1</v></c>'
        '<c r="D1" t="s"><v>2</v></c>'
        '<c r="E1" t="s"><v>3</v></c>'
        '<c r="F1" t="s"><v>4</v></c>'
        "</row>"
    ]
    row_number = 2
    for index, _key in enumerate(keys):
        key_index = 8 + index
        rows.append(
            f'<row r="{row_number}">'
            f'<c r="A{row_number}" t="s"><v>{key_index}</v></c>'
            f'<c r="C{row_number}" t="s"><v>5</v></c>'
            f'<c r="D{row_number}"><v>0.1</v></c>'
            f'<c r="E{row_number}"><v>0.2</v></c>'
            f'<c r="F{row_number}"><v>0.3</v></c>'
            "</row>"
        )
        row_number += 1
        if index < finite_totalpoly_count:
            statistics = (
                f'<c r="D{row_number}"><v>-0.1</v></c>'
                f'<c r="E{row_number}"><v>0.4</v></c>'
                f'<c r="F{row_number}"><v>0.5</v></c>'
            )
        else:
            statistics = (
                f'<c r="D{row_number}" t="s"><v>7</v></c>'
                f'<c r="E{row_number}" t="s"><v>7</v></c>'
                f'<c r="F{row_number}" t="s"><v>7</v></c>'
            )
        rows.append(
            f'<row r="{row_number}">'
            f'<c r="A{row_number}" t="s"><v>{key_index}</v></c>'
            f'<c r="C{row_number}" t="s"><v>6</v></c>'
            f"{statistics}</row>"
        )
        row_number += 1
    worksheet = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        "<sheetData>"
        + "".join(rows)
        + "</sheetData></worksheet>"
    ).encode("utf-8")
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("xl/worksheets/sheet1.xml", worksheet)
        archive.writestr("xl/sharedStrings.xml", shared_xml)
    return output.getvalue(), keys


def matrix_header() -> list[str]:
    families = ("80S_RNA", "High_Poly", "Low_Poly", "pDNA", "Total_RNA")
    result = ["barcode"]
    for arm in ("WT", "Mutant"):
        for family in families:
            for replicate in range(1, 7):
                result.append(f"{family}_{replicate}_S{replicate}_{arm}")
    return result


def make_matrix_plain(keys: list[str]) -> bytes:
    lines = ["\t".join(matrix_header())]
    values = ["1.25"] * 60
    lines.extend("\t".join([key, *values]) for key in keys)
    return ("\n".join(lines) + "\n").encode("utf-8")


class FakeResponse(io.BytesIO):
    status = 200

    def __init__(self, payload: bytes, url: str):
        super().__init__(payload)
        self._url = url
        self.headers = {
            "Content-Length": str(len(payload)),
            "Content-Encoding": "identity",
        }

    def geturl(self) -> str:
        return self._url


def configure_test_sources(
    monkeypatch: pytest.MonkeyPatch,
    *,
    jats_payload: bytes,
    soft_payload: bytes,
    soft_plain: bytes,
    matrix_payload: bytes,
    matrix_plain: bytes,
    matrix_row_count: int,
    matrix_finite_count: int,
) -> dict[str, Any]:
    config = frozen_config()
    source = config["public_sources"]
    for key, payload in (
        (producer.JATS_CONFIG_KEY, jats_payload),
        (producer.SOFT_CONFIG_KEY, soft_payload),
        (producer.MATRIX_CONFIG_KEY, matrix_payload),
    ):
        source[key]["bytes"] = len(payload)
        source[key]["sha256"] = hashlib.sha256(payload).hexdigest()
    source[producer.SOFT_CONFIG_KEY]["plain_bytes"] = len(soft_plain)
    source[producer.SOFT_CONFIG_KEY]["plain_sha256"] = hashlib.sha256(
        soft_plain
    ).hexdigest()
    matrix_spec = source[producer.MATRIX_CONFIG_KEY]
    matrix_spec["plain_bytes"] = len(matrix_plain)
    matrix_spec["plain_sha256"] = hashlib.sha256(matrix_plain).hexdigest()
    matrix_spec["row_count"] = matrix_row_count
    matrix_spec["s3_key_count"] = matrix_row_count
    matrix_spec["s3_finite_totalpoly_key_count"] = matrix_finite_count

    binding = config["implementation_binding"]
    binding.update(
        {
            "status": producer.BOUND,
            "implementation_commit": "1" * 40,
            "implementation_script_sha256": "2" * 64,
            "implementation_test_sha256": "3" * 64,
        }
    )
    core = producer.config_core_sha256(config)
    binding["config_core_sha256"] = core
    monkeypatch.setattr(producer, "FROZEN_CONFIG_CORE_SHA256", core)
    producer.validate_static_config(config)
    return config


def publication_case(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    dict[str, Any],
    dict[str, str],
    producer.PredecessorSummary,
    Any,
]:
    jats_payload = make_jats()
    soft_plain = make_soft_plain()
    soft_payload = gzip.compress(soft_plain, mtime=0)
    xlsx, keys = make_s3_xlsx(pair_count=2, finite_totalpoly_count=1)
    s3 = producer.audit_table_s3_selective(
        xlsx,
        expected_pair_count=2,
        expected_finite_totalpoly_count=1,
    )
    matrix_plain = make_matrix_plain(keys)
    matrix_payload = gzip.compress(matrix_plain, mtime=0)
    config = configure_test_sources(
        monkeypatch,
        jats_payload=jats_payload,
        soft_payload=soft_payload,
        soft_plain=soft_plain,
        matrix_payload=matrix_payload,
        matrix_plain=matrix_plain,
        matrix_row_count=2,
        matrix_finite_count=1,
    )
    binding = {
        "status": "PASS_BOUND_IMPLEMENTATION",
        "implementation_commit": "1" * 40,
        "binding_commit": "8" * 40,
        "implementation_script_sha256": "2" * 64,
        "implementation_test_sha256": "3" * 64,
        "config_core_sha256": config["implementation_binding"][
            "config_core_sha256"
        ],
    }
    source_payloads = {
        config["public_sources"][producer.JATS_CONFIG_KEY]["url"]: jats_payload,
        config["public_sources"][producer.SOFT_CONFIG_KEY]["url"]: soft_payload,
        config["public_sources"][producer.MATRIX_CONFIG_KEY]["url"]: matrix_payload,
    }

    def open_url(url: str) -> FakeResponse:
        return FakeResponse(source_payloads[url], url)

    return config, binding, dummy_predecessor(s3), open_url


def test_disk_config_preserves_exact6_and_append_only_history() -> None:
    config = frozen_config()
    producer.validate_static_config(config)
    binding = config["implementation_binding"]
    assert binding["status"] in {producer.UNKNOWN, producer.BOUND}
    repository = config["repository_authority"]
    assert repository["historical_base0_commit"] == (
        "0b95ac77a44644e57cc4d0bfb31a9154238fdca6"
    )
    assert repository["historical_i1_commit"] == (
        "9844246dd4b3874a9ecfcf03a233278c5d3a02e0"
    )
    assert repository["implementation_repair_base_commit"] == (
        "9844246dd4b3874a9ecfcf03a233278c5d3a02e0"
    )
    assert repository["historical_i1_blobs"] == producer.HISTORICAL_I1_BLOBS
    assert config["output_contract"]["exact_member_names"] == [
        "PMC10540565_EUROPE_PMC_FULLTEXT.xml",
        "GSE200302_family.soft.gz",
        "GSE200302_log2_cpm_counts_all_samples.txt.gz",
        producer.AUDIT_NAME,
        producer.CHECKSUMS_NAME,
        producer.MARKER_NAME,
    ]
    matrix = config["public_sources"][producer.MATRIX_CONFIG_KEY]
    assert (
        matrix["bytes"],
        matrix["sha256"],
        matrix["plain_bytes"],
        matrix["plain_sha256"],
        matrix["row_count"],
        matrix["header_field_count"],
        matrix["s3_key_count"],
        matrix["s3_finite_totalpoly_key_count"],
    ) == (
        2_843_042,
        "ed93162f9540676138cfba05af2841c90619ac4335eb55ee3d956a3cd8aace3c",
        7_028_853,
        "66b933967a5628cfd8e76ea3ae8ad8240f80f14d5479b7bf3563ccac45c9f260",
        6_772,
        61,
        6_772,
        6_547,
    )
    endpoint = config["viability_contract"][
        "canonical_reported_endpoint_semantics"
    ]
    assert endpoint["effect_definition"] == (
        "log2((mutant total-poly/total-RNA)/(WT total-poly/total-RNA))"
    )
    assert endpoint["endpoint_id"] == "TOTAL_POLYSOME_TRANSLATION_EFFICIENCY"
    assert endpoint["table_s3_comparison"] == "TotalPoly:RNA"
    assert endpoint["table_s3_field"] == "xtail_log2FC_TE"
    assert endpoint["positive_direction"] == (
        "MUTANT_HIGHER_TOTAL_POLYSOME_TRANSLATION_EFFICIENCY_THAN_WT"
    )
    assert endpoint["attrition"] == {
        "designed_pair_count": 6_885,
        "published_pair_count": 6_772,
        "finite_primary_pair_count": 6_547,
        "equation": "6885_TO_6772_TO_6547",
    }
    assert endpoint["status_if_all_source_checks_pass"] == (
        "READY_FOR_PASS_RECORD_NOT_YET_BOUND"
    )
    assert endpoint["consumer_gate_pass"] is False
    replicate = config["viability_contract"]["row_replicate_or_valid_se"]
    assert replicate["status_if_all_source_checks_pass"] == (
        "READY_FOR_REPLICATE_BRANCH_PASS_RECORD_NOT_YET_BOUND"
    )
    assert replicate["consumer_gate_pass"] is False
    assert replicate["study_reported_biological_replicate_count"] == 6
    assert replicate["standard_error_status"] == (
        "ABSENT_NOT_REPORTED_NOT_DERIVED_NOT_USED"
    )
    rights = config["viability_contract"]["license_rights"]
    assert rights["status_if_all_source_checks_pass"] == (
        "READY_FOR_PRIVATE_CANONICAL_ONLY_PASS_RECORD_NOT_YET_BOUND"
    )
    assert rights["consumer_gate_pass"] is False
    group = config["viability_contract"]["biological_group_authority"]
    assert group["status"] == "BLOCKED_PENDING_AUTHOR_SOURCE_GROUP_MAPPING_ROOT"
    decision = config["decision_boundary"]
    assert decision["qualified"] is False
    assert decision["canonical_record_count"] == 0
    assert decision["gate_records_written"] == 0
    assert decision["training_allowed"] is False
    assert decision["model_selection_allowed"] is False
    assert decision["next_phase_authorized"] is False


def test_disk_i2_or_b2_lifecycle_and_synthetic_pair_regression() -> None:
    disk = frozen_config()
    producer.validate_static_config(disk)
    unknown_i2 = exact_unknown_i(disk)
    disk_binding = disk["implementation_binding"]
    if disk_binding["status"] == producer.UNKNOWN:
        assert disk == unknown_i2
    else:
        producer.validate_i_to_b_config_pair(
            unknown_i2,
            disk,
            implementation_commit=disk_binding["implementation_commit"],
            script_sha256=disk_binding["implementation_script_sha256"],
            test_sha256=disk_binding["implementation_test_sha256"],
        )

    synthetic_b2 = bound_config(unknown_i2)
    producer.validate_i_to_b_config_pair(
        unknown_i2,
        synthetic_b2,
        implementation_commit="1" * 40,
        script_sha256="2" * 64,
        test_sha256="3" * 64,
    )


def test_i_to_b_exact_four_scalars_and_unknown_stops_before_output(
    tmp_path: Path,
) -> None:
    unknown = exact_unknown_i()
    bound = bound_config(unknown)
    producer.validate_i_to_b_config_pair(
        unknown,
        bound,
        implementation_commit="1" * 40,
        script_sha256="2" * 64,
        test_sha256="3" * 64,
    )
    s3 = producer.S3SelectiveState(
        keys=frozenset({"one"}),
        finite_totalpoly_keys=frozenset({"one"}),
        row_count=2,
    )
    target = tmp_path / "must_not_exist"
    with pytest.raises(producer.BindingError, match="stopped before source/output"):
        producer.publish_bundle(
            unknown,
            {},
            dummy_predecessor(s3),
            target,
            open_url=lambda _url: pytest.fail("network must not be touched"),
        )
    assert not target.exists()


def test_jats_and_soft_exact_semantics() -> None:
    config = frozen_config()
    jats = producer.audit_jats(
        make_jats(), config["public_sources"][producer.JATS_CONFIG_KEY]
    )
    assert jats["identity"] == {
        "doi": "10.1016/j.celrep.2023.112840",
        "pmcid": "PMC10540565",
        "pmid": "37516102",
    }
    assert jats["normalized_paragraphs"][
        "endpoint_and_six_biological_replicates"
    ]["sha256"] == "45dd0d8b9c7976748615f2c7b620bcc403fe7bf5c832b2dbb8516d758b27ac3d"

    plain = make_soft_plain()
    payload = gzip.compress(plain, mtime=0)
    spec = copy.deepcopy(config["public_sources"][producer.SOFT_CONFIG_KEY])
    spec["plain_bytes"] = len(plain)
    spec["plain_sha256"] = hashlib.sha256(plain).hexdigest()
    soft = producer.audit_soft(payload, spec)
    assert soft["sample_count"] == 24
    assert soft["sample_supplementary_none_count"] == 24
    assert soft["series_processed_matrix_reference_count"] == 1
    assert soft["geo_dataset_restriction_field_count"] == 0


def test_full_6772_by_61_matrix_crosscheck_and_6547_finite_coverage() -> None:
    xlsx, keys = make_s3_xlsx(pair_count=6_772, finite_totalpoly_count=6_547)
    s3 = producer.audit_table_s3_selective(xlsx)
    assert len(s3.keys) == 6_772
    assert len(s3.finite_totalpoly_keys) == 6_547

    plain = make_matrix_plain(keys)
    payload = gzip.compress(plain, mtime=0)
    spec = copy.deepcopy(
        frozen_config()["public_sources"][producer.MATRIX_CONFIG_KEY]
    )
    spec["plain_bytes"] = len(plain)
    spec["plain_sha256"] = hashlib.sha256(plain).hexdigest()
    audit, state = producer.audit_matrix(payload, spec, s3=s3)
    assert state.row_count == 6_772
    assert state.header_field_count == 61
    assert audit["matrix_key_set_equals_s3_key_set"] is True
    assert audit["finite_totalpoly_key_count"] == 6_547
    assert audit["matrix_covers_every_finite_totalpoly_key"] is True
    assert audit["standard_error_status"] == (
        "ABSENT_NOT_REPORTED_NOT_DERIVED_NOT_USED"
    )
    assert audit["endpoint_excluded_families"] == ["80S_RNA", "pDNA"]


def test_exact6_publication_idempotence_and_partial_preservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, binding, predecessor, open_url = publication_case(monkeypatch)

    target = tmp_path / "exact6"
    result = producer.publish_bundle(
        config,
        binding,
        predecessor,
        target,
        open_url=open_url,
    )
    assert result["publication_state"] == "COMMITTED_ACCEPTED_AUDIT_ONLY"
    assert result["marker_created"] is True
    assert result["canonical_identity_reconfirmed"] is True
    assert result["durability_reconfirmed"] is True
    assert sorted(path.name for path in target.iterdir()) == sorted(
        config["output_contract"]["exact_member_names"]
    )
    marker = json.loads((target / producer.MARKER_NAME).read_text(encoding="ascii"))
    assert marker["terminal_marker_written_last"] is True
    assert marker["exact_final_member_count"] == 6
    assert marker["preterminal_member_count"] == 5
    audit = json.loads((target / producer.AUDIT_NAME).read_text(encoding="ascii"))
    assert audit["endpoint_crosswalk"]["status_if_all_source_checks_pass"] == (
        "READY_FOR_PASS_RECORD_NOT_YET_BOUND"
    )
    assert audit["replicate_branch"]["status_if_all_source_checks_pass"] == (
        "READY_FOR_REPLICATE_BRANCH_PASS_RECORD_NOT_YET_BOUND"
    )
    assert audit["private_only_rights"]["status_if_all_source_checks_pass"] == (
        "READY_FOR_PRIVATE_CANONICAL_ONLY_PASS_RECORD_NOT_YET_BOUND"
    )
    assert audit["biological_group_authority"]["status"] == (
        "BLOCKED_PENDING_AUTHOR_SOURCE_GROUP_MAPPING_ROOT"
    )
    assert audit["decision_boundary"]["canonical_record_count"] == 0
    assert audit["decision_boundary"]["training_allowed"] is False

    idempotent = producer.publish_bundle(
        config,
        binding,
        predecessor,
        target,
        open_url=lambda _url: pytest.fail("idempotent path must not redownload"),
    )
    assert idempotent["publication_state"] == (
        "IDEMPOTENT_EXACT_EXISTING_DURABILITY_RECONFIRMED"
    )
    assert idempotent["canonical_identity_reconfirmed"] is True
    assert idempotent["durability_reconfirmed"] is True

    def transient_existing_fault(event: str) -> None:
        assert event == "before_existing_fresh_validation"
        raise RuntimeError("synthetic existing durability interruption")

    recovered_existing = producer.publish_bundle(
        config,
        binding,
        predecessor,
        target,
        open_url=lambda _url: pytest.fail("existing recovery must not redownload"),
        fault_hook=transient_existing_fault,
    )
    assert recovered_existing["publication_state"] == (
        "IDEMPOTENT_EXACT_EXISTING_AFTER_DURABILITY_RECOVERY"
    )

    partial = tmp_path / "partial_preserved"
    partial.mkdir()
    with pytest.raises(producer.ExistingOutputRequiresManualReview):
        producer.publish_bundle(
            config,
            binding,
            predecessor,
            partial,
            open_url=lambda _url: pytest.fail("existing partial must not redownload"),
        )
    assert partial.is_dir() and list(partial.iterdir()) == []


def test_canonical_entrypoints_and_worktree_files_fail_closed(
    tmp_path: Path,
) -> None:
    external_script = tmp_path / "producer-copy.py"
    external_script.write_bytes(SCRIPT.read_bytes())
    with pytest.raises(producer.BindingError, match="canonical production script"):
        producer._validate_production_entrypoint_paths(
            config_path=producer.PRODUCTION_CONFIG_PATH,
            script_path=external_script,
        )
    with pytest.raises(producer.BindingError, match="canonical authority path"):
        producer._validate_production_entrypoint_paths(
            config_path=tmp_path / "config-copy.json",
            script_path=producer.PRODUCTION_SCRIPT_PATH,
        )

    repo = tmp_path / "worktree"
    for relative in ("configs", "scripts/route_a_v3", "tests/route_a_v3"):
        (repo / relative).mkdir(parents=True, exist_ok=True)

    real_config = repo / "real-config.json"
    real_config.write_bytes(b"config-authority\n")
    symlink_config = repo / "configs/symlink.json"
    symlink_config.symlink_to(real_config)
    with pytest.raises(producer.InputIntegrityError):
        producer._read_exact_relative(
            repo,
            "configs/symlink.json",
            expected_bytes=real_config.stat().st_size,
            expected_sha256=hashlib.sha256(real_config.read_bytes()).hexdigest(),
            collect=True,
            label="symlink config",
        )

    hard_config_source = repo / "hard-config-source.json"
    hard_config_source.write_bytes(b"hard-config\n")
    hard_config = repo / "configs/hard.json"
    os.link(hard_config_source, hard_config)
    with pytest.raises(producer.InputIntegrityError):
        producer._read_exact_relative(
            repo,
            "configs/hard.json",
            expected_bytes=len(b"hard-config\n"),
            expected_sha256=hashlib.sha256(b"hard-config\n").hexdigest(),
            collect=True,
            label="hardlink config",
        )

    hard_script_source = repo / "hard-script-source.py"
    hard_script_source.write_bytes(b"script-head\n")
    hard_script = repo / "scripts/route_a_v3/hard.py"
    os.link(hard_script_source, hard_script)
    with pytest.raises(producer.InputIntegrityError):
        producer._read_exact_relative(
            repo,
            "scripts/route_a_v3/hard.py",
            expected_bytes=len(b"script-head\n"),
            expected_sha256=hashlib.sha256(b"script-head\n").hexdigest(),
            collect=True,
            label="hardlink script",
        )

    drift_test = repo / "tests/route_a_v3/drift.py"
    drift_test.write_bytes(b"abcdeg")
    with pytest.raises(producer.InputIntegrityError, match="byte/hash authority"):
        producer._read_exact_relative(
            repo,
            "tests/route_a_v3/drift.py",
            expected_bytes=len(b"abcdef"),
            expected_sha256=hashlib.sha256(b"abcdef").hexdigest(),
            collect=True,
            label="drifted test",
        )


def test_wrong_i2_parent_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    implementation = "c" * 40

    def wrong_parent_git(_repo: Path, *arguments: str) -> str:
        assert arguments == (
            "rev-list",
            "--parents",
            "-n",
            "1",
            implementation,
        )
        return f"{implementation} {'d' * 40}"

    monkeypatch.setattr(producer, "_git", wrong_parent_git)
    with pytest.raises(producer.BindingError, match="required direct child"):
        producer._validate_i2_implementation_commit(tmp_path, implementation)


def test_historical_i1_blob_drift_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(producer, "_single_parent", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        producer,
        "_changed_paths",
        lambda _repo, _commit: sorted(producer.EXPECTED_I_PATHS),
    )
    monkeypatch.setattr(
        producer,
        "_git_blob",
        lambda _repo, _commit, _path, _expected_sha256=None: b"wrong-history",
    )
    with pytest.raises(producer.BindingError, match="historical pushed I1 blob"):
        producer._validate_historical_i1_authority(tmp_path)


def test_closed_preterminal_member_replacement_is_preserved_not_committed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, binding, predecessor, open_url = publication_case(monkeypatch)
    target = tmp_path / "preterminal-replacement"
    moved_audit = tmp_path / "original-audit.json"

    def replace_closed_member(event: str) -> None:
        if event != "after_preterminal_members_written":
            return
        os.rename(target / producer.AUDIT_NAME, moved_audit)
        (target / producer.AUDIT_NAME).write_bytes(b"{}\n")

    with pytest.raises(producer.PartialPublicationError):
        producer.publish_bundle(
            config,
            binding,
            predecessor,
            target,
            open_url=open_url,
            fault_hook=replace_closed_member,
        )
    assert target.is_dir()
    assert moved_audit.is_file()
    assert not (target / producer.MARKER_NAME).exists()


def test_created_bundle_late_parent_rename_is_indeterminate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, binding, predecessor, open_url = publication_case(monkeypatch)
    container = tmp_path / "created-parent-case"
    parent = container / "parent"
    old_parent = container / "parent-original"
    parent.mkdir(parents=True)
    target = parent / "bundle"

    def rename_parent(event: str) -> None:
        if event != "before_fresh_final_validation":
            return
        os.rename(parent, old_parent)
        parent.mkdir()

    with pytest.raises(
        producer.PostMarkerCommitOutcomeIndeterminate,
        match="POST_MARKER_COMMIT_OUTCOME_INDETERMINATE",
    ):
        producer.publish_bundle(
            config,
            binding,
            predecessor,
            target,
            open_url=open_url,
            fault_hook=rename_parent,
        )
    assert (old_parent / "bundle" / producer.MARKER_NAME).is_file()
    assert not target.exists()


def test_existing_bundle_late_parent_rename_requires_manual_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, binding, predecessor, open_url = publication_case(monkeypatch)
    container = tmp_path / "existing-parent-case"
    parent = container / "parent"
    old_parent = container / "parent-original"
    parent.mkdir(parents=True)
    target = parent / "bundle"
    producer.publish_bundle(
        config,
        binding,
        predecessor,
        target,
        open_url=open_url,
    )

    def rename_parent(event: str) -> None:
        if event != "before_existing_fresh_validation":
            return
        os.rename(parent, old_parent)
        parent.mkdir()

    with pytest.raises(producer.ExistingOutputRequiresManualReview):
        producer.publish_bundle(
            config,
            binding,
            predecessor,
            target,
            open_url=lambda _url: pytest.fail("existing path must not redownload"),
            fault_hook=rename_parent,
        )
    assert (old_parent / "bundle" / producer.MARKER_NAME).is_file()
    assert not target.exists()


def test_post_marker_fault_recovery_is_explicit_or_indeterminate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, binding, predecessor, open_url = publication_case(monkeypatch)
    recovered_target = tmp_path / "post-marker-recovered"
    real_fsync = producer.os.fsync
    fsync_fault_injected = False

    def fail_one_post_marker_directory_fsync(descriptor: int) -> None:
        nonlocal fsync_fault_injected
        descriptor_stat = producer.os.fstat(descriptor)
        if (
            not fsync_fault_injected
            and producer.stat.S_ISDIR(descriptor_stat.st_mode)
            and (recovered_target / producer.MARKER_NAME).exists()
        ):
            fsync_fault_injected = True
            raise OSError("synthetic post-marker directory fsync fault")
        real_fsync(descriptor)

    monkeypatch.setattr(producer.os, "fsync", fail_one_post_marker_directory_fsync)

    recovered = producer.publish_bundle(
        config,
        binding,
        predecessor,
        recovered_target,
        open_url=open_url,
    )
    assert recovered["publication_state"] == (
        "COMMITTED_ACCEPTED_AFTER_POST_MARKER_RECOVERY"
    )
    assert fsync_fault_injected is True
    assert recovered["recovered_from_post_marker_error"] == "OSError"
    assert recovered["canonical_identity_reconfirmed"] is True
    assert recovered["durability_reconfirmed"] is True
    monkeypatch.setattr(producer.os, "fsync", real_fsync)

    indeterminate_target = tmp_path / "post-marker-indeterminate"
    moved_marker = tmp_path / "original-marker.json"

    def replace_marker_then_interrupt(event: str) -> None:
        if event != "after_marker_created_before_directory_fsync":
            return
        os.rename(indeterminate_target / producer.MARKER_NAME, moved_marker)
        (indeterminate_target / producer.MARKER_NAME).write_bytes(b"{}\n")
        raise RuntimeError("synthetic post-marker replacement")

    with pytest.raises(
        producer.PostMarkerCommitOutcomeIndeterminate,
        match="POST_MARKER_COMMIT_OUTCOME_INDETERMINATE",
    ):
        producer.publish_bundle(
            config,
            binding,
            predecessor,
            indeterminate_target,
            open_url=open_url,
            fault_hook=replace_marker_then_interrupt,
        )
    assert moved_marker.is_file()
    assert (indeterminate_target / producer.MARKER_NAME).read_bytes() == b"{}\n"


def test_matrix_key_drift_fails_closed() -> None:
    xlsx, keys = make_s3_xlsx(pair_count=2, finite_totalpoly_count=1)
    s3 = producer.audit_table_s3_selective(
        xlsx,
        expected_pair_count=2,
        expected_finite_totalpoly_count=1,
    )
    plain = make_matrix_plain([keys[0], "chr2:999_A-C"])
    payload = gzip.compress(plain, mtime=0)
    spec = copy.deepcopy(
        frozen_config()["public_sources"][producer.MATRIX_CONFIG_KEY]
    )
    spec.update(
        {
            "plain_bytes": len(plain),
            "plain_sha256": hashlib.sha256(plain).hexdigest(),
            "row_count": 2,
            "s3_key_count": 2,
            "s3_finite_totalpoly_key_count": 1,
        }
    )
    with pytest.raises(producer.SourceSemanticConflict, match="key set differs"):
        producer.audit_matrix(payload, spec, s3=s3)
