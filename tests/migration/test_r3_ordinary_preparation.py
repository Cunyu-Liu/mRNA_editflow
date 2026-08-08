"""Synthetic R3-0 ordinary preparation and provenance tests."""
import hashlib
import json
import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "b0x"))

import build_effect_dataset as B  # noqa: E402
from scripts.repair.prepare_r3_ordinary import (  # noqa: E402
    CROSSWALK_SCHEMA_VERSION,
    EDIT_REPAIR_METHOD,
    ESTIMAND_SCHEMA_VERSION,
    prepare_rows,
    load_frozen_crosswalk,
    load_frozen_estimand_registry,
    write_bundle_exclusive,
)


def _crosswalk_entry(**overrides):
    entry = {
        "study": "S1",
        "endpoint": "ep",
        "assay_arm": "raw_arm",
        "library": "lib_a",
        "source_asset": "ordinary_asset.tsv",
        "region": "5'UTR",
        "source_anchor_rule": "diff_cand_src:raw_library",
        "publication_context_id": "cell_line_a|assay_a|lib_a",
        "context_evidence_id": "context_evidence_a",
        "source_relation_evidence_id": "relation_evidence_a",
        "allowed_row_relation_evidence": [{
            "evidence_id": "row_relation_1",
            "evidence_sha256": "b" * 64,
            "relation_status": "ACCEPTED",
        }],
    }
    entry.update(overrides)
    return entry


def _crosswalk(*entries):
    rows = list(entries or [_crosswalk_entry()])
    return {
        "path": "ordinary_crosswalk.json",
        "sha256": "c" * 64,
        "crosswalk_id": "cw1",
        "version": "1",
        "frozen_at_utc": "2026-08-07T00:00:00Z",
        "entries": rows,
        "index": {
            tuple(row[field] for field in (
                "study", "endpoint", "assay_arm", "library",
                "source_asset", "region", "source_anchor_rule",
            )): row
            for row in rows
        },
    }


def _observation(value=2.0, replicate="r1", **overrides):
    row = {
        "observation_id": "obs_%s" % replicate,
        "sequence_id": "S1_candidate__cand",
        "endpoint_id": "ep",
        "context_id": "raw_arm",
        "replicate": replicate,
        "unit": "unit",
        "value": value,
        "source_file": "ordinary_asset.tsv",
        "input_row_sha256": "a" * 64,
    }
    row.update(overrides)
    return row


def _row(
    source="AAAAAA",
    candidate="CAAAAA",
    edit_list=None,
    study="S1",
    endpoint="ep",
    library="lib_a",
    assay_arm="raw_arm",
    source_value=1.0,
    candidate_value=2.0,
    legacy_source_id="legacy_1",
    anchor_identity="anchor_1",
    **overrides,
):
    if edit_list is None:
        edit_list = [{"op": "SUB", "pos": 0, "token": candidate[0]}]
    row = {
        "pair_id": "pair_%s" % legacy_source_id,
        "record_id": "record_%s" % legacy_source_id,
        "study": study,
        "benchmark": "5U-A1",
        "endpoint": endpoint,
        "assay_arm": assay_arm,
        "library": library,
        "source_asset": "ordinary_asset.tsv",
        "region": "5'UTR",
        "source_anchor_rule": "diff_cand_src:raw_library",
        "source_anchor_identity": anchor_identity,
        "source_anchor_status": "MEASURED_SINGLE_VALUE",
        "source_anchor_evidence": [{"evidence": "raw_anchor"}],
        "source_sequence": source,
        "candidate_sequence": candidate,
        "edit_list": edit_list,
        "edit_count": len(edit_list),
        "source_id": legacy_source_id,
        "legacy_source_id": legacy_source_id,
        "candidate_id": "cand_%s" % legacy_source_id,
        "source_value": source_value,
        "candidate_value": candidate_value,
        "delta": candidate_value - source_value,
        "functional_observations": [
            _observation(
                candidate_value,
                overrides.get("replicate", "r1"),
                context_id=assay_arm,
                sequence_id="%s_candidate__cand" % study,
                endpoint_id=endpoint,
            )
        ],
        "functional_observation_hashes": ["a" * 64],
        "pair_provenance": {
            "evidence_id": "row_relation_1",
            "evidence_sha256": "b" * 64,
            "relation_status": "ACCEPTED",
        },
    }
    row.update(overrides)
    return row


def _estimand_entry(**overrides):
    entry = {
        "study": "GSE217518",
        "endpoint": "ep_stability_hek",
        "raw_source_field": "source_value",
        "raw_candidate_field": "candidate_value",
        "raw_unit": "half_life",
        "direction": "HIGHER_IS_MORE_STABLE",
        "transform": "IDENTITY_DIFFERENCE",
        "valid_domain": "STRICTLY_POSITIVE",
        "role": "PRIMARY",
        "replicate_policy": "SINGLE_OBSERVATION_ONLY",
        "source_citation": "author-reviewed-source",
        "source_evidence_sha256": "e" * 64,
    }
    entry.update(overrides)
    return entry


def _estimand_registry(*entries):
    rows = list(entries or [_estimand_entry()])
    return {
        "path": "ordinary_estimand_registry.json",
        "sha256": "d" * 64,
        "registry_id": "er1",
        "version": "1",
        "frozen_at_utc": "2026-08-07T00:00:00Z",
        "entries": rows,
        "index": {(row["study"], row["endpoint"]): row for row in rows},
    }


def _s1_estimand_registry(**overrides):
    entry = _estimand_entry(
        study="S1",
        endpoint="ep",
        raw_unit="unit",
        direction="SIGNED_EFFECT_AS_REPORTED",
        transform="IDENTITY_DIFFERENCE",
        valid_domain="FINITE",
        role="PRIMARY",
        replicate_policy="SINGLE_OBSERVATION_ONLY",
    )
    entry.update(overrides)
    return _estimand_registry(entry)


def _prepare(rows, crosswalk=None, registry=None):
    return prepare_rows(
        rows,
        crosswalk or _crosswalk(),
        registry or _s1_estimand_registry(),
    )


def _write_json(path, payload):
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_functional_observations_preserve_replicates_and_provenance(tmp_path):
    path = tmp_path / "ordinary_observations.jsonl"
    rows = [
        _observation(1.0, "r1"),
        _observation(2.0, "r2", observation_id="obs_r2"),
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    index = B.load_candidate_observations(
        path, {"S1": {"S1_candidate__cand"}}
    )
    loaded = index["S1_candidate__cand"]["ep"]
    assert {row["replicate"]: row["value"] for row in loaded} == {
        "r1": 1.0,
        "r2": 2.0,
    }
    assert all(len(row["input_row_sha256"]) == 64 for row in loaded)


def test_functional_observation_value_conflict_fails_closed(tmp_path):
    path = tmp_path / "ordinary_observations.jsonl"
    rows = [
        _observation(1.0, "r1", observation_id="obs_a"),
        _observation(2.0, "r1", observation_id="obs_b"),
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(B.ObservationConflictError, match="multiple values"):
        B.load_candidate_observations(
            path, {"S1": {"S1_candidate__cand"}}
        )


def test_functional_observation_missing_value_fails_closed(tmp_path):
    path = tmp_path / "ordinary_observations.jsonl"
    path.write_text(json.dumps(_observation(None, "r1")) + "\n")
    with pytest.raises(B.ObservationConflictError, match="non-numeric"):
        B.load_candidate_observations(
            path, {"S1": {"S1_candidate__cand"}}
        )


def test_delta_derivation_emits_each_resolvable_replicate():
    sid = "S1_candidate__cand"
    observations = {sid: {"ep": [
        _observation(2.0, "r1"),
        _observation(3.0, "r2", observation_id="obs_r2"),
    ]}}
    record = {
        "record_id": "record",
        "dataset": "dataset",
        "accession": "S1",
        "region": "5'UTR",
        "source_sequence": "AAAAAA",
        "candidate_sequence": "CAAAAA",
        "edit_script": [{"op": "SUB", "pos": 0, "token": "C"}],
        "metadata": {"library": "lib_a", "source_file": "ordinary_asset.tsv"},
    }
    spec = {
        "type": "diff_cand_src", "cand_endpoint": "ep",
        "src_source": "raw_library", "benchmark": "5U-A1",
    }
    anchor = {("lib_a", "AAAAAA"): {
        "source_anchor_identity": "anchor_1",
        "source_anchor_status": "MEASURED_SINGLE_VALUE",
        "value": 1.0,
        "measured_values": [1.0],
        "source_anchor_evidence": [{"row": 1}],
    }}
    output = B.derive_delta("S1", spec, record, observations, anchor, {}, sid)
    assert [row["candidate_value"] for row in output] == [2.0, 3.0]
    assert [row["delta"] for row in output] == [1.0, 2.0]
    assert all(len(row["functional_observations"]) == 1 for row in output)
    assert all(row["legacy_source_id"].endswith("__src") for row in output)


def test_wt_metadata_anchor_identity_is_source_based_not_candidate_record():
    spec = {
        "type": "diff_wt_meta",
        "benchmark": "5U-A1",
        "endpoints": [{
            "cand_endpoint": "ep_stability_hek",
            "wt_meta": "stability_wt_hek",
        }],
    }
    base_record = {
        "dataset": "dataset",
        "accession": "GSE217518",
        "region": "5'UTR",
        "source_sequence": "AAAAAA",
        "candidate_sequence": "CAAAAA",
        "edit_script": [{"op": "SUB", "pos": 0, "token": "C"}],
        "metadata": {
            "library": "lib_a",
            "source_file": "ordinary_asset.tsv",
        },
    }
    identities = []
    for index, wt_value in enumerate((10.0, 20.0), start=1):
        record = dict(base_record)
        record["record_id"] = "record_%d" % index
        record["metadata"] = dict(base_record["metadata"])
        record["metadata"]["stability_wt_hek"] = wt_value
        sid = "GSE217518_candidate_%d__cand" % index
        observations = {sid: {"ep_stability_hek": 30.0}}
        output = B.derive_delta(
            "GSE217518", spec, record, observations, {}, {}, sid
        )
        identities.append(output[0]["source_anchor_identity"])
    assert identities[0] == identities[1]


def test_multi_replicate_rows_hold_without_bound_hierarchical_model():
    rows = [
        _row(legacy_source_id="same", replicate="r1"),
        _row(legacy_source_id="same", replicate="r2"),
    ]
    registry = _s1_estimand_registry(
        replicate_policy="PRESERVE_FOR_HIERARCHICAL_MODEL"
    )
    ready, hold, report = _prepare(rows, registry=registry)
    assert not ready
    assert len(hold) == 2
    assert all(
        "HOLD_REQUIRES_REPLICATE_MODEL" in row["r3_hold_reasons"]
        for row in hold
    )
    assert report["replicate_groups"] == {
        "n_multi_replicate_groups": 1,
        "n_rows_in_multi_replicate_groups": 2,
        "policy": (
            "No replicate averaging or independent-row weighting is "
            "performed by R3 preparation."
        ),
    }


def test_incomplete_equal_length_sub_script_is_repaired_exactly():
    row = _row(edit_list=[])
    ready, hold, report = _prepare([row])
    assert not hold
    assert report["n_primary_ready_rows"] == 1
    repaired = ready[0]
    assert repaired["edit_script_status"] == EDIT_REPAIR_METHOD
    assert repaired["edit_list"] == [{"op": "SUB", "pos": 0, "token": "C"}]
    assert repaired["edit_repair"]["reason"] == "INCOMPLETE_EDIT_SCRIPT"
    assert repaired["edit_repair"]["sub_replay_exact"] is True


def test_nonempty_mismatched_sub_script_is_not_reconstructed():
    row = _row(
        source="AAAAAA",
        candidate="CCAAAA",
        edit_list=[{"op": "SUB", "pos": 0, "token": "C"}],
    )
    ready, hold, _ = _prepare([row])
    assert not ready
    assert "NONEMPTY_MISMATCHED_SUB_SCRIPT_NOT_REPAIRABLE" in hold[0][
        "r3_hold_reasons"
    ]
    assert "edit_repair" not in hold[0]


@pytest.mark.parametrize(
    "row,reason",
    [
        (_row(edit_list=[{"op": "INS", "pos": 0, "token": "C"}]), "NON_SUB_EDIT"),
        (_row(source="AAAA", candidate="CAAAA", edit_list=[]), "UNEQUAL_LENGTH"),
    ],
)
def test_non_sub_and_unequal_rows_are_never_coerced(row, reason):
    ready, hold, _ = _prepare([row])
    assert not ready
    assert reason in hold[0]["r3_hold_reasons"]
    assert "edit_repair" not in hold[0]


@pytest.mark.parametrize("edit_count", [2, 4, 6])
def test_out_of_contract_edit_budgets_never_enter_primary(edit_count):
    source = "A" * 10
    candidate = "C" * edit_count + "A" * (10 - edit_count)
    edits = [
        {"op": "SUB", "pos": index, "token": "C"}
        for index in range(edit_count)
    ]
    ready, hold, report = _prepare([
        _row(source=source, candidate=candidate, edit_list=edits)
    ])
    assert not ready
    assert hold[0]["r3_disposition"] == "DEVELOPMENT_OUT_OF_CONTRACT"
    assert hold[0]["observed_edit_count"] == edit_count
    assert report["edit_budget_disposition_counts"] == {
        "DEVELOPMENT_OUT_OF_CONTRACT": 1
    }


def test_missing_crosswalk_atom_is_hold():
    row = _row()
    row.pop("library")
    ready, hold, _ = _prepare([row])
    assert not ready
    assert "MISSING_CONTEXT_ATOM:library" in hold[0]["r3_hold_reasons"]


def test_row_relation_evidence_must_match_frozen_crosswalk_allowlist():
    row = _row()
    row["pair_provenance"]["evidence_id"] = "unbound_relation"
    ready, hold, _ = _prepare([row])
    assert not ready
    assert "ROW_SOURCE_RELATION_EVIDENCE_NOT_BOUND" in hold[0][
        "r3_hold_reasons"
    ]


def test_unreviewed_row_relation_cannot_unlock_derived_representation():
    row = _row(edit_list=[])
    row["pair_provenance"]["relation_status"] = "UNREVIEWED"
    ready, hold, _ = _prepare([row])
    assert not ready
    assert hold[0]["edit_script_status"] == EDIT_REPAIR_METHOD
    assert "ROW_SOURCE_RELATION_NOT_ACCEPTED" in hold[0][
        "r3_hold_reasons"
    ]


def test_sentinel_publication_context_is_rejected(tmp_path):
    payload = {
        "schema_version": CROSSWALK_SCHEMA_VERSION,
        "status": "FROZEN",
        "crosswalk_id": "cw1",
        "version": "1",
        "frozen_at_utc": "2026-08-07T00:00:00Z",
        "entries": [_crosswalk_entry(
            study="GSE114002",
            publication_context_id="ctx_gse114002",
        )],
    }
    path = tmp_path / "ordinary_crosswalk.json"
    digest = _write_json(path, payload)
    with pytest.raises(ValueError, match="sentinel"):
        load_frozen_crosswalk(path, digest)


def test_source_group_v2_merges_same_mother_only_with_same_frozen_lineage():
    first = _row(candidate="CAAAAA", legacy_source_id="legacy_1")
    second = _row(
        candidate="GAAAAA",
        edit_list=[{"op": "SUB", "pos": 0, "token": "G"}],
        legacy_source_id="legacy_2",
    )
    ready, hold, report = _prepare([first, second])
    assert not hold
    assert len({row["source_group_id_v2"] for row in ready}) == 1
    assert len({row["legacy_source_id"] for row in ready}) == 2
    groups = report["source_groups_v2"]
    assert groups["n_groups_with_row_repeats"] == 1
    assert groups["n_groups_with_at_least_2_distinct_candidates"] == 1
    assert groups["n_conflict_free_distinct_candidate_groups"] == 1
    assert report["primary_ready_source_groups_v2"][
        "publication_qualified"
    ] is False
    assert report["primary_ready_source_groups_v2"][
        "development_availability"
    ] == "DEV_AVAILABLE"


def test_source_group_v2_does_not_merge_different_context_or_library():
    entries = [
        _crosswalk_entry(),
        _crosswalk_entry(
            assay_arm="raw_arm_b",
            publication_context_id="cell_line_b|assay_b|lib_a",
            context_evidence_id="context_b",
        ),
        _crosswalk_entry(
            library="lib_b",
            publication_context_id="cell_line_a|assay_a|lib_b",
            context_evidence_id="context_c",
        ),
    ]
    rows = [
        _row(legacy_source_id="l1"),
        _row(assay_arm="raw_arm_b", legacy_source_id="l2"),
        _row(library="lib_b", legacy_source_id="l3"),
    ]
    ready, hold, _ = _prepare(rows, _crosswalk(*entries))
    assert not hold
    assert len({row["source_group_id_v2"] for row in ready}) == 3


def test_source_value_conflict_holds_entire_v2_group():
    rows = [
        _row(candidate="CAAAAA", source_value=1.0, legacy_source_id="l1"),
        _row(
            candidate="GAAAAA", source_value=1.5, candidate_value=2.5,
            edit_list=[{"op": "SUB", "pos": 0, "token": "G"}],
            legacy_source_id="l2",
        ),
    ]
    ready, hold, report = _prepare(rows)
    assert not ready
    assert len(hold) == 2
    assert all(
        "SOURCE_GROUP_VALUE_CONFLICT" in row["r3_hold_reasons"]
        for row in hold
    )
    assert report["n_source_group_value_conflicts"] == 1


def _gse217518_row(source_value=10.0, candidate_value=20.0):
    row = _row(
        study="GSE217518",
        endpoint="ep_stability_hek",
        assay_arm="raw_gse217518_arm",
        source_value=source_value,
        candidate_value=candidate_value,
    )
    row["delta"] = candidate_value - source_value
    return row


def _gse217518_crosswalk():
    return _crosswalk(_crosswalk_entry(
        study="GSE217518",
        endpoint="ep_stability_hek",
        assay_arm="raw_gse217518_arm",
    ))


def test_gse217518_missing_estimand_registry_is_primary_hold():
    ready, hold, _ = prepare_rows(
        [_gse217518_row()], _gse217518_crosswalk(), None
    )
    assert not ready
    assert "MISSING_FROZEN_ENDPOINT_ESTIMAND" in hold[0]["r3_hold_reasons"]
    assert hold[0]["raw_source_value"] == 10.0
    assert hold[0]["raw_candidate_value"] == 20.0


def test_gse217518_primary_identity_preserves_raw_half_life_delta():
    registry = _estimand_registry(_estimand_entry())
    ready, hold, _ = prepare_rows(
        [_gse217518_row()], _gse217518_crosswalk(), registry
    )
    assert not hold
    assert ready[0]["raw_delta"] == 10.0
    assert ready[0]["delta"] == 10.0
    assert ready[0]["estimand_binding"]["formula"] == (
        "candidate_value - source_value"
    )
    assert ready[0]["estimand_binding"]["role"] == "PRIMARY"


def test_estimand_registry_log_ratio_is_secondary_only_and_preserves_raw():
    registry = _estimand_registry(_estimand_entry(
        transform="NATURAL_LOG_RATIO",
        valid_domain="STRICTLY_POSITIVE",
        direction="HIGHER_IS_MORE_STABLE",
        role="SECONDARY_SENSITIVITY",
    ))
    ready, hold, _ = prepare_rows(
        [_gse217518_row()], _gse217518_crosswalk(), registry
    )
    assert not ready
    assert hold[0]["raw_delta"] == 10.0
    assert hold[0]["delta"] == pytest.approx(math.log(2.0))
    assert hold[0]["estimand_binding"]["formula"] == (
        "ln(candidate_value / source_value)"
    )
    assert "ENDPOINT_ESTIMAND_SECONDARY_SENSITIVITY_ONLY" in hold[0][
        "r3_hold_reasons"
    ]


def test_estimand_registry_rejects_nonpositive_transform_domain():
    registry = _estimand_registry(_estimand_entry(
        transform="NATURAL_LOG_RATIO",
        valid_domain="STRICTLY_POSITIVE",
        role="SECONDARY_SENSITIVITY",
    ))
    ready, hold, _ = prepare_rows(
        [_gse217518_row(source_value=0.0)],
        _gse217518_crosswalk(),
        registry,
    )
    assert not ready
    assert "ENDPOINT_ESTIMAND_REQUIRES_POSITIVE_VALUES" in hold[0]["r3_hold_reasons"]


def test_estimand_direction_orients_lower_is_more_stable():
    entry = _estimand_entry(
        study="S1",
        endpoint="ep",
        direction="LOWER_IS_MORE_STABLE",
        transform="IDENTITY_DIFFERENCE",
        valid_domain="FINITE",
    )
    registry = _estimand_registry(entry)
    ready, hold, _ = prepare_rows(
        [_row(source_value=20.0, candidate_value=10.0)],
        _crosswalk(),
        registry,
    )
    assert not hold
    assert ready[0]["delta"] == 10.0
    assert ready[0]["estimand_binding"]["direction_multiplier"] == -1.0


def test_invalid_estimand_direction_fails_registry_load(tmp_path):
    payload = {
        "schema_version": ESTIMAND_SCHEMA_VERSION,
        "status": "FROZEN",
        "registry_id": "er1",
        "version": "1",
        "frozen_at_utc": "2026-08-07T00:00:00Z",
        "entries": [_estimand_entry(direction="UNKNOWN")],
    }
    path = tmp_path / "ordinary_estimand_registry.json"
    digest = _write_json(path, payload)
    with pytest.raises(ValueError, match="direction"):
        load_frozen_estimand_registry(path, digest)


def test_registry_loader_rejects_primary_log_transform(tmp_path):
    payload = {
        "schema_version": ESTIMAND_SCHEMA_VERSION,
        "status": "FROZEN",
        "registry_id": "er1",
        "version": "1",
        "frozen_at_utc": "2026-08-07T00:00:00Z",
        "entries": [_estimand_entry(
            transform="NATURAL_LOG_RATIO",
            role="PRIMARY",
        )],
    }
    path = tmp_path / "ordinary_estimand_registry.json"
    digest = _write_json(path, payload)
    with pytest.raises(ValueError, match="PRIMARY"):
        load_frozen_estimand_registry(path, digest)


def test_prepare_rows_is_input_order_deterministic():
    first = _row(legacy_source_id="l1", anchor_identity="a1")
    second = _row(
        source="CCCCCC", candidate="GCCCCC",
        edit_list=[{"op": "SUB", "pos": 0, "token": "G"}],
        legacy_source_id="l2", anchor_identity="a2",
    )
    ready_a, hold_a, report_a = _prepare([first, second])
    ready_b, hold_b, report_b = _prepare([second, first])
    assert ready_a == ready_b
    assert hold_a == hold_b
    assert report_a == report_b


def test_bundle_outputs_are_exclusive_and_hash_bound(tmp_path):
    dataset = tmp_path / "ordinary_dataset.jsonl"
    dataset.write_text(json.dumps(_row()) + "\n")
    ready, hold, report = _prepare([_row()])
    out_dir = tmp_path / "ordinary_bundle"
    manifest = write_bundle_exclusive(
        out_dir, ready, hold, report, dataset, _crosswalk()
    )
    assert manifest["outputs"]["primary_ready"]["n_rows"] == 1
    before = {
        path.name: path.read_bytes() for path in out_dir.iterdir()
    }
    with pytest.raises(FileExistsError):
        write_bundle_exclusive(
            out_dir, ready, hold, report, dataset, _crosswalk()
        )
    assert before == {
        path.name: path.read_bytes() for path in out_dir.iterdir()
    }


def test_bundle_refuses_protected_output_path(tmp_path):
    dataset = tmp_path / "ordinary_dataset.jsonl"
    dataset.write_text(json.dumps(_row()) + "\n")
    ready, hold, report = _prepare([_row()])
    with pytest.raises(ValueError, match="protected path"):
        write_bundle_exclusive(
            tmp_path / "restricted_output", ready, hold, report,
            dataset, _crosswalk(),
        )
