"""Synthetic adversarial tests for the proposal-only R3 Phase-0 inventory."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.repair.build_r3_phase0_proposal import (  # noqa: E402
    BINDING_SCHEMA_VERSION,
    D1_FILE_NAMES,
    DIFF_REF_ALT_AUDIT_FIELDS,
    DIFF_REF_ALT_AUDIT_SCHEMA_VERSION,
    DIFF_REF_ALT_POLICY_FIELDS,
    DIFF_REF_ALT_POLICY_SCHEMA_VERSION,
    EFFECT_DATASET_MANIFEST_VERSION,
    EFFECT_DATASET_SCHEMA_VERSION,
    EFFECT_MANIFEST_FIELDS,
    EFFECT_QUALIFICATION_STATUS,
    FUNCTIONAL_OBSERVATION_SCHEMA_VERSION,
    IMPLEMENTATION_BINDING_FIELDS,
    IMPLEMENTATION_BINDING_SCHEMA_VERSION,
    INPUT_PRESENCE_PROVENANCE_SCHEMA_VERSION,
    JSONL_D1_AUTHORITIES,
    MANIFEST_QUALIFICATION_STATUS,
    OUTCOME_COMMITMENT_SCHEMA_VERSION,
    Phase0ProposalError,
    REQUIRED_DIFF_REF_ALT_POLICY_SHA256,
    SCHEMA_VERSION,
    build_phase0_proposal,
    validate_phase0_proposal,
    write_phase0_proposal_exclusive,
)
from scripts.b0x import build_effect_dataset as b0x_builder  # noqa: E402
from scripts.b0x import config as b0x_config  # noqa: E402


def _canonical(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_bytes(
        b"".join((_canonical(row) + "\n").encode("utf-8") for row in rows)
    )


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _group_row(group_id: str, atom: str, members: list[str]) -> dict:
    normalized = sorted(set(members))
    material = {
        "group_id": group_id,
        "grouping_atom": atom,
        "member_ids": normalized,
    }
    return {
        **material,
        "group_sha256": hashlib.sha256(
            _canonical(material).encode("utf-8")
        ).hexdigest(),
    }


def _preserved_observation(raw_observation: dict) -> dict:
    raw = copy.deepcopy(raw_observation)
    raw_sha256 = _json_sha(raw)
    preserved = copy.deepcopy(raw)
    upstream_declared = preserved.pop("input_row_sha256", None)
    if upstream_declared is not None:
        preserved["upstream_declared_input_row_sha256"] = upstream_declared
    presence = {
        "schema_version": INPUT_PRESENCE_PROVENANCE_SCHEMA_VERSION,
        "assay_context": (
            "ABSENT_CANONICALIZED_TO_NULL"
            if "assay_context" not in raw
            else "EXPLICIT_NULL"
            if raw["assay_context"] is None
            else "EXPLICIT_VALUE"
        ),
        "replicate": (
            "ABSENT_CANONICALIZED_TO_NULL"
            if "replicate" not in raw
            else "EXPLICIT_NULL"
            if raw["replicate"] is None
            else "EXPLICIT_VALUE"
        ),
    }
    for field in ("assay_context", "replicate"):
        preserved.setdefault(field, None)
    preserved["input_presence_provenance"] = presence
    preserved["raw_input_row_sha256"] = raw_sha256
    preserved["raw_input_value_before_numeric_normalization"] = raw.get(
        "value"
    )
    preserved["value"] = float(raw.get("value"))
    preserved["input_row_sha256"] = _json_sha(preserved)
    return preserved


def _empty_diff_ref_alt_audits() -> dict:
    empty_multiset_sha = _json_sha([])
    audits = {}
    for study, expected_policy_sha in sorted(
        REQUIRED_DIFF_REF_ALT_POLICY_SHA256.items()
    ):
        policy = copy.deepcopy(b0x_config.DELTA_SPECS[study]["join_policy"])
        assert _json_sha(policy) == expected_policy_sha
        audits[study] = {
            "schema_version": DIFF_REF_ALT_AUDIT_SCHEMA_VERSION,
            "verification_status": (
                "PRESENCE_AWARE_INDEPENDENT_REPLAY_VERIFIED"
            ),
            "join_policy": policy,
            "join_policy_sha256": expected_policy_sha,
            "policy_id": policy["policy_id"],
            "comparison_scale_id": policy["comparison_scale_id"],
            "n_materialized_pairs": 0,
            "n_output_pairs": 0,
            "n_output_join_rows": 0,
            "n_expected_missing_both_pairs": 0,
            "n_consumed_missing_both_rows": 0,
            "n_ref_observation_groups": 0,
            "n_alt_observation_groups": 0,
            "n_input_side_groups": 0,
            "n_consumed_ref_observation_groups": 0,
            "n_consumed_alt_observation_groups": 0,
            "n_consumed_side_groups": 0,
            "n_matched_groups": 0,
            "n_unmatched_ref_groups": 0,
            "n_unmatched_alt_groups": 0,
            "n_missing_both_rows": 0,
            "n_derived_rows": 0,
            "n_unavailable_rows": 0,
            "expected_output_rows_from_group_conservation": 0,
            "join_disposition_counts": {},
            "delta_unavailable_reason_counts": {},
            "replicate_pairing_status_counts": {},
            "ref_assay_context_input_presence_group_counts": {},
            "alt_assay_context_input_presence_group_counts": {},
            "ref_replicate_input_presence_group_counts": {},
            "alt_replicate_input_presence_group_counts": {},
            "ref_raw_unit_token_group_counts": {},
            "alt_raw_unit_token_group_counts": {},
            "expected_input_side_group_multiset_sha256": empty_multiset_sha,
            "consumed_output_side_group_multiset_sha256": empty_multiset_sha,
            "expected_missing_both_multiset_sha256": empty_multiset_sha,
            "consumed_missing_both_multiset_sha256": empty_multiset_sha,
            "materialized_pair_registry_sha256": empty_multiset_sha,
            "output_pair_registry_sha256": empty_multiset_sha,
        }
    return audits


def _effect_rows(*, reverse: bool = False, outcome_shift: float = 0.0) -> list[dict]:
    rows = [
        {
            "effect_dataset_schema_version": EFFECT_DATASET_SCHEMA_VERSION,
            "qualification_status": EFFECT_QUALIFICATION_STATUS,
            "functional_observation_schema_version": (
                FUNCTIONAL_OBSERVATION_SCHEMA_VERSION
            ),
            "pair_id": "pair_alpha",
            "record_id": "record_alpha",
            "study": "GSE114002",
            "endpoint": "ep_rl",
            "assay_arm": "HEK293T",
            "library": "designed_library",
            "source_asset": "ordinary_source_a.csv",
            "region": "5UTR",
            "source_anchor_rule": "diff_cand_src:raw_library",
            "context_id": "legacy_context_a",
            "assay_context": "legacy_context_a",
            "unit": "MRL",
            "functional_observations": [
                _preserved_observation({
                    "endpoint_id": "ep_rl",
                    "assay_context": "legacy_context_a",
                    "replicate": 1,
                    "value": 3.0 + outcome_shift,
                })
            ],
            "source_sequence": "AAAA",
            "candidate_sequence": "CAAA",
            "edit_list": [{"pos": 0, "token": "C"}],
            "edit_count": 1,
            "source_value": 2.0 + outcome_shift,
            "candidate_value": 3.0 + outcome_shift,
            "delta": 1.0 + outcome_shift,
            "sign": 1,
            "pair_provenance": {"relation_status": "ACCEPTED"},
        },
        {
            "effect_dataset_schema_version": EFFECT_DATASET_SCHEMA_VERSION,
            "qualification_status": EFFECT_QUALIFICATION_STATUS,
            "functional_observation_schema_version": (
                FUNCTIONAL_OBSERVATION_SCHEMA_VERSION
            ),
            "pair_id": "pair_beta",
            "record_id": "record_beta",
            "study": "GSE114002",
            "endpoint": "ep_rl",
            "assay_arm": "HEK293T",
            "library": None,
            "source_asset": "ordinary_source_a.csv",
            "region": "5UTR",
            "source_anchor_rule": "diff_cand_src:raw_library",
            "context_id": "legacy_context_b",
            "assay_context": "legacy_context_c",
            "unit": "MRL",
            "functional_observations": [
                _preserved_observation({
                    "endpoint_id": "ep_rl",
                    "replicate": None,
                    "value": str(4.0 - outcome_shift),
                    "input_row_sha256": "a" * 64,
                })
            ],
            "source_sequence": "GGGG",
            "candidate_sequence": "UGGG",
            "edit_list": [{"op": "SUB", "pos": 0, "ref": "G", "alt": "U"}],
            "edit_count": 1,
            "source_value": 5.0 - outcome_shift,
            "candidate_value": 4.0 - outcome_shift,
            "delta": -1.0 - outcome_shift,
            "sign": -1,
            "pair_provenance": {"relation_status": "ACCEPTED"},
        },
    ]
    return list(reversed(rows)) if reverse else rows


def _fixture(
    base: Path,
    *,
    reverse_effect: bool = False,
    reverse_registry: bool = False,
    reverse_assignments: bool = False,
    outcome_shift: float = 0.0,
) -> dict:
    base.mkdir(parents=True, exist_ok=True)
    paths = {
        "effect": base / "effect_dataset.jsonl",
        "effect_manifest": base / "effect_dataset_manifest.json",
        "d1_sha256s": base / "D1_SHA256SUMS",
        "utr_edit_pairs": base / D1_FILE_NAMES["utr_edit_pairs"],
        "functional_observations": base / D1_FILE_NAMES["functional_observations"],
        "group_assignments": base / D1_FILE_NAMES["group_assignments"],
        "group_registry": base / D1_FILE_NAMES["group_registry"],
        "sequence_entities": base / D1_FILE_NAMES["sequence_entities"],
        "object_attributes": base / D1_FILE_NAMES["object_attributes"],
        "grouping_atom_projection": base / D1_FILE_NAMES[
            "grouping_atom_projection"
        ],
    }
    _write_jsonl(
        paths["utr_edit_pairs"],
        [
            {
                "pair_id": "pair_alpha",
                "source_sequence_id": "seq_alpha_source",
                "candidate_sequence_id": "seq_alpha_candidate",
                "relation_acceptance_status": "ACCEPTED",
                "relation_status": "ACCEPTED",
            },
            {
                "pair_id": "pair_beta",
                "source_sequence_id": "seq_beta_source",
                "candidate_sequence_id": "seq_beta_candidate",
                "relation_acceptance_status": "ACCEPTED",
                "relation_status": "ACCEPTED",
            },
        ],
    )
    _write_jsonl(
        paths["functional_observations"],
        [
            {
                "observation_id": "obs_alpha",
                "endpoint_id": "ep_rl",
                "context_id": "legacy_context_a",
                "replicate": 1,
                "unit": "MRL",
                "value": 3.0,
            },
            {
                "observation_id": "obs_beta",
                "endpoint_id": "ep_rl",
                "context_id": "legacy_context_b",
                "replicate": 1,
                "unit": "MRL",
                "value": 4.0,
            },
        ],
    )
    registry_rows = [
        _group_row("group_pair_alpha", "PAIR", ["pair_alpha"]),
        _group_row("group_pair_beta", "PAIR", ["pair_beta"]),
        _group_row(
            "group_source_shared", "SOURCE", ["pair_alpha", "pair_beta"]
        ),
        _group_row("group_study", "STUDY", ["pair_alpha", "pair_beta"]),
    ]
    if reverse_registry:
        registry_rows.reverse()
    _write_jsonl(paths["group_registry"], registry_rows)
    assignments = []
    for pair_id, pair_group in (
        ("pair_alpha", "group_pair_alpha"),
        ("pair_beta", "group_pair_beta"),
    ):
        for atom, group_id in (
            ("PAIR", pair_group),
            ("SOURCE", "group_source_shared"),
            ("STUDY", "group_study"),
        ):
            assignments.append(
                {
                    "assignment_id": f"assignment_{pair_id}_{atom.lower()}",
                    "object_id": pair_id,
                    "object_type": "PAIR",
                    "group_id": group_id,
                    "grouping_atom": atom,
                    "assignment_algorithm_id": "D1_PROVENANCE_ATOM_PROJECTION_V1",
                }
            )
    if reverse_assignments:
        assignments.reverse()
    _write_jsonl(paths["group_assignments"], assignments)
    _write_jsonl(
        paths["sequence_entities"],
        [
            {"sequence_id": "seq_alpha_source", "normalized_sequence": "AAAA"},
            {"sequence_id": "seq_alpha_candidate", "normalized_sequence": "CAAA"},
            {"sequence_id": "seq_beta_source", "normalized_sequence": "GGGG"},
            {"sequence_id": "seq_beta_candidate", "normalized_sequence": "UGGG"},
        ],
    )
    _write_jsonl(
        paths["object_attributes"],
        [
            {
                "object_id": "pair_alpha",
                "object_type": "PAIR",
                "group_ids_by_atom": {
                    "PAIR": ["group_pair_alpha"],
                    "SOURCE": ["group_source_shared"],
                    "STUDY": ["group_study"],
                },
            },
            {
                "object_id": "pair_beta",
                "object_type": "PAIR",
                "group_ids_by_atom": {
                    "PAIR": ["group_pair_beta"],
                    "SOURCE": ["group_source_shared"],
                    "STUDY": ["group_study"],
                },
            },
        ],
    )
    _write_json(
        paths["grouping_atom_projection"],
        {
            "artifact_id": "d1_grouping_atom_projection_v1",
            "schema_version": "3.1",
            "grouping_atom_projection_rule_sha256": "a" * 64,
            "assignment_algorithm_id": "D1_PROVENANCE_ATOM_PROJECTION_V1",
            "group_id_algorithm_id": "SHA256_ATOM_VALUE_V1",
            "invented_atom_forbidden": True,
            "projection_policy": {
                "PAIR": "pair_id for PAIR only",
                "SOURCE": "source_sequence_id for PAIR only",
                "STUDY": "explicit accession",
            },
            "counters": {},
        },
    )
    _write_jsonl(
        paths["effect"],
        _effect_rows(reverse=reverse_effect, outcome_shift=outcome_shift),
    )
    d1_expected = {
        logical: _sha(paths[logical]) for logical in D1_FILE_NAMES
    }
    paths["d1_sha256s"].write_text(
        "".join(
            f"{d1_expected[logical]}  {D1_FILE_NAMES[logical]}\n"
            for logical in sorted(JSONL_D1_AUTHORITIES)
        ),
        encoding="utf-8",
    )
    _write_json(
        paths["effect_manifest"],
        {
            "manifest_version": EFFECT_DATASET_MANIFEST_VERSION,
            "effect_dataset_schema_version": EFFECT_DATASET_SCHEMA_VERSION,
            "functional_observation_schema_version": (
                FUNCTIONAL_OBSERVATION_SCHEMA_VERSION
            ),
            "functional_observation_input_presence_schema_version": (
                INPUT_PRESENCE_PROVENANCE_SCHEMA_VERSION
            ),
            "qualification": {
                "status": MANIFEST_QUALIFICATION_STATUS,
                "dataset_without_manifest": "NOT_QUALIFIED",
                "replay_audit_completed_before_dataset_write": True,
                "scientific_qualification": False,
                "training_ready": False,
                "split_ready": False,
            },
            "implementation_binding": {
                "schema_version": IMPLEMENTATION_BINDING_SCHEMA_VERSION,
                "build_effect_dataset_path": "scripts/b0x/build_effect_dataset.py",
                "build_effect_dataset_sha256": "b" * 64,
                "config_path": "scripts/b0x/config.py",
                "config_sha256": "c" * 64,
                "delta_specs_sha256": "d" * 64,
                "diff_ref_alt_policy_sha256_by_study": dict(
                    REQUIRED_DIFF_REF_ALT_POLICY_SHA256
                ),
                "effect_dataset_schema_version": EFFECT_DATASET_SCHEMA_VERSION,
                "manifest_version": EFFECT_DATASET_MANIFEST_VERSION,
                "diff_ref_alt_join_policy_schema_version": (
                    DIFF_REF_ALT_POLICY_SCHEMA_VERSION
                ),
                "diff_ref_alt_replay_audit_schema_version": (
                    DIFF_REF_ALT_AUDIT_SCHEMA_VERSION
                ),
                "functional_observation_input_presence_schema_version": (
                    INPUT_PRESENCE_PROVENANCE_SCHEMA_VERSION
                ),
                "ordinary_only": True,
                "scientific_unit_authority_created": False,
                "estimand_authority_created": False,
            },
            "n_records": len(_effect_rows()),
            "n_pairs_scanned": 2,
            "per_asset": {},
            "source_anchor_status_counts": {},
            "pair_relation_provenance": {},
            "sha256": _sha(paths["effect"]),
            "inputs": {
                "pairs": {"sha256": d1_expected["utr_edit_pairs"]},
                "functional_observations": {
                    "sha256": d1_expected["functional_observations"]
                },
            },
            "diff_ref_alt_join_audits": _empty_diff_ref_alt_audits(),
            "legacy_source_lineage": {},
            "config": {},
        },
    )
    return {
        "effect_path": paths["effect"],
        "effect_sha256": _sha(paths["effect"]),
        "effect_manifest_path": paths["effect_manifest"],
        "effect_manifest_sha256": _sha(paths["effect_manifest"]),
        "d1_sha256s_path": paths["d1_sha256s"],
        "d1_sha256s_sha256": _sha(paths["d1_sha256s"]),
        "d1_paths": {logical: paths[logical] for logical in D1_FILE_NAMES},
        "d1_expected_sha256": d1_expected,
    }


def _build(args: dict) -> dict:
    return build_phase0_proposal(**args)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _refresh_d1_bindings(args: dict) -> None:
    expected = {
        logical: _sha(args["d1_paths"][logical]) for logical in D1_FILE_NAMES
    }
    args["d1_expected_sha256"] = expected
    args["d1_sha256s_path"].write_text(
        "".join(
            f"{expected[logical]}  {D1_FILE_NAMES[logical]}\n"
            for logical in sorted(JSONL_D1_AUTHORITIES)
        ),
        encoding="utf-8",
    )
    args["d1_sha256s_sha256"] = _sha(args["d1_sha256s_path"])
    manifest = json.loads(
        args["effect_manifest_path"].read_text(encoding="utf-8")
    )
    manifest["inputs"]["pairs"]["sha256"] = expected["utr_edit_pairs"]
    manifest["inputs"]["functional_observations"]["sha256"] = expected[
        "functional_observations"
    ]
    _write_json(args["effect_manifest_path"], manifest)
    args["effect_manifest_sha256"] = _sha(args["effect_manifest_path"])


def _walk(value: object):
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key), child
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def _assert_no_raw_identity_values(proposal: dict) -> None:
    rendered = _canonical(proposal)
    for raw_identifier in (
        "pair_alpha",
        "pair_beta",
        "record_alpha",
        "record_beta",
        "seq_alpha_source",
        "group_pair_alpha",
        "group_source_shared",
        "assignment_pair_beta_source",
        "AAAA",
        "CAAA",
    ):
        assert raw_identifier not in rendered


def _rehash_forged_proposal(proposal: dict) -> None:
    binding_material = {
        "schema_version": BINDING_SCHEMA_VERSION,
        "input_authorities": proposal["input_authorities"],
        "binding_checks": proposal["binding_checks"],
    }
    proposal["authority_binding_sha256"] = _json_sha(binding_material)
    outcome_free_material = {
        "schema_version": OUTCOME_COMMITMENT_SCHEMA_VERSION,
        "inventory_policy": proposal["inventory_policy"],
        "context_candidate_inventory": proposal["context_candidate_inventory"],
        "estimand_review_queue": proposal["estimand_review_queue"],
        "relation_atom_inventory": proposal["relation_atom_inventory"],
        "mechanical_candidate_inventory": proposal[
            "mechanical_candidate_inventory"
        ],
    }
    proposal["outcome_free_inventory_sha256"] = _json_sha(outcome_free_material)
    material = dict(proposal)
    material.pop("proposal_sha256")
    proposal["proposal_sha256"] = _json_sha(material)


def _forge_context_identity(proposal: dict) -> None:
    candidate = proposal["context_candidate_inventory"]["candidates"][0]
    candidate["mechanical_key"]["library"] = "public pair_alpha"
    candidate["candidate_key_sha256"] = _json_sha(
        {
            "schema_version": "r3_phase0_context_candidate_key_v1",
            **candidate["mechanical_key"],
        }
    )
    proposal["context_candidate_inventory"]["candidates"].sort(
        key=lambda row: row["candidate_key_sha256"]
    )


def _forge_estimand_outcome_text(proposal: dict) -> None:
    item = proposal["estimand_review_queue"][0]
    item["observed_unit"] = "reported delta=999"
    item["candidate_key_sha256"] = _json_sha(
        {
            "schema_version": "r3_phase0_estimand_candidate_key_v1",
            "study": item["study"],
            "endpoint": item["endpoint"],
            "observed_unit": item["observed_unit"],
        }
    )
    proposal["estimand_review_queue"].sort(
        key=lambda row: row["candidate_key_sha256"]
    )


def test_presence_aware_v4_contract_exactly_matches_upstream_builder():
    assert SCHEMA_VERSION == (
        "r3_phase0_mechanical_proposal_"
        "v3_presence_aware_exact_nested_replay"
    )
    assert BINDING_SCHEMA_VERSION == (
        "r3_phase0_exact_input_binding_"
        "v3_presence_aware_manifest_replay"
    )
    assert OUTCOME_COMMITMENT_SCHEMA_VERSION == (
        "r3_phase0_outcome_free_inventory_commitment_"
        "v3_presence_aware_exact_nested_replay"
    )
    exact_pairs = (
        (
            EFFECT_DATASET_MANIFEST_VERSION,
            b0x_builder.EFFECT_DATASET_MANIFEST_VERSION,
        ),
        (EFFECT_DATASET_SCHEMA_VERSION, b0x_builder.EFFECT_DATASET_SCHEMA_VERSION),
        (
            FUNCTIONAL_OBSERVATION_SCHEMA_VERSION,
            b0x_builder.OBSERVATION_SCHEMA_VERSION,
        ),
        (
            INPUT_PRESENCE_PROVENANCE_SCHEMA_VERSION,
            b0x_builder.INPUT_PRESENCE_PROVENANCE_SCHEMA_VERSION,
        ),
        (
            IMPLEMENTATION_BINDING_SCHEMA_VERSION,
            b0x_builder.IMPLEMENTATION_BINDING_SCHEMA_VERSION,
        ),
        (
            EFFECT_QUALIFICATION_STATUS,
            b0x_builder.EFFECT_ROW_QUALIFICATION_STATUS,
        ),
        (
            MANIFEST_QUALIFICATION_STATUS,
            b0x_builder.MANIFEST_QUALIFICATION_STATUS,
        ),
        (
            DIFF_REF_ALT_POLICY_SCHEMA_VERSION,
            b0x_builder.DIFF_REF_ALT_JOIN_POLICY_SCHEMA_VERSION,
        ),
        (
            DIFF_REF_ALT_AUDIT_SCHEMA_VERSION,
            b0x_builder.DIFF_REF_ALT_JOIN_AUDIT_SCHEMA_VERSION,
        ),
    )
    assert all(local == upstream for local, upstream in exact_pairs)
    assert IMPLEMENTATION_BINDING_FIELDS == (
        b0x_builder.IMPLEMENTATION_BINDING_KEYS
    )
    assert len(IMPLEMENTATION_BINDING_FIELDS) == 15
    assert DIFF_REF_ALT_POLICY_FIELDS == (
        b0x_builder.DIFF_REF_ALT_JOIN_POLICY_KEYS
    )
    assert len(DIFF_REF_ALT_POLICY_FIELDS) == 18
    assert DIFF_REF_ALT_AUDIT_FIELDS == (
        b0x_builder.DIFF_REF_ALT_JOIN_AUDIT_KEYS
    )
    assert len(DIFF_REF_ALT_AUDIT_FIELDS) == 39
    assert "functional_observation_input_presence_schema_version" in (
        EFFECT_MANIFEST_FIELDS
    )
    assert REQUIRED_DIFF_REF_ALT_POLICY_SHA256 == {
        study: _json_sha(b0x_config.DELTA_SPECS[study]["join_policy"])
        for study in sorted(REQUIRED_DIFF_REF_ALT_POLICY_SHA256)
    }


def test_builds_only_non_active_outcome_free_mechanical_proposal(tmp_path):
    proposal = _build(_fixture(tmp_path / "ordinary"))
    assert validate_phase0_proposal(proposal) == proposal
    assert proposal["status"] == "PROPOSAL_NON_ACTIVE"
    assert proposal["training_ready"] is False
    assert proposal["phase0_ready"] is False
    assert proposal["split_created"] is False
    assert proposal["ordinary_only"] is True
    assert proposal["sealed_access"] is False
    assert proposal["internal_test_access"] is False
    assert proposal["mechanical_candidate_inventory"]["effect_row_count"] == 2
    assert proposal["mechanical_candidate_inventory"][
        "legacy_token_edit_script_row_count"
    ] == 1
    assert proposal["context_candidate_inventory"][
        "library_missing_policy"
    ] == "PRESERVE_NULL_NO_IMPUTATION"
    assert proposal["context_candidate_inventory"]["missing_field_counts"][
        "library"
    ] == 1
    mrl = next(
        item
        for item in proposal["estimand_review_queue"]
        if item["endpoint"] == "ep_rl"
    )
    assert mrl["mechanical_endpoint_family_hint"] == "RIBOSOME_LOADING_MRL"
    assert "stabil" not in _canonical(mrl).lower()
    assert all(
        atom["edge_decision"] == "UNDECIDED_HUMAN_REVIEW_REQUIRED"
        for atom in proposal["relation_atom_inventory"]["coverage_by_atom"]
    )
    prohibited_keys = {
        "delta",
        "sign",
        "outcome",
        "source_value",
        "candidate_value",
        "row_id",
        "row_ids",
        "record_id",
        "record_ids",
        "pair_id",
        "pair_ids",
        "member_ids",
        "members",
        "source_sequence",
        "candidate_sequence",
        "normalized_sequence",
    }
    assert not {key.lower() for key, _ in _walk(proposal)} & prohibited_keys
    _assert_no_raw_identity_values(proposal)


def test_pair_status_uses_canonical_authority_and_missing_stays_unreviewed(
    tmp_path,
):
    args = _fixture(tmp_path / "ordinary")
    pair_rows = _read_jsonl(args["d1_paths"]["utr_edit_pairs"])
    pair_rows[1].pop("relation_acceptance_status")
    assert pair_rows[1]["relation_status"] == "ACCEPTED"
    _write_jsonl(args["d1_paths"]["utr_edit_pairs"], pair_rows)
    _refresh_d1_bindings(args)

    proposal = _build(args)
    audit = proposal["relation_atom_inventory"]["pair_join_audit"]
    assert audit["relation_status_authority_field"] == (
        "relation_acceptance_status"
    )
    assert audit["relation_status_allowlist"] == [
        "ACCEPTED",
        "AMBIGUOUS",
        "CANDIDATE",
        "REJECTED",
    ]
    assert audit["d1_relation_status_counts"] == {
        "ACCEPTED": 1,
        "UNREVIEWED": 1,
    }
    assert audit["d1_relation_status_source_field_counts"] == {
        "MISSING_NOT_ASSERTED": 1,
        "relation_acceptance_status": 1,
    }
    assert audit["d1_accepted_pair_match_count"] == 1
    _assert_no_raw_identity_values(proposal)


@pytest.mark.parametrize(
    ("canonical", "legacy", "error"),
    [
        ("REJECTED", "ACCEPTED", "conflicts with legacy"),
        ("APPROVED", None, "unsupported canonical"),
        (" ACCEPTED ", None, "trim-equal"),
    ],
)
def test_pair_status_conflict_or_non_allowlisted_canonical_fails_closed(
    canonical,
    legacy,
    error,
    tmp_path,
):
    args = _fixture(tmp_path / f"ordinary_{canonical.lower()}")
    pair_rows = _read_jsonl(args["d1_paths"]["utr_edit_pairs"])
    pair_rows[0]["relation_acceptance_status"] = canonical
    if legacy is None:
        pair_rows[0].pop("relation_status")
    else:
        pair_rows[0]["relation_status"] = legacy
    _write_jsonl(args["d1_paths"]["utr_edit_pairs"], pair_rows)
    _refresh_d1_bindings(args)

    with pytest.raises(Phase0ProposalError, match=error):
        _build(args)


def test_assignment_to_existing_group_requires_registry_membership(tmp_path):
    args = _fixture(tmp_path / "ordinary")
    registry_rows = _read_jsonl(args["d1_paths"]["group_registry"])
    registry_rows = [
        _group_row("group_source_shared", "SOURCE", ["pair_alpha"])
        if row["group_id"] == "group_source_shared"
        else row
        for row in registry_rows
    ]
    _write_jsonl(args["d1_paths"]["group_registry"], registry_rows)
    _refresh_d1_bindings(args)

    proposal = _build(args)
    relation = proposal["relation_atom_inventory"]
    assert relation["assignment_missing_registry_group_count"] == 0
    assert relation["assignment_pair_not_registry_member_count"] == 1
    assert relation["review_disposition"] == "HOLD_FOR_HUMAN_REVIEW"
    assert "ASSIGNMENT_PAIR_NOT_REGISTRY_MEMBER" in relation["hold_reasons"]
    source = next(
        row for row in relation["coverage_by_atom"] if row["atom_type"] == "SOURCE"
    )
    assert source["effect_pair_count_with_assignment"] == 1
    assert relation["registry_effect_pair_membership_count"] == 5
    _assert_no_raw_identity_values(proposal)


def test_assignment_algorithm_mismatch_is_hold_and_excluded_from_coverage(
    tmp_path,
):
    args = _fixture(tmp_path / "ordinary")
    assignments = _read_jsonl(args["d1_paths"]["group_assignments"])
    for row in assignments:
        if row["assignment_id"] == "assignment_pair_beta_source":
            row["assignment_algorithm_id"] = "UNBOUND_ALGORITHM"
    _write_jsonl(args["d1_paths"]["group_assignments"], assignments)
    _refresh_d1_bindings(args)

    proposal = _build(args)
    relation = proposal["relation_atom_inventory"]
    assert relation["assignment_algorithm_mismatch_count"] == 1
    assert relation["review_disposition"] == "HOLD_FOR_HUMAN_REVIEW"
    assert "ASSIGNMENT_ALGORITHM_MISMATCH" in relation["hold_reasons"]
    source = next(
        row for row in relation["coverage_by_atom"] if row["atom_type"] == "SOURCE"
    )
    assert source["effect_pair_count_with_assignment"] == 1
    assert relation["registry_effect_pair_membership_count"] == 6
    _assert_no_raw_identity_values(proposal)


def test_registry_member_ids_and_group_hash_are_verified_before_replay(tmp_path):
    args = _fixture(tmp_path / "ordinary")
    registry_rows = _read_jsonl(args["d1_paths"]["group_registry"])
    for row in registry_rows:
        if row["group_id"] == "group_source_shared":
            row["member_ids"] = ["pair_beta", "pair_alpha", "pair_alpha"]
        elif row["group_id"] == "group_study":
            row["group_sha256"] = "0" * 64
    _write_jsonl(args["d1_paths"]["group_registry"], registry_rows)
    _refresh_d1_bindings(args)

    proposal = _build(args)
    relation = proposal["relation_atom_inventory"]
    assert relation["registry_member_ids_invalid_count"] == 1
    assert relation["registry_group_sha_mismatch_count"] == 1
    assert relation["assignment_registry_group_invalid_count"] == 4
    assert relation["registry_effect_pair_membership_count"] == 2
    assert relation["review_disposition"] == "HOLD_FOR_HUMAN_REVIEW"
    assert set(relation["hold_reasons"]) >= {
        "REGISTRY_MEMBER_IDS_INVALID",
        "REGISTRY_GROUP_SHA_MISMATCH",
        "ASSIGNMENT_REGISTRY_GROUP_INVALID",
    }
    coverage = {
        row["atom_type"]: row for row in relation["coverage_by_atom"]
    }
    assert coverage["SOURCE"]["effect_pair_count_with_assignment"] == 0
    assert coverage["STUDY"]["effect_pair_count_with_assignment"] == 0
    _assert_no_raw_identity_values(proposal)


def test_inventory_is_row_order_and_numeric_outcome_invariant(tmp_path):
    first = _build(_fixture(tmp_path / "first"))
    changed = _build(
        _fixture(
            tmp_path / "changed",
            reverse_effect=True,
            outcome_shift=1000.0,
        )
    )
    assert first["input_authorities"]["effect_dataset"]["sha256"] != changed[
        "input_authorities"
    ]["effect_dataset"]["sha256"]
    assert (
        first["outcome_free_inventory_sha256"]
        == changed["outcome_free_inventory_sha256"]
    )
    for field in (
        "context_candidate_inventory",
        "estimand_review_queue",
        "relation_atom_inventory",
        "mechanical_candidate_inventory",
    ):
        assert first[field] == changed[field]


def test_registry_and_assignment_row_order_do_not_change_inventory(tmp_path):
    first = _build(_fixture(tmp_path / "first"))
    reversed_d1 = _build(
        _fixture(
            tmp_path / "reversed",
            reverse_registry=True,
            reverse_assignments=True,
        )
    )
    for authority in ("group_registry", "group_assignments"):
        assert first["input_authorities"][authority]["sha256"] != reversed_d1[
            "input_authorities"
        ][authority]["sha256"]
    assert first["outcome_free_inventory_sha256"] == reversed_d1[
        "outcome_free_inventory_sha256"
    ]
    for field in (
        "context_candidate_inventory",
        "estimand_review_queue",
        "relation_atom_inventory",
        "mechanical_candidate_inventory",
    ):
        assert first[field] == reversed_d1[field]
    _assert_no_raw_identity_values(reversed_d1)


def test_exact_hash_and_manifest_tamper_fail_before_output(tmp_path):
    args = _fixture(tmp_path / "ordinary")
    args["effect_sha256"] = "0" * 64
    output = tmp_path / "proposal.json"
    with pytest.raises(Phase0ProposalError, match="SHA-256"):
        _build(args)
    assert not output.exists()

    args = _fixture(tmp_path / "ordinary_two")
    args["d1_expected_sha256"] = dict(args["d1_expected_sha256"])
    args["d1_expected_sha256"]["group_registry"] = "f" * 64
    with pytest.raises(Phase0ProposalError, match="SHA-256"):
        _build(args)


@pytest.mark.parametrize("protected_name", ["sealed_input", "GSE246381", "internal_test_data"])
def test_protected_paths_are_rejected(protected_name, tmp_path):
    args = _fixture(tmp_path / protected_name)
    with pytest.raises(Phase0ProposalError, match="protected"):
        _build(args)


def test_nested_internal_test_or_sealed_row_is_rejected(tmp_path):
    args = _fixture(tmp_path / "ordinary")
    rows = _effect_rows()
    rows[0]["metadata"] = {"future_partition": "internal_test"}
    _write_jsonl(args["effect_path"], rows)
    args["effect_sha256"] = _sha(args["effect_path"])
    manifest = json.loads(args["effect_manifest_path"].read_text(encoding="utf-8"))
    manifest["sha256"] = args["effect_sha256"]
    _write_json(args["effect_manifest_path"], manifest)
    args["effect_manifest_sha256"] = _sha(args["effect_manifest_path"])
    with pytest.raises(Phase0ProposalError, match="protected"):
        _build(args)


def test_proposal_cannot_be_relabelled_active_or_ready(tmp_path):
    proposal = _build(_fixture(tmp_path / "ordinary"))
    for mutation in (
        lambda value: value.__setitem__("status", "FROZEN"),
        lambda value: value.__setitem__("training_ready", True),
        lambda value: value.__setitem__("phase0_ready", True),
        lambda value: value.__setitem__("split_created", True),
    ):
        forged = copy.deepcopy(proposal)
        mutation(forged)
        with pytest.raises(Phase0ProposalError):
            validate_phase0_proposal(forged)


def test_rehashed_nested_capability_and_text_attacks_fail_closed(tmp_path):
    proposal = _build(_fixture(tmp_path / "ordinary"))
    attacks = [
        (
            "input authority extra capability",
            lambda value: value["input_authorities"]["effect_dataset"].update(
                training_ready=False
            ),
        ),
        (
            "binding check extra capability",
            lambda value: value["binding_checks"].update(split_created=False),
        ),
        (
            "binding bool replaced by integer",
            lambda value: value["binding_checks"].update(
                all_jsonl_inputs_streamed=1
            ),
        ),
        (
            "false all-seven D1 claim",
            lambda value: value["binding_checks"].update(
                d1_sha256s_binds_all_required_d1_authorities=True
            ),
        ),
        (
            "policy changed to allowed",
            lambda value: value["inventory_policy"].update(
                context_inference="ALLOWED"
            ),
        ),
        (
            "policy bool replaced by integer",
            lambda value: value["inventory_policy"].update(ordinary_only=1),
        ),
        (
            "context inventory nested status",
            lambda value: value["context_candidate_inventory"].update(
                training_ready=False
            ),
        ),
        (
            "context candidate attacker note",
            lambda value: value["context_candidate_inventory"]["candidates"][
                0
            ].update(attacker_note="capability enabled"),
        ),
        (
            "estimand attacker note",
            lambda value: value["estimand_review_queue"][0].update(
                attacker_note="scientific admission"
            ),
        ),
        (
            "relation extra capability",
            lambda value: value["relation_atom_inventory"].update(
                training_ready=False
            ),
        ),
        (
            "pair join nested status",
            lambda value: value["relation_atom_inventory"][
                "pair_join_audit"
            ].update(status="PROPOSAL_NON_ACTIVE"),
        ),
        (
            "projection attacker note",
            lambda value: value["relation_atom_inventory"][
                "projection_metadata"
            ].update(attacker_note="authority admitted"),
        ),
        (
            "coverage extra capability",
            lambda value: value["relation_atom_inventory"]["coverage_by_atom"][
                0
            ].update(training_ready=False),
        ),
        (
            "hub attacker note",
            lambda value: value["relation_atom_inventory"]["hub_audit"].update(
                attacker_note="edge activated"
            ),
        ),
        (
            "mechanical flag upgrade",
            lambda value: value["mechanical_candidate_inventory"].update(
                outcome_fields_used_for_inventory=True
            ),
        ),
        (
            "mechanical attacker note",
            lambda value: value["mechanical_candidate_inventory"].update(
                attacker_note="raw outcomes included"
            ),
        ),
        (
            "human review list weakened",
            lambda value: value["human_review_required"].pop(),
        ),
        (
            "limitations weakened",
            lambda value: value["proposal_limitations"].pop(),
        ),
        ("raw identity in allowed-shaped context", _forge_context_identity),
        ("raw outcome text in allowed-shaped estimand", _forge_estimand_outcome_text),
    ]
    for label, mutate in attacks:
        forged = copy.deepcopy(proposal)
        mutate(forged)
        _rehash_forged_proposal(forged)
        try:
            validate_phase0_proposal(forged)
        except Phase0ProposalError:
            pass
        else:
            pytest.fail(f"rehashed nested attack was accepted: {label}")


@pytest.mark.parametrize(
    "old_schema",
    [
        "r3_phase0_mechanical_proposal_v1",
        "r3_phase0_mechanical_proposal_v2_exact_nested_replay",
    ],
)
def test_old_proposal_schema_is_rejected_even_when_self_hash_is_recomputed(
    old_schema,
    tmp_path,
):
    proposal = _build(_fixture(tmp_path / "ordinary"))
    proposal["schema_version"] = old_schema
    _rehash_forged_proposal(proposal)
    with pytest.raises(Phase0ProposalError, match="schema_version"):
        validate_phase0_proposal(proposal)


@pytest.mark.parametrize(
    "tamper",
    [
        "manifest_version_v2",
        "manifest_version_v3",
        "effect_schema_v3",
        "functional_schema_v1",
        "manifest_presence_schema_missing",
        "manifest_presence_schema_old",
        "n_records_wrong",
        "n_records_bool",
        "n_records_negative",
        "missing_required_audit",
        "policy_schema_v2",
        "policy_presence_field_missing",
        "audit_schema_v3",
        "audit_verification_v3",
        "audit_policy_hash",
        "audit_conservation",
        "audit_presence_map_missing",
        "audit_presence_unknown_status",
        "audit_presence_zero_entry",
        "audit_presence_conservation",
        "qualification_status_old",
        "qualification_scientific_true",
        "qualification_training_true",
        "qualification_split_true",
        "qualification_extra_key",
        "implementation_extra_key",
        "implementation_missing_key",
        "implementation_schema_v1",
        "implementation_presence_missing",
        "implementation_presence_old",
        "implementation_ordinary_integer",
        "manifest_extra_key",
    ],
)
def test_manifest_v4_schema_presence_and_replay_audit_tamper_fail_closed(
    tamper,
    tmp_path,
):
    args = _fixture(tmp_path / tamper)
    manifest = json.loads(
        args["effect_manifest_path"].read_text(encoding="utf-8")
    )
    if tamper == "manifest_version_v2":
        manifest["manifest_version"] = "b0x_effect_dataset_manifest_v2"
    elif tamper == "manifest_version_v3":
        manifest["manifest_version"] = (
            "b0x_effect_dataset_manifest_v3_diff_ref_alt_replay"
        )
    elif tamper == "effect_schema_v3":
        manifest["effect_dataset_schema_version"] = (
            "b0x_effect_row_v3_diff_ref_alt_independent_replay"
        )
    elif tamper == "functional_schema_v1":
        manifest["functional_observation_schema_version"] = (
            "lossless_functional_observations_v1"
        )
    elif tamper == "manifest_presence_schema_missing":
        manifest.pop("functional_observation_input_presence_schema_version")
    elif tamper == "manifest_presence_schema_old":
        manifest[
            "functional_observation_input_presence_schema_version"
        ] = "functional_observation_input_presence_v0"
    elif tamper == "n_records_wrong":
        manifest["n_records"] += 1
    elif tamper == "n_records_bool":
        manifest["n_records"] = True
    elif tamper == "n_records_negative":
        manifest["n_records"] = -1
    elif tamper == "missing_required_audit":
        manifest["diff_ref_alt_join_audits"].pop("GSE232572")
    elif tamper == "policy_schema_v2":
        manifest["diff_ref_alt_join_audits"]["GSE232572"]["join_policy"][
            "schema_version"
        ] = "diff_ref_alt_join_policy_v2_explicit_null_semantics"
    elif tamper == "policy_presence_field_missing":
        manifest["diff_ref_alt_join_audits"]["GSE232572"]["join_policy"].pop(
            "presence_match"
        )
    elif tamper == "audit_schema_v3":
        manifest["diff_ref_alt_join_audits"]["GSE232572"][
            "schema_version"
        ] = "diff_ref_alt_join_audit_v3_independent_replay"
    elif tamper == "audit_verification_v3":
        manifest["diff_ref_alt_join_audits"]["GSE232572"][
            "verification_status"
        ] = "INDEPENDENT_REPLAY_VERIFIED"
    elif tamper == "audit_policy_hash":
        manifest["diff_ref_alt_join_audits"]["GSE232572"][
            "join_policy_sha256"
        ] = "0" * 64
    elif tamper == "audit_conservation":
        manifest["diff_ref_alt_join_audits"]["GSE232572"][
            "n_output_join_rows"
        ] = 1
    elif tamper == "audit_presence_map_missing":
        manifest["diff_ref_alt_join_audits"]["GSE232572"].pop(
            "ref_assay_context_input_presence_group_counts"
        )
    elif tamper == "audit_presence_unknown_status":
        manifest["diff_ref_alt_join_audits"]["GSE232572"][
            "ref_assay_context_input_presence_group_counts"
        ] = {"NOT_APPLICABLE_NO_OBSERVATION": 1}
    elif tamper == "audit_presence_zero_entry":
        manifest["diff_ref_alt_join_audits"]["GSE232572"][
            "ref_assay_context_input_presence_group_counts"
        ] = {"EXPLICIT_VALUE": 0}
    elif tamper == "audit_presence_conservation":
        manifest["diff_ref_alt_join_audits"]["GSE232572"][
            "ref_assay_context_input_presence_group_counts"
        ] = {"EXPLICIT_VALUE": 1}
    elif tamper == "qualification_status_old":
        manifest["qualification"]["status"] = (
            "MECHANICAL_REPLAY_VERIFIED_WITH_MATCHING_V3_MANIFEST_"
            "SCIENTIFIC_HOLD"
        )
    elif tamper == "qualification_scientific_true":
        manifest["qualification"]["scientific_qualification"] = True
    elif tamper == "qualification_training_true":
        manifest["qualification"]["training_ready"] = True
    elif tamper == "qualification_split_true":
        manifest["qualification"]["split_ready"] = True
    elif tamper == "qualification_extra_key":
        manifest["qualification"]["attacker_note"] = "still non-active"
    elif tamper == "implementation_extra_key":
        manifest["implementation_binding"]["training_ready"] = False
    elif tamper == "implementation_missing_key":
        manifest["implementation_binding"].pop("estimand_authority_created")
    elif tamper == "implementation_schema_v1":
        manifest["implementation_binding"]["schema_version"] = (
            "b0x_implementation_binding_v1"
        )
    elif tamper == "implementation_presence_missing":
        manifest["implementation_binding"].pop(
            "functional_observation_input_presence_schema_version"
        )
    elif tamper == "implementation_presence_old":
        manifest["implementation_binding"][
            "functional_observation_input_presence_schema_version"
        ] = "functional_observation_input_presence_v0"
    elif tamper == "implementation_ordinary_integer":
        manifest["implementation_binding"]["ordinary_only"] = 1
    elif tamper == "manifest_extra_key":
        manifest["attacker_note"] = "unknown top-level metadata"
    else:  # pragma: no cover - parametrization is exact
        raise AssertionError(tamper)
    _write_json(args["effect_manifest_path"], manifest)
    args["effect_manifest_sha256"] = _sha(args["effect_manifest_path"])

    with pytest.raises(Phase0ProposalError):
        _build(args)


def test_effect_row_v3_schema_tamper_fails_closed(tmp_path):
    args = _fixture(tmp_path / "ordinary")
    rows = _read_jsonl(args["effect_path"])
    rows[0]["effect_dataset_schema_version"] = (
        "b0x_effect_row_v3_diff_ref_alt_independent_replay"
    )
    _write_jsonl(args["effect_path"], rows)
    args["effect_sha256"] = _sha(args["effect_path"])
    manifest = json.loads(
        args["effect_manifest_path"].read_text(encoding="utf-8")
    )
    manifest["sha256"] = args["effect_sha256"]
    _write_json(args["effect_manifest_path"], manifest)
    args["effect_manifest_sha256"] = _sha(args["effect_manifest_path"])

    with pytest.raises(Phase0ProposalError, match="schema"):
        _build(args)


def test_effect_row_old_qualification_status_fails_closed(tmp_path):
    args = _fixture(tmp_path / "ordinary")
    rows = _read_jsonl(args["effect_path"])
    rows[0]["qualification_status"] = (
        "MECHANICAL_REPLAY_REQUIRES_MATCHING_V3_MANIFEST_SCIENTIFIC_HOLD"
    )
    _write_jsonl(args["effect_path"], rows)
    args["effect_sha256"] = _sha(args["effect_path"])
    manifest = json.loads(
        args["effect_manifest_path"].read_text(encoding="utf-8")
    )
    manifest["sha256"] = args["effect_sha256"]
    _write_json(args["effect_manifest_path"], manifest)
    args["effect_manifest_sha256"] = _sha(args["effect_manifest_path"])

    with pytest.raises(Phase0ProposalError, match="schema binding"):
        _build(args)


@pytest.mark.parametrize(
    "tamper",
    [
        "row_functional_schema_v1",
        "functional_observations_missing",
        "presence_missing",
        "presence_extra_key",
        "presence_schema_old",
        "presence_status_unknown",
        "explicit_null_value_conflict",
        "explicit_value_null_conflict",
        "canonicalized_field_missing",
    ],
)
def test_effect_observation_v2_presence_tamper_fails_closed(tamper, tmp_path):
    args = _fixture(tmp_path / tamper)
    rows = _read_jsonl(args["effect_path"])
    observation = rows[0]["functional_observations"][0]
    presence = observation["input_presence_provenance"]
    if tamper == "row_functional_schema_v1":
        rows[0]["functional_observation_schema_version"] = (
            "lossless_functional_observations_v1"
        )
    elif tamper == "functional_observations_missing":
        rows[0].pop("functional_observations")
    elif tamper == "presence_missing":
        observation.pop("input_presence_provenance")
    elif tamper == "presence_extra_key":
        presence["attacker_note"] = "presence upgraded"
    elif tamper == "presence_schema_old":
        presence["schema_version"] = "functional_observation_input_presence_v0"
    elif tamper == "presence_status_unknown":
        presence["assay_context"] = "INFERRED_VALUE"
    elif tamper == "explicit_null_value_conflict":
        presence["assay_context"] = "EXPLICIT_NULL"
    elif tamper == "explicit_value_null_conflict":
        observation["assay_context"] = None
    elif tamper == "canonicalized_field_missing":
        observation.pop("replicate")
    else:  # pragma: no cover - parametrization is exact
        raise AssertionError(tamper)
    _write_jsonl(args["effect_path"], rows)
    args["effect_sha256"] = _sha(args["effect_path"])
    manifest = json.loads(
        args["effect_manifest_path"].read_text(encoding="utf-8")
    )
    manifest["sha256"] = args["effect_sha256"]
    _write_json(args["effect_manifest_path"], manifest)
    args["effect_manifest_sha256"] = _sha(args["effect_manifest_path"])

    with pytest.raises(Phase0ProposalError):
        _build(args)


@pytest.mark.parametrize(
    ("tamper", "error"),
    [
        ("normalized_hash", "normalized content hash mismatch"),
        ("raw_hash", "raw content hash mismatch"),
    ],
)
def test_embedded_observation_hash_tamper_with_outer_rehash_fails_closed(
    tamper,
    error,
    tmp_path,
):
    args = _fixture(tmp_path / tamper)
    rows = _read_jsonl(args["effect_path"])
    observation = rows[0]["functional_observations"][0]
    if tamper == "normalized_hash":
        observation["input_row_sha256"] = "0" * 64
    elif tamper == "raw_hash":
        observation["raw_input_row_sha256"] = "0" * 64
        normalized = dict(observation)
        normalized.pop("input_row_sha256")
        observation["input_row_sha256"] = _json_sha(normalized)
    else:  # pragma: no cover - parametrization is exact
        raise AssertionError(tamper)
    _write_jsonl(args["effect_path"], rows)
    args["effect_sha256"] = _sha(args["effect_path"])
    manifest = json.loads(
        args["effect_manifest_path"].read_text(encoding="utf-8")
    )
    manifest["sha256"] = args["effect_sha256"]
    _write_json(args["effect_manifest_path"], manifest)
    args["effect_manifest_sha256"] = _sha(args["effect_manifest_path"])

    with pytest.raises(Phase0ProposalError, match=error):
        _build(args)


def test_embedded_observation_extra_reserved_field_fails_closed(tmp_path):
    args = _fixture(tmp_path / "ordinary")
    rows = _read_jsonl(args["effect_path"])
    observation = rows[0]["functional_observations"][0]
    observation["raw_input_attacker_note"] = "forged reserved provenance"
    normalized = dict(observation)
    normalized.pop("input_row_sha256")
    observation["input_row_sha256"] = _json_sha(normalized)
    _write_jsonl(args["effect_path"], rows)
    args["effect_sha256"] = _sha(args["effect_path"])
    manifest = json.loads(
        args["effect_manifest_path"].read_text(encoding="utf-8")
    )
    manifest["sha256"] = args["effect_sha256"]
    _write_json(args["effect_manifest_path"], manifest)
    args["effect_manifest_sha256"] = _sha(args["effect_manifest_path"])

    with pytest.raises(Phase0ProposalError, match="extra reserved"):
        _build(args)


def _configure_one_matched_presence_aware_diff(args: dict) -> tuple[list, dict]:
    rows = _read_jsonl(args["effect_path"])
    manifest = json.loads(
        args["effect_manifest_path"].read_text(encoding="utf-8")
    )
    audit = manifest["diff_ref_alt_join_audits"]["GSE232572"]
    rows[0].update(
        {
            "study": "GSE232572",
            "diff_ref_alt_join_policy_schema_version": (
                DIFF_REF_ALT_POLICY_SCHEMA_VERSION
            ),
            "diff_ref_alt_join_policy_id": audit["policy_id"],
            "diff_ref_alt_join_policy_sha256": audit["join_policy_sha256"],
            "comparison_scale_id": audit["comparison_scale_id"],
            "diff_ref_alt_join_disposition": "MATCHED",
            "ref_observation_group_present": True,
            "alt_observation_group_present": True,
            "observation_join_key": {
                "context_id": "legacy_context_a",
                "assay_context": "legacy_context_a",
                "assay_context_input_presence": "EXPLICIT_VALUE",
                "replicate": 1,
                "replicate_input_presence": "EXPLICIT_VALUE",
                "comparison_scale_id": audit["comparison_scale_id"],
            },
        }
    )
    audit.update(
        {
            "n_materialized_pairs": 1,
            "n_output_pairs": 1,
            "n_output_join_rows": 1,
            "n_ref_observation_groups": 1,
            "n_alt_observation_groups": 1,
            "n_input_side_groups": 2,
            "n_consumed_ref_observation_groups": 1,
            "n_consumed_alt_observation_groups": 1,
            "n_consumed_side_groups": 2,
            "n_matched_groups": 1,
            "n_derived_rows": 1,
            "expected_output_rows_from_group_conservation": 1,
            "join_disposition_counts": {"MATCHED": 1},
            "ref_assay_context_input_presence_group_counts": {
                "EXPLICIT_VALUE": 1
            },
            "alt_assay_context_input_presence_group_counts": {
                "EXPLICIT_VALUE": 1
            },
            "ref_replicate_input_presence_group_counts": {
                "EXPLICIT_VALUE": 1
            },
            "alt_replicate_input_presence_group_counts": {
                "EXPLICIT_VALUE": 1
            },
        }
    )
    _write_jsonl(args["effect_path"], rows)
    args["effect_sha256"] = _sha(args["effect_path"])
    manifest["sha256"] = args["effect_sha256"]
    _write_json(args["effect_manifest_path"], manifest)
    args["effect_manifest_sha256"] = _sha(args["effect_manifest_path"])
    return rows, manifest


def test_materialized_diff_dispositions_are_crosschecked_without_outcomes(
    tmp_path,
):
    args = _fixture(tmp_path / "ordinary")
    _, manifest = _configure_one_matched_presence_aware_diff(args)

    proposal = _build(args)
    assert proposal["binding_checks"][
        "diff_ref_alt_materialized_dispositions_crosschecked"
    ] is True
    assert proposal["binding_checks"][
        "diff_ref_alt_materialized_presence_crosschecked"
    ] is True

    forged_manifest = copy.deepcopy(manifest)
    forged_manifest["diff_ref_alt_join_audits"]["GSE232572"] = (
        _empty_diff_ref_alt_audits()["GSE232572"]
    )
    _write_json(args["effect_manifest_path"], forged_manifest)
    args["effect_manifest_sha256"] = _sha(args["effect_manifest_path"])
    with pytest.raises(Phase0ProposalError, match="mechanical audit"):
        _build(args)


@pytest.mark.parametrize(
    "tamper",
    [
        "join_key_missing_field",
        "join_presence_unknown",
        "join_presence_value_conflict",
        "observation_presence_disagrees",
        "audit_presence_map_disagrees",
    ],
)
def test_materialized_diff_presence_tamper_fails_closed(tamper, tmp_path):
    args = _fixture(tmp_path / tamper)
    rows, manifest = _configure_one_matched_presence_aware_diff(args)
    join_key = rows[0]["observation_join_key"]
    if tamper == "join_key_missing_field":
        join_key.pop("replicate_input_presence")
    elif tamper == "join_presence_unknown":
        join_key["assay_context_input_presence"] = "INFERRED_VALUE"
    elif tamper == "join_presence_value_conflict":
        join_key["assay_context_input_presence"] = "EXPLICIT_NULL"
    elif tamper == "observation_presence_disagrees":
        observation = rows[0]["functional_observations"][0]
        observation["replicate"] = None
        observation["input_presence_provenance"]["replicate"] = (
            "EXPLICIT_NULL"
        )
    elif tamper == "audit_presence_map_disagrees":
        manifest["diff_ref_alt_join_audits"]["GSE232572"][
            "ref_assay_context_input_presence_group_counts"
        ] = {"EXPLICIT_NULL": 1}
    else:  # pragma: no cover - parametrization is exact
        raise AssertionError(tamper)
    _write_jsonl(args["effect_path"], rows)
    args["effect_sha256"] = _sha(args["effect_path"])
    manifest["sha256"] = args["effect_sha256"]
    _write_json(args["effect_manifest_path"], manifest)
    args["effect_manifest_sha256"] = _sha(args["effect_manifest_path"])

    with pytest.raises(Phase0ProposalError):
        _build(args)


def test_d1_sha_manifest_binds_six_jsonl_and_projection_is_separate(tmp_path):
    args = _fixture(tmp_path / "ordinary")
    sums_text = args["d1_sha256s_path"].read_text(encoding="utf-8")
    assert D1_FILE_NAMES["grouping_atom_projection"] not in sums_text
    assert len(sums_text.splitlines()) == len(JSONL_D1_AUTHORITIES)
    proposal = _build(args)
    assert proposal["binding_checks"][
        "d1_sha256s_binds_all_required_d1_jsonl_authorities"
    ] is True
    assert proposal["binding_checks"][
        "grouping_atom_projection_exact_sha256_verified_separately"
    ] is True
    assert "d1_sha256s_binds_all_required_d1_authorities" not in proposal[
        "binding_checks"
    ]

    missing = _fixture(tmp_path / "missing_jsonl_sum")
    lines = missing["d1_sha256s_path"].read_text(encoding="utf-8").splitlines()
    missing["d1_sha256s_path"].write_text(
        "\n".join(lines[1:]) + "\n", encoding="utf-8"
    )
    missing["d1_sha256s_sha256"] = _sha(missing["d1_sha256s_path"])
    with pytest.raises(Phase0ProposalError, match="does not bind exact"):
        _build(missing)

    wrong_projection = _fixture(tmp_path / "wrong_projection")
    wrong_projection["d1_expected_sha256"] = dict(
        wrong_projection["d1_expected_sha256"]
    )
    wrong_projection["d1_expected_sha256"]["grouping_atom_projection"] = (
        "f" * 64
    )
    with pytest.raises(Phase0ProposalError, match="SHA-256"):
        _build(wrong_projection)


def test_canonical_output_is_exclusive_and_replays(tmp_path):
    proposal = _build(_fixture(tmp_path / "ordinary"))
    output = tmp_path / "proposal.json"
    digest = write_phase0_proposal_exclusive(output, proposal)
    assert digest == _sha(output)
    assert json.loads(output.read_text(encoding="utf-8")) == proposal
    original = output.read_bytes()
    with pytest.raises(Exception, match="already exists"):
        write_phase0_proposal_exclusive(output, proposal)
    assert output.read_bytes() == original
