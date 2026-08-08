"""B0-X effect dataset builder tests: delta derivation per asset type and the
feature module.  Tests the pure functions with synthetic inputs (no remote
data)."""
import json
import copy
import sys
from pathlib import Path

import numpy as np
import pytest

B0X = Path(__file__).resolve().parents[2] / "scripts" / "b0x"
sys.path.insert(0, str(B0X))

import build_effect_dataset as B  # noqa: E402
import features as F  # noqa: E402


def _record(src="ACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGU",
            cand="ACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACG",
            metadata=None, rid="X_rec"):
    return {
        "record_id": rid,
        "source_sequence": src,
        "candidate_sequence": cand,
        "edit_script": [{"op": "SUB", "pos": 3, "token": "G"}],
        "metadata": metadata or {},
    }


def _diff_ref_alt_spec(
    ref_endpoint="ep_activity_ref_mean",
    alt_endpoint="ep_activity_alt_mean",
    *,
    policy_id=(
        "SYNTHETIC_EXPLICIT_PAIRED_ENDPOINT_SCALE_"
        "V2_EXPLICIT_NULL_SEMANTICS"
    ),
    comparison_scale_id="SYNTHETIC_ACTIVITY_PAIRED_ASSAY_SCALE_V1",
):
    return {
        "type": "diff_ref_alt",
        "alt_endpoint": alt_endpoint,
        "ref_endpoint": ref_endpoint,
        "benchmark": "3U-A1",
        "join_policy": {
            "schema_version": (
                "diff_ref_alt_join_policy_v2_explicit_null_semantics"
            ),
            "policy_id": policy_id,
            "key_fields": ["context_id", "assay_context", "replicate"],
            "context_match": "EXACT",
            "context_required_non_null": True,
            "assay_context_match": "EXACT",
            "assay_context_null_policy": (
                "ALLOW_EXACT_NULL_MECHANICAL_NO_ASSAY_CONTEXT_CLAIM"
            ),
            "replicate_match": "EXACT_CANONICAL_TOKEN_INCLUDING_NULL",
            "replicate_field": "replicate",
            "replicate_null_policy": (
                "ALLOW_EXACT_NULL_NO_REPLICATE_CLAIM"
            ),
            "ref_allowed_raw_unit_tokens": [ref_endpoint],
            "alt_allowed_raw_unit_tokens": [alt_endpoint],
            "comparison_scale_id": comparison_scale_id,
            "raw_unit_semantics": (
                "ENDPOINT_SPECIFIC_RAW_LABEL_TOKEN_NOT_PHYSICAL_UNIT"
            ),
        },
    }


def _functional_observation(
    sequence_id,
    endpoint_id,
    value,
    *,
    unit=None,
    context_id="synthetic_context",
    assay_context="synthetic_assay",
    replicate=None,
    observation_id=None,
):
    return {
        "observation_id": observation_id,
        "sequence_id": sequence_id,
        "endpoint_id": endpoint_id,
        "context_id": context_id,
        "assay_context": assay_context,
        "replicate": replicate,
        "value": value,
        "unit": endpoint_id if unit is None else unit,
    }


def _paired_replay_fixture(*, contexts=("synthetic_context",)):
    study = "GSE232572"
    sid = "GSE232572_X__cand"
    pair_id = "pair_1"
    spec = _diff_ref_alt_spec()
    cand_values = {sid: {
        "ep_activity_ref_mean": [
            _functional_observation(
                sid,
                "ep_activity_ref_mean",
                float(index + 1),
                context_id=context,
                observation_id="ref_%s" % context,
            )
            for index, context in enumerate(contexts)
        ],
        "ep_activity_alt_mean": [
            _functional_observation(
                sid,
                "ep_activity_alt_mean",
                float((index + 1) * 10),
                context_id=context,
                observation_id="alt_%s" % context,
            )
            for index, context in enumerate(contexts)
        ],
    }}
    records = B.derive_delta(
        study, spec, _record(), cand_values, {}, {}, sid
    )
    for row in records:
        row["pair_id"] = pair_id
    materialized_pairs = [{
        "study": study,
        "pair_id": pair_id,
        "candidate_sequence_id": sid,
    }]
    return study, spec, sid, cand_values, materialized_pairs, records


def _replay_fixture_audit(spec, cand_values, materialized_pairs, records):
    return B.summarize_diff_ref_alt_join_audits(
        records,
        {"GSE232572": spec},
        cand_values=cand_values,
        materialized_pairs=materialized_pairs,
    )["GSE232572"]


def test_delta_diff_cand_src_gse114002():
    rec = _record(src="ACGU" * 20, cand="ACGU" * 12 + "ACGU", metadata=None)
    rec["source_sequence"] = "ACGU" * 20
    rec["candidate_sequence"] = "ACGU" * 20
    spec = {"type": "diff_cand_src", "cand_endpoint": "ep_rl", "src_source": "raw_library", "benchmark": "5U-A1"}
    cand_values = {"GSE114002_X__cand": {"ep_rl": 6.0}}
    src_anchors = {"ACGU" * 20: 5.0}
    out = B.derive_delta("GSE114002", spec, rec, cand_values, src_anchors, {}, "GSE114002_X__cand")
    assert len(out) == 1
    assert out[0]["delta_source_status"] == "derived"
    assert out[0]["delta"] == 1.0
    assert out[0]["source_value"] == 5.0
    assert out[0]["candidate_value"] == 6.0


def test_diff_ref_alt_endpoint_units_map_to_registered_comparison_scale():
    rec = _record()
    spec = _diff_ref_alt_spec()
    sid = "GSE232572_X__cand"
    cand_values = {sid: {
        "ep_activity_alt_mean": [_functional_observation(
            sid, "ep_activity_alt_mean", 8.0, observation_id="alt",
        )],
        "ep_activity_ref_mean": [_functional_observation(
            sid, "ep_activity_ref_mean", 5.0, observation_id="ref",
        )],
    }}
    # This is the original failure mode: the lossless global identity must
    # distinguish these observations because their raw unit tokens differ.
    # Only the explicit paired policy may map them to one comparison scale.
    assert B._observation_join_key(
        cand_values[sid]["ep_activity_ref_mean"][0]
    ) != B._observation_join_key(
        cand_values[sid]["ep_activity_alt_mean"][0]
    )
    out = B.derive_delta("GSE232572", spec, rec, cand_values, {}, {}, "GSE232572_X__cand")
    assert len(out) == 1
    assert out[0]["delta"] == 3.0
    assert out[0]["delta_source_status"] == "derived"
    assert out[0]["delta_unavailable_reason"] is None
    assert out[0]["diff_ref_alt_join_disposition"] == "MATCHED"
    assert out[0]["comparison_scale_id"] == (
        "SYNTHETIC_ACTIVITY_PAIRED_ASSAY_SCALE_V1"
    )
    assert out[0]["ref_observation_unit_tokens"] == [
        "ep_activity_ref_mean"
    ]
    assert out[0]["alt_observation_unit_tokens"] == [
        "ep_activity_alt_mean"
    ]
    assert out[0]["observation_join_key"]["replicate"] is None
    assert out[0]["replicate_pairing_status"] == (
        "EXACT_NULL_SLOT_NO_REPLICATE_CLAIM"
    )


def test_diff_ref_alt_unregistered_raw_unit_fails_closed():
    rec = _record()
    spec = _diff_ref_alt_spec()
    sid = "GSE232572_X__cand"
    cand_values = {sid: {
        "ep_activity_ref_mean": [_functional_observation(
            sid,
            "ep_activity_ref_mean",
            5.0,
            unit="activity_ref_mean",
        )],
        "ep_activity_alt_mean": [_functional_observation(
            sid, "ep_activity_alt_mean", 8.0,
        )],
    }}

    with pytest.raises(B.DiffRefAltJoinPolicyError, match="unregistered ref"):
        B.derive_delta(
            "GSE232572", spec, rec, cand_values, {}, {}, sid
        )


@pytest.mark.parametrize(
    ("ref_overrides", "alt_overrides", "expected_reason"),
    [
        (
            {"context_id": "context_ref"},
            {"context_id": "context_alt"},
            "REF_ALT_CONTEXT_ID_MISMATCH",
        ),
        (
            {"assay_context": "assay_ref"},
            {"assay_context": "assay_alt"},
            "REF_ALT_ASSAY_CONTEXT_MISMATCH",
        ),
        (
            {"replicate": None},
            {"replicate": "replicate_alt"},
            "REF_ALT_REPLICATE_MISMATCH",
        ),
    ],
)
def test_diff_ref_alt_exact_pairing_mismatch_never_derives_delta(
    ref_overrides, alt_overrides, expected_reason,
):
    rec = _record()
    spec = _diff_ref_alt_spec()
    sid = "GSE232572_X__cand"
    ref_kwargs = {
        "context_id": "synthetic_context",
        "assay_context": "synthetic_assay",
        "replicate": None,
    }
    alt_kwargs = dict(ref_kwargs)
    ref_kwargs.update(ref_overrides)
    alt_kwargs.update(alt_overrides)
    cand_values = {sid: {
        "ep_activity_ref_mean": [_functional_observation(
            sid, "ep_activity_ref_mean", 5.0, **ref_kwargs,
        )],
        "ep_activity_alt_mean": [_functional_observation(
            sid, "ep_activity_alt_mean", 8.0, **alt_kwargs,
        )],
    }}

    out = B.derive_delta(
        "GSE232572", spec, rec, cand_values, {}, {}, sid
    )

    assert len(out) == 2
    assert {row["diff_ref_alt_join_disposition"] for row in out} == {
        "UNMATCHED_REF_ONLY", "UNMATCHED_ALT_ONLY",
    }
    assert {row["delta_unavailable_reason"] for row in out} == {
        expected_reason
    }
    assert all(row["delta"] is None for row in out)
    assert all(
        row["delta_source_status"] == "source_anchor_unavailable"
        for row in out
    )
    assert sum(len(row["functional_observations"]) for row in out) == 2
    if expected_reason == "REF_ALT_REPLICATE_MISMATCH":
        null_slot = next(
            row for row in out
            if row["observation_join_key"]["replicate"] is None
        )
        assert null_slot["replicate_pairing_status"] == (
            "EXACT_NULL_SLOT_NO_REPLICATE_CLAIM"
        )


def test_diff_ref_alt_one_sided_orphan_emits_one_unavailable_row():
    rec = _record()
    spec = _diff_ref_alt_spec()
    sid = "GSE232572_X__cand"
    cand_values = {sid: {
        "ep_activity_ref_mean": [_functional_observation(
            sid, "ep_activity_ref_mean", 5.0,
        )],
    }}

    out = B.derive_delta(
        "GSE232572", spec, rec, cand_values, {}, {}, sid
    )

    assert len(out) == 1
    assert out[0]["diff_ref_alt_join_disposition"] == "UNMATCHED_REF_ONLY"
    assert out[0]["delta_unavailable_reason"] == "MISSING_ALT_ENDPOINT"
    assert out[0]["delta"] is None
    assert len(out[0]["functional_observations"]) == 1


def test_diff_ref_alt_pairing_is_input_order_invariant():
    rec = _record()
    spec = _diff_ref_alt_spec()
    sid = "GSE232572_X__cand"
    observations_by_endpoint = {
        "ep_activity_ref_mean": [
            _functional_observation(
                sid, "ep_activity_ref_mean", 5.0,
                context_id="context_a", observation_id="ref_a",
            ),
            _functional_observation(
                sid, "ep_activity_ref_mean", 10.0,
                context_id="context_b", observation_id="ref_b",
            ),
        ],
        "ep_activity_alt_mean": [
            _functional_observation(
                sid, "ep_activity_alt_mean", 8.0,
                context_id="context_a", observation_id="alt_a",
            ),
            _functional_observation(
                sid, "ep_activity_alt_mean", 14.0,
                context_id="context_b", observation_id="alt_b",
            ),
        ],
    }
    reverse = {
        endpoint: list(reversed(rows))
        for endpoint, rows in reversed(list(observations_by_endpoint.items()))
    }

    forward_out = B.derive_delta(
        "GSE232572", spec, rec, {sid: observations_by_endpoint},
        {}, {}, sid,
    )
    reverse_out = B.derive_delta(
        "GSE232572", spec, rec, {sid: reverse}, {}, {}, sid,
    )

    assert [B._canonical_json(row) for row in forward_out] == [
        B._canonical_json(row) for row in reverse_out
    ]
    assert [row["delta"] for row in forward_out] == [3.0, 4.0]


def test_all_four_diff_ref_alt_configs_bind_verified_raw_unit_tokens():
    expected = {
        "ENCSR854RUF": {
            "policy_id": (
                "ENCSR854RUF_EXPLICIT_PAIRED_ENDPOINT_SCALE_"
                "V2_EXPLICIT_NULL_SEMANTICS"
            ),
            "ref_endpoint": "ep_log2FoldChange_Ref_HEK293FT",
            "alt_endpoint": "ep_log2FoldChange_Alt_HEK293FT",
            "ref_unit": "log2FoldChange_Ref_HEK293FT",
            "alt_unit": "log2FoldChange_Alt_HEK293FT",
            "scale": "ENCSR854RUF_LOG2_FOLD_CHANGE_PAIRED_ASSAY_SCALE_V1",
        },
        "GSE186455": {
            "policy_id": (
                "GSE186455_EXPLICIT_PAIRED_ENDPOINT_SCALE_"
                "V2_EXPLICIT_NULL_SEMANTICS"
            ),
            "ref_endpoint": "ep_n2a_activity_ref_mean",
            "alt_endpoint": "ep_n2a_activity_alt_mean",
            "ref_unit": "n2a_activity_ref_mean",
            "alt_unit": "n2a_activity_alt_mean",
            "scale": "GSE186455_N2A_ACTIVITY_PAIRED_ASSAY_SCALE_V1",
        },
        "GSE232571": {
            "policy_id": (
                "GSE232571_EXPLICIT_PAIRED_ENDPOINT_SCALE_"
                "V2_EXPLICIT_NULL_SEMANTICS"
            ),
            "ref_endpoint": "ep_activity_HEK293_ref_mean",
            "alt_endpoint": "ep_activity_HEK293_alt_mean",
            "ref_unit": "activity_HEK293_ref_mean",
            "alt_unit": "activity_HEK293_alt_mean",
            "scale": "GSE232571_HEK293_ACTIVITY_PAIRED_ASSAY_SCALE_V1",
        },
        "GSE232572": {
            "policy_id": (
                "GSE232572_EXPLICIT_PAIRED_ENDPOINT_SCALE_"
                "V2_EXPLICIT_NULL_SEMANTICS"
            ),
            "ref_endpoint": "ep_activity_ref_mean",
            "alt_endpoint": "ep_activity_alt_mean",
            "ref_unit": "activity_ref_mean",
            "alt_unit": "activity_alt_mean",
            "scale": "GSE232572_ACTIVITY_PAIRED_ASSAY_SCALE_V1",
        },
    }

    configured = {
        study: spec for study, spec in B.config.DELTA_SPECS.items()
        if spec["type"] == "diff_ref_alt"
    }

    assert set(configured) == set(expected)
    assert len({
        spec["join_policy"]["policy_id"] for spec in configured.values()
    }) == 4
    assert len({
        spec["join_policy"]["comparison_scale_id"]
        for spec in configured.values()
    }) == 4
    for study, authority in expected.items():
        spec = configured[study]
        policy = B.validate_diff_ref_alt_join_policy(spec)
        assert policy["schema_version"] == (
            "diff_ref_alt_join_policy_v2_explicit_null_semantics"
        )
        assert policy["policy_id"] == authority["policy_id"]
        assert spec["ref_endpoint"] == authority["ref_endpoint"]
        assert spec["alt_endpoint"] == authority["alt_endpoint"]
        assert policy["ref_allowed_raw_unit_tokens"] == [
            authority["ref_unit"]
        ]
        assert policy["alt_allowed_raw_unit_tokens"] == [
            authority["alt_unit"]
        ]
        assert policy["comparison_scale_id"] == authority["scale"]
        assert policy["key_fields"] == [
            "context_id", "assay_context", "replicate"
        ]
        assert policy["context_match"] == "EXACT"
        assert policy["context_required_non_null"] is True
        assert policy["assay_context_match"] == "EXACT"
        assert policy["assay_context_null_policy"] == (
            "ALLOW_EXACT_NULL_MECHANICAL_NO_ASSAY_CONTEXT_CLAIM"
        )
        assert policy["replicate_match"] == (
            "EXACT_CANONICAL_TOKEN_INCLUDING_NULL"
        )
        assert policy["replicate_field"] == "replicate"
        assert policy["replicate_null_policy"] == (
            "ALLOW_EXACT_NULL_NO_REPLICATE_CLAIM"
        )
        # Endpoint ids and raw unit labels are distinct authority fields.  The
        # builder must never infer one from an ``ep_`` prefix or suffix.
        assert authority["ref_unit"] != authority["ref_endpoint"]
        assert authority["alt_unit"] != authority["alt_endpoint"]


def test_diff_ref_alt_requires_an_explicit_versioned_join_policy():
    spec = {
        "type": "diff_ref_alt",
        "ref_endpoint": "ep_ref",
        "alt_endpoint": "ep_alt",
        "benchmark": "3U-A1",
    }

    with pytest.raises(B.DiffRefAltJoinPolicyError, match="join_policy"):
        B.derive_delta(
            "SYNTHETIC", spec, _record(), {}, {}, {}, "SYNTHETIC_X__cand"
        )


def test_diff_ref_alt_n_plus_m_side_group_conservation():
    rec = _record()
    spec = _diff_ref_alt_spec()
    sid = "GSE232572_X__cand"
    ref_rows = [
        _functional_observation(
            sid, "ep_activity_ref_mean", value,
            context_id=context, observation_id="ref_" + context,
        )
        for context, value in (
            ("context_a", 1.0),
            ("context_b", 2.0),
            ("context_ref_only", 3.0),
        )
    ]
    alt_rows = [
        _functional_observation(
            sid, "ep_activity_alt_mean", value,
            context_id=context, observation_id="alt_" + context,
        )
        for context, value in (
            ("context_a", 11.0),
            ("context_b", 22.0),
            ("context_alt_only", 33.0),
        )
    ]

    cand_values = {sid: {
        "ep_activity_ref_mean": ref_rows,
        "ep_activity_alt_mean": alt_rows,
    }}
    out = B.derive_delta(
        "GSE232572",
        spec,
        rec,
        cand_values,
        {},
        {},
        sid,
    )
    for row in out:
        row["pair_id"] = "pair_1"
    audit = B.summarize_diff_ref_alt_join_audits(
        out,
        {"GSE232572": spec},
        cand_values=cand_values,
        materialized_pairs=[{
            "study": "GSE232572",
            "pair_id": "pair_1",
            "candidate_sequence_id": sid,
        }],
    )["GSE232572"]

    # N=3 ref groups and M=3 alt groups consume all six raw observations.
    # Two exact matches collapse to two output rows; the two side-orphans are
    # each emitted once, so output = N + M - matched = 4.
    assert sum(len(row["functional_observations"]) for row in out) == 6
    assert len(out) == 4
    assert audit["n_ref_observation_groups"] == 3
    assert audit["n_alt_observation_groups"] == 3
    assert audit["n_input_side_groups"] == 6
    assert audit["n_matched_groups"] == 2
    assert audit["n_unmatched_ref_groups"] == 1
    assert audit["n_unmatched_alt_groups"] == 1
    assert audit["n_output_join_rows"] == 4
    assert audit["expected_output_rows_from_group_conservation"] == 4
    assert audit["n_derived_rows"] == 2
    assert audit["n_unavailable_rows"] == 2


def test_replay_audit_reconstructs_evidence_and_exact_schema_keys():
    _, spec, _, cand_values, materialized_pairs, records = (
        _paired_replay_fixture()
    )

    audit = _replay_fixture_audit(
        spec, cand_values, materialized_pairs, records
    )

    assert set(audit) == B.DIFF_REF_ALT_JOIN_AUDIT_KEYS
    assert audit["schema_version"] == (
        "diff_ref_alt_join_audit_v3_independent_replay"
    )
    assert audit["verification_status"] == "INDEPENDENT_REPLAY_VERIFIED"
    assert audit["expected_input_side_group_multiset_sha256"] == (
        audit["consumed_output_side_group_multiset_sha256"]
    )
    assert audit["materialized_pair_registry_sha256"] == (
        audit["output_pair_registry_sha256"]
    )


def test_replay_audit_rejects_cleared_functional_observations():
    _, spec, _, cand_values, materialized_pairs, records = (
        _paired_replay_fixture()
    )
    tampered = copy.deepcopy(records)
    tampered[0]["functional_observations"] = []
    tampered[0]["functional_observation_hashes"] = []

    with pytest.raises(B.DiffRefAltJoinPolicyError):
        _replay_fixture_audit(
            spec, cand_values, materialized_pairs, tampered
        )


def test_replay_audit_rejects_coordinated_delta_and_value_tampering():
    _, spec, _, cand_values, materialized_pairs, records = (
        _paired_replay_fixture()
    )
    tampered = copy.deepcopy(records)
    row = tampered[0]
    ref_rows = []
    for observation in row["functional_observations"]:
        if observation["endpoint_id"] == spec["ref_endpoint"]:
            observation["value"] = 1000.0
            ref_rows.append(observation)
        else:
            observation["value"] = 1999.0
    row["source_value"] = 1000.0
    row["candidate_value"] = 1999.0
    row["delta"] = 999.0
    row["functional_observation_hashes"] = sorted(
        B._json_sha256(observation)
        for observation in row["functional_observations"]
    )
    row["source_anchor_evidence"] = sorted(
        ref_rows, key=B._canonical_json
    )
    row["source_anchor_measured_values"] = [1000.0]

    with pytest.raises(
        B.DiffRefAltJoinPolicyError, match="hash multiset"
    ):
        _replay_fixture_audit(
            spec, cand_values, materialized_pairs, tampered
        )


@pytest.mark.parametrize(
    "tamper_kind", [
        "endpoint", "unit", "key", "hash", "observation_ids", "top_unit",
    ],
)
def test_replay_audit_rejects_wrong_endpoint_unit_key_or_hash(tamper_kind):
    _, spec, _, cand_values, materialized_pairs, records = (
        _paired_replay_fixture()
    )
    tampered = copy.deepcopy(records)
    row = tampered[0]
    if tamper_kind == "endpoint":
        row["functional_observations"][0]["endpoint_id"] = "ep_unexpected"
    elif tamper_kind == "unit":
        row["functional_observations"][0]["unit"] = "unregistered_unit"
    elif tamper_kind == "key":
        row["observation_join_key"]["context_id"] = "wrong_context"
    elif tamper_kind == "hash":
        row["functional_observations"][0]["input_row_sha256"] = "0" * 64
        row["functional_observation_hashes"] = sorted(
            observation.get("input_row_sha256")
            or B._json_sha256(observation)
            for observation in row["functional_observations"]
        )
    elif tamper_kind == "observation_ids":
        row["observation_ids"] = ["forged_id"]
    else:
        row["unit"] = "forged_common_unit"

    with pytest.raises(B.DiffRefAltJoinPolicyError):
        _replay_fixture_audit(
            spec, cand_values, materialized_pairs, tampered
        )


@pytest.mark.parametrize("tamper_kind", ["value", "context_key"])
def test_replay_rehash_rejects_content_tamper_with_declared_hash_unchanged(
    tamper_kind,
):
    study, spec, sid, cand_values, materialized_pairs, _ = (
        _paired_replay_fixture()
    )
    for endpoint_rows in cand_values[sid].values():
        for observation in endpoint_rows:
            observation["input_row_sha256"] = B._json_sha256(observation)
    records = B.derive_delta(
        study, spec, _record(), cand_values, {}, {}, sid
    )
    records[0]["pair_id"] = "pair_1"
    tampered = copy.deepcopy(records)
    observation = tampered[0]["functional_observations"][0]
    if tamper_kind == "value":
        observation["value"] = float(observation["value"]) + 1.0
    else:
        observation["context_id"] = "tampered_context"

    with pytest.raises(
        B.DiffRefAltJoinPolicyError, match="content disagrees"
    ):
        _replay_fixture_audit(
            spec, cand_values, materialized_pairs, tampered
        )


def test_loader_namespaces_predeclared_hash_and_mints_self_verifiable_hash(
    tmp_path,
):
    sid = "GSE232572_X__cand"
    observation = _functional_observation(
        sid, "ep_activity_ref_mean", 1.0
    )
    observation["input_row_sha256"] = B._json_sha256(observation)
    path = tmp_path / "functional_observations.jsonl"
    path.write_text(json.dumps(observation) + "\n", encoding="utf-8")

    loaded = B.load_candidate_observations(
        path, {"GSE232572": {sid}}
    )[sid]["ep_activity_ref_mean"][0]

    assert loaded["upstream_declared_input_row_sha256"] != (
        loaded["input_row_sha256"]
    )
    assert B._replay_input_row_hash(loaded) == loaded["input_row_sha256"]


def test_replay_audit_rejects_wrong_ref_anchor_evidence():
    _, spec, _, cand_values, materialized_pairs, records = (
        _paired_replay_fixture()
    )
    tampered = copy.deepcopy(records)
    tampered[0]["source_anchor_evidence"] = []

    with pytest.raises(
        B.DiffRefAltJoinPolicyError, match="ref anchor"
    ):
        _replay_fixture_audit(
            spec, cand_values, materialized_pairs, tampered
        )


def test_replay_audit_rejects_duplicate_group_consumption():
    _, spec, _, cand_values, materialized_pairs, records = (
        _paired_replay_fixture()
    )
    duplicated = copy.deepcopy(records) + copy.deepcopy(records)

    with pytest.raises(
        B.DiffRefAltJoinPolicyError, match="hash multiset"
    ):
        _replay_fixture_audit(
            spec, cand_values, materialized_pairs, duplicated
        )


def test_replay_audit_rejects_splitting_an_exact_match_into_orphans():
    _, spec, _, cand_values, materialized_pairs, records = (
        _paired_replay_fixture()
    )
    ref_row = copy.deepcopy(records[0])
    alt_row = copy.deepcopy(records[0])
    ref_row["functional_observations"] = [
        observation for observation in ref_row["functional_observations"]
        if observation["endpoint_id"] == spec["ref_endpoint"]
    ]
    alt_row["functional_observations"] = [
        observation for observation in alt_row["functional_observations"]
        if observation["endpoint_id"] == spec["alt_endpoint"]
    ]

    with pytest.raises(B.DiffRefAltJoinPolicyError, match="split an exact"):
        _replay_fixture_audit(
            spec, cand_values, materialized_pairs, [ref_row, alt_row]
        )


def test_replay_audit_rejects_derive_output_that_dropped_one_group():
    _, spec, _, cand_values, materialized_pairs, records = (
        _paired_replay_fixture(contexts=("context_a", "context_b"))
    )
    assert len(records) == 2

    with pytest.raises(
        B.DiffRefAltJoinPolicyError, match="hash multiset"
    ):
        _replay_fixture_audit(
            spec, cand_values, materialized_pairs, records[:-1]
        )


def test_replay_audit_accounts_for_one_missing_both_row_per_pair():
    study = "GSE232572"
    spec = _diff_ref_alt_spec()
    sid = "GSE232572_EMPTY__cand"
    cand_values = {sid: {}}
    records = B.derive_delta(
        study, spec, _record(), cand_values, {}, {}, sid
    )
    records[0]["pair_id"] = "pair_empty"
    audit = B.summarize_diff_ref_alt_join_audits(
        records,
        {study: spec},
        cand_values=cand_values,
        materialized_pairs=[{
            "study": study,
            "pair_id": "pair_empty",
            "candidate_sequence_id": sid,
        }],
    )[study]

    assert len(records) == 1
    assert audit["n_expected_missing_both_pairs"] == 1
    assert audit["n_consumed_missing_both_rows"] == 1
    assert audit["n_missing_both_rows"] == 1
    assert audit["expected_missing_both_multiset_sha256"] == (
        audit["consumed_missing_both_multiset_sha256"]
    )


def test_paired_join_uses_explicit_replicate_field_without_id_fallback():
    study, spec, sid, cand_values, materialized_pairs, records = (
        _paired_replay_fixture()
    )
    cand_values[sid][spec["ref_endpoint"]][0]["replicate_id"] = "ref_id"
    cand_values[sid][spec["alt_endpoint"]][0]["replicate_id"] = "alt_id"
    records = B.derive_delta(
        study, spec, _record(), cand_values, {}, {}, sid
    )
    records[0]["pair_id"] = "pair_1"
    audit = _replay_fixture_audit(
        spec, cand_values, materialized_pairs, records
    )

    assert len(records) == 1
    assert records[0]["delta"] == 9.0
    assert records[0]["replicate"] is None
    assert records[0]["replicate_pairing_status"] == (
        "EXACT_NULL_SLOT_NO_REPLICATE_CLAIM"
    )
    assert audit["n_matched_groups"] == 1


def test_paired_policy_allows_exact_null_assay_but_forbids_null_context():
    study, spec, sid, cand_values, _, _ = _paired_replay_fixture()
    for endpoint_rows in cand_values[sid].values():
        endpoint_rows[0]["assay_context"] = None
    matched = B.derive_delta(
        study, spec, _record(), cand_values, {}, {}, sid
    )
    assert matched[0]["delta"] == 9.0
    for endpoint_rows in cand_values[sid].values():
        endpoint_rows[0]["context_id"] = None

    with pytest.raises(B.DiffRefAltJoinPolicyError, match="context_id"):
        B.derive_delta(
            study, spec, _record(), cand_values, {}, {}, sid
        )


def test_pair_provenance_requires_original_trim_equal_canonical_status():
    for invalid in (" ACCEPTED", "ACCEPTED ", "", 1):
        with pytest.raises(ValueError, match="trim-equal"):
            B.build_pair_provenance({
                "pair_id": "p1",
                "relation_acceptance_status": invalid,
            })


def test_diff_ref_alt_registry_rejects_aliases_and_shared_authority_ids():
    base = _diff_ref_alt_spec()
    same_endpoint = copy.deepcopy(base)
    same_endpoint["alt_endpoint"] = same_endpoint["ref_endpoint"]
    with pytest.raises(B.DiffRefAltJoinPolicyError, match="must differ"):
        B.validate_diff_ref_alt_join_policy(same_endpoint)

    overlapping_units = copy.deepcopy(base)
    overlapping_units["join_policy"]["alt_allowed_raw_unit_tokens"] = (
        overlapping_units["join_policy"]["ref_allowed_raw_unit_tokens"]
    )
    with pytest.raises(B.DiffRefAltJoinPolicyError, match="disjoint"):
        B.validate_diff_ref_alt_join_policy(overlapping_units)

    second = _diff_ref_alt_spec(
        ref_endpoint="ep_second_ref",
        alt_endpoint="ep_second_alt",
        comparison_scale_id="SECOND_SCALE_V1",
    )
    second["join_policy"]["policy_id"] = base["join_policy"]["policy_id"]
    with pytest.raises(B.DiffRefAltJoinPolicyError, match="policy_id"):
        B.validate_diff_ref_alt_policy_registry({
            "STUDY_A": base,
            "STUDY_B": second,
        })

    second["join_policy"]["policy_id"] = "SECOND_POLICY_V2"
    second["join_policy"]["comparison_scale_id"] = (
        base["join_policy"]["comparison_scale_id"]
    )
    with pytest.raises(
        B.DiffRefAltJoinPolicyError, match="comparison_scale_id"
    ):
        B.validate_diff_ref_alt_policy_registry({
            "STUDY_A": base,
            "STUDY_B": second,
        })


def test_build_manifest_contains_per_study_join_policy_hash_and_counts(
    tmp_path, monkeypatch,
):
    study = "GSE232572"
    sid = "GSE232572_A__cand"
    spec = _diff_ref_alt_spec()
    pair = {
        "pair_id": "pair_a",
        "candidate_sequence_id": sid,
        "source_sequence_id": "GSE232572_A__src",
        "scientific_track": "E",
        "relation_acceptance_status": "ACCEPTED",
    }
    canonical = {
        "record_id": "GSE232572_A",
        "dataset": "synthetic",
        "accession": study,
        "region": "3'UTR",
        "source_sequence": "AAAA",
        "candidate_sequence": "CAAA",
        "edit_script": [{"op": "SUB", "pos": 0, "token": "C"}],
        "metadata": {},
    }
    observations = [
        _functional_observation(
            sid,
            "ep_activity_ref_mean",
            5.0,
            observation_id="ref_a",
        ),
        _functional_observation(
            sid,
            "ep_activity_alt_mean",
            8.0,
            observation_id="alt_a",
        ),
    ]
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "utr_edit_pairs.jsonl").write_text(
        json.dumps(pair) + "\n", encoding="utf-8"
    )
    (staging / "functional_observations.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in observations),
        encoding="utf-8",
    )
    canonical_path = tmp_path / "canonical.jsonl"
    canonical_path.write_text(json.dumps(canonical) + "\n", encoding="utf-8")
    monkeypatch.setattr(B.config, "STAGING_DIR", str(staging))
    monkeypatch.setattr(B.config, "CANONICAL_RECORDS", str(canonical_path))
    monkeypatch.setattr(B.config, "ACTIVE_STUDIES", [study])
    monkeypatch.setattr(B.config, "RECONSTRUCTED_PAIRS", {})
    monkeypatch.setattr(B.config, "DELTA_SPECS", {study: spec})

    manifest = B.build(tmp_path / "out")
    audit = manifest["diff_ref_alt_join_audits"][study]
    policy = B.validate_diff_ref_alt_join_policy(spec)
    effect_row = json.loads(
        (tmp_path / "out" / "effect_dataset.jsonl").read_text(
            encoding="utf-8"
        )
    )

    assert manifest["manifest_version"] == (
        "b0x_effect_dataset_manifest_v3_diff_ref_alt_replay"
    )
    assert manifest["manifest_version"] != "b0x_effect_dataset_manifest_v2"
    assert manifest["effect_dataset_schema_version"] == (
        "b0x_effect_row_v3_diff_ref_alt_independent_replay"
    )
    assert effect_row["effect_dataset_schema_version"] == (
        manifest["effect_dataset_schema_version"]
    )
    assert effect_row["qualification_status"] == (
        "MECHANICAL_REPLAY_REQUIRES_MATCHING_V3_MANIFEST_SCIENTIFIC_HOLD"
    )
    assert manifest["qualification"] == {
        "status": (
            "MECHANICAL_REPLAY_VERIFIED_WITH_MATCHING_V3_MANIFEST_"
            "SCIENTIFIC_HOLD"
        ),
        "dataset_without_manifest": "NOT_QUALIFIED",
        "replay_audit_completed_before_dataset_write": True,
        "scientific_qualification": False,
        "training_ready": False,
        "split_ready": False,
    }
    binding = manifest["implementation_binding"]
    assert set(binding) == B.IMPLEMENTATION_BINDING_KEYS
    assert binding["build_effect_dataset_sha256"] == B.sha256_file(
        Path(B.__file__).resolve()
    )
    assert binding["config_sha256"] == B.sha256_file(
        Path(B.config.__file__).resolve()
    )
    assert binding["delta_specs_sha256"] == B._json_sha256(
        B.config.DELTA_SPECS
    )
    assert binding["diff_ref_alt_policy_sha256_by_study"] == {
        study: B._json_sha256(policy)
    }
    assert binding["ordinary_only"] is True
    assert binding["scientific_unit_authority_created"] is False
    assert binding["estimand_authority_created"] is False
    assert binding["diff_ref_alt_replay_audit_schema_version"] == (
        "diff_ref_alt_join_audit_v3_independent_replay"
    )
    assert set(audit) == B.DIFF_REF_ALT_JOIN_AUDIT_KEYS
    assert audit["join_policy"] == policy
    assert audit["join_policy_sha256"] == B._json_sha256(policy)
    assert audit["verification_status"] == "INDEPENDENT_REPLAY_VERIFIED"
    assert audit["n_materialized_pairs"] == 1
    assert audit["n_ref_observation_groups"] == 1
    assert audit["n_alt_observation_groups"] == 1
    assert audit["n_consumed_ref_observation_groups"] == 1
    assert audit["n_consumed_alt_observation_groups"] == 1
    assert audit["n_matched_groups"] == 1
    assert audit["n_output_join_rows"] == 1
    assert audit["n_derived_rows"] == 1
    assert audit["n_unavailable_rows"] == 0
    assert audit["ref_raw_unit_token_group_counts"] == {
        "ep_activity_ref_mean": 1
    }
    assert audit["alt_raw_unit_token_group_counts"] == {
        "ep_activity_alt_mean": 1
    }


def test_delta_log2fc_direct():
    rec = _record()
    spec = {"type": "log2fc", "endpoint": "ep_log2fc", "benchmark": "3U-A1"}
    cand_values = {"GSE298114_X__cand": {"ep_log2fc": 1.5}}
    out = B.derive_delta("GSE298114", spec, rec, cand_values, {}, {}, "GSE298114_X__cand")
    assert out[0]["delta"] == 1.5
    assert out[0]["delta_source_status"] == "log2fc_direct"


def test_delta_diff_wt_meta_two_endpoints():
    rec = _record(metadata={"stability_wt_hek": 70.0, "stability_wt_sh": 20.0})
    spec = {"type": "diff_wt_meta",
            "endpoints": [{"cand_endpoint": "ep_stability_hek", "wt_meta": "stability_wt_hek"},
                          {"cand_endpoint": "ep_stability_sh", "wt_meta": "stability_wt_sh"}],
            "benchmark": "5U-A1"}
    cand_values = {"GSE217518_X__cand": {"ep_stability_hek": 80.0, "ep_stability_sh": 25.0}}
    out = B.derive_delta("GSE217518", spec, rec, cand_values, {}, {}, "GSE217518_X__cand")
    assert len(out) == 2
    assert {r["endpoint"] for r in out} == {"ep_stability_hek", "ep_stability_sh"}
    assert out[0]["delta"] in (10.0, 5.0)
    assert all(r["delta_source_status"] == "derived" for r in out)


def test_delta_source_anchor_unavailable():
    rec = _record(src="ACGU" * 20, cand="ACGU" * 20)
    spec = {"type": "diff_cand_src", "cand_endpoint": "ep_rl", "src_source": "raw_library", "benchmark": "5U-A1"}
    cand_values = {"GSE114002_X__cand": {"ep_rl": 6.0}}
    # source anchor NOT in the raw library map -> unavailable
    out = B.derive_delta("GSE114002", spec, rec, cand_values, {}, {}, "GSE114002_X__cand")
    assert out[0]["delta_source_status"] == "source_anchor_unavailable"
    assert out[0]["delta"] is None


def test_load_pairs_filters_active(tmp_path):
    p = tmp_path / "pairs.jsonl"
    rows = [
        {"pair_id": "p1", "candidate_sequence_id": "GSE114002_A__cand", "source_sequence_id": "GSE114002_A__src", "scientific_track": "E"},
        {"pair_id": "p2", "candidate_sequence_id": "GSE114002_B__cand", "source_sequence_id": "GSE114002_B__src", "scientific_track": "F"},
        {"pair_id": "p3", "candidate_sequence_id": "GSE149487_C__cand", "source_sequence_id": "GSE149487_C__src", "scientific_track": "E"},
        {"pair_id": "p4", "candidate_sequence_id": "GSE232571_D__cand", "source_sequence_id": "GSE232571_D__src", "scientific_track": "E"},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    out = B.load_pairs(p)
    ids = {r["pair_id"] for r in out}
    assert ids == {"p1", "p4"}  # p2 not E, p3 not active


def test_pair_provenance_maps_canonical_accepted_and_preserves_input():
    pair = {
        "pair_id": "p1",
        "relation_acceptance_status": "ACCEPTED",
        "relation_status": "ACCEPTED",
        "canonical_evidence": {"evidence_id": "e1"},
    }

    provenance = B.build_pair_provenance(pair)

    assert provenance["relation_status"] == "ACCEPTED"
    assert provenance["relation_status_source_field"] == (
        "relation_acceptance_status"
    )
    assert provenance["relation_acceptance_status"] == "ACCEPTED"
    assert provenance["legacy_relation_status_input"] == "ACCEPTED"
    assert provenance["canonical_evidence"] == pair["canonical_evidence"]
    assert provenance["evidence_sha256"] == B._json_sha256(pair)


def test_pair_provenance_rejects_canonical_legacy_conflict():
    pair = {
        "pair_id": "p1",
        "relation_acceptance_status": "ACCEPTED",
        "relation_status": "REJECTED",
    }

    with pytest.raises(B.RelationStatusConflictError, match="conflicts"):
        B.build_pair_provenance(pair)


def test_pair_provenance_missing_canonical_never_upgrades_legacy():
    pair = {
        "pair_id": "p1",
        "relation_status": "ACCEPTED",
        "canonical_evidence": {"evidence_id": "e1"},
    }

    provenance = B.build_pair_provenance(pair)

    assert provenance["relation_status"] == "UNREVIEWED"
    assert provenance["relation_status_source_field"] == (
        "MISSING_NOT_ASSERTED"
    )
    assert provenance["legacy_relation_status_input"] == "ACCEPTED"
    assert provenance["canonical_evidence"] == pair["canonical_evidence"]


def test_relation_provenance_summary_is_order_independent():
    provenances = [
        B.build_pair_provenance({
            "pair_id": "accepted",
            "relation_acceptance_status": "ACCEPTED",
        }),
        B.build_pair_provenance({
            "pair_id": "ambiguous",
            "relation_acceptance_status": "AMBIGUOUS",
        }),
        B.build_pair_provenance({
            "pair_id": "missing",
            "relation_status": "ACCEPTED",
        }),
    ]

    forward = B.summarize_pair_relation_provenance(provenances)
    reverse = B.summarize_pair_relation_provenance(
        list(reversed(provenances))
    )

    assert forward == reverse
    assert forward["relation_status_counts"] == {
        "ACCEPTED": 1,
        "AMBIGUOUS": 1,
        "UNREVIEWED": 1,
    }
    assert forward["relation_status_source_field_counts"] == {
        "MISSING_NOT_ASSERTED": 1,
        "relation_acceptance_status": 2,
    }


def test_build_manifest_relation_counts_and_effect_bytes_ignore_pair_order(
    tmp_path, monkeypatch,
):
    pairs = [
        {
            "pair_id": "pair_accepted",
            "candidate_sequence_id": "GSE298114_A__cand",
            "source_sequence_id": "GSE298114_A__src",
            "scientific_track": "E",
            "relation_acceptance_status": "ACCEPTED",
        },
        {
            "pair_id": "pair_missing",
            "candidate_sequence_id": "GSE298114_B__cand",
            "source_sequence_id": "GSE298114_B__src",
            "scientific_track": "E",
            "relation_status": "ACCEPTED",
        },
    ]
    canonical_rows = [
        {
            "record_id": "GSE298114_A",
            "dataset": "synthetic",
            "accession": "GSE298114",
            "region": "3'UTR",
            "source_sequence": "AAAA",
            "candidate_sequence": "CAAA",
            "edit_script": [{"op": "SUB", "pos": 0, "token": "C"}],
            "metadata": {},
        },
        {
            "record_id": "GSE298114_B",
            "dataset": "synthetic",
            "accession": "GSE298114",
            "region": "3'UTR",
            "source_sequence": "AAAA",
            "candidate_sequence": "GAAA",
            "edit_script": [{"op": "SUB", "pos": 0, "token": "G"}],
            "metadata": {},
        },
    ]
    observations = [
        {
            "observation_id": "obs_a",
            "sequence_id": "GSE298114_A__cand",
            "endpoint_id": "ep_log2fc",
            "context_id": "synthetic_context",
            "value": 1.0,
            "unit": "log2fc",
            "replicate": None,
        },
        {
            "observation_id": "obs_b",
            "sequence_id": "GSE298114_B__cand",
            "endpoint_id": "ep_log2fc",
            "context_id": "synthetic_context",
            "value": -1.0,
            "unit": "log2fc",
            "replicate": None,
        },
    ]

    canonical_path = tmp_path / "canonical.jsonl"
    canonical_path.write_text(
        "".join(json.dumps(row) + "\n" for row in canonical_rows),
        encoding="utf-8",
    )
    monkeypatch.setattr(B.config, "CANONICAL_RECORDS", str(canonical_path))
    monkeypatch.setattr(B.config, "ACTIVE_STUDIES", ["GSE298114"])
    monkeypatch.setattr(B.config, "RECONSTRUCTED_PAIRS", {})
    monkeypatch.setattr(B.config, "DELTA_SPECS", {
        "GSE298114": {
            "type": "log2fc",
            "endpoint": "ep_log2fc",
            "benchmark": "3U-A1",
        },
    })

    def run_build(name, pair_rows):
        staging = tmp_path / (name + "_staging")
        staging.mkdir()
        (staging / "utr_edit_pairs.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in pair_rows),
            encoding="utf-8",
        )
        (staging / "functional_observations.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in observations),
            encoding="utf-8",
        )
        monkeypatch.setattr(B.config, "STAGING_DIR", str(staging))
        out_dir = tmp_path / (name + "_out")
        manifest = B.build(out_dir)
        return manifest, (out_dir / "effect_dataset.jsonl").read_bytes()

    forward_manifest, forward_bytes = run_build("forward", pairs)
    reverse_manifest, reverse_bytes = run_build(
        "reverse", list(reversed(pairs))
    )

    assert forward_bytes == reverse_bytes
    expected = {
        "relation_status_counts": {"ACCEPTED": 1, "UNREVIEWED": 1},
        "relation_status_source_field_counts": {
            "MISSING_NOT_ASSERTED": 1,
            "relation_acceptance_status": 1,
        },
    }
    assert forward_manifest["pair_relation_provenance"] == (
        reverse_manifest["pair_relation_provenance"]
    )
    assert forward_manifest["pair_relation_provenance"] == {
        "authority_field": "relation_acceptance_status",
        "materialized_pairs": expected,
        "effect_records": expected,
    }


def test_features_shapes():
    src = "ACGUACGUACGUACGUACGU"
    cand = "ACGUACGUACGUACGUACGC"
    fx = F.extract_features(src, cand, [{"op": "SUB", "pos": 5, "token": "G"}])
    assert fx["source_feat"].shape == (20,)
    assert fx["candidate_feat"].shape == (20,)
    assert fx["diff_feat"].shape == (20,)
    assert fx["edit_feat"].shape == (12,)
    assert fx["source_onehot"].shape == (100, 4)
    # kmers vector: 2 * (4 + 16 + 64) = 168
    assert F.kmers_vector(src, cand).shape == (168,)


def test_gc_computation():
    seq = "GGGGCCCC"  # 8 nt, 8/8 GC
    f = F.sequence_features(seq)
    # GC is index 1 in the 4-dim header [len, gc, gc_first10, aug]
    assert abs(f[1] - 1.0) < 1e-5


def test_edit_features_empty_token_no_index_error():
    # An edit with an empty/none token must not raise IndexError.
    empty = F.edit_features([{"op": "SUB", "pos": 3, "token": ""}], 100)
    missing = F.edit_features([{"op": "SUB", "pos": 3}], 100)
    assert empty.shape == (12,)
    assert missing.shape == (12,)
    assert np.all(np.isfinite(empty))
    assert np.all(np.isfinite(missing))
