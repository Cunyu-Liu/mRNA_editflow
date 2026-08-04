#!/usr/bin/env python3
"""CLI validator for the v3.1 benchmark-first contract (C3).

Reimplements the C3-03 checks and exits non-zero on any failure. It is
definition-only: it never touches data files.
"""
import hashlib
import json
import sys
from pathlib import Path

import yaml
from jsonschema import Draft202012Validator

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_DIR = REPO_ROOT / "schemas" / "v3_1"
EXEC_DIR = REPO_ROOT / "docs" / "execution"
CONFIG_FILE = REPO_ROOT / "configs" / "utr_editflow_contract_v3_1.yaml"

CONTRACT_ID = "utr_editflow_goal_v3.1_benchmark_first"

FROZEN = {
    "schema_filename_set_sha256": "d2e5ddaef3665214007422638df3cc6b0357747aad3911efd4f29319647b1762",
    "task_id_set_sha256": "b0b43cb76f39b32009e3a6ef8ae6d05395d61bf7baa7480743587e6772447207",
    "task_descriptor_set_sha256": "8f42ef044d8de1a26b9b587587c2de99c6068f67f37e269e226e143333245ba3",
    "split_id_set_sha256": "b8c6fb2718875862da500c949481d04db08d1d21f94e3d13da49e3ace64ff487",
    "split_descriptor_set_sha256": "c8a6c82a9a1ab687ef2c3cb912ed96aae26c73a0662b0ae0911040c37e8ef1fa",
    "task_split_allowlist_sha256": "02b25e4717e4a7192b658d5e69cdbb198e5b696b3ea520b7a0a887fcf89097ab",
    "grouping_atom_rule_sha256": "bd8395ab0ec23d98d7c1b717e7fcb0bdd3df6d18002985624cd9eb41f8bd7983",
    "activation_calibration_rule_sha256": "b2652abda7a2dbb7001e7fb655db9b6ac19f2b8f80fbc65362dc1236fd9781e9",
    "diagnostic_registry_expected_set_sha256": "f25c0adc643f38ff26c5e08bf07e4175a4e2571eaae939d61daa91fc6f2aabb2",
    "sealed_cohort_set_sha256": "275774a99cbe46ccd3084747f7a6efa4ac9af04ed841b2932c318f3682f07df0",
}

TASK_IDS = [
    "CROSS_REGION_PROPERTY_F_OBSERVATION", "CROSS_REGION_RECONSTRUCT_E_PAIR",
    "F3_OUTCOME_AUX_OBSERVATION", "F5_OUTCOME_AUX_OBSERVATION",
    "T3_EFFECT_DELTA_E_PAIR", "T3_PROPERTY_E_PAIR", "T3_RANK_EXPLORATORY_E_PAIR",
    "T3_RECONSTRUCT_E_PAIR", "T5_CONTEXT_E_PAIR", "T5_CONTEXT_F_OBSERVATION",
    "T5_GEN_RECONSTRUCT_E_PAIR", "T5_RANK_CLOSED_SELECT_E_PAIR",
]

SPLIT_IDS = [
    "3utr_sequence_cluster_disjoint", "3utr_source_or_variant_disjoint",
    "3utr_study_disjoint", "5utr_sequence_cluster_disjoint", "5utr_source_disjoint",
    "5utr_study_disjoint", "cross_region_3_to_5", "cross_region_5_to_3",
    "heldout_context", "sealed_final_v1",
]

EXPECTED_21_FILENAMES = [
    "dataset_asset.schema.json", "edit_path_set.schema.json",
    "eligibility_record.schema.json", "exposure_record.schema.json",
    "functional_observation.schema.json", "generation_task.schema.json",
    "group_assignment.schema.json", "group_registry.schema.json",
    "rejection_record.schema.json", "relation_role_transition.schema.json",
    "reporter_artifact_assessment.schema.json", "sequence_entity.schema.json",
    "split_assignment.schema.json", "split_registry.schema.json",
    "task_eligibility_cell.schema.json", "task_registry.schema.json",
    "task_split_applicability.schema.json", "transformation_edge.schema.json",
    "use_role.schema.json", "utr_edit_pair.schema.json",
    "utr_edit_relation_candidate.schema.json",
]

TASK_ALLOWLIST = {
    "CROSS_REGION_PROPERTY_F_OBSERVATION": ["cross_region_3_to_5", "cross_region_5_to_3"],
    "CROSS_REGION_RECONSTRUCT_E_PAIR": ["cross_region_3_to_5", "cross_region_5_to_3"],
    "F3_OUTCOME_AUX_OBSERVATION": ["3utr_sequence_cluster_disjoint", "3utr_study_disjoint"],
    "F5_OUTCOME_AUX_OBSERVATION": ["5utr_sequence_cluster_disjoint", "5utr_study_disjoint"],
    "T3_EFFECT_DELTA_E_PAIR": ["3utr_source_or_variant_disjoint", "3utr_study_disjoint"],
    "T3_PROPERTY_E_PAIR": ["3utr_source_or_variant_disjoint", "3utr_study_disjoint"],
    "T3_RANK_EXPLORATORY_E_PAIR": ["3utr_source_or_variant_disjoint", "3utr_study_disjoint"],
    "T3_RECONSTRUCT_E_PAIR": ["3utr_source_or_variant_disjoint", "3utr_study_disjoint"],
    "T5_CONTEXT_E_PAIR": ["heldout_context"],
    "T5_CONTEXT_F_OBSERVATION": ["heldout_context"],
    "T5_GEN_RECONSTRUCT_E_PAIR": ["5utr_source_disjoint", "5utr_study_disjoint", "sealed_final_v1"],
    "T5_RANK_CLOSED_SELECT_E_PAIR": ["5utr_source_disjoint", "5utr_study_disjoint", "sealed_final_v1"],
}

GROUPING_RULE_BYTES = (
    "PAIR_OR_OBSERVATION_GROUPING_ATOMS_V1\n"
    "PAIR:REQUIRE_ALL_LISTED\n"
    "OBSERVATION:PAIR=NOT_APPLICABLE;SOURCE=NOT_APPLICABLE;BIOLOGICAL_PARENT=NOT_APPLICABLE;REQUIRE_OTHER_LISTED\n"
    "MISSING_REQUIRED_ATOM=TASK_CELL_INELIGIBLE_NO_ASSIGNMENT\n"
    "INVENTED_ATOM_FORBIDDEN=true\n"
)
CALIBRATION_RULE_BYTES = (
    "ACTIVATION_CALIBRATION_MASK_V1\n"
    "ELIGIBLE_SCOPE=ORDINARY_NONSEALED_CURRENT_LEAF_TECHNICAL_ACCEPTED\n"
    "COMPONENT_ATOMS=BIOLOGICAL_PARENT,GENE,LIBRARY_LINEAGE,SEQUENCE_CLUSTER,STUDY,TILE_FAMILY\n"
    "COMPONENT_ID=SHA256_SORTED_MEMBER_IDS\n"
    "SELECT=UINT64_BE(SHA256(UTR_EDITFLOW_V3_1_CALIBRATION|COMPONENT_ID)[0:8])%5==0\n"
    "CALIBRATION_PARTITION=DEVELOPMENT_ONLY\n"
    "OUTCOME_BLIND=true\n"
)
SEALED_RULE_BYTES = (
    "SEALED_FINAL_COHORT_SET_SHA256=275774a99cbe46ccd3084747f7a6efa4ac9af04ed841b2932c318f3682f07df0;"
    "OTHER_PARTITIONS_FORBIDDEN=true\n"
)

REQUIRED_DEFS = {
    "group_registry.schema.json": ["NoEditSamplingFrameRow"],
    "functional_observation.schema.json": ["FunctionalObservationCandidate", "EndpointRegistryRow"],
    "exposure_record.schema.json": ["AccessIntent", "AccessCompletion", "AccessAbort", "FoundationExposureAuditRow", "EffectiveExposureProjection"],
    "transformation_edge.schema.json": ["SupersessionEdge", "CurrentCanonicalObjectProjection"],
    "relation_role_transition.schema.json": ["EffectiveRoleProjection", "B0PreparedManifest", "B0TransactionCommit"],
    "task_registry.schema.json": ["ActivationCalibrationMaskRow", "TaskActivationDecision"],
    "split_registry.schema.json": ["SplitActivationDecision"],
    "task_split_applicability.schema.json": ["TaskSplitDefinitionRow", "TaskSplitApplicabilityDecision"],
    "eligibility_record.schema.json": ["B0RoleDecisionEvidence", "GlobalEligibilityDecisionEvidence"],
    "generation_task.schema.json": ["DiagnosticRegistryRow"],
}


def _sha256(b) -> str:
    if isinstance(b, str):
        b = b.encode("utf-8")
    return hashlib.sha256(b).hexdigest()


def _load_yaml(path):
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _load_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _gen_value(schema, defs):
    """Recursively build an instance that satisfies a schema (for positive fixtures)."""
    if not isinstance(schema, dict):
        return "x"
    if "$ref" in schema:
        ref = schema["$ref"]
        if ref.startswith("#/$defs/"):
            dname = ref.split("/")[-1]
            if dname in defs:
                return _gen_value(defs[dname], defs)
        return "x"
    if "const" in schema:
        return schema["const"]
    if "enum" in schema:
        return schema["enum"][0]
    t = schema.get("type")
    if isinstance(t, list):
        t = t[0] if t else "string"
    if t == "object":
        props = schema.get("properties", {})
        out = {}
        for k in schema.get("required", []):
            out[k] = _gen_value(props.get(k, {"type": "string"}), defs)
        return out
    if t == "array":
        items = schema.get("items", {})
        n = schema.get("minItems", 1)
        return [_gen_value(items, defs) for _ in range(max(n, 1))]
    if t == "integer":
        return 1
    if t == "number":
        return 1.0
    if t == "boolean":
        return True
    if t == "string":
        return "x"
    return "x"


def _check(name, cond, failures):
    if not cond:
        failures.append(name)
        print(f"  [FAIL] {name}")
    else:
        print(f"  [ok]   {name}")


def main() -> int:
    failures = []
    print("Validating v3.1 contract (C3)")

    # 1. schema filename set
    files = sorted(f.name for f in SCHEMA_DIR.glob("*.schema.json"))
    _check("schema filename set == expected 21", files == EXPECTED_21_FILENAMES and len(files) == 21, failures)
    _check("schema filename-set hash", _sha256("".join(f + "\n" for f in files).encode()) == FROZEN["schema_filename_set_sha256"], failures)

    # 2. per-file schema id/version/contract/hash
    manifest = _load_json(SCHEMA_DIR / "SCHEMA_MANIFEST.json")
    sums = {}
    for line in (SCHEMA_DIR / "SCHEMA_SHA256SUMS").read_text(encoding="utf-8").splitlines():
        sha, _, name = line.partition("  ")
        sums[name] = sha.strip()
    seen_ids = set()
    _check("manifest schema_count == 21", len(manifest["schemas"]) == 21, failures)
    ok = True
    for e in manifest["schemas"]:
        fname = e["filename"]
        ok = ok and (e["sha256"] == sums.get(fname)) and (e["schema_version"] == "3.1") \
            and (e["contract_id"] == CONTRACT_ID) and (e["$id"].endswith(fname)) \
            and (e["$id"] not in seen_ids) and (_sha256((SCHEMA_DIR / fname).read_bytes()) == e["sha256"])
        seen_ids.add(e["$id"])
    _check("per-file schema id/version/contract/hash unique", ok, failures)

    # 3. required $defs
    ok = True
    for fname, defs in REQUIRED_DEFS.items():
        doc = _load_json(SCHEMA_DIR / fname)
        for d in defs:
            if d not in doc.get("$defs", {}):
                ok = False
    _check("required $defs present", ok, failures)

    # 4. task registry
    task_reg = _load_yaml(EXEC_DIR / "task_registry_v3_1.yaml")
    ids = sorted(t["task_id"] for t in task_reg["tasks"])
    _check("task ID set", ids == sorted(TASK_IDS) and len(ids) == 12, failures)
    _check("task ID-set hash",
           _sha256("".join(i + "\n" for i in sorted(ids)).encode()) == FROZEN["task_id_set_sha256"], failures)
    lines = []
    for t in task_reg["tasks"]:
        lines.append("|".join([t["task_id"], t["task_kind"], t["object_type"], t["scientific_track"],
                               t["region_scope"], t["estimand_id"], t["activation_rule_id"],
                               t["analysis_unit_id"], t["species_scope_policy_id"]]))
    _check("task descriptor-set hash",
           _sha256("".join(l + "\n" for l in sorted(lines)).encode()) == FROZEN["task_descriptor_set_sha256"], failures)
    _check("activation-calibration rule hash",
           _sha256(task_reg["activation_calibration_rule_bytes"].encode()) == FROZEN["activation_calibration_rule_sha256"], failures)

    # 5. split registry
    split_reg = _load_yaml(EXEC_DIR / "split_registry_v3_1.yaml")
    sids = sorted(s["split_contract_id"] for s in split_reg["splits"])
    _check("split ID set", sids == sorted(SPLIT_IDS) and len(sids) == 10, failures)
    _check("split ID-set hash",
           _sha256("".join(i + "\n" for i in sorted(sids)).encode()) == FROZEN["split_id_set_sha256"], failures)
    lines = []
    for s in split_reg["splits"]:
        pr = ",".join(sorted(p["partition_role"] for p in s["partitions"]))
        ga = ",".join(sorted(s["grouping_atoms"]))
        lines.append("|".join([s["split_contract_id"], s["activation_rule_id"], s["region_scope"],
                               s["object_scope"], s["direction_or_cohort_rule_id"],
                               s["direction_or_cohort_rule_sha256"], pr, ga, str(s["sealed_final"]).lower()]))
    _check("split descriptor-set hash",
           _sha256("".join(l + "\n" for l in sorted(lines)).encode()) == FROZEN["split_descriptor_set_sha256"], failures)
    _check("grouping-atom rule hash",
           _sha256(split_reg["grouping_atom_projection_rule_bytes"].encode()) == FROZEN["grouping_atom_rule_sha256"], failures)
    _check("sealed cohort rule hash",
           _sha256(SEALED_RULE_BYTES) == "f0ced6dc8869b040f1197b519403691bd97f07e59906c3c82434606a9861262a", failures)

    # 6. allowlist + 120-row matrix
    lines = []
    for task in sorted(TASK_ALLOWLIST):
        lines.append(f"{task}|{','.join(sorted(TASK_ALLOWLIST[task]))}")
    _check("allowlist hash",
           _sha256("".join(l + "\n" for l in lines).encode()) == FROZEN["task_split_allowlist_sha256"], failures)
    matrix = _load_yaml(EXEC_DIR / "task_split_contract_matrix_v3_1.yaml")
    rows = matrix["rows"]
    keys = {(r["task_id"], r["split_contract_id"]) for r in rows}
    _check("matrix has exactly 120 rows", len(rows) == 120 and len(keys) == 120, failures)
    ok = True
    for task in TASK_IDS:
        for split in SPLIT_IDS:
            if (task, split) not in keys:
                ok = False
    _check("matrix key set == 12x10 Cartesian product", ok, failures)
    ok = True
    for r in rows:
        exp = "ALLOWED" if r["split_contract_id"] in TASK_ALLOWLIST[r["task_id"]] else "NOT_ALLOWED"
        if r["contract_mapping"] != exp:
            ok = False
    _check("matrix mapping consistent with allowlist", ok, failures)

    # 7. diagnostic registry
    diag = _load_yaml(EXEC_DIR / "diagnostic_registry_v3_1.yaml")
    dids = [r["diagnostic_id"] for r in diag["rows"]]
    _check("diagnostic registry expected set",
           dids == ["OPEN_GENERATION_DIAGNOSTIC_E_GENERATION_TASK"] and
           _sha256("".join(i + "\n" for i in sorted(dids)).encode()) == FROZEN["diagnostic_registry_expected_set_sha256"], failures)
    _check("diagnostic object type GENERATION_TASK", diag["rows"][0]["object_type"] == "GENERATION_TASK", failures)

    # 8. GSE246381 truth lock
    user = _load_yaml(EXEC_DIR / "USER_DECISION_GSE246381.yaml")
    a = user["axes"]
    _check("GSE246381 four-axis truth lock",
           a["project_sequence_analytic_exposure"] == "NONE_CONFIRMED" and
           a["project_sequence_analytic_use_types"] == ["NONE_CONFIRMED"] and
           a["project_label_analytic_exposure"] == "NONE_CONFIRMED" and
           a["project_label_analytic_use_types"] == ["NONE_CONFIRMED"] and
           a["pipeline_sequence_materialization"] == "PRESENT" and
           a["pipeline_label_materialization"] == "PRESENT" and
           a["foundation_requirement"] == "REQUIRED_FM0_A" and
           a["foundation_audit_status"] == "DEFERRED_TO_FM0_A" and
           a["future_role"] == "SEALED_EXTERNAL_FINAL_CANDIDATE", failures)
    _check("GSE246381 E4 revoked",
           user["truth_lock"]["e4_interpretation_revoked"] is True and
           user["truth_lock"]["evidence_grade_E4X_positive_assignment_count"] == 0, failures)

    # 9. Track U prohibition
    text = json.dumps(_load_yaml(EXEC_DIR / "task_registry_v3_1.yaml"), ensure_ascii=False)
    _check("Track U prohibited", "scientific_track: U" not in text and '"scientific_track": "U"' not in text
           and "project_unlabeled_pretraining_enabled" not in text and "fallback_to_U_track" not in text, failures)

    # 10. schema fixtures
    ok = True
    for fname in EXPECTED_21_FILENAMES:
        doc = _load_json(SCHEMA_DIR / fname)
        validator = Draft202012Validator(doc)
        pos = _gen_value(doc, doc.get("$defs", {}))
        if not validator.is_valid(pos):
            print(f"    [fixture] positive failed for {fname}")
            ok = False
        neg = dict(pos)
        if neg:
            neg.pop(next(iter(neg)))
        if validator.is_valid(neg):
            print(f"    [fixture] negative unexpectedly valid for {fname}")
            ok = False
    _check("schema positive/negative fixtures", ok, failures)

    # 11. config frozen hashes consistent
    cfg = _load_yaml(CONFIG_FILE)
    ok = all(cfg["frozen_hashes"].get(k) == v for k, v in FROZEN.items())
    _check("config frozen hashes consistent", ok, failures)

    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s) failed")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())