"""C3 contract tests for utr_editflow_goal_v3.1_benchmark_first.

These tests assert the frozen hashes, schema validity, task/split registries,
the 120-row definition matrix, the GSE246381 truth lock, and the required
$defs fixtures. They are definition-only: they never touch data files.
"""
import hashlib
import json
import os
from pathlib import Path

import pytest
import yaml
from jsonschema import Draft202012Validator

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_DIR = REPO_ROOT / "schemas" / "v3_1"
EXEC_DIR = REPO_ROOT / "docs" / "execution"
CONFIG_FILE = REPO_ROOT / "configs" / "utr_editflow_contract_v3_1.yaml"

CONTRACT_ID = "utr_editflow_goal_v3.1_benchmark_first"

# --- frozen hash constants (from 5.1 / 5.7 of the authoritative contract) ---
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
    "CROSS_REGION_PROPERTY_F_OBSERVATION",
    "CROSS_REGION_RECONSTRUCT_E_PAIR",
    "F3_OUTCOME_AUX_OBSERVATION",
    "F5_OUTCOME_AUX_OBSERVATION",
    "T3_EFFECT_DELTA_E_PAIR",
    "T3_PROPERTY_E_PAIR",
    "T3_RANK_EXPLORATORY_E_PAIR",
    "T3_RECONSTRUCT_E_PAIR",
    "T5_CONTEXT_E_PAIR",
    "T5_CONTEXT_F_OBSERVATION",
    "T5_GEN_RECONSTRUCT_E_PAIR",
    "T5_RANK_CLOSED_SELECT_E_PAIR",
]

SPLIT_IDS = [
    "3utr_sequence_cluster_disjoint",
    "3utr_source_or_variant_disjoint",
    "3utr_study_disjoint",
    "5utr_sequence_cluster_disjoint",
    "5utr_source_disjoint",
    "5utr_study_disjoint",
    "cross_region_3_to_5",
    "cross_region_5_to_3",
    "heldout_context",
    "sealed_final_v1",
]

EXPECTED_21_FILENAMES = [
    "dataset_asset.schema.json",
    "edit_path_set.schema.json",
    "eligibility_record.schema.json",
    "exposure_record.schema.json",
    "functional_observation.schema.json",
    "generation_task.schema.json",
    "group_assignment.schema.json",
    "group_registry.schema.json",
    "rejection_record.schema.json",
    "relation_role_transition.schema.json",
    "reporter_artifact_assessment.schema.json",
    "sequence_entity.schema.json",
    "split_assignment.schema.json",
    "split_registry.schema.json",
    "task_eligibility_cell.schema.json",
    "task_registry.schema.json",
    "task_split_applicability.schema.json",
    "transformation_edge.schema.json",
    "use_role.schema.json",
    "utr_edit_pair.schema.json",
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


def _load_yaml(path: Path):
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _load_json(path: Path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
# 8.1 Schema filename set
# --------------------------------------------------------------------------- #
def test_schema_filename_set():
    files = sorted(f.name for f in SCHEMA_DIR.glob("*.schema.json"))
    assert files == EXPECTED_21_FILENAMES
    assert len(files) == 21
    payload = "".join(f + "\n" for f in files).encode("utf-8")
    assert _sha256(payload) == FROZEN["schema_filename_set_sha256"]


def test_schema_expected_set_hashes():
    manifest = _load_json(SCHEMA_DIR / "SCHEMA_MANIFEST.json")
    sums = {}
    for line in (SCHEMA_DIR / "SCHEMA_SHA256SUMS").read_text(encoding="utf-8").splitlines():
        sha, _, name = line.partition("  ")
        sums[name] = sha.strip()
    assert len(manifest["schemas"]) == 21
    assert manifest["filename_set_sha256"] == FROZEN["schema_filename_set_sha256"]
    seen_ids = set()
    for entry in manifest["schemas"]:
        fname = entry["filename"]
        assert fname in sums
        assert entry["sha256"] == sums[fname]
        assert entry["schema_version"] == "3.1"
        assert entry["contract_id"] == CONTRACT_ID
        assert entry["$id"].endswith(fname)
        assert entry["$id"] not in seen_ids
        seen_ids.add(entry["$id"])
        # per-file hash matches on disk
        raw = (SCHEMA_DIR / fname).read_bytes()
        assert _sha256(raw) == entry["sha256"]


def test_schema_required_defs_present():
    for fname, defs in REQUIRED_DEFS.items():
        doc = _load_json(SCHEMA_DIR / fname)
        for d in defs:
            assert d in doc.get("$defs", {}), f"{fname} missing $defs/{d}"


def test_schema_has_no_obsolete_sealed_access_event():
    # No obsolete SealedAccessEvent $defs may remain in the active schemas.
    for f in SCHEMA_DIR.glob("*.schema.json"):
        doc = _load_json(f)
        assert "SealedAccessEvent" not in doc.get("$defs", {}), f"obsolete $defs in {f.name}"


# --------------------------------------------------------------------------- #
# 8.2 Task registry
# --------------------------------------------------------------------------- #
def test_task_id_set():
    task_reg = _load_yaml(EXEC_DIR / "task_registry_v3_1.yaml")
    ids = sorted(t["task_id"] for t in task_reg["tasks"])
    assert ids == sorted(TASK_IDS)
    assert len(ids) == 12
    payload = "".join(i + "\n" for i in sorted(ids)).encode("utf-8")
    assert _sha256(payload) == FROZEN["task_id_set_sha256"]


def test_task_descriptor_set():
    task_reg = _load_yaml(EXEC_DIR / "task_registry_v3_1.yaml")
    lines = []
    for t in task_reg["tasks"]:
        lines.append("|".join([
            t["task_id"], t["task_kind"], t["object_type"], t["scientific_track"],
            t["region_scope"], t["estimand_id"], t["activation_rule_id"],
            t["analysis_unit_id"], t["species_scope_policy_id"],
        ]))
    payload = "".join(l + "\n" for l in sorted(lines)).encode("utf-8")
    assert _sha256(payload) == FROZEN["task_descriptor_set_sha256"]


def test_activation_calibration_rule():
    task_reg = _load_yaml(EXEC_DIR / "task_registry_v3_1.yaml")
    rule_bytes = task_reg["activation_calibration_rule_bytes"].encode("utf-8")
    assert rule_bytes == CALIBRATION_RULE_BYTES.encode("utf-8")
    assert _sha256(rule_bytes) == FROZEN["activation_calibration_rule_sha256"]


# --------------------------------------------------------------------------- #
# 8.3 Split registry
# --------------------------------------------------------------------------- #
def test_split_id_set():
    split_reg = _load_yaml(EXEC_DIR / "split_registry_v3_1.yaml")
    ids = sorted(s["split_contract_id"] for s in split_reg["splits"])
    assert ids == sorted(SPLIT_IDS)
    assert len(ids) == 10
    payload = "".join(i + "\n" for i in sorted(ids)).encode("utf-8")
    assert _sha256(payload) == FROZEN["split_id_set_sha256"]


def test_split_descriptor_set():
    split_reg = _load_yaml(EXEC_DIR / "split_registry_v3_1.yaml")
    lines = []
    for s in split_reg["splits"]:
        partition_roles = ",".join(sorted(p["partition_role"] for p in s["partitions"]))
        grouping_atoms = ",".join(sorted(s["grouping_atoms"]))
        lines.append("|".join([
            s["split_contract_id"], s["activation_rule_id"], s["region_scope"],
            s["object_scope"], s["direction_or_cohort_rule_id"],
            s["direction_or_cohort_rule_sha256"], partition_roles,
            grouping_atoms, str(s["sealed_final"]).lower(),
        ]))
    payload = "".join(l + "\n" for l in sorted(lines)).encode("utf-8")
    assert _sha256(payload) == FROZEN["split_descriptor_set_sha256"]


def test_grouping_atom_projection_rule():
    split_reg = _load_yaml(EXEC_DIR / "split_registry_v3_1.yaml")
    rule_bytes = split_reg["grouping_atom_projection_rule_bytes"].encode("utf-8")
    assert rule_bytes == GROUPING_RULE_BYTES.encode("utf-8")
    assert _sha256(rule_bytes) == FROZEN["grouping_atom_rule_sha256"]


def test_sealed_cohort_rule():
    # sealed_final_v1 must carry the sealed cohort rule and its hash.
    split_reg = _load_yaml(EXEC_DIR / "split_registry_v3_1.yaml")
    sealed = next(s for s in split_reg["splits"] if s["split_contract_id"] == "sealed_final_v1")
    assert sealed["sealed_final"] is True
    assert sealed["direction_or_cohort_rule_id"] == "SEALED_COHORT_IDS_V1_SHA256_275774A99CBE46CD"
    # canonical sealed rule bytes hash
    assert _sha256(SEALED_RULE_BYTES) == sealed["direction_or_cohort_rule_sha256"]
    assert _sha256(SEALED_RULE_BYTES) == "f0ced6dc8869b040f1197b519403691bd97f07e59906c3c82434606a9861262a"
    assert FROZEN["sealed_cohort_set_sha256"] in SEALED_RULE_BYTES


# --------------------------------------------------------------------------- #
# 8.4 Task x split allowlist + 120-row matrix
# --------------------------------------------------------------------------- #
def test_task_split_allowlist():
    lines = []
    for task in sorted(TASK_ALLOWLIST):
        lines.append(f"{task}|{','.join(sorted(TASK_ALLOWLIST[task]))}")
    payload = "".join(l + "\n" for l in lines).encode("utf-8")
    assert _sha256(payload) == FROZEN["task_split_allowlist_sha256"]


def test_task_split_matrix_120_rows():
    matrix = _load_yaml(EXEC_DIR / "task_split_contract_matrix_v3_1.yaml")
    rows = matrix["rows"]
    assert len(rows) == 120
    # exact key set = 12 x 10 Cartesian product
    keys = {(r["task_id"], r["split_contract_id"]) for r in rows}
    assert len(keys) == 120
    for task in TASK_IDS:
        for split in SPLIT_IDS:
            assert (task, split) in keys, f"missing {task} x {split}"
    # contract_mapping consistent with allowlist
    for r in rows:
        expected = "ALLOWED" if r["split_contract_id"] in TASK_ALLOWLIST[r["task_id"]] else "NOT_ALLOWED"
        assert r["contract_mapping"] == expected, f"mapping mismatch {r['task_id']} {r['split_contract_id']}"


# --------------------------------------------------------------------------- #
# 8.5 Diagnostic registry
# --------------------------------------------------------------------------- #
def test_diagnostic_registry_expected_set():
    diag = _load_yaml(EXEC_DIR / "diagnostic_registry_v3_1.yaml")
    ids = [r["diagnostic_id"] for r in diag["rows"]]
    assert ids == ["OPEN_GENERATION_DIAGNOSTIC_E_GENERATION_TASK"]
    payload = "".join(i + "\n" for i in sorted(ids)).encode("utf-8")
    assert _sha256(payload) == FROZEN["diagnostic_registry_expected_set_sha256"]
    # must not be deleted or change object type
    assert diag["rows"][0]["object_type"] == "GENERATION_TASK"


# --------------------------------------------------------------------------- #
# 8.6 GSE246381 truth lock
# --------------------------------------------------------------------------- #
def test_gse246381_truth_lock():
    user = _load_yaml(EXEC_DIR / "USER_DECISION_GSE246381.yaml")
    a = user["axes"]
    assert a["project_sequence_analytic_exposure"] == "NONE_CONFIRMED"
    assert a["project_sequence_analytic_use_types"] == ["NONE_CONFIRMED"]
    assert a["project_label_analytic_exposure"] == "NONE_CONFIRMED"
    assert a["project_label_analytic_use_types"] == ["NONE_CONFIRMED"]
    assert a["pipeline_sequence_materialization"] == "PRESENT"
    assert a["pipeline_label_materialization"] == "PRESENT"
    assert a["foundation_requirement"] == "REQUIRED_FM0_A"
    assert a["foundation_audit_status"] == "DEFERRED_TO_FM0_A"
    assert a["future_role"] == "SEALED_EXTERNAL_FINAL_CANDIDATE"
    assert user["truth_lock"]["e4_interpretation_revoked"] is True
    assert user["truth_lock"]["evidence_grade_E4X_positive_assignment_count"] == 0


def test_gse246381_project_exposure_absent_in_schemas():
    # active schema/assignments must not contain legacy project_exposure field
    for f in SCHEMA_DIR.glob("*.schema.json"):
        doc = _load_json(f)
        assert "project_exposure" not in doc.get("properties", {}), f"legacy field in {f.name}"


def test_gse246381_truth_lock_in_config():
    cfg = _load_yaml(CONFIG_FILE)
    lock = cfg["gse246381_truth_lock"]
    assert lock["project_sequence_analytic_exposure"] == "NONE_CONFIRMED"
    assert lock["future_role"] == "SEALED_EXTERNAL_FINAL_CANDIDATE"
    assert lock["e4_interpretation_revoked"] is True


# --------------------------------------------------------------------------- #
# 8.7 Track U prohibition
# --------------------------------------------------------------------------- #
def test_track_u_prohibited():
    # scientific_track=U, project_unlabeled_pretraining_enabled, fallback_to_U_track must be 0/absent.
    for path in [EXEC_DIR / "task_registry_v3_1.yaml", EXEC_DIR / "split_registry_v3_1.yaml"]:
        doc = _load_yaml(path)
        text = json.dumps(doc, ensure_ascii=False)
        assert "scientific_track: U" not in text and '"scientific_track": "U"' not in text
        assert "project_unlabeled_pretraining_enabled" not in text
        assert "fallback_to_U_track" not in text


# --------------------------------------------------------------------------- #
# 8.8 Schema positive/negative fixtures (Draft-07 compatible)
# --------------------------------------------------------------------------- #
def test_schema_fixtures():
    draft = "https://json-schema.org/draft/2020-12/schema"
    for fname in EXPECTED_21_FILENAMES:
        doc = _load_json(SCHEMA_DIR / fname)
        assert doc.get("$schema") == draft
        validator = Draft202012Validator(doc)
        # positive fixture: a minimal instance satisfying the schema
        pos = _gen_value(doc, doc.get("$defs", {}))
        assert validator.is_valid(pos), f"{fname} positive fixture invalid"
        # negative fixture: an instance missing a required field
        neg = dict(pos)
        if neg:
            neg.pop(next(iter(neg)))
        assert not validator.is_valid(neg), f"{fname} negative fixture unexpectedly valid"


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


# --------------------------------------------------------------------------- #
# 8.9 Config/registry consistency
# --------------------------------------------------------------------------- #
def test_config_frozen_hashes_consistent():
    cfg = _load_yaml(CONFIG_FILE)
    for key, val in FROZEN.items():
        assert cfg["frozen_hashes"][key] == val, f"config frozen hash {key} mismatch"


def test_contract_id_consistent_across_docs():
    docs = [
        EXEC_DIR / "task_registry_v3_1.yaml",
        EXEC_DIR / "split_registry_v3_1.yaml",
        EXEC_DIR / "task_split_contract_matrix_v3_1.yaml",
        EXEC_DIR / "diagnostic_registry_v3_1.yaml",
        EXEC_DIR / "decision_log_v3_1.yaml",
        EXEC_DIR / "claim_matrix_v3_1.yaml",
    ]
    for d in docs:
        doc = _load_yaml(d)
        assert doc.get("contract_id") == CONTRACT_ID, f"{d.name} contract_id mismatch"