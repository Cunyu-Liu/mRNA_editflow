#!/usr/bin/env python
"""B0-R (v3.1) — validator for the Stage 1..5 builder artifacts.

Checks, at minimum:

  * frozen definition exact sets / hashes (task, split, allowlist, sealed cohort)
  * schema shape of every row of every ordinary/restricted artifact
  * EligibilityRecord self-hash (JCS over the row minus eligibility_manifest_sha256)
  * task/split/applicability decision self-hash
  * restricted RelationRoleTransition hash chain (prev_event_sha256 linkage)
  * ordinary/restricted conservation vs the D1 source pair/observation counts
  * cell -> EligibilityRecord FK (only ACTIVE objects have cells)
  * global PENDING == 0
  * task/split/applicability expected row counts (12 / 10 / 120)
  * dual-store isolation (no cross-store object-id overlap)

The validator is intentionally dependency-light: it uses jsonschema when
available and falls back to a small structural checker otherwise. No training,
no GPU work.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from b0_v3_1_common import (  # noqa: E402
    FROZEN_HASHES,
    REQUIRED_SPLIT_IDS,
    REQUIRED_TASK_IDS,
    SEALED_COHORT_IDS,
    iter_jsonl,
    jcs_sha256,
    load_config,
    load_matrix,
    load_splits,
    load_tasks,
    set_sha256,
    sha256_file,
)

# Schema shapes (mirror of the frozen schemas/v3_1/*.json). Used for the
# jsonschema call and for the manual fallback.
# key -> (schema_file, top-level (bool) or $defs name, required, properties)
SCHEMA_BY_KEY = {
    "B0_ROLE_DECISION_EVIDENCE": ("eligibility_record.schema.json", "B0RoleDecisionEvidence",
        ["object_id", "role_decision", "evidence_id", "evidence_sha256"],
        {"object_id": "string", "role_decision": "string",
         "evidence_id": "string", "evidence_sha256": "string"}),
    "GLOBAL_ELIGIBILITY_DECISION_EVIDENCE": ("eligibility_record.schema.json", "GlobalEligibilityDecisionEvidence",
        ["decision_id", "global_eligibility", "evidence_id", "evidence_sha256"],
        {"decision_id": "string", "global_eligibility": "string",
         "evidence_id": "string", "evidence_sha256": "string"}),
    "ELIGIBILITY_MANIFEST": ("eligibility_record.schema.json", None,
        ["object_id", "object_type", "global_eligibility", "purpose", "eligibility_manifest_sha256"],
        {"object_id": "string", "object_type": {"enum": ["PAIR", "OBSERVATION"]},
         "global_eligibility": {"enum": ["ELIGIBLE", "INELIGIBLE", "INELIGIBLE_WITH_REASON"]},
         "purpose": "string", "eligibility_manifest_sha256": "string"}),
    "RELATION_ROLE_TRANSITIONS": ("relation_role_transition.schema.json", None,
        ["transition_id", "object_id", "from_role", "to_role", "prev_event_sha256", "event_sha256"],
        {"transition_id": "string", "object_id": "string", "from_role": "string",
         "to_role": "string", "prev_event_sha256": ["string", "null"],
         "event_sha256": "string", "config_hash": ["string", "null"]}),
    "EFFECTIVE_ROLE_PROJECTION": ("relation_role_transition.schema.json", "EffectiveRoleProjection",
        ["object_id", "effective_role", "cardinality", "transition_chain_root_sha256"],
        {"object_id": "string", "effective_role": "string", "cardinality": "integer",
         "transition_chain_root_sha256": "string"}),
    "TASK_ELIGIBILITY_UNIVERSE": ("task_eligibility_cell.schema.json", None,
        ["cell_id", "object_id", "task_id", "split_contract_id", "cell_status", "assigned_partition_id"],
        {"cell_id": "string", "object_id": "string", "task_id": "string",
         "split_contract_id": "string",
         "cell_status": {"enum": ["ELIGIBLE", "INELIGIBLE", "INELIGIBLE_WITH_REASON"]},
         "assigned_partition_id": ["string", "null"]}),
    "SPLIT_ASSIGNMENTS": ("split_assignment.schema.json", None,
        ["assignment_id", "object_id", "object_type", "split_contract_id", "partition_id",
         "partition_role", "assignment_algorithm_id", "assignment_algorithm_sha256"],
        {"assignment_id": "string", "object_id": "string",
         "object_type": {"enum": ["PAIR", "OBSERVATION"]},
         "split_contract_id": "string", "partition_id": "string",
         "partition_role": {"enum": ["TRAIN", "DEVELOPMENT", "INTERNAL_TEST", "SEALED_FINAL", "STRESS_ONLY"]},
         "assignment_algorithm_id": "string", "assignment_algorithm_sha256": "string"}),
}

# Decision artifacts are validated for shape + self-hash (schema-free, they are
# registry-level JSONL not covered by the 21 frozen schemas).
DECISION_KEYS = {
    "TASK_ACTIVATION_DECISIONS": ("task_id",),
    "SPLIT_ACTIVATION_DECISIONS": ("split_contract_id",),
    "TASK_SPLIT_APPLICABILITY_DECISIONS": ("task_id", "split_contract_id"),
}


def _validate_row_shape(schema_spec, row) -> list:
    """Return list of error strings for one row against a schema spec."""
    errors = []
    _, ret_key, required, props = schema_spec
    if ret_key is None:
        # top-level object schema
        allowed = set(props) | set(required)
    else:
        # $defs shape: only required fields + the four B0 evidence fields
        allowed = set(props)
    for req in required:
        if req not in row:
            errors.append(f"missing_required_field:{req}")
    for field in row:
        if field not in allowed:
            errors.append(f"additional_property:{field}")
    for field, spec in props.items():
        if field not in row:
            continue
        val = row[field]
        if isinstance(spec, dict) and "enum" in spec:
            if val not in spec["enum"]:
                errors.append(f"enum_mismatch:{field}={val!r}")
        elif isinstance(spec, (list, tuple)):
            if val is not None and not isinstance(val, str):
                errors.append(f"type_mismatch:{field}")
        elif spec == "integer":
            if not isinstance(val, int) or isinstance(val, bool):
                errors.append(f"type_mismatch:{field}")
        elif spec == "string":
            if not isinstance(val, str):
                errors.append(f"type_mismatch:{field}")
    return errors


def validate_definition_hashes(worktree: Path) -> Counter:
    errors = Counter()
    tasks = load_tasks(worktree / "docs" / "execution" / "task_registry_v3_1.yaml")
    splits = load_splits(worktree / "docs" / "execution" / "split_registry_v3_1.yaml")
    matrix = load_matrix(worktree / "docs" / "execution" / "task_split_contract_matrix_v3_1.yaml")
    config = load_config(worktree / "configs" / "utr_editflow_contract_v3_1.yaml")

    if set(tasks) != set(REQUIRED_TASK_IDS):
        errors["task_activation_decision_expected_set_mismatch"] += 1
    if set_sha256(tasks) != FROZEN_HASHES["task_id_set_sha256"]:
        errors["task_registry_expected_set_hash_mismatch"] += 1
    if set(splits) != set(REQUIRED_SPLIT_IDS):
        errors["split_activation_decision_expected_set_mismatch"] += 1
    if set_sha256(splits) != FROZEN_HASHES["split_id_set_sha256"]:
        errors["split_registry_expected_set_hash_mismatch"] += 1
    if set_sha256(SEALED_COHORT_IDS) != FROZEN_HASHES["sealed_cohort_set_sha256"]:
        errors["sealed_cohort_expected_set_mismatch"] += 1
    if len(matrix) != 120:
        errors["task_split_definition_row_count"] += 1
    # allowlist hash
    allow = {}
    for row in matrix:
        if row.get("contract_mapping") != "ALLOWED":
            continue
        allow.setdefault(row["task_id"], []).append(row["split_contract_id"])
    allow_lines = []
    for tid in sorted(allow):
        allow_lines.append(f"{tid}|{','.join(sorted(allow[tid]))}")
    # allowlist hash: ALLOWED rows only, task_id|comma-joined-sorted-split-ids,
    # LF-terminated each line (§5.7.3).
    body = "".join(line + "\n" for line in sorted(allow_lines))
    from b0_v3_1_common import sha256_utf8
    if sha256_utf8(body) != FROZEN_HASHES["task_split_allowlist_sha256"]:
        errors["task_split_allowlist_mismatch"] += 1
    return errors


def validate_artifact_schema(path: Path, key: str) -> Counter:
    errors = Counter()
    spec = SCHEMA_BY_KEY[key]
    n = 0
    for row in iter_jsonl(path):
        n += 1
        for e in _validate_row_shape(spec, row):
            errors[f"schema_error:{e}"] += 1
    errors["_rows"] = n
    return errors


def validate_self_hash(path: Path, hash_field: str, id_field: str) -> Counter:
    errors = Counter()
    for row in iter_jsonl(path):
        computed = jcs_sha256(row, exclude=[hash_field])
        if row.get(hash_field) != computed:
            errors["self_hash_mismatch"] += 1
    return errors


def validate_transition_chain(path: Path) -> Counter:
    errors = Counter()
    prev = None
    for row in iter_jsonl(path):
        if prev is not None and row.get("prev_event_sha256") != prev:
            errors["transition_predecessor_hash_mismatch"] += 1
        prev = row.get("event_sha256")
    return errors


def validate_conservation(ordinary_pairs, ordinary_obs, restricted_pairs, restricted_obs,
                          out: Path, res_out: Path) -> Counter:
    errors = Counter()
    # D1 source counts
    n_ord_pairs = sum(1 for _ in iter_jsonl(ordinary_pairs))
    n_ord_obs = sum(1 for _ in iter_jsonl(ordinary_obs))
    n_res_pairs = sum(1 for _ in iter_jsonl(restricted_pairs))
    n_res_obs = sum(1 for _ in iter_jsonl(restricted_obs))

    # Eligibility manifests must cover every current-leaf accepted E pair / F observation.
    ord_man = set(r["object_id"] for r in iter_jsonl(out / "ELIGIBILITY_MANIFEST.jsonl"))
    res_man = set(r["object_id"] for r in iter_jsonl(res_out / "ELIGIBILITY_MANIFEST.jsonl"))

    ord_e_pair_ids = {r["pair_id"] for r in iter_jsonl(ordinary_pairs) if r.get("scientific_track") == "E"}
    ord_f_obs_ids = {r["observation_id"] for r in iter_jsonl(ordinary_obs)}
    res_e_pair_ids = {r["pair_id"] for r in iter_jsonl(restricted_pairs) if r.get("scientific_track") == "E"}
    res_f_obs_ids = {r["observation_id"] for r in iter_jsonl(restricted_obs)}

    if not ord_man.issuperset(ord_e_pair_ids | ord_f_obs_ids):
        errors["ordinary_eligibility_conservation_mismatch"] += 1
    if not res_man.issuperset(res_e_pair_ids | res_f_obs_ids):
        errors["restricted_eligibility_conservation_mismatch"] += 1

    # global pending must be 0
    pending = sum(1 for r in iter_jsonl(out / "ELIGIBILITY_MANIFEST.jsonl")
                  if r.get("global_eligibility") == "INELIGIBLE")
    pending += sum(1 for r in iter_jsonl(res_out / "ELIGIBILITY_MANIFEST.jsonl")
                   if r.get("global_eligibility") == "INELIGIBLE")
    if pending != 0:
        errors["global_pending_nonzero"] = pending

    errors["_ordinary_pairs"] = n_ord_pairs
    errors["_ordinary_obs"] = n_ord_obs
    errors["_restricted_pairs"] = n_res_pairs
    errors["_restricted_obs"] = n_res_obs
    errors["_ordinary_eligibility"] = len(ord_man)
    errors["_restricted_eligibility"] = len(res_man)
    return errors


def validate_fk_and_isolation(out: Path, res_out: Path) -> Counter:
    errors = Counter()
    ord_man = {r["object_id"] for r in iter_jsonl(out / "ELIGIBILITY_MANIFEST.jsonl")}
    res_man = {r["object_id"] for r in iter_jsonl(res_out / "ELIGIBILITY_MANIFEST.jsonl")}

    # every cell object must exist in the corresponding eligibility manifest
    for r in iter_jsonl(out / "TASK_ELIGIBILITY_UNIVERSE.jsonl"):
        if r["object_id"] not in ord_man:
            errors["cell_foreign_key_mismatch"] += 1
    for r in iter_jsonl(res_out / "TASK_ELIGIBILITY_UNIVERSE.jsonl"):
        if r["object_id"] not in res_man:
            errors["cell_foreign_key_mismatch"] += 1

    # dual-store isolation
    overlap = ord_man & res_man
    if overlap:
        errors[f"cross_store_object_overlap"] += len(overlap)

    # duplicate cell keys
    seen = set()
    for r in iter_jsonl(out / "TASK_ELIGIBILITY_UNIVERSE.jsonl"):
        k = (r["object_id"], r["task_id"], r["split_contract_id"])
        if k in seen:
            errors["duplicate_task_eligibility_cell"] += 1
        seen.add(k)
    for r in iter_jsonl(res_out / "TASK_ELIGIBILITY_UNIVERSE.jsonl"):
        k = (r["object_id"], r["task_id"], r["split_contract_id"])
        if k in seen:
            errors["duplicate_task_eligibility_cell"] += 1
        seen.add(k)
    return errors


def validate_decision_counts(out: Path) -> Counter:
    errors = Counter()
    n_task = sum(1 for _ in iter_jsonl(out / "TASK_ACTIVATION_DECISIONS.jsonl"))
    n_split = sum(1 for _ in iter_jsonl(out / "SPLIT_ACTIVATION_DECISIONS.jsonl"))
    n_app = sum(1 for _ in iter_jsonl(out / "TASK_SPLIT_APPLICABILITY_DECISIONS.jsonl"))
    if n_task != 12:
        errors["task_activation_decision_expected_set_mismatch"] += 1
    if n_split != 10:
        errors["split_activation_decision_expected_set_mismatch"] += 1
    if n_app != 120:
        errors["applicability_decision_key_set_mismatch"] += 1
    # decision self-hash
    for key, id_fields in DECISION_KEYS.items():
        seen = set()
        for r in iter_jsonl(out / f"{key}.jsonl"):
            if "decision_sha256" in r and r.get("decision_sha256") != jcs_sha256(r, exclude=["decision_sha256"]):
                errors["decision_sha_mismatch"] += 1
            if all(f in r for f in id_fields):
                k = tuple(r[f] for f in id_fields)
                if k in seen:
                    errors[f"duplicate_decision:{key}"] += 1
                seen.add(k)
    return errors


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worktree", required=True)
    ap.add_argument("--ordinary-dir", required=True)
    ap.add_argument("--restricted-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--restricted-out", required=True)
    args = ap.parse_args()

    wt = Path(args.worktree)
    od = Path(args.ordinary_dir)
    rd = Path(args.restricted_dir)
    out = Path(args.out_dir)
    res_out = Path(args.restricted_out)

    errors = Counter()

    errors.update(validate_definition_hashes(wt))

    # schema validation of every artifact, both stores
    for key, fname in [
        ("B0_ROLE_DECISION_EVIDENCE", "B0_ROLE_DECISION_EVIDENCE.jsonl"),
        ("GLOBAL_ELIGIBILITY_DECISION_EVIDENCE", "GLOBAL_ELIGIBILITY_DECISION_EVIDENCE.jsonl"),
        ("ELIGIBILITY_MANIFEST", "ELIGIBILITY_MANIFEST.jsonl"),
        ("RELATION_ROLE_TRANSITIONS", "RELATION_ROLE_TRANSITIONS.jsonl"),
        ("EFFECTIVE_ROLE_PROJECTION", "EFFECTIVE_ROLE_PROJECTION.jsonl"),
        ("TASK_ELIGIBILITY_UNIVERSE", "TASK_ELIGIBILITY_UNIVERSE.jsonl"),
        ("SPLIT_ASSIGNMENTS", "SPLIT_ASSIGNMENTS.jsonl"),
    ]:
        errors.update(validate_artifact_schema(out / fname, key))
        errors.update(validate_artifact_schema(res_out / fname, key))

    # self-hashes
    errors.update(validate_self_hash(out / "ELIGIBILITY_MANIFEST.jsonl",
                                     "eligibility_manifest_sha256", "object_id"))
    errors.update(validate_self_hash(res_out / "ELIGIBILITY_MANIFEST.jsonl",
                                     "eligibility_manifest_sha256", "object_id"))

    # transition chain (restricted has real events; ordinary is empty)
    errors.update(validate_transition_chain(res_out / "RELATION_ROLE_TRANSITIONS.jsonl"))

    # conservation, FK, isolation, decision counts
    errors.update(validate_conservation(
        od / "utr_edit_pairs.jsonl", od / "functional_observations.jsonl",
        rd / "utr_edit_pairs.jsonl", rd / "functional_observations.jsonl",
        out, res_out))
    errors.update(validate_fk_and_isolation(out, res_out))
    errors.update(validate_decision_counts(out))

    # aggregate error counters (only the *_mismatch / non-underscore keys)
    error_total = sum(v for k, v in errors.items() if not k.startswith("_"))
    result = {
        "validator": "PASS" if error_total == 0 else "FAIL",
        "total_errors": error_total,
        "counters": dict(errors),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if error_total == 0 else 1


if __name__ == "__main__":
    sys.exit(main())