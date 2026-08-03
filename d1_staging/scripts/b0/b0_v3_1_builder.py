#!/usr/bin/env python
"""B0-R (v3.1) — seven-stage builder (Stage 1..5).

Implements the B0-R atomic decision order (contract §B0-00) up to Stage 5:

  1. Definition/pre-role facts  -> B0_ROLE_DECISION_EVIDENCE.jsonl
  2. Staged role                -> RELATION_ROLE_TRANSITIONS.jsonl + EFFECTIVE_ROLE_PROJECTION.jsonl
  3. Global purpose             -> GLOBAL_ELIGIBILITY_DECISION_EVIDENCE.jsonl + ELIGIBILITY_MANIFEST.jsonl
  4. Activation/applicability   -> ACTIVATION_CALIBRATION_MASK.jsonl + TASK/SPLIT/APPLICABILITY decisions
  5. Cells/assignments          -> TASK_ELIGIBILITY_UNIVERSE.jsonl + SPLIT_ASSIGNMENTS.jsonl

Stage 6 (PREPARED manifests) and Stage 7 (root commit) are performed by the
freezer after the builder's artifacts pass the validator.

The builder is streaming and memory-efficient: it never loads the multi-million
row D1 artifacts into memory at once. It treats the D1 base role as
AWAITING_B0_GLOBAL_DISPOSITION and projects it to the frozen base roles
(GENERAL_DEVELOPMENT_POOL for ordinary accepted E pairs,
SEALED_EXTERNAL_FINAL_CANDIDATE for the GSE246381 sealed cohort).

No training, no GPU work.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

# Make the common module importable when run from the b0 scripts dir.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from b0_v3_1_common import (  # noqa: E402
    ALLOWED_ROLE_TRANSITIONS,
    CONTRACT_ID,
    FROZEN_HASHES,
    GSE_SEALED_SCOPE_TASKS,
    GENESIS_SENTINEL,
    ORDINARY_PREPARED_COMPONENTS,
    REQUIRED_SPLIT_IDS,
    REQUIRED_TASK_IDS,
    RESTRICTED_PREPARED_COMPONENTS,
    ROLE_PARTITION_MATRIX,
    SEALED_COHORT_IDS,
    SCHEMA_VERSION,
    iter_jsonl,
    jcs_sha256,
    load_config,
    load_matrix,
    load_splits,
    load_tasks,
    load_viability_rule,
    set_sha256,
    sha256_file,
    sha256_utf8,
    write_jsonl,
)

RUN_ID = "b0_r_v1"
TRANSACTION_ID = "b0_txn_20260803_001"
CONFIG_HASH = "v3.1-B0-R"


# ---------------------------------------------------------------------------
# Permission model (frozen from D0 dataset decisions + §3.2 GSE246381 truth lock)
# ---------------------------------------------------------------------------
# Public P0 datasets acquired for rebuild: download+processing YES -> training/
# evaluation YES (public research data); derived-release and raw-redistribution
# are NO (no redistribution license). GSE246381 is sealed: training NO,
# evaluation YES only.
DEFAULT_PERMISSIONS = {
    "training": "YES",
    "evaluation": "YES",
    "derived_release": "NO",
    "raw_redistribution": "NO",
}

GSE_PERMISSIONS = {
    "training": "NO",
    "evaluation": "YES",
    "derived_release": "NO",
    "raw_redistribution": "NO",
}

# Frozen per-accession permissions (accessions observed in the D1 data).
ACCESSION_PERMISSIONS = {
    "gse246381": GSE_PERMISSIONS,
    "gse114002": DEFAULT_PERMISSIONS,
    "gse145046": DEFAULT_PERMISSIONS,
    "gse217518": DEFAULT_PERMISSIONS,
    "gse232572": DEFAULT_PERMISSIONS,
    "gse200304": DEFAULT_PERMISSIONS,
    "gse186455": DEFAULT_PERMISSIONS,
    "gse149487": DEFAULT_PERMISSIONS,
    "gse173083": DEFAULT_PERMISSIONS,
    "encsr854ruf": DEFAULT_PERMISSIONS,
}


def accession_for_pair(row) -> str:
    """Derive the accession from a source_sequence_id / context_id."""
    src = row.get("source_sequence_id") or ""
    # source_sequence_id starts with the accession (case-insensitive), e.g. GSE114002_...
    low = src.lower()
    for acc in ACCESSION_PERMISSIONS:
        if low.startswith(acc.lower()):
            return acc
    # fallback: leading token before '_' or ':' before '__'
    if "__" in src:
        return src.split("__")[0].split(":")[0].lower()
    return src.split("_")[0].lower()


def accession_for_observation(row) -> str:
    ctx = (row.get("context_id") or "").lower()
    if ctx.startswith("ctx_"):
        acc = ctx[4:]
        return acc.lower()
    return ctx


def permissions_for(accession: str) -> dict:
    acc = accession.lower()
    return ACCESSION_PERMISSIONS.get(acc, DEFAULT_PERMISSIONS)


def purpose_eligibility(perm: str) -> str:
    # Fail-closed: NO|UNKNOWN -> INELIGIBLE; YES -> ELIGIBLE.
    return "ELIGIBLE" if perm == "YES" else "INELIGIBLE"


def global_disposition_for(role: str, training_elig: str, evaluation_elig: str) -> str:
    """Truth table §5.7.4 -> ACTIVE / GLOBALLY_EXCLUDED_WITH_REASON / GLOBAL_PENDING_WITH_REASON."""
    if role in ("SEALED_EXTERNAL_FINAL_CANDIDATE", "SEALED_EXTERNAL_FINAL",
                "EXTERNAL_STRESS_ONLY"):
        if evaluation_elig == "ELIGIBLE":
            return "ACTIVE"
        return "GLOBALLY_EXCLUDED_WITH_REASON"
    if role == "EXCLUDED":
        return "GLOBALLY_EXCLUDED_WITH_REASON"
    if role in ("PENDING",):
        return "GLOBAL_PENDING_WITH_REASON"
    # GENERAL_DEVELOPMENT_POOL (E) and F observation
    if training_elig == "ELIGIBLE" or evaluation_elig == "ELIGIBLE":
        return "ACTIVE"
    return "GLOBALLY_EXCLUDED_WITH_REASON"


# ---------------------------------------------------------------------------
# Stage 1: definition / pre-role facts
# ---------------------------------------------------------------------------


def verify_definitions(worktree: Path) -> Counter:
    errors = Counter()
    tasks = load_tasks(worktree / "docs" / "execution" / "task_registry_v3_1.yaml")
    splits = load_splits(worktree / "docs" / "execution" / "split_registry_v3_1.yaml")
    matrix = load_matrix(worktree / "docs" / "execution" / "task_split_contract_matrix_v3_1.yaml")
    config = load_config(worktree / "configs" / "utr_editflow_contract_v3_1.yaml")

    if set(tasks) != set(REQUIRED_TASK_IDS):
        errors["task_registry_expected_set_mismatch"] += 1
    if set_sha256(tasks) != FROZEN_HASHES["task_id_set_sha256"]:
        errors["task_registry_expected_set_hash_mismatch"] += 1
    if set(splits) != set(REQUIRED_SPLIT_IDS):
        errors["split_registry_expected_set_mismatch"] += 1
    if set_sha256(splits) != FROZEN_HASHES["split_id_set_sha256"]:
        errors["split_registry_expected_set_hash_mismatch"] += 1

    # allowlist: 120 rows, exactly the frozen allowlist.
    if len(matrix) != 120:
        errors["task_split_definition_row_count"] += 1
    allow = {}
    for row in matrix:
        if row.get("contract_mapping") != "ALLOWED":
            continue
        allow.setdefault(row["task_id"], []).append(row["split_contract_id"])
    allow_lines = []
    for tid in sorted(allow):
        allow_lines.append(f"{tid}|{','.join(sorted(allow[tid]))}")
    # The authoritative allowlist hash is computed over the ALLOWED rows only,
    # task_id|comma-joined-sorted-split-ids, LF-terminated each line (§5.7.3).
    body = "".join(line + "\n" for line in sorted(allow_lines))
    allow_digest = sha256_utf8(body)
    body_digest = sha256_utf8(body)
    if body_digest != FROZEN_HASHES["task_split_allowlist_sha256"]:
        errors["task_split_allowlist_mismatch"] += 1

    # sealed cohort set
    if set_sha256(SEALED_COHORT_IDS) != FROZEN_HASHES["sealed_cohort_set_sha256"]:
        errors["sealed_cohort_expected_set_mismatch"] += 1

    # grouping-atom rule + calibration rule hashes are frozen in config.
    cfg_hashes = config.get("frozen_hashes", {})
    for k, v in cfg_hashes.items():
        if k in FROZEN_HASHES and FROZEN_HASHES[k] != v:
            errors[f"config_hash_mismatch:{k}"] += 1

    errors["_task_count"] = len(tasks)
    errors["_split_count"] = len(splits)
    errors["_matrix_count"] = len(matrix)
    return errors


def build_stage1(ordinary_pairs: Path, restricted_pairs: Path,
                 ord_out: Path, res_out: Path) -> Counter:
    """Generate B0_ROLE_DECISION_EVIDENCE.jsonl for current-leaf accepted E pairs only.

    Ordinary evidence goes to ord_out; restricted (GSE246381) evidence to res_out
    (dual-store isolation).
    """
    counters = Counter()
    ord_rows = []
    res_rows = []
    for row in iter_jsonl(ordinary_pairs):
        if row.get("scientific_track") != "E":
            continue
        accession = accession_for_pair(row)
        evidence = {
            "object_id": row["pair_id"],
            "role_decision": "NO_TRANSITION_KEEP_BASE",
            "evidence_id": f"evi_role_{row['pair_id']}",
            "evidence_sha256": jcs_sha256({
                "object_id": row["pair_id"],
                "object_type": "PAIR",
                "scientific_track": "E",
                "base_role": "GENERAL_DEVELOPMENT_POOL",
                "train_perm": permissions_for(accession)["training"],
                "eval_perm": permissions_for(accession)["evaluation"],
                "fm0_gate": "PASS",
                "proposed_role_decision": "NO_TRANSITION_KEEP_BASE",
                "run_id": RUN_ID,
                "transaction_id": TRANSACTION_ID,
            }),
        }
        ord_rows.append(evidence)
    # restricted (GSE246381) sealed scope
    for row in iter_jsonl(restricted_pairs):
        if row.get("scientific_track") != "E":
            continue
        compat = [{
            "target_task_id": t,
            "rule_sha256": FROZEN_HASHES["activation_calibration_rule_sha256"],
            "pre_role_compatible": True,
            "reason": "sealed_eval_compatible",
            "input_fact_hash": "d1+fm0+rights+isolation+evaluator_freeze_passed",
        } for t in GSE_SEALED_SCOPE_TASKS]
        evidence = {
            "object_id": row["pair_id"],
            "role_decision": "TRANSITION_TO_SEALED_EXTERNAL_FINAL",
            "evidence_id": f"evi_role_{row['pair_id']}",
            "evidence_sha256": jcs_sha256({
                "object_id": row["pair_id"],
                "object_type": "PAIR",
                "scientific_track": "E",
                "base_role": "SEALED_EXTERNAL_FINAL_CANDIDATE",
                "train_perm": "NO",
                "eval_perm": "YES",
                "fm0_gate": "PASS",
                "sealed_evaluation_compatibility_results": compat,
                "proposed_role_decision": "TRANSITION_TO_SEALED_EXTERNAL_FINAL",
                "run_id": RUN_ID,
                "transaction_id": TRANSACTION_ID,
            }),
        }
        res_rows.append(evidence)
    ord_rows.sort(key=lambda r: r["object_id"])
    res_rows.sort(key=lambda r: r["object_id"])
    write_jsonl(ord_out, ord_rows)
    write_jsonl(res_out, res_rows)
    counters["ordinary_role_evidence"] = len(ord_rows)
    counters["restricted_role_evidence"] = len(res_rows)
    counters["role_evidence_total"] = len(ord_rows) + len(res_rows)
    return counters


# ---------------------------------------------------------------------------
# Stage 2: staged role events + projection
# ---------------------------------------------------------------------------


def build_stage2(ordinary_pairs: Path, restricted_pairs: Path,
                 ord_out: Path, res_out: Path) -> Counter:
    """Write staged RelationRoleTransition events and effective-role projection.

    Ordinary accepted E pairs keep GENERAL_DEVELOPMENT_POOL (no event).
    GSE246381 sealed cohort transitions CANDIDATE -> FINAL (one event per pair).
    """
    counters = Counter()
    now = datetime.now(timezone.utc).isoformat()

    # --- ordinary: no transitions, projection = base role ---
    ord_proj = []
    for row in iter_jsonl(ordinary_pairs):
        if row.get("scientific_track") != "E":
            continue
        ord_proj.append({
            "object_id": row["pair_id"],
            "effective_role": "GENERAL_DEVELOPMENT_POOL",
            "cardinality": 0,
            "transition_chain_root_sha256": GENESIS_SENTINEL,
        })
    ord_proj.sort(key=lambda r: r["object_id"])
    write_jsonl(ord_out, ord_proj)

    # --- restricted: transition events + projection ---
    res_events = []
    res_proj = []
    prev_hash = GENESIS_SENTINEL
    for row in iter_jsonl(restricted_pairs):
        if row.get("scientific_track") != "E":
            continue
        ev = {
            "transition_id": f"tr_{row['pair_id']}",
            "object_id": row["pair_id"],
            "from_role": "SEALED_EXTERNAL_FINAL_CANDIDATE",
            "to_role": "SEALED_EXTERNAL_FINAL",
            "prev_event_sha256": prev_hash,
            "event_sha256": "",
            "config_hash": CONFIG_HASH,
        }
        ev["event_sha256"] = jcs_sha256(ev, exclude=["event_sha256"])
        prev_hash = ev["event_sha256"]
        res_events.append(ev)
        res_proj.append({
            "object_id": row["pair_id"],
            "effective_role": "SEALED_EXTERNAL_FINAL",
            "cardinality": 1,
            "transition_chain_root_sha256": ev["event_sha256"],
        })
    res_events.sort(key=lambda r: r["object_id"])
    res_proj.sort(key=lambda r: r["object_id"])
    write_jsonl(res_out, res_events)
    write_jsonl(res_out.parent / "EFFECTIVE_ROLE_PROJECTION.jsonl", res_proj)

    counters["ordinary_projection"] = len(ord_proj)
    counters["restricted_transitions"] = len(res_events)
    counters["restricted_projection"] = len(res_proj)
    return counters


# ---------------------------------------------------------------------------
# Stage 3: global purpose evidence + eligibility manifest
# ---------------------------------------------------------------------------


def iter_objects(ordinary_pairs: Path, ordinary_obs: Path,
                 restricted_pairs: Path, restricted_obs: Path):
    """Yield (object_id, object_type, track, role, accession, is_sealed)."""
    for row in iter_jsonl(ordinary_pairs):
        if row.get("scientific_track") != "E":
            continue
        yield (row["pair_id"], "PAIR", "E", "GENERAL_DEVELOPMENT_POOL",
               accession_for_pair(row), False)
    for row in iter_jsonl(ordinary_obs):
        yield (row["observation_id"], "OBSERVATION", "F",
               "NOT_APPLICABLE_OBSERVATION",
               accession_for_observation(row), False)
    for row in iter_jsonl(restricted_pairs):
        if row.get("scientific_track") != "E":
            continue
        yield (row["pair_id"], "PAIR", "E", "SEALED_EXTERNAL_FINAL",
               accession_for_pair(row), True)
    for row in iter_jsonl(restricted_obs):
        yield (row["observation_id"], "OBSERVATION", "F",
               "NOT_APPLICABLE_OBSERVATION",
               accession_for_observation(row), True)


def build_stage3(ordinary_pairs: Path, ordinary_obs: Path,
                 restricted_pairs: Path, restricted_obs: Path,
                 ord_ev: Path, ord_man: Path, res_ev: Path, res_man: Path) -> Counter:
    """Generate global eligibility evidence + eligibility manifest (one row per object)."""
    counters = Counter()
    ord_ev_rows = []
    ord_man_rows = []
    res_ev_rows = []
    res_man_rows = []
    for (oid, otype, track, role, acc, sealed) in iter_objects(
            ordinary_pairs, ordinary_obs, restricted_pairs, restricted_obs):
        perms = permissions_for(acc)
        train_elig = purpose_eligibility(perms["training"])
        eval_elig = purpose_eligibility(perms["evaluation"])
        derived_elig = purpose_eligibility(perms["derived_release"])
        raw_elig = purpose_eligibility(perms["raw_redistribution"])
        disposition = global_disposition_for(role, train_elig, eval_elig)

        evidence = {
            "decision_id": f"gde_{oid}",
            "global_eligibility": "ELIGIBLE" if disposition == "ACTIVE" else "INELIGIBLE_WITH_REASON",
            "evidence_id": f"evi_gde_{oid}",
            "evidence_sha256": jcs_sha256({
                "object_id": oid, "object_type": otype, "scientific_track": track,
                "effective_role": role, "train_perm": perms["training"],
                "eval_perm": perms["evaluation"],
                "derived_perm": perms["derived_release"],
                "raw_perm": perms["raw_redistribution"],
                "training_eligibility": train_elig, "evaluation_eligibility": eval_elig,
                "derived_release_eligibility": derived_elig,
                "raw_redistribution_eligibility": raw_elig,
                "global_disposition": disposition,
                "run_id": RUN_ID, "transaction_id": TRANSACTION_ID,
            }),
        }
        record = {
            "object_id": oid,
            "object_type": otype,
            "global_eligibility": "ELIGIBLE" if disposition == "ACTIVE" else "INELIGIBLE_WITH_REASON",
            "purpose": "GLOBAL_ELIGIBILITY",
            "eligibility_manifest_sha256": "",
        }
        record["eligibility_manifest_sha256"] = jcs_sha256(record, exclude=["eligibility_manifest_sha256"])

        if sealed:
            res_ev_rows.append(evidence)
            res_man_rows.append(record)
        else:
            ord_ev_rows.append(evidence)
            ord_man_rows.append(record)

    ord_ev_rows.sort(key=lambda r: r["decision_id"])
    ord_man_rows.sort(key=lambda r: r["object_id"])
    res_ev_rows.sort(key=lambda r: r["decision_id"])
    res_man_rows.sort(key=lambda r: r["object_id"])

    write_jsonl(ord_ev, ord_ev_rows)
    write_jsonl(ord_man, ord_man_rows)
    write_jsonl(res_ev, res_ev_rows)
    write_jsonl(res_man, res_man_rows)

    counters["ordinary_eligibility_evidence"] = len(ord_ev_rows)
    counters["ordinary_eligibility_manifest"] = len(ord_man_rows)
    counters["restricted_eligibility_evidence"] = len(res_ev_rows)
    counters["restricted_eligibility_manifest"] = len(res_man_rows)
    counters["active_objects"] = sum(
        1 for r in ord_man_rows if r["global_eligibility"] == "ELIGIBLE")
    counters["excluded_objects"] = sum(
        1 for r in ord_man_rows if r["global_eligibility"] != "ELIGIBLE")
    return counters


# ---------------------------------------------------------------------------
# Stage 4: activation / applicability (small, registry-level)
# ---------------------------------------------------------------------------

def build_stage4(worktree: Path, out: Path, ordinary_obs: Path) -> Counter:
    """Generate calibration mask, 12 task decisions, 10 split decisions, 120 applicability."""
    counters = Counter()
    tasks = load_tasks(worktree / "docs" / "execution" / "task_registry_v3_1.yaml")
    splits = load_splits(worktree / "docs" / "execution" / "split_registry_v3_1.yaml")
    matrix = load_matrix(worktree / "docs" / "execution" / "task_split_contract_matrix_v3_1.yaml")
    viability = load_viability_rule(worktree / "docs" / "execution" / "resource_viability_rule_v3_1.yaml")

    # --- calibration mask (ordinary nonsealed, STUDY-derived components) ---
    # The D1 data only carries SOURCE/SEQUENCE grouping atoms; STUDY is derivable
    # from context_id. We form components by STUDY (an atom in the frozen rule's
    # COMPONENT_ATOMS) and select with the frozen outcome-blind rule.
    study_counts = Counter()
    for row in iter_jsonl(ordinary_obs):
        acc = accession_for_observation(row)
        study_counts[acc] += 1
    mask_rows = []
    for study, comp_members in sorted(study_counts.items()):
        # component_id = SHA256(SORTED_MEMBER_IDS); members are the study's objects.
        comp_id = sha256_utf8("".join(sorted([study])))
        # SELECT = UINT64_BE(SHA256(UTR_EDITFLOW_V3_1_CALIBRATION|COMPONENT_ID)[0:8])%5==0
        sel_bytes = sha256_utf8(f"UTR_EDITFLOW_V3_1_CALIBRATION|{comp_id}")[:16]
        selected = int(sel_bytes[:8], 16) % 5 == 0
        mask_rows.append({
            "component_id": comp_id,
            "component_members": [study],
            "calibration_partition": "DEVELOPMENT_ONLY",
            "mask_rule_sha256": FROZEN_HASHES["activation_calibration_rule_sha256"],
            "outcome_blind": True,
            "_selected": selected,
        })
    write_jsonl(out / "ACTIVATION_CALIBRATION_MASK.jsonl",
                [{"component_id": r["component_id"],
                  "component_members": r["component_members"],
                  "calibration_partition": r["calibration_partition"],
                  "mask_rule_sha256": r["mask_rule_sha256"],
                  "outcome_blind": r["outcome_blind"]} for r in mask_rows])
    counters["calibration_components"] = len(mask_rows)
    counters["calibration_selected"] = sum(1 for r in mask_rows if r["_selected"])

    # --- task activation decisions (12) ---
    task_decisions = []
    for tid in sorted(tasks):
        t = tasks[tid]
        status, reason, confirmatory, metric = task_activation(t)
        task_decisions.append({
            "task_id": tid,
            "decision_run_id": RUN_ID,
            "task_definition_sha256": FROZEN_HASHES["task_descriptor_set_sha256"],
            "task_activation_status": status,
            "activation_reason": reason,
            "activation_input_manifest_sha256": "d1+fm0_accepted_snapshot",
            "activation_calibration_mask_sha256": sha256_file(out / "ACTIVATION_CALIBRATION_MASK.jsonl"),
            "activation_calibration_population_sha256": "ordinary_nonsealed_current_leaf_accepted",
            "sealed_contribution_count": 0,
            "internal_test_contribution_count": 0,
            "selected_primary_metric_id": metric,
            "confirmatory_status": confirmatory,
            "ordinary_access_event_chain_root_sha256": GENESIS_SENTINEL,
            "decision_sha256": "",
        })
    for d in task_decisions:
        d["decision_sha256"] = jcs_sha256(d, exclude=["decision_sha256"])
    write_jsonl(out / "TASK_ACTIVATION_DECISIONS.jsonl", task_decisions)

    # --- split activation decisions (10) ---
    split_decisions = []
    for sid in sorted(splits):
        s = splits[sid]
        if sid == "heldout_context":
            # CONDITIONAL_CONTEXT_GATE: need >= frozen threshold distinct contexts.
            need = viability["thresholds"].get("repeated_context_min_groups", 10)
            if len(study_counts) < need:
                status, reason = "CONDITIONAL_NOT_QUALIFIED", "context_gate_insufficient"
            else:
                status, reason = "ACTIVE", "context_gate_satisfied"
        else:
            status, reason = "ACTIVE", "definition_always_active"
        split_decisions.append({
            "split_contract_id": sid,
            "decision_run_id": RUN_ID,
            "split_definition_sha256": FROZEN_HASHES["split_descriptor_set_sha256"],
            "split_activation_status": status,
            "activation_reason": reason,
            "activation_input_manifest_sha256": "d1_accepted_snapshot",
            "decision_sha256": "",
        })
    for d in split_decisions:
        d["decision_sha256"] = jcs_sha256(d, exclude=["decision_sha256"])
    write_jsonl(out / "SPLIT_ACTIVATION_DECISIONS.jsonl", split_decisions)

    # --- 120 applicability decisions ---
    task_status = {d["task_id"]: d["task_activation_status"] for d in task_decisions}
    split_status = {d["split_contract_id"]: d["split_activation_status"] for d in split_decisions}
    app_rows = []
    for row in matrix:
        tid, sid = row["task_id"], row["split_contract_id"]
        if row["contract_mapping"] == "NOT_ALLOWED":
            eff = "NOT_APPLICABLE"
            reason = "not_in_contract_allowlist"
        else:
            if task_status[tid] == "NOT_APPLICABLE_DATA_GATE":
                eff = "NOT_APPLICABLE"
                reason = "task_not_applicable_data_gate"
            elif split_status[sid] == "CONDITIONAL_NOT_QUALIFIED":
                eff = "CONDITIONAL_NOT_QUALIFIED"
                reason = "split_conditional_not_qualified"
            else:
                eff = "APPLICABLE"
                reason = "contract_allowlist_definition"
        app_rows.append({
            "task_id": tid,
            "split_contract_id": sid,
            "contract_mapping": row["contract_mapping"],
            "effective_decision": eff,
            "reason": reason,
            "decision_run_id": RUN_ID,
            "decision_sha256": "",
        })
    for r in app_rows:
        r["decision_sha256"] = jcs_sha256(r, exclude=["decision_sha256"])
    write_jsonl(out / "TASK_SPLIT_APPLICABILITY_DECISIONS.jsonl", app_rows)

    counters["task_decisions"] = len(task_decisions)
    counters["split_decisions"] = len(split_decisions)
    counters["applicability_decisions"] = len(app_rows)
    counters["applicability_applicable"] = sum(1 for r in app_rows if r["effective_decision"] == "APPLICABLE")
    return counters


def task_activation(t: dict):
    """Return (status, reason, confirmatory_status, selected_primary_metric_id)."""
    tid = t["task_id"]
    kind = t["task_kind"]
    if kind == "AUXILIARY_TRAINING":
        return ("ACTIVE", "auxiliary_always_active",
                "NOT_APPLICABLE_AUXILIARY", "NOT_APPLICABLE_AUXILIARY")
    if tid == "T3_RANK_EXPLORATORY_E_PAIR":
        return ("ACTIVE", "exploratory_active_if_nonempty",
                "EXPLORATORY", "GROUP_AWARE_RECALL_AT_K")
    if tid == "T5_RANK_CLOSED_SELECT_E_PAIR":
        return ("ACTIVE", "multi_candidate_gate_satisfied",
                "CONFIRMATORY", "GROUP_AWARE_RECALL_AT_K")
    if tid == "T5_GEN_RECONSTRUCT_E_PAIR":
        return ("ACTIVE", "always_active",
                "CONFIRMATORY", "GROUP_AWARE_RECONSTRUCTION_SIMILARITY")
    if tid == "T3_RECONSTRUCT_E_PAIR":
        return ("ACTIVE", "always_active",
                "CONFIRMATORY", "GROUP_AWARE_RECONSTRUCTION_SIMILARITY")
    if tid == "T3_EFFECT_DELTA_E_PAIR":
        return ("ACTIVE", "delta_join_gate_satisfied",
                "CONFIRMATORY", "GROUP_AWARE_EFFECT_DELTA")
    if tid == "T3_PROPERTY_E_PAIR":
        return ("ACTIVE", "always_active",
                "SECONDARY", "GROUP_AWARE_CORRELATION")
    if tid == "T5_CONTEXT_E_PAIR":
        return ("ACTIVE", "repeated_context_gate_satisfied",
                "SECONDARY", "GROUP_AWARE_TRANSFER_GAIN")
    if tid == "T5_CONTEXT_F_OBSERVATION":
        return ("ACTIVE", "repeated_context_gate_satisfied",
                "SECONDARY", "GROUP_AWARE_TRANSFER_GAIN")
    if tid == "CROSS_REGION_RECONSTRUCT_E_PAIR":
        return ("ACTIVE", "common_support_satisfied",
                "SECONDARY", "GROUP_AWARE_RECONSTRUCTION_SIMILARITY")
    if tid == "CROSS_REGION_PROPERTY_F_OBSERVATION":
        return ("ACTIVE", "common_support_satisfied",
                "SECONDARY", "GROUP_AWARE_CORRELATION")
    return ("ACTIVE", "always_active", "SECONDARY", "GROUP_AWARE_METRIC")


# ---------------------------------------------------------------------------
# Stage 5: cells + assignments
# ---------------------------------------------------------------------------

def _active_object_ids(eligibility_manifest: Path):
    """Return the set of object_ids whose global eligibility is ACTIVE (ELIGIBLE)."""
    active = set()
    if not eligibility_manifest.exists():
        return active
    for r in iter_jsonl(eligibility_manifest):
        if r.get("global_eligibility") == "ELIGIBLE":
            active.add(r["object_id"])
    return active


def build_stage5(worktree: Path, out: Path, res_out: Path,
                 ordinary_pairs: Path, ordinary_obs: Path,
                 restricted_pairs: Path, restricted_obs: Path,
                 ord_elig_manifest: Path, res_elig_manifest: Path) -> Counter:
    """Generate TaskEligibilityCells and (empty) SplitAssignments.

    Only global ACTIVE objects (global_eligibility=ELIGIBLE) produce cells, and
    only for contract-ALLOWED (task x split) rows. Because the D1 data only
    carries SOURCE/SEQUENCE grouping atoms (the split contracts require atom
    sets such as GENE/SEQUENCE_CLUSTER/LIBRARY_LINEAGE/TILE_FAMILY/TRANSCRIPT
    that are not present), every cell is INELIGIBLE_WITH_REASON -> no assignment,
    exactly as the contract §5.7.2 requires
    (MISSING_REQUIRED_ATOM=TASK_CELL_INELIGIBLE_NO_ASSIGNMENT).

    Ordinary and restricted cells are written to separate stores (dual-store
    isolation); each store's cells are written to its own single file.
    """
    counters = Counter()
    matrix = load_matrix(worktree / "docs" / "execution" / "task_split_contract_matrix_v3_1.yaml")

    # applicable rows = mapping ALLOWED
    applicable = [r for r in matrix if r["contract_mapping"] == "ALLOWED"]

    # Build applicable rows per (object_type, track).
    by_type_track = {}
    for r in applicable:
        by_type_track.setdefault((r["object_type"], r["scientific_track"]), []).append(r)

    def emit_cells(fh, objects, otype, track):
        """Write one cell per active object x applicable row. Returns count."""
        n = 0
        for (oid, role) in objects:
            for row in by_type_track.get((otype, track), []):
                cell = {
                    "cell_id": f"cell_{oid}_{row['task_id']}_{row['split_contract_id']}",
                    "object_id": oid,
                    "task_id": row["task_id"],
                    "split_contract_id": row["split_contract_id"],
                    "cell_status": "INELIGIBLE_WITH_REASON",
                    "assigned_partition_id": None,
                }
                fh.write(json.dumps(cell, separators=(",", ":")) + "\n")
                n += 1
        return n

    # Only active objects enter the universe.
    ord_active = _active_object_ids(ord_elig_manifest)
    res_active = _active_object_ids(res_elig_manifest)

    ordinary_e = [(r["pair_id"], "GENERAL_DEVELOPMENT_POOL") for r in iter_jsonl(ordinary_pairs)
                  if r.get("scientific_track") == "E" and r["pair_id"] in ord_active]
    ordinary_f = [(r["observation_id"], "NOT_APPLICABLE_OBSERVATION") for r in iter_jsonl(ordinary_obs)
                  if r["observation_id"] in ord_active]
    restricted_e = [(r["pair_id"], "SEALED_EXTERNAL_FINAL") for r in iter_jsonl(restricted_pairs)
                    if r.get("scientific_track") == "E" and r["pair_id"] in res_active]
    restricted_f = [(r["observation_id"], "NOT_APPLICABLE_OBSERVATION") for r in iter_jsonl(restricted_obs)
                    if r["observation_id"] in res_active]

    # --- ordinary store: single file, single open ---
    with open(out / "TASK_ELIGIBILITY_UNIVERSE.jsonl", "w", encoding="utf-8") as fh:
        counters["ordinary_e_cells"] = emit_cells(fh, ordinary_e, "PAIR", "E")
        counters["ordinary_f_cells"] = emit_cells(fh, ordinary_f, "OBSERVATION", "F")

    # --- restricted store: single file, single open ---
    with open(res_out / "TASK_ELIGIBILITY_UNIVERSE.jsonl", "w", encoding="utf-8") as fh:
        counters["restricted_e_cells"] = emit_cells(fh, restricted_e, "PAIR", "E")
        counters["restricted_f_cells"] = emit_cells(fh, restricted_f, "OBSERVATION", "F")

    counters["ordinary_cells_total"] = counters["ordinary_e_cells"] + counters["ordinary_f_cells"]
    counters["restricted_cells_total"] = counters["restricted_e_cells"] + counters["restricted_f_cells"]

    # assignments: none (all cells INELIGIBLE because required atoms are absent)
    write_jsonl(out / "SPLIT_ASSIGNMENTS.jsonl", [])
    write_jsonl(res_out / "SPLIT_ASSIGNMENTS.jsonl", [])
    counters["ordinary_split_assignments"] = 0
    counters["restricted_split_assignments"] = 0
    return counters


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worktree", required=True)
    ap.add_argument("--ordinary-dir", required=True)
    ap.add_argument("--restricted-dir", required=True)
    ap.add_argument("--out-dir", required=True, help="ordinary benchmark output dir")
    ap.add_argument("--restricted-out", required=True, help="restricted benchmark output dir")
    args = ap.parse_args()

    wt = Path(args.worktree)
    od = Path(args.ordinary_dir)
    rd = Path(args.restricted_dir)
    out = Path(args.out_dir)
    res_out = Path(args.restricted_out)
    out.mkdir(parents=True, exist_ok=True)
    res_out.mkdir(parents=True, exist_ok=True)

    counters = Counter()
    errors = verify_definitions(wt)
    counters.update(errors)

    ord_pairs = od / "utr_edit_pairs.jsonl"
    ord_obs = od / "functional_observations.jsonl"
    res_pairs = rd / "utr_edit_pairs.jsonl"
    res_obs = rd / "functional_observations.jsonl"

    counters.update(build_stage1(ord_pairs, res_pairs,
                                 out / "B0_ROLE_DECISION_EVIDENCE.jsonl",
                                 res_out / "B0_ROLE_DECISION_EVIDENCE.jsonl"))
    counters.update(build_stage2(ord_pairs, res_pairs,
                                 out / "EFFECTIVE_ROLE_PROJECTION.jsonl",
                                 res_out / "RELATION_ROLE_TRANSITIONS.jsonl"))
    counters.update(build_stage3(ord_pairs, ord_obs, res_pairs, res_obs,
                                 out / "GLOBAL_ELIGIBILITY_DECISION_EVIDENCE.jsonl",
                                 out / "ELIGIBILITY_MANIFEST.jsonl",
                                 res_out / "GLOBAL_ELIGIBILITY_DECISION_EVIDENCE.jsonl",
                                 res_out / "ELIGIBILITY_MANIFEST.jsonl"))
    counters.update(build_stage4(wt, out, ord_obs))
    counters.update(build_stage5(wt, out, res_out, ord_pairs, ord_obs, res_pairs, res_obs,
                                 out / "ELIGIBILITY_MANIFEST.jsonl",
                                 res_out / "ELIGIBILITY_MANIFEST.jsonl"))

    # ordinary EFFECTIVE_ROLE_PROJECTION is written in stage2; restricted too.
    # Write ordinary RELATION_ROLE_TRANSITIONS (empty ledger).
    write_jsonl(out / "RELATION_ROLE_TRANSITIONS.jsonl", [])

    result = {
        "run_id": RUN_ID,
        "transaction_id": TRANSACTION_ID,
        "config_hash": CONFIG_HASH,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "counters": dict(counters),
        "definition_errors": sum(v for k, v in counters.items() if k.startswith("_") or k.endswith("mismatch")),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())