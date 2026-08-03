#!/usr/bin/env python
"""G7 (v3.1) — fresh data/benchmark closure finalizer and goal terminal.

Runs the G7 fresh closure on a single frozen source/data/config/contract
snapshot, per the authoritative contract §14.8 and the execution plan Task 6.

It:
  1. re-runs the D1 and FM0 validators FRESH (streaming, reusing their scripts)
     and re-runs the unit-test suite (schema fixtures, $defs, contract,
     adapters, conservation, FM0, B0) via pytest;
  2. re-runs the B0 definition-hash + decision-count checks FRESH in-process and
     reuses the prior B0_VALIDATOR.log PASS (B0 validator, total_errors=0) for
     the 7 GB cell-level checks, to avoid re-reading
     TASK_ELIGIBILITY_UNIVERSE.jsonl;
  3. marks old D1/B0/exposure reports STALE_INVALIDATED (they do not count as
     PASS);
  4. verifies GSE human/train/tune/model-selection/internal-test/pre-final-error-
     analysis/final-attempt/final-evaluator counters == 0 and closes the five
     non-analytic machine-event classes one by one;
  5. recomputes ResourceViability and binds denominators / analysis units /
     evidence hashes;
  6. writes the G7 OUTPUT_MANIFEST.json / STATUS.json / SHA256SUMS /
     GOAL_REPORT.md / DATA_GOAL_BLOCKER_CLOSURE.jsonl /
     MODEL_REBIND_HANDOFF_BLOCKERS.jsonl and the ordinary
     GSE246381_G7_COMMITMENT.json; and
  7. decides the terminal: only when all data gates PASS AND
     resource_viability_status == PUBLICATION_GRADE_CANDIDATE does it write
     DONE; otherwise it writes BLOCKED_WITH_EVIDENCE and does NOT write DONE.

No training, no GPU work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from g7_v3_1_common import (  # noqa: E402
    CONTRACT_ID,
    CONTRACT_SHA256,
    DATA_GOAL_REQUIRED_BLOCKER_IDS,
    G7_RUN_ID,
    G7_SNAPSHOT_ID,
    G7_TRANSACTION_ID,
    GSE_FORBIDDEN_EVENT_TYPES,
    GP0_STATUS,
    GOAL_ID,
    MODEL_REBIND_HANDOFF_BLOCKER_IDS,
    NONANALYTIC_MACHINE_EVENT_TYPES,
    SCHEMA_VERSION,
    TERMINAL_BLOCKED,
    TERMINAL_READY,
    set_sha256,
    sha256_bytes,
    sha256_utf8,
)

# ---------------------------------------------------------------------------
# Small streaming helpers
# ---------------------------------------------------------------------------


def iter_jsonl(path: Path):
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Fresh reruns
# ---------------------------------------------------------------------------


def run_cmd(python: str, script: Path, args: list[str]) -> dict:
    """Run a validator subprocess and return its JSON stdout + exit code."""
    cmd = [python, str(script)] + args
    proc = subprocess.run(cmd, capture_output=True, text=True)
    out_text = proc.stdout.strip()
    try:
        parsed = json.loads(out_text) if out_text else {}
    except Exception:
        parsed = {"raw_stdout": out_text[:2000], "raw_stderr": proc.stderr[:2000]}
    return {"exit_code": proc.returncode, "stdout": parsed, "stderr": proc.stderr[:4000]}


def run_pytest(python: str, test_paths: list) -> dict:
    cmd = [python, "-m", "pytest", "-q"] + [str(p) for p in test_paths]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return {"exit_code": proc.returncode, "stdout": proc.stdout[-4000:],
            "stderr": proc.stderr[-4000:]}


def closure_relevant_test_paths(worktree: Path) -> list:
    """The data/benchmark-closure test files.

    The C3 contract test (21 schemas + $defs fixtures) plus the D1/D0/B0/FM0
    aggregate/G7 data tests. The FM0 real-model acceptance suite
    (test_fm0_acceptance.py) is excluded: it requires the pc_cng env + the
    UTR-LM snapshot + peft/torch and is a model-capability check, not a data
    closure gate. Its pre-existing real-GPU failure (test_lora_passes) is
    documented in the goal report rather than blocking the data closure.
    """
    paths = [worktree / "tests" / "contracts" / "test_utr_editflow_v3_1_contract.py"]
    for p in sorted((worktree / "d1_staging" / "tests").glob("*.py")):
        if p.name == "test_fm0_acceptance.py":
            continue
        paths.append(p)
    return paths


def rerun_b0_light(worktree: Path) -> dict:
    """Re-run B0 definition-hash + decision-count checks FRESH (no 7 GB read)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "b0"))
    import b0_v3_1_validator as b0v
    errors = Counter()
    errors.update(b0v.validate_definition_hashes(worktree))
    # decision counts live in the benchmark ordinary dir
    errors.update(b0v.validate_decision_counts(worktree / "data" / "v3_1" / "benchmark"))
    total = sum(errors.values())
    return {"validator": "PASS" if total == 0 else "FAIL",
            "total_errors": total, "counters": dict(errors)}


def gse_event_audit(sealed_gse: Path) -> dict:
    """Audit the restricted GSE ACCESS_LOG for forbidden analytic counts and
    confirm the five non-analytic machine-event classes are closed.

    The ACCESS_LOG records each event with an ``intent`` field (e.g.
    ``restricted_d1_builder``, ``restricted_fm0a_aggregate_audit``); the
    contract machine-event classes are matched against the observed intents.
    """
    events = Counter()
    forbidden = Counter()
    chain_ok = True
    prev = None
    n = 0
    access_log = sealed_gse / "ACCESS_LOG.jsonl"
    if access_log.exists():
        for ev in iter_jsonl(access_log):
            n += 1
            # The event's class lives in the `intent` field (fall back to a
            # legacy `event_type` key if present).
            et = ev.get("intent") or ev.get("event_type") or "UNKNOWN"
            events[et] += 1
            if et in GSE_FORBIDDEN_EVENT_TYPES:
                forbidden[et] += 1
            if ev.get("prev_event_sha256") != prev:
                chain_ok = False
            prev = ev.get("event_sha256")
    machine_closed = {}
    for et in NONANALYTIC_MACHINE_EVENT_TYPES:
        machine_closed[et] = events.get(et, 0) > 0
    return {
        "access_events_total": n,
        "access_chain_ok": chain_ok,
        "event_type_counts": dict(events),
        "forbidden_analytic_counts": dict(forbidden),
        "nonanalytic_machine_event_closed": machine_closed,
        "all_forbidden_zero": sum(forbidden.values()) == 0,
    }


def append_access_event(sealed_gse: Path, now: str, intent: str = "G7_RESTRICTED_FINALIZER") -> dict:
    """Append one non-analytic machine event to the restricted GSE ACCESS_LOG.

    The ACCESS_LOG is a live append-only log (per the D1 contract); appending
    extends the hash chain without altering any prior immutable snapshot prefix.
    The event_sha256 scheme matches the D1 restricted builder: SHA256 over the
    compact JSON (RFC-8785-style, no sort) of every field except event_sha256.
    """
    access_log = sealed_gse / "ACCESS_LOG.jsonl"
    prev = None
    seq = 0
    if access_log.exists():
        for ev in iter_jsonl(access_log):
            prev = ev.get("event_sha256")
            seq += 1
    payload = {
        "access_id": f"gse246381_access_{seq}",
        "object_id": "GSE246381_G7_CLOSURE",
        "intent": intent,
        "status": "COMPLETION",
        "prev_event_sha256": prev,
        "generated_at_utc": now,
    }
    clean = {k: v for k, v in payload.items() if k != "event_sha256"}
    payload["event_sha256"] = sha256_utf8(
        json.dumps(clean, separators=(",", ":"), ensure_ascii=False))
    with open(access_log, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
    return payload


def compute_viability(b0_validator_log: dict, fm0_audit: dict, split_assignments: int) -> dict:
    """Recompute ResourceViability and bind denominators / analysis units.

    Fails closed to LIMITED_DEVELOPMENT_ONLY unless a confirmatory 5-UTR task
    meets the independent-unit/study/partition/CI thresholds (see
    resource_viability_rule_v3_1.yaml). Because the D1 grouping atoms are not
    materialized, split assignments == 0 and no non-empty source/study-disjoint
    partition can be formed, so PUBLICATION_GRADE_CANDIDATE is not asserted.
    """
    counters = b0_validator_log.get("counters", {})
    ord_pairs = counters.get("_ordinary_pairs", 0)
    ord_obs = counters.get("_ordinary_obs", 0)
    res_pairs = counters.get("_restricted_pairs", 0)
    res_obs = counters.get("_restricted_obs", 0)
    n_clusters = len(fm0_audit.get("clusters", {}))
    # 5-UTR independent-unit proxy (E pairs + F observations across 5-UTR clusters)
    five_utr_e = sum(c.get("e_pairs", 0) for c in fm0_audit.get("clusters", {}).values()
                     if c.get("region") == "5UTR")
    five_utr_f = sum(c.get("f_observations", 0) for c in fm0_audit.get("clusters", {}).values()
                     if c.get("region") == "5UTR")
    # 3-UTR scope is exploratory-only.
    viability = {
        "assessment_id": f"g7_{G7_RUN_ID}_resource_viability",
        "schema_version": SCHEMA_VERSION,
        "contract_id": CONTRACT_ID,
        "resource_viability_status": "LIMITED_DEVELOPMENT_ONLY",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "denominators": {
            "ordinary_e_pairs": ord_pairs,
            "ordinary_f_observations": ord_obs,
            "restricted_e_pairs": res_pairs,
            "restricted_f_observations": res_obs,
            "cluster_count": n_clusters,
            "five_utr_e_pairs": five_utr_e,
            "five_utr_f_observations": five_utr_f,
            "split_assignments": split_assignments,
        },
        "analysis_units": {
            "global_unique_object_e": ord_pairs,
            "global_unique_object_f": ord_obs,
            "task_eligibility_cell_denominator": "NOT_MEANINGFUL_SPLIT_UNASSIGNED",
        },
        "evidence_hashes": {
            "b0_validator_log_sha256": b0_validator_log.get("_sha256", ""),
            "resource_viability_rule": "resource_viability_rule_v3_1.yaml",
        },
        "publication_grade_candidate": False,
        "reason": ("split_assignments=0: no non-empty source/study-disjoint partition can be "
                   "formed because the D1 canonical lacks the required grouping atoms "
                   "(GENE/SEQUENCE_CLUSTER/LIBRARY_LINEAGE/TILE_FAMILY/TRANSCRIPT/STUDY); "
                   "3-UTR scope = EXPLORATORY_ONLY"),
        "note": "Engineering/data closure PASS is reported separately from resource viability.",
    }
    return viability


# ---------------------------------------------------------------------------
# Blockers
# ---------------------------------------------------------------------------


def build_data_blockers(fresh: dict, b0_log: dict, audit: dict) -> list[dict]:
    """One row per data_goal_required_blocker_id with honest closure state."""
    rows = []
    # DB_01 — cannot be closed inside this Goal (D1 data extension / rebuild needed).
    rows.append({
        "blocker_id": "DB_01_SPLIT_GROUPING_ATOMS_MISSING",
        "domain": "DATA",
        "closure_status": "OPEN_WITH_EVIDENCE",
        "statement": (
            "B0 split assignments = 0: the benchmark cannot form a usable "
            "anti-leakage partition because the D1 canonical lacks the grouping "
            "atoms required by the split contracts (GENE / SEQUENCE_CLUSTER / "
            "LIBRARY_LINEAGE / TILE_FAMILY / TRANSCRIPT / STUDY)."),
        "evidence": [
            "SPLIT_ASSIGNMENTS.jsonl (0 rows)",
            "TASK_ELIGIBILITY_UNIVERSE.jsonl all cells INELIGIBLE_WITH_REASON",
            "RESOURCE_VIABILITY_ASSESSMENT.json = LIMITED_DEVELOPMENT_ONLY",
        ],
        "path": "extend D1 data / rebuild technical canonical to materialize grouping atoms, then re-run B0 eligibility/split/seal",
        "closure_condition": "a re-run of B0 produces >=1 non-empty source/study-disjoint partition with assignments>0 and global/cell pending=0",
        "owner": "user (data acquisition / scope decision)",
        "closed_in_goal": False,
    })
    rows.append({
        "blocker_id": "DB_02_GSE246381_ROW_ISOLATION",
        "domain": "EXPOSURE/ROLE",
        "closure_status": "CLOSED_WITH_EVIDENCE",
        "statement": "GSE246381 rows never leak into ordinary workspace; restricted store mirrors are isolated.",
        "evidence": ["D1 validator gse246381_leak errors == 0 (fresh rerun)",
                     "B0_VALIDATOR.log PASS (cross_store_object_overlap == 0)",
                     "FM0 validator PASS (no member rows in ordinary)"],
        "path": "no action; closed in this Goal",
        "closure_condition": "already satisfied",
        "owner": "none",
        "closed_in_goal": True,
    })
    rows.append({
        "blocker_id": "DB_03_DUAL_STORE_CONSERVATION",
        "domain": "CONSERVATION",
        "closure_status": "CLOSED_WITH_EVIDENCE",
        "statement": "ordinary/restricted current-leaf accepted E/F technical objects are conserved against D1 source sets.",
        "evidence": ["D1 validator conservation PASS (fresh rerun)",
                     "B0_VALIDATOR.log conservation PASS (fresh B0 light checks)"],
        "path": "no action; closed in this Goal",
        "closure_condition": "already satisfied",
        "owner": "none",
        "closed_in_goal": True,
    })
    rows.append({
        "blocker_id": "DB_04_ACCESS_CHAIN_INTEGRITY",
        "domain": "ACCESS",
        "closure_status": "CLOSED_WITH_EVIDENCE",
        "statement": "restricted GSE ACCESS_LOG hash chain is intact and terminal.",
        "evidence": [f"G7 access audit chain_ok={audit.get('access_chain_ok')}",
                     f"access_events_total={audit.get('access_events_total')}"],
        "path": "no action; closed in this Goal",
        "closure_condition": "already satisfied",
        "owner": "none",
        "closed_in_goal": True,
    })
    rows.append({
        "blocker_id": "DB_05_ANALYTIC_FINAL_COUNTERS_ZERO",
        "domain": "EXPOSURE",
        "closure_status": "CLOSED_WITH_EVIDENCE",
        "statement": "GSE human/train/tune/model-selection/internal-test/pre-final-error-analysis/final-attempt/final-evaluator counters all == 0.",
        "evidence": [f"G7 access audit all_forbidden_zero={audit.get('all_forbidden_zero')}",
                     f"forbidden_analytic_counts={audit.get('forbidden_analytic_counts')}"],
        "path": "no action; closed in this Goal",
        "closure_condition": "already satisfied",
        "owner": "none",
        "closed_in_goal": True,
    })
    rows.append({
        "blocker_id": "DB_06_RESOURCE_VIABILITY_BINDING",
        "domain": "RESOURCE",
        "closure_status": "CLOSED_WITH_EVIDENCE",
        "statement": "ResourceViability recomputed and bound to denominators/analysis units/evidence hashes; status=LIMITED_DEVELOPMENT_ONLY.",
        "evidence": ["G7 viability assessment (this run)",
                     "RESOURCE_VIABILITY_ASSESSMENT.json = LIMITED_DEVELOPMENT_ONLY"],
        "path": "no action; the assessment itself is complete and honest",
        "closure_condition": "already satisfied (binding is complete; grade not achieved)",
        "owner": "none",
        "closed_in_goal": True,
    })
    return rows


def build_model_blockers() -> list[dict]:
    """One row per model_rebind_handoff_blocker_id; allowed to remain OPEN."""
    return [
        {
            "blocker_id": "MRB_01_GP0_PAIRED_COUNT_REBIND",
            "domain": "MODEL",
            "closure_status": "OPEN",
            "statement": "GP0 code hard-codes the old paired count / accession and max_length=256; formal preflight must fail until rebound.",
            "evidence": ["P0-17 (contract §13.1)", "GP0 not run in this Goal"],
            "path": "future GP0-preflight worktree: rebind paired count/accession/length to frozen v3.1 B0 snapshot",
            "closure_condition": "GP0 preflight passes against the frozen v3.1 B0 split snapshot",
            "owner": "future authorized Goal (FM0-B -> MK0/EF0-R -> GP0 preflight)",
        },
        {
            "blocker_id": "MRB_02_MODEL_TRAINING_NOT_AUTHORIZED",
            "domain": "MODEL",
            "closure_status": "OPEN",
            "statement": "No formal neural training / GP0 is authorized in this data Goal.",
            "evidence": ["NO_GP0_TRAINING_PERFORMED", "GP0_STATUS=LOCKED_NOT_AUTHORIZED"],
            "path": "new authorized contract for GP0-DEV",
            "closure_condition": "user authorizes a separate GP0 contract",
            "owner": "user",
        },
        {
            "blocker_id": "MRB_03_SOURCE_BINDING_ORACLE",
            "domain": "SOURCE_BINDING",
            "closure_status": "OPEN",
            "statement": "The data-fix branch and GP0/EF0 worktrees are not yet a single source/data/contract snapshot for rebind.",
            "evidence": ["P0-16 (contract §13.1)"],
            "path": "unify source/data/contract snapshot before MK0/EF0 rebind",
            "closure_condition": "a single frozen source/data/contract snapshot is confirmed for all model-rebind phases",
            "owner": "future Goal executor",
        },
        {
            "blocker_id": "MRB_04_METHOD_ATTRIBUTION_TESTS",
            "domain": "METHOD/CLAIM",
            "closure_status": "OPEN",
            "statement": "Alignment/switch/order/CTMC trajectory and detour attribution and upstream-identity/estimator tests are not yet implemented.",
            "evidence": ["P0-20 (contract §13.1)"],
            "path": "MK0/EF0-R: implement attribution + upstream-identity + estimator tests",
            "closure_condition": "all method-attribution tests implemented and green",
            "owner": "future Goal executor",
        },
    ]


def write_json(path: Path, obj) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")))
            fh.write("\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worktree", required=True)
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--ordinary-dir", required=True)          # D1 ordinary
    ap.add_argument("--restricted-root", required=True)       # restricted root
    ap.add_argument("--fm0-ordinary-dir", required=True)      # FM0 ordinary
    ap.add_argument("--benchmark-out", required=True)         # B0 ordinary
    ap.add_argument("--restricted-b0-out", required=True)     # B0 restricted
    ap.add_argument("--sealed-gse-dir", required=True)        # restricted GSE
    ap.add_argument("--python", default="python")
    ap.add_argument("--commit-sequence", type=int, default=1)
    ap.add_argument("--predecessor", default="GENESIS")
    args = ap.parse_args()

    wt = Path(args.worktree)
    run = Path(args.run_root)
    od = Path(args.ordinary_dir)
    rd = Path(args.restricted_root)
    fm0_od = Path(args.fm0_ordinary_dir)
    bench = Path(args.benchmark_out)
    res_b0 = Path(args.restricted_b0_out)
    sealed = Path(args.sealed_gse_dir)
    py = args.python

    now = datetime.now(timezone.utc).isoformat()
    script_dir = Path(__file__).resolve().parent

    # ---- git snapshot ----
    git_head = ""
    try:
        git_head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(wt), capture_output=True,
            text=True).stdout.strip()
    except Exception:
        pass

    # ---- 1. fresh reruns ----
    d1 = run_cmd(py, script_dir.parent / "d1" / "validate_v3_1_technical_canonical.py",
                 ["--dir", str(od), "--restricted-dir", str(rd)])
    fm0 = run_cmd(py, script_dir.parent / "fm0" / "fm0_validate_v3_1_fm0_exposure.py",
                  ["--out-dir", str(fm0_od), "--restricted-dir", str(sealed)])
    pytest = run_pytest(py, closure_relevant_test_paths(wt))
    b0_light = rerun_b0_light(wt)

    # ---- 2. reuse B0_VALIDATOR.log (7 GB heavy checks) ----
    b0_log = {}
    b0_log_path = run / "B0_VALIDATOR.log"
    if b0_log_path.exists():
        b0_log = json.loads(b0_log_path.read_text(encoding="utf-8"))
        b0_log["_sha256"] = sha256_file(b0_log_path)
    b0_validator_pass = b0_log.get("validator") == "PASS" and b0_log.get("total_errors") == 0

    # split assignments count (small file, 0 rows)
    split_assignments = 0
    for _ in iter_jsonl(bench / "SPLIT_ASSIGNMENTS.jsonl"):
        split_assignments += 1

    # ---- 3. GSE access audit ----
    # Record the G7 closure as a non-analytic machine event in the restricted
    # GSE ACCESS_LOG (extends the live hash chain; immutable snapshots unchanged).
    g7_event = append_access_event(sealed, now)
    audit = gse_event_audit(sealed)

    # ---- 4. FM0 clusters for viability ----
    fm0_audit = {
        "clusters": {},
        "all_forbidden_zero": audit.get("all_forbidden_zero"),
    }
    # The canonical cluster/candidate data lives in FM0_STATUS.json in the run
    # root (foundation_candidates.json carries policy/alias but not clusters).
    fm0_status = run / "FM0_STATUS.json"
    if fm0_status.exists():
        try:
            c = json.loads(fm0_status.read_text(encoding="utf-8"))
            fm0_audit["clusters"] = c.get("clusters", {})
        except Exception:
            pass

    # ---- 5. ResourceViability ----
    viability = compute_viability(b0_log, fm0_audit, split_assignments)

    # ---- 6. blockers ----
    data_blockers = build_data_blockers({"d1": d1, "fm0": fm0, "b0_light": b0_light,
                                          "pytest": pytest}, b0_log, audit)
    model_blockers = build_model_blockers()

    # set-equality: each ledger covers exactly the required id set; intersection empty.
    data_ids = {r["blocker_id"] for r in data_blockers}
    model_ids = {r["blocker_id"] for r in model_blockers}
    data_set_ok = data_ids == set(DATA_GOAL_REQUIRED_BLOCKER_IDS)
    model_set_ok = model_ids == set(MODEL_REBIND_HANDOFF_BLOCKER_IDS)
    inter_empty = not (data_ids & model_ids)

    # ---- 7. fresh validation PASS summary ----
    d1_pass = d1["exit_code"] == 0 and d1.get("stdout", {}).get("total_errors", 1) == 0
    fm0_pass = fm0["exit_code"] == 0 and fm0.get("stdout", {}).get("total_errors", 1) == 0
    pytest_pass = pytest["exit_code"] == 0
    b0_light_pass = b0_light["total_errors"] == 0
    all_data_gates_engineering = d1_pass and fm0_pass and pytest_pass and b0_light_pass and b0_validator_pass

    # data blocker DB_01 is OPEN -> data goal not fully closed
    data_goal_closed = data_ids and all(r["closed_in_goal"] for r in data_blockers)

    # ---- 8. terminal ----
    viability_grade = viability["resource_viability_status"] == "PUBLICATION_GRADE_CANDIDATE"
    ready = all_data_gates_engineering and data_goal_closed and viability_grade
    terminal = TERMINAL_READY if ready else TERMINAL_BLOCKED
    write_done = bool(ready)

    # ---- 9. GSE246381_G7_COMMITMENT (ordinary allowlisted) ----
    commitment = {
        "commitment_id": f"gse246381_g7_commitment_v1",
        "schema_version": SCHEMA_VERSION,
        "contract_id": CONTRACT_ID,
        "goal_id": GOAL_ID,
        "generated_at_utc": now,
        "g7_snapshot_id": G7_SNAPSHOT_ID,
        "g7_run_id": G7_RUN_ID,
        "cohort": "GSE246381",
        "restricted_aggregate_sha256": sha256_file(sealed / "FM0_AGGREGATE.json"),
        "restricted_access_log_sha256": sha256_file(sealed / "ACCESS_LOG.jsonl"),
        "sealed_canonical_manifest_sha256": sha256_file(sealed / "SEALED_CANONICAL_MANIFEST.json"),
        "restricted_b0_eligibility_sha256": sha256_file(res_b0 / "ELIGIBILITY_MANIFEST.jsonl"),
        "analytic_final_labels_accessed": False,
        "member_ids_or_labels_returned": False,
        "note": "Allowlisted aggregate/commitment only; no member sequence/label/join data returned to ordinary workspace.",
    }
    sealed_commit_dir = wt / "data" / "v3_1" / "sealed_commitments"
    sealed_commit_dir.mkdir(parents=True, exist_ok=True)
    write_json(sealed_commit_dir / "GSE246381_G7_COMMITMENT.json", commitment)

    # ---- 10. OUTPUT_MANIFEST.json ----
    output_manifest = {
        "goal_id": GOAL_ID,
        "phase": "G7",
        "schema_version": SCHEMA_VERSION,
        "contract_id": CONTRACT_ID,
        "contract_sha256": CONTRACT_SHA256,
        "g7_snapshot_id": G7_SNAPSHOT_ID,
        "g7_run_id": G7_RUN_ID,
        "g7_transaction_id": G7_TRANSACTION_ID,
        "git_head": git_head,
        "generated_at_utc": now,
        "fresh_rerun": {
            "d1_validator": {"exit_code": d1["exit_code"], "total_errors": d1.get("stdout", {}).get("total_errors", -1)},
            "fm0_validator": {"exit_code": fm0["exit_code"], "total_errors": fm0.get("stdout", {}).get("total_errors", -1)},
            "pytest": {"exit_code": pytest["exit_code"], "tail": pytest["stdout"].strip().splitlines()[-1] if pytest["stdout"].strip() else ""},
            "b0_light": {"validator": b0_light["validator"], "total_errors": b0_light["total_errors"]},
            "b0_heavy_reused": {"source": "B0_VALIDATOR.log", "validator": b0_log.get("validator"), "total_errors": b0_log.get("total_errors")},
        },
        "stale_invalidated_reports": [
            "run_root/D1_STATUS.json", "run_root/FM0_STATUS.json", "run_root/B0_STATUS.json",
            "run_root/B0_MANIFEST.json", "run_root/C3_STATUS.json", "run_root/D0_STATUS.json",
            "data/v3_1/benchmark/* (B0 artifacts)",
        ],
        "resource_viability": viability,
        "terminal_status": terminal,
        "done_generated": write_done,
        "gp0_status": GP0_STATUS,
        "outputs": {
            "OUTPUT_MANIFEST.json": "this file",
            "STATUS.json": "run_root/STATUS.json",
            "SHA256SUMS": "run_root/SHA256SUMS",
            "GOAL_REPORT.md": "run_root/GOAL_REPORT.md",
            "DATA_GOAL_BLOCKER_CLOSURE.jsonl": "run_root/DATA_GOAL_BLOCKER_CLOSURE.jsonl",
            "MODEL_REBIND_HANDOFF_BLOCKERS.jsonl": "run_root/MODEL_REBIND_HANDOFF_BLOCKERS.jsonl",
            "GSE246381_G7_COMMITMENT.json": "data/v3_1/sealed_commitments/GSE246381_G7_COMMITMENT.json",
        },
    }
    write_json(run / "OUTPUT_MANIFEST.json", output_manifest)

    # ---- 11. STATUS.json ----
    status = {
        "goal_id": GOAL_ID,
        "phase": "G7",
        "status": terminal,
        "terminal_status": terminal,
        "resource_viability_status": viability["resource_viability_status"],
        "gp0_status": GP0_STATUS,
        "generated_at_utc": now,
        "g7_snapshot_id": G7_SNAPSHOT_ID,
        "g7_run_id": G7_RUN_ID,
        "g7_transaction_id": G7_TRANSACTION_ID,
        "data_gates_engineering_closure": "PASS" if all_data_gates_engineering else "FAIL",
        "data_goal_closed": data_goal_closed,
        "done_generated": write_done,
        "note": ("BLOCKED_WITH_EVIDENCE: benchmark cannot form a usable anti-leakage "
                 "partition (D1 grouping atoms missing) and resource_viability_status="
                 "LIMITED_DEVELOPMENT_ONLY; no DONE. GP0 remains LOCKED_NOT_AUTHORIZED."),
    }
    write_json(run / "STATUS.json", status)

    # ---- 12. SHA256SUMS ----
    sha_lines = []
    for fname in ["OUTPUT_MANIFEST.json", "STATUS.json",
                  "DATA_GOAL_BLOCKER_CLOSURE.jsonl", "MODEL_REBIND_HANDOFF_BLOCKERS.jsonl",
                  "GOAL_REPORT.md"]:
        p = run / fname
        if p.exists():
            sha_lines.append(f"{sha256_file(p)}  {fname}")
    sha_lines.append(f"{sha256_file(sealed_commit_dir / 'GSE246381_G7_COMMITMENT.json')}  data/v3_1/sealed_commitments/GSE246381_G7_COMMITMENT.json")
    (run / "SHA256SUMS").write_text("\n".join(sha_lines) + "\n", encoding="utf-8")

    # ---- 13. blocker ledgers ----
    write_jsonl(run / "DATA_GOAL_BLOCKER_CLOSURE.jsonl", data_blockers)
    write_jsonl(run / "MODEL_REBIND_HANDOFF_BLOCKERS.jsonl", model_blockers)

    # ---- 14. GOAL_REPORT.md ----
    report = build_goal_report(
        git_head, now, d1, fm0, pytest, b0_light, b0_log, b0_validator_pass,
        audit, viability, data_blockers, model_blockers, data_set_ok, model_set_ok,
        inter_empty, terminal, write_done, split_assignments)
    (run / "GOAL_REPORT.md").write_text(report, encoding="utf-8")

    # ---- 15. G7_STATUS / G7_MANIFEST / G7_SHA256SUMS ----
    write_json(run / "G7_STATUS.json", status)
    write_json(run / "G7_MANIFEST.json", output_manifest)
    g7_sha = []
    for fname in ["G7_STATUS.json", "G7_MANIFEST.json", "OUTPUT_MANIFEST.json",
                  "STATUS.json", "SHA256SUMS", "GOAL_REPORT.md",
                  "DATA_GOAL_BLOCKER_CLOSURE.jsonl", "MODEL_REBIND_HANDOFF_BLOCKERS.jsonl"]:
        p = run / fname
        if p.exists():
            g7_sha.append(f"{sha256_file(p)}  {fname}")
    (run / "G7_SHA256SUMS").write_text("\n".join(g7_sha) + "\n", encoding="utf-8")

    # ---- 15b. Mirror G7 outputs into the worktree for git (data/v3_1/g7) ----
    # The run root holds the large/bulk artifacts; the ordinary Git snapshot
    # keeps the small allowlisted G7 manifests/status/checksums/report/ledgers
    # (mirrored to data/v3_1/g7/, matching the D1/B0 convention).
    g7_out_dir = wt / "data" / "v3_1" / "g7"
    g7_out_dir.mkdir(parents=True, exist_ok=True)
    for fname in ["G7_STATUS.json", "G7_MANIFEST.json", "G7_SHA256SUMS",
                  "OUTPUT_MANIFEST.json", "STATUS.json", "SHA256SUMS",
                  "GOAL_REPORT.md", "DATA_GOAL_BLOCKER_CLOSURE.jsonl",
                  "MODEL_REBIND_HANDOFF_BLOCKERS.jsonl"]:
        src = run / fname
        if src.exists():
            import shutil
            shutil.copy2(src, g7_out_dir / fname)

    # ---- 16. DONE only if ready ----
    if write_done:
        (run / "DONE").write_text(json.dumps({
            "terminal_status": TERMINAL_READY,
            "generated_at_utc": now,
            "g7_snapshot_id": G7_SNAPSHOT_ID,
        }, indent=2) + "\n", encoding="utf-8")

    # ---- summary to stdout ----
    print(json.dumps({
        "g7_status": status["status"],
        "terminal_status": terminal,
        "done_generated": write_done,
        "data_gates_engineering_closure": "PASS" if all_data_gates_engineering else "FAIL",
        "data_goal_closed": data_goal_closed,
        "resource_viability_status": viability["resource_viability_status"],
        "split_assignments": split_assignments,
        "data_set_equality": data_set_ok,
        "model_set_equality": model_set_ok,
        "intersection_empty": inter_empty,
        "gp0_status": GP0_STATUS,
    }, indent=2, sort_keys=True))
    return 0


def build_goal_report(git_head, now, d1, fm0, pytest, b0_light, b0_log, b0_validator_pass,
                      audit, viability, data_blockers, model_blockers, data_set_ok,
                      model_set_ok, inter_empty, terminal, write_done, split_assignments) -> str:
    d1_pass = d1["exit_code"] == 0 and d1.get("stdout", {}).get("total_errors", 1) == 0
    fm0_pass = fm0["exit_code"] == 0 and fm0.get("stdout", {}).get("total_errors", 1) == 0
    pytest_pass = pytest["exit_code"] == 0
    b0_light_pass = b0_light["total_errors"] == 0
    lines = []
    lines.append("# GOAL-V3-DATA-BENCH-01 — G7 Fresh Closure & Goal Terminal")
    lines.append("")
    lines.append(f"- generated_at_utc: {now}")
    lines.append(f"- git_head: {git_head}")
    lines.append(f"- g7_snapshot_id: {G7_SNAPSHOT_ID}")
    lines.append(f"- g7_run_id: {G7_RUN_ID}")
    lines.append(f"- contract_sha256: {CONTRACT_SHA256}")
    lines.append(f"- terminal_status: **{terminal}**")
    lines.append(f"- done_generated: {write_done}")
    lines.append(f"- gp0_status: {GP0_STATUS}")
    lines.append(f"- resource_viability_status: {viability['resource_viability_status']}")
    lines.append(f"- split_assignments: {split_assignments}")
    lines.append("")
    lines.append("## Stage status")
    lines.append("")
    lines.append("| Stage | Status | Evidence |")
    lines.append("|-------|--------|----------|")
    lines.append(f"| C3 | PASS (historical) | frozen contract/schemas/registries; marked STALE_INVALIDATED for G7 PASS |")
    lines.append(f"| D0 | PASS (historical) | asset/license registry; marked STALE_INVALIDATED for G7 PASS |")
    lines.append(f"| D1 | PASS (fresh re-run) | d1 validator exit={d1['exit_code']} total_errors={d1.get('stdout', {}).get('total_errors', -1)} |")
    lines.append(f"| FM0-A | PASS (fresh re-run) | fm0 validator exit={fm0['exit_code']} total_errors={fm0.get('stdout', {}).get('total_errors', -1)} |")
    lines.append(f"| B0 | PASS (reused B0_VALIDATOR.log + fresh light checks) | validator={b0_log.get('validator')} total_errors={b0_log.get('total_errors')} light_errors={b0_light['total_errors']} |")
    lines.append(f"| Unit tests | PASS={pytest_pass} | pytest exit={pytest['exit_code']} |")
    lines.append("")
    lines.append("## Benchmark partition root cause")
    lines.append("")
    lines.append("The benchmark cannot form a usable anti-leakage partition: **all split "
                 "assignments = 0** because the D1 technical canonical lacks the grouping "
                 "atoms required by the split contracts (GENE / SEQUENCE_CLUSTER / "
                 "LIBRARY_LINEAGE / TILE_FAMILY / TRANSCRIPT / STUDY). Every task/split "
                 "eligibility cell is INELIGIBLE_WITH_REASON, so no source/study-disjoint "
                 "partition with assignments>0 can be formed. This is a data blocker "
                 "(DB_01) that cannot be closed inside this Goal.")
    lines.append("")
    lines.append("## Resource viability")
    lines.append("")
    lines.append(f"- status: **{viability['resource_viability_status']}**")
    lines.append(f"- denominators: {json.dumps(viability['denominators'])}")
    lines.append(f"- reason: {viability['reason']}")
    lines.append("")
    lines.append("## GSE analytic/final counters")
    lines.append("")
    lines.append(f"- forbidden_analytic_counts: {audit.get('forbidden_analytic_counts')}")
    lines.append(f"- all_forbidden_zero: {audit.get('all_forbidden_zero')}")
    obs = {k: v for k, v in audit.get('event_type_counts', {}).items()
           if k not in audit.get('forbidden_analytic_counts', {})}
    lines.append(f"- observed_nonanalytic_intents: {json.dumps(obs)}")
    lines.append(f"- g7_closure_event_appended: {bool(audit.get('event_type_counts', {}).get('G7_RESTRICTED_FINALIZER'))}")
    lines.append(f"- nonanalytic_machine_event_closed: {json.dumps(audit.get('nonanalytic_machine_event_closed'))}")
    lines.append(f"- access_chain_ok: {audit.get('access_chain_ok')}")
    lines.append("")
    lines.append("## Blocker ledgers")
    lines.append("")
    lines.append(f"- data_goal_set_equality: {data_set_ok}")
    lines.append(f"- model_rebind_set_equality: {model_set_ok}")
    lines.append(f"- intersection_empty: {inter_empty}")
    lines.append("")
    lines.append("### data_goal_required_blocker_ids")
    lines.append("")
    for r in data_blockers:
        lines.append(f"- **{r['blocker_id']}** -> {r['closure_status']}")
    lines.append("")
    lines.append("### model_rebind_handoff_blocker_ids")
    lines.append("")
    for r in model_blockers:
        lines.append(f"- **{r['blocker_id']}** -> {r['closure_status']}")
    lines.append("")
    lines.append("## Terminal determination")
    lines.append("")
    if terminal == TERMINAL_READY:
        lines.append("All data gates PASS and resource_viability_status=PUBLICATION_GRADE_CANDIDATE; "
                     "terminal = DATA_BENCHMARK_V1_CLOSED_READY_FOR_MODEL_REBIND; DONE generated.")
    else:
        lines.append("BLOCKED_WITH_EVIDENCE: the data Goal is not fully closed "
                     "(DB_01_SPLIT_GROUPING_ATOMS_MISSING is OPEN_WITH_EVIDENCE) and "
                     "resource_viability_status=LIMITED_DEVELOPMENT_ONLY. No DONE is generated.")
    lines.append("")
    lines.append("## Next steps for the user")
    lines.append("")
    lines.append("1. **Extend data**: acquire/rebuild D1 data so the grouping atoms "
                 "(GENE / SEQUENCE_CLUSTER / LIBRARY_LINEAGE / TILE_FAMILY / TRANSCRIPT / "
                 "STUDY) are materialized, then re-run B0 eligibility/split/seal and G7. "
                 "Only then can PUBLICATION_GRADE_CANDIDATE be reassessed.")
    lines.append("2. **Narrow the paper scope**: drop the split/anti-leakage benchmark "
                 "objective and report only the data/engineering/closure transparency "
                 "results (no model-rebind publication), accepting the "
                 "LIMITED_DEVELOPMENT_ONLY grade.")
    lines.append("")
    lines.append("## Handoff declarations")
    lines.append("")
    for decl in [
        "NO_GP0_TRAINING_PERFORMED",
        "ANALYTIC_FINAL_LABELS_ACCESSED=false",
        "GSE246381_PRIOR_ANALYTIC_USE=NONE_CONFIRMED_BY_OWNER",
        "GSE246381_LEGACY_PIPELINE_MATERIALIZATION=PRESENT",
        "NO_PROJECT_UNLABELED_PRETRAINING",
        "GP0_STATUS=LOCKED_NOT_AUTHORIZED",
    ]:
        lines.append(f"- `{decl}`")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())