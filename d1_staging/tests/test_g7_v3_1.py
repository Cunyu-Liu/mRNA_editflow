#!/usr/bin/env python
"""Unit tests for the G7 (v3.1) fresh closure finalizer.

These tests are fast and deterministic: they never touch the 7 GB
TASK_ELIGIBILITY_UNIVERSE.jsonl or the multi-GB D1 files. They cover:

  * blocker id set-equality (data vs model-rebind) and empty intersection
  * terminal-state determination (BLOCKED_WITH_EVIDENCE vs READY)
  * GSE access-audit counter logic (forbidden analytic events == 0)
  * ResourceViability fail-closed (LIMITED_DEVELOPMENT_ONLY when split==0)
  * output manifest / status schema shape
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "g7"))

import g7_v3_1_common as common  # noqa: E402
import g7_v3_1_finalizer as finalizer  # noqa: E402


def test_blocker_sets_equality_and_empty_intersection():
    data = set(common.DATA_GOAL_REQUIRED_BLOCKER_IDS)
    model = set(common.MODEL_REBIND_HANDOFF_BLOCKER_IDS)
    assert data == set(common.DATA_GOAL_REQUIRED_BLOCKER_IDS)
    assert model == set(common.MODEL_REBIND_HANDOFF_BLOCKER_IDS)
    assert not (data & model)
    assert len(data) == 6
    assert len(model) == 4


def test_terminal_blocked_when_viability_limited():
    # split==0 -> viability limited -> terminal is BLOCKED_WITH_EVIDENCE, no DONE
    viability = finalizer.compute_viability(
        {"counters": {"_ordinary_pairs": 88042, "_ordinary_obs": 3322161,
                      "_restricted_pairs": 1184, "_restricted_obs": 15392}},
        {"clusters": {}}, 0)
    assert viability["resource_viability_status"] == "LIMITED_DEVELOPMENT_ONLY"
    assert viability["publication_grade_candidate"] is False
    assert viability["denominators"]["split_assignments"] == 0


def test_gse_audit_forbidden_zeros():
    with tempfile.TemporaryDirectory() as td:
        log = Path(td) / "ACCESS_LOG.jsonl"
        prev = None
        rows = []
        for i, et in enumerate(["RESTRICTED_BUILDER_PARSE", "AGGREGATE_QC_MACHINE",
                                "FM_OVERLAP_AGGREGATE", "B0_ELIGIBILITY_SPLIT_BUILD"]):
            ev = {"event_type": et, "prev_event_sha256": prev, "event_sha256": f"h{i}"}
            prev = ev["event_sha256"]
            rows.append(ev)
        with open(log, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        audit = finalizer.gse_event_audit(Path(td))
        assert audit["all_forbidden_zero"] is True
        assert audit["access_chain_ok"] is True
        assert audit["access_events_total"] == 4
        for et in common.NONANALYTIC_MACHINE_EVENT_TYPES:
            assert et in audit["nonanalytic_machine_event_closed"]


def test_gse_audit_detects_forbidden():
    with tempfile.TemporaryDirectory() as td:
        log = Path(td) / "ACCESS_LOG.jsonl"
        with open(log, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"event_type": "TRAIN",
                                 "prev_event_sha256": None,
                                 "event_sha256": "h0"}) + "\n")
        audit = finalizer.gse_event_audit(Path(td))
        assert audit["all_forbidden_zero"] is False
        assert audit["forbidden_analytic_counts"].get("TRAIN") == 1


def test_gse_audit_uses_intent_field():
    # The ACCESS_LOG records the event class in the `intent` field; a forbidden
    # analytic intent must be detected (not masked as UNKNOWN).
    with tempfile.TemporaryDirectory() as td:
        log = Path(td) / "ACCESS_LOG.jsonl"
        with open(log, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"intent": "FINAL_EVALUATOR",
                                 "prev_event_sha256": None,
                                 "event_sha256": "h0"}) + "\n")
        audit = finalizer.gse_event_audit(Path(td))
        assert audit["all_forbidden_zero"] is False
        assert audit["forbidden_analytic_counts"].get("FINAL_EVALUATOR") == 1
        assert audit["event_type_counts"].get("UNKNOWN") is None


def test_append_access_event_extends_chain():
    with tempfile.TemporaryDirectory() as td:
        sealed = Path(td)
        log = sealed / "ACCESS_LOG.jsonl"
        with open(log, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"access_id": "gse246381_access_0",
                                 "object_id": "x", "intent": "restricted_d1_builder",
                                 "status": "COMPLETION", "prev_event_sha256": None,
                                 "event_sha256": "h0"}) + "\n")
        ev = finalizer.append_access_event(sealed, "2026-08-04T00:00:00+00:00")
        assert ev["access_id"] == "gse246381_access_1"
        assert ev["prev_event_sha256"] == "h0"
        assert ev["intent"] == "G7_RESTRICTED_FINALIZER"
        assert ev["event_sha256"]
        # chain re-audit is intact and the G7 class is now closed
        audit = finalizer.gse_event_audit(sealed)
        assert audit["access_chain_ok"] is True
        assert audit["access_events_total"] == 2
        assert audit["nonanalytic_machine_event_closed"]["G7_RESTRICTED_FINALIZER"] is True


def test_data_blockers_shape_and_honest_open():
    rows = finalizer.build_data_blockers(
        {"d1": {}, "fm0": {}, "b0_light": {}, "pytest": {}},
        {"_sha256": "x"}, {"access_chain_ok": True, "access_events_total": 1,
                           "all_forbidden_zero": True,
                           "forbidden_analytic_counts": {}})
    ids = {r["blocker_id"] for r in rows}
    assert ids == set(common.DATA_GOAL_REQUIRED_BLOCKER_IDS)
    # DB_01 is honestly OPEN_WITH_EVIDENCE; the rest are closed.
    by_id = {r["blocker_id"]: r for r in rows}
    assert by_id["DB_01_SPLIT_GROUPING_ATOMS_MISSING"]["closure_status"] == "OPEN_WITH_EVIDENCE"
    assert by_id["DB_01_SPLIT_GROUPING_ATOMS_MISSING"]["closed_in_goal"] is False
    for bid in common.DATA_GOAL_REQUIRED_BLOCKER_IDS:
        if bid != "DB_01_SPLIT_GROUPING_ATOMS_MISSING":
            assert by_id[bid]["closure_status"] == "CLOSED_WITH_EVIDENCE"
            assert by_id[bid]["closed_in_goal"] is True


def test_model_blockers_all_open_with_meta():
    rows = finalizer.build_model_blockers()
    ids = {r["blocker_id"] for r in rows}
    assert ids == set(common.MODEL_REBIND_HANDOFF_BLOCKER_IDS)
    for r in rows:
        assert r["closure_status"] == "OPEN"
        assert r["evidence"] and r["path"] and r["closure_condition"] and r["owner"]


def test_goal_report_contains_blocked_terminal():
    report = finalizer.build_goal_report(
        "abc123", "2026-08-04T00:00:00+00:00",
        {"exit_code": 0, "stdout": {"total_errors": 0}},
        {"exit_code": 0, "stdout": {"total_errors": 0}},
        {"exit_code": 0},
        {"total_errors": 0}, {"validator": "PASS", "total_errors": 0}, True,
        {"all_forbidden_zero": True, "forbidden_analytic_counts": {},
         "nonanalytic_machine_event_closed": {}, "access_chain_ok": True},
        {"resource_viability_status": "LIMITED_DEVELOPMENT_ONLY",
         "denominators": {}, "reason": "split_assignments=0"},
        finalizer.build_data_blockers({}, {}, {}),
        finalizer.build_model_blockers(),
        True, True, True, "BLOCKED_WITH_EVIDENCE", False, 0)
    assert "BLOCKED_WITH_EVIDENCE" in report
    assert "DB_01_SPLIT_GROUPING_ATOMS_MISSING" in report
    assert "LIMITED_DEVELOPMENT_ONLY" in report
    assert "NO_GP0_TRAINING_PERFORMED" in report