#!/usr/bin/env python
"""G7 (v3.1) — shared constants for the fresh closure finalizer.

Supplies the authoritative blocker IDs (data-goal vs model-rebind), the terminal
state machine, and frozen G7 snapshot identifiers. No training, no GPU work.
"""

from __future__ import annotations

import hashlib
import sys

CONTRACT_ID = "utr_editflow_goal_v3.1_benchmark_first"
SCHEMA_VERSION = "3.1"
GOAL_ID = "GOAL-V3-DATA-BENCH-01"
G7_RUN_ID = "g7_r_v1"
G7_TRANSACTION_ID = "g7_txn_20260803_001"
G7_SNAPSHOT_ID = "g7_snapshot_20260803_001"

# Authoritative contract SHA256 (task statement; must match frozen contract).
CONTRACT_SHA256 = "35dd4bf27a3c7d574ab777f5d858ad1b13dcb9273bdb4961e4c30a1a94bf8759"

GENESIS_SENTINEL = "GENESIS"

# ---------------------------------------------------------------------------
# Blocker semantics
# ---------------------------------------------------------------------------
# data_goal_required_blocker_ids: data blockers this Goal must close. For the
# current project the benchmark cannot form a usable anti-leakage partition
# because the D1 canonical lacks the grouping atoms required by the split
# contracts (GENE / SEQUENCE_CLUSTER / LIBRARY_LINEAGE / TILE_FAMILY /
# TRANSCRIPT / STUDY). That blocker cannot be closed inside this Goal (it needs
# a D1 data extension / rebuild to materialize the grouping atoms), so it is
# recorded honestly as OPEN_WITH_EVIDENCE with evidence / path / condition /
# owner, and the Goal is therefore BLOCKED_WITH_EVIDENCE.
DATA_GOAL_REQUIRED_BLOCKER_IDS = sorted([
    "DB_01_SPLIT_GROUPING_ATOMS_MISSING",
    "DB_02_GSE246381_ROW_ISOLATION",
    "DB_03_DUAL_STORE_CONSERVATION",
    "DB_04_ACCESS_CHAIN_INTEGRITY",
    "DB_05_ANALYTIC_FINAL_COUNTERS_ZERO",
    "DB_06_RESOURCE_VIABILITY_BINDING",
])

# model_rebind_handoff_blocker_ids: model-rebind related blockers (GP0 hard-coded
# old paired count, model not trained, etc.). These are allowed to stay OPEN in
# this data Goal, but each must carry evidence / current paths / closure
# condition / owner.
MODEL_REBIND_HANDOFF_BLOCKER_IDS = sorted([
    "MRB_01_GP0_PAIRED_COUNT_REBIND",
    "MRB_02_MODEL_TRAINING_NOT_AUTHORIZED",
    "MRB_03_SOURCE_BINDING_ORACLE",
    "MRB_04_METHOD_ATTRIBUTION_TESTS",
])

# GP0 is permanently LOCKED in this Goal.
GP0_STATUS = "LOCKED_NOT_AUTHORIZED"

# Terminal states.
TERMINAL_BLOCKED = "BLOCKED_WITH_EVIDENCE"
TERMINAL_READY = "DATA_BENCHMARK_V1_CLOSED_READY_FOR_MODEL_REBIND"

# The five non-analytic machine-event classes that G7 must close one-by-one.
NONANALYTIC_MACHINE_EVENT_TYPES = [
    "RESTRICTED_BUILDER_PARSE",
    "AGGREGATE_QC_MACHINE",
    "FM_OVERLAP_AGGREGATE",
    "B0_ELIGIBILITY_SPLIT_BUILD",
    "G7_RESTRICTED_FINALIZER",
]

# GSE analytic / human event types that must all have count == 0.
GSE_FORBIDDEN_EVENT_TYPES = [
    "HUMAN_SEQUENCE_VIEW",
    "HUMAN_LABEL_VIEW",
    "TRAIN",
    "TUNE",
    "MODEL_SELECTION",
    "INTERNAL_TEST_EVALUATOR",
    "PRE_FINAL_ERROR_ANALYSIS",
    "FINAL_ATTEMPT",
    "FINAL_EVALUATOR",
]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_utf8(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def set_sha256(ids) -> str:
    """SHA256 over UTF-8, lexicographic, LF-terminated each element."""
    lines = "".join(sorted(str(i) + "\n" for i in ids))
    return sha256_utf8(lines)


def main() -> int:
    print("data_goal_required_blocker_ids_sha256:",
          set_sha256(DATA_GOAL_REQUIRED_BLOCKER_IDS))
    print("model_rebind_handoff_blocker_ids_sha256:",
          set_sha256(MODEL_REBIND_HANDOFF_BLOCKER_IDS))
    # intersection must be empty
    inter = set(DATA_GOAL_REQUIRED_BLOCKER_IDS) & set(MODEL_REBIND_HANDOFF_BLOCKER_IDS)
    print("intersection:", sorted(inter))
    return 0


if __name__ == "__main__":
    sys.exit(main())