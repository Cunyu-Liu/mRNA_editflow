#!/usr/bin/env python
"""B0-05: Data Card for each evaluation track.

Contract: utr_editflow_contract_v2 (FROZEN) §10.
Task: B0-05
Acceptance: data card complete for all tracks

Consumes the B0-04 eval track manifest and produces a per-track data card
covering the five required dimensions:

  1. counts       — record counts by split, role, region, accession, exposure
  2. bias         — sequence length, GC content, edit distance, label stats
  3. exposure     — exposure status, data role, historically-exposed flag
  4. allowed_claims    — ledger allowed/forbidden claims + track-supported claims
  5. unsupported_capabilities — what this track CANNOT support

The data card is emitted as a single JSON report with one entry per track,
plus an overall summary.

Outputs:
  data/b0_05_data_card.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from canonical_schemas import ALLOWED_CLAIMS_POOL, EVAL_TRACKS  # noqa: E402

# ---------------------------------------------------------------------------
# Track semantics — supported claims + unsupported capabilities
# ---------------------------------------------------------------------------

# Claims that each track can actually evaluate (subset of the scientific claims
# the track's evaluation procedure supports). These are intersected with the
# ledger's per-record allowed_claims to determine the effective claim set.
TRACK_SUPPORTED_CLAIMS: Dict[str, Tuple[str, ...]] = {
    "closed_measured_pool": (
        "edit_effect",            # measured source→candidate pair shows edit effect
        "generation_grounding",   # likelihood/recovery grounded in measured candidate
    ),
    "heldout_generative": (
        "generation_grounding",   # generate from held-out source
        "generative_denoising",   # recover measured candidate from held-out source
    ),
    "open_legal_generation": (
        "generation_grounding",   # legal generation under constraints
        "foundation_adaptation",  # can evaluate foundation model transfer
        "region_representation",  # can evaluate cross-region generation
    ),
}

# Capabilities each track CANNOT support (with reasons).
UNSUPPORTED_CAPABILITIES: Dict[str, List[Dict[str, str]]] = {
    "closed_measured_pool": [
        {
            "capability": "open_ended_generation_diversity",
            "reason": "candidate pool is closed to measured candidates only; "
                      "cannot evaluate diversity of novel candidates",
        },
        {
            "capability": "generative_denoising_from_unmeasured_sources",
            "reason": "requires a measured (source, candidate) pair; cannot "
                      "denoise from sources without a measured target",
        },
        {
            "capability": "cross_region_transfer_quality",
            "reason": "evaluates measured-pair likelihood, not transfer; "
                      "cross-region splits are present but the candidate pool "
                      "remains closed per region",
        },
    ],
    "heldout_generative": [
        {
            "capability": "fully_open_generation_without_target",
            "reason": "always has a measured candidate as the recovery target; "
                      "cannot evaluate generation without a target",
        },
        {
            "capability": "edit_effect_on_unmeasured_candidates",
            "reason": "recovery targets are measured candidates only; cannot "
                      "assess edit effect on candidates that were never measured",
        },
        {
            "capability": "candidate_quality_beyond_measured_recovery",
            "reason": "evaluation is recovery-oriented; cannot claim candidate "
                      "quality beyond measured-candidate likelihood",
        },
    ],
    "open_legal_generation": [
        {
            "capability": "recovery_against_measured_candidates",
            "reason": "no measured target to compare against; generation is "
                      "open-support and unconstrained by a measured pool",
        },
        {
            "capability": "edit_effect_claims",
            "reason": "no measured (source, candidate) pair to define an edit "
                      "effect; candidates are generated, not measured",
        },
        {
            "capability": "measured_pair_likelihood",
            "reason": "no closed candidate pool; cannot compute likelihood of a "
                      "specific measured candidate",
        },
    ],
}


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


# ---------------------------------------------------------------------------
# Statistics helpers (stdlib only)
# ---------------------------------------------------------------------------

def _percentile(sorted_vals: Sequence[float], p: float) -> float:
    """Linear-interpolation percentile on a pre-sorted sequence."""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    k = (len(sorted_vals) - 1) * p
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return float(sorted_vals[f])
    return float(sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f))


def numeric_stats(values: Sequence[float]) -> Dict[str, float]:
    """Compute min/max/mean/median/std/p25/p75 for a numeric sequence."""
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return {"n": 0, "min": 0, "max": 0, "mean": 0, "median": 0,
                "std": 0, "p25": 0, "p75": 0}
    return {
        "n": len(vals),
        "min": vals[0],
        "max": vals[-1],
        "mean": statistics.fmean(vals),
        "median": statistics.median(vals),
        "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
        "p25": _percentile(vals, 0.25),
        "p75": _percentile(vals, 0.75),
    }


def gc_content(seq: str) -> float:
    """Fraction of G+C in a sequence (0..1). Empty → 0."""
    if not seq:
        return 0.0
    gc = seq.count("G") + seq.count("C")
    return gc / len(seq)


# ---------------------------------------------------------------------------
# Data card builders
# ---------------------------------------------------------------------------

def build_counts(
    track: str,
    track_entries: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Count records by split, role, region, accession, exposure class."""
    by_split: Dict[str, Counter] = defaultdict(Counter)
    by_region: Counter = Counter()
    by_accession: Counter = Counter()
    by_exposure_class: Counter = Counter()
    unique_rids: set = set()
    for e in track_entries:
        by_split[e["split"]][e["split_role"]] += 1
        by_region[e.get("region") or "?"] += 1
        by_accession[e.get("accession") or "?"] += 1
        by_exposure_class[e["exposure_class"]] += 1
        unique_rids.add(e["record_id"])
    return {
        "total_assignments": len(track_entries),
        "unique_records": len(unique_rids),
        "by_split": {k: dict(v) for k, v in sorted(by_split.items())},
        "by_region": dict(by_region),
        "by_accession": dict(by_accession),
        "by_exposure_class": dict(by_exposure_class),
    }


def build_bias(
    track: str,
    track_entries: List[Dict[str, Any]],
    canonical_records: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute bias stats: sequence length, GC, edit distance, labels.

    Computed on unique records (deduplicated by record_id) to avoid
    double-counting records that appear in multiple splits.
    """
    seen: set = set()
    src_lens: List[int] = []
    cand_lens: List[int] = []
    src_gc: List[float] = []
    cand_gc: List[float] = []
    edit_dists: List[int] = []
    n_ins: List[int] = []
    n_del: List[int] = []
    n_sub: List[int] = []
    label_values: Dict[str, List[float]] = defaultdict(list)

    for e in track_entries:
        rid = e["record_id"]
        if rid in seen:
            continue
        seen.add(rid)
        rec = canonical_records.get(rid)
        if rec is None:
            continue
        src = rec.get("source_sequence") or ""
        cand = rec.get("candidate_sequence") or ""
        if src:
            src_lens.append(len(src))
            src_gc.append(gc_content(src))
        if cand:
            cand_lens.append(len(cand))
            cand_gc.append(gc_content(cand))
        ed = rec.get("edit_distance", 0)
        edit_dists.append(ed)
        n_ins.append(rec.get("n_ins", 0))
        n_del.append(rec.get("n_del", 0))
        n_sub.append(rec.get("n_sub", 0))
        for k, v in rec.get("labels", {}).items():
            if isinstance(v, (int, float)):
                label_values[k].append(float(v))

    ed_counter = Counter(edit_dists)
    label_stats = {k: numeric_stats(v) for k, v in label_values.items()}

    return {
        "source_length": numeric_stats([float(x) for x in src_lens]),
        "candidate_length": numeric_stats([float(x) for x in cand_lens]),
        "gc_content_source": numeric_stats(src_gc),
        "gc_content_candidate": numeric_stats(cand_gc),
        "edit_distance": {
            **numeric_stats([float(x) for x in edit_dists]),
            "distribution": {str(k): v for k, v in sorted(ed_counter.items())},
        },
        "n_ins_total": sum(n_ins),
        "n_del_total": sum(n_del),
        "n_sub_total": sum(n_sub),
        "n_ins_mean_per_record": (sum(n_ins) / len(n_ins)) if n_ins else 0.0,
        "n_del_mean_per_record": (sum(n_del) / len(n_del)) if n_del else 0.0,
        "n_sub_mean_per_record": (sum(n_sub) / len(n_sub)) if n_sub else 0.0,
        "label_keys": sorted(label_values.keys()),
        "label_stats": label_stats,
    }


def build_exposure(
    track: str,
    track_entries: List[Dict[str, Any]],
    ledger: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Exposure status, data role, historically-exposed flag (unique records)."""
    seen: set = set()
    by_exposure_status: Counter = Counter()
    by_data_role: Counter = Counter()
    by_record_type: Counter = Counter()
    historically_exposed = 0
    labels_allowed_train = 0
    labels_allowed_hp = 0
    for e in track_entries:
        rid = e["record_id"]
        if rid in seen:
            continue
        seen.add(rid)
        led = ledger.get(rid, {})
        by_exposure_status[led.get("exposure_status", "unknown")] += 1
        by_data_role[led.get("data_role", "unknown")] += 1
        by_record_type[led.get("record_type", "unknown")] += 1
        if led.get("historically_exposed"):
            historically_exposed += 1
        if led.get("labels_allowed_for_new_training"):
            labels_allowed_train += 1
        if led.get("labels_allowed_for_new_hyperparameter_selection"):
            labels_allowed_hp += 1
    return {
        "by_exposure_status": dict(by_exposure_status),
        "by_data_role": dict(by_data_role),
        "by_record_type": dict(by_record_type),
        "historically_exposed_count": historically_exposed,
        "labels_allowed_for_new_training": labels_allowed_train,
        "labels_allowed_for_new_hyperparameter_selection": labels_allowed_hp,
    }


def build_allowed_claims(
    track: str,
    track_entries: List[Dict[str, Any]],
    ledger: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Allowed/forbidden claims from the ledger + track-supported claims."""
    seen: set = set()
    allowed_counter: Counter = Counter()
    forbidden_counter: Counter = Counter()
    for e in track_entries:
        rid = e["record_id"]
        if rid in seen:
            continue
        seen.add(rid)
        led = ledger.get(rid, {})
        for c in led.get("allowed_claims", []):
            allowed_counter[c] += 1
        for c in led.get("forbidden_claims", []):
            forbidden_counter[c] += 1
    supported = TRACK_SUPPORTED_CLAIMS[track]
    # Effective = supported AND allowed by ledger (intersection)
    effective = [c for c in supported if allowed_counter.get(c, 0) > 0]
    return {
        "track_supported_claims": list(supported),
        "effective_supported_claims": effective,
        "ledger_allowed_claims_per_claim_counts": dict(
            sorted(allowed_counter.items())
        ),
        "ledger_forbidden_claims_per_claim_counts": dict(
            sorted(forbidden_counter.items())
        ),
        "allowed_claims_pool_schema": list(ALLOWED_CLAIMS_POOL),
        "note": (
            "track_supported_claims = claims this track's evaluation procedure "
            "can support. effective_supported_claims = supported AND present in "
            "the ledger's allowed_claims for at least one record. "
            "ledger_*_per_claim_counts = how many unique records allow/forbid "
            "each claim."
        ),
    }


def build_track_data_card(
    track: str,
    all_entries: List[Dict[str, Any]],
    canonical_records: Dict[str, Dict[str, Any]],
    ledger: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Build the full data card for one track."""
    track_entries = [
        e for e in all_entries if e["track_roles"].get(track) != "none"
    ]
    return {
        "track": track,
        "description": _track_description(track),
        "n_assignments": len(track_entries),
        "counts": build_counts(track, track_entries),
        "bias": build_bias(track, track_entries, canonical_records),
        "exposure": build_exposure(track, track_entries, ledger),
        "allowed_claims": build_allowed_claims(track, track_entries, ledger),
        "unsupported_capabilities": UNSUPPORTED_CAPABILITIES[track],
    }


def _track_description(track: str) -> str:
    return {
        "closed_measured_pool": "measured source-candidate pairs, closed support",
        "heldout_generative": "held-out source→candidate generative likelihood/recovery",
        "open_legal_generation": "open-support legal generation under constraints",
    }[track]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_data_card(
    manifest_path: str,
    canonical_records_path: str,
    exposure_ledger_path: str,
) -> Dict[str, Any]:
    print("Loading eval track manifest...")
    manifest = load_jsonl(manifest_path)
    print(f"  {len(manifest)} assignments")

    print("Loading canonical records...")
    canonical: Dict[str, Dict[str, Any]] = {}
    for r in load_jsonl(canonical_records_path):
        canonical[r["record_id"]] = r
    print(f"  {len(canonical)} records")

    print("Loading exposure ledger...")
    ledger: Dict[str, Dict[str, Any]] = {}
    for r in load_jsonl(exposure_ledger_path):
        ledger[r["record_id"]] = r
    print(f"  {len(ledger)} entries")

    cards: Dict[str, Any] = {}
    for track in EVAL_TRACKS:
        print(f"\nBuilding data card for {track}...")
        cards[track] = build_track_data_card(
            track, manifest, canonical, ledger
        )
        c = cards[track]
        print(f"  n_assignments={c['n_assignments']} "
              f"unique={c['counts']['unique_records']} "
              f"effective_claims={c['allowed_claims']['effective_supported_claims']}")

    # Completeness check: every track has all 5 dimensions non-empty
    required_sections = ["counts", "bias", "exposure", "allowed_claims",
                         "unsupported_capabilities"]
    complete = True
    for track in EVAL_TRACKS:
        for sec in required_sections:
            if not cards[track].get(sec):
                complete = False
                print(f"  WARNING: {track}.{sec} is empty")
        # unsupported_capabilities must have at least 1 entry
        if not cards[track].get("unsupported_capabilities"):
            complete = False

    return {
        "task": "B0-05",
        "contract": "utr_editflow_contract_v2",
        "tracks_documented": list(EVAL_TRACKS),
        "required_dimensions": required_sections,
        "track_cards": cards,
        "acceptance": {
            "data_card_complete_for_all_tracks": complete,
        },
        "overall_pass": complete,
        "manifest_source": manifest_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="B0-05: Data Card")
    parser.add_argument(
        "--manifest", default="data/b0_04_eval_track_manifest.jsonl",
    )
    parser.add_argument(
        "--canonical-records", default="data/d1_canonical_records.jsonl",
    )
    parser.add_argument(
        "--exposure-ledger", default="data/data_exposure_ledger.jsonl",
    )
    parser.add_argument("--output", default="data/b0_05_data_card.json")
    args = parser.parse_args()

    report = run_data_card(
        args.manifest, args.canonical_records, args.exposure_ledger,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nData card written to {out}")
    print(f"Overall pass: {report['overall_pass']}")
    sys.exit(0 if report["overall_pass"] else 1)


if __name__ == "__main__":
    main()
