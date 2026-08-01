"""Unit tests for B0-05 Data Card.

B0-05 acceptance:
  - data card complete for all tracks

Covers:
  - gc_content
  - numeric_stats / _percentile
  - build_counts
  - build_bias (lengths, GC, edit distance, labels)
  - build_exposure
  - build_allowed_claims (supported + effective + ledger counts)
  - build_track_data_card (full per-track card)
  - run_data_card (integration: completeness check)
  - TRACK_SUPPORTED_CLAIMS / UNSUPPORTED_CAPABILITIES sanity

Run: pytest d1_staging/tests/test_b0_data_card.py -v
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
B0_SCRIPTS = os.path.join(HERE, "..", "scripts", "b0")
sys.path.insert(0, B0_SCRIPTS)

from canonical_schemas import ALLOWED_CLAIMS_POOL, EVAL_TRACKS  # noqa: E402
from data_card import (  # noqa: E402
    TRACK_SUPPORTED_CLAIMS,
    UNSUPPORTED_CAPABILITIES,
    _percentile,
    build_allowed_claims,
    build_bias,
    build_counts,
    build_exposure,
    build_track_data_card,
    gc_content,
    numeric_stats,
    run_data_card,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rec(rid, src="ACGTACGT", cand="ACGTACGA", ed=1, labels=None,
         region="5'UTR", accession="GSE114002"):
    return {
        "record_id": rid,
        "accession": accession,
        "region": region,
        "source_sequence": src,
        "candidate_sequence": cand,
        "edit_distance": ed,
        "n_ins": 0,
        "n_del": 0,
        "n_sub": ed,
        "labels": labels or {"rl": 4.5},
    }


def _manifest_entry(rid, split="5utr_source_disjoint", role="train",
                    region="5'UTR", accession="GSE114002",
                    track_roles=None):
    if track_roles is None:
        track_roles = {t: f"{role}_pair" if t == "closed_measured_pool"
                       else f"{role}_heldout_source" if t == "heldout_generative"
                       else f"{role}_gen_source" for t in EVAL_TRACKS}
    return {
        "split": split,
        "record_id": rid,
        "accession": accession,
        "region": region,
        "split_role": role,
        "exposure_class": {"train": "TRAIN", "val": "EVAL_VAL",
                           "test": "EVAL_TEST"}.get(role, "NONE"),
        "track_roles": track_roles,
    }


def _ledger(rid, data_role="D_C", record_type="paired",
            exposure_status="unexposed", allowed=None, forbidden=None,
            hist_exp=False):
    return {
        "record_id": rid,
        "data_role": data_role,
        "record_type": record_type,
        "exposure_status": exposure_status,
        "historically_exposed": hist_exp,
        "labels_allowed_for_new_training": True,
        "labels_allowed_for_new_hyperparameter_selection": True,
        "allowed_claims": allowed or ["edit_effect", "generation_grounding"],
        "forbidden_claims": forbidden or [],
    }


# ---------------------------------------------------------------------------
# gc_content
# ---------------------------------------------------------------------------

class TestGCContent:
    def test_empty(self):
        assert gc_content("") == 0.0

    def test_all_gc(self):
        assert gc_content("GCGC") == 1.0

    def test_no_gc(self):
        assert gc_content("ATAT") == 0.0

    def test_half(self):
        assert gc_content("ACGT") == 0.5

    def test_lowercase_not_counted(self):
        # Only uppercase G/C counted
        assert gc_content("gcgc") == 0.0


# ---------------------------------------------------------------------------
# numeric_stats / _percentile
# ---------------------------------------------------------------------------

class TestNumericStats:
    def test_empty(self):
        s = numeric_stats([])
        assert s["n"] == 0

    def test_single(self):
        s = numeric_stats([5.0])
        assert s["n"] == 1
        assert s["min"] == 5.0
        assert s["max"] == 5.0
        assert s["mean"] == 5.0
        assert s["std"] == 0.0

    def test_range(self):
        s = numeric_stats([1.0, 2.0, 3.0, 4.0, 5.0])
        assert s["min"] == 1.0
        assert s["max"] == 5.0
        assert s["mean"] == 3.0
        assert s["median"] == 3.0
        assert s["p25"] == 2.0
        assert s["p75"] == 4.0

    def test_filters_none(self):
        s = numeric_stats([1.0, None, 3.0])
        assert s["n"] == 2
        assert s["min"] == 1.0


class TestPercentile:
    def test_p25(self):
        assert _percentile([1, 2, 3, 4, 5], 0.25) == 2.0

    def test_p75(self):
        assert _percentile([1, 2, 3, 4, 5], 0.75) == 4.0

    def test_p0(self):
        assert _percentile([10, 20, 30], 0.0) == 10

    def test_p100(self):
        assert _percentile([10, 20, 30], 1.0) == 30

    def test_empty(self):
        assert _percentile([], 0.5) == 0.0

    def test_single(self):
        assert _percentile([42], 0.5) == 42.0


# ---------------------------------------------------------------------------
# build_counts
# ---------------------------------------------------------------------------

class TestBuildCounts:
    def test_basic_counts(self):
        entries = [
            _manifest_entry("r1", role="train"),
            _manifest_entry("r2", role="test"),
            _manifest_entry("r1", split="study_disjoint", role="train"),
        ]
        c = build_counts("closed_measured_pool", entries)
        assert c["total_assignments"] == 3
        assert c["unique_records"] == 2
        assert c["by_split"]["5utr_source_disjoint"]["train"] == 1
        assert c["by_split"]["5utr_source_disjoint"]["test"] == 1
        assert c["by_split"]["study_disjoint"]["train"] == 1
        assert c["by_region"]["5'UTR"] == 3
        assert c["by_exposure_class"]["TRAIN"] == 2
        assert c["by_exposure_class"]["EVAL_TEST"] == 1

    def test_region_accession_breakdown(self):
        entries = [
            _manifest_entry("r1", region="5'UTR", accession="GSE114002"),
            _manifest_entry("r2", region="3'UTR", accession="GSE200304"),
        ]
        c = build_counts("closed_measured_pool", entries)
        assert c["by_region"]["5'UTR"] == 1
        assert c["by_region"]["3'UTR"] == 1
        assert c["by_accession"]["GSE114002"] == 1
        assert c["by_accession"]["GSE200304"] == 1


# ---------------------------------------------------------------------------
# build_bias
# ---------------------------------------------------------------------------

class TestBuildBias:
    def test_lengths_and_gc(self):
        recs = {
            "r1": _rec("r1", src="ACGT", cand="ACGA", ed=1),
            "r2": _rec("r2", src="ACGTACGT", cand="ACGTACGA", ed=1),
        }
        entries = [_manifest_entry("r1"), _manifest_entry("r2")]
        b = build_bias("closed_measured_pool", entries, recs)
        assert b["source_length"]["n"] == 2
        assert b["source_length"]["min"] == 4
        assert b["source_length"]["max"] == 8
        assert b["candidate_length"]["n"] == 2
        assert b["gc_content_source"]["mean"] == 0.5

    def test_edit_distance_distribution(self):
        recs = {
            "r1": _rec("r1", ed=0),
            "r2": _rec("r2", ed=1),
            "r3": _rec("r3", ed=1),
        }
        entries = [_manifest_entry("r1"), _manifest_entry("r2"), _manifest_entry("r3")]
        b = build_bias("closed_measured_pool", entries, recs)
        assert b["edit_distance"]["distribution"]["0"] == 1
        assert b["edit_distance"]["distribution"]["1"] == 2

    def test_label_stats(self):
        recs = {
            "r1": _rec("r1", labels={"rl": 1.0}),
            "r2": _rec("r2", labels={"rl": 3.0}),
        }
        entries = [_manifest_entry("r1"), _manifest_entry("r2")]
        b = build_bias("closed_measured_pool", entries, recs)
        assert "rl" in b["label_stats"]
        assert b["label_stats"]["rl"]["mean"] == 2.0
        assert b["label_stats"]["rl"]["n"] == 2

    def test_deduplication(self):
        """Record appearing in multiple splits is counted once for bias."""
        recs = {"r1": _rec("r1", ed=1)}
        entries = [
            _manifest_entry("r1", split="5utr_source_disjoint"),
            _manifest_entry("r1", split="study_disjoint"),
        ]
        b = build_bias("closed_measured_pool", entries, recs)
        assert b["source_length"]["n"] == 1  # deduplicated

    def test_missing_record_skipped(self):
        entries = [_manifest_entry("ghost")]
        b = build_bias("closed_measured_pool", entries, {})
        assert b["source_length"]["n"] == 0

    def test_ins_del_sub_totals(self):
        recs = {
            "r1": {**_rec("r1"), "n_ins": 2, "n_del": 1, "n_sub": 0, "edit_distance": 3},
            "r2": {**_rec("r2"), "n_ins": 0, "n_del": 0, "n_sub": 1, "edit_distance": 1},
        }
        entries = [_manifest_entry("r1"), _manifest_entry("r2")]
        b = build_bias("closed_measured_pool", entries, recs)
        assert b["n_ins_total"] == 2
        assert b["n_del_total"] == 1
        assert b["n_sub_total"] == 1


# ---------------------------------------------------------------------------
# build_exposure
# ---------------------------------------------------------------------------

class TestBuildExposure:
    def test_basic(self):
        ledger = {
            "r1": _ledger("r1", exposure_status="unexposed"),
            "r2": _ledger("r2", exposure_status="historically_exposed", hist_exp=True),
        }
        entries = [_manifest_entry("r1"), _manifest_entry("r2")]
        e = build_exposure("closed_measured_pool", entries, ledger)
        assert e["by_exposure_status"]["unexposed"] == 1
        assert e["by_exposure_status"]["historically_exposed"] == 1
        assert e["historically_exposed_count"] == 1
        assert e["by_data_role"]["D_C"] == 2

    def test_deduplication(self):
        ledger = {"r1": _ledger("r1")}
        entries = [
            _manifest_entry("r1", split="s1"),
            _manifest_entry("r1", split="s2"),
        ]
        e = build_exposure("closed_measured_pool", entries, ledger)
        assert e["by_exposure_status"]["unexposed"] == 1  # deduplicated

    def test_labels_allowed_flags(self):
        ledger = {
            "r1": _ledger("r1"),
            "r2": {**_ledger("r2"),
                    "labels_allowed_for_new_training": False,
                    "labels_allowed_for_new_hyperparameter_selection": False},
        }
        entries = [_manifest_entry("r1"), _manifest_entry("r2")]
        e = build_exposure("closed_measured_pool", entries, ledger)
        assert e["labels_allowed_for_new_training"] == 1
        assert e["labels_allowed_for_new_hyperparameter_selection"] == 1


# ---------------------------------------------------------------------------
# build_allowed_claims
# ---------------------------------------------------------------------------

class TestAllowedClaims:
    def test_supported_claims_listed(self):
        ledger = {"r1": _ledger("r1")}
        entries = [_manifest_entry("r1")]
        ac = build_allowed_claims("closed_measured_pool", entries, ledger)
        assert "edit_effect" in ac["track_supported_claims"]
        assert "generation_grounding" in ac["track_supported_claims"]

    def test_effective_claims_intersection(self):
        # r1 allows edit_effect; r2 only allows generation_grounding
        ledger = {
            "r1": _ledger("r1", allowed=["edit_effect", "generation_grounding"]),
            "r2": _ledger("r2", allowed=["generation_grounding"]),
        }
        entries = [_manifest_entry("r1"), _manifest_entry("r2")]
        ac = build_allowed_claims("closed_measured_pool", entries, ledger)
        # closed_measured_pool supports edit_effect + generation_grounding
        # both are present in at least one record -> both effective
        assert set(ac["effective_supported_claims"]) == {
            "edit_effect", "generation_grounding"
        }

    def test_effective_claims_empty_when_ledger_disagrees(self):
        ledger = {"r1": _ledger("r1", allowed=["dense_pretraining"])}
        entries = [_manifest_entry("r1")]
        ac = build_allowed_claims("closed_measured_pool", entries, ledger)
        # closed_measured_pool supports edit_effect + generation_grounding,
        # but the ledger only allows dense_pretraining -> no effective claims
        assert ac["effective_supported_claims"] == []

    def test_per_claim_counts(self):
        ledger = {
            "r1": _ledger("r1", allowed=["edit_effect", "generation_grounding"]),
            "r2": _ledger("r2", allowed=["edit_effect"]),
        }
        entries = [_manifest_entry("r1"), _manifest_entry("r2")]
        ac = build_allowed_claims("closed_measured_pool", entries, ledger)
        assert ac["ledger_allowed_claims_per_claim_counts"]["edit_effect"] == 2
        assert ac["ledger_allowed_claims_per_claim_counts"]["generation_grounding"] == 1

    def test_forbidden_claims_counted(self):
        ledger = {"r1": _ledger("r1", forbidden=["intervention_claim"])}
        entries = [_manifest_entry("r1")]
        ac = build_allowed_claims("closed_measured_pool", entries, ledger)
        assert ac["ledger_forbidden_claims_per_claim_counts"]["intervention_claim"] == 1

    def test_heldout_generative_supports_denoising(self):
        ledger = {"r1": _ledger("r1")}
        entries = [_manifest_entry("r1")]
        ac = build_allowed_claims("heldout_generative", entries, ledger)
        assert "generative_denoising" in ac["track_supported_claims"]


# ---------------------------------------------------------------------------
# build_track_data_card (full card)
# ---------------------------------------------------------------------------

class TestBuildTrackDataCard:
    def test_full_card_has_all_dimensions(self):
        recs = {"r1": _rec("r1"), "r2": _rec("r2")}
        ledger = {"r1": _ledger("r1"), "r2": _ledger("r2")}
        entries = [
            _manifest_entry("r1", role="train"),
            _manifest_entry("r2", role="test"),
        ]
        card = build_track_data_card(
            "closed_measured_pool", entries, recs, ledger
        )
        assert card["track"] == "closed_measured_pool"
        for sec in ["counts", "bias", "exposure", "allowed_claims",
                    "unsupported_capabilities"]:
            assert sec in card
            assert card[sec]  # non-empty
        assert card["n_assignments"] == 2
        assert len(card["unsupported_capabilities"]) >= 1

    def test_filters_none_roles(self):
        """Entries where the track role is 'none' are excluded."""
        recs = {"r1": _rec("r1")}
        ledger = {"r1": _ledger("r1")}
        entries = [_manifest_entry("r1")]
        # Set closed_measured_pool to none
        entries[0]["track_roles"]["closed_measured_pool"] = "none"
        card = build_track_data_card(
            "closed_measured_pool", entries, recs, ledger
        )
        assert card["n_assignments"] == 0


# ---------------------------------------------------------------------------
# TRACK_SUPPORTED_CLAIMS / UNSUPPORTED_CAPABILITIES sanity
# ---------------------------------------------------------------------------

class TestTrackSemantics:
    def test_all_tracks_have_supported_claims(self):
        for track in EVAL_TRACKS:
            assert track in TRACK_SUPPORTED_CLAIMS
            assert len(TRACK_SUPPORTED_CLAIMS[track]) >= 2

    def test_all_tracks_have_unsupported_capabilities(self):
        for track in EVAL_TRACKS:
            assert track in UNSUPPORTED_CAPABILITIES
            assert len(UNSUPPORTED_CAPABILITIES[track]) >= 1
            for cap in UNSUPPORTED_CAPABILITIES[track]:
                assert "capability" in cap
                assert "reason" in cap

    def test_supported_claims_in_schema_pool_or_ledger_ext(self):
        # supported claims should be either in ALLOWED_CLAIMS_POOL or a known
        # ledger claim; all current ones are in the pool
        for track, claims in TRACK_SUPPORTED_CLAIMS.items():
            for c in claims:
                assert c in ALLOWED_CLAIMS_POOL, f"{c} not in ALLOWED_CLAIMS_POOL"


# ---------------------------------------------------------------------------
# run_data_card (integration)
# ---------------------------------------------------------------------------

class TestRunDataCard:
    def _setup(self, tmp_path):
        recs = {"r1": _rec("r1"), "r2": _rec("r2")}
        canonical = tmp_path / "canonical.jsonl"
        with open(canonical, "w") as f:
            for r in recs.values():
                f.write(json.dumps(r) + "\n")
        ledger = tmp_path / "ledger.jsonl"
        with open(ledger, "w") as f:
            for rid in recs:
                f.write(json.dumps(_ledger(rid)) + "\n")
        manifest = tmp_path / "manifest.jsonl"
        with open(manifest, "w") as f:
            f.write(json.dumps(_manifest_entry("r1", role="train")) + "\n")
            f.write(json.dumps(_manifest_entry("r2", role="test")) + "\n")
        return str(manifest), str(canonical), str(ledger)

    def test_completeness_pass(self, tmp_path):
        m, c, l = self._setup(tmp_path)
        report = run_data_card(m, c, l)
        assert report["overall_pass"] is True
        assert report["acceptance"]["data_card_complete_for_all_tracks"] is True
        assert set(report["tracks_documented"]) == set(EVAL_TRACKS)
        for track in EVAL_TRACKS:
            assert track in report["track_cards"]
            card = report["track_cards"][track]
            for sec in ["counts", "bias", "exposure", "allowed_claims",
                        "unsupported_capabilities"]:
                assert card[sec]

    def test_all_tracks_present(self, tmp_path):
        m, c, l = self._setup(tmp_path)
        report = run_data_card(m, c, l)
        assert len(report["track_cards"]) == 3
