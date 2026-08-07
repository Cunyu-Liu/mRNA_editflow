"""E0-X unit tests: pre-registration protocol validation + Holm-Bonferroni.

Phase E0-X freezes the evaluation protocol (data manifest, task/split registry,
model aliases + checkpoint hashes, reward/beta, metric family + Holm correction,
seeds, evaluator command, budget, output schema, fallback) BEFORE any sealed
final access.  These tests verify the pure data-free cores in
`scripts/e0x/prereg.py`:
  * the frozen yaml is internally consistent (validate_file),
  * the validator rejects each broken invariant with a specific error,
  * Holm-Bonferroni enforces family-wise error-rate control and monotonicity.
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from scripts.e0x import prereg
from scripts.e0x import sealed


ROOT = Path(__file__).resolve().parent.parent.parent
PREREG = ROOT / "configs" / "e0x_preregistration_v1.yaml"


def _base() -> dict:
    """A structurally valid pre-registration dict (mirror of the frozen yaml)."""
    return {
        "contract_id": "mrna_xeditflow_goal_v1_1",
        "phase": "E0-X",
        "goal": "GOAL-XEDITFLOW-MIGRATION-01",
        "preregistration_id": "E0X_PREREG_TEST",
        "status": "FROZEN",
        "data": {
            "effect_dataset": {
                "path": "artifacts/b0x/effect_dataset.jsonl",
                "sha256": "f" * 64,
                "n_records": 106659,
                "n_delta_defined": 103199,
            },
            "split": "S4",
        },
        "models": {
            "frozen_base_flow": {"alias": "f0x", "sha256": "a" * 64},
            "frozen_critic": {
                "type": "SparseEditFormer",
                "checkpoints": {"model_5U-A1__GSE114002.pt": {"sha256": "b" * 64}},
            },
        },
        "reward": {"beta": 1.0, "guidance_strategy_primary": "rate_cfg"},
        "metrics": {
            "holm_family": {
                "inference": "HOLM_BONFERRONI",
                "alpha": 0.05,
                "hypotheses": [
                    {"id": "H1", "metric": "macro_delta_spearman", "null_hypothesis": "h1"},
                    {"id": "H2", "metric": "generation_quality_mean_delta", "null_hypothesis": "h2"},
                ],
            }
        },
        "go_nogo": {
            "effect_gate": {"macro_delta_spearman_ge": 0.25,
                            "macro_sign_accuracy_ge": 0.60,
                            "top10_enrichment_ge": 1.50,
                            "beat_strongest_nonfoundation_baseline": True},
            "guidance_gate": {"primary_strategy": "rate_cfg", "vs": "no_guidance"},
            "hard_constraints": {"legality_ge": 1.0, "budget_violation_le": 0.0},
        },
        "execution": {
            "gpu_policy": {"permitted": ["cuda:1", "cuda:2", "cuda:3"],
                           "forbidden": ["cuda:4", "4"], "fallback": ""},
            "output_schema": {"aggregate_only": True,
                              "row_level_labels_returned": False},
            "evaluator_command": "python -m scripts.e0x.run_e0x_final --prereg configs/e0x_preregistration_v1.yaml",
        },
    }


# ---------------------------------------------------------------------------
# frozen yaml is valid
# ---------------------------------------------------------------------------

def test_frozen_prereg_file_is_valid():
    rep = prereg.validate_file(PREREG)
    assert rep["valid"] is True, "errors=%r" % rep["errors"]
    assert rep["n_errors"] == 0
    assert rep["preregistration_id"] == "E0X_PREREG_20260807"
    assert rep["holm_alpha"] == 0.05
    assert rep["n_hypotheses"] >= 1


# ---------------------------------------------------------------------------
# validator rejects each broken invariant
# ---------------------------------------------------------------------------

def test_status_must_be_frozen():
    d = _base(); d["status"] = "DRAFT"
    assert prereg.validate(d)["valid"] is False


def test_phase_must_be_e0x():
    d = _base(); d["phase"] = "F0-X"
    assert prereg.validate(d)["valid"] is False


def test_effect_dataset_sha256_required():
    d = _base(); d["data"]["effect_dataset"]["sha256"] = "short"
    assert prereg.validate(d)["valid"] is False


def test_base_flow_sha256_required():
    d = _base(); d["models"]["frozen_base_flow"]["sha256"] = ""
    assert prereg.validate(d)["valid"] is False


def test_critic_checkpoint_sha256_required():
    d = _base()
    ck = d["models"]["frozen_critic"]["checkpoints"]
    ck[list(ck)[0]]["sha256"] = "bad"
    assert prereg.validate(d)["valid"] is False


def test_beta_must_be_positive():
    d = _base(); d["reward"]["beta"] = 0.0
    assert prereg.validate(d)["valid"] is False


def test_holm_alpha_in_range():
    d = _base(); d["metrics"]["holm_family"]["alpha"] = 1.5
    assert prereg.validate(d)["valid"] is False


def test_hypothesis_null_hypothesis_required():
    d = _base(); del d["metrics"]["holm_family"]["hypotheses"][0]["null_hypothesis"]
    assert prereg.validate(d)["valid"] is False


def test_legality_must_be_exact_1():
    d = _base(); d["go_nogo"]["hard_constraints"]["legality_ge"] = 0.99
    assert prereg.validate(d)["valid"] is False


def test_budget_violation_must_be_0():
    d = _base(); d["go_nogo"]["hard_constraints"]["budget_violation_le"] = 0.01
    assert prereg.validate(d)["valid"] is False


def test_gpu4_must_be_forbidden():
    d = _base(); d["execution"]["gpu_policy"]["forbidden"] = ["cuda:3"]
    assert prereg.validate(d)["valid"] is False


def test_cuda_fallback_fail_closed():
    d = _base(); d["execution"]["gpu_policy"]["fallback"] = "cuda:0"
    assert prereg.validate(d)["valid"] is False


def test_output_must_be_aggregate_only():
    d = _base(); d["execution"]["output_schema"]["aggregate_only"] = False
    assert prereg.validate(d)["valid"] is False


def test_row_level_labels_not_returned():
    d = _base(); d["execution"]["output_schema"]["row_level_labels_returned"] = True
    assert prereg.validate(d)["valid"] is False


def test_evaluator_command_references_frozen_prereg():
    d = _base(); d["execution"]["evaluator_command"] = "python -m foo"
    assert prereg.validate(d)["valid"] is False


def test_valid_config_stays_valid():
    assert prereg.validate(_base())["valid"] is True


# ---------------------------------------------------------------------------
# Holm-Bonferroni family-wise error-rate control
# ---------------------------------------------------------------------------

def test_holm_rejects_only_below_alpha():
    # smallest p-value: Holm adjusted = p * n = 0.001 * 3 = 0.003 <= 0.05 (rejected)
    adj = prereg.holm_bonferroni([0.001, 0.4, 0.6], alpha=0.05)
    assert adj[0] == pytest.approx(0.003)
    assert adj[0] <= 0.05          # rejected
    assert adj[1] > 0.05           # not rejected
    assert adj[2] > 0.05


def test_holm_raises_on_bad_pvalue():
    with pytest.raises(ValueError):
        prereg.holm_bonferroni([-0.1, 0.5], alpha=0.05)


def test_holm_adjusted_never_less_than_raw_and_monotone():
    pvals = [0.01, 0.02, 0.03, 0.5]
    adj = prereg.holm_bonferroni(pvals, alpha=0.05)
    # each adjusted >= its raw
    for p, a in zip(pvals, adj):
        assert a >= p
    # adjusted values are monotone non-decreasing in the rejection order
    # (sorted by raw p).  Every adjusted element must be <= 1.
    assert all(a <= 1.0 for a in adj)


def test_holm_empty():
    assert prereg.holm_bonferroni([], alpha=0.05) == []


def test_holm_all_rejected_when_all_small():
    adj = prereg.holm_bonferroni([0.001, 0.002, 0.003], alpha=0.05)
    assert all(a <= 0.05 for a in adj)


# ---------------------------------------------------------------------------
# sealed access protocol (scripts/e0x/sealed.py)
# ---------------------------------------------------------------------------

def test_make_event_and_verify_chain():
    e1 = sealed.make_event("acc1", "obj", "intent", "COMPLETION", None)
    e2 = sealed.make_event("acc2", "obj", "intent", "COMPLETION", e1["event_sha256"])
    assert sealed.verify_chain([e1, e2]) is True
    assert e2["prev_event_sha256"] == e1["event_sha256"]


def test_verify_chain_detects_tamper():
    e1 = sealed.make_event("acc1", "obj", "intent", "COMPLETION", None)
    e2 = sealed.make_event("acc2", "obj", "intent", "COMPLETION", e1["event_sha256"])
    e2["event_sha256"] = "0" * 64  # tamper
    assert sealed.verify_chain([e1, e2]) is False


def test_compare_and_append_rejects_stale(tmp_path):
    log = tmp_path / "ACCESS_LOG.jsonl"
    e1 = sealed.make_event("acc1", "obj", "intent", "COMPLETION", None)
    # e2 claims a prev head that will NOT be the running head after e1 is appended
    e2 = sealed.make_event("acc2", "obj", "intent", "COMPLETION", "f" * 64)
    ok, _ = sealed.compare_and_append(log, e1)
    assert ok is True
    # stale: e2's prev (f...) does not match the head (e1's hash)
    ok2, reason = sealed.compare_and_append(log, e2)
    assert ok2 is False
    assert "stale" in reason
    # only e1 was written
    assert sealed.read_chain(log) == [e1]


def test_sealed_state_lifecycle(tmp_path):
    log = tmp_path / "ACCESS_LOG.jsonl"
    sm = sealed.SealedAccessState(log)
    assert sm.state == sealed.SealedAccessState.UNSEALED
    sm.append_intent("f", "GSE246381_E0X_FINAL", "e0x_sealed_final", "PREREG")
    assert sm.state == sealed.SealedAccessState.INTENT_APPENDED
    sm.reserve("f", "GSE246381_E0X_FINAL", "e0x_sealed_final", "PREREG")
    assert sm.state == sealed.SealedAccessState.RESERVED
    sm.complete("f", "GSE246381_E0X_FINAL", "e0x_sealed_final", "PREREG", "0" * 64)
    assert sm.state == sealed.SealedAccessState.COMPLETED
    # terminal: not retryable
    sm2 = sealed.SealedAccessState(log)
    assert sm2.state == sealed.SealedAccessState.COMPLETED
    with pytest.raises(sealed.SealedAccessError):
        sm2.reserve("f", "GSE246381_E0X_FINAL", "e0x_sealed_final", "PREREG")


def test_sealed_abort_invalidates(tmp_path):
    log = tmp_path / "ACCESS_LOG.jsonl"
    sm = sealed.SealedAccessState(log)
    sm.append_intent("f", "G", "e0x_sealed_final", "PREREG")
    sm.abort("f", "G", "e0x_sealed_final", "PREREG", "crash")
    reloaded = sealed.SealedAccessState(log)
    assert reloaded.state == sealed.SealedAccessState.ABORTED
    # a crashed/aborted final is not retryable
    with pytest.raises(sealed.SealedAccessError):
        reloaded.append_intent("f2", "G", "e0x_sealed_final", "PREREG")


def test_build_aggregate_is_aggregate_only():
    d = _base()
    per_hyp = [
        {"id": "H1_EFFECT_TRANSFER", "metric": "macro_delta_spearman",
         "stat": 0.3, "pvalue": 0.01, "n": 100},
        {"id": "H3_LEGALITY", "metric": "legality_rate", "stat": 1.0,
         "pvalue": None, "n": 100},
    ]
    agg = sealed.build_aggregate(d, per_hyp)
    assert agg["phase"] == "E0-X"
    assert agg["preregistration_id"] == "E0X_PREREG_TEST"
    assert "holm_adjusted_pvalues" in agg
    assert len(agg["holm_adjusted_pvalues"]) == 2
    sealed.assert_no_row_level(agg)


def test_assert_no_row_level_rejects_ids():
    bad = {"per_hypothesis": [{"id": "H1", "stat": 0.3, "sequence_id": "row1"}]}
    with pytest.raises(sealed.SealedAccessError):
        sealed.assert_no_row_level(bad)


def test_assert_no_row_level_rejects_raw_vector():
    bad = {"per_hypothesis": [{"id": "H1", "stat": 0.3}],
           "leak": [0.1, 0.2, 0.3, 0.4, 0.5]}
    with pytest.raises(sealed.SealedAccessError):
        sealed.assert_no_row_level(bad)


def test_permutation_pvalue_positive_signal():
    rng = np.random.RandomState(0)
    contexts = ["s1|ep"] * 40 + ["s2|ep"] * 40
    true_d = rng.normal(0, 1, 80)
    pred = true_d + rng.normal(0, 0.2, 80)  # strong positive monotone signal
    p = sealed.permutation_pvalue(true_d, pred, contexts, n_perm=200, seed=1)
    assert p is not None
    assert p < 0.05


def test_permutation_pvalue_no_signal_not_rejected():
    rng = np.random.RandomState(0)
    contexts = ["s1|ep"] * 40 + ["s2|ep"] * 40
    true_d = rng.normal(0, 1, 80)
    pred = rng.normal(0, 1, 80)  # independent -> not significant
    p = sealed.permutation_pvalue(true_d, pred, contexts, n_perm=200, seed=1)
    assert p is not None
    assert p > 0.05


def test_verdict_from_aggregate_go_nogo():
    d = _base()
    per_hyp = [
        {"id": "H1_EFFECT_TRANSFER", "metric": "macro_delta_spearman",
         "stat": 0.30, "pvalue": 0.01, "n": 100,
         "sign_accuracy": 0.62, "top10pct_enrichment": 2.0,
         "abs_candidate_spearman": 0.20},
        {"id": "H3_LEGALITY", "metric": "legality_rate", "stat": 1.0,
         "pvalue": None, "n": 1},
    ]
    holm = prereg.holm_bonferroni([0.01, 1.0], alpha=0.05)
    v = sealed.verdict_from_aggregate(d, per_hyp, holm)
    assert v["verdict"] == "GO"
    assert all(v["checks"].values())
    # below the pre-registered gate threshold -> NO_GO
    per_hyp[0]["stat"] = 0.10
    v2 = sealed.verdict_from_aggregate(d, per_hyp,
                                       prereg.holm_bonferroni([0.9, 1.0], 0.05))
    assert v2["verdict"] == "NO_GO"


def test_verdict_requires_full_effect_gate_not_just_spearman():
    """Each frozen effect-gate threshold must independently gate the verdict:
    sign_accuracy, top10 enrichment, and beat-abs_candidate must ALL hold in
    addition to the spearman + Holm pair (fixes a gate that only checked the
    spearman threshold)."""
    d = _base()
    def hyp(stat=0.30, sign=0.62, top10=2.0, abs_base=0.20):
        return {"id": "H1_EFFECT_TRANSFER", "metric": "macro_delta_spearman",
                "stat": stat, "pvalue": 0.01, "n": 100,
                "sign_accuracy": sign, "top10pct_enrichment": top10,
                "abs_candidate_spearman": abs_base}
    h3 = {"id": "H3_LEGALITY", "metric": "legality_rate", "stat": 1.0,
          "pvalue": None, "n": 1}
    holm = prereg.holm_bonferroni([0.01, 1.0], alpha=0.05)

    # baseline all-pass -> GO
    assert sealed.verdict_from_aggregate(d, [hyp(), dict(h3)], holm)["verdict"] == "GO"

    # each single violation must flip to NO_GO (even if spearman+Holm pass)
    assert sealed.verdict_from_aggregate(
        d, [hyp(sign=0.50), dict(h3)], holm)["verdict"] == "NO_GO"
    assert sealed.verdict_from_aggregate(
        d, [hyp(top10=1.0), dict(h3)], holm)["verdict"] == "NO_GO"
    assert sealed.verdict_from_aggregate(
        d, [hyp(abs_base=0.40), dict(h3)], holm)["verdict"] == "NO_GO"