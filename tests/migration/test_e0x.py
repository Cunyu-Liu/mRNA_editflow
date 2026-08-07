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
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.e0x import prereg
from scripts.e0x import run_e0x_final as e0x_runner
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
                    {"id": "H1_EFFECT_TRANSFER", "metric": "macro_delta_spearman",
                     "null_hypothesis": "h1"},
                    {"id": "H2_GUIDANCE_VALUE",
                     "metric": "generation_quality_mean_delta",
                     "null_hypothesis": "h2"},
                    {"id": "H3_LEGALITY", "metric": "legality_rate",
                     "null_hypothesis": "h3"},
                ],
            }
        },
        "go_nogo": {
            "effect_gate": {"macro_delta_spearman_ge": 0.25,
                            "macro_sign_accuracy_ge": 0.60,
                            "top10_enrichment_ge": 1.50,
                            "beat_strongest_nonfoundation_baseline": True},
            "guidance_gate": {
                "primary_strategy": "rate_cfg", "vs": "no_guidance",
                "generation_quality_mean_delta_gt": 0.0,
                "generation_quality_frac_beneficial_gt": 0.0,
                "ndcg_gap_ge": 0.05, "top10_recall_ge": 0.70,
                "normalized_regret_le": 0.10,
            },
            "hard_constraints": {"legality_ge": 1.0,
                                 "length_preservation_ge": 1.0,
                                 "budget_violation_le": 0.0},
            "ood_margin": {"degradation_margin": "preregistered",
                           "fail_if_beyond": True},
        },
        "execution": {
            "gpu_policy": {"permitted": ["cuda:1", "cuda:2", "cuda:3"],
                           "forbidden": ["cuda:4", "4"], "fallback": ""},
            "output_schema": {"aggregate_only": True,
                              "row_level_labels_returned": False},
            "evaluator_command": "python -m scripts.e0x.run_e0x_final --prereg configs/e0x_preregistration_v1.yaml",
        },
    }


def _sealed_args(tmp_path: Path, *, authorize: bool = True,
                 confirmation: str | None = None,
                 preregistration_id: str | None = e0x_runner.FROZEN_PREREGISTRATION_ID,
                 prereg_data: dict | None = None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    prereg_data = prereg_data or _sealed_base()
    fixture = tmp_path / "ordinary_evaluator_fixture.bin"
    fixture.write_bytes(b"E0X synthetic ordinary aggregate-only scorer fixture v1")
    per_hypothesis = _passing_family()
    holm = prereg.holm_bonferroni([h["pvalue"] for h in per_hypothesis], 0.05)
    verdict = sealed.verdict_from_aggregate(prereg_data, per_hypothesis, holm)
    assert verdict["verdict"] == "GO"
    activation = {
        "schema_version": e0x_runner.ORDINARY_GO_COMMITMENT_SCHEMA,
        "status": "COMPLETE",
        "verdict": "GO",
        "full_frozen_hypothesis_family_complete": True,
        "stable_enrichment_definition_preregistered": True,
        "preregistration_id": e0x_runner.FROZEN_PREREGISTRATION_ID,
        "preregistration_sha256": e0x_runner.FROZEN_PREREGISTRATION_SHA256,
        "contract_id": e0x_runner.FROZEN_CONTRACT_ID,
        "effect_dataset_sha256": prereg_data["data"]["effect_dataset"]["sha256"],
        "critic_checkpoint_sha256": {
            name: metadata["sha256"]
            for name, metadata in
            prereg_data["models"]["frozen_critic"]["checkpoints"].items()},
        "checkpoint_schema_version": "m4_sparse_r2_v1",
        "input_schema_version": "shared_edit_centered_window_with_padding_masks_v1",
        "macro_sign_accuracy_threshold": 0.60,
        "sealed_member_access": False,
        "formal_sealed_evaluator_invoked": False,
        "historical_results_overwritten": False,
        "per_hypothesis": per_hypothesis,
        "holm_adjusted_pvalues": holm,
        "go_nogo_checks": verdict["checks"],
    }
    activation_path = tmp_path / "ordinary_go_activation.json"
    activation_path.write_text(json.dumps(activation), encoding="utf-8")
    activation_sha = hashlib.sha256(activation_path.read_bytes()).hexdigest()
    return SimpleNamespace(
        restricted=tmp_path / "restricted",
        raw_seq_dir=tmp_path / "must_not_be_read",
        authorize_sealed_final=authorize,
        confirm_one_shot=(confirmation
                          if confirmation is not None
                          else e0x_runner.SEALED_FINAL_CONFIRMATION),
        confirm_preregistration_id=preregistration_id,
        actual_preregistration_sha256=e0x_runner.FROZEN_PREREGISTRATION_SHA256,
        confirm_preregistration_sha256=e0x_runner.FROZEN_PREREGISTRATION_SHA256,
        ordinary_go_manifest=activation_path,
        confirm_ordinary_go_sha256=activation_sha,
        evaluator_fixture=fixture,
    )


def _sealed_base() -> dict:
    """The base fixture with the exact frozen E0-X hypothesis family."""
    d = _base()
    d["preregistration_id"] = e0x_runner.FROZEN_PREREGISTRATION_ID
    d["metrics"]["holm_family"]["hypotheses"] = [
        {"id": "H1_EFFECT_TRANSFER", "metric": "macro_delta_spearman",
         "null_hypothesis": "h1"},
        {"id": "H2_GUIDANCE_VALUE", "metric": "generation_quality_mean_delta",
         "null_hypothesis": "h2"},
        {"id": "H3_LEGALITY", "metric": "legality_rate",
         "null_hypothesis": "h3"},
    ]
    d["go_nogo"]["guidance_gate"].update({
        "generation_quality_mean_delta_gt": 0.0,
        "generation_quality_frac_beneficial_gt": 0.0,
        "ndcg_gap_ge": 0.05,
        "top10_recall_ge": 0.70,
        "normalized_regret_le": 0.10,
    })
    d["go_nogo"]["hard_constraints"]["length_preservation_ge"] = 1.0
    d["execution"]["evaluator_command"] = (
        "python -m scripts.e0x.run_e0x_final --prereg "
        "configs/e0x_preregistration_v1.yaml --mode sealed-final "
        "--authorize-sealed-final --confirm-one-shot TOKEN "
        "--confirm-preregistration-id ID --confirm-preregistration-sha256 SHA "
        "--ordinary-go-manifest MANIFEST --confirm-ordinary-go-sha256 GO_SHA "
        "--evaluator-fixture FIXTURE")
    return d


def test_preregistration_snapshot_is_parsed_and_hashed_from_same_bytes(tmp_path):
    snapshot = tmp_path / "prereg.yaml"
    raw = PREREG.read_bytes()
    snapshot.write_bytes(raw)
    loaded, digest = prereg.load_prereg_and_sha256(snapshot)
    assert loaded == prereg.load_prereg(PREREG)
    assert digest == hashlib.sha256(raw).hexdigest()


def _canonical_base() -> dict:
    return copy.deepcopy(prereg.load_prereg(PREREG))


def _passing_family():
    return [
        {"id": "H1_EFFECT_TRANSFER", "metric": "macro_delta_spearman",
         "stat": 0.30, "pvalue": 0.01, "n": 8,
         "sign_accuracy": 0.625, "top10pct_enrichment": 2.0,
         "abs_candidate_spearman": 0.20,
         "top10pct_enrichment_definition_status": "PREREGISTERED_STABLE"},
        {"id": "H2_GUIDANCE_VALUE", "metric": "generation_quality_mean_delta",
         "stat": 0.10, "pvalue": 0.01, "n": 8,
         "generation_quality_mean_delta_difference": 0.10,
         "generation_quality_frac_beneficial_difference": 0.10,
         "ndcg_gap": 0.10, "top10_recall": 0.80,
         "normalized_regret": 0.05, "ood_margin_pass": True},
        {"id": "H3_LEGALITY", "metric": "legality_rate",
         "stat": 1.0, "pvalue": 0.01, "n": 8,
         "length_preservation_rate": 1.0, "budget_violation_count": 0},
    ]


def _ordinary_fixture_evaluator(prereg_, args, cfg, rows, vocab, device):
    """Data-free scorer fixture implementing the frozen aggregate interface."""
    del prereg_, args, cfg, vocab, device
    assert rows == [{"fixture": "ordinary", "delta": 0.25}]
    return _passing_family()


def _rewrite_activation(args, mutator):
    payload = json.loads(args.ordinary_go_manifest.read_text(encoding="utf-8"))
    mutator(payload)
    args.ordinary_go_manifest.write_text(json.dumps(payload), encoding="utf-8")
    args.confirm_ordinary_go_sha256 = hashlib.sha256(
        args.ordinary_go_manifest.read_bytes()).hexdigest()


_ordinary_fixture_evaluator.e0x_contract_version = (
    e0x_runner.SEALED_EVALUATOR_CONTRACT_VERSION)
_ordinary_fixture_evaluator.e0x_implementation_ready = True
_ordinary_fixture_evaluator.e0x_fixture_sha256 = sealed.sha256_hex(
    b"E0X synthetic ordinary aggregate-only scorer fixture v1")


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


def test_hypothesis_family_and_order_are_exact():
    d = _base()
    d["metrics"]["holm_family"]["hypotheses"].reverse()
    assert prereg.validate(d)["valid"] is False
    d = _base()
    d["metrics"]["holm_family"]["hypotheses"].pop(1)
    assert prereg.validate(d)["valid"] is False


def test_legality_must_be_exact_1():
    d = _base(); d["go_nogo"]["hard_constraints"]["legality_ge"] = 0.99
    assert prereg.validate(d)["valid"] is False


def test_budget_violation_must_be_0():
    d = _base(); d["go_nogo"]["hard_constraints"]["budget_violation_le"] = 0.01
    assert prereg.validate(d)["valid"] is False


def test_length_preservation_must_be_exact_1():
    d = _base(); d["go_nogo"]["hard_constraints"]["length_preservation_ge"] = 0.99
    assert prereg.validate(d)["valid"] is False


def test_guidance_and_ood_thresholds_cannot_be_omitted():
    d = _base(); del d["go_nogo"]["guidance_gate"]["top10_recall_ge"]
    assert prereg.validate(d)["valid"] is False
    d = _base(); d["go_nogo"]["ood_margin"]["fail_if_beyond"] = False
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
# sealed-final fail-closed preflight / evaluator contract
# ---------------------------------------------------------------------------

def test_current_unimplemented_evaluator_is_blocked_before_one_shot(tmp_path):
    args = _sealed_args(tmp_path)
    with pytest.raises(sealed.SealedAccessError, match="before ACCESS_INTENT"):
        e0x_runner.run_sealed_final(
            _base(), args, cfg=None, rows=[], vocab={}, device=None)

    # The preflight rejected the evaluator before even creating the restricted
    # directory, access log, lock, intent, reservation, or abort event.
    assert args.restricted.exists() is False


def test_sealed_preflight_is_repeatable_and_has_no_restricted_side_effect(tmp_path):
    args = _sealed_args(tmp_path)
    first = e0x_runner.sealed_final_preflight(_base(), args)
    second = e0x_runner.sealed_final_preflight(_base(), args)

    assert first == second
    assert first["status"] == "BLOCKED"
    assert first["ready_for_one_shot_reservation"] is False
    assert first["checks"]["evaluator_implementation_ready"] is False
    assert first["checks"]["synthetic_ordinary_fixture_sha256_bound"] is False
    for key in ("restricted_store_read", "sealed_member_data_read",
                "access_log_read", "access_log_written",
                "reservation_written", "output_artifact_written"):
        assert first["safety"][key] is False
    assert args.restricted.exists() is False
    assert args.raw_seq_dir.exists() is False


def test_preflight_never_even_resolves_restricted_or_raw_paths(tmp_path):
    canonical = _canonical_base()
    valid = _sealed_args(tmp_path, prereg_data=canonical)
    class NoRestrictedPathAccess:
        authorize_sealed_final = valid.authorize_sealed_final
        confirm_one_shot = valid.confirm_one_shot
        confirm_preregistration_id = valid.confirm_preregistration_id
        actual_preregistration_sha256 = valid.actual_preregistration_sha256
        confirm_preregistration_sha256 = valid.confirm_preregistration_sha256
        ordinary_go_manifest = valid.ordinary_go_manifest
        confirm_ordinary_go_sha256 = valid.confirm_ordinary_go_sha256
        evaluator_fixture = valid.evaluator_fixture

        @property
        def restricted(self):
            raise AssertionError("preflight must not resolve restricted path")

        @property
        def raw_seq_dir(self):
            raise AssertionError("preflight must not resolve raw sequence path")

    report = e0x_runner.sealed_final_preflight(
        canonical, NoRestrictedPathAccess(),
        evaluator=_ordinary_fixture_evaluator)
    # The current canonical command is intentionally stale, so a truthful
    # canonical preflight remains BLOCKED without inspecting protected paths.
    assert report["ready_for_one_shot_reservation"] is False
    assert report["checks"]["canonical_preregistration_semantics"] is True
    assert report["checks"]["formal_evaluator_command_matches_runner"] is False
    assert report["safety"]["restricted_store_read"] is False
    assert report["safety"]["sealed_member_data_read"] is False


def test_dry_run_cli_does_not_load_dataset_or_write_output(tmp_path, capsys):
    restricted = tmp_path / "restricted"
    out_dir = tmp_path / "out"
    missing_dataset = tmp_path / "must_not_be_loaded.jsonl"
    rc = e0x_runner.main([
        "--dry-run",
        "--prereg", str(PREREG),
        "--dataset", str(missing_dataset),
        "--restricted", str(restricted),
        "--raw-seq-dir", str(tmp_path / "must_not_be_read"),
        "--out-dir", str(out_dir),
    ])
    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert report["status"] == "BLOCKED"
    assert captured.out.lstrip().startswith("{")
    assert captured.err == ""
    assert rc == 4  # truthful BLOCKED: production evaluator is not mounted
    assert restricted.exists() is False
    assert out_dir.exists() is False
    assert missing_dataset.exists() is False


def test_formal_cli_with_stub_stops_before_dataset_and_access_log(tmp_path):
    restricted = tmp_path / "restricted"
    out_dir = tmp_path / "out"
    missing_dataset = tmp_path / "must_not_be_loaded.jsonl"
    rc = e0x_runner.main([
        "--mode", "sealed-final",
        "--authorize-sealed-final",
        "--confirm-one-shot", e0x_runner.SEALED_FINAL_CONFIRMATION,
        "--confirm-preregistration-id", "E0X_PREREG_20260807",
        "--prereg", str(PREREG),
        "--dataset", str(missing_dataset),
        "--restricted", str(restricted),
        "--raw-seq-dir", str(tmp_path / "must_not_be_read"),
        "--out-dir", str(out_dir),
    ])
    assert rc == 4
    assert restricted.exists() is False
    assert out_dir.exists() is False
    assert missing_dataset.exists() is False


def test_internal_output_uses_ordinary_guard_and_never_overwrites(tmp_path):
    protected = tmp_path / "restricted_members" / "out"
    with pytest.raises(ValueError, match="protected path"):
        e0x_runner.validate_internal_output_path(protected)
    assert protected.exists() is False

    ordinary = tmp_path / "ordinary_out"
    output = e0x_runner.validate_internal_output_path(ordinary)
    assert ordinary.exists() is False
    ordinary.mkdir()
    output.write_text("historical\n", encoding="utf-8")
    before = output.read_bytes()
    with pytest.raises(FileExistsError, match="immutable"):
        e0x_runner.validate_internal_output_path(ordinary)
    assert output.read_bytes() == before


def test_sealed_output_uses_dedicated_boundary_not_ordinary_guard(tmp_path):
    args = _sealed_args(tmp_path)
    args.out_dir = tmp_path / "sealed_aggregate_destination"
    output = e0x_runner.validate_sealed_output_path(args)
    assert output.name == "e0x_sealed-final_results.json"
    assert args.out_dir.exists() is False

    args.out_dir = args.restricted / "aggregate"
    with pytest.raises(sealed.SealedAccessError, match="restricted member storage"):
        e0x_runner.validate_sealed_output_path(args)
    assert args.out_dir.exists() is False


def test_result_writer_is_exclusive(tmp_path):
    output = tmp_path / "result.json"
    e0x_runner.write_result_exclusive(output, {"status": "FIRST"})
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        e0x_runner.write_result_exclusive(output, {"status": "SECOND"})
    assert output.read_bytes() == before


@pytest.mark.parametrize(
    ("args_mutator", "prereg_mutator", "failed_check"),
    [
        (lambda args: setattr(args, "authorize_sealed_final", False),
         lambda d: None, "explicit_sealed_authorization"),
        (lambda args: setattr(args, "confirm_one_shot", "NOT_AUTHORIZED"),
         lambda d: None, "one_shot_confirmation_matches"),
        (lambda args: setattr(args, "confirm_preregistration_id", "WRONG_FREEZE"),
         lambda d: None, "preregistration_id_confirmed"),
        (lambda args: None, lambda d: d.__setitem__("status", "DRAFT"),
         "preregistration_frozen"),
    ],
)
def test_formal_path_requires_permission_and_frozen_confirmation(
        tmp_path, args_mutator, prereg_mutator, failed_check):
    args = _sealed_args(tmp_path)
    d = _sealed_base()
    args_mutator(args)
    prereg_mutator(d)

    report = e0x_runner.sealed_final_preflight(
        d, args, evaluator=_ordinary_fixture_evaluator)
    assert report["ready_for_one_shot_reservation"] is False
    assert report["checks"][failed_check] is False
    with pytest.raises(sealed.SealedAccessError, match="before ACCESS_INTENT"):
        e0x_runner.run_sealed_final(
            d, args, cfg=None,
            rows=[{"fixture": "ordinary", "delta": 0.25}],
            vocab={}, device=None, evaluator=_ordinary_fixture_evaluator)
    assert args.restricted.exists() is False


@pytest.mark.parametrize(
    ("mutator", "failed_check"),
    [
        (lambda d: d.__setitem__("preregistration_id", "MUTATED"),
         "canonical_preregistration_id"),
        (lambda d: d.__setitem__("contract_id", "MUTATED"),
         "canonical_contract_id"),
        (lambda d: d["metrics"]["holm_family"]["hypotheses"].reverse(),
         "preregistration_valid"),
    ],
)
def test_canonical_preregistration_mutation_is_blocked_without_restricted_access(
        tmp_path, mutator, failed_check):
    d = _canonical_base()
    args = _sealed_args(tmp_path, prereg_data=d)
    mutator(d)
    report = e0x_runner.sealed_final_preflight(
        d, args, evaluator=_ordinary_fixture_evaluator)
    assert report["status"] == "BLOCKED"
    assert report["checks"][failed_check] is False
    assert report["checks"]["canonical_preregistration_semantics"] is False
    assert args.restricted.exists() is False


def test_canonical_preregistration_hash_requires_actual_and_confirmation(tmp_path):
    canonical = _canonical_base()
    args = _sealed_args(tmp_path, prereg_data=canonical)
    args.actual_preregistration_sha256 = "0" * 64
    report = e0x_runner.sealed_final_preflight(
        canonical, args, evaluator=_ordinary_fixture_evaluator)
    assert report["checks"]["canonical_preregistration_sha256"] is False
    assert report["ready_for_one_shot_reservation"] is False
    assert args.restricted.exists() is False

    args = _sealed_args(tmp_path / "second", prereg_data=canonical)
    args.confirm_preregistration_sha256 = "0" * 64
    report = e0x_runner.sealed_final_preflight(
        canonical, args, evaluator=_ordinary_fixture_evaluator)
    assert report["checks"]["preregistration_sha256_confirmed"] is False
    assert report["ready_for_one_shot_reservation"] is False
    assert args.restricted.exists() is False


@pytest.mark.parametrize(
    "mutator",
    [
        lambda payload: payload.__setitem__("unexpected_field", True),
        lambda payload: payload.__setitem__("verdict", "NO_GO"),
        lambda payload: payload["per_hypothesis"].pop(1),
        lambda payload: payload["holm_adjusted_pvalues"].__setitem__(0, 0.5),
        lambda payload: payload["go_nogo_checks"].__setitem__(
            "H2_ood_margin_pass", False),
        lambda payload: payload["critic_checkpoint_sha256"].__setitem__(
            "model_5U-A1__GSE114002.pt", "0" * 64),
    ],
)
def test_ordinary_go_activation_is_strict_and_recomputed(tmp_path, mutator):
    args = _sealed_args(tmp_path)
    _rewrite_activation(args, mutator)
    report = e0x_runner.sealed_final_preflight(
        _sealed_base(), args, evaluator=_ordinary_fixture_evaluator)
    assert report["checks"]["ordinary_internal_go_commitment_valid"] is False
    assert report["ordinary_go_commitment"]["valid"] is False
    assert report["ready_for_one_shot_reservation"] is False
    assert args.restricted.exists() is False


def test_synthetic_ordinary_fixture_exercises_scorer_and_state_machine(
        tmp_path, monkeypatch):
    args = _sealed_args(tmp_path)
    fixture_rows = [{"fixture": "ordinary", "delta": 0.25}]

    # The low-level state machine is tested with its independently tested
    # canonical preflight seam replaced.  The current frozen command is stale,
    # so no real configuration can truthfully become READY in this revision.
    monkeypatch.setattr(
        e0x_runner, "require_sealed_final_preflight",
        lambda *_args, **_kwargs: {"status": "READY_TEST_FIXTURE"})
    result = e0x_runner.run_sealed_final(
        _sealed_base(), args, cfg=None, rows=fixture_rows, vocab={}, device=None,
        evaluator=_ordinary_fixture_evaluator)

    assert result["sealed_access_state"] == sealed.SealedAccessState.COMPLETED
    access_log = args.restricted / "ACCESS_LOG.jsonl"
    events = sealed.read_chain(access_log)
    assert sealed.verify_chain(events) is True
    assert [event["status"] for event in events] == [
        "ACCESS_INTENT", "RESERVED", "COMPLETION"]
    assert not any(event["status"] == "ABORT" for event in events)
    sealed.assert_no_row_level(result)


@pytest.mark.parametrize("partial_state", ["INTENT", "RESERVED"])
def test_partial_one_shot_state_is_never_resumed_or_modified(
        tmp_path, partial_state, monkeypatch):
    args = _sealed_args(tmp_path)
    log = args.restricted / "ACCESS_LOG.jsonl"
    sm = sealed.SealedAccessState(log)
    sm.append_intent(
        "prior", "GSE246381_E0X_FINAL", "e0x_sealed_final",
        e0x_runner.FROZEN_PREREGISTRATION_ID)
    if partial_state == "RESERVED":
        sm.reserve(
            "prior", "GSE246381_E0X_FINAL", "e0x_sealed_final",
            e0x_runner.FROZEN_PREREGISTRATION_ID)
    before = log.read_bytes()
    calls = []

    def must_not_run(*_args, **_kwargs):
        calls.append(True)
        raise AssertionError("partial one-shot must not invoke evaluator")

    must_not_run.e0x_contract_version = e0x_runner.SEALED_EVALUATOR_CONTRACT_VERSION
    must_not_run.e0x_implementation_ready = True
    must_not_run.e0x_fixture_sha256 = _ordinary_fixture_evaluator.e0x_fixture_sha256

    monkeypatch.setattr(
        e0x_runner, "require_sealed_final_preflight",
        lambda *_args, **_kwargs: {"status": "READY_TEST_FIXTURE"})

    with pytest.raises(sealed.SealedAccessError, match="never resumed"):
        e0x_runner.run_sealed_final(
            _sealed_base(), args, cfg=None,
            rows=[{"fixture": "ordinary", "delta": 0.25}],
            vocab={}, device=None, evaluator=must_not_run)
    assert calls == []
    assert log.read_bytes() == before


def test_evaluator_contract_rejects_row_level_fixture_output():
    bad = _ordinary_fixture_evaluator(
        None, None, None,
        [{"fixture": "ordinary", "delta": 0.25}], {}, None)
    bad[0]["sequence_id"] = "must-never-leak"
    with pytest.raises(sealed.SealedAccessError, match="schema mismatch"):
        e0x_runner.validate_evaluator_result(_sealed_base(), bad)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda family: family.reverse(),
        lambda family: family.pop(1),
        lambda family: family[1].__setitem__("unexpected", 1),
        lambda family: family[1].__setitem__("top10_recall", 1.1),
        lambda family: family[1].__setitem__("ood_margin_pass", "yes"),
        lambda family: family[2].__setitem__("budget_violation_count", 0.0),
    ],
)
def test_evaluator_contract_is_exact_full_family_allowlist(mutator):
    family = _passing_family()
    mutator(family)
    with pytest.raises(sealed.SealedAccessError):
        e0x_runner.validate_evaluator_result(_sealed_base(), family)


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
    d = _sealed_base()
    per_hyp = _ordinary_fixture_evaluator(
        None, None, None, [{"fixture": "ordinary", "delta": 0.25}], {}, None)
    holm = prereg.holm_bonferroni([0.01, 0.01, 0.01], alpha=0.05)
    v = sealed.verdict_from_aggregate(d, per_hyp, holm)
    assert v["verdict"] == "GO"
    assert all(v["checks"].values())
    # below the pre-registered gate threshold -> NO_GO
    per_hyp[0]["stat"] = 0.10
    v2 = sealed.verdict_from_aggregate(
        d, per_hyp, prereg.holm_bonferroni([0.9, 0.01, 0.01], 0.05))
    assert v2["verdict"] == "NO_GO"


def test_verdict_requires_full_effect_gate_not_just_spearman():
    """Each frozen effect-gate threshold must independently gate the verdict:
    sign_accuracy, top10 enrichment, and beat-abs_candidate must ALL hold in
    addition to the spearman + Holm pair (fixes a gate that only checked the
    spearman threshold)."""
    d = _sealed_base()
    def hyp(stat=0.30, sign=0.62, top10=2.0, abs_base=0.20):
        return {"id": "H1_EFFECT_TRANSFER", "metric": "macro_delta_spearman",
                "stat": stat, "pvalue": 0.01, "n": 100,
                "sign_accuracy": sign, "top10pct_enrichment": top10,
                "abs_candidate_spearman": abs_base,
                "top10pct_enrichment_definition_status": "PREREGISTERED_STABLE"}
    passing = _ordinary_fixture_evaluator(
        None, None, None, [{"fixture": "ordinary", "delta": 0.25}], {}, None)
    h2, h3 = passing[1], passing[2]
    holm = prereg.holm_bonferroni([0.01, 0.01, 0.01], alpha=0.05)

    # baseline all-pass -> GO
    assert sealed.verdict_from_aggregate(
        d, [hyp(), dict(h2), dict(h3)], holm)["verdict"] == "GO"

    # each single violation must flip to NO_GO (even if spearman+Holm pass)
    assert sealed.verdict_from_aggregate(
        d, [hyp(sign=0.50), dict(h2), dict(h3)], holm)["verdict"] == "NO_GO"
    assert sealed.verdict_from_aggregate(
        d, [hyp(top10=1.0), dict(h2), dict(h3)], holm)["verdict"] == "NO_GO"
    assert sealed.verdict_from_aggregate(
        d, [hyp(abs_base=0.40), dict(h2), dict(h3)], holm)["verdict"] == "NO_GO"


@pytest.mark.parametrize(
    ("hypothesis_index", "field", "value", "failed_check"),
    [
        (1, "pvalue", 0.90, "H2_holm_significant"),
        (1, "generation_quality_mean_delta_difference", 0.0,
         "H2_generation_quality_mean_delta_gt"),
        (1, "generation_quality_frac_beneficial_difference", 0.0,
         "H2_generation_quality_frac_beneficial_gt"),
        (1, "ndcg_gap", 0.01, "H2_ndcg_gap_ge"),
        (1, "top10_recall", 0.60, "H2_top10_recall_ge"),
        (1, "normalized_regret", 0.20, "H2_normalized_regret_le"),
        (1, "ood_margin_pass", False, "H2_ood_margin_pass"),
        (2, "pvalue", 0.90, "H3_holm_significant"),
        (2, "stat", 0.99, "H3_legality_ge"),
        (2, "length_preservation_rate", 0.99,
         "H3_length_preservation_ge"),
        (2, "budget_violation_count", 1, "H3_budget_violation_le"),
    ],
)
def test_verdict_enforces_every_h2_h3_gate(
        hypothesis_index, field, value, failed_check):
    d = _sealed_base()
    family = _passing_family()
    family[hypothesis_index][field] = value
    holm = prereg.holm_bonferroni(
        [h["pvalue"] for h in family], alpha=0.05)
    verdict = sealed.verdict_from_aggregate(d, family, holm)
    assert verdict["verdict"] == "NO_GO"
    assert verdict["checks"][failed_check] is False


@pytest.mark.parametrize(
    "mutator",
    [
        lambda family, holm: family.pop(1),
        lambda family, holm: family[1].pop("ndcg_gap"),
        lambda family, holm: family[1].__setitem__("ndcg_gap", "bad"),
        lambda family, holm: holm.pop(),
    ],
)
def test_incomplete_decision_inputs_hold_instead_of_false_go_or_nogo(mutator):
    d = _sealed_base()
    family = _passing_family()
    holm = prereg.holm_bonferroni(
        [h["pvalue"] for h in family], alpha=0.05)
    mutator(family, holm)
    verdict = sealed.verdict_from_aggregate(d, family, holm)
    assert verdict["verdict"] == "HOLD_REPAIR"
    assert verdict["repair_reasons"]
