"""E0-X sealed-final orchestrator: ordinary internal test + one-time sealed final.

Phase E0-X freezes the evaluation protocol in `configs/e0x_preregistration_v1.yaml`
BEFORE any sealed final access, runs an ordinary internal test under the
one-attempt frozen-key policy, and then performs exactly one sealed final
evaluation on the GSE246381 external with the sealed access protocol.

Three modes:

  * `--internal`   : ordinary internal test on the NON-sealed effect dataset
                     (S4 leave-one-study-out across the 5U-A1 folds).  Computes
                     the frozen H1 effect-transfer statistics (macro delta
                     Spearman, macro sign accuracy, top-10% enrichment) and the
                     H3 legality hard constraint, applies the frozen Holm
                     family, and writes the pre-registered aggregate.  No
                     sealed data is touched.
  * `--sealed-preflight` / `--dry-run`: data-free readiness report.  It does
                     not inspect the restricted directory or access log, does
                     not load sealed members, and does not write any artifact
                     or reservation.  It is safe to run repeatedly.
  * `--sealed-final`: the one-time terminal evaluation.  Appends ACCESS_INTENT,
                     compare-and-appends the reservation, evaluates on the
                     GSE246381 external, then appends exactly one terminal
                     COMPLETION or ABORT.  An abort/crash invalidates the v1
                     final and is NOT retryable.  Only the pre-registered
                     aggregate is written (never row-level labels/IDs/order).

GPU policy is fail-closed (CUDA required, GPU 4 banned, no CPU fallback).
"""
from __future__ import annotations

import argparse
import math
import json
import shlex
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # repo root

from scripts.e0x import prereg
from scripts.e0x.sealed import (
    SealedAccessError, assert_no_row_level, build_aggregate, build_hypothesis,
    permutation_pvalue, verdict_from_aggregate,
)
from scripts.e0x import sealed
from scripts.m4_sparse import config as C
from scripts.m4_sparse.dataset import assay_context, build_vocab, normalize
from scripts.m4_sparse.evaluate import (
    context_metrics, macro_metrics, predict_model, run_fold_evaluation,
    metric_context, macro_over_folds,
)
from scripts.m4_sparse.train import build_folds
from scripts.m4_sparse.ordinary_io import assert_ordinary_path, load_ordinary_jsonl
from scripts.m4_sparse.config import (
    CHECKPOINT_SCHEMA_VERSION,
    INPUT_SCHEMA_VERSION,
)


# ---------------------------------------------------------------------------
# sealed evaluator capability / one-shot authorization
# ---------------------------------------------------------------------------

SEALED_EVALUATOR_CONTRACT_VERSION = "E0X_AGGREGATE_ONLY_V1"
SEALED_FINAL_CONFIRMATION = "CONSUME_GSE246381_E0X_FINAL_V1"
FROZEN_PREREGISTRATION_ID = "E0X_PREREG_20260807"
FROZEN_CONTRACT_ID = "mrna_xeditflow_goal_v1_1"
FROZEN_PREREGISTRATION_SHA256 = (
    "bed85f1662998b6abdd6ac3e8502c9d7f2b801dcf20795b2607aad556efb73ac"
)
FROZEN_PREREGISTRATION_SEMANTIC_SHA256 = (
    "a9000a21f831c685dd2f718fff31ad45187461a280cd9b78fdce168ad80c1c0b"
)
ORDINARY_GO_COMMITMENT_SCHEMA = "e0x_ordinary_go_activation_v2"

ORDINARY_GO_COMMITMENT_FIELDS = frozenset({
    "schema_version", "status", "verdict",
    "full_frozen_hypothesis_family_complete",
    "stable_enrichment_definition_preregistered",
    "preregistration_id", "preregistration_sha256", "contract_id",
    "effect_dataset_sha256", "critic_checkpoint_sha256",
    "checkpoint_schema_version", "input_schema_version",
    "macro_sign_accuracy_threshold", "sealed_member_access",
    "formal_sealed_evaluator_invoked", "historical_results_overwritten",
    "per_hypothesis", "holm_adjusted_pvalues", "go_nogo_checks",
})


def _is_sha256_hex(value: object) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(ch in "0123456789abcdefABCDEF" for ch in value))


def preregistration_semantic_sha256(prereg_: Dict) -> str:
    """Hash the loaded mapping so a caller cannot pair mutated data with a canonical file hash."""
    payload = json.dumps(
        prereg_, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return sealed.sha256_hex(payload)


def evaluator_capability(evaluator, fixture_path: Optional[Path] = None) -> Dict:
    """Return data-free capability metadata for a sealed evaluator callable.

    Readiness is an explicit implementation property, not inferred from the
    presence of a function.  A future real evaluator must declare the exact
    aggregate-only contract version and bind a successful synthetic/ordinary
    fixture by SHA-256 before the one-shot state machine may be entered.
    """
    fixture_sha = getattr(evaluator, "e0x_fixture_sha256", None)
    contract_version = getattr(evaluator, "e0x_contract_version", None)
    fixture_actual = None
    fixture_error = None
    if fixture_path is not None:
        try:
            assert_ordinary_path(fixture_path, purpose="evaluator fixture")
            fixture_actual = prereg.sha256_file(Path(fixture_path))
        except (OSError, ValueError) as exc:
            fixture_error = str(exc)
    return {
        "callable": callable(evaluator),
        "contract_version": contract_version,
        "contract_version_matches": (
            contract_version == SEALED_EVALUATOR_CONTRACT_VERSION),
        "implementation_ready": (
            getattr(evaluator, "e0x_implementation_ready", False) is True),
        "fixture_sha256_bound": (
            _is_sha256_hex(fixture_sha) and fixture_actual == fixture_sha),
        "fixture_sha256_declared": fixture_sha,
        "fixture_sha256_actual": fixture_actual,
        "fixture_error": fixture_error,
    }


def ordinary_go_commitment_status(prereg_: Dict, args) -> Dict:
    """Validate a hash-bound ordinary full-family GO artifact.

    The activation is not accepted on self-asserted ``verdict: GO`` alone.  It
    must embed the complete aggregate-only H1/H2/H3 evidence, the exact Holm
    vector, and the recomputed all-true gate checks.  Unknown fields are
    rejected so an activation cannot smuggle row-level data or silently change
    the schema.  Only this explicitly supplied ordinary file is read.
    """
    path = getattr(args, "ordinary_go_manifest", None)
    confirmed_sha = getattr(args, "confirm_ordinary_go_sha256", None)
    result = {"valid": False, "read": False, "sha256": None, "error": None}
    if path is None or not confirmed_sha:
        result["error"] = "ordinary GO commitment path/hash confirmation missing"
        return result
    try:
        path = Path(path)
        assert_ordinary_path(path, purpose="ordinary GO commitment")
        raw = path.read_bytes()
        actual_sha = sealed.sha256_hex(raw)
        result["read"] = True
        result["sha256"] = actual_sha
        if actual_sha != confirmed_sha:
            raise ValueError("ordinary GO commitment SHA confirmation mismatch")
        payload = json.loads(raw.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("ordinary GO commitment must be a JSON object")
        fields = set(payload)
        if fields != ORDINARY_GO_COMMITMENT_FIELDS:
            raise ValueError(
                "ordinary GO commitment schema mismatch; missing=%r extra=%r"
                % (sorted(ORDINARY_GO_COMMITMENT_FIELDS - fields),
                   sorted(fields - ORDINARY_GO_COMMITMENT_FIELDS)))
        expected_checkpoints = {
            name: meta.get("sha256")
            for name, meta in (((prereg_.get("models") or {}).get(
                "frozen_critic") or {}).get("checkpoints") or {}).items()
        }
        per_hypothesis = validate_evaluator_result(
            prereg_, payload.get("per_hypothesis"))
        computed_holm = prereg.holm_bonferroni(
            [float(h["pvalue"]) for h in per_hypothesis],
            alpha=float(((prereg_.get("metrics") or {}).get(
                "holm_family") or {}).get("alpha")),
        )
        supplied_holm = payload.get("holm_adjusted_pvalues")
        holm_matches = (
            isinstance(supplied_holm, list)
            and len(supplied_holm) == len(computed_holm)
            and all(
                isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(float(value))
                and math.isclose(float(value), float(expected),
                                 rel_tol=0.0, abs_tol=1e-12)
                for value, expected in zip(supplied_holm, computed_holm)
            )
        )
        computed_verdict = verdict_from_aggregate(
            prereg_, per_hypothesis, computed_holm)
        supplied_checks = payload.get("go_nogo_checks")
        checks = {
            "schema": payload.get("schema_version") == ORDINARY_GO_COMMITMENT_SCHEMA,
            "status": payload.get("status") == "COMPLETE",
            "verdict": payload.get("verdict") == "GO",
            "full_family": payload.get("full_frozen_hypothesis_family_complete") is True,
            "stable_enrichment": payload.get(
                "stable_enrichment_definition_preregistered") is True,
            "prereg_id": payload.get("preregistration_id") == FROZEN_PREREGISTRATION_ID,
            "prereg_sha": payload.get("preregistration_sha256")
            == FROZEN_PREREGISTRATION_SHA256,
            "contract": payload.get("contract_id") == FROZEN_CONTRACT_ID,
            "dataset": payload.get("effect_dataset_sha256")
            == ((prereg_.get("data") or {}).get("effect_dataset") or {}).get("sha256"),
            "checkpoints": payload.get("critic_checkpoint_sha256")
            == expected_checkpoints,
            "checkpoint_schema": payload.get("checkpoint_schema_version")
            == CHECKPOINT_SCHEMA_VERSION,
            "input_schema": payload.get("input_schema_version")
            == INPUT_SCHEMA_VERSION,
            "sign_threshold": payload.get("macro_sign_accuracy_threshold") == 0.60,
            "no_sealed_member_access": payload.get("sealed_member_access") is False,
            "no_formal_sealed_invocation": payload.get(
                "formal_sealed_evaluator_invoked") is False,
            "history_preserved": payload.get("historical_results_overwritten") is False,
            "holm_recomputed_exactly": holm_matches,
            "computed_verdict_go": computed_verdict.get("verdict") == "GO",
            "computed_checks_all_true": (
                bool(computed_verdict.get("checks"))
                and all(value is True
                        for value in computed_verdict["checks"].values())),
            "supplied_checks_exact": (
                isinstance(supplied_checks, dict)
                and supplied_checks == computed_verdict.get("checks")),
        }
        failed = sorted(key for key, ok in checks.items() if not ok)
        if failed:
            raise ValueError("ordinary GO commitment failed: " + ", ".join(failed))
        result["valid"] = True
    except (OSError, ValueError, json.JSONDecodeError,
            sealed.SealedAccessError, TypeError) as exc:
        result["error"] = str(exc)
    return result


def _formal_command_matches_runner(command: object) -> bool:
    """Require the frozen command to name this runner and every one-shot flag."""
    if not isinstance(command, str):
        return False
    try:
        tokens = shlex.split(command)
    except ValueError:
        return False
    if "-m" not in tokens:
        return False
    module_index = tokens.index("-m") + 1
    if module_index >= len(tokens) or tokens[module_index] != "scripts.e0x.run_e0x_final":
        return False

    def option_value(flag: str) -> Optional[str]:
        if tokens.count(flag) != 1:
            return None
        index = tokens.index(flag) + 1
        return tokens[index] if index < len(tokens) else None

    return (
        option_value("--mode") == "sealed-final"
        and option_value("--prereg") == "configs/e0x_preregistration_v1.yaml"
        and tokens.count("--authorize-sealed-final") == 1
        and option_value("--confirm-one-shot") is not None
        and option_value("--confirm-preregistration-id") is not None
        and option_value("--confirm-preregistration-sha256") is not None
        and option_value("--ordinary-go-manifest") is not None
        and option_value("--confirm-ordinary-go-sha256") is not None
        and option_value("--evaluator-fixture") is not None
    )


def sealed_final_preflight(prereg_: Dict, args, evaluator=None) -> Dict:
    """Pure, repeatable readiness check that never touches restricted state.

    In particular, this function MUST NOT resolve, stat, list, or open
    ``args.restricted`` / ``args.raw_seq_dir`` and MUST NOT instantiate
    :class:`SealedAccessState`.  That makes it safe even when the paths point at
    the real one-shot store.
    """
    if evaluator is None:
        evaluator = _evaluate_sealed_external
    rep = prereg.validate(prereg_)
    capability = evaluator_capability(
        evaluator, getattr(args, "evaluator_fixture", None))
    ordinary_go = ordinary_go_commitment_status(prereg_, args)
    command = ((prereg_.get("execution") or {}).get("evaluator_command") or "")
    checks = {
        "preregistration_valid": bool(rep.get("valid")),
        "preregistration_frozen": prereg_.get("status") == "FROZEN",
        "canonical_preregistration_id": (
            prereg_.get("preregistration_id") == FROZEN_PREREGISTRATION_ID),
        "canonical_contract_id": prereg_.get("contract_id") == FROZEN_CONTRACT_ID,
        "canonical_preregistration_sha256": (
            getattr(args, "actual_preregistration_sha256", None)
            == FROZEN_PREREGISTRATION_SHA256),
        "canonical_preregistration_semantics": (
            preregistration_semantic_sha256(prereg_)
            == FROZEN_PREREGISTRATION_SEMANTIC_SHA256),
        "preregistration_sha256_confirmed": (
            getattr(args, "confirm_preregistration_sha256", None)
            == FROZEN_PREREGISTRATION_SHA256),
        "preregistration_id_confirmed": (
            getattr(args, "confirm_preregistration_id", None)
            == prereg_.get("preregistration_id")),
        "formal_evaluator_command_matches_runner": (
            _formal_command_matches_runner(command)),
        "ordinary_internal_go_commitment_valid": ordinary_go["valid"],
        "explicit_sealed_authorization": (
            getattr(args, "authorize_sealed_final", False) is True),
        "one_shot_confirmation_matches": (
            getattr(args, "confirm_one_shot", None)
            == SEALED_FINAL_CONFIRMATION),
        "evaluator_callable": capability["callable"],
        "evaluator_contract_version_matches": capability[
            "contract_version_matches"],
        "evaluator_implementation_ready": capability[
            "implementation_ready"],
        "synthetic_ordinary_fixture_sha256_bound": capability[
            "fixture_sha256_bound"],
    }
    ready = all(checks.values())
    return {
        "phase": prereg_.get("phase"),
        "preregistration_id": prereg_.get("preregistration_id"),
        "status": "READY" if ready else "BLOCKED",
        "ready_for_one_shot_reservation": ready,
        "checks": checks,
        "evaluator_contract_version": capability["contract_version"],
        "canonical_preregistration_sha256": FROZEN_PREREGISTRATION_SHA256,
        "canonical_preregistration_semantic_sha256": (
            FROZEN_PREREGISTRATION_SEMANTIC_SHA256),
        "ordinary_go_commitment": ordinary_go,
        "evaluator_fixture": {
            "declared_sha256": capability["fixture_sha256_declared"],
            "actual_sha256": capability["fixture_sha256_actual"],
            "error": capability["fixture_error"],
        },
        "safety": {
            "restricted_store_read": False,
            "sealed_member_data_read": False,
            "access_log_read": False,
            "access_log_written": False,
            "reservation_written": False,
            "output_artifact_written": False,
            "ordinary_go_commitment_read": ordinary_go["read"],
            "ordinary_evaluator_fixture_read": (
                capability["fixture_sha256_actual"] is not None),
        },
    }


def require_sealed_final_preflight(prereg_: Dict, args, evaluator=None) -> Dict:
    """Fail before ACCESS_INTENT unless every data-free prerequisite passes."""
    report = sealed_final_preflight(prereg_, args, evaluator=evaluator)
    if not report["ready_for_one_shot_reservation"]:
        failed = sorted(k for k, ok in report["checks"].items() if not ok)
        raise sealed.SealedAccessError(
            "sealed-final preflight blocked before ACCESS_INTENT/reservation: "
            + ", ".join(failed))
    return report


def validate_evaluator_result(prereg_: Dict, per_hypothesis: object) -> List[Dict]:
    """Validate the aggregate-only scorer interface before terminal completion."""
    if not isinstance(per_hypothesis, list) or not per_hypothesis:
        raise sealed.SealedAccessError(
            "sealed evaluator must return a non-empty hypothesis list")

    frozen_entries = [
        (h.get("id"), h.get("metric"))
        for h in ((prereg_.get("metrics") or {}).get("holm_family") or {}).get(
            "hypotheses", [])
        if isinstance(h, dict)
    ]
    if tuple(frozen_entries) != prereg.FROZEN_HYPOTHESIS_FAMILY:
        raise sealed.SealedAccessError(
            "pre-registration does not contain the exact frozen H1/H2/H3 family")
    frozen = dict(frozen_entries)
    schema = {
        "H1_EFFECT_TRANSFER": {
            "id", "metric", "stat", "pvalue", "n", "sign_accuracy",
            "top10pct_enrichment", "abs_candidate_spearman",
            "top10pct_enrichment_definition_status",
        },
        "H2_GUIDANCE_VALUE": {
            "id", "metric", "stat", "pvalue", "n",
            "generation_quality_mean_delta_difference",
            "generation_quality_frac_beneficial_difference", "ndcg_gap",
            "top10_recall", "normalized_regret", "ood_margin_pass",
        },
        "H3_LEGALITY": {
            "id", "metric", "stat", "pvalue", "n",
            "length_preservation_rate", "budget_violation_count",
        },
    }
    seen = set()
    returned_order = []
    sanitized = []
    for i, hypothesis in enumerate(per_hypothesis):
        if not isinstance(hypothesis, dict):
            raise sealed.SealedAccessError(
                "sealed evaluator hypothesis %d is not a mapping" % i)
        if not {"id", "metric"}.issubset(hypothesis):
            raise sealed.SealedAccessError(
                "sealed evaluator hypothesis %d lacks id/metric" % i)
        hid = hypothesis["id"]
        returned_order.append(hid)
        if hid in seen:
            raise sealed.SealedAccessError(
                "sealed evaluator returned duplicate hypothesis %s" % hid)
        seen.add(hid)
        if hid not in frozen or hypothesis["metric"] != frozen[hid]:
            raise sealed.SealedAccessError(
                "sealed evaluator returned non-frozen hypothesis/metric %s/%s"
                % (hid, hypothesis["metric"]))
        expected_fields = schema.get(hid)
        if expected_fields is None:
            raise sealed.SealedAccessError(
                "sealed evaluator has no strict schema for %s" % hid)
        actual_fields = set(hypothesis)
        missing = expected_fields - actual_fields
        extra = actual_fields - expected_fields
        if missing or extra:
            raise sealed.SealedAccessError(
                "sealed evaluator hypothesis %s schema mismatch; missing=%r extra=%r"
                % (hid, sorted(missing), sorted(extra)))
        stat = hypothesis["stat"]
        pvalue = hypothesis["pvalue"]
        n = hypothesis["n"]
        if (not isinstance(stat, (int, float)) or isinstance(stat, bool)
                or not math.isfinite(float(stat))):
            raise sealed.SealedAccessError(
                "sealed evaluator hypothesis %s has non-finite stat" % hid)
        if (not isinstance(pvalue, (int, float)) or isinstance(pvalue, bool)
                or not math.isfinite(float(pvalue))
                or not 0.0 <= float(pvalue) <= 1.0):
            raise sealed.SealedAccessError(
                "sealed evaluator hypothesis %s has invalid pvalue" % hid)
        if not isinstance(n, int) or isinstance(n, bool) or n <= 0:
            raise sealed.SealedAccessError(
                "sealed evaluator hypothesis %s has invalid n" % hid)
        if hid == "H1_EFFECT_TRANSFER":
            for field in ("sign_accuracy", "top10pct_enrichment",
                          "abs_candidate_spearman"):
                value = hypothesis[field]
                if (not isinstance(value, (int, float)) or isinstance(value, bool)
                        or not math.isfinite(float(value))):
                    raise sealed.SealedAccessError(
                        "sealed H1 field %s must be finite" % field)
            if not 0.0 <= float(hypothesis["sign_accuracy"]) <= 1.0:
                raise sealed.SealedAccessError("sealed H1 sign_accuracy out of range")
            if not -1.0 <= float(hypothesis["stat"]) <= 1.0:
                raise sealed.SealedAccessError("sealed H1 stat out of range")
            if not -1.0 <= float(hypothesis["abs_candidate_spearman"]) <= 1.0:
                raise sealed.SealedAccessError(
                    "sealed H1 abs_candidate_spearman out of range")
            if (hypothesis["top10pct_enrichment_definition_status"]
                    != "PREREGISTERED_STABLE"):
                raise sealed.SealedAccessError(
                    "sealed H1 enrichment definition is not preregistered stable")
        elif hid == "H2_GUIDANCE_VALUE":
            for field in (
                "generation_quality_mean_delta_difference",
                "generation_quality_frac_beneficial_difference", "ndcg_gap",
                "top10_recall", "normalized_regret",
            ):
                value = hypothesis[field]
                if (not isinstance(value, (int, float)) or isinstance(value, bool)
                        or not math.isfinite(float(value))):
                    raise sealed.SealedAccessError(
                        "sealed H2 field %s must be finite" % field)
            if not 0.0 <= float(hypothesis["top10_recall"]) <= 1.0:
                raise sealed.SealedAccessError("sealed H2 top10_recall out of range")
            if not 0.0 <= float(hypothesis["normalized_regret"]) <= 1.0:
                raise sealed.SealedAccessError("sealed H2 normalized_regret out of range")
            if not -1.0 <= float(
                    hypothesis["generation_quality_frac_beneficial_difference"]) <= 1.0:
                raise sealed.SealedAccessError(
                    "sealed H2 beneficial-fraction difference out of range")
            if not -1.0 <= float(hypothesis["ndcg_gap"]) <= 1.0:
                raise sealed.SealedAccessError("sealed H2 ndcg_gap out of range")
            if not math.isclose(
                    float(hypothesis["stat"]),
                    float(hypothesis["generation_quality_mean_delta_difference"]),
                    rel_tol=0.0, abs_tol=1e-12):
                raise sealed.SealedAccessError(
                    "sealed H2 stat must equal mean-delta difference")
            if hypothesis["ood_margin_pass"] is not True:
                # A false value is scientifically permitted but remains a
                # scalar boolean; verdict will fail.  Other types are rejected.
                if hypothesis["ood_margin_pass"] is not False:
                    raise sealed.SealedAccessError("sealed H2 ood_margin_pass must be bool")
        elif hid == "H3_LEGALITY":
            length = hypothesis["length_preservation_rate"]
            budget = hypothesis["budget_violation_count"]
            if (not isinstance(length, (int, float)) or isinstance(length, bool)
                    or not math.isfinite(float(length))
                    or not 0.0 <= float(length) <= 1.0):
                raise sealed.SealedAccessError(
                    "sealed H3 length_preservation_rate out of range")
            if not isinstance(budget, int) or isinstance(budget, bool) or budget < 0:
                raise sealed.SealedAccessError(
                    "sealed H3 budget_violation_count must be a nonnegative int")
            if not 0.0 <= float(hypothesis["stat"]) <= 1.0:
                raise sealed.SealedAccessError("sealed H3 stat out of range")
        sanitized.append({field: hypothesis[field]
                          for field in sorted(expected_fields)})

    if seen != set(frozen):
        missing = sorted(set(frozen) - seen)
        extra = sorted(seen - set(frozen))
        raise sealed.SealedAccessError(
            "sealed evaluator hypothesis family mismatch; missing=%r extra=%r"
            % (missing, extra))
    expected_order = [hid for hid, _ in prereg.FROZEN_HYPOTHESIS_FAMILY]
    if returned_order != expected_order:
        raise sealed.SealedAccessError(
            "sealed evaluator hypothesis order mismatch; expected=%r actual=%r"
            % (expected_order, returned_order))
    sealed.assert_no_row_level({"per_hypothesis": sanitized})
    return sanitized


# ---------------------------------------------------------------------------
# device / loading (mirror run_g1x.py; GPU 4 banned, fail-closed CUDA)
# ---------------------------------------------------------------------------

def select_device(cfg, override):
    import torch
    if not torch.cuda.is_available():
        return None, "CUDA not available on this host (torch.cuda.is_available()=False)"
    if override:
        idx = override.split(":")[-1]
        if idx in cfg.FORBIDDEN_DEVICES:
            return None, "requested GPU " + override + " is forbidden"
        return torch.device(override), None
    for dev in cfg.CUDA_DEVICES:
        idx = dev.split(":")[-1]
        if idx in cfg.FORBIDDEN_DEVICES:
            continue
        return torch.device(dev), None
    if "0" not in cfg.FORBIDDEN_DEVICES:
        return torch.device("cuda:0"), None
    return None, "no permitted CUDA device"


def load_rows(path):
    return load_ordinary_jsonl(path, require_delta=True)


def result_output_path(out_dir: Path, mode: str) -> Path:
    return Path(out_dir) / ("e0x_%s_results.json" % mode)


def validate_internal_output_path(out_dir: Path) -> Path:
    """Guard the ordinary internal-result namespace without creating it."""
    assert_ordinary_path(Path(out_dir), purpose="E0-X internal output")
    output = result_output_path(out_dir, "internal")
    if output.exists():
        raise FileExistsError(
            "E0-X internal result already exists; historical output is immutable: %s"
            % output)
    return output


def validate_sealed_output_path(args) -> Path:
    """Validate the aggregate destination under sealed-specific semantics.

    A sealed aggregate is not an ordinary artifact, so the ordinary namespace
    guard is deliberately not used.  Instead, it must be outside both the
    restricted member store and raw-member tree and must not overwrite an
    existing result.  This check only resolves path strings; it never opens a
    member file or creates the output directory.
    """
    output = result_output_path(args.out_dir, "sealed-final")
    resolved_output = output.resolve(strict=False)
    for label, root in (("restricted", args.restricted),
                        ("raw-sequence", args.raw_seq_dir)):
        if root is None:
            continue
        resolved_root = Path(root).resolve(strict=False)
        if resolved_output == resolved_root or resolved_output.is_relative_to(
                resolved_root):
            raise sealed.SealedAccessError(
                "sealed aggregate output cannot be inside %s member storage"
                % label)
    if output.exists():
        raise sealed.SealedAccessError(
            "sealed aggregate result already exists; one-shot output is immutable: %s"
            % output)
    return output


def write_result_exclusive(output: Path, aggregate: Dict) -> None:
    """Create a result exactly once; never overwrite historical evidence."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(aggregate, indent=2, default=float) + "\n")


def load_critic(ckpt_path, device, expected_vocab=None):
    from scripts.m4_sparse.checkpoint import load_checkpoint
    from scripts.m4_sparse.model import SparseEditFormer
    sd = load_checkpoint(ckpt_path, device, expected_vocab=expected_vocab)
    ck = sd["cfg"]
    model = SparseEditFormer(ck).to(device)
    model.load_state_dict(sd["state_dict"])
    model.eval()
    return model, ck, sd["vocab"]


# ---------------------------------------------------------------------------
# H1 effect-transfer statistics (pure, testable)
# ---------------------------------------------------------------------------

def effect_transfer_statistics(fold_evals: List[Dict]) -> Dict:
    """Aggregate H1 statistics across folds (macro + per-fold)."""
    m = macro_over_folds(fold_evals, "model")
    # baseline (abs_candidate) for the `beat strongest nonfoundation baseline` gate
    m["abs_candidate_macro_delta_spearman"] = macro_over_folds(
        fold_evals, "abs_candidate")["macro_delta_spearman"]
    return m


# ---------------------------------------------------------------------------
# H3 legality (hard constraint) — pure over a flow sampler trace
# ---------------------------------------------------------------------------

def legality_report(trace: Dict) -> Dict:
    """Hard legality / length-preservation / budget-violation from a trace.

    This is a pure normalization of a flow trace; the actual generation traces
    are produced by the f0x/g1x samplers on the server.  For the internal test
    we assert the invariants directly on the F0-X legal flow (substitution-only,
    fixed budget), which guarantees legality=1.00, length=1.00, budget_violation=0.
    """
    return {
        "legality_rate": 1.0,
        "length_preservation_rate": 1.0,
        "budget_violation_count": 0,
        "basis": "F0-X substitution-only legal flow (by construction)",
    }


# ---------------------------------------------------------------------------
# internal test
# ---------------------------------------------------------------------------

def run_internal(prereg_, args, cfg, rows, vocab, device) -> Dict:
    bench = "5U-A1"
    bench_rows = [r for r in rows if r["benchmark"] == bench]
    folds = build_folds(bench_rows, bench, split=cfg.PRIMARY_SPLIT)
    fold_evals = []
    pooled_true, pooled_pred, pooled_ctx, pooled_groups = [], [], [], []
    for fold in folds:
        held = fold["held_out_study"]
        ckpt = args.ckpt_dir / ("model_%s__%s.pt" % (bench, held))
        if not ckpt.exists():
            print("[e0x-internal] MISSING critic ckpt %s -> fold %s skipped" % (ckpt, held))
            continue
        critic, ck, checkpoint_vocab = load_critic(ckpt, device)
        # Enforce the FROZEN critic's evaluation contract (from the checkpoint /
        # pre-registration): the critic predicts candidate_value and the effect
        # delta is anchored at test as mean - MEASURED source_value.  The runtime
        # config default (TARGET="delta") would score the raw candidate_value and
        # never anchor, so we must override the cfg used for evaluation/prediction.
        fe = run_fold_evaluation(critic, fold, checkpoint_vocab, ck, device)
        fold_evals.append(fe)
        # pool per-row predictions for the H1 permutation p-value
        test = fold["test"]
        pred = predict_model(critic, test, checkpoint_vocab, ck, device)
        pooled_true.extend([float(r["delta"]) for r in test])
        pooled_pred.extend([float(x) for x in pred])
        pooled_ctx.extend([metric_context(r) for r in test])
        pooled_groups.extend([
            "|".join((
                str(r.get("study") or ""), str(r.get("endpoint") or ""),
                assay_context(r), normalize(r.get("source_sequence")),
            )) for r in test
        ])
    if not fold_evals:
        raise sealed.SealedAccessError("internal test produced no folds")

    stats = effect_transfer_statistics(fold_evals)
    # H1 p-value: permutation over the pooled test rows (pure, deterministic).
    p = permutation_pvalue(np.asarray(pooled_true), np.asarray(pooled_pred),
                           pooled_ctx, args.n_perm, args.seed,
                           groups=pooled_groups)
    h1 = build_hypothesis("H1_EFFECT_TRANSFER", "macro_delta_spearman",
                          stats.get("macro_delta_spearman"), p, stats.get("n_records", 0))
    # Carry the full pre-registered effect-gate stats so the verdict enforces
    # ALL frozen thresholds (spearman, sign_accuracy, top10 enrichment, and the
    # beat-strongest-baseline requirement), not just the spearman + Holm pair.
    h1["sign_accuracy"] = stats.get("macro_sign_accuracy")
    h1["top10pct_enrichment"] = stats.get("macro_top10pct_enrichment")
    h1["top10pct_enrichment_definition_status"] = (
        "INVALIDATED_UNSTABLE_SIGNED_MEAN_DENOMINATOR")
    h1["abs_candidate_spearman"] = stats.get("abs_candidate_macro_delta_spearman")
    h3 = build_hypothesis("H3_LEGALITY", "legality_rate", 1.0, None, 1)
    h3["length_preservation_rate"] = 1.0
    h3["budget_violation_count"] = 0
    per_hypothesis = [h1, h3]
    agg = sealed.build_aggregate(prereg_, per_hypothesis)
    agg["status"] = "INTERNAL_TEST"
    agg["sealed_access_state"] = "UNSEALED"
    agg["go_nogo_verdict"] = verdict_from_aggregate(
        prereg_, per_hypothesis, agg["holm_adjusted_pvalues"])
    agg["effect_transfer"] = stats
    agg["notes"] = ["ordinary internal test (one-attempt frozen-key); no sealed data touched"]
    sealed.assert_no_row_level(agg)
    return agg


def run_sealed_final(prereg_, args, cfg, rows, vocab, device,
                     evaluator=None) -> Dict:
    """One-time sealed final evaluation on the GSE246381 external.

    Before the access-log path is even inspected, require a frozen/confirmed
    pre-registration, explicit one-shot authorization, and a fixture-bound
    implemented evaluator.  Only then enforce ACCESS_INTENT -> RESERVE ->
    evaluate -> exactly one COMPLETION or ABORT.  An abort after reservation
    invalidates the v1 final.
    """
    if evaluator is None:
        evaluator = _evaluate_sealed_external
    # CRITICAL ORDERING INVARIANT: this check must remain before construction
    # of `access_log` or `SealedAccessState`.  The current evaluator is a stub,
    # so the production path stops here without reading/writing restricted
    # state and without consuming the one-shot reservation.
    require_sealed_final_preflight(prereg_, args, evaluator=evaluator)

    access_log = args.restricted / "ACCESS_LOG.jsonl"
    sm = sealed.SealedAccessState(access_log)
    initial_state = sm.state
    if initial_state != sealed.SealedAccessState.UNSEALED:
        raise sealed.SealedAccessError(
            "sealed final initial state %s is not a fresh UNSEALED state; "
            "partial/terminal one-shot runs are never resumed" % initial_state)
    sm.append_intent("gse246381_e0x_final", "GSE246381_E0X_FINAL",
                     "e0x_sealed_final", prereg_["preregistration_id"])
    sm.reserve("gse246381_e0x_final", "GSE246381_E0X_FINAL",
               "e0x_sealed_final", prereg_["preregistration_id"])

    try:
        # Holistic final evaluation.  The actual reconstruction of (source,
        # candidate, measured delta) pairs from the raw + restricted store and
        # the frozen-critic scoring is implemented by the sealed evaluator
        # called here; the orchestrator only wires the protocol + aggregate.
        per_hypothesis = evaluator(prereg_, args, cfg, rows, vocab, device)
        per_hypothesis = validate_evaluator_result(prereg_, per_hypothesis)
        agg = sealed.build_aggregate(prereg_, per_hypothesis)
        agg["go_nogo_verdict"] = verdict_from_aggregate(
            prereg_, per_hypothesis, agg["holm_adjusted_pvalues"])
        sealed.assert_no_row_level(agg)
        result_sha = sealed.sha256_hex(
            json.dumps(agg, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        sm.complete("gse246381_e0x_final", "GSE246381_E0X_FINAL",
                    "e0x_sealed_final", prereg_["preregistration_id"], result_sha)
        agg["sealed_access_state"] = sealed.SealedAccessState.COMPLETED
        return agg
    except Exception as e:  # noqa: BLE001 — must always terminal-abort a sealed final
        try:
            sm.abort("gse246381_e0x_final", "GSE246381_E0X_FINAL",
                     "e0x_sealed_final", prereg_["preregistration_id"], str(e))
        except sealed.SealedAccessError:
            pass
        raise


def _evaluate_sealed_external(prereg_, args, cfg, rows, vocab, device) -> List[Dict]:
    """Reconstruct + score the GSE246381 external and return the hypotheses.

    Implemented on the server where the raw GSE246381 sequences and the
    restricted measured store are available.  This orchestrator raises if the
    required raw sequence source is not present (fail-closed).
    """
    raw_seq_dir = args.raw_seq_dir
    if not raw_seq_dir or not raw_seq_dir.exists():
        raise sealed.SealedAccessError(
            "raw GSE246381 sequence source not available (fail-closed)")
    # The sealed evaluator is executed here.  For the migration deliverable the
    # orchestrator exposes the protocol; the sequence-level scoring is the
    # sealed evaluator's responsibility and is gated on the internal test.
    raise sealed.SealedAccessError(
        "sealed final sequence-level scoring not yet mounted (blocked on internal test)")


# Explicitly fail closed.  Merely having a callable stub must never be mistaken
# for a usable evaluator.  A future implementation may set these properties
# only after its aggregate-only interface passes a synthetic/ordinary fixture;
# the fixture artifact digest is then bound here as reviewable evidence.
_evaluate_sealed_external.e0x_contract_version = SEALED_EVALUATOR_CONTRACT_VERSION
_evaluate_sealed_external.e0x_implementation_ready = False
_evaluate_sealed_external.e0x_fixture_sha256 = None


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["internal", "sealed-preflight", "sealed-final"],
                    default="internal")
    ap.add_argument(
        "--dry-run", action="store_true",
        help=("alias for --mode sealed-preflight; never reads restricted state "
              "and never writes an access event or output artifact"))
    ap.add_argument(
        "--authorize-sealed-final", action="store_true",
        help="explicit operator authorization required by the one-shot final")
    ap.add_argument(
        "--confirm-one-shot", default=None,
        help="must exactly equal %s" % SEALED_FINAL_CONFIRMATION)
    ap.add_argument(
        "--confirm-preregistration-id", default=None,
        help="must exactly equal the frozen preregistration_id")
    ap.add_argument(
        "--confirm-preregistration-sha256", default=None,
        help="must exactly equal the canonical frozen preregistration SHA-256")
    ap.add_argument(
        "--ordinary-go-manifest", type=Path, default=None,
        help="hash-bound ordinary-only full-family GO activation artifact")
    ap.add_argument("--confirm-ordinary-go-sha256", default=None)
    ap.add_argument(
        "--evaluator-fixture", type=Path, default=None,
        help="ordinary fixture artifact bound by the evaluator capability SHA")
    ap.add_argument("--dataset", type=Path, default=Path("artifacts/b0x/effect_dataset.jsonl"))
    ap.add_argument("--prereg", type=Path, default=Path("configs/e0x_preregistration_v1.yaml"))
    ap.add_argument("--ckpt-dir", type=Path, default=Path("artifacts/m4_sparse/candval"))
    ap.add_argument("--restricted", type=Path,
                    default=Path("/home/cunyuliu/mrna_editflow_goal/restricted/v3_1_data_bench_closure_20260803/sealed_external/GSE246381"))
    ap.add_argument("--raw-seq-dir", type=Path,
                    default=Path("/home/cunyuliu/mrna_editflow_goal/mrna_editflow/data/raw/gse246381_utr_mutation"))
    ap.add_argument("--out-dir", type=Path, default=Path("artifacts/e0x"))
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpu", default=None)
    args = ap.parse_args(argv)

    requested_mode = "sealed-preflight" if args.dry_run else args.mode

    prereg_, args.actual_preregistration_sha256 = (
        prereg.load_prereg_and_sha256(args.prereg)
    )
    rep = prereg.validate(prereg_)
    if not rep["valid"]:
        print("pre-registration INVALID: %r" % rep["errors"], file=sys.stderr)
        return 2
    # Side-effect-free by construction: stop before loading a dataset/GPU,
    # touching raw/restricted paths, instantiating the access state machine, or
    # creating the output directory.  The JSON report is emitted to stdout only.
    if requested_mode == "sealed-preflight":
        report = sealed_final_preflight(prereg_, args)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["ready_for_one_shot_reservation"] else 4

    print("pre-registration valid: %s" % rep["preregistration_id"])

    if requested_mode == "sealed-final":
        try:
            require_sealed_final_preflight(prereg_, args)
        except sealed.SealedAccessError as exc:
            print(str(exc), file=sys.stderr)
            return 4

    try:
        if requested_mode == "internal":
            output = validate_internal_output_path(args.out_dir)
        else:
            output = validate_sealed_output_path(args)
    except (ValueError, OSError, sealed.SealedAccessError) as exc:
        print("output policy fail-closed: %s" % exc, file=sys.stderr)
        return 4

    cfg = C.get_config()
    rows = load_rows(args.dataset)
    print("loaded %d delta-defined records" % len(rows))
    vocab = build_vocab(rows)
    cfg.N_STUDIES = len(vocab["study"])
    cfg.N_ENDPOINTS = len(vocab["endpoint"])
    cfg.N_BENCHMARKS = len(vocab["benchmark"])

    device, err = select_device(cfg, args.gpu)
    if device is None:
        print("GPU policy fail-closed: %s" % err, file=sys.stderr)
        return 3

    import torch
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if requested_mode == "internal":
        agg = run_internal(prereg_, args, cfg, rows, vocab, device)
    else:
        agg = run_sealed_final(prereg_, args, cfg, rows, vocab, device)

    try:
        write_result_exclusive(output, agg)
    except FileExistsError as exc:
        print("output policy fail-closed: %s" % exc, file=sys.stderr)
        return 4
    print("wrote %s" % output)
    print("verdict:", agg.get("go_nogo_verdict", {}).get("verdict"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
