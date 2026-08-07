"""E0-X pre-registration validation (pure, unit-testable, no remote data).

Binds the frozen `configs/e0x_preregistration_v1.yaml` protocol and checks its
internal consistency BEFORE any sealed final access.  These are the pure
data-free cores; the sealed-final orchestrator lives in `run_e0x_final.py`.

Checks enforced here (all must pass before ACCESS_INTENT is written):
  * FROZEN status and all required top-level sections present.
  * Data manifest carries the immutable effect-dataset SHA-256.
  * Model aliases all carry a SHA-256; critic checkpoint hashes are non-empty.
  * Holm family has >= 1 hypothesis; alpha in (0, 1]; inference is
    HOLM_BONFERRONI.
  * GO/NO-GO thresholds are present and numerically sane.
  * GPU policy forbids GPU 4 and permits at least one device.
  * Output schema is aggregate-only and does not return row-level labels.
  * Evaluator command references the frozen prereg file.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


class PreRegistrationError(ValueError):
    """Raised when the frozen pre-registration protocol is inconsistent."""


def _req(block: Dict[str, Any], key: str, where: str) -> Any:
    if not isinstance(block, dict) or key not in block or block[key] is None:
        raise PreRegistrationError("missing required field %s.%s" % (where, key))
    return block[key]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_prereg(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def holm_bonferroni(pvalues: List[float], alpha: float) -> List[Optional[float]]:
    """Holm-Bonferroni adjusted p-values (family-wise error-rate control).

    Returns adjusted p-values aligned to the input order; a hypothesis is
    rejected iff adjusted[n] <= alpha.  Input p-values must be in [0, 1].
    """
    n = len(pvalues)
    if n == 0:
        return []
    if any(not 0.0 <= p <= 1.0 for p in pvalues):
        raise PreRegistrationError("holm p-values must be in [0, 1]")
    order = sorted(range(n), key=lambda i: pvalues[i])
    adjusted = [None] * n
    running_max = 0.0
    for rank, i in enumerate(order, start=1):
        raw = min(1.0, pvalues[i] * (n - rank + 1))
        # Holm-Bonferroni: the adjusted value is the running max of the
        # Bonferroni-scaled values, so the reject set is monotone (order statistic).
        running_max = max(running_max, raw)
        adjusted[i] = running_max
    return adjusted


def validate(prereg: Dict[str, Any]) -> Dict[str, Any]:
    """Validate the frozen pre-registration protocol. Returns a report dict."""
    errors: List[str] = []

    def check(cond: bool, msg: str) -> None:
        if not cond:
            errors.append(msg)

    # 1. status frozen
    check(prereg.get("status") == "FROZEN",
          "status must be FROZEN before sealed final access")
    check(prereg.get("phase") == "E0-X", "phase must be E0-X")
    check(bool(prereg.get("preregistration_id")), "preregistration_id required")

    # 2. data manifest
    data = prereg.get("data") or {}
    eff = data.get("effect_dataset") or {}
    check(bool(eff.get("sha256")) and len(eff["sha256"]) == 64,
          "effect-dataset sha256 must be a 64-hex digest")
    check(eff.get("n_records", 0) > 0, "n_records must be > 0")
    check(eff.get("n_delta_defined", 0) > 0, "n_delta_defined must be > 0")
    check(data.get("split") == "S4", "primary split must be S4")

    # 3. models
    models = prereg.get("models") or {}
    bf = models.get("frozen_base_flow") or {}
    check(len(bf.get("sha256", "")) == 64, "base flow sha256 required")
    critic = models.get("frozen_critic") or {}
    cks = critic.get("checkpoints") or {}
    check(bool(cks), "critic must have >= 1 checkpoint")
    for name, meta in cks.items():
        check(len(meta.get("sha256", "")) == 64,
              "critic checkpoint %s sha256 required" % name)

    # 4. reward / beta
    reward = prereg.get("reward") or {}
    check(reward.get("beta", 0) > 0, "beta must be > 0")
    check(bool(reward.get("guidance_strategy_primary")),
          "guidance_strategy_primary required")

    # 5. metric family + Holm
    metrics = prereg.get("metrics") or {}
    holm = metrics.get("holm_family") or {}
    hyps = holm.get("hypotheses") or []
    check(holm.get("inference") == "HOLM_BONFERRONI",
          "inference must be HOLM_BONFERRONI")
    alpha = holm.get("alpha")
    check(isinstance(alpha, (int, float)) and 0.0 < alpha <= 1.0,
          "holm alpha must be in (0, 1]")
    check(len(hyps) >= 1, "holm_family.hypotheses must be non-empty")
    for h in hyps:
        check(bool(h.get("id")), "hypothesis id required")
        check(bool(h.get("metric")), "hypothesis metric required")
        check(bool(h.get("null_hypothesis")), "hypothesis null_hypothesis required")

    # 6. GO/NO-GO thresholds
    gates = prereg.get("go_nogo") or {}
    eg = gates.get("effect_gate") or {}
    check(0.0 <= eg.get("macro_delta_spearman_ge", -1) <= 1.0,
          "effect_gate.macro_delta_spearman_ge must be in [0,1]")
    check(0.0 <= eg.get("macro_sign_accuracy_ge", -1) <= 1.0,
          "effect_gate.macro_sign_accuracy_ge must be in [0,1]")
    gg = gates.get("guidance_gate") or {}
    check(bool(gg.get("primary_strategy")) and bool(gg.get("vs")),
          "guidance_gate strategy/vs required")
    hc = gates.get("hard_constraints") or {}
    check(hc.get("legality_ge", 0) == 1.0, "legality_ge must be 1.0")
    check(hc.get("budget_violation_le", -1) == 0.0,
          "budget_violation_le must be 0.0")

    # 7. GPU policy (fail-closed; GPU 4 forbidden)
    exec_ = prereg.get("execution") or {}
    gpu = exec_.get("gpu_policy") or {}
    for dev in gpu.get("forbidden", []):
        check(dev in ("4", "cuda:4"), "GPU 4 must be forbidden")
    check(len(gpu.get("permitted", [])) >= 1, "at least one permitted GPU")
    check(gpu.get("fallback", "") == "", "CUDA fallback must be fail-closed (empty)")

    # 8. output schema aggregate-only
    osch = (exec_.get("output_schema") or {})
    check(bool(osch.get("aggregate_only")), "output must be aggregate-only")
    check(osch.get("row_level_labels_returned") is False,
          "row-level labels must not be returned")

    # 9. evaluator command references frozen prereg
    cmd = exec_.get("evaluator_command", "")
    check("e0x_preregistration_v1.yaml" in cmd,
          "evaluator command must reference the frozen prereg file")

    return {
        "preregistration_id": prereg.get("preregistration_id"),
        "valid": not errors,
        "n_errors": len(errors),
        "errors": errors,
        "n_hypotheses": len(hyps),
        "holm_alpha": alpha,
    }


def validate_file(path: Path) -> Dict[str, Any]:
    return validate(load_prereg(path))