"""P2-05: RL-2 Group-Normalized Policy Gradient (GRPO) pilot entry point.

This script wires up the GRPO trainer (rl/grpo.py) with:
  * split-contract enforcement (--train-idx/--val-idx/--test-idx required)
  * Oracle #3 manifest verification (--oracle-manifest required in paper mode)
  * warm-start from a P2-02 Stage A checkpoint (--checkpoint)
  * optional reference policy for KL penalty (--ref-checkpoint, defaults to
    --checkpoint)
  * KL penalty and entropy bonus coefficients (--kl-coef, --entropy-coef)
  * trajectory JSONL + checkpoint + provenance sidecar outputs

Reward signal: ``delta predicted TE`` from Oracle #3 (P1-05 GBT regressor).
This is a *predicted / internal proxy* for translation efficiency. All log
lines and the results JSON use the ``predicted_te_internal_proxy`` field
name, per project constraint (any "improves TE" claim must be qualified
until P2-01 multi-region oracle validation completes).

Usage (development mode, preliminary):
    python -m scripts.run_p2_05_grpo_pilot \\
        --checkpoint benchmark/paper/stage_a_recovery_p2_02_seed42/stage_a_step10000.pt \\
        --checkpoint-sha256 <sha256_of_checkpoint> \\
        --oracle-manifest benchmark/paper/leakage_free_headline/oracle_manifest.json \\
        --records-jsonl data/reconstructed/p0_data_reconstruction_v1/combined/combined_model_view.records.jsonl \\
        --split-manifest benchmark/dev/p0_data_reconstruction_v1/combined_family/split_manifest.json \\
        --split-role train \\
        --train-idx benchmark/dev/p0_data_reconstruction_v1/combined_family/train.idx \\
        --val-idx   benchmark/dev/p0_data_reconstruction_v1/combined_family/val.idx \\
        --test-idx  benchmark/dev/p0_data_reconstruction_v1/combined_family/test.idx \\
        --task cds \\
        --group-size 8 \\
        --kl-coef 0.05 \\
        --entropy-coef 0.01 \\
        --n-iter 500 \\
        --n-groups 4 \\
        --policy-seed 0 \\
        --rollout-seeds 0 1 2 3 4 5 6 7 8 9 \\
        --out-dir benchmark/dev/grpo_pilot_preliminary/cds_seed0 \\
        --device cuda:2 \\
        --run-mode development

Status (2026-07-21):
  * Split-contract enforcement: IMPLEMENTED + tested.
  * Oracle-manifest verification: IMPLEMENTED + tested.
  * CLI parsing + arg validation: IMPLEMENTED + tested.
  * Checkpoint SHA-256 verification: IMPLEMENTED + tested.
  * GRPO trainer wiring (KL + entropy + ref_policy): IMPLEMENTED (in rl/grpo.py).
  * Real mRNA MDP construction (full 5'UTR/CDS/3'UTR + Oracle #3 reward):
    PENDING — depends on P2-02 10k checkpoint structure. The
    ``_build_real_mdp`` function below is a placeholder that raises
    NotImplementedError; it will be filled in once P2-02 completes and
    the checkpoint's model/backbone interface is verified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PACKAGE_PARENT = _REPO_ROOT.parent
for _p in (str(_PACKAGE_PARENT), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mrna_editflow.core.schema import MRNARecord
from mrna_editflow.data.download_mrna import load_records_jsonl
from mrna_editflow.data.split_contract import (
    VerifiedSplitContract,
    load_and_verify_split_manifest,
)
from mrna_editflow.eval.artifact_contract import (
    build_run_metadata,
    load_and_verify_oracle_manifest,
    normalize_run_mode,
    require_paper_cli_inputs,
    validate_output_namespace,
)
from mrna_editflow.rl.grpo import GRPOConfig, GRPOREINFORCE
from mrna_editflow.rl.policy import Policy, PolicyConfig
from mrna_editflow.rl.real_mdp import OracleLike, RealMRNAMDP
from mrna_editflow.rl.tiny_mdp import TinyMDP


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class P205ContractError(ValueError):
    """Split-contract or oracle-manifest verification failure."""


class P205CheckpointError(ValueError):
    """Checkpoint SHA-256 mismatch or file-not-found."""


class P205MDPNotReadyError(RuntimeError):
    """Real mRNA MDP construction is pending P2-02 checkpoint completion."""


# ---------------------------------------------------------------------------
# SHA-256 verification
# ---------------------------------------------------------------------------


def sha256_file(path: str | Path) -> str:
    """Compute SHA-256 of a file (64-char hex digest)."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checkpoint_sha256(
    checkpoint_path: str,
    expected_sha256: Optional[str],
) -> str:
    """Verify the SHA-256 of ``checkpoint_path`` against ``expected_sha256``.

    Returns the computed SHA-256. Raises ``P205CheckpointError`` if the file
    is missing or the digest does not match.
    """
    if not checkpoint_path:
        raise P205CheckpointError("--checkpoint is required")
    if not Path(checkpoint_path).exists():
        raise P205CheckpointError(
            f"checkpoint not found: {checkpoint_path} "
            "(P2-02 10k checkpoint not yet produced?)"
        )
    computed = sha256_file(checkpoint_path)
    if expected_sha256:
        expected = expected_sha256.lower()
        if computed.lower() != expected:
            raise P205CheckpointError(
                f"checkpoint SHA-256 mismatch: expected {expected}, "
                f"computed {computed}"
            )
    return computed


# ---------------------------------------------------------------------------
# Split-contract verification
# ---------------------------------------------------------------------------


def verify_split_contract_cli(
    args: argparse.Namespace,
) -> VerifiedSplitContract:
    """Load and verify the split contract from --split-manifest.

    Also verifies that --train-idx/--val-idx/--test-idx, if provided, match
    the contract indices exactly. Raises ``P205ContractError`` on mismatch.
    """
    if not args.split_manifest:
        raise P205ContractError(
            "--split-manifest is required (split-contract enforcement)"
        )
    contract = load_and_verify_split_manifest(
        args.split_manifest, records_path=args.records_jsonl
    )
    # Verify provided idx files match the contract.
    idx_paths = {
        "train": args.train_idx,
        "val": args.val_idx,
        "test": args.test_idx,
    }
    for role, path_str in idx_paths.items():
        if path_str is None:
            raise P205ContractError(
                f"--{role}-idx is required (split-contract enforcement)"
            )
        path = Path(path_str)
        if not path.exists():
            raise P205ContractError(f"--{role}-idx file not found: {path}")
        with path.open("r") as fh:
            indices = [int(line.strip()) for line in fh if line.strip()]
        if not indices:
            raise P205ContractError(f"--{role}-idx file is empty: {path}")
        contract_indices = contract.roles[role].indices
        if len(indices) != len(contract_indices):
            raise P205ContractError(
                f"--{role}-idx has {len(indices)} indices but split contract "
                f"has {len(contract_indices)} for role '{role}'; mismatch."
            )
        if set(indices) != set(contract_indices):
            raise P205ContractError(
                f"--{role}-idx indices do not match split contract for role "
                f"'{role}'."
            )
    return contract


# ---------------------------------------------------------------------------
# Oracle-manifest verification
# ---------------------------------------------------------------------------


def verify_oracle_manifest_cli(
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Verify the Oracle #3 manifest.

    In paper mode, --oracle-manifest is required. In development mode, it is
    optional but recommended.
    """
    return load_and_verify_oracle_manifest(
        args.oracle_manifest, run_mode=args.run_mode
    )


# ---------------------------------------------------------------------------
# Output namespace verification
# ---------------------------------------------------------------------------


def verify_output_namespace_cli(args: argparse.Namespace) -> str:
    """Verify the output directory namespace is allowed for the run mode."""
    return validate_output_namespace(args.out_dir, run_mode=args.run_mode)


# ---------------------------------------------------------------------------
# GRPO config construction
# ---------------------------------------------------------------------------


@dataclass
class P205RunConfig:
    """Parsed and validated P2-05 run configuration."""

    checkpoint_path: str
    checkpoint_sha256: str
    ref_checkpoint_path: str
    ref_checkpoint_sha256: str
    oracle_manifest_path: Optional[str]
    oracle_metadata: Dict[str, Any]
    split_contract: VerifiedSplitContract
    split_role: str
    task: str  # "cds" or "five_utr"
    group_size: int
    kl_coef: float
    entropy_coef: float
    lr: float
    n_iter: int
    n_groups: int
    policy_seed: int
    rollout_seeds: Tuple[int, ...]
    out_dir: str
    device: str
    run_mode: str
    limit: int
    records_count: int
    max_steps: Optional[int] = None

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_sha256": self.checkpoint_sha256,
            "ref_checkpoint_path": self.ref_checkpoint_path,
            "ref_checkpoint_sha256": self.ref_checkpoint_sha256,
            "oracle_manifest_path": self.oracle_manifest_path,
            "oracle_artifact_sha256": self.oracle_metadata.get("artifact_sha256"),
            "split_manifest_path": self.split_contract.manifest_path,
            "split_manifest_sha256": self.split_contract.manifest_sha256,
            "split_role": self.split_role,
            "task": self.task,
            "group_size": self.group_size,
            "kl_coef": self.kl_coef,
            "entropy_coef": self.entropy_coef,
            "lr": self.lr,
            "n_iter": self.n_iter,
            "n_groups": self.n_groups,
            "policy_seed": self.policy_seed,
            "rollout_seeds": list(self.rollout_seeds),
            "out_dir": self.out_dir,
            "device": self.device,
            "run_mode": self.run_mode,
            "limit": self.limit,
            "max_steps": self.max_steps,
            "records_count": self.records_count,
            "reward_field": "predicted_te_internal_proxy",
        }


def build_run_config(args: argparse.Namespace) -> P205RunConfig:
    """Parse args, verify contracts, and build a P205RunConfig."""
    args.run_mode = normalize_run_mode(args.run_mode)
    require_paper_cli_inputs(
        run_mode=args.run_mode,
        split_manifest=args.split_manifest,
        split_role=args.split_role,
        allowed_roles=("train",),
        oracle_manifest=args.oracle_manifest,
        require_oracle=(args.run_mode == "paper"),
    )
    # Split contract.
    split_contract = verify_split_contract_cli(args)
    # Oracle manifest.
    oracle_metadata = verify_oracle_manifest_cli(args)
    # Output namespace.
    verify_output_namespace_cli(args)
    # Checkpoint SHA-256.
    ckpt_sha = verify_checkpoint_sha256(args.checkpoint, args.checkpoint_sha256)
    ref_ckpt_path = args.ref_checkpoint or args.checkpoint
    ref_ckpt_sha_expected = args.ref_checkpoint_sha256 or args.checkpoint_sha256
    ref_ckpt_sha = verify_checkpoint_sha256(ref_ckpt_path, ref_ckpt_sha_expected)
    # Records count.
    if args.records_jsonl:
        records = load_records_jsonl(args.records_jsonl)
        records_count = len(records)
    else:
        records_count = 0
    return P205RunConfig(
        checkpoint_path=args.checkpoint,
        checkpoint_sha256=ckpt_sha,
        ref_checkpoint_path=ref_ckpt_path,
        ref_checkpoint_sha256=ref_ckpt_sha,
        oracle_manifest_path=args.oracle_manifest,
        oracle_metadata=oracle_metadata,
        split_contract=split_contract,
        split_role=args.split_role,
        task=args.task,
        group_size=args.group_size,
        kl_coef=args.kl_coef,
        entropy_coef=args.entropy_coef,
        lr=args.lr,
        n_iter=args.n_iter,
        n_groups=args.n_groups,
        policy_seed=args.policy_seed,
        rollout_seeds=tuple(args.rollout_seeds),
        out_dir=args.out_dir,
        device=args.device,
        run_mode=args.run_mode,
        limit=args.limit,
        max_steps=args.max_steps,
        records_count=records_count,
    )


# ---------------------------------------------------------------------------
# Real mRNA MDP construction
# ---------------------------------------------------------------------------


def _load_records_from_split(
    run_config: P205RunConfig,
    limit: Optional[int] = None,
) -> List[MRNARecord]:
    """Load records from the split contract's records_path and select via train.idx.

    Returns a list of :class:`MRNARecord` indexed by the split contract's
    train role. If ``limit`` is provided, only the first ``limit`` records
    are returned (deterministic first-N).
    """
    records_path = run_config.split_contract.records_path
    records = load_records_jsonl(records_path)
    # The split contract's train.idx indexes into the full records list.
    train_role = run_config.split_contract.roles.get("train")
    if train_role is None:
        raise P205ContractError(
            "split contract missing train role; "
            "cannot select training records"
        )
    train_idx = train_role.indices
    selected: List[MRNARecord] = []
    for i in train_idx:
        if 0 <= i < len(records):
            selected.append(records[i])
        if limit is not None and len(selected) >= limit:
            break
    return selected


def _load_oracle_from_manifest(
    run_config: P205RunConfig,
) -> Optional[OracleLike]:
    """Load Oracle #3 (GBT regressor) from the manifest path.

    Returns ``None`` if no oracle manifest is configured (allowed in
    development mode). In paper mode, ``build_run_config`` already
    enforces that the manifest is provided and verified.

    The oracle is loaded lazily (inside this function) so that the module
    can be imported without a hard dependency on ``lightgbm``.
    """
    if not run_config.oracle_metadata:
        return None
    artifact_path = run_config.oracle_metadata.get("artifact_path")
    if not artifact_path:
        return None
    artifact_dir = str(Path(artifact_path).parent)
    # Lazy import: avoids hard lightgbm dependency at module load time.
    from models.oracle_final.gbt_regressor import GBTOracle  # type: ignore

    return GBTOracle.load(Path(artifact_dir))


def _load_policy_from_checkpoint(
    run_config: P205RunConfig,
    device: torch.device,
) -> Policy:
    """Load a Stage A checkpoint into a Policy for GRPO warm-start.

    Reconstructs the (backbone, model) pair via ``build_stage_a_model``
    from ``train_backbone.py`` (the same builder used by
    ``eval_stage_a_heldout.py``), then loads the checkpoint's
    ``model_state`` and (if present) ``backbone_state`` with
    ``strict=False`` (key mismatches are logged as warnings, since the
    Stage A checkpoint may contain auxiliary heads that the RL policy
    does not need).

    Parameters
    ----------
    run_config : P205RunConfig
        Carries ``checkpoint_path`` (already SHA-256-verified by
        ``build_run_config``).
    device : torch.device
        Where to place the model.

    Returns
    -------
    Policy
        A ready-to-train ``Policy`` wrapping the warm-started Stage A
        model. The model is in ``train()`` mode; the backbone is in
        ``eval()`` mode (frozen).

    Raises
    ------
    P205CheckpointError
        If the checkpoint file is missing, lacks required keys
        (``config`` or ``model_state``), or ``train_backbone`` cannot
        be imported.
    """
    ckpt_path = run_config.checkpoint_path
    if not ckpt_path or not Path(ckpt_path).exists():
        raise P205CheckpointError(
            f"checkpoint not found: {ckpt_path} "
            "(P2-02 10k checkpoint not yet produced?)"
        )

    # Lazy import: train_backbone is a top-level script at the repo root,
    # not a package module. The repo root is already on sys.path via the
    # _REPO_ROOT insertion at the top of this file.
    try:
        from train_backbone import _coerce_config, build_stage_a_model  # type: ignore
    except ImportError as exc:
        raise P205CheckpointError(
            f"cannot import build_stage_a_model from train_backbone: {exc}. "
            "Ensure the repo root is on PYTHONPATH."
        ) from exc

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "config" not in ckpt:
        raise P205CheckpointError(
            f"checkpoint missing required 'config' field: {ckpt_path}"
        )
    if "model_state" not in ckpt:
        raise P205CheckpointError(
            f"checkpoint missing required 'model_state' field: {ckpt_path}"
        )
    cfg = _coerce_config(ckpt["config"])
    # Disable aux struct supervision for RL: it requires an explicit target
    # tensor that we do not have at RL training time. We only need the
    # generation head for action sampling.
    cfg.model.use_aux_struct = False

    backbone, model = build_stage_a_model(cfg, device)
    missing, unexpected = model.load_state_dict(
        ckpt["model_state"], strict=False
    )
    if missing:
        print(
            f"[warn] model_state missing keys: {missing[:5]}"
            f"{'...' if len(missing) > 5 else ''}",
            file=sys.stderr,
        )
    if unexpected:
        print(
            f"[warn] model_state unexpected keys: {unexpected[:5]}"
            f"{'...' if len(unexpected) > 5 else ''}",
            file=sys.stderr,
        )

    # Backbone state may or may not be present (frozen vs. trainable).
    backbone_state = ckpt.get("backbone_state")
    if backbone_state:
        missing_b, unexpected_b = backbone.load_state_dict(
            backbone_state, strict=False
        )
        if missing_b:
            print(
                f"[warn] backbone_state missing keys: {missing_b[:5]}"
                f"{'...' if len(missing_b) > 5 else ''}",
                file=sys.stderr,
            )
        if unexpected_b:
            print(
                f"[warn] backbone_state unexpected keys: {unexpected_b[:5]}"
                f"{'...' if len(unexpected_b) > 5 else ''}",
                file=sys.stderr,
            )

    model.train()    # RL training mode (gradients flow through the model)
    backbone.eval()  # backbone is frozen
    return Policy(
        model=model,
        backbone=backbone,
        cfg=PolicyConfig(),
        device=device,
    )


def _build_real_mdp(
    run_config: P205RunConfig,
    records: Sequence[MRNARecord],
    oracle: Optional[OracleLike],
) -> RealMRNAMDP:
    """Build a real mRNA MDP for the GRPO pilot.

    The MDP takes the first record from ``records`` as the initial state
    and uses ``oracle`` to compute the terminal reward
    (``delta predicted_te_internal_proxy``). The MDP implements the same
    interface as :class:`TinyMDP`, so :class:`GRPOREINFORCE` can be used
    unchanged.

    Parameters
    ----------
    run_config : P205RunConfig
        Carries ``task`` (region), ``limit``, etc.
    records : Sequence[MRNARecord]
        Records selected from the split contract's train role. Must be
        non-empty.
    oracle : OracleLike or None
        The reward oracle (e.g. :class:`GBTOracle`). If ``None``, raises
        :class:`P205MDPNotReadyError`.

    Raises
    ------
    P205MDPNotReadyError
        If ``records`` is empty or ``oracle`` is None.
    """
    if not records:
        raise P205MDPNotReadyError(
            "no records available for MDP construction; "
            "split contract train role is empty or --limit is 0"
        )
    if oracle is None:
        raise P205MDPNotReadyError(
            "no oracle available for MDP construction; "
            "provide --oracle-manifest (Oracle #3) or run in development "
            "mode with a mock oracle (not supported in the entry point)"
        )
    # Map task -> region. "cds" and "five_utr" are the two supported tasks
    # per the P2-05 spec ("先 D2 CDS, 再 D1 5'UTR").
    task_to_region = {
        "cds": "cds",
        "five_utr": "five_utr",
    }
    region = task_to_region.get(run_config.task, "full")
    initial_record = records[0]
    return RealMRNAMDP(
        initial_record=initial_record,
        oracle=oracle,
        max_steps=(run_config.max_steps if run_config.max_steps is not None
                  else (run_config.limit if run_config.limit > 0 else 3)),
        gamma=0.99,
        region=region,
        reward_field="predicted_te_internal_proxy",
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def run_grpo_pilot(run_config: P205RunConfig) -> Dict[str, Any]:
    """Run the GRPO pilot.

    Returns a dict with:
      * ``metadata``: the run metadata (for provenance sidecar).
      * ``trajectory_jsonl``: path to the trajectory JSONL.
      * ``checkpoint_path``: path to the final policy checkpoint.
      * ``curves``: path to the training-curve JSONL.
      * ``status``: "completed" | "mdp_not_ready" | "failed".
    """
    out_dir = Path(run_config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trajectory_jsonl = out_dir / "trajectories.jsonl"
    curves_jsonl = out_dir / "curves.jsonl"
    final_ckpt = out_dir / "policy_final.pt"
    metadata_path = out_dir / "run_metadata.json"

    # Write metadata sidecar first (provenance).
    metadata = run_config.to_metadata()
    metadata["start_time"] = time.time()
    metadata["status"] = "starting"
    with metadata_path.open("w") as fh:
        json.dump(metadata, fh, indent=2, sort_keys=True)

    # Build GRPO config.
    grpo_cfg = GRPOConfig(
        group_size=run_config.group_size,
        eps=1e-8,
        clip_advantage=0.0,
        kl_coef=run_config.kl_coef,
        entropy_coef=run_config.entropy_coef,
    )

    # Stage 1: Load records from the split contract.
    try:
        records = _load_records_from_split(run_config, limit=1)
    except (P205ContractError, FileNotFoundError, OSError) as exc:
        metadata["status"] = "mdp_not_ready"
        metadata["error"] = f"records loading failed: {exc}"
        metadata["end_time"] = time.time()
        with metadata_path.open("w") as fh:
            json.dump(metadata, fh, indent=2, sort_keys=True)
        return {
            "metadata": metadata,
            "trajectory_jsonl": str(trajectory_jsonl),
            "checkpoint_path": str(final_ckpt),
            "curves": str(curves_jsonl),
            "status": "mdp_not_ready",
            "error": str(exc),
        }

    # Stage 2: Load Oracle #3 from the manifest (may be None in dev mode).
    try:
        oracle = _load_oracle_from_manifest(run_config)
    except Exception as exc:
        metadata["status"] = "mdp_not_ready"
        metadata["error"] = f"oracle loading failed: {exc}"
        metadata["end_time"] = time.time()
        with metadata_path.open("w") as fh:
            json.dump(metadata, fh, indent=2, sort_keys=True)
        return {
            "metadata": metadata,
            "trajectory_jsonl": str(trajectory_jsonl),
            "checkpoint_path": str(final_ckpt),
            "curves": str(curves_jsonl),
            "status": "mdp_not_ready",
            "error": str(exc),
        }

    # Stage 3: Build the real MDP (records + oracle).
    try:
        mdp = _build_real_mdp(run_config, records=records, oracle=oracle)
    except P205MDPNotReadyError as exc:
        metadata["status"] = "mdp_not_ready"
        metadata["error"] = str(exc)
        metadata["end_time"] = time.time()
        with metadata_path.open("w") as fh:
            json.dump(metadata, fh, indent=2, sort_keys=True)
        return {
            "metadata": metadata,
            "trajectory_jsonl": str(trajectory_jsonl),
            "checkpoint_path": str(final_ckpt),
            "curves": str(curves_jsonl),
            "status": "mdp_not_ready",
            "error": str(exc),
        }

    # Stage 4: Load the policy from the P2-02 checkpoint.
    # _load_policy_from_checkpoint is now implemented (v5); it raises
    # P205CheckpointError on missing file / missing keys / import failure,
    # and returns a ready-to-train Policy on success. We catch both
    # P205MDPNotReadyError (legacy) and P205CheckpointError here so that
    # any policy-loading failure still returns "mdp_not_ready" status
    # (preserving the contract that TestRunGrpoPilotMDPNotReady verifies).
    device = torch.device(run_config.device)
    try:
        policy = _load_policy_from_checkpoint(run_config, device)
    except (P205MDPNotReadyError, P205CheckpointError) as exc:
        metadata["status"] = "mdp_not_ready"
        metadata["error"] = str(exc)
        metadata["mdp_metadata"] = mdp.to_metadata()
        metadata["end_time"] = time.time()
        with metadata_path.open("w") as fh:
            json.dump(metadata, fh, indent=2, sort_keys=True)
        return {
            "metadata": metadata,
            "trajectory_jsonl": str(trajectory_jsonl),
            "checkpoint_path": str(final_ckpt),
            "curves": str(curves_jsonl),
            "status": "mdp_not_ready",
            "error": str(exc),
        }

    # Stage 5: Build reference policy (for KL penalty).
    ref_policy = policy
    if run_config.ref_checkpoint_path and run_config.ref_checkpoint_path != run_config.checkpoint_path:
        ref_policy = _load_policy_from_checkpoint(run_config, device)

    # Stage 6: Construct GRPO trainer and run training loop.
    trainer = GRPOREINFORCE(
        policy=policy,
        mdp=mdp,
        cfg=grpo_cfg,
        lr=run_config.lr,
        ref_policy=ref_policy,
    )
    torch.manual_seed(run_config.policy_seed)

    curves: List[Dict[str, Any]] = []
    with trajectory_jsonl.open("w") as traj_fh, curves_jsonl.open("w") as curves_fh:
        for it in range(run_config.n_iter):
            groups = [
                trainer.collect_group(generator=_make_generator(run_config.policy_seed + it * 1000 + g, device=device))
                for g in range(run_config.n_groups)
            ]
            metrics = trainer.step(groups)
            metrics["iter"] = it
            curves.append(metrics)
            curves_fh.write(json.dumps(metrics) + "\n")
            curves_fh.flush()
            for g_idx, group in enumerate(groups):
                traj_fh.write(json.dumps({
                    "iter": it,
                    "group": g_idx,
                    "transitions": [
                        {
                            "step": t.step,
                            "action": _action_to_dict(t.action),
                            "reward": t.reward,
                        }
                        for t in group[0].transitions  # log first trajectory per group
                    ],
                }) + "\n")
            traj_fh.flush()

        # Save final checkpoint.
        torch.save(
            {
                "stage": "RL-2-GRPO",
                "iter": run_config.n_iter,
                "policy_state": policy.model.state_dict(),
                "grpo_config": {
                    "group_size": grpo_cfg.group_size,
                    "kl_coef": grpo_cfg.kl_coef,
                    "entropy_coef": grpo_cfg.entropy_coef,
                },
                "run_metadata": metadata,
            },
            final_ckpt,
        )

    metadata["status"] = "completed"
    metadata["end_time"] = time.time()
    with metadata_path.open("w") as fh:
        json.dump(metadata, fh, indent=2, sort_keys=True)
    return {
        "metadata": metadata,
        "trajectory_jsonl": str(trajectory_jsonl),
        "checkpoint_path": str(final_ckpt),
        "curves": str(curves_jsonl),
        "status": "completed",
    }


def _make_generator(seed: int, device: Optional[torch.device] = None) -> torch.Generator:
    """Create a torch.Generator seeded with the given seed on the specified device.

    The generator device must match the device of tensors it's used with
    (e.g., torch.multinomial requires generator and probs on the same device).
    Defaults to CPU for backward compatibility.
    """
    dev = device if device is not None else torch.device("cpu")
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed)
    return gen


def _action_to_dict(action) -> Dict[str, Any]:
    """Serialize an Action to a JSON-compatible dict."""
    if action.is_stop():
        return {"op": "stop", "pos": -1, "nt": -1}
    return {"op": action.op, "pos": action.pos, "nt": action.nt}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="P2-05: RL-2 GRPO pilot (group-normalized PG + KL + entropy)"
    )
    parser.add_argument("--checkpoint", required=True,
        help="Path to P2-02 Stage A 10k checkpoint (.pt).")
    parser.add_argument("--checkpoint-sha256", default=None,
        help="Expected SHA-256 of --checkpoint. If omitted, computed and logged.")
    parser.add_argument("--ref-checkpoint", default=None,
        help="Reference policy checkpoint for KL penalty. Defaults to --checkpoint.")
    parser.add_argument("--ref-checkpoint-sha256", default=None,
        help="Expected SHA-256 of --ref-checkpoint. Defaults to --checkpoint-sha256.")
    parser.add_argument("--oracle-manifest", default=None,
        help="Path to Oracle #3 manifest JSON. Required in paper mode.")
    parser.add_argument("--records-jsonl", required=True,
        help="Path to combined_model_view.records.jsonl.")
    parser.add_argument("--split-manifest", default=None,
        help="Path to split_manifest.json.")
    parser.add_argument("--split-role", choices=("train", "val", "test"),
        default="train",
        help="Split role to sample initial states from. Default: train.")
    parser.add_argument("--train-idx", default=None,
        help="Path to train.idx (required).")
    parser.add_argument("--val-idx", default=None,
        help="Path to val.idx (required).")
    parser.add_argument("--test-idx", default=None,
        help="Path to test.idx (required).")
    parser.add_argument("--task", choices=("cds", "five_utr"), default="cds",
        help="Edit region: 'cds' (D2) or 'five_utr' (D1). Default: cds.")
    parser.add_argument("--group-size", type=int, default=8,
        help="GRPO group size N (trajectories per starting state). Default: 8.")
    parser.add_argument("--kl-coef", type=float, default=0.0,
        help="KL penalty coefficient. 0 disables. Default: 0.0.")
    parser.add_argument("--entropy-coef", type=float, default=0.0,
        help="Entropy bonus coefficient. 0 disables. Default: 0.0.")
    parser.add_argument("--lr", type=float, default=0.01,
        help="SGD learning rate. Default: 0.01.")
    parser.add_argument("--n-iter", type=int, default=500,
        help="Number of GRPO gradient steps. Default: 500.")
    parser.add_argument("--n-groups", type=int, default=4,
        help="Number of groups per gradient step. Default: 4.")
    parser.add_argument("--policy-seed", type=int, default=0,
        help="Policy/training seed. Default: 0.")
    parser.add_argument("--rollout-seeds", type=int, nargs="+",
        default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        help="Rollout seeds. Default: 0..9.")
    parser.add_argument("--out-dir", required=True,
        help="Output directory for trajectories, checkpoints, curves.")
    parser.add_argument("--device", default="cuda:0",
        help="Torch device. Default: cuda:0.")
    parser.add_argument("--run-mode", choices=("development", "paper"),
        default="development",
        help="Run mode. 'paper' requires oracle-manifest + paper-eligible ckpt.")
    parser.add_argument("--max-steps", type=int, default=None,
        help="Max MDP steps per trajectory. If None, uses --limit (backward-compatible). "
             "Decouple from --limit to control memory: each step keeps a forward-pass "
             "computation graph, so n_groups * group_size * max_steps must fit in GPU memory.")
    parser.add_argument("--limit", type=int, default=1024,
        help="Max number of source sequences to sample. Default: 1024.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    run_config = build_run_config(args)
    result = run_grpo_pilot(run_config)
    print(json.dumps({
        "status": result["status"],
        "out_dir": run_config.out_dir,
        "checkpoint_sha256": run_config.checkpoint_sha256,
        "oracle_artifact_sha256": run_config.oracle_metadata.get("artifact_sha256"),
        "split_manifest_sha256": run_config.split_contract.manifest_sha256,
        "reward_field": "predicted_te_internal_proxy",
    }, sort_keys=True))
    if result["status"] == "mdp_not_ready":
        # Exit 0 so the wrapper script can continue; the metadata sidecar
        # records that the MDP is not ready.
        return 0
    if result["status"] == "failed":
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
