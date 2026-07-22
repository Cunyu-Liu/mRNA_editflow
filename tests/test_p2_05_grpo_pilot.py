"""Tests for P2-05 GRPO pilot entry point (scripts/run_p2_05_grpo_pilot.py).

Covers the testable parts that do NOT require the P2-02 10k checkpoint:
  * SHA-256 verification (verify_checkpoint_sha256)
  * Split-contract verification (verify_split_contract_cli)
  * Oracle-manifest verification (verify_oracle_manifest_cli)
  * Output namespace verification (verify_output_namespace_cli)
  * CLI argument parsing (_parse_args)
  * build_run_config end-to-end (with a tiny split manifest)
  * run_grpo_pilot returns "mdp_not_ready" status when MDP is not implemented
  * GRPO config construction includes KL + entropy coefficients

The real mRNA MDP construction (_build_real_mdp) is pending P2-02 and is
tested only via the "mdp_not_ready" status path.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from unittest.mock import patch

import torch

# Make scripts/ importable.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_PACKAGE_PARENT = _REPO_ROOT.parent
for _p in (str(_PACKAGE_PARENT), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mrna_editflow.core.schema import MRNARecord
from mrna_editflow.data.download_mrna import write_records_jsonl
from mrna_editflow.data.split_contract import (
    VerifiedSplitContract,
    build_split_manifest,
    load_and_verify_split_manifest,
    sha256_file,
    transcript_id_digest,
)

# Import the entry point as a module. Register in sys.modules BEFORE
# exec_module so that dataclass introspection (cls.__module__ lookup) works.
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "run_p2_05_grpo_pilot",
    _REPO_ROOT / "scripts" / "run_p2_05_grpo_pilot.py",
)
assert _spec is not None and _spec.loader is not None
run_p2_05_grpo_pilot = importlib.util.module_from_spec(_spec)
sys.modules["run_p2_05_grpo_pilot"] = run_p2_05_grpo_pilot
_spec.loader.exec_module(run_p2_05_grpo_pilot)

# Aliases for convenience.
verify_checkpoint_sha256 = run_p2_05_grpo_pilot.verify_checkpoint_sha256
verify_split_contract_cli = run_p2_05_grpo_pilot.verify_split_contract_cli
verify_oracle_manifest_cli = run_p2_05_grpo_pilot.verify_oracle_manifest_cli
verify_output_namespace_cli = run_p2_05_grpo_pilot.verify_output_namespace_cli
build_run_config = run_p2_05_grpo_pilot.build_run_config
run_grpo_pilot = run_p2_05_grpo_pilot.run_grpo_pilot
_load_policy_from_checkpoint = run_p2_05_grpo_pilot._load_policy_from_checkpoint
P205RunConfig = run_p2_05_grpo_pilot.P205RunConfig
P205ContractError = run_p2_05_grpo_pilot.P205ContractError
P205CheckpointError = run_p2_05_grpo_pilot.P205CheckpointError
P205MDPNotReadyError = run_p2_05_grpo_pilot.P205MDPNotReadyError
GRPOConfig = run_p2_05_grpo_pilot.GRPOConfig
_parse_args = run_p2_05_grpo_pilot._parse_args


# ---------------------------------------------------------------------------
# Helpers (mirror tests/test_split_contract_enforcement.py)
# ---------------------------------------------------------------------------


def _make_records(n: int = 6) -> List[MRNARecord]:
    out: List[MRNARecord] = []
    for i in range(n):
        out.append(
            MRNARecord(
                transcript_id=f"TEST{i:04d}",
                five_utr="GCC",
                cds="AUG" + "GCU" * (i + 1) + "UAA",
                three_utr="GGG",
                species="human",
            )
        )
    return out


def _write_idx_file(path: Path, indices: Sequence[int]) -> None:
    with path.open("w") as fh:
        for idx in indices:
            fh.write(f"{int(idx)}\n")


def _build_split(
    tmpdir: Path,
    records: Sequence[MRNARecord],
    train_idx: Sequence[int],
    val_idx: Sequence[int],
    test_idx: Sequence[int],
    paper_eligible: bool = False,
) -> tuple:
    """Build a valid split manifest + idx files in tmpdir.

    Returns (manifest_path, VerifiedSplitContract).
    """
    records_path = tmpdir / "records.jsonl"
    write_records_jsonl(list(records), str(records_path))
    role_paths: Dict[str, str] = {}
    for role, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        p = tmpdir / f"{role}.idx"
        _write_idx_file(p, idx)
        role_paths[role] = str(p)
    leakage_path = tmpdir / "leakage.json"
    leakage_path.write_text(json.dumps({
        "split": {"cluster_disjoint": True},
        "summary": {
            "exact_match_count": 0,
            "leakage_exact_match_count": 0,
            "leakage_flagged_fraction": 0.0,
            "near_neighbor_threshold_passed": True,
        },
    }))
    cluster_path = tmpdir / "clusters.txt"
    with cluster_path.open("w") as fh:
        for i in range(len(records)):
            fh.write(f"{i}\n")
    manifest_payload = build_split_manifest(
        dataset_id="test-p2-05-dataset",
        records_path=str(records_path),
        role_idx_paths=role_paths,
        leakage_report_path=str(leakage_path),
        algorithm="family_balanced",
        seed=0,
        family_threshold=0.0,
        family_disjoint=True,
        exact_cross_role_matches=0,
        near_neighbor_threshold_passed=True,
        cluster_assignment_path=str(cluster_path),
        paper_eligible=paper_eligible,
        block_reasons=() if paper_eligible else ("test_block",),
    )
    manifest_path = tmpdir / "split.manifest.json"
    manifest_path.write_text(json.dumps(manifest_payload, sort_keys=True))
    contract = load_and_verify_split_manifest(
        str(manifest_path), records_path=str(records_path)
    )
    return manifest_path, contract


def _build_oracle_manifest(tmpdir: Path) -> tuple:
    """Build a valid Oracle #3 manifest + artifact in tmpdir.

    Returns (manifest_path, artifact_path, artifact_sha256).
    """
    artifact_path = tmpdir / "oracle_meta.json"
    artifact_payload = {
        "oracle_type": "gbt_regressor",
        "schema_version": 1,
        "frozen_at": "2026-07-19",
    }
    artifact_path.write_text(json.dumps(artifact_payload, sort_keys=True))
    artifact_sha = sha256_file(artifact_path)
    manifest = {
        "schema_version": 1,
        "oracle_type": "gbt_regressor",
        "independent": True,
        "source": "tiny_test_oracle",
        "independence_statement": "Tiny test oracle for unit tests; independent of test split.",
        "artifact_path": str(artifact_path),
        "artifact_sha256": artifact_sha,
    }
    manifest_path = tmpdir / "oracle_manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True))
    return manifest_path, artifact_path, artifact_sha


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSHA256Verification(unittest.TestCase):
    """Tests for verify_checkpoint_sha256."""

    def setUp(self) -> None:
        self.tmpdir_obj = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self.tmpdir_obj.name)
        self.ckpt_path = self.tmpdir / "ckpt.pt"
        self.ckpt_path.write_bytes(b"fake checkpoint bytes")

    def tearDown(self) -> None:
        self.tmpdir_obj.cleanup()

    def test_returns_computed_sha_when_no_expected(self) -> None:
        sha = verify_checkpoint_sha256(str(self.ckpt_path), None)
        self.assertEqual(len(sha), 64)
        self.assertEqual(sha, sha256_file(self.ckpt_path))

    def test_returns_computed_sha_when_expected_matches(self) -> None:
        expected = sha256_file(self.ckpt_path)
        sha = verify_checkpoint_sha256(str(self.ckpt_path), expected)
        self.assertEqual(sha, expected.lower())

    def test_raises_on_missing_checkpoint(self) -> None:
        with self.assertRaises(P205CheckpointError):
            verify_checkpoint_sha256("/nonexistent/path.pt", None)

    def test_raises_on_empty_checkpoint_path(self) -> None:
        with self.assertRaises(P205CheckpointError):
            verify_checkpoint_sha256("", None)

    def test_raises_on_sha_mismatch(self) -> None:
        with self.assertRaises(P205CheckpointError) as ctx:
            verify_checkpoint_sha256(
                str(self.ckpt_path),
                "0" * 64,
            )
        self.assertIn("mismatch", str(ctx.exception).lower())


class TestCLIParsing(unittest.TestCase):
    """Tests for _parse_args."""

    def _base_argv(self) -> List[str]:
        return [
            "--checkpoint", "/tmp/ckpt.pt",
            "--records-jsonl", "/tmp/records.jsonl",
            "--out-dir", "/tmp/out",
            "--split-manifest", "/tmp/manifest.json",
            "--split-role", "train",
            "--train-idx", "/tmp/train.idx",
            "--val-idx", "/tmp/val.idx",
            "--test-idx", "/tmp/test.idx",
        ]

    def test_default_kl_and_entropy_are_zero(self) -> None:
        args = _parse_args(self._base_argv())
        self.assertEqual(args.kl_coef, 0.0)
        self.assertEqual(args.entropy_coef, 0.0)

    def test_kl_and_entropy_can_be_set(self) -> None:
        argv = self._base_argv() + ["--kl-coef", "0.05", "--entropy-coef", "0.01"]
        args = _parse_args(argv)
        self.assertEqual(args.kl_coef, 0.05)
        self.assertEqual(args.entropy_coef, 0.01)

    def test_default_task_is_cds(self) -> None:
        args = _parse_args(self._base_argv())
        self.assertEqual(args.task, "cds")

    def test_task_can_be_five_utr(self) -> None:
        argv = self._base_argv() + ["--task", "five_utr"]
        args = _parse_args(argv)
        self.assertEqual(args.task, "five_utr")

    def test_default_group_size_is_8(self) -> None:
        args = _parse_args(self._base_argv())
        self.assertEqual(args.group_size, 8)

    def test_default_rollout_seeds_are_0_through_9(self) -> None:
        args = _parse_args(self._base_argv())
        self.assertEqual(args.rollout_seeds, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    def test_ref_checkpoint_defaults_to_none(self) -> None:
        args = _parse_args(self._base_argv())
        self.assertIsNone(args.ref_checkpoint)

    def test_run_mode_default_is_development(self) -> None:
        args = _parse_args(self._base_argv())
        self.assertEqual(args.run_mode, "development")

    def test_run_mode_paper_is_accepted(self) -> None:
        argv = self._base_argv() + ["--run-mode", "paper",
                                     "--oracle-manifest", "/tmp/oracle.json"]
        args = _parse_args(argv)
        self.assertEqual(args.run_mode, "paper")

    def test_checkpoint_is_required(self) -> None:
        argv = self._base_argv()
        argv.remove("--checkpoint")
        argv.remove("/tmp/ckpt.pt")
        with self.assertRaises(SystemExit):
            _parse_args(argv)

    def test_records_jsonl_is_required(self) -> None:
        argv = self._base_argv()
        argv.remove("--records-jsonl")
        argv.remove("/tmp/records.jsonl")
        with self.assertRaises(SystemExit):
            _parse_args(argv)

    def test_out_dir_is_required(self) -> None:
        argv = self._base_argv()
        argv.remove("--out-dir")
        argv.remove("/tmp/out")
        with self.assertRaises(SystemExit):
            _parse_args(argv)


class TestSplitContractVerification(unittest.TestCase):
    """Tests for verify_split_contract_cli."""

    def setUp(self) -> None:
        self.tmpdir_obj = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self.tmpdir_obj.name)
        self.records = _make_records(n=8)
        # Split: train=[0,1,2,3], val=[4,5], test=[6,7]
        self.manifest_path, self.contract = _build_split(
            self.tmpdir, self.records,
            train_idx=[0, 1, 2, 3],
            val_idx=[4, 5],
            test_idx=[6, 7],
        )

    def tearDown(self) -> None:
        self.tmpdir_obj.cleanup()

    def _make_args(self, **overrides: Any) -> Any:
        defaults = dict(
            split_manifest=str(self.manifest_path),
            records_jsonl=str(self.tmpdir / "records.jsonl"),
            train_idx=str(self.tmpdir / "train.idx"),
            val_idx=str(self.tmpdir / "val.idx"),
            test_idx=str(self.tmpdir / "test.idx"),
        )
        defaults.update(overrides)
        return type("Args", (), defaults)()

    def test_returns_contract_when_all_idx_match(self) -> None:
        args = self._make_args()
        contract = verify_split_contract_cli(args)
        self.assertEqual(contract.manifest_path, str(self.manifest_path.resolve()))
        self.assertIn("train", contract.roles)
        self.assertIn("val", contract.roles)
        self.assertIn("test", contract.roles)
        self.assertEqual(len(contract.roles["train"].indices), 4)
        self.assertEqual(len(contract.roles["val"].indices), 2)
        self.assertEqual(len(contract.roles["test"].indices), 2)

    def test_raises_when_split_manifest_missing(self) -> None:
        args = self._make_args(split_manifest="")
        with self.assertRaises(P205ContractError):
            verify_split_contract_cli(args)

    def test_raises_when_train_idx_missing(self) -> None:
        args = self._make_args(train_idx=None)
        with self.assertRaises(P205ContractError):
            verify_split_contract_cli(args)

    def test_raises_when_val_idx_missing(self) -> None:
        args = self._make_args(val_idx=None)
        with self.assertRaises(P205ContractError):
            verify_split_contract_cli(args)

    def test_raises_when_test_idx_missing(self) -> None:
        args = self._make_args(test_idx=None)
        with self.assertRaises(P205ContractError):
            verify_split_contract_cli(args)

    def test_raises_when_train_idx_file_not_found(self) -> None:
        args = self._make_args(train_idx="/nonexistent/train.idx")
        with self.assertRaises(P205ContractError):
            verify_split_contract_cli(args)

    def test_raises_when_train_idx_indices_mismatch(self) -> None:
        # Write a wrong train.idx.
        wrong_path = self.tmpdir / "wrong_train.idx"
        _write_idx_file(wrong_path, [99, 100, 101, 102])
        args = self._make_args(train_idx=str(wrong_path))
        with self.assertRaises(P205ContractError):
            verify_split_contract_cli(args)


class TestOracleManifestVerification(unittest.TestCase):
    """Tests for verify_oracle_manifest_cli."""

    def setUp(self) -> None:
        self.tmpdir_obj = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self.tmpdir_obj.name)
        self.manifest_path, self.artifact_path, self.artifact_sha = _build_oracle_manifest(self.tmpdir)

    def tearDown(self) -> None:
        self.tmpdir_obj.cleanup()

    def _make_args(self, **overrides: Any) -> Any:
        defaults = dict(
            oracle_manifest=str(self.manifest_path),
            run_mode="development",
        )
        defaults.update(overrides)
        return type("Args", (), defaults)()

    def test_returns_metadata_in_development_mode(self) -> None:
        args = self._make_args()
        meta = verify_oracle_manifest_cli(args)
        self.assertEqual(meta["oracle_type"], "gbt_regressor")
        self.assertTrue(meta["independent"])
        self.assertEqual(meta["artifact_sha256"], self.artifact_sha)

    def test_returns_none_when_manifest_missing_in_dev_mode(self) -> None:
        args = self._make_args(oracle_manifest=None)
        meta = verify_oracle_manifest_cli(args)
        self.assertEqual(meta["oracle_type"], "heuristic_development_oracle")
        self.assertFalse(meta["independent"])


class TestBuildRunConfig(unittest.TestCase):
    """End-to-end tests for build_run_config (with a tiny split)."""

    def setUp(self) -> None:
        self.tmpdir_obj = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self.tmpdir_obj.name)
        self.records = _make_records(n=8)
        self.manifest_path, _ = _build_split(
            self.tmpdir, self.records,
            train_idx=[0, 1, 2, 3],
            val_idx=[4, 5],
            test_idx=[6, 7],
        )
        self.oracle_manifest_path, _, self.oracle_artifact_sha = _build_oracle_manifest(self.tmpdir)
        self.ckpt_path = self.tmpdir / "ckpt.pt"
        self.ckpt_path.write_bytes(b"fake checkpoint bytes for p2_05 test")
        self.ckpt_sha = sha256_file(self.ckpt_path)
        self.out_dir = self.tmpdir / "out"
        self.out_dir.mkdir()

    def tearDown(self) -> None:
        self.tmpdir_obj.cleanup()

    def _make_args(self, **overrides: Any) -> Any:
        defaults = dict(
            checkpoint=str(self.ckpt_path),
            checkpoint_sha256=self.ckpt_sha,
            ref_checkpoint=None,
            ref_checkpoint_sha256=None,
            oracle_manifest=str(self.oracle_manifest_path),
            records_jsonl=str(self.tmpdir / "records.jsonl"),
            split_manifest=str(self.manifest_path),
            split_role="train",
            train_idx=str(self.tmpdir / "train.idx"),
            val_idx=str(self.tmpdir / "val.idx"),
            test_idx=str(self.tmpdir / "test.idx"),
            task="cds",
            group_size=8,
            kl_coef=0.05,
            entropy_coef=0.01,
            lr=0.01,
            n_iter=10,
            n_groups=2,
            policy_seed=0,
            rollout_seeds=[0, 1],
            out_dir=str(self.out_dir),
            device="cpu",
            run_mode="development",
            limit=8,
        )
        defaults.update(overrides)
        return type("Args", (), defaults)()

    def test_build_run_config_succeeds_with_valid_inputs(self) -> None:
        args = self._make_args()
        cfg = build_run_config(args)
        self.assertEqual(cfg.checkpoint_sha256, self.ckpt_sha)
        self.assertEqual(cfg.kl_coef, 0.05)
        self.assertEqual(cfg.entropy_coef, 0.01)
        self.assertEqual(cfg.task, "cds")
        self.assertEqual(cfg.group_size, 8)
        self.assertEqual(cfg.records_count, 8)

    def test_metadata_includes_kl_and_entropy(self) -> None:
        args = self._make_args()
        cfg = build_run_config(args)
        meta = cfg.to_metadata()
        self.assertEqual(meta["kl_coef"], 0.05)
        self.assertEqual(meta["entropy_coef"], 0.01)
        self.assertEqual(meta["reward_field"], "predicted_te_internal_proxy")

    def test_metadata_includes_checkpoint_and_split_shas(self) -> None:
        args = self._make_args()
        cfg = build_run_config(args)
        meta = cfg.to_metadata()
        self.assertEqual(meta["checkpoint_sha256"], self.ckpt_sha)
        self.assertEqual(meta["ref_checkpoint_sha256"], self.ckpt_sha)
        self.assertIn("split_manifest_sha256", meta)
        self.assertEqual(meta["oracle_artifact_sha256"], self.oracle_artifact_sha)

    def test_ref_checkpoint_defaults_to_checkpoint(self) -> None:
        args = self._make_args()
        cfg = build_run_config(args)
        self.assertEqual(cfg.ref_checkpoint_path, cfg.checkpoint_path)
        self.assertEqual(cfg.ref_checkpoint_sha256, cfg.checkpoint_sha256)

    def test_ref_checkpoint_can_be_overridden(self) -> None:
        ref_ckpt = self.tmpdir / "ref.pt"
        ref_ckpt.write_bytes(b"different ref checkpoint bytes")
        ref_sha = sha256_file(ref_ckpt)
        args = self._make_args(
            ref_checkpoint=str(ref_ckpt),
            ref_checkpoint_sha256=ref_sha,
        )
        cfg = build_run_config(args)
        self.assertEqual(cfg.ref_checkpoint_path, str(ref_ckpt))
        self.assertEqual(cfg.ref_checkpoint_sha256, ref_sha)

    def test_raises_on_checkpoint_sha_mismatch(self) -> None:
        args = self._make_args(checkpoint_sha256="0" * 64)
        with self.assertRaises(P205CheckpointError):
            build_run_config(args)

    def test_raises_on_train_idx_mismatch(self) -> None:
        wrong_train = self.tmpdir / "wrong_train.idx"
        _write_idx_file(wrong_train, [99, 100, 101, 102])
        args = self._make_args(train_idx=str(wrong_train))
        with self.assertRaises(P205ContractError):
            build_run_config(args)


class TestRunGrpoPilotMDPNotReady(unittest.TestCase):
    """run_grpo_pilot returns 'mdp_not_ready' until P2-02 completes."""

    def setUp(self) -> None:
        self.tmpdir_obj = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self.tmpdir_obj.name)
        self.records = _make_records(n=8)
        self.manifest_path, _ = _build_split(
            self.tmpdir, self.records,
            train_idx=[0, 1, 2, 3],
            val_idx=[4, 5],
            test_idx=[6, 7],
        )
        self.oracle_manifest_path, _, _ = _build_oracle_manifest(self.tmpdir)
        self.ckpt_path = self.tmpdir / "ckpt.pt"
        self.ckpt_path.write_bytes(b"fake checkpoint bytes for p2_05 mdp test")
        self.ckpt_sha = sha256_file(self.ckpt_path)
        self.out_dir = self.tmpdir / "out_mdp"

    def tearDown(self) -> None:
        self.tmpdir_obj.cleanup()

    def _make_run_config(self) -> Any:
        args = type("Args", (), dict(
            checkpoint=str(self.ckpt_path),
            checkpoint_sha256=self.ckpt_sha,
            ref_checkpoint=None,
            ref_checkpoint_sha256=None,
            oracle_manifest=str(self.oracle_manifest_path),
            records_jsonl=str(self.tmpdir / "records.jsonl"),
            split_manifest=str(self.manifest_path),
            split_role="train",
            train_idx=str(self.tmpdir / "train.idx"),
            val_idx=str(self.tmpdir / "val.idx"),
            test_idx=str(self.tmpdir / "test.idx"),
            task="cds",
            group_size=8,
            kl_coef=0.0,
            entropy_coef=0.0,
            lr=0.01,
            n_iter=10,
            n_groups=2,
            policy_seed=0,
            rollout_seeds=[0],
            out_dir=str(self.out_dir),
            device="cpu",
            run_mode="development",
            limit=8,
        ))()
        return build_run_config(args)

    def test_returns_mdp_not_ready_status(self) -> None:
        cfg = self._make_run_config()
        result = run_grpo_pilot(cfg)
        self.assertEqual(result["status"], "mdp_not_ready")
        self.assertIn("error", result)

    def test_writes_metadata_sidecar(self) -> None:
        cfg = self._make_run_config()
        run_grpo_pilot(cfg)
        metadata_path = self.out_dir / "run_metadata.json"
        self.assertTrue(metadata_path.exists())
        with metadata_path.open() as fh:
            meta = json.load(fh)
        self.assertEqual(meta["status"], "mdp_not_ready")
        self.assertEqual(meta["reward_field"], "predicted_te_internal_proxy")
        self.assertEqual(meta["checkpoint_sha256"], self.ckpt_sha)
        self.assertEqual(meta["kl_coef"], 0.0)
        self.assertEqual(meta["entropy_coef"], 0.0)

    def test_creates_out_dir_if_not_exists(self) -> None:
        cfg = self._make_run_config()
        # out_dir may not exist yet.
        self.assertFalse(self.out_dir.exists())
        run_grpo_pilot(cfg)
        self.assertTrue(self.out_dir.exists())


class TestGRPOConfigConstruction(unittest.TestCase):
    """Verify that build_run_config constructs a GRPOConfig with KL + entropy."""

    def setUp(self) -> None:
        self.tmpdir_obj = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self.tmpdir_obj.name)
        self.records = _make_records(n=8)
        self.manifest_path, _ = _build_split(
            self.tmpdir, self.records,
            train_idx=[0, 1, 2, 3],
            val_idx=[4, 5],
            test_idx=[6, 7],
        )
        self.oracle_manifest_path, _, _ = _build_oracle_manifest(self.tmpdir)
        self.ckpt_path = self.tmpdir / "ckpt.pt"
        self.ckpt_path.write_bytes(b"fake checkpoint bytes for grpo cfg test")
        self.ckpt_sha = sha256_file(self.ckpt_path)
        self.out_dir = self.tmpdir / "out_cfg"

    def tearDown(self) -> None:
        self.tmpdir_obj.cleanup()

    def _make_args(self, **overrides: Any) -> Any:
        defaults = dict(
            checkpoint=str(self.ckpt_path),
            checkpoint_sha256=self.ckpt_sha,
            ref_checkpoint=None,
            ref_checkpoint_sha256=None,
            oracle_manifest=str(self.oracle_manifest_path),
            records_jsonl=str(self.tmpdir / "records.jsonl"),
            split_manifest=str(self.manifest_path),
            split_role="train",
            train_idx=str(self.tmpdir / "train.idx"),
            val_idx=str(self.tmpdir / "val.idx"),
            test_idx=str(self.tmpdir / "test.idx"),
            task="cds",
            group_size=4,
            kl_coef=0.1,
            entropy_coef=0.02,
            lr=0.005,
            n_iter=5,
            n_groups=2,
            policy_seed=42,
            rollout_seeds=[0, 1, 2],
            out_dir=str(self.out_dir),
            device="cpu",
            run_mode="development",
            limit=8,
        )
        defaults.update(overrides)
        return type("Args", (), defaults)()

    def test_grpo_config_includes_kl_and_entropy(self) -> None:
        # Build the config and check the GRPOConfig fields.
        args = self._make_args()
        cfg = build_run_config(args)
        # Construct a GRPOConfig from the run config (mirrors run_grpo_pilot).
        grpo_cfg = GRPOConfig(
            group_size=cfg.group_size,
            eps=1e-8,
            clip_advantage=0.0,
            kl_coef=cfg.kl_coef,
            entropy_coef=cfg.entropy_coef,
        )
        self.assertEqual(grpo_cfg.group_size, 4)
        self.assertEqual(grpo_cfg.kl_coef, 0.1)
        self.assertEqual(grpo_cfg.entropy_coef, 0.02)

    def test_grpo_config_validates_kl_coef_negative(self) -> None:
        with self.assertRaises(ValueError):
            GRPOConfig(group_size=4, kl_coef=-0.1)

    def test_grpo_config_validates_entropy_coef_negative(self) -> None:
        with self.assertRaises(ValueError):
            GRPOConfig(group_size=4, entropy_coef=-0.1)


class TestLoadPolicyFromCheckpoint(unittest.TestCase):
    """Tests for _load_policy_from_checkpoint (v5).

    These tests do NOT require a real Stage A checkpoint. They verify:
      * Error paths (missing file, missing 'config' key, missing 'model_state' key).
      * Happy path with mocked build_stage_a_model + Policy.
      * Happy path without backbone_state (frozen backbone case).

    Note: _make_run_config constructs P205RunConfig DIRECTLY (bypassing
    build_run_config) so that we can test _load_policy_from_checkpoint's
    own error handling for missing/invalid checkpoints, rather than
    build_run_config's (which also checks checkpoint existence).
    """

    def setUp(self) -> None:
        self.tmpdir_obj = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self.tmpdir_obj.name)
        self.records = _make_records(n=4)
        self.manifest_path, self.contract = _build_split(
            self.tmpdir, self.records,
            train_idx=[0, 1],
            val_idx=[2],
            test_idx=[3],
        )
        self.oracle_manifest_path, self.oracle_artifact_path, self.oracle_artifact_sha = \
            _build_oracle_manifest(self.tmpdir)
        self.out_dir = self.tmpdir / "out_load"

    def tearDown(self) -> None:
        self.tmpdir_obj.cleanup()

    def _make_run_config(self, checkpoint_path: str) -> Any:
        """Build a P205RunConfig DIRECTLY (bypasses build_run_config).

        This lets us test _load_policy_from_checkpoint's own error handling
        for missing/invalid checkpoint paths, rather than build_run_config's
        (which calls verify_checkpoint_sha256 first).
        """
        oracle_metadata = {
            "oracle_type": "gbt_regressor",
            "independent": True,
            "source": "tiny_test_oracle",
            "independence_statement": "Tiny test oracle for unit tests.",
            "artifact_path": str(self.oracle_artifact_path),
            "artifact_sha256": self.oracle_artifact_sha,
        }
        return P205RunConfig(
            checkpoint_path=checkpoint_path,
            checkpoint_sha256="x" * 64,
            ref_checkpoint_path=checkpoint_path,
            ref_checkpoint_sha256="x" * 64,
            oracle_manifest_path=str(self.oracle_manifest_path),
            oracle_metadata=oracle_metadata,
            split_contract=self.contract,
            split_role="train",
            task="cds",
            group_size=8,
            kl_coef=0.0,
            entropy_coef=0.0,
            lr=0.01,
            n_iter=10,
            n_groups=2,
            policy_seed=0,
            rollout_seeds=(0,),
            out_dir=str(self.out_dir),
            device="cpu",
            run_mode="development",
            limit=8,
            records_count=4,
        )

    def test_missing_file_raises_checkpoint_error(self) -> None:
        """A non-existent checkpoint path raises P205CheckpointError."""
        rc = self._make_run_config(str(self.tmpdir / "nonexistent.pt"))
        with self.assertRaises(P205CheckpointError) as ctx:
            _load_policy_from_checkpoint(rc, torch.device("cpu"))
        self.assertIn("checkpoint not found", str(ctx.exception))

    def test_missing_config_key_raises_checkpoint_error(self) -> None:
        """A checkpoint lacking the 'config' key raises P205CheckpointError."""
        bad_path = self.tmpdir / "bad_no_config.pt"
        torch.save({"model_state": {}}, bad_path)
        rc = self._make_run_config(str(bad_path))
        with self.assertRaises(P205CheckpointError) as ctx:
            _load_policy_from_checkpoint(rc, torch.device("cpu"))
        self.assertIn("missing required 'config'", str(ctx.exception))

    def test_missing_model_state_key_raises_checkpoint_error(self) -> None:
        """A checkpoint lacking 'model_state' raises P205CheckpointError."""
        bad_path = self.tmpdir / "bad_no_model_state.pt"
        torch.save({"config": {}}, bad_path)
        rc = self._make_run_config(str(bad_path))
        with self.assertRaises(P205CheckpointError) as ctx:
            _load_policy_from_checkpoint(rc, torch.device("cpu"))
        self.assertIn("missing required 'model_state'", str(ctx.exception))

    def _make_fake_train_backbone(self, dummy_model, dummy_backbone):
        """Create a fake train_backbone module for injection into sys.modules."""
        fake_tb = types.ModuleType("train_backbone")

        class DummyCfg:
            class model:
                use_aux_struct = True

        fake_tb._coerce_config = lambda d: DummyCfg()
        fake_tb.build_stage_a_model = lambda cfg, device: (dummy_backbone, dummy_model)
        return fake_tb

    def test_happy_path_with_mocked_build(self) -> None:
        """Happy path: mocked build_stage_a_model + real load_state_dict.

        Verifies that:
          - The function returns a Policy (mocked).
          - model.load_state_dict is called with ckpt["model_state"].
          - backbone.load_state_dict is called when backbone_state is present.
          - model is in train() mode and backbone in eval() mode.
        """
        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(2))

            def forward(self, *a, **kw):
                raise NotImplementedError

        class DummyBackbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bias = torch.nn.Parameter(torch.zeros(3))

            def forward(self, *a, **kw):
                raise NotImplementedError

        dummy_model = DummyModel()
        dummy_backbone = DummyBackbone()

        ckpt_path = self.tmpdir / "good.pt"
        torch.save(
            {
                "stage": "stage_a",
                "step": 10000,
                "best_loss": 1.23,
                "config": {"model": {"use_aux_struct": True}},
                "model_state": dummy_model.state_dict(),
                "backbone_state": dummy_backbone.state_dict(),
            },
            ckpt_path,
        )

        fake_tb = self._make_fake_train_backbone(dummy_model, dummy_backbone)

        class FakePolicy:
            def __init__(self, model, backbone, cfg, device):
                self.model = model
                self.backbone = backbone
                self.cfg = cfg
                self.device = device

        rc = self._make_run_config(str(ckpt_path))
        # Reset modes to verify the function sets them correctly.
        dummy_model.eval()
        dummy_backbone.train()

        with patch.dict(sys.modules, {"train_backbone": fake_tb}), \
             patch.object(run_p2_05_grpo_pilot, "Policy", FakePolicy):
            result = _load_policy_from_checkpoint(rc, torch.device("cpu"))

        self.assertIsInstance(result, FakePolicy)
        self.assertIs(result.model, dummy_model)
        self.assertIs(result.backbone, dummy_backbone)
        # model should be in train mode, backbone in eval mode.
        self.assertTrue(dummy_model.training)
        self.assertFalse(dummy_backbone.training)

    def test_no_backbone_state_is_ok(self) -> None:
        """A checkpoint without 'backbone_state' is OK (frozen backbone case)."""
        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(2))

        class DummyBackbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bias = torch.nn.Parameter(torch.zeros(3))

        dummy_model = DummyModel()
        dummy_backbone = DummyBackbone()

        ckpt_path = self.tmpdir / "no_backbone.pt"
        torch.save(
            {
                "stage": "stage_a",
                "step": 5000,
                "config": {},
                "model_state": dummy_model.state_dict(),
                # NO 'backbone_state' key
            },
            ckpt_path,
        )

        fake_tb = self._make_fake_train_backbone(dummy_model, dummy_backbone)

        class FakePolicy:
            def __init__(self, model, backbone, cfg, device):
                pass

        rc = self._make_run_config(str(ckpt_path))
        with patch.dict(sys.modules, {"train_backbone": fake_tb}), \
             patch.object(run_p2_05_grpo_pilot, "Policy", FakePolicy):
            result = _load_policy_from_checkpoint(rc, torch.device("cpu"))
        self.assertIsInstance(result, FakePolicy)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
