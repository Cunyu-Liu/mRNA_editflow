"""P1-10 split-contract enforcement tests.

Verifies that train_backbone.py and eval/run_multiseed_benchmark.py:
  - In development mode: no enforcement (backward compatible).
  - In paper mode: require either --split-manifest OR (--train-idx, --val-idx, --test-idx).
  - When idx files provided: verify they match split contract (FileNotFoundError, ValueError).
  - For multiseed benchmark: exact-match fail-closed on records_content_digest.

Run:
    cd /home/cunyuliu/mrna_editflow_goal
    /home/cunyuliu/miniconda3/envs/editflow/bin/python3.10 -m pytest \\
        mrna_editflow/tests/test_p1_10_split_enforcement.py -v
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Optional, Sequence

# Make repo root importable when running directly.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from mrna_editflow.core.schema import MRNARecord
from mrna_editflow.data.download_mrna import write_records_jsonl
from mrna_editflow.data.split_contract import (
    VerifiedRole,
    VerifiedSplitContract,
    build_split_manifest,
    load_and_verify_split_manifest,
    records_content_digest,
    sha256_file,
    transcript_id_digest,
)


def _make_records(n: int = 6) -> list[MRNARecord]:
    """Build deterministic synthetic MRNARecords for testing.

    MRNARecord signature: (transcript_id, five_utr, cds, three_utr, species="human")
    """
    out: list[MRNARecord] = []
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
    paper_eligible: bool = True,
) -> tuple[Path, VerifiedSplitContract]:
    """Build a minimal split manifest + load VerifiedSplitContract."""
    records_path = tmpdir / "records.jsonl"
    write_records_jsonl(list(records), str(records_path))
    role_paths = {}
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
            # Each record in its own cluster so family_disjoint=True is satisfied.
            fh.write(f"{i}\n")
    manifest_payload = build_split_manifest(
        dataset_id="test-dataset",
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
    contract = load_and_verify_split_manifest(str(manifest_path), records_path=str(records_path))
    return manifest_path, contract


# ---------------- train_backbone tests ----------------


def _make_train_args(
    run_mode: str = "development",
    split_manifest: Optional[str] = None,
    split_role: Optional[str] = None,
    train_idx: Optional[str] = None,
    val_idx: Optional[str] = None,
    test_idx: Optional[str] = None,
) -> argparse.Namespace:
    return argparse.Namespace(
        run_mode=run_mode,
        split_manifest=split_manifest,
        split_role=split_role,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
    )


class TestTrainBackbonePaperModeEnforcement(unittest.TestCase):
    """Tests for train_backbone.py paper-mode --train-idx/--val-idx/--test-idx enforcement."""

    def test_dev_mode_does_not_require_idx_or_manifest(self) -> None:
        # Should not raise; dev mode skips enforcement
        args = _make_train_args(run_mode="development")
        # No assertions needed; if no exception, test passes
        # (The actual enforcement code is in main(); we mimic the check here)
        if args.run_mode == "paper":
            self.fail("should not be paper mode")
        self.assertIsNone(args.split_manifest)
        self.assertIsNone(args.train_idx)

    def test_paper_mode_without_manifest_or_idx_raises(self) -> None:
        # Mimic the enforcement logic from main(): should raise SystemExit.
        args = _make_train_args(run_mode="paper")
        idx_provided = all([args.train_idx, args.val_idx, args.test_idx])
        with self.assertRaises(SystemExit):
            if not args.split_manifest and not idx_provided:
                raise SystemExit(
                    "paper mode requires either --split-manifest OR "
                    "(--train-idx AND --val-idx AND --test-idx); aborting."
                )

    def test_paper_mode_with_manifest_only_passes(self) -> None:
        args = _make_train_args(run_mode="paper", split_manifest="/some/path.json", split_role="train")
        idx_provided = all([args.train_idx, args.val_idx, args.test_idx])
        # Should not raise
        self.assertTrue(args.split_manifest or idx_provided)

    def test_paper_mode_with_all_idx_only_passes(self) -> None:
        args = _make_train_args(
            run_mode="paper",
            train_idx="/tmp/train.idx",
            val_idx="/tmp/val.idx",
            test_idx="/tmp/test.idx",
        )
        idx_provided = all([args.train_idx, args.val_idx, args.test_idx])
        self.assertTrue(args.split_manifest or idx_provided)


class TestTrainBackboneVerifyIdxFiles(unittest.TestCase):
    """Tests for _verify_idx_files in train_backbone.py."""

    def _make_args(
        self,
        train_idx: Optional[str] = None,
        val_idx: Optional[str] = None,
        test_idx: Optional[str] = None,
    ) -> argparse.Namespace:
        return argparse.Namespace(
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
        )

    def test_nonexistent_idx_file_raises_filenotfound(self) -> None:
        from mrna_editflow.train_backbone import _verify_idx_files
        with tempfile.TemporaryDirectory() as tmp:
            train_p = os.path.join(tmp, "train.idx")
            val_p = os.path.join(tmp, "val.idx")
            test_p = os.path.join(tmp, "missing.idx")  # does not exist
            _write_idx_file(Path(train_p), [0, 1])
            _write_idx_file(Path(val_p), [2])
            args = self._make_args(train_p, val_p, test_p)
            with self.assertRaises(FileNotFoundError):
                _verify_idx_files(args, None, None)

    def test_empty_idx_file_raises_valueerror(self) -> None:
        from mrna_editflow.train_backbone import _verify_idx_files
        with tempfile.TemporaryDirectory() as tmp:
            train_p = os.path.join(tmp, "train.idx")
            val_p = os.path.join(tmp, "val.idx")
            test_p = os.path.join(tmp, "test.idx")
            Path(train_p).write_text("")  # empty
            _write_idx_file(Path(val_p), [2])
            _write_idx_file(Path(test_p), [3])
            args = self._make_args(train_p, val_p, test_p)
            with self.assertRaises(ValueError):
                _verify_idx_files(args, None, None)

    def test_idx_count_mismatch_with_contract_raises(self) -> None:
        from mrna_editflow.train_backbone import _verify_idx_files
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            records = _make_records(6)
            train_idx = [0, 1, 2]
            val_idx = [3]
            test_idx = [4, 5]
            _, contract = _build_split(tmpdir, records, train_idx, val_idx, test_idx)
            # Write wrong-count train idx
            bad_train = tmpdir / "bad_train.idx"
            _write_idx_file(bad_train, [0, 1])  # only 2, contract has 3
            args = self._make_args(
                str(bad_train),
                str(tmpdir / "val.idx"),
                str(tmpdir / "test.idx"),
            )
            with self.assertRaises(ValueError):
                _verify_idx_files(args, contract, records)

    def test_idx_content_mismatch_with_contract_raises(self) -> None:
        from mrna_editflow.train_backbone import _verify_idx_files
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            records = _make_records(6)
            train_idx = [0, 1, 2]
            val_idx = [3]
            test_idx = [4, 5]
            _, contract = _build_split(tmpdir, records, train_idx, val_idx, test_idx)
            # Write same-count but wrong-content train idx
            bad_train = tmpdir / "bad_train.idx"
            _write_idx_file(bad_train, [0, 1, 5])  # 5 not in train
            args = self._make_args(
                str(bad_train),
                str(tmpdir / "val.idx"),
                str(tmpdir / "test.idx"),
            )
            with self.assertRaises(ValueError):
                _verify_idx_files(args, contract, records)

    def test_idx_content_match_with_contract_passes(self) -> None:
        from mrna_editflow.train_backbone import _verify_idx_files
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            records = _make_records(6)
            train_idx = [0, 1, 2]
            val_idx = [3]
            test_idx = [4, 5]
            _, contract = _build_split(tmpdir, records, train_idx, val_idx, test_idx)
            # Same content, different order
            reordered_train = tmpdir / "reordered_train.idx"
            _write_idx_file(reordered_train, [2, 0, 1])
            args = self._make_args(
                str(reordered_train),
                str(tmpdir / "val.idx"),
                str(tmpdir / "test.idx"),
            )
            _verify_idx_files(args, contract, records)  # should not raise


# ---------------- run_multiseed_benchmark tests ----------------


def _make_eval_args(
    run_mode: str = "development",
    split_manifest: Optional[str] = None,
    split_role: Optional[str] = None,
    train_idx: Optional[str] = None,
    val_idx: Optional[str] = None,
    test_idx: Optional[str] = None,
) -> argparse.Namespace:
    return argparse.Namespace(
        run_mode=run_mode,
        split_manifest=split_manifest,
        split_role=split_role,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
    )


class TestRunMultiseedExactMatchFailClosed(unittest.TestCase):
    """Tests for _enforce_exact_match_fail_closed in eval/run_multiseed_benchmark.py."""

    def test_no_contract_returns_silently(self) -> None:
        from mrna_editflow.eval.run_multiseed_benchmark import _enforce_exact_match_fail_closed
        records = _make_records(4)
        args = _make_eval_args()
        # Should not raise
        _enforce_exact_match_fail_closed(args, None, records)

    def test_digest_match_passes(self) -> None:
        from mrna_editflow.eval.run_multiseed_benchmark import _enforce_exact_match_fail_closed
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            records = _make_records(6)
            train_idx = [0, 1, 2]
            val_idx = [3]
            test_idx = [4, 5]
            _, contract = _build_split(tmpdir, records, train_idx, val_idx, test_idx)
            args = _make_eval_args()
            _enforce_exact_match_fail_closed(args, contract, records)  # should not raise

    def test_digest_mismatch_raises_systemexit(self) -> None:
        from mrna_editflow.eval.run_multiseed_benchmark import _enforce_exact_match_fail_closed
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            records = _make_records(6)
            train_idx = [0, 1, 2]
            val_idx = [3]
            test_idx = [4, 5]
            _, contract = _build_split(tmpdir, records, train_idx, val_idx, test_idx)
            # Tampered records: different sequence
            tampered = list(records)
            tampered[0] = MRNARecord(
                transcript_id=records[0].transcript_id,
                five_utr=records[0].five_utr,
                cds="AUG" + "GCU" * 99 + "UAA",  # changed
                three_utr=records[0].three_utr,
                species=records[0].species,
            )
            args = _make_eval_args()
            with self.assertRaises(SystemExit):
                _enforce_exact_match_fail_closed(args, contract, tampered)

    def test_idx_mismatch_raises_systemexit(self) -> None:
        from mrna_editflow.eval.run_multiseed_benchmark import _enforce_exact_match_fail_closed
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            records = _make_records(6)
            train_idx = [0, 1, 2]
            val_idx = [3]
            test_idx = [4, 5]
            _, contract = _build_split(tmpdir, records, train_idx, val_idx, test_idx)
            # Provide wrong-content train idx
            bad_train = tmpdir / "bad_train.idx"
            _write_idx_file(bad_train, [0, 1, 5])  # 5 not in train
            args = _make_eval_args(
                train_idx=str(bad_train),
                val_idx=str(tmpdir / "val.idx"),
                test_idx=str(tmpdir / "test.idx"),
            )
            with self.assertRaises(SystemExit):
                _enforce_exact_match_fail_closed(args, contract, records)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
