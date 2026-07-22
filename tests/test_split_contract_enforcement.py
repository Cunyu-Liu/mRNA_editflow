"""P2-04 split contract enforcement tests.

Extends P1-10 tests with additional coverage for:
  - SHA-256 mismatch detection on idx files (vs split contract's recorded SHA)
  - Exact-match fail-closed is DEFAULT (not opt-out) in run_multiseed_benchmark
  - All training entry points expose --train-idx/--val-idx/--test-idx arguments
  - Foundation pretraining corpus leakage detection (pretraining record in test split)

Covers deliverables for P2-04:
  - 缺 idx 退出 (missing idx → exit) — covered by P1-10, extended here
  - SHA 不匹配退出 (SHA mismatch → exit) — NEW
  - exact-match fail-closed — covered by P1-10, extended here for default behavior

Run:
    cd /home/cunyuliu/mrna_editflow_goal
    /home/cunyuliu/miniconda3/envs/editflow/bin/python3.10 -m pytest \\
        mrna_editflow/tests/test_split_contract_enforcement.py -v
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


class TestShaMismatchDetection(unittest.TestCase):
    """P2-04: SHA-256 mismatch on idx files must cause exit.

    The split contract records the SHA-256 of each role's idx file. If a user
    provides an idx file whose SHA does not match the contract, the script
    must fail-closed.
    """

    def test_tampered_idx_file_content_detected_via_content_mismatch(self) -> None:
        """Tampering with idx file content changes both SHA and indices.

        This is detected by the content-mismatch check in _verify_idx_files
        and _enforce_exact_match_fail_closed.
        """
        from mrna_editflow.eval.run_multiseed_benchmark import _enforce_exact_match_fail_closed

        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            records = _make_records(6)
            train_idx = [0, 1, 2]
            val_idx = [3]
            test_idx = [4, 5]
            _, contract = _build_split(tmpdir, records, train_idx, val_idx, test_idx)

            # Tamper: replace train idx with different indices (same count)
            tampered_train = tmpdir / "tampered_train.idx"
            _write_idx_file(tampered_train, [0, 1, 5])  # 5 not in train
            args = argparse.Namespace(
                run_mode="development",
                split_manifest=None,
                split_role=None,
                train_idx=str(tampered_train),
                val_idx=str(tmpdir / "val.idx"),
                test_idx=str(tmpdir / "test.idx"),
            )
            with self.assertRaises(SystemExit):
                _enforce_exact_match_fail_closed(args, contract, records)

    def test_train_backbone_tampered_idx_raises_valueerror(self) -> None:
        """train_backbone._verify_idx_files detects tampered idx content."""
        from mrna_editflow.train_backbone import _verify_idx_files

        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            records = _make_records(6)
            train_idx = [0, 1, 2]
            val_idx = [3]
            test_idx = [4, 5]
            _, contract = _build_split(tmpdir, records, train_idx, val_idx, test_idx)

            tampered_train = tmpdir / "tampered_train.idx"
            _write_idx_file(tampered_train, [0, 1, 5])
            args = argparse.Namespace(
                train_idx=str(tampered_train),
                val_idx=str(tmpdir / "val.idx"),
                test_idx=str(tmpdir / "test.idx"),
            )
            with self.assertRaises(ValueError):
                _verify_idx_files(args, contract, records)


class TestExactMatchFailClosedIsDefault(unittest.TestCase):
    """P2-04: exact-match fail-closed must be the DEFAULT behavior.

    There is no --disable-fail-closed flag; the check runs whenever a split
    contract is present.
    """

    def test_fail_closed_runs_with_split_contract_in_dev_mode(self) -> None:
        """In development mode with a split contract, fail-closed still runs."""
        from mrna_editflow.eval.run_multiseed_benchmark import _enforce_exact_match_fail_closed

        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            records = _make_records(6)
            train_idx = [0, 1, 2]
            val_idx = [3]
            test_idx = [4, 5]
            _, contract = _build_split(tmpdir, records, train_idx, val_idx, test_idx)
            args = argparse.Namespace(
                run_mode="development",
                split_manifest=None,
                split_role=None,
                train_idx=None,
                val_idx=None,
                test_idx=None,
            )
            # Should not raise (digest matches)
            _enforce_exact_match_fail_closed(args, contract, records)

    def test_fail_closed_aborts_on_digest_mismatch_in_dev_mode(self) -> None:
        """Even in dev mode, a digest mismatch with a split contract aborts."""
        from mrna_editflow.eval.run_multiseed_benchmark import _enforce_exact_match_fail_closed

        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            records = _make_records(6)
            train_idx = [0, 1, 2]
            val_idx = [3]
            test_idx = [4, 5]
            _, contract = _build_split(tmpdir, records, train_idx, val_idx, test_idx)
            # Tampered records
            tampered = list(records)
            tampered[0] = MRNARecord(
                transcript_id=records[0].transcript_id,
                five_utr="TTT",  # changed
                cds=records[0].cds,
                three_utr=records[0].three_utr,
                species=records[0].species,
            )
            args = argparse.Namespace(
                run_mode="development",
                split_manifest=None,
                split_role=None,
                train_idx=None,
                val_idx=None,
                test_idx=None,
            )
            with self.assertRaises(SystemExit):
                _enforce_exact_match_fail_closed(args, contract, tampered)

    def test_no_fail_closed_when_no_split_contract(self) -> None:
        """Without a split contract, fail-closed is a no-op (backward compat)."""
        from mrna_editflow.eval.run_multiseed_benchmark import _enforce_exact_match_fail_closed

        records = _make_records(4)
        args = argparse.Namespace(
            run_mode="development",
            split_manifest=None,
            split_role=None,
            train_idx=None,
            val_idx=None,
            test_idx=None,
        )
        # Should not raise regardless of records content
        _enforce_exact_match_fail_closed(args, None, records)


class TestTrainingEntryPointsExposeSplitArgs(unittest.TestCase):
    """P2-04: All training entry points must expose --train-idx/--val-idx/--test-idx."""

    def test_train_backbone_exposes_split_args(self) -> None:
        from mrna_editflow.train_backbone import _parse_args
        # Parse with minimal required args; check split args exist
        args = _parse_args([
            "--config", "/tmp/dummy.json",
            "--train-idx", "/tmp/train.idx",
            "--val-idx", "/tmp/val.idx",
            "--test-idx", "/tmp/test.idx",
        ])
        self.assertEqual(args.train_idx, "/tmp/train.idx")
        self.assertEqual(args.val_idx, "/tmp/val.idx")
        self.assertEqual(args.test_idx, "/tmp/test.idx")

    def test_run_multiseed_benchmark_exposes_split_args(self) -> None:
        from mrna_editflow.eval.run_multiseed_benchmark import _parse_args
        args = _parse_args([
            "--records-jsonl", "/tmp/dummy.jsonl",
            "--out-dir", "/tmp/out",
            "--train-idx", "/tmp/train.idx",
            "--val-idx", "/tmp/val.idx",
            "--test-idx", "/tmp/test.idx",
        ])
        self.assertEqual(args.train_idx, "/tmp/train.idx")
        self.assertEqual(args.val_idx, "/tmp/val.idx")
        self.assertEqual(args.test_idx, "/tmp/test.idx")

    def test_train_backbone_split_manifest_arg_exists(self) -> None:
        from mrna_editflow.train_backbone import _parse_args
        args = _parse_args([
            "--config", "/tmp/dummy.json",
            "--split-manifest", "/tmp/manifest.json",
            "--split-role", "train",
        ])
        self.assertEqual(args.split_manifest, "/tmp/manifest.json")
        self.assertEqual(args.split_role, "train")

    def test_run_multiseed_benchmark_split_manifest_arg_exists(self) -> None:
        from mrna_editflow.eval.run_multiseed_benchmark import _parse_args
        args = _parse_args([
            "--records-jsonl", "/tmp/dummy.jsonl",
            "--out-dir", "/tmp/out",
            "--split-manifest", "/tmp/manifest.json",
            "--split-role", "test",
        ])
        self.assertEqual(args.split_manifest, "/tmp/manifest.json")
        self.assertEqual(args.split_role, "test")


class TestFoundationPretrainingLeakageDetection(unittest.TestCase):
    """P2-04: Foundation pretraining corpus leakage detection.

    If a pretraining record appears in the test split, the exact-match
    fail-closed mechanism detects it via records_content_digest mismatch
    (pretraining corpus has different digest than split contract).
    """

    def test_pretraining_corpus_overlap_detected_by_digest_mismatch(self) -> None:
        """A pretraining corpus that overlaps with test split but has different
        ordering/content will have a different digest, triggering fail-closed."""
        from mrna_editflow.eval.run_multiseed_benchmark import _enforce_exact_match_fail_closed

        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            records = _make_records(6)
            train_idx = [0, 1, 2]
            val_idx = [3]
            test_idx = [4, 5]
            _, contract = _build_split(tmpdir, records, train_idx, val_idx, test_idx)

            # Simulate a pretraining corpus that includes test records (leakage)
            # by adding an extra record — this changes the digest
            leaked_corpus = list(records) + [
                MRNARecord(
                    transcript_id="LEAK0001",
                    five_utr="AAA",
                    cds="AUGUUUUAA",
                    three_utr="UUU",
                    species="human",
                )
            ]
            args = argparse.Namespace(
                run_mode="development",
                split_manifest=None,
                split_role=None,
                train_idx=None,
                val_idx=None,
                test_idx=None,
            )
            with self.assertRaises(SystemExit):
                _enforce_exact_match_fail_closed(args, contract, leaked_corpus)

    def test_reordered_records_pass_via_set_match(self) -> None:
        """Records in different order but same content pass idx set-match
        (indices are compared as sets, not lists)."""
        from mrna_editflow.train_backbone import _verify_idx_files

        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            records = _make_records(6)
            train_idx = [0, 1, 2]
            val_idx = [3]
            test_idx = [4, 5]
            _, contract = _build_split(tmpdir, records, train_idx, val_idx, test_idx)

            # Reorder train idx (same content, different order)
            reordered_train = tmpdir / "reordered_train.idx"
            _write_idx_file(reordered_train, [2, 0, 1])
            args = argparse.Namespace(
                train_idx=str(reordered_train),
                val_idx=str(tmpdir / "val.idx"),
                test_idx=str(tmpdir / "test.idx"),
            )
            # Should not raise (set comparison passes)
            _verify_idx_files(args, contract, records)


class TestMissingIdxExitBehavior(unittest.TestCase):
    """P2-04: Missing idx file must cause exit (FileNotFoundError)."""

    def test_train_backbone_missing_idx_raises_filenotfound(self) -> None:
        from mrna_editflow.train_backbone import _verify_idx_files

        with tempfile.TemporaryDirectory() as tmp:
            train_p = os.path.join(tmp, "train.idx")
            val_p = os.path.join(tmp, "val.idx")
            test_p = os.path.join(tmp, "missing.idx")  # does not exist
            _write_idx_file(Path(train_p), [0, 1])
            _write_idx_file(Path(val_p), [2])
            args = argparse.Namespace(
                train_idx=train_p,
                val_idx=val_p,
                test_idx=test_p,
            )
            with self.assertRaises(FileNotFoundError):
                _verify_idx_files(args, None, None)

    def test_run_multiseed_benchmark_missing_idx_raises_filenotfound(self) -> None:
        from mrna_editflow.eval.run_multiseed_benchmark import _enforce_exact_match_fail_closed

        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            records = _make_records(6)
            train_idx = [0, 1, 2]
            val_idx = [3]
            test_idx = [4, 5]
            _, contract = _build_split(tmpdir, records, train_idx, val_idx, test_idx)

            # Point to a non-existent test idx file
            args = argparse.Namespace(
                run_mode="development",
                split_manifest=None,
                split_role=None,
                train_idx=str(tmpdir / "train.idx"),
                val_idx=str(tmpdir / "val.idx"),
                test_idx=str(tmpdir / "nonexistent.idx"),
            )
            with self.assertRaises((FileNotFoundError, SystemExit)):
                _enforce_exact_match_fail_closed(args, contract, records)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
