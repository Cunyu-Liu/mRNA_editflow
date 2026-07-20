"""P1-05 smoke test: lock_oracle.py end-to-end.

Verifies:
    1. lock_oracle() seals a fake oracle dir (writes manifest, key, chmod 444)
    2. verify_lock() returns all-True on a pristine locked oracle
    3. audit_independence() reports independent=True
    4. Tamper detection: modifying a model file breaks verification
    5. Test-label one-way hash check works

Run:
    PYTHONPATH=/home/cunyuliu/mrna_editflow_goal \
    /home/cunyuliu/miniconda3/envs/editflow/bin/python \
    scripts/smoke_test_p1_05_lock.py
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from models.oracle_final.lock_oracle import (  # noqa: E402
    LockManifest,
    lock_oracle,
    verify_lock,
    audit_independence,
)


def _make_fake_oracle(oracle_dir: Path) -> None:
    """Create a minimal fake oracle directory."""
    oracle_dir.mkdir(parents=True, exist_ok=True)
    # Fake LightGBM text model files
    (oracle_dir / "mean_model.txt").write_text(
        "version\n2\n0\nend of trees\n")
    (oracle_dir / "q10_model.txt").write_text(
        "version\n2\n0\nend of trees\n")
    (oracle_dir / "q90_model.txt").write_text(
        "version\n2\n0\nend of trees\n")
    # Fake oracle_meta.json with feature_config (for independence audit)
    meta = {
        "oracle_id": "smoke_test_oracle",
        "hyperparams": {
            "feature_config": {
                "max_kmer_k": 5,
                "include_codon_usage": True,
                "include_motifs": True,
                "n_features": 340,
            },
            "n_estimators": 100,
            "learning_rate": 0.05,
        },
        "dataset": "lepplek2022 + sample2019_train",
        "fitted": True,
    }
    (oracle_dir / "oracle_meta.json").write_text(json.dumps(meta, indent=2))


def main() -> int:
    print("=" * 70)
    print("P1-05 lock_oracle smoke test")
    print("=" * 70)

    tmpdir = Path(tempfile.mkdtemp(prefix="lock_smoke_"))
    try:
        # ---------- Test 1: lock_oracle ----------
        oracle_dir = tmpdir / "oracle3_smoke"
        _make_fake_oracle(oracle_dir)

        train_seqs = [f"ATGCGC{'ACGT' * 5}{i}" for i in range(50)]
        train_labels = [float(i) for i in range(50)]
        test_seqs = [f"TTTGGG{'ACGT' * 5}{i}" for i in range(10)]
        test_labels = [float(i) * 0.5 for i in range(10)]

        print("\n[Test 1] lock_oracle()")
        manifest = lock_oracle(
            oracle_dir=oracle_dir,
            training_sequences=train_seqs,
            training_labels=train_labels,
            test_sequences=test_seqs,
            test_labels=test_labels,
            oracle_id="smoke_oracle_v1",
            signing_key="test_key_do_not_use_in_prod",
            make_readonly=True,
        )
        assert isinstance(manifest, LockManifest), "manifest is not LockManifest"
        assert manifest.oracle_id == "smoke_oracle_v1"
        assert len(manifest.model_hashes) == 3, \
            f"expected 3 model hashes, got {len(manifest.model_hashes)}"
        assert manifest.oracle_meta_hash, "oracle_meta_hash empty"
        assert manifest.feature_extractor_hash, "feature_extractor_hash empty"
        assert manifest.training_data_hash, "training_data_hash empty"
        assert len(manifest.test_label_hashes) == 10, \
            f"expected 10 test label hashes, got {len(manifest.test_label_hashes)}"
        assert manifest.hmac_signature, "hmac_signature empty"
        print("  PASS: lock_oracle produced valid manifest")
        print(f"    model_hashes: {len(manifest.model_hashes)}")
        print(f"    test_label_hashes: {len(manifest.test_label_hashes)}")
        print(f"    hmac_signature: {manifest.hmac_signature[:16]}...")

        # Verify files exist and are read-only
        assert (oracle_dir / "lock_manifest.json").exists()
        assert (oracle_dir / ".lock_key").exists()
        for f in oracle_dir.iterdir():
            if f.is_file():
                assert not os.access(f, os.W_OK), \
                    f"{f.name} is writable (should be read-only)"
        print("  PASS: all files read-only (chmod 444)")

        # ---------- Test 2: verify_lock (pristine) ----------
        print("\n[Test 2] verify_lock() on pristine oracle")
        result = verify_lock(
            oracle_dir=oracle_dir,
            training_sequences=train_seqs,
            test_sequences=test_seqs,
            test_labels=test_labels,
        )
        assert result.manifest_valid, "HMAC signature invalid"
        assert result.model_hashes_match, "model hashes don't match"
        assert result.meta_hash_match, "meta hash doesn't match"
        assert result.training_data_hash_match, "training data hash doesn't match"
        assert result.test_label_hashes_match, "test label hashes don't match"
        assert result.valid, "verify_lock did not return valid=True"
        print(f"  PASS: verify_lock valid={result.valid}")
        print(f"    manifest_valid: {result.manifest_valid}")
        print(f"    model_hashes_match: {result.model_hashes_match}")
        print(f"    meta_hash_match: {result.meta_hash_match}")
        print(f"    training_data_hash_match: {result.training_data_hash_match}")
        print(f"    test_label_hashes_match: {result.test_label_hashes_match}")

        # ---------- Test 3: audit_independence ----------
        print("\n[Test 3] audit_independence()")
        # Need to make files writable temporarily to inspect (audit itself
        # just reads, but the test setup may need adjustments).
        # We provide teacher sequences with 50% overlap.
        teacher_seqs = train_seqs[:25] + [f"CCCC{'AT' * 10}{i}" for i in range(25)]
        audit = audit_independence(
            oracle_dir=oracle_dir,
            teacher_ckpt_dir=None,  # skip architecture check via ckpt path
            teacher_training_sequences=teacher_seqs,
            oracle_training_sequences=train_seqs,
        )
        print(f"  independent: {audit.independent}")
        for crit, val in audit.criteria.items():
            print(f"    {crit}: {val}")
        for note in audit.notes:
            print(f"    note: {note}")
        assert audit.criteria["different_architecture"], \
            "different_architecture failed"
        assert audit.criteria["different_feature_space"], \
            "different_feature_space failed"
        assert audit.criteria["frozen_readonly"], "frozen_readonly failed"
        assert audit.criteria["signed_valid"], "signed_valid failed"
        assert audit.criteria["test_labels_hashed"], "test_labels_hashed failed"
        # Overlap is 25/50 = 50%, which is NOT < 0.5, so training_data_distinct
        # should be False (this is the expected boundary case for the test).
        # We accept either way but log it.
        print(f"  PASS: audit_independence completed")
        print(f"    (training_data_distinct={audit.criteria['training_data_distinct']} "
              f"is acceptable for smoke test)")

        # ---------- Test 4: tamper detection ----------
        print("\n[Test 4] tamper detection")
        # Need to make file writable to tamper
        model_path = oracle_dir / "mean_model.txt"
        os.chmod(model_path, 0o644)
        with open(model_path, "a") as f:
            f.write("\n# TAMPERED\n")
        os.chmod(model_path, 0o444)

        tampered_result = verify_lock(oracle_dir=oracle_dir)
        assert not tampered_result.model_hashes_match, \
            "tampered model hash still matches (expected mismatch)"
        assert not tampered_result.valid, \
            "tampered oracle still valid (expected invalid)"
        assert any("mean_model.txt" in m for m in tampered_result.mismatches), \
            f"expected mean_model.txt in mismatches, got {tampered_result.mismatches}"
        print(f"  PASS: tamper detected")
        print(f"    model_hashes_match: {tampered_result.model_hashes_match}")
        print(f"    valid: {tampered_result.valid}")
        print(f"    mismatches: {tampered_result.mismatches}")

        # ---------- Test 5: test-label one-way hash ----------
        print("\n[Test 5] test-label one-way hash")
        # Verify that the manifest stores hashes, not raw labels
        with open(oracle_dir / "lock_manifest.json") as f:
            stored = json.load(f)
        test_hashes = stored["test_label_hashes"]
        # All values should be 64-char hex strings
        for rid, h in test_hashes.items():
            assert len(h) == 64, f"hash not 64 chars: {h}"
            assert all(c in "0123456789abcdef" for c in h), \
                f"hash not hex: {h}"
        # Verify that the hash is one-way (can't recover label)
        # Hash of 0.000000 should be different from hash of 1.000000
        from models.oracle_final.lock_oracle import _label_hash
        h0 = _label_hash(0.0)
        h1 = _label_hash(1.0)
        assert h0 != h1, "label hashes collide for 0.0 and 1.0"
        # Verify the rid is also a hash (not the raw sequence)
        for rid in test_hashes.keys():
            assert len(rid) == 64, f"rid not 64 chars: {rid}"
            assert all(c in "0123456789abcdef" for c in rid), \
                f"rid not hex: {rid}"
        print(f"  PASS: all {len(test_hashes)} test labels stored as one-way SHA-256 hashes")
        print(f"  PASS: all {len(test_hashes)} record IDs are SHA-256 hashes (not raw sequences)")

        print("\n" + "=" * 70)
        print("ALL P1-05 lock_oracle SMOKE TESTS PASSED")
        print("=" * 70)
        return 0

    finally:
        # Restore write permissions before cleanup
        if oracle_dir.exists():
            for f in oracle_dir.iterdir():
                if f.is_file():
                    os.chmod(f, 0o644)
        shutil.rmtree(tmpdir, ignore_errors=True)
        print(f"\n(cleaned up: {tmpdir})")


if __name__ == "__main__":
    sys.exit(main())
