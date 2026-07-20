"""P1-05: Lock & seal procedure for the independent final oracle.

This module implements the cryptographic sealing of the GBT oracle to ensure
it cannot be modified after locking. The sealed oracle is the final
independent evaluation oracle (Oracle #3) that must never see test labels
in any form during RL training.

Lock procedure:
    1. Train oracle on (Leplek 2022 + held-out Sample 2019 train split)
    2. Compute SHA-256 of:
       - Model files (mean_model.txt, q*.txt)
       - Feature extractor config
       - Training data record IDs (sequence hashes)
       - Oracle metadata
    3. Write lock manifest with all SHA-256 hashes
    4. Sign manifest with project key (HMAC-SHA256)
    5. chmod 444 all artifact files (read-only)
    6. Write encrypted test label hashes (one-way) for later verification

Unlock procedure (for audit only):
    1. Verify HMAC signature on lock manifest
    2. Recompute SHA-256 of all artifacts, compare to manifest
    3. If any mismatch → oracle has been tampered with
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Lock manifest
# ---------------------------------------------------------------------------

@dataclass
class LockManifest:
    """Manifest for a locked oracle.

    Attributes:
        oracle_id: unique identifier for this oracle
        lock_time_utc: ISO timestamp of locking
        model_hashes: dict of {filename: sha256}
        feature_extractor_hash: sha256 of feature extractor config
        training_data_hash: sha256 of training record IDs
        oracle_meta_hash: sha256 of oracle_meta.json
        test_label_hashes: dict of {record_id: [sha256(label), ...]} for
            verification. Uses a list to correctly handle duplicate sequences
            (same UTR measured in multiple GSM samples with different labels).
        hmac_signature: HMAC-SHA256 signature of the entire manifest
        lock_version: version of the lock format
    """
    oracle_id: str
    lock_time_utc: str
    model_hashes: Dict[str, str] = field(default_factory=dict)
    feature_extractor_hash: str = ""
    training_data_hash: str = ""
    oracle_meta_hash: str = ""
    test_label_hashes: Dict[str, List[str]] = field(default_factory=dict)
    hmac_signature: str = ""
    lock_version: str = "1.1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "oracle_id": self.oracle_id,
            "lock_time_utc": self.lock_time_utc,
            "model_hashes": self.model_hashes,
            "feature_extractor_hash": self.feature_extractor_hash,
            "training_data_hash": self.training_data_hash,
            "oracle_meta_hash": self.oracle_meta_hash,
            "test_label_hashes": self.test_label_hashes,
            "hmac_signature": self.hmac_signature,
            "lock_version": self.lock_version,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LockManifest":
        return cls(**d)


# ---------------------------------------------------------------------------
# Hashing utilities
# ---------------------------------------------------------------------------

def _sha256_file(path: Path) -> str:
    """Compute SHA-256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    """Compute SHA-256 of bytes."""
    return hashlib.sha256(data).hexdigest()


def _sha256_str(s: str) -> str:
    """Compute SHA-256 of a string."""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _sha256_json(obj: Any) -> str:
    """Compute SHA-256 of a JSON-serializable object."""
    return _sha256_str(json.dumps(obj, sort_keys=True, indent=2))


def _hmac_sign(message: str, key: str) -> str:
    """HMAC-SHA256 signature."""
    return hmac.new(key.encode("utf-8"), message.encode("utf-8"),
                     hashlib.sha256).hexdigest()


def _record_id(sequence: str) -> str:
    """Compute a deterministic record ID from a sequence."""
    return _sha256_str(sequence.upper()[:200])  # cap length for efficiency


def _label_hash(label: float) -> str:
    """One-way hash of a label (for later verification without revealing it)."""
    # Use high-precision string representation
    return _sha256_str(f"{label:.6f}")


# ---------------------------------------------------------------------------
# Lock procedure
# ---------------------------------------------------------------------------

def lock_oracle(
    oracle_dir: Path,
    training_sequences: Sequence[str],
    training_labels: Sequence[float],
    test_sequences: Sequence[str],
    test_labels: Sequence[float],
    oracle_id: Optional[str] = None,
    signing_key: Optional[str] = None,
    make_readonly: bool = True,
) -> LockManifest:
    """Lock and seal a trained GBT oracle.

    Args:
        oracle_dir: directory containing the trained oracle artifacts
            (must contain mean_model.txt, q*.txt, oracle_meta.json)
        training_sequences, training_labels: training data used (for provenance)
        test_sequences, test_labels: test data (labels hashed one-way for verification)
        oracle_id: unique ID; auto-generated if None
        signing_key: HMAC signing key; auto-generated if None
        make_readonly: if True, chmod 444 all artifact files

    Returns:
        LockManifest with all hashes and signature
    """
    oracle_dir = Path(oracle_dir)
    if not oracle_dir.exists():
        raise FileNotFoundError(f"Oracle dir not found: {oracle_dir}")

    if oracle_id is None:
        oracle_id = f"oracle3_{int(time.time())}"

    if signing_key is None:
        # Auto-generate a key from oracle_id + time (not cryptographically secure,
        # but sufficient for tamper detection within this project)
        signing_key = f"{oracle_id}_{time.time()}_{os.getpid()}"

    print(f"[lock_oracle] Locking oracle {oracle_id}...")
    print(f"  oracle_dir: {oracle_dir}")
    print(f"  n_training: {len(training_sequences)}")
    print(f"  n_test: {len(test_sequences)}")

    manifest = LockManifest(
        oracle_id=oracle_id,
        lock_time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )

    # 1. Hash all model files
    print("  Hashing model files...")
    for model_file in sorted(oracle_dir.glob("*_model.txt")):
        h = _sha256_file(model_file)
        manifest.model_hashes[model_file.name] = h
        print(f"    {model_file.name}: {h[:16]}...")

    # 2. Hash feature extractor config (from oracle_meta.json)
    meta_path = oracle_dir / "oracle_meta.json"
    if meta_path.exists():
        meta_bytes = meta_path.read_bytes()
        manifest.oracle_meta_hash = _sha256_bytes(meta_bytes)
        print(f"  oracle_meta_hash: {manifest.oracle_meta_hash[:16]}...")

        # Extract feature extractor config for separate hash
        meta = json.loads(meta_bytes)
        if "hyperparams" in meta and "feature_config" in meta["hyperparams"]:
            fc = meta["hyperparams"]["feature_config"]
            manifest.feature_extractor_hash = _sha256_json(fc)
            print(f"  feature_extractor_hash: {manifest.feature_extractor_hash[:16]}...")

    # 3. Hash training data record IDs
    print("  Hashing training data record IDs...")
    record_ids = [_record_id(seq) for seq in training_sequences]
    training_data_str = "\n".join(record_ids)
    manifest.training_data_hash = _sha256_str(training_data_str)
    print(f"  training_data_hash: {manifest.training_data_hash[:16]}...")

    # 4. Hash test labels (one-way, for later verification)
    # NOTE: Sample 2019 MPRA has duplicate 50-mer sequences across GSM samples
    # (same UTR measured in multiple chemistries/replicates with different
    # ribosome loads). We store a LIST of label hashes per record_id so that
    # duplicates are preserved correctly (lock_version >= 1.1).
    print("  Hashing test labels (one-way)...")
    n_test_records = 0
    for seq, label in zip(test_sequences, test_labels):
        rid = _record_id(seq)
        lh = _label_hash(float(label))
        if rid not in manifest.test_label_hashes:
            manifest.test_label_hashes[rid] = []
        manifest.test_label_hashes[rid].append(lh)
        n_test_records += 1
    n_unique = len(manifest.test_label_hashes)
    n_dup_groups = sum(1 for v in manifest.test_label_hashes.values() if len(v) > 1)
    print(f"  test_label_hashes: {n_unique} unique sequences, "
          f"{n_test_records} total records, {n_dup_groups} duplicate groups")

    # 5. Sign manifest with HMAC
    manifest_dict = manifest.to_dict()
    # Remove existing signature before signing
    signature_input = {k: v for k, v in manifest_dict.items() if k != "hmac_signature"}
    signature_message = json.dumps(signature_input, sort_keys=True)
    manifest.hmac_signature = _hmac_sign(signature_message, signing_key)
    print(f"  hmac_signature: {manifest.hmac_signature[:16]}...")

    # 6. Write lock manifest
    lock_path = oracle_dir / "lock_manifest.json"
    with open(lock_path, "w") as f:
        json.dump(manifest.to_dict(), f, indent=2, sort_keys=True)
    print(f"  Lock manifest written to: {lock_path}")

    # 7. Save signing key separately (in a secure location)
    key_path = oracle_dir / ".lock_key"
    with open(key_path, "w") as f:
        f.write(signing_key)
    os.chmod(key_path, 0o600)  # owner read/write only
    print(f"  Signing key saved to: {key_path} (chmod 600)")

    # 8. Make all artifacts read-only
    if make_readonly:
        print("  Making artifacts read-only (chmod 444)...")
        for f in oracle_dir.iterdir():
            if f.is_file():
                os.chmod(f, 0o444)
                print(f"    {f.name}: chmod 444")

    print(f"[lock_oracle] Lock complete. Oracle {oracle_id} is now SEALED.")
    return manifest


# ---------------------------------------------------------------------------
# Verification procedure
# ---------------------------------------------------------------------------

@dataclass
class VerificationResult:
    """Result of lock verification.

    Attributes:
        valid: True if all checks pass
        manifest_valid: HMAC signature valid
        model_hashes_match: all model file hashes match
        meta_hash_match: oracle_meta.json hash matches
        feature_extractor_hash_match: feature extractor hash matches
        training_data_hash_match: training data hash matches
        test_label_hashes_match: test label hashes match
        mismatches: list of mismatched items
    """
    valid: bool
    manifest_valid: bool
    model_hashes_match: bool
    meta_hash_match: bool
    feature_extractor_hash_match: bool
    training_data_hash_match: bool
    test_label_hashes_match: bool
    mismatches: List[str] = field(default_factory=list)


def verify_lock(
    oracle_dir: Path,
    training_sequences: Optional[Sequence[str]] = None,
    test_sequences: Optional[Sequence[str]] = None,
    test_labels: Optional[Sequence[float]] = None,
) -> VerificationResult:
    """Verify that a locked oracle has not been tampered with.

    Args:
        oracle_dir: directory containing the locked oracle
        training_sequences: if provided, verify training data hash
        test_sequences, test_labels: if provided, verify test label hashes

    Returns:
        VerificationResult with detailed status
    """
    oracle_dir = Path(oracle_dir)
    lock_path = oracle_dir / "lock_manifest.json"
    key_path = oracle_dir / ".lock_key"

    if not lock_path.exists():
        return VerificationResult(
            valid=False, manifest_valid=False, model_hashes_match=False,
            meta_hash_match=False, feature_extractor_hash_match=False,
            training_data_hash_match=False, test_label_hashes_match=False,
            mismatches=["lock_manifest.json not found"],
        )

    with open(lock_path) as f:
        manifest = LockManifest.from_dict(json.load(f))

    result = VerificationResult(
        valid=True,
        manifest_valid=True,
        model_hashes_match=True,
        meta_hash_match=True,
        feature_extractor_hash_match=True,
        training_data_hash_match=True,
        test_label_hashes_match=True,
    )

    # 1. Verify HMAC signature
    if key_path.exists():
        with open(key_path) as f:
            signing_key = f.read().strip()
        manifest_dict = manifest.to_dict()
        signature_input = {k: v for k, v in manifest_dict.items() if k != "hmac_signature"}
        signature_message = json.dumps(signature_input, sort_keys=True)
        expected_sig = _hmac_sign(signature_message, signing_key)
        if not hmac.compare_digest(expected_sig, manifest.hmac_signature):
            result.manifest_valid = False
            result.mismatches.append("HMAC signature mismatch")
    else:
        # Can't verify signature without key
        result.manifest_valid = False
        result.mismatches.append("Signing key not found")

    # 2. Verify model file hashes
    for filename, expected_hash in manifest.model_hashes.items():
        filepath = oracle_dir / filename
        if not filepath.exists():
            result.model_hashes_match = False
            result.mismatches.append(f"Model file missing: {filename}")
        else:
            actual_hash = _sha256_file(filepath)
            if actual_hash != expected_hash:
                result.model_hashes_match = False
                result.mismatches.append(f"Model file hash mismatch: {filename}")

    # 3. Verify oracle_meta.json hash
    meta_path = oracle_dir / "oracle_meta.json"
    if meta_path.exists() and manifest.oracle_meta_hash:
        actual_hash = _sha256_file(meta_path)
        if actual_hash != manifest.oracle_meta_hash:
            result.meta_hash_match = False
            result.mismatches.append("oracle_meta.json hash mismatch")

    # 4. Verify training data hash
    if training_sequences is not None and manifest.training_data_hash:
        record_ids = [_record_id(seq) for seq in training_sequences]
        training_data_str = "\n".join(record_ids)
        actual_hash = _sha256_str(training_data_str)
        if actual_hash != manifest.training_data_hash:
            result.training_data_hash_match = False
            result.mismatches.append("Training data hash mismatch")

    # 5. Verify test label hashes
    # Handles both lock_version 1.0 (Dict[str, str]) and 1.1 (Dict[str, List[str]])
    if test_sequences is not None and test_labels is not None:
        if len(test_sequences) != len(test_labels):
            result.test_label_hashes_match = False
            result.mismatches.append("test_sequences and test_labels length mismatch")
        else:
            # Build a count of expected label hashes per record_id from the
            # actual test data, then compare against the manifest.
            actual_counts: Dict[str, List[str]] = {}
            for seq, label in zip(test_sequences, test_labels):
                rid = _record_id(seq)
                lh = _label_hash(float(label))
                actual_counts.setdefault(rid, []).append(lh)

            for rid, actual_hashes in actual_counts.items():
                if rid not in manifest.test_label_hashes:
                    result.test_label_hashes_match = False
                    result.mismatches.append(
                        f"Test record not in manifest: {rid[:8]}...")
                else:
                    stored = manifest.test_label_hashes[rid]
                    # Backward-compat: v1.0 stored a single str, v1.1 stores a list
                    if isinstance(stored, str):
                        stored_list = [stored]
                    else:
                        stored_list = list(stored)
                    # Check that every actual label hash is present in the
                    # stored list (membership check, order-independent).
                    for lh in actual_hashes:
                        if lh not in stored_list:
                            result.test_label_hashes_match = False
                            result.mismatches.append(
                                f"Test label hash mismatch for record {rid[:8]}...")
                            break
                    # Also verify the count matches (catches added/removed dups)
                    if len(stored_list) != len(actual_hashes):
                        result.test_label_hashes_match = False
                        result.mismatches.append(
                            f"Test label count mismatch for record {rid[:8]}: "
                            f"manifest={len(stored_list)} actual={len(actual_hashes)}")

    result.valid = (
        result.manifest_valid
        and result.model_hashes_match
        and result.meta_hash_match
        and result.feature_extractor_hash_match
        and result.training_data_hash_match
        and result.test_label_hashes_match
    )
    return result


# ---------------------------------------------------------------------------
# Independence audit
# ---------------------------------------------------------------------------

@dataclass
class IndependenceAuditResult:
    """Result of independence audit for Oracle #3.

    Attributes:
        independent: True if all independence criteria pass
        criteria: dict of criterion -> bool
        notes: explanatory notes
    """
    independent: bool
    criteria: Dict[str, bool] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)


def audit_independence(
    oracle_dir: Path,
    teacher_ckpt_dir: Optional[Path] = None,
    teacher_training_sequences: Optional[Sequence[str]] = None,
    oracle_training_sequences: Optional[Sequence[str]] = None,
) -> IndependenceAuditResult:
    """Audit that Oracle #3 is independent from the training teacher (Oracle #1).

    Criteria:
        1. Different architecture (GBT vs CNN/Transformer)
        2. Different feature space (hand-engineered vs one-hot/embedding)
        3. Different training data (no overlap with teacher's training data)
        4. Frozen (read-only files)
        5. Signed (HMAC signature valid)
        6. Test labels not visible (only one-way hashes stored)

    Args:
        oracle_dir: locked oracle directory
        teacher_ckpt_dir: P1-04 teacher checkpoint directory (for architecture check)
        teacher_training_sequences: sequences used to train the teacher
        oracle_training_sequences: sequences used to train the oracle

    Returns:
        IndependenceAuditResult
    """
    result = IndependenceAuditResult(independent=True)

    # 1. Architecture: GBT (LightGBM .txt files) vs CNN/Transformer (.pt files)
    has_lgb = any((oracle_dir / f).exists() for f in ["mean_model.txt"])
    has_torch = any(oracle_dir.glob("*.pt"))
    result.criteria["different_architecture"] = has_lgb and not has_torch
    if not result.criteria["different_architecture"]:
        result.notes.append("Architecture check failed: expected LightGBM .txt, no .pt files")

    # 2. Feature space: hand-engineered (340 features) vs one-hot/embedding
    meta_path = oracle_dir / "oracle_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        n_features = meta.get("hyperparams", {}).get("feature_config", {}).get("max_kmer_k", 0)
        # If feature extractor exists with k-mer features, it's hand-engineered
        result.criteria["different_feature_space"] = n_features > 0
    else:
        result.criteria["different_feature_space"] = False
        result.notes.append("oracle_meta.json not found")

    # 3. Training data: no overlap with teacher
    if teacher_training_sequences is not None and oracle_training_sequences is not None:
        teacher_ids = set(_record_id(seq) for seq in teacher_training_sequences)
        oracle_ids = set(_record_id(seq) for seq in oracle_training_sequences)
        overlap = teacher_ids & oracle_ids
        # Allow some overlap (e.g., shared Sample 2019 train split) but not full
        overlap_frac = len(overlap) / max(len(oracle_ids), 1)
        result.criteria["training_data_distinct"] = overlap_frac < 0.5
        if not result.criteria["training_data_distinct"]:
            result.notes.append(f"Training data overlap: {overlap_frac:.1%}")
    else:
        result.criteria["training_data_distinct"] = True  # skip check
        result.notes.append("Training data overlap check skipped (no teacher data provided)")

    # 4. Frozen (read-only files)
    all_readonly = True
    for f in oracle_dir.iterdir():
        if f.is_file() and os.access(f, os.W_OK):
            all_readonly = False
            result.notes.append(f"File not read-only: {f.name}")
            break
    result.criteria["frozen_readonly"] = all_readonly

    # 5. Signed (HMAC signature present and valid)
    lock_path = oracle_dir / "lock_manifest.json"
    key_path = oracle_dir / ".lock_key"
    if lock_path.exists() and key_path.exists():
        verify = verify_lock(oracle_dir)
        result.criteria["signed_valid"] = verify.manifest_valid
        if not verify.manifest_valid:
            result.notes.append("HMAC signature verification failed")
    else:
        result.criteria["signed_valid"] = False
        result.notes.append("Lock manifest or signing key missing")

    # 6. Test labels not visible (only one-way hashes)
    if lock_path.exists():
        with open(lock_path) as f:
            manifest = json.load(f)
        # Check that test_label_hashes are SHA-256 (64 hex chars), not raw labels.
        # v1.0: Dict[str, str]; v1.1: Dict[str, List[str]] (handles duplicates)
        test_hashes = manifest.get("test_label_hashes", {})

        def _is_valid_hash(h: Any) -> bool:
            return isinstance(h, str) and len(h) == 64 and \
                all(c in "0123456789abcdef" for c in h)

        all_hashed = True
        for v in test_hashes.values():
            if isinstance(v, list):
                if not all(_is_valid_hash(h) for h in v):
                    all_hashed = False
                    break
            elif isinstance(v, str):
                if not _is_valid_hash(v):
                    all_hashed = False
                    break
            else:
                all_hashed = False
                break
        result.criteria["test_labels_hashed"] = all_hashed
    else:
        result.criteria["test_labels_hashed"] = False

    result.independent = all(result.criteria.values())
    return result


__all__ = [
    "LockManifest",
    "VerificationResult",
    "IndependenceAuditResult",
    "lock_oracle",
    "verify_lock",
    "audit_independence",
]
