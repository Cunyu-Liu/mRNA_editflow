"""FM0-01 tests: hash/license manifest, exposure audit, from-scratch, LoRA,
partial unfreeze, determinism, input-length.

Run on server:
    cd /home/cunyuliu/mrna_editflow_goal/mrna_editflow
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    /home/cunyuliu/miniconda3/envs/pc_cng/bin/python -m pytest \
        d1_staging/tests/test_fm0_acceptance.py -v
"""

import json
import os
import sys
from pathlib import Path

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
FM0_SCRIPTS = os.path.join(HERE, "..", "scripts", "fm0")
sys.path.insert(0, FM0_SCRIPTS)

from fm0_common import (  # noqa: E402
    ensure_offline_env,
    get_snapshot_dir,
    load_config,
)


def _snapshot_available() -> bool:
    try:
        ensure_offline_env()
        get_snapshot_dir()
        return True
    except Exception:
        return False


SNAPSHOT_OK = _snapshot_available()
fm0_real = pytest.mark.skipif(
    not SNAPSHOT_OK,
    reason="UTR-LM snapshot not in HF cache.",
)


# ---------------------------------------------------------------------------
# Hash / license manifest
# ---------------------------------------------------------------------------

@fm0_real
def test_hash_license_manifest_builds(tmp_path):
    from fm0_hash_license_manifest import build_manifest
    ensure_offline_env()
    manifest = build_manifest()
    assert manifest["model_id"] == "multimolecule/utrlm-mrl"
    assert manifest["license"]["type"] == "agpl-3.0"
    # All expected files present
    assert not manifest["missing_files"], f"missing: {manifest['missing_files']}"
    # Each file has a sha256
    for f in manifest["files"]:
        assert len(f["sha256"]) == 64
        assert f["size_bytes"] > 0
    # License text non-empty
    assert len(manifest["license"]["license_text"]) > 100


@fm0_real
def test_hash_license_manifest_main_writes_json(tmp_path):
    from fm0_hash_license_manifest import main as hash_main
    out = tmp_path / "manifest.json"
    sys.argv = ["fm0_hash_license_manifest.py", "--output", str(out)]
    try:
        hash_main()
    finally:
        sys.argv = [sys.argv[0]]
    assert out.exists()
    data = json.loads(out.read_text())
    assert data["task_id"] == "FM0-01"
    assert data["license"]["type"] == "agpl-3.0"


# ---------------------------------------------------------------------------
# Load + tokenize
# ---------------------------------------------------------------------------

@fm0_real
def test_load_and_tokenize_report_passes(tmp_path):
    from fm0_load_and_tokenize import run_load_and_tokenize
    report = run_load_and_tokenize()
    assert report["pass"] is True
    assert report["config_mismatches_vs_yaml"] == []
    assert report["forward_smoke"]["device"].startswith("cuda")
    # T->U applied for at least one example
    assert any(ex["T_to_U_applied"] for ex in report["tokenization_examples"])


# ---------------------------------------------------------------------------
# Embedding determinism (GPU required by the active contract)
# ---------------------------------------------------------------------------

@fm0_real
def test_embedding_determinism_passes_gpu():
    from fm0_embedding_determinism import run_determinism
    report = run_determinism("auto")
    assert report["pass"] is True
    assert report["device"].startswith("cuda")
    assert report["bit_exact_two_runs"] is True
    assert report["max_abs_diff"] == 0.0
    for mode, r in report["pooling_results"].items():
        assert r["bit_exact"] is True, f"mode={mode} not bit-exact"


# ---------------------------------------------------------------------------
# Input-length behavior (GPU required by the active contract)
# ---------------------------------------------------------------------------

@fm0_real
def test_input_length_behavior_passes_gpu():
    from fm0_input_length_behavior import run_input_length
    report = run_input_length("auto")
    assert report["pass"] is True
    assert report["device"].startswith("cuda")
    # No NaN/Inf anywhere
    for r in report["results"]:
        assert not r["any_nan"]
        assert not r["any_inf"]
    # Over-max inputs truncated (1022+50 and 2000)
    long_results = [r for r in report["results"] if r["input_nt_count"] > 1022]
    assert len(long_results) >= 2
    for r in long_results:
        assert r["truncated_by_tokenizer"] is True


# ---------------------------------------------------------------------------
# From-scratch control
# ---------------------------------------------------------------------------

@fm0_real
def test_from_scratch_control_passes_gpu():
    from fm0_from_scratch_control import run_from_scratch
    report = run_from_scratch("auto", seed=20260801)
    assert report["pass"] is True
    assert report["device"].startswith("cuda")
    assert report["architecture_matches"] is True
    assert report["weights_differ"] is True
    assert report["forward_check"]["deterministic_two_runs"] is True
    # Scratch output should differ from pretrained (proves weights differ)
    assert report["forward_check"]["max_abs_diff_vs_pretrained"] > 1e-3


# ---------------------------------------------------------------------------
# Exposure audit
# ---------------------------------------------------------------------------

def test_exposure_audit_passes_with_data(tmp_path):
    """Exposure audit can run even without the checkpoint (uses config + data)."""
    from fm0_exposure_audit import run_exposure_audit, build_exposure_entries

    entries = build_exposure_entries()
    assert len(entries) >= 4
    gse114002 = next(e for e in entries if e["accession"] == "GSE114002")
    assert gse114002["historically_exposed_to_utrlm"] is True
    assert gse114002["exposure_type"] == "sequence_prior_only"
    assert gse114002["labels_exposed_to_utrlm"] is False
    assert gse114002["evidence_grade_for_foundation_eval"] == "E4"

    gse200304 = next(e for e in entries if e["accession"] == "GSE200304")
    assert gse200304["historically_exposed_to_utrlm"] is False
    assert gse200304["evidence_grade_for_foundation_eval"] == "E5"


def test_exposure_audit_main_writes_json(tmp_path):
    from fm0_exposure_audit import main as audit_main
    # Use repo-root data paths if present, else use empty paths (audit still works)
    repo_root = Path(__file__).resolve().parents[2]
    records = repo_root / "data" / "d1_canonical_records.jsonl"
    b0 = repo_root / "data" / "b0_splits"
    out = tmp_path / "audit.json"
    sys.argv = [
        "fm0_exposure_audit.py",
        "--canonical-records", str(records),
        "--b0-splits-dir", str(b0),
        "--output", str(out),
    ]
    try:
        audit_main()
    finally:
        sys.argv = [sys.argv[0]]
    assert out.exists()
    data = json.loads(out.read_text())
    assert data["pass"] is True
    assert "GSE114002" in data["summary"]["exposed_accessions"]


# ---------------------------------------------------------------------------
# GPU-only tests (skip if no CUDA). These cover acceptance gates that MUST run
# on GPU per contract: GPU forward, memory/latency, frozen cache, LoRA, partial
# unfreeze.
# ---------------------------------------------------------------------------

def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


needs_cuda = pytest.mark.skipif(
    not _cuda_available(),
    reason="CUDA not available; contract requires GPU_only for these gates.",
)


@fm0_real
@needs_cuda
def test_gpu_forward_passes():
    from fm0_gpu_forward import run_gpu_forward
    report = run_gpu_forward("auto")
    assert report["pass"] is True
    assert report["device"].startswith("cuda")
    for r in report["test_cases"]:
        assert r["output_device"].startswith("cuda")
        assert r["output_dtype"] == "torch.float32"


@fm0_real
@needs_cuda
def test_memory_latency_passes():
    from fm0_memory_latency import run_memory_latency
    report = run_memory_latency("auto", warmup=1, iters=2)
    assert report["pass"] is True
    assert len(report["results"]) >= 5
    for r in report["results"]:
        assert r["per_batch_ms"] > 0
        assert r["peak_allocated_mb"] > 0


@fm0_real
@needs_cuda
def test_lora_passes():
    from fm0_lora import run_lora
    report = run_lora("auto")
    assert report["pass"] is True
    assert report["after_peft"]["num_parameters_trainable"] > 0
    assert report["after_peft"]["num_parameters_trainable"] == report["after_peft"]["expected_trainable"]
    assert len(report["non_lora_trainable_params"]) == 0


@fm0_real
@needs_cuda
def test_partial_unfreeze_passes():
    from fm0_partial_unfreeze import run_partial_unfreeze
    report = run_partial_unfreeze("auto", unfreeze_last_n=2)
    assert report["pass"] is True
    assert report["num_parameters_trainable"] > 0
    assert report["num_parameters_trainable"] < report["num_parameters_total"]
    assert report["embedding_trainable_count"] == 0
    assert report["pooler_trainable_count"] == 0
    assert report["num_parameters_trainable"] == report["expected_trainable"]


@fm0_real
@needs_cuda
def test_frozen_cache_smoke(tmp_path):
    """Frozen cache smoke test: tiny max_records to keep it fast."""
    from fm0_frozen_cache import run_frozen_cache
    repo_root = Path(__file__).resolve().parents[2]
    records = repo_root / "data" / "d1_canonical_records.jsonl"
    if not records.exists():
        pytest.skip("d1_canonical_records.jsonl not present")
    cache_dir = tmp_path / "frozen_cache"
    report = run_frozen_cache(records, cache_dir, "auto", batch_size=8, max_records=50)
    assert report["pass"] is True
    assert report["num_total_embedded"] > 0
    # Each .npz file should be loadable and have correct shape
    import numpy as np
    for w in report["written_files"]:
        d = np.load(w["path"], allow_pickle=True)
        assert d["embedding"].shape == (w["num_records"], w["hidden_size"])
        assert d["hidden_size"] == 128
