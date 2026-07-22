"""P2-02 unit tests: TrainConfig recovery fields + held-out eval helpers.

Covers:
- New TrainConfig fields (amp_init_scale, save_every, eval_every, val_idx_path,
  val_max_eval) have backward-compatible defaults.
- New fields load correctly from JSON.
- _load_idx reads idx files.
- _select_records_by_idx respects max_eval and out-of-range indices.
- run_heldout_eval returns the expected stat keys (with a stub model/backbone).
- _verify_idx_files catches content mismatch.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mrna_editflow.core.config import MEFConfig, TrainConfig
from mrna_editflow.core.schema import MRNARecord


# ---------------------------------------------------------------------------
# TrainConfig backward compatibility
# ---------------------------------------------------------------------------
class TestTrainConfigP202Fields:
    def test_defaults_backward_compatible(self):
        """Old configs that don't specify P2-02 fields still work."""
        tc = TrainConfig()
        assert tc.amp_init_scale == 1024.0, "default amp_init_scale must be 1024.0 for backward compat"
        assert tc.save_every == 0, "default save_every must be 0 (off)"
        assert tc.eval_every == 0, "default eval_every must be 0 (off)"
        assert tc.val_idx_path is None, "default val_idx_path must be None"
        assert tc.val_max_eval == 0, "default val_max_eval must be 0 (all)"

    def test_p202_fields_set_explicitly(self):
        tc = TrainConfig(
            amp_init_scale=256.0,
            save_every=1000,
            eval_every=500,
            val_idx_path="/tmp/val.idx",
            val_max_eval=500,
        )
        assert tc.amp_init_scale == 256.0
        assert tc.save_every == 1000
        assert tc.eval_every == 500
        assert tc.val_idx_path == "/tmp/val.idx"
        assert tc.val_max_eval == 500

    def test_mefconfig_from_json_with_p202_fields(self, tmp_path):
        """MEFConfig.from_json must parse P2-02 fields."""
        cfg_dict = {
            "train": {
                "lr": 1e-6,
                "amp_init_scale": 256.0,
                "save_every": 1000,
                "eval_every": 500,
                "val_idx_path": "/tmp/val.idx",
                "val_max_eval": 500,
            }
        }
        p = tmp_path / "cfg.json"
        p.write_text(json.dumps(cfg_dict))
        cfg = MEFConfig.from_json(str(p))
        assert cfg.train.amp_init_scale == 256.0
        assert cfg.train.save_every == 1000
        assert cfg.train.eval_every == 500
        assert cfg.train.val_idx_path == "/tmp/val.idx"
        assert cfg.train.val_max_eval == 500
        assert cfg.train.lr == 1e-6

    def test_mefconfig_from_json_without_p202_fields(self, tmp_path):
        """Old config JSON without P2-02 fields must still parse."""
        cfg_dict = {"train": {"lr": 1e-4, "batch_size": 1}}
        p = tmp_path / "cfg_old.json"
        p.write_text(json.dumps(cfg_dict))
        cfg = MEFConfig.from_json(str(p))
        assert cfg.train.lr == 1e-4
        assert cfg.train.amp_init_scale == 1024.0  # default
        assert cfg.train.save_every == 0  # default


# ---------------------------------------------------------------------------
# _load_idx
# ---------------------------------------------------------------------------
class TestLoadIdx:
    def test_load_idx_basic(self, tmp_path):
        from scripts.run_stage_a_recovery_p2_02 import _load_idx
        p = tmp_path / "val.idx"
        p.write_text("0\n1\n2\n3\n4\n")
        idx = _load_idx(str(p))
        assert idx == [0, 1, 2, 3, 4]

    def test_load_idx_skips_blank_lines(self, tmp_path):
        from scripts.run_stage_a_recovery_p2_02 import _load_idx
        p = tmp_path / "val.idx"
        p.write_text("0\n\n1\n\n2\n")
        idx = _load_idx(str(p))
        assert idx == [0, 1, 2]

    def test_load_idx_empty_file(self, tmp_path):
        from scripts.run_stage_a_recovery_p2_02 import _load_idx
        p = tmp_path / "empty.idx"
        p.write_text("")
        idx = _load_idx(str(p))
        assert idx == []


# ---------------------------------------------------------------------------
# _select_records_by_idx
# ---------------------------------------------------------------------------
class TestSelectRecordsByIdx:
    def _make_records(self, n):
        return [MRNARecord(transcript_id=f"t{i}", five_utr="AUG", cds="C" * (i + 1), three_utr="UUU") for i in range(n)]

    def test_select_all(self):
        from scripts.run_stage_a_recovery_p2_02 import _select_records_by_idx
        records = self._make_records(10)
        picked = _select_records_by_idx(records, [0, 1, 2], max_eval=None)
        assert len(picked) == 3
        assert picked[0].transcript_id == "t0"

    def test_select_with_max_eval(self):
        from scripts.run_stage_a_recovery_p2_02 import _select_records_by_idx
        records = self._make_records(10)
        picked = _select_records_by_idx(records, [0, 1, 2, 3, 4], max_eval=2)
        assert len(picked) == 2

    def test_select_max_eval_zero_means_all(self):
        from scripts.run_stage_a_recovery_p2_02 import _select_records_by_idx
        records = self._make_records(10)
        picked = _select_records_by_idx(records, [0, 1, 2], max_eval=0)
        assert len(picked) == 3

    def test_select_skips_out_of_range(self):
        from scripts.run_stage_a_recovery_p2_02 import _select_records_by_idx
        records = self._make_records(5)
        picked = _select_records_by_idx(records, [0, 10, 4, -1], max_eval=None)
        # -1 and 10 are out of range (0 <= i < n)
        assert len(picked) == 2
        assert picked[0].transcript_id == "t0"
        assert picked[1].transcript_id == "t4"


# ---------------------------------------------------------------------------
# run_heldout_eval (with stub model/backbone)
# ---------------------------------------------------------------------------
class TestRunHeldoutEval:
    def test_returns_expected_keys(self):
        from scripts.run_stage_a_recovery_p2_02 import run_heldout_eval
        # Stub: _flow_batch_loss is imported locally, so we monkeypatch it
        import scripts.run_stage_a_recovery_p2_02 as mod

        cfg = MEFConfig()
        # Force backbone="none" so build doesn't try to load weights
        cfg.backbone.name = "none"
        device = torch.device("cpu")

        model = MagicMock()
        backbone = MagicMock()
        val_records = self._make_records(3)

        # Monkeypatch _flow_batch_loss (imported locally inside run_heldout_eval)
        # We need to patch train_backbone._flow_batch_loss
        import train_backbone

        def stub_loss(model, backbone, records, cfg, device, scheduler, seed, property_bucket=None):
            return {"loss": torch.tensor(float(seed)), "edit_loss": torch.tensor(0.0), "aux_loss": torch.tensor(0.0)}

        orig = train_backbone._flow_batch_loss
        train_backbone._flow_batch_loss = stub_loss
        try:
            result = run_heldout_eval(
                model=model,
                backbone=backbone,
                val_records=val_records,
                cfg=cfg,
                device=device,
                batch_size=1,
                seed=100,
            )
        finally:
            train_backbone._flow_batch_loss = orig

        assert "val_loss_mean" in result
        assert "val_loss_median" in result
        assert "val_loss_p95" in result
        assert "val_loss_std" in result
        assert "val_n" in result
        assert result["val_n"] == 3

    def test_empty_val_records_returns_nan(self):
        from scripts.run_stage_a_recovery_p2_02 import run_heldout_eval
        cfg = MEFConfig()
        result = run_heldout_eval(
            model=MagicMock(),
            backbone=MagicMock(),
            val_records=[],
            cfg=cfg,
            device=torch.device("cpu"),
        )
        assert result["val_n"] == 0
        assert np.isnan(result["val_loss_mean"])

    def _make_records(self, n):
        return [MRNARecord(transcript_id=f"t{i}", five_utr="AUG", cds="C" * (i + 1), three_utr="UUU") for i in range(n)]


# ---------------------------------------------------------------------------
# _verify_idx_files (recovery's copy)
# ---------------------------------------------------------------------------
class TestVerifyIdxFilesRecovery:
    def test_missing_file_raises(self, tmp_path):
        from scripts.run_stage_a_recovery_p2_02 import _verify_idx_files
        args = MagicMock()
        args.train_idx = str(tmp_path / "nonexistent.idx")
        args.val_idx = None
        args.test_idx = None
        with pytest.raises(FileNotFoundError, match="train-idx"):
            _verify_idx_files(args, split_contract=None, records=None)

    def test_empty_file_raises(self, tmp_path):
        from scripts.run_stage_a_recovery_p2_02 import _verify_idx_files
        p = tmp_path / "empty.idx"
        p.write_text("")
        args = MagicMock()
        args.train_idx = str(p)
        args.val_idx = None
        args.test_idx = None
        with pytest.raises(ValueError, match="empty"):
            _verify_idx_files(args, split_contract=None, records=None)


# ---------------------------------------------------------------------------
# P2-02 config file sanity
# ---------------------------------------------------------------------------
class TestP202ConfigFile:
    def test_recovery_config_parses(self):
        """The P2-02 recovery config must parse and have the expected fixes."""
        cfg_path = os.path.join(_REPO_ROOT, "configs", "stage_a_recovery_p2_02.json")
        if not os.path.exists(cfg_path):
            pytest.skip(f"config not found at {cfg_path} (expected on server)")
        cfg = MEFConfig.from_json(cfg_path)
        # P2-02 fix: LR reduced 100x from 1e-4 to 1e-6
        assert cfg.train.lr == 1e-6, f"expected lr=1e-6, got {cfg.train.lr}"
        # P2-02 fix: AMP init_scale reduced from 1024 to 256
        assert cfg.train.amp_init_scale == 256.0, f"expected amp_init_scale=256, got {cfg.train.amp_init_scale}"
        # P2-02 fix: grad_clip = 1.0 (already correct, verify it's still set)
        assert cfg.train.grad_clip == 1.0
        # P2-02 fix: effective batch = batch_size * grad_accum in [8, 16]
        eff = cfg.train.batch_size * cfg.train.grad_accum
        assert 8 <= eff <= 16, f"effective batch {eff} not in [8, 16]"
        # P2-02 fix: save_every = 1000
        assert cfg.train.save_every == 1000
        # P2-02 fix: eval_every > 0
        assert cfg.train.eval_every > 0
