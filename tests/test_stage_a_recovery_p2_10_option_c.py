"""Unit tests for P2-10 Option C: Stage A recovery with stricter fixes.

Tests cover:
1. ``_apply_lr_warmup`` — linear warmup schedule correctness.
2. Config validation — amp=false, batch_size=8, grad_accum=8, lr=1e-6.
3. CLI argument parsing — --warmup-steps default and override.
4. Launcher script structure — required paths and parameters present.

All tests are offline (no GPU, no model loading, no data loading).
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest
import torch

# Ensure repo root is on sys.path for imports.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_PACKAGE_PARENT = _REPO_ROOT.parent
for _p in (str(_PACKAGE_PARENT), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scripts.run_stage_a_recovery_p2_10_option_c import (
    _apply_lr_warmup,
    _parse_args,
)


# ---------------------------------------------------------------------------
# 1. _apply_lr_warmup tests
# ---------------------------------------------------------------------------
class TestApplyLrWarmup:
    """Test the linear LR warmup schedule."""

    def _make_optimizer(self, initial_lr: float = 0.0) -> torch.optim.Optimizer:
        param = torch.nn.Parameter(torch.zeros(1))
        return torch.optim.SGD([param], lr=initial_lr)

    def test_warmup_disabled_returns_target(self):
        """When warmup_steps=0, LR should always be target_lr."""
        opt = self._make_optimizer()
        lr = _apply_lr_warmup(opt, step=1, warmup_steps=0, target_lr=1e-6)
        assert lr == 1e-6
        assert opt.param_groups[0]["lr"] == 1e-6

    def test_warmup_step_zero_returns_zero(self):
        """At step 0, LR should be 0 (start of warmup)."""
        opt = self._make_optimizer()
        lr = _apply_lr_warmup(opt, step=0, warmup_steps=500, target_lr=1e-6)
        assert lr == 0.0
        assert opt.param_groups[0]["lr"] == 0.0

    def test_warmup_step_one_returns_fraction(self):
        """At step 1 of 500, LR should be target_lr * 1/500."""
        opt = self._make_optimizer()
        lr = _apply_lr_warmup(opt, step=1, warmup_steps=500, target_lr=1e-6)
        expected = 1e-6 * 1.0 / 500.0
        assert abs(lr - expected) < 1e-15
        assert abs(opt.param_groups[0]["lr"] - expected) < 1e-15

    def test_warmup_midpoint_returns_half(self):
        """At step 250 of 500, LR should be target_lr * 0.5."""
        opt = self._make_optimizer()
        lr = _apply_lr_warmup(opt, step=250, warmup_steps=500, target_lr=1e-6)
        expected = 1e-6 * 250.0 / 500.0
        assert abs(lr - expected) < 1e-15

    def test_warmup_complete_returns_target(self):
        """At step >= warmup_steps, LR should be target_lr."""
        opt = self._make_optimizer()
        lr = _apply_lr_warmup(opt, step=500, warmup_steps=500, target_lr=1e-6)
        assert lr == 1e-6
        assert opt.param_groups[0]["lr"] == 1e-6

    def test_warmup_past_end_returns_target(self):
        """At step > warmup_steps, LR should still be target_lr (constant)."""
        opt = self._make_optimizer()
        lr = _apply_lr_warmup(opt, step=9999, warmup_steps=500, target_lr=1e-6)
        assert lr == 1e-6

    def test_warmup_monotonic_increase(self):
        """LR should increase monotonically during warmup."""
        opt = self._make_optimizer()
        lrs: List[float] = []
        for step in range(1, 501):
            lrs.append(_apply_lr_warmup(opt, step=step, warmup_steps=500, target_lr=1e-6))
        for i in range(1, len(lrs)):
            assert lrs[i] >= lrs[i - 1], f"LR decreased at step {i+1}"

    def test_warmup_updates_all_param_groups(self):
        """If optimizer has multiple param_groups, all should be updated."""
        p1 = torch.nn.Parameter(torch.zeros(1))
        p2 = torch.nn.Parameter(torch.zeros(1))
        opt = torch.optim.SGD(
            [{"params": [p1], "lr": 0.0}, {"params": [p2], "lr": 0.0}],
            lr=0.0,
        )
        lr = _apply_lr_warmup(opt, step=250, warmup_steps=500, target_lr=1e-6)
        expected = 1e-6 * 250.0 / 500.0
        assert abs(opt.param_groups[0]["lr"] - expected) < 1e-15
        assert abs(opt.param_groups[1]["lr"] - expected) < 1e-15

    def test_warmup_negative_step_returns_zero(self):
        """Negative step should be treated as step 0 (LR=0)."""
        opt = self._make_optimizer()
        lr = _apply_lr_warmup(opt, step=-5, warmup_steps=500, target_lr=1e-6)
        assert lr == 0.0

    def test_warmup_negative_warmup_steps_disables(self):
        """Negative warmup_steps should disable warmup (return target)."""
        opt = self._make_optimizer()
        lr = _apply_lr_warmup(opt, step=1, warmup_steps=-1, target_lr=1e-6)
        assert lr == 1e-6


# ---------------------------------------------------------------------------
# 2. Config validation tests
# ---------------------------------------------------------------------------
class TestP2_10Config:
    """Test that the P2-10 Option C config has the correct settings."""

    @pytest.fixture
    def config_path(self) -> str:
        return str(_REPO_ROOT / "configs" / "stage_a_recovery_p2_10_option_c.json")

    def test_config_file_exists(self, config_path: str):
        """Config file must exist."""
        assert os.path.exists(config_path), f"Config not found: {config_path}"

    def test_config_loads_as_json(self, config_path: str):
        """Config must be valid JSON."""
        with open(config_path) as fh:
            cfg = json.load(fh)
        assert isinstance(cfg, dict)

    def test_config_amp_disabled(self, config_path: str):
        """AMP must be disabled (the #1 fix for numerical stability)."""
        with open(config_path) as fh:
            cfg = json.load(fh)
        assert cfg["train"]["amp"] is False, "AMP must be false for P2-10 Option C"

    def test_config_batch_size_is_8(self, config_path: str):
        """batch_size must be 8 (2x larger than P2-02's 4)."""
        with open(config_path) as fh:
            cfg = json.load(fh)
        assert cfg["train"]["batch_size"] == 8

    def test_config_grad_accum_is_8(self, config_path: str):
        """grad_accum must be 8 (2x larger than P2-02's 4, effective batch=64)."""
        with open(config_path) as fh:
            cfg = json.load(fh)
        assert cfg["train"]["grad_accum"] == 8

    def test_config_effective_batch_is_64(self, config_path: str):
        """Effective batch (batch_size * grad_accum) must be 64."""
        with open(config_path) as fh:
            cfg = json.load(fh)
        effective = cfg["train"]["batch_size"] * cfg["train"]["grad_accum"]
        assert effective == 64, f"Expected effective batch 64, got {effective}"

    def test_config_lr_is_1e_6(self, config_path: str):
        """LR must be 1e-6 (same as P2-02, already 100x lower than original 1e-4)."""
        with open(config_path) as fh:
            cfg = json.load(fh)
        assert cfg["train"]["lr"] == 0.000001

    def test_config_grad_clip_is_1(self, config_path: str):
        """grad_clip must be 1.0 (same as P2-02, already enforced correctly)."""
        with open(config_path) as fh:
            cfg = json.load(fh)
        assert cfg["train"]["grad_clip"] == 1.0

    def test_config_eval_every_is_250(self, config_path: str):
        """eval_every must be 250 (more frequent than P2-02's 500)."""
        with open(config_path) as fh:
            cfg = json.load(fh)
        assert cfg["train"]["eval_every"] == 250

    def test_config_save_every_is_500(self, config_path: str):
        """save_every must be 500 (more frequent than P2-02's 1000)."""
        with open(config_path) as fh:
            cfg = json.load(fh)
        assert cfg["train"]["save_every"] == 500

    def test_config_save_dir_is_p2_10(self, config_path: str):
        """save_dir must reference p2_10_option_c (not p2_02)."""
        with open(config_path) as fh:
            cfg = json.load(fh)
        assert "p2_10_option_c" in cfg["train"]["save_dir"]

    def test_config_aux_struct_disabled(self, config_path: str):
        """use_aux_struct must be false (same as P2-02, no target tensor at eval)."""
        with open(config_path) as fh:
            cfg = json.load(fh)
        assert cfg["model"]["use_aux_struct"] is False

    def test_config_oom_ladder_starts_at_8(self, config_path: str):
        """oom_batch_ladder must start at 8 (matching batch_size=8)."""
        with open(config_path) as fh:
            cfg = json.load(fh)
        ladder = cfg["train"]["oom_batch_ladder"]
        assert ladder[0] == 8, f"Expected ladder[0]=8, got {ladder[0]}"


# ---------------------------------------------------------------------------
# 3. CLI argument parsing tests
# ---------------------------------------------------------------------------
class TestParseArgs:
    """Test CLI argument parsing for P2-10 Option C."""

    def _base_args(self) -> List[str]:
        return [
            "--config", "configs/stage_a_recovery_p2_10_option_c.json",
            "--records-jsonl", "data/combined.jsonl",
            "--train-idx", "train.idx",
            "--val-idx", "val.idx",
            "--test-idx", "test.idx",
        ]

    def test_warmup_steps_default_is_500(self):
        """--warmup-steps must default to 500."""
        args = _parse_args(self._base_args())
        assert args.warmup_steps == 500

    def test_warmup_steps_override(self):
        """--warmup-steps must accept custom values."""
        argv = self._base_args() + ["--warmup-steps", "1000"]
        args = _parse_args(argv)
        assert args.warmup_steps == 1000

    def test_warmup_steps_zero(self):
        """--warmup-steps 0 must be accepted (disables warmup)."""
        argv = self._base_args() + ["--warmup-steps", "0"]
        args = _parse_args(argv)
        assert args.warmup_steps == 0

    def test_config_required(self):
        """--config must be required."""
        with pytest.raises(SystemExit):
            _parse_args([])

    def test_records_jsonl_required(self):
        """--records-jsonl must be required."""
        argv = ["--config", "foo.json"]
        with pytest.raises(SystemExit):
            _parse_args(argv)

    def test_run_mode_development(self):
        """--run-mode development must be accepted."""
        argv = self._base_args() + ["--run-mode", "development"]
        args = _parse_args(argv)
        assert args.run_mode == "development"

    def test_run_mode_paper_default(self):
        """--run-mode must default to paper."""
        args = _parse_args(self._base_args())
        assert args.run_mode == "paper"

    def test_all_split_idx_args(self):
        """All split idx arguments must be parsed."""
        args = _parse_args(self._base_args())
        assert args.train_idx == "train.idx"
        assert args.val_idx == "val.idx"
        assert args.test_idx == "test.idx"


# ---------------------------------------------------------------------------
# 4. Launcher script structure tests
# ---------------------------------------------------------------------------
class TestLauncherScript:
    """Test that the launcher script has the correct structure."""

    @pytest.fixture
    def launcher_path(self) -> str:
        return str(_REPO_ROOT / "scripts" / "launch_p2_10_option_c.sh")

    def test_launcher_exists(self, launcher_path: str):
        """Launcher script must exist."""
        assert os.path.exists(launcher_path)

    def test_launcher_references_p2_10_script(self, launcher_path: str):
        """Launcher must reference run_stage_a_recovery_p2_10_option_c."""
        with open(launcher_path) as fh:
            content = fh.read()
        assert "run_stage_a_recovery_p2_10_option_c" in content

    def test_launcher_references_p2_10_config(self, launcher_path: str):
        """Launcher must reference the P2-10 config file."""
        with open(launcher_path) as fh:
            content = fh.read()
        assert "stage_a_recovery_p2_10_option_c.json" in content

    def test_launcher_has_warmup_steps(self, launcher_path: str):
        """Launcher must pass --warmup-steps."""
        with open(launcher_path) as fh:
            content = fh.read()
        assert "--warmup-steps" in content

    def test_launcher_has_split_contract(self, launcher_path: str):
        """Launcher must enforce split contract (--train-idx/--val-idx/--test-idx)."""
        with open(launcher_path) as fh:
            content = fh.read()
        assert "--train-idx" in content
        assert "--val-idx" in content
        assert "--test-idx" in content

    def test_launcher_has_run_mode_paper(self, launcher_path: str):
        """Launcher must use --run-mode paper."""
        with open(launcher_path) as fh:
            content = fh.read()
        assert "--run-mode paper" in content

    def test_launcher_default_gpu_is_0(self, launcher_path: str):
        """Launcher must default to GPU 0."""
        with open(launcher_path) as fh:
            content = fh.read()
        assert "P2_10_GPU:-0" in content or 'P2_10_GPU:-"0"' in content

    def test_launcher_does_not_use_gpu_4(self, launcher_path: str):
        """Launcher must NOT default to GPU 4 (forbidden — calibrate)."""
        with open(launcher_path) as fh:
            content = fh.read()
        # GPU 4 should not appear as a default
        assert 'P2_10_GPU:-4' not in content
        assert 'P2_10_GPU:-"4"' not in content

    def test_launcher_sets_cuda_visible_devices(self, launcher_path: str):
        """Launcher must set CUDA_VISIBLE_DEVICES."""
        with open(launcher_path) as fh:
            content = fh.read()
        assert "CUDA_VISIBLE_DEVICES" in content
