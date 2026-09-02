"""Unit tests for the Saluki PyTorch port (core/route2_saluki_port_v1.py)."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from core.route2_saluki_port_v1 import (
    SALUKI_FILTERS,
    SALUKI_INPUT_CHANNELS,
    SALUKI_MIN_LENGTH,
    SalukiGRUV1,
    SalukiPortError,
    encode_saluki_six_channel_v1,
)

SIZE = SALUKI_FILTERS
GATES = SALUKI_FILTERS * 3


def _write_synthetic_checkpoint(path: Path, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    saved: dict[str, np.ndarray] = {}

    with h5py.File(path, "w") as handle:
        weights = handle.create_group("model_weights")

        def write(rel: str, array: np.ndarray) -> None:
            group_name, dataset_name = rel.rsplit("/", 1)
            group = weights.require_group(group_name)
            group.create_dataset(dataset_name, data=array)
            saved[rel] = array

        write("conv1d/conv1d/kernel:0", rng.standard_normal((5, 6, SIZE)).astype(np.float32) * 0.05)
        for index in range(1, 7):
            write(f"conv1d_{index}/conv1d_{index}/kernel:0", rng.standard_normal((5, SIZE, SIZE)).astype(np.float32) * 0.05)
            write(f"conv1d_{index}/conv1d_{index}/bias:0", rng.standard_normal(SIZE).astype(np.float32) * 0.05)
        for name in ["layer_normalization"] + [f"layer_normalization_{i}" for i in range(1, 7)]:
            write(f"{name}/{name}/gamma:0", np.ones(SIZE, dtype=np.float32))
            write(f"{name}/{name}/beta:0", np.zeros(SIZE, dtype=np.float32))
        write("gru/gru/gru_cell/kernel:0", rng.standard_normal((SIZE, GATES)).astype(np.float32) * 0.05)
        write("gru/gru/gru_cell/recurrent_kernel:0", rng.standard_normal((SIZE, GATES)).astype(np.float32) * 0.05)
        write("gru/gru/gru_cell/bias:0", rng.standard_normal((2, GATES)).astype(np.float32) * 0.05)
        for name in ("batch_normalization", "batch_normalization_1"):
            write(f"{name}/{name}/gamma:0", np.ones(SIZE, dtype=np.float32))
            write(f"{name}/{name}/beta:0", np.zeros(SIZE, dtype=np.float32))
            write(f"{name}/{name}/moving_mean:0", np.zeros(SIZE, dtype=np.float32))
            write(f"{name}/{name}/moving_variance:0", np.ones(SIZE, dtype=np.float32))
        write("dense/dense/kernel:0", rng.standard_normal((SIZE, SIZE)).astype(np.float32) * 0.05)
        write("dense/dense/bias:0", rng.standard_normal(SIZE).astype(np.float32) * 0.05)
        write("dense_1/dense_1/kernel:0", rng.standard_normal((SIZE, 1)).astype(np.float32) * 0.05)
        write("dense_1/dense_1/bias:0", rng.standard_normal(1).astype(np.float32) * 0.05)
    return saved


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def test_port_loads_and_forwards(tmp_path: Path) -> None:
    checkpoint = tmp_path / "saluki_synth.h5"
    _write_synthetic_checkpoint(checkpoint)
    model = SalukiGRUV1(checkpoint)
    model.eval()
    generator = torch.Generator().manual_seed(7)
    inputs = torch.bernoulli(torch.full((2, 320, SALUKI_INPUT_CHANNELS), 0.25), generator=generator)
    with torch.no_grad():
        first = model(inputs)
        second = model(inputs.clone())
    assert first.shape == (2,)
    assert torch.isfinite(first).all().item()
    assert torch.equal(first, second)


def test_port_rejects_short_and_malformed_inputs(tmp_path: Path) -> None:
    checkpoint = tmp_path / "saluki_synth.h5"
    _write_synthetic_checkpoint(checkpoint)
    model = SalukiGRUV1(checkpoint)
    model.eval()
    short = torch.zeros(1, SALUKI_MIN_LENGTH - 1, SALUKI_INPUT_CHANNELS)
    with pytest.raises(SalukiPortError, match="minimum"):
        model(short)
    wrong_channels = torch.zeros(1, 512, SALUKI_INPUT_CHANNELS - 1)
    with pytest.raises(SalukiPortError, match="must be"):
        model(wrong_channels)


def test_encoder_matches_official_channel_order() -> None:
    encoded = encode_saluki_six_channel_v1("ACGUN")
    assert encoded.shape == (5, SALUKI_INPUT_CHANNELS)
    assert encoded.dtype == np.float32
    assert encoded[0, 0] == 1.0 and encoded[0, 1:4].sum() == 0.0  # A
    assert encoded[1, 1] == 1.0 and encoded[1, :4].sum() == 1.0  # C
    assert encoded[2, 2] == 1.0                                  # G
    assert encoded[3, 3] == 1.0                                  # U maps onto the T slot
    assert encoded[4, :4].sum() == 0.0                           # unknown base stays zero
    # 2022-era raw indicator semantics: ordinary positions are 0.0 on the
    # coding/splice channels (verified against the official test set).
    assert np.all(encoded[:, 4] == 0.0) and np.all(encoded[:, 5] == 0.0)
    padded = encode_saluki_six_channel_v1("ACGU", seq_len=8)
    assert padded.shape == (8, SALUKI_INPUT_CHANNELS)
    assert padded[4:].sum() == 0.0  # official right-padding is all-zero
    cropped = encode_saluki_six_channel_v1("A" * 12, seq_len=4)
    assert cropped.shape == (4, SALUKI_INPUT_CHANNELS)
    assert cropped[:, 0].sum() == 4.0


def test_gru_step_matches_keras_reset_after_formula(tmp_path: Path) -> None:
    """The decisive porting hazard: gate order and reset_after semantics."""
    checkpoint = tmp_path / "saluki_synth.h5"
    saved = _write_synthetic_checkpoint(checkpoint, seed=11)
    model = SalukiGRUV1(checkpoint)
    model.eval()

    kernel = saved["gru/gru/gru_cell/kernel:0"]          # (64, 192) columns [z, r, h]
    recurrent = saved["gru/gru/gru_cell/recurrent_kernel:0"]
    bias = saved["gru/gru/gru_cell/bias:0"]               # (2, 192): [input, recurrent]

    rng = np.random.default_rng(3)
    x = rng.standard_normal(SIZE).astype(np.float32)
    h_prev = rng.standard_normal(SIZE).astype(np.float32)

    def split_gates(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if matrix.ndim == 1:
            return matrix[:SIZE], matrix[SIZE : 2 * SIZE], matrix[2 * SIZE :]
        return matrix[:, :SIZE], matrix[:, SIZE : 2 * SIZE], matrix[:, 2 * SIZE :]

    wz, wr, wh = split_gates(kernel)
    uz, ur, uh = split_gates(recurrent)
    bz, br, bh = split_gates(bias[0])
    bz2, br2, bh2 = split_gates(bias[1])

    z = _sigmoid(x @ wz + bz + h_prev @ uz + bz2)
    r = _sigmoid(x @ wr + br + h_prev @ ur + br2)
    n = np.tanh(x @ wh + bh + r * (h_prev @ uh + bh2))
    expected = z * h_prev + (1.0 - z) * n

    with torch.no_grad():
        output, _ = model.gru(
            torch.from_numpy(x).view(1, 1, SIZE),
            torch.from_numpy(h_prev).view(1, 1, SIZE),
        )
    actual = output[0, 0].numpy()
    assert np.allclose(expected, actual, atol=1e-5), np.abs(expected - actual).max()


def test_forward_is_batch_consistent(tmp_path: Path) -> None:
    checkpoint = tmp_path / "saluki_synth.h5"
    _write_synthetic_checkpoint(checkpoint)
    model = SalukiGRUV1(checkpoint)
    model.eval()
    generator = torch.Generator().manual_seed(5)
    inputs = torch.bernoulli(torch.full((3, 350, SALUKI_INPUT_CHANNELS), 0.25), generator=generator)
    with torch.no_grad():
        batched = model(inputs)
        singles = torch.stack([model(inputs[i : i + 1]) for i in range(3)])
    # CPU GEMM blocking differs across batch sizes; the deep LayerNorm/GRU
    # stack amplifies the ~1e-6 reassociation noise to ~1e-3 on random
    # synthetic weights. The authoritative correctness gate is the official
    # prediction parity check (run_route2_saluki_port_smoke_v1.py --parity).
    assert torch.allclose(batched, singles, atol=0.01), (batched - singles).abs().max()
