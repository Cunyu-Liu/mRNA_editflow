"""PyTorch inference port of the official Saluki GRU model (Agarwal & Kelley 2022).

Source of truth: the frozen ``model_config`` embedded in the official Zenodo
checkpoints (``datasets/deeplearning/train_gru/f*_c*/train/model{0,1}_best.h5``
of DOI 10.5281/zenodo.6326409), 43 layers:

    Conv1D(64, k=5, valid, no bias) -> LN(eps 0.007) -> ReLU
    x6 [Conv1D(64, k=5, valid, bias) -> Dropout(0.3, train-only)
        -> MaxPool1D(2, 2) -> LN(eps 0.007) -> ReLU]
    GRU(64, reset_after=True) -> BatchNorm(eps 0.001) -> ReLU
    Dense(64) -> Dropout -> BatchNorm(eps 0.001) -> ReLU -> Dense(1)

Input: ``(B, L, 6)`` float tensor. Channel layout (official ``dna_1hot`` /
``RnaDataset`` parse order): ``[A, C, G, T/U, codon-frame, splice5p]``. All
sequences in one batch must share one length (no internal padding; callers
group equal-length sequences). Minimum length 320 so the pooled trunk
(conv k=5 + six [conv k=5, pool /2] blocks) keeps at least one position;
shorter UTR inputs must be right-padded via ``seq_len`` (the official
inference length is 12288).

Keras -> PyTorch mapping notes (the two real porting hazards):
- GRU gate order: Keras kernel columns are ``[z, r, h]`` (update, reset, new)
  while ``torch.nn.GRU`` rows are ``[r, z, n]``; rows are reordered on load.
- Keras ``reset_after=True`` semantics coincide with the PyTorch GRU update
  rule ``n_t = tanh(W_in x + b_in + r_t * (W_hn h + b_hn))``.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

SALUKI_PORT_SCHEMA_V1 = "route_a_v3_route2_saluki_port.v1"
SALUKI_INPUT_CHANNELS = 6
SALUKI_FILTERS = 64
SALUKI_KERNEL = 5
SALUKI_POOL_BLOCKS = 6
SALUKI_LN_EPSILON = 0.007
SALUKI_BN_EPSILON = 0.001
SALUKI_MIN_LENGTH = 68  # (L - 4) // 2**6 >= 1


class SalukiPortError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SalukiPortError(message)


def _reorder_gru_gates(matrix: np.ndarray, axis: int) -> np.ndarray:
    """Map Keras GRU gate order [z, r, h] onto PyTorch [r, z, n] along ``axis``."""
    size = SALUKI_FILTERS * 3
    _require(matrix.shape[axis] == size, "Saluki GRU matrix gate dimension changed")
    z, r, h = np.split(matrix, 3, axis=axis)
    return np.concatenate([r, z, h], axis=axis)


def encode_saluki_six_channel_v1(sequence: str, seq_len: int | None = None) -> np.ndarray:
    """Encode a UTR sequence into the official 6-channel layout.

    Channels 0-3 are the A/C/G/T(U) one-hot in the official ``dna_1hot`` order.
    Channels 4-5 replicate the official ``tf.one_hot(v, 1)`` indicator
    semantics: the channel value is 1.0 at ordinary positions and 0.0 where the
    annotation marks a codon-first / splice5p position. A UTR-only input
    without CDS/splice annotation is therefore all ordinary positions, i.e.
    channels 4-5 are all ones. ``seq_len`` right-pads with zeros or crops to
    the last ``seq_len`` positions, matching the official helper.
    """
    text = str(sequence).upper().replace("U", "T")
    _require(len(text) > 0, "Saluki input sequence is empty")
    table = {"A": 0, "C": 1, "G": 2, "T": 3}
    encoded = np.zeros((len(text), SALUKI_INPUT_CHANNELS), dtype=np.float32)
    for index, nucleotide in enumerate(text):
        slot = table.get(nucleotide)
        if slot is not None:
            encoded[index, slot] = 1.0
    encoded[:, 4] = 1.0  # ordinary-position frame indicator (no CDS annotation)
    encoded[:, 5] = 1.0  # ordinary-position splice5p indicator (no splice annotation)
    if seq_len is not None:
        if len(text) < seq_len:
            padding = np.zeros((seq_len - len(text), SALUKI_INPUT_CHANNELS), dtype=np.float32)
            encoded = np.concatenate([encoded, padding], axis=0)
        else:
            encoded = encoded[-seq_len:]
    return encoded


class SalukiGRUV1(nn.Module):
    """Frozen inference port of one official Saluki fold checkpoint."""

    def __init__(self, weight_path: Path):
        super().__init__()
        arrays = self._read_checkpoint(weight_path)

        self.register_buffer(
            "conv0_weight",
            torch.from_numpy(arrays["conv0_kernel"].astype(np.float32).transpose(2, 1, 0).copy()),
        )
        self.convs = nn.ModuleList(
            nn.Conv1d(SALUKI_FILTERS, SALUKI_FILTERS, SALUKI_KERNEL)
            for _ in range(SALUKI_POOL_BLOCKS)
        )
        with torch.no_grad():
            for index, module in enumerate(self.convs):
                module.weight.copy_(
                    torch.from_numpy(
                        arrays["conv_kernels"][index].astype(np.float32).transpose(2, 1, 0).copy()
                    )
                )
                module.bias.copy_(torch.from_numpy(arrays["conv_biases"][index].astype(np.float32).copy()))

        for index in range(SALUKI_POOL_BLOCKS + 1):
            self.register_buffer(
                f"ln{index}_weight",
                torch.from_numpy(arrays["ln_gammas"][index].astype(np.float32).copy()),
            )
            self.register_buffer(
                f"ln{index}_bias",
                torch.from_numpy(arrays["ln_betas"][index].astype(np.float32).copy()),
            )

        self.gru = nn.GRU(SALUKI_FILTERS, SALUKI_FILTERS, batch_first=False)
        weight_ih = _reorder_gru_gates(arrays["gru_kernel"], axis=1).T
        weight_hh = _reorder_gru_gates(arrays["gru_recurrent"], axis=1).T
        bias_input = _reorder_gru_gates(arrays["gru_bias"][0], axis=0)
        bias_recurrent = _reorder_gru_gates(arrays["gru_bias"][1], axis=0)
        with torch.no_grad():
            self.gru.weight_ih_l0.copy_(torch.from_numpy(weight_ih.astype(np.float32).copy()))
            self.gru.weight_hh_l0.copy_(torch.from_numpy(weight_hh.astype(np.float32).copy()))
            self.gru.bias_ih_l0.copy_(torch.from_numpy(bias_input.astype(np.float32).copy()))
            self.gru.bias_hh_l0.copy_(torch.from_numpy(bias_recurrent.astype(np.float32).copy()))

        for key, prefix in (("bn0", "bn0"), ("bn1", "bn1")):
            for tag, name in (
                ("weight", "gamma"),
                ("bias", "beta"),
                ("mean", "moving_mean"),
                ("var", "moving_variance"),
            ):
                self.register_buffer(
                    f"{key}_{tag}",
                    torch.from_numpy(arrays[f"{prefix}_{name}"].astype(np.float32).copy()),
                )

        self.register_buffer("dense_weight", torch.from_numpy(arrays["dense_kernel"].astype(np.float32).T.copy()))
        self.register_buffer("dense_bias", torch.from_numpy(arrays["dense_bias"].astype(np.float32).copy()))
        self.register_buffer("dense1_weight", torch.from_numpy(arrays["dense1_kernel"].astype(np.float32).T.copy()))
        self.register_buffer("dense1_bias", torch.from_numpy(arrays["dense1_bias"].astype(np.float32).copy()))

    @staticmethod
    def _read_checkpoint(weight_path: Path) -> dict[str, np.ndarray]:
        _require(Path(weight_path).is_file(), f"Saluki checkpoint is absent: {weight_path}")
        arrays: dict[str, np.ndarray] = {}

        def load(handle: h5py.File, path: str) -> np.ndarray:
            dataset = handle[f"model_weights/{path}"]
            _require(isinstance(dataset, h5py.Dataset), f"Saluki h5 path is not a dataset: {path}")
            return np.asarray(dataset)

        with h5py.File(weight_path, "r") as handle:
            conv0 = load(handle, "conv1d/conv1d/kernel:0")
            _require(
                tuple(conv0.shape) == (SALUKI_KERNEL, SALUKI_INPUT_CHANNELS, SALUKI_FILTERS),
                "Saluki conv0 geometry changed",
            )
            arrays["conv0_kernel"] = conv0

            kernels, biases = [], []
            for index in range(1, SALUKI_POOL_BLOCKS + 1):
                kernel = load(handle, f"conv1d_{index}/conv1d_{index}/kernel:0")
                bias = load(handle, f"conv1d_{index}/conv1d_{index}/bias:0")
                _require(
                    tuple(kernel.shape) == (SALUKI_KERNEL, SALUKI_FILTERS, SALUKI_FILTERS),
                    f"Saluki conv{index} geometry changed",
                )
                _require(tuple(bias.shape) == (SALUKI_FILTERS,), f"Saluki conv{index} bias geometry changed")
                kernels.append(kernel)
                biases.append(bias)
            arrays["conv_kernels"] = kernels
            arrays["conv_biases"] = biases

            ln_names = ["layer_normalization"] + [
                f"layer_normalization_{index}" for index in range(1, SALUKI_POOL_BLOCKS + 1)
            ]
            gammas, betas = [], []
            for name in ln_names:
                gamma = load(handle, f"{name}/{name}/gamma:0")
                beta = load(handle, f"{name}/{name}/beta:0")
                _require(
                    tuple(gamma.shape) == (SALUKI_FILTERS,) and tuple(beta.shape) == (SALUKI_FILTERS,),
                    f"Saluki {name} geometry changed",
                )
                gammas.append(gamma)
                betas.append(beta)
            arrays["ln_gammas"] = gammas
            arrays["ln_betas"] = betas

            arrays["gru_kernel"] = load(handle, "gru/gru/gru_cell/kernel:0")
            arrays["gru_recurrent"] = load(handle, "gru/gru/gru_cell/recurrent_kernel:0")
            arrays["gru_bias"] = load(handle, "gru/gru/gru_cell/bias:0")
            _require(tuple(arrays["gru_kernel"].shape) == (SALUKI_FILTERS, SALUKI_FILTERS * 3), "Saluki GRU kernel geometry changed")
            _require(tuple(arrays["gru_recurrent"].shape) == (SALUKI_FILTERS, SALUKI_FILTERS * 3), "Saluki GRU recurrent geometry changed")
            _require(tuple(arrays["gru_bias"].shape) == (2, SALUKI_FILTERS * 3), "Saluki GRU bias geometry changed")

            for key, name in (("bn0", "batch_normalization"), ("bn1", "batch_normalization_1")):
                for tag, h5_name in (
                    ("gamma", "gamma:0"),
                    ("beta", "beta:0"),
                    ("moving_mean", "moving_mean:0"),
                    ("moving_variance", "moving_variance:0"),
                ):
                    values = load(handle, f"{name}/{name}/{h5_name}")
                    _require(tuple(values.shape) == (SALUKI_FILTERS,), f"Saluki {name} {tag} geometry changed")
                    arrays[f"{key}_{tag}"] = values

            arrays["dense_kernel"] = load(handle, "dense/dense/kernel:0")
            arrays["dense_bias"] = load(handle, "dense/dense/bias:0")
            arrays["dense1_kernel"] = load(handle, "dense_1/dense_1/kernel:0")
            arrays["dense1_bias"] = load(handle, "dense_1/dense_1/bias:0")
            _require(tuple(arrays["dense_kernel"].shape) == (SALUKI_FILTERS, SALUKI_FILTERS), "Saluki dense geometry changed")
            _require(tuple(arrays["dense1_kernel"].shape) == (SALUKI_FILTERS, 1), "Saluki dense_1 geometry changed")

        return arrays

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        _require(
            inputs.ndim == 3 and inputs.shape[2] == SALUKI_INPUT_CHANNELS,
            "Saluki input must be (B, L, 6)",
        )
        length = int(inputs.shape[1])
        _require(
            length >= SALUKI_MIN_LENGTH,
            f"Saluki input length {length} is below the pooled-trunk minimum {SALUKI_MIN_LENGTH}",
        )
        x = F.conv1d(inputs.transpose(1, 2), self.conv0_weight)
        x = x.transpose(1, 2)
        x = F.layer_norm(x, (SALUKI_FILTERS,), self.ln0_weight, self.ln0_bias, SALUKI_LN_EPSILON)
        x = F.relu(x)
        for index in range(SALUKI_POOL_BLOCKS):
            x = self.convs[index](x.transpose(1, 2)).transpose(1, 2)
            x = F.max_pool1d(x.transpose(1, 2), kernel_size=2, stride=2).transpose(1, 2)
            x = F.layer_norm(
                x,
                (SALUKI_FILTERS,),
                getattr(self, f"ln{index + 1}_weight"),
                getattr(self, f"ln{index + 1}_bias"),
                SALUKI_LN_EPSILON,
            )
            x = F.relu(x)
        sequence = x.transpose(0, 1)
        output, _ = self.gru(sequence)
        hidden = output[-1]
        hidden = F.batch_norm(
            hidden,
            self.bn0_mean,
            self.bn0_var,
            self.bn0_weight,
            self.bn0_bias,
            training=False,
            eps=SALUKI_BN_EPSILON,
        )
        hidden = F.relu(hidden)
        hidden = F.linear(hidden, self.dense_weight, self.dense_bias)
        hidden = F.batch_norm(
            hidden,
            self.bn1_mean,
            self.bn1_var,
            self.bn1_weight,
            self.bn1_bias,
            training=False,
            eps=SALUKI_BN_EPSILON,
        )
        hidden = F.relu(hidden)
        return F.linear(hidden, self.dense1_weight, self.dense1_bias).squeeze(-1)
