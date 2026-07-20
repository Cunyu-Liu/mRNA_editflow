"""P1-04 Architecture A: CNN-50mer (Optimus100K-style).

Replicates Sample 2019 NBT architecture for 5'UTR -> ribosome load regression.

Architecture:
    Input: (N, 4, L) one-hot (ACGU)
    Conv1d(4->64, k=8) + BN + ReLU
    Conv1d(64->64, k=8) + BN + ReLU
    Conv1d(64->128, k=8) + BN + ReLU
    MaxPool1d(2)
    Conv1d(128->128, k=8) + BN + ReLU
    Conv1d(128->256, k=8) + BN + ReLU
    MaxPool1d(2)
    Flatten
    Linear(256 * (L//4) -> 256) + ReLU + Dropout(0.3)
    Linear(256 -> 1)

Training: AdamW lr=1e-3 weight_decay=1e-5 batch=512 30 epochs MSE on standardized log1p(label)
MC dropout for uncertainty: enable dropout at inference, n_mc_samples forward passes.
"""
from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Relative imports (when installed under models/predictors/)
try:
    from .base import PredictorBase, PredictionResult
except ImportError:
    from base import PredictorBase, PredictionResult  # type: ignore


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

@dataclass
class CNN50merHyperparams:
    """Hyperparameters for CNN-50mer predictor.

    Defaults follow Sample 2019 NBT architecture and training recipe.
    """
    max_len: int = 50                # sequence length (pad/truncate)
    alphabet: str = "ACGU"
    conv_channels: Tuple[int, int, int, int, int] = (64, 64, 128, 128, 256)
    kernel_size: int = 8
    pool_size: int = 2
    dropout: float = 0.3
    hidden_dim: int = 256
    # Training
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 512
    n_epochs: int = 30
    patience: int = 5                # early stopping patience
    log_transform: bool = True       # log1p(label) before standardization
    # MC dropout
    n_mc_samples: int = 30
    # Runtime
    device: str = "cuda"
    num_workers: int = 2

    def to_dict(self) -> Dict[str, Any]:
        d = self.__dict__.copy()
        d["conv_channels"] = list(self.conv_channels)
        return d


# ---------------------------------------------------------------------------
# Torch model
# ---------------------------------------------------------------------------

def _build_torch_model(hp: CNN50merHyperparams) -> Any:
    """Build the torch CNN model."""
    import torch
    import torch.nn as nn

    class CNN50mer(nn.Module):
        def __init__(self, hp: CNN50merHyperparams):
            super().__init__()
            self.hp = hp
            in_ch = len(hp.alphabet)
            chans = hp.conv_channels
            k = hp.kernel_size
            pad = k // 2

            self.conv1 = nn.Conv1d(in_ch, chans[0], k, padding=pad)
            self.bn1 = nn.BatchNorm1d(chans[0])
            self.conv2 = nn.Conv1d(chans[0], chans[1], k, padding=pad)
            self.bn2 = nn.BatchNorm1d(chans[1])
            self.conv3 = nn.Conv1d(chans[1], chans[2], k, padding=pad)
            self.bn3 = nn.BatchNorm1d(chans[2])
            self.pool1 = nn.MaxPool1d(hp.pool_size, hp.pool_size)

            self.conv4 = nn.Conv1d(chans[2], chans[3], k, padding=pad)
            self.bn4 = nn.BatchNorm1d(chans[3])
            self.conv5 = nn.Conv1d(chans[3], chans[4], k, padding=pad)
            self.bn5 = nn.BatchNorm1d(chans[4])
            self.pool2 = nn.MaxPool1d(hp.pool_size, hp.pool_size)

            # Compute flattened dim by dry forward (pad=k//2 doesn't preserve L exactly
            # for even kernel sizes, so we measure the actual output shape)
            L = hp.max_len
            with torch.no_grad():
                dummy = torch.zeros(1, in_ch, L)
                x = torch.relu(self.bn1(self.conv1(dummy)))
                x = torch.relu(self.bn2(self.conv2(x)))
                x = torch.relu(self.bn3(self.conv3(x)))
                x = self.pool1(x)
                x = torch.relu(self.bn4(self.conv4(x)))
                x = torch.relu(self.bn5(self.conv5(x)))
                x = self.pool2(x)
            self.flatten_dim = int(x.numel())  # 1 * C * L_flat

            self.fc1 = nn.Linear(self.flatten_dim, hp.hidden_dim)
            self.drop = nn.Dropout(hp.dropout)
            self.fc_out = nn.Linear(hp.hidden_dim, 1)
            self.relu = nn.ReLU()

        def forward(self, x: Any) -> Any:
            # x: (N, in_ch, L)
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.relu(self.bn2(self.conv2(x)))
            x = self.relu(self.bn3(self.conv3(x)))
            x = self.pool1(x)
            x = self.relu(self.bn4(self.conv4(x)))
            x = self.relu(self.bn5(self.conv5(x)))
            x = self.pool2(x)
            x = x.flatten(1)
            x = self.relu(self.fc1(x))
            x = self.drop(x)
            x = self.fc_out(x)
            return x.squeeze(-1)

    return CNN50mer(hp)


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------

class CNN50merPredictor(PredictorBase):
    """CNN-50mer predictor (Architecture A).

    Implements PredictorBase interface. Trains via AdamW + MSE on
    standardized log1p(label). Uncertainty via MC dropout.
    """

    def __init__(
        self,
        name: str,
        dataset_name: str,
        fold_idx: int,
        seed: int = 42,
        device: str = "cuda",
        hyperparams: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name, dataset_name, fold_idx, seed, device, hyperparams)
        # Parse hyperparams
        hp_dict = self.hyperparams or {}
        self.hp = CNN50merHyperparams(**{k: v for k, v in hp_dict.items()
                                          if k in CNN50merHyperparams.__dataclass_fields__})
        if "device" in hp_dict:
            self.hp.device = hp_dict["device"]
        self.device = self.hp.device
        # Standardization params (computed at fit time, persisted via hyperparams)
        self._y_mean: float = float(hp_dict.get("_y_mean", 0.0))
        self._y_std: float = float(hp_dict.get("_y_std", 1.0))
        self._model: Any = None

    # ------------------------------------------------------------------
    # Required abstract methods
    # ------------------------------------------------------------------

    def _build_model(self) -> Any:
        self._model = _build_torch_model(self.hp)
        self._model.to(self.device)

    def _set_seed(self) -> None:
        import torch
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def _encode(self, sequences: Sequence[str]) -> Any:
        """Encode list of sequences to (N, 4, L) one-hot tensor."""
        import torch
        L = self.hp.max_len
        A = len(self.hp.alphabet)
        char_to_idx = {c: i for i, c in enumerate(self.hp.alphabet)}
        arr = np.zeros((len(sequences), A, L), dtype=np.float32)
        for i, seq in enumerate(sequences):
            for j, ch in enumerate(seq[:L]):
                if ch in char_to_idx:
                    arr[i, char_to_idx[ch], j] = 1.0
        return torch.from_numpy(arr)

    def _transform_label(self, y: np.ndarray) -> np.ndarray:
        if self.hp.log_transform:
            y = np.log1p(np.maximum(y, 0))
        return (y - self._y_mean) / max(self._y_std, 1e-8)

    def _inverse_transform(self, y: np.ndarray) -> np.ndarray:
        return np.expm1(y * self._y_std + self._y_mean) if self.hp.log_transform \
            else (y * self._y_std + self._y_mean)

    def fit(
        self,
        X_train: Sequence[str],
        y_train: Sequence[float],
        X_val: Optional[Sequence[str]] = None,
        y_val: Optional[Sequence[float]] = None,
    ) -> Dict[str, List[float]]:
        """Train via AdamW + MSE."""
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        self._set_seed()
        if self._model is None:
            self._build_model()

        y_train_arr = np.asarray(y_train, dtype=np.float32)
        # Compute standardization on train labels
        if self.hp.log_transform:
            yt = np.log1p(np.maximum(y_train_arr, 0))
        else:
            yt = y_train_arr
        self._y_mean = float(yt.mean())
        self._y_std = float(yt.std() + 1e-8)
        yt_norm = (yt - self._y_mean) / self._y_std

        # Build tensors
        X_train_t = self._encode(X_train)
        y_train_t = torch.from_numpy(yt_norm)
        train_ds = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(
            train_ds, batch_size=self.hp.batch_size, shuffle=True,
            num_workers=self.hp.num_workers, drop_last=False,
        )

        # Val
        val_loader = None
        if X_val is not None and y_val is not None:
            X_val_t = self._encode(X_val)
            y_val_arr = np.asarray(y_val, dtype=np.float32)
            y_val_t = torch.from_numpy(self._transform_label(y_val_arr))
            val_ds = TensorDataset(X_val_t, y_val_t)
            val_loader = DataLoader(val_ds, batch_size=self.hp.batch_size,
                                     shuffle=False, num_workers=self.hp.num_workers)

        optimizer = torch.optim.AdamW(
            self._model.parameters(), lr=self.hp.learning_rate,
            weight_decay=self.hp.weight_decay,
        )
        criterion = nn.MSELoss()

        history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "val_pearson": []}
        best_val_loss = float("inf")
        best_state = None
        epochs_since_improve = 0

        for epoch in range(self.hp.n_epochs):
            self._model.train()
            epoch_loss = 0.0
            n_batches = 0
            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                optimizer.zero_grad()
                pred = self._model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item())
                n_batches += 1
            train_loss = epoch_loss / max(n_batches, 1)
            history["train_loss"].append(train_loss)

            # Val
            if val_loader is not None:
                self._model.eval()
                val_loss_sum = 0.0
                preds_list, targets_list = [], []
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb = xb.to(self.device)
                        yb = yb.to(self.device)
                        pred = self._model(xb)
                        loss = criterion(pred, yb)
                        val_loss_sum += float(loss.item())
                        preds_list.append(pred.cpu().numpy())
                        targets_list.append(yb.cpu().numpy())
                val_loss = val_loss_sum / max(len(val_loader), 1)
                preds = np.concatenate(preds_list)
                targets = np.concatenate(targets_list)
                try:
                    from scipy.stats import pearsonr
                    val_r = float(pearsonr(preds, targets)[0])
                except Exception:
                    val_r = 0.0
                history["val_loss"].append(val_loss)
                history["val_pearson"].append(val_r)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.detach().clone() for k, v in self._model.state_dict().items()}
                    epochs_since_improve = 0
                else:
                    epochs_since_improve += 1
                    if epochs_since_improve >= self.hp.patience:
                        break
            else:
                history["val_loss"].append(float("nan"))
                history["val_pearson"].append(float("nan"))

        # Restore best
        if best_state is not None:
            self._model.load_state_dict(best_state)

        # Persist standardization params via hyperparams (so save/load roundtrips)
        self.hyperparams["_y_mean"] = self._y_mean
        self.hyperparams["_y_std"] = self._y_std

        self._fitted = True
        return history

    def predict(self, X: Sequence[str]) -> np.ndarray:
        """Mean prediction (no MC dropout)."""
        import torch
        if not self._fitted or self._model is None:
            raise RuntimeError("Predictor not fitted")
        self._model.eval()
        with torch.no_grad():
            xt = self._encode(X).to(self.device)
            # Forward in batches
            preds = []
            bs = max(self.hp.batch_size, 1)
            for i in range(0, len(xt), bs):
                batch = xt[i:i + bs]
                pred = self._model(batch).cpu().numpy()
                preds.append(pred)
        y_pred = np.concatenate(preds) if preds else np.array([])
        return self._inverse_transform(y_pred)

    def predict_with_uncertainty(
        self, X: Sequence[str], n_mc_samples: int = 30,
    ) -> PredictionResult:
        """MC dropout uncertainty estimation.

        Enable dropout at inference, run n_mc_samples forward passes,
        return mean + std across samples.
        """
        import torch
        if not self._fitted or self._model is None:
            raise RuntimeError("Predictor not fitted")

        n_mc = max(n_mc_samples or self.hp.n_mc_samples, 1)
        # Enable dropout modules
        self._model.train()
        # But disable BatchNorm updates by setting eval mode on BN layers only
        for m in self._model.modules():
            if isinstance(m, type(self._model.pool1)):  # keep as-is
                pass
        # Re-enable BN in eval mode (we want BN to use running stats, not update them)
        import torch.nn as nn
        for m in self._model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                m.eval()

        bs = max(self.hp.batch_size, 1)
        xt = self._encode(X).to(self.device)
        n = len(xt)
        all_preds = np.zeros((n_mc, n), dtype=np.float32)
        with torch.no_grad():
            for s in range(n_mc):
                preds = []
                for i in range(0, n, bs):
                    batch = xt[i:i + bs]
                    pred = self._model(batch).cpu().numpy()
                    preds.append(pred)
                all_preds[s] = np.concatenate(preds) if preds else np.array([])

        # Inverse transform each sample
        all_preds_inv = np.stack([self._inverse_transform(all_preds[s]) for s in range(n_mc)])
        mean = all_preds_inv.mean(axis=0)
        std = all_preds_inv.std(axis=0)
        return PredictionResult(
            mean=mean.astype(np.float32),
            std=std.astype(np.float32),
            abstain_mask=None,
            metadata={"n_mc_samples": n_mc, "method": "mc_dropout"},
        )


__all__ = ["CNN50merPredictor", "CNN50merHyperparams"]
