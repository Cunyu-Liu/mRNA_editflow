"""P1-04 Architecture B: Transformer-UTR (UTR-LM-style).

Captures long-range dependencies in variable-length UTR sequences.

Architecture:
    Input: variable-length UTR (N, L_max) tokenized as character-level tokens
    Token embedding (vocab=4 + special tokens, dim=128)
    + Learned positional embedding (max_len=128)
    4 x TransformerEncoderLayer(d_model=128, nhead=8, dim_ff=512, dropout=0.1)
    Mean-pool over valid tokens (mask-aware)
    Linear(128 -> 2) -> (mu, log_var)  # Gaussian head

Training: AdamW lr=3e-4 warmup 1k steps + cosine decay, batch=256, 50 epochs
          NLL loss for Gaussian head: -0.5 * (log(2*pi) + log_var + (y-mu)^2 / exp(log_var))
MC uncertainty: inherent via Gaussian head (predict mu, sigma directly)
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from .base import PredictorBase, PredictionResult
except ImportError:
    from base import PredictorBase, PredictionResult  # type: ignore


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

@dataclass
class TransformerUTRHyperparams:
    """Hyperparameters for Transformer-UTR predictor."""
    max_len: int = 100               # max sequence length (pad/truncate)
    alphabet: str = "ACGU"
    d_model: int = 128
    n_heads: int = 8
    n_layers: int = 4
    dim_ff: int = 512
    dropout: float = 0.1
    # Training
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    batch_size: int = 256
    n_epochs: int = 50
    patience: int = 7
    warmup_steps: int = 1000
    log_transform: bool = True
    # Runtime
    device: str = "cuda"
    num_workers: int = 2
    # Gaussian head
    min_log_var: float = -6.0        # clamp log_var for numerical stability
    max_log_var: float = 4.0

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


# ---------------------------------------------------------------------------
# Torch model
# ---------------------------------------------------------------------------

def _build_torch_model(hp: TransformerUTRHyperparams) -> Any:
    """Build the torch Transformer model with Gaussian head."""
    import torch
    import torch.nn as nn

    class TransformerUTR(nn.Module):
        def __init__(self, hp: TransformerUTRHyperparams):
            super().__init__()
            self.hp = hp
            vocab_size = len(hp.alphabet) + 1  # +1 for padding token (idx 0 = pad)
            self.pad_idx = 0
            self.vocab_size = vocab_size

            self.token_emb = nn.Embedding(vocab_size, hp.d_model, padding_idx=self.pad_idx)
            self.pos_emb = nn.Embedding(hp.max_len, hp.d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hp.d_model, nhead=hp.n_heads,
                dim_feedforward=hp.dim_ff, dropout=hp.dropout,
                batch_first=True, activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=hp.n_layers)
            self.dropout = nn.Dropout(hp.dropout)
            # Gaussian head: outputs (mu, log_var)
            self.head = nn.Linear(hp.d_model, 2)
            # Init head small to start near N(0, 1)
            nn.init.zeros_(self.head.bias)
            nn.init.normal_(self.head.weight, std=1e-3)

        def forward(self, tokens: Any, mask: Any) -> Any:
            """tokens: (N, L) long, mask: (N, L) bool (True = valid, False = pad)
            Returns: (mu (N,), log_var (N,))
            """
            import torch
            N, L = tokens.shape
            positions = torch.arange(L, device=tokens.device).unsqueeze(0).expand(N, L)
            x = self.token_emb(tokens) + self.pos_emb(positions)
            # TransformerEncoder with src_key_padding_mask: True = ignore
            pad_mask = ~mask  # (N, L) True = pad (to be ignored)
            x = self.encoder(x, src_key_padding_mask=pad_mask)
            x = self.dropout(x)
            # Mean-pool over valid tokens
            valid_counts = mask.float().sum(dim=1, keepdim=True).clamp(min=1.0)
            pooled = (x * mask.unsqueeze(-1).float()).sum(dim=1) / valid_counts
            out = self.head(pooled)  # (N, 2)
            mu = out[:, 0]
            log_var = out[:, 1].clamp(self.hp.min_log_var, self.hp.max_log_var)
            return mu, log_var

    return TransformerUTR(hp)


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------

class TransformerUTRPredictor(PredictorBase):
    """Transformer-UTR predictor (Architecture B)."""

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
        hp_dict = self.hyperparams or {}
        self.hp = TransformerUTRHyperparams(**{k: v for k, v in hp_dict.items()
                                                if k in TransformerUTRHyperparams.__dataclass_fields__})
        if "device" in hp_dict:
            self.hp.device = hp_dict["device"]
        self.device = self.hp.device
        self._y_mean: float = float(hp_dict.get("_y_mean", 0.0))
        self._y_std: float = float(hp_dict.get("_y_std", 1.0))
        self._model: Any = None

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

    def _encode(self, sequences: Sequence[str]) -> Tuple[Any, Any]:
        """Encode sequences to (tokens, mask) tensors.

        tokens: (N, L) long, 0 = pad, 1..A = alphabet
        mask: (N, L) bool, True = valid
        """
        import torch
        L = self.hp.max_len
        char_to_idx = {c: i + 1 for i, c in enumerate(self.hp.alphabet)}  # 0 = pad
        tokens = np.zeros((len(sequences), L), dtype=np.int64)
        mask = np.zeros((len(sequences), L), dtype=bool)
        for i, seq in enumerate(sequences):
            for j, ch in enumerate(seq[:L]):
                if ch in char_to_idx:
                    tokens[i, j] = char_to_idx[ch]
                    mask[i, j] = True
        return torch.from_numpy(tokens), torch.from_numpy(mask)

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
        """Train with NLL loss for Gaussian head + cosine LR schedule."""
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        self._set_seed()
        if self._model is None:
            self._build_model()

        y_train_arr = np.asarray(y_train, dtype=np.float32)
        if self.hp.log_transform:
            yt = np.log1p(np.maximum(y_train_arr, 0))
        else:
            yt = y_train_arr
        self._y_mean = float(yt.mean())
        self._y_std = float(yt.std() + 1e-8)
        yt_norm = (yt - self._y_mean) / self._y_std

        tokens_t, mask_t = self._encode(X_train)
        y_train_t = torch.from_numpy(yt_norm)
        train_ds = TensorDataset(tokens_t, mask_t, y_train_t)
        train_loader = DataLoader(train_ds, batch_size=self.hp.batch_size,
                                   shuffle=True, num_workers=self.hp.num_workers)

        val_loader = None
        if X_val is not None and y_val is not None:
            tv_t, mv_t = self._encode(X_val)
            y_val_arr = np.asarray(y_val, dtype=np.float32)
            y_val_t = torch.from_numpy(self._transform_label(y_val_arr))
            val_ds = TensorDataset(tv_t, mv_t, y_val_t)
            val_loader = DataLoader(val_ds, batch_size=self.hp.batch_size,
                                     shuffle=False, num_workers=self.hp.num_workers)

        optimizer = torch.optim.AdamW(self._model.parameters(),
                                       lr=self.hp.learning_rate,
                                       weight_decay=self.hp.weight_decay)

        # Compute total steps for cosine schedule
        steps_per_epoch = max(len(train_loader), 1)
        total_steps = steps_per_epoch * self.hp.n_epochs
        warmup = self.hp.warmup_steps

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return step / max(warmup, 1)
            progress = (step - warmup) / max(total_steps - warmup, 1)
            return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        def gaussian_nll(mu: Any, log_var: Any, y: Any) -> Any:
            return 0.5 * (math.log(2 * math.pi) + log_var
                          + (y - mu) ** 2 / torch.exp(log_var) + 1e-8).mean()

        history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "val_pearson": []}
        best_val_loss = float("inf")
        best_state = None
        epochs_since_improve = 0
        global_step = 0

        for epoch in range(self.hp.n_epochs):
            self._model.train()
            epoch_loss = 0.0
            n_batches = 0
            for tokens_b, mask_b, yb in train_loader:
                tokens_b = tokens_b.to(self.device)
                mask_b = mask_b.to(self.device)
                yb = yb.to(self.device)
                optimizer.zero_grad()
                mu, log_var = self._model(tokens_b, mask_b)
                loss = gaussian_nll(mu, log_var, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                global_step += 1
                epoch_loss += float(loss.item())
                n_batches += 1
            train_loss = epoch_loss / max(n_batches, 1)
            history["train_loss"].append(train_loss)

            if val_loader is not None:
                self._model.eval()
                val_loss_sum = 0.0
                preds_list, targets_list = [], []
                with torch.no_grad():
                    for tokens_b, mask_b, yb in val_loader:
                        tokens_b = tokens_b.to(self.device)
                        mask_b = mask_b.to(self.device)
                        yb = yb.to(self.device)
                        mu, log_var = self._model(tokens_b, mask_b)
                        loss = gaussian_nll(mu, log_var, yb)
                        val_loss_sum += float(loss.item())
                        preds_list.append(mu.cpu().numpy())
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

        if best_state is not None:
            self._model.load_state_dict(best_state)

        self.hyperparams["_y_mean"] = self._y_mean
        self.hyperparams["_y_std"] = self._y_std

        self._fitted = True
        return history

    def predict(self, X: Sequence[str]) -> np.ndarray:
        """Mean prediction (mu)."""
        import torch
        if not self._fitted or self._model is None:
            raise RuntimeError("Predictor not fitted")
        self._model.eval()
        tokens_t, mask_t = self._encode(X)
        tokens_t = tokens_t.to(self.device)
        mask_t = mask_t.to(self.device)
        bs = max(self.hp.batch_size, 1)
        preds = []
        with torch.no_grad():
            for i in range(0, len(tokens_t), bs):
                mu, _ = self._model(tokens_t[i:i+bs], mask_t[i:i+bs])
                preds.append(mu.cpu().numpy())
        y_pred = np.concatenate(preds) if preds else np.array([])
        return self._inverse_transform(y_pred)

    def predict_with_uncertainty(
        self, X: Sequence[str], n_mc_samples: int = 30,
    ) -> PredictionResult:
        """Predict mu + sigma from Gaussian head.

        For Transformer-UTR, uncertainty is inherent via the Gaussian head:
        sigma = exp(0.5 * log_var). No MC sampling needed.
        """
        import torch
        if not self._fitted or self._model is None:
            raise RuntimeError("Predictor not fitted")
        self._model.eval()
        tokens_t, mask_t = self._encode(X)
        tokens_t = tokens_t.to(self.device)
        mask_t = mask_t.to(self.device)
        bs = max(self.hp.batch_size, 1)
        preds_mu, preds_sigma = [], []
        with torch.no_grad():
            for i in range(0, len(tokens_t), bs):
                mu, log_var = self._model(tokens_t[i:i+bs], mask_t[i:i+bs])
                sigma = torch.exp(0.5 * log_var)
                preds_mu.append(mu.cpu().numpy())
                preds_sigma.append(sigma.cpu().numpy())
        mu_arr = np.concatenate(preds_mu) if preds_mu else np.array([])
        sigma_arr = np.concatenate(preds_sigma) if preds_sigma else np.array([])

        # Inverse transform: y = expm1(mu * y_std + y_mean), so
        # var(y) = (expm1)'(mu_norm * y_std + y_mean)^2 * sigma_norm^2 * y_std^2
        # For log_transform: dy/dmu_norm = y_std * exp(mu_norm * y_std + y_mean)
        mu_orig = self._inverse_transform(mu_arr)
        if self.hp.log_transform:
            # y = expm1(z), z = mu_norm * y_std + y_mean
            z = mu_arr * self._y_std + self._y_mean
            dydz = np.exp(z)  # derivative of expm1
            std_orig = np.abs(dydz) * sigma_arr * self._y_std
        else:
            std_orig = sigma_arr * self._y_std
        return PredictionResult(
            mean=mu_orig.astype(np.float32),
            std=std_orig.astype(np.float32),
            abstain_mask=None,
            metadata={"method": "gaussian_head", "n_mc_samples": 1},
        )


__all__ = ["TransformerUTRPredictor", "TransformerUTRHyperparams"]
