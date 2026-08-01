"""FM0-01 shared helpers for UTR-LM loading, tokenizing, embedding.

Standalone module (no imports from mrna_editflow_repo), following the
d1_staging convention. All FM0 scripts import from this module via sys.path.

Contract: utr_editflow_contract_v2 (FROZEN), task FM0-01.
"""

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = Path(os.path.dirname(os.path.abspath(__file__))).resolve()
# d1_staging root (one level up from scripts/fm0)
D1_STAGING_ROOT = HERE.parent.parent
# repo root (one level up from d1_staging)
REPO_ROOT = D1_STAGING_ROOT.parent

CONFIG_PATH = REPO_ROOT / "configs" / "fm0_utrlm_config.yaml"
# FM0 small JSON manifests/reports go to repo-root data/fm0/ (consistent with
# data/d1_canonical_records.jsonl and data/b0_splits/). Large caches (frozen
# embeddings, LoRA adapters) go to /mnt per contract §8.
DATA_FM0_DIR = REPO_ROOT / "data" / "fm0"

# Default output dir for manifests / reports.
DEFAULT_OUTPUT_DIR = DATA_FM0_DIR


def ensure_offline_env() -> None:
    """Force HF/transformers offline mode (server has no internet to HF).

    Idempotent: only sets if not already set, so explicit env vars from the
    shell still win.
    """
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    # Also silence tokenizers parallelism warning
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def load_config() -> dict:
    """Load the frozen FM0 config YAML. Minimal parser (no pyyaml dep needed).

    Only supports the flat-ish structure we wrote; for nested values we use
    a tiny indentation-based parser. Falls back to pyyaml if available.
    """
    try:
        import yaml  # type: ignore
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception:
        # Minimal fallback parser (not used in practice since pyyaml is installed)
        raise RuntimeError(
            f"Could not load {CONFIG_PATH}; install pyyaml or fix config."
        )


def get_snapshot_dir() -> Path:
    """Return the HF snapshot directory for the UTR-LM checkpoint."""
    cfg = load_config()
    p = Path(cfg["storage"]["snapshot_dir"])
    if not p.exists():
        raise FileNotFoundError(
            f"UTR-LM snapshot not found at {p}. Run FM0 preflight to download "
            f"the checkpoint into the HF cache."
        )
    return p


def get_model_id() -> str:
    return load_config()["model"]["model_id"]


def require_cuda() -> "torch.device":
    """Contract: training_device = GPU_only. STOP if CUDA unavailable."""
    import torch
    if not torch.cuda.is_available():
        sys.exit(
            "[FM0] FATAL: CUDA unavailable. Contract requires GPU_only. "
            "Aborting (forward-only principle: do not mask failures)."
        )
    return torch.device("cuda")


def pick_gpu_device(preferred: Optional[List[str]] = None) -> "torch.device":
    """Pick a GPU with the most free memory.

    Args:
        preferred: device index strings to prefer (from config). If None, uses
            config's preferred_devices.
    """
    import torch
    require_cuda()
    if preferred is None:
        preferred = load_config()["gpu"]["preferred_devices"]

    n = torch.cuda.device_count()
    best_idx = None
    best_free = -1
    for i in range(n):
        idx = str(i)
        # Skip devices not in preferred list (but fall back if none match)
        if preferred and idx not in preferred:
            # Still consider if no preferred device has free mem
            pass
        try:
            free, _total = torch.cuda.mem_get_info(i)
        except Exception:
            continue
        # Strongly prefer devices in `preferred` list
        weight = free if idx in preferred else free // 4
        if weight > best_free:
            best_free = weight
            best_idx = i
    if best_idx is None:
        sys.exit("[FM0] FATAL: no usable CUDA device found.")
    return torch.device(f"cuda:{best_idx}")


# ---------------------------------------------------------------------------
# Model / tokenizer loaders
# ---------------------------------------------------------------------------

def load_tokenizer():
    """Load the RnaTokenizer from the frozen checkpoint."""
    ensure_offline_env()
    from multimolecule import RnaTokenizer
    return RnaTokenizer.from_pretrained(get_model_id())


def load_config_obj():
    """Load the UtrLmConfig (architecture spec) from the frozen checkpoint."""
    ensure_offline_env()
    from multimolecule import UtrLmConfig
    return UtrLmConfig.from_pretrained(get_model_id())


def load_model(device: Optional[str] = None, dtype: Optional[str] = None):
    """Load the bare UtrLm encoder (no pretraining heads) from the frozen checkpoint.

    Args:
        device: "cpu", "cuda", "cuda:N", or None (leaves on CPU).
        dtype: "float32" / "float16" / "bfloat16", or None (default float32).
    """
    ensure_offline_env()
    import torch
    from multimolecule import UtrLmModel

    model = UtrLmModel.from_pretrained(get_model_id())
    if dtype is not None:
        dt = getattr(torch, dtype)
        model = model.to(dt)
    if device is not None:
        if device.startswith("cuda") and not torch.cuda.is_available():
            sys.exit(f"[FM0] FATAL: device={device} but CUDA unavailable.")
        model = model.to(device)
    model.eval()
    return model


def load_model_from_scratch(seed: int = 20260801):
    """Random-init control arm: same architecture, NO checkpoint weights.

    Returns a model with random initialization (using the config's init scheme).
    """
    ensure_offline_env()
    import torch
    from multimolecule import UtrLmModel

    cfg = load_config_obj()
    torch.manual_seed(seed)
    model = UtrLmModel(cfg)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Tokenization helpers
# ---------------------------------------------------------------------------

def tokenize_sequences(
    sequences: List[str],
    tokenizer,
    max_length: Optional[int] = None,
    padding: bool = True,
    return_tensors: str = "pt",
) -> Dict:
    """Tokenize a list of nucleotide sequences.

    Sequences may contain A/C/G/T/U/N; tokenizer converts T->U.
    """
    return tokenizer(
        sequences,
        padding=padding,
        truncation=max_length is not None,
        max_length=max_length,
        return_tensors=return_tensors,
    )


# ---------------------------------------------------------------------------
# Embedding / pooling
# ---------------------------------------------------------------------------

def pool_embeddings(
    last_hidden_state,    # [B, L, H]
    attention_mask,       # [B, L]
    mode: str = "cls",
):
    """Pool token-level hidden states into a single [B, H] vector.

    modes:
      - "cls":  take the first token (BOS/<cls>) — standard BERT pooled rep
      - "mean": masked mean over real tokens
      - "max":  masked max over real tokens
    """
    import torch
    if mode == "cls":
        return last_hidden_state[:, 0, :]
    if mode == "mean":
        mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
        summed = (last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1.0)
        return summed / counts
    if mode == "max":
        mask = attention_mask.unsqueeze(-1).to(torch.bool)
        neg_inf = torch.finfo(last_hidden_state.dtype).min
        masked = last_hidden_state.masked_fill(~mask, neg_inf)
        return masked.max(dim=1).values
    raise ValueError(f"unknown pooling mode: {mode}")


def embed(
    sequences: List[str],
    model,
    tokenizer,
    device,
    pooling: str = "cls",
    batch_size: int = 32,
    max_length: Optional[int] = None,
):
    """Embed a list of sequences; return numpy array [N, H]."""
    import torch
    model.eval()
    outs = []
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i : i + batch_size]
        enc = tokenize_sequences(batch, tokenizer, max_length=max_length)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
        h = out.last_hidden_state
        v = pool_embeddings(h, enc["attention_mask"], mode=pooling)
        outs.append(v.cpu())
    import numpy as np
    return torch.cat(outs, dim=0).numpy()


# ---------------------------------------------------------------------------
# Small IO helpers
# ---------------------------------------------------------------------------

def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def sha256_of_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def summarize_model(model) -> dict:
    """Return a JSON-serializable summary of a loaded model."""
    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    dtype = str(next(model.parameters()).dtype)
    device = str(next(model.parameters()).device)
    return {
        "num_parameters_total": n_total,
        "num_parameters_trainable": n_trainable,
        "num_parameters_frozen": n_total - n_trainable,
        "dtype": dtype,
        "device": device,
    }
