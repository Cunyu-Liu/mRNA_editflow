"""Route A retraining configuration for M4 SparseEditFormer.

This is an ISOLATED development experiment (no publication claim).  It is
based on the frozen migration worktree baseline (the E0-X official source:
candval TARGET="candidate_value", sign_accuracy=0.510, macro_delta_spearman
=0.297, top10pct_enrichment=9.92) so the gate measurement is strictly
comparable.

New levers vs the baseline (baseline candval already used
TARGET="candidate_value", ANCHOR_AT_TEST=True, LAMBDA_RANK=0.5 active):

  * SOURCE_ANCHOR_INJECT : explicitly feed the measured source_value into the
    model context head.  The baseline only used source_value as a test-time
    anchor (delta = mean - source); it never let the model learn from it.
    This is the primary "stronger source anchoring" lever of Route A.
  * EFFECTIVE_LAMBDA_RANK : kept explicit (>0) so the rank gradient is
    unambiguously active and auditable.

Governance: no sealed GSE246381 data is consumed; the frozen M4 baseline
artifacts/checkpoints are never modified; output goes to a new isolated
directory (artifacts/m4_routea_v1).  CUDA is required for training (no CPU
fallback); GPU 4 is forbidden.
"""
from __future__ import annotations

from types import SimpleNamespace

# ---- encoding conventions (mirror the frozen migration baseline) ----
NUC_ORDER = "ACGU"
NUC_TO_IDX = {ch: i for i, ch in enumerate(NUC_ORDER)}
MAX_SEQ_LEN = 100
EDIT_FEAT_DIM = 12

# ---- split / seed ----
PRIMARY_SPLIT = "S4"
SECONDARY_SPLIT = "S1"
SEED = 42

# ---- model ----
HIDDEN_DIM = 64
NHEAD = 4
N_LAYERS = 2
DIM_FF = 128
CONV_KS = 5
DROP = 0.1

# ---- training ----
BATCH_SIZE = 256
LR = 1e-3
EPOCHS = 4
DEV_FRAC = 0.15
WEIGHT_DECAY = 1e-5
MARGIN = 0.5  # pairwise ranking margin

# ---- loss weights (identical to the frozen baseline) ----
LAMBDA_MEAN = 1.0
LAMBDA_VAR = 0.5
LAMBDA_SIGN = 0.5
LAMBDA_CONSISTENCY = 0.25
LAMBDA_RANK = 0.5
# Route-A audit: rank gradient explicitly active (the frozen baseline already
# had LAMBDA_RANK=0.5; we keep it explicit and non-zero here).
EFFECTIVE_LAMBDA_RANK = 0.5

# ---- target / honesty (matches E0-X candval) ----
TARGET = "candidate_value"
ANCHOR_AT_TEST = True  # delta = cand_pred - MEASURED source_value at test time

# ---- Route-A primary lever ----
# If True, the measured source_value is injected into the model context vector
# so the model can learn a source-relative representation.
SOURCE_ANCHOR_INJECT = True

# ---- gpu selection ----
# Prefer the free GPU 5; GPU 4 is forbidden (owner).  No CPU fallback: run.py
# records not_run if CUDA is unavailable.
CUDA_DEVICES = ["cuda:5", "cuda:1", "cuda:2", "cuda:3"]
FORBIDDEN_DEVICES = {"4", "cuda:4"}

# ---- data / output (relative to repo root) ----
DATASET = "artifacts/b0x/effect_dataset.jsonl"
OUT_DIR = "artifacts/m4_routea_v1"

# cap for smoke-testing (None = use all delta-defined records)
LIMIT = None
MAX_TRAIN_CAP = None


def get_config() -> SimpleNamespace:
    cfg = SimpleNamespace()
    for k in ("NUC_ORDER", "NUC_TO_IDX", "MAX_SEQ_LEN", "EDIT_FEAT_DIM",
              "PRIMARY_SPLIT", "SECONDARY_SPLIT", "SEED", "HIDDEN_DIM", "NHEAD",
              "N_LAYERS", "DIM_FF", "CONV_KS", "DROP", "BATCH_SIZE", "LR",
              "EPOCHS", "DEV_FRAC", "WEIGHT_DECAY", "MARGIN", "LAMBDA_MEAN",
              "LAMBDA_VAR", "LAMBDA_SIGN", "LAMBDA_CONSISTENCY", "LAMBDA_RANK",
              "EFFECTIVE_LAMBDA_RANK", "TARGET", "ANCHOR_AT_TEST",
              "SOURCE_ANCHOR_INJECT", "CUDA_DEVICES", "FORBIDDEN_DEVICES",
              "DATASET", "OUT_DIR", "LIMIT", "MAX_TRAIN_CAP"):
        setattr(cfg, k, globals()[k])
    cfg.N_STUDIES = 0
    cfg.N_ENDPOINTS = 0
    cfg.N_BENCHMARKS = 0
    return cfg
