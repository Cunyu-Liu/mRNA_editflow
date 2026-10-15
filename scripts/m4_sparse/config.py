"""M4 SparseEditFormer configuration.

Uses the same nucleotide conventions as B0-X (NUC_ORDER="ACGU") and an
edit-centred MAX_SEQ_LEN=100 window so edits beyond position 99 remain visible.
It uses the same S4 (leave-one-study-out) primary macro structure.  Only ACTIVE A1
benchmarks are trained (5U-A1 + 3U-A1).  EditBench-5U-A2-Dense is DORMANT (no
qualified A2 dense asset) and is intentionally NOT configured: training is
A1-NATURAL, adapted without an A2 dense pretraining stage.  GSE246381 is SEALED
and excluded.
"""
from __future__ import annotations

from types import SimpleNamespace

# ---- encoding conventions (mirror scripts/b0x/config.py) ----
NUC_ORDER = "ACGU"
NUC_TO_IDX = {ch: i for i, ch in enumerate(NUC_ORDER)}
MAX_SEQ_LEN = 100
EDIT_FEAT_DIM = 12
CHECKPOINT_SCHEMA_VERSION = "m4_sparse_r3_p0_v1"
INPUT_SCHEMA_VERSION = "shared_edit_centered_window_with_padding_masks_v1"
RANK_PAIR_DEFINITION = (
    "within_source_study_endpoint_context_distinct_delta_all_pairs_v1"
)
RANK_SCORE_DEFINITION = "predicted_delta_mean_r3_p0_v1"
SIGN_SCORE_DEFINITION = "predicted_delta_mean_over_sigma_r3_p0_v1"
SIGN_DECISION_RULE = "predicted_delta_mean_ge_0_r3_p0_v1"
SIGN_CALIBRATION_METHOD = (
    "positive_temperature_probability_only_grouped_dev_r3_p0_v1"
)
CHECKPOINT_SELECTION_POLICY = (
    "study_macro_lexicographic_sign_then_spearman_then_nll_r3_p0_v1"
)
DEV_SIGN_ACCURACY_GATE = 0.60

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
HUBER_DELTA = 1.0
CONTEXT_DROPOUT = 0.10  # train the reserved unknown study/endpoint embeddings

# ---- loss weights ----
LAMBDA_MEAN = 1.0
LAMBDA_VAR = 0.5
LAMBDA_SIGN = 0.5
LAMBDA_CONSISTENCY = 0.25
LAMBDA_RANK = 0.5

# ---- target / honesty ----
# TARGET selects the regression target.
#   "delta"            : predict delta directly from sequence (pure sequence).
#   "candidate_value"  : predict candidate_value; at test time
#                        delta = cand_pred - MEASURED source_value (honest
#                        anchor setting, same inputs as the abs_candidate
#                        reference ceiling).  Disclosed in the report.
TARGET = "delta"
ANCHOR_AT_TEST = True  # if TARGET=="candidate_value", subtract measured source

# ---- gpu selection with fallback ----
CUDA_DEVICES = ["cuda:1", "cuda:2", "cuda:3"]  # avoid GPU 4 (owned)
FORBIDDEN_DEVICES = {"4", "cuda:4"}

# ---- data / output (relative to repo root) ----
DATASET = "artifacts/b0x/effect_dataset.jsonl"
OUT_DIR = "artifacts/m4_sparse"

# cap for smoke-testing (None = use all delta-defined records)
LIMIT = None
MAX_TRAIN_CAP = None  # optional cap on per-fold train records (None = all)


def get_config() -> SimpleNamespace:
    cfg = SimpleNamespace()
    for k in ("NUC_ORDER", "NUC_TO_IDX", "MAX_SEQ_LEN", "EDIT_FEAT_DIM",
              "CHECKPOINT_SCHEMA_VERSION", "INPUT_SCHEMA_VERSION",
              "RANK_PAIR_DEFINITION", "RANK_SCORE_DEFINITION",
              "SIGN_SCORE_DEFINITION", "SIGN_DECISION_RULE",
              "SIGN_CALIBRATION_METHOD", "CHECKPOINT_SELECTION_POLICY",
              "DEV_SIGN_ACCURACY_GATE",
              "PRIMARY_SPLIT", "SECONDARY_SPLIT", "SEED", "HIDDEN_DIM", "NHEAD",
              "N_LAYERS", "DIM_FF", "CONV_KS", "DROP", "BATCH_SIZE", "LR",
              "EPOCHS", "DEV_FRAC", "WEIGHT_DECAY", "MARGIN", "HUBER_DELTA",
              "CONTEXT_DROPOUT", "LAMBDA_MEAN",
              "LAMBDA_VAR", "LAMBDA_SIGN", "LAMBDA_CONSISTENCY", "LAMBDA_RANK",
              "TARGET", "ANCHOR_AT_TEST", "CUDA_DEVICES", "FORBIDDEN_DEVICES", "DATASET", "OUT_DIR",
              "LIMIT", "MAX_TRAIN_CAP"):
        setattr(cfg, k, globals()[k])
    # filled at runtime by run.py from the data
    cfg.N_STUDIES = 0
    cfg.N_ENDPOINTS = 0
    cfg.N_BENCHMARKS = 0
    return cfg
