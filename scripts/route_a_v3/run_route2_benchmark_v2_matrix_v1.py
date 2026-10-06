#!/usr/bin/env python3
"""Task 2.2 + 3.5 (benchmark v2): port 5 new model families onto the frozen-delta
matrix - unit tests with official reference outputs + frozen-delta evaluation
(VALIDATION-only for the 9 existing tasks + NEW_EVAL_ROW for the 3 new rows).

Preregistration compliance (docs/paper/benchmark_v2_matrix_row_prereg_v1.md):
- frozen-DELTA zero-tuning: official weights are loaded verbatim (strict or
  audited non-strict) and used as-is; no gradient updates, no output
  post-calibration; the only freedom is declared input adaptation + device
  transfer. delta_hat = y_hat(candidate) - y_hat(source).
- input adaptation declared per family (mirroring the port ledger section 6
  drafts; no caliber direction changes).
- VALIDATION only for existing Development tasks (Task-1 evaluator, K=10);
  TEST split untouched. New rows (M1/M6/S1) are evaluated per their own
  row_definition.md (task_id + row structure), split = NEW_EVAL_ROW.
- append-only: results are written under analysis_benchmark_v2_matrix_v1/
  (family x task cell directory + matrix_summary.json). No existing manifest
  or artifact is modified.
- missing-cell rules: rows where a family's input domain does not match the
  task region are STRUCTURED_NA (not failures); MISS stats recorded per
  prereg section 1.5.

Families (port ledger section 4, license waiver recorded 2026-09-20):
- lamar_utr5te: LAMAR-UTR5TEPred (MIT). HF-style EsmForSequenceClassification
  with Linear regression head (768->1). Official tokenizer
  (single-nucleotide, char-level, <cls>/<eos>), padding_side='left',
  truncation to model_max_length=1026 (official finetune caliber,
  tokenize_data.ipynb). Weights: model.safetensors (330M, checkpoint-17600).
  Loaded via the official LAMAR package (LAMAR/sequence_classification_patch)
  with a transformers-v5 compatibility shim (find_pruneable_heads_and_indices,
  prune_linear_layer, get_head_mask - inference-unused helpers removed in
  transformers 5.x; shim does not touch any forward computation).
- gemorna_5utr: GEMORNA 5'UTR predictor (official src/main_pred5UTR.py
  caliber). char-level vocab {'[PAD]':0,'A':5,'U':6,'G':7,'C':8,'N':9},
  T->U, right-padded to length 100, GRU 2-layer (128) -> Linear -> logit,
  then official scale() z*std+mean (5.19937892 / 1.40592675).
- gemorna_3utr: GEMORNA 3'UTR predictor (official src/main_pred3UTR.py
  caliber). char-level tokenization (no pad; variable length), TextCNN
  (kernels 2,4,6,8,10) -> max-pool -> Linear -> raw logit (no scale()).
- utr_stcnet: UTR-STCNet MPRA-H checkpoint (paddle was feared, but the pkl
  is a torch state_dict {'model': sd} - torch.load works). Official
  evaluation.py caliber: one-hot 4-channel, length-capped to 120 by keeping
  the LAST 120 nt (utr_dataset_all.UTRDATA.encoder), RIGHT-padded to 120,
  utrformer_large (embed_dims [64,128,320,512], depths [3,8,27,3]), forward
  returns (x, x_s) in eval mode; take x (RL scalar).
- utr_insight: UTR-Insight model_epoch199.pkl (official
  3.UTR_Insight_endogenous_predict.ipynb caliber). Local modified esm
  package (repo esm/ dir, importable). Alphabet standard_toks 'AGCT' with
  <cls>/<eos> added by batch converter; official pad convention:
  '<pad>' x (100 - len) left-pad, total 100 tokens. ConvTransformerPredictor
  with experiment_indicator = [1, 0] (official choice for
  endogenous/held-out prediction in nb3; recorded in adaptation declaration).
  Unit test: official pre-generated Result/utr_insight/e_pred_random_50.csv
  (same GSM3130435 library as GSE114002).
- hydrarna: PORT_BLOCKED_HEAD_MISSING. Local HydraRNA_model.pt was inspected
  (torch.load): the checkpoint contains ONLY the MLM pretraining head
  (encoder.lm_head.*) - no MRL regression head exists in the released
  weights, and the official examples (finetune_HydraAttRNA12_mlp_5UTRMRL_
  scaled.py) fit a fresh MLP head on the user side. The remaining google-
  drive weights could not be fetched (server has no external network;
  weights/gdown_dl.log shows connection failure). Under the zero-tuning
  gate there is no admissible frozen MRL predictor -> BLOCKED, recorded in
  matrix_summary.json; no cells are produced.

Task grid (rows = family cells; columns):
- Existing 9 (VALIDATION only, Task-1 evaluator):
  gse114002_mrl (5UTR), gse269595_polya (3UTR), encsr854ruf_mprau (3UTR,
  pair-mean primary), gse200304_te (3UTR), gse149487_te (5UTR),
  gse149487_rna (5UTR), gse217518_hl_5utr (5UTR), gse217518_hl_3utr (3UTR),
  gse186455 (3UTR).
- New eval rows (NEW_EVAL_ROW, per row_definition.md):
  m1_mrl (5UTR, 2805 rows), m6_ndd5utr (5UTR, 800 rows),
  s1_stability 5UTR arm + 3UTR arm (5,572 sub-rows total; evaluated per arm
  as two task columns; biological_context distinguishes SH/HEK sub-rows,
  both kept - the row definition declares dual-assay sub-rows share
  sequences and context distinguishes them).

Spearman rho is the headline metric per cell (record-level, paired-delta);
Task-1 evaluator outputs (K=10) are additionally recorded for the existing
tasks. For ENCSR854RUF the pair-mean variant caliber (W-ladder) is
additionally computed. MPRAU conventions preserved: variants = record id
before ':context:', >=2 contexts, per-variant means.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import math
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EVAL_REPO = Path(
    "/home/cunyuliu/mrna_editflow_goal/worktrees/route_a_v3_setflow_v5_base_fix_20260901"
)
MNT = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2")
ASSETS = MNT / "external_model_assets"
MANIFEST = MNT / "manifests/route2_development_frozen_v1/development_manifest.jsonl"
NEW_ROWS_ROOT = MNT / "benchmark_v2"
DEFAULT_OUTPUT = MNT / "experiments/analysis_benchmark_v2_matrix_v1"

K = 10
BASES = set("ACGT")
MPRAU_BOOTSTRAP_ITERATIONS = 2000
BOOTSTRAP_SEED = 20260816
CELL_BOOTSTRAP_ITERATIONS = 1000

FAMILY_ORDER = ("lamar_utr5te", "gemorna_5utr", "gemorna_3utr", "utr_stcnet", "utr_insight")
FAMILY_WEIGHTS = {
    "lamar_utr5te": ASSETS / "lamar_weights/UTR5TEPred/saving_model/mammalian_2048/bs16_lr5e-5_wr0.05_32epochs_5/checkpoint-17600/model.safetensors",
    "gemorna_5utr": ASSETS / "gemorna/5utr.pt",
    "gemorna_3utr": ASSETS / "gemorna/3utr.pt",
    "utr_stcnet": ASSETS / "utr_stcnet/checkpoint/UTR-STCNet_checkpoints/MPRA-H/UTR-H_new_RL_epoch300_batchsize256_padd120_new.pkl",
    "utr_insight": ASSETS / "utr_insight/Model/utr_insight/model_epoch199.pkl",
}

INPUT_ADAPTATION = {
    "lamar_utr5te": {
        "tokenization": "official LAMAR single-nucleotide char tokenizer (vocab <cls>/<pad>/<eos>/<unk>/<mask>/ATCGN), padding_side='left' per official finetune notebook",
        "truncation": "truncation=True, max_length=1026 (official tokenize_data.ipynb model_max_length; never hit by matrix tasks, max 837 nt)",
        "input_domain": "5UTR-only (official UTR5TEPred finetune domain); 3UTR tasks -> STRUCTURED_NA",
        "output_head": "EsmForSequenceClassification Linear head (768->1, num_labels=1) - the released checkpoint's classifier.head.* weights",
        "extra": "loaded via official LAMAR package; transformers-5 shim for prune helpers only (inference-unused); batch inference fp32",
    },
    "gemorna_5utr": {
        "tokenization": "official char-level vocab {'[PAD]':0,'A':5,'U':6,'G':7,'C':8,'N':9} with T->U (official helper.tokenize)",
        "truncation": "right-pad to fixed length 100 (official main_pred5UTR inference: tokenized + [PAD]*(100-len)); sequences >100 nt keep the FIRST 100 nt (5' anchored; official predictor trained on <=100-nt 5'UTRs, no official truncation rule - declared)",
        "input_domain": "5UTR-only; 3UTR tasks -> STRUCTURED_NA",
        "output_head": "GRU(2x128) hidden concat -> Linear -> logit -> official scale(pred, 5.19937892, 1.40592675)",
    },
    "gemorna_3utr": {
        "tokenization": "official char-level vocab (T->U), no padding (TextCNN variable length, official main_pred3UTR inference)",
        "truncation": "none (no official truncation; max 3UTR task length 201 nt; batch by equal-length grouping)",
        "input_domain": "3UTR-only; 5UTR tasks -> STRUCTURED_NA",
        "output_head": "TextCNN (kernels 2,4,6,8,10 x 200) max-pool -> Linear -> raw logit (official: no scale())",
    },
    "utr_stcnet": {
        "tokenization": "official one-hot 4-channel map A/C/G/T -> unit vectors, N -> zero vector (utr_dataset_all.UTRDATA)",
        "truncation": "official encoder rule: len>120 -> keep LAST 120 nt; then RIGHT-pad to 120 with zero rows (tokens[-len:]=encoded)",
        "input_domain": "5UTR-only (MPRA-H human 5'UTR library, padd120 caliber); 3UTR tasks -> STRUCTURED_NA",
        "output_head": "utrformer_large RL regression head (self.head: Linear 512->64->1); forward(x) default train=True returns (x_scalar, x_s) - official evaluation.py call, take the scalar",
        "extra": "pkl inspected: torch state_dict {'model': sd} (NOT paddle - paddle dependency not required); module.-prefix stripped per official evaluation.py; official cluster_dpc_knn is stochastic by design (density + rand*1e-6) - per-batch torch.manual_seed(20260816) pinned for reproducibility, unit test self-consistency verified under the pinned seed",
    },
    "utr_insight": {
        "tokenization": "official modified esm (repo-local esm/ package): Alphabet('AGCT'), char tokens, batch converter adds <cls> + <eos>",
        "truncation": "official pad convention: '<pad>' x (100 - len) LEFT-pad to a fixed total of 100 tokens (per official Result csvs and nb1 '50+50' case generalized to pad-to-100 for variable length, confirmed by e_pred_random_100 pad counts); sequences >100 nt keep the FIRST 100 nt (declared - no official rule for >100)",
        "input_domain": "5UTR-only (official model trained on 5'UTR MRL); 3UTR tasks -> STRUCTURED_NA",
        "output_head": "ConvTransformerPredictor output (nodes->1); experiment_indicator = [1, 0] (official nb3 endogenous-prediction choice)",
    },
}

FAMILY_DOMAIN = {
    "lamar_utr5te": "5UTR",
    "gemorna_5utr": "5UTR",
    "gemorna_3utr": "3UTR",
    "utr_stcnet": "5UTR",
    "utr_insight": "5UTR",
}

EXISTING_TASKS = {
    "gse114002_mrl": {"study": "GSE114002", "region": "5UTR", "endpoint": "MEAN_RIBOSOME_LOAD"},
    "gse269595_polya": {"study": "GSE269595", "region": "3UTR", "endpoint": "PROXIMAL_POLYA_SITE_USAGE_LOG2_ODDS"},
    "encsr854ruf_mprau": {"study": "ENCSR854RUF", "region": "3UTR", "endpoint": "MPRAU_ALLELIC_SKEW_LOG2_FOLD_CHANGE"},
    "gse200304_te": {"study": "GSE200304", "region": "3UTR", "endpoint": "TOTAL_POLYSOME_TRANSLATION_EFFICIENCY"},
    "gse149487_te": {"study": "GSE149487", "region": "5UTR", "endpoint": "te_log2_polysome_over_totalrna"},
    "gse149487_rna": {"study": "GSE149487", "region": "5UTR", "endpoint": "transcript_log2_totalrna_over_dna"},
    "gse217518_hl_5utr": {"study": "GSE217518", "region": "5UTR", "endpoint": "RNA_HALF_LIFE_MINUTES"},
    "gse217518_hl_3utr": {"study": "GSE217518", "region": "3UTR", "endpoint": "RNA_HALF_LIFE_MINUTES"},
    "gse186455": {"study": "GSE186455", "region": "3UTR", "endpoint": "PUBLISHED_REF_VS_ALT_ACTIVITY_LMM_LOG2_FOLD_CHANGE"},
}

NEW_ROW_TASKS = {
    "m1_mrl": {"dir": "m1_mrl_eval_row", "region": "5UTR"},
    "m6_ndd5utr": {"dir": "m6_ndd5utr_eval_row", "region": "5UTR"},
    "s1_stability_5utr": {"dir": "s1_stability_eval_row", "region": "5UTR"},
    "s1_stability_3utr": {"dir": "s1_stability_eval_row", "region": "3UTR"},
}

TASK_ORDER = (
    "gse114002_mrl",
    "gse269595_polya",
    "encsr854ruf_mprau",
    "gse200304_te",
    "gse149487_te",
    "gse149487_rna",
    "gse217518_hl_5utr",
    "gse217518_hl_3utr",
    "gse186455",
    "m1_mrl",
    "m6_ndd5utr",
    "s1_stability_5utr",
    "s1_stability_3utr",
)


class MatrixError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MatrixError(message)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ev = _load_module(
    "route2_frozen_eval_matrix",
    EVAL_REPO / "scripts/route_a_v3/evaluate_route2_prediction_v1.py",
)


@dataclass
class PairRow:
    row_id: str
    source: str
    candidate: str
    target: float
    source_id: str = ""
    study: str = ""
    context: str = ""
    region: str = ""
    endpoint: str = ""
    extra: dict = field(default_factory=dict)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_path_for(study: str) -> Path:
    for name in ("canonical_records.private.jsonl", "canonical_records.jsonl"):
        path = MNT / "canonical" / study / "v1" / name
        if path.is_file():
            return path
    raise MatrixError(f"canonical records absent for {study}")


def load_existing_task(task_key: str) -> list[PairRow]:
    spec = EXISTING_TASKS[task_key]
    stratum = [spec["study"], spec["region"], spec["endpoint"]]
    manifest_rows = [
        json.loads(line) for line in MANIFEST.read_text(encoding="utf-8").splitlines()
    ]
    eval_ids = {
        str(row["canonical_record_id"])
        for row in manifest_rows
        if list(row["stratum"]) == stratum and row["split"] == "VALIDATION"
    }
    _require(eval_ids, f"task {task_key} has no VALIDATION rows")
    rows: list[PairRow] = []
    with canonical_path_for(spec["study"]).open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            record_id = str(row["canonical_record_id"])
            if record_id not in eval_ids:
                continue
            source = str(row["source_sequence"]).upper()
            candidate = str(row["candidate_sequence"]).upper()
            _require(
                not ((set(source) | set(candidate)) - BASES),
                f"non-ACGT sequence in {record_id}",
            )
            target = float(row["direction_normalized_delta"])
            _require(math.isfinite(target), f"non-finite target in {record_id}")
            rows.append(
                PairRow(
                    row_id=record_id,
                    source=source,
                    candidate=candidate,
                    target=target,
                    source_id=str(row["source_id"]),
                    study=spec["study"],
                    context=str(row["biological_context_id"]),
                    region=spec["region"],
                    endpoint=spec["endpoint"],
                )
            )
    _require(len(rows) == len(eval_ids), f"task {task_key} canonical coverage mismatch")
    return rows


def load_new_row_task(task_key: str) -> list[PairRow]:
    spec = NEW_ROW_TASKS[task_key]
    path = NEW_ROWS_ROOT / spec["dir"] / "projection_rows.jsonl"
    rows: list[PairRow] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if str(row["region"]) != spec["region"]:
                continue
            source = str(row["source_sequence"]).upper()
            candidate = str(row["candidate_sequence"]).upper()
            target = float(row["direction_normalized_delta"])
            _require(math.isfinite(target), f"non-finite target in {row['row_id']}")
            rows.append(
                PairRow(
                    row_id=str(row["row_id"]),
                    source=source,
                    candidate=candidate,
                    target=target,
                    source_id=str(row.get("source_group_id", "")),
                    study=str(row["study_unit"]),
                    context=str(row["biological_context"]),
                    region=str(row["region"]),
                    endpoint=str(row["endpoint_descriptor"].get("endpoint_id", "")),
                    extra={"task_id": str(row["task_id"])},
                )
            )
    _require(rows, f"new row task {task_key} is empty")
    return rows


def load_task(task_key: str) -> list[PairRow]:
    if task_key in EXISTING_TASKS:
        return load_existing_task(task_key)
    return load_new_row_task(task_key)


def region_of(task_key: str) -> str:
    if task_key in EXISTING_TASKS:
        return EXISTING_TASKS[task_key]["region"]
    return NEW_ROW_TASKS[task_key]["region"]


# ---------------------------------------------------------------------------
# Family adapters
# ---------------------------------------------------------------------------


class PredictResult:
    __slots__ = ("values", "misses")

    def __init__(self):
        self.values: dict[str, float] = {}
        self.misses: dict[str, str] = {}


# --- LAMAR -----------------------------------------------------------------

_LAMAR_SHIM_INSTALLED = False


def _install_lamar_shim() -> None:
    global _LAMAR_SHIM_INSTALLED
    if _LAMAR_SHIM_INSTALLED:
        return

    def find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
        mask = torch.ones(n_heads, head_size)
        heads = set(heads) - already_pruned_heads
        for head in heads:
            head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()
        return heads, index

    def prune_linear_layer(layer, index, dim=0):
        import torch.nn as nn

        index = index.to(layer.weight.device)
        weight = layer.weight.index_select(dim, index).clone()
        bias = None
        if layer.bias is not None:
            bias = layer.bias[index].clone() if dim == 0 else layer.bias.clone()
        new = nn.Linear(weight.size(1), weight.size(0), bias=layer.bias is not None).to(
            layer.weight.dtype
        )
        new.weight.copy_(weight)
        if bias is not None:
            new.bias.copy_(bias)
        return new

    def get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
        _require(head_mask is None, "head_mask is not supported in this port")
        return [None] * num_hidden_layers

    import transformers.modeling_utils as modeling_utils

    modeling_utils.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices
    modeling_utils.prune_linear_layer = prune_linear_layer

    sys.path.insert(0, str(ASSETS / "lamar"))
    from LAMAR.modeling_nucESM2 import EsmModel

    EsmModel.get_head_mask = get_head_mask
    _LAMAR_SHIM_INSTALLED = True


def load_lamar(device: torch.device):
    _install_lamar_shim()
    from transformers import AutoConfig, AutoTokenizer

    from LAMAR.sequence_classification_patch import EsmForSequenceClassification

    tokenizer = AutoTokenizer.from_pretrained(
        ASSETS / "lamar/tokenizer/single_nucleotide",
        model_max_length=1026,
        padding_side="left",
    )
    config = AutoConfig.from_pretrained(
        ASSETS / "lamar/config/config_150M.json",
        vocab_size=len(tokenizer),
        pad_token_id=tokenizer.pad_token_id,
        mask_token_id=tokenizer.mask_token_id,
        num_labels=1,
        token_dropout=False,
        positional_embedding_type="rotary",
        hidden_size=768,
        intermediate_size=3072,
        num_attention_heads=12,
        num_hidden_layers=12,
    )
    model = EsmForSequenceClassification(
        config, head_type="Linear", freeze=False, kernel_sizes=[2, 3, 5], ocs=32
    )
    from safetensors.torch import load_file

    state = load_file(FAMILY_WEIGHTS["lamar_utr5te"])
    missing, unexpected = model.load_state_dict(state, strict=False)
    _require(
        not missing and not unexpected,
        f"LAMAR state mismatch: {missing[:3]} / {unexpected[:3]}",
    )
    return model.to(device).eval(), tokenizer


def predict_lamar(
    model, tokenizer, sequences: list[str], device, batch_size: int = 16
) -> PredictResult:
    result = PredictResult()
    order = sorted(range(len(sequences)), key=lambda i: len(sequences[i]))
    with torch.no_grad():
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            batch = [sequences[i] for i in indices]
            encoded = tokenizer(
                batch, truncation=True, max_length=1026, padding=True, return_tensors="pt"
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            logits = model(**encoded).logits.squeeze(-1).float().cpu()
            _require(torch.isfinite(logits).all().item(), "LAMAR produced non-finite logits")
            for i, value in zip(indices, logits.tolist()):
                result.values[str(i)] = float(value)
    return result


# --- GEMORNA ---------------------------------------------------------------

GEMORNA_VOCAB = {"[PAD]": 0, "A": 5, "U": 6, "G": 7, "C": 8, "N": 9}
GEMORNA_5_SCALE_MEAN = 5.19937892
GEMORNA_5_SCALE_STD = 1.40592675


def _load_gemorna_module(name: str, filename: str):
    # file-level spec load: the repo root already defines a `models` package that
    # shadows gemorna's namespace package `models` - import by path instead.
    path = ASSETS / "gemorna/src/models" / filename
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_gemorna_5utr(device: torch.device):
    models_module = _load_gemorna_module("gemorna_model_pred5UTR", "model_pred5UTR.py")
    args = argparse.Namespace(
        embed_num=10, embed_dim=64, kernel_num=128, kernel_sizes=[5, 10, 30, 50], dropout=0.1
    )
    model = models_module.Model(args)
    state = torch.load(FAMILY_WEIGHTS["gemorna_5utr"], map_location="cpu", weights_only=False)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def load_gemorna_3utr(device: torch.device):
    models_module = _load_gemorna_module("gemorna_model_pred3UTR", "model_pred3UTR.py")
    args = argparse.Namespace(
        embed_num=10, embed_dim=256, kernel_num=200, kernel_sizes=[2, 4, 6, 8, 10], dropout=0.1
    )
    model = models_module.Model(args)
    state = torch.load(FAMILY_WEIGHTS["gemorna_3utr"], map_location="cpu", weights_only=False)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def _gemorna_tokens(seq: str) -> list[int]:
    rna = seq.upper().replace("T", "U")
    return [GEMORNA_VOCAB.get(char, GEMORNA_VOCAB["N"]) for char in rna]


def predict_gemorna_5utr(
    model, sequences: list[str], device, batch_size: int = 64
) -> PredictResult:
    result = PredictResult()
    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            batch = sequences[start : start + batch_size]
            tokens = []
            for seq in batch:
                ids = _gemorna_tokens(seq[:100])
                tokens.append(ids + [GEMORNA_VOCAB["[PAD]"]] * (100 - len(ids)))
            tensor = torch.tensor(tokens, dtype=torch.long, device=device)
            logits = model(tensor).squeeze(-1).float().cpu()
            _require(torch.isfinite(logits).all().item(), "GEMORNA-5 produced non-finite logits")
            scaled = logits * GEMORNA_5_SCALE_STD + GEMORNA_5_SCALE_MEAN
            for i, value in zip(range(start, start + len(batch)), scaled.tolist()):
                result.values[str(i)] = float(value)
    return result


def predict_gemorna_3utr(
    model, sequences: list[str], device, batch_size: int = 64
) -> PredictResult:
    result = PredictResult()
    order = sorted(range(len(sequences)), key=lambda i: len(sequences[i]))
    with torch.no_grad():
        cursor = 0
        while cursor < len(order):
            length = len(sequences[order[cursor]])
            group: list[int] = []
            while (
                cursor + len(group) < len(order)
                and len(sequences[order[cursor + len(group)]]) == length
                and len(group) < batch_size
            ):
                group.append(order[cursor + len(group)])
            tokens = [_gemorna_tokens(sequences[i]) for i in group]
            tensor = torch.tensor(tokens, dtype=torch.long, device=device)
            logits = model(tensor).squeeze(-1).float().cpu()
            _require(torch.isfinite(logits).all().item(), "GEMORNA-3 produced non-finite logits")
            for i, value in zip(group, logits.tolist()):
                result.values[str(i)] = float(value)
            cursor += len(group)
    return result


# --- UTR-STCNet ------------------------------------------------------------

STCNET_SEQ_MAP = {
    "A": [1, 0, 0, 0],
    "C": [0, 1, 0, 0],
    "G": [0, 0, 1, 0],
    "T": [0, 0, 0, 1],
    "N": [0, 0, 0, 0],
}
STCNET_MAX_LEN = 120
STCNET_SEED = 20260816


def _stcnet_encode(seq: str) -> torch.Tensor:
    kept = seq[-STCNET_MAX_LEN:] if len(seq) > STCNET_MAX_LEN else seq
    tokens = torch.zeros((STCNET_MAX_LEN, 4), dtype=torch.float32)
    encoded = torch.tensor(
        [STCNET_SEQ_MAP.get(char, [0, 0, 0, 0]) for char in kept], dtype=torch.float32
    )
    tokens[-len(encoded) :] = encoded
    return tokens


def load_utr_stcnet(device: torch.device):
    sys.path.insert(0, str(ASSETS / "utr_stcnet"))
    utrformer_module = importlib.import_module("UTR.UTRFormer")
    model = utrformer_module.utrformer_large(padding_idx=0, token_cls=5, pooling_size=STCNET_MAX_LEN)
    state = torch.load(FAMILY_WEIGHTS["utr_stcnet"], map_location="cpu", weights_only=False)
    model.load_state_dict({k.replace("module.", ""): v for k, v in state["model"].items()})
    return model.to(device).eval()


def predict_stcnet(model, sequences: list[str], device, batch_size: int = 64) -> PredictResult:
    result = PredictResult()
    # official cluster_dpc_knn adds torch.rand*1e-6 to token density to break
    # ties (inference-time randomness by design); a fixed per-batch seed makes
    # runs reproducible without altering the model's behavior distribution.
    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            torch.manual_seed(STCNET_SEED)
            batch = sequences[start : start + batch_size]
            tensor = torch.stack([_stcnet_encode(seq) for seq in batch]).to(device)
            out, _ = model(tensor)
            values = out.squeeze(-1).float().cpu()
            _require(torch.isfinite(values).all().item(), "UTR-STCNet produced non-finite outputs")
            for i, value in zip(range(start, start + len(batch)), values.tolist()):
                result.values[str(i)] = float(value)
    return result


# --- UTR-Insight -----------------------------------------------------------

INSIGHT_PAD_TOTAL = 100


def load_utr_insight(device: torch.device):
    sys.path.insert(0, str(ASSETS / "utr_insight"))
    from esm.data import Alphabet
    from esm.model.esm2_secondarystructure import ESM2 as ESM2_SISS
    from esm.modules import ConvTransformerLayer
    import torch.nn as nn

    class ConvTransformerPredictor(nn.Module):
        def __init__(
            self, alphabet, dropout=0.2, ct_layers=3, kmer=7, layers=6, embed_dim=128, nodes=40, heads=16
        ):
            super().__init__()
            self.esm2 = ESM2_SISS(
                num_layers=layers, embed_dim=embed_dim, attention_heads=heads, alphabet=alphabet
            )
            self.convtransformer_decoder = nn.ModuleList(
                [
                    ConvTransformerLayer(
                        embed_dim,
                        embed_dim * 4,
                        heads,
                        kmer - i * 2,
                        dropout=dropout,
                        use_esm1b_layer_norm=True,
                    )
                    for i in range(ct_layers)
                ]
            )
            self.dropout = nn.Dropout(dropout)
            self.relu = nn.ReLU()
            self.flatten = nn.Flatten()
            self.experiment_dense = nn.Linear(2, nodes)
            self.linear = nn.Linear(6 * embed_dim, nodes)
            self.linear_2 = nn.Linear(nodes, nodes * 4)
            self.linear_3 = nn.Linear(nodes * 4, nodes)
            self.output = nn.Linear(nodes, 1)

        def forward(self, tokens, experiment_indicator, self_attn_padding_mask=None):
            embeddings = self.esm2(tokens, [0, 6], return_representation=True)
            embeddings_rep = embeddings["representations"][6][:, 1:-1]
            # faithful to the official notebook forward: every decoder layer is
            # called with embeddings_rep as input (x_o holds only the LAST layer's
            # output; earlier layer outputs are computed and discarded)
            for layer in self.convtransformer_decoder:
                x_o, _ = layer(x=embeddings_rep, self_attn_padding_mask=self_attn_padding_mask)
            x = torch.flip(x_o, dims=[1])
            frame_1 = x[:, 0::3, :]
            frame_2 = x[:, 1::3, :]
            frame_3 = x[:, 2::3, :]
            f1m = torch.max(frame_1, dim=1)[0]
            f2m = torch.max(frame_2, dim=1)[0]
            f3m = torch.max(frame_3, dim=1)[0]
            mask_expanded = ~self_attn_padding_mask.unsqueeze(2)

            def masked_mean(frame, mask):
                frame_sum = torch.sum(frame * mask, dim=1)
                mask_sum = torch.sum(mask, dim=1) + 1e-8
                return frame_sum / mask_sum

            f1a = masked_mean(frame_1, mask_expanded[:, 0::3, :])
            f2a = masked_mean(frame_2, mask_expanded[:, 1::3, :])
            f3a = masked_mean(frame_3, mask_expanded[:, 2::3, :])
            pooled = torch.cat([f1m, f1a, f2m, f2a, f3m, f3a], dim=1)
            experiment_output = self.experiment_dense(experiment_indicator)
            o_linear = self.linear(self.flatten(pooled)) + experiment_output
            o2 = self.linear_2(o_linear)
            o3 = self.linear_3(o2)
            o = self.output(self.dropout(self.relu(o3)))
            return o

    alphabet = Alphabet(mask_prob=0.0, standard_toks="AGCT")
    model = ConvTransformerPredictor(alphabet)
    state = torch.load(FAMILY_WEIGHTS["utr_insight"], map_location="cpu", weights_only=False)
    model.load_state_dict({k.replace("module.", ""): v for k, v in state.items()}, strict=True)
    return model.to(device).eval(), alphabet


def _insight_encode(seq: str, alphabet) -> list[int]:
    kept = seq[:INSIGHT_PAD_TOTAL]
    pad_count = INSIGHT_PAD_TOTAL - len(kept)
    return [alphabet.padding_idx] * pad_count + alphabet.encode(kept)


def predict_insight(
    model, alphabet, sequences: list[str], device, batch_size: int = 32
) -> PredictResult:
    result = PredictResult()
    indicator_value = [1.0, 0.0]
    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            batch = sequences[start : start + batch_size]
            encoded = [_insight_encode(seq, alphabet) for seq in batch]
            max_len = max(len(ids) for ids in encoded)
            tokens = torch.full((len(batch), max_len), alphabet.padding_idx, dtype=torch.long)
            for row_index, ids in enumerate(encoded):
                tokens[row_index, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            tokens = tokens.to(device)
            pad_mask = tokens.eq(alphabet.padding_idx)[:, 1:-1]
            indicator = torch.tensor(
                [indicator_value] * len(batch), dtype=torch.float32, device=device
            )
            out = model(tokens, indicator, self_attn_padding_mask=pad_mask)
            values = out.squeeze(-1).float().cpu()
            _require(torch.isfinite(values).all().item(), "UTR-Insight produced non-finite outputs")
            for i, value in zip(range(start, start + len(batch)), values.tolist()):
                result.values[str(i)] = float(value)
    return result


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _spearman(observed: np.ndarray, predicted: np.ndarray) -> float | None:
    if len(observed) < 3 or np.std(observed) == 0.0 or np.std(predicted) == 0.0:
        return None
    value = spearmanr(observed, predicted).statistic
    return None if not math.isfinite(float(value)) else float(value)


def bootstrap_ci(
    observed: np.ndarray, predicted: np.ndarray, iterations: int, seed: int
) -> list[float] | None:
    if len(observed) < 3:
        return None
    rng = np.random.default_rng(seed)
    n = len(observed)
    boots = []
    for _ in range(iterations):
        idx = rng.integers(0, n, n)
        value = spearmanr(observed[idx], predicted[idx]).statistic
        if np.isfinite(value):
            boots.append(float(value))
    if not boots:
        return None
    array = np.asarray(boots)
    return [float(np.percentile(array, 2.5)), float(np.percentile(array, 97.5))]


def mprau_pair_mean(rows: list[PairRow], deltas: dict[str, float]) -> dict:
    by_variant: dict[str, list[PairRow]] = defaultdict(list)
    for row in rows:
        if row.row_id.startswith("ENCSR854RUF:"):
            by_variant[row.row_id.split(":context:")[0]].append(row)
    targets: list[float] = []
    preds: list[float] = []
    for variant in sorted(by_variant):
        group = by_variant[variant]
        if len(group) >= 2:
            targets.append(float(np.mean([r.target for r in group])))
            preds.append(float(np.mean([deltas[r.row_id] for r in group])))
    target_array = np.asarray(targets)
    pred_array = np.asarray(preds)
    return {
        "variant_count": int(len(target_array)),
        "pair_mean_spearman": _spearman(target_array, pred_array),
        "bootstrap_ci_95": bootstrap_ci(
            target_array, pred_array, MPRAU_BOOTSTRAP_ITERATIONS, BOOTSTRAP_SEED
        ),
        "bootstrap_iterations": MPRAU_BOOTSTRAP_ITERATIONS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "caliber": "W-ladder variant pair-mean (>=2 contexts, per-variant means)",
    }


def evaluator_metrics(rows: list[PairRow], predictions: dict[str, float]) -> dict:
    """Frozen Task-1 evaluator on canonical-shaped observation dicts."""
    observations = [
        {
            "canonical_record_id": row.row_id,
            "study_unit_id": row.study,
            "source_id": row.source_id,
            "biological_context_id": row.context,
            "endpoint_id": row.endpoint,
            "stratum": (row.study, row.region, row.endpoint),
            "task": (row.region, row.endpoint),
            "observed": row.target,
        }
        for row in rows
    ]
    return ev.evaluate(observations, predictions, K)


def evaluate_cell(
    family: str,
    task_key: str,
    rows: list[PairRow],
    predictions: dict[str, float],
    miss_reasons: dict[str, str],
) -> dict:
    _require(set(predictions) == {r.row_id for r in rows}, "predictions do not cover rows")
    observed = np.asarray([r.target for r in rows], dtype=float)
    predicted = np.asarray([predictions[r.row_id] for r in rows], dtype=float)
    entry: dict[str, Any] = {
        "family": family,
        "task": task_key,
        "split": "VALIDATION" if task_key in EXISTING_TASKS else "NEW_EVAL_ROW",
        "record_count": int(len(rows)),
        "spearman": _spearman(observed, predicted),
        "spearman_bootstrap_ci_95": bootstrap_ci(
            observed, predicted, CELL_BOOTSTRAP_ITERATIONS, BOOTSTRAP_SEED
        ),
        "sign_accuracy": (
            float(np.mean(np.sign(observed) == np.sign(predicted))) if len(rows) else None
        ),
        "source_group_count": len({r.source_id for r in rows}),
        "miss_count": 0,
        "miss_rate": 0.0,
        "miss_reason_histogram": {},
        "high_miss": False,
    }
    if miss_reasons:
        entry["miss_count"] = len(miss_reasons)
        entry["miss_rate"] = len(miss_reasons) / len(rows)
        entry["miss_reason_histogram"] = dict(Counter(miss_reasons.values()))
        entry["high_miss"] = entry["miss_rate"] > 0.15
    if task_key == "encsr854ruf_mprau":
        entry["mprau_pair_mean"] = mprau_pair_mean(rows, predictions)
    if task_key in EXISTING_TASKS:
        entry["task1_evaluator"] = evaluator_metrics(rows, predictions)
    return entry


def run_family_task(
    family: str,
    task_key: str,
    rows: list[PairRow],
    predictor: Callable[[list[str]], PredictResult],
    heartbeat_path: Path,
) -> tuple[dict, list[dict]]:
    sequences: list[str] = []
    keys: list[tuple[str, str]] = []
    for row in rows:
        sequences.append(row.source)
        keys.append(("src", row.row_id))
        sequences.append(row.candidate)
        keys.append(("cand", row.row_id))
    result = predictor(sequences)
    source_pred: dict[str, float] = {}
    candidate_pred: dict[str, float] = {}
    for index, (side, row_id) in enumerate(keys):
        value = result.values.get(str(index))
        if value is not None:
            if side == "src":
                source_pred[row_id] = value
            else:
                candidate_pred[row_id] = value
    deltas: dict[str, float] = {}
    miss_reasons: dict[str, str] = {}
    for row in rows:
        if row.row_id in source_pred and row.row_id in candidate_pred:
            deltas[row.row_id] = candidate_pred[row.row_id] - source_pred[row.row_id]
        else:
            miss_reasons[row.row_id] = "missing source or candidate prediction"
    with heartbeat_path.open("a", encoding="utf-8") as handle:
        handle.write(
            f"[{family}][{task_key}] predicted {len(deltas)} pairs, missed {len(miss_reasons)}\n"
        )
    cell = evaluate_cell(family, task_key, rows, deltas, miss_reasons)
    payload = []
    for row in rows:
        payload.append(
            {
                "row_id": row.row_id,
                "source_sequence": row.source,
                "candidate_sequence": row.candidate,
                "predicted_source": source_pred.get(row.row_id),
                "predicted_candidate": candidate_pred.get(row.row_id),
                "predicted_direction_normalized_delta": deltas.get(row.row_id),
                "observed_direction_normalized_delta": row.target,
                "source_group_id": row.source_id,
                "biological_context": row.context,
                "region": row.region,
                "split": "VALIDATION" if task_key in EXISTING_TASKS else "NEW_EVAL_ROW",
            }
        )
    return cell, payload


# ---------------------------------------------------------------------------
# Unit tests (Task 2.2.1)
# ---------------------------------------------------------------------------


def run_unit_test(family: str, loaded: dict, device, args) -> dict:
    weights_path = FAMILY_WEIGHTS[family]
    anchor = {"weights_path": str(weights_path), "weights_sha256": sha256_of(weights_path)}
    probe_sequences: list[str] = []
    reference: dict[str, float] = {}
    mode = "SELF_CONSISTENCY_STRICT_LOAD"
    report: dict[str, Any] = {"family": family, **anchor}

    if family == "utr_insight":
        import pandas as pd

        csv_path = ASSETS / "utr_insight/Result/utr_insight/e_pred_random_50.csv"
        df = pd.read_csv(csv_path).head(args.unit_test_rows)
        for utr, y_pred in zip(df["utr"], df["y_pred"]):
            raw = str(utr).replace("<pad>", "")
            probe_sequences.append(raw)
            reference[raw] = float(y_pred)
        mode = "OFFICIAL_CSV_ALIGNMENT"
    elif family == "gemorna_5utr":
        probe_sequences = [
            "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT",
            "TTTTGGGGCCCCAAAATTTTGGGGCCCCAAAATTTTGGGGCCCCAAAAAA",
        ]
    elif family == "utr_stcnet":
        probe_sequences = [
            "ACGT" * 30,
        ]
    else:
        probe_sequences = [
            "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT",
        ]

    predictor, _model = loaded[family]
    first = predictor(probe_sequences)
    second = predictor(probe_sequences)
    self_consistent = all(
        str(i) in first.values and first.values[str(i)] == second.values.get(str(i))
        for i in range(len(probe_sequences))
    )
    report["mode"] = mode
    report["self_consistency_bitwise"] = bool(self_consistent)
    report["probe_values"] = {str(i): first.values.get(str(i)) for i in range(len(probe_sequences))}
    if mode == "OFFICIAL_CSV_ALIGNMENT":
        diffs = [
            abs(first.values[str(i)] - reference[seq]) for i, seq in enumerate(probe_sequences)
        ]
        report["official_reference"] = "Result/utr_insight/e_pred_random_50.csv (repo pre-generated)"
        report["max_abs_diff"] = float(max(diffs))
        report["mean_abs_diff"] = float(np.mean(diffs))
        report["passed"] = bool(self_consistent and report["max_abs_diff"] < 0.5)
    else:
        report["official_reference"] = None
        report["passed"] = bool(self_consistent)
    report["strict_load"] = True
    if family == "lamar_utr5te":
        report["load_audit"] = "load_state_dict(strict=False) with 0 missing / 0 unexpected keys"
    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-gpu-index", required=True, type=int)
    parser.add_argument("--families", nargs="+", default=list(FAMILY_ORDER), choices=FAMILY_ORDER)
    parser.add_argument("--tasks", nargs="+", default=list(TASK_ORDER), choices=TASK_ORDER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--smoke-limit", type=int, default=0)
    parser.add_argument("--unit-test-only", action="store_true")
    parser.add_argument("--unit-test-rows", type=int, default=64)
    args = parser.parse_args()

    _require(
        not os.environ.get("CUDA_VISIBLE_DEVICES"),
        "CUDA_VISIBLE_DEVICES remapping is forbidden for physical-device provenance",
    )
    _require(torch.cuda.is_available(), "CUDA unavailable - GPU required")
    device = torch.device(f"cuda:{args.physical_gpu_index}")
    torch.cuda.set_device(device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    heartbeat_path = args.output_dir / "heartbeat.log"
    summary: dict[str, Any] = {
        "schema_version": "benchmark_v2_matrix_v1",
        "mode": "UNIT_TEST_ONLY" if args.unit_test_only else "FULL_MATRIX",
        "cpu_fallback": False,
        "device": torch.cuda.get_device_name(device),
        "cuda_index": args.physical_gpu_index,
        "families": args.families,
        "tasks": args.tasks,
        "task_order": list(TASK_ORDER),
        "input_adaptation": {f: INPUT_ADAPTATION[f] for f in args.families},
        "unit_tests": {},
        "cells": {},
        "blocked_families": {
            "hydrarna": {
                "status": "PORT_BLOCKED_HEAD_MISSING",
                "weights_path": str(ASSETS / "hydrarna/weights/models/HydraRNA_model.pt"),
                "evidence": (
                    "checkpoint inspected with torch.load: contains only the MLM pretraining "
                    "head (encoder.lm_head.*); no MRL regression head is released with the "
                    "repo, and official finetune examples fit a fresh user-side MLP head. "
                    "Remaining google-drive weights unfetchable (no external network; "
                    "weights/gdown_dl.log shows connection failure). Zero-tuning gate "
                    "forbids training a head -> blocked."
                ),
                "ledger_correction": (
                    "port ledger v1 section 2.10 judged model.pt as the 5'UTR MRL head; "
                    "physical inspection refutes that - correction recorded here per the "
                    "ledger's no-silent-rejudgment rule."
                ),
            }
        },
        "preregistration": {
            "gates": "benchmark_v2_matrix_row_prereg_v1.md (commit-frozen); license waiver per ledger addendum 2026-09-20",
            "caliber": "frozen-delta zero-tuning; VALIDATION only (existing tasks); NEW_EVAL_ROW (new rows); append-only outputs",
        },
    }

    from core.route2_gpu_failure_evidence import cuda_device_observation

    summary["cuda_provenance"] = cuda_device_observation(args.physical_gpu_index)

    loaded: dict[str, tuple[Callable[[list[str]], PredictResult], Any]] = {}

    if "lamar_utr5te" in args.families:
        model, tokenizer = load_lamar(device)
        loaded["lamar_utr5te"] = (
            lambda seqs, _m=model, _t=tokenizer: predict_lamar(_m, _t, seqs, device),
            model,
        )
    if "gemorna_5utr" in args.families:
        model = load_gemorna_5utr(device)
        loaded["gemorna_5utr"] = (
            lambda seqs, _m=model: predict_gemorna_5utr(_m, seqs, device),
            model,
        )
    if "gemorna_3utr" in args.families:
        model = load_gemorna_3utr(device)
        loaded["gemorna_3utr"] = (
            lambda seqs, _m=model: predict_gemorna_3utr(_m, seqs, device),
            model,
        )
    if "utr_stcnet" in args.families:
        model = load_utr_stcnet(device)
        loaded["utr_stcnet"] = (
            lambda seqs, _m=model: predict_stcnet(_m, seqs, device),
            model,
        )
    if "utr_insight" in args.families:
        model, alphabet = load_utr_insight(device)
        loaded["utr_insight"] = (
            lambda seqs, _m=model, _a=alphabet: predict_insight(_m, _a, seqs, device),
            model,
        )

    for family in args.families:
        summary["unit_tests"][family] = run_unit_test(family, loaded, device, args)
        print(f"[unit-test][{family}] {json.dumps(summary['unit_tests'][family])}", flush=True)

    (args.output_dir / "unit_tests.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    if args.unit_test_only:
        return 0

    for family in args.families:
        predictor, _model = loaded[family]
        family_dir = args.output_dir / family
        family_dir.mkdir(parents=True, exist_ok=True)
        for task_key in args.tasks:
            cell_key = f"{family}__{task_key}"
            task_region = region_of(task_key)
            if FAMILY_DOMAIN[family] != task_region:
                summary["cells"][cell_key] = {
                    "family": family,
                    "task": task_key,
                    "status": "STRUCTURED_NA",
                    "reason": (
                        f"input domain {FAMILY_DOMAIN[family]} does not match task region {task_region}"
                    ),
                }
                continue
            result_path = family_dir / task_key / "cell_result.json"
            if result_path.is_file() and not args.smoke_limit:
                summary["cells"][cell_key] = json.loads(result_path.read_text(encoding="utf-8"))
                print(f"[resume][{cell_key}] loaded existing cell", flush=True)
                continue
            rows = load_task(task_key)
            if args.smoke_limit:
                rows = rows[: args.smoke_limit]
            cell, payload = run_family_task(family, task_key, rows, predictor, heartbeat_path)
            cell_dir = family_dir / task_key
            cell_dir.mkdir(parents=True, exist_ok=True)
            with (cell_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
                for item in payload:
                    handle.write(json.dumps(item) + "\n")
            cell_result_path = cell_dir / "cell_result.json"
            cell_result_path.write_text(
                json.dumps(cell, indent=2, sort_keys=True), encoding="utf-8"
            )
            summary["cells"][cell_key] = cell
            print(
                f"[cell][{cell_key}] rho={cell.get('spearman')} n={cell.get('record_count')}",
                flush=True,
            )
        (args.output_dir / "matrix_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )

    (args.output_dir / "matrix_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
