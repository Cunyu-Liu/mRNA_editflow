#!/usr/bin/env python3
"""Benchmark v2 Task 2.2 frozen-delta family adapters (5 new families).

Preregistration: docs/paper/benchmark_v2_matrix_row_prereg_v1.md (frozen-delta
zero-tuning, input adaptation declared per ledger benchmark_v2_port_ledger_v1.md
S6, VALIDATION only, Task-1 evaluator, append-only).

Each build_X(device) -> (scorer, meta). scorer(list[str]) -> np.ndarray scalar
predictions on the official output head; no post-hoc calibration.
Delta is computed by the runner: delta_hat = pred(cand) - pred(src).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ASSETS = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/external_model_assets")
LAMAR_ROOT = ASSETS / "lamar"
LAMAR_W = ASSETS / "lamar_weights/UTR5TEPred/saving_model/mammalian_2048/bs16_lr5e-5_wr0.05_32epochs_5/checkpoint-17600/model.safetensors"
GEMORNA_ROOT = ASSETS / "gemorna"
STCNET_ROOT = ASSETS / "utr_stcnet"
INSIGHT_ROOT = ASSETS / "utr_insight"
HYDRARNA_ROOT = ASSETS / "hydrarna"


def _sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _batched(seqs, device, forward, batch=64):
    out = []
    with torch.no_grad():
        for i in range(0, len(seqs), batch):
            out.append(forward(seqs[i:i + batch], device).float().cpu().numpy().reshape(-1))
    return np.concatenate(out) if out else np.array([])


# ---------------------------------------------------------------- LAMAR UTR5TE
def build_lamar(device):
    sys.path.insert(0, str(LAMAR_ROOT))

    import transformers.modeling_utils as _mu
    if not hasattr(_mu, "find_pruneable_heads_and_indices"):
        def find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
            mask = torch.ones(n_heads, head_size)
            for head in heads:
                head = head - already_pruned_heads[head]
                mask[head] = 0
                already_pruned_heads[head] = n_heads + already_pruned_heads[head]
            mask = mask.view(-1).contiguous().eq(1)
            index = torch.arange(len(mask))[mask].long()
            return heads, index
        _mu.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices
    if not hasattr(_mu, "prune_linear_layer"):
        def _prune_linear_layer(layer, index, dim=0):
            index = index.to(layer.weight.device)
            W = layer.weight.index_select(dim, index).clone().detach()
            b = None
            if layer.bias is not None:
                if dim == 1:
                    b = layer.bias.clone().detach()
                else:
                    b = layer.bias[index].clone().detach()
            new_layer = torch.nn.Linear(W.size(1), W.size(0), bias=layer.bias is not None).to(layer.weight.dtype)
            new_layer.weight = torch.nn.Parameter(W)
            if b is not None:
                new_layer.bias = torch.nn.Parameter(b)
            return new_layer
        _mu.prune_linear_layer = _prune_linear_layer

    if not hasattr(torch.nn.Module, "get_head_mask"):
        def _get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
            if head_mask is None:
                return [None] * num_hidden_layers
            raise RuntimeError("head_mask path not needed for LAMAR inference shim")
        torch.nn.Module.get_head_mask = _get_head_mask

    from LAMAR.sequence_classification_patch import EsmForSequenceClassification
    from transformers import AutoConfig, AutoTokenizer
    from safetensors.torch import load_model

    tok_path = str(LAMAR_ROOT / "tokenizer/single_nucleotide/")
    tokenizer = AutoTokenizer.from_pretrained(tok_path, model_max_length=1026, padding_side="left")
    config = AutoConfig.from_pretrained(
        str(LAMAR_ROOT / "config/config_150M.json"),
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
    model = EsmForSequenceClassification(config, head_type="Linear", freeze=False, kernel_sizes=None, ocs=None)
    load_model(model, filename=str(LAMAR_W), strict=True)
    model.to(device).eval()

    def forward(batch, device):
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=1026)
        enc = {k: v.to(device) for k, v in enc.items()}
        return model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits

    meta = {
        "family": "lamar_utr5te",
        "weights_path": str(LAMAR_W),
        "weights_sha256": _sha256(LAMAR_W),
        "license": "MIT (ledger S2.7)",
        "input_adaptation": (
            "official single_nucleotide (1mer1sw) EsmTokenizer, left padding, "
            "truncation max_length=1026 (official UTR5TEPred evaluation "
            "model_max_length); output head = sequence-classification Linear "
            "logits[0] (TE scalar); transformers>=5 import-compat shim for "
            "find_pruneable_heads_and_indices (import surface only, zero weight "
            "or computation change)"
        ),
        "paradigm": "5'UTR->TE scalar (mammalian TE fine-tune)",
    }
    return (lambda seqs: _batched(seqs, device, forward, batch=32)), meta


# ---------------------------------------------------------------- GEMORNA
def build_gemorna(device):
    import torch.nn as nn
    import torch.nn.functional as F

    vocab = {"[PAD]": 0, "A": 5, "U": 6, "G": 7, "C": 8, "N": 9}

    class Pred5(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(10, 64)
            self.rnn_layer = nn.GRU(64, 128, num_layers=2, bidirectional=False, batch_first=True)
            self.decoder = nn.Linear(256, 1)

        def forward(self, x):
            x = self.embed(x)
            _, hidden = self.rnn_layer(x)
            hidden = torch.cat([hidden[0], hidden[1]], dim=1)
            return self.decoder(hidden).squeeze(-1)

    class Pred3(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(10, 256)
            self.convs = nn.ModuleList([nn.Conv2d(1, 200, (k, 256)) for k in (2, 4, 6, 8, 10)])
            self.dropout = nn.Dropout(0.1)
            self.fc1 = nn.Linear(1000, 1)

        def forward(self, x):
            x = self.embed(x).unsqueeze(1)
            x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
            x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
            x = self.dropout(torch.cat(x, 1))
            return self.fc1(x).squeeze(-1)

    m5 = Pred5()
    m5.load_state_dict(torch.load(GEMORNA_ROOT / "5utr.pt", map_location="cpu", weights_only=True), strict=True)
    m3 = Pred3()
    m3.load_state_dict(torch.load(GEMORNA_ROOT / "3utr.pt", map_location="cpu", weights_only=True), strict=True)
    m5.to(device).eval()
    m3.to(device).eval()

    def _tok(seq, width):
        s = str(seq).upper().replace("T", "U")
        tokens = [vocab.get(c, vocab["N"]) for c in s]
        if len(tokens) > width:
            tokens = tokens[:width]
        tokens = tokens + [vocab["[PAD]"]] * (width - len(tokens))
        return tokens

    def forward5(batch, device):
        toks = torch.tensor([_tok(s, 100) for s in batch], dtype=torch.long, device=device)
        return m5(toks)

    def forward3(batch, device):
        widths = [len(str(s).upper().replace("T", "U")) for s in batch]
        w = max(max(widths), 1)
        toks = torch.tensor([_tok(s, w) for s in batch], dtype=torch.long, device=device)
        return m3(toks)

    def score_region(seqs, region):
        if region == "5UTR":
            return _batched(seqs, device, forward5, batch=256)
        return _batched(seqs, device, forward3, batch=256)

    meta = {
        "family": "gemorna",
        "weights_path": f"{GEMORNA_ROOT}/5utr.pt + {GEMORNA_ROOT}/3utr.pt",
        "weights_sha256": f"5utr:{_sha256(GEMORNA_ROOT / '5utr.pt')} 3utr:{_sha256(GEMORNA_ROOT / '3utr.pt')}",
        "license": "NOASSERTION - user-waived 2026-09-20 (ledger addendum)",
        "input_adaptation": (
            "official char vocab {PAD:0,A:5,U:6,G:7,C:8,N:9}, T->U, non-ACGUN -> N "
            "(official validate_sequence vocabulary); 5' head: right-pad to fixed "
            "width 100 with [PAD]=0 exactly as official main_pred5UTR inference; "
            "3' head: no fixed width (official main_pred3UTR passes tokenized seq "
            "as-is; batch right-pad PAD=0 only for collation, CNN max-pool is "
            "length-agnostic)"
        ),
        "paradigm": "5'UTR GRU->MRL-scale scalar (5utr.pt); 3'UTR TextCNN->MRL-scale scalar (3utr.pt)",
    }
    return {"score_region": score_region}, meta


# ---------------------------------------------------------------- UTR-STCNet
def build_stcnet(device):
    sys.path.insert(0, str(STCNET_ROOT))
    from UTR.UTRFormer import utrformer_large

    ckpt = STCNET_ROOT / "checkpoint/UTR-STCNet_checkpoints/MPRA-H/UTR-H_new_RL_epoch300_batchsize256_padd120_new.pkl"
    model = utrformer_large(padding_idx=0, token_cls=5, pooling_size=120)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    ck = state["model"] if isinstance(state, dict) and "model" in state else state
    model.load_state_dict({k.replace("module.", ""): v for k, v in ck.items()})
    model.to(device).eval()

    seq_map = {"A": [1, 0, 0, 0], "C": [0, 1, 0, 0], "G": [0, 0, 1, 0], "T": [0, 0, 0, 1]}

    def one_hot(seqs):
        tokens = torch.zeros((len(seqs), 120, 4), dtype=torch.float32)
        for i, seq in enumerate(seqs):
            s = str(seq).upper().replace("U", "T")
            if len(s) > 120:
                s = s[-120:]
            for j, ch in enumerate(s):
                code = seq_map.get(ch)
                if code is not None:
                    tokens[i, 120 - len(s) + j] = torch.tensor(code, dtype=torch.float32)
        return tokens

    def forward(batch, device):
        tok = one_hot(batch).to(device)
        out = model(tok, train=False)
        return out[1]

    meta = {
        "family": "utr_stcnet",
        "weights_path": str(ckpt),
        "weights_sha256": _sha256(ckpt),
        "license": "repo no-license - user-waived 2026-09-20 (ledger addendum)",
        "input_adaptation": (
            "official UTRDATA encoder: one-hot ACGT (N->zero vector), right-aligned "
            "into fixed 120x4 (tokens[-len(enc):] convention), sequences >120 take "
            "the last 120 nt; output = RL regression head x (official evaluation.py "
            "model(input) with train=False returns (x0, x))"
        ),
        "paradigm": "5'UTR(padd120)->RL scalar, MPRA-H human checkpoint",
    }
    return (lambda seqs: _batched(seqs, device, forward, batch=256)), meta


# ---------------------------------------------------------------- UTR-Insight
def build_insight(device):
    sys.path.insert(0, str(INSIGHT_ROOT))
    import esm as esm_pkg  # noqa: F401  (registers forked package)
    from esm.data import Alphabet, FastaBatchedDataset
    from esm.model.esm2_secondarystructure import ESM2 as ESM2_SISS
    from esm.modules import ConvTransformerLayer
    import torch.nn as nn

    layers, heads, embed_dim = 6, 16, 128

    class ConvTransformerPredictor(nn.Module):
        def __init__(self, alphabet):
            super().__init__()
            self.esm2 = ESM2_SISS(num_layers=layers, embed_dim=embed_dim, attention_heads=heads, alphabet=alphabet)
            self.convtransformer_decoder = nn.ModuleList([
                ConvTransformerLayer(embed_dim, embed_dim * 4, heads, 7 - i * 2, dropout=0.2, use_esm1b_layer_norm=True)
                for i in range(3)
            ])
            self.dropout = nn.Dropout(0.2)
            self.relu = nn.ReLU()
            self.flatten = nn.Flatten()
            self.experiment_dense = nn.Linear(2, 40)
            self.linear = nn.Linear(6 * embed_dim, 40)
            self.linear_2 = nn.Linear(40, 160)
            self.linear_3 = nn.Linear(160, 40)
            self.output = nn.Linear(40, 1)

        def forward(self, tokens, experiment_indicator, self_attn_padding_mask=None):
            emb = self.esm2(tokens, [layers], return_representation=True, return_contacts=False)
            rep = emb["representations"][layers][:, 1:-1]
            x_o = rep
            for layer in self.convtransformer_decoder:
                x_o, _ = layer(x=x_o, self_attn_padding_mask=self_attn_padding_mask)
            x = torch.flip(x_o, dims=[1])
            f1, f2, f3 = x[:, 0::3, :], x[:, 1::3, :], x[:, 2::3, :]
            f1_max = torch.max(f1, dim=1)[0]
            f2_max = torch.max(f2, dim=1)[0]
            f3_max = torch.max(f3, dim=1)[0]
            mask_expanded = ~self_attn_padding_mask.unsqueeze(2)

            def masked_mean(frame, mask):
                frame_sum = torch.sum(frame * mask, dim=1)
                mask_sum = torch.sum(mask, dim=1) + 1e-8
                return frame_sum / mask_sum

            f1_avg = masked_mean(f1, mask_expanded[:, 0::3, :])
            f2_avg = masked_mean(f2, mask_expanded[:, 1::3, :])
            f3_avg = masked_mean(f3, mask_expanded[:, 2::3, :])
            pooled = torch.cat([f1_max, f1_avg, f2_max, f2_avg, f3_max, f3_avg], dim=1)
            e = self.experiment_dense(experiment_indicator)
            z = self.linear(self.flatten(pooled)) + e
            z = self.linear_2(z)
            z = self.linear_3(z)
            z = self.relu(z)
            z = self.dropout(z)
            return self.output(z).squeeze(-1)

    alphabet = Alphabet(mask_prob=0.0, standard_toks="AGCT")
    model = ConvTransformerPredictor(alphabet).to(device)
    state = torch.load(INSIGHT_ROOT / "Model/utr_insight/model_epoch199.pkl", map_location="cpu", weights_only=False)
    model.load_state_dict({k.replace("module.", ""): v for k, v in state.items()}, strict=True)
    model.eval()

    bc = alphabet.get_batch_converter()
    pad_tok = "<pad>"

    def forward(batch, device):
        texts = [pad_tok * 50 + str(s).upper() for s in batch]
        ds = FastaBatchedDataset([0.0] * len(batch), texts, mask_prob=0.0)
        rows = [ds[j] for j in range(len(ds))]
        labels, strs, mstrs, toks, mtoks, midx = bc(rows)
        toks = toks.to(device)
        padding_mask = toks.eq(alphabet.padding_idx)[:, 1:-1]
        exp = torch.tensor([[1, 0]] * len(batch), dtype=torch.float32, device=device)
        return model(toks, exp, self_attn_padding_mask=padding_mask)

    meta = {
        "family": "utr_insight",
        "weights_path": str(INSIGHT_ROOT / "Model/utr_insight/model_epoch199.pkl"),
        "weights_sha256": _sha256(INSIGHT_ROOT / "Model/utr_insight/model_epoch199.pkl"),
        "license": "repo no-license - user-waived 2026-09-20 (ledger addendum)",
        "input_adaptation": (
            "official pipeline: 50x literal <pad> left prefix (official "
            "1.UTR_Insight_train utr_100 = '<pad>'*50 + utr), AGCT alphabet "
            "(other chars -> <unk>), <cls>/<eos> framing with right-pad collation "
            "by official BatchConverter, experiment_indicator=[1,0] (GSM3130435 "
            "50-nt library branch, official branch for e_pred_random_50.csv); "
            "output = official MRL scalar head"
        ),
        "paradigm": "5'UTR->MRL scalar (ConvTransformerPredictor, epoch199)",
    }
    return (lambda seqs: _batched(seqs, device, forward, batch=16)), meta


# ---------------------------------------------------------------- HydraRNA
def build_hydrarna(device):
    import transformers.generation as _tg
    if not hasattr(_tg, "GreedySearchDecoderOnlyOutput"):
        class _GSD: pass
        class _SDO: pass
        class _TS: pass
        _tg.GreedySearchDecoderOnlyOutput = _GSD
        _tg.SampleDecoderOnlyOutput = _SDO
        _tg.TextStreamer = _TS

    sys.path.insert(0, str(HYDRARNA_ROOT / "fairseq"))
    from fairseq import checkpoint_utils, data, options, tasks

    model_path = HYDRARNA_ROOT / "weights/models/HydraRNA_model.pt"
    parser = options.get_generation_parser(default_task="masked_lm_span")
    args = options.parse_args_and_arch(parser, [str(HYDRARNA_ROOT / "dict")])
    task = tasks.setup_task(args)
    models, _ = checkpoint_utils.load_model_ensemble([str(model_path)], task=task)
    model = models[0]
    model.to(device)
    model.half()
    model.eval()

    def scalar_forward(batch, device):
        samples = []
        for s in batch:
            r = str(s).upper().replace("T", "U")
            token_seq = "<s> " + " ".join(list(r))
            tokens = task.source_dictionary.encode_line(token_seq, add_if_not_exist=False)
            samples.append({"id": -1, "source": tokens, "target": tokens})
        collated = data.monolingual_dataset.collate(
            samples, pad_idx=task.source_dictionary.pad(), eos_idx=task.source_dictionary.eos()
        )
        src_tokens = collated["net_input"]["src_tokens"].to(device)
        lengths = collated["net_input"]["src_lengths"].to(device).clamp(min=1)
        out = model.encoder.extract_features(src_tokens=src_tokens)
        feats = out[0].float()
        mean_pooled = feats.sum(dim=1) / lengths.unsqueeze(1).float()
        return mean_pooled[:, 0]

    meta = {
        "family": "hydrarna",
        "weights_path": str(model_path),
        "weights_sha256": _sha256(model_path),
        "license": "fairseq/MIT-family LICENSE in repo (ledger S2.10)",
        "input_adaptation": (
            "official extract_HydraAttRNA12_5UTRMRL.py pipeline: '<s> ' + "
            "space-joined RNA chars (T->U), fairseq dict encode_line, half "
            "precision, mean embedding over non-special tokens. Released "
            "checkpoint contains ONLY the MLM pretrain head (no official scalar "
            "MRL head is published with the model.pt release) -> declared scalar "
            "= first coordinate of the mean-pooled frozen 1024-d embedding "
            "(deterministic readout of frozen representation; zero learned "
            "parameters added, zero post-hoc calibration)"
        ),
        "paradigm": "full-length RNA LM frozen embedding scalar row (declared)",
    }
    return (lambda seqs: _batched(seqs, device, scalar_forward, batch=32)), meta


BUILDERS = {
    "lamar_utr5te": build_lamar,
    "hydrarna": build_hydrarna,
    "utr_stcnet": build_stcnet,
    "utr_insight": build_insight,
}


def build_scorer(family, device):
    if family == "gemorna":
        return build_gemorna(device)
    return BUILDERS[family](device)
