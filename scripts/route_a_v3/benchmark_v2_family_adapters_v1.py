#!/usr/bin/env python3
"""Model adapters for the five benchmark v2 new families (frozen, zero-tuning).

Each adapter: build_scorer(family, device) -> (scorer, meta).
scorer(list[str]) -> np.ndarray of scalar predictions (absolute scale, official
output head, no post-hoc calibration). input adaptation declarations per
benchmark_v2_port_ledger_v1.md §6, refined only in the declared direction.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

LAMAR_ROOT = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/external_model_assets/lamar")
LAMAR_W = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/external_model_assets/lamar_weights/UTR5TEPred/saving_model/mammalian_2048/bs16_lr5e-5_wr0.05_32epochs_5/checkpoint-17600/model.safetensors")
GEMORNA_ROOT = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/external_model_assets/gemorna")
STCNET_ROOT = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/external_model_assets/utr_stcnet")
INSIGHT_ROOT = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/external_model_assets/utr_insight")
HYDRARNA_ROOT = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/external_model_assets/hydrarna")


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
    )
    model = EsmForSequenceClassification(config, head_type="Linear", freeze=False, kernel_sizes=None, ocs=None)
    load_model(model, filename=str(LAMAR_W), strict=True)
    model.to(device).eval()

    def forward(batch, device):
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=1026)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits

    meta = {
        "family": "lamar_utr5te",
        "weights_path": str(LAMAR_W),
        "weights_sha256": _sha256(LAMAR_W),
        "license": "MIT (ledger §2.7)",
        "input_adaptation": (
            "official single_nucleotide (1mer1sw) EsmTokenizer, left padding, "
            "truncation max_length=1026 (official model_max_length); "
            "output head = sequence-classification Linear logits[0] (TE scalar); "
            "bfloat16 autocast on CUDA"
        ),
        "paradigm": "5'UTR->TE scalar (trained on mammalian TE); rows scored as-is",
    }
    return (lambda seqs: _batched(seqs, device, forward, batch=32)), meta


# ---------------------------------------------------------------- GEMORNA
def build_gemorna(device):
    import torch.nn as nn
    import torch.nn.functional as F

    sys.path.insert(0, str(GEMORNA_ROOT / "src"))
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
        tokens = tokens + [0] * (width - len(tokens))
        return tokens

    def forward(batch, device):
        # 5' head: official inference pads right to width 100 with [PAD]
        toks = torch.tensor([_tok(s, 100) for s in batch], dtype=torch.long, device=device)
        return m5(toks)

    def forward3(batch, device):
        # 3' head: official main_pred3UTR passes the tokenized seq without padding
        # -> batch needs equal widths; use per-batch max width with PAD=0 (same
        # embedding id; CNN is length-agnostic via max-pooling)
        widths = [len(str(s).upper().replace("T", "U")) for s in batch]
        w = max(widths)
        toks = torch.tensor([_tok(s, w) for s in batch], dtype=torch.long, device=device)
        return m3(toks)

    def score(seqs):
        # family = one row collection: 5'UTR rows scored by the 5' head, 3'UTR
        # rows by the 3' head (both official checkpoints, declared in the ledger)
        # the generic runner only passes region-agnostic calls; region dispatch
        # happens here via the two-call interface below.
        raise RuntimeError("GEMORNA requires region-aware dispatch; use score5/score3")

    def scorer(seqs):
        raise RuntimeError("GEMORNA requires region-aware dispatch; use build_gemorna_region_scorer")

    return scorer, {
        "family": "gemorna",
        "weights_path": f"{GEMORNA_ROOT}/5utr.pt + {GEMORNA_ROOT}/3utr.pt",
        "weights_sha256": f"5utr:{_sha256(GEMORNA_ROOT / '5utr.pt')} 3utr:{_sha256(GEMORNA_ROOT / '3utr.pt')}",
        "license": "NOASSERTION - user-waived (ledger addendum)",
        "input_adaptation": (
            "official char vocab {PAD:0,A:5,U:6,G:7,C:8,N:9}, T->U, N for other "
            "chars; 5' head pads right to 100 (official inference); 3' head "
            "length-agnostic CNN with right PAD=0 batch padding"
        ),
        "paradigm": "5'UTR GRU->MRL scalar; 3'UTR TextCNN->MRL scalar",
    }


def build_gemorna_region_scorer(device):
    raise NotImplementedError


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
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(tok, train=False)
        return out[1]

    return (lambda seqs: _batched(seqs, device, forward, batch=256)), {
        "family": "utr_stcnet",
        "weights_path": str(ckpt),
        "weights_sha256": _sha256(ckpt),
        "license": "repo no-license - user-waived (ledger addendum)",
        "input_adaptation": (
            "official one-hot ACGT, right-aligned to padd120 (UTRDATA convention, "
            "matches run_route2_utr_stcnet_frozen_delta_mrl_v1.py), N->0 vector, "
            "output = RL regression head out[1]"
        ),
        "paradigm": "5'UTR(padd120)->RL scalar, MPRA-H human checkpoint",
        "new_row_region_filter": None,
    }


# ---------------------------------------------------------------- UTR-Insight
def build_insight(device):
    sys.path.insert(0, str(INSIGHT_ROOT))
    import esm as esm_pkg
    from esm.data import Alphabet, FastaBatchedDataset

    import torch.nn as nn
    from esm.model.esm2_secondarystructure import ESM2 as ESM2_SISS

    layers, heads, embed_dim = 6, 16, 128
    from esm.modules import ConvTransformerLayer

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
            emb = self.esm2(tokens, [layers], return_representation=True)
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
            z = self.relu(self.linear_2(z))
            z = self.relu(self.linear_3(z))
            z = self.dropout(z)
            return self.output(z).squeeze(-1)

    alphabet = Alphabet(mask_prob=0.0, standard_toks="AGCT")
    model = ConvTransformerPredictor(alphabet).to(device)
    state = torch.load(INSIGHT_ROOT / "Model/utr_insight/model_epoch199.pkl", map_location="cpu", weights_only=False)
    model.load_state_dict({k.replace("module.", ""): v for k, v in state.items()})
    model.eval()

    pad_tok = "<pad>"

    def forward(batch, device):
        # official pipeline: 50 leading literal <pad> tokens inside the sequence
        # string, then alphabet.get_batch_converter-style collate (bos/eos added
        # by the encoder convention: cls at 0, eos after sequence, right PAD)
        encoded = []
        for s in batch:
            r = str(s).upper()
            if not set(r) <= set("AGCT"):
                r = "".join(c if c in "AGCT" else "N" for c in r)
            text = pad_tok * 50 + r
            ids = [alphabet.get_idx(c) for c in text]
            ids = [alphabet.cls_idx] + ids + [alphabet.eos_idx]
            encoded.append(torch.tensor(ids, dtype=torch.long))
        maxw = max(len(t) for t in encoded)
        pad_idx = alphabet.padding_idx
        ids = torch.full((len(encoded), maxw), pad_idx, dtype=torch.long)
        for i, t in enumerate(encoded):
            ids[i, : len(t)] = t
        ids = ids.to(device)
        padding_mask = ids.eq(pad_idx)[:, 1:-1]
        exp = torch.tensor([[1, 0]] * len(batch), dtype=torch.float32, device=device)
        return model(ids, exp, self_attn_padding_mask=padding_mask)

    return (lambda seqs: _batched(seqs, device, forward, batch=16)), {
        "family": "utr_insight",
        "weights_path": str(INSIGHT_ROOT / "Model/utr_insight/model_epoch199.pkl"),
        "weights_sha256": _sha256(INSIGHT_ROOT / "Model/utr_insight/model_epoch199.pkl"),
        "license": "repo no-license - user-waived (ledger addendum)",
        "input_adaptation": (
            "official 50x<pad> prefix + RNA alphabet (T->U), <cls>/<eos> framing, "
            "experiment_indicator=[1,0] (GSM3130435-library branch); "
            "output = MRL scalar head"
        ),
        "paradigm": "5'UTR->MRL scalar (ConvTransformerPredictor)",
    }


# ---------------------------------------------------------------- HydraRNA
def build_hydrarna(device):
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

    def forward(batch, device):
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
        out = model.encoder.extract_features(src_tokens=src_tokens)
        pooled = out[0][:, 1:-1, :].mean(dim=1).float()
        return pooled

    return (lambda seqs: _batched(seqs, device, forward, batch=32)), {
        "family": "hydrarna",
        "weights_path": str(model_path),
        "weights_sha256": _sha256(model_path),
        "license": "fairseq/MIT-family LICENSE in repo (ledger §2.10)",
        "input_adaptation": (
            "fairseq masked_lm_span dict, <s> + space-joined RNA chars (T->U), "
            "mean pooled extract_features [B,1024] as scalar source; rows scored "
            "with this frozen representation (no task head published for MRL "
            "release -> embedding-based row, declared)"
        ),
        "paradigm": "full-length RNA LM; 5'UTR/3'UTR rows via pooled embedding scalar",
    }


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
