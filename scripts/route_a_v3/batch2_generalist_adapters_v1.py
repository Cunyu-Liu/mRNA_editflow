#!/usr/bin/env python3
"""Bespoke embedding adapters for the batch-2 generalist models (benchmark v2 amendment v2).

One callable per family, each returning a uniform interface for the frozen-delta probe pipeline:

    build(model_key, device) -> (embed_fn, meta)
    embed_fn(sequences: list[str]) -> torch.Tensor  (N, D) fp32 CUDA mean-pooled embeddings

All backbones are frozen (eval + no grad). Pooling follows the existing frozen-row caliber
(mean over non-special tokens) wherever the tokenizer exposes an attention mask; for models
whose official embedding API is already sequence-level mean-pooled (Orthrus `representation`),
the official API is used verbatim.

Input adaptation declarations (frozen, per port-ledger discipline; refined only in the
declared direction):

  orthrus_4track / orthrus_6track
      official `orthrus_hf.py` (trust_remote_code, Mamba backbone, needs CUDA + mamba-ssm);
      input = model.seq_to_oh(seq) one-hot (A,C,G,T ordering, U->T); embedding =
      model.representation(x, lengths=None, channel_last=True) -> (B, 512) official mean-pool.
  codonfm_80m
      official NVIDIA-Digital-Bio/CodonFM source: `src.models.components.encodon.EnCodon`
      + `src.tokenizer.tokenizer.Tokenizer` (codon 3-mer, seq_type='dna'); weights loaded from
      the safetensors checkpoint with key remap model.* -> *; embedding = last hidden state
      mean over non-special tokens (CLS/SEP from tokenizer mask).
  mrnalm_5utr / mrnalm_3utr
      official Sanofi `OneModel.py` tokenizer recipe replicated verbatim: WordLevel vocab
      ['[PAD]','[UNK]','[CLS]','[SEP]','<MASK>'] + bases (A,U,G,C,N; input T->U), BertProcessing
      specials; BertForMaskedLM backbone from the CDN checkpoint; embedding = mean over
      non-special tokens of the last hidden state.
  calm
      multimolecule re-export (provenance note in the acquisition report): CaLmModel +
      RnaTokenizer, exactly the RNA-FM loading path; mean over non-special tokens.
  lucaone
      official LucaOne src (models.lucaone_gplm.LucaGPLM + Alphabet) with the official
      checkpoint-step17600000; gene input path = gene_seq_replace (A->1,T->2,C->3,G->4,N->5) +
      tokenizer.encode(seq_type='gene'); output = official 'represent' tensor, mean over
      non-special tokens (BOS/EOS excluded via mask lengths).

Discipline: frozen backbones, no gradient, VALIDATION-only usage by the caller, protected
TEST reads = 0 (this module never touches data), no threshold edits. This file is import-only
infrastructure: running it directly executes the per-model smoke test (one forward pass).

Usage (smoke test): python batch2_generalist_adapters_v1.py --models orthrus_4track calm ...
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

HF = Path("/mnt/cunyuliu/hf_home/models")
CODONFM_SRC = Path("/tmp/codonfm_src")           # NVIDIA official repo (cloned 2026-10-03)
MRNALM_SRC = Path("/tmp/mrnalm_src")             # Sanofi official repo (cloned 2026-10-03)
LUCAONE_SRC = HF / "LucaOne-Official/src_official"   # official src tree (contains models/, alphabet, etc.) copied next to ckpt


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"[adapter] {message}")


def _mean_pool(hidden: torch.Tensor, attention: torch.Tensor) -> torch.Tensor:
    keep = attention.bool()
    return (hidden * keep.unsqueeze(-1)).sum(dim=1) / keep.sum(dim=1, keepdim=True).clamp_min(1)


# --------------------------------------------------------------------------- Orthrus
def build_orthrus(track: int, device):
    model_dir = HF / f"antichronology--orthrus-{track}-track/snapshots/main"
    _require(model_dir.is_dir(), f"orthrus-{track}-track dir missing")
    # mamba_ssm is installed but its package __init__ is incompatible with transformers 5.x
    # (it imports legacy generation symbols at import time). Pre-stub the package entry so the
    # real mamba_ssm.modules.* submodules can be imported directly without executing __init__.
    import importlib
    import types
    if "mamba_ssm" not in sys.modules or not hasattr(sys.modules["mamba_ssm"], "__path__"):
        real_spec = importlib.util.find_spec("mamba_ssm")
        _require(real_spec is not None and real_spec.submodule_search_locations,
                 "mamba_ssm package not importable")
        pkg = types.ModuleType("mamba_ssm")
        pkg.__path__ = list(real_spec.submodule_search_locations)
        sys.modules["mamba_ssm"] = pkg

    # transformers 5.x AutoModel.from_pretrained hits an API incompatibility inside
    # orthrus_hf.py (all_tied_weights_keys); constructing the model class directly from the
    # official file and loading the safetensors checkpoint reproduces the official loading
    # path with missing=0/unexpected=0 (verified 2026-10-03).
    sys.path.insert(0, str(model_dir))
    from orthrus_hf import OrthrusPretrainedModel, OrthrusConfig
    from safetensors.torch import load_file

    config = OrthrusConfig.from_pretrained(str(model_dir), local_files_only=True)
    _require(config.n_tracks == track, "track mismatch vs registry")
    model = OrthrusPretrainedModel(config)
    sd = load_file(str(model_dir / "model.safetensors"))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    _require(len(missing) == 0 and len(unexpected) == 0,
             f"orthrus state mismatch: {missing[:3]} / {unexpected[:3]}")
    model = model.to(device).eval()
    model.requires_grad_(False)
    n_params = sum(p.numel() for p in model.parameters())

    def embed(sequences: list[str]) -> torch.Tensor:
        # length-sorted batching not needed: sequences are short (<=164 nt) and Mamba is
        # sequential anyway; one-by-one keeps the official API exactly.
        outs = []
        with torch.no_grad():
            for seq in sequences:
                oh = model.seq_to_oh(seq)  # (L, n_tracks)
                if oh.shape[1] != config.n_tracks:
                    # Official 6-track definition (orthrus/data.py encode_6_track): channels
                    # 5-6 are the CDS-frame track and the splice-site track. Our evaluation
                    # records are isolated UTR fragments (no transcript context), so both
                    # tracks are all zeros — the official encoding of a UTR-only sequence.
                    zeros = torch.zeros((len(seq), config.n_tracks - 4), dtype=torch.float32)
                    oh = torch.cat([oh, zeros], dim=1)
                _require(oh.shape[1] == config.n_tracks,
                         f"one-hot channels {oh.shape[1]} != n_tracks {config.n_tracks}")
                oh = oh.unsqueeze(0).to(device)  # (1, L, C)
                lengths = torch.tensor([oh.shape[1]], device=device)
                rep = model.representation(oh, lengths=lengths, channel_last=True)  # (1, D)
                _require(rep.is_cuda and torch.isfinite(rep).all().item(), "orthrus rep nonfinite")
                outs.append(rep.float().squeeze(0))
        return torch.stack(outs)

    meta = {
        "model_path": str(model_dir),
        "pretrained_parameter_count": n_params,
        "official_api": "direct class construct (AutoModel incompatible w/ transformers 5.x) + seq_to_oh + representation(lengths) mean-pool",
        "input_adaptation": "ACGT one-hot (U->T per official helper); 6-track adds zero CDS-frame + splice tracks (official UTR-only encoding, orthrus/data.py); no chunking",
        "pooling": "official representation() mean-pool",
    }
    return embed, meta


# --------------------------------------------------------------------------- CodonFM
def build_codonfm(device):
    weights = HF / "nvidia--NV-CodonFM-Encodon-80M-v1/snapshots/main"
    _require((weights / "NV-CodonFM-Encodon-80M-v1.safetensors").exists(), "codonfm weights missing")
    _require((CODONFM_SRC / "src").is_dir(), "codonfm official src not cloned (see docstring)")
    # official encodon_layer.py imports apply_chunking_to_forward from
    # transformers.modeling_utils (4.x path); in transformers 5.x it lives in
    # transformers.pytorch_utils — pre-patch the old name so the official file imports cleanly.
    import transformers.modeling_utils as _tm
    if not hasattr(_tm, "apply_chunking_to_forward"):
        from transformers.pytorch_utils import apply_chunking_to_forward as _actf
        _tm.apply_chunking_to_forward = _actf
    # the official mha.py hard-depends on xformers.memory_efficient_attention, which is not
    # installed in this env. PyTorch SDPA (scaled_dot_product_attention) is mathematically
    # identical for a full additive bias; we pre-stub the xformers module with an SDPA-backed
    # reimplementation of memory_efficient_attention so the official code path runs unchanged.
    try:
        import xformers.ops  # noqa: F401
    except ImportError:
        import types as _types
        from torch.nn import functional as _F

        def _mea(query, key, value, op=None, attn_bias=None, p=0.0, **_kw):
            # xformers memory_efficient_attention signature: (B, L, H, D) tensors, additive
            # bias broadcastable to (B, H, Lq, Lk). Convert to SDPA layout (B, H, L, D).
            q = query.transpose(1, 2)
            k = key.transpose(1, 2)
            v = value.transpose(1, 2)
            if attn_bias is not None:
                if attn_bias.dim() == 4 and attn_bias.shape[1] == 1 and q.shape[1] != 1:
                    bias = attn_bias
                else:
                    bias = attn_bias
                # attn_bias arrives as (B, H, Lq, Lk) after the official repeat calls.
                out = _F.scaled_dot_product_attention(q, k, v, attn_mask=bias, dropout_p=p)
            else:
                out = _F.scaled_dot_product_attention(q, k, v, dropout_p=p)
            return out.transpose(1, 2)

        xops = _types.ModuleType("xformers.ops")
        xops.memory_efficient_attention = _mea
        xf = sys.modules.get("xformers") or _types.ModuleType("xformers")
        xf.ops = xops
        sys.modules["xformers"] = xf
        sys.modules["xformers.ops"] = xops
    sys.path.insert(0, str(CODONFM_SRC))

    from src.models.components.encodon import EnCodon
    from src.models.components.encodon_config import EnCodonConfig
    from src.tokenizer.tokenizer import Tokenizer as CodonTokenizer
    from safetensors.torch import load_file

    config = json.load(open(weights / "config.json"))
    enc_cfg = EnCodonConfig(
        vocab_size=config["vocab_size"],
        hidden_size=config["hidden_size"],
        num_hidden_layers=config["num_hidden_layers"],
        num_attention_heads=config["num_attention_heads"],
        intermediate_size=config["intermediate_size"],
        hidden_act=config["hidden_act"],
        hidden_dropout_prob=config["hidden_dropout_prob"],
        attention_probs_dropout_prob=config["attention_probs_dropout_prob"],
        initializer_range=config["initializer_range"],
        layer_norm_eps=config["layer_norm_eps"],
        pad_token_id=config["pad_token_id"],
        rotary_theta=config.get("rotary_theta", 1e5),
        max_position_embeddings=config.get("max_position_embeddings", 2048),
        # the HF config's position_embedding_type=rotary is the EnCodon default (rotary
        # embeddings are built into the official MHA module); no extra arg exists.
    )
    model = EnCodon(enc_cfg)
    sd = load_file(str(weights / "NV-CodonFM-Encodon-80M-v1.safetensors"))
    sd = {k.removeprefix("model."): v for k, v in sd.items() if k.startswith("model.")}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    _require(not [k for k in missing if "cls" not in k], f"codonfm missing encoder keys: {missing[:5]}")
    model = model.to(device).eval()
    model.requires_grad_(False)
    n_params = sum(p.numel() for p in model.parameters())

    tokenizer = CodonTokenizer(seq_type="dna")

    def embed(sequences: list[str]) -> torch.Tensor:
        outs = []
        with torch.no_grad():
            for seq in sequences:
                ids = tokenizer.build_inputs_with_special_tokens(tokenizer.encode(seq))
                # official mha.py asserts sequence length divisible by 8 (Tensor Core
                # constraint in their xformers path; our SDPA shim keeps the assert).
                # Declared adaptation: right-pad the token sequence to the next multiple
                # of 8 with the pad token, mask them out of pooling.
                pad_to = (len(ids) + 7) // 8 * 8
                ids = ids + [tokenizer.pad_token_id] * (pad_to - len(ids))
                input_ids = torch.tensor([ids], dtype=torch.long, device=device)
                n_special = len(tokenizer.special_tokens)
                mask = (input_ids >= n_special).long()
                _require(mask.sum() > 0, "codonfm empty effective mask")
                out = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                            extract_embeddings_only=True)
                hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
                pooled = _mean_pool(hidden, mask)
                _require(pooled.is_cuda and torch.isfinite(pooled).all().item(), "codonfm nonfinite")
                outs.append(pooled.float().squeeze(0))
        return torch.stack(outs)

    meta = {
        "model_path": str(weights),
        "pretrained_parameter_count": n_params,
        "official_source": "github.com/NVIDIA-Digital-Bio/CodonFM (src.models.components.encodon)",
        "input_adaptation": "3-mer codon tokenizer (DNA ACGT), CLS/SEP specials, right-pad to multiple-of-8 (declared, official TensorCore assert), mean over codon tokens",
        "provenance_note": "license:other (NVIDIA Open Model License); HF id carries -v1 suffix",
    }
    return embed, meta


# --------------------------------------------------------------------------- mRNA-LM
def build_mrnalm(region: str, device):
    ckpt = HF / f"Sanofi-Public--mRNA-LM/{region}"
    _require((ckpt / "pytorch_model.bin").exists(), f"mrnalm {region} checkpoint missing")
    from transformers import BertConfig, BertForMaskedLM
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.processors import BertProcessing
    from transformers import PreTrainedTokenizerFast

    # official OneModel.build_tokenizer recipe, verbatim
    lst_ele = list("AUGCN")
    lst_voc = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "<MASK>"]
    lst_voc.extend([b for b in lst_ele])
    dic_voc = dict(zip(lst_voc, range(len(lst_voc))))
    tok = Tokenizer(WordLevel(vocab=dic_voc, unk_token="[UNK]"))
    tok.add_special_tokens(["[PAD]", "[CLS]", "[UNK]", "[SEP]", "<MASK>"])
    tok.pre_tokenizer = Whitespace()
    tok.post_processor = BertProcessing(("[SEP]", dic_voc["[SEP]"]), ("[CLS]", dic_voc["[CLS]"]))
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tok, unk_token="[UNK]", sep_token="[SEP]", pad_token="[PAD]",
        cls_token="[CLS]", mask_token="<MASK>")

    config = BertConfig.from_pretrained(str(ckpt), local_files_only=True)
    model = BertForMaskedLM(config)
    sd = torch.load(ckpt / "pytorch_model.bin", map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    _require(len(missing) == 0, f"mrnalm missing keys: {missing[:5]}")
    model = model.to(device).eval()
    model.requires_grad_(False)
    n_params = sum(p.numel() for p in model.parameters())
    max_length = 512 if region == "5utr" else 1024

    def embed(sequences: list[str]) -> torch.Tensor:
        # official encode_string: T->U then whitespace-split char tokens, CLS/SEP specials.
        # WordLevel vocab + Whitespace pre-tokenizer requires space-joined characters,
        # otherwise the whole sequence maps to a single [UNK] token.
        joined = [" ".join(s.replace("T", "U")) for s in sequences]
        enc = tokenizer(joined, truncation=True,
                        padding=True, max_length=max_length, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attention = enc["attention_mask"].to(device)
        with torch.no_grad():
            hidden = model.bert(input_ids=input_ids, attention_mask=attention).last_hidden_state
            pooled = _mean_pool(hidden, attention)
            _require(pooled.is_cuda and torch.isfinite(pooled).all().item(), "mrnalm nonfinite")
        return pooled.float()

    meta = {
        "model_path": str(ckpt),
        "pretrained_parameter_count": n_params,
        "official_source": "github.com/Sanofi-Public/mRNA-LM (OneModel.py tokenizer recipe, CDN checkpoint)",
        "input_adaptation": f"T->U, char-level WordLevel vocab (A,U,G,C,N), max_length {max_length}",
        "pooling": "mean over non-special tokens (official get_mean_token_embeddings excludes CLS/SEP)",
    }
    return embed, meta


# --------------------------------------------------------------------------- CaLM
def build_calm(device):
    model_dir = HF / "multimolecule--calm/snapshots/main"
    _require(model_dir.is_dir(), "calm dir missing")
    from multimolecule.models import calm as calm_mod

    tokenizer = calm_mod.AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model = calm_mod.AutoModel.from_pretrained(str(model_dir), local_files_only=True).to(device).eval()
    model.requires_grad_(False)
    n_params = sum(p.numel() for p in model.parameters())

    def embed(sequences: list[str]) -> torch.Tensor:
        # CaLM is a *codon* model: its tokenizer requires lengths divisible by 3.
        # Declared adaptation: truncate each sequence to its largest multiple of 3
        # (UTR benchmark sequences are 50-164 nt; at most 2 trailing bases dropped).
        trimmed = [s[: len(s) // 3 * 3] for s in sequences]
        enc = tokenizer([s.replace("T", "U") for s in trimmed], padding=True, return_tensors="pt")
        tokens = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            hidden = model(**tokens).last_hidden_state
            attention = tokens["attention_mask"]
            special = torch.zeros_like(attention, dtype=torch.bool)
            special[:, 0] = True
            special[torch.arange(len(sequences), device=device), attention.sum(dim=1) - 1] = True
            keep = attention.bool() & ~special
            pooled = (hidden * keep.unsqueeze(-1)).sum(dim=1) / keep.sum(dim=1, keepdim=True).clamp_min(1)
            _require(pooled.is_cuda and torch.isfinite(pooled).all().item(), "calm nonfinite")
        return pooled.float()

    meta = {
        "model_path": str(model_dir),
        "pretrained_parameter_count": n_params,
        "official_source": "multimolecule/calm (re-export of oxpig CaLM architecture; OPIG original 403)",
        "input_adaptation": "codon tokenizer (multimolecule re-export): T->U, truncate to multiple-of-3 (<=2 trailing bases; declared), dynamic padding",
        "pooling": "mean over non-special tokens (fp32)",
        "provenance_note": "AGPL-3.0; re-export provenance declared in batch2 acquisition report",
    }
    return embed, meta


# --------------------------------------------------------------------------- LucaOne
def build_lucaone(device):
    ckpt = HF / "LucaOne-Official/checkpoint-step17600000"
    _require((ckpt / "pytorch.pth").exists(), "lucaone official checkpoint missing")
    # transformers 4.26-era helpers used by the official modeling_bert.py were removed in
    # 5.x; supply the two legacy helpers FIRST (official 4.26 implementations) so that the
    # official import chain (`from transformers.pytorch_utils import ...`) resolves.
    import transformers.pytorch_utils as _tpu
    if not hasattr(_tpu, "find_pruneable_heads_and_indices"):
        def _fphai(heads, n_heads, head_size, already_pruned_heads):
            mask = torch.ones(n_heads, head_size)
            heads = set(heads) - already_pruned_heads
            for head in heads:
                mask[head] = 0
            mask = mask.view(-1).contiguous().eq(1)
            index = torch.arange(len(mask))[mask].long()
            return heads, index

        def _prune_linear_layer(layer, index, dim=0):
            layer.weight = torch.nn.Parameter(
                layer.weight.data.index_select(dim, index).clone().detach())
            if layer.bias is not None and dim == 0:
                layer.bias = torch.nn.Parameter(layer.bias.data[index].clone().detach())
            return layer

        _tpu.find_pruneable_heads_and_indices = _fphai
        _tpu.prune_linear_layer = _prune_linear_layer

    # The official modules import each other as `from models.x import ...` / `from src.models.x`.
    # The host repo already has a `models` package (W0 worktree) which shadows the name, so
    # alias both `models` and `src` to the LucaOne src tree before importing.
    import types as _t
    _lucaone_models = _t.ModuleType("models")
    _lucaone_models.__path__ = [str(LUCAONE_SRC / "models")]
    sys.modules["models"] = _lucaone_models
    _src_alias = _t.ModuleType("src")
    _src_alias.__path__ = [str(LUCAONE_SRC)]
    sys.modules["src"] = _src_alias
    # the fallback import style `from src.models.x import ...` also needs a `src.models`
    # submodule alias, otherwise a stale `models` module in sys.modules wins the lookup
    _src_models = _t.ModuleType("src.models")
    _src_models.__path__ = [str(LUCAONE_SRC / "models")]
    sys.modules["src.models"] = _src_models
    if str(LUCAONE_SRC) not in sys.path:
        sys.path.insert(0, str(LUCAONE_SRC))
    from models.lucaone_gplm import LucaGPLM
    from models.lucaone_gplm_config import LucaGPLMConfig
    from models.alphabet import Alphabet

    config = LucaGPLMConfig.from_pretrained(str(ckpt))
    model = LucaGPLM(config)
    sd = torch.load(ckpt / "pytorch.pth", map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    _require(isinstance(sd, dict) and sd, "lucaone checkpoint is not a state dict")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # core encoder keys must all load; only downstream heads may be absent
    core_missing = [k for k in missing if not any(
        tag in k for tag in ("classifier", "lm_head", "contact_head"))]
    _require(not core_missing, f"lucaone missing core keys: {core_missing[:5]}")
    model = model.to(device).eval()
    model.requires_grad_(False)
    n_params = sum(p.numel() for p in model.parameters())
    tokenizer = Alphabet.from_pretrained(str(ckpt / "tokenizer"))

    def _gene_replace(seq: str) -> str:
        table = {"A": "1", "T": "2", "U": "2", "C": "3", "G": "4"}
        return "".join(table.get(ch, "5") for ch in seq)

    def embed(sequences: list[str]) -> torch.Tensor:
        outs = []
        with torch.no_grad():
            for seq in sequences:
                enc = tokenizer.encode(seq_type="gene", seq=_gene_replace(seq))
                ids = torch.tensor([enc], dtype=torch.long, device=device)
                if tokenizer.prepend_bos:
                    ids = torch.cat([torch.tensor([[tokenizer.cls_idx]], device=device), ids], dim=1)
                if tokenizer.append_eos:
                    ids = torch.cat([ids, torch.tensor([[tokenizer.eos_idx]], device=device)], dim=1)
                # non-special mask: everything except pad/bos/eos
                special_ids = {tokenizer.cls_idx, tokenizer.eos_idx, tokenizer.padding_idx}
                keep = torch.ones_like(ids)
                for sid in special_ids:
                    keep[ids == sid] = 0
                _require(keep.sum() > 0, "lucaone empty effective mask")
                out = model(input_ids=ids, repr_layers=[config.num_hidden_layers])
                # LucaGPLM (inference/embedding_inference mode) returns a dict whose
                # 'hidden_states' entry is already the requested-layer (B, L, E) tensor
                # (repr_layers selects the layer; verified against the official forward).
                hidden = out["hidden_states"]  # (B, L, E)
                _require(hidden.dim() == 3 and hidden.shape[0] == 1,
                         f"lucaone hidden shape unexpected: {tuple(hidden.shape)}")
                pooled = _mean_pool(hidden, keep)
                _require(pooled.is_cuda and torch.isfinite(pooled).all().item(), "lucaone nonfinite")
                outs.append(pooled.float().squeeze(0))
        return torch.stack(outs)

    meta = {
        "model_path": str(ckpt),
        "pretrained_parameter_count": n_params,
        "official_source": "github.com/LucaOne/LucaOne src + checkpoint-step17600000 from the official FTP (47.93.21.181)",
        "input_adaptation": "gene path: A->1,T->2,C->3,G->4,N->5 + Alphabet tokenizer, gene token_type",
        "pooling": "mean over non-special tokens of the final layer representation",
        "sha256": "373c1ce812fcd74aaf58a4aa3b0df8418095fa41750c0ec5437752a03f53a46b",
    }
    return embed, meta


# --------------------------------------------------------------------------- registry
REGISTRY = {
    "orthrus_4track": lambda d: build_orthrus(4, d),
    "orthrus_6track": lambda d: build_orthrus(6, d),
    "codonfm_80m": build_codonfm,
    "mrnalm_5utr": lambda d: build_mrnalm("5utr", d),
    "mrnalm_3utr": lambda d: build_mrnalm("3utr", d),
    "calm": build_calm,
    "lucaone": build_lucaone,
}


def smoke_test(model_keys: list[str], device) -> int:
    seqs = [
        "GCUCGUUUCGGGAGCUCGGUUUCGGGAGCUCGGUUUCGGGAGCUCGGUUUCG".replace("U", "T"),
        "ACGTACGTACGTACGTACGTACGTACGTACGTACGT",
    ]
    failures = []
    for key in model_keys:
        try:
            embed, meta = REGISTRY[key](device)
            emb = embed(seqs)
            ok = emb.shape[0] == len(seqs) and emb.is_cuda and bool(torch.isfinite(emb).all())
            print(f"[{key}] OK: {tuple(emb.shape)} finite={bool(torch.isfinite(emb).all())} "
                  f"params={meta['pretrained_parameter_count']:,}", flush=True)
            if not ok:
                failures.append(key)
            del embed
            torch.cuda.empty_cache()
        except Exception as exc:
            print(f"[{key}] FAIL: {exc}", flush=True)
            failures.append(key)
    print("SMOKE", "ALL_PASS" if not failures else f"FAILURES={failures}")
    return 0 if not failures else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=list(REGISTRY), choices=list(REGISTRY))
    parser.add_argument("--physical-gpu-index", type=int, default=5)
    args = parser.parse_args()
    _require(torch.cuda.is_available(), "CUDA unavailable - GPU required")
    device = torch.device(f"cuda:{args.physical_gpu_index}")
    torch.cuda.set_device(device)
    return smoke_test(args.models, device)


if __name__ == "__main__":
    raise SystemExit(main())