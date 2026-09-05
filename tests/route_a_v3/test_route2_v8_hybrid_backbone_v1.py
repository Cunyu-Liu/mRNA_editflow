"""Unit tests for core/route2_v8_hybrid_backbone_v1.py (V8 Stage 1 arms S/H)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.route2_v8_hybrid_backbone_v1 import (  # noqa: E402
    CNNMotifStem,
    NUM_DOMAINS,
    V8JointRegressor,
    nucleotide_one_hot,
    parameter_report,
    verify_vocab_alignment,
)

MRNABERT_PATH = Path(
    "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/external_model_assets/"
    "mrnabert_a1eb7df25804d23f08646e1cb996b234d7208a40"
)
requires_assets = pytest.mark.skipif(not MRNABERT_PATH.exists(), reason="mRNABERT assets not mounted")


def build_tiny_base():
    """2-layer/64d instance of the pinned custom BertModel (same code path)."""
    from transformers import AutoConfig, AutoModel

    config = AutoConfig.from_pretrained(MRNABERT_PATH, local_files_only=True, trust_remote_code=True)
    config.num_hidden_layers = 2
    config.hidden_size = 64
    config.intermediate_size = 128
    config.num_attention_heads = 4
    base = AutoModel.from_config(config, trust_remote_code=True, add_pooling_layer=False)
    modeling_module = sys.modules[base.__class__.__module__]
    modeling_module.flash_attn_qkvpacked_func = None
    return base


def _sample_ids(batch: int = 3, length: int = 52) -> torch.Tensor:
    """CLS + random ACGT + SEP, token ids from the pinned mRNABERT vocab."""
    torch.manual_seed(0)
    body = torch.randint(5, 9, (batch, length - 2))
    ids = torch.cat([torch.full((batch, 1), 2), body, torch.full((batch, 1), 3)], dim=1)  # CLS=2, SEP=3
    return ids


def test_nucleotide_one_hot_values() -> None:
    ids = torch.tensor([[2, 5, 6, 7, 8, 9, 0, 1]])  # CLS A T C G N PAD UNK
    one_hot = nucleotide_one_hot(ids)
    assert one_hot.shape == (1, 8, 4)
    assert torch.equal(one_hot[0, 1], torch.tensor([1.0, 0.0, 0.0, 0.0]))  # A
    assert torch.equal(one_hot[0, 2], torch.tensor([0.0, 1.0, 0.0, 0.0]))  # T
    assert torch.equal(one_hot[0, 3], torch.tensor([0.0, 0.0, 1.0, 0.0]))  # C
    assert torch.equal(one_hot[0, 4], torch.tensor([0.0, 0.0, 0.0, 1.0]))  # G
    for special in (0, 5, 6, 7):  # CLS, N, PAD, UNK -> zeros
        assert torch.equal(one_hot[0, special], torch.zeros(4))


def test_stem_shape_and_zero_init() -> None:
    stem = CNNMotifStem(output_dim=768)
    ids = _sample_ids(batch=3, length=52)
    out = stem(ids)
    assert out.shape == (3, 52, 768)
    assert torch.count_nonzero(out) == 0, "zero-initialised final projection must make the stem a no-op at init"
    assert stem.parameter_count() == 537_056


def test_stem_masks_special_tokens_after_training() -> None:
    stem = CNNMotifStem(output_dim=768)
    with torch.no_grad():
        stem.proj2.weight.normal_(0.0, 0.02)  # un-zero the projection
        stem.proj2.bias.normal_(0.0, 0.02)
    ids = torch.tensor([[2, 5, 6, 7, 3, 0, 0]])  # CLS A T C SEP PAD PAD
    out = stem(ids)
    assert out.shape == (1, 7, 768)
    for position in (0, 4, 5, 6):  # CLS / SEP / PAD positions stay exactly zero
        assert torch.count_nonzero(out[0, position]) == 0
    for position in (1, 2, 3):  # real nucleotides receive features
        assert torch.count_nonzero(out[0, position]) > 0


def test_stem_padding_invariance() -> None:
    """Stem features at real positions must not depend on right-pad length."""
    stem = CNNMotifStem(output_dim=64)
    with torch.no_grad():
        stem.proj2.weight.normal_(0.0, 0.05)
    ids_short = torch.tensor([[2, 5, 6, 7, 8, 6, 5, 3]])
    ids_padded = torch.cat([ids_short, torch.zeros(1, 10, dtype=torch.long)], dim=1)
    out_short = stem(ids_short)
    out_padded = stem(ids_padded)
    assert torch.allclose(out_short[0, :8], out_padded[0, :8], atol=1e-6)


@requires_assets
def test_regressor_forward_shapes_and_gradients() -> None:
    base = build_tiny_base()
    for arch, use_stem in (("s", False), ("h", True)):
        model = V8JointRegressor(base, use_stem=use_stem, num_domains=NUM_DOMAINS)
        ids = _sample_ids(batch=4, length=52)
        mask = torch.ones_like(ids)
        domains = torch.tensor([0, 1, 2, 0])
        out = model(ids, mask, domains)
        assert out.shape == (4,)
        loss = out.float().pow(2).mean()
        loss.backward()
        assert model.head.weight.grad is not None
        assert model.domain_embeddings.weight.grad is not None
        if use_stem:
            assert model.stem.proj2.weight.grad is not None
        base.zero_grad(set_to_none=True)


@requires_assets
def test_arm_h_equals_arm_s_at_init() -> None:
    """Zero-init stem projection: H must be functionally identical to S at step 0."""
    base = build_tiny_base()
    arm_s = V8JointRegressor(base, use_stem=False, num_domains=NUM_DOMAINS)
    arm_h = V8JointRegressor(base, use_stem=True, num_domains=NUM_DOMAINS)
    arm_h.load_state_dict(arm_s.state_dict(), strict=False)  # shares base/head/domain weights
    arm_s.eval()
    arm_h.eval()
    ids = _sample_ids(batch=4, length=52)
    mask = torch.ones_like(ids)
    domains = torch.tensor([0, 1, 0, 2])
    out_s = arm_s(ids, mask, domains)
    out_h = arm_h(ids, mask, domains)
    assert torch.allclose(out_s, out_h, atol=1e-5), (out_s - out_h).abs().max()


@requires_assets
def test_domain_conditioning_changes_output() -> None:
    base = build_tiny_base()
    model = V8JointRegressor(base, use_stem=True, num_domains=NUM_DOMAINS)
    ids = _sample_ids(batch=2, length=52)
    mask = torch.ones_like(ids)
    out_mrl = model(ids, mask, torch.zeros(2, dtype=torch.long))
    out_polya = model(ids, mask, torch.ones(2, dtype=torch.long))
    assert not torch.allclose(out_mrl, out_polya)


@requires_assets
def test_encode_pooled_matches_manual() -> None:
    base = build_tiny_base()
    model = V8JointRegressor(base, use_stem=False, num_domains=NUM_DOMAINS)
    model.eval()
    ids = _sample_ids(batch=2, length=20)
    mask = torch.ones_like(ids)
    mask[1, 12:] = 0  # right padding on the second row
    ids[1, 12:] = 0
    domains = torch.tensor([0, 1])
    with torch.no_grad():
        pooled = model.encode_pooled(ids, mask, domains)
        hidden = base(input_ids=ids, attention_mask=mask)[0]
        m3 = mask.unsqueeze(-1)
        manual = (hidden * m3).sum(1) / m3.sum(1).clamp(min=1)
        manual = manual + model.domain_embeddings(domains)
    assert torch.allclose(pooled, manual, atol=1e-5)
    # padded positions must not leak into the pooled representation
    with torch.no_grad():
        polluted = hidden.clone()
        polluted[1, 12:] = 12345.0
        manual_polluted = (polluted * m3).sum(1) / m3.sum(1).clamp(min=1) + model.domain_embeddings(domains)
    assert torch.allclose(manual[1], manual_polluted[1], atol=1e-5)


@requires_assets
def test_forward_right_padding_invariance() -> None:
    base = build_tiny_base()
    model = V8JointRegressor(base, use_stem=True, num_domains=NUM_DOMAINS)
    model.eval()
    with torch.no_grad():
        model.stem.proj2.weight.normal_(0.0, 0.02)
    ids = _sample_ids(batch=1, length=20)
    ids_padded = torch.cat([ids, torch.zeros(1, 9, dtype=torch.long)], dim=1)
    mask = torch.ones_like(ids)
    mask_padded = torch.ones_like(ids_padded)
    mask_padded[:, 20:] = 0  # right padding must be attention-masked (tokenizer convention)
    domains = torch.zeros(1, dtype=torch.long)
    with torch.no_grad():
        out = model(ids, mask, domains)
        out_padded = model(ids_padded, mask_padded, domains)
    assert torch.allclose(out, out_padded, atol=1e-4)


@requires_assets
def test_parameter_report_accounting() -> None:
    base = build_tiny_base()
    arm_s = V8JointRegressor(base, use_stem=False, num_domains=NUM_DOMAINS)
    arm_h = V8JointRegressor(base, use_stem=True, num_domains=NUM_DOMAINS)
    report_s = parameter_report(arm_s)
    report_h = parameter_report(arm_h)
    assert report_s["cnn_stem"] == 0
    # tiny base: proj2 is 512->64 instead of 512->768
    expected_stem = 3_168 + 73_856 + (128 * 512 + 512) + (512 * 64 + 64)
    assert report_h["cnn_stem"] == expected_stem == arm_h.stem.parameter_count()
    assert report_h["trainable_total"] - report_s["trainable_total"] == expected_stem
    assert report_h["domain_embeddings"] == NUM_DOMAINS * 64
    assert report_h["linear_head"] == 64 + 1


@requires_assets
def test_verify_vocab_alignment_real_tokenizer() -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MRNABERT_PATH, local_files_only=True)
    verify_vocab_alignment(tokenizer)  # must not raise
    encoded = tokenizer("A T C G N", add_special_tokens=False, return_tensors=None)
    assert encoded["input_ids"] == [5, 6, 7, 8, 9]


def test_build_rejects_unknown_arch() -> None:
    from core.route2_v8_hybrid_backbone_v1 import build_v8_regressor

    with pytest.raises(ValueError):
        build_v8_regressor(MRNABERT_PATH, "x")  # arch validated before any checkpoint I/O
