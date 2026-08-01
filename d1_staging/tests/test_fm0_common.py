"""FM0-01 tests: shared fixtures + offline env + tokenizer/model load + forward.

These tests require the actual UTR-LM checkpoint in the HF cache. They are
marked @pytest.mark.fm0_real; skip if HF_HUB_OFFLINE_CACHE_AVAILABLE is unset
or snapshot is missing.

Run on server:
    cd /home/cunyuliu/mrna_editflow_goal/mrna_editflow
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    /home/cunyuliu/miniconda3/envs/pc_cng/bin/python -m pytest \
        d1_staging/tests/test_fm0_common.py -v
"""

import os
import sys
from pathlib import Path

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
FM0_SCRIPTS = os.path.join(HERE, "..", "scripts", "fm0")
sys.path.insert(0, FM0_SCRIPTS)

from fm0_common import (  # noqa: E402
    CONFIG_PATH,
    ensure_offline_env,
    get_snapshot_dir,
    load_config,
    load_config_obj,
    load_model,
    load_tokenizer,
    pool_embeddings,
    summarize_model,
    tokenize_sequences,
)


# ---------------------------------------------------------------------------
# Skip machinery: real-checkpoint tests only run when the snapshot is present.
# ---------------------------------------------------------------------------
def _snapshot_available() -> bool:
    try:
        ensure_offline_env()
        get_snapshot_dir()
        return True
    except Exception:
        return False


SNAPSHOT_OK = _snapshot_available()
fm0_real = pytest.mark.skipif(
    not SNAPSHOT_OK,
    reason="UTR-LM snapshot not in HF cache; set HF_HUB_OFFLINE=1 on server.",
)


# ---------------------------------------------------------------------------
# Config / snapshot tests (always run, even without checkpoint)
# ---------------------------------------------------------------------------

def test_config_yaml_exists():
    assert CONFIG_PATH.exists(), f"missing config: {CONFIG_PATH}"


def test_config_yaml_loads():
    cfg = load_config()
    assert cfg["meta"]["task_id"] == "FM0-01"
    assert cfg["meta"]["frozen"] is True
    assert cfg["model"]["model_id"] == "multimolecule/utrlm-mrl"


def test_config_yaml_required_fields():
    cfg = load_config()
    # Spot-check critical fields
    assert cfg["model"]["num_hidden_layers"] == 6
    assert cfg["model"]["hidden_size"] == 128
    assert cfg["model"]["vocab_size"] == 28
    assert cfg["model"]["max_position_embeddings"] == 1026
    assert cfg["license"]["type"] == "agpl-3.0"
    assert cfg["gpu"]["require_cuda"] is True
    # H7 arms
    arms = cfg["adaptation"]
    for arm in ["frozen", "lora", "partial_unfreeze", "from_scratch"]:
        assert arms[arm]["enabled"] is True, f"arm {arm} not enabled"
    # Exposure
    assert "GSE114002" in cfg["exposure"]["historically_exposed_accessions"]
    assert cfg["exposure"]["labels_exposed"] is False


# ---------------------------------------------------------------------------
# Real-checkpoint tests
# ---------------------------------------------------------------------------

@fm0_real
def test_tokenizer_loads():
    tok = load_tokenizer()
    assert tok.vocab_size == 28
    assert tok.special_tokens_map["bos_token"] == "<cls>"
    assert tok.special_tokens_map["eos_token"] == "<eos>"


@fm0_real
def test_tokenizer_T_to_U_conversion():
    tok = load_tokenizer()
    enc = tokenize_sequences(["ACGT"], tok, padding=False)
    ids = enc["input_ids"].tolist()[0]
    # BOS + A C G U + EOS  (T converted to U)
    assert len(ids) == 6
    decoded = tok.decode(ids, skip_special_tokens=True)
    assert "T" not in decoded
    assert "U" in decoded


@fm0_real
def test_config_obj_matches_yaml():
    cfg = load_config()
    config_obj = load_config_obj()
    assert config_obj.model_type == cfg["model"]["model_type"]
    assert config_obj.hidden_size == cfg["model"]["hidden_size"]
    assert config_obj.num_hidden_layers == cfg["model"]["num_hidden_layers"]
    assert config_obj.num_attention_heads == cfg["model"]["num_attention_heads"]
    assert config_obj.intermediate_size == cfg["model"]["intermediate_size"]
    assert config_obj.vocab_size == cfg["model"]["vocab_size"]
    assert config_obj.max_position_embeddings == cfg["model"]["max_position_embeddings"]


@fm0_real
def test_model_loads_on_cpu():
    model = load_model(device="cpu")
    s = summarize_model(model)
    cfg = load_config()
    assert s["num_parameters_total"] == cfg["model"]["num_parameters"]
    assert s["dtype"] == "torch.float32"
    assert s["device"] == "cpu"


@fm0_real
def test_forward_smoke_cpu():
    import torch
    tok = load_tokenizer()
    model = load_model(device="cpu")
    enc = tokenize_sequences(["ACGUACGUAC"], tok)
    with torch.no_grad():
        out = model(**enc)
    h = out.last_hidden_state
    assert h.shape == (1, enc["input_ids"].shape[1], 128)
    assert not torch.isnan(h).any()
    assert not torch.isinf(h).any()


@fm0_real
def test_pooling_modes_produce_correct_shape():
    import torch
    tok = load_tokenizer()
    model = load_model(device="cpu")
    enc = tokenize_sequences(["ACGUACGU", "GCCAUUACGGCC"], tok)
    with torch.no_grad():
        out = model(**enc)
    h = out.last_hidden_state
    mask = enc["attention_mask"]
    for mode in ["cls", "mean", "max"]:
        v = pool_embeddings(h, mask, mode=mode)
        assert v.shape == (2, 128), f"mode={mode} shape={v.shape}"
        assert not torch.isnan(v).any()


@fm0_real
def test_offline_env_set():
    ensure_offline_env()
    assert os.environ.get("HF_HUB_OFFLINE") == "1"
    assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"
