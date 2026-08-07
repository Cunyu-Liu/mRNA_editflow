"""M4 SparseEditFormer architecture / encoding unit tests (pure, no remote data)."""
import copy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.m4_sparse import config as C  # noqa: E402
from scripts.m4_sparse.dataset import (  # noqa: E402
    EffectDataset, IncompatibleEditEncodingError, UNKNOWN_CONTEXT, build_vocab,
    edit_features, encode_pair, invert_edits, one_hot,
)
from scripts.m4_sparse.model import SparseEditFormer  # noqa: E402
from scripts.m4_sparse.checkpoint import (  # noqa: E402
    IncompatibleCheckpointError, build_checkpoint_payload, load_checkpoint,
    validate_checkpoint_payload, vocab_sha256,
)
from scripts.m4_sparse.train import (  # noqa: E402
    GroupBatchSampler, _pairwise_rank_loss, build_folds, compute_loss,
    grouped_train_dev_split,
)


def _rec(delta, study="S1", endpoint="ep_rl", benchmark="5U-A1", source_id="src1"):
    s = "ACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGU"
    c = s[:10] + "C" + s[11:]
    return {
        "pair_id": "p", "record_id": "r", "study": study, "benchmark": benchmark,
        "source_sequence": s, "candidate_sequence": c,
        "edit_list": [{"op": "SUB", "pos": 10, "token": "C"}], "edit_count": 1,
        "source_id": source_id, "candidate_id": "c", "endpoint": endpoint,
        "context_id": "ctx",
        "candidate_value": float(delta) + 0.5, "source_value": 0.5,
        "delta": float(delta), "delta_source_status": "derived",
    }


def _cfg(rows):
    cfg = C.get_config()
    vocab = build_vocab(rows)
    cfg.N_STUDIES = len(vocab["study"])
    cfg.N_ENDPOINTS = len(vocab["endpoint"])
    cfg.N_BENCHMARKS = len(vocab["benchmark"])
    cfg.STUDY_UNK_ID = vocab["study"][UNKNOWN_CONTEXT]
    cfg.ENDPOINT_UNK_ID = vocab["endpoint"][UNKNOWN_CONTEXT]
    return cfg, vocab


def _rows(n=16):
    return [_rec((i % 5) - 2.0, study="S%d" % (i % 3), source_id="src%d" % i) for i in range(n)]


def test_m4_runner_direct_cli_help_resolves_repo_imports():
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts/m4_sparse/run.py"), "--help"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--out-dir" in result.stdout


def test_m4_runner_refuses_protected_output_before_dataset_read(tmp_path):
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts/m4_sparse/run.py"),
         "--dataset", str(tmp_path / "missing.jsonl"),
         "--out-dir", str(tmp_path / "restricted_output")],
        cwd=REPO_ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode != 0
    assert "ordinary-only ordinary output refuses protected" in result.stderr


# ---- one-hot / edit features ----
def test_one_hot_shape_and_t2u():
    a = one_hot("ACGU", max_len=100)
    assert a.shape == (100, 4)
    assert a[0].tolist() == [1.0, 0.0, 0.0, 0.0]  # A
    b = one_hot("ACGT", max_len=100)  # T -> U
    assert b[3].tolist() == [0.0, 0.0, 0.0, 1.0]  # U


def test_edit_features_empty_and_dim():
    assert edit_features([], 100).shape == (12,)
    assert np.allclose(edit_features([], 100), np.zeros(12))
    f = edit_features([{"op": "SUB", "pos": 10, "token": "G"}], 100)
    assert f.shape == (12,)
    assert f[0] == 1  # n_edits


def test_invert_edits_sub_uses_source_nucleotide():
    src = "ACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGU"
    cand = src[:10] + "G" + src[11:]
    inv = invert_edits([{"op": "SUB", "pos": 10, "token": "G"}], src, cand)
    assert inv == [{"op": "SUB", "pos": 10, "token": "G"}]  # src[10] == "G"


def test_invert_edits_ins_del_swap():
    inv_ins = invert_edits([{"op": "INS", "pos": 3, "token": "G"}], "ACGUAC", "ACGGUAC")
    assert inv_ins[0]["op"] == "DEL" and inv_ins[0]["pos"] == 3
    inv_del = invert_edits([{"op": "DEL", "pos": 3, "token": "G"}], "ACGUAC", "ACUAC")
    assert inv_del[0]["op"] == "INS"


def test_edit_centred_window_exposes_edit_beyond_first_100():
    src = "A" * 180
    cand = src[:150] + "C" + src[151:]
    encoded = encode_pair(src, cand,
                          [{"op": "SUB", "pos": 150, "token": "C"}],
                          max_len=100)
    src_x, cand_x, src_pad, cand_pad, start = encoded
    local_pos = 150 - start
    assert start > 0
    assert 0 <= local_pos < 100
    assert src_x[local_pos].tolist() == [1.0, 0.0, 0.0, 0.0]
    assert cand_x[local_pos].tolist() == [0.0, 1.0, 0.0, 0.0]
    assert not src_pad.any() and not cand_pad.any()


def test_shared_window_fails_closed_when_edit_span_is_too_wide():
    src = "A" * 220
    cand = "C" + src[1:180] + "G" + src[181:]
    edits = [
        {"op": "SUB", "pos": 0, "token": "C"},
        {"op": "SUB", "pos": 180, "token": "G"},
    ]
    with pytest.raises(IncompatibleEditEncodingError, match="cannot represent edit span"):
        encode_pair(src, cand, edits, max_len=100)


@pytest.mark.parametrize("bad_edits", [
    [{"op": "INS", "pos": 2, "token": "C"}],
    [{"op": "DEL", "pos": 2, "token": "C"}],
])
def test_shared_window_rejects_indel_ops_even_for_short_equal_length_sequences(
        bad_edits):
    with pytest.raises(IncompatibleEditEncodingError, match="SUB edits only"):
        encode_pair("AAAA", "AACA", bad_edits, max_len=100)


def test_shared_window_rejects_incomplete_or_noop_edit_scripts():
    with pytest.raises(IncompatibleEditEncodingError, match="does not exactly match"):
        encode_pair("AAAA", "AACA", [], max_len=100)
    with pytest.raises(IncompatibleEditEncodingError, match="no-op SUB"):
        encode_pair("AAAA", "AAAA",
                    [{"op": "SUB", "pos": 2, "token": "A"}], max_len=100)


def test_dataset_separates_effect_labels_and_rank_groups():
    rows = [_rec(1.0, source_id="shared"), _rec(-1.0, source_id="shared")]
    rows[1]["candidate_sequence"] = rows[1]["source_sequence"][:11] + "A" + rows[1]["source_sequence"][12:]
    rows[1]["edit_list"] = [{"op": "SUB", "pos": 11, "token": "A"}]
    cfg, vocab = _cfg(rows)
    ds = EffectDataset(rows, vocab, target="candidate_value")
    a, b = ds[0], ds[1]
    assert a["candidate_value"].item() == pytest.approx(1.5)
    assert a["source_value"].item() == pytest.approx(0.5)
    assert a["delta"].item() == pytest.approx(1.0)
    assert a["sign"].item() == 1.0
    assert b["sign"].item() == 0.0
    assert a["source_group"].item() == b["source_group"].item()
    assert a["y"].item() == pytest.approx(a["candidate_value"].item())


def test_rank_groups_do_not_cross_assay_contexts():
    rows = [_rec(1.0, source_id="shared"), _rec(-1.0, source_id="shared")]
    rows[1]["context_id"] = "different_context"
    rows[1]["candidate_sequence"] = (
        rows[1]["source_sequence"][:11] + "A" + rows[1]["source_sequence"][12:])
    rows[1]["edit_list"] = [{"op": "SUB", "pos": 11, "token": "A"}]
    _, vocab = _cfg(rows)
    ds = EffectDataset(rows, vocab)
    assert ds[0]["source_group"].item() != ds[1]["source_group"].item()
    assert not ds[0]["rank_group_eligible"].item()
    assert not ds[1]["rank_group_eligible"].item()


def test_vocab_has_reserved_unknown_and_s4_dataset_uses_it():
    rows = _rows()
    _, vocab = _cfg(rows)
    assert vocab["study"][UNKNOWN_CONTEXT] == len(vocab["study"]) - 1
    ds = EffectDataset(rows[:2], vocab, force_unk_study=True)
    assert {ds[i]["study"].item() for i in range(len(ds))} == {
        vocab["study"][UNKNOWN_CONTEXT]
    }


def test_dataset_rejects_legacy_vocab_without_reserved_unknown():
    row = _rec(1.0)
    legacy = {"study": {"S1": 0}, "endpoint": {"ep_rl": 0},
              "benchmark": {"5U-A1": 0}}
    with pytest.raises(ValueError, match="reserved __UNK__"):
        EffectDataset([row], legacy)


def test_padding_masks_mark_only_unfilled_suffix():
    row = _rec(1.0)
    row["source_sequence"] = "ACGN"
    row["candidate_sequence"] = "ACGU"
    row["edit_list"] = [{"op": "SUB", "pos": 3, "token": "U"}]
    _, vocab = _cfg([row])
    x = EffectDataset([row], vocab, max_len=8)[0]
    assert x["src_pad_mask"].tolist() == [False, False, False, False,
                                            True, True, True, True]
    assert x["cand_pad_mask"].tolist() == [False, False, False, False,
                                             True, True, True, True]


# ---- model forward ----
def test_model_forward_shapes():
    rows = _rows()
    cfg, vocab = _cfg(rows)
    model = SparseEditFormer(cfg)
    ds = EffectDataset(rows, vocab, target="delta")
    x = ds[0]
    out = model(x["src"].unsqueeze(0), x["cand"].unsqueeze(0), x["edit"].unsqueeze(0),
                x["study"].unsqueeze(0), x["endpoint"].unsqueeze(0),
                x["bench"].unsqueeze(0),
                src_pad_mask=x["src_pad_mask"].unsqueeze(0),
                cand_pad_mask=x["cand_pad_mask"].unsqueeze(0))
    assert out["mean"].shape == (1,)
    assert out["logvar"].shape == (1,)
    assert out["sign"].shape == (1,)
    assert out["rank"].shape == (1,)


def test_model_loss_decreases_when_overfitting():
    rows = _rows(64)
    cfg, vocab = _cfg(rows)
    model = SparseEditFormer(cfg)
    torch.manual_seed(0)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    ds = EffectDataset(rows, vocab, target="delta")
    from torch.utils.data import DataLoader
    dl = DataLoader(ds, batch_size=16, shuffle=True)
    # Mean loss over whole data passes (robust to batch sampling noise).
    def epoch_loss():
        model.train()
        tot, nb = 0.0, 0
        for b in dl:
            bb = {k: v for k, v in b.items()}
            opt.zero_grad()
            losses = compute_loss(model, bb, cfg, torch.device("cpu"))
            losses["loss"].backward()
            opt.step()
            tot += losses["loss"].item()
            nb += 1
        return tot / max(nb, 1)
    first = epoch_loss()
    last = None
    for _ in range(3):
        last = epoch_loss()
    assert last < first, "loss did not decrease on tiny overfit set"

def test_pairwise_rank_loss_zero_for_perfect_order():
    score = torch.tensor([1.0, 2.0, 3.0, 4.0])
    target = torch.tensor([1.0, 2.0, 3.0, 4.0])
    torch.manual_seed(0)
    loss = _pairwise_rank_loss(score, target, 0.5, torch.device("cpu"))
    assert float(loss) == pytest.approx(0.0)


def test_pairwise_rank_loss_never_crosses_source_groups():
    score = torch.tensor([0.0, 0.0])
    target = torch.tensor([-1.0, 1.0])
    separate = torch.tensor([0, 1])
    together = torch.tensor([0, 0])
    assert _pairwise_rank_loss(score, target, 0.5, torch.device("cpu"),
                               groups=separate).item() == 0.0
    assert _pairwise_rank_loss(score, target, 0.5, torch.device("cpu"),
                               groups=together).item() == pytest.approx(0.5)


def test_candidate_value_loss_uses_delta_sign_and_disables_inverse():
    rows = [_rec(1.0, source_id="a"), _rec(-1.0, source_id="b")]
    rows[0]["source_value"], rows[0]["candidate_value"] = 2.0, 3.0
    rows[1]["source_value"], rows[1]["candidate_value"] = 2.0, 1.0
    cfg, vocab = _cfg(rows)
    cfg.TARGET = "candidate_value"
    cfg.CONTEXT_DROPOUT = 0.0

    class RecordingModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, src, cand, edit, study, endpoint, bench, **kwargs):
            self.calls += 1
            z = src[:, 0, 0] * 0.0
            logits = torch.tensor([10.0, -10.0], device=src.device)[:src.size(0)]
            return {"mean": z, "logvar": z, "sign": logits, "rank": z}

    from torch.utils.data import DataLoader
    batch = next(iter(DataLoader(EffectDataset(rows, vocab,
                                               target="candidate_value"),
                                 batch_size=2)))
    model = RecordingModel()
    losses = compute_loss(model, batch, cfg, torch.device("cpu"))
    assert losses["bce"].item() < 1e-3
    assert losses["consis"].item() == 0.0
    assert model.calls == 1


def test_variance_nll_does_not_duplicate_mean_huber_gradient():
    rows = [_rec(1.0, source_id="a")]
    cfg, vocab = _cfg(rows)
    cfg.TARGET = "candidate_value"
    cfg.CONTEXT_DROPOUT = 0.0
    cfg.LAMBDA_MEAN = 1.0
    cfg.LAMBDA_VAR = 0.5
    cfg.LAMBDA_SIGN = 0.0
    cfg.LAMBDA_CONSISTENCY = 0.0
    cfg.LAMBDA_RANK = 0.0

    class ScalarMeanModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mean = torch.nn.Parameter(torch.tensor(0.0))
            self.logvar = torch.nn.Parameter(torch.tensor(0.0))

        def forward(self, src, cand, edit, study, endpoint, bench, **kwargs):
            mean = self.mean.expand(src.size(0))
            logvar = self.logvar.expand(src.size(0))
            zero = mean * 0.0
            return {"mean": mean, "logvar": logvar, "sign": zero, "rank": zero}

    from torch.utils.data import DataLoader
    batch = next(iter(DataLoader(EffectDataset(rows, vocab,
                                               target="candidate_value"),
                                 batch_size=1)))
    model = ScalarMeanModel()
    losses = compute_loss(model, batch, cfg, torch.device("cpu"))
    losses["loss"].backward()
    # y=1.5 and Huber delta=1 -> dL/dmean=-1.  The variance NLL must not add a
    # second mean-regression gradient on top of that robust objective.
    assert model.mean.grad.item() == pytest.approx(-1.0)
    assert "huber" in losses and torch.isfinite(losses["nll"])


def test_context_dropout_routes_training_calls_to_reserved_unk():
    rows = _rows(4)
    cfg, vocab = _cfg(rows)
    cfg.CONTEXT_DROPOUT = 1.0

    class RecordingModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.seen = []

        def forward(self, src, cand, edit, study, endpoint, bench, **kwargs):
            self.seen.append((study.detach().clone(), endpoint.detach().clone()))
            z = src[:, 0, 0] * 0.0
            return {"mean": z, "logvar": z, "sign": z, "rank": z}

    from torch.utils.data import DataLoader
    batch = next(iter(DataLoader(EffectDataset(rows, vocab), batch_size=4)))
    model = RecordingModel()
    compute_loss(model, batch, cfg, torch.device("cpu"))
    for study, endpoint in model.seen:
        assert torch.all(study == vocab["study"][UNKNOWN_CONTEXT])
        assert torch.all(endpoint == vocab["endpoint"][UNKNOWN_CONTEXT])


# ---- grouped split / batch supervision ----
def test_grouped_train_dev_split_keeps_source_alias_component_together():
    rows = [
        {**_rec(1.0, source_id="id_a"), "record_id": "a1"},
        {**_rec(2.0, source_id="id_a"), "record_id": "a2",
         "source_sequence": "G" * 72},
        {**_rec(3.0, source_id="id_b"), "record_id": "b1",
         "source_sequence": "G" * 72},
        {**_rec(4.0, source_id="id_c"), "record_id": "c1",
         "source_sequence": "U" * 72},
        {**_rec(5.0, source_id="id_d"), "record_id": "d1",
         "source_sequence": "C" * 72},
    ]
    train, dev = grouped_train_dev_split(rows, dev_frac=0.34, seed=7)
    location = {
        row["record_id"]: "train" for row in train
    } | {
        row["record_id"]: "dev" for row in dev
    }
    # id_a links a1<->a2; the shared sequence then links a2<->b1.
    assert {location[name] for name in ("a1", "a2", "b1")} == {
        location["a1"]
    }
    assert train and dev


def test_group_batch_sampler_never_splits_oversized_rank_group():
    group_ids = np.asarray([0] * 130 + [1] * 3 + [2] * 2)
    batches = list(GroupBatchSampler(group_ids, batch_size=64, seed=3))
    for group_id in np.unique(group_ids):
        expected = set(np.flatnonzero(group_ids == group_id).tolist())
        containing = [set(batch) for batch in batches if expected & set(batch)]
        assert len(containing) == 1
        assert expected <= containing[0]


# ---- versioned checkpoint boundary ----
def _r2_checkpoint_payload():
    rows = [_rec(1.0)]
    cfg, vocab = _cfg(rows)
    cfg.SIGN_DECISION_RULE = "sign_head_logit_ge_0_r2_prospective_v1"
    cfg.SIGN_TEMPERATURE = 1.25
    cfg.SIGN_CALIBRATION_METHOD = "temperature_scaling_grouped_dev_v1"
    model = SparseEditFormer(cfg)
    payload = build_checkpoint_payload(
        model.state_dict(), cfg, held_out_study="held", vocab=vocab)
    return payload, vocab


def test_r2_checkpoint_roundtrip_binds_schema_and_vocab_hash(tmp_path):
    payload, vocab = _r2_checkpoint_payload()
    assert payload["vocab_sha256"] == vocab_sha256(vocab)
    path = tmp_path / "r2.pt"
    torch.save(payload, path)
    loaded = load_checkpoint(path, torch.device("cpu"), expected_vocab=vocab)
    assert loaded["checkpoint_schema_version"] == C.CHECKPOINT_SCHEMA_VERSION
    assert loaded["input_schema_version"] == C.INPUT_SCHEMA_VERSION


def test_legacy_checkpoint_and_schema_drift_are_rejected():
    payload, _ = _r2_checkpoint_payload()
    with pytest.raises(IncompatibleCheckpointError, match="legacy checkpoint"):
        validate_checkpoint_payload({"state_dict": payload["state_dict"],
                                     "cfg": payload["cfg"]})
    drift = copy.deepcopy(payload)
    drift["input_schema_version"] = "legacy_prefix_window"
    with pytest.raises(IncompatibleCheckpointError, match="input schema"):
        validate_checkpoint_payload(drift)


def test_checkpoint_rejects_vocab_hash_tampering_and_runtime_vocab_drift():
    payload, vocab = _r2_checkpoint_payload()
    tampered = copy.deepcopy(payload)
    tampered["vocab"]["study"]["injected"] = len(tampered["vocab"]["study"])
    with pytest.raises(IncompatibleCheckpointError, match="vocabulary hash"):
        validate_checkpoint_payload(tampered)

    runtime_vocab = copy.deepcopy(vocab)
    runtime_vocab["study"] = {"different": 0, UNKNOWN_CONTEXT: 1}
    with pytest.raises(IncompatibleCheckpointError, match="runtime M4 vocabulary"):
        validate_checkpoint_payload(payload, expected_vocab=runtime_vocab)


# ---- inverse consistency helper ----
def test_inverse_consistency_antisymmetry_term_is_finite():
    rows = _rows(16)
    cfg, vocab = _cfg(rows)
    model = SparseEditFormer(cfg)
    ds = EffectDataset(rows, vocab, target="delta")
    from torch.utils.data import DataLoader
    batch = next(iter(DataLoader(ds, batch_size=8)))
    b = {k: v for k, v in batch.items()}
    out = model(b["src"], b["cand"], b["edit"], b["study"], b["endpoint"],
                b["bench"], src_pad_mask=b["src_pad_mask"],
                cand_pad_mask=b["cand_pad_mask"])
    out_inv = model(b["cand"], b["src"], b["inv_edit"], b["study"],
                    b["endpoint"], b["bench"],
                    src_pad_mask=b["cand_pad_mask"],
                    cand_pad_mask=b["src_pad_mask"])
    consis = torch.mean((out["mean"] + out_inv["mean"]) ** 2)
    assert torch.isfinite(consis)
