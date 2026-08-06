"""G1-X real-mRNA guidance integration unit tests (pure, CPU, no remote data).

Covers the guidance head (GuidanceRatioNet), the guided-flow sampler, the
guidance step policies, and the runner's measured-candidate scoring +
aggregation math.  These are the unit-testable, data-free cores; the full
GPU/real-data orchestration lives in scripts/g1x/run_g1x.py.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.g1x.sampler import (  # noqa: E402
    GuidedFlowSampler, first_step_preference, preference_scores,
)
from scripts.g1x.guidance import (  # noqa: E402
    GuidanceRatioNet, base_step_policy, first_order_step_policy,
    rate_cfg_step_policy, latent_cfg_step_policy, dgm_learned_step_policy,
    POLICY_BUILDERS, _single_edit_rows,
)
from scripts.f0x.flow import (  # noqa: E402
    LegalAction, EditFlowState, build_state, enumerate_legal_actions,
    apply_action,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _state(seq="ACGUACGU", budget=1):
    return build_state(seq, [True] * len(seq), budget)


def _dummy_base_policy(state, actions):
    """Deterministic base policy: uniform over legal actions."""
    return np.zeros(len(actions), dtype=float)


def _dummy_guided_policy(state, actions):
    """Guided policy: put all mass on the first legal action."""
    gl = np.full(len(actions), -1.0, dtype=float)
    gl[0] = 5.0
    return {"guided_logits": gl, "base_logits":
            np.zeros(len(actions), dtype=float), "ratio": None,
            "critic_mean": 0.5, "critic_logvar": -1.0}


# ---------------------------------------------------------------------------
# guidance head
# ---------------------------------------------------------------------------

def test_guidance_head_input_output_shape():
    import torch
    head = GuidanceRatioNet(hidden=8)
    src = torch.zeros(2, 8, 4)  # [B,L,4] one-hot
    out = head(src)
    assert out.shape == (2, 8, 4)


def test_guidance_head_effects_returns_numpy():
    import torch
    head = GuidanceRatioNet(hidden=8)
    head.eval()
    src = torch.zeros(1, 8, 4)
    e = head.effects(src)
    assert isinstance(e, np.ndarray)
    assert e.shape == (1, 8, 4)


def test_single_edit_rows_filters_multiedit():
    rows = [
        {"edit_list": [{"op": "SUB", "pos": 0, "token": "A"}]},
        {"edit_list": [{"op": "SUB", "pos": 0, "token": "A"},
                       {"op": "SUB", "pos": 2, "token": "C"}]},
        {"edit_list": [{"op": "SUB", "pos": 1, "token": "G"}]},
    ]
    kept = _single_edit_rows(rows)
    assert [r for r in kept] == [rows[0], rows[2]]


# ---------------------------------------------------------------------------
# pre-registered policy builders
# ---------------------------------------------------------------------------

def test_policy_builders_are_pre_registered():
    expected = {"no_guidance", "first_order", "rate_cfg",
                "latent_cfg", "dgm_learned"}
    assert expected <= set(POLICY_BUILDERS)


def test_base_step_policy_guided_equals_base():
    st = _state()
    acts = enumerate_legal_actions(st)
    net = _DummyNet()
    pol = base_step_policy(net, None)
    info = pol(st, acts)
    assert np.allclose(info["guided_logits"], info["base_logits"])
    # softmax of uniform logits = uniform
    gl = np.exp(info["guided_logits"] - info["guided_logits"].max())
    assert np.allclose(gl / gl.sum(), np.full(len(acts), 1.0 / len(acts)))


def test_first_order_guidance_shifts_mass_toward_high_effect():
    import torch
    st = _state("ACGU")
    acts = enumerate_legal_actions(st)
    # guidance head that strongly prefers position 0
    class Head:
        def effects(self, vec):
            e = np.zeros((1, 4, 4))
            e[0, 0, :] = [3.0, 3.0, 3.0, 3.0]  # position 0 high effect
            return e
    net = _DummyNet()
    pol = first_order_step_policy(net, Head(), beta=2.0, device=None)
    info = pol(st, acts)
    pos0_idxs = [i for i, a in enumerate(acts) if a.pos == 0]
    guided = np.exp(info["guided_logits"] - info["guided_logits"].max())
    guided = guided / guided.sum()
    assert guided[pos0_idxs].sum() > 0.5  # dominated by high-effect position


def test_rate_cfg_and_latent_cfg_return_dict_contract():
    st = _state("ACGU")
    acts = enumerate_legal_actions(st)
    net = _DummyNet()
    critic = _DummyCritic()
    vocab = {"study": {}, "endpoint": {}, "benchmark": {}}
    ctx = {"study": None, "endpoint": None, "bench": "5U-A1",
           "source_value": 1.0}
    for name, builder in [("rate_cfg", rate_cfg_step_policy),
                          ("latent_cfg", latent_cfg_step_policy)]:
        pol = builder(net, critic, vocab, beta=1.0, device=None, ctx=ctx)
        info = pol(st, acts)
        assert "guided_logits" in info and "base_logits" in info
        assert len(info["guided_logits"]) == len(acts)


def test_dgm_learned_has_no_base_logits():
    st = _state("ACGU")
    acts = enumerate_legal_actions(st)
    critic = _DummyCritic()
    vocab = {"study": {}, "endpoint": {}, "benchmark": {}}
    ctx = {"study": None, "endpoint": None, "bench": "5U-A1",
           "source_value": 1.0}
    pol = dgm_learned_step_policy(critic, vocab, beta=1.0, device=None, ctx=ctx)
    info = pol(st, acts)
    assert info["base_logits"] is None
    assert len(info["guided_logits"]) == len(acts)


# ---------------------------------------------------------------------------
# guided-flow sampler
# ---------------------------------------------------------------------------

def test_sampler_respects_fixed_budget():
    st = _state("ACGUACGU", budget=3)
    samp = GuidedFlowSampler(_dummy_guided_policy, seed=1)
    out = samp.sample(st)
    assert out["n_steps"] == 3
    assert out["budget_remaining"] == 0
    assert out["length"] == out["source_length"] == 8


def test_sampler_records_legality_and_guidance_quantities():
    st = _state("ACGU", budget=1)
    samp = GuidedFlowSampler(_dummy_guided_policy, seed=1)
    out = samp.sample(st)
    assert out["n_steps"] == 1
    t = out["trajectory"][0]
    assert t["legal"] is True
    assert t["guided_prob"] > 0.9  # dominant mass on the chosen action
    assert t["critic_mean"] == 0.5
    assert t["critic_logvar"] == -1.0
    assert t["policy_entropy"] >= 0.0


def test_sampler_reproducible_with_seed():
    st = _state("AAAA", budget=20)
    s1 = GuidedFlowSampler(_dummy_guided_policy, seed=7)
    s2 = GuidedFlowSampler(_dummy_guided_policy, seed=7)
    o1 = s1.sample(st)
    o2 = s2.sample(st)
    assert o1["trajectory"] == o2["trajectory"]
    assert o1["final_seq"] == o2["final_seq"]


def test_sampler_base_prob_null_when_no_base_logits():
    st = _state("AAAA", budget=1)
    def pol(state, actions):
        return {"guided_logits": np.zeros(len(actions)),
                "base_logits": None}
    samp = GuidedFlowSampler(pol, seed=1)
    out = samp.sample(st)
    assert out["trajectory"][0]["base_prob"] is None


# ---------------------------------------------------------------------------
# first-step preference / ranking maps
# ---------------------------------------------------------------------------

def test_preference_scores_maps_by_pos_and_target():
    st = _state("ACGU", budget=1)
    acts = enumerate_legal_actions(st)
    info = preference_scores(_dummy_guided_policy, st)
    assert "guided" in info and "base" in info
    # first action dominant mass at its (pos, target)
    a0 = acts[0]
    assert info["guided"][a0.pos][a0.target_nt] > 0.9


def test_first_step_preference_returns_prob_for_edit():
    st = _state("ACGU", budget=1)
    acts = enumerate_legal_actions(st)
    a0 = acts[0]
    r = first_step_preference(_dummy_guided_policy, st, a0.pos, a0.target_nt)
    assert r["guided_prob"] and r["guided_prob"] > 0.9
    assert r["base_prob"] is not None


def test_first_step_preference_missing_edit_returns_none():
    st = _state("ACGU", budget=1)
    r = first_step_preference(_dummy_guided_policy, st, 99, "X")
    assert r["guided_prob"] is None


# ---------------------------------------------------------------------------
# runner scoring / aggregation (pure math)
# ---------------------------------------------------------------------------

def test_runner_candidate_score_uses_pref_maps():
    """The runner reads guided/base maps from preference_scores, not from the
    step-info dict.  Regression: previously it passed info (which lacks the
    maps) -> all non-critic strategies returned -inf."""
    from scripts.g1x import run_g1x as R
    st = _state("ACGU", budget=1)
    acts = enumerate_legal_actions(st)
    a0 = acts[0]
    rec = {"candidate_sequence": apply_action(st, a0).seq,
           "source_sequence": st.seq,
           "edit_list": [{"op": "SUB", "pos": a0.pos, "token": a0.target_nt}],
           "delta": 1.0, "study": "S1", "endpoint": "ep_x",
           "source_value": 0.5}
    pref = {"guided": {a0.pos: {a0.target_nt: 0.9}},
            "base": None}
    params = {"critic": None, "vocab": None, "bench": "5U-A1"}
    score = R._strategy_candidate_score("first_order", None, st, rec,
                                        params, None, pref)
    assert score == pytest.approx(0.9)


def test_runner_candidate_score_no_guidance_uses_base_map():
    from scripts.g1x import run_g1x as R
    st = _state("ACGU", budget=1)
    acts = enumerate_legal_actions(st)
    a0 = acts[0]
    rec = {"candidate_sequence": apply_action(st, a0).seq,
           "source_sequence": st.seq,
           "edit_list": [{"op": "SUB", "pos": a0.pos, "token": a0.target_nt}],
           "delta": 1.0, "study": "S1", "endpoint": "ep_x",
           "source_value": 0.5}
    pref = {"guided": {a0.pos: {a0.target_nt: 0.9}},
            "base": {a0.pos: {a0.target_nt: 0.4}}}
    params = {"critic": None, "vocab": None, "bench": "5U-A1"}
    score = R._strategy_candidate_score("no_guidance", None, st, rec,
                                        params, None, pref)
    assert score == pytest.approx(0.4)


def test_runner_candidate_score_missing_edit_returns_neginf():
    from scripts.g1x import run_g1x as R
    st = _state("ACGU", budget=1)
    rec = {"candidate_sequence": "ACGU", "source_sequence": st.seq,
           "edit_list": [], "delta": 1.0, "study": "S1",
           "endpoint": "ep_x", "source_value": 0.5}
    pref = {"guided": {}, "base": {}}
    score = R._strategy_candidate_score("first_order", None, st, rec,
                                        {}, None, pref)
    assert score == -np.inf


def test_runner_candidate_edit_extracts_sub():
    from scripts.g1x import run_g1x as R
    rec = {"edit_list": [{"op": "SUB", "pos": 3, "token": "t"}]}  # t->U
    assert R._candidate_edit(rec) == (3, "U")


def test_gtr_builder_policy_returns_dict_contract():
    """generate_then_rerank policy must return a dict (not a plain array) and
    base_logits must equal the base-flow guided logits.  Regression: the base
    step policy returns a dict, so `np.asarray(base_policy(...))` raised
    TypeError; we must read `["guided_logits"]`."""
    from scripts.g1x import run_g1x as R
    st = _state("ACGU", budget=1)
    acts = enumerate_legal_actions(st)
    net = _DummyNet()
    critic = _DummyCritic()
    vocab = {"study": {}, "endpoint": {}, "benchmark": {}}
    ctx = {"study": None, "endpoint": None, "bench": "5U-A1",
           "source_value": 1.0}
    builder = R._gtr_builder(net, None, critic, vocab, beta=1.0,
                             device=torch.device("cpu"), ctx=ctx)
    info = builder(st, acts)
    assert isinstance(info, dict)
    assert "guided_logits" in info and "base_logits" in info
    assert len(info["guided_logits"]) == len(acts)
    # base flow is uniform -> base_logits are all-equal (uniform preference)
    bl = np.asarray(info["base_logits"])
    assert np.allclose(bl, bl[0])


def test_generation_quality_returns_structure_and_respects_budget():
    """evaluate_generation_quality samples guided trajectories and reports the
    critic-judged delta of the final generated sequence over the source.  With a
    uniform base policy and a constant critic, delta is 0 but the structure and
    budget (n_steps == budget) must hold."""
    from scripts.g1x import run_g1x as R

    def simple_builder(base_net, head, critic, vocab, beta, device, ctx=None):
        def policy(state, actions):
            gl = np.zeros(len(actions), dtype=float)
            return {"guided_logits": gl, "base_logits": gl.copy(),
                    "ratio": None, "critic_mean": 0.0, "critic_logvar": -1.0}
        return policy

    builders = {"base": simple_builder, "guided": simple_builder}
    params = {"base_net": _DummyNet(), "head": None, "critic": _DummyCritic(),
              "vocab": {"study": {}, "endpoint": {}, "benchmark": {}},
              "beta": 1.0, "bench": "5U-A1"}
    sources = ["ACGUACGU", "GACGUACU"]
    res = R.evaluate_generation_quality(builders, params, torch.device("cpu"),
                                        sources, budget=2, seed=1)
    assert set(res.keys()) == {"base", "guided"}
    s = res["base"]
    assert s["n_sources"] == 2
    assert s["budget"] == 2
    assert s["mean_delta"] == pytest.approx(0.0)
    assert s["median_delta"] == pytest.approx(0.0)
    assert s["frac_beneficial"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# dummy model stand-ins (CPU, no torch.backends needed)
# ---------------------------------------------------------------------------

class _DummyNet:
    """Minimal stand-in for FlowRateNet.policy_fn used by the base policy."""

    def policy_fn(self, state, actions):
        return _dummy_base_policy(state, actions)


class _DummyCritic:
    """Minimal stand-in for the frozen SparseEditFormer critic."""

    def __call__(self, src, cand, ef, sid, eid, bid):
        B = src.shape[0]
        return {
            "mean": torch.zeros(B, 1),
            "logvar": torch.zeros(B, 1),
            "rank": torch.zeros(B, 1),
        }