"""B0-X effect dataset builder tests: delta derivation per asset type and the
feature module.  Tests the pure functions with synthetic inputs (no remote
data)."""
import json
import sys
from pathlib import Path

import numpy as np

B0X = Path(__file__).resolve().parents[2] / "scripts" / "b0x"
sys.path.insert(0, str(B0X))

import build_effect_dataset as B  # noqa: E402
import features as F  # noqa: E402


def _record(src="ACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGU",
            cand="ACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACG",
            metadata=None, rid="X_rec"):
    return {
        "record_id": rid,
        "source_sequence": src,
        "candidate_sequence": cand,
        "edit_script": [{"op": "SUB", "pos": 3, "token": "G"}],
        "metadata": metadata or {},
    }


def test_delta_diff_cand_src_gse114002():
    rec = _record(src="ACGU" * 20, cand="ACGU" * 12 + "ACGU", metadata=None)
    rec["source_sequence"] = "ACGU" * 20
    rec["candidate_sequence"] = "ACGU" * 20
    spec = {"type": "diff_cand_src", "cand_endpoint": "ep_rl", "src_source": "raw_library", "benchmark": "5U-A1"}
    cand_values = {"GSE114002_X__cand": {"ep_rl": 6.0}}
    src_anchors = {"ACGU" * 20: 5.0}
    out = B.derive_delta("GSE114002", spec, rec, cand_values, src_anchors, {}, "GSE114002_X__cand")
    assert len(out) == 1
    assert out[0]["delta_source_status"] == "derived"
    assert out[0]["delta"] == 1.0
    assert out[0]["source_value"] == 5.0
    assert out[0]["candidate_value"] == 6.0


def test_delta_diff_ref_alt():
    rec = _record()
    spec = {"type": "diff_ref_alt", "alt_endpoint": "ep_activity_alt_mean",
            "ref_endpoint": "ep_activity_ref_mean", "benchmark": "3U-A1"}
    cand_values = {"GSE232572_X__cand": {"ep_activity_alt_mean": 8.0, "ep_activity_ref_mean": 5.0}}
    out = B.derive_delta("GSE232572", spec, rec, cand_values, {}, {}, "GSE232572_X__cand")
    assert out[0]["delta"] == 3.0
    assert out[0]["delta_source_status"] == "derived"


def test_delta_log2fc_direct():
    rec = _record()
    spec = {"type": "log2fc", "endpoint": "ep_log2fc", "benchmark": "3U-A1"}
    cand_values = {"GSE298114_X__cand": {"ep_log2fc": 1.5}}
    out = B.derive_delta("GSE298114", spec, rec, cand_values, {}, {}, "GSE298114_X__cand")
    assert out[0]["delta"] == 1.5
    assert out[0]["delta_source_status"] == "log2fc_direct"


def test_delta_diff_wt_meta_two_endpoints():
    rec = _record(metadata={"stability_wt_hek": 70.0, "stability_wt_sh": 20.0})
    spec = {"type": "diff_wt_meta",
            "endpoints": [{"cand_endpoint": "ep_stability_hek", "wt_meta": "stability_wt_hek"},
                          {"cand_endpoint": "ep_stability_sh", "wt_meta": "stability_wt_sh"}],
            "benchmark": "5U-A1"}
    cand_values = {"GSE217518_X__cand": {"ep_stability_hek": 80.0, "ep_stability_sh": 25.0}}
    out = B.derive_delta("GSE217518", spec, rec, cand_values, {}, {}, "GSE217518_X__cand")
    assert len(out) == 2
    assert {r["endpoint"] for r in out} == {"ep_stability_hek", "ep_stability_sh"}
    assert out[0]["delta"] in (10.0, 5.0)
    assert all(r["delta_source_status"] == "derived" for r in out)


def test_delta_source_anchor_unavailable():
    rec = _record(src="ACGU" * 20, cand="ACGU" * 20)
    spec = {"type": "diff_cand_src", "cand_endpoint": "ep_rl", "src_source": "raw_library", "benchmark": "5U-A1"}
    cand_values = {"GSE114002_X__cand": {"ep_rl": 6.0}}
    # source anchor NOT in the raw library map -> unavailable
    out = B.derive_delta("GSE114002", spec, rec, cand_values, {}, {}, "GSE114002_X__cand")
    assert out[0]["delta_source_status"] == "source_anchor_unavailable"
    assert out[0]["delta"] is None


def test_load_pairs_filters_active(tmp_path):
    p = tmp_path / "pairs.jsonl"
    rows = [
        {"pair_id": "p1", "candidate_sequence_id": "GSE114002_A__cand", "source_sequence_id": "GSE114002_A__src", "scientific_track": "E"},
        {"pair_id": "p2", "candidate_sequence_id": "GSE114002_B__cand", "source_sequence_id": "GSE114002_B__src", "scientific_track": "F"},
        {"pair_id": "p3", "candidate_sequence_id": "GSE149487_C__cand", "source_sequence_id": "GSE149487_C__src", "scientific_track": "E"},
        {"pair_id": "p4", "candidate_sequence_id": "GSE232571_D__cand", "source_sequence_id": "GSE232571_D__src", "scientific_track": "E"},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    out = B.load_pairs(p)
    ids = {r["pair_id"] for r in out}
    assert ids == {"p1", "p4"}  # p2 not E, p3 not active


def test_features_shapes():
    src = "ACGUACGUACGUACGUACGU"
    cand = "ACGUACGUACGUACGUACGC"
    fx = F.extract_features(src, cand, [{"op": "SUB", "pos": 5, "token": "G"}])
    assert fx["source_feat"].shape == (20,)
    assert fx["candidate_feat"].shape == (20,)
    assert fx["diff_feat"].shape == (20,)
    assert fx["edit_feat"].shape == (12,)
    assert fx["source_onehot"].shape == (100, 4)
    # kmers vector: 2 * (4 + 16 + 64) = 168
    assert F.kmers_vector(src, cand).shape == (168,)


def test_gc_computation():
    seq = "GGGGCCCC"  # 8 nt, 8/8 GC
    f = F.sequence_features(seq)
    # GC is index 1 in the 4-dim header [len, gc, gc_first10, aug]
    assert abs(f[1] - 1.0) < 1e-5


def test_edit_features_empty_token_no_index_error():
    # An edit with an empty/none token must not raise IndexError.
    empty = F.edit_features([{"op": "SUB", "pos": 3, "token": ""}], 100)
    missing = F.edit_features([{"op": "SUB", "pos": 3}], 100)
    assert empty.shape == (12,)
    assert missing.shape == (12,)
    assert np.all(np.isfinite(empty))
    assert np.all(np.isfinite(missing))