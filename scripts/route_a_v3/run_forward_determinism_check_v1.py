#!/usr/bin/env python3
"""P2-1 (rigor audit A2+A3): forward-determinism check for the five new
benchmark-v2 families (GEMORNA / LAMAR-UTR5TEPred / UTR-Insight / UTR-STCNet /
HydraRNA), MIG-inference caliber.

Protocol (frozen before execution):
- For each family: take 100 matrix-cell input sequences (50 canonical source +
  50 candidate sequences, GSE114002 VALIDATION rows -- the exact sequences the
  leaderboard cells scored), run inference TWICE with the same weights, same
  input, same process (same CUDA device), compare run1 vs run2 per sequence.
- Metrics: max |delta| over the 100 sequences; Spearman rank correlation
  between the two prediction vectors; bitwise-identical flag.
- Verdict DETERMINISTIC if bitwise identical OR (max|delta| <= 1e-6 AND rank
  corr >= 0.999999); NONDETERMINISTIC otherwise.
- STCNet focus: official cluster_dpc_knn adds torch.rand*1e-6 tie-breaking
  noise (inference-time randomness by design, UTRFormer_layers.py L315-318).
  Tested BOTH: (a) pinned-seed protocol (matrix runner pinned per-batch
  torch.manual_seed(20260816)) and (b) unpinned native behavior, to quantify
  the raw nondeterminism level.
- HydraRNA focus: fairseq/Triton path runs in a SUBPROCESS with
  CUDA_VISIBLE_DEVICES=<gpu uuid> so the MIG slice appears as cuda:0 (the
  documented Triton/MIG single-device fix; weights/numerics unchanged, fp16).
- GEMORNA region dispatch: 5UTR cells scored by the 5' head, 3UTR by the 3'
  head; both heads tested (100 5UTR seqs for head-5, 50 3UTR seqs for head-3).
- Impact rule: if NONDETERMINISTIC with max|delta| > 1e-6 (perturbs beyond the
  4th decimal), classify REQUIRES_LEADERBOARD_NOTE; else NO_IMPACT / BENIGN.

Discipline: reads canonical VALIDATION sequences only (label-free usage);
no training; no weight changes; MIG inference on GPU 6/7 only (never competes
with the M1 watcher full-card queue on GPU 0-5); protected TEST reads = 0;
append-only output determinism.json.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
ADAPTERS = REPO_ROOT / "scripts/route_a_v3/benchmark_v2_matrix/family_adapters_v1.py"
MNT = Path("/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2")
OUT_DIR = MNT / "experiments/analysis_forward_determinism_v1"

MANIFEST = MNT / "manifests/route2_development_frozen_v1/development_manifest.jsonl"
CANON_GSE114002 = MNT / "canonical/GSE114002/v1/canonical_records.private.jsonl"
CANON_GSE269595 = MNT / "canonical/GSE269595/v1/canonical_records.private.jsonl"

N_SEQS = 100
VERDICT_TOL = 1e-6
RANK_FLOOR = 0.999999
STCNET_PIN_SEED = 20260816

PY = sys.executable


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_sequences():
    def validation_ids(study):
        ids = set()
        with MANIFEST.open() as handle:
            for line in handle:
                row = json.loads(line)
                if row["split"] == "VALIDATION" and row["study_unit_id"] == study:
                    ids.add(str(row["canonical_record_id"]))
        return ids

    rows = []
    ids = validation_ids("GSE114002")
    with CANON_GSE114002.open() as handle:
        for line in handle:
            row = json.loads(line)
            if str(row["canonical_record_id"]) in ids:
                rows.append(row)
    rows = sorted(rows, key=lambda r: str(r["canonical_record_id"]))
    five = [r["source_sequence"] for r in rows[:50]] + [
        r["candidate_sequence"] for r in rows[:50]
    ]

    rows3 = []
    ids3 = validation_ids("GSE269595")
    with CANON_GSE269595.open() as handle:
        for line in handle:
            row = json.loads(line)
            if str(row["canonical_record_id"]) in ids3:
                rows3.append(row)
    rows3 = sorted(rows3, key=lambda r: str(r["canonical_record_id"]))
    three = [r["source_sequence"] for r in rows3[:25]] + [
        r["candidate_sequence"] for r in rows3[:25]
    ]
    return five, three, {
        "gse114002_validation_rows": len(rows),
        "gse269595_validation_rows": len(rows3),
    }


def compare(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    bitwise = bool(np.array_equal(a, b))
    max_abs = float(np.abs(a - b).max()) if len(a) else 0.0
    rank = None
    if len(a) >= 3 and a.std() > 0 and b.std() > 0:
        from scipy.stats import spearmanr

        value = spearmanr(a, b).statistic
        rank = float(value) if np.isfinite(value) else None
    verdict = (
        "DETERMINISTIC"
        if bitwise or (max_abs <= VERDICT_TOL and rank is not None and rank >= RANK_FLOOR)
        else "NONDETERMINISTIC"
    )
    return {
        "n": int(len(a)),
        "bitwise_identical": bitwise,
        "max_abs_diff": max_abs,
        "rank_spearman_run1_vs_run2": rank,
        "verdict": verdict,
    }


def run_family(family, five, three, device_index):
    import torch

    device = torch.device(f"cuda:{device_index}")
    ad = _load_module("family_adapters_det", ADAPTERS)
    family_obj, meta = ad.build_scorer(family, device)

    def score(seqs):
        if isinstance(family_obj, dict) and "score_region" in family_obj:
            return np.asarray(family_obj["score_region"](seqs, "5UTR"), dtype=float)
        return np.asarray(family_obj(seqs), dtype=float)

    run1 = score(five)
    run2 = score(five)
    entry = {
        "family": family,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "mode": "SAME_PROCESS_TWO_FORWARDS",
        "n_seqs": len(five),
        **compare(run1, run2),
        "meta_weights_sha256": meta.get("weights_sha256"),
        "input_adaptation": meta.get("input_adaptation"),
    }

    if family == "gemorna":
        run3a = np.asarray(family_obj["score_region"](three, "3UTR"), dtype=float)
        run3b = np.asarray(family_obj["score_region"](three, "3UTR"), dtype=float)
        entry["gemorna_3utr_head"] = {"n_seqs": len(three), **compare(run3a, run3b)}
    return entry


def run_stcnet_pinned_and_unpinned(five, device_index):
    import torch

    device = torch.device(f"cuda:{device_index}")
    ad = _load_module("family_adapters_det", ADAPTERS)
    scorer, meta = ad.build_scorer("utr_stcnet", device)

    a = np.asarray(scorer(five), dtype=float)
    b = np.asarray(scorer(five), dtype=float)
    c = np.asarray(scorer(five), dtype=float)
    unpinned = compare(a, b)
    unpinned_ac = compare(a, c)

    def run_pinned():
        out = []
        for i in range(0, len(five), 256):
            torch.manual_seed(STCNET_PIN_SEED)
            chunk = five[i : i + 256]
            out.extend(np.asarray(scorer(chunk), dtype=float).tolist())
        return np.asarray(out, dtype=float)

    p1 = run_pinned()
    p2 = run_pinned()
    pinned = compare(p1, p2)

    if pinned["bitwise_identical"] or pinned["max_abs_diff"] <= VERDICT_TOL:
        verdict = "NONDETERMINISTIC_BY_DESIGN__PINNED_PROTOCOL_DETERMINISTIC"
    else:
        verdict = "NONDETERMINISTIC"
    return {
        "family": "utr_stcnet",
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "mode": "SAME_PROCESS_TWO_FORWARDS",
        "n_seqs": len(five),
        "unpinned_run1_vs_run2": unpinned,
        "unpinned_run1_vs_run3": unpinned_ac,
        "pinned_seed_20260816_run1_vs_run2": pinned,
        "noise_source": (
            "official UTRFormer_layers.cluster_dpc_knn adds torch.rand*1e-6 "
            "tie-breaking noise to token density at inference (official design, "
            "not a port bug); matrix runner pinned per-batch "
            f"torch.manual_seed({STCNET_PIN_SEED})"
        ),
        "verdict": verdict,
        "meta_weights_sha256": meta.get("weights_sha256"),
        "input_adaptation": meta.get("input_adaptation"),
    }


HYDRARNA_SUBPROCESS_CODE = r"""
import json, sys
import importlib.util
import numpy as np
import torch

adapters = "/home/cunyuliu/mrna_editflow_goal/worktrees/route_a_v3_w0_diagnosis_20260902/scripts/route_a_v3/benchmark_v2_matrix/family_adapters_v1.py"
spec = importlib.util.spec_from_file_location("family_adapters_hydra", adapters)
ad = importlib.util.module_from_spec(spec)
sys.modules["family_adapters_hydra"] = ad
spec.loader.exec_module(ad)

device = torch.device("cuda:0")
scorer, meta = ad.build_scorer("hydrarna", device)
seqs = json.load(open(sys.argv[1]))
run1 = np.asarray(scorer(seqs), dtype=float)
run2 = np.asarray(scorer(seqs), dtype=float)
json.dump({"run1": run1.tolist(), "run2": run2.tolist(), "gpu_name": torch.cuda.get_device_name(device)}, open(sys.argv[2], "w"))
print("hydrarna subprocess done", flush=True)
"""


def run_hydrarna_subprocess(five, gpu_uuid):
    import os
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        seq_path = Path(tmp) / "seqs.json"
        out_path = Path(tmp) / "out.json"
        code_path = Path(tmp) / "sub.py"
        seq_path.write_text(json.dumps(five))
        code_path.write_text(HYDRARNA_SUBPROCESS_CODE)
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = gpu_uuid
        result = subprocess.run(
            [PY, str(code_path), str(seq_path), str(out_path)],
            capture_output=True,
            text=True,
            env=env,
            timeout=1800,
        )
        if result.returncode != 0:
            return {
                "family": "hydrarna",
                "mode": "SUBPROCESS_CUDA_VISIBLE_DEVICES_SINGLE_DEVICE",
                "verdict": "PORT_FAILURE",
                "stderr_tail": (result.stderr or "")[-800:],
            }
        payload = json.loads(out_path.read_text())
    return {
        "family": "hydrarna",
        "mode": "SUBPROCESS_CUDA_VISIBLE_DEVICES_SINGLE_DEVICE (Triton/MIG fix)",
        "gpu_name": payload["gpu_name"],
        "cuda_visible_devices": gpu_uuid,
        "n_seqs": len(five),
        **compare(payload["run1"], payload["run2"]),
        "note": (
            "fairseq load_model_ensemble + extract_features mean-pooled scalar, "
            "half precision, subprocess with CUDA_VISIBLE_DEVICES pinned to the "
            "MIG slice uuid (single-device visibility; weights/numerics unchanged)"
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-gpu-index", type=int, default=6)
    parser.add_argument("--hydrarna-gpu-index", type=int, default=7)
    args = parser.parse_args()

    five, three, counts = load_sequences()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    import torch

    gpu_uuid = None
    for idx in (args.hydrarna_gpu_index, args.physical_gpu_index):
        try:
            props = torch.cuda.get_device_properties(idx)
            raw = str(getattr(props, "uuid", ""))
            gpu_uuid = raw.replace("GPU-", "") if raw else None
            if gpu_uuid:
                break
        except Exception:
            continue

    results = {
        "schema_version": "route_a_v3_forward_determinism_check_v1",
        "task": "P2-1 rigor audit A2+A3: five-family forward determinism",
        "protocol": {
            "n_sequences": N_SEQS,
            "sequence_source": (
                "GSE114002 VALIDATION 50 src + 50 cand (matrix cell input "
                "sequences); GEMORNA 3' head additionally 25+25 GSE269595 "
                "polyA 3UTR VALIDATION sequences"
            ),
            "determinism_verdict_rule": (
                f"DETERMINISTIC iff bitwise identical OR (max|d| <= {VERDICT_TOL} "
                f"AND rank spearman >= {RANK_FLOOR})"
            ),
            "same_weights_same_inputs_same_process": True,
            "gpu_scope": "MIG slices GPU 6/7 only (inference; never touches GPU 0-5 full-card queue)",
            "protected_test_reads": 0,
        },
        "sequence_counts": counts,
        "families": {},
    }

    order = ["lamar_utr5te", "gemorna", "utr_insight", "utr_stcnet", "hydrarna"]
    for family in order:
        print(f"[{family}] starting", flush=True)
        if family == "hydrarna":
            entry = run_hydrarna_subprocess(five, gpu_uuid)
        elif family == "utr_stcnet":
            entry = run_stcnet_pinned_and_unpinned(five, args.physical_gpu_index)
        else:
            entry = run_family(family, five, three, args.physical_gpu_index)
        results["families"][family] = entry
        print(
            json.dumps(
                {k: v for k, v in entry.items() if k != "input_adaptation"}, indent=1
            ),
            flush=True,
        )

    for family, entry in results["families"].items():
        if entry.get("verdict") == "PORT_FAILURE":
            entry["leaderboard_impact"] = {"classification": "PORT_FAILURE", "statement": "check failed to run"}
            continue
        max_d = entry.get("max_abs_diff")
        sub = entry.get("gemorna_3utr_head")
        if sub and sub.get("max_abs_diff", 0) > (max_d or 0):
            max_d = sub["max_abs_diff"]
        unp = entry.get("unpinned_run1_vs_run2", {})
        max_unp = unp.get("max_abs_diff")
        nondet = entry.get("verdict", "").startswith("NONDETERMINISTIC")
        if nondet and ((max_d is not None and max_d > 1e-6) or (max_unp is not None and max_unp > 1e-6)):
            impact = "REQUIRES_LEADERBOARD_NOTE"
        elif nondet:
            impact = "BENIGN_NONDETERMINISM"
        else:
            impact = "NO_IMPACT"
        entry["leaderboard_impact"] = {
            "statement": (
                "matrix cell numbers are single-forward single-run; max|d| "
                "between two same-input runs exceeds 1e-6 -> the cell's 4th "
                "decimal may not be bitwise reproducible -> leaderboard "
                "declaration note required"
                if impact == "REQUIRES_LEADERBOARD_NOTE"
                else "cell numbers bitwise/near-bitwise reproducible under the "
                "recorded protocol; no caveat needed"
                if impact == "NO_IMPACT"
                else "nondeterministic in raw scalar output but within tolerance"
            ),
            "classification": impact,
            "max_abs_diff_pinned_protocol": max_d,
            "max_abs_diff_unpinned_native": max_unp,
        }

    out_path = OUT_DIR / "determinism.json"
    if out_path.exists():
        raise SystemExit("determinism.json already exists (append-only discipline)")
    out_path.write_text(json.dumps(results, indent=1, sort_keys=True))
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
